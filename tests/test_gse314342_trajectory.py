from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import pytest

from analysis.extract_gse314342_effects import _rest_gene_aggregate
from credo.data.core import (
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from credo.data.gse314342 import (
    LateTimeResolution,
    build_mass_and_count_tables,
    build_support_metadata,
    canonicalize_obs,
)
from credo.data.trajectory_view import TrajectoryView
from credo.config.schema import RunConfig, TrajectoryTrainingConfig
from credo.losses.counts import (
    CountBlock,
    FitnessBank,
    GroupedMultiTimeCountLikelihood,
)
from credo.models.full_model import FullDynamicsModel
from credo.models.background_trajectory_counterfactual import BackgroundTrajectoryCounterfactualEngine
from credo.models.ecology import EcologicalPayoff
from credo.models.weighted_sde import WeightedParticleSimulator
from credo.training.trajectory_batch import TargetBalancedTrajectorySampler
from credo.training.trajectory_batch import initialise_particles_from_trajectory
from credo.training.trajectory_trainer import TrajectoryTrainer
from runners.run_credo_gse314342 import configured_args
from runners.run_credo_trajectory import (
    _load_continuation_model,
    build_config,
    export_trajectory_counterfactuals,
    parse_args as parse_trajectory_args,
    split_validation_study,
)
from scripts.make_gse314342_pilot import _select_guides


pytestmark = pytest.mark.integration


def test_support_metadata_matches_trajectory_contract() -> None:
    late_time = LateTimeResolution(
        physical_time_hours=48.0,
        status="resolved",
        processed_label="Stim48hr",
        raw_label="24 hr stim",
        rationale="processed release",
        source="test",
    )

    metadata = build_support_metadata(
        latent_key="X_credo",
        latent_dim=32,
        support_atoms_cap=64,
        late_time=late_time,
    )

    assert metadata["finite_measure_key"] == ["sample_id", "guide_id"]
    assert metadata["physical_times_hours"] == {
        "Rest": 0.0,
        "Stim8hr": 8.0,
        "Stim48hr": 48.0,
    }
    assert metadata["mass_mode"] == "group_total"


def test_pilot_selection_is_deterministic_and_requires_complete_keys() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["D3"] * 5,
            "guide_id": ["ntc1", "ntc2", "g1", "g2", "incomplete"],
            "embedding_id": ["__NTC__", "__NTC__", "A", "B", "C"],
            "is_control": [True, True, False, False, False],
            "has_Rest": [True] * 5,
            "has_Stim8hr": [True, True, True, True, False],
            "has_Stim48hr": [True] * 5,
        }
    )

    first, controls, genes = _select_guides(
        manifest,
        donor="D3",
        target_genes=1,
        controls=1,
        seed=7,
    )
    second, _, _ = _select_guides(
        manifest,
        donor="D3",
        target_genes=1,
        controls=1,
        seed=7,
    )

    assert first["guide_id"].tolist() == second["guide_id"].tolist()
    assert "incomplete" not in set(first["guide_id"])
    assert len(controls) == 1
    assert len(genes) == 1


def _mapped_study() -> PerturbSeqDynamicsData:
    rows = []
    latent = []
    for donor in ["D1", "D2"]:
        for guide, gene, control in [
            ("NTC-1", "__NTC__", True),
            ("GATA3-1", "GATA3", False),
            ("GATA3-2", "GATA3", False),
            ("STAT1-1", "STAT1", False),
        ]:
            for time_idx, label in enumerate(["Rest", "Stim8hr", "Stim48hr"]):
                for cell_idx in range(5):
                    rows.append(
                        {
                            "cell_id": f"{donor}_{guide}_{label}_{cell_idx}",
                            "perturbation_id": guide,
                            "time_label": label,
                            "sample_id": donor,
                            "guide_id": guide,
                            "embedding_id": gene,
                            "target_gene": gene,
                            "context_group_id": donor,
                            "view_id": f"{donor}::{guide}",
                            "is_control": control,
                        }
                    )
                    latent.append([float(time_idx), float(cell_idx) / 10])
    cells = pd.DataFrame(rows)
    masses = (
        cells.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .astype(float)
        .rename("mass")
        .reset_index()
    )
    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(["Rest", "Stim8hr", "Stim48hr"], [0, 8, 48]),
        catalog=PerturbationCatalog(
            ["NTC-1", "GATA3-1", "GATA3-2", "STAT1-1"],
            ["NTC-1"],
        ),
        cell_state=CellStateTable(cells, np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(masses),
    )


def test_measure_keys_share_target_gene_embedding_and_batch_together() -> None:
    trajectory = _mapped_study().to_sparse_trajectory_problem(by_sample=True)
    view = TrajectoryView(trajectory, "Rest", ["Stim8hr", "Stim48hr"])
    assert view.embedding_id(("D1", "GATA3-1")) == "GATA3"
    assert view.embedding_id(("D2", "GATA3-2")) == "GATA3"
    assert view.embedding_id(("D1", "NTC-1")) == "__NTC__"

    sampler = TargetBalancedTrajectorySampler(
        view,
        genes_per_batch=1,
        controls_per_batch=2,
        max_active_measure_keys=8,
        seed=3,
    )
    batches = list(sampler.batches())
    gata3 = next(batch for batch in batches if ("D1", "GATA3-1") in batch)
    assert {("D1", "GATA3-1"), ("D1", "GATA3-2"), ("D2", "GATA3-1"), ("D2", "GATA3-2")} <= set(gata3)


def test_target_support_cap_is_deterministic_and_preserves_mass() -> None:
    trajectory = _mapped_study().to_sparse_trajectory_problem(by_sample=True)
    view = TrajectoryView(trajectory, "Rest", ["Stim8hr", "Stim48hr"])
    support_a, logw_a = view.target_tensors(max_atoms=3, seed=9)
    support_b, logw_b = view.target_tensors(max_atoms=3, seed=9)
    key = ("D1", "GATA3-1")
    assert torch.equal(support_a["Stim8hr"][key], support_b["Stim8hr"][key])
    assert support_a["Stim8hr"][key].shape[0] == 3
    assert torch.isclose(logw_a["Stim8hr"][key].logsumexp(0).exp(), torch.tensor(5.0))


def test_optional_atom_weights_define_conditional_measure_weights() -> None:
    cells = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "perturbation_id": ["g1", "g1"],
            "time_label": ["Rest", "Rest"],
            "sample_id": ["D1", "D1"],
            "atom_weight": [1.0, 3.0],
        }
    )
    study = PerturbSeqDynamicsData(
        time_axis=TimeAxis(["Rest", "Stim"], [0, 1]),
        catalog=PerturbationCatalog(["g1"], ["g1"]),
        cell_state=CellStateTable(cells, np.zeros((2, 1), dtype=np.float32)),
        mass_table=MassTable(
            pd.DataFrame(
                {"perturbation_id": ["g1"], "time_label": ["Rest"], "sample_id": ["D1"], "mass": [8.0]}
            )
        ),
    )
    measure = study.build_measure("g1", "Rest", "D1")
    assert np.allclose(measure.weights, [2.0, 6.0])
    sparse_measure = study.to_sparse_trajectory_problem(by_sample=True).get(
        "Rest", ("D1", "g1")
    )
    assert np.allclose(sparse_measure.weights, measure.weights)


def test_grouped_ecology_conditions_payoff_on_grouped_mediators() -> None:
    payoff = EcologicalPayoff(2, 2, n_ranks=2, mediator_dim=2)
    eta = torch.randn(3, 4, 2)
    embedding = torch.randn(3, 2)
    q = torch.softmax(torch.randn(3, 2), dim=-1)
    s = torch.randn(3, 2, requires_grad=True)
    value = payoff(eta, embedding, q, s)
    value.sum().backward()
    assert value.shape == (3, 4)
    assert s.grad is not None and torch.count_nonzero(s.grad) > 0


def test_intrinsic_checkpoint_can_initialize_ecological_continuation() -> None:
    kwargs = dict(
        perturbation_ids=["ctrl", "gene"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
    )
    intrinsic = FullDynamicsModel(**kwargs, context_kind="none", ecological_growth=False)
    ecological = FullDynamicsModel(**kwargs, context_kind="mlp", ecological_growth=True)
    _load_continuation_model(ecological, {"model_state_dict": intrinsic.state_dict()})
    assert torch.allclose(
        ecological.embedding.reference_embedding,
        intrinsic.embedding.reference_embedding,
    )


def test_cohort_config_is_executable_and_cli_flags_win(tmp_path) -> None:
    config = tmp_path / "pilot.yaml"
    config.write_text(json.dumps({"args": ["--epochs", "2"]}), encoding="utf-8")
    args = parse_trajectory_args(
        configured_args(
            [
                "--config", str(config),
                "--data-path", "support.h5ad",
                "--output-dir", "run",
                "--epochs", "3",
            ]
        )
    )
    assert args.epochs == 3
    assert args.source_label == "Rest"
    assert args.context_protocol == "none"
    assert args.mass_scope == "full_obs"
    cfg = build_config(args, latent_dim=2)
    assert cfg.data.mass_scope == "full_obs"
    assert cfg.model.context_kind == "none"
    assert cfg.model.ecological_growth is False


def test_trajectory_config_has_only_executable_options() -> None:
    inert = {
        "context_batch_mode",
        "save_particles_every",
        "save_rollouts",
        "teacher_forced_weight",
        "trajectory_mode",
        "validation_source",
    }
    assert inert.isdisjoint(TrajectoryTrainingConfig.model_fields)


def test_trajectory_contract_rejects_batched_global_context(tmp_path) -> None:
    args = parse_trajectory_args(
        [
            "--data-path", "support.h5ad",
            "--output-dir", str(tmp_path),
            "--context-protocol", "global_self_consistent",
            "--context-kind", "mlp",
            "--max-active-measure-keys", "8",
        ]
    )
    with pytest.raises(ValueError, match="global_self_consistent.*batch"):
        build_config(args, latent_dim=2)


def test_no_context_model_rejects_ecological_payoff() -> None:
    with pytest.raises(ValueError, match="context_kind='none'.*ecological_growth"):
        FullDynamicsModel(
            ["ctrl", "gene"],
            ["ctrl"],
            latent_dim=2,
            embedding_dim=2,
            n_programs=2,
            mediator_dim=2,
            hidden_dim=8,
            depth=1,
            context_kind="none",
            ecological_growth=True,
        )


def test_leave_one_guide_out_keeps_partner_guide_for_shared_embedding() -> None:
    args = parse_trajectory_args(
        [
            "--data-path", "support.h5ad",
            "--output-dir", "run",
            "--validation-guide-ids", "GATA3-2",
            "--context-protocol", "none",
        ]
    )
    train, validation = split_validation_study(_mapped_study(), args)
    assert validation is not None
    assert "GATA3-2" not in set(train.cell_state.df["guide_id"])
    assert set(validation.cell_state.df["guide_id"]) == {"GATA3-2"}
    assert "GATA3-1" in set(train.cell_state.df["guide_id"])
    train_view = TrajectoryView(
        train.to_sparse_trajectory_problem(by_sample=True),
        "Rest",
        ["Stim8hr", "Stim48hr"],
    )
    val_view = TrajectoryView(
        validation.to_sparse_trajectory_problem(by_sample=True),
        "Rest",
        ["Stim8hr", "Stim48hr"],
    )
    assert set(val_view.embedding_id_list) == {"GATA3"}
    assert "GATA3" in set(train_view.embedding_id_list)


def test_rest_priming_aggregate_preserves_guide_and_donor_support() -> None:
    frame = pd.DataFrame(
        {
            "target_gene": ["GATA3", "GATA3"],
            "guide_id": ["GATA3-1", "GATA3-2"],
            "sample_id": ["D1", "D2"],
            "latent_mean_shift_norm": [1.0, 3.0],
        }
    )
    summary = _rest_gene_aggregate(frame).iloc[0]
    assert summary["n_guide_views"] == 2
    assert summary["n_donors"] == 2
    assert summary["latent_mean_shift_norm_median"] == pytest.approx(2.0)


def test_none_and_grouped_context_semantics() -> None:
    z = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]], [[10.0, 0.0]]])
    logw = torch.zeros(3, 1)
    log_m0 = torch.zeros(3)
    grouped = torch.tensor([0, 0, 1])

    intrinsic = FullDynamicsModel(
        ["ctrl", "gene"], ["ctrl"], latent_dim=2, embedding_dim=2,
        n_programs=2, mediator_dim=2, hidden_dim=8, depth=1,
        context_kind="none", ecological_growth=False,
    )
    _, intrinsic_state = intrinsic.step(
        z, torch.tensor(0.0), logw, log_m0,
        perturbation_ids=["a", "b", "c"],
        embedding_ids=["ctrl", "gene", "gene"],
        context_group_index=grouped,
    )
    assert intrinsic_state.context.shape == (3, 4)
    assert torch.count_nonzero(intrinsic_state.context) == 0

    contextual = FullDynamicsModel(
        ["ctrl", "gene"], ["ctrl"], latent_dim=2, embedding_dim=2,
        n_programs=2, mediator_dim=2, hidden_dim=8, depth=1,
        context_kind="mlp", ecological_growth=False,
    )
    _, grouped_state = contextual.step(
        z, torch.tensor(0.0), logw, log_m0,
        perturbation_ids=["a", "b", "c"],
        embedding_ids=["ctrl", "gene", "gene"],
        context_group_index=grouped,
    )
    assert torch.allclose(grouped_state.context[0], grouped_state.context[1])
    assert not torch.allclose(grouped_state.context[0], grouped_state.context[2])
    assert torch.allclose(grouped_state.freq_g[:2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(grouped_state.freq_g[2:], torch.tensor([1.0]))


def test_grouped_count_bank_keeps_donor_denominator_and_active_gradient() -> None:
    growth = torch.tensor([[[0.2]], [[-0.1]]], requires_grad=True)
    logw = torch.zeros(3, 1, 1)
    tau = torch.tensor([0.0, 0.5, 1.0])
    block = CountBlock(
        context_group_id="D1",
        time_label="Stim48hr",
        key_indices=torch.tensor([0, 1]),
        exposure=torch.tensor([0.5, 0.5]),
        counts=torch.tensor([7.0, 3.0]),
        n_total=torch.tensor(10.0),
    )
    bank = FitnessBank(["Stim48hr"], 2)
    bank.values[0, 1] = 1.5
    likelihood = GroupedMultiTimeCountLikelihood()
    loss, logs = likelihood.forward_with_logs(
        growth_steps=growth,
        logw_steps=logw,
        tau_steps=tau,
        blocks=[block],
        checkpoint_indices={"Stim48hr": 2},
        active_key_indices=torch.tensor([0]),
        fitness_bank=bank,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert growth.grad is not None and torch.count_nonzero(growth.grad) > 0
    assert int(logs["counts/n_blocks"]) == 1


def test_gse314342_qc_mass_and_late_time_provenance(tmp_path) -> None:
    resolution_path = tmp_path / "late.json"
    resolution_path.write_text(
        json.dumps(
            {
                "accession": "GSE314342",
                "status": "resolved",
                "processed_label": "Stim48hr",
                "raw_label": "24 hr stim",
                "physical_time_hours": 48,
                "rationale": "processed release",
                "source": "test",
            }
        )
    )
    resolution = LateTimeResolution.load(resolution_path)
    obs = pd.DataFrame(
        {
            "guide_id": ["g1", "ntc", "multi-guide", None, "bad"],
            "guide_type": ["targeting", "non-targeting", "targeting", "targeting", "targeting"],
            "target_gene_name": ["GATA3", "ignored", "STAT1", "STAT1", "STAT1"],
            "donor_id": ["CE1"] * 5,
            "condition": ["Stim48hr"] * 5,
            "low_quality": [False, False, False, False, True],
        }
    )
    canonical = canonicalize_obs(
        obs,
        donor_aliases={"CE1": "D1"},
        late_time=resolution,
        source_name="d1_48.h5ad",
    )
    assert canonical["guide_id"].tolist() == ["g1", "ntc"]
    assert canonical.loc[canonical["guide_id"].eq("ntc"), "embedding_id"].item() == "__NTC__"
    assert set(canonical["physical_time"]) == {48.0}

    repeated = pd.concat(
        [canonical.assign(time_label=label) for label in ["Rest", "Stim8hr", "Stim48hr"]],
        ignore_index=True,
    )
    counts, blocks = build_mass_and_count_tables(repeated, alpha=0.5)
    assert np.allclose(
        counts.groupby(["sample_id", "time_label"])["mass_value"].sum().to_numpy(),
        1.0,
    )
    assert set(blocks["time_label"]) == {"Stim8hr", "Stim48hr"}


def test_gse314342_release_schema_uses_authoritative_single_guide_group(tmp_path) -> None:
    resolution_path = tmp_path / "late.json"
    resolution_path.write_text(
        json.dumps(
            {
                "accession": "GSE314342",
                "status": "resolved",
                "processed_label": "Stim48hr",
                "raw_label": "24 hr stim",
                "physical_time_hours": 48,
                "rationale": "processed release",
                "source": "test",
            }
        )
    )
    obs = pd.DataFrame(
        {
            "guide_id": ["GATA3-1", "NTC-1", "multi_sgRNA", None, "STAT1-1"],
            "guide_type": ["targeting", "non-targeting", "targeting", None, "targeting"],
            "guide_group": [
                "targeting single sgRNA",
                "targeting single sgRNA",
                "multi sgRNA",
                "no sgRNA",
                "targeting single sgRNA",
            ],
            "perturbed_gene_name": ["GATA3", "NTC", "STAT1", None, "STAT1"],
            "perturbed_gene_id": ["ENSG1", "NTC", "ENSG2", None, "ENSG2"],
            "top_guide_UMI_counts": [10, 11, 20, 0, 5],
            "lane_id": ["L1"] * 5,
            "low_quality": [False, False, False, False, True],
        }
    )
    canonical = canonicalize_obs(
        obs,
        donor_aliases=None,
        late_time=LateTimeResolution.load(resolution_path),
        source_name="D1_Stim8hr.assigned_guide.h5ad",
        sample_id="D1",
        original_donor_id="CE0008162",
        time_label="Stim8hr",
        run_id="CD4i_R1",
    )
    assert canonical["guide_id"].tolist() == ["GATA3-1", "NTC-1"]
    assert canonical["sample_id"].unique().tolist() == ["D1"]
    assert canonical["original_donor_id"].unique().tolist() == ["CE0008162"]
    assert canonical["run_id"].unique().tolist() == ["CD4i_R1"]
    assert canonical["guide_umi_count"].tolist() == [10, 11]
    assert canonical.loc[canonical["is_control"], "target_gene"].item() == "__NTC__"
    assert canonical.loc[~canonical["is_control"], "target_gene_id"].item() == "ENSG1"


def _trajectory_config(tmp_path, context_kind: str, context_protocol: str) -> RunConfig:
    cfg = RunConfig(output_dir=str(tmp_path), device="cpu")
    cfg.latent.dim = 2
    cfg.model.embedding_dim = 2
    cfg.model.n_programs = 2
    cfg.model.mediator_dim = 2
    cfg.model.hidden_dim = 8
    cfg.model.depth = 1
    cfg.model.context_kind = context_kind
    cfg.model.ecological_growth = False
    cfg.simulation.n_particles = 2
    cfg.eval.n_eval_particles = 2
    cfg.training.epochs = 1
    cfg.training.lambda_weak = 0
    cfg.training.lambda_count = 0
    cfg.training.lambda_reg_net = 0
    cfg.training.lambda_reg_diffusion = 0
    cfg.training.lambda_reg_embed = 0
    cfg.training.lambda_reg_growth_bias = 0
    cfg.training.sinkhorn_max_iter = 4
    cfg.trajectory_training.source_label = "Rest"
    cfg.trajectory_training.target_labels = ["Stim8hr", "Stim48hr"]
    cfg.trajectory_training.steps_per_interval = 1
    cfg.trajectory_training.max_active_measure_keys = 8
    cfg.trajectory_training.genes_per_batch = 1
    cfg.trajectory_training.controls_per_batch = 2
    cfg.trajectory_training.context_protocol = context_protocol
    return cfg


def _trajectory_model(context_kind: str) -> FullDynamicsModel:
    return FullDynamicsModel(
        ["__NTC__", "GATA3", "STAT1"],
        ["__NTC__"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        context_kind=context_kind,
        ecological_growth=False,
    )


@pytest.mark.parametrize(
    ("context_kind", "context_protocol"),
    [("none", "none"), ("mlp", "grouped_self_consistent")],
)
def test_target_balanced_trainer_runs_with_gene_embeddings_and_context_bank(
    tmp_path,
    context_kind: str,
    context_protocol: str,
) -> None:
    trajectory = _mapped_study().to_sparse_trajectory_problem(by_sample=True)
    cfg = _trajectory_config(tmp_path, context_kind, context_protocol)
    model = _trajectory_model(context_kind)
    trainer = TrajectoryTrainer(
        model,
        cfg,
        trajectory,
        "Rest",
        ["Stim8hr", "Stim48hr"],
        output_dir=str(tmp_path),
        ema_decay=0,
    )
    trainer.train()
    predictions = pd.read_csv(tmp_path / "predicted_metrics_by_key_time.csv")
    assert predictions["measure_key"].nunique() == 8
    assert set(predictions["embedding_id"]) == {"__NTC__", "GATA3", "STAT1"}
    assert (trainer.context_bank is not None) is (context_protocol == "grouped_self_consistent")
    counterfactuals = export_trajectory_counterfactuals(
        trainer,
        output_path=tmp_path / "counterfactuals.csv",
        n_particles=2,
        max_keys=1,
        seed=11,
    )
    assert set(counterfactuals["target_label"]) == {"Stim8hr", "Stim48hr"}
    assert {"guide_id", "target_gene", "context_group_id", "is_control"} <= set(
        counterfactuals.columns
    )


def test_grouped_guide_validation_context_retains_heldout_rest_views(tmp_path) -> None:
    args = parse_trajectory_args(
        [
            "--data-path", "support.h5ad",
            "--output-dir", "run",
            "--validation-guide-ids", "GATA3-2",
            "--context-protocol", "grouped_self_consistent",
        ]
    )
    train, validation = split_validation_study(_mapped_study(), args)
    assert validation is not None
    train_trajectory = train.to_sparse_trajectory_problem(by_sample=True)
    validation_trajectory = validation.to_sparse_trajectory_problem(by_sample=True)
    cfg = _trajectory_config(tmp_path, "mlp", "grouped_self_consistent")
    trainer = TrajectoryTrainer(
        _trajectory_model("mlp"),
        cfg,
        train_trajectory,
        "Rest",
        ["Stim8hr", "Stim48hr"],
        validation_trajectory=validation_trajectory,
        output_dir=str(tmp_path),
        ema_decay=0,
    )

    train_keys = set(TrajectoryView(train_trajectory, "Rest", ["Stim8hr", "Stim48hr"]).source_keys)
    heldout_keys = set(
        TrajectoryView(
            validation_trajectory,
            "Rest",
            ["Stim8hr", "Stim48hr"],
        ).source_keys
    )
    assert set(trainer.context_measure_keys) == train_keys | heldout_keys
    assert trainer.context_bank is not None
    assert trainer.context_bank.log_n_steps.shape[1] == len(train_keys | heldout_keys)
    result = trainer.evaluate()
    assert result["metrics"]["validation_source"] == "held_out"
    assert set(result["predictions"]["validation_source"]) == {"held_out"}
    assert set(result["predictions"]["guide_id"]) == {"GATA3-2"}


def test_background_counterfactual_preserves_full_donor_context() -> None:
    trajectory = _mapped_study().to_sparse_trajectory_problem(by_sample=True)
    view = TrajectoryView(trajectory, "Rest", ["Stim8hr", "Stim48hr"])
    model = FullDynamicsModel(
        ["__NTC__", "GATA3", "STAT1"],
        ["__NTC__"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=2,
        hidden_dim=8,
        depth=1,
        context_kind="mlp",
        ecological_growth=False,
    )
    z0, logw0, log_m0 = initialise_particles_from_trajectory(
        trajectory, "Rest", view.source_keys, 2, seed=5
    )
    stats, _, _ = model.context_agg.summarize_groups(z0, logw0, log_m0)
    group_index = torch.tensor([0 if view.context_group(key) == "D1" else 1 for key in view.source_keys])
    focal = ("D1", "GATA3-1")
    engine = BackgroundTrajectoryCounterfactualEngine(
        model,
        WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=2,
    )
    result = engine.run(
        trajectory,
        source_label="Rest",
        target_labels=["Stim8hr", "Stim48hr"],
        focal_measure_key=focal,
        focal_embedding_id="GATA3",
        tau_grid=torch.tensor([0.0, 8 / 48, 1.0]),
        checkpoint_indices={"Rest": 0, "Stim8hr": 1, "Stim48hr": 2},
        background_group_statistics_steps=[stats, stats],
        background_context_group_index=group_index,
        focal_global_index=view.source_keys.index(focal),
        seed=7,
    )
    assert result.factual.context_steps.shape == (2, 1, 4)
    assert result.reference_clamped is not None
    assert set(result.metrics_by_time["time_label"] if "time_label" in result.metrics_by_time else result.metrics_by_time["target_label"]) == {"Stim8hr", "Stim48hr"}

    with pytest.raises(ValueError, match="exactly one"):
        engine.run(
            trajectory,
            source_label="Rest",
            target_labels=["Stim8hr", "Stim48hr"],
            focal_measure_key=focal,
            focal_embedding_id="GATA3",
            tau_grid=torch.tensor([0.0, 8 / 48, 1.0]),
            checkpoint_indices={"Rest": 0, "Stim8hr": 1, "Stim48hr": 2},
            background_group_statistics_steps=[stats, stats],
            background_context_steps=torch.zeros(2, 1, 4),
            background_context_group_index=group_index,
            focal_global_index=view.source_keys.index(focal),
        )
