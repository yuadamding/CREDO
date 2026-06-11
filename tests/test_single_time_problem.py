from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from credo.cli.validate_data import main as validate_data_main
from credo.config.schema import RunConfig
from credo.data import (
    SingleTimeProblem,
    TimeAxis,
    build_single_time_problem_from_anndata,
    validate_anndata_schema,
)
from credo.losses import (
    control_null_effect_loss,
    minimal_effect_action_loss,
    single_time_guide_concordance_loss,
)
from credo.models import (
    FullDynamicsModel,
    SingleTimeCounterfactualEngine,
    WeightedParticleSimulator,
    resolve_single_time_context_tau,
)
from credo.models.context import ContextState
from credo.models.single_time_context import SingleTimeContextProvider
from credo.training import SingleTimeTrainer
from credo.training.trainer import Trainer


pytestmark = pytest.mark.unit


def _single_time_adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "perturbation_id": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_b", "gene_b"],
            "is_control": [True, True, False, False, True, True, False, False],
            "sample_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    data = ad.AnnData(X=np.ones((8, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.0, 1.0],
            [0.0, 1.1],
        ],
        dtype=np.float32,
    )
    return data


def _model() -> FullDynamicsModel:
    torch.manual_seed(0)
    return FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=1,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
    )


def _target_gene_model() -> FullDynamicsModel:
    torch.manual_seed(0)
    return FullDynamicsModel(
        perturbation_ids=["ctrl", "gene_a", "gene_b"],
        control_ids=["ctrl"],
        latent_dim=2,
        embedding_dim=2,
        n_programs=2,
        mediator_dim=1,
        hidden_dim=8,
        depth=1,
        ecological_growth=False,
        control_mode="soft_ref",
    )


def test_single_time_does_not_relax_longitudinal_time_axis() -> None:
    with pytest.raises(ValueError, match="Need at least two time points"):
        TimeAxis(["snapshot"], [0.0])


def test_single_time_schema_profile_accepts_snapshot_with_sample_id(tmp_path) -> None:
    path = tmp_path / "single_time.h5ad"
    _single_time_adata().write_h5ad(path)

    report = validate_anndata_schema(path, schema="single_time")

    assert report["ok"] is True
    assert report["schema"] == "single_time"
    assert report["n_controls"] == 4
    assert report["n_non_controls"] == 4


def test_single_time_schema_profile_rejects_missing_control_flag(tmp_path) -> None:
    path = tmp_path / "missing_control.h5ad"
    data = _single_time_adata()
    del data.obs["is_control"]
    data.write_h5ad(path)

    report = validate_anndata_schema(path, schema="single_time")

    assert report["ok"] is False
    assert "is_control" in report["obs_columns_missing"]


def test_validate_data_cli_accepts_schema_profile_alias(tmp_path, capsys) -> None:
    path = tmp_path / "single_time.h5ad"
    _single_time_adata().write_h5ad(path)

    exit_code = validate_data_main([
        "--data-path",
        str(path),
        "--schema-profile",
        "single_time",
        "--json",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["schema"] == "single_time"


def test_single_time_problem_builds_nonphysical_effect_endpoint() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    endpoint = problem.to_effect_endpoint_problem()

    assert isinstance(problem, SingleTimeProblem)
    assert endpoint.time_axis.labels == ["control_reference", "observed_snapshot"]
    assert endpoint.metadata["problem_mode"] == "single_time"
    assert endpoint.metadata["effect_axis_is_physical_time"] is False
    assert endpoint.metadata["claim_level"] == "single_time_effect_path"
    assert endpoint.metadata["context_protocol"] == "observed_snapshot"
    assert endpoint.perturbation_ids == ["s1::ctrl", "s2::ctrl", "s1::gene_a", "s2::gene_b"]
    assert endpoint.metadata["abundance_claim_grade"] == "none"
    assert endpoint.metadata["view_level"] == "view"
    assert endpoint.metadata["measure_to_embedding"]["s1::gene_a"] == "gene_a"
    assert endpoint.metadata["measure_to_original_perturbation"]["s1::gene_a"] == "gene_a"
    assert endpoint.initial["s1::gene_a"].n_atoms == 2
    assert endpoint.terminal["s1::gene_a"].n_atoms == 2
    assert endpoint.initial["s1::ctrl"].n_atoms == 1
    assert endpoint.terminal["s1::ctrl"].n_atoms == 1


def test_single_time_effect_endpoint_can_pool_by_embedding() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    endpoint = problem.to_effect_endpoint_problem(view_level="embedding")

    assert endpoint.perturbation_ids == ["ctrl", "gene_a", "gene_b"]
    assert endpoint.metadata["view_level"] == "embedding"
    assert endpoint.metadata["measure_to_embedding"]["gene_a"] == "gene_a"
    assert endpoint.initial["gene_a"].n_atoms == 2
    assert endpoint.terminal["gene_a"].n_atoms == 2


def test_single_time_problem_uses_batch_matching_without_sample_id() -> None:
    data = _single_time_adata()
    data.obs["batch_id"] = data.obs["sample_id"].replace({"s1": "b1", "s2": "b2"})
    del data.obs["sample_id"]

    problem = build_single_time_problem_from_anndata(data, reference_scope="batch")

    assert {view.batch_id for view in problem.views} == {"b1", "b2"}
    assert {
        view.reference_scope
        for view in problem.views
        if not view.is_control
    } == {"batch"}
    assert {
        view.reference_scope
        for view in problem.views
        if view.is_control
    } == {"control_cell_split"}


def test_single_time_unit_mass_disables_abundance_claims() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata())

    assert problem.abundance_claims_allowed is False
    assert problem.abundance_claim_grade == "none"
    assert problem.to_effect_endpoint_problem().metadata["abundance_claims_allowed"] is False


def test_single_time_cell_count_is_diagnostic_not_claim_grade() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), mass_mode="cell_count")

    assert problem.abundance_claims_allowed is False
    assert problem.abundance_claim_grade == "diagnostic"
    assert problem.to_effect_endpoint_problem().metadata["abundance_claim_grade"] == "diagnostic"


def test_single_time_config_rejects_abundance_claims_without_claim_grade_mass() -> None:
    with pytest.raises(ValueError, match="abundance_claims"):
        RunConfig(
            single_time={
                "enabled": True,
                "mass_mode": "unavailable",
                "abundance_claims": "enabled",
            }
        )
    with pytest.raises(ValueError, match="cell_count"):
        RunConfig(
            single_time={
                "enabled": True,
                "mass_mode": "cell_count",
                "abundance_claims": "enabled",
            }
        )


def test_single_time_builder_stores_guide_target_metadata() -> None:
    data = _single_time_adata()
    data.obs["guide_id"] = ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1", "ctrl_g1", "ctrl_g1", "gb_g1", "gb_g1"]
    data.obs["target_gene"] = ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_b", "gene_b"]

    problem = build_single_time_problem_from_anndata(data, embedding_level="target_gene")
    endpoint = problem.to_effect_endpoint_problem()

    gene_a = next(view for view in problem.views if view.embedding_id == "gene_a")
    assert gene_a.guide_id == "ga_g1"
    assert gene_a.target_gene == "gene_a"
    assert endpoint.metadata["target_ids"]["s1::gene_a"] == "gene_a"
    assert endpoint.metadata["measure_to_guide"]["s1::gene_a"] == "ga_g1"


def test_single_time_target_plus_guide_residual_is_rejected_until_modeled() -> None:
    data = _single_time_adata()
    data.obs["guide_id"] = ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1", "ctrl_g1", "ctrl_g1", "gb_g1", "gb_g1"]
    data.obs["target_gene"] = ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_b", "gene_b"]

    with pytest.raises(NotImplementedError, match="hierarchical"):
        build_single_time_problem_from_anndata(data, embedding_level="target_plus_guide_residual")


def test_single_time_counterfactual_uses_same_reference_source_and_noise() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    engine = SingleTimeCounterfactualEngine(
        model=_model(),
        simulator=WeightedParticleSimulator(n_steps=2, store_history=True),
        n_particles=4,
    )

    result = engine.run(problem, ["gene_a"], seed=9, common_noise=True)[0]

    assert result.metadata["counterfactual_type"] == "single_time_effect_path"
    assert result.metadata["same_reference_source"] is True
    assert result.metadata["same_start_semantics"] == "constructed_reference_source"
    assert result.metadata["same_noise"] is True
    assert result.metadata["context_protocol"] == "observed_snapshot"
    assert result.metadata["target_measure_key"] == "s1::gene_a"
    assert result.metadata["target_embedding_id"] == "gene_a"
    assert torch.equal(result.rollout_perturb.z_steps[0], result.rollout_control.z_steps[0])
    assert torch.equal(result.rollout_perturb.logw_steps[0], result.rollout_control.logw_steps[0])
    assert torch.equal(result.rollout_perturb.noise_steps, result.rollout_control.noise_steps)


def test_single_time_context_tau_defaults_match_protocol() -> None:
    assert resolve_single_time_context_tau("observed_snapshot", "auto") == 1.0
    assert resolve_single_time_context_tau("source_reference", "auto") == 0.0
    assert resolve_single_time_context_tau("observed_snapshot", "midpoint") == 0.5
    assert resolve_single_time_context_tau("observed_snapshot", 0.25) == 0.25


def test_single_time_context_provider_fixed_cache_reuses_context() -> None:
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    endpoint = problem.to_effect_endpoint_problem(view_level="view")
    provider = SingleTimeContextProvider(
        problem=problem,
        endpoint=endpoint,
        n_particles=4,
        protocol="observed_snapshot",
        context_sampling="fixed",
    )
    model = _model()

    ctx1 = provider.build(model, seed=1, perturbation_ids=endpoint.perturbation_ids)
    ctx2 = provider.build(model, seed=999, perturbation_ids=endpoint.perturbation_ids)

    assert ctx1 is ctx2


def test_context_state_override_recomputes_mass_diagnostics() -> None:
    model = _model()
    z = torch.zeros(2, 3, 2)
    logw = torch.full((2, 3), -np.log(3.0))
    log_m0 = torch.tensor([0.0, 2.0])
    override = ContextState(
        q=torch.tensor([0.5, 0.5]),
        s=torch.tensor([0.0]),
        context=torch.tensor([0.5, 0.5, 0.0]),
        mass_g=torch.ones(2) * 999.0,
        freq_g=torch.tensor([0.5, 0.5]),
        log_mass_g=torch.ones(2) * 999.0,
        log_total_mass=torch.tensor(999.0),
    )

    _, ctx = model.step(
        z=z,
        tau=torch.tensor(0.0),
        logw=logw,
        log_m0=log_m0,
        perturbation_ids=["gene_a", "ctrl"],
        context_override=override,
    )

    assert torch.allclose(ctx.log_mass_g, log_m0, atol=1e-6)
    assert not torch.allclose(ctx.mass_g, override.mass_g)


def test_single_time_trainer_wires_context_and_extra_losses(tmp_path) -> None:
    config = RunConfig(
        simulation={"n_particles": 4, "n_steps": 2, "store_history": True},
        training={
            "epochs": 1,
            "lambda_count": 0.0,
            "lambda_weak": 0.0,
            "lambda_reg_net": 0.0,
            "lambda_reg_diffusion": 0.0,
            "lambda_reg_embed": 0.0,
        },
        single_time={
            "enabled": True,
            "lambda_control_null": 0.1,
            "lambda_minimal_action": 0.1,
            "lambda_guide_concordance": 0.1,
        },
    )
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    trainer = SingleTimeTrainer(
        model=_model(),
        config=config,
        problem=problem,
        output_dir=str(tmp_path),
        warmup_epochs=0,
    )

    result = trainer.train(n_epochs=1)

    assert result.history.loss_extra[0] >= 0.0
    assert result.claim_report["effect_axis_is_physical_time"] is False
    assert result.claim_report["view_level"] == "view"
    assert trainer.endpoint.perturbation_ids == ["s1::ctrl", "s2::ctrl", "s1::gene_a", "s2::gene_b"]
    assert trainer.trainer._embedding_ids_for_pids(["s1::gene_a"]) == ["gene_a"]


def test_single_time_trainer_can_use_target_gene_embedding_level(tmp_path) -> None:
    data = _single_time_adata()
    data.obs["guide_id"] = ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1", "ctrl_g1", "ctrl_g1", "gb_g1", "gb_g1"]
    data.obs["target_gene"] = ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_b", "gene_b"]
    problem = build_single_time_problem_from_anndata(
        data,
        reference_scope="sample",
        embedding_level="target_gene",
    )
    config = RunConfig(
        simulation={"n_particles": 4, "n_steps": 1, "store_history": True},
        training={"epochs": 1, "lambda_count": 0.0, "lambda_weak": 0.0},
        single_time={"enabled": True},
    )

    trainer = SingleTimeTrainer(
        model=_target_gene_model(),
        config=config,
        problem=problem,
        output_dir=str(tmp_path),
        warmup_epochs=0,
    )

    assert trainer.endpoint.metadata["measure_to_embedding"]["s1::gene_a"] == "gene_a"
    assert trainer.trainer._embedding_ids_for_pids(["s1::gene_a", "s2::gene_b"]) == ["gene_a", "gene_b"]


def test_single_time_hooks_are_rejected_in_chunked_trainer(tmp_path) -> None:
    config = RunConfig(
        simulation={"n_particles": 4, "n_steps": 1, "store_history": True},
        training={
            "epochs": 1,
            "lambda_count": 0.0,
            "lambda_weak": 0.0,
            "max_active_perturbations": 1,
        },
        single_time={"enabled": True},
    )
    problem = build_single_time_problem_from_anndata(_single_time_adata(), reference_scope="sample")
    endpoint = problem.to_effect_endpoint_problem(view_level="view")
    trainer = Trainer(
        model=_model(),
        config=config,
        endpoint=endpoint,
        supported_pids=endpoint.perturbation_ids,
        output_dir=str(tmp_path),
        particle_sampling="measure_weights",
        context_override_provider=lambda **_: None,
        extra_loss_callback=lambda **_: (torch.tensor(0.0), {}),
    )
    optimizer = trainer._build_optimizer(stage="all")

    with pytest.raises(ValueError, match="full perturbation rollout|chunked trainer"):
        trainer._one_epoch(
            optimizer,
            epoch=0,
            stage="all",
            perturbation_ids=endpoint.perturbation_ids,
        )


def test_control_cell_split_is_seeded() -> None:
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(12)],
            "perturbation_id": ["ctrl"] * 6 + ["gene_a"] * 6,
            "is_control": [True] * 6 + [False] * 6,
            "sample_id": ["s1"] * 12,
        },
        index=[f"cell_{i}" for i in range(12)],
    )
    data = ad.AnnData(X=np.ones((12, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.stack([np.asarray([float(i), 0.0]) for i in range(12)]).astype(np.float32)

    p1 = build_single_time_problem_from_anndata(data, control_split_seed=1, reference_scope="sample")
    p2 = build_single_time_problem_from_anndata(data, control_split_seed=1, reference_scope="sample")
    p3 = build_single_time_problem_from_anndata(data, control_split_seed=9, reference_scope="sample")
    ctrl1 = next(view for view in p1.views if view.is_control)
    ctrl2 = next(view for view in p2.views if view.is_control)
    ctrl3 = next(view for view in p3.views if view.is_control)

    assert np.array_equal(ctrl1.source.support, ctrl2.source.support)
    assert not np.array_equal(ctrl1.source.support, ctrl3.source.support)


def test_single_time_regularizer_helpers() -> None:
    effects = torch.tensor([0.1, 2.0, 4.0])
    is_control = torch.tensor([True, False, False])

    assert torch.isclose(control_null_effect_loss(effects, is_control), torch.tensor(0.01))
    assert minimal_effect_action_loss(growth_steps=torch.ones(2, 3, 4)).item() == pytest.approx(1.0)
    assert minimal_effect_action_loss(
        sigma_steps=torch.ones(2, 3, 4),
        growth_steps=torch.ones(2, 3, 4),
    ).item() == pytest.approx(2.0)
    assert single_time_guide_concordance_loss(
        torch.tensor([1.0, 3.0, 10.0]),
        ["gene_a", "gene_a", "gene_b"],
    ).item() == pytest.approx(1.0)
