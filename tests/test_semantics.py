from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from credo.contracts import Axis, FiniteMeasure, MassSemantics, SplitSpec, TrajectoryData
from credo.counterfactual import COMMON_COUNTERFACTUAL_COLUMNS
from credo.io import RunConfig, load_data, validate_run_data
from credo.model import CREDOModel
from credo.objective import (
    CountBlock,
    checkpoint_geometry_mass_loss,
    count_block_loss,
    total_objective,
)
from credo.particles import (
    CatalogContextProvider,
    DynamicsStep,
    NoContextProvider,
    ParticleState,
    axis_grid,
    euler_maruyama_rollout,
    rollout,
    sample_initial_particles,
    sample_noise,
)
from credo.training import CatalogBank, _representation_scope, _validation_split


def _model(data: TrajectoryData, *, context: str = "none") -> CREDOModel:
    return CREDOModel(
        embedding_ids=data.embedding_ids,
        control_embedding_ids=data.control_embedding_ids,
        latent_dim=data.latent_dim,
        embedding_dim=4,
        n_programs=4,
        hidden_dim=16,
        context_mode=context,
    )


def test_measure_metadata_is_one_to_one_and_ids_are_opaque(tiny_data) -> None:
    assert tiny_data.measure_meta["measure_id"].is_unique
    assert set(tiny_data.measure_ids) == set(tiny_data.measures[tiny_data.axis.source])
    assert all(isinstance(measure_id, str) for measure_id in tiny_data.measure_ids)
    assert any(
        row.measure_id != row.embedding_id for row in tiny_data.measure_meta.itertuples(index=False)
    )


def test_embedding_control_status_must_be_consistent(tiny_data) -> None:
    metadata = tiny_data.measure_meta.copy()
    metadata.loc[metadata["measure_id"].eq("D1::GENE1-1"), "is_control"] = True
    with pytest.raises(ValueError, match="mixes control and non-control"):
        replace(tiny_data, measure_meta=metadata)


def test_representation_and_split_provenance_are_strict(tiny_data) -> None:
    representation = replace(tiny_data.representation, latent_dim=tiny_data.latent_dim + 1)
    with pytest.raises(ValueError, match="latent_dim disagrees"):
        replace(tiny_data, representation=representation)
    with pytest.raises(ValueError, match="must be unique"):
        SplitSpec(strategy="sample", train_values=("D1", "D1"))
    with pytest.raises(ValueError, match="must be disjoint"):
        SplitSpec(strategy="sample", train_values=("D1",), validation_values=("D1",))


def test_representation_scope_rejects_nested_holdout_leakage(tiny_config, tiny_data) -> None:
    settings = tiny_config.recipe_config
    validation = type(settings.validation).model_validate(
        {
            "strategy": "context_group",
            "values": ["D1"],
            "fraction": 0,
            "representation_scope": "nested",
        }
    )
    config = tiny_config.model_copy(
        update={"recipe_config": settings.model_copy(update={"validation": validation})}
    )
    split = _validation_split(tiny_data, config)
    leaked = replace(
        tiny_data.representation,
        fit_scope="training_fold_source",
        included_samples=("D1", "D2"),
    )
    nested_data = replace(tiny_data, representation=leaked)
    with pytest.raises(ValueError, match="held-out representation samples"):
        _representation_scope(nested_data, split, config)


def test_all_checkpoint_representation_is_explicitly_shared(tiny_config, tiny_data) -> None:
    representation = replace(
        tiny_data.representation,
        fit_scope="all_checkpoints",
        included_samples=("D1", "D2"),
        included_time_labels=tuple(tiny_data.axis.labels),
    )
    shared_data = replace(tiny_data, representation=representation)
    split = _validation_split(shared_data, tiny_config)
    assert _representation_scope(shared_data, split, tiny_config) == "shared"


def test_training_split_representation_supports_strict_checkpoint_holdout(
    tiny_config, tiny_data
) -> None:
    settings = tiny_config.recipe_config
    validation = type(settings.validation).model_validate(
        {
            "strategy": "checkpoint",
            "values": ["Stim8hr"],
            "fraction": 0,
            "representation_scope": "nested",
        }
    )
    config = tiny_config.model_copy(
        update={"recipe_config": settings.model_copy(update={"validation": validation})}
    )
    split = _validation_split(tiny_data, config)
    representation = replace(
        tiny_data.representation,
        fit_scope="training_split",
        included_samples=("D1", "D2"),
        included_time_labels=("Rest", "Stim48hr"),
    )
    nested_data = replace(tiny_data, representation=representation)
    assert _representation_scope(nested_data, split, config) == "nested"

    leaked = replace(
        nested_data.representation,
        included_time_labels=("Rest", "Stim8hr", "Stim48hr"),
    )
    with pytest.raises(ValueError, match="held-out representation times"):
        _representation_scope(replace(nested_data, representation=leaked), split, config)


def test_control_residual_is_zero_and_controls_share_reference(tiny_data) -> None:
    model = _model(tiny_data)
    controls = list(tiny_data.control_embedding_ids) * 2
    residual = model.residuals(controls)
    effective = model.effective_embeddings(controls)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(effective, model.reference_embedding.unsqueeze(0).expand_as(effective))
    model.assert_soft_reference()


def test_reference_branch_removes_only_selected_residual(tiny_data) -> None:
    model = _model(tiny_data)
    embedding_ids = ("__control__", "GENE1", "GENE2")
    mask = torch.tensor([False, True, False])
    model.assert_reference_branch(embedding_ids, mask)


def test_absolute_mass_is_invariant_to_logweight_stabilization(tiny_data) -> None:
    model = _model(tiny_data)
    state = sample_initial_particles(tiny_data, tiny_data.measure_ids[:3], 8, seed=11)
    shift = torch.tensor([5.0, -2.0, 1.5])
    stabilized = replace(
        state,
        logw=state.logw + shift[:, None],
        log_m0=state.log_m0 - shift,
    )
    assert torch.allclose(state.absolute_log_weight, stabilized.absolute_log_weight)
    original = model.summarize_context(state.z, state.absolute_log_weight)
    shifted = model.summarize_context(stabilized.z, stabilized.absolute_log_weight)
    assert all(torch.allclose(left, right) for left, right in zip(original, shifted, strict=False))


def test_ecological_context_changes_growth_only(tiny_data) -> None:
    model = _model(tiny_data, context="catalog_bank")
    with torch.no_grad():
        model.payoff_reference_left.zero_()
        model.payoff_reference_right.zero_()
        model.payoff_reference_left[0, 0] = 1.0
        model.payoff_reference_right[0, 0] = 1.0
        model.payoff_residual_left.zero_()
        model.payoff_residual_right.zero_()
    embedding_ids = tuple(tiny_data.embedding_ids[:3])
    z = torch.randn(3, 5, tiny_data.latent_dim)
    empty = torch.zeros(3, model.n_programs)
    shifted = empty.clone()
    shifted[:, 0] = 1.0
    without_context = model(z, torch.tensor(0.5), embedding_ids, empty)
    with_context = model(z, torch.tensor(0.5), embedding_ids, shifted)
    assert torch.equal(without_context.drift, with_context.drift)
    assert torch.equal(without_context.sigma_diag, with_context.sigma_diag)
    assert not torch.equal(without_context.growth, with_context.growth)


def test_mass_rows_require_explicit_denominators(tiny_config, tmp_path) -> None:
    masses = pd.read_parquet(tiny_config.data.masses)
    assert masses["denominator"].str.len().gt(0).all()
    broken_path = tmp_path / "masses.parquet"
    masses.drop(columns="denominator").to_parquet(broken_path, index=False)
    data_config = tiny_config.data.model_copy(update={"masses": broken_path})
    broken_config = tiny_config.model_copy(update={"data": data_config})
    with pytest.raises(ValueError, match="denominator"):
        load_data(broken_config)


def test_dataset_manifest_is_required_and_self_describing(tiny_config, tmp_path) -> None:
    missing = tmp_path / "missing.json"
    data_config = tiny_config.data.model_copy(update={"dataset": missing})
    with pytest.raises(FileNotFoundError, match="Canonical dataset manifest not found"):
        load_data(tiny_config.model_copy(update={"data": data_config}))

    incomplete = tmp_path / "dataset.json"
    incomplete.write_text('{"schema_version": 1}\n', encoding="utf-8")
    data_config = tiny_config.data.model_copy(update={"dataset": incomplete})
    with pytest.raises(ValueError, match="missing required keys"):
        load_data(tiny_config.model_copy(update={"data": data_config}))


def test_effect_axis_config_blocks_reaction_training(tiny_config) -> None:
    raw = tiny_config.model_dump()
    raw["data"]["counts"] = None
    raw["axis"] = {
        "kind": "effect",
        "source": "reference",
        "labels": ("reference", "observed"),
        "values": (0.0, 1.0),
    }
    raw["recipe_config"]["model"]["context"] = "none"
    raw["recipe_config"]["training"]["epochs"] = {"state": 1, "mass": 1, "context": 0}
    raw["recipe_config"]["loss"] = {"mass": 1.0, "count": 0.0}
    with pytest.raises(ValueError, match="Growth and mass fitting"):
        RunConfig.model_validate(raw)


def test_context_phase_requires_a_trained_mass_phase(tiny_config) -> None:
    raw = tiny_config.model_dump()
    raw["recipe_config"]["training"]["epochs"] = {"state": 1, "mass": 0, "context": 1}
    with pytest.raises(ValueError, match="positive mass stage"):
        RunConfig.model_validate(raw)


def test_unit_mass_allows_state_geometry_only(tiny_data, tiny_config) -> None:
    measures = {
        label: {
            measure_id: FiniteMeasure(
                measure.support,
                measure.normalized_weights,
                1.0,
            )
            for measure_id, measure in by_measure.items()
        }
        for label, by_measure in tiny_data.measures.items()
    }
    unit_data = replace(
        tiny_data,
        measures=measures,
        mass_semantics=MassSemantics.UNIT,
        count_blocks=(),
    )
    raw = tiny_config.model_dump()
    raw["data"]["counts"] = None
    raw["recipe_config"]["training"]["epochs"] = {"state": 1, "mass": 1, "context": 0}
    raw["recipe_config"]["loss"] = {"mass": 1.0, "count": 0.0}
    reaction_config = RunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="state geometry training only"):
        validate_run_data(reaction_config, unit_data)


def test_captured_counts_are_diagnostic_not_abundance_claims(tiny_data) -> None:
    captured = replace(tiny_data, mass_semantics=MassSemantics.CAPTURED_COUNT)
    assert captured.claim_policy["relative_abundance"] is False
    assert captured.claim_policy["abundance_claim"] == "diagnostic_only"
    with pytest.raises(ValueError, match="requires informative mass"):
        captured.require_mass_claim("relative expansion")


def test_sparse_target_masks_missing_target_only(tiny_data) -> None:
    missing_id = "D2::GENE2-2"
    assert missing_id in tiny_data.measures["Rest"]
    assert missing_id in tiny_data.measures["Stim8hr"]
    assert missing_id not in tiny_data.measures["Stim48hr"]
    assert len(tiny_data.available_measure_ids("Stim48hr")) == len(tiny_data.measure_ids) - 1


def test_count_blocks_require_complete_denominators(tiny_data) -> None:
    block = tiny_data.count_blocks[0]
    broken = CountBlock(
        context_group_id=block.context_group_id,
        time_label=block.time_label,
        measure_indices=block.measure_indices[:-1],
        exposure=block.exposure[:-1],
        counts=block.counts[:-1],
    )
    with pytest.raises(ValueError, match="every source-supported measure"):
        replace(tiny_data, count_blocks=(broken, *tiny_data.count_blocks[1:]))


def test_effect_axis_blocks_physical_claims_and_counts(tiny_data) -> None:
    effect_axis = Axis(
        kind="effect",
        source="reference",
        labels=("reference", "observed"),
        values=(0.0, 1.0),
    )
    effect = replace(
        tiny_data,
        axis=effect_axis,
        measures={
            "reference": tiny_data.measures["Rest"],
            "observed": tiny_data.measures["Stim8hr"],
        },
        count_blocks=(),
    )
    assert effect.claim_policy["physical_interpolation"] is False
    assert effect.claim_policy["absolute_growth"] is False
    assert effect.claim_policy["abundance_claim"] == "none"
    with pytest.raises(ValueError, match="physical axis"):
        effect.require_mass_claim("growth")
    with pytest.raises(ValueError, match="Count likelihood"):
        replace(effect, count_blocks=tiny_data.count_blocks)


def test_endpoint_is_a_two_checkpoint_trajectory(tiny_data) -> None:
    endpoint = replace(
        tiny_data,
        axis=Axis(
            kind="physical",
            source="Rest",
            labels=("Rest", "Stim8hr"),
            values=(0.0, 8.0),
        ),
        measures={
            "Rest": tiny_data.measures["Rest"],
            "Stim8hr": tiny_data.measures["Stim8hr"],
        },
        count_blocks=(),
    )
    model = _model(endpoint)
    model.set_phase("state")
    state = sample_initial_particles(endpoint, None, 4, seed=2)
    grid = axis_grid(endpoint.axis, 1, device="cpu", dtype=torch.float32)
    result = rollout(model, state, grid, context_provider=NoContextProvider())
    objective = checkpoint_geometry_mass_loss(
        result,
        endpoint,
        mass_weight=1.0,
        include_mass=True,
        validation_source="train_self_eval",
    )
    assert objective.observation_count == len(endpoint.measure_ids)
    assert len(objective.rows) == len(endpoint.measure_ids)


def test_no_context_chunks_equal_the_full_rollout(tiny_data) -> None:
    selected = tiny_data.measure_ids[:3]
    model = _model(tiny_data)
    model.set_phase("state")
    state = sample_initial_particles(tiny_data, selected, 4, seed=19)
    grid = axis_grid(tiny_data.axis, 1, device="cpu", dtype=torch.float32)
    noise = sample_noise(state, grid, seed=20)
    full = rollout(model, state, grid, context_provider=NoContextProvider(), noise=noise)
    terminals = []
    for index in range(len(selected)):
        item = slice(index, index + 1)
        chunk = ParticleState(
            z=state.z[item],
            logw=state.logw[item],
            log_m0=state.log_m0[item],
            measure_ids=(state.measure_ids[index],),
            embedding_ids=(state.embedding_ids[index],),
            context_group_ids=(state.context_group_ids[index],),
            measure_indices=state.measure_indices[item],
            residual_scale=state.residual_scale[item],
        )
        terminals.append(
            rollout(
                model,
                chunk,
                grid,
                context_provider=NoContextProvider(),
                noise=noise[:, item],
            ).terminal_z
        )
    assert torch.allclose(full.terminal_z, torch.cat(terminals, dim=0), atol=1e-6)


def test_common_rollout_rejects_malformed_kernel_output(tiny_data) -> None:
    state = sample_initial_particles(tiny_data, tiny_data.measure_ids[:2], 4, seed=23)
    grid = axis_grid(tiny_data.axis, 1, device="cpu", dtype=torch.float32)

    class BrokenKernel:
        def step(self, *, z, logw, **kwargs):
            del kwargs
            return DynamicsStep(
                drift=torch.zeros_like(z),
                sigma_diag=torch.ones_like(z),
                growth=torch.zeros(logw.shape[0]),
                context=torch.zeros(z.shape[0], 1),
            )

    with pytest.raises(ValueError, match="growth must match"):
        euler_maruyama_rollout(BrokenKernel(), state, grid)

    class BrokenContextKernel:
        def step(self, *, z, logw, **kwargs):
            del kwargs
            return DynamicsStep(
                drift=torch.zeros_like(z),
                sigma_diag=torch.ones_like(z),
                growth=torch.zeros_like(logw),
                context=torch.zeros(z.shape[0] + 1, 1),
            )

    with pytest.raises(ValueError, match="global vector or have one leading row"):
        euler_maruyama_rollout(BrokenContextKernel(), state, grid)

    class GlobalContextKernel:
        def step(self, *, z, logw, **kwargs):
            del kwargs
            return DynamicsStep(
                drift=torch.zeros_like(z),
                sigma_diag=torch.ones_like(z),
                growth=torch.zeros_like(logw),
                context=torch.zeros(3),
            )

    result = euler_maruyama_rollout(GlobalContextKernel(), state, grid)
    assert result.context_steps.shape == (len(grid) - 1, 3)


def test_count_loss_is_finite_and_differentiable(tiny_data) -> None:
    model = _model(tiny_data)
    model.set_phase("mass")
    state = sample_initial_particles(tiny_data, None, 4, seed=29)
    grid = axis_grid(tiny_data.axis, 1, device="cpu", dtype=torch.float32)
    result = rollout(model, state, grid, context_provider=NoContextProvider())
    concentration = torch.tensor(np.log(100.0), requires_grad=True)
    value = count_block_loss(
        result,
        tiny_data,
        log_concentration=concentration,
    )
    value.backward()
    assert torch.isfinite(value)
    assert concentration.grad is not None and torch.isfinite(concentration.grad)


def test_count_loss_uses_only_active_context_groups(tiny_data) -> None:
    model = _model(tiny_data)
    model.set_phase("mass")
    d1_ids = tuple(
        row.measure_id
        for row in tiny_data.measure_meta.itertuples(index=False)
        if row.context_group_id == "D1"
    )
    state = sample_initial_particles(tiny_data, d1_ids, 4, seed=31)
    grid = axis_grid(tiny_data.axis, 1, device="cpu", dtype=torch.float32)
    result = rollout(model, state, grid, context_provider=NoContextProvider())
    concentration = torch.tensor(np.log(100.0))
    original = count_block_loss(
        result,
        tiny_data,
        log_concentration=concentration,
    )
    changed_blocks = []
    for block in tiny_data.count_blocks:
        counts = block.counts.clone()
        if block.context_group_id == "D2":
            counts[0] += 10_000
        changed_blocks.append(
            CountBlock(
                context_group_id=block.context_group_id,
                time_label=block.time_label,
                measure_indices=block.measure_indices,
                exposure=block.exposure,
                counts=counts,
            )
        )
    changed = count_block_loss(
        result,
        replace(tiny_data, count_blocks=tuple(changed_blocks)),
        log_concentration=concentration,
    )
    assert torch.equal(original, changed)


def test_count_only_batch_uses_complete_bank(tiny_data, trained_run) -> None:
    source_only_id = "D2::GENE2-2"
    measures = {
        label: {
            measure_id: measure
            for measure_id, measure in by_measure.items()
            if measure_id != source_only_id or label == tiny_data.axis.source
        }
        for label, by_measure in tiny_data.measures.items()
    }
    count_only_data = replace(tiny_data, measures=measures)
    state = sample_initial_particles(count_only_data, (source_only_id,), 4, seed=37)
    particle_rollout = rollout(
        trained_run.model,
        state,
        trained_run.grid,
        context_provider=CatalogContextProvider(trained_run.bank),
    )
    result = total_objective(
        particle_rollout,
        count_only_data,
        mass_weight=1.0,
        count_weight=0.01,
        include_mass=True,
        log_concentration=trained_run.log_count_concentration,
        fitness_bank=trained_run.bank,
    )
    assert result.checkpoint.observation_count == 0
    assert torch.isfinite(result.count)
    assert torch.isfinite(result.total)


def test_catalog_bank_is_complete_before_optimization(tiny_data, trained_run) -> None:
    empty = CatalogBank.empty(
        tiny_data,
        trained_run.model,
        len(trained_run.grid) - 1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        empty.assert_complete()
    assert trained_run.bank.diagnostics()["bank_seen_fraction"] == 1.0


def test_count_validation_holds_out_complete_context_groups(tiny_data, trained_run) -> None:
    metadata = tiny_data.measure_meta.set_index("measure_id")
    train_groups = set(metadata.loc[list(trained_run.train_measure_ids), "context_group_id"])
    validation_groups = set(
        metadata.loc[list(trained_run.validation_measure_ids), "context_group_id"]
    )
    assert trained_run.validation_source == "held_out"
    assert trained_run.validation_strategy == "context_group_holdout"
    assert train_groups.isdisjoint(validation_groups)


def test_explicit_context_group_and_checkpoint_validation(tiny_config, tiny_data) -> None:
    settings = tiny_config.recipe_config
    validation_type = type(settings.validation)
    donor_validation = validation_type.model_validate(
        {"strategy": "context_group", "values": ["D1"], "fraction": 0}
    )
    donor_split = _validation_split(
        tiny_data,
        tiny_config.model_copy(
            update={"recipe_config": settings.model_copy(update={"validation": donor_validation})}
        ),
    )
    assert donor_split.strategy == "context_group_holdout"
    assert {
        tiny_data.measure_meta.set_index("measure_id").loc[value, "context_group_id"]
        for value in donor_split.validation_measure_ids
    } == {"D1"}

    checkpoint_validation = validation_type.model_validate(
        {"strategy": "checkpoint", "values": ["Stim8hr"], "fraction": 0}
    )
    checkpoint_split = _validation_split(
        tiny_data,
        tiny_config.model_copy(
            update={
                "recipe_config": settings.model_copy(update={"validation": checkpoint_validation})
            }
        ),
    )
    assert checkpoint_split.strategy == "checkpoint_holdout"
    assert checkpoint_split.train_time_labels == ("Stim48hr",)
    assert checkpoint_split.validation_time_labels == ("Stim8hr",)


def test_target_batching_modes_are_deterministic_permutations(trained_run) -> None:
    ids = trained_run.data.measure_ids
    for order in ("target_round_robin", "target_blocked"):
        first = list(trained_run._batches(ids, seed=17, batch_size=5, order=order))
        second = list(trained_run._batches(ids, seed=17, batch_size=5, order=order))
        assert first == second
        assert sorted(value for batch in first for value in batch) == sorted(ids)
    metadata = trained_run.data.measure_meta.set_index("measure_id")
    blocked = list(trained_run._batches(ids, seed=17, batch_size=5, order="target_blocked"))
    batch_by_id = {measure_id: index for index, batch in enumerate(blocked) for measure_id in batch}
    for _, rows in metadata.loc[list(ids)].groupby("embedding_id", observed=True):
        if not rows["is_control"].any():
            assert len({batch_by_id[measure_id] for measure_id in rows.index}) == 1


def test_reference_counterfactual_uses_same_start_and_noise(trained_run, monkeypatch) -> None:
    module = importlib.import_module("credo.counterfactual")
    original_rollout = module.rollout
    calls = []

    def capture(model, initial_state, grid, **kwargs):
        calls.append(
            (
                initial_state.z.clone(),
                initial_state.logw.clone(),
                initial_state.log_m0.clone(),
                kwargs["noise"].clone(),
            )
        )
        return original_rollout(model, initial_state, grid, **kwargs)

    monkeypatch.setattr(module, "rollout", capture)
    result = module.counterfactual(trained_run, "D1::GENE1-1")
    assert len(calls) == 2
    assert torch.equal(calls[0][0], calls[1][0])
    assert torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[0][2], calls[1][2])
    assert torch.equal(calls[0][3], calls[1][3])
    assert result.columns.tolist() == list(COMMON_COUNTERFACTUAL_COLUMNS)
    with pytest.raises(ValueError, match="same_noise=True"):
        module.counterfactual(trained_run, "D1::GENE1-1", same_noise=False)
    control = module.counterfactual(trained_run, "D1::NTC-1", n_particles=4)
    assert control["delta_log_mass"].eq(0).all()
    assert control["mean_shift_l2"].eq(0).all()
    assert control["energy_distance"].eq(0).all()


def test_intrinsic_counterfactual_samples_only_the_focal_measure(trained_run, monkeypatch) -> None:
    module = importlib.import_module("credo.counterfactual")
    original = module.sample_initial_particles
    sampled_ids = []

    def capture(data, measure_ids, n_particles, **kwargs):
        sampled_ids.append(tuple(measure_ids))
        return original(data, measure_ids, n_particles, **kwargs)

    monkeypatch.setattr(module, "sample_initial_particles", capture)
    monkeypatch.setattr(trained_run.model, "context_enabled", False)
    module.counterfactual(trained_run, "D1::GENE1-1", n_particles=4)
    assert sampled_ids == [("D1::GENE1-1",)]
