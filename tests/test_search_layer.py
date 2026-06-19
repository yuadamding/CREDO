from __future__ import annotations

import dataclasses
import math

import pytest

from credo.search import (
    ABLATION_ONLY,
    CREDOTrainOutput,
    CREDOTrialMetrics,
    CREDOTrialSpec,
    DEFAULT_THRESHOLDS,
    FROZEN,
    NoOpReporter,
    RecordingReporter,
    SEARCHABLE,
    TrialPrunedError,
    append_trial_record,
    assert_frozen_semantics,
    claim_grade_thresholds,
    constraints_satisfied,
    hard_constraints,
    load_trial_records,
    metrics_from_history,
    objective_vector,
    pruner_score,
    run_credo_trial,
    spec_to_run_config,
)
from credo.search.manifests import pareto_front, setting_sha256, spec_sha256
from credo.search.objective import DIVERGENCE_PENALTY, feasible_pruner_score


pytestmark = pytest.mark.unit


BRANCH_PARTICLE_OK = {
    "source_ess_frac": 0.4,
    "factual_terminal_ess_frac": 0.4,
    "reference_terminal_ess_frac": 0.4,
    "factual_min_ess_frac_over_time": 0.4,
    "reference_min_ess_frac_over_time": 0.4,
    "factual_max_weight_frac": 0.2,
    "reference_max_weight_frac": 0.2,
    "factual_logw_range": 3.0,
    "reference_logw_range": 3.0,
}


BUILDER_METADATA = {
    "builder_name": "test_builder",
    "builder_version": "1",
    "data_path_hash": "data",
    "mass_table_hash": "mass",
    "split_file_hash": "split",
    "fold_assignment_hash": "folds",
    "latent_source": "pca",
    "latent_key": "X_pca",
    "gene_panel_hash": "genes",
    "normalization_hash": "norm",
    "hvg_preprocessing_hash": "hvg",
    "encoder_checkpoint_hash": "encoder",
    "representation_config_sha256": "rep",
    "dataset_organism": "mouse",
    "gene_symbol_namespace": "MGI",
    "expression_gene_universe_hash": "expr-universe",
    "decoder_gene_panel_hash": "decoder-genes",
    "fold_grid_sha256": "fold-grid",
    "seed_grid": "0,1,2",
    "split_manifest_sha256": "split-manifest",
}
REQUIRED_FOLDS = ("fold0", "fold1", "fold2", "fold3")
REQUIRED_SEEDS = (0, 1, 2)
CLAIM_GRADE_TEST_THRESHOLDS = claim_grade_thresholds(
    control_null_max=0.05,
    log_mass_error_max=0.2,
)
CLAIM_GRADE_GUIDE_TEST_THRESHOLDS = claim_grade_thresholds(
    control_null_max=0.05,
    log_mass_error_max=0.2,
    guide_concordance_max=0.1,
    require_guide_concordance=True,
)


def _claim_grade_output(metrics: CREDOTrialMetrics, **kwargs) -> CREDOTrainOutput:
    return CREDOTrainOutput(metrics=metrics, builder_metadata=BUILDER_METADATA, **kwargs)


def _complete_claim_grade_metrics(**overrides) -> CREDOTrialMetrics:
    payload = {
        "endpoint_geom_mass": 0.2,
        "mass_error_value": 0.05,
        "mass_error_kind": "abs_log_residual",
        "terminal_ess_frac_min": 0.4,
        "min_ess_frac_over_time": 0.35,
        "max_weight_frac_mean": 0.2,
        "converged": True,
        "validation_source": "held_out",
        "control_null_gap": 0.01,
        **BRANCH_PARTICLE_OK,
    }
    payload.update(overrides)
    return CREDOTrialMetrics(**payload)


def _claim_grade_grid_kwargs(**overrides):
    kwargs = {
        "required_folds": REQUIRED_FOLDS,
        "required_seeds": REQUIRED_SEEDS,
    }
    kwargs.update(overrides)
    return kwargs


# --- space ------------------------------------------------------------------

def test_field_classes_partition_tunable_fields() -> None:
    identity = {
        "dataset_kind",
        "claim_type",
        "split_type",
        "data_id",
        "seed",
        "fold_id",
        "latent_dim",
        "latent_source",
        "latent_key",
        "encoder_checkpoint_sha256",
        "representation_config_sha256",
        "gene_panel_sha256",
        "normalization_sha256",
        "hvg_selection_sha256",
        "batch_correction_sha256",
        "input_data_sha256",
        "mass_table_sha256",
        "split_file_sha256",
        "parent_setting_sha256",
        "parent_search_profile",
        "parent_objective_front_id",
    }
    all_fields = {f.name for f in dataclasses.fields(CREDOTrialSpec)}
    classified = set(SEARCHABLE) | set(ABLATION_ONLY) | set(FROZEN)
    # Every spec field is either identity or classified, with no overlaps.
    assert classified | identity == all_fields
    assert not (set(SEARCHABLE) & set(ABLATION_ONLY))
    assert not (set(SEARCHABLE) & set(FROZEN))
    assert not (set(ABLATION_ONLY) & set(FROZEN))


def test_spec_to_run_config_maps_searchable_fields() -> None:
    spec = CREDOTrialSpec(
        hidden_dim=256,
        depth=4,
        embedding_dim=16,
        n_programs=16,
        mediator_dim=16,
        n_particles=64,
        n_steps=8,
        eval_particles=128,
        lr_net=1e-3,
        weight_decay=1e-4,
        lambda_weak=0.5,
        lambda_count=0.0,
        sinkhorn_tau=2.5,
        sinkhorn_epsilon=0.2,
        epochs=42,
        seed=7,
    )
    cfg = spec_to_run_config(spec, output_dir="/tmp/run", latent_dim=12)

    assert cfg.model.hidden_dim == 256
    assert cfg.model.depth == 4
    assert cfg.model.embedding_dim == 16
    assert cfg.simulation.n_particles == 64
    assert cfg.simulation.n_steps == 8
    assert cfg.eval.n_eval_particles == 128
    assert cfg.training.lr_net == pytest.approx(1e-3)
    assert cfg.training.sinkhorn_tau == pytest.approx(2.5)
    assert cfg.training.epochs == 42
    assert cfg.training.seed == 7
    assert cfg.latent.dim == 12
    assert cfg.model.control_mode == "soft_ref"
    # Single-device by construction: search parallelizes across trials.
    assert cfg.multi_gpu_devices == []


def test_assert_frozen_semantics_rejects_mutated_contract() -> None:
    bad_control = CREDOTrialSpec(control_mode="free")
    with pytest.raises(ValueError, match="frozen method semantics"):
        assert_frozen_semantics(bad_control)
    with pytest.raises(ValueError):
        spec_to_run_config(CREDOTrialSpec(same_start_counterfactuals=False))


# --- objective / constraints ------------------------------------------------

def test_constraints_use_intra_trajectory_ess_minimum() -> None:
    spec = CREDOTrialSpec()
    # Terminal ESS is fine, but the run collapsed mid-trajectory below the floor.
    metrics = CREDOTrialMetrics(
        terminal_ess_frac_min=0.25,
        min_ess_frac_over_time=0.05,
        max_weight_frac_mean=0.2,
    )
    constraints = hard_constraints(metrics, spec, DEFAULT_THRESHOLDS)
    assert constraints["ess_ok"] is False
    assert constraints_satisfied(constraints) is False

    healthy = CREDOTrialMetrics(
        terminal_ess_frac_min=0.25,
        min_ess_frac_over_time=0.18,
        max_weight_frac_mean=0.2,
    )
    assert hard_constraints(healthy, spec, DEFAULT_THRESHOLDS)["ess_ok"] is True


def test_heldout_provenance_constraint_and_objective() -> None:
    spec = CREDOTrialSpec()
    self_eval = CREDOTrialMetrics(
        terminal_ess_frac_min=0.3,
        min_ess_frac_over_time=0.3,
        max_weight_frac_mean=0.2,
        heldout_score=0.4,
        validation_source="train_self_eval",
    )
    constraints = hard_constraints(self_eval, spec, DEFAULT_THRESHOLDS)
    assert constraints["heldout_provenance_ok"] is False
    # A self-eval score must not enter the multi-objective vector as generalization.
    assert "heldout_generalization" not in objective_vector(self_eval)

    held_out = dataclasses.replace(self_eval, validation_source="held_out")
    assert hard_constraints(held_out, spec, DEFAULT_THRESHOLDS)["heldout_provenance_ok"] is True
    assert objective_vector(held_out)["heldout_generalization"] == pytest.approx(0.4)


def test_pruner_score_dominated_by_divergence() -> None:
    diverged = CREDOTrialMetrics(endpoint_geom_mass=0.1, diverged=True)
    assert pruner_score(diverged) == pytest.approx(DIVERGENCE_PENALTY)
    spec = CREDOTrialSpec()
    assert feasible_pruner_score(diverged, spec) >= DIVERGENCE_PENALTY


def test_pruner_score_default_weights_none_does_not_crash() -> None:
    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=1.0,
        mass_error_value=0.1,
        mass_error_kind="abs_log_residual",
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    assert math.isfinite(pruner_score(metrics))


# --- metrics extraction -----------------------------------------------------

def test_metrics_from_history_takes_last_epoch_values() -> None:
    history = {
        "loss_end": [1.0, 0.7, 0.5],
        "loss_count": [2.0, 1.5, 1.2],
        "loss_weak": [0.2, 0.1, 0.05],
        "terminal_ess_frac_min": [0.5, 0.4, 0.35],
        "min_ess_frac_mean": [0.45, 0.3, 0.28],
        "max_weight_frac_mean": [0.2, 0.25, 0.27],
        "logw_range_max": [3.0, 4.0, 4.5],
        "validation_source": ["held_out", "held_out", "held_out"],
    }
    m = metrics_from_history(history, wall_seconds=12.0, diverged=False)
    assert m.endpoint_geom_mass == pytest.approx(0.5)
    # Training history has no separate mass term (loss_end is the combined proxy),
    # so relative mass error is only available from an eval summary -> NaN here.
    assert math.isnan(m.log_mass_error)
    assert math.isnan(m.mass_error_value)
    assert m.mass_error_kind == "unknown"
    assert m.count_nll == pytest.approx(1.2)
    assert m.min_ess_frac_over_time == pytest.approx(0.28)
    assert m.validation_source == "held_out"
    assert m.wall_seconds == pytest.approx(12.0)

    # Eval summary (held out) supplies endpoint fit and relative mass error.
    m2 = metrics_from_history(
        history, eval_summary={"mean_endpoint_geom_mass": 0.61, "mean_mass_rel_error": 0.12}
    )
    assert m2.endpoint_geom_mass == pytest.approx(0.61)
    assert m2.log_mass_error == pytest.approx(0.12)
    assert m2.mass_error_value == pytest.approx(0.12)
    assert m2.mass_error_kind == "relative_error"


def test_mean_mass_error_without_explicit_kind_is_not_claim_grade() -> None:
    m = metrics_from_history(
        {"loss_end": [0.5]},
        eval_summary={"mean_endpoint_geom_mass": 0.4, "mean_mass_error": 0.2},
    )
    assert m.mass_error_value == pytest.approx(0.2)
    assert m.log_mass_error == pytest.approx(0.2)
    assert m.mass_error_kind == "unknown"

    explicit = metrics_from_history(
        {"loss_end": [0.5]},
        eval_summary={
            "mean_endpoint_geom_mass": 0.4,
            "mass_error_value": 0.2,
            "mass_error_kind": "abs_log_residual",
        },
    )
    assert explicit.mass_error_kind == "abs_log_residual"


def test_epoch_generic_mass_error_requires_explicit_kind() -> None:
    from credo.search.metrics import metrics_from_epoch

    generic = metrics_from_epoch({"loss_end": 0.5, "loss_total": 0.5, "mass_error": 0.2})
    assert generic.mass_error_value == pytest.approx(0.2)
    assert generic.mass_error_kind == "unknown"

    typed = metrics_from_epoch(
        {
            "loss_end": 0.5,
            "loss_total": 0.5,
            "mass_error": 0.2,
            "mass_error_kind": "abs_log_residual",
        }
    )
    assert typed.mass_error_kind == "abs_log_residual"
    explicit_key = metrics_from_epoch(
        {
            "loss_end": 0.5,
            "loss_total": 0.5,
            "mass_error": 9.0,
            "abs_log_mass_residual": 0.2,
        }
    )
    assert explicit_key.mass_error_value == pytest.approx(0.2)
    assert explicit_key.mass_error_kind == "abs_log_residual"


def test_branch_metric_nan_summary_falls_back_to_history() -> None:
    m = metrics_from_history(
        {
            "loss_end": [0.5],
            "factual_terminal_ess_frac": [0.33],
            "reference_logw_range": [4.5],
        },
        eval_summary={
            "mean_endpoint_geom_mass": 0.4,
            "factual_terminal_ess_frac": float("nan"),
            "reference_logw_range": float("nan"),
        },
    )
    assert m.factual_terminal_ess_frac == pytest.approx(0.33)
    assert m.reference_logw_range == pytest.approx(4.5)


# --- runner -----------------------------------------------------------------

def _good_metrics() -> CREDOTrialMetrics:
    return CREDOTrialMetrics(
        endpoint_geom_mass=0.3,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.3,
        max_weight_frac_mean=0.2,
        heldout_score=0.25,
        validation_source="held_out",
        converged=True,
    )


def test_run_credo_trial_scores_and_constrains() -> None:
    seen = {}

    def train_fn(cfg, spec, reporter):
        seen["lr_net"] = cfg.training.lr_net
        reporter.report(0, _good_metrics())
        return _good_metrics()

    spec = CREDOTrialSpec(lr_net=2e-4)
    reporter = RecordingReporter()
    result = run_credo_trial(
        spec, train_fn=train_fn, output_dir="/tmp/trial0", reporter=reporter
    )

    assert seen["lr_net"] == pytest.approx(2e-4)
    assert result.feasible is True
    # The combined proxy is exposed under its honest name (not "endpoint_geometry").
    assert "endpoint_geom_mass" in result.objective_vector
    assert "endpoint_geometry" not in result.objective_vector
    assert result.run_dir == "/tmp/trial0"
    assert len(reporter.history) == 1


def test_run_credo_trial_propagates_pruning() -> None:
    def train_fn(cfg, spec, reporter):
        reporter.report(0, _good_metrics())
        if reporter.should_prune():
            raise TrialPrunedError(epoch=0)
        return _good_metrics()

    with pytest.raises(TrialPrunedError):
        run_credo_trial(
            CREDOTrialSpec(),
            train_fn=train_fn,
            output_dir="/tmp/trial1",
            reporter=RecordingReporter(prune_after=1),
        )


def test_run_credo_trial_rejects_non_metrics_return() -> None:
    with pytest.raises(TypeError, match="CREDOTrialMetrics"):
        run_credo_trial(
            CREDOTrialSpec(),
            train_fn=lambda cfg, spec, rep: {"loss": 0.1},
            output_dir="/tmp/trial2",
        )


# --- manifest database ------------------------------------------------------

def test_trial_manifest_roundtrip_and_pareto(tmp_path) -> None:
    db = tmp_path / "trials.jsonl"

    def make(geo, mass, ess_min, feasible_seed):
        spec = CREDOTrialSpec(seed=feasible_seed)
        metrics = CREDOTrialMetrics(
            endpoint_geom_mass=geo,
            log_mass_error=mass,
            terminal_ess_frac_min=ess_min,
            min_ess_frac_over_time=ess_min,
            max_weight_frac_mean=0.2,
            converged=True,
        )
        return run_credo_trial(
            spec, train_fn=lambda c, s, r: metrics, output_dir=str(tmp_path / f"r{feasible_seed}")
        )

    r_good = make(0.2, 0.1, 0.4, 1)       # feasible, low objectives
    r_tradeoff = make(0.1, 0.3, 0.4, 2)    # feasible, different trade-off
    r_dominated = make(0.5, 0.5, 0.4, 3)   # feasible but dominated
    r_infeasible = make(0.05, 0.05, 0.02, 4)  # best objectives but ESS-infeasible

    for r in (r_good, r_tradeoff, r_dominated, r_infeasible):
        append_trial_record(db, r)

    records = load_trial_records(db)
    assert len(records) == 4
    assert spec_sha256(r_good.spec) != spec_sha256(r_tradeoff.spec)

    front = pareto_front(
        records,
        ["objective.endpoint_geom_mass", "objective.mass_error"],
        feasible_only=True,
    )
    seeds = {r["spec.seed"] for r in front}
    # The two non-dominated feasible trade-offs are on the front; the dominated
    # one is not, and the infeasible (ESS-collapsed) one is excluded entirely.
    assert seeds == {1, 2}


def test_select_final_candidates_refuses_claim_grade_scalar_ranking(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=0.2,
        mass_error_value=0.05,
        mass_error_kind="abs_log_residual",
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        validation_source="held_out",
        control_null_gap=0.01,
        **BRANCH_PARTICLE_OK,
    )
    result = run_credo_trial(
        CREDOTrialSpec(seed=1, fold_id="fold0"),
        train_fn=lambda c, s, r: _claim_grade_output(metrics),
        output_dir=str(tmp_path / "r"),
        thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
    )
    with pytest.raises(ValueError, match="pruner_score"):
        select_final_candidates([trial_record(result)], profile="claim_grade", sort_by="pruner_score")


def test_select_final_candidates_aggregates_folds_and_seeds(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    def make_record(hidden_dim: int, fold: int, seed: int, geom: float, mass: float):
        metrics = CREDOTrialMetrics(
            endpoint_geom_mass=geom + fold * 0.001 + seed * 0.0001,
            mass_error_value=mass + fold * 0.0001,
            mass_error_kind="abs_log_residual",
            terminal_ess_frac_min=0.4,
            min_ess_frac_over_time=0.35,
            max_weight_frac_mean=0.2,
            converged=True,
            validation_source="held_out",
            control_null_gap=0.01,
            **BRANCH_PARTICLE_OK,
        )
        spec = CREDOTrialSpec(hidden_dim=hidden_dim, fold_id=f"fold{fold}", seed=seed)
        result = run_credo_trial(
            spec,
            train_fn=lambda c, s, r: _claim_grade_output(metrics),
            output_dir=str(tmp_path / f"{hidden_dim}_{fold}_{seed}"),
            thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
        )
        return trial_record(result)

    records = [
        make_record(128, fold, seed, geom=0.20, mass=0.08)
        for fold in range(4)
        for seed in range(3)
    ]
    records += [
        make_record(256, fold, seed, geom=0.25, mass=0.04)
        for fold in range(4)
        for seed in range(3)
    ]

    front = select_final_candidates(
        records,
        profile="claim_grade",
        aggregate_by=("fold_id", "seed"),
        objectives=["endpoint_geom_mass", "mass_error"],
        **_claim_grade_grid_kwargs(),
    )
    assert len(front) == 2
    assert {row["fold_count"] for row in front} == {4}
    assert {row["seed_count"] for row in front} == {3}
    assert all(row["n_trials"] == 12 for row in front)
    assert all("objective.endpoint_geom_mass.se" in row for row in front)

    with pytest.raises(ValueError, match="4 folds x 3 seeds"):
        select_final_candidates(records[:9], profile="claim_grade", **_claim_grade_grid_kwargs())


def test_final_selector_accepts_decomposed_endpoint_objectives_by_default(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = CREDOTrialMetrics(
                endpoint_geom_mass=float("nan"),
                endpoint_sinkhorn=0.2 + fold * 0.001,
                endpoint_mass_penalty=0.03,
                mass_error_value=0.05,
                mass_error_kind="abs_log_residual",
                terminal_ess_frac_min=0.4,
                min_ess_frac_over_time=0.35,
                max_weight_frac_mean=0.2,
                converged=True,
                validation_source="held_out",
                control_null_gap=0.01,
                **BRANCH_PARTICLE_OK,
            )
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    front = select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs())
    assert len(front) == 1
    row = front[0]
    assert "objective.endpoint_geometry.mean" in row
    assert "objective.endpoint_mass_penalty.mean" in row
    assert "objective.endpoint_geom_mass.mean" not in row


def test_final_selector_keeps_geometry_when_mass_penalty_missing(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics(
                endpoint_geom_mass=float("nan"),
                endpoint_sinkhorn=0.2 + fold * 0.001,
                endpoint_mass_penalty=None,
                gpu_seconds=10.0,
            )
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    front = select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs())
    assert len(front) == 1
    assert "objective.endpoint_geometry.mean" in front[0]
    assert "objective.endpoint_mass_penalty.mean" not in front[0]
    assert "objective.endpoint_geom_mass.mean" not in front[0]
    assert "objective.gpu_seconds.mean" not in front[0]


def test_claim_grade_requires_full_fold_seed_grid_not_distinct_counts_only(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    pairs = [(fold, seed) for fold in range(4) for seed in range(3)]
    pairs.remove((3, 2))
    pairs.append((0, 0))
    records = []
    for idx, (fold, seed) in enumerate(pairs):
        metrics = CREDOTrialMetrics(
            endpoint_geom_mass=0.2,
            mass_error_value=0.05,
            mass_error_kind="abs_log_residual",
            terminal_ess_frac_min=0.4,
            min_ess_frac_over_time=0.35,
            max_weight_frac_mean=0.2,
            converged=True,
            validation_source="held_out",
            control_null_gap=0.01,
            **BRANCH_PARTICLE_OK,
        )
        result = run_credo_trial(
            CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
            train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
            output_dir=str(tmp_path / f"{idx}"),
            thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
        )
        records.append(trial_record(result))

    with pytest.raises(ValueError, match="4 folds x 3 seeds"):
        select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs())


def test_claim_grade_uses_explicit_required_fold_seed_grid(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in (0, 1, 7):
            metrics = _complete_claim_grade_metrics()
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    with pytest.raises(ValueError, match="explicit required_folds"):
        select_final_candidates(records, profile="claim_grade")
    assert select_final_candidates(
        records,
        profile="claim_grade",
        required_folds=("fold0", "fold1", "fold2", "fold3"),
        required_seeds=(0, 1, 7),
    )
    with pytest.raises(ValueError, match="4 folds x 3 seeds"):
        select_final_candidates(
            records,
            profile="claim_grade",
            required_folds=("fold0", "fold1", "fold2", "fold3"),
            required_seeds=(0, 1, 2),
        )


def test_guide_claim_requires_guide_concordance_gap(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics()
            result = run_credo_trial(
                CREDOTrialSpec(
                    claim_type="guide_generalization",
                    split_type="guide_holdout",
                    fold_id=f"fold{fold}",
                    seed=seed,
                ),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"missing_{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    assert select_final_candidates(
        records,
        profile="claim_grade",
        require_guide_concordance=False,
        require_finite_thresholds=False,
        **_claim_grade_grid_kwargs(),
    )
    assert select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs()) == []

    complete = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics(guide_concordance_gap=0.02)
            result = run_credo_trial(
                CREDOTrialSpec(
                    claim_type="guide_generalization",
                    split_type="guide_holdout",
                    fold_id=f"fold{fold}",
                    seed=seed,
                ),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"complete_{fold}_{seed}"),
                thresholds=CLAIM_GRADE_GUIDE_TEST_THRESHOLDS,
            )
            complete.append(trial_record(result))
    assert select_final_candidates(complete, profile="claim_grade", **_claim_grade_grid_kwargs())


def test_claim_grade_requires_builder_metadata_by_default(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics(guide_concordance_gap=0.0)
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: metrics,
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    assert select_final_candidates(
        records,
        profile="claim_grade",
        require_builder_metadata=False,
        **_claim_grade_grid_kwargs(),
    )
    assert select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs()) == []


def test_claim_grade_rejects_presence_only_threshold_profiles(tmp_path) -> None:
    from credo.search import CLAIM_GRADE_PRESENCE_THRESHOLDS, select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics(guide_concordance_gap=0.0)
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_PRESENCE_THRESHOLDS,
            )
            records.append(trial_record(result))

    assert select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs()) == []
    assert select_final_candidates(
        records,
        profile="claim_grade",
        require_finite_thresholds=False,
        **_claim_grade_grid_kwargs(),
    )


def test_claim_grade_requires_biology_builder_metadata(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    incomplete_builder = dict(BUILDER_METADATA)
    incomplete_builder.pop("dataset_organism")
    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = _complete_claim_grade_metrics()
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: CREDOTrainOutput(
                    metrics=metrics,
                    builder_metadata=incomplete_builder,
                ),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    assert select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs()) == []


def test_mixed_endpoint_records_use_canonical_default_objective(tmp_path) -> None:
    from credo.search import select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            if seed == 0:
                metrics = _complete_claim_grade_metrics(
                    endpoint_geom_mass=float("nan"),
                    endpoint_sinkhorn=0.20 + fold * 0.001,
                    endpoint_mass_penalty=0.03,
                )
            else:
                metrics = _complete_claim_grade_metrics(endpoint_geom_mass=0.22 + fold * 0.001)
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    front = select_final_candidates(records, profile="claim_grade", **_claim_grade_grid_kwargs())
    assert len(front) == 1
    assert "objective.endpoint_geometry_or_proxy.mean" in front[0]
    assert "objective.endpoint_geometry.mean" not in front[0]
    legacy_records = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "objective.endpoint_geometry_or_proxy",
                "objective.endpoint_metric_kind",
                "objective.endpoint_mass_penalty_or_nan",
            }
        }
        for record in records
    ]
    assert "objective.endpoint_geometry_or_proxy.mean" in select_final_candidates(
        legacy_records, profile="claim_grade", **_claim_grade_grid_kwargs()
    )[0]


def test_final_selector_reports_attempted_and_feasible_trial_counts(tmp_path) -> None:
    from credo.search import CREDOTrainOutput, select_final_candidates
    from credo.search.manifests import trial_record

    records = []
    for fold in range(4):
        for seed in range(3):
            metrics = CREDOTrialMetrics(
                endpoint_geom_mass=0.2,
                mass_error_value=0.05,
                mass_error_kind="abs_log_residual",
                terminal_ess_frac_min=0.4,
                min_ess_frac_over_time=0.35,
                max_weight_frac_mean=0.2,
                converged=True,
                validation_source="held_out",
                control_null_gap=0.01,
                **BRANCH_PARTICLE_OK,
            )
            result = run_credo_trial(
                CREDOTrialSpec(fold_id=f"fold{fold}", seed=seed),
                train_fn=lambda c, s, r, metrics=metrics: _claim_grade_output(metrics),
                output_dir=str(tmp_path / f"{fold}_{seed}"),
                thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
            )
            records.append(trial_record(result))

    failed = run_credo_trial(
        CREDOTrialSpec(fold_id="fold0", seed=0),
        train_fn=lambda c, s, r: _claim_grade_output(
            CREDOTrialMetrics(
                endpoint_geom_mass=0.2,
                mass_error_value=0.05,
                mass_error_kind="abs_log_residual",
                terminal_ess_frac_min=0.4,
                min_ess_frac_over_time=0.35,
                max_weight_frac_mean=0.2,
                converged=True,
                validation_source="held_out",
                control_null_gap=0.01,
                **BRANCH_PARTICLE_OK,
            ),
            failure_type="RuntimeError",
        ),
        output_dir=str(tmp_path / "failed"),
        thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
    )
    records.append(trial_record(failed))

    front = select_final_candidates(
        records,
        profile="claim_grade",
        objectives=["endpoint_geom_mass", "mass_error"],
        **_claim_grade_grid_kwargs(),
    )
    assert len(front) == 1
    assert front[0]["n_trials"] == 12
    assert front[0]["n_trials_total"] == 13
    assert front[0]["n_trials_feasible"] == 12
    assert front[0]["worst_source_ess_frac"] == pytest.approx(0.4)
    assert front[0]["worst_factual_max_weight_frac"] == pytest.approx(0.2)
    assert front[0]["worst_reference_logw_range"] == pytest.approx(3.0)


def test_particle_step_convergence_diagnostics_gate_fidelities() -> None:
    from credo.search import (
        ConvergenceThresholds,
        FidelityRecord,
        claim_grade_convergence_thresholds,
        estimate_convergence_thresholds_from_pilot,
        particle_step_convergence_diagnostics,
    )

    thresholds = ConvergenceThresholds(endpoint_drift_median_max=0.05, top_k=2)
    records = [
        FidelityRecord(
            n_particles=512,
            n_steps=24,
            delta_log_mass_by_id={"a": 0.1, "b": 0.3, "c": -0.2},
            endpoint_metric_by_id={"a": 0.5, "b": 0.6, "c": 0.7},
            biology_axis_score_by_axis={"tnf": 0.5, "cis": -0.2},
            top_hit_scores={"a": 0.1, "b": 0.9, "c": 0.8},
            ess_min=0.30,
        ),
        FidelityRecord(
            n_particles=1024,
            n_steps=48,
            delta_log_mass_by_id={"a": 0.2, "b": 0.4, "c": -0.1},
            endpoint_metric_by_id={"a": 0.51, "b": 0.61, "c": 0.70},
            biology_axis_score_by_axis={"tnf": 0.6, "cis": -0.1},
            top_hit_scores={"a": 0.2, "b": 0.95, "c": 0.85},
            ess_min=0.25,
        ),
        FidelityRecord(
            n_particles=2048,
            n_steps=96,
            delta_log_mass_by_id={"a": 0.3, "b": 0.5, "c": -0.05},
            endpoint_metric_by_id={"a": 0.52, "b": 0.60, "c": 0.69},
            biology_axis_score_by_axis={"tnf": 0.7, "cis": -0.3},
            top_hit_scores={"a": 0.3, "b": 0.99, "c": 0.80},
            ess_min=0.20,
        ),
    ]
    summary = particle_step_convergence_diagnostics(records, thresholds=thresholds)
    assert summary["passes"] is True
    assert summary["reference_n_particles"] == 2048
    assert summary["reference_n_steps"] == 96
    assert summary["rank_correlation_delta_log_mass_min"] >= 0.9
    assert summary["top_hit_jaccard_min"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="endpoint_drift_median_max"):
        claim_grade_convergence_thresholds(endpoint_drift_median_max=math.inf)
    claim_thresholds = claim_grade_convergence_thresholds(endpoint_drift_median_max=0.05)
    assert claim_thresholds.endpoint_drift_median_max == pytest.approx(0.05)
    calibrated = estimate_convergence_thresholds_from_pilot(
        between_perturbation_endpoint_distances=[1.0, 2.0, 3.0],
        within_perturbation_fold_endpoint_distances=[0.2, 0.4, 0.6],
    )
    assert calibrated.endpoint_drift_median_max == pytest.approx(0.1)

    unstable = dataclasses.replace(records[-1], biology_axis_score_by_axis={"tnf": -0.7, "cis": 0.3})
    failed = particle_step_convergence_diagnostics(records[:-1] + [unstable], thresholds=thresholds)
    assert failed["passes"] is False
    assert "biology_axis_sign_stability" in failed["failed_gates"]


def test_null_suite_baseline_and_biology_axis_helpers() -> None:
    from credo.search import (
        BiologyAxisSpec,
        DEFAULT_BIOLOGY_AXES,
        baseline_export_manifest,
        baseline_export_record,
        evaluate_biology_axis_gates,
        required_baselines_for_claim,
        summarize_null_suite,
    )

    nulls = summarize_null_suite(
        {"control_null": 2.0, "guide_shuffle": 0.1},
        {
            "control_null": [0.1, 0.2, 0.3, 0.4],
            "guide_shuffle": [0.2, 0.3, 0.4, 0.5],
        },
    )
    assert nulls["control_null"]["empirical_p"] == pytest.approx(0.2)
    assert nulls["control_null"]["fdr"] <= 0.4
    assert nulls["guide_shuffle"]["null_gap_p95"] == pytest.approx(0.485)

    credo = baseline_export_record(
        "credo_endpoint_proxy",
        artifact_path="/tmp/credo.csv",
        metrics={"endpoint_geom_mass": 0.2},
        status="available",
    )
    with pytest.raises(ValueError, match="Unknown baseline_kind"):
        baseline_export_record("not_a_baseline")
    manifest = baseline_export_manifest([credo], required=("credo_endpoint_proxy", "moscot_time"))
    assert manifest["complete"] is False
    assert manifest["required_missing"] == ["moscot_time"]
    assert required_baselines_for_claim("biology") == ("credo_endpoint_proxy",)
    assert "wfr_mfm_unbalanced_transport" in required_baselines_for_claim("method_superiority")
    default_axis_by_name = {axis.name: axis for axis in DEFAULT_BIOLOGY_AXES}
    assert default_axis_by_name["renz_expansion_anchor_perturbations"].axis_kind == "perturbation_anchor"
    assert default_axis_by_name["renz_expansion_anchor_perturbations"].null_alternative == "greater"
    assert default_axis_by_name["renz_tnf_ap1_nfkb_expression_module"].axis_kind == "expression_module"
    assert "Notch1" not in default_axis_by_name["renz_tnf_ap1_nfkb_expression_module"].markers

    axes = (
        BiologyAxisSpec(
            "tnf",
            ("Notch1",),
            expected_direction="positive",
            organism="mouse",
            gene_symbol_namespace="MGI",
            module_source="Renz",
            min_coverage=0.7,
        ),
        BiologyAxisSpec("artifact", ("gene_size",), expected_direction="negative"),
    )
    gates = evaluate_biology_axis_gates(
        {"tnf": 1.2, "artifact": 0.4},
        null_summaries={"tnf": {"empirical_p": 0.01, "fdr": 0.02}},
        coverage_by_axis={"tnf": 0.5},
        axes=axes,
        score_abs_min=0.1,
    )
    assert gates["tnf"]["pass"] is False
    assert gates["tnf"]["organism"] == "mouse"
    assert gates["tnf"]["axis_kind"] == "expression_module"
    assert gates["tnf"]["null_alternative"] == "two_sided"
    assert gates["tnf"]["gene_symbol_namespace"] == "MGI"
    assert "marker_coverage" in gates["tnf"]["failed_gates"]
    assert gates["artifact"]["pass"] is False
    assert "wrong_direction" in gates["artifact"]["failed_gates"]

    missing_coverage = evaluate_biology_axis_gates(
        {"tnf": 1.2},
        null_summaries={"tnf": {"empirical_p": 0.01, "fdr": 0.02}},
        axes=axes[:1],
    )
    assert "missing_marker_coverage" in missing_coverage["tnf"]["failed_gates"]

    mismatch = evaluate_biology_axis_gates(
        {"tnf": 1.2},
        null_summaries={"tnf": {"empirical_p": 0.01, "fdr": 0.02}},
        coverage_by_axis={"tnf": 1.0},
        axes=axes[:1],
        dataset_organism="human",
    )
    assert "organism_mismatch" in mismatch["tnf"]["failed_gates"]
    mapped = evaluate_biology_axis_gates(
        {"tnf": 1.2},
        null_summaries={"tnf": {"empirical_p": 0.01, "fdr": 0.02}},
        coverage_by_axis={"tnf": 1.0},
        axes=axes[:1],
        dataset_organism="human",
        homolog_mapped_axes=("tnf",),
    )
    assert "organism_mismatch" not in mapped["tnf"]["failed_gates"]


def test_problem_builder_registry_builds_from_config_without_namespace() -> None:
    from types import SimpleNamespace

    from credo.search import (
        ProblemBuilderMetadata,
        ProblemBuilderRegistry,
        build_endpoint_problem_from_config,
        build_single_time_problem_from_config,
        problem_builder_metadata,
    )

    registry = ProblemBuilderRegistry()
    metadata = ProblemBuilderMetadata(
        builder_name="tiny_builder",
        builder_version="1",
        data_path_hash="data",
        mass_table_hash="mass",
        split_file_hash="split",
        fold_assignment_hash="folds",
        latent_source="pca",
        latent_key="X_pca",
        gene_panel_hash="genes",
        normalization_hash="norm",
        hvg_preprocessing_hash="hvg",
        encoder_checkpoint_hash="encoder",
        representation_config_sha256="rep",
        dataset_organism="mouse",
        gene_symbol_namespace="MGI",
        expression_gene_universe_hash="expr-universe",
        decoder_gene_panel_hash="decoder-genes",
        fold_grid_sha256="fold-grid",
        seed_grid="0,1,2",
        split_manifest_sha256="split-manifest",
    )
    registry.register(
        "endpoint",
        "dataset-a",
        lambda cfg: {"kind": "endpoint", "data_id": cfg.data_id},
        metadata=metadata,
    )
    cfg = SimpleNamespace(data_id="dataset-a")

    assert build_endpoint_problem_from_config(cfg, registry=registry) == {
        "kind": "endpoint",
        "data_id": "dataset-a",
    }
    record = problem_builder_metadata("endpoint", cfg, registry=registry).to_record()
    assert record["builder_name"] == "tiny_builder"
    assert record["mass_table_hash"] == "mass"
    assert record["dataset_organism"] == "mouse"
    with pytest.raises(KeyError, match="single_time"):
        build_single_time_problem_from_config(cfg, registry=registry)
    with pytest.raises(ValueError, match="data_id"):
        build_endpoint_problem_from_config(SimpleNamespace(data_id=None), registry=registry)


def test_trial_record_persists_builder_metadata_and_hashes_identity(tmp_path) -> None:
    from credo.search.manifests import setting_sha256, trial_record

    spec = CREDOTrialSpec(fold_id="fold0", seed=1)
    metrics = _complete_claim_grade_metrics()
    result = run_credo_trial(
        spec,
        train_fn=lambda c, s, r: _claim_grade_output(metrics),
        output_dir=str(tmp_path / "r"),
        thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
    )
    record = trial_record(result)

    assert record["builder.builder_name"] == "test_builder"
    assert record["builder.fold_assignment_hash"] == "folds"
    assert record["builder.representation_config_sha256"] == "rep"
    assert record["builder.dataset_organism"] == "mouse"
    assert record["builder.expression_gene_universe_hash"] == "expr-universe"
    assert record["constraints.threshold_profile"] == "claim_grade_finite"
    assert record["constraints.control_null_max"] == pytest.approx(0.05)
    assert record["constraints.log_mass_error_max"] == pytest.approx(0.2)
    assert record["constraints.thresholds_sha256"]
    assert record["setting_sha256"] == setting_sha256(spec, BUILDER_METADATA)
    changed = dict(BUILDER_METADATA, gene_panel_hash="other-genes")
    assert setting_sha256(spec, BUILDER_METADATA) != setting_sha256(spec, changed)

    bare_result = run_credo_trial(
        spec,
        train_fn=lambda c, s, r: metrics,
        output_dir=str(tmp_path / "bare"),
        builder_metadata=BUILDER_METADATA,
        thresholds=CLAIM_GRADE_TEST_THRESHOLDS,
    )
    bare_record = trial_record(bare_result)
    assert bare_record["builder.builder_name"] == "test_builder"


def test_metrics_validation_source_prefers_eval_summary() -> None:
    # A stale history label must not override a fresh held-out eval summary.
    history = {"loss_end": [0.5], "validation_source": ["train_self_eval"]}
    summary = {"mean_endpoint_geom_mass": 0.4, "validation_source": "held_out"}
    m = metrics_from_history(history, eval_summary=summary)
    assert m.endpoint_geom_mass == pytest.approx(0.4)
    assert m.validation_source == "held_out"


def test_pruner_penalizes_missing_endpoint_metric() -> None:
    good = CREDOTrialMetrics(
        endpoint_geom_mass=0.3,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    missing = dataclasses.replace(good, endpoint_geom_mass=float("nan"))
    # Missing core fit metric is a large penalty, not a neutral zero.
    assert pruner_score(missing) > pruner_score(good) + 100.0


def test_hard_constraints_reject_nan_fit_metrics() -> None:
    spec = CREDOTrialSpec()
    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=float("nan"),
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    constraints = hard_constraints(metrics, spec, DEFAULT_THRESHOLDS)
    assert constraints["fit_metrics_finite"] is False
    assert constraints_satisfied(constraints) is False


def test_objective_uses_pure_geometry_only_when_decomposed() -> None:
    decomposed = CREDOTrialMetrics(
        endpoint_geom_mass=0.5,
        endpoint_sinkhorn=0.3,
        endpoint_mass_penalty=0.2,
        log_mass_error=0.1,
    )
    vector = objective_vector(decomposed)
    assert vector["endpoint_geometry"] == pytest.approx(0.3)
    assert vector["endpoint_geometry_or_proxy"] == pytest.approx(0.3)
    assert vector["endpoint_mass_penalty"] == pytest.approx(0.2)
    assert "endpoint_geom_mass" not in vector


def test_suggest_spec_allows_base_with_searched_key() -> None:
    from credo.search.schedulers import suggest_spec

    class _FakeTrial:
        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

    # hidden_dim is a searched key; passing it in base must not crash.
    base = {"data_id": "d", "seed": 1, "hidden_dim": 9999}
    spec = suggest_spec(_FakeTrial(), base)
    assert spec.hidden_dim == 128  # suggested value wins
    assert spec.data_id == "d"


def test_suggest_spec_samples_lambda_count_from_mixture() -> None:
    from credo.search.schedulers import suggest_spec

    class _FakeTrial:
        def __init__(self, regime):
            self.regime = regime
            self.requested = []

        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, log=False):
            self.requested.append((name, low, high, log))
            if name == "lambda_count_regime":
                return self.regime
            return low

    zero = _FakeTrial(0.10)
    low = _FakeTrial(0.50)
    high = _FakeTrial(0.95)

    assert suggest_spec(zero, {}).lambda_count == pytest.approx(0.0)
    assert suggest_spec(low, {}).lambda_count == pytest.approx(1e-3)
    assert suggest_spec(high, {}).lambda_count == pytest.approx(1.0)
    assert not any(name == "lambda_count_low" for name, *_ in zero.requested)
    assert any(name == "lambda_count_low" for name, *_ in low.requested)
    assert any(name == "lambda_count_high" for name, *_ in high.requested)


def test_scheduler_profiles_use_separate_fidelity_ranges() -> None:
    from credo.search.schedulers import (
        suggest_ablation_spec,
        suggest_claim_grade_refit_spec,
        suggest_pareto_refit_spec,
    )

    class _FakeTrial:
        def suggest_categorical(self, name, choices):
            return choices[-1]

        def suggest_int(self, name, low, high):
            return high

        def suggest_float(self, name, low, high, log=False):
            if name == "lambda_count_regime":
                return 0.1
            return low

    pareto = suggest_pareto_refit_spec(_FakeTrial(), {"data_id": "d"})
    claim = suggest_claim_grade_refit_spec(
        _FakeTrial(),
        {
            "data_id": "d",
            "hidden_dim": 256,
            "depth": 3,
            "embedding_dim": 16,
            "n_programs": 16,
            "mediator_dim": 16,
            "parent_setting_sha256": "parent",
            "parent_search_profile": "pareto_refit",
            "parent_objective_front_id": "front",
        },
    )
    ablation = suggest_ablation_spec(_FakeTrial(), {"data_id": "d"})

    assert pareto.n_particles == 1024
    assert pareto.eval_particles == 2048
    assert claim.n_particles == 2048
    assert claim.eval_particles == 4096
    assert claim.parent_setting_sha256 == "parent"
    assert ablation.context_kind == "causal_attention"
    assert ablation.ecological_growth is False
    with pytest.raises(ValueError, match="selected architecture"):
        suggest_claim_grade_refit_spec(_FakeTrial(), {"data_id": "d"})
    with pytest.raises(ValueError, match="parent_search_profile"):
        suggest_claim_grade_refit_spec(
            _FakeTrial(),
            {
                "data_id": "d",
                "hidden_dim": 256,
                "depth": 3,
                "embedding_dim": 16,
                "n_programs": 16,
                "mediator_dim": 16,
                "parent_setting_sha256": "parent",
                "parent_search_profile": "light_screen",
                "parent_objective_front_id": "front",
            },
        )


def test_spec_to_run_config_sets_latent_dim_via_construction() -> None:
    cfg = spec_to_run_config(CREDOTrialSpec(), output_dir="x", latent_dim=9)
    assert cfg.latent.dim == 9


def test_spec_post_init_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        CREDOTrialSpec(hidden_dim=0)
    with pytest.raises(ValueError, match="lr_net"):
        CREDOTrialSpec(lr_net=0.0)
    with pytest.raises(ValueError, match="lambda_weak"):
        CREDOTrialSpec(lambda_weak=-1.0)


def test_write_trial_dir_is_parallel_safe_source_of_truth(tmp_path) -> None:
    from credo.search import reduce_trial_dirs, write_trial_dir

    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=0.2,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    result = run_credo_trial(
        CREDOTrialSpec(seed=5), train_fn=lambda c, s, r: metrics, output_dir=str(tmp_path / "r")
    )
    trial_dir = write_trial_dir(tmp_path / "trials", result, index=0)
    assert (trial_dir / "result.json").exists()

    jsonl = reduce_trial_dirs(tmp_path / "trials", tmp_path / "all.jsonl")
    records = load_trial_records(jsonl)
    assert len(records) == 1
    assert records[0]["spec.seed"] == 5


def test_run_credo_trial_translates_trainer_prune_signal() -> None:
    # A trainer raises a duck-typed pruned exception (marker _credo_pruned) so
    # credo.training need not import credo.search; run_credo_trial must translate
    # it into TrialPrunedError (a pruned trial, not a short completed one).
    class _TrainerPruned(RuntimeError):
        _credo_pruned = True
        epoch = 3

    def train_fn(cfg, spec, reporter):
        raise _TrainerPruned()

    with pytest.raises(TrialPrunedError):
        run_credo_trial(CREDOTrialSpec(), train_fn=train_fn, output_dir="/tmp/tp")


def test_metrics_from_epoch_prefers_validation_endpoint_loss() -> None:
    from credo.search.metrics import metrics_from_epoch

    m = metrics_from_epoch(
        {
            "loss_end": 10.0,
            "val_endpoint_loss": 1.0,
            "validation_source": "held_out",
            "loss_total": 10.0,
        }
    )
    # Pruning ranks on the held-out endpoint loss, not the training loss.
    assert m.endpoint_geom_mass == pytest.approx(1.0)
    assert m.train_endpoint_geom_mass == pytest.approx(10.0)
    assert m.validation_source == "held_out"

    # Falls back to training loss when no validation signal is present.
    m2 = metrics_from_epoch({"loss_end": 5.0, "loss_total": 5.0})
    assert m2.endpoint_geom_mass == pytest.approx(5.0)


def test_claim_grade_constraints_require_mass_and_diagnostics() -> None:
    from credo.search import CLAIM_GRADE_PRESENCE_THRESHOLDS

    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    # Missing mass + missing diagnostics: feasible for screening, NOT claim-grade.
    sparse = CREDOTrialMetrics(**base)
    assert constraints_satisfied(hard_constraints(sparse, spec, DEFAULT_THRESHOLDS)) is True

    cg = hard_constraints(sparse, spec, CLAIM_GRADE_PRESENCE_THRESHOLDS)
    assert cg["mass_metric_finite"] is False
    assert cg["mass_error_kind_ok"] is False
    assert cg["control_null_ok"] is False
    assert cg["guide_concordance_ok"] is False
    assert cg["factual_terminal_ess_ok"] is False
    assert constraints_satisfied(cg) is False

    # A fully-specified claim-grade trial passes.
    complete = CREDOTrialMetrics(
        **base,
        mass_error_value=0.1,
        mass_error_kind="abs_log_residual",
        control_null_gap=0.0,
        guide_concordance_gap=0.0,
        heldout_score=0.2,
        validation_source="held_out",
        **BRANCH_PARTICLE_OK,
    )
    assert constraints_satisfied(hard_constraints(complete, spec, CLAIM_GRADE_PRESENCE_THRESHOLDS)) is True


def test_claim_grade_thresholds_alias_is_not_open_ceiling_gate() -> None:
    from credo.search import CLAIM_GRADE_THRESHOLDS

    assert CLAIM_GRADE_THRESHOLDS is None
    with pytest.raises(ValueError, match="CLAIM_GRADE_THRESHOLDS"):
        hard_constraints(CREDOTrialMetrics(), CREDOTrialSpec(), CLAIM_GRADE_THRESHOLDS)


def test_suggest_spec_rejects_unknown_base_key() -> None:
    from credo.search.schedulers import suggest_spec

    class _FakeTrial:
        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, log=False):
            return low

    with pytest.raises(ValueError, match="Unknown CREDOTrialSpec base keys"):
        suggest_spec(_FakeTrial(), {"not_a_real_field": 1})


def test_write_trial_dir_is_unique_per_trial_id(tmp_path) -> None:
    from credo.search import reduce_trial_dirs, write_trial_dir

    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=0.2,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    # Same spec, two trials (e.g. reruns) must not overwrite each other.
    result = run_credo_trial(
        CREDOTrialSpec(seed=1), train_fn=lambda c, s, r: metrics, output_dir=str(tmp_path / "r")
    )
    d1 = write_trial_dir(tmp_path / "trials", result, trial_id="a")
    d2 = write_trial_dir(tmp_path / "trials", result, trial_id="b")
    assert d1 != d2 and d1.exists() and d2.exists()

    records = load_trial_records(reduce_trial_dirs(tmp_path / "trials", tmp_path / "all.jsonl"))
    assert len(records) == 2
    assert records[0]["schema_version"] == "credo.search.v2"


def test_claim_grade_requires_heldout_endpoint() -> None:
    from credo.search import claim_grade_thresholds

    th = claim_grade_thresholds(control_null_max=0.05, log_mass_error_max=0.2)
    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        mass_error_value=0.1,
        mass_error_kind="abs_log_residual",
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        control_null_gap=0.0,
        guide_concordance_gap=0.0,
        **BRANCH_PARTICLE_OK,
    )
    self_eval = CREDOTrialMetrics(**base, validation_source="train_self_eval")
    cg = hard_constraints(self_eval, spec, th)
    assert cg["heldout_endpoint_ok"] is False
    assert constraints_satisfied(cg) is False

    held_out = CREDOTrialMetrics(**base, validation_source="held_out")
    assert constraints_satisfied(hard_constraints(held_out, spec, th)) is True


def test_claim_grade_thresholds_factory_enforces_finite_ceilings() -> None:
    from credo.search import claim_grade_thresholds

    th = claim_grade_thresholds(
        control_null_max=0.05,
        log_mass_error_max=0.2,
        guide_concordance_max=0.10,
        require_guide_concordance=True,
    )
    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        mass_error_value=0.1,
        mass_error_kind="abs_log_residual",
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        validation_source="held_out",
        **BRANCH_PARTICLE_OK,
    )
    big_gap = CREDOTrialMetrics(**base, control_null_gap=0.20, guide_concordance_gap=0.0)
    c = hard_constraints(big_gap, spec, th)
    assert c["control_null_ok"] is False
    assert constraints_satisfied(c) is False

    within = CREDOTrialMetrics(**base, control_null_gap=0.01, guide_concordance_gap=0.05)
    assert constraints_satisfied(hard_constraints(within, spec, th)) is True


def test_latent_dim_is_part_of_spec_hash() -> None:
    assert spec_sha256(CREDOTrialSpec(latent_dim=16)) != spec_sha256(CREDOTrialSpec(latent_dim=32))
    cfg = spec_to_run_config(CREDOTrialSpec(latent_dim=20), output_dir="x")
    assert cfg.latent.dim == 20


def test_representation_and_claim_identity_are_hashed_separately_from_seed_fold() -> None:
    base = CREDOTrialSpec(
        data_id="dataset-a",
        claim_type="guide_generalization",
        split_type="guide_holdout",
        latent_dim=16,
        latent_source="pca",
        latent_key="X_pca",
        representation_config_sha256="rep1",
        input_data_sha256="input1",
        mass_table_sha256="mass1",
        split_file_sha256="split1",
        seed=1,
        fold_id="fold0",
    )
    same_setting = dataclasses.replace(base, seed=2, fold_id="fold1")
    different_representation = dataclasses.replace(base, representation_config_sha256="rep2")
    different_parent = dataclasses.replace(base, parent_setting_sha256="parent-a")

    assert spec_sha256(base) != spec_sha256(same_setting)
    assert setting_sha256(base) == setting_sha256(same_setting)
    assert setting_sha256(base) != setting_sha256(different_representation)
    assert setting_sha256(base) != setting_sha256(different_parent)


def test_run_credo_trial_propagates_train_output_provenance() -> None:
    from credo.search import CREDOTrainOutput

    def train_fn(cfg, spec, reporter):
        return CREDOTrainOutput(
            metrics=_good_metrics(),
            run_dir="/tmp/rd",
            checkpoint_path="/tmp/rd/best.pt",
            eval_summary_path="/tmp/rd/eval.json",
        )

    result = run_credo_trial(CREDOTrialSpec(), train_fn=train_fn, output_dir="/tmp/x")
    assert result.checkpoint_path == "/tmp/rd/best.pt"
    assert result.run_dir == "/tmp/rd"
    assert result.eval_summary_path == "/tmp/rd/eval.json"
    assert result.feasible is True


def test_metrics_from_history_records_train_endpoint_with_eval_summary() -> None:
    history = {"loss_end": [1.0, 0.7, 0.5]}
    m = metrics_from_history(history, eval_summary={"mean_endpoint_geom_mass": 0.61})
    assert m.endpoint_geom_mass == pytest.approx(0.61)  # held-out for selection
    assert m.train_endpoint_geom_mass == pytest.approx(0.5)  # training value retained


def test_summary_nan_metrics_are_sanitized() -> None:
    m = metrics_from_history(
        {"loss_end": [0.5]},
        eval_summary={
            "mean_endpoint_geom_mass": 0.5,
            "mean_count_nll": float("nan"),
            "mean_endpoint_sinkhorn": float("nan"),
        },
    )
    assert m.count_nll is None
    assert m.endpoint_sinkhorn is None
    # No NaN leaks into the objective vector.
    assert all(v == v for v in objective_vector(m).values())


def test_summary_source_does_not_label_training_endpoint_as_heldout() -> None:
    from credo.search import CLAIM_GRADE_PRESENCE_THRESHOLDS

    # The eval summary carries a validation_source but no endpoint metric, so the
    # endpoint comes from training history and must NOT inherit "held_out".
    m = metrics_from_history({"loss_end": [0.5]}, eval_summary={"validation_source": "held_out"})
    assert m.endpoint_geom_mass == pytest.approx(0.5)
    assert m.validation_source == "train_self_eval"
    constraints = hard_constraints(m, CREDOTrialSpec(), CLAIM_GRADE_PRESENCE_THRESHOLDS)
    assert constraints["heldout_endpoint_ok"] is False


def test_run_credo_trial_reconciles_latent_dim_into_spec_hash() -> None:
    train_fn = lambda c, s, r: _good_metrics()  # noqa: E731
    # Conflicting override is an error, not a silent change.
    with pytest.raises(ValueError, match="conflicts with spec.latent_dim"):
        run_credo_trial(
            CREDOTrialSpec(latent_dim=16), train_fn=train_fn, output_dir="/tmp/x", latent_dim=32
        )
    # A bare override is folded into the (hashed) spec.
    result = run_credo_trial(CREDOTrialSpec(), train_fn=train_fn, output_dir="/tmp/x", latent_dim=24)
    assert result.spec.latent_dim == 24


def test_claim_grade_thresholds_reject_infinite_ceilings() -> None:
    from credo.search import claim_grade_thresholds

    with pytest.raises(ValueError, match="control_null_max"):
        claim_grade_thresholds(control_null_max=math.inf, log_mass_error_max=0.2)
    with pytest.raises(ValueError, match="log_mass_error_max"):
        claim_grade_thresholds(control_null_max=0.05)
    with pytest.raises(ValueError, match="guide_concordance_max"):
        claim_grade_thresholds(
            control_null_max=0.05,
            log_mass_error_max=0.2,
            require_guide_concordance=True,
        )
    th = claim_grade_thresholds(control_null_max=0.05, log_mass_error_max=0.2)
    assert th.control_null_max == pytest.approx(0.05)
    assert th.log_mass_error_max == pytest.approx(0.2)


def test_trial_record_includes_provenance_fields() -> None:
    from credo.search import CREDOTrainOutput
    from credo.search.manifests import trial_record

    def train_fn(cfg, spec, reporter):
        return CREDOTrainOutput(
            metrics=_good_metrics(),
            run_dir="/tmp/rd",
            checkpoint_path="/tmp/rd/ck.pt",
            history_path="/tmp/rd/h.csv",
            eval_summary_path="/tmp/rd/e.json",
            resolved_config_path="/tmp/rd/c.json",
        )

    record = trial_record(run_credo_trial(CREDOTrialSpec(), train_fn=train_fn, output_dir="/tmp/x"))
    for key in ("history_path", "eval_summary_path", "resolved_config_path", "failure_type", "failure_message"):
        assert key in record
    assert record["history_path"] == "/tmp/rd/h.csv"


def test_failure_metadata_makes_trial_infeasible() -> None:
    from credo.search import CREDOTrainOutput

    def train_fn(cfg, spec, reporter):
        return CREDOTrainOutput(
            metrics=_good_metrics(), failure_type="RuntimeError", failure_message="boom after eval"
        )

    result = run_credo_trial(CREDOTrialSpec(), train_fn=train_fn, output_dir="/tmp/x")
    assert result.constraints["no_failure"] is False
    assert result.feasible is False
    assert result.failure_type == "RuntimeError"


def test_metrics_from_epoch_falls_back_when_val_endpoint_is_nan() -> None:
    from credo.search.metrics import metrics_from_epoch

    # NaN val_endpoint_loss must fall through to eval_endpoint_loss, then training.
    m = metrics_from_epoch(
        {"loss_end": 5.0, "val_endpoint_loss": float("nan"), "eval_endpoint_loss": 2.0, "loss_total": 5.0}
    )
    assert m.endpoint_geom_mass == pytest.approx(2.0)
    m2 = metrics_from_epoch({"loss_end": 5.0, "val_endpoint_loss": float("nan"), "loss_total": 5.0})
    assert m2.endpoint_geom_mass == pytest.approx(5.0)


def test_objective_vector_omits_nan_from_external_metrics() -> None:
    m = CREDOTrialMetrics(
        endpoint_geom_mass=0.3,
        endpoint_sinkhorn=float("nan"),
        count_nll=float("nan"),
        guide_concordance_gap=0.1,
    )
    vector = objective_vector(m)
    assert "endpoint_geometry" not in vector  # NaN pure-geometry dropped
    assert vector["endpoint_geom_mass"] == pytest.approx(0.3)  # fell back to combined
    assert "count_nll" not in vector  # NaN dropped
    assert vector["guide_concordance_gap"] == pytest.approx(0.1)  # now exposed
    assert all(v == v for v in vector.values())  # no NaN


def test_write_trial_dir_sanitizes_trial_id(tmp_path) -> None:
    from credo.search import write_trial_dir

    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=0.2,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    result = run_credo_trial(
        CREDOTrialSpec(), train_fn=lambda c, s, r: metrics, output_dir=str(tmp_path / "r")
    )
    trial_dir = write_trial_dir(tmp_path / "trials", result, trial_id="../../evil id")
    # The path stays under trials/ (no traversal) and the name is sanitized.
    assert (tmp_path / "trials") in trial_dir.parents
    assert ".." not in trial_dir.name
    assert (trial_dir / "result.json").exists()


def test_failure_metadata_penalizes_pruner_score() -> None:
    from credo.search import CREDOTrainOutput

    result = run_credo_trial(
        CREDOTrialSpec(),
        train_fn=lambda c, s, r: CREDOTrainOutput(
            metrics=_good_metrics(), failure_type="RuntimeError", failure_message="boom"
        ),
        output_dir="/tmp/x",
    )
    assert result.feasible is False
    assert result.constraints["no_failure"] is False
    # The scalar score reflects the SAME (augmented) feasibility verdict.
    assert result.pruner_score >= DIVERGENCE_PENALTY


def test_metrics_from_epoch_training_fallback_is_not_heldout() -> None:
    from credo.search.metrics import metrics_from_epoch

    m = metrics_from_epoch(
        {"loss_end": 5.0, "val_endpoint_loss": float("nan"), "loss_total": 5.0, "validation_source": "held_out"}
    )
    assert m.endpoint_geom_mass == pytest.approx(5.0)
    # Fell back to the training loss, so it is NOT held out despite the label.
    assert m.validation_source == "train_self_eval"


def test_last_accepts_numpy_scalars() -> None:
    import numpy as np

    from credo.search.metrics import _last

    assert _last(np.float32(1.25)) == pytest.approx(1.25)
    assert _last(np.float64(2.5)) == pytest.approx(2.5)
    assert _last(np.asarray(3.0)) == pytest.approx(3.0)  # 0-D array
    assert math.isnan(_last(np.float32("nan")))


def test_constraint_thresholds_reject_invalid_direct_construction() -> None:
    from credo.search import ConstraintThresholds

    with pytest.raises(ValueError, match="ess_floor"):
        ConstraintThresholds(ess_floor=-0.1)
    with pytest.raises(ValueError, match="max_weight_ceiling"):
        ConstraintThresholds(max_weight_ceiling=2.0)
    with pytest.raises(ValueError, match="control_null_max"):
        ConstraintThresholds(control_null_max=-1.0)


def test_fit_metrics_finite_accepts_endpoint_sinkhorn_when_combined_absent() -> None:
    m = CREDOTrialMetrics(
        endpoint_geom_mass=float("nan"),
        endpoint_sinkhorn=0.3,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    constraints = hard_constraints(m, CREDOTrialSpec(), DEFAULT_THRESHOLDS)
    assert constraints["fit_metrics_finite"] is True


def test_write_trial_dir_refuses_duplicate_without_overwrite(tmp_path) -> None:
    from credo.search import write_trial_dir

    metrics = CREDOTrialMetrics(
        endpoint_geom_mass=0.2,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    result = run_credo_trial(
        CREDOTrialSpec(seed=1), train_fn=lambda c, s, r: metrics, output_dir=str(tmp_path / "r")
    )
    write_trial_dir(tmp_path / "trials", result, trial_id="dup", index=0)
    with pytest.raises(FileExistsError):
        write_trial_dir(tmp_path / "trials", result, trial_id="dup", index=0)
    overwritten = write_trial_dir(tmp_path / "trials", result, trial_id="dup", index=0, overwrite=True)
    assert (overwritten / "result.json").exists()


def test_pruner_uses_endpoint_sinkhorn_when_combined_proxy_absent() -> None:
    from credo.search.objective import MISSING_METRIC_PENALTY

    m = CREDOTrialMetrics(
        endpoint_geom_mass=float("nan"),
        endpoint_sinkhorn=0.3,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
    )
    # Feasible via the pure-geometry fallback, and NOT penalized for a missing
    # combined proxy (pruner and feasibility now agree on the headline endpoint).
    assert hard_constraints(m, CREDOTrialSpec())["fit_metrics_finite"] is True
    assert pruner_score(m) < MISSING_METRIC_PENALTY


def test_decomposed_eval_summary_sets_heldout_endpoint_provenance() -> None:
    # A held-out summary that supplies only the pure-geometry endpoint (no combined
    # proxy) is still a held-out headline endpoint.
    m = metrics_from_history(
        {"loss_end": [0.5]},
        eval_summary={
            "mean_endpoint_sinkhorn": 0.3,
            "mean_log_mass_error": 0.1,
            "validation_source": "held_out",
        },
    )
    assert m.endpoint_sinkhorn == pytest.approx(0.3)
    assert m.validation_source == "held_out"
    assert m.mass_error_kind == "abs_log_residual"


def test_claim_grade_mass_error_ceiling_rejects_large_mass_error() -> None:
    from credo.search import claim_grade_thresholds

    th = claim_grade_thresholds(control_null_max=0.05, log_mass_error_max=0.2)
    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        validation_source="held_out",
        control_null_gap=0.0,
        **BRANCH_PARTICLE_OK,
    )
    too_large = CREDOTrialMetrics(**base, mass_error_value=0.5, mass_error_kind="abs_log_residual")
    constraints = hard_constraints(too_large, spec, th)
    assert constraints["mass_error_ok"] is False
    assert constraints_satisfied(constraints) is False

    relative = CREDOTrialMetrics(**base, mass_error_value=0.1, mass_error_kind="relative_error")
    constraints = hard_constraints(relative, spec, th)
    assert constraints["mass_error_kind_ok"] is False

    within = CREDOTrialMetrics(**base, mass_error_value=0.1, mass_error_kind="abs_log_residual")
    assert constraints_satisfied(hard_constraints(within, spec, th)) is True


def test_constrained_score_helper_shares_run_credo_trial_semantics() -> None:
    from credo.search import constrained_score_from_constraints

    metrics = _good_metrics()
    feasible_constraints = dict(hard_constraints(metrics, CREDOTrialSpec(), DEFAULT_THRESHOLDS))
    feasible_constraints["no_failure"] = True
    assert constrained_score_from_constraints(metrics, feasible_constraints) == pytest.approx(
        pruner_score(metrics)
    )

    failed_constraints = dict(feasible_constraints)
    failed_constraints["no_failure"] = False
    assert constrained_score_from_constraints(metrics, failed_constraints) >= DIVERGENCE_PENALTY


def test_optuna_studies_when_installed() -> None:
    pytest.importorskip("optuna")
    from credo.search.schedulers import make_multiobjective_study, make_study

    assert make_study(study_name="smoke").direction.name == "MINIMIZE"
    assert len(make_multiobjective_study(directions=["minimize", "minimize"]).directions) == 2


def test_schedulers_import_without_optuna() -> None:
    # Importing the adapters must not require optuna to be installed.
    import credo.search.schedulers as sched

    assert hasattr(sched, "OptunaReporter")
    assert hasattr(sched, "make_study")
