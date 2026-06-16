from __future__ import annotations

import dataclasses

import pytest

from credo.search import (
    ABLATION_ONLY,
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
    constraints_satisfied,
    hard_constraints,
    load_trial_records,
    metrics_from_history,
    objective_vector,
    pruner_score,
    run_credo_trial,
    spec_to_run_config,
)
from credo.search.manifests import pareto_front, spec_sha256
from credo.search.objective import DIVERGENCE_PENALTY, feasible_pruner_score


pytestmark = pytest.mark.unit


# --- space ------------------------------------------------------------------

def test_field_classes_partition_tunable_fields() -> None:
    identity = {"dataset_kind", "data_id", "seed", "fold_id"}
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


# --- metrics extraction -----------------------------------------------------

def test_metrics_from_history_takes_last_epoch_values() -> None:
    history = {
        "loss_end": [1.0, 0.7, 0.5],
        "loss_mass": [0.9, 0.4, 0.3],
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
    assert m.log_mass_error == pytest.approx(0.3)
    assert m.count_nll == pytest.approx(1.2)
    assert m.min_ess_frac_over_time == pytest.approx(0.28)
    assert m.validation_source == "held_out"
    assert m.wall_seconds == pytest.approx(12.0)

    # Eval summary (held out) overrides the training loss for endpoint fit.
    m2 = metrics_from_history(history, eval_summary={"mean_endpoint_geom_mass": 0.61})
    assert m2.endpoint_geom_mass == pytest.approx(0.61)


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
    assert "endpoint_geometry" in result.objective_vector
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
        ["objective.endpoint_geometry", "objective.mass_error"],
        feasible_only=True,
    )
    seeds = {r["spec.seed"] for r in front}
    # The two non-dominated feasible trade-offs are on the front; the dominated
    # one is not, and the infeasible (ESS-collapsed) one is excluded entirely.
    assert seeds == {1, 2}


def test_schedulers_import_without_optuna() -> None:
    # Importing the adapters must not require optuna to be installed.
    import credo.search.schedulers as sched

    assert hasattr(sched, "OptunaReporter")
    assert hasattr(sched, "make_study")
