from __future__ import annotations

import dataclasses
import math

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
    identity = {"dataset_kind", "data_id", "seed", "fold_id", "latent_dim"}
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
    from credo.search import CLAIM_GRADE_THRESHOLDS

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

    cg = hard_constraints(sparse, spec, CLAIM_GRADE_THRESHOLDS)
    assert cg["mass_metric_finite"] is False
    assert cg["control_null_ok"] is False
    assert cg["guide_concordance_ok"] is False
    assert constraints_satisfied(cg) is False

    # A fully-specified claim-grade trial passes.
    complete = CREDOTrialMetrics(
        **base,
        log_mass_error=0.1,
        control_null_gap=0.0,
        guide_concordance_gap=0.0,
        heldout_score=0.2,
        validation_source="held_out",
    )
    assert constraints_satisfied(hard_constraints(complete, spec, CLAIM_GRADE_THRESHOLDS)) is True


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
    assert records[0]["schema_version"] == "credo.search.v1"


def test_claim_grade_requires_heldout_endpoint() -> None:
    from credo.search import CLAIM_GRADE_THRESHOLDS

    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        control_null_gap=0.0,
        guide_concordance_gap=0.0,
    )
    self_eval = CREDOTrialMetrics(**base, validation_source="train_self_eval")
    cg = hard_constraints(self_eval, spec, CLAIM_GRADE_THRESHOLDS)
    assert cg["heldout_endpoint_ok"] is False
    assert constraints_satisfied(cg) is False

    held_out = CREDOTrialMetrics(**base, validation_source="held_out")
    assert constraints_satisfied(hard_constraints(held_out, spec, CLAIM_GRADE_THRESHOLDS)) is True


def test_claim_grade_thresholds_factory_enforces_finite_ceilings() -> None:
    from credo.search import claim_grade_thresholds

    th = claim_grade_thresholds(
        control_null_max=0.05, guide_concordance_max=0.10, require_guide_concordance=True
    )
    spec = CREDOTrialSpec()
    base = dict(
        endpoint_geom_mass=0.3,
        log_mass_error=0.1,
        terminal_ess_frac_min=0.4,
        min_ess_frac_over_time=0.4,
        max_weight_frac_mean=0.2,
        converged=True,
        validation_source="held_out",
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
        claim_grade_thresholds(control_null_max=math.inf)
    with pytest.raises(ValueError, match="guide_concordance_max"):
        claim_grade_thresholds(control_null_max=0.05, require_guide_concordance=True)
    th = claim_grade_thresholds(control_null_max=0.05)
    assert th.control_null_max == pytest.approx(0.05)


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


def test_schedulers_import_without_optuna() -> None:
    # Importing the adapters must not require optuna to be installed.
    import credo.search.schedulers as sched

    assert hasattr(sched, "OptunaReporter")
    assert hasattr(sched, "make_study")
