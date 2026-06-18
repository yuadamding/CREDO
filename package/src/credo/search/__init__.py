"""Dataset-agnostic setting-search layer around CREDO.

This package wraps a CREDO trainer as a black-box function of a validated
configuration so that multi-objective / multi-fidelity hyperparameter
optimization (Optuna, Ray Tune) - and optionally a scheduler-style RL controller
- can search *training settings* without touching CREDO's model semantics.

Design split:

* ``space``      - ``CREDOTrialSpec`` and the SEARCHABLE / ABLATION_ONLY / FROZEN
  field classes; ``spec_to_run_config`` builds a validated ``RunConfig``.
* ``metrics``    - ``CREDOTrialMetrics`` / ``CREDOTrialResult`` and extraction
  from a trainer history + eval summary.
* ``objective``  - scalar pruner score, multi-objective vector, and hard
  feasibility constraints (wired to intra-trajectory ESS and held-out
  provenance).
* ``pruning``    - the ``SearchReporter`` protocol and ``TrialPrunedError``.
* ``runner``     - ``run_credo_trial``: the callable trial entry point.
* ``schedulers`` - optional Optuna adapters (lazy import).
* ``manifests``  - the per-trial reproducibility database + Pareto extraction.

Frozen method semantics (control reference, same-start counterfactuals,
mass-faithful context, exact full-context batching) are never searchable;
``assert_frozen_semantics`` enforces that.
"""
from __future__ import annotations

from .metrics import (
    CREDOTrainOutput,
    CREDOTrialMetrics,
    CREDOTrialResult,
    MassErrorKind,
    metrics_from_epoch,
    metrics_from_history,
)
from .objective import (
    CLAIM_GRADE_PRESENCE_THRESHOLDS,
    CLAIM_GRADE_THRESHOLDS,
    ConstraintThresholds,
    DEFAULT_THRESHOLDS,
    SearchProfile,
    Standardizer,
    claim_grade_thresholds,
    constrained_score_from_constraints,
    constraints_satisfied,
    feasible_pruner_score,
    hard_constraints,
    objective_vector,
    pruner_score,
    thresholds_for_profile,
)
from .pruning import (
    NoOpReporter,
    RecordingReporter,
    SearchReporter,
    TrialPrunedError,
)
from .diagnostics import (
    BASELINE_KINDS,
    BASELINE_STATUSES,
    DEFAULT_BIOLOGY_AXES,
    BiologyAxisSpec,
    ConvergenceThresholds,
    FidelityRecord,
    baseline_export_manifest,
    baseline_export_record,
    evaluate_biology_axis_gates,
    particle_step_convergence_diagnostics,
    summarize_null_distribution,
    summarize_null_suite,
)
from .problem_builders import (
    DEFAULT_PROBLEM_BUILDERS,
    ProblemBuilderRegistry,
    build_endpoint_problem_from_config,
    build_single_time_problem_from_config,
    build_trajectory_problem_from_config,
    clear_problem_builders,
    register_problem_builder,
)
from .runner import run_credo_trial
from .space import (
    ABLATION_ONLY,
    CREDOTrialSpec,
    FROZEN,
    SEARCHABLE,
    assert_frozen_semantics,
    spec_to_run_config,
)
from .manifests import (
    append_trial_record,
    load_trial_records,
    pareto_front,
    reduce_trial_dirs,
    select_final_candidates,
    setting_sha256,
    spec_sha256,
    trial_record,
    write_trial_dir,
)

__all__ = [
    "ABLATION_ONLY",
    "BASELINE_KINDS",
    "BASELINE_STATUSES",
    "BiologyAxisSpec",
    "CLAIM_GRADE_PRESENCE_THRESHOLDS",
    "CLAIM_GRADE_THRESHOLDS",
    "CREDOTrainOutput",
    "CREDOTrialMetrics",
    "CREDOTrialResult",
    "CREDOTrialSpec",
    "ConstraintThresholds",
    "ConvergenceThresholds",
    "DEFAULT_BIOLOGY_AXES",
    "DEFAULT_PROBLEM_BUILDERS",
    "DEFAULT_THRESHOLDS",
    "FidelityRecord",
    "FROZEN",
    "MassErrorKind",
    "NoOpReporter",
    "ProblemBuilderRegistry",
    "RecordingReporter",
    "SEARCHABLE",
    "SearchProfile",
    "SearchReporter",
    "Standardizer",
    "TrialPrunedError",
    "append_trial_record",
    "assert_frozen_semantics",
    "baseline_export_manifest",
    "baseline_export_record",
    "build_endpoint_problem_from_config",
    "build_single_time_problem_from_config",
    "build_trajectory_problem_from_config",
    "claim_grade_thresholds",
    "clear_problem_builders",
    "constrained_score_from_constraints",
    "constraints_satisfied",
    "evaluate_biology_axis_gates",
    "feasible_pruner_score",
    "hard_constraints",
    "load_trial_records",
    "metrics_from_epoch",
    "metrics_from_history",
    "objective_vector",
    "pareto_front",
    "pruner_score",
    "particle_step_convergence_diagnostics",
    "reduce_trial_dirs",
    "register_problem_builder",
    "run_credo_trial",
    "select_final_candidates",
    "setting_sha256",
    "spec_sha256",
    "spec_to_run_config",
    "summarize_null_distribution",
    "summarize_null_suite",
    "thresholds_for_profile",
    "trial_record",
    "write_trial_dir",
]
