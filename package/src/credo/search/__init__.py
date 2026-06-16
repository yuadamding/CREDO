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
    CREDOTrialMetrics,
    CREDOTrialResult,
    metrics_from_epoch,
    metrics_from_history,
)
from .objective import (
    CLAIM_GRADE_THRESHOLDS,
    ConstraintThresholds,
    DEFAULT_THRESHOLDS,
    Standardizer,
    constraints_satisfied,
    feasible_pruner_score,
    hard_constraints,
    objective_vector,
    pruner_score,
)
from .pruning import (
    NoOpReporter,
    RecordingReporter,
    SearchReporter,
    TrialPrunedError,
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
    spec_sha256,
    trial_record,
    write_trial_dir,
)

__all__ = [
    "ABLATION_ONLY",
    "CLAIM_GRADE_THRESHOLDS",
    "CREDOTrialMetrics",
    "CREDOTrialResult",
    "CREDOTrialSpec",
    "ConstraintThresholds",
    "DEFAULT_THRESHOLDS",
    "FROZEN",
    "NoOpReporter",
    "RecordingReporter",
    "SEARCHABLE",
    "SearchReporter",
    "Standardizer",
    "TrialPrunedError",
    "append_trial_record",
    "assert_frozen_semantics",
    "constraints_satisfied",
    "feasible_pruner_score",
    "hard_constraints",
    "load_trial_records",
    "metrics_from_epoch",
    "metrics_from_history",
    "objective_vector",
    "pareto_front",
    "pruner_score",
    "reduce_trial_dirs",
    "run_credo_trial",
    "spec_sha256",
    "spec_to_run_config",
    "trial_record",
    "write_trial_dir",
]
