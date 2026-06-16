"""Callable trial runner: treat CREDO as a black-box function of a spec.

``run_credo_trial`` maps a :class:`CREDOTrialSpec` to a validated ``RunConfig``,
hands it (plus the reporter) to an injected ``train_fn``, and turns the returned
metrics into a :class:`CREDOTrialResult` with the objective vector, hard
constraints, and feasible pruner score attached.

``train_fn`` is the only thing that touches data, torch, and the real trainer,
so the search layer itself stays light and unit-testable. A real endpoint
adapter looks like::

    def endpoint_train_fn(cfg, spec, reporter):
        problem = build_endpoint_problem(...)          # data -> EndpointProblem
        model = build_model_from_config(cfg, problem)  # FullDynamicsModel
        trainer = Trainer(model=model, problem=problem, config=cfg,
                          reporter=reporter, output_dir=cfg.output_dir)
        history = trainer.train()                      # raises TrialPrunedError if pruned
        summary = summarize_eval(trainer.evaluate())   # held-out fit
        return metrics_from_history(history.to_dict(), eval_summary=summary,
                                    wall_seconds=..., diverged=trainer.diverged)

The data -> problem step is the part that still needs a ``Namespace``-free entry
point extracted from the CLI runners; everything downstream of ``cfg`` is already
config-driven.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Union

from .metrics import CREDOTrainOutput, CREDOTrialMetrics, CREDOTrialResult
from .objective import (
    ConstraintThresholds,
    DEFAULT_THRESHOLDS,
    DIVERGENCE_PENALTY,
    Standardizer,
    constraints_satisfied,
    hard_constraints,
    objective_vector,
    pruner_score,
)
from .pruning import NoOpReporter, SearchReporter, TrialPrunedError
from .space import CREDOTrialSpec, spec_to_run_config


# train_fn(run_config, spec, reporter) -> CREDOTrialMetrics | CREDOTrainOutput
TrainFn = Callable[[object, CREDOTrialSpec, SearchReporter], Union[CREDOTrialMetrics, CREDOTrainOutput]]


def run_credo_trial(
    spec: CREDOTrialSpec,
    *,
    train_fn: TrainFn,
    output_dir: str,
    reporter: Optional[SearchReporter] = None,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
    standardizer: Optional[Standardizer] = None,
    latent_dim: Optional[int] = None,
    device: str = "cpu",
    build_config: Callable[..., object] = spec_to_run_config,
) -> CREDOTrialResult:
    """Run one CREDO trial end-to-end and return a scored, constrained result.

    ``train_fn`` must return :class:`CREDOTrialMetrics`. It may raise
    :class:`credo.search.pruning.TrialPrunedError` for early stopping; that
    propagates to the caller (the optimizer adapter translates it). Frozen
    method semantics are enforced when the config is built.
    """
    reporter = reporter or NoOpReporter()
    # Representation identity has one source of truth: the (hashed) spec. An
    # explicit latent_dim is folded into the spec so it cannot diverge from the
    # spec hash; a conflicting override is an error rather than a silent change.
    if latent_dim is not None:
        if spec.latent_dim is not None and int(spec.latent_dim) != int(latent_dim):
            raise ValueError(
                f"latent_dim override {latent_dim} conflicts with spec.latent_dim="
                f"{spec.latent_dim}; set latent_dim once on CREDOTrialSpec so it is hashed."
            )
        spec = dataclasses.replace(spec, latent_dim=int(latent_dim))
    cfg = build_config(spec, output_dir=output_dir, device=device, latent_dim=None)

    try:
        out = train_fn(cfg, spec, reporter)
    except TrialPrunedError:
        raise
    except Exception as exc:  # noqa: BLE001 - translate the trainer's prune signal
        # The trainers raise a duck-typed pruned exception (marker attribute
        # _credo_pruned) so credo.training need not import credo.search. Translate
        # it into the search-native TrialPrunedError; re-raise anything else.
        if getattr(exc, "_credo_pruned", False):
            raise TrialPrunedError(getattr(exc, "epoch", None)) from exc
        raise

    # train_fn may return rich provenance (CREDOTrainOutput) or just metrics.
    if isinstance(out, CREDOTrainOutput):
        output = out
    elif isinstance(out, CREDOTrialMetrics):
        output = CREDOTrainOutput(metrics=out, run_dir=output_dir)
    else:
        raise TypeError(
            "train_fn must return a CREDOTrialMetrics or CREDOTrainOutput; got "
            f"{type(out).__name__}. Use metrics_from_history to build the metrics."
        )

    metrics = output.metrics
    constraints = dict(hard_constraints(metrics, spec, thresholds))
    # A train_fn that reports a failure must not yield a feasible trial.
    constraints["no_failure"] = output.failure_type is None and output.failure_message is None
    feasible = constraints_satisfied(constraints)
    # Derive the scalar score from the SAME (augmented) feasibility verdict, so a
    # failed-but-otherwise-fine trial gets the infeasibility penalty rather than a
    # competitive score. (feasible_pruner_score recomputes only hard_constraints
    # and would miss the no_failure constraint added here.)
    score = pruner_score(metrics, standardizer=standardizer)
    if not feasible:
        score += DIVERGENCE_PENALTY
    return CREDOTrialResult(
        spec=spec,
        metrics=metrics,
        objective_vector=objective_vector(metrics),
        constraints=constraints,
        pruner_score=score,
        feasible=feasible,
        run_dir=output.run_dir or output_dir,
        checkpoint_path=output.checkpoint_path,
        history_path=output.history_path,
        eval_summary_path=output.eval_summary_path,
        resolved_config_path=output.resolved_config_path,
        failure_type=output.failure_type,
        failure_message=output.failure_message,
    )


__all__ = ["TrainFn", "run_credo_trial"]
