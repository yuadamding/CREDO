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

from typing import Callable, Optional

from .metrics import CREDOTrialMetrics, CREDOTrialResult
from .objective import (
    ConstraintThresholds,
    DEFAULT_THRESHOLDS,
    Standardizer,
    constraints_satisfied,
    feasible_pruner_score,
    hard_constraints,
    objective_vector,
)
from .pruning import NoOpReporter, SearchReporter
from .space import CREDOTrialSpec, spec_to_run_config


# train_fn(run_config, spec, reporter) -> CREDOTrialMetrics
TrainFn = Callable[[object, CREDOTrialSpec, SearchReporter], CREDOTrialMetrics]


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
    cfg = build_config(spec, output_dir=output_dir, device=device, latent_dim=latent_dim)

    metrics = train_fn(cfg, spec, reporter)
    if not isinstance(metrics, CREDOTrialMetrics):
        raise TypeError(
            "train_fn must return a CREDOTrialMetrics; got "
            f"{type(metrics).__name__}. Use metrics_from_history to build one."
        )

    constraints = hard_constraints(metrics, spec, thresholds)
    result = CREDOTrialResult(
        spec=spec,
        metrics=metrics,
        objective_vector=objective_vector(metrics),
        constraints=constraints,
        pruner_score=feasible_pruner_score(
            metrics, spec, thresholds=thresholds, standardizer=standardizer
        ),
        feasible=constraints_satisfied(constraints),
        run_dir=output_dir,
    )
    return result


__all__ = ["TrainFn", "run_credo_trial"]
