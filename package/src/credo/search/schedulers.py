"""Optional Optuna adapters for CREDO setting search.

Importing this module never requires Optuna; the adapters import it lazily and
raise a clear error only when actually used. This keeps ``credo.search`` usable
(spec/metrics/objective/manifest) without the optimizer dependency installed.

Recommended engine: Optuna with TPE (or NSGA-II for multi-objective) plus a
Hyperband/SuccessiveHalving pruner driven by ``pruner_score`` via the reporter.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .metrics import CREDOTrialMetrics
from .objective import pruner_score
from .pruning import SearchReporter


def _require_optuna() -> Any:
    try:
        import optuna  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - exercised only without optuna
        raise ImportError(
            "Optuna is required for credo.search.schedulers. "
            "Install it with `pip install optuna` (it is an optional search "
            "dependency, not a core CREDO requirement)."
        ) from exc
    return optuna


class OptunaReporter:
    """Bridge a CREDO trial's intermediate metrics to an Optuna trial.

    ``report`` feeds the scalar ``pruner_score`` to ``trial.report`` and
    ``should_prune`` defers to Optuna's configured pruner.
    """

    def __init__(self, trial: Any, *, score_fn: Callable[[CREDOTrialMetrics], float] = pruner_score) -> None:
        self.trial = trial
        self._score_fn = score_fn

    def report(self, step: int, metrics: CREDOTrialMetrics) -> None:
        self.trial.report(self._score_fn(metrics), step=step)

    def should_prune(self) -> bool:
        return bool(self.trial.should_prune())


def suggest_spec(trial: Any, base: dict[str, Any]) -> "CREDOTrialSpec":
    """Sample a :class:`CREDOTrialSpec` from an Optuna trial over searchable dims.

    ``base`` supplies the fixed fields (dataset_kind, data_id, seed, frozen
    semantics, ablation choices). Only SEARCHABLE dimensions are suggested here;
    extend as needed for a given study.
    """
    from .space import CREDOTrialSpec  # local import to avoid cycle at module load

    return CREDOTrialSpec(
        **base,
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512, 768]),
        depth=trial.suggest_int("depth", 2, 5),
        embedding_dim=trial.suggest_categorical("embedding_dim", [8, 16, 32, 64]),
        n_programs=trial.suggest_categorical("n_programs", [8, 16, 24, 32]),
        mediator_dim=trial.suggest_categorical("mediator_dim", [8, 16, 32]),
        lr_net=trial.suggest_float("lr_net", 1e-5, 3e-3, log=True),
        lr_embed=trial.suggest_float("lr_embed", 1e-5, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        lambda_weak=trial.suggest_float("lambda_weak", 1e-4, 10.0, log=True),
        lambda_count=trial.suggest_float("lambda_count", 0.0, 10.0),
        sinkhorn_tau=trial.suggest_float("sinkhorn_tau", 0.1, 10.0, log=True),
        n_particles=trial.suggest_categorical("n_particles", [64, 128, 256]),
        n_steps=trial.suggest_categorical("n_steps", [8, 16, 24]),
    )


def make_study(
    *,
    direction: str = "minimize",
    sampler: Optional[Any] = None,
    pruner: Optional[Any] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Any:
    """Create an Optuna study with sensible CREDO defaults (TPE + Hyperband)."""
    optuna = _require_optuna()
    sampler = sampler or optuna.samplers.TPESampler(multivariate=True, group=True)
    pruner = pruner or optuna.pruners.HyperbandPruner()
    return optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )


__all__ = ["OptunaReporter", "make_study", "suggest_spec"]
