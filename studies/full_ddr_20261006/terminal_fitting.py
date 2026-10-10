"""Study-specific terminal fitting, with assessment deferred to sealed forecasts.

This adapter does not authorize execution or choose an exposure budget. It keeps
the inherited objective/optimizer plan but prevents its automatic holdout score.
Input compilation and structural validation are not an outcome-access firewall;
the phase worker must bind exact inputs and roles before constructing a trainer.
Only D3/D4 architecture fitting is implemented here. The selected all-four
refit needs its own later phase admission, not a relaxed donor check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch

from credo.problems import FiniteMeasureDynamicsProblem
from credo.recipes.compact_sde_v3.particles import (
    NoContextProvider, rollout, sample_initial_particles, sample_noise,
)
from credo.recipes.compact_sde_v3.recipe import CompactSDEV3Recipe
from credo.recipes.compact_sde_v3.training import Trainer


POLICY_ID = "main_full_ddr_terminal_fit_deferred_assessment_v1"
REFRESH_GUIDES = 256


def _validate_architecture_fit(problem, config) -> None:
    if not isinstance(problem, FiniteMeasureDynamicsProblem):
        raise ValueError("A phase-compiled finite-measure problem is required.")
    for data, expected in ((problem.training, {"D3", "D4"}), (problem.validation, {"D2"})):
        metadata = data.measure_meta.set_index("measure_id")
        if set(metadata.sample_id.astype(str)) != expected:
            raise ValueError("Architecture fitting requires D3/D4 and D2 selection roles.")
        if any(not str(key).startswith(f"{metadata.loc[key, 'sample_id']}::") for key in data.measure_ids):
            raise ValueError("Measure identity and donor role disagree.")
        if tuple(data.axis.labels) != ("Stim8hr", "Stim48hr") or tuple(data.axis.values) != (8.0, 48.0):
            raise ValueError("Only the declared stimulated 40-hour interval is admitted.")
    settings = config.recipe_config
    if settings.training.checkpoint_selection != "last":
        raise ValueError("Only terminal checkpoint selection is admitted before sealing.")
    if settings.model.context != "none" or settings.training.epochs.context != 0:
        raise ValueError("This architecture study cannot enable ecological context.")
    if settings.training.epochs.mass < 1:
        raise ValueError("Full DDR requires a positive reaction-stage exposure.")
    if settings.loss.mass != 0 or settings.loss.count <= 0 or settings.loss.count_likelihood != "categorical_ce":
        raise ValueError("Captured categorical counts, not absolute mass or concentration fitting, are required.")
    if settings.loss.mass_increment != "particle_log_mass":
        raise ValueError("Exact terminal-minus-initial particle log-mass increments are required.")
    if Path(config.output).exists():
        raise FileExistsError("Fitting output already exists; do not resume or overwrite implicitly.")


class TerminalFittingTrainer(Trainer):
    """No public assessment API; terminal core evaluation is explicitly deferred."""

    def _fit(self) -> None:
        _validate_architecture_fit(self.problem, self.config)
        self._deferred_terminal_calls = 0
        self._inside_terminal_fit = True
        try:
            super()._fit()
        finally:
            self._inside_terminal_fit = False
        requested = sum(stage.epochs for stage in self.training_plan.stages)
        if self._deferred_terminal_calls != 1 or self.completed_epochs != requested:
            raise RuntimeError("Terminal fitting did not execute the exact plan and one deferred final evaluation.")
        if not self.metrics.empty or any(row["validation_monitored"] for row in self.history_rows):
            raise RuntimeError("Unexpected holdout assessment during terminal fitting.")
        self.execution_trace.append(dict(event=POLICY_ID, assessment_performed=False,
                                         terminal_evaluation_deferred=True,
                                         execution_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                         catalog_refresh_guides=REFRESH_GUIDES,
                                         refresh_rng="plan_seed+epoch*1009+guide_chunk_index_v1"))
        self._terminal_fitting_complete = True

    def evaluate(self, data=None, *, particles=None, seed=None) -> pd.DataFrame:
        # The core trainer calls this exactly once after terminal fitting. It
        # must not turn that call into an unsealed D2 assessment. All other
        # uses fail, rather than silently presenting an empty result as success.
        if (getattr(self, "_inside_terminal_fit", False) and data is None and particles is None
                and seed is None and self.completed_epochs == sum(stage.epochs for stage in self.training_plan.stages)
                and self._deferred_terminal_calls == 0):
            self._deferred_terminal_calls += 1
            return pd.DataFrame(columns=["measure_id", "time_label"])
        raise RuntimeError("Assessment is deferred: publish and seal forecasts before the separate scorer.")

    def _evaluate_ids(self, *args, **kwargs):
        raise RuntimeError("Holdout evaluation cannot run through a fitting-only trainer.")

    def _validation_count_loss(self, *args, **kwargs):
        raise RuntimeError("Holdout count scoring cannot run through a fitting-only trainer.")

    def _validation_score(self, *args, **kwargs):
        raise RuntimeError("Holdout checkpoint selection cannot run through a fitting-only trainer.")

    @torch.no_grad()
    def _refresh_bank_for(self, data, bank, *, epoch: int, grid=None) -> None:
        """Detached no-context refresh, matching the capacity path's batch bound.

        Chunking changes the sampling stream relative to the historical full-
        context-group refresh. This is an explicit new RNG policy, not exact
        historical replay. Every admitted fitting guide is still refreshed.
        """
        if data is not self.data or bank is not self.bank or self.model.context_mode != "none":
            raise ValueError("Only the fitting bank may be refreshed, with context disabled.")
        rollout_grid = self.training_grid if grid is None else grid
        if int(bank.tensors["context_log_mass"].shape[0]) != len(rollout_grid) - 1:
            raise ValueError("Catalog bank and refresh integration grids disagree.")
        if bank.has_fixed_background or self.model.context_enabled:
            raise ValueError("This bounded refresh cannot substitute for a coupled ecological rollout.")
        was_training = self.model.training
        self.model.eval()
        try:
            bank.reset_coverage()
            bank.refresh_fixed_background(self.model)
            particles = self._catalog_refresh_particle_count()
            bank.refresh_particles = particles
            for chunk, start in enumerate(range(0, len(data.measure_ids), REFRESH_GUIDES)):
                initial = sample_initial_particles(data, data.measure_ids[start:start + REFRESH_GUIDES], particles,
                    device=self.device, dtype=self.dtype, seed=self.training_plan.seed + epoch * 1009 + chunk)
                result = rollout(self.model, initial, rollout_grid, context_provider=NoContextProvider(),
                    noise=sample_noise(initial, rollout_grid,
                                       seed=self.training_plan.seed + epoch * 1009 + chunk + 2_000_003))
                bank.update_from_rollout(result, self.model, data, full_refresh=True)
                del result, initial
            bank.last_full_refresh_epoch = int(epoch)
            bank.assert_complete()
        finally:
            self.model.train(was_training)

    def _manifest(self):
        manifest = super()._manifest()
        manifest["study_execution_policy"] = POLICY_ID
        manifest["scientific_assessment"] = "not_performed_fitting_only"
        manifest["terminal_evaluation_deferred"] = True
        return manifest

    def save(self):
        if not getattr(self, "_terminal_fitting_complete", False) or not self.metrics.empty:
            raise RuntimeError("Only a complete terminal fit with no assessment rows may be published.")
        if Path(self.config.output).exists():
            raise FileExistsError("Preserve existing or failed publications; never overwrite.")
        return super().save()

    @classmethod
    def load(cls, *args, **kwargs):
        raise RuntimeError("Use a separately admitted source-only checkpoint loader; inherited load evaluates outcomes.")


class TerminalFittingRecipe(CompactSDEV3Recipe):
    """Same estimator/immutable plan; explicit study-specific execution policy."""

    def execute_training(self, study, *, model, plan, objectives, run_config, **kwargs):
        _validate_architecture_fit(study, run_config)
        return TerminalFittingTrainer.from_plan(study, model, run_config, plan, objectives, **kwargs)


recipe = TerminalFittingRecipe()
