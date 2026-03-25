from __future__ import annotations

"""End-to-end CAMFND pipeline runner.

This module chains the benchmark workflow into a single callable:

    data_contract             – finite-measure endpoint benchmark validation
    simulator_validation      – trusted Euler-Maruyama reference checks
    single_screen_model       – control-anchored model without screen context
    multiscreen_context_model – context-aware model on the joint benchmark
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from camfnd.data.contract import PerturbSeqDynamicsData
from camfnd.data.multiscreen_benchmark import MultiscreenBenchmarkConfig, Stage2BenchmarkConfig, generate_multiscreen_dataset
from camfnd.data.single_screen_benchmark import SingleScreenBenchmarkConfig, Stage1BenchmarkConfig, generate_single_screen_dataset
from camfnd.evaluation.data_contract import DataContractEvaluation, evaluate_data_contract
from camfnd.evaluation.multiscreen_context_model import MultiscreenContextModelEvaluation, evaluate_multiscreen_context_model
from camfnd.evaluation.simulator_validation import SimulatorValidationEvaluation, evaluate_simulator_validation
from camfnd.evaluation.single_screen_model import SingleScreenModelEvaluation, evaluate_single_screen_model
from camfnd.training.multiscreen_context_model import MultiscreenContextTrainConfig, Stage2TrainConfig
from camfnd.training.single_screen_model import SingleScreenTrainConfig, Stage1TrainConfig


@dataclass(slots=True)
class PipelineResult:
    """Aggregated results from the complete benchmark-oriented CAMFND pipeline."""

    data_contract: DataContractEvaluation
    simulator_validation: SimulatorValidationEvaluation
    single_screen_model: SingleScreenModelEvaluation
    multiscreen_context_model: MultiscreenContextModelEvaluation
    single_screen_dataset: PerturbSeqDynamicsData
    multiscreen_dataset: PerturbSeqDynamicsData

    @property
    def step1(self) -> DataContractEvaluation:
        return self.data_contract

    @property
    def step2(self) -> SimulatorValidationEvaluation:
        return self.simulator_validation

    @property
    def step3(self) -> SingleScreenModelEvaluation:
        return self.single_screen_model

    @property
    def step4(self) -> MultiscreenContextModelEvaluation:
        return self.multiscreen_context_model

    @property
    def stage1_dataset(self) -> PerturbSeqDynamicsData:
        return self.single_screen_dataset

    @property
    def stage2_dataset(self) -> PerturbSeqDynamicsData:
        return self.multiscreen_dataset

    @property
    def all_pass(self) -> bool:
        return bool(
            self.data_contract.ok
            and self.simulator_validation.ok
            and self.single_screen_model.ok
            and self.multiscreen_context_model.ok
        )

    def to_dict(self) -> dict:
        return {
            "data_contract": self.data_contract.to_dict(),
            "simulator_validation": self.simulator_validation.to_dict(),
            "single_screen_model": self.single_screen_model.to_dict(),
            "multiscreen_context_model": self.multiscreen_context_model.to_dict(),
            "all_pass": self.all_pass,
        }

    def save(self, output_dir: str | Path) -> None:
        """Write all evaluation artifacts to `output_dir`."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # data_contract
        data_contract_dir = out / "data_contract"
        data_contract_dir.mkdir(exist_ok=True)
        self.single_screen_dataset.cells.obs.to_csv(data_contract_dir / "cells_obs.csv", index=False)
        self.single_screen_dataset.masses.table.to_csv(data_contract_dir / "masses.csv", index=False)
        if self.single_screen_dataset.truth and self.single_screen_dataset.truth.truth_params is not None:
            self.single_screen_dataset.truth.truth_params.to_csv(data_contract_dir / "truth_params.csv", index=False)
        if self.single_screen_dataset.truth and self.single_screen_dataset.truth.analytic_summary is not None:
            self.single_screen_dataset.truth.analytic_summary.to_csv(data_contract_dir / "analytic_summary.csv", index=False)
        self.data_contract.count_summary.to_csv(data_contract_dir / "count_summary.csv", index=False)
        self.data_contract.empirical_terminal_summary.to_csv(data_contract_dir / "empirical_terminal_summary.csv", index=False)
        self.data_contract.analytic_comparison.to_csv(data_contract_dir / "analytic_comparison.csv", index=False)
        (data_contract_dir / "evaluation.json").write_text(json.dumps(self.data_contract.to_dict(), indent=2))

        # simulator_validation
        simulator_dir = out / "simulator_validation"
        simulator_dir.mkdir(exist_ok=True)
        self.simulator_validation.default_run_summary.to_csv(simulator_dir / "default_run_summary.csv", index=False)
        self.simulator_validation.default_analytic_comparison.to_csv(simulator_dir / "default_analytic_comparison.csv", index=False)
        self.simulator_validation.convergence_table.to_csv(simulator_dir / "convergence_table.csv", index=False)
        (simulator_dir / "evaluation.json").write_text(json.dumps(self.simulator_validation.to_dict(), indent=2))

        # single_screen_model
        single_screen_dir = out / "single_screen_model"
        single_screen_dir.mkdir(exist_ok=True)
        self.single_screen_model.summary_table.to_csv(single_screen_dir / "summary_table.csv", index=False)
        self.single_screen_model.full_result.history.to_csv(single_screen_dir / "full_history.csv", index=False)
        self.single_screen_model.full_result.final_simulation.summary.to_csv(single_screen_dir / "full_terminal_summary.csv", index=False)
        (single_screen_dir / "evaluation.json").write_text(json.dumps(self.single_screen_model.to_dict(), indent=2))

        # multiscreen_context_model
        multiscreen_dir = out / "multiscreen_context_model"
        multiscreen_dir.mkdir(exist_ok=True)
        self.multiscreen_context_model.summary_table.to_csv(multiscreen_dir / "summary_table.csv", index=False)
        self.multiscreen_context_model.full_result.history.to_csv(multiscreen_dir / "full_history.csv", index=False)
        self.multiscreen_context_model.no_context_result.history.to_csv(multiscreen_dir / "no_context_history.csv", index=False)
        self.multiscreen_context_model.full_result.final_loss_table.to_csv(multiscreen_dir / "full_terminal_summary.csv", index=False)
        self.multiscreen_context_model.no_context_result.final_loss_table.to_csv(multiscreen_dir / "no_context_terminal_summary.csv", index=False)
        self.multiscreen_context_model.full_result.final_simulation.context_summary.to_csv(multiscreen_dir / "full_context_summary.csv", index=False)
        self.multiscreen_context_model.no_context_result.final_simulation.context_summary.to_csv(multiscreen_dir / "no_context_context_summary.csv", index=False)
        if self.multiscreen_dataset.truth is not None and self.multiscreen_dataset.truth.context_trajectories is not None:
            self.multiscreen_dataset.truth.context_trajectories.to_csv(multiscreen_dir / "truth_context_trajectories.csv", index=False)
        (multiscreen_dir / "evaluation.json").write_text(json.dumps(self.multiscreen_context_model.to_dict(), indent=2))

        # Combined summary
        (out / "pipeline_summary.json").write_text(json.dumps(self.to_dict(), indent=2))


def run_full_pipeline(
    *,
    single_screen_config: Optional[SingleScreenBenchmarkConfig] = None,
    multiscreen_config: Optional[MultiscreenBenchmarkConfig] = None,
    single_screen_train_config: Optional[SingleScreenTrainConfig] = None,
    multiscreen_train_config: Optional[MultiscreenContextTrainConfig] = None,
    stage1_config: Optional[Stage1BenchmarkConfig] = None,
    stage2_config: Optional[Stage2BenchmarkConfig] = None,
    stage1_train_config: Optional[Stage1TrainConfig] = None,
    stage2_train_config: Optional[Stage2TrainConfig] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the complete benchmark-oriented CAMFND pipeline end to end.

    Parameters
    ----------
    single_screen_config:
        Configuration for the single-screen benchmark data generation.
    multiscreen_config:
        Configuration for the multi-screen benchmark data generation.
    single_screen_train_config:
        Training configuration for the single-screen learnable model benchmark.
    multiscreen_train_config:
        Training configuration for the multi-screen context-aware benchmark.
    stage1_config, stage2_config, stage1_train_config, stage2_train_config:
        Backward-compatible legacy parameter names.
    output_dir:
        If provided, all evaluation artifacts are saved here.
    verbose:
        Print evaluation summaries to stdout.

    Returns
    -------
    PipelineResult
        All evaluation results and training outputs from the benchmark pipeline.
    """

    single_screen_config = single_screen_config or stage1_config or SingleScreenBenchmarkConfig()
    multiscreen_config = multiscreen_config or stage2_config or MultiscreenBenchmarkConfig(
        seed=29,
        n_obs_p4=32,
        n_obs_p60=32,
        n_truth_particles=1024,
        n_steps=48,
    )
    single_screen_train_config = single_screen_train_config or stage1_train_config or SingleScreenTrainConfig()
    multiscreen_train_config = multiscreen_train_config or stage2_train_config or MultiscreenContextTrainConfig()

    # --- Data-contract validation ---
    if verbose:
        print("=" * 60)
        print("data_contract: finite-measure endpoint benchmark")
        print("=" * 60)
    single_screen_dataset = generate_single_screen_dataset(single_screen_config)
    data_contract_eval = evaluate_data_contract(single_screen_dataset)
    if verbose:
        print(json.dumps(data_contract_eval.to_dict(), indent=2))

    # --- Simulator validation ---
    if verbose:
        print("\n" + "=" * 60)
        print("simulator_validation: trusted Euler-Maruyama reference")
        print("=" * 60)
    simulator_validation_eval = evaluate_simulator_validation(single_screen_dataset)
    if verbose:
        print(json.dumps(simulator_validation_eval.to_dict(), indent=2))

    # --- Single-screen model benchmark ---
    if verbose:
        print("\n" + "=" * 60)
        print("single_screen_model: control-anchored benchmark")
        print("=" * 60)
    single_screen_model_eval = evaluate_single_screen_model(single_screen_dataset, full_config=single_screen_train_config)
    if verbose:
        print(json.dumps(single_screen_model_eval.to_dict(), indent=2))

    # --- Multi-screen context benchmark ---
    if verbose:
        print("\n" + "=" * 60)
        print("multiscreen_context_model: context-aware benchmark")
        print("=" * 60)
    multiscreen_dataset = generate_multiscreen_dataset(multiscreen_config)
    multiscreen_context_eval = evaluate_multiscreen_context_model(multiscreen_dataset, full_config=multiscreen_train_config)
    if verbose:
        print(json.dumps(multiscreen_context_eval.to_dict(), indent=2))

    result = PipelineResult(
        data_contract=data_contract_eval,
        simulator_validation=simulator_validation_eval,
        single_screen_model=single_screen_model_eval,
        multiscreen_context_model=multiscreen_context_eval,
        single_screen_dataset=single_screen_dataset,
        multiscreen_dataset=multiscreen_dataset,
    )

    if output_dir is not None:
        result.save(output_dir)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Pipeline complete. All pass: {result.all_pass}")
        print("=" * 60)

    return result
