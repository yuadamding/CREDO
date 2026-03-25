from __future__ import annotations

"""Acceptance checks for the single-screen learnable model."""

from dataclasses import dataclass, replace
from typing import Dict, Optional

import pandas as pd

from camfnd.data.contract import PerturbSeqDynamicsData
from camfnd.data.single_screen_benchmark import Stage1BenchmarkConfig, generate_stage1_dataset
from camfnd.training.single_screen_model import Stage1TrainConfig, Stage1TrainingResult, train_stage1_model


@dataclass(slots=True)
class Step3Evaluation:
    full_result: Stage1TrainingResult
    no_growth_result: Stage1TrainingResult
    shared_diffusion_result: Stage1TrainingResult
    normalized_only_result: Stage1TrainingResult
    summary_table: pd.DataFrame
    thresholds: Dict[str, float]

    @property
    def full_pass(self) -> bool:
        row = self.summary_table.set_index("model_name").loc["full"]
        return bool(
            row["qualitative_pass"]
            and row["mean_abs_mass_error"] <= self.thresholds["full_mass_error_max"]
            and row["mean_abs_mean_error"] <= self.thresholds["full_mean_error_max"]
            and row["mean_abs_variance_error"] <= self.thresholds["full_variance_error_max"]
        )

    @property
    def ablations_fail_as_expected(self) -> bool:
        table = self.summary_table.set_index("model_name")
        no_growth_ok = bool(table.loc["no_growth", "react_mass_error"] >= self.thresholds["react_mass_failure_min"])
        normalized_ok = bool(table.loc["normalized_only", "react_mass_error"] >= self.thresholds["react_mass_failure_min"])
        shared_diff_ok = bool(table.loc["shared_diffusion", "diff_variance_error"] >= self.thresholds["diff_variance_failure_min"])
        return no_growth_ok and normalized_ok and shared_diff_ok

    @property
    def ok(self) -> bool:
        return bool(self.full_pass and self.ablations_fail_as_expected)

    def to_dict(self) -> dict:
        return {
            "full_pass": self.full_pass,
            "ablations_fail_as_expected": self.ablations_fail_as_expected,
            "ok": self.ok,
            "thresholds": dict(self.thresholds),
        }


def _truth_table(dataset: PerturbSeqDynamicsData) -> pd.DataFrame:
    problem = dataset.to_endpoint_problem(by_sample=True)
    rows = []
    for key, measure in problem.terminal.items():
        rows.append(
            {
                "sample_id": measure.sample_id,
                "perturbation_id": measure.perturbation_id,
                "terminal_mean": float(measure.mean()[0]),
                "terminal_variance": float(measure.variance_trace()),
                "terminal_mass": float(measure.total_mass),
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _summarize_result(model_name: str, result: Stage1TrainingResult, truth: pd.DataFrame) -> dict:
    pred = result.final_simulation.summary[
        [
            "sample_id",
            "perturbation_id",
            "terminal_mass",
            "terminal_mean_0",
            "terminal_var_trace",
        ]
    ].copy().rename(
        columns={
            "terminal_mass": "terminal_mass_pred",
            "terminal_mean_0": "terminal_mean_pred",
            "terminal_var_trace": "terminal_variance_pred",
        }
    )
    merged = pred.merge(truth, on=["sample_id", "perturbation_id"], how="left")
    merged["abs_mass_error"] = (merged["terminal_mass_pred"] - merged["terminal_mass"]).abs()
    merged["abs_mean_error"] = (merged["terminal_mean_pred"] - merged["terminal_mean"]).abs()
    merged["abs_variance_error"] = (merged["terminal_variance_pred"] - merged["terminal_variance"]).abs()

    by_pert = merged.set_index("perturbation_id")
    ctrl_mean = float(by_pert.loc["ctrl", "terminal_mean_pred"])
    drift_mean = float(by_pert.loc["drift", "terminal_mean_pred"])
    ctrl_var = float(by_pert.loc["ctrl", "terminal_variance_pred"])
    diff_var = float(by_pert.loc["diff", "terminal_variance_pred"])
    ctrl_mass = float(by_pert.loc["ctrl", "terminal_mass_pred"])
    react_mass = float(by_pert.loc["react", "terminal_mass_pred"])
    react_mass_truth = float(by_pert.loc["react", "terminal_mass"])
    diff_var_truth = float(by_pert.loc["diff", "terminal_variance"])

    qualitative_pass = bool(
        drift_mean > ctrl_mean + 0.10
        and diff_var > ctrl_var + 0.005
        and react_mass < ctrl_mass - 0.10
    )

    return {
        "model_name": model_name,
        "qualitative_pass": qualitative_pass,
        "mean_abs_mass_error": float(merged["abs_mass_error"].mean()),
        "mean_abs_mean_error": float(merged["abs_mean_error"].mean()),
        "mean_abs_variance_error": float(merged["abs_variance_error"].mean()),
        "react_mass_error": abs(react_mass - react_mass_truth),
        "diff_variance_error": abs(diff_var - diff_var_truth),
        "drift_mean_minus_ctrl": drift_mean - ctrl_mean,
        "diff_variance_minus_ctrl": diff_var - ctrl_var,
        "ctrl_mass_minus_react": ctrl_mass - react_mass,
        "endpoint_loss_mean": float(result.final_loss_table["endpoint_loss"].mean()),
        "best_total_loss": float(result.history["total_loss"].min()),
    }


def evaluate_step3(
    dataset: Optional[PerturbSeqDynamicsData] = None,
    *,
    full_config: Optional[Stage1TrainConfig] = None,
) -> Step3Evaluation:
    """Train the full Step-3 model and the required ablations on Stage I."""

    if dataset is None:
        dataset = generate_stage1_dataset(
            Stage1BenchmarkConfig(seed=7, n_obs_p4=32, n_obs_p60=32, infer_latent_transform=True)
        )
    dataset.validate()
    truth = _truth_table(dataset)

    full_config = full_config or Stage1TrainConfig()
    full_config.validate()

    full_result = train_stage1_model(dataset, config=full_config)
    no_growth_result = train_stage1_model(dataset, config=replace(full_config, use_growth=False))
    shared_diffusion_result = train_stage1_model(dataset, config=replace(full_config, shared_diffusion=True))
    normalized_only_result = train_stage1_model(
        dataset,
        config=replace(full_config, loss_mode="normalized_only", aux_mass_weight=0.0),
    )

    summary_rows = [
        _summarize_result("full", full_result, truth),
        _summarize_result("no_growth", no_growth_result, truth),
        _summarize_result("shared_diffusion", shared_diffusion_result, truth),
        _summarize_result("normalized_only", normalized_only_result, truth),
    ]
    summary_table = pd.DataFrame(summary_rows).sort_values("model_name").reset_index(drop=True)

    thresholds = {
        "full_mass_error_max": 0.02,
        "full_mean_error_max": 0.03,
        "full_variance_error_max": 0.005,
        "react_mass_failure_min": 0.20,
        "diff_variance_failure_min": 0.010,
    }

    return Step3Evaluation(
        full_result=full_result,
        no_growth_result=no_growth_result,
        shared_diffusion_result=shared_diffusion_result,
        normalized_only_result=normalized_only_result,
        summary_table=summary_table,
        thresholds=thresholds,
    )


# Semantic aliases for the clearer software-facing API.
SingleScreenModelEvaluation = Step3Evaluation
evaluate_single_screen_model = evaluate_step3
