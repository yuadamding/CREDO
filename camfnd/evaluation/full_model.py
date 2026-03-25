from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import pandas as pd

from camfnd.data.contract import PerturbSeqDynamicsData
from camfnd.data.multiscreen_benchmark import Stage2BenchmarkConfig, generate_stage2_dataset
from camfnd.training.full_model import FullModelTrainConfig, FullModelTrainingResult, train_full_model


@dataclass(slots=True)
class FullModelEvaluation:
    full_result: FullModelTrainingResult
    no_context_result: FullModelTrainingResult
    summary_table: pd.DataFrame
    thresholds: Dict[str, float]

    @property
    def full_pass(self) -> bool:
        row = self.summary_table.set_index("model_name").loc["full"]
        return bool(
            row["stable"]
            and row["control_anchor_exact"]
            and row["control_screen2_minus_screen1"] >= self.thresholds["control_shift_min"]
            and row["control_screen2_minus_screen1_error"] <= self.thresholds["control_shift_error_max"]
            and row["mean_abs_screen_delta_error"] <= self.thresholds["screen_delta_error_max"]
            and row["mean_abs_mean_error"] <= self.thresholds["full_mean_error_max"]
            and row["mean_abs_mass_error"] <= self.thresholds["full_mass_error_max"]
            and row["mean_abs_variance_error"] <= self.thresholds["full_variance_error_max"]
        )

    @property
    def no_context_fails_as_expected(self) -> bool:
        table = self.summary_table.set_index("model_name")
        no_ctx = table.loc["no_context"]
        full = table.loc["full"]
        return bool(
            no_ctx["mean_abs_screen_delta_error"] >= self.thresholds["no_context_screen_delta_error_min"]
            and no_ctx["control_screen2_minus_screen1_error"] >= self.thresholds["no_context_control_delta_error_min"]
            and full["mean_abs_screen_delta_error"] < no_ctx["mean_abs_screen_delta_error"]
            and full["endpoint_loss_mean"] < no_ctx["endpoint_loss_mean"]
        )

    @property
    def ok(self) -> bool:
        return bool(self.full_pass and self.no_context_fails_as_expected)

    def to_dict(self) -> dict:
        return {
            "full_pass": self.full_pass,
            "no_context_fails_as_expected": self.no_context_fails_as_expected,
            "ok": self.ok,
            "thresholds": dict(self.thresholds),
        }


def _truth_table(dataset: PerturbSeqDynamicsData) -> pd.DataFrame:
    problem = dataset.to_endpoint_problem(by_sample=True)
    rows = []
    for key, measure in problem.terminal.items():
        row = {
            "sample_id": measure.sample_id,
            "perturbation_id": measure.perturbation_id,
            "terminal_mass": float(measure.total_mass),
            "terminal_variance": float(measure.variance_trace()),
        }
        mean = measure.mean().reshape(-1)
        for dim_idx, value in enumerate(mean):
            row[f"terminal_mean_{dim_idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _mean_abs_context_value(result: FullModelTrainingResult) -> float:
    context = result.final_simulation.context_summary
    context_cols = [column for column in context.columns if column.startswith("context_")]
    if not context_cols:
        return 0.0
    return float(context[context_cols].abs().mean().mean())


def _summarize_result(model_name: str, result: FullModelTrainingResult, truth: pd.DataFrame) -> dict:
    pred = result.final_simulation.summary[
        ["sample_id", "perturbation_id", "terminal_mass", "terminal_mean_0", "terminal_var_trace"]
    ].copy().rename(
        columns={
            "terminal_mass": "terminal_mass_pred",
            "terminal_mean_0": "terminal_mean_pred",
            "terminal_var_trace": "terminal_variance_pred",
        }
    )
    merged = pred.merge(
        truth[["sample_id", "perturbation_id", "terminal_mass", "terminal_mean_0", "terminal_variance"]],
        on=["sample_id", "perturbation_id"],
        how="left",
    )
    merged["abs_mass_error"] = (merged["terminal_mass_pred"] - merged["terminal_mass"]).abs()
    merged["abs_mean_error"] = (merged["terminal_mean_pred"] - merged["terminal_mean_0"]).abs()
    merged["abs_variance_error"] = (merged["terminal_variance_pred"] - merged["terminal_variance"]).abs()

    delta_rows = []
    for pid in truth["perturbation_id"].unique():
        sub = merged.loc[merged["perturbation_id"] == pid].sort_values("sample_id")
        if sub.shape[0] != 2:
            continue
        s1 = sub.iloc[0]
        s2 = sub.iloc[1]
        delta_rows.append(
            {
                "perturbation_id": pid,
                "pred_delta_mean": float(s2["terminal_mean_pred"] - s1["terminal_mean_pred"]),
                "truth_delta_mean": float(s2["terminal_mean_0"] - s1["terminal_mean_0"]),
                "abs_delta_mean_error": float(
                    abs(
                        (s2["terminal_mean_pred"] - s1["terminal_mean_pred"])
                        - (s2["terminal_mean_0"] - s1["terminal_mean_0"])
                    )
                ),
            }
        )
    delta = pd.DataFrame(delta_rows).set_index("perturbation_id")
    score_perturbations = [pid for pid in ("ctrl", "drift", "diff", "react") if pid in delta.index]
    if not score_perturbations:
        score_perturbations = list(delta.index)

    return {
        "model_name": model_name,
        "stable": bool(result.final_simulation.stable),
        "control_anchor_exact": bool(result.model.control_anchor_is_exact()),
        "mean_abs_mass_error": float(merged["abs_mass_error"].mean()),
        "mean_abs_mean_error": float(merged["abs_mean_error"].mean()),
        "mean_abs_variance_error": float(merged["abs_variance_error"].mean()),
        "mean_abs_screen_delta_error": float(delta.loc[score_perturbations, "abs_delta_mean_error"].mean()),
        "control_screen2_minus_screen1": float(delta.loc["ctrl", "pred_delta_mean"]),
        "control_screen2_minus_screen1_error": float(delta.loc["ctrl", "abs_delta_mean_error"]),
        "drift_screen2_minus_screen1": float(delta.loc["drift", "pred_delta_mean"]),
        "diff_screen2_minus_screen1": float(delta.loc["diff", "pred_delta_mean"]),
        "react_screen2_minus_screen1": float(delta.loc["react", "pred_delta_mean"]),
        "driver_screen2_minus_screen1": float(delta.loc["driver", "pred_delta_mean"]),
        "endpoint_loss_mean": float(result.final_loss_table["endpoint_loss"].mean()),
        "best_total_loss": float(result.history["total_loss"].min()),
        "mean_abs_context_value": _mean_abs_context_value(result),
    }


def evaluate_full_model(
    dataset: Optional[PerturbSeqDynamicsData] = None,
    *,
    full_config: Optional[FullModelTrainConfig] = None,
) -> FullModelEvaluation:
    """Direct acceptance harness for the full model path on the Stage-II benchmark."""

    if dataset is None:
        dataset = generate_stage2_dataset(
            Stage2BenchmarkConfig(
                seed=29,
                n_obs_p4=32,
                n_obs_p60=32,
                n_truth_particles=1024,
                n_steps=48,
                infer_latent_transform=True,
            )
        )
    dataset.validate()
    truth = _truth_table(dataset)

    full_config = full_config or FullModelTrainConfig(device="cpu")
    full_config.validate()

    full_result = train_full_model(dataset, config=full_config)
    no_context_result = train_full_model(
        dataset,
        config=replace(full_config, use_context=False, aux_screen_delta_mean_weight=0.0),
    )

    summary_rows = [
        _summarize_result("full", full_result, truth),
        _summarize_result("no_context", no_context_result, truth),
    ]
    summary_table = pd.DataFrame(summary_rows).sort_values("model_name").reset_index(drop=True)

    thresholds = {
        "control_shift_min": 0.02,
        "control_shift_error_max": 0.03,
        "screen_delta_error_max": 0.03,
        "full_mean_error_max": 0.05,
        "full_mass_error_max": 0.05,
        "full_variance_error_max": 0.03,
        "no_context_screen_delta_error_min": 0.05,
        "no_context_control_delta_error_min": 0.045,
    }

    return FullModelEvaluation(
        full_result=full_result,
        no_context_result=no_context_result,
        summary_table=summary_table,
        thresholds=thresholds,
    )
