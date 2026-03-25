from __future__ import annotations

"""Visualization pipeline for CAMFND benchmark phases."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from camfnd.data.contract import PerturbSeqDynamicsData
from camfnd.pipeline import PipelineResult, run_full_pipeline


PHASE_COLORS = {
    "data_contract": "#0f766e",
    "simulator_validation": "#b45309",
    "single_screen_model": "#1d4ed8",
    "multiscreen_context_model": "#b91c1c",
}

MODEL_COLORS = {
    "full": "#0f766e",
    "no_growth": "#b45309",
    "shared_diffusion": "#7c3aed",
    "normalized_only": "#dc2626",
    "no_context": "#b91c1c",
}

TIME_COLORS = {
    "P4": "#2563eb",
    "P60": "#ea580c",
}

PERTURBATION_COLORS = {
    "ctrl": "#475569",
    "drift": "#0ea5e9",
    "diff": "#8b5cf6",
    "react": "#ef4444",
    "driver": "#f59e0b",
}


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    dpi: int = 160
    image_format: str = "png"
    transparent: bool = False
    close_figures: bool = True
    style: str = "default"


@dataclass(slots=True)
class VisualizationArtifacts:
    output_dir: Path
    report_path: Path
    manifest_path: Path
    figure_paths: Dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "manifest_path": str(self.manifest_path),
            "figure_paths": {key: str(path) for key, path in sorted(self.figure_paths.items())},
        }


@dataclass(slots=True)
class VisualizationPipelineResult:
    pipeline_result: PipelineResult
    artifacts: VisualizationArtifacts


def generate_pipeline_visualizations(
    result: PipelineResult,
    output_dir: str | Path,
    *,
    config: Optional[VisualizationConfig] = None,
) -> VisualizationArtifacts:
    """Render a complete benchmark visualization report from a PipelineResult."""

    cfg = config or VisualizationConfig()
    _apply_plot_style(cfg)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase_dirs = {
        "overview": out / "overview",
        "data_contract": out / "data_contract",
        "simulator_validation": out / "simulator_validation",
        "single_screen_model": out / "single_screen_model",
        "multiscreen_context_model": out / "multiscreen_context_model",
    }
    for directory in phase_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    figure_paths: Dict[str, Path] = {}

    _collect_paths(figure_paths, "overview", _render_overview_figures(result, phase_dirs["overview"], cfg))
    _collect_paths(
        figure_paths,
        "data_contract",
        _render_data_contract_figures(result.data_contract, result.single_screen_dataset, phase_dirs["data_contract"], cfg),
    )
    _collect_paths(
        figure_paths,
        "simulator_validation",
        _render_simulator_validation_figures(
            result.simulator_validation,
            result.single_screen_dataset,
            phase_dirs["simulator_validation"],
            cfg,
        ),
    )
    _collect_paths(
        figure_paths,
        "single_screen_model",
        _render_single_screen_model_figures(
            result.single_screen_model,
            result.single_screen_dataset,
            phase_dirs["single_screen_model"],
            cfg,
        ),
    )
    _collect_paths(
        figure_paths,
        "multiscreen_context_model",
        _render_multiscreen_context_figures(
            result.multiscreen_context_model,
            result.multiscreen_dataset,
            phase_dirs["multiscreen_context_model"],
            cfg,
        ),
    )

    report_path = out / "VISUALIZATION_REPORT.md"
    manifest_path = out / "visualization_manifest.json"
    _write_visualization_report(result, figure_paths, out, report_path)
    manifest_path.write_text(json.dumps({k: str(v) for k, v in sorted(figure_paths.items())}, indent=2))

    return VisualizationArtifacts(
        output_dir=out,
        report_path=report_path,
        manifest_path=manifest_path,
        figure_paths=figure_paths,
    )


def run_pipeline_with_visualizations(
    *,
    pipeline_output_dir: str | Path | None = None,
    visualization_output_dir: str | Path | None = None,
    visualization_config: Optional[VisualizationConfig] = None,
    **pipeline_kwargs,
) -> VisualizationPipelineResult:
    """Run the benchmark pipeline and immediately render the full visualization report."""

    pipeline_result = run_full_pipeline(output_dir=pipeline_output_dir, **pipeline_kwargs)
    viz_dir = Path(visualization_output_dir or Path(pipeline_output_dir or "./camfnd_outputs") / "visualizations")
    artifacts = generate_pipeline_visualizations(pipeline_result, viz_dir, config=visualization_config)
    return VisualizationPipelineResult(pipeline_result=pipeline_result, artifacts=artifacts)


def _apply_plot_style(config: VisualizationConfig) -> None:
    plt.style.use(config.style)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.facecolor": "#fffdf8",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def _collect_paths(target: Dict[str, Path], prefix: str, items: Mapping[str, Path]) -> None:
    for key, path in items.items():
        target[f"{prefix}/{key}"] = path


def _save_figure(fig: plt.Figure, path: Path, config: VisualizationConfig) -> Path:
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight", transparent=config.transparent)
    if config.close_figures:
        plt.close(fig)
    return path


def _placeholder(ax: plt.Axes, message: str, *, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#525252")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _group_labels(frame: pd.DataFrame, *, include_sample: bool = True) -> pd.Series:
    if include_sample and "sample_id" in frame.columns:
        return frame["sample_id"].astype(str) + "\n" + frame["perturbation_id"].astype(str)
    return frame["perturbation_id"].astype(str)


def _grouped_bar(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    category_col: str,
    hue_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    palette: Optional[Mapping[str, str]] = None,
) -> None:
    if frame.empty:
        _placeholder(ax, "No data available.", title=title)
        return

    pivot = frame.pivot_table(index=category_col, columns=hue_col, values=value_col, aggfunc="first")
    categories = pivot.index.tolist()
    hues = pivot.columns.tolist()
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(hues), 1)

    for idx, hue in enumerate(hues):
        values = pivot[hue].to_numpy(dtype=float)
        offset = (idx - (len(hues) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=(palette or {}).get(str(hue), "#64748b"),
            label=str(hue),
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y")


def _single_metric_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    color_col: Optional[str],
    title: str,
    ylabel: str,
    palette: Optional[Mapping[str, str]] = None,
    threshold: float | None = None,
) -> None:
    if frame.empty:
        _placeholder(ax, "No data available.", title=title)
        return

    colors = None
    if color_col is not None:
        colors = [(palette or {}).get(str(value), "#64748b") for value in frame[color_col].tolist()]

    x = np.arange(frame.shape[0], dtype=float)
    ax.bar(x, frame[value_col].astype(float).to_numpy(), color=colors or "#64748b")
    if threshold is not None:
        ax.axhline(threshold, color="#111827", linestyle="--", linewidth=1.2, label="threshold")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(frame[label_col].astype(str).tolist(), rotation=35, ha="right")
    ax.grid(axis="y")


def _parity_scatter(
    ax: plt.Axes,
    truth: Iterable[float],
    pred: Iterable[float],
    labels: Iterable[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#0f766e",
) -> None:
    truth_arr = np.asarray(list(truth), dtype=float)
    pred_arr = np.asarray(list(pred), dtype=float)
    if truth_arr.size == 0:
        _placeholder(ax, "No truth targets available.", title=title)
        return

    lo = float(min(truth_arr.min(), pred_arr.min()))
    hi = float(max(truth_arr.max(), pred_arr.max()))
    padding = 0.05 * max(hi - lo, 1e-8)
    ax.scatter(truth_arr, pred_arr, s=55, color=color, edgecolor="white", linewidth=0.7, alpha=0.95)
    ax.plot([lo - padding, hi + padding], [lo - padding, hi + padding], linestyle="--", color="#111827")
    for x, y, label in zip(truth_arr, pred_arr, labels):
        ax.annotate(str(label), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lo - padding, hi + padding)
    ax.set_ylim(lo - padding, hi + padding)
    ax.grid(True)


def _overlay_parity_scatter(
    ax: plt.Axes,
    truth: Iterable[float],
    series: Mapping[str, Iterable[float]],
    labels: Iterable[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    palette: Mapping[str, str],
) -> None:
    truth_arr = np.asarray(list(truth), dtype=float)
    series_arrays = {name: np.asarray(list(values), dtype=float) for name, values in series.items()}
    if truth_arr.size == 0:
        _placeholder(ax, "No truth targets available.", title=title)
        return

    lo = float(min([truth_arr.min(), *[values.min() for values in series_arrays.values()]]))
    hi = float(max([truth_arr.max(), *[values.max() for values in series_arrays.values()]]))
    padding = 0.05 * max(hi - lo, 1e-8)
    ax.plot([lo - padding, hi + padding], [lo - padding, hi + padding], linestyle="--", color="#111827")
    labels_list = list(labels)
    for model_name, pred_arr in series_arrays.items():
        ax.scatter(
            truth_arr,
            pred_arr,
            s=55,
            color=palette.get(model_name, "#64748b"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=model_name,
        )
        for x, y, label in zip(truth_arr, pred_arr, labels_list):
            ax.annotate(str(label), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lo - padding, hi + padding)
    ax.set_ylim(lo - padding, hi + padding)
    ax.legend(frameon=False)
    ax.grid(True)


def _render_overview_figures(
    result: PipelineResult,
    output_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    status_frame = pd.DataFrame(
        [
            {"phase": "data_contract", "ok": int(result.data_contract.ok)},
            {"phase": "simulator_validation", "ok": int(result.simulator_validation.ok)},
            {"phase": "single_screen_model", "ok": int(result.single_screen_model.ok)},
            {"phase": "multiscreen_context_model", "ok": int(result.multiscreen_context_model.ok)},
        ]
    )
    axes[0, 0].barh(
        status_frame["phase"],
        status_frame["ok"],
        color=[PHASE_COLORS[name] if ok else "#ef4444" for name, ok in zip(status_frame["phase"], status_frame["ok"])],
    )
    axes[0, 0].set_xlim(0, 1.05)
    axes[0, 0].set_title("Phase Status")
    axes[0, 0].set_xlabel("pass flag")

    full_row = result.single_screen_model.summary_table.set_index("model_name").loc["full"]
    full_metrics = pd.DataFrame(
        {
            "metric": ["mass", "mean", "variance"],
            "value": [
                float(full_row["mean_abs_mass_error"]),
                float(full_row["mean_abs_mean_error"]),
                float(full_row["mean_abs_variance_error"]),
            ],
            "threshold": [
                result.single_screen_model.thresholds["full_mass_error_max"],
                result.single_screen_model.thresholds["full_mean_error_max"],
                result.single_screen_model.thresholds["full_variance_error_max"],
            ],
        }
    )
    axes[0, 1].bar(full_metrics["metric"], full_metrics["value"], color=PHASE_COLORS["single_screen_model"])
    axes[0, 1].plot(full_metrics["metric"], full_metrics["threshold"], color="#111827", linestyle="--", marker="o")
    axes[0, 1].set_title("Single-Screen Full Model Errors")
    axes[0, 1].set_ylabel("absolute error")

    convergence = result.simulator_validation.convergence_table
    axes[1, 0].plot(convergence["n_steps"], convergence["mean_abs_error_mean"], marker="o", label="mean")
    axes[1, 0].plot(convergence["n_steps"], convergence["mean_abs_error_variance"], marker="o", label="variance")
    axes[1, 0].plot(convergence["n_steps"], convergence["mean_abs_error_mass"], marker="o", label="mass")
    axes[1, 0].set_title("Simulator Convergence")
    axes[1, 0].set_xlabel("n_steps")
    axes[1, 0].set_ylabel("mean absolute error")
    axes[1, 0].legend(frameon=False)

    screen_delta = result.multiscreen_context_model.summary_table.set_index("model_name")
    compare = pd.DataFrame(
        {
            "model_name": ["full", "no_context"],
            "screen_delta_error": [
                float(screen_delta.loc["full", "mean_abs_screen_delta_error"]),
                float(screen_delta.loc["no_context", "mean_abs_screen_delta_error"]),
            ],
        }
    )
    axes[1, 1].bar(
        compare["model_name"],
        compare["screen_delta_error"],
        color=[MODEL_COLORS["full"], MODEL_COLORS["no_context"]],
    )
    axes[1, 1].axhline(
        result.multiscreen_context_model.thresholds["screen_delta_error_max"],
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="full threshold",
    )
    axes[1, 1].set_title("Context-Sensitive Screen Delta Error")
    axes[1, 1].set_ylabel("absolute delta-mean error")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("CAMFND Benchmark Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    paths["benchmark_overview"] = _save_figure(fig, output_dir / f"benchmark_overview.{config.image_format}", config)
    return paths


def _render_data_contract_figures(
    evaluation,
    dataset: PerturbSeqDynamicsData,
    output_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    summary = evaluation.count_summary.copy()
    summary["group"] = _group_labels(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _grouped_bar(
        axes[0],
        summary,
        category_col="group",
        hue_col="time_label",
        value_col="n_cells",
        title="Observed Cell Counts by Endpoint Group",
        ylabel="cell count",
        palette=TIME_COLORS,
    )
    _grouped_bar(
        axes[1],
        summary,
        category_col="group",
        hue_col="time_label",
        value_col="mass",
        title="Guide-Abundance Mass by Endpoint Group",
        ylabel="mass",
        palette=TIME_COLORS,
    )
    fig.suptitle("Data Contract: Counts and Masses", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["counts_and_mass"] = _save_figure(fig, output_dir / f"counts_and_mass.{config.image_format}", config)

    terminal = evaluation.empirical_terminal_summary.copy()
    terminal["group"] = _group_labels(terminal)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _single_metric_bars(
        axes[0],
        terminal,
        label_col="group",
        value_col="terminal_mean",
        color_col="perturbation_id",
        title="Terminal Mean",
        ylabel="mean",
        palette=PERTURBATION_COLORS,
    )
    _single_metric_bars(
        axes[1],
        terminal,
        label_col="group",
        value_col="terminal_variance",
        color_col="perturbation_id",
        title="Terminal Variance",
        ylabel="variance",
        palette=PERTURBATION_COLORS,
    )
    _single_metric_bars(
        axes[2],
        terminal,
        label_col="group",
        value_col="terminal_mass",
        color_col="perturbation_id",
        title="Terminal Mass",
        ylabel="mass",
        palette=PERTURBATION_COLORS,
    )
    fig.suptitle("Data Contract: Terminal Measure Moments", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["terminal_moments"] = _save_figure(fig, output_dir / f"terminal_moments.{config.image_format}", config)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    analytic = evaluation.analytic_comparison.copy()
    labels = _group_labels(analytic).tolist() if not analytic.empty else []
    _parity_scatter(
        axes[0],
        analytic.get("truth_terminal_mean", []),
        analytic.get("empirical_terminal_mean", []),
        labels,
        title="Empirical vs Analytic Mean",
        xlabel="analytic mean",
        ylabel="empirical mean",
        color=PHASE_COLORS["data_contract"],
    )
    _parity_scatter(
        axes[1],
        analytic.get("truth_terminal_variance", []),
        analytic.get("empirical_terminal_variance", []),
        labels,
        title="Empirical vs Analytic Variance",
        xlabel="analytic variance",
        ylabel="empirical variance",
        color=PHASE_COLORS["data_contract"],
    )
    _parity_scatter(
        axes[2],
        analytic.get("truth_terminal_mass", []),
        analytic.get("empirical_terminal_mass", []),
        labels,
        title="Empirical vs Analytic Mass",
        xlabel="analytic mass",
        ylabel="empirical mass",
        color=PHASE_COLORS["data_contract"],
    )
    fig.suptitle("Data Contract: Analytic Parity Checks", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["analytic_parity"] = _save_figure(fig, output_dir / f"analytic_parity.{config.image_format}", config)

    checks = pd.DataFrame(
        {
            "check": list(evaluation.signature_checks.keys()),
            "passed": [int(value) for value in evaluation.signature_checks.values()],
        }
    )
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.6 * max(checks.shape[0], 1))))
    if checks.empty:
        _placeholder(ax, "No signature checks available.", title="Qualitative Signature Checks")
    else:
        ax.barh(
            checks["check"],
            checks["passed"],
            color=["#16a34a" if value else "#dc2626" for value in checks["passed"]],
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass flag")
        ax.set_title("Qualitative Signature Checks")
    fig.tight_layout()
    paths["signature_checks"] = _save_figure(fig, output_dir / f"signature_checks.{config.image_format}", config)

    return paths


def _render_simulator_validation_figures(
    evaluation,
    dataset: PerturbSeqDynamicsData,
    output_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    init_table = evaluation.initialization_table.copy()
    init_table["group"] = _group_labels(init_table)
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.6 * max(init_table.shape[0], 1))))
    if init_table.empty:
        _placeholder(ax, "No initialization table available.", title="Initialization Exactness")
    else:
        ax.barh(
            init_table["group"],
            init_table["exact"].astype(int),
            color=["#16a34a" if flag else "#dc2626" for flag in init_table["exact"]],
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("exact reconstruction flag")
        ax.set_title("Simulator Validation: Initialization Exactness")
    fig.tight_layout()
    paths["initialization_exactness"] = _save_figure(fig, output_dir / f"initialization_exactness.{config.image_format}", config)

    comp = evaluation.default_analytic_comparison.copy()
    labels = _group_labels(comp).tolist() if not comp.empty else []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _parity_scatter(
        axes[0],
        comp.get("truth_terminal_mean", []),
        comp.get("terminal_mean_0", []),
        labels,
        title="Default Run Mean",
        xlabel="analytic mean",
        ylabel="simulated mean",
        color=PHASE_COLORS["simulator_validation"],
    )
    _parity_scatter(
        axes[1],
        comp.get("truth_terminal_variance", []),
        comp.get("terminal_var_trace", []),
        labels,
        title="Default Run Variance",
        xlabel="analytic variance",
        ylabel="simulated variance",
        color=PHASE_COLORS["simulator_validation"],
    )
    _parity_scatter(
        axes[2],
        comp.get("truth_terminal_mass", []),
        comp.get("terminal_mass", []),
        labels,
        title="Default Run Mass",
        xlabel="analytic mass",
        ylabel="simulated mass",
        color=PHASE_COLORS["simulator_validation"],
    )
    fig.suptitle("Simulator Validation: Default Run Parity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["default_run_parity"] = _save_figure(fig, output_dir / f"default_run_parity.{config.image_format}", config)

    convergence = evaluation.convergence_table.copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("mean_abs_error_mean", "Mean Error"),
        ("mean_abs_error_variance", "Variance Error"),
        ("mean_abs_error_mass", "Mass Error"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(convergence["n_steps"], convergence[metric], marker="o", color=PHASE_COLORS["simulator_validation"])
        ax.set_title(title)
        ax.set_xlabel("n_steps")
        ax.set_ylabel("mean absolute error")
    fig.suptitle("Simulator Validation: Convergence with Step Count", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["convergence"] = _save_figure(fig, output_dir / f"convergence.{config.image_format}", config)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    comp["group"] = _group_labels(comp)
    error_specs = [
        ("abs_error_mean", "Mean Error"),
        ("abs_error_variance", "Variance Error"),
        ("abs_error_mass", "Mass Error"),
    ]
    for ax, (metric, title) in zip(axes, error_specs):
        _single_metric_bars(
            ax,
            comp,
            label_col="group",
            value_col=metric,
            color_col="perturbation_id",
            title=title,
            ylabel="absolute error",
            palette=PERTURBATION_COLORS,
        )
    fig.suptitle("Simulator Validation: Default Run Error Profile", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["default_run_errors"] = _save_figure(fig, output_dir / f"default_run_errors.{config.image_format}", config)

    return paths


def _render_single_screen_model_figures(
    evaluation,
    dataset: PerturbSeqDynamicsData,
    output_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    histories = {
        "full": evaluation.full_result.history,
        "no_growth": evaluation.no_growth_result.history,
        "shared_diffusion": evaluation.shared_diffusion_result.history,
        "normalized_only": evaluation.normalized_only_result.history,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    history_metrics = [
        ("total_loss", "Total Loss"),
        ("endpoint_loss", "Endpoint Loss"),
        ("aux_loss", "Auxiliary Loss"),
        ("reg_total", "Regularization"),
    ]
    for ax, (metric, title) in zip(axes.flat, history_metrics):
        for model_name, history in histories.items():
            if metric in history.columns:
                ax.plot(history["epoch"], history[metric], label=model_name, color=MODEL_COLORS.get(model_name, "#64748b"))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Single-Screen Model: Training Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["training_dynamics"] = _save_figure(fig, output_dir / f"training_dynamics.{config.image_format}", config)

    full_table = evaluation.full_result.final_loss_table.copy()
    labels = _group_labels(full_table).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _parity_scatter(
        axes[0],
        full_table.get("target_mean_0", []),
        full_table.get("pred_mean_0", []),
        labels,
        title="Full Model Mean",
        xlabel="truth mean",
        ylabel="predicted mean",
        color=MODEL_COLORS["full"],
    )
    _parity_scatter(
        axes[1],
        full_table.get("target_var_trace", []),
        full_table.get("pred_var_trace", []),
        labels,
        title="Full Model Variance",
        xlabel="truth variance",
        ylabel="predicted variance",
        color=MODEL_COLORS["full"],
    )
    _parity_scatter(
        axes[2],
        full_table.get("target_mass", []),
        full_table.get("pred_mass", []),
        labels,
        title="Full Model Mass",
        xlabel="truth mass",
        ylabel="predicted mass",
        color=MODEL_COLORS["full"],
    )
    fig.suptitle("Single-Screen Model: Full-Model Terminal Parity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["full_model_parity"] = _save_figure(fig, output_dir / f"full_model_parity.{config.image_format}", config)

    summary = evaluation.summary_table.copy()
    metric_specs = [
        ("mean_abs_mass_error", "Mean Mass Error"),
        ("mean_abs_mean_error", "Mean Mean Error"),
        ("mean_abs_variance_error", "Mean Variance Error"),
        ("react_mass_error", "React Mass Error"),
        ("diff_variance_error", "Diff Variance Error"),
        ("endpoint_loss_mean", "Endpoint Loss Mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        _single_metric_bars(
            ax,
            summary,
            label_col="model_name",
            value_col=metric,
            color_col="model_name",
            title=title,
            ylabel=metric,
            palette=MODEL_COLORS,
        )
    fig.suptitle("Single-Screen Model: Ablation Metric Scorecard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["ablation_metrics"] = _save_figure(fig, output_dir / f"ablation_metrics.{config.image_format}", config)

    signature_specs = [
        ("drift_mean_minus_ctrl", "Drift Mean - Ctrl", 0.10),
        ("diff_variance_minus_ctrl", "Diff Variance - Ctrl", 0.005),
        ("ctrl_mass_minus_react", "Ctrl Mass - React", 0.10),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title, threshold) in zip(axes, signature_specs):
        _single_metric_bars(
            ax,
            summary,
            label_col="model_name",
            value_col=metric,
            color_col="model_name",
            title=title,
            ylabel="gap",
            palette=MODEL_COLORS,
            threshold=threshold,
        )
    fig.suptitle("Single-Screen Model: Signature Recovery Gaps", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["signature_gaps"] = _save_figure(fig, output_dir / f"signature_gaps.{config.image_format}", config)

    return paths


def _render_multiscreen_context_figures(
    evaluation,
    dataset: PerturbSeqDynamicsData,
    output_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    histories = {
        "full": evaluation.full_result.history,
        "no_context": evaluation.no_context_result.history,
    }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    history_metrics = [
        ("total_loss", "Total Loss"),
        ("endpoint_loss", "Endpoint Loss"),
        ("aux_loss", "Auxiliary Loss"),
        ("screen_delta_loss", "Screen Delta Loss"),
        ("eta", "Estimated Eta"),
        ("kappa", "Estimated Kappa"),
    ]
    for ax, (metric, title) in zip(axes.flat, history_metrics):
        for model_name, history in histories.items():
            if metric in history.columns:
                ax.plot(history["epoch"], history[metric], label=model_name, color=MODEL_COLORS.get(model_name, "#64748b"))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Multiscreen Context Model: Training Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["training_dynamics"] = _save_figure(fig, output_dir / f"training_dynamics.{config.image_format}", config)

    truth = _terminal_truth_table(dataset)
    full_delta = _screen_delta_table(evaluation.full_result.final_simulation.summary, truth, model_name="full")
    no_context_delta = _screen_delta_table(evaluation.no_context_result.final_simulation.summary, truth, model_name="no_context")
    delta_frame = pd.concat([full_delta, no_context_delta], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    if delta_frame.empty:
        _placeholder(axes[0], "No two-screen delta information available.", title="Predicted Screen Deltas")
        _placeholder(axes[1], "No two-screen delta information available.", title="Screen Delta Errors")
    else:
        pred_plot = delta_frame[["perturbation_id", "model_name", "pred_delta_mean"]].copy()
        pred_plot = pd.concat(
            [
                pred_plot,
                full_delta[["perturbation_id", "truth_delta_mean"]].assign(model_name="truth", pred_delta_mean=full_delta["truth_delta_mean"])[
                    ["perturbation_id", "model_name", "pred_delta_mean"]
                ],
            ],
            ignore_index=True,
        )
        _grouped_bar(
            axes[0],
            pred_plot,
            category_col="perturbation_id",
            hue_col="model_name",
            value_col="pred_delta_mean",
            title="Screen-2 Minus Screen-1 Mean Shift",
            ylabel="delta mean",
            palette={**MODEL_COLORS, "truth": "#111827"},
        )
        _grouped_bar(
            axes[1],
            delta_frame,
            category_col="perturbation_id",
            hue_col="model_name",
            value_col="abs_delta_mean_error",
            title="Absolute Screen Delta Error",
            ylabel="absolute error",
            palette=MODEL_COLORS,
        )
        axes[1].axhline(
            evaluation.thresholds["screen_delta_error_max"],
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label="full threshold",
        )
        axes[1].legend(frameon=False)
    fig.suptitle("Multiscreen Context Model: Cross-Screen Effect Recovery", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["screen_delta_comparison"] = _save_figure(fig, output_dir / f"screen_delta_comparison.{config.image_format}", config)

    fig = _plot_context_trajectories(
        evaluation.full_result.final_simulation.context_summary,
        evaluation.no_context_result.final_simulation.context_summary,
        dataset.truth.context_trajectories if dataset.truth is not None else None,
    )
    fig.suptitle("Multiscreen Context Model: Context Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["context_trajectories"] = _save_figure(fig, output_dir / f"context_trajectories.{config.image_format}", config)

    full_table = evaluation.full_result.final_loss_table.copy()
    no_context_table = evaluation.no_context_result.final_loss_table.copy()
    labels = _group_labels(full_table).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _overlay_parity_scatter(
        axes[0],
        full_table.get("target_mean_0", []),
        {"full": full_table.get("pred_mean_0", []), "no_context": no_context_table.get("pred_mean_0", [])},
        labels,
        title="Terminal Mean Parity",
        xlabel="truth mean",
        ylabel="predicted mean",
        palette=MODEL_COLORS,
    )
    _overlay_parity_scatter(
        axes[1],
        full_table.get("target_var_trace", []),
        {"full": full_table.get("pred_var_trace", []), "no_context": no_context_table.get("pred_var_trace", [])},
        labels,
        title="Terminal Variance Parity",
        xlabel="truth variance",
        ylabel="predicted variance",
        palette=MODEL_COLORS,
    )
    _overlay_parity_scatter(
        axes[2],
        full_table.get("target_mass", []),
        {"full": full_table.get("pred_mass", []), "no_context": no_context_table.get("pred_mass", [])},
        labels,
        title="Terminal Mass Parity",
        xlabel="truth mass",
        ylabel="predicted mass",
        palette=MODEL_COLORS,
    )
    fig.suptitle("Multiscreen Context Model: Full vs No-Context Terminal Parity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths["terminal_parity"] = _save_figure(fig, output_dir / f"terminal_parity.{config.image_format}", config)

    return paths


def _plot_context_trajectories(
    full_context: pd.DataFrame,
    no_context: pd.DataFrame,
    truth_context: Optional[pd.DataFrame],
) -> plt.Figure:
    sample_ids = sorted(set(full_context["sample_id"].astype(str).tolist()))
    n_samples = max(len(sample_ids), 1)
    fig, axes = plt.subplots(1, n_samples, figsize=(7.2 * n_samples, 4.8), squeeze=False)
    for ax, sample_id in zip(axes.flat, sample_ids):
        full_sub = full_context.loc[full_context["sample_id"].astype(str) == sample_id]
        no_sub = no_context.loc[no_context["sample_id"].astype(str) == sample_id]
        ax.plot(full_sub["time"], full_sub["context"], color=MODEL_COLORS["full"], linewidth=2.0, label="full")
        ax.plot(no_sub["time"], no_sub["context"], color=MODEL_COLORS["no_context"], linewidth=2.0, label="no_context")
        if truth_context is not None and not truth_context.empty:
            truth_sub = truth_context.loc[truth_context["sample_id"].astype(str) == sample_id]
            if not truth_sub.empty:
                ax.plot(truth_sub["time"], truth_sub["context"], color="#111827", linestyle="--", linewidth=1.7, label="truth")
        ax.set_title(f"sample {sample_id}")
        ax.set_xlabel("time")
        ax.set_ylabel("context")
        ax.legend(frameon=False)
    return fig


def _terminal_truth_table(dataset: PerturbSeqDynamicsData) -> pd.DataFrame:
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


def _screen_delta_table(summary: pd.DataFrame, truth: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    pred = summary[
        ["sample_id", "perturbation_id", "terminal_mean_0"]
    ].rename(columns={"terminal_mean_0": "pred_terminal_mean"}).copy()
    merged = pred.merge(truth, on=["sample_id", "perturbation_id"], how="left")
    rows = []
    for perturbation_id, sub in merged.groupby("perturbation_id"):
        sub = sub.sort_values("sample_id").reset_index(drop=True)
        if sub.shape[0] != 2:
            continue
        pred_delta = float(sub.loc[1, "pred_terminal_mean"] - sub.loc[0, "pred_terminal_mean"])
        truth_delta = float(sub.loc[1, "terminal_mean"] - sub.loc[0, "terminal_mean"])
        rows.append(
            {
                "model_name": model_name,
                "perturbation_id": str(perturbation_id),
                "pred_delta_mean": pred_delta,
                "truth_delta_mean": truth_delta,
                "abs_delta_mean_error": abs(pred_delta - truth_delta),
            }
        )
    return pd.DataFrame(rows).sort_values("perturbation_id").reset_index(drop=True)


def _write_visualization_report(
    result: PipelineResult,
    figure_paths: Mapping[str, Path],
    output_dir: Path,
    report_path: Path,
) -> None:
    def rel(key: str) -> str:
        return figure_paths[key].relative_to(output_dir).as_posix()

    lines = [
        "# CAMFND Visualization Report",
        "",
        "This report is generated by `camfnd.visualization.pipeline` and is organized around the four benchmark phases.",
        "",
        "## Phase Summary",
        "",
        f"- `data_contract`: `{result.data_contract.ok}`",
        f"- `simulator_validation`: `{result.simulator_validation.ok}`",
        f"- `single_screen_model`: `{result.single_screen_model.ok}`",
        f"- `multiscreen_context_model`: `{result.multiscreen_context_model.ok}`",
        "",
        "## Overview",
        "",
        f"![Benchmark Overview]({rel('overview/benchmark_overview')})",
        "",
        "## Data Contract",
        "",
        "Questions answered:",
        "- Are endpoint cell counts fixed while masses remain distinct?",
        "- Do terminal mean, variance, and mass signatures match the intended perturbation semantics?",
        "- Do empirical summaries agree with analytic truth?",
        "",
        f"![Counts And Mass]({rel('data_contract/counts_and_mass')})",
        "",
        f"![Terminal Moments]({rel('data_contract/terminal_moments')})",
        "",
        f"![Analytic Parity]({rel('data_contract/analytic_parity')})",
        "",
        f"![Signature Checks]({rel('data_contract/signature_checks')})",
        "",
        "## Simulator Validation",
        "",
        "Questions answered:",
        "- Does particle initialization reconstruct the input measure exactly?",
        "- Does the default simulator match the analytic benchmark moments?",
        "- Do errors decrease with finer time discretization?",
        "",
        f"![Initialization Exactness]({rel('simulator_validation/initialization_exactness')})",
        "",
        f"![Default Run Parity]({rel('simulator_validation/default_run_parity')})",
        "",
        f"![Convergence]({rel('simulator_validation/convergence')})",
        "",
        f"![Default Run Errors]({rel('simulator_validation/default_run_errors')})",
        "",
        "## Single-Screen Model",
        "",
        "Questions answered:",
        "- How do the full model and ablations optimize during training?",
        "- Does the full model recover the truth terminal measures?",
        "- Which ablations fail on mass or variance in the expected way?",
        "",
        f"![Training Dynamics]({rel('single_screen_model/training_dynamics')})",
        "",
        f"![Full Model Parity]({rel('single_screen_model/full_model_parity')})",
        "",
        f"![Ablation Metrics]({rel('single_screen_model/ablation_metrics')})",
        "",
        f"![Signature Gaps]({rel('single_screen_model/signature_gaps')})",
        "",
        "## Multiscreen Context Model",
        "",
        "Questions answered:",
        "- How do the context-aware and no-context models optimize over training?",
        "- Does context coupling recover the cross-screen delta-mean effects?",
        "- Do the learned context trajectories match the truth trajectory shape?",
        "",
        f"![Training Dynamics]({rel('multiscreen_context_model/training_dynamics')})",
        "",
        f"![Screen Delta Comparison]({rel('multiscreen_context_model/screen_delta_comparison')})",
        "",
        f"![Context Trajectories]({rel('multiscreen_context_model/context_trajectories')})",
        "",
        f"![Terminal Parity]({rel('multiscreen_context_model/terminal_parity')})",
        "",
    ]
    report_path.write_text("\n".join(lines))
