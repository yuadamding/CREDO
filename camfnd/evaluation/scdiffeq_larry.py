from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch

from camfnd.data.contract import (
    CellStateTable,
    FiniteMeasure,
    Key,
    LatentTransform,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from camfnd.models.sinkhorn import normalized_geometry_loss, unbalanced_sinkhorn_divergence
from camfnd.training.full_model import FullModelTrainConfig, FullModelTrainingResult, train_full_model


LARRY_DEFAULT_PATH = Path("/home/yding1995/opscc_sc/scDiffeq/KleinLabData/scdiffeq_data/larry/larry.h5ad")
_SAMPLE_ID = "larry_quickstart_nm_clone_fate"
_TIME_INITIAL = "D2"
_TIME_TERMINAL = "D6"
_DOMINANT_FATE_TO_PERTURBATION = {
    "Undifferentiated": "ctrl",
    "Monocyte": "monocyte",
    "Neutrophil": "neutrophil",
}


@dataclass(slots=True)
class ScDiffEqLarryEvaluation:
    train_dataset: PerturbSeqDynamicsData
    dataset_summary: pd.DataFrame
    full_result: FullModelTrainingResult
    no_context_result: FullModelTrainingResult
    holdout_terminal: Dict[Key, FiniteMeasure]
    summary_table: pd.DataFrame
    baseline_detail_table: pd.DataFrame
    full_detail_table: pd.DataFrame
    no_context_detail_table: pd.DataFrame
    metadata: Dict[str, object]

    @property
    def ok(self) -> bool:
        table = self.summary_table.set_index("model_name")
        return bool(table.loc["full", "stable"] and table.loc["no_context", "stable"])

    @property
    def context_better_on_holdout(self) -> bool:
        table = self.summary_table.set_index("model_name")
        return bool(
            table.loc["full", "mean_holdout_endpoint_loss"] < table.loc["no_context", "mean_holdout_endpoint_loss"]
            and table.loc["full", "mean_l2_mean_error"] <= table.loc["no_context", "mean_l2_mean_error"]
        )

    @property
    def best_camfnd_beats_identity(self) -> bool:
        table = self.summary_table.set_index("model_name")
        best_camfnd = min(
            float(table.loc["full", "mean_holdout_normalized_loss"]),
            float(table.loc["no_context", "mean_holdout_normalized_loss"]),
        )
        baseline = float(table.loc["identity_baseline", "mean_holdout_normalized_loss"])
        return bool(best_camfnd < baseline)


def _mean_abs_context_value(result: FullModelTrainingResult) -> float:
    context = result.final_simulation.context_summary
    context_cols = [column for column in context.columns if column.startswith("context_")]
    if not context_cols:
        return 0.0
    return float(context[context_cols].abs().mean().mean())


def _read_larry_obs(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    x_pca = np.asarray(adata.obsm["X_pca"], dtype=float)
    fate_counts = adata.uns["fate_counts"][["Monocyte", "Neutrophil", "Undifferentiated"]].copy()
    return obs, x_pca, fate_counts


def _quickstart_clone_subset(obs: pd.DataFrame, fate_counts: pd.DataFrame) -> pd.DataFrame:
    dominant_counts = fate_counts.fillna(0.0)
    dominant_fate = dominant_counts.idxmax(axis=1)
    dominant_fate = dominant_fate[dominant_counts.max(axis=1) > 0]
    nm_clones = fate_counts[["Monocyte", "Neutrophil"]].dropna().index

    out = obs.copy()
    out["nm_clones"] = out["clone_idx"].isin(nm_clones)
    out["dominant_fate"] = out["clone_idx"].map(dominant_fate)

    mask = (
        out["Cell type annotation"].isin(["Monocyte", "Neutrophil", "Undifferentiated"])
        & out["nm_clones"]
        & out["dominant_fate"].isin(_DOMINANT_FATE_TO_PERTURBATION)
    )
    out = out.loc[mask].copy()
    out["perturbation_id"] = out["dominant_fate"].map(_DOMINANT_FATE_TO_PERTURBATION)
    out["sample_id"] = _SAMPLE_ID
    out["time_label"] = np.where(out["Time point"].eq(2.0), _TIME_INITIAL, np.where(out["Time point"].eq(6.0), _TIME_TERMINAL, "OTHER"))
    out = out[out["time_label"].isin([_TIME_INITIAL, _TIME_TERMINAL])].copy()
    out["cell_id"] = out.index.astype(str)
    return out


def _split_terminal_indices(indices: np.ndarray, *, rng: np.random.Generator, train_n: int, eval_n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < train_n + eval_n:
        raise ValueError(
            f"Terminal group has {len(indices)} cells, but train_n + eval_n = {train_n + eval_n}."
        )
    perm = rng.permutation(indices)
    return perm[:train_n], perm[train_n : train_n + eval_n]


def _build_measure(
    *,
    support: np.ndarray,
    perturbation_id: str,
    time_label: str,
    sample_id: str,
    total_mass: float,
) -> FiniteMeasure:
    weights = np.full(support.shape[0], float(total_mass) / support.shape[0], dtype=float)
    measure = FiniteMeasure(
        support=support.astype(float),
        weights=weights,
        total_mass=float(total_mass),
        perturbation_id=str(perturbation_id),
        time_label=str(time_label),
        sample_id=str(sample_id),
    )
    measure.validate()
    return measure


def build_scdiffeq_larry_camfnd_dataset(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    max_initial_cells_per_measure: Optional[int] = None,
    train_terminal_cells_per_measure: int = 96,
    eval_terminal_cells_per_measure: int = 256,
    seed: int = 17,
) -> tuple[PerturbSeqDynamicsData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"scDiffEq LARRY dataset not found at {data_path}.")
    if int(latent_dims) <= 0:
        raise ValueError("latent_dims must be positive.")

    obs, x_pca, fate_counts = _read_larry_obs(data_path)
    subset = _quickstart_clone_subset(obs, fate_counts)
    rng = np.random.default_rng(seed)

    if latent_dims > x_pca.shape[1]:
        raise ValueError(f"Requested latent_dims={latent_dims}, but X_pca only has {x_pca.shape[1]} columns.")

    obs_rows: list[dict] = []
    z_rows: list[np.ndarray] = []
    mass_rows: list[dict] = []
    holdout_terminal: Dict[Key, FiniteMeasure] = {}
    summary_rows: list[dict] = []

    train_df = subset.copy()
    train_df["row_index"] = np.arange(train_df.shape[0], dtype=int)
    train_df["source_index"] = train_df.index.astype(str)

    for perturbation_id in ["ctrl", "monocyte", "neutrophil"]:
        initial_group = train_df[
            (train_df["perturbation_id"] == perturbation_id) & (train_df["time_label"] == _TIME_INITIAL)
        ]
        terminal_group = train_df[
            (train_df["perturbation_id"] == perturbation_id) & (train_df["time_label"] == _TIME_TERMINAL)
        ]
        if initial_group.empty or terminal_group.empty:
            raise ValueError(f"Missing required cells for perturbation {perturbation_id!r}.")

        initial_mass = float(initial_group.shape[0] / train_df[train_df["time_label"] == _TIME_INITIAL].shape[0])
        terminal_mass = float(terminal_group.shape[0] / train_df[train_df["time_label"] == _TIME_TERMINAL].shape[0])

        initial_indices = initial_group["row_index"].to_numpy(dtype=int)
        if max_initial_cells_per_measure is not None and initial_indices.shape[0] > max_initial_cells_per_measure:
            initial_indices = rng.choice(initial_indices, size=max_initial_cells_per_measure, replace=False)
        terminal_indices = terminal_group["row_index"].to_numpy(dtype=int)
        terminal_train_indices, terminal_eval_indices = _split_terminal_indices(
            terminal_indices,
            rng=rng,
            train_n=int(train_terminal_cells_per_measure),
            eval_n=int(eval_terminal_cells_per_measure),
        )

        for time_label, use_indices, total_mass in (
            (_TIME_INITIAL, initial_indices, initial_mass),
            (_TIME_TERMINAL, terminal_train_indices, terminal_mass),
        ):
            sampled = train_df.iloc[np.sort(use_indices)]
            for _, row in sampled.iterrows():
                obs_rows.append(
                    {
                        "cell_id": f"{row['source_index']}::{row['perturbation_id']}::{time_label}",
                        "perturbation_id": perturbation_id,
                        "time_label": time_label,
                        "sample_id": _SAMPLE_ID,
                    }
                )
                z_rows.append(x_pca[int(row["row_index"]), :latent_dims].astype(float))
            mass_rows.append(
                {
                    "perturbation_id": perturbation_id,
                    "time_label": time_label,
                    "sample_id": _SAMPLE_ID,
                    "mass": total_mass,
                }
            )

        holdout_support = np.asarray(x_pca[np.sort(terminal_eval_indices), :latent_dims], dtype=float)
        holdout_terminal[(_SAMPLE_ID, perturbation_id)] = _build_measure(
            support=holdout_support,
            perturbation_id=perturbation_id,
            time_label=_TIME_TERMINAL,
            sample_id=_SAMPLE_ID,
            total_mass=terminal_mass,
        )

        summary_rows.extend(
            [
                {
                    "sample_id": _SAMPLE_ID,
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_INITIAL,
                    "n_cells_full": int(initial_group.shape[0]),
                    "n_cells_train": int(initial_indices.shape[0]),
                    "n_cells_holdout": 0,
                    "mass": initial_mass,
                },
                {
                    "sample_id": _SAMPLE_ID,
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_TERMINAL,
                    "n_cells_full": int(terminal_group.shape[0]),
                    "n_cells_train": int(terminal_train_indices.shape[0]),
                    "n_cells_holdout": int(terminal_eval_indices.shape[0]),
                    "mass": terminal_mass,
                },
            ]
        )

    cells = CellStateTable(obs=pd.DataFrame(obs_rows), Z=np.vstack(z_rows))
    masses = MassTable(table=pd.DataFrame(mass_rows))
    time_axis = TimeAxis(
        initial_label=_TIME_INITIAL,
        terminal_label=_TIME_TERMINAL,
        normalized_time={_TIME_INITIAL: 0.0, _TIME_TERMINAL: 1.0},
        physical_time={_TIME_INITIAL: 2.0, _TIME_TERMINAL: 6.0},
    )
    catalog = PerturbationCatalog(
        pd.DataFrame(
            [
                {"perturbation_id": "ctrl", "is_control": True},
                {"perturbation_id": "monocyte", "is_control": False},
                {"perturbation_id": "neutrophil", "is_control": False},
            ]
        )
    )
    latent_transform = LatentTransform.from_array(cells.Z)
    dataset = PerturbSeqDynamicsData(
        time_axis=time_axis,
        catalog=catalog,
        cells=cells,
        masses=masses,
        latent_transform=latent_transform,
        metadata={
            "source_dataset": str(data_path),
            "source_example": "scDiffEq quickstart LARRY clone subset",
            "adapter": "dominant_clone_fate_within_nm_clone_subset",
            "latent_key": "X_pca",
            "latent_dims": int(latent_dims),
            "mass_mode": "time_fraction",
        },
    )
    dataset.validate()

    metadata = {
        "source_dataset": str(data_path),
        "source_example": "scDiffEq quickstart LARRY clone subset",
        "adapter": "dominant_clone_fate_within_nm_clone_subset",
        "latent_key": "X_pca",
        "latent_dims": int(latent_dims),
        "mass_mode": "time_fraction",
        "max_initial_cells_per_measure": None if max_initial_cells_per_measure is None else int(max_initial_cells_per_measure),
        "train_terminal_cells_per_measure": int(train_terminal_cells_per_measure),
        "eval_terminal_cells_per_measure": int(eval_terminal_cells_per_measure),
        "seed": int(seed),
    }
    dataset_summary = pd.DataFrame(summary_rows).sort_values(["time_label", "perturbation_id"]).reset_index(drop=True)
    return dataset, holdout_terminal, dataset_summary, metadata


def _detail_table(
    *,
    model_name: str,
    result: FullModelTrainingResult,
    holdout_terminal: Dict[Key, FiniteMeasure],
) -> pd.DataFrame:
    rows = []
    dtype = result.final_simulation.terminal_particles[next(iter(result.final_simulation.terminal_particles))].z.dtype
    device = result.final_simulation.terminal_particles[next(iter(result.final_simulation.terminal_particles))].z.device

    for key, measure in holdout_terminal.items():
        pred_state = result.final_simulation.terminal_particles[key]
        pred_support = pred_state.z
        pred_weights = pred_state.atom_weights()
        target_support = torch.as_tensor(measure.support, dtype=dtype, device=device)
        target_weights = torch.as_tensor(measure.weights, dtype=dtype, device=device)
        endpoint_loss = unbalanced_sinkhorn_divergence(
            pred_support,
            pred_weights,
            target_support,
            target_weights,
            epsilon=result.config.epsilon,
            tau=result.config.tau,
            max_iters=result.config.sinkhorn_iters,
        )
        normalized_loss = normalized_geometry_loss(
            pred_support,
            pred_weights,
            target_support,
            target_weights,
            epsilon=result.config.epsilon,
            tau=result.config.tau,
            max_iters=result.config.sinkhorn_iters,
        )
        pred_mass = float(pred_state.total_mass().detach().cpu())
        pred_mean = pred_state.mean().detach().cpu().numpy()
        pred_var = float(pred_state.variance_trace().detach().cpu())
        target_mean = measure.mean()
        target_var = float(measure.variance_trace())
        rows.append(
            {
                "model_name": model_name,
                "sample_id": measure.sample_id,
                "perturbation_id": measure.perturbation_id,
                "holdout_endpoint_loss": float(endpoint_loss.detach().cpu()),
                "holdout_normalized_loss": float(normalized_loss.detach().cpu()),
                "pred_mass": pred_mass,
                "target_mass": float(measure.total_mass),
                "abs_mass_error": abs(pred_mass - float(measure.total_mass)),
                "l2_mean_error": float(np.linalg.norm(pred_mean - target_mean)),
                "abs_variance_error": abs(pred_var - target_var),
                "pred_var_trace": pred_var,
                "target_var_trace": target_var,
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _summary_row(model_name: str, result: FullModelTrainingResult, detail_table: pd.DataFrame) -> dict:
    return {
        "model_name": model_name,
        "stable": bool(result.final_simulation.stable),
        "control_anchor_exact": bool(result.model.control_anchor_is_exact()),
        "mean_holdout_endpoint_loss": float(detail_table["holdout_endpoint_loss"].mean()),
        "mean_holdout_normalized_loss": float(detail_table["holdout_normalized_loss"].mean()),
        "mean_abs_mass_error": float(detail_table["abs_mass_error"].mean()),
        "mean_l2_mean_error": float(detail_table["l2_mean_error"].mean()),
        "mean_abs_variance_error": float(detail_table["abs_variance_error"].mean()),
        "train_endpoint_loss_mean": float(result.final_loss_table["endpoint_loss"].mean()),
        "best_total_loss": float(result.history["total_loss"].min()),
        "mean_abs_context_value": _mean_abs_context_value(result),
    }


def _identity_baseline_detail_table(
    *,
    train_dataset: PerturbSeqDynamicsData,
    holdout_terminal: Dict[Key, FiniteMeasure],
    epsilon: float,
    tau: float,
    max_iters: int,
) -> pd.DataFrame:
    problem = train_dataset.to_endpoint_problem(by_sample=True)
    rows = []
    for key, measure in holdout_terminal.items():
        initial = problem.initial[key]
        pred_support = torch.as_tensor(initial.support, dtype=torch.float64)
        pred_weights = torch.as_tensor(initial.weights, dtype=torch.float64)
        target_support = torch.as_tensor(measure.support, dtype=torch.float64)
        target_weights = torch.as_tensor(measure.weights, dtype=torch.float64)
        endpoint_loss = unbalanced_sinkhorn_divergence(
            pred_support,
            pred_weights,
            target_support,
            target_weights,
            epsilon=epsilon,
            tau=tau,
            max_iters=max_iters,
        )
        normalized_loss = normalized_geometry_loss(
            pred_support,
            pred_weights,
            target_support,
            target_weights,
            epsilon=epsilon,
            tau=tau,
            max_iters=max_iters,
        )
        pred_mean = initial.mean()
        pred_var = initial.variance_trace()
        target_mean = measure.mean()
        target_var = measure.variance_trace()
        rows.append(
            {
                "model_name": "identity_baseline",
                "sample_id": measure.sample_id,
                "perturbation_id": measure.perturbation_id,
                "holdout_endpoint_loss": float(endpoint_loss.detach().cpu()),
                "holdout_normalized_loss": float(normalized_loss.detach().cpu()),
                "pred_mass": float(initial.total_mass),
                "target_mass": float(measure.total_mass),
                "abs_mass_error": abs(float(initial.total_mass) - float(measure.total_mass)),
                "l2_mean_error": float(np.linalg.norm(pred_mean - target_mean)),
                "abs_variance_error": abs(float(pred_var) - float(target_var)),
                "pred_var_trace": float(pred_var),
                "target_var_trace": float(target_var),
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _baseline_summary_row(detail_table: pd.DataFrame) -> dict:
    return {
        "model_name": "identity_baseline",
        "stable": True,
        "control_anchor_exact": True,
        "mean_holdout_endpoint_loss": float(detail_table["holdout_endpoint_loss"].mean()),
        "mean_holdout_normalized_loss": float(detail_table["holdout_normalized_loss"].mean()),
        "mean_abs_mass_error": float(detail_table["abs_mass_error"].mean()),
        "mean_l2_mean_error": float(detail_table["l2_mean_error"].mean()),
        "mean_abs_variance_error": float(detail_table["abs_variance_error"].mean()),
        "train_endpoint_loss_mean": float("nan"),
        "best_total_loss": float("nan"),
        "mean_abs_context_value": 0.0,
    }


def evaluate_camfnd_on_scdiffeq_larry(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    max_initial_cells_per_measure: Optional[int] = None,
    train_terminal_cells_per_measure: int = 96,
    eval_terminal_cells_per_measure: int = 256,
    dataset_seed: int = 17,
    full_config: Optional[FullModelTrainConfig] = None,
) -> ScDiffEqLarryEvaluation:
    train_dataset, holdout_terminal, dataset_summary, metadata = build_scdiffeq_larry_camfnd_dataset(
        data_path=data_path,
        latent_dims=latent_dims,
        max_initial_cells_per_measure=max_initial_cells_per_measure,
        train_terminal_cells_per_measure=train_terminal_cells_per_measure,
        eval_terminal_cells_per_measure=eval_terminal_cells_per_measure,
        seed=dataset_seed,
    )

    full_config = full_config or FullModelTrainConfig(
        device="cpu",
        hidden_dim=24,
        depth=1,
        context_dim=2,
        summary_dim=8,
        summary_hidden_dim=16,
        context_hidden_dim=8,
        epochs=100,
        lr=0.01,
        n_steps=8,
        loss_mode="normalized_only",
        aux_mass_weight=0.5,
        aux_mean_weight=5.0,
        aux_variance_weight=1.0,
        aux_screen_delta_mean_weight=0.0,
        dtype="float64",
    )
    full_config.validate()

    full_result = train_full_model(train_dataset, config=full_config)
    no_context_result = train_full_model(
        train_dataset,
        config=replace(full_config, use_context=False, aux_screen_delta_mean_weight=0.0),
    )

    baseline_detail = _identity_baseline_detail_table(
        train_dataset=train_dataset,
        holdout_terminal=holdout_terminal,
        epsilon=full_config.epsilon,
        tau=full_config.tau,
        max_iters=full_config.sinkhorn_iters,
    )
    full_detail = _detail_table(model_name="full", result=full_result, holdout_terminal=holdout_terminal)
    no_context_detail = _detail_table(model_name="no_context", result=no_context_result, holdout_terminal=holdout_terminal)
    summary_table = pd.DataFrame(
        [
            _baseline_summary_row(baseline_detail),
            _summary_row("full", full_result, full_detail),
            _summary_row("no_context", no_context_result, no_context_detail),
        ]
    ).sort_values("model_name").reset_index(drop=True)

    return ScDiffEqLarryEvaluation(
        train_dataset=train_dataset,
        dataset_summary=dataset_summary,
        full_result=full_result,
        no_context_result=no_context_result,
        holdout_terminal=holdout_terminal,
        summary_table=summary_table,
        baseline_detail_table=baseline_detail,
        full_detail_table=full_detail,
        no_context_detail_table=no_context_detail,
        metadata=metadata,
    )
