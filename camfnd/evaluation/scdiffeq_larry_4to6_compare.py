from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

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
_SAMPLE_ID = "larry_4to6_celltype"
_TIME_INITIAL = "D4"
_TIME_TERMINAL = "D6"
_CELLTYPE_TO_PERTURBATION = {
    "Undifferentiated": "ctrl",
    "Monocyte": "monocyte",
    "Neutrophil": "neutrophil",
}
_PERTURBATION_ORDER = ["ctrl", "monocyte", "neutrophil"]


@dataclass(slots=True)
class Larry4to6MethodComparison:
    train_dataset: PerturbSeqDynamicsData
    train_adata: ad.AnnData
    holdout_terminal: Dict[Key, FiniteMeasure]
    dataset_summary: pd.DataFrame
    detail_table: pd.DataFrame
    summary_table: pd.DataFrame
    camfnd_result: FullModelTrainingResult
    metadata: Dict[str, object]


@dataclass(slots=True)
class Larry4to6CVComparison:
    fold_comparisons: list[Larry4to6MethodComparison]
    fold_summary_table: pd.DataFrame
    detail_table: pd.DataFrame
    summary_table: pd.DataFrame
    metadata: Dict[str, object]


@dataclass(slots=True)
class ScDiffEqTuningResult:
    candidate_table: pd.DataFrame
    best_by_metric: Dict[str, dict]
    metadata: Dict[str, object]


def _read_larry_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    obs["source_row_index"] = np.arange(obs.shape[0], dtype=int)
    x_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    fate_counts = adata.uns["fate_counts"].copy()
    return obs, x_pca, fate_counts


def _subset_quickstart_cells(obs: pd.DataFrame, fate_counts: pd.DataFrame) -> pd.DataFrame:
    nm_clones = fate_counts[["Monocyte", "Neutrophil"]].dropna().index
    out = obs.copy()
    mask = (
        out["Cell type annotation"].isin(list(_CELLTYPE_TO_PERTURBATION))
        & out["clone_idx"].isin(nm_clones)
        & out["Time point"].isin([4.0, 6.0])
    )
    out = out.loc[mask].copy()
    out["perturbation_id"] = out["Cell type annotation"].map(_CELLTYPE_TO_PERTURBATION)
    out["sample_id"] = _SAMPLE_ID
    out["time_label"] = np.where(out["Time point"].eq(4.0), _TIME_INITIAL, _TIME_TERMINAL)
    out["cell_id"] = out.index.astype(str)
    return out


def _build_measure(
    *,
    support: np.ndarray,
    total_mass: float,
    perturbation_id: str,
    sample_id: str,
    time_label: str,
) -> FiniteMeasure:
    weights = np.full(support.shape[0], float(total_mass) / float(support.shape[0]), dtype=float)
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


def _build_benchmark_from_terminal_split(
    *,
    data_path: Path,
    subset: pd.DataFrame,
    x_pca: np.ndarray,
    latent_dims: int,
    terminal_train_source_indices_by_pid: Dict[str, np.ndarray],
    terminal_eval_source_indices_by_pid: Dict[str, np.ndarray],
    metadata_extra: Optional[Dict[str, object]] = None,
) -> tuple[PerturbSeqDynamicsData, ad.AnnData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    obs_rows: list[dict] = []
    z_rows: list[np.ndarray] = []
    mass_rows: list[dict] = []
    summary_rows: list[dict] = []
    adata_obs_rows: list[pd.DataFrame] = []
    adata_z_rows: list[np.ndarray] = []
    holdout_terminal: Dict[Key, FiniteMeasure] = {}

    initial_total = subset[subset["time_label"] == _TIME_INITIAL].shape[0]
    terminal_total = subset[subset["time_label"] == _TIME_TERMINAL].shape[0]

    for perturbation_id in _PERTURBATION_ORDER:
        initial_group = subset[
            (subset["perturbation_id"] == perturbation_id) & (subset["time_label"] == _TIME_INITIAL)
        ].copy()
        terminal_group = subset[
            (subset["perturbation_id"] == perturbation_id) & (subset["time_label"] == _TIME_TERMINAL)
        ].copy()
        if initial_group.empty or terminal_group.empty:
            raise ValueError(f"Missing required cells for perturbation {perturbation_id!r}.")

        initial_mass = float(initial_group.shape[0] / initial_total)
        terminal_mass = float(terminal_group.shape[0] / terminal_total)

        terminal_train_source_indices = np.asarray(
            terminal_train_source_indices_by_pid[perturbation_id], dtype=int
        )
        terminal_eval_source_indices = np.asarray(
            terminal_eval_source_indices_by_pid[perturbation_id], dtype=int
        )

        initial_support = x_pca[initial_group["source_row_index"].to_numpy(dtype=int), :latent_dims]
        terminal_train_support = x_pca[terminal_train_source_indices, :latent_dims]
        terminal_eval_support = x_pca[terminal_eval_source_indices, :latent_dims]
        terminal_train_group = (
            terminal_group.set_index("source_row_index").loc[terminal_train_source_indices].reset_index()
        )

        for row in initial_group.itertuples(index=False):
            obs_rows.append(
                {
                    "cell_id": f"{row.cell_id}::{perturbation_id}::{_TIME_INITIAL}",
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_INITIAL,
                    "sample_id": _SAMPLE_ID,
                }
            )
        for row in terminal_train_group.itertuples(index=False):
            obs_rows.append(
                {
                    "cell_id": f"{row.cell_id}::{perturbation_id}::{_TIME_TERMINAL}",
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_TERMINAL,
                    "sample_id": _SAMPLE_ID,
                }
            )

        z_rows.extend(initial_support.astype(float))
        z_rows.extend(terminal_train_support.astype(float))
        mass_rows.extend(
            [
                {
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_INITIAL,
                    "sample_id": _SAMPLE_ID,
                    "mass": initial_mass,
                },
                {
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_TERMINAL,
                    "sample_id": _SAMPLE_ID,
                    "mass": terminal_mass,
                },
            ]
        )

        adata_initial = initial_group[
            ["Time point", "Cell type annotation", "perturbation_id", "sample_id", "time_label"]
        ].copy()
        adata_terminal = terminal_train_group[
            ["Time point", "Cell type annotation", "perturbation_id", "sample_id", "time_label"]
        ].copy()
        adata_obs_rows.extend([adata_initial, adata_terminal])
        adata_z_rows.extend([initial_support.astype(np.float32), terminal_train_support.astype(np.float32)])

        holdout_terminal[(_SAMPLE_ID, perturbation_id)] = _build_measure(
            support=terminal_eval_support,
            total_mass=terminal_mass,
            perturbation_id=perturbation_id,
            sample_id=_SAMPLE_ID,
            time_label=_TIME_TERMINAL,
        )

        summary_rows.extend(
            [
                {
                    "sample_id": _SAMPLE_ID,
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_INITIAL,
                    "n_cells_full": int(initial_group.shape[0]),
                    "n_cells_train": int(initial_group.shape[0]),
                    "n_cells_holdout": 0,
                    "mass": initial_mass,
                },
                {
                    "sample_id": _SAMPLE_ID,
                    "perturbation_id": perturbation_id,
                    "time_label": _TIME_TERMINAL,
                    "n_cells_full": int(terminal_group.shape[0]),
                    "n_cells_train": int(terminal_train_support.shape[0]),
                    "n_cells_holdout": int(terminal_eval_support.shape[0]),
                    "mass": terminal_mass,
                },
            ]
        )

    cells = CellStateTable(obs=pd.DataFrame(obs_rows), Z=np.vstack(z_rows))
    masses = MassTable(table=pd.DataFrame(mass_rows))
    dataset = PerturbSeqDynamicsData(
        time_axis=TimeAxis(
            initial_label=_TIME_INITIAL,
            terminal_label=_TIME_TERMINAL,
            normalized_time={_TIME_INITIAL: 0.0, _TIME_TERMINAL: 1.0},
            physical_time={_TIME_INITIAL: 4.0, _TIME_TERMINAL: 6.0},
        ),
        catalog=PerturbationCatalog(
            pd.DataFrame(
                [
                    {"perturbation_id": "ctrl", "is_control": True},
                    {"perturbation_id": "monocyte", "is_control": False},
                    {"perturbation_id": "neutrophil", "is_control": False},
                ]
            )
        ),
        cells=cells,
        masses=masses,
        latent_transform=LatentTransform.from_array(cells.Z),
        metadata={
            "source_dataset": str(data_path),
            "source_example": "scDiffEq quickstart LARRY clone subset",
            "adapter": "cell_type_4to6_within_nm_clone_subset",
            "latent_key": "X_pca",
            "latent_dims": int(latent_dims),
            "mass_mode": "time_fraction",
        },
    )
    dataset.validate()

    adata_obs = pd.concat(adata_obs_rows, axis=0).copy()
    adata_obs.index = [f"cell_{i}" for i in range(adata_obs.shape[0])]
    adata_z = np.vstack(adata_z_rows).astype(np.float32)
    train_adata = ad.AnnData(X=adata_z.copy(), obs=adata_obs)
    train_adata.obsm["X_pca"] = adata_z.copy()

    metadata = {
        "source_dataset": str(data_path),
        "source_example": "scDiffEq quickstart LARRY clone subset",
        "adapter": "cell_type_4to6_within_nm_clone_subset",
        "latent_key": "X_pca",
        "latent_dims": int(latent_dims),
        "mass_mode": "time_fraction",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    dataset_summary = pd.DataFrame(summary_rows).sort_values(["time_label", "perturbation_id"]).reset_index(drop=True)
    return dataset, train_adata, holdout_terminal, dataset_summary, metadata


def _build_kfold_terminal_splits(
    *,
    subset: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> Dict[str, list[np.ndarray]]:
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2.")

    rng = np.random.default_rng(seed)
    fold_source_indices: Dict[str, list[np.ndarray]] = {}
    for perturbation_id in _PERTURBATION_ORDER:
        terminal_source_indices = subset[
            (subset["perturbation_id"] == perturbation_id) & (subset["time_label"] == _TIME_TERMINAL)
        ]["source_row_index"].to_numpy(dtype=int)
        if terminal_source_indices.shape[0] < n_folds:
            raise ValueError(
                f"Terminal group {perturbation_id!r} has only {terminal_source_indices.shape[0]} cells, "
                f"which is too few for n_folds={n_folds}."
            )
        perm = rng.permutation(terminal_source_indices)
        splits = [np.sort(split.astype(int)) for split in np.array_split(perm, n_folds)]
        if any(split.size == 0 for split in splits):
            raise ValueError(f"At least one fold for {perturbation_id!r} is empty.")
        fold_source_indices[perturbation_id] = splits
    return fold_source_indices


def build_larry_4to6_celltype_benchmark(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    train_terminal_cells_per_measure: int = 96,
    eval_terminal_cells_per_measure: int = 256,
    seed: int = 17,
) -> tuple[PerturbSeqDynamicsData, ad.AnnData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"LARRY dataset not found at {data_path}.")
    if int(latent_dims) <= 0:
        raise ValueError("latent_dims must be positive.")

    obs, x_pca, fate_counts = _read_larry_data(data_path)
    subset = _subset_quickstart_cells(obs, fate_counts)
    rng = np.random.default_rng(seed)

    if latent_dims > x_pca.shape[1]:
        raise ValueError(f"Requested latent_dims={latent_dims}, but X_pca only has {x_pca.shape[1]} columns.")

    terminal_train_source_indices_by_pid: Dict[str, np.ndarray] = {}
    terminal_eval_source_indices_by_pid: Dict[str, np.ndarray] = {}
    for perturbation_id in _PERTURBATION_ORDER:
        terminal_source_indices = subset[
            (subset["perturbation_id"] == perturbation_id) & (subset["time_label"] == _TIME_TERMINAL)
        ]["source_row_index"].to_numpy(dtype=int)
        if terminal_source_indices.shape[0] < train_terminal_cells_per_measure + eval_terminal_cells_per_measure:
            raise ValueError(
                f"Terminal group {perturbation_id!r} has {terminal_source_indices.shape[0]} cells, "
                f"but requires {train_terminal_cells_per_measure + eval_terminal_cells_per_measure}."
            )
        perm = rng.permutation(terminal_source_indices)
        terminal_train_source_indices_by_pid[perturbation_id] = np.sort(perm[:train_terminal_cells_per_measure])
        terminal_eval_source_indices_by_pid[perturbation_id] = np.sort(
            perm[
                train_terminal_cells_per_measure : train_terminal_cells_per_measure + eval_terminal_cells_per_measure
            ]
        )

    return _build_benchmark_from_terminal_split(
        data_path=data_path,
        subset=subset,
        x_pca=x_pca,
        latent_dims=latent_dims,
        terminal_train_source_indices_by_pid=terminal_train_source_indices_by_pid,
        terminal_eval_source_indices_by_pid=terminal_eval_source_indices_by_pid,
        metadata_extra={
            "seed": int(seed),
            "train_terminal_cells_per_measure": int(train_terminal_cells_per_measure),
            "eval_terminal_cells_per_measure": int(eval_terminal_cells_per_measure),
            "split_protocol": "single_holdout",
        },
    )


def _weighted_summary(
    support: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[float, np.ndarray, float]:
    total_mass = float(weights.sum().detach().cpu())
    normalized = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    mean = (normalized[:, None] * support).sum(dim=0)
    centered = support - mean
    variance_trace = float((normalized * centered.pow(2).sum(dim=1)).sum().detach().cpu())
    return total_mass, mean.detach().cpu().numpy(), variance_trace


def _metric_row(
    *,
    model_name: str,
    perturbation_id: str,
    pred_support: torch.Tensor,
    pred_weights: torch.Tensor,
    target: FiniteMeasure,
    epsilon: float,
    tau: float,
    max_iters: int,
) -> dict:
    target_support = torch.as_tensor(target.support, dtype=pred_support.dtype, device=pred_support.device)
    target_weights = torch.as_tensor(target.weights, dtype=pred_weights.dtype, device=pred_weights.device)
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
    pred_mass, pred_mean, pred_var = _weighted_summary(pred_support, pred_weights)
    return {
        "model_name": model_name,
        "sample_id": target.sample_id,
        "perturbation_id": perturbation_id,
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "normalized_loss": float(normalized_loss.detach().cpu()),
        "abs_mass_error": abs(pred_mass - float(target.total_mass)),
        "l2_mean_error": float(np.linalg.norm(pred_mean - target.mean())),
        "abs_variance_error": abs(pred_var - float(target.variance_trace())),
        "pred_mass": pred_mass,
        "target_mass": float(target.total_mass),
    }


def _identity_detail_table(
    *,
    train_dataset: PerturbSeqDynamicsData,
    holdout_terminal: Dict[Key, FiniteMeasure],
    epsilon: float,
    tau: float,
    max_iters: int,
) -> pd.DataFrame:
    problem = train_dataset.to_endpoint_problem(by_sample=True)
    rows = []
    for key in sorted(holdout_terminal):
        target = holdout_terminal[key]
        initial = problem.initial[key]
        rows.append(
            _metric_row(
                model_name="identity_baseline",
                perturbation_id=target.perturbation_id,
                pred_support=torch.as_tensor(initial.support, dtype=torch.float64),
                pred_weights=torch.as_tensor(initial.weights, dtype=torch.float64),
                target=target,
                epsilon=epsilon,
                tau=tau,
                max_iters=max_iters,
            )
        )
    return pd.DataFrame(rows).sort_values("perturbation_id").reset_index(drop=True)


def _camfnd_detail_table(
    *,
    result: FullModelTrainingResult,
    holdout_terminal: Dict[Key, FiniteMeasure],
) -> pd.DataFrame:
    rows = []
    for key in sorted(holdout_terminal):
        target = holdout_terminal[key]
        pred_state = result.final_simulation.terminal_particles[key]
        rows.append(
            _metric_row(
                model_name="camfnd",
                perturbation_id=target.perturbation_id,
                pred_support=pred_state.z,
                pred_weights=pred_state.atom_weights(),
                target=target,
                epsilon=result.config.epsilon,
                tau=result.config.tau,
                max_iters=result.config.sinkhorn_iters,
            )
        )
    return pd.DataFrame(rows).sort_values("perturbation_id").reset_index(drop=True)


def _scdiffeq_detail_table(
    *,
    train_adata: ad.AnnData,
    holdout_terminal: Dict[Key, FiniteMeasure],
    epsilon: float,
    tau: float,
    max_iters: int,
    seed: int,
    train_epochs: int,
    batch_size: int,
    dt: float,
    simulation_repeats: int = 1,
    model_kwargs: Optional[Dict[str, Any]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    try:
        import scdiffeq as sdq
    except ImportError as exc:
        raise ImportError(
            "scdiffeq is not installed in this environment. "
            "Run this evaluator from /home/yding1995/opscc_sc/scDiffeq/scDiffEq/.venv/bin/python."
        ) from exc

    latent_dims = int(train_adata.obsm["X_pca"].shape[1])
    model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})
    model = sdq.scDiffEq(
        train_adata,
        latent_dim=latent_dims,
        use_key="X_pca",
        time_key="Time point",
        seed=int(seed),
        build_kNN=False,
        silent=True,
        train_epochs=int(train_epochs),
        train_step_size=10,
        batch_size=int(batch_size),
        **model_kwargs,
    )
    model.fit(train_epochs=int(train_epochs), accelerator="auto", devices=1, **fit_kwargs)

    rows = []
    initial_total_mass = {}
    for perturbation_id in _PERTURBATION_ORDER:
        initial_mask = (train_adata.obs["perturbation_id"] == perturbation_id) & (
            train_adata.obs["time_label"] == _TIME_INITIAL
        )
        initial_total_mass[perturbation_id] = float(initial_mask.sum() / (train_adata.obs["time_label"] == _TIME_INITIAL).sum())

    for key in sorted(holdout_terminal):
        target = holdout_terminal[key]
        perturbation_id = target.perturbation_id
        idx = train_adata.obs.index[
            (train_adata.obs["perturbation_id"] == perturbation_id) & (train_adata.obs["time_label"] == _TIME_INITIAL)
        ]
        adata_sim = sdq.tl.simulate(
            train_adata,
            idx=idx,
            N=int(simulation_repeats),
            diffeq=model.DiffEq,
            use_key="X_pca",
            time_key="Time point",
            dt=float(dt),
        )
        final = adata_sim[np.isclose(adata_sim.obs["t"].to_numpy(dtype=float), 6.0)].copy()
        pred_support = torch.as_tensor(np.asarray(final.X, dtype=float), dtype=torch.float64)
        pred_mass = initial_total_mass[perturbation_id]
        pred_weights = torch.full((pred_support.shape[0],), pred_mass / pred_support.shape[0], dtype=torch.float64)
        rows.append(
            _metric_row(
                model_name="scdiffeq",
                perturbation_id=perturbation_id,
                pred_support=pred_support,
                pred_weights=pred_weights,
                target=target,
                epsilon=epsilon,
                tau=tau,
                max_iters=max_iters,
            )
        )
    return pd.DataFrame(rows).sort_values("perturbation_id").reset_index(drop=True)


def _summary_row(model_name: str, detail: pd.DataFrame) -> dict:
    return {
        "model_name": model_name,
        "mean_endpoint_loss": float(detail["endpoint_loss"].mean()),
        "mean_normalized_loss": float(detail["normalized_loss"].mean()),
        "mean_abs_mass_error": float(detail["abs_mass_error"].mean()),
        "mean_l2_mean_error": float(detail["l2_mean_error"].mean()),
        "mean_abs_variance_error": float(detail["abs_variance_error"].mean()),
    }


def _default_camfnd_config() -> FullModelTrainConfig:
    return FullModelTrainConfig(
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


def _evaluate_from_benchmark(
    *,
    train_dataset: PerturbSeqDynamicsData,
    train_adata: ad.AnnData,
    holdout_terminal: Dict[Key, FiniteMeasure],
    dataset_summary: pd.DataFrame,
    metadata: Dict[str, object],
    camfnd_config: Optional[FullModelTrainConfig],
    dataset_seed: int,
    scdiffeq_train_epochs: int,
    scdiffeq_batch_size: int,
    scdiffeq_dt: float,
    scdiffeq_simulation_repeats: int = 1,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Larry4to6MethodComparison:
    camfnd_config = replace(camfnd_config or _default_camfnd_config())
    camfnd_config.validate()

    camfnd_result = train_full_model(train_dataset, config=camfnd_config)
    identity_detail = _identity_detail_table(
        train_dataset=train_dataset,
        holdout_terminal=holdout_terminal,
        epsilon=camfnd_config.epsilon,
        tau=camfnd_config.tau,
        max_iters=camfnd_config.sinkhorn_iters,
    )
    camfnd_detail = _camfnd_detail_table(result=camfnd_result, holdout_terminal=holdout_terminal)
    scdiffeq_detail = _scdiffeq_detail_table(
        train_adata=train_adata,
        holdout_terminal=holdout_terminal,
        epsilon=camfnd_config.epsilon,
        tau=camfnd_config.tau,
        max_iters=camfnd_config.sinkhorn_iters,
        seed=dataset_seed,
        train_epochs=scdiffeq_train_epochs,
        batch_size=scdiffeq_batch_size,
        dt=scdiffeq_dt,
        simulation_repeats=scdiffeq_simulation_repeats,
        model_kwargs=scdiffeq_model_kwargs,
        fit_kwargs=scdiffeq_fit_kwargs,
    )

    detail_table = (
        pd.concat([identity_detail, camfnd_detail, scdiffeq_detail], ignore_index=True)
        .sort_values(["model_name", "perturbation_id"])
        .reset_index(drop=True)
    )
    summary_table = pd.DataFrame(
        [
            _summary_row("identity_baseline", identity_detail),
            _summary_row("camfnd", camfnd_detail),
            _summary_row("scdiffeq", scdiffeq_detail),
        ]
    ).sort_values("model_name").reset_index(drop=True)

    result_metadata = dict(metadata)
    result_metadata.update(
        {
            "comparison_protocol": "shared_4_to_6_celltype_endpoint_benchmark",
            "scdiffeq_train_epochs": int(scdiffeq_train_epochs),
            "scdiffeq_batch_size": int(scdiffeq_batch_size),
            "scdiffeq_dt": float(scdiffeq_dt),
            "scdiffeq_simulation_repeats": int(scdiffeq_simulation_repeats),
            "scdiffeq_model_kwargs": dict(scdiffeq_model_kwargs or {}),
            "scdiffeq_fit_kwargs": dict(scdiffeq_fit_kwargs or {}),
        }
    )
    return Larry4to6MethodComparison(
        train_dataset=train_dataset,
        train_adata=train_adata,
        holdout_terminal=holdout_terminal,
        dataset_summary=dataset_summary,
        detail_table=detail_table,
        summary_table=summary_table,
        camfnd_result=camfnd_result,
        metadata=result_metadata,
    )


def evaluate_camfnd_vs_scdiffeq_larry_4to6(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    train_terminal_cells_per_measure: int = 96,
    eval_terminal_cells_per_measure: int = 256,
    dataset_seed: int = 17,
    camfnd_config: Optional[FullModelTrainConfig] = None,
    scdiffeq_train_epochs: int = 100,
    scdiffeq_batch_size: int = 256,
    scdiffeq_dt: float = 0.5,
    scdiffeq_simulation_repeats: int = 1,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Larry4to6MethodComparison:
    train_dataset, train_adata, holdout_terminal, dataset_summary, metadata = build_larry_4to6_celltype_benchmark(
        data_path=data_path,
        latent_dims=latent_dims,
        train_terminal_cells_per_measure=train_terminal_cells_per_measure,
        eval_terminal_cells_per_measure=eval_terminal_cells_per_measure,
        seed=dataset_seed,
    )
    return _evaluate_from_benchmark(
        train_dataset=train_dataset,
        train_adata=train_adata,
        holdout_terminal=holdout_terminal,
        dataset_summary=dataset_summary,
        metadata=metadata,
        camfnd_config=camfnd_config,
        dataset_seed=dataset_seed,
        scdiffeq_train_epochs=scdiffeq_train_epochs,
        scdiffeq_batch_size=scdiffeq_batch_size,
        scdiffeq_dt=scdiffeq_dt,
        scdiffeq_simulation_repeats=scdiffeq_simulation_repeats,
        scdiffeq_model_kwargs=scdiffeq_model_kwargs,
        scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
    )


def evaluate_camfnd_vs_scdiffeq_larry_4to6_cv(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    n_folds: int = 5,
    dataset_seed: int = 17,
    camfnd_config: Optional[FullModelTrainConfig] = None,
    scdiffeq_train_epochs: int = 100,
    scdiffeq_batch_size: int = 256,
    scdiffeq_dt: float = 0.5,
    scdiffeq_simulation_repeats: int = 1,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Larry4to6CVComparison:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"LARRY dataset not found at {data_path}.")
    if int(latent_dims) <= 0:
        raise ValueError("latent_dims must be positive.")

    obs, x_pca, fate_counts = _read_larry_data(data_path)
    subset = _subset_quickstart_cells(obs, fate_counts)
    if latent_dims > x_pca.shape[1]:
        raise ValueError(f"Requested latent_dims={latent_dims}, but X_pca only has {x_pca.shape[1]} columns.")

    fold_splits = _build_kfold_terminal_splits(subset=subset, n_folds=n_folds, seed=dataset_seed)

    fold_comparisons: list[Larry4to6MethodComparison] = []
    fold_summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []

    for fold_index in range(int(n_folds)):
        terminal_train_source_indices_by_pid: Dict[str, np.ndarray] = {}
        terminal_eval_source_indices_by_pid: Dict[str, np.ndarray] = {}
        for perturbation_id in _PERTURBATION_ORDER:
            terminal_eval_source_indices_by_pid[perturbation_id] = fold_splits[perturbation_id][fold_index]
            terminal_train_source_indices_by_pid[perturbation_id] = np.sort(
                np.concatenate(
                    [
                        fold_splits[perturbation_id][other_index]
                        for other_index in range(int(n_folds))
                        if other_index != fold_index
                    ]
                )
            )

        train_dataset, train_adata, holdout_terminal, dataset_summary, metadata = _build_benchmark_from_terminal_split(
            data_path=data_path,
            subset=subset,
            x_pca=x_pca,
            latent_dims=latent_dims,
            terminal_train_source_indices_by_pid=terminal_train_source_indices_by_pid,
            terminal_eval_source_indices_by_pid=terminal_eval_source_indices_by_pid,
            metadata_extra={
                "seed": int(dataset_seed),
                "split_protocol": "kfold",
                "n_folds": int(n_folds),
                "fold_index": int(fold_index),
            },
        )
        fold_result = _evaluate_from_benchmark(
            train_dataset=train_dataset,
            train_adata=train_adata,
            holdout_terminal=holdout_terminal,
            dataset_summary=dataset_summary,
            metadata=metadata,
            camfnd_config=camfnd_config,
            dataset_seed=dataset_seed,
            scdiffeq_train_epochs=scdiffeq_train_epochs,
            scdiffeq_batch_size=scdiffeq_batch_size,
            scdiffeq_dt=scdiffeq_dt,
            scdiffeq_simulation_repeats=scdiffeq_simulation_repeats,
            scdiffeq_model_kwargs=scdiffeq_model_kwargs,
            scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
        )
        fold_comparisons.append(fold_result)

        fold_summary = fold_result.summary_table.copy()
        fold_summary.insert(0, "fold_index", int(fold_index))
        fold_summary_frames.append(fold_summary)

        fold_detail = fold_result.detail_table.copy()
        fold_detail.insert(0, "fold_index", int(fold_index))
        detail_frames.append(fold_detail)

    fold_summary_table = pd.concat(fold_summary_frames, ignore_index=True)
    detail_table = pd.concat(detail_frames, ignore_index=True)

    metric_columns = [
        "mean_endpoint_loss",
        "mean_normalized_loss",
        "mean_abs_mass_error",
        "mean_l2_mean_error",
        "mean_abs_variance_error",
    ]
    summary_table = (
        fold_summary_table.groupby("model_name")[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary_table.columns = [
        "model_name" if column == ("model_name", "") else f"{column[0]}_{column[1]}"
        for column in summary_table.columns.to_flat_index()
    ]
    summary_table = summary_table.sort_values("model_name").reset_index(drop=True)

    metadata = {
        "source_dataset": str(data_path),
        "comparison_protocol": "shared_4_to_6_celltype_endpoint_5fold_cv",
        "latent_dims": int(latent_dims),
        "n_folds": int(n_folds),
        "seed": int(dataset_seed),
        "scdiffeq_train_epochs": int(scdiffeq_train_epochs),
        "scdiffeq_batch_size": int(scdiffeq_batch_size),
        "scdiffeq_dt": float(scdiffeq_dt),
        "scdiffeq_simulation_repeats": int(scdiffeq_simulation_repeats),
        "scdiffeq_model_kwargs": dict(scdiffeq_model_kwargs or {}),
        "scdiffeq_fit_kwargs": dict(scdiffeq_fit_kwargs or {}),
    }
    return Larry4to6CVComparison(
        fold_comparisons=fold_comparisons,
        fold_summary_table=fold_summary_table,
        detail_table=detail_table,
        summary_table=summary_table,
        metadata=metadata,
    )


def tune_scdiffeq_on_larry_4to6(
    *,
    data_path: str | Path = LARRY_DEFAULT_PATH,
    latent_dims: int = 4,
    train_terminal_cells_per_measure: int = 96,
    eval_terminal_cells_per_measure: int = 256,
    dataset_seed: int = 17,
    candidates: Optional[list[Dict[str, Any]]] = None,
) -> ScDiffEqTuningResult:
    candidate_list = candidates or [
        {
            "name": "baseline",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 1,
            "model_kwargs": {},
            "fit_kwargs": {},
        },
        {
            "name": "full_train_no_val",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 1,
            "model_kwargs": {"train_val_split": [1.0, 0.0]},
            "fit_kwargs": {},
        },
        {
            "name": "full_train_lr1e3",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 1,
            "model_kwargs": {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
            "fit_kwargs": {},
        },
        {
            "name": "full_train_longer",
            "train_epochs": 200,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 1,
            "model_kwargs": {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
            "fit_kwargs": {},
        },
        {
            "name": "smaller_dt",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.25,
            "simulation_repeats": 1,
            "model_kwargs": {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
            "fit_kwargs": {},
        },
        {
            "name": "repeat4",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 4,
            "model_kwargs": {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
            "fit_kwargs": {},
        },
        {
            "name": "wider_drift",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 1,
            "model_kwargs": {
                "train_val_split": [1.0, 0.0],
                "train_lr": 1e-3,
                "mu_hidden": [256, 256],
                "sigma_hidden": [64, 64],
            },
            "fit_kwargs": {},
        },
        {
            "name": "wider_repeat4",
            "train_epochs": 100,
            "batch_size": 256,
            "dt": 0.5,
            "simulation_repeats": 4,
            "model_kwargs": {
                "train_val_split": [1.0, 0.0],
                "train_lr": 1e-3,
                "mu_hidden": [256, 256],
                "sigma_hidden": [64, 64],
            },
            "fit_kwargs": {},
        },
    ]

    train_dataset, train_adata, holdout_terminal, dataset_summary, metadata = build_larry_4to6_celltype_benchmark(
        data_path=data_path,
        latent_dims=latent_dims,
        train_terminal_cells_per_measure=train_terminal_cells_per_measure,
        eval_terminal_cells_per_measure=eval_terminal_cells_per_measure,
        seed=dataset_seed,
    )
    metric_config = _default_camfnd_config()
    rows = []
    for candidate in candidate_list:
        scdiffeq_detail = _scdiffeq_detail_table(
            train_adata=train_adata,
            holdout_terminal=holdout_terminal,
            epsilon=metric_config.epsilon,
            tau=metric_config.tau,
            max_iters=metric_config.sinkhorn_iters,
            seed=dataset_seed,
            train_epochs=int(candidate["train_epochs"]),
            batch_size=int(candidate["batch_size"]),
            dt=float(candidate["dt"]),
            simulation_repeats=int(candidate.get("simulation_repeats", 1)),
            model_kwargs=dict(candidate.get("model_kwargs", {})),
            fit_kwargs=dict(candidate.get("fit_kwargs", {})),
        )
        scdiffeq_summary = _summary_row("scdiffeq", scdiffeq_detail)
        row = {
            "candidate_name": str(candidate["name"]),
            "train_epochs": int(candidate["train_epochs"]),
            "batch_size": int(candidate["batch_size"]),
            "dt": float(candidate["dt"]),
            "simulation_repeats": int(candidate.get("simulation_repeats", 1)),
            "model_kwargs": dict(candidate.get("model_kwargs", {})),
            "fit_kwargs": dict(candidate.get("fit_kwargs", {})),
            "mean_endpoint_loss": float(scdiffeq_summary["mean_endpoint_loss"]),
            "mean_normalized_loss": float(scdiffeq_summary["mean_normalized_loss"]),
            "mean_abs_mass_error": float(scdiffeq_summary["mean_abs_mass_error"]),
            "mean_l2_mean_error": float(scdiffeq_summary["mean_l2_mean_error"]),
            "mean_abs_variance_error": float(scdiffeq_summary["mean_abs_variance_error"]),
            "ctrl_normalized_loss": float(
                scdiffeq_detail.loc[scdiffeq_detail["perturbation_id"] == "ctrl", "normalized_loss"].iloc[0]
            ),
            "monocyte_normalized_loss": float(
                scdiffeq_detail.loc[scdiffeq_detail["perturbation_id"] == "monocyte", "normalized_loss"].iloc[0]
            ),
            "neutrophil_normalized_loss": float(
                scdiffeq_detail.loc[scdiffeq_detail["perturbation_id"] == "neutrophil", "normalized_loss"].iloc[0]
            ),
        }
        rows.append(row)

    candidate_table = pd.DataFrame(rows).sort_values("candidate_name").reset_index(drop=True)
    best_by_metric = {}
    for metric in (
        "mean_endpoint_loss",
        "mean_normalized_loss",
        "mean_abs_mass_error",
        "mean_l2_mean_error",
        "mean_abs_variance_error",
    ):
        best_index = candidate_table[metric].idxmin()
        best_by_metric[metric] = candidate_table.loc[int(best_index)].to_dict()

    return ScDiffEqTuningResult(
        candidate_table=candidate_table,
        best_by_metric=best_by_metric,
        metadata={
            "source_dataset": str(data_path),
            "latent_dims": int(latent_dims),
            "dataset_seed": int(dataset_seed),
            "n_candidates": int(len(candidate_list)),
            "train_cells": int(train_dataset.cells.obs.shape[0]),
            "holdout_measures": int(len(holdout_terminal)),
        },
    )
