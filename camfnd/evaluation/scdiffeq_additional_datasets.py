from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

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
from camfnd.evaluation.scdiffeq_larry_4to6_compare import (
    _camfnd_detail_table,
    _default_camfnd_config,
    _identity_detail_table,
    _metric_row,
    _summary_row,
)
from camfnd.training.full_model import FullModelTrainConfig, FullModelTrainingResult, train_full_model


SCDIFFEQ_DATA_ROOT = Path("/home/yding1995/opscc_sc/scDiffeq/KleinLabData/scdiffeq_data")
PANCREAS_DEFAULT_PATH = SCDIFFEQ_DATA_ROOT / "pancreatic_endocrinogenesis" / "adata.pancreatic_endocrinogenesis.cytotrace.h5ad"
HUMAN_HEMATOPOIESIS_DEFAULT_PATH = (
    SCDIFFEQ_DATA_ROOT / "human_hematopoiesis" / "human_hematopoiesis.processed.h5ad"
)


@dataclass(slots=True)
class ScDiffEqDatasetMethodComparison:
    dataset_name: str
    train_dataset: PerturbSeqDynamicsData
    train_adata: ad.AnnData
    holdout_terminal: Dict[Key, FiniteMeasure]
    dataset_summary: pd.DataFrame
    detail_table: pd.DataFrame
    summary_table: pd.DataFrame
    camfnd_result: FullModelTrainingResult
    metadata: Dict[str, object]


@dataclass(slots=True)
class ScDiffEqAdditionalDatasetSuite:
    comparisons: list[ScDiffEqDatasetMethodComparison]
    summary_table: pd.DataFrame
    detail_table: pd.DataFrame
    metadata: Dict[str, object]


def _format_time_label(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return f"T{int(round(value))}"
    return f"T{str(value).replace('.', 'p')}"


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


def _read_latent_adata(path: Path, latent_key: str, latent_dims: int) -> tuple[pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    obs["source_row_index"] = np.arange(obs.shape[0], dtype=int)
    if latent_key == "X":
        Z = np.asarray(adata.X, dtype=np.float32)
    else:
        Z = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if latent_dims > Z.shape[1]:
        raise ValueError(f"Requested latent_dims={latent_dims}, but {latent_key} has only {Z.shape[1]} columns.")
    return obs, Z[:, :latent_dims]


def _split_terminal_indices(
    *,
    subset: pd.DataFrame,
    perturbation_ids: Sequence[str],
    terminal_time: float,
    holdout_fraction: float,
    seed: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train_indices: Dict[str, np.ndarray] = {}
    eval_indices: Dict[str, np.ndarray] = {}
    for perturbation_id in perturbation_ids:
        terminal_source_indices = subset[
            (subset["perturbation_id"] == perturbation_id)
            & np.isclose(subset["time_value"].to_numpy(dtype=float), float(terminal_time))
        ]["source_row_index"].to_numpy(dtype=int)
        if terminal_source_indices.shape[0] < 2:
            raise ValueError(
                f"Terminal group {perturbation_id!r} has only {terminal_source_indices.shape[0]} cells; need at least 2."
            )
        perm = rng.permutation(terminal_source_indices)
        n_eval = int(np.floor(holdout_fraction * terminal_source_indices.shape[0]))
        n_eval = max(1, min(n_eval, terminal_source_indices.shape[0] - 1))
        eval_indices[perturbation_id] = np.sort(perm[:n_eval].astype(int))
        train_indices[perturbation_id] = np.sort(perm[n_eval:].astype(int))
    return train_indices, eval_indices


def build_scdiffeq_endpoint_benchmark(
    *,
    data_path: str | Path,
    dataset_name: str,
    time_key: str,
    initial_time: float,
    terminal_time: float,
    latent_key: str = "X_pca",
    latent_dims: int = 10,
    group_col: Optional[str] = None,
    include_groups: Optional[Sequence[str]] = None,
    min_initial_cells: int = 1,
    min_terminal_cells: int = 2,
    control_perturbation: Optional[str] = None,
    holdout_fraction: float = 0.25,
    seed: int = 17,
) -> tuple[PerturbSeqDynamicsData, ad.AnnData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    if not (0.0 < float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")

    obs, Z = _read_latent_adata(data_path, latent_key=latent_key, latent_dims=int(latent_dims))
    obs["time_value"] = obs[time_key].astype(float)
    mask = (obs["time_value"] >= float(initial_time)) & (obs["time_value"] <= float(terminal_time))
    subset = obs.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No cells found in time interval [{initial_time}, {terminal_time}].")

    if group_col is None:
        sample_id = f"{dataset_name}_t{_format_time_label(initial_time)[1:]}_to_{_format_time_label(terminal_time)[1:]}_overall"
        subset["perturbation_id"] = "ctrl"
        perturbation_ids = ["ctrl"]
        control_perturbation = "ctrl"
        adapter = "overall_endpoint"
    else:
        sample_id = f"{dataset_name}_t{_format_time_label(initial_time)[1:]}_to_{_format_time_label(terminal_time)[1:]}_{group_col}"
        subset["perturbation_id"] = subset[group_col].astype(str)
        init_counts = subset[np.isclose(subset["time_value"], float(initial_time))]["perturbation_id"].value_counts()
        term_counts = subset[np.isclose(subset["time_value"], float(terminal_time))]["perturbation_id"].value_counts()
        if include_groups is None:
            include_groups = [
                str(group)
                for group in sorted(set(init_counts.index).intersection(term_counts.index))
                if int(init_counts.get(group, 0)) >= int(min_initial_cells)
                and int(term_counts.get(group, 0)) >= int(min_terminal_cells)
            ]
        include_groups = [str(value) for value in include_groups]
        subset = subset[subset["perturbation_id"].isin(include_groups)].copy()
        perturbation_ids = sorted(include_groups)
        if not perturbation_ids:
            raise ValueError("No perturbation groups satisfy the requested filters.")
        adapter = f"grouped_by_{group_col}"

    subset["sample_id"] = sample_id
    subset["cell_id"] = subset.index.astype(str)
    subset["time_label"] = subset["time_value"].map(_format_time_label)

    if control_perturbation is None:
        control_perturbation = perturbation_ids[0]
    control_perturbation = str(control_perturbation)
    if control_perturbation not in perturbation_ids:
        raise ValueError(f"control_perturbation {control_perturbation!r} is not in the included groups.")

    train_terminal_indices_by_pid, eval_terminal_indices_by_pid = _split_terminal_indices(
        subset=subset,
        perturbation_ids=perturbation_ids,
        terminal_time=float(terminal_time),
        holdout_fraction=float(holdout_fraction),
        seed=int(seed),
    )

    all_time_values = sorted(subset["time_value"].astype(float).unique().tolist())
    obs_rows: list[dict] = []
    z_rows: list[np.ndarray] = []
    mass_rows: list[dict] = []
    summary_rows: list[dict] = []
    adata_obs_rows: list[pd.DataFrame] = []
    adata_z_rows: list[np.ndarray] = []
    holdout_terminal: Dict[Key, FiniteMeasure] = {}

    for time_value in all_time_values:
        time_label = _format_time_label(time_value)
        time_subset = subset[np.isclose(subset["time_value"], float(time_value))].copy()
        time_total = float(time_subset.shape[0])
        for perturbation_id in perturbation_ids:
            group_time = time_subset[time_subset["perturbation_id"] == perturbation_id].copy()
            if group_time.empty:
                continue
            group_full_n = int(group_time.shape[0])
            terminal_holdout_n = 0
            if np.isclose(time_value, float(terminal_time)):
                train_indices = train_terminal_indices_by_pid[perturbation_id]
                holdout_indices = eval_terminal_indices_by_pid[perturbation_id]
                group_train = (
                    group_time.set_index("source_row_index").loc[train_indices].reset_index()
                )
                group_eval_support = Z[holdout_indices, :]
                terminal_holdout_n = int(holdout_indices.shape[0])
                terminal_mass = float(group_full_n / time_total)
                holdout_terminal[(sample_id, perturbation_id)] = _build_measure(
                    support=group_eval_support,
                    total_mass=terminal_mass,
                    perturbation_id=perturbation_id,
                    sample_id=sample_id,
                    time_label=time_label,
                )
            else:
                group_train = group_time

            for row in group_train.itertuples(index=False):
                obs_rows.append(
                    {
                        "cell_id": f"{row.cell_id}::{row.perturbation_id}::{time_label}",
                        "perturbation_id": perturbation_id,
                        "time_label": time_label,
                        "sample_id": sample_id,
                    }
                )
            z_rows.extend(Z[group_train["source_row_index"].to_numpy(dtype=int), :].astype(float))

            mass_rows.append(
                {
                    "perturbation_id": perturbation_id,
                    "time_label": time_label,
                    "sample_id": sample_id,
                    "mass": float(group_full_n / time_total),
                }
            )
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "perturbation_id": perturbation_id,
                    "time_label": time_label,
                    "time_value": float(time_value),
                    "n_cells_full": group_full_n,
                    "n_cells_train": int(group_train.shape[0]),
                    "n_cells_holdout": terminal_holdout_n,
                    "mass": float(group_full_n / time_total),
                }
            )

            adata_train_obs = group_train[["perturbation_id", "sample_id", "time_label"]].copy()
            adata_train_obs[time_key] = float(time_value)
            if group_col is not None and group_col in group_train.columns:
                adata_train_obs[group_col] = group_train[group_col].astype(str).to_numpy()
            adata_obs_rows.append(adata_train_obs)
            adata_z_rows.append(Z[group_train["source_row_index"].to_numpy(dtype=int), :].astype(np.float32))

    if not z_rows:
        raise ValueError("No training cells were collected for the benchmark.")

    cells = CellStateTable(obs=pd.DataFrame(obs_rows), Z=np.vstack(z_rows))
    masses = MassTable(table=pd.DataFrame(mass_rows))
    initial_label = _format_time_label(float(initial_time))
    terminal_label = _format_time_label(float(terminal_time))
    dataset = PerturbSeqDynamicsData(
        time_axis=TimeAxis(
            initial_label=initial_label,
            terminal_label=terminal_label,
            normalized_time={initial_label: 0.0, terminal_label: 1.0},
            physical_time={initial_label: float(initial_time), terminal_label: float(terminal_time)},
        ),
        catalog=PerturbationCatalog(
            pd.DataFrame(
                [
                    {"perturbation_id": perturbation_id, "is_control": bool(perturbation_id == control_perturbation)}
                    for perturbation_id in perturbation_ids
                ]
            )
        ),
        cells=cells,
        masses=masses,
        latent_transform=LatentTransform.from_array(cells.Z),
        metadata={
            "source_dataset": str(data_path),
            "source_example": dataset_name,
            "adapter": adapter,
            "latent_key": latent_key,
            "latent_dims": int(latent_dims),
            "group_col": group_col,
            "control_perturbation": control_perturbation,
            "mass_mode": "time_fraction_within_included_groups",
            "holdout_fraction": float(holdout_fraction),
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
        "source_example": dataset_name,
        "adapter": adapter,
        "latent_key": latent_key,
        "latent_dims": int(latent_dims),
        "group_col": group_col,
        "control_perturbation": control_perturbation,
        "sample_id": sample_id,
        "initial_time": float(initial_time),
        "terminal_time": float(terminal_time),
        "holdout_fraction": float(holdout_fraction),
        "perturbation_ids": list(perturbation_ids),
    }
    dataset_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["time_value", "perturbation_id"])
        .reset_index(drop=True)
    )
    return dataset, train_adata, holdout_terminal, dataset_summary, metadata


def build_pancreas_overall_benchmark(
    *,
    data_path: str | Path = PANCREAS_DEFAULT_PATH,
    initial_time: float = 4.0,
    terminal_time: float = 10.0,
    latent_dims: int = 10,
    holdout_fraction: float = 0.25,
    seed: int = 17,
) -> tuple[PerturbSeqDynamicsData, ad.AnnData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    return build_scdiffeq_endpoint_benchmark(
        data_path=data_path,
        dataset_name="scDiffEq pancreas",
        time_key="t",
        initial_time=initial_time,
        terminal_time=terminal_time,
        latent_key="X_pca",
        latent_dims=latent_dims,
        group_col=None,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )


def build_human_hematopoiesis_celltype_benchmark(
    *,
    data_path: str | Path = HUMAN_HEMATOPOIESIS_DEFAULT_PATH,
    initial_time: float = 4.0,
    terminal_time: float = 7.0,
    latent_dims: int = 10,
    holdout_fraction: float = 0.25,
    seed: int = 17,
    include_groups: Optional[Sequence[str]] = None,
) -> tuple[PerturbSeqDynamicsData, ad.AnnData, Dict[Key, FiniteMeasure], pd.DataFrame, Dict[str, object]]:
    return build_scdiffeq_endpoint_benchmark(
        data_path=data_path,
        dataset_name="scDiffEq human hematopoiesis",
        time_key="t",
        initial_time=initial_time,
        terminal_time=terminal_time,
        latent_key="X_pca",
        latent_dims=latent_dims,
        group_col="cell_type",
        include_groups=include_groups or ("Bas", "GMP-like", "HSC", "MEP-like", "Mon"),
        min_initial_cells=10,
        min_terminal_cells=24,
        control_perturbation="HSC",
        holdout_fraction=holdout_fraction,
        seed=seed,
    )


def _scdiffeq_detail_table_generic(
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
    time_key: str,
    terminal_time: float,
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
        time_key=time_key,
        seed=int(seed),
        build_kNN=False,
        silent=True,
        train_epochs=int(train_epochs),
        train_step_size=10,
        batch_size=int(batch_size),
        **model_kwargs,
    )
    model.fit(train_epochs=int(train_epochs), accelerator="auto", devices=1, **fit_kwargs)

    initial_time = float(train_adata.obs[time_key].astype(float).min())
    initial_mask_all = np.isclose(train_adata.obs[time_key].to_numpy(dtype=float), initial_time)
    rows = []
    initial_total_mass: Dict[str, float] = {}
    for perturbation_id in sorted(train_adata.obs["perturbation_id"].astype(str).unique().tolist()):
        initial_mask = (train_adata.obs["perturbation_id"].astype(str) == perturbation_id) & initial_mask_all
        initial_total_mass[perturbation_id] = float(initial_mask.sum() / max(int(initial_mask_all.sum()), 1))

    for key in sorted(holdout_terminal):
        target = holdout_terminal[key]
        perturbation_id = target.perturbation_id
        idx = train_adata.obs.index[
            (train_adata.obs["perturbation_id"].astype(str) == perturbation_id)
            & initial_mask_all
        ]
        adata_sim = sdq.tl.simulate(
            train_adata,
            idx=idx,
            N=int(simulation_repeats),
            diffeq=model.DiffEq,
            use_key="X_pca",
            time_key=time_key,
            dt=float(dt),
        )
        final = adata_sim[np.isclose(adata_sim.obs[time_key].to_numpy(dtype=float), float(terminal_time))].copy()
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


def _evaluate_from_benchmark(
    *,
    dataset_name: str,
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
) -> ScDiffEqDatasetMethodComparison:
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
    scdiffeq_detail = _scdiffeq_detail_table_generic(
        train_adata=train_adata,
        holdout_terminal=holdout_terminal,
        epsilon=camfnd_config.epsilon,
        tau=camfnd_config.tau,
        max_iters=camfnd_config.sinkhorn_iters,
        seed=dataset_seed,
        train_epochs=scdiffeq_train_epochs,
        batch_size=scdiffeq_batch_size,
        dt=scdiffeq_dt,
        time_key="t",
        terminal_time=float(metadata["terminal_time"]),
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
    )
    summary_table.insert(0, "dataset_name", dataset_name)
    metadata = {
        **metadata,
        "scdiffeq_train_epochs": int(scdiffeq_train_epochs),
        "scdiffeq_batch_size": int(scdiffeq_batch_size),
        "scdiffeq_dt": float(scdiffeq_dt),
        "scdiffeq_simulation_repeats": int(scdiffeq_simulation_repeats),
        "scdiffeq_model_kwargs": dict(scdiffeq_model_kwargs or {}),
        "scdiffeq_fit_kwargs": dict(scdiffeq_fit_kwargs or {}),
        "camfnd_config": camfnd_config,
    }
    return ScDiffEqDatasetMethodComparison(
        dataset_name=dataset_name,
        train_dataset=train_dataset,
        train_adata=train_adata,
        holdout_terminal=holdout_terminal,
        dataset_summary=dataset_summary,
        detail_table=detail_table,
        summary_table=summary_table,
        camfnd_result=camfnd_result,
        metadata=metadata,
    )


def evaluate_camfnd_vs_scdiffeq_pancreas(
    *,
    data_path: str | Path = PANCREAS_DEFAULT_PATH,
    initial_time: float = 4.0,
    terminal_time: float = 10.0,
    latent_dims: int = 10,
    holdout_fraction: float = 0.25,
    dataset_seed: int = 17,
    camfnd_config: Optional[FullModelTrainConfig] = None,
    scdiffeq_train_epochs: int = 100,
    scdiffeq_batch_size: int = 256,
    scdiffeq_dt: float = 0.5,
    scdiffeq_simulation_repeats: int = 4,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> ScDiffEqDatasetMethodComparison:
    train_dataset, train_adata, holdout_terminal, dataset_summary, metadata = build_pancreas_overall_benchmark(
        data_path=data_path,
        initial_time=initial_time,
        terminal_time=terminal_time,
        latent_dims=latent_dims,
        holdout_fraction=holdout_fraction,
        seed=dataset_seed,
    )
    return _evaluate_from_benchmark(
        dataset_name="pancreas",
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
        scdiffeq_model_kwargs=scdiffeq_model_kwargs or {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
        scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
    )


def evaluate_camfnd_vs_scdiffeq_human_hematopoiesis(
    *,
    data_path: str | Path = HUMAN_HEMATOPOIESIS_DEFAULT_PATH,
    initial_time: float = 4.0,
    terminal_time: float = 7.0,
    latent_dims: int = 10,
    holdout_fraction: float = 0.25,
    dataset_seed: int = 17,
    include_groups: Optional[Sequence[str]] = None,
    camfnd_config: Optional[FullModelTrainConfig] = None,
    scdiffeq_train_epochs: int = 100,
    scdiffeq_batch_size: int = 256,
    scdiffeq_dt: float = 0.5,
    scdiffeq_simulation_repeats: int = 4,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> ScDiffEqDatasetMethodComparison:
    train_dataset, train_adata, holdout_terminal, dataset_summary, metadata = build_human_hematopoiesis_celltype_benchmark(
        data_path=data_path,
        initial_time=initial_time,
        terminal_time=terminal_time,
        latent_dims=latent_dims,
        holdout_fraction=holdout_fraction,
        seed=dataset_seed,
        include_groups=include_groups,
    )
    return _evaluate_from_benchmark(
        dataset_name="human_hematopoiesis",
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
        scdiffeq_model_kwargs=scdiffeq_model_kwargs or {"train_val_split": [1.0, 0.0], "train_lr": 1e-3},
        scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
    )


def evaluate_camfnd_vs_scdiffeq_additional_datasets(
    *,
    camfnd_config: Optional[FullModelTrainConfig] = None,
    scdiffeq_train_epochs: int = 100,
    scdiffeq_batch_size: int = 256,
    scdiffeq_dt: float = 0.5,
    scdiffeq_simulation_repeats: int = 4,
    scdiffeq_model_kwargs: Optional[Dict[str, Any]] = None,
    scdiffeq_fit_kwargs: Optional[Dict[str, Any]] = None,
) -> ScDiffEqAdditionalDatasetSuite:
    comparisons = [
        evaluate_camfnd_vs_scdiffeq_human_hematopoiesis(
            camfnd_config=camfnd_config,
            scdiffeq_train_epochs=scdiffeq_train_epochs,
            scdiffeq_batch_size=scdiffeq_batch_size,
            scdiffeq_dt=scdiffeq_dt,
            scdiffeq_simulation_repeats=scdiffeq_simulation_repeats,
            scdiffeq_model_kwargs=scdiffeq_model_kwargs,
            scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
        ),
        evaluate_camfnd_vs_scdiffeq_pancreas(
            camfnd_config=camfnd_config,
            scdiffeq_train_epochs=scdiffeq_train_epochs,
            scdiffeq_batch_size=scdiffeq_batch_size,
            scdiffeq_dt=scdiffeq_dt,
            scdiffeq_simulation_repeats=scdiffeq_simulation_repeats,
            scdiffeq_model_kwargs=scdiffeq_model_kwargs,
            scdiffeq_fit_kwargs=scdiffeq_fit_kwargs,
        ),
    ]
    summary_table = pd.concat([comparison.summary_table for comparison in comparisons], ignore_index=True)
    detail_frames = []
    for comparison in comparisons:
        detail = comparison.detail_table.copy()
        detail.insert(0, "dataset_name", comparison.dataset_name)
        detail_frames.append(detail)
    detail_table = pd.concat(detail_frames, ignore_index=True)
    return ScDiffEqAdditionalDatasetSuite(
        comparisons=comparisons,
        summary_table=summary_table,
        detail_table=detail_table,
        metadata={
            "datasets": [comparison.dataset_name for comparison in comparisons],
            "scdiffeq_train_epochs": int(scdiffeq_train_epochs),
            "scdiffeq_batch_size": int(scdiffeq_batch_size),
            "scdiffeq_dt": float(scdiffeq_dt),
            "scdiffeq_simulation_repeats": int(scdiffeq_simulation_repeats),
            "scdiffeq_model_kwargs": dict(scdiffeq_model_kwargs or {"train_val_split": [1.0, 0.0], "train_lr": 1e-3}),
            "scdiffeq_fit_kwargs": dict(scdiffeq_fit_kwargs or {}),
        },
    )
