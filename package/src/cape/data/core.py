"""Core data structures for the P4/P60 Perturb-seq dynamics model.

The canonical study object is PerturbSeqDynamicsData, which combines
TimeAxis, PerturbationCatalog, CellStateTable, MassTable, and optional
ExposureTable / ReplicateCountTable.

All measure construction keeps total_mass distinct from normalized weights
to preserve the unbalanced OT semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# TimeAxis
# ---------------------------------------------------------------------------

@dataclass
class TimeAxis:
    """Ordered physical time labels with normalized tau in [0, 1]."""
    labels: List[str]             # e.g. ["P4", "P60"]
    physical_times: List[float]   # e.g. [4.0, 60.0]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.physical_times):
            raise ValueError("labels/times length mismatch")
        if len(self.labels) < 2:
            raise ValueError("Need at least two time points")
        for i in range(1, len(self.physical_times)):
            if not self.physical_times[i] > self.physical_times[i - 1]:
                raise ValueError(f"Times must be strictly increasing: {self.physical_times}")

    @property
    def t_min(self) -> float:
        return self.physical_times[0]

    @property
    def t_max(self) -> float:
        return self.physical_times[-1]

    def tau(self, label: str) -> float:
        """Normalized time for label."""
        idx = self.labels.index(label)
        t = self.physical_times[idx]
        return (t - self.t_min) / (self.t_max - self.t_min)

    def physical(self, label: str) -> float:
        idx = self.labels.index(label)
        return self.physical_times[idx]

    @classmethod
    def p4_p60(cls) -> "TimeAxis":
        return cls(labels=["P4", "P60"], physical_times=[4.0, 60.0])


# ---------------------------------------------------------------------------
# PerturbationCatalog
# ---------------------------------------------------------------------------

@dataclass
class PerturbationCatalog:
    """Registry of perturbation ids and which ones are controls."""
    perturbation_ids: List[str]
    control_ids: List[str]

    def __post_init__(self) -> None:
        ids = set(self.perturbation_ids)
        if len(ids) != len(self.perturbation_ids):
            raise ValueError("Duplicate perturbation_ids")
        if len(self.control_ids) == 0:
            raise ValueError("Must have at least one control")
        for c in self.control_ids:
            if c not in ids:
                raise KeyError(f"Control {c!r} not in perturbation_ids")

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbation_ids)

    @property
    def non_control_ids(self) -> List[str]:
        ctrl = set(self.control_ids)
        return [p for p in self.perturbation_ids if p not in ctrl]

    def is_control(self, pid: str) -> bool:
        return pid in set(self.control_ids)

    def index_of(self, pid: str) -> int:
        return self.perturbation_ids.index(pid)


# ---------------------------------------------------------------------------
# CellStateTable
# ---------------------------------------------------------------------------

@dataclass
class CellStateTable:
    """Single-cell latent states with required metadata columns."""
    df: pd.DataFrame     # must contain: cell_id, perturbation_id, time_label, sample_id
    latent: np.ndarray   # shape [n_cells, d]

    REQUIRED_COLS = {"cell_id", "perturbation_id", "time_label", "sample_id"}

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise KeyError(f"CellStateTable missing columns: {missing}")
        if len(self.df) != len(self.latent):
            raise ValueError(f"df rows {len(self.df)} != latent rows {len(self.latent)}")

    @property
    def n_cells(self) -> int:
        return len(self.df)

    @property
    def latent_dim(self) -> int:
        return self.latent.shape[1]

    def filter(self, mask: np.ndarray) -> "CellStateTable":
        return CellStateTable(df=self.df[mask].reset_index(drop=True), latent=self.latent[mask])

    def select_time(self, label: str) -> "CellStateTable":
        mask = self.df["time_label"].values == label
        return self.filter(mask)

    def select_perturbation(self, pid: str) -> "CellStateTable":
        mask = self.df["perturbation_id"].values == pid
        return self.filter(mask)


# ---------------------------------------------------------------------------
# MassTable
# ---------------------------------------------------------------------------

@dataclass
class MassTable:
    """Guide-abundance masses for each (perturbation_id, time_label, sample_id)."""
    df: pd.DataFrame  # must contain: perturbation_id, time_label, sample_id, mass

    REQUIRED_COLS = {"perturbation_id", "time_label", "sample_id", "mass"}

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise KeyError(f"MassTable missing columns: {missing}")
        if not (self.df["mass"] > 0).all():
            raise ValueError("All masses must be positive")

    def get(self, perturbation_id: str, time_label: str, sample_id: str) -> float:
        row = self.df[
            (self.df["perturbation_id"] == perturbation_id) &
            (self.df["time_label"] == time_label) &
            (self.df["sample_id"] == sample_id)
        ]
        if len(row) != 1:
            raise KeyError(
                f"Expected 1 row for ({perturbation_id}, {time_label}, {sample_id}), got {len(row)}"
            )
        return float(row["mass"].iloc[0])

    def get_pooled(self, perturbation_id: str, time_label: str) -> float:
        """Sum mass across all sample_ids for this perturbation and time."""
        row = self.df[
            (self.df["perturbation_id"] == perturbation_id) &
            (self.df["time_label"] == time_label)
        ]
        return float(row["mass"].sum())


# ---------------------------------------------------------------------------
# ExposureTable
# ---------------------------------------------------------------------------

@dataclass
class ExposureTable:
    """T0 matched-library guide exposure frequencies/counts."""
    df: pd.DataFrame  # required: perturbation_id, library_batch, exposure

    REQUIRED_COLS = {"perturbation_id", "library_batch", "exposure"}

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise KeyError(f"ExposureTable missing columns: {missing}")

    def get(self, perturbation_id: str, library_batch: str) -> float:
        row = self.df[
            (self.df["perturbation_id"] == perturbation_id) &
            (self.df["library_batch"] == library_batch)
        ]
        if len(row) != 1:
            raise KeyError(
                f"Expected 1 exposure row for ({perturbation_id}, {library_batch}), got {len(row)}"
            )
        return float(row["exposure"].iloc[0])

    def get_batch_vector(self, perturbation_ids: List[str], library_batch: str) -> np.ndarray:
        """Return exposure vector for given perturbation list and batch."""
        return np.array([self.get(p, library_batch) for p in perturbation_ids])


# ---------------------------------------------------------------------------
# ReplicateCountTable
# ---------------------------------------------------------------------------

@dataclass
class ReplicateCountTable:
    """Replicate-level guide counts at P4 and P60."""
    df: pd.DataFrame
    # required: sample_id, time_label, library_batch, perturbation_id, count, n_total_sample

    REQUIRED_COLS = {"sample_id", "time_label", "library_batch", "perturbation_id",
                     "count", "n_total_sample"}

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise KeyError(f"ReplicateCountTable missing columns: {missing}")

    def get_count_matrix(
        self, time_label: str, perturbation_ids: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Return count matrix [n_samples, n_perturbations], sample_ids, n_totals."""
        sub = self.df[self.df["time_label"] == time_label]
        sample_ids = sorted(sub["sample_id"].unique().tolist())
        counts = np.zeros((len(sample_ids), len(perturbation_ids)))
        n_totals = np.zeros(len(sample_ids))
        for i, sid in enumerate(sample_ids):
            for j, pid in enumerate(perturbation_ids):
                row = sub[(sub["sample_id"] == sid) & (sub["perturbation_id"] == pid)]
                if len(row) > 0:
                    counts[i, j] = float(row["count"].iloc[0])
                    n_totals[i] = float(row["n_total_sample"].iloc[0])
        return counts, sample_ids, n_totals


# ---------------------------------------------------------------------------
# LatentTransform
# ---------------------------------------------------------------------------

@dataclass
class LatentTransform:
    """Optional whitening/scaling for the latent space."""
    mean: np.ndarray     # [d]
    scale: np.ndarray    # [d]  (std or eigenvalue-based)

    def apply(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean) / (self.scale + 1e-8)

    def inverse(self, z_white: np.ndarray) -> np.ndarray:
        return z_white * self.scale + self.mean

    @classmethod
    def from_array(cls, z: np.ndarray) -> "LatentTransform":
        mean = z.mean(axis=0)
        scale = z.std(axis=0) + 1e-8
        return cls(mean=mean, scale=scale)

    def apply_torch(self, z: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=z.dtype, device=z.device)
        scale = torch.tensor(self.scale, dtype=z.dtype, device=z.device)
        return (z - mean) / (scale + 1e-8)


# ---------------------------------------------------------------------------
# FiniteMeasure
# ---------------------------------------------------------------------------

@dataclass
class FiniteMeasure:
    """Discrete finite measure: weights need not sum to 1."""
    support: np.ndarray   # [n_atoms, d]
    weights: np.ndarray   # [n_atoms], sum = total_mass
    total_mass: float

    def __post_init__(self) -> None:
        if self.support.ndim != 2:
            raise ValueError(f"support must be 2D, got {self.support.ndim}D")
        if len(self.weights) != len(self.support):
            raise ValueError("support/weights length mismatch")
        if not self.total_mass > 0:
            raise ValueError("total_mass must be positive")
        weight_sum = float(np.asarray(self.weights, dtype=float).sum())
        if not np.isclose(weight_sum, float(self.total_mass), rtol=1e-4, atol=1e-8):
            raise ValueError(
                "weights.sum() must equal total_mass "
                f"(got {weight_sum:.6g} vs {float(self.total_mass):.6g})"
            )

    @property
    def n_atoms(self) -> int:
        return len(self.support)

    @property
    def latent_dim(self) -> int:
        return self.support.shape[1]

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.total_mass

    def mean(self) -> np.ndarray:
        return (self.normalized_weights[:, None] * self.support).sum(axis=0)

    def covariance(self) -> np.ndarray:
        mu = self.mean()
        diff = self.support - mu
        return (self.normalized_weights[:, None] * diff).T @ diff

    def variance_trace(self) -> float:
        return float(np.diag(self.covariance()).sum())

    def to_torch(self, device: str = "cpu", dtype: torch.dtype = torch.float32
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        sup = torch.tensor(self.support, dtype=dtype, device=device)
        w = torch.tensor(self.weights, dtype=dtype, device=device)
        return sup, w


# ---------------------------------------------------------------------------
# EndpointProblem
# ---------------------------------------------------------------------------

@dataclass
class EndpointProblem:
    """Paired initial and terminal finite measures per perturbation."""
    initial: Dict[str, FiniteMeasure]   # keyed by perturbation_id
    terminal: Dict[str, FiniteMeasure]
    time_axis: TimeAxis
    perturbation_ids: List[str]

    def __post_init__(self) -> None:
        for pid in self.perturbation_ids:
            if pid not in self.initial:
                raise KeyError(f"Missing initial measure for {pid}")
            if pid not in self.terminal:
                raise KeyError(f"Missing terminal measure for {pid}")

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbation_ids)


# ---------------------------------------------------------------------------
# SimulationTruth  (for synthetic benchmarks)
# ---------------------------------------------------------------------------

@dataclass
class SimulationTruth:
    """Ground truth from synthetic data generation."""
    truth_params: Dict[str, Any] = field(default_factory=dict)
    analytic_summary: Optional[pd.DataFrame] = None
    hidden_paths: Optional[np.ndarray] = None          # [n_perturb, n_particles, n_steps, d]
    context_trajectories: Optional[np.ndarray] = None  # [n_steps, C]
    simulator_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProgramScoreTable
# ---------------------------------------------------------------------------

@dataclass
class ProgramScoreTable:
    """Cell-level or pooled program scores."""
    df: pd.DataFrame
    # cell-level: cell_id, program_name, score
    # pooled: perturbation_id, time_label, program_name, score
    level: str = "cell"  # "cell" or "pooled"

    def get_scores(self, program_name: str) -> pd.DataFrame:
        return self.df[self.df["program_name"] == program_name].copy()


# ---------------------------------------------------------------------------
# PerturbSeqDynamicsData
# ---------------------------------------------------------------------------

@dataclass
class PerturbSeqDynamicsData:
    """Canonical study object combining all data sources."""
    time_axis: TimeAxis
    catalog: PerturbationCatalog
    cell_state: CellStateTable
    mass_table: MassTable
    latent_transform: Optional[LatentTransform] = None
    exposure_table: Optional[ExposureTable] = None
    replicate_counts: Optional[ReplicateCountTable] = None
    program_scores: Optional[ProgramScoreTable] = None
    truth: Optional[SimulationTruth] = None

    @property
    def latent_dim(self) -> int:
        return self.cell_state.latent_dim

    def build_measure(
        self,
        perturbation_id: str,
        time_label: str,
        sample_id: str = "pooled",
        density_correct: bool = False,
    ) -> FiniteMeasure:
        """Build a FiniteMeasure for a given perturbation / time."""
        cells = self.cell_state.select_time(time_label).select_perturbation(perturbation_id)
        if sample_id != "pooled":
            sample_mask = cells.df["sample_id"].astype(str).eq(str(sample_id)).to_numpy()
            cells = cells.filter(sample_mask)
        n = cells.n_cells
        if n <= 0:
            raise ValueError(f"No cells for ({perturbation_id}, {time_label}, {sample_id})")

        if sample_id == "pooled":
            total_mass = self.mass_table.get_pooled(perturbation_id, time_label)
        else:
            total_mass = self.mass_table.get(perturbation_id, time_label, sample_id)

        support = cells.latent.copy()
        weights = np.full(n, total_mass / n)
        return FiniteMeasure(support=support, weights=weights, total_mass=total_mass)

    def to_endpoint_problem(
        self,
        perturbation_ids: Optional[List[str]] = None,
        initial_label: Optional[str] = None,
        terminal_label: Optional[str] = None,
    ) -> EndpointProblem:
        """Construct pooled EndpointProblem across replicates."""
        pids = perturbation_ids or self.catalog.perturbation_ids
        init_label = initial_label or self.time_axis.labels[0]
        term_label = terminal_label or self.time_axis.labels[-1]

        initial, terminal = {}, {}
        for pid in pids:
            initial[pid] = self.build_measure(pid, init_label)
            terminal[pid] = self.build_measure(pid, term_label)

        return EndpointProblem(
            initial=initial,
            terminal=terminal,
            time_axis=self.time_axis,
            perturbation_ids=list(pids),
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for pid in self.catalog.perturbation_ids:
            for tl in self.time_axis.labels:
                mask = (
                    (self.cell_state.df["perturbation_id"] == pid) &
                    (self.cell_state.df["time_label"] == tl)
                )
                n_cells = int(mask.sum())
                mass = self.mass_table.get_pooled(pid, tl) if n_cells > 0 else 0.0
                rows.append({
                    "perturbation_id": pid,
                    "time_label": tl,
                    "n_cells": n_cells,
                    "pooled_mass": mass,
                    "is_control": self.catalog.is_control(pid),
                })
        return pd.DataFrame(rows)
