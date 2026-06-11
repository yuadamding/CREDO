"""Core finite-measure data structures for CREDO dynamics.

The canonical study object is PerturbSeqDynamicsData, which combines
TimeAxis, PerturbationCatalog, CellStateTable, MassTable, and optional
ExposureTable / ReplicateCountTable.

All measure construction keeps total_mass distinct from normalized weights
to preserve finite-measure mass semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


MeasureKey = str | Tuple[str, str]
POOLED_SAMPLE_ID = "__pooled__"
LEGACY_POOLED_SAMPLE_ID = "pooled"
POOLED_SAMPLE_IDS = frozenset({POOLED_SAMPLE_ID, LEGACY_POOLED_SAMPLE_ID})


def is_pooled_sample_id(sample_id: str) -> bool:
    """Return True for canonical or legacy pooled-sample sentinels."""
    return str(sample_id) in POOLED_SAMPLE_IDS


def _is_valid_measure_key(key: MeasureKey) -> bool:
    return isinstance(key, str) or (
        isinstance(key, tuple)
        and len(key) == 2
        and all(isinstance(part, str) for part in key)
    )


def _format_measure_keys(keys: set[MeasureKey]) -> List[str]:
    return sorted(str(key) for key in keys)


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
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("TimeAxis labels must be unique")
        if not np.isfinite(np.asarray(self.physical_times, dtype=float)).all():
            raise ValueError("physical_times must be finite")
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
        self.perturbation_ids = [str(pid) for pid in self.perturbation_ids]
        self.control_ids = [str(pid) for pid in self.control_ids]
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
        self.df = self.df.copy()
        for col in sorted(self.REQUIRED_COLS):
            self.df[col] = self.df[col].astype(str)
        self.latent = np.asarray(self.latent)
        if len(self.df) != len(self.latent):
            raise ValueError(f"df rows {len(self.df)} != latent rows {len(self.latent)}")
        if self.latent.ndim != 2:
            raise ValueError(f"latent must be 2D, got {self.latent.ndim}D")
        if not np.isfinite(np.asarray(self.latent)).all():
            raise ValueError("latent contains NaN or inf")
        if self.df["cell_id"].duplicated().any():
            dup = self.df.loc[self.df["cell_id"].duplicated(), "cell_id"].iloc[0]
            raise ValueError(f"Duplicate cell_id in CellStateTable: {dup!r}")

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
        mass = self.df["mass"].astype(float)
        if not np.isfinite(mass.to_numpy()).all():
            raise ValueError("All masses must be finite")
        if not (mass > 0).all():
            raise ValueError("All masses must be positive")
        key_frame = pd.DataFrame(
            {
                "perturbation_id": self.df["perturbation_id"].astype(str),
                "time_label": self.df["time_label"].astype(str),
                "sample_id": self.df["sample_id"].astype(str),
            }
        )
        duplicated = key_frame.duplicated(["perturbation_id", "time_label", "sample_id"])
        if duplicated.any():
            row = self.df.loc[duplicated].iloc[0]
            raise ValueError(
                "Duplicate MassTable row for "
                f"({row['perturbation_id']}, {row['time_label']}, {row['sample_id']})"
            )
        for (pid, time_label), sub in key_frame.groupby(["perturbation_id", "time_label"], observed=True):
            sample_ids = sub["sample_id"]
            pooled_mask = sample_ids.map(is_pooled_sample_id)
            if int(pooled_mask.sum()) > 1:
                raise ValueError(
                    "MassTable has multiple pooled sentinel rows for "
                    f"({pid}, {time_label}); use only {POOLED_SAMPLE_ID!r}"
                )
            has_pooled = pooled_mask.any()
            has_sample_specific = (~pooled_mask).any()
            if has_pooled and has_sample_specific:
                raise ValueError(
                    "MassTable mixes pooled and sample-specific rows for "
                    f"({pid}, {time_label}); this can double-count pooled mass"
                )

    def get(self, perturbation_id: str, time_label: str, sample_id: str) -> float:
        if is_pooled_sample_id(sample_id):
            return self.get_pooled(perturbation_id, time_label)
        row = self.df[
            self.df["perturbation_id"].astype(str).eq(str(perturbation_id)) &
            self.df["time_label"].astype(str).eq(str(time_label)) &
            self.df["sample_id"].astype(str).eq(str(sample_id))
        ]
        if len(row) != 1:
            raise KeyError(
                f"Expected 1 row for ({perturbation_id}, {time_label}, {sample_id}), got {len(row)}"
            )
        return float(row["mass"].iloc[0])

    def get_pooled(self, perturbation_id: str, time_label: str) -> float:
        """Sum mass across all sample_ids for this perturbation and time."""
        row = self.df[
            self.df["perturbation_id"].astype(str).eq(str(perturbation_id)) &
            self.df["time_label"].astype(str).eq(str(time_label))
        ]
        if len(row) == 0:
            raise KeyError(f"No mass rows for pooled ({perturbation_id}, {time_label})")
        pooled = row[row["sample_id"].astype(str).map(is_pooled_sample_id)]
        if len(pooled) == 1:
            return float(pooled["mass"].iloc[0])
        if len(pooled) > 1:
            raise ValueError(f"Multiple pooled mass rows for ({perturbation_id}, {time_label})")
        return float(row["mass"].sum())

    def has_pooled(self, perturbation_id: str, time_label: str) -> bool:
        row = self.df[
            self.df["perturbation_id"].astype(str).eq(str(perturbation_id)) &
            self.df["time_label"].astype(str).eq(str(time_label))
        ]
        return bool(row["sample_id"].astype(str).map(is_pooled_sample_id).any())


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
        self.df = self.df.copy()
        self.df["perturbation_id"] = self.df["perturbation_id"].astype(str)
        self.df["library_batch"] = self.df["library_batch"].astype(str)
        exposure = self.df["exposure"].astype(float)
        if not np.isfinite(exposure.to_numpy()).all() or np.any(exposure.to_numpy() <= 0):
            raise ValueError("ExposureTable exposures must be positive and finite")
        duplicated = self.df.duplicated(["perturbation_id", "library_batch"])
        if duplicated.any():
            row = self.df.loc[duplicated].iloc[0]
            raise ValueError(
                "Duplicate ExposureTable row for "
                f"({row['perturbation_id']}, {row['library_batch']})"
            )

    def get(self, perturbation_id: str, library_batch: str) -> float:
        row = self.df[
            self.df["perturbation_id"].eq(str(perturbation_id)) &
            self.df["library_batch"].eq(str(library_batch))
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
        self.df = self.df.copy()
        for col in ["sample_id", "time_label", "library_batch", "perturbation_id"]:
            self.df[col] = self.df[col].astype(str)
        counts = self.df["count"].astype(float)
        totals = self.df["n_total_sample"].astype(float)
        if not np.isfinite(counts.to_numpy()).all() or np.any(counts.to_numpy() < 0):
            raise ValueError("ReplicateCountTable counts must be nonnegative and finite")
        if not np.allclose(counts.to_numpy(), np.round(counts.to_numpy()), rtol=0.0, atol=1e-6):
            raise ValueError("ReplicateCountTable counts must be integer-like")
        if not np.isfinite(totals.to_numpy()).all() or np.any(totals.to_numpy() <= 0):
            raise ValueError("ReplicateCountTable n_total_sample must be positive and finite")
        if not np.allclose(totals.to_numpy(), np.round(totals.to_numpy()), rtol=0.0, atol=1e-6):
            raise ValueError("ReplicateCountTable n_total_sample must be integer-like")
        duplicated = self.df.duplicated(["sample_id", "time_label", "library_batch", "perturbation_id"])
        if duplicated.any():
            row = self.df.loc[duplicated].iloc[0]
            raise ValueError(
                "Duplicate ReplicateCountTable row for "
                f"({row['sample_id']}, {row['time_label']}, "
                f"{row['library_batch']}, {row['perturbation_id']})"
            )

    def get_count_matrix(
        self, time_label: str, perturbation_ids: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Return count matrix [n_samples, n_perturbations], sample_ids, n_totals."""
        perturbation_ids = [str(pid) for pid in perturbation_ids]
        sub = self.df[self.df["time_label"].eq(str(time_label))]
        sample_ids = sorted(sub["sample_id"].unique().tolist())
        counts = np.zeros((len(sample_ids), len(perturbation_ids)))
        n_totals = np.zeros(len(sample_ids))
        for i, sid in enumerate(sample_ids):
            for j, pid in enumerate(perturbation_ids):
                row = sub[sub["sample_id"].eq(str(sid)) & sub["perturbation_id"].eq(str(pid))]
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
        if not np.isfinite(np.asarray(self.support)).all():
            raise ValueError("support contains NaN or inf")
        weights = np.asarray(self.weights, dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError("weights contains NaN or inf")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        if not np.isfinite(float(self.total_mass)):
            raise ValueError("total_mass must be finite")
        if not self.total_mass > 0:
            raise ValueError("total_mass must be positive")
        weight_sum = float(weights.sum())
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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for pid in self.perturbation_ids:
            if pid not in self.initial:
                raise KeyError(f"Missing initial measure for {pid}")
            if pid not in self.terminal:
                raise KeyError(f"Missing terminal measure for {pid}")

    @property
    def n_perturbations(self) -> int:
        return len(self.perturbation_ids)


@dataclass
class TrajectoryProblem:
    """Multi-time finite-measure problem.

    ``measures[time_label][key]`` stores the observed finite measure at a
    checkpoint.  For pooled data, ``key`` is a perturbation id.  For
    sample-aware data, ``key`` is ``(sample_id, perturbation_id)``.
    """
    measures: Dict[str, Dict[MeasureKey, FiniteMeasure]]
    catalog: PerturbationCatalog
    time_axis: TimeAxis
    time_labels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.time_labels) < 2:
            raise ValueError("TrajectoryProblem requires at least two time labels")
        missing = [label for label in self.time_labels if label not in self.measures]
        if missing:
            raise ValueError(f"Missing measure dictionaries for time labels: {missing}")
        for label in self.time_labels:
            self.time_axis.tau(label)

        ref_keys = set(self.measures[self.time_labels[0]].keys())
        if not ref_keys:
            raise ValueError("TrajectoryProblem requires at least one measure key")
        invalid = [key for key in ref_keys if not _is_valid_measure_key(key)]
        if invalid:
            raise ValueError(f"Invalid trajectory measure keys: {_format_measure_keys(set(invalid))}")
        ref_sample_aware = {isinstance(key, tuple) for key in ref_keys}
        if len(ref_sample_aware) != 1:
            raise ValueError("TrajectoryProblem measure keys must be all pooled ids or all sample-aware tuples")
        for label in self.time_labels[1:]:
            keys = set(self.measures[label].keys())
            invalid = [key for key in keys if not _is_valid_measure_key(key)]
            if invalid:
                raise ValueError(f"Invalid trajectory measure keys at time {label}: {_format_measure_keys(set(invalid))}")
            sample_aware = {isinstance(key, tuple) for key in keys}
            if len(sample_aware) != 1 or sample_aware != ref_sample_aware:
                raise ValueError("TrajectoryProblem measure key type must be consistent across time labels")
            if keys != ref_keys:
                raise ValueError(
                    f"Measure keys differ at time {label}: "
                    f"missing={_format_measure_keys(ref_keys - keys)}, "
                    f"extra={_format_measure_keys(keys - ref_keys)}"
                )

    @property
    def keys(self) -> List[MeasureKey]:
        return list(self.measures[self.time_labels[0]].keys())

    @property
    def perturbation_ids(self) -> List[str]:
        if all(isinstance(key, str) for key in self.keys):
            return [str(key) for key in self.keys]
        return sorted({str(key[1]) for key in self.keys if isinstance(key, tuple)})

    def tau(self, time_label: str) -> float:
        return self.time_axis.tau(time_label)

    @property
    def observed_taus(self) -> List[float]:
        return [self.tau(label) for label in self.time_labels]

    def get(self, time_label: str, key: MeasureKey) -> FiniteMeasure:
        return self.measures[time_label][key]

    def interval_pairs(self) -> List[Tuple[str, str]]:
        return list(zip(self.time_labels[:-1], self.time_labels[1:]))

    def to_endpoint_problem(
        self,
        initial_label: Optional[str] = None,
        terminal_label: Optional[str] = None,
    ) -> EndpointProblem:
        """Backward-compatible pooled endpoint view.

        Existing endpoint code expects perturbation-id keys, so sample-aware
        trajectory keys must be pooled before taking this view.
        """
        init_label = initial_label or self.time_labels[0]
        term_label = terminal_label or self.time_labels[-1]
        initial = self.measures[init_label]
        terminal = self.measures[term_label]
        if not all(isinstance(key, str) for key in initial):
            raise ValueError("Sample-aware TrajectoryProblem cannot be viewed as an EndpointProblem")
        perturbation_ids = [str(key) for key in initial.keys()]
        return EndpointProblem(
            initial={str(key): value for key, value in initial.items()},
            terminal={str(key): value for key, value in terminal.items()},
            time_axis=self.time_axis,
            perturbation_ids=perturbation_ids,
        )


@dataclass
class SparseTrajectoryProblem:
    """Multi-time finite-measure problem with missing sample/time keys allowed.

    ``measures[time_label][key]`` stores only observed finite measures.  This is
    useful for donor-aware time courses where not every donor/perturbation
    combination exists at every observed time.
    """
    measures: Dict[str, Dict[MeasureKey, FiniteMeasure]]
    catalog: PerturbationCatalog
    time_axis: TimeAxis
    time_labels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.time_labels) < 2:
            raise ValueError("SparseTrajectoryProblem requires at least two time labels")
        missing = [label for label in self.time_labels if label not in self.measures]
        if missing:
            raise ValueError(f"Missing measure dictionaries for time labels: {missing}")
        for label in self.time_labels:
            self.time_axis.tau(label)

        all_keys: set[MeasureKey] = set()
        key_modes: set[bool] = set()
        for label in self.time_labels:
            keys = set(self.measures[label].keys())
            invalid = [key for key in keys if not _is_valid_measure_key(key)]
            if invalid:
                raise ValueError(
                    f"Invalid sparse trajectory measure keys at time {label}: "
                    f"{_format_measure_keys(set(invalid))}"
                )
            all_keys.update(keys)
            key_modes.update(isinstance(key, tuple) for key in keys)
        if not all_keys:
            raise ValueError("SparseTrajectoryProblem requires at least one observed measure")
        if len(key_modes) > 1:
            raise ValueError("SparseTrajectoryProblem measure keys must be all pooled ids or all sample-aware tuples")

    @property
    def keys(self) -> List[MeasureKey]:
        out: set[MeasureKey] = set()
        for label in self.time_labels:
            out.update(self.measures[label].keys())
        return sorted(out, key=str)

    @property
    def perturbation_ids(self) -> List[str]:
        if all(isinstance(key, str) for key in self.keys):
            return [str(key) for key in self.keys]
        return sorted({str(key[1]) for key in self.keys if isinstance(key, tuple)})

    @property
    def observed_taus(self) -> List[float]:
        return [self.tau(label) for label in self.time_labels]

    def tau(self, time_label: str) -> float:
        return self.time_axis.tau(time_label)

    def available_keys(self, time_label: str) -> set[MeasureKey]:
        if time_label not in self.measures:
            raise KeyError(f"Unknown trajectory time label: {time_label!r}")
        return set(self.measures[time_label].keys())

    def target_keys(self, source_label: str, target_label: str) -> set[MeasureKey]:
        return self.available_keys(source_label) & self.available_keys(target_label)

    def get(self, time_label: str, key: MeasureKey) -> FiniteMeasure:
        return self.measures[time_label][key]

    def interval_pairs(self) -> List[Tuple[str, str]]:
        return list(zip(self.time_labels[:-1], self.time_labels[1:]))

    def to_endpoint_problem(
        self,
        initial_label: Optional[str] = None,
        terminal_label: Optional[str] = None,
    ) -> EndpointProblem:
        init_label = initial_label or self.time_labels[0]
        term_label = terminal_label or self.time_labels[-1]
        common_keys = self.target_keys(init_label, term_label)
        if not common_keys:
            raise ValueError(f"No common measure keys for endpoint view {init_label!r}->{term_label!r}")
        if not all(isinstance(key, str) for key in common_keys):
            raise ValueError("Sample-aware SparseTrajectoryProblem cannot be viewed as an EndpointProblem")
        perturbation_ids = sorted(str(key) for key in common_keys)
        return EndpointProblem(
            initial={pid: self.measures[init_label][pid] for pid in perturbation_ids},
            terminal={pid: self.measures[term_label][pid] for pid in perturbation_ids},
            time_axis=self.time_axis,
            perturbation_ids=perturbation_ids,
        )


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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate cross-table consistency for measure construction."""
        catalog_ids = set(self.catalog.perturbation_ids)
        time_labels = set(self.time_axis.labels)

        cell_pids = set(self.cell_state.df["perturbation_id"].astype(str))
        unknown_cell_pids = cell_pids - catalog_ids
        if unknown_cell_pids:
            raise ValueError(f"CellStateTable has perturbations outside catalog: {sorted(unknown_cell_pids)}")

        mass_pids = set(self.mass_table.df["perturbation_id"].astype(str))
        unknown_mass_pids = mass_pids - catalog_ids
        if unknown_mass_pids:
            raise ValueError(f"MassTable has perturbations outside catalog: {sorted(unknown_mass_pids)}")

        cell_times = set(self.cell_state.df["time_label"].astype(str))
        unknown_cell_times = cell_times - time_labels
        if unknown_cell_times:
            raise ValueError(f"CellStateTable has time labels outside TimeAxis: {sorted(unknown_cell_times)}")

        mass_times = set(self.mass_table.df["time_label"].astype(str))
        unknown_mass_times = mass_times - time_labels
        if unknown_mass_times:
            raise ValueError(f"MassTable has time labels outside TimeAxis: {sorted(unknown_mass_times)}")

        cell_keys = set(
            zip(
                self.cell_state.df["perturbation_id"].astype(str),
                self.cell_state.df["time_label"].astype(str),
                self.cell_state.df["sample_id"].astype(str),
            )
        )
        mass_keys = set(
            zip(
                self.mass_table.df["perturbation_id"].astype(str),
                self.mass_table.df["time_label"].astype(str),
                self.mass_table.df["sample_id"].astype(str),
            )
        )
        pooled_mass_keys = {
            (pid, time_label)
            for pid, time_label, sample_id in mass_keys
            if is_pooled_sample_id(sample_id)
        }
        observed_samples = (
            self.cell_state.df.assign(
                perturbation_id=self.cell_state.df["perturbation_id"].astype(str),
                time_label=self.cell_state.df["time_label"].astype(str),
                sample_id=self.cell_state.df["sample_id"].astype(str),
            )
            .groupby(["perturbation_id", "time_label"], observed=True)["sample_id"]
            .agg(lambda values: set(values))
            .to_dict()
        )
        missing_mass = set()
        unsafe_pooled_fallback = set()
        for pid, time_label, sample_id in cell_keys:
            if (pid, time_label, sample_id) in mass_keys:
                continue
            if (pid, time_label) in pooled_mass_keys:
                samples = observed_samples.get((pid, time_label), set())
                if len(samples) <= 1 or is_pooled_sample_id(sample_id):
                    continue
                unsafe_pooled_fallback.add((pid, time_label, sample_id))
                continue
            missing_mass.add((pid, time_label, sample_id))
        if missing_mass:
            preview = sorted(str(key) for key in missing_mass)[:5]
            raise ValueError(f"Missing MassTable rows for observed cell groups: {preview}")
        if unsafe_pooled_fallback:
            preview = sorted(str(key) for key in unsafe_pooled_fallback)[:5]
            raise ValueError(
                "Sample-specific MassTable rows are required when multiple samples "
                f"share a perturbation/time group: {preview}"
            )

        if self.exposure_table is not None:
            exposure_pids = set(self.exposure_table.df["perturbation_id"].astype(str))
            unknown_exposure_pids = exposure_pids - catalog_ids
            if unknown_exposure_pids:
                raise ValueError(f"ExposureTable has perturbations outside catalog: {sorted(unknown_exposure_pids)}")
            exposure = self.exposure_table.df["exposure"].astype(float).to_numpy()
            if not np.isfinite(exposure).all() or np.any(exposure <= 0):
                raise ValueError("ExposureTable exposures must be positive and finite")

        if self.replicate_counts is not None:
            count_pids = set(self.replicate_counts.df["perturbation_id"].astype(str))
            unknown_count_pids = count_pids - catalog_ids
            if unknown_count_pids:
                raise ValueError(f"ReplicateCountTable has perturbations outside catalog: {sorted(unknown_count_pids)}")
            count_times = set(self.replicate_counts.df["time_label"].astype(str))
            unknown_count_times = count_times - time_labels
            if unknown_count_times:
                raise ValueError(f"ReplicateCountTable has time labels outside TimeAxis: {sorted(unknown_count_times)}")

    @property
    def latent_dim(self) -> int:
        return self.cell_state.latent_dim

    def build_measure(
        self,
        perturbation_id: str,
        time_label: str,
        sample_id: str = POOLED_SAMPLE_ID,
        density_correct: bool = False,
    ) -> FiniteMeasure:
        """Build a FiniteMeasure for a given perturbation / time."""
        if density_correct:
            raise NotImplementedError(
                "density_correct=True is not implemented for PerturbSeqDynamicsData.build_measure; "
                "use explicit precomputed cell weights or leave density_correct=False."
            )
        cells_all = self.cell_state.select_time(time_label).select_perturbation(perturbation_id)
        cells = cells_all
        if not is_pooled_sample_id(sample_id):
            sample_mask = cells.df["sample_id"].astype(str).eq(str(sample_id)).to_numpy()
            cells = cells.filter(sample_mask)
        n = cells.n_cells
        if n <= 0:
            raise ValueError(f"No cells for ({perturbation_id}, {time_label}, {sample_id})")

        if is_pooled_sample_id(sample_id):
            mass_rows = self.mass_table.df[
                self.mass_table.df["perturbation_id"].astype(str).eq(str(perturbation_id)) &
                self.mass_table.df["time_label"].astype(str).eq(str(time_label))
            ].copy()
            if len(mass_rows) == 0:
                raise KeyError(f"No mass rows for pooled ({perturbation_id}, {time_label})")
            has_explicit_pooled = mass_rows["sample_id"].astype(str).map(is_pooled_sample_id).any()
            support = cells.latent.copy()
            if has_explicit_pooled:
                total_mass = self.mass_table.get_pooled(perturbation_id, time_label)
                weights = np.full(n, total_mass / n)
                return FiniteMeasure(support=support, weights=weights, total_mass=total_mass)

            # If pooled geometry is built from sample-specific mass rows, keep
            # each sample's contribution proportional to its finite-measure
            # mass instead of raw captured-cell recovery.
            sample_ids = cells.df["sample_id"].astype(str).to_numpy()
            weights = np.zeros(n, dtype=float)
            total_mass = 0.0
            for sid in sorted(set(sample_ids)):
                mask = sample_ids == sid
                mass_s = self.mass_table.get(perturbation_id, time_label, sid)
                weights[mask] = mass_s / float(mask.sum())
                total_mass += mass_s
            return FiniteMeasure(support=support, weights=weights, total_mass=total_mass)
        else:
            try:
                total_mass = self.mass_table.get(perturbation_id, time_label, sample_id)
            except KeyError as exc:
                observed = set(cells_all.df["sample_id"].astype(str))
                if self.mass_table.has_pooled(perturbation_id, time_label) and len(observed) <= 1:
                    total_mass = self.mass_table.get_pooled(perturbation_id, time_label)
                else:
                    raise KeyError(
                        "Missing sample-specific mass row for "
                        f"({perturbation_id}, {time_label}, {sample_id}); "
                        "pooled fallback is only allowed for single-sample groups."
                    ) from exc

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

    def to_trajectory_problem(
        self,
        *,
        by_sample: bool = False,
        time_labels: Optional[Sequence[str]] = None,
        perturbations: Optional[Sequence[str]] = None,
        require_all_times: bool = True,
    ) -> TrajectoryProblem | SparseTrajectoryProblem:
        """Construct a multi-time finite-measure view.

        By default this returns pooled perturbation-keyed measures for every
        observed time label in ``self.time_axis``.  With ``by_sample=True``,
        keys are ``(sample_id, perturbation_id)`` and only sample/perturbation
        pairs with support at every requested time are kept.
        """
        if not require_all_times:
            return self.to_sparse_trajectory_problem(
                by_sample=by_sample,
                time_labels=time_labels,
                perturbations=perturbations,
            )
        labels = list(self.time_axis.labels if time_labels is None else time_labels)
        if len(labels) < 2:
            raise ValueError("Need at least two time labels for a trajectory problem")
        for label in labels:
            self.time_axis.tau(label)

        requested_pids = list(perturbations or self.catalog.perturbation_ids)
        df = self.cell_state.df
        requested_pid_set = set(requested_pids)

        def has_cells(pid: str, label: str, sample_id: Optional[str] = None) -> bool:
            mask = (df["perturbation_id"] == pid) & (df["time_label"] == label)
            if sample_id is not None:
                mask = mask & (df["sample_id"].astype(str) == str(sample_id))
            return bool(mask.any())

        keys: List[MeasureKey] = []
        if by_sample:
            pairs = (
                df.loc[df["perturbation_id"].isin(requested_pid_set), ["sample_id", "perturbation_id"]]
                .drop_duplicates()
                .sort_values(["sample_id", "perturbation_id"])
            )
            for row in pairs.itertuples(index=False):
                sample_id = str(row.sample_id)
                pid = str(row.perturbation_id)
                if require_all_times and not all(has_cells(pid, label, sample_id) for label in labels):
                    continue
                keys.append((sample_id, pid))
        else:
            for pid in requested_pids:
                if require_all_times and not all(has_cells(pid, label) for label in labels):
                    continue
                keys.append(str(pid))

        if not keys:
            raise ValueError("No perturbation/sample keys have support at the requested time labels")

        measures: Dict[str, Dict[MeasureKey, FiniteMeasure]] = {label: {} for label in labels}
        for label in labels:
            for key in keys:
                if isinstance(key, tuple):
                    sample_id, pid = key
                    measures[label][key] = self.build_measure(pid, label, sample_id=sample_id)
                else:
                    measures[label][key] = self.build_measure(str(key), label)

        return TrajectoryProblem(
            measures=measures,
            catalog=self.catalog,
            time_axis=self.time_axis,
            time_labels=labels,
            metadata={
                "latent_dim": self.latent_dim,
                "by_sample": by_sample,
                "require_all_times": require_all_times,
            },
        )

    def to_sparse_trajectory_problem(
        self,
        *,
        by_sample: bool = False,
        time_labels: Optional[Sequence[str]] = None,
        perturbations: Optional[Sequence[str]] = None,
    ) -> SparseTrajectoryProblem:
        """Construct a trajectory view that keeps incomplete sample/time keys."""
        labels = list(self.time_axis.labels if time_labels is None else time_labels)
        if len(labels) < 2:
            raise ValueError("Need at least two time labels for a sparse trajectory problem")
        for label in labels:
            self.time_axis.tau(label)

        requested_pids = list(perturbations or self.catalog.perturbation_ids)
        requested_pid_set = set(requested_pids)
        df = self.cell_state.df

        def has_cells(pid: str, label: str, sample_id: Optional[str] = None) -> bool:
            mask = (df["perturbation_id"] == pid) & (df["time_label"] == label)
            if sample_id is not None:
                mask = mask & (df["sample_id"].astype(str) == str(sample_id))
            return bool(mask.any())

        measures: Dict[str, Dict[MeasureKey, FiniteMeasure]] = {label: {} for label in labels}
        if by_sample:
            pairs = (
                df.loc[df["perturbation_id"].isin(requested_pid_set), ["sample_id", "perturbation_id"]]
                .drop_duplicates()
                .sort_values(["sample_id", "perturbation_id"])
            )
            for row in pairs.itertuples(index=False):
                sample_id = str(row.sample_id)
                pid = str(row.perturbation_id)
                key: MeasureKey = (sample_id, pid)
                for label in labels:
                    if has_cells(pid, label, sample_id):
                        measures[label][key] = self.build_measure(pid, label, sample_id=sample_id)
        else:
            for pid in requested_pids:
                for label in labels:
                    if has_cells(str(pid), label):
                        measures[label][str(pid)] = self.build_measure(str(pid), label)

        return SparseTrajectoryProblem(
            measures=measures,
            catalog=self.catalog,
            time_axis=self.time_axis,
            time_labels=labels,
            metadata={
                "latent_dim": self.latent_dim,
                "by_sample": by_sample,
                "require_all_times": False,
            },
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
