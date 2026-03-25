from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Key = Union[str, Tuple[str, str]]  # perturbation_id or (sample_id, perturbation_id)


@dataclass(slots=True)
class TimeAxis:
    """Normalized and optional physical time metadata for the P4 -> P60 interval."""

    initial_label: str = "P4"
    terminal_label: str = "P60"
    normalized_time: Mapping[str, float] = field(
        default_factory=lambda: {"P4": 0.0, "P60": 1.0}
    )
    physical_time: Mapping[str, float] = field(
        default_factory=lambda: {"P4": 4.0, "P60": 60.0}
    )

    def validate(self) -> None:
        if self.initial_label not in self.normalized_time:
            raise ValueError(f"Missing normalized time for {self.initial_label!r}.")
        if self.terminal_label not in self.normalized_time:
            raise ValueError(f"Missing normalized time for {self.terminal_label!r}.")
        if self.normalized_time[self.initial_label] >= self.normalized_time[self.terminal_label]:
            raise ValueError("Initial time must be strictly smaller than terminal time.")

    def t(self, label: str) -> float:
        if label not in self.normalized_time:
            raise KeyError(f"Unknown time label {label!r}.")
        return float(self.normalized_time[label])


@dataclass(slots=True)
class PerturbationCatalog:
    """Catalog of perturbations and control membership."""

    table: pd.DataFrame

    REQUIRED_COLUMNS = ("perturbation_id", "is_control")

    def validate(self) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.table.columns]
        if missing:
            raise ValueError(f"PerturbationCatalog missing required columns: {missing}")
        ids = self.table["perturbation_id"].astype(str)
        if ids.duplicated().any():
            raise ValueError("perturbation_id values must be unique in the catalog.")
        if not self.table["is_control"].isin([True, False]).all():
            raise ValueError("Column 'is_control' must contain only booleans.")
        if self.table.shape[0] == 0:
            raise ValueError("PerturbationCatalog must contain at least one perturbation.")
        if self.table.loc[self.table["is_control"], :].shape[0] == 0:
            raise ValueError("PerturbationCatalog must contain at least one control perturbation.")

    @property
    def perturbation_ids(self) -> list[str]:
        return self.table["perturbation_id"].astype(str).tolist()

    @property
    def controls(self) -> list[str]:
        return self.table.loc[self.table["is_control"], "perturbation_id"].astype(str).tolist()

    def is_control(self, perturbation_id: str) -> bool:
        sub = self.table.loc[self.table["perturbation_id"].astype(str) == str(perturbation_id), "is_control"]
        if sub.shape[0] != 1:
            raise KeyError(f"Unknown perturbation_id {perturbation_id!r}.")
        return bool(sub.iloc[0])


@dataclass(slots=True)
class CellStateTable:
    """Observed single-cell latent states with aligned metadata."""

    obs: pd.DataFrame
    Z: np.ndarray  # shape [n_cells, d]

    REQUIRED_COLUMNS = ("cell_id", "perturbation_id", "time_label", "sample_id")

    def validate(self) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.obs.columns]
        if missing:
            raise ValueError(f"CellStateTable missing required columns: {missing}")
        if self.obs.shape[0] != self.Z.shape[0]:
            raise ValueError("obs rows and Z rows must match.")
        if self.Z.ndim != 2:
            raise ValueError("Z must have shape [n_cells, d].")
        if self.obs.shape[0] == 0:
            raise ValueError("CellStateTable must contain at least one cell.")
        if self.obs["cell_id"].astype(str).duplicated().any():
            raise ValueError("cell_id values must be unique.")
        for column in ("perturbation_id", "time_label", "sample_id"):
            if self.obs[column].isna().any():
                raise ValueError(f"Column {column!r} contains missing values.")
        if not np.isfinite(self.Z).all():
            raise ValueError("Z contains non-finite values.")

    @property
    def n_cells(self) -> int:
        return int(self.obs.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.Z.shape[1])


@dataclass(slots=True)
class MassTable:
    """Guide-abundance masses, separate from single-cell sample counts."""

    table: pd.DataFrame

    REQUIRED_COLUMNS = ("perturbation_id", "time_label", "sample_id", "mass")

    def validate(self) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.table.columns]
        if missing:
            raise ValueError(f"MassTable missing required columns: {missing}")
        if self.table.shape[0] == 0:
            raise ValueError("MassTable must contain at least one row.")
        if (self.table["mass"].astype(float) <= 0).any():
            raise ValueError("All masses must be strictly positive.")
        if not np.isfinite(self.table["mass"].astype(float).to_numpy()).all():
            raise ValueError("MassTable contains non-finite mass values.")
        key_cols = ["perturbation_id", "time_label", "sample_id"]
        if self.table.duplicated(key_cols).any():
            raise ValueError(f"MassTable has duplicate entries for key {key_cols}.")

    def lookup(self, perturbation_id: str, time_label: str, sample_id: str) -> float:
        mask = (
            self.table["perturbation_id"].astype(str).eq(str(perturbation_id))
            & self.table["time_label"].astype(str).eq(str(time_label))
            & self.table["sample_id"].astype(str).eq(str(sample_id))
        )
        sub = self.table.loc[mask, "mass"]
        if sub.shape[0] != 1:
            raise KeyError(
                "Mass not found uniquely for "
                f"perturbation={perturbation_id!r}, time_label={time_label!r}, sample_id={sample_id!r}."
            )
        return float(sub.iloc[0])


@dataclass(slots=True)
class LatentTransform:
    """Optional latent-space scaling/whitening metadata for downstream OT costs."""

    mean: np.ndarray
    covariance: np.ndarray
    whitening: np.ndarray
    feature_names: Tuple[str, ...] = field(default_factory=tuple)
    epsilon: float = 1e-6

    def validate(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("LatentTransform.mean must be one-dimensional.")
        d = self.mean.shape[0]
        if self.covariance.shape != (d, d):
            raise ValueError("LatentTransform.covariance must have shape [d, d].")
        if self.whitening.shape != (d, d):
            raise ValueError("LatentTransform.whitening must have shape [d, d].")
        if self.feature_names and len(self.feature_names) != d:
            raise ValueError("feature_names length must equal latent dimension.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not np.isfinite(self.mean).all():
            raise ValueError("LatentTransform.mean contains non-finite values.")
        if not np.isfinite(self.covariance).all():
            raise ValueError("LatentTransform.covariance contains non-finite values.")
        if not np.isfinite(self.whitening).all():
            raise ValueError("LatentTransform.whitening contains non-finite values.")

    @classmethod
    def from_array(
        cls,
        Z: np.ndarray,
        epsilon: float = 1e-6,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "LatentTransform":
        if Z.ndim != 2:
            raise ValueError("Z must have shape [n_cells, d].")
        if Z.shape[0] < 2:
            raise ValueError("At least two rows are required to estimate a covariance matrix.")
        mean = Z.mean(axis=0)
        centered = Z - mean
        covariance = np.cov(centered, rowvar=False)
        covariance = np.atleast_2d(covariance)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.clip(eigvals, epsilon, None)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        transform = cls(
            mean=mean.astype(float),
            covariance=covariance.astype(float),
            whitening=whitening.astype(float),
            feature_names=tuple(feature_names or []),
            epsilon=float(epsilon),
        )
        transform.validate()
        return transform

    def apply(self, Z: np.ndarray) -> np.ndarray:
        self.validate()
        if Z.ndim != 2 or Z.shape[1] != self.mean.shape[0]:
            raise ValueError(
                "Z must have shape [n, d] with d matching the latent transform dimension."
            )
        return (Z - self.mean) @ self.whitening.T


@dataclass(slots=True)
class SimulationTruth:
    """Optional container for synthetic ground-truth metadata."""

    truth_params: Optional[pd.DataFrame] = None
    analytic_summary: Optional[pd.DataFrame] = None
    hidden_paths: Optional[Dict[str, Any]] = None
    context_trajectories: Optional[pd.DataFrame] = None
    simulator_config: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        return any(
            item is not None
            for item in (self.truth_params, self.analytic_summary, self.hidden_paths, self.context_trajectories)
        )


@dataclass(slots=True)
class FiniteMeasure:
    """Finite nonnegative measure on latent state space with explicit total mass."""

    support: np.ndarray  # [n, d]
    weights: np.ndarray  # [n]
    total_mass: float
    perturbation_id: str
    time_label: str
    sample_id: str

    def validate(self) -> None:
        if self.support.ndim != 2:
            raise ValueError("support must have shape [n, d].")
        if self.weights.ndim != 1:
            raise ValueError("weights must have shape [n].")
        if self.support.shape[0] != self.weights.shape[0]:
            raise ValueError("support rows and weights length must match.")
        if self.support.shape[0] == 0:
            raise ValueError("FiniteMeasure must contain at least one support atom.")
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative.")
        if self.total_mass <= 0:
            raise ValueError("total_mass must be strictly positive.")
        if not np.isfinite(self.total_mass):
            raise ValueError("total_mass must be finite.")
        if not np.isfinite(self.support).all():
            raise ValueError("support contains non-finite values.")
        if not np.isfinite(self.weights).all():
            raise ValueError("weights contain non-finite values.")
        weight_sum = float(self.weights.sum())
        if not np.isclose(weight_sum, self.total_mass, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"weights sum {weight_sum} does not match total_mass {self.total_mass}."
            )

    @property
    def n_atoms(self) -> int:
        return int(self.support.shape[0])

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.total_mass

    def mean(self) -> np.ndarray:
        p = self.normalized_weights[:, None]
        return np.sum(p * self.support, axis=0)

    def covariance(self) -> np.ndarray:
        mu = self.mean()
        centered = self.support - mu
        return centered.T @ (centered * self.normalized_weights[:, None])

    def variance_trace(self) -> float:
        return float(np.trace(self.covariance()))


@dataclass(slots=True)
class EndpointProblem:
    """Paired initial and terminal finite-measure endpoints for each perturbation key."""

    initial: Dict[Key, FiniteMeasure]
    terminal: Dict[Key, FiniteMeasure]
    catalog: PerturbationCatalog
    time_axis: TimeAxis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.catalog.validate()
        self.time_axis.validate()
        if set(self.initial.keys()) != set(self.terminal.keys()):
            raise ValueError("Initial and terminal measure keys must match exactly.")
        if not self.initial:
            raise ValueError("EndpointProblem must contain at least one key.")
        known_perts = set(self.catalog.perturbation_ids)
        for key, measure in list(self.initial.items()) + list(self.terminal.items()):
            measure.validate()
            perturbation_id = key[1] if isinstance(key, tuple) else key
            if str(perturbation_id) not in known_perts:
                raise ValueError(f"Endpoint key {key!r} contains perturbation outside catalog.")

    @property
    def keys(self) -> list[Key]:
        return list(self.initial.keys())

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for key in self.keys:
            init = self.initial[key]
            term = self.terminal[key]
            rows.append(
                {
                    "key": key,
                    "sample_id": init.sample_id,
                    "perturbation_id": init.perturbation_id,
                    "initial_n_atoms": init.n_atoms,
                    "terminal_n_atoms": term.n_atoms,
                    "initial_mass": init.total_mass,
                    "terminal_mass": term.total_mass,
                    "initial_mean_0": float(init.mean()[0]),
                    "terminal_mean_0": float(term.mean()[0]),
                    "initial_var_trace": init.variance_trace(),
                    "terminal_var_trace": term.variance_trace(),
                }
            )
        return pd.DataFrame(rows)


@dataclass(slots=True)
class PerturbSeqDynamicsData:
    """Canonical study object for P4 -> P60 perturbation-indexed endpoint data."""

    time_axis: TimeAxis
    catalog: PerturbationCatalog
    cells: CellStateTable
    masses: MassTable
    latent_transform: Optional[LatentTransform] = None
    truth: Optional[SimulationTruth] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.time_axis.validate()
        self.catalog.validate()
        self.cells.validate()
        self.masses.validate()
        if self.latent_transform is not None:
            self.latent_transform.validate()
            if self.latent_transform.mean.shape[0] != self.cells.latent_dim:
                raise ValueError("latent_transform dimension does not match cell latent dimension.")

        known_perts = set(self.catalog.perturbation_ids)
        obs_perts = set(self.cells.obs["perturbation_id"].astype(str))
        mass_perts = set(self.masses.table["perturbation_id"].astype(str))
        if not obs_perts.issubset(known_perts):
            missing = sorted(obs_perts - known_perts)
            raise ValueError(f"Cells contain perturbations missing from catalog: {missing}")
        if not mass_perts.issubset(known_perts):
            missing = sorted(mass_perts - known_perts)
            raise ValueError(f"Masses contain perturbations missing from catalog: {missing}")

        labels = set(self.cells.obs["time_label"].astype(str)) | set(
            self.masses.table["time_label"].astype(str)
        )
        for required in (self.time_axis.initial_label, self.time_axis.terminal_label):
            if required not in labels:
                raise ValueError(f"Missing required time label: {required}")

        key_cols = ["perturbation_id", "time_label", "sample_id"]
        obs_keys = (
            self.cells.obs[key_cols].astype(str).drop_duplicates().sort_values(key_cols).reset_index(drop=True)
        )
        mass_keys = (
            self.masses.table[key_cols].astype(str).drop_duplicates().sort_values(key_cols).reset_index(drop=True)
        )
        merged = obs_keys.merge(mass_keys, on=key_cols, how="left", indicator=True)
        missing_mass = merged.loc[merged["_merge"] != "both", key_cols]
        if not missing_mass.empty:
            raise ValueError(
                "MassTable is missing rows for some observed cell groups: "
                f"{missing_mass.to_dict(orient='records')}"
            )

    @property
    def latent_dim(self) -> int:
        return self.cells.latent_dim

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.cells.obs["sample_id"].astype(str).unique().tolist())

    def with_inferred_latent_transform(self, epsilon: float = 1e-6) -> "PerturbSeqDynamicsData":
        transform = LatentTransform.from_array(self.cells.Z, epsilon=epsilon)
        out = replace(self, latent_transform=transform)
        out.validate()
        return out

    def subset(
        self,
        perturbations: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        time_labels: Optional[Sequence[str]] = None,
    ) -> "PerturbSeqDynamicsData":
        obs = self.cells.obs.copy()
        mask = np.ones(obs.shape[0], dtype=bool)

        if perturbations is not None:
            perturbations = [str(value) for value in perturbations]
            mask &= obs["perturbation_id"].astype(str).isin(perturbations).to_numpy()
        if sample_ids is not None:
            sample_ids = [str(value) for value in sample_ids]
            mask &= obs["sample_id"].astype(str).isin(sample_ids).to_numpy()
        if time_labels is not None:
            time_labels = [str(value) for value in time_labels]
            mask &= obs["time_label"].astype(str).isin(time_labels).to_numpy()

        new_obs = obs.loc[mask].reset_index(drop=True)
        new_Z = self.cells.Z[mask]

        mass_df = self.masses.table.copy()
        if perturbations is not None:
            mass_df = mass_df[mass_df["perturbation_id"].astype(str).isin(perturbations)]
        if sample_ids is not None:
            mass_df = mass_df[mass_df["sample_id"].astype(str).isin(sample_ids)]
        if time_labels is not None:
            mass_df = mass_df[mass_df["time_label"].astype(str).isin(time_labels)]
        mass_df = mass_df.reset_index(drop=True)

        cat_df = self.catalog.table.copy()
        if perturbations is not None:
            cat_df = cat_df[cat_df["perturbation_id"].astype(str).isin(perturbations)].reset_index(drop=True)

        out = replace(
            self,
            catalog=PerturbationCatalog(cat_df),
            cells=CellStateTable(new_obs, new_Z),
            masses=MassTable(mass_df),
        )
        out.validate()
        return out

    def build_measure(self, perturbation_id: str, time_label: str, sample_id: str) -> FiniteMeasure:
        obs = self.cells.obs
        mask = (
            obs["perturbation_id"].astype(str).eq(str(perturbation_id))
            & obs["time_label"].astype(str).eq(str(time_label))
            & obs["sample_id"].astype(str).eq(str(sample_id))
        ).to_numpy()
        support = self.cells.Z[mask]
        if support.shape[0] == 0:
            raise ValueError(
                "No cells found for "
                f"perturbation={perturbation_id!r}, time_label={time_label!r}, sample_id={sample_id!r}."
            )

        total_mass = self.masses.lookup(
            perturbation_id=str(perturbation_id),
            time_label=str(time_label),
            sample_id=str(sample_id),
        )
        weights = np.full(support.shape[0], total_mass / support.shape[0], dtype=float)
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

    def to_endpoint_problem(self, by_sample: bool = True) -> EndpointProblem:
        initial: Dict[Key, FiniteMeasure] = {}
        terminal: Dict[Key, FiniteMeasure] = {}

        if not by_sample:
            raise NotImplementedError(
                "Step 1 keeps the interface sample-aware by default. Pooling should be an explicit later step."
            )

        for sample_id in self.sample_ids:
            for perturbation_id in self.catalog.perturbation_ids:
                key: Key = (sample_id, perturbation_id)
                initial[key] = self.build_measure(
                    perturbation_id=perturbation_id,
                    time_label=self.time_axis.initial_label,
                    sample_id=sample_id,
                )
                terminal[key] = self.build_measure(
                    perturbation_id=perturbation_id,
                    time_label=self.time_axis.terminal_label,
                    sample_id=sample_id,
                )

        problem = EndpointProblem(
            initial=initial,
            terminal=terminal,
            catalog=self.catalog,
            time_axis=self.time_axis,
            metadata={
                "latent_dim": self.latent_dim,
                "has_latent_transform": self.latent_transform is not None,
                **self.metadata,
            },
        )
        problem.validate()
        return problem

    def summary(self) -> pd.DataFrame:
        obs = self.cells.obs.copy()
        cell_counts = (
            obs.groupby(["sample_id", "time_label", "perturbation_id"]).size().rename("n_cells").reset_index()
        )
        out = cell_counts.merge(
            self.masses.table,
            on=["sample_id", "time_label", "perturbation_id"],
            how="left",
        )
        return out.sort_values(["sample_id", "time_label", "perturbation_id"]).reset_index(drop=True)
