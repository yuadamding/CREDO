'''
PerturbSeqDynamicsData is the canonical study object; 
it stores observed cells, perturbation metadata, 
abundance-scale masses, and optional simulation truth, 
and it exposes perturbation-indexed finite-measure 
endpoint views for model training and evaluation.
'''
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


Key = Union[str, Tuple[str, str]]  # perturbation_id or (sample_id, perturbation_id)


@dataclass(slots=True)
class TimeAxis:
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
            raise ValueError(f"Missing normalized time for {self.initial_label}.")
        if self.terminal_label not in self.normalized_time:
            raise ValueError(f"Missing normalized time for {self.terminal_label}.")
        if self.normalized_time[self.initial_label] >= self.normalized_time[self.terminal_label]:
            raise ValueError("Initial time must be smaller than terminal time.")

    def t(self, label: str) -> float:
        return float(self.normalized_time[label])


@dataclass(slots=True)
class PerturbationCatalog:
    table: pd.DataFrame

    REQUIRED_COLUMNS = ("perturbation_id", "is_control")

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(f"PerturbationCatalog missing columns: {missing}")
        if self.table["perturbation_id"].duplicated().any():
            raise ValueError("perturbation_id must be unique in PerturbationCatalog.")
        if not self.table["is_control"].isin([True, False]).all():
            raise ValueError("Column 'is_control' must be boolean.")

    @property
    def perturbation_ids(self) -> list[str]:
        return self.table["perturbation_id"].astype(str).tolist()

    @property
    def controls(self) -> list[str]:
        return self.table.loc[self.table["is_control"], "perturbation_id"].astype(str).tolist()


@dataclass(slots=True)
class CellStateTable:
    obs: pd.DataFrame
    Z: np.ndarray  # shape [n_cells, d]

    REQUIRED_COLUMNS = ("cell_id", "perturbation_id", "time_label", "sample_id")

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.obs.columns]
        if missing:
            raise ValueError(f"CellStateTable missing columns: {missing}")
        if self.obs.shape[0] != self.Z.shape[0]:
            raise ValueError("obs rows and Z rows must match.")
        if self.Z.ndim != 2:
            raise ValueError("Z must have shape [n_cells, d].")
        if self.obs["cell_id"].duplicated().any():
            raise ValueError("cell_id must be unique.")
        if self.obs["time_label"].isna().any():
            raise ValueError("time_label contains NA.")
        if self.obs["perturbation_id"].isna().any():
            raise ValueError("perturbation_id contains NA.")
        if self.obs["sample_id"].isna().any():
            raise ValueError("sample_id contains NA.")

    @property
    def n_cells(self) -> int:
        return int(self.obs.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.Z.shape[1])


@dataclass(slots=True)
class MassTable:
    table: pd.DataFrame

    REQUIRED_COLUMNS = ("perturbation_id", "time_label", "sample_id", "mass")

    def validate(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(f"MassTable missing columns: {missing}")
        if (self.table["mass"] <= 0).any():
            raise ValueError("All masses must be strictly positive.")
        dup_cols = ["perturbation_id", "time_label", "sample_id"]
        if self.table.duplicated(dup_cols).any():
            raise ValueError(f"MassTable has duplicate rows for key {dup_cols}.")

    def lookup(self, perturbation_id: str, time_label: str, sample_id: str) -> float:
        sub = self.table[
            (self.table["perturbation_id"] == perturbation_id)
            & (self.table["time_label"] == time_label)
            & (self.table["sample_id"] == sample_id)
        ]
        if sub.shape[0] != 1:
            raise KeyError(
                f"Mass not found uniquely for perturbation={perturbation_id}, "
                f"time_label={time_label}, sample_id={sample_id}"
            )
        return float(sub["mass"].iloc[0])


@dataclass(slots=True)
class SimulationTruth:
    truth_params: Optional[pd.DataFrame] = None
    hidden_paths: Optional[Dict[str, Any]] = None
    context_trajectories: Optional[pd.DataFrame] = None
    simulator_config: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        return (
            self.truth_params is not None
            or self.hidden_paths is not None
            or self.context_trajectories is not None
        )


@dataclass(slots=True)
class FiniteMeasure:
    support: np.ndarray        # [n, d]
    weights: np.ndarray        # [n]
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
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative.")
        if self.total_mass <= 0:
            raise ValueError("total_mass must be positive.")
        s = float(self.weights.sum())
        if not np.isclose(s, self.total_mass, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"weights sum {s} does not match total_mass {self.total_mass}."
            )

    @property
    def n_atoms(self) -> int:
        return int(self.support.shape[0])

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.total_mass


@dataclass(slots=True)
class EndpointProblem:
    initial: Dict[Key, FiniteMeasure]
    terminal: Dict[Key, FiniteMeasure]
    catalog: PerturbationCatalog
    time_axis: TimeAxis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if set(self.initial.keys()) != set(self.terminal.keys()):
            raise ValueError("Initial and terminal measure keys must match.")
        for m in list(self.initial.values()) + list(self.terminal.values()):
            m.validate()

    @property
    def keys(self) -> list[Key]:
        return list(self.initial.keys())


@dataclass(slots=True)
class PerturbSeqDynamicsData:
    time_axis: TimeAxis
    catalog: PerturbationCatalog
    cells: CellStateTable
    masses: MassTable
    truth: Optional[SimulationTruth] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.time_axis.validate()
        self.catalog.validate()
        self.cells.validate()
        self.masses.validate()

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
        for required in [self.time_axis.initial_label, self.time_axis.terminal_label]:
            if required not in labels:
                raise ValueError(f"Missing required time label: {required}")

    @property
    def latent_dim(self) -> int:
        return self.cells.latent_dim

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.cells.obs["sample_id"].astype(str).unique().tolist())

    def subset(
        self,
        perturbations: Optional[Sequence[str]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        time_labels: Optional[Sequence[str]] = None,
    ) -> "PerturbSeqDynamicsData":
        obs = self.cells.obs.copy()
        mask = np.ones(obs.shape[0], dtype=bool)

        if perturbations is not None:
            mask &= obs["perturbation_id"].isin(perturbations).to_numpy()
        if sample_ids is not None:
            mask &= obs["sample_id"].isin(sample_ids).to_numpy()
        if time_labels is not None:
            mask &= obs["time_label"].isin(time_labels).to_numpy()

        new_obs = obs.loc[mask].reset_index(drop=True)
        new_Z = self.cells.Z[mask]

        mass_df = self.masses.table.copy()
        if perturbations is not None:
            mass_df = mass_df[mass_df["perturbation_id"].isin(perturbations)]
        if sample_ids is not None:
            mass_df = mass_df[mass_df["sample_id"].isin(sample_ids)]
        if time_labels is not None:
            mass_df = mass_df[mass_df["time_label"].isin(time_labels)]
        mass_df = mass_df.reset_index(drop=True)

        cat_df = self.catalog.table.copy()
        if perturbations is not None:
            cat_df = cat_df[cat_df["perturbation_id"].isin(perturbations)].reset_index(drop=True)

        out = replace(
            self,
            catalog=PerturbationCatalog(cat_df),
            cells=CellStateTable(new_obs, new_Z),
            masses=MassTable(mass_df),
        )
        out.validate()
        return out

    def build_measure(
        self,
        perturbation_id: str,
        time_label: str,
        sample_id: str,
    ) -> FiniteMeasure:
        obs = self.cells.obs
        mask = (
            (obs["perturbation_id"] == perturbation_id)
            & (obs["time_label"] == time_label)
            & (obs["sample_id"] == sample_id)
        ).to_numpy()

        support = self.cells.Z[mask]
        if support.shape[0] == 0:
            raise ValueError(
                f"No cells found for perturbation={perturbation_id}, "
                f"time_label={time_label}, sample_id={sample_id}"
            )

        total_mass = self.masses.lookup(
            perturbation_id=perturbation_id,
            time_label=time_label,
            sample_id=sample_id,
        )
        weights = np.full(support.shape[0], total_mass / support.shape[0], dtype=float)

        measure = FiniteMeasure(
            support=support,
            weights=weights,
            total_mass=total_mass,
            perturbation_id=str(perturbation_id),
            time_label=str(time_label),
            sample_id=str(sample_id),
        )
        measure.validate()
        return measure

    def to_endpoint_problem(self, by_sample: bool = True) -> EndpointProblem:
        initial: Dict[Key, FiniteMeasure] = {}
        terminal: Dict[Key, FiniteMeasure] = {}

        samples = self.sample_ids if by_sample else ["__pooled__"]
        perts = self.catalog.perturbation_ids

        if by_sample:
            for sample_id in samples:
                for g in perts:
                    key: Key = (sample_id, g)
                    initial[key] = self.build_measure(g, self.time_axis.initial_label, sample_id)
                    terminal[key] = self.build_measure(g, self.time_axis.terminal_label, sample_id)
        else:
            # pooled view across samples; requires pre-aggregation in masses and cells
            # or a dedicated pooling method if sample-aware data are present
            raise NotImplementedError(
                "Use by_sample=True for multi-sample data, or add an explicit pooling method."
            )

        problem = EndpointProblem(
            initial=initial,
            terminal=terminal,
            catalog=self.catalog,
            time_axis=self.time_axis,
            metadata={"latent_dim": self.latent_dim},
        )
        problem.validate()
        return problem

    def summary(self) -> pd.DataFrame:
        obs = self.cells.obs.copy()
        cell_counts = (
            obs.groupby(["sample_id", "time_label", "perturbation_id"])
            .size()
            .rename("n_cells")
            .reset_index()
        )
        out = cell_counts.merge(
            self.masses.table,
            on=["sample_id", "time_label", "perturbation_id"],
            how="left",
        )
        return out.sort_values(["sample_id", "time_label", "perturbation_id"]).reset_index(drop=True)