"""Single-time Perturb-seq views for CREDO effect-path modeling.

This module intentionally does not relax :class:`TimeAxis` or
``TrajectoryProblem``.  A one-snapshot assay is represented as a distinct
``SingleTimeProblem`` and adapted internally to a two-point, non-physical
control-reference -> observed-snapshot effect axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import EndpointProblem, FiniteMeasure, PerturbationCatalog, TimeAxis


SingleTimeMassMode = Literal["cell_count", "unit_mass", "obs_column", "unavailable"]
SingleTimeEmbeddingLevel = Literal["perturbation", "guide", "target_gene"]
SingleTimeEndpointViewLevel = Literal["embedding", "perturbation", "view"]
ControlReferenceScope = Literal["auto", "sample", "batch", "global"]
SingleTimeContextProtocol = Literal[
    "observed_snapshot",
    "source_reference",
    "self_consistent",
    "clamped_external",
]


@dataclass(frozen=True)
class SingleTimeView:
    """One observed perturbation snapshot and its matched control source."""

    view_id: str
    perturbation_id: str
    embedding_id: str
    source: FiniteMeasure
    target: FiniteMeasure
    is_control: bool = False
    guide_id: str | None = None
    target_gene: str | None = None
    guide_residual_id: str | None = None
    sample_id: str | None = None
    batch_id: str | None = None
    reference_scope: str = "global"
    reference_control_ids: tuple[str, ...] = ()


def _finite_measure(
    latent: np.ndarray,
    *,
    mass_mode: SingleTimeMassMode,
    mass_values: np.ndarray | None = None,
) -> FiniteMeasure:
    support = np.asarray(latent, dtype=np.float32)
    if support.ndim != 2 or support.shape[0] == 0:
        raise ValueError("A finite measure requires at least one latent cell and a 2D latent array.")
    if mass_mode in {"unit_mass", "unavailable"}:
        weights = np.full(support.shape[0], 1.0 / support.shape[0], dtype=np.float64)
    elif mass_mode == "cell_count":
        weights = np.ones(support.shape[0], dtype=np.float64)
    elif mass_mode == "obs_column":
        if mass_values is None:
            raise ValueError("mass_mode='obs_column' requires mass_values.")
        weights = np.asarray(mass_values, dtype=np.float64)
        if weights.shape != (support.shape[0],):
            raise ValueError("mass_values must have one entry per latent row.")
    else:  # pragma: no cover - kept for defensive runtime use
        raise ValueError(f"Unsupported mass_mode: {mass_mode!r}")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Single-time finite-measure weights must be positive and finite.")
    return FiniteMeasure(support=support, weights=weights, total_mass=float(weights.sum()))


def _pool_measures(measures: Sequence[FiniteMeasure]) -> FiniteMeasure:
    if not measures:
        raise ValueError("Cannot pool an empty measure sequence.")
    latent_dim = measures[0].latent_dim
    if any(mu.latent_dim != latent_dim for mu in measures):
        raise ValueError("Cannot pool finite measures with different latent dimensions.")
    support = np.concatenate([mu.support for mu in measures], axis=0)
    weights = np.concatenate([mu.weights for mu in measures], axis=0)
    return FiniteMeasure(support=support, weights=weights, total_mass=float(weights.sum()))


@dataclass
class ControlReferenceBuilder:
    """Build matched control reference measures for single-time snapshots."""

    control_ids: Sequence[str]
    scope: ControlReferenceScope = "auto"
    allow_control_self_reference: bool = False

    def __post_init__(self) -> None:
        if not self.control_ids:
            raise ValueError("Single-time reference construction requires at least one control id.")
        self.control_ids = tuple(str(pid) for pid in self.control_ids)
        if self.scope not in {"auto", "sample", "batch", "global"}:
            raise ValueError("scope must be 'auto', 'sample', 'batch', or 'global'.")

    def select_control_mask(
        self,
        obs: pd.DataFrame,
        *,
        perturbation_id: str,
        sample_id: str | None = None,
        batch_id: str | None = None,
    ) -> tuple[np.ndarray, str, tuple[str, ...]]:
        """Return a matched-control mask, the scope used, and control ids used."""
        controls = obs["perturbation_id"].astype(str).isin(self.control_ids)
        if str(perturbation_id) in self.control_ids and not self.allow_control_self_reference:
            controls = controls & ~obs["perturbation_id"].astype(str).eq(str(perturbation_id))

        candidates: list[tuple[str, np.ndarray]] = []
        if self.scope in {"auto", "sample"} and sample_id is not None and "sample_id" in obs:
            candidates.append(("sample", controls & obs["sample_id"].astype(str).eq(str(sample_id))))
        if self.scope in {"auto", "batch"} and batch_id is not None and "batch_id" in obs:
            candidates.append(("batch", controls & obs["batch_id"].astype(str).eq(str(batch_id))))
        if self.scope in {"auto", "global"}:
            candidates.append(("global", controls))

        if self.scope != "auto":
            requested = [item for item in candidates if item[0] == self.scope]
            if requested:
                candidates = requested
        for used_scope, mask in candidates:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.any():
                used = tuple(sorted(obs.loc[mask_array, "perturbation_id"].astype(str).unique().tolist()))
                return mask_array, used_scope, used

        raise ValueError(
            "No matched control cells available for "
            f"perturbation_id={perturbation_id!r}, sample_id={sample_id!r}, batch_id={batch_id!r}."
        )


@dataclass
class SingleTimeProblem:
    """A one-timepoint perturbation snapshot with matched control references.

    The adapted endpoint problem is an effect-path problem, not a physical
    time-course.  Downstream reports should use ``claim_level`` and
    ``abundance_claims_allowed`` to avoid longitudinal claims.
    """

    views: list[SingleTimeView]
    catalog: PerturbationCatalog
    context_protocol: SingleTimeContextProtocol = "observed_snapshot"
    mass_mode: SingleTimeMassMode = "unit_mass"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_protocol not in {
            "observed_snapshot",
            "source_reference",
            "self_consistent",
            "clamped_external",
        }:
            raise ValueError("Invalid single-time context_protocol.")
        if not self.views:
            raise ValueError("SingleTimeProblem requires at least one view.")
        view_ids = [view.view_id for view in self.views]
        if len(set(view_ids)) != len(view_ids):
            raise ValueError("SingleTimeProblem view_id values must be unique.")
        catalog_ids = set(self.catalog.perturbation_ids)
        for view in self.views:
            if view.embedding_id not in catalog_ids:
                raise KeyError(f"View {view.view_id!r} embedding_id {view.embedding_id!r} is not in catalog.")
            if view.source.latent_dim != view.target.latent_dim:
                raise ValueError(f"View {view.view_id!r} source/target latent dimensions differ.")

    @property
    def perturbation_ids(self) -> list[str]:
        return [pid for pid in self.catalog.perturbation_ids if any(v.embedding_id == pid for v in self.views)]

    @property
    def abundance_claims_allowed(self) -> bool:
        return self.abundance_claim_grade == "claim_grade"

    @property
    def abundance_claim_grade(self) -> Literal["none", "diagnostic", "claim_grade"]:
        if self.mass_mode in {"unit_mass", "unavailable"}:
            return "none"
        if self.mass_mode == "cell_count":
            return "diagnostic"
        return "claim_grade"

    @property
    def claim_level(self) -> str:
        if self.context_protocol == "self_consistent":
            return "single_time_effect_path_self_consistent_diagnostic"
        return "single_time_effect_path"

    def to_effect_endpoint_problem(
        self,
        *,
        view_level: SingleTimeEndpointViewLevel = "view",
    ) -> EndpointProblem:
        """Return a two-point control-reference -> observed effect problem."""
        if view_level not in {"embedding", "perturbation", "view"}:
            raise ValueError("view_level must be 'view', 'embedding', or 'perturbation'.")
        pooled_by_embedding = view_level in {"embedding", "perturbation"}
        metadata_view_level = "embedding" if pooled_by_embedding else "view"
        if not pooled_by_embedding:
            initial = {view.view_id: view.source for view in self.views}
            terminal = {view.view_id: view.target for view in self.views}
            pids = [view.view_id for view in self.views]
            embedding_ids = {view.view_id: view.embedding_id for view in self.views}
            views_by_key = {view.view_id: [view] for view in self.views}
        else:
            by_embedding: dict[str, list[SingleTimeView]] = {}
            for view in self.views:
                by_embedding.setdefault(view.embedding_id, []).append(view)
            pids = [pid for pid in self.catalog.perturbation_ids if pid in by_embedding]
            initial = {pid: _pool_measures([view.source for view in by_embedding[pid]]) for pid in pids}
            terminal = {pid: _pool_measures([view.target for view in by_embedding[pid]]) for pid in pids}
            embedding_ids = {pid: pid for pid in pids}
            views_by_key = by_embedding

        def _unique_or_none(values: list[str | None]) -> str | None:
            unique = sorted({str(value) for value in values if value is not None})
            if len(unique) == 1:
                return unique[0]
            return None

        measure_to_original = {
            key: _unique_or_none([view.perturbation_id for view in views_by_key[key]]) or key
            for key in pids
        }
        measure_to_target = {
            key: _unique_or_none([view.target_gene for view in views_by_key[key]])
            for key in pids
        }
        measure_to_guide = {
            key: _unique_or_none([view.guide_id for view in views_by_key[key]])
            for key in pids
        }
        guide_residual_ids = {
            key: _unique_or_none([view.guide_residual_id for view in views_by_key[key]])
            for key in pids
        }
        control_measure_keys = [
            key
            for key in pids
            if any(view.is_control for view in views_by_key[key])
        ]

        metadata = {
            **self.metadata,
            "problem_mode": "single_time",
            "effect_axis": True,
            "effect_axis_labels": ["control_reference", "observed_snapshot"],
            "effect_axis_is_physical_time": False,
            "claim_level": self.claim_level,
            "context_protocol": self.context_protocol,
            "mass_mode": self.mass_mode,
            "abundance_claims_allowed": self.abundance_claims_allowed,
            "abundance_claim_grade": self.abundance_claim_grade,
            "single_time_abundance_claim_grade": self.abundance_claim_grade,
            "view_level": metadata_view_level,
            "measure_keys": list(pids),
            "embedding_ids": embedding_ids,
            "measure_to_embedding": dict(embedding_ids),
            "measure_to_original_perturbation": measure_to_original,
            "measure_to_target_gene": measure_to_target,
            "measure_to_guide": measure_to_guide,
            "guide_residual_ids": guide_residual_ids,
            "control_measure_keys": control_measure_keys,
            "control_embedding_ids": list(self.catalog.control_ids),
            "target_ids": {
                key: measure_to_target[key] or measure_to_original[key]
                for key in pids
            },
            "axis_interpretation": "effect_axis",
            "counterfactual_source_semantics": "same_constructed_reference_source",
            "same_start_semantics": "constructed_reference_source",
        }
        return EndpointProblem(
            initial=initial,
            terminal=terminal,
            time_axis=TimeAxis(["control_reference", "observed_snapshot"], [0.0, 1.0]),
            perturbation_ids=pids,
            metadata=metadata,
        )

    def claim_report(self) -> dict[str, Any]:
        return {
            "problem_mode": "single_time",
            "effect_axis_is_physical_time": False,
            "claim_level": self.claim_level,
            "context_protocol": self.context_protocol,
            "mass_mode": self.mass_mode,
            "abundance_claims_allowed": self.abundance_claims_allowed,
            "abundance_claim_grade": self.abundance_claim_grade,
            "n_views": len(self.views),
            "n_perturbations": len(self.perturbation_ids),
            "control_ids": list(self.catalog.control_ids),
        }


def _normalise_single_time_obs(
    obs: pd.DataFrame,
    *,
    perturbation_col: str,
    control_col: str,
    sample_col: str | None,
    batch_col: str | None,
    guide_col: str | None,
    target_gene_col: str | None,
) -> pd.DataFrame:
    out = obs.copy()
    if perturbation_col not in out:
        raise KeyError(f"AnnData obs missing perturbation column {perturbation_col!r}.")
    if control_col not in out:
        raise KeyError(f"AnnData obs missing control column {control_col!r}.")
    out["perturbation_id"] = out[perturbation_col].astype(str)
    if guide_col is not None and guide_col in out:
        out["guide_id"] = out[guide_col].astype(str)
    if target_gene_col is not None and target_gene_col in out:
        out["target_gene"] = out[target_gene_col].astype(str)
    if pd.api.types.is_bool_dtype(out[control_col]):
        out["is_control"] = out[control_col].to_numpy(dtype=bool)
    else:
        normalized = out[control_col].astype(str).str.strip().str.lower()
        valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
        if not bool(valid.all()):
            raise ValueError("Single-time control column must be boolean-like.")
        out["is_control"] = normalized.isin({"true", "1", "yes"})
    if "cell_id" not in out:
        out["cell_id"] = out.index.astype(str)
    else:
        out["cell_id"] = out["cell_id"].astype(str)
    if sample_col is not None:
        if sample_col not in out:
            raise KeyError(f"AnnData obs missing sample column {sample_col!r}.")
        out["sample_id"] = out[sample_col].astype(str)
    if batch_col is not None:
        if batch_col not in out:
            raise KeyError(f"AnnData obs missing batch column {batch_col!r}.")
        out["batch_id"] = out[batch_col].astype(str)
    if out["cell_id"].duplicated().any():
        raise ValueError("Single-time AnnData requires unique cell_id values.")
    return out


def build_single_time_problem_from_anndata(
    adata_or_path: Any,
    *,
    latent_key: str = "X_pca",
    perturbation_col: str = "perturbation_id",
    guide_col: str | None = "guide_id",
    target_gene_col: str | None = "target_gene",
    embedding_level: SingleTimeEmbeddingLevel = "perturbation",
    control_col: str = "is_control",
    sample_col: str | None = "sample_id",
    batch_col: str | None = "batch_id",
    mass_mode: SingleTimeMassMode = "unit_mass",
    mass_value_col: str | None = None,
    reference_scope: ControlReferenceScope = "auto",
    context_protocol: SingleTimeContextProtocol = "observed_snapshot",
    min_cells: int = 1,
    control_split_seed: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> SingleTimeProblem:
    """Build a :class:`SingleTimeProblem` from an AnnData object or path."""
    import anndata as ad

    if mass_mode not in {"cell_count", "unit_mass", "obs_column", "unavailable"}:
        raise ValueError("Invalid single-time mass_mode.")
    if mass_mode == "obs_column" and not mass_value_col:
        raise ValueError("mass_value_col is required when mass_mode='obs_column'.")
    if embedding_level == "target_plus_guide_residual":
        raise NotImplementedError(
            "embedding_level='target_plus_guide_residual' requires a hierarchical "
            "target-plus-guide residual embedding module. Use 'target_gene' and "
            "treat guide metadata as concordance diagnostics for now."
        )
    if embedding_level not in {"perturbation", "guide", "target_gene"}:
        raise ValueError("embedding_level must be 'perturbation', 'guide', or 'target_gene'.")
    if isinstance(adata_or_path, (str, Path)):
        adata = ad.read_h5ad(adata_or_path)
    else:
        adata = adata_or_path
    if latent_key not in adata.obsm:
        raise KeyError(f"AnnData missing obsm[{latent_key!r}].")
    latent = np.asarray(adata.obsm[latent_key])
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs:
        raise ValueError(f"AnnData obsm[{latent_key!r}] must have shape [n_obs, latent_dim].")
    if not np.isfinite(latent).all():
        raise ValueError(f"AnnData obsm[{latent_key!r}] contains non-finite values.")

    obs = _normalise_single_time_obs(
        adata.obs,
        perturbation_col=perturbation_col,
        control_col=control_col,
        sample_col=sample_col if sample_col in adata.obs else None,
        batch_col=batch_col if batch_col in adata.obs else None,
        guide_col=guide_col if guide_col is not None and guide_col in adata.obs else None,
        target_gene_col=target_gene_col if target_gene_col is not None and target_gene_col in adata.obs else None,
    )
    if "sample_id" not in obs and "batch_id" not in obs:
        raise ValueError("Single-time AnnData requires obs['sample_id'] or obs['batch_id'].")
    if mass_value_col is not None and mass_value_col not in obs:
        raise KeyError(f"AnnData obs missing mass_value_col {mass_value_col!r}.")

    control_ids = sorted(obs.loc[obs["is_control"], "perturbation_id"].astype(str).unique().tolist())
    all_pids = sorted(obs["perturbation_id"].astype(str).unique().tolist())
    ref_builder = ControlReferenceBuilder(control_ids=control_ids, scope=reference_scope)

    views: list[SingleTimeView] = []
    control_embedding_ids: set[str] = set()
    for pid in all_pids:
        target_mask = obs["perturbation_id"].astype(str).eq(pid).to_numpy()
        if int(target_mask.sum()) < int(min_cells):
            continue
        sub = obs.loc[target_mask]
        group_col = "sample_id" if "sample_id" in obs else "batch_id" if "batch_id" in obs else None
        group_values = sorted(sub[group_col].astype(str).unique().tolist()) if group_col is not None else [None]
        for group_value in group_values:
            sample_mask = target_mask
            if group_col is not None and group_value is not None:
                sample_mask = sample_mask & obs[group_col].astype(str).eq(str(group_value)).to_numpy()
            sample_sub = obs.loc[sample_mask]
            if len(sample_sub) < int(min_cells):
                continue
            guide_values = sorted(sample_sub["guide_id"].astype(str).unique().tolist()) if "guide_id" in obs else []
            target_values_unique = (
                sorted(sample_sub["target_gene"].astype(str).unique().tolist())
                if "target_gene" in obs
                else []
            )
            guide_id = guide_values[0] if len(guide_values) == 1 else None
            target_gene = target_values_unique[0] if len(target_values_unique) == 1 else None
            if embedding_level == "guide":
                embedding_id = guide_id or pid
            elif embedding_level == "target_gene":
                embedding_id = target_gene or pid
            else:
                embedding_id = pid
            guide_residual_id = None
            if pid in control_ids:
                control_embedding_ids.add(embedding_id)
            sample_id = str(group_value) if group_col == "sample_id" and group_value is not None else None
            batch_id = str(group_value) if group_col == "batch_id" and group_value is not None else None
            if "batch_id" in obs and len(sample_sub) > 0:
                batch_values = sorted(sample_sub["batch_id"].astype(str).unique().tolist())
                batch_id = batch_values[0] if len(batch_values) == 1 else None
            try:
                source_mask, used_scope, used_controls = ref_builder.select_control_mask(
                    obs,
                    perturbation_id=pid,
                    sample_id=sample_id,
                    batch_id=batch_id,
                )
            except ValueError:
                if pid not in control_ids:
                    raise
                control_indices = np.flatnonzero(sample_mask)
                if len(control_indices) < 2:
                    continue
                split_key = f"{pid}|{group_col}|{group_value}|{control_split_seed}"
                split_offset = int(hashlib.sha256(split_key.encode("utf-8")).hexdigest()[:12], 16)
                rng = np.random.default_rng(int(control_split_seed) + split_offset)
                control_indices = rng.permutation(control_indices)
                split = max(1, len(control_indices) // 2)
                source_indices = control_indices[:split]
                target_indices = control_indices[split:]
                if len(target_indices) == 0:
                    continue
                source_mask = np.zeros(len(obs), dtype=bool)
                source_mask[source_indices] = True
                sample_mask = np.zeros(len(obs), dtype=bool)
                sample_mask[target_indices] = True
                used_scope = "control_cell_split"
                used_controls = (pid,)
                sample_sub = obs.loc[sample_mask]
            target_values = None if mass_value_col is None else obs.loc[sample_mask, mass_value_col].to_numpy()
            source_values = None if mass_value_col is None else obs.loc[source_mask, mass_value_col].to_numpy()
            target = _finite_measure(latent[sample_mask], mass_mode=mass_mode, mass_values=target_values)
            source = _finite_measure(latent[source_mask], mass_mode=mass_mode, mass_values=source_values)
            view_group = sample_id if sample_id is not None else batch_id
            view_id = pid if view_group is None else f"{view_group}::{pid}"
            views.append(
                SingleTimeView(
                    view_id=view_id,
                    perturbation_id=pid,
                    embedding_id=embedding_id,
                    source=source,
                    target=target,
                    is_control=pid in control_ids,
                    guide_id=guide_id,
                    target_gene=target_gene,
                    guide_residual_id=guide_residual_id,
                    sample_id=sample_id,
                    batch_id=batch_id,
                    reference_scope=used_scope,
                    reference_control_ids=used_controls,
                )
            )

    catalog_ids = sorted({view.embedding_id for view in views} | control_embedding_ids)
    catalog = PerturbationCatalog(catalog_ids, sorted(control_embedding_ids))
    return SingleTimeProblem(
        views=views,
        catalog=catalog,
        context_protocol=context_protocol,
        mass_mode=mass_mode,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "ControlReferenceBuilder",
    "ControlReferenceScope",
    "SingleTimeContextProtocol",
    "SingleTimeEndpointViewLevel",
    "SingleTimeEmbeddingLevel",
    "SingleTimeMassMode",
    "SingleTimeProblem",
    "SingleTimeView",
    "build_single_time_problem_from_anndata",
]
