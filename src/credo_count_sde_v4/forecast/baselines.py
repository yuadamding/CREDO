"""Training-only fits and source-only prediction of RNA means and captured mass."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..canonical import contract_id
from ..data.prepared_shards import PreparedShardReader
from ..errors import ContractError
from .aggregation import verify_runtime
from .artifacts import publish_bundle, read_spec, verify_bundle, write_json
from .contracts import FAMILIES, ForecastSpec


def _frequency(n: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return (n + 0.5) / (n.sum() + 0.5 * len(n))


def _normalize(values: np.ndarray[Any, Any], epsilon: float) -> np.ndarray[Any, Any]:
    values = np.asarray(values, dtype=np.float64) + epsilon
    return values / values.sum(axis=-1, keepdims=True)


def _response(
    source: np.ndarray[Any, Any], log_change: np.ndarray[Any, Any], epsilon: float
) -> np.ndarray[Any, Any]:
    logits = np.log(source + epsilon) + log_change
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


def numeric_h5_identity(handle: h5py.File) -> str:
    """Hash numerical predictions/parameters independently of packaging provenance."""
    digest = hashlib.sha256()
    names: list[str] = []
    handle.visititems(
        lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None
    )
    for name in sorted(names):
        array = handle[name]
        digest.update(contract_id([name, list(array.shape), str(array.dtype)]).encode())
        if array.ndim == 0:
            digest.update(np.asarray(array[()]).tobytes())
        else:
            for start in range(0, len(array), 32):
                digest.update(np.asarray(array[start : start + 32]).tobytes(order="C"))
    return digest.hexdigest()


def _open_sources(
    stack: ExitStack, sources: dict[str, Path], spec: ForecastSpec, allowed: tuple[str, ...]
) -> tuple[dict[str, h5py.File], dict[str, str]]:
    if set(sources) != set(allowed):
        raise ContractError("Source summaries must exactly match the authorized role.")
    result, parents = {}, {}
    for source, path in sources.items():
        receipt = verify_bundle(path, stage="source_summary", specification=spec.identity())
        if receipt.facts["source_id"] != source or not receipt.facts["complete_denominator"]:
            raise ContractError("Source summary identity or denominator mismatch.")
        expected_access = spec.fitting if allowed == spec.fitting.source_ids else spec.query
        if receipt.parents["access"] != expected_access.identity():
            raise ContractError("Source summary was produced under another access role.")
        handle = stack.enter_context(h5py.File(path / "statistics.h5", "r"))
        if (
            handle.attrs["guide_catalog_sha256"] != spec.guide_catalog_sha256
            or handle.attrs["rna_order_sha256"] != spec.rna_order_sha256
        ):
            raise ContractError("Source summary biological order mismatch.")
        result[source], parents[source] = handle, receipt.identity()
    return result, parents


def _means(handle: h5py.File, rows: Any) -> np.ndarray[Any, Any]:
    return handle["composition_sum"][rows] / np.maximum(handle["n_rna_cells"][rows], 1)[:, None]


def _control_mean(handle: h5py.File, controls: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result = np.zeros(handle["composition_sum"].shape[1])
    total = 0
    for start in range(0, len(controls), 32):
        rows = controls[start : start + 32]
        result += handle["composition_sum"][rows].sum(0)
        total += int(handle["n_rna_cells"][rows].sum())
    if total == 0:
        raise ContractError("Declared control reference has no positive-RNA observations.")
    return result / total


def fit_baselines(
    spec: ForecastSpec, catalog: pd.DataFrame, fitting_sources: dict[str, Path], destination: Path
) -> None:
    """No query or endpoint outcome argument is accepted by the numerical fit."""
    verify_runtime(spec)
    if contract_id(catalog.to_dict("records")) != spec.guide_catalog_sha256:
        raise ContractError("Fitting guide catalog identity mismatch.")
    g, f = len(catalog), len(spec.rna_positions)
    targets = sorted(set(catalog.target_id))
    target_lookup = {target: index for index, target in enumerate(targets)}
    target_index = np.asarray([target_lookup[t] for t in catalog.target_id], dtype=np.int64)
    controls = np.flatnonzero(catalog.is_control.to_numpy())
    lookup = {(r.donor_id, r.condition_role): r.source_id for r in spec.source_roles}
    epsilon = spec.rules.expression_pseudocount
    with ExitStack() as stack:
        sources, parents = _open_sources(stack, fitting_sources, spec, spec.fitting.source_ids)
        for source, handle in sources.items():
            if int(handle["n_cells"][:].sum()) != sum(
                s.selected_rows for s in spec.fitting.shards if s.source_id == source
            ):
                raise ContractError("Fitting summary omitted authorized cells.")
        pairs = [
            (sources[lookup[d, "source"]], sources[lookup[d, "destination"]])
            for d in spec.fitting_donors
        ]
        control_source = np.mean([_control_mean(a, controls) for a, _ in pairs], axis=0)
        control_endpoint = np.mean([_control_mean(b, controls) for _, b in pairs], axis=0)
        control_response = np.mean(
            [
                np.log(_control_mean(b, controls) + epsilon)
                - np.log(_control_mean(a, controls) + epsilon)
                for a, b in pairs
            ],
            axis=0,
        )
        mass_endpoint = np.mean([_frequency(b["n_cells"][:]) for _, b in pairs], axis=0)
        mass_change = np.mean(
            [
                np.log(_frequency(b["n_cells"][:])) - np.log(_frequency(a["n_cells"][:]))
                for a, b in pairs
            ],
            axis=0,
        )
        mass_support = np.sum(
            [np.minimum(a["n_cells"][:], b["n_cells"][:]) for a, b in pairs], axis=0
        )
        target_sizes = np.bincount(target_index, minlength=len(targets))
        target_mass_endpoint = (
            np.bincount(target_index, weights=mass_endpoint, minlength=len(targets))[target_index]
            / target_sizes[target_index]
        )
        target_mass_change = (
            np.bincount(target_index, weights=mass_change, minlength=len(targets))[target_index]
            / target_sizes[target_index]
        )
        mass_weight = mass_support / (mass_support + spec.rules.shrinkage_cells)
        hierarchical_mass_change = (
            mass_weight * mass_change + (1 - mass_weight) * target_mass_change
        )

        def write(path: Path) -> dict[str, Any]:
            write_json(path / "specification.json", spec.model_dump(mode="json"))
            catalog.to_parquet(path / "guide_catalog.parquet", index=False)
            target_endpoint = np.zeros((len(targets), f))
            target_response = np.zeros((len(targets), f))
            target_endpoint_n = np.zeros(len(targets), dtype=np.int64)
            target_response_n = np.zeros(len(targets), dtype=np.int64)
            with h5py.File(path / "parameters.h5", "x") as out:
                guide_endpoint = out.create_dataset(
                    "guide_endpoint", shape=(g, f), dtype="float32", chunks=(min(32, g), f)
                )
                guide_response = out.create_dataset(
                    "guide_response_log", shape=(g, f), dtype="float32", chunks=(min(32, g), f)
                )
                endpoint_donors = np.zeros(g, dtype=np.int64)
                pair_support = np.zeros(g, dtype=np.int64)
                for start in range(0, g, 32):
                    rows = slice(start, min(start + 32, g))
                    count = rows.stop - rows.start
                    end_sum, response_sum = np.zeros((count, f)), np.zeros((count, f))
                    n_end, n_pair, support = (
                        np.zeros(count),
                        np.zeros(count),
                        np.zeros(count, dtype=np.int64),
                    )
                    for a, b in pairs:
                        source_mean, endpoint_mean = _means(a, rows), _means(b, rows)
                        source_n, endpoint_n = a["n_rna_cells"][rows], b["n_rna_cells"][rows]
                        available = endpoint_n > 0
                        paired = available & (source_n > 0)
                        end_sum += endpoint_mean * available[:, None]
                        response_sum += (
                            np.log(endpoint_mean + epsilon) - np.log(source_mean + epsilon)
                        ) * paired[:, None]
                        n_end += available
                        n_pair += paired
                        support += np.minimum(source_n, endpoint_n)
                    end_mean = end_sum / np.maximum(n_end, 1)[:, None]
                    response_mean = response_sum / np.maximum(n_pair, 1)[:, None]
                    guide_endpoint[rows], guide_response[rows] = end_mean, response_mean
                    endpoint_donors[rows], pair_support[rows] = n_end.astype(np.int64), support
                    for local, t in enumerate(target_index[rows]):
                        if n_end[local] > 0:
                            target_endpoint[t] += end_mean[local]
                            target_endpoint_n[t] += 1
                        if n_pair[local] > 0:
                            target_response[t] += response_mean[local]
                            target_response_n[t] += 1
                target_endpoint /= np.maximum(target_endpoint_n, 1)[:, None]
                target_response /= np.maximum(target_response_n, 1)[:, None]
                out.create_dataset("target_endpoint", data=target_endpoint.astype(np.float32))
                out.create_dataset("target_response_log", data=target_response.astype(np.float32))
                out.create_dataset("target_endpoint_guides", data=target_endpoint_n)
                out.create_dataset("target_response_guides", data=target_response_n)
                out.create_dataset("guide_endpoint_donors", data=endpoint_donors)
                out.create_dataset("guide_pair_cells", data=pair_support)
                out.create_dataset("guide_target_index", data=target_index)
                out.create_dataset("control_source", data=control_source)
                out.create_dataset("control_endpoint", data=control_endpoint)
                out.create_dataset("control_response_log", data=control_response)
                out.create_dataset("mass_guide_endpoint", data=mass_endpoint)
                out.create_dataset("mass_target_endpoint", data=target_mass_endpoint)
                out.create_dataset("mass_hierarchical_response_log", data=hierarchical_mass_change)
                numerical_sha256 = numeric_h5_identity(out)
            return dict(
                numerical_sha256=numerical_sha256,
                fitting_sources=list(spec.fitting.source_ids),
                fitting_cells=sum(s.selected_rows for s in spec.fitting.shards),
                query_or_endpoint_outcomes_used=False,
                guide_catalog_sha256=spec.guide_catalog_sha256,
                rna_order_sha256=spec.rna_order_sha256,
                families=list(FAMILIES),
            )

        publish_bundle(
            destination,
            stage="fitted_baselines",
            specification=spec.identity(),
            parents=parents,
            writer=write,
        )


def predict_baselines(fitted: Path, query_sources: dict[str, Path], destination: Path) -> None:
    """Only a fitted artifact and source-query summaries enter this interface."""
    fit = verify_bundle(fitted, stage="fitted_baselines")
    spec = ForecastSpec.model_validate(read_spec(fitted))
    verify_runtime(spec)
    if fit.specification_sha256 != spec.identity():
        raise ContractError("Fitted model specification changed.")
    catalog = pd.read_parquet(fitted / "guide_catalog.parquet")
    if contract_id(catalog.to_dict("records")) != spec.guide_catalog_sha256:
        raise ContractError("Prediction guide catalog mismatch.")
    controls = np.flatnonzero(catalog.is_control.to_numpy())
    g, f = len(catalog), len(spec.rna_positions)
    epsilon = spec.rules.expression_pseudocount
    with ExitStack() as stack:
        queries, parents = _open_sources(stack, query_sources, spec, spec.query.source_ids)
        query = queries[spec.query.source_ids[0]]
        parameters = stack.enter_context(h5py.File(fitted / "parameters.h5", "r"))
        parents["fitted_baselines"] = fit.identity()
        if int(query["n_cells"][:].sum()) != sum(s.selected_rows for s in spec.query.shards):
            raise ContractError("Query summary omitted authorized source cells.")
        query_control = _control_mean(query, controls)
        n_source, n_rna = query["n_cells"][:], query["n_rna_cells"][:]
        source_frequency = _frequency(n_source)
        predicted_mass = {
            "source_persistence": source_frequency,
            "fitting_control_response": source_frequency,
            "guide_endpoint_transfer": parameters["mass_guide_endpoint"][:],
            "target_endpoint_transfer": parameters["mass_target_endpoint"][:],
            "hierarchical_source_response": _response(
                source_frequency, parameters["mass_hierarchical_response_log"][:], 0.0
            ),
        }
        control_change = parameters["control_response_log"][:]
        controls_predicted = {
            "source_persistence": _normalize(query_control, epsilon),
            "fitting_control_response": _response(query_control, control_change, epsilon),
            "guide_endpoint_transfer": _normalize(parameters["control_endpoint"][:], epsilon),
            "target_endpoint_transfer": _normalize(parameters["control_endpoint"][:], epsilon),
            "hierarchical_source_response": _response(query_control, control_change, epsilon),
        }

        def write(path: Path) -> dict[str, Any]:
            write_json(path / "specification.json", spec.model_dump(mode="json"))
            catalog.to_parquet(path / "guide_catalog.parquet", index=False)
            coverage: list[dict[str, Any]] = []
            with h5py.File(path / "predictions.h5", "x") as out:
                out.create_dataset("source_n_cells", data=n_source)
                out.create_dataset("source_n_rna_cells", data=n_rna)
                out.create_dataset("source_frequency", data=source_frequency)
                datasets = {}
                for family in FAMILIES:
                    group = out.create_group(family)
                    group.create_dataset("abundance_frequency", data=predicted_mass[family])
                    group.create_dataset("reference_composition", data=controls_predicted[family])
                    datasets[family] = group.create_dataset(
                        "mean_composition", shape=(g, f), dtype="float32", chunks=(min(32, g), f)
                    )
                    group.create_dataset("expression_available", shape=(g,), dtype="bool")
                target_index = parameters["guide_target_index"][:]
                for start in range(0, g, 32):
                    rows = slice(start, min(start + 32, g))
                    targets = target_index[rows]
                    # h5py requires unique sorted keys; restore duplicates explicitly.
                    unique, inverse = np.unique(targets, return_inverse=True)
                    source_mean = _means(query, rows)
                    endpoint = parameters["guide_endpoint"][rows]
                    target_endpoint = parameters["target_endpoint"][unique][inverse]
                    guide_change = parameters["guide_response_log"][rows]
                    target_change = parameters["target_response_log"][unique][inverse]
                    target_pair = parameters["target_response_guides"][unique][inverse]
                    target_change[target_pair == 0] = control_change
                    support = parameters["guide_pair_cells"][rows]
                    weight = support / (support + spec.rules.shrinkage_cells)
                    change = weight[:, None] * guide_change + (1 - weight[:, None]) * target_change
                    control_rows = catalog.is_control.to_numpy()[rows]
                    change[control_rows] = control_change
                    means = {
                        "source_persistence": _normalize(source_mean, epsilon),
                        "fitting_control_response": _response(source_mean, control_change, epsilon),
                        "guide_endpoint_transfer": _normalize(endpoint, epsilon),
                        "target_endpoint_transfer": _normalize(target_endpoint, epsilon),
                        "hierarchical_source_response": _response(source_mean, change, epsilon),
                    }
                    for family in FAMILIES:
                        valid = n_rna[rows] > 0
                        if family == "guide_endpoint_transfer":
                            valid &= parameters["guide_endpoint_donors"][rows] > 0
                        if family == "target_endpoint_transfer":
                            valid &= parameters["target_endpoint_guides"][unique][inverse] > 0
                        values = means[family]
                        values[~valid] = 0.0
                        datasets[family][rows] = values
                        out[family]["expression_available"][rows] = valid
                        for offset, index in enumerate(range(rows.start, rows.stop)):
                            coverage.append(
                                dict(
                                    guide_index=index,
                                    guide_id=catalog.guide_id.iloc[index],
                                    target_id=catalog.target_id.iloc[index],
                                    family=family,
                                    source_cells=int(n_source[index]),
                                    source_RNA_cells=int(n_rna[index]),
                                    expression_available=bool(valid[offset]),
                                    expression_status="source_absent"
                                    if n_source[index] == 0
                                    else "zero_RNA_source"
                                    if n_rna[index] == 0
                                    else "fitting_endpoint_unavailable"
                                    if not valid[offset]
                                    else "predicted",
                                    source_support_stratum="0"
                                    if n_source[index] == 0
                                    else "1-19"
                                    if n_source[index] < 20
                                    else "20-99"
                                    if n_source[index] < 100
                                    else ">=100",
                                    abundance_status="mass_only_source_prior"
                                    if n_source[index] == 0
                                    else "source_conditioned",
                                    hierarchical_response_support="guide_target_shrinkage"
                                    if support[offset] > 0
                                    else "target_fallback"
                                    if target_pair[offset] > 0
                                    else "fitting_control_fallback",
                                )
                            )
                numerical_sha256 = numeric_h5_identity(out)
            pd.DataFrame(coverage).to_parquet(path / "coverage.parquet", index=False)
            return dict(
                numerical_sha256=numerical_sha256,
                query_sources=list(spec.query.source_ids),
                query_cells=int(n_source.sum()),
                guide_catalog_sha256=spec.guide_catalog_sha256,
                rna_order_sha256=spec.rna_order_sha256,
                query_access_sha256=spec.query.identity(),
                evaluation_view_sha256=spec.evaluation_view.sha256,
                endpoint_outcomes_used=False,
                guides=g,
                RNA_features=f,
                complete_abundance_denominator=True,
                fitted_parameters_numerical_sha256=fit.facts["numerical_sha256"],
            )

        publish_bundle(
            destination,
            stage="prediction",
            specification=spec.identity(),
            parents=parents,
            writer=write,
        )


def resolve_catalog(package_root: Path, spec: ForecastSpec) -> pd.DataFrame:
    return PreparedShardReader(package_root, spec.fitting).guide_catalog()
