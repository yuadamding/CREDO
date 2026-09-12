"""Post-publication keyed scoring; no endpoint observation enters prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..canonical import contract_id
from ..data.prepared_shards import PreparedEvaluationAccess
from ..errors import ContractError
from .aggregation import verify_runtime
from .artifacts import publish_bundle, read_spec, verify_bundle, write_json
from .baselines import _control_mean, _frequency, _means, _normalize
from .contracts import FAMILIES, BundleManifest, ForecastSpec


def prediction_barrier(prediction: Path) -> tuple[BundleManifest, ForecastSpec]:
    """Fully verify an already published prediction before granting endpoint use."""
    receipt = verify_bundle(prediction, stage="prediction")
    spec = ForecastSpec.model_validate(read_spec(prediction))
    if receipt.specification_sha256 != spec.identity():
        raise ContractError("Prediction specification identity mismatch.")
    if (
        receipt.facts["endpoint_outcomes_used"] is not False
        or receipt.facts["query_access_sha256"] != spec.query.identity()
        or receipt.facts["evaluation_view_sha256"] != spec.evaluation_view.sha256
        or receipt.facts["guide_catalog_sha256"] != spec.guide_catalog_sha256
        or receipt.facts["rna_order_sha256"] != spec.rna_order_sha256
    ):
        raise ContractError("Prediction information-set or biological-order mismatch.")
    verify_runtime(spec)
    return receipt, spec


def verify_evaluation_access(
    prediction: Path, access: PreparedEvaluationAccess, spec: ForecastSpec
) -> BundleManifest:
    receipt, saved = prediction_barrier(prediction)
    if saved.identity() != spec.identity() or access.prediction_seal_sha256 != receipt.identity():
        raise ContractError("Endpoint access is not bound to these published predictions.")
    for key in (
        "package_completion_sha256",
        "package_inventory_sha256",
        "amendment_sha256",
        "feature_order_sha256",
        "guide_catalog",
        "n_features",
        "task_id",
    ):
        if getattr(access, key) != getattr(spec.query, key):
            raise ContractError("Endpoint capability belongs to another package or task.")
    if access.parent_view_sha256 != spec.evaluation_view.sha256 or len(access.source_ids) != 1:
        raise ContractError("Endpoint capability must use exactly the frozen endpoint view.")
    if set(access.source_ids) & (set(spec.fitting.source_ids) | set(spec.query.source_ids)):
        raise ContractError("Endpoint source overlaps fitting/query observations.")
    return receipt


def _macro(frame: pd.DataFrame, metric: str) -> float | None:
    values = frame.groupby("target_id", sort=True)[metric].mean().dropna()
    return None if not len(values) else float(values.mean())


def evaluate_baselines(
    prediction: Path, access: PreparedEvaluationAccess, endpoint_source: Path, destination: Path
) -> None:
    """Score population means and complete-catalog captured abundance independently.

    The observed count-depth conditional score omits multinomial factorial terms.
    It is RNA-UMI-weighted cross entropy, NOT an unconditional future-count
    likelihood and NOT a per-cell distributional/variance qualification.
    """
    sealed, spec = prediction_barrier(prediction)
    verify_evaluation_access(prediction, access, spec)
    truth = verify_bundle(endpoint_source, stage="source_summary", specification=spec.identity())
    if (
        truth.parents["access"] != access.identity()
        or truth.facts["source_id"] != access.source_ids[0]
        or not truth.facts["includes_counts"]
        or not truth.facts["complete_denominator"]
    ):
        raise ContractError("Evaluator summary is not the bound complete endpoint.")
    catalog = pd.read_parquet(prediction / "guide_catalog.parquet")
    if contract_id(catalog.to_dict("records")) != spec.guide_catalog_sha256:
        raise ContractError("Prediction guide labels changed.")
    coverage = pd.read_parquet(prediction / "coverage.parquet")
    g, f = len(catalog), len(spec.rna_positions)
    expected_keys = {(family, index) for family in FAMILIES for index in range(g)}
    if (
        len(coverage) != len(expected_keys)
        or set(zip(coverage.family, coverage.guide_index, strict=True)) != expected_keys
    ):
        raise ContractError("Missing, duplicate or unknown prediction coverage keys.")
    for column in ("guide_id", "target_id"):
        if not np.array_equal(coverage[column], catalog[column].iloc[coverage.guide_index]):
            raise ContractError("Coverage biological labels disagree with the catalog.")
    epsilon = spec.rules.expression_pseudocount
    controls = np.flatnonzero(catalog.is_control.to_numpy())
    keyed: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    with (
        h5py.File(prediction / "predictions.h5", "r") as predicted,
        h5py.File(endpoint_source / "statistics.h5", "r") as observed,
    ):
        if (
            observed.attrs["rna_order_sha256"] != spec.rna_order_sha256
            or observed.attrs["guide_catalog_sha256"] != spec.guide_catalog_sha256
        ):
            raise ContractError("Endpoint numerical biological order mismatch.")
        n_end, n_rna = observed["n_cells"][:], observed["n_rna_cells"][:]
        if int(n_end.sum()) != sum(s.selected_rows for s in access.shards):
            raise ContractError("Endpoint summary changed the declared population denominator.")
        endpoint_frequency = _frequency(n_end)
        source_frequency = predicted["source_frequency"][:]
        observed_control = _control_mean(observed, controls)
        for family in FAMILIES:
            group = predicted[family]
            if group["mean_composition"].shape != (g, f):
                raise ContractError("Prediction dimensions disagree with biological identities.")
            available = group["expression_available"][:]
            cov = coverage[coverage.family.eq(family)].sort_values("guide_index")
            if not np.array_equal(cov.expression_available, available):
                raise ContractError("Prediction support flags disagree with coverage records.")
            mass = group["abundance_frequency"][:]
            reference = group["reference_composition"][:]
            if (
                not np.isfinite(mass).all()
                or np.any(mass <= 0)
                or not np.isclose(mass.sum(), 1.0, atol=1e-12)
            ):
                raise ContractError("Predicted abundance lost its full-catalog denominator.")
            if (
                not np.isfinite(reference).all()
                or np.any(reference <= 0)
                or not np.isclose(reference.sum(), 1.0)
            ):
                raise ContractError("Invalid independent predicted control reference.")
            for index in range(g):
                mass_rows.append(
                    dict(
                        family=family,
                        guide_index=index,
                        guide_id=catalog.guide_id.iloc[index],
                        target_id=catalog.target_id.iloc[index],
                        is_control=bool(catalog.is_control.iloc[index]),
                        source_cells=int(predicted["source_n_cells"][index]),
                        endpoint_cells=int(n_end[index]),
                        source_support_stratum=cov.source_support_stratum.iloc[index],
                        prediction_frequency=float(mass[index]),
                        endpoint_frequency=float(endpoint_frequency[index]),
                        source_frequency=float(source_frequency[index]),
                        log_frequency_squared_error=float(
                            (np.log(mass[index]) - np.log(endpoint_frequency[index])) ** 2
                        ),
                        observed_interval_log_effect=float(
                            np.log(endpoint_frequency[index]) - np.log(source_frequency[index])
                        ),
                        predicted_interval_log_effect=float(
                            np.log(mass[index]) - np.log(source_frequency[index])
                        ),
                        conditional_nll_without_constant=float(-n_end[index] * np.log(mass[index])),
                    )
                )
            for start in range(0, g, 32):
                rows = slice(start, min(start + 32, g))
                means = group["mean_composition"][rows].astype(np.float64)
                supported = available[rows]
                if (
                    not np.isfinite(means).all()
                    or np.any(means < 0)
                    or not np.allclose(means[supported].sum(1), 1, atol=2e-6)
                    or np.any(means[~supported] != 0)
                ):
                    raise ContractError("Invalid supported or abstained expression predictions.")
                observed_mean = _means(observed, rows)
                # Renormalize float32 serialization round-off, not biological subsets.
                means[supported] /= means[supported].sum(1, keepdims=True)
                actual = _normalize(observed_mean, epsilon)
                observed_effect = np.log(observed_mean + epsilon) - np.log(
                    observed_control + epsilon
                )
                predicted_effect = np.log(means + epsilon) - np.log(reference + epsilon)
                raw_counts = observed["count_sum"][rows]
                for local, index in enumerate(range(rows.start, rows.stop)):
                    valid = bool(supported[local] and n_rna[index] > 0)
                    informative = (
                        np.abs(observed_effect[local]) >= spec.rules.informative_log_effect
                    )
                    umi = int(raw_counts[local].sum())
                    ce = (
                        float(
                            -np.dot(
                                raw_counts[local],
                                np.log(np.maximum(means[local], np.finfo(float).tiny)),
                            )
                        )
                        if valid
                        else None
                    )
                    keyed.append(
                        dict(
                            family=family,
                            guide_index=index,
                            guide_id=catalog.guide_id.iloc[index],
                            target_id=catalog.target_id.iloc[index],
                            is_control=bool(catalog.is_control.iloc[index]),
                            source_support_stratum=cov.source_support_stratum.iloc[index],
                            prediction_status=cov.expression_status.iloc[index],
                            endpoint_cells=int(n_end[index]),
                            endpoint_RNA_cells=int(n_rna[index]),
                            endpoint_RNA_UMIs=umi,
                            endpoint_status="endpoint_absent"
                            if n_end[index] == 0
                            else "zero_RNA_endpoint"
                            if n_rna[index] == 0
                            else "observed",
                            scored=valid,
                            composition_mse=float(np.mean((means[local] - actual[local]) ** 2))
                            if valid
                            else None,
                            hellinger_squared=float(
                                0.5 * np.sum((np.sqrt(means[local]) - np.sqrt(actual[local])) ** 2)
                            )
                            if valid
                            else None,
                            log1p_10000_composition_mse=float(
                                np.mean(
                                    (
                                        np.log1p(10000 * means[local])
                                        - np.log1p(10000 * actual[local])
                                    )
                                    ** 2
                                )
                            )
                            if valid
                            else None,
                            endpoint_control_log_effect_mse=float(
                                np.mean((predicted_effect[local] - observed_effect[local]) ** 2)
                            )
                            if valid
                            else None,
                            informative_effect_genes=int(informative.sum()),
                            effect_sign_accuracy=float(
                                np.mean(
                                    np.sign(predicted_effect[local][informative])
                                    == np.sign(observed_effect[local][informative])
                                )
                            )
                            if valid and informative.any()
                            else None,
                            conditional_nll_without_constant=ce,
                            conditional_cross_entropy_per_RNA_UMI=ce / umi
                            if valid and umi and ce is not None
                            else None,
                        )
                    )
    expression = pd.DataFrame(keyed)
    mass_frame = pd.DataFrame(mass_rows)
    common = expression.groupby("guide_index").scored.all()
    expression["common_family_support"] = expression.guide_index.map(common)
    metrics = [
        "composition_mse",
        "hellinger_squared",
        "log1p_10000_composition_mse",
        "endpoint_control_log_effect_mse",
        "effect_sign_accuracy",
    ]
    summary: dict[str, Any] = {}
    for family in FAMILIES:
        all_rows = expression[expression.family.eq(family)]
        mass = mass_frame[mass_frame.family.eq(family)]
        entries: dict[str, Any] = {}
        for label, subset in (
            ("targeting", all_rows[~all_rows.is_control]),
            ("controls", all_rows[all_rows.is_control]),
            (
                "targeting_common_support",
                all_rows[~all_rows.is_control & all_rows.common_family_support],
            ),
        ):
            scored = subset[subset.scored]
            entries[label] = dict(
                designed_guides=len(subset),
                scored_guides=len(scored),
                scored_targets=int(scored.target_id.nunique()),
                macro_target={metric: _macro(scored, metric) for metric in metrics},
            )
            for column in ("prediction_status", "endpoint_status"):
                entries[label][column] = {
                    str(k): int(v) for k, v in subset[column].value_counts().items()
                }
        scored = all_rows[all_rows.scored]
        total_umis = int(scored.endpoint_RNA_UMIs.sum())
        entries["expression_conditional_cross_entropy_per_RNA_UMI"] = (
            float(scored.conditional_nll_without_constant.sum() / total_umis)
            if total_umis
            else None
        )
        entries["abundance"] = dict(
            guides=g,
            observed_cells=int(mass.endpoint_cells.sum()),
            denominator="complete_bound_guide_catalog",
            log_frequency_RMSE=float(np.sqrt(mass.log_frequency_squared_error.mean())),
            interval_log_effect_RMSE=float(np.sqrt(mass.log_frequency_squared_error.mean())),
            conditional_cross_entropy_per_cell=float(
                mass.conditional_nll_without_constant.sum() / mass.endpoint_cells.sum()
            ),
            jeffreys_KL_truth_to_prediction=float(
                np.sum(
                    mass.endpoint_frequency
                    * np.log(mass.endpoint_frequency / mass.prediction_frequency)
                )
            ),
            by_source_support={
                str(key): dict(
                    guides=len(group),
                    log_frequency_RMSE=float(np.sqrt(group.log_frequency_squared_error.mean())),
                )
                for key, group in mass.groupby("source_support_stratum")
            },
        )
        summary[family] = entries

    def write(path: Path) -> dict[str, Any]:
        expression.to_parquet(path / "expression_by_guide.parquet", index=False)
        mass_frame.to_parquet(path / "abundance_by_guide.parquet", index=False)
        write_json(path / "metrics.json", summary)
        write_json(path / "specification.json", spec.model_dump(mode="json"))
        return dict(
            guides=g,
            RNA_features=f,
            endpoint_cells=int(n_end.sum()),
            query_donor=spec.query_donor,
            prediction_seal_sha256=sealed.identity(),
            complete_abundance_denominator=True,
            observed_counts_accessed_only_after_prediction_publication=True,
            composition_scope="equal_cell_population_means_only",
            effect_definition="endpoint_log_composition_relative_to_independent_control_reference",
            likelihood_definition="conditional_cross_entropy_factorial_constant_omitted",
            abundance_scope="relative_captured_cell_abundance_not_absolute_growth",
            scientific_promotion=False,
            storage_isolation_qualified=False,
            prediction_support_does_not_require_endpoint_cell_correspondence=True,
        )

    publish_bundle(
        destination,
        stage="evaluation",
        specification=spec.identity(),
        parents={
            "prediction": sealed.identity(),
            "endpoint_summary": truth.identity(),
            "evaluation_access": access.identity(),
        },
        writer=write,
    )
    if verify_bundle(prediction, stage="prediction").identity() != sealed.identity():
        raise ContractError("Evaluation modified published predictions.")
