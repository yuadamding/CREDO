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
from ..runtime_identity import environment_identity, implementation_tree_hash
from .aggregation import verify_runtime
from .artifacts import publish_bundle, read_spec, verify_bundle, write_json
from .baselines import (
    _control_mean,
    _frequency,
    _means,
    _normalize,
    abundance_prediction_basis,
)
from .contracts import (
    FAMILIES,
    BundleManifest,
    EvaluationCorrection,
    ForecastSpec,
    process_environment,
)


def _verified_prediction(prediction: Path) -> tuple[BundleManifest, ForecastSpec]:
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
    return receipt, spec


def prediction_barrier(prediction: Path) -> tuple[BundleManifest, ForecastSpec]:
    """Fully verify publication AND original runtime before granting endpoint use."""
    receipt, spec = _verified_prediction(prediction)
    verify_runtime(spec)
    return receipt, spec


def verify_evaluation_access(
    prediction: Path, access: PreparedEvaluationAccess, spec: ForecastSpec
) -> BundleManifest:
    receipt, saved = prediction_barrier(prediction)
    if saved.identity() != spec.identity():
        raise ContractError("Endpoint access is not bound to these published predictions.")
    _verify_access_binding(receipt, access, spec)
    return receipt


def _verify_access_binding(
    receipt: BundleManifest, access: PreparedEvaluationAccess, spec: ForecastSpec
) -> None:
    if access.prediction_seal_sha256 != receipt.identity():
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


def _macro(frame: pd.DataFrame, metric: str) -> float | None:
    values = frame.groupby("target_id", sort=True)[metric].mean().dropna()
    return None if not len(values) else float(values.mean())


def _conditional_score(frame: pd.DataFrame) -> dict[str, Any]:
    scored = frame[frame.scored]
    umis = int(scored.endpoint_RNA_UMIs.sum())
    return dict(
        scored_guides=len(scored),
        endpoint_RNA_UMIs=umis,
        conditional_cross_entropy_per_RNA_UMI=(
            float(scored.conditional_nll_without_constant.sum() / umis) if umis else None
        ),
    )


def rescore_baselines(
    prediction: Path,
    access: PreparedEvaluationAccess,
    endpoint_source: Path,
    previous_evaluation: Path,
    destination: Path,
    correction: EvaluationCorrection,
) -> None:
    """Versioned score-only correction; cannot mint access, aggregate, fit or predict.

    Both original runtime identity and new evaluator identity are explicit. The
    numerical environment must still equal the frozen original environment.
    Ordinary fitting/prediction/evaluation runtime gates are not relaxed.
    """
    sealed, spec = _verified_prediction(prediction)
    _verify_access_binding(sealed, access, spec)
    previous = verify_bundle(previous_evaluation, stage="evaluation", specification=spec.identity())
    expected = dict(
        prediction=sealed.identity(),
        endpoint_summary=correction.endpoint_summary_sha256,
        evaluation_access=access.identity(),
    )
    if (
        correction.original_specification_sha256 != spec.identity()
        or correction.original_implementation_sha256 != spec.implementation_sha256
        or correction.evaluator_implementation_sha256 != implementation_tree_hash()
        or correction.prediction_seal_sha256 != sealed.identity()
        or correction.evaluation_access_sha256 != access.identity()
        or correction.previous_evaluation_sha256 != previous.identity()
        or previous.parents != expected
        or previous.facts.get("scoring_version", 1) != 1
        or environment_identity() != spec.environment
        or process_environment() != spec.process_environment
    ):
        raise ContractError("Evaluation correction identity, lineage or runtime mismatch.")
    _evaluate_baselines(prediction, access, endpoint_source, destination, sealed, spec, correction)
    if verify_bundle(previous_evaluation).identity() != previous.identity():
        raise ContractError("Correction modified the original evaluation.")


def evaluate_baselines(
    prediction: Path, access: PreparedEvaluationAccess, endpoint_source: Path, destination: Path
) -> None:
    """Score population means and complete-catalog captured abundance independently.

    The observed count-depth conditional score omits multinomial factorial terms.
    It is RNA-UMI-weighted cross entropy, NOT an unconditional future-count
    likelihood and NOT a per-cell distributional/variance qualification.
    """
    sealed, spec = prediction_barrier(prediction)
    _verify_access_binding(sealed, access, spec)
    _evaluate_baselines(prediction, access, endpoint_source, destination, sealed, spec)


def _evaluate_baselines(
    prediction: Path,
    access: PreparedEvaluationAccess,
    endpoint_source: Path,
    destination: Path,
    sealed: BundleManifest,
    spec: ForecastSpec,
    correction: EvaluationCorrection | None = None,
) -> None:
    truth = verify_bundle(endpoint_source, stage="source_summary", specification=spec.identity())
    if correction is not None and truth.identity() != correction.endpoint_summary_sha256:
        raise ContractError("Correction endpoint summary identity mismatch.")
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
        observed_control = _normalize(_control_mean(observed, controls), epsilon)
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
            reference /= reference.sum()
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
                        query_source_support="present"
                        if predicted["source_n_cells"][index] > 0
                        else "absent",
                        abundance_prediction_basis=abundance_prediction_basis(family),
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
                observed_effect = np.log(actual) - np.log(observed_control)
                # Published predictions/references already contain their declared
                # smoothing. Handle serialization underflow, not a second pseudocount.
                predicted_effect = np.log(np.maximum(means, np.finfo(float).tiny)) - np.log(
                    reference
                )
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
                conditional_count_score=_conditional_score(scored),
            )
            for column in ("prediction_status", "endpoint_status"):
                entries[label][column] = {
                    str(k): int(v) for k, v in subset[column].value_counts().items()
                }
        own_score = _conditional_score(all_rows)
        entries["expression_conditional_cross_entropy_per_RNA_UMI"] = own_score[
            "conditional_cross_entropy_per_RNA_UMI"
        ]
        entries["expression_conditional_count_score"] = own_score
        entries["expression_common_support_conditional_count_score"] = _conditional_score(
            all_rows[all_rows.common_family_support]
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
        if correction is not None:
            write_json(path / "correction.json", correction.model_dump(mode="json"))
        return dict(
            scoring_version=2,
            evaluator_implementation_sha256=implementation_tree_hash(),
            guides=g,
            RNA_features=f,
            endpoint_cells=int(n_end.sum()),
            query_donor=spec.query_donor,
            prediction_seal_sha256=sealed.identity(),
            complete_abundance_denominator=True,
            observed_counts_accessed_only_after_prediction_publication=True,
            composition_scope="equal_cell_population_means_only",
            effect_definition="log_positive_composition_minus_log_positive_independent_reference",
            truth_smoothing="expression_pseudocount_once_then_normalize",
            prediction_smoothing="already_published_no_added_pseudocount",
            numerical_zero_policy="float64_tiny_floor_for_prediction_serialization_underflow",
            abundance_provenance_definition="query_support_separate_from_family_prediction_basis",
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
            **(
                {
                    "previous_evaluation": correction.previous_evaluation_sha256,
                    "correction": correction.identity(),
                }
                if correction is not None
                else {}
            ),
        },
        writer=write,
    )
    if verify_bundle(prediction, stage="prediction").identity() != sealed.identity():
        raise ContractError("Evaluation modified published predictions.")
