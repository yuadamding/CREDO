"""D2 source-only population lease and bounded batched, identity-seeded inference."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from itertools import zip_longest
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from credo.contracts import Axis, FiniteMeasure
from credo.recipes.compact_sde_v3.particles import ParticleState, axis_grid, sample_noise
from development_inputs import frozen_binding
from full_forecast import ForecastCatalog, ForecastSources, _model_digest, _prefixes, measure_seed, terminal_summary
from gpu_native import TensorDynamics, tensor_rollout
from latent_support import FrozenLatentSupportStore, PhaseAccess
from native_study import Guide


FORECAST_NUMERICAL_POLICY = "architecture_fixed_v4_mlp_batch16_residual_single"
FORECAST_ATOL = 2e-5
FORECAST_RTOL = 1e-5
_AGREEMENT_FIELDS = ("state", "log_weight", "mass_increment", "conditional_features")


class ForecastAgreementError(RuntimeError):
    """An unchanged numerical gate failed; retain the complete comparison."""

    def __init__(self, report):
        self.report = report
        failed = [name for name, value in report["fields"].items() if not value["passed"]]
        super().__init__("Fixed scalar/production forecast agreement failed: " + ", ".join(failed or ["inventory"]))


def forecast_numerical_policy(model, *, diagnostic_legacy16=False):
    """Architecture-bound arithmetic, never a result-selected fallback.

    The MLP route retains the previous batch shape. Residual trunks always use
    the scalar reference's single-guide numerical shape. This changes neither
    FP32 parameters nor model equations, source/noise streams, or integration.
    """
    if model.trunk_kind not in {"mlp", "residual_mlp"}:
        raise ValueError("Unbound forecast trunk architecture")
    return dict(
        policy=("diagnostic_legacy16_not_production" if diagnostic_legacy16 else FORECAST_NUMERICAL_POLICY),
        trunk_kind=model.trunk_kind, source_chunk_maximum_guides=16,
        dynamics_maximum_guides=(16 if diagnostic_legacy16 or model.trunk_kind == "mlp" else 1),
        precision="strict_fp32", automatic_fallback=False,
        diagnostic_only=bool(diagnostic_legacy16),
    )


@contextmanager
def d2_source_lease(inputs, metadata, handoff):
    # Deliberately do not call prepared_development: it opens D2 endpoints.
    entries = [entry for entry in inputs["stores"] if entry["source_ids"] == ["D2_Stim8hr"]]
    if len(entries) != 1:
        raise ValueError("One exact D2 source store is required")
    entry = entries[0]
    binding = frozen_binding(inputs, entry)
    store = FrozenLatentSupportStore(binding, PhaseAccess("selection"), expected_representation=binding.representation)
    try:
        with store.cached_snapshot(maximum_bytes=inputs["maximum_snapshot_bytes"]):
            rows = metadata["canonical_guides"]
            if len(rows) != 26504 or [r["guide_index"] for r in rows] != list(range(26504)):
                raise ValueError("Canonical catalog axis differs")
            guides = tuple(Guide(r["sgRNA"], "__NTC__" if r["is_design_ntc"] else r["primary_target_id"],
                                 r["is_design_ntc"]) for r in rows)
            records = [store.support_record("D2_Stim8hr", i) for i in range(26504)]
            counts = tuple(r["captured_cells"] for r in records)
            catalog = ForecastCatalog("D2", guides, counts, entry["completion_sha256"],
                                      inputs["representation_sha256"], handoff["query_inventory_sha256"])
            probability = catalog.source_frequency
            laws, meta = {}, []
            for row in handoff["query_records"]:
                index, key = row["guide_index"], row["measure_id"]
                law = store.read(store.ref("D2_Stim8hr", index))
                # SupportStore.read returns a normalized EmpiricalLaw, not the
                # mass-bearing FiniteMeasure consumed by the particle sampler.
                # Preserve its probabilities and apply captured source mass once.
                laws[key] = FiniteMeasure(law.coordinates, law.probabilities * probability[index],
                                          float(probability[index]))
                meta.append(dict(measure_id=key, guide_id=row["guide_id"], embedding_id=row["embedding_id"],
                    is_control=row["is_control"], sample_id="D2", context_group_id="captured-aggregate::D2"))
            data = SimpleNamespace(axis=Axis("physical", ("Stim8hr", "Stim48hr"), (8., 48.), "Stim8hr"),
                measures={"Stim8hr": laws}, measure_meta=pd.DataFrame(meta), latent_dim=48)
            yield ForecastSources.from_compiled(data, catalog)
    finally:
        store.close()


def iter_batched_forecasts(model, sources, *, prefixes=(4096, 8192), steps_per_interval=8,
                           seed=0, batch_guides=16, progress=lambda _: None, diagnostic_legacy16=False):
    """Same per-guide RNG/rollout contract; no repeated full-panel dataframe index.

    No ecology couples guides. Source/noise streams are constructed per guide,
    then handled by the architecture-bound numerical policy. Residual trunks
    use single-guide dynamics regardless of the source chunk; MLP retains the
    original multi-guide arithmetic. The diagnostic-only legacy route is never
    selected from numerical results and cannot earn a production certificate.
    """
    prefixes = _prefixes(prefixes)
    if type(batch_guides) is not int or not 1 <= batch_guides <= 16:
        raise ValueError("Forecast guide batch exceeds its frozen memory shape")
    policy = forecast_numerical_policy(model, diagnostic_legacy16=diagnostic_legacy16)
    if diagnostic_legacy16 and batch_guides != 16:
        raise ValueError("Legacy diagnostic requires its exact 16-guide chunk")
    if model.context_mode != "none" or model.context_enabled or not model.growth_enabled:
        raise ValueError("Full DDR without ecology required")
    if not set(sources.measure_meta.embedding_id) <= set(model.embedding_ids):
        raise ValueError("Unfitted targets cannot receive forecasts")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32 or model.latent_dim != sources.latent_dim:
        raise ValueError("Frozen FP32 model and state dimension required")
    device = parameter.device
    metadata = sources.measure_meta.set_index("measure_id")
    indices = {key: index for index, key in enumerate(sources.measure_ids)}
    before, was_training = _model_digest(model), model.training
    model.eval()
    dynamics = TensorDynamics(model, precision="strict_fp32")
    try:
        with torch.inference_mode():
            grid = axis_grid(sources.axis, steps_per_interval, device=device, dtype=torch.float32)
            for start in range(0, len(sources.measure_ids), batch_guides):
                ids = sources.measure_ids[start:start+batch_guides]
                states, noises = [], []
                for key in ids:
                    law = sources.measures["Stim8hr"][key]
                    support = torch.tensor(np.array(law.support, copy=True), dtype=torch.float32, device=device)
                    weights = torch.tensor(law.normalized_weights, dtype=torch.float32, device=device)
                    generator = torch.Generator(device=device).manual_seed(measure_seed(seed, key, "source"))
                    draws = torch.multinomial(weights, prefixes[-1], replacement=True, generator=generator)
                    initial = ParticleState(support[draws].unsqueeze(0),
                        torch.full((1, prefixes[-1]), -math.log(prefixes[-1]), device=device, dtype=torch.float32),
                        torch.tensor([math.log(law.total_mass)], device=device, dtype=torch.float32), (key,),
                        (metadata.loc[key, "embedding_id"],), (metadata.loc[key, "context_group_id"],),
                        torch.tensor([indices[key]], device=device), torch.ones(1, device=device, dtype=torch.float32))
                    states.append(initial)
                    noises.append(sample_noise(initial, grid, seed=measure_seed(seed, key, "noise")))
                    del support, weights, draws
                if policy["dynamics_maximum_guides"] == 1:
                    # Preserve the reference [1, particles, dimension] shape
                    # for every residual Linear/LayerNorm/head evaluation.
                    for initial, noise in zip(states, noises, strict=True):
                        key = initial.measure_ids[0]
                        residual_indices = torch.tensor(
                            [model._residual_index.get(initial.embedding_ids[0], -1)+1],
                            device=device, dtype=torch.long)
                        result = tensor_rollout(model, dynamics, initial, grid, noise, residual_indices)
                        yield terminal_summary(result, prefixes, sources.guide_indices[indices[key]], seed=seed)
                        del result, residual_indices
                    del initial, noise
                else:
                    state = ParticleState(torch.cat([s.z for s in states]), torch.cat([s.logw for s in states]),
                        torch.cat([s.log_m0 for s in states]), ids,
                        tuple(s.embedding_ids[0] for s in states), tuple(s.context_group_ids[0] for s in states),
                        torch.cat([s.measure_indices for s in states]), torch.cat([s.residual_scale for s in states]))
                    residual_indices = torch.tensor([model._residual_index.get(key, -1)+1 for key in state.embedding_ids],
                                                    device=device, dtype=torch.long)
                    result = tensor_rollout(model, dynamics, state, grid, torch.cat(noises, dim=1), residual_indices)
                    for local, key in enumerate(ids):
                        yield terminal_summary(result.slice_measure(local), prefixes,
                                               sources.guide_indices[indices[key]], seed=seed)
                    del result, state, residual_indices
                del states, noises
                progress(dict(event="forecast_guides", completed=min(start+batch_guides, len(sources.measure_ids)),
                              total=len(sources.measure_ids), seed=seed, parameter_updates=0))
    finally:
        model.train(was_training)
        if _model_digest(model) != before:
            raise RuntimeError("Frozen model changed during prediction")


def _finite_float(value):
    return float(value) if np.isfinite(value) else None


def _field_agreement(a, b, *, measure_id, guide_index):
    """JSON-safe diagnostics using the original np.allclose directional rule."""
    a, b = np.asarray(a), np.asarray(b)
    report = dict(reference_shape=list(a.shape), candidate_shape=list(b.shape),
        elements_compared=0, violation_count=0, nonfinite_count=0,
        maximum_absolute_difference=0., maximum_normalized_tolerance_ratio=0.,
        worst_absolute_error=None, worst_tolerance_error=None,
        shape_mismatch=a.shape != b.shape, empty_array=a.size == 0 or b.size == 0)
    if report["shape_mismatch"] or report["empty_array"]:
        report["passed"] = False
        return report
    finite = np.isfinite(a) & np.isfinite(b)
    # Preserve the v3 gate, including its direction: b is the tolerance scale.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = np.abs(a-b)
        allowance = FORECAST_ATOL + FORECAST_RTOL * np.abs(b)
        ratio = difference.astype(np.float64) / allowance.astype(np.float64)
        violated = ~finite | ~np.isclose(a, b, atol=FORECAST_ATOL, rtol=FORECAST_RTOL, equal_nan=False)
    absolute_rank = np.where(finite, difference.astype(np.float64), np.inf)
    tolerance_rank = np.where(finite, ratio, np.inf)

    def worst(rank):
        coordinate = np.unravel_index(int(np.argmax(rank)), rank.shape)
        return dict(measure_id=measure_id, guide_index=int(guide_index),
            coordinate=[int(i) for i in coordinate], reference_value=_finite_float(a[coordinate]),
            candidate_value=_finite_float(b[coordinate]), finite=bool(finite[coordinate]),
            absolute_difference=_finite_float(difference[coordinate]),
            allowed_difference=_finite_float(allowance[coordinate]),
            normalized_tolerance_ratio=_finite_float(ratio[coordinate]))

    report.update(elements_compared=int(a.size), violation_count=int(violated.sum()),
        nonfinite_count=int((~finite).sum()), passed=not bool(violated.any()),
        maximum_absolute_difference=_finite_float(absolute_rank.max()),
        maximum_normalized_tolerance_ratio=_finite_float(tolerance_rank.max()),
        worst_absolute_error=worst(absolute_rank), worst_tolerance_error=worst(tolerance_rank))
    return report


def _compare_forecast_streams(expected, actual, *, expected_guides):
    """Collect all fields/guides before deciding; always close both streams.

    This numerical helper makes no device-qualification claim. Its public
    caller enforces the production CUDA boundary.
    """
    rows, inventory_errors = [], []
    missing = object()
    try:
        for left, right in zip_longest(expected, actual, fillvalue=missing):
            if left is missing or right is missing:
                inventory_errors.append("missing_reference_row" if left is missing else "missing_candidate_row")
                continue
            left_id = (left.measure_id, left.guide_index, left.seed)
            right_id = (right.measure_id, right.guide_index, right.seed)
            if left_id != right_id:
                inventory_errors.append(dict(reference=list(left_id), candidate=list(right_id)))
            rows.append(dict(measure_id=left.measure_id, guide_index=int(left.guide_index), seed=int(left.seed),
                fields={name: _field_agreement(getattr(left, name), getattr(right, name),
                    measure_id=left.measure_id, guide_index=left.guide_index) for name in _AGREEMENT_FIELDS}))
    finally:
        # Close the later-opened production iterator first, so an original
        # training/eval flag is restored by the outer reference iterator last.
        try:
            close = getattr(actual, "close", None)
            if close is not None:
                close()
        finally:
            close = getattr(expected, "close", None)
            if close is not None:
                close()
    if len(rows) != expected_guides:
        inventory_errors.append(dict(expected_guides=expected_guides, compared_guides=len(rows)))
    fields = {}
    for name in _AGREEMENT_FIELDS:
        parts = [row["fields"][name] for row in rows]
        def worst(field, point):
            candidates = [part for part in parts if part[point] is not None]
            if not candidates:
                return 0., None
            selected = max(candidates, key=lambda p: math.inf if p[field] is None else p[field])
            return selected[field], selected[point]
        absolute, worst_absolute = worst("maximum_absolute_difference", "worst_absolute_error")
        ratio, worst_ratio = worst("maximum_normalized_tolerance_ratio", "worst_tolerance_error")
        fields[name] = dict(passed=bool(parts) and all(p["passed"] for p in parts),
            elements_compared=sum(p["elements_compared"] for p in parts),
            violation_count=sum(p["violation_count"] for p in parts),
            nonfinite_count=sum(p["nonfinite_count"] for p in parts),
            shape_mismatch_count=sum(p["shape_mismatch"] for p in parts),
            empty_array_count=sum(p["empty_array"] for p in parts),
            maximum_absolute_difference=absolute, maximum_normalized_tolerance_ratio=ratio,
            worst_absolute_error=worst_absolute, worst_tolerance_error=worst_ratio)
    return dict(agreement_passed=not inventory_errors and all(p["passed"] for p in fields.values()),
        atol=FORECAST_ATOL, rtol=FORECAST_RTOL, compared_guides=len(rows),
        inventory_errors=inventory_errors, fields=fields, guide_errors=rows,
        maximum_absolute_difference={name: value["maximum_absolute_difference"] for name, value in fields.items()})


def qualify_forecast(model, sources, *, seed, device, diagnostic_legacy16=False):
    """Fixed source-only gate, or explicitly nonqualifying legacy diagnosis."""
    from full_forecast import iter_forecasts
    if torch.device(device).type != "cuda" or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Real forecast precision qualification requires CUDA")
    count = min(16, len(sources.measure_ids))
    chosen = replace(sources, measure_ids=sources.measure_ids[:count], guide_indices=sources.guide_indices[:count])
    expected = iter_forecasts(model, chosen, prefixes=(4096, 8192), steps_per_interval=8, seed=seed)
    actual = iter_batched_forecasts(model, chosen, prefixes=(4096, 8192), steps_per_interval=8, seed=seed,
                                   diagnostic_legacy16=diagnostic_legacy16)
    was_training = model.training
    try:
        report = _compare_forecast_streams(expected, actual, expected_guides=count)
    finally:
        # Interleaved lazy iterators can finish in either order. Do not let
        # their nested eval/restore bookkeeping alter the caller's mode.
        model.train(was_training)
    report.update(status=("diagnostic_only" if diagnostic_legacy16 else
                          "passed" if report["agreement_passed"] else "failed"),
        scope=("legacy_batch16_diagnosis_not_a_production_certificate" if diagnostic_legacy16 else
               "source_only_scalar_vs_production_not_endpoint_precision"),
        numerical_policy=forecast_numerical_policy(model, diagnostic_legacy16=diagnostic_legacy16),
        guides=count, particles=8192, steps=8, seed=seed,
        production_qualified=bool(report["agreement_passed"] and not diagnostic_legacy16),
        parameter_updates=0, endpoint_observations_read=False)
    if not diagnostic_legacy16 and not report["agreement_passed"]:
        raise ForecastAgreementError(report)
    return report
