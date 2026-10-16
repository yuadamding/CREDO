"""Fitting-only persistence normalizers, separate training and selection maps.

Consumes an already admitted in-memory cache; never opens files, selects an
observation checkpoint, loads query outcomes or changes model parameters.
Production input leases, runtime qualification and an operation/output budget
belong to the caller. A successful arithmetic call is not launch authority.
"""

from dataclasses import asdict
import math

import numpy as np
import torch

from .fitting_provider import FoldFittingCache, _hash, _sha
from .geometry import ObservedLaw, RaggedSinkhornGeometry
from .measure_ops import source_log_frequencies
from .molecular import endpoint_effect_terms, transform_composition
from .sampling import (LawIdentity, LawRequest, SamplingContext, SourceLaw,
                       SamplerSpecification, sample_scale_sources)
from .selection import CHANNELS, complete_vector


PROTOCOL_SHA256 = "a9df0ad879087b6e9538dfd4de72a6f6f8bdac52c3c03d4219c1de2f0147b587"
STATE_CHANNELS = ("state_energy", "spread_absolute_error", "shape_mmd2", "joint_mse")
TARGET_CHANNELS = ("target_endpoint_mse", "pooled_effect_mse")


def _mean(values):
    values = tuple(values)
    if not values:
        raise ValueError("Required fitting support is empty; no replacement scale")
    return math.fsum(values) / len(values)


def _positive(values, names):
    if set(values) != set(names) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in values.values()):
        raise ValueError("Every fitting-only scale must be finite and strictly positive; no floor")
    return {key: float(values[key]) for key in names}


def _sinkhorn(raw):
    if not math.isfinite(raw) or raw < -1e-5:
        raise ValueError("Persistence Sinkhorn violates the bound numerical rule")
    return {"raw_risk": raw, "risk": max(0., raw),
            "negative_roundoff_corrected": raw < 0, "negative_tolerance": 1e-5,
            "training_loss_modified": False}


def reduce_normalizers(cache, law_rows, donor_rows, *, sampler_binding, producer_sha256):
    """Reconcile exact cache support before reducing computed scalar rows.

    Training state/shape are law-equal; selection state/shape/joint are
    donor-equal then law-equal. Molecular channels retain guide -> target ->
    donor reduction. Count and pooled control each contribute once per donor.
    Supplied numeric rows alone do not prove correct upstream calculations.
    """
    if not isinstance(cache, FoldFittingCache):
        raise TypeError("Exact validated fitting cache required")
    cache.assert_unchanged()
    _sha(producer_sha256)
    donors = cache.manifest.fold.training_donors
    rows = tuple(law_rows)
    expected = {(d, g) for i in cache.paired for d, g in (cache.laws[i],)}
    indexed = {}
    for row in rows:
        key = (row.get("donor"), row.get("guide_index"))
        if key not in expected or key in indexed:
            raise ValueError("Duplicated, foreign or nonpaired normalization law")
        donor, guide = key
        entry = cache.manifest.catalog[guide]
        if (row.get("guide_id") != entry.guide_id or row.get("target_id") != entry.target_id
                or row.get("is_control") is not entry.is_control):
            raise ValueError("Normalization law identity differs from fixed cache routing")
        required = set(STATE_CHANNELS) | {"sinkhorn"}
        if not entry.is_control:
            required.update(TARGET_CHANNELS)
        for name in required:
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Unavailable required law score; no denominator repair")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Incomplete paired-law support; no favorable subset normalizer")
    catalog = {}
    for row in donor_rows:
        donor = row.get("donor")
        if donor not in donors or donor in catalog:
            raise ValueError("Exactly one catalog/control score row per fitting donor required")
        if row.get("catalog_categories") != len(cache.manifest.catalog):
            raise ValueError("Normalization omitted complete-catalog categories")
        for key in ("count_ce_minus_entropy", "count_quadratic", "pooled_control_endpoint_mse"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Unavailable required donor score")
        catalog[donor] = row
    if set(catalog) != set(donors):
        raise ValueError("Missing fitting donor catalog/control scores")
    reduced = {}
    for donor in donors:
        laws = [indexed[key] for key in sorted(indexed) if key[0] == donor]
        values = {name: _mean(row[name] for row in laws) for name in STATE_CHANNELS}
        targets = {}
        for row in laws:
            if not row["is_control"]:
                targets.setdefault(row["target_id"], []).append(row)
        for name in TARGET_CHANNELS:
            values[name] = _mean(_mean(row[name] for row in group) for group in targets.values())
        for name in ("count_ce_minus_entropy", "count_quadratic", "pooled_control_endpoint_mse"):
            values[name] = catalog[donor][name]
        reduced[donor] = complete_vector(values)
    selection = _positive({name: _mean(reduced[d][name] for d in donors) for name in CHANNELS}, CHANNELS)
    training = _positive({
        "state": _mean(row["sinkhorn"] for row in indexed.values()),
        "count": selection["count_ce_minus_entropy"],
        "shape": _mean(row["shape_mmd2"] for row in indexed.values()),
        "endpoint": .5 * (selection["target_endpoint_mse"] + selection["pooled_control_endpoint_mse"]),
        "effect": selection["pooled_effect_mse"],
    }, ("state", "count", "shape", "endpoint", "effect"))
    support = [{"donor": d, "guide_index": g, "guide_id": cache.manifest.catalog[g].guide_id,
                "source_positive_RNA_cells": int(cache.populations[d, "Stim8hr"].positive_counts[g]),
                "endpoint_positive_RNA_cells": int(cache.populations[d, "Stim48hr"].positive_counts[g])}
               for d, g in sorted(indexed)]
    payload = dict(schema="credo.frontier.computed-normalizers.v1", protocol_sha256=PROTOCOL_SHA256,
        objective_rules_sha256=cache.manifest.rules_sha256, fitting_donors=list(donors),
        fitting_input_sha256=cache.manifest.input_sha256, cache_manifest_sha256=cache.manifest.identity,
        observation_binding=asdict(cache.manifest.observation_binding),
        producer_sha256=producer_sha256, sampler_binding=sampler_binding,
        support_inventory=support, support_inventory_sha256=_hash(support),
        per_law=[indexed[key] for key in sorted(indexed)], per_donor=[catalog[d] for d in donors],
        donor_selection_risks=reduced, training_values=training, selection_values=selection,
        synthetic_fixture=cache.manifest.synthetic_fixture,
        scientific_updates=0, query_outcomes_used=False, scientific_launch_authority=False)
    # The producer retains full per-law numerical correction records, not just
    # their mean. Hashing rejects nonfinite/non-JSON hidden metadata as well.
    return {**payload, "content_sha256": _hash(payload)}


def _tensor(array, device, dtype=torch.float64):
    return torch.tensor(array, dtype=dtype, device=device)


@torch.no_grad()
def compute_normalizers(cache, geometry, *, study_id, study_seed, producer_sha256,
                        sampler_source_sha256, protocol_payload, device="cpu",
                        source_particles=512, block=128):
    """Bounded one-law-at-a-time persistence calculations, no optimizer.

    A real call is legal only in a separately admitted numerical phase. Tests
    use tiny fabricated caches, <=128 source particles and CPU. The function
    preserves every observed endpoint atom and every captured catalog entry.
    """
    import hashlib
    from .evaluation import score_state_law, score_catalog, score_joint
    if (not isinstance(cache, FoldFittingCache) or not isinstance(geometry, RaggedSinkhornGeometry)
            or type(protocol_payload) is not bytes
            or hashlib.sha256(protocol_payload).hexdigest() != PROTOCOL_SHA256):
        raise ValueError("Exact cache, retained geometry and frozen normalization protocol required")
    for value in (producer_sha256, sampler_source_sha256):
        _sha(value)
    dev = torch.device(device)
    if type(source_particles) is not int or type(block) is not int or block < 1:
        raise ValueError("Explicit positive integer numerical dimensions required")
    if cache.manifest.synthetic_fixture:
        if dev.type != "cpu" or not 1 <= source_particles <= 128 or block > 128:
            raise ValueError("Fabricated normalization is bounded CPU work only")
    elif source_particles != 512 or dev.type not in ("cpu", "cuda"):
        raise ValueError("Scientific normalizers require the fixed512 source draws")
    context = SamplingContext(study_id, cache.manifest.fold.training_donors, "F0-V0",
                              study_seed, 0, 0, "scales")
    spec = SamplerSpecification(maximum_laws=1)
    sampler_binding = dict(context=asdict(context), specification=asdict(spec),
                           source_sha256=sampler_source_sha256, particles=source_particles,
                           pairwise_block=block, geometry=geometry.backend_identity,
                           unused_noise_omitted=True, unused_full_source_diagnostics_omitted=True)
    cache.assert_unchanged()
    donor_rows, source_pools, endpoint_pools, frequencies = [], {}, {}, {}
    controls = [i for i, g in enumerate(cache.manifest.catalog) if g.is_control]
    for donor in cache.manifest.fold.training_donors:
        src, end = (cache.populations[donor, condition] for condition in ("Stim8hr", "Stim48hr"))
        logq = source_log_frequencies(_tensor(src.captured_counts, dev, torch.int64))
        p = _tensor(end.captured_counts, dev)
        p = p / p.sum()
        frequencies[donor] = (logq.exp(), p)
        catalog = score_catalog(logq, _tensor(end.captured_counts, dev, torch.int64),
                                expected_catalog_size=len(cache.manifest.catalog))
        for population, out in ((src, source_pools), (end, endpoint_pools)):
            total = sum(int(population.positive_counts[i]) for i in controls)
            if total <= 0:
                raise ValueError("Missing required pooled control; no renormalized substitute")
            mean = sum(population.mean_compositions[i] * (int(population.positive_counts[i]) / total)
                       for i in controls if population.positive_counts[i])
            out[donor] = _tensor(mean, dev)
        error = transform_composition(source_pools[donor]) - transform_composition(endpoint_pools[donor])
        donor_rows.append(dict(donor=donor, catalog_categories=len(cache.manifest.catalog),
            count_ce_minus_entropy=catalog["count_ce_minus_entropy"],
            count_quadratic=catalog["count_quadratic"],
            pooled_control_endpoint_mse=float(error.square().mean()), count_details=catalog))
    law_rows = []
    for index in cache.paired:
        donor, guide = cache.laws[index]
        entry = cache.manifest.catalog[guide]
        src, end = (cache.populations[donor, c] for c in ("Stim8hr", "Stim48hr"))
        a, b = map(int, src.offsets[guide:guide + 2])
        c, d = map(int, end.offsets[guide:guide + 2])
        source = src.states[a:b]
        law = SourceLaw(LawIdentity(donor, "Stim8hr", entry.guide_id,
            src.reference.source_id, src.reference.replicate_id), source,
            np.full(b - a, 1. / (b - a), dtype=np.float64))
        sampled = sample_scale_sources((LawRequest(law),), context, particles=source_particles,
                               specification=spec, device=dev)
        x = sampled.states[0]
        y = _tensor(end.states[c:d], dev, torch.float32)
        logs = torch.full((1, source_particles), -math.log(source_particles), device=dev, dtype=torch.float32)
        actual = ObservedLaw(f"{donor}::{entry.guide_id}", y,
                             torch.full((d - c,), 1. / (d - c), device=dev, dtype=torch.float32))
        sinkhorn = _sinkhorn(float(geometry(x[None], logs, law_ids=(actual.law_id,),
                                           observations=(actual,)).per_law_losses[0]))
        xd, yd = x.double(), y.double()
        wx = torch.full((len(xd),), 1. / len(xd), device=dev, dtype=torch.float64)
        wy = torch.full((len(yd),), 1. / len(yd), device=dev, dtype=torch.float64)
        # The14 normalizers use the fixed scale draw, not additional full-source
        # energy diagnostics. Preserve those diagnostics in assessment defaults,
        # but avoid an otherwise unused source^2/source*endpoint calculation here.
        scores = score_state_law(xd, wx, yd, wy, None, None, block=block,
                                 include_source_diagnostics=False)
        q, p = frequencies[donor]
        joint = score_joint(xd, wx, float(q[guide]), yd, wy, float(p[guide]))
        row = dict(donor=donor, guide_index=guide, guide_id=entry.guide_id,
                   target_id=entry.target_id, is_control=entry.is_control,
                   sinkhorn=sinkhorn["risk"], sinkhorn_details=sinkhorn,
                   state_details=scores, joint_details=joint,
                   **{name: scores[name] for name in STATE_CHANNELS if name != "joint_mse"},
                   joint_mse=joint["joint_mse"])
        if not entry.is_control:
            terms = endpoint_effect_terms(_tensor(src.mean_compositions[guide], dev)[None],
                _tensor(end.mean_compositions[guide], dev)[None], source_pools[donor], endpoint_pools[donor])
            row.update(target_endpoint_mse=float(terms["target_mse"]), pooled_effect_mse=float(terms["effect_mse"]),
                       molecular_interaction=float(terms["interaction_per_law"][0]),
                       zero_effect_mse=float(terms["zero_effect_mse"]))
        law_rows.append(row)
    cache.assert_unchanged()
    return reduce_normalizers(cache, law_rows, donor_rows, sampler_binding=sampler_binding,
                              producer_sha256=producer_sha256)
