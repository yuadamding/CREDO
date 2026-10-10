"""Source-only, zero-update full-DDR forecasts and two-stage bank publication.

The caller owns phase authorization, checked input leases, frozen checkpoint
loading and GPU admission. This module does not open endpoint observations,
score candidates, train, or grant access. BANK.json is not an assessment seal;
seal_bank requires a separately published post-prediction input-check receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from scipy.special import logsumexp

from credo.recipes.compact_sde_v3.particles import (
    NoContextProvider, axis_grid, rollout, sample_initial_particles, sample_noise,
)
from encode_sources import publish_json, sha_file
from native_study import Guide, count_vector_sha256, series_id


def _digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("A bound lowercase SHA256 is required.")
    return value


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ForecastCatalog:
    donor: str
    guides: tuple[Guide, ...]
    counts: tuple[int, ...]
    source_artifact_sha256: str
    representation_sha256: str
    support_sha256: str

    def __post_init__(self):
        if self.donor not in {"D1", "D2", "D3", "D4"}:
            raise ValueError("Only explicitly authorized GSE314342 donors are supported.")
        if not self.guides or len(self.counts) != len(self.guides):
            raise ValueError("The complete ordered count/guide catalog is required.")
        if len({g.guide_id for g in self.guides}) != len(self.guides):
            raise ValueError("Duplicate actual guide identities.")
        if any(not g.guide_id or not g.target_id or type(g.is_control) is not bool for g in self.guides):
            raise ValueError("Exact guide, target and control identities are required.")
        if any(type(n) is not int or n < 0 for n in self.counts) or sum(self.counts) <= 0:
            raise ValueError("Preserve complete nonnegative integer counts and a positive capture total.")
        for value in (self.source_artifact_sha256, self.representation_sha256, self.support_sha256):
            _digest(value)

    @property
    def source_frequency(self) -> np.ndarray:
        counts = np.asarray(self.counts, dtype=np.float64)
        return (counts + 0.5) / (counts.sum() + 0.5 * len(counts))

    @property
    def identity(self) -> dict:
        return dict(donor=self.donor, source=f"{self.donor}_Stim8hr", catalog_size=len(self.guides),
                    catalog_sha256=_hash_json([[g.guide_id, g.target_id, g.is_control] for g in self.guides]),
                    source_counts_sha256=count_vector_sha256(self.counts),
                    source_artifact_sha256=self.source_artifact_sha256,
                    representation_sha256=self.representation_sha256,
                    support_sha256=self.support_sha256, source_pseudocount=0.5)


@dataclass(frozen=True)
class ForecastSources:
    """Minimal sampler payload: its measures contain only the source checkpoint."""
    catalog: ForecastCatalog
    axis: object
    measures: Mapping
    measure_meta: object
    latent_dim: int
    measure_ids: tuple[str, ...]
    guide_indices: tuple[int, ...]

    @classmethod
    def from_compiled(cls, data, catalog: ForecastCatalog):
        if data.axis.labels != ("Stim8hr", "Stim48hr") or data.axis.values != (8.0, 48.0):
            raise ValueError("Forecasting requires the exact stimulated 40-hour interval.")
        metadata = data.measure_meta.copy()
        if metadata.empty or set(metadata.sample_id) != {catalog.donor}:
            raise ValueError("One exact query donor is required; no fitting-source substitution.")
        source = data.measures["Stim8hr"]  # Never inspect an endpoint or count block.
        guide_lookup = {guide.guide_id: i for i, guide in enumerate(catalog.guides)}
        source_frequency = catalog.source_frequency  # One complete reduction, not O(G**2).
        metadata = metadata.sort_values("measure_id", kind="stable").reset_index(drop=True)
        indices = []
        for row in metadata.itertuples(index=False):
            if row.guide_id not in guide_lookup:
                raise ValueError("A source guide is absent from the complete catalog.")
            index = guide_lookup[row.guide_id]
            if row.measure_id != series_id(catalog.donor, index):
                raise ValueError("Source donor/guide identity cannot be relabeled.")
            if bool(row.is_control) != catalog.guides[index].is_control:
                raise ValueError("Source control status differs from the catalog.")
            expected_embedding = ("shared-NTC-reference" if catalog.guides[index].is_control
                                  else f"target::{catalog.guides[index].target_id}")
            if row.embedding_id != expected_embedding:
                raise ValueError("Source target binding differs from the actual guide catalog.")
            law = source[row.measure_id]
            if law.latent_dim != data.latent_dim or not math.isclose(
                law.total_mass, source_frequency[index], rel_tol=3e-6, abs_tol=0.0
            ):
                raise ValueError("Source law does not use this catalog's regularized captured mass.")
            indices.append(index)
        if len(set(indices)) != len(indices) or set(source) != set(metadata.measure_id):
            raise ValueError("Source laws and exact modeled-guide inventory disagree.")
        ids = tuple(metadata.measure_id)
        # Copy the mapping, not the atoms. The bound lease must outlive prediction.
        laws = MappingProxyType({key: source[key] for key in ids})
        return cls(catalog, data.axis, MappingProxyType({"Stim8hr": laws}), metadata,
                   data.latent_dim, ids, tuple(indices))


def _prefixes(values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(values)
    if not values or any(type(n) is not int or n < 2 for n in values) or tuple(sorted(set(values))) != values:
        raise ValueError("Distinct increasing particle prefixes of at least two are required.")
    return values


def measure_seed(seed: int, measure_id: str, purpose: str) -> int:
    if type(seed) is not int or seed < 0 or purpose not in {"source", "noise"}:
        raise ValueError("A nonnegative seed and exact random-stream purpose are required.")
    # Independent of iteration order, architecture and output chunking. These
    # are deliberately new forecast RNG identities, not historical bank reuse.
    raw = f"full-ddr-forecast-v1\0{seed}\0{measure_id}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class GuideForecast:
    measure_id: str
    guide_index: int
    seed: int
    steps_per_interval: int
    state: np.ndarray
    # Centered unnormalized particle weights, at the largest requested prefix.
    log_weight: np.ndarray
    mass_increment: np.ndarray
    conditional_features: np.ndarray
    ess_fraction: np.ndarray
    max_weight_fraction: np.ndarray


def terminal_summary(result, prefixes: Sequence[int], guide_index: int, *, seed: int = 0) -> GuideForecast:
    prefixes = _prefixes(prefixes)
    if len(result.measure_ids) != 1 or result.terminal_z.shape[1] != prefixes[-1]:
        raise ValueError("One guide and the exact largest particle prefix are required.")
    state = result.terminal_z[0].detach().cpu().numpy().astype(np.float32, copy=True)
    initial = result.logw_steps[0, 0].detach().double().cpu().numpy()
    terminal = result.logw_steps[-1, 0].detach().double().cpu().numpy()
    if not np.isfinite(state).all() or not np.isfinite(initial).all() or not np.isfinite(terminal).all():
        raise ValueError("Nonfinite predicted particles or weights.")
    origin = initial.max()
    initial, terminal = initial - origin, terminal - origin
    increments, features, ess, maximum = [], [], [], []
    phi = np.column_stack((np.ones(len(state)), np.tanh(state.astype(np.float64))))
    for n in prefixes:
        normalizer = logsumexp(terminal[:n])
        probability = np.exp(terminal[:n] - normalizer)
        if not np.isfinite(probability).all() or np.any(probability <= 0):
            raise ValueError("Particle-probability underflow is unresolved, not an observed zero.")
        increments.append(normalizer - logsumexp(initial[:n]))
        features.append(probability @ phi[:n])
        ess.append(1.0 / (n * np.square(probability).sum()))
        maximum.append(probability.max())
    return GuideForecast(result.measure_ids[0], guide_index, seed, len(result.axis_grid) - 1, state, terminal,
                         np.asarray(increments), np.asarray(features), np.asarray(ess), np.asarray(maximum))


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        array = value.detach().cpu().numpy()
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def iter_forecasts(model, sources: ForecastSources, *, prefixes=(4096, 8192),
                   steps_per_interval=8, seed=0) -> Iterator[GuideForecast]:
    """Bound GPU memory to one guide; retain all source support and exact weights.

    Local CPU execution is for synthetic tests only. A production worker must
    require the qualified H100 before calling this numerical component.
    """
    prefixes = _prefixes(prefixes)
    if type(steps_per_interval) is not int or steps_per_interval < 1:
        raise ValueError("Positive integration-step count required.")
    if model.context_mode != "none" or model.context_enabled or not model.growth_enabled:
        raise ValueError("This study requires reaction and forbids ecological feedback.")
    if not set(sources.measure_meta.embedding_id) <= set(model.embedding_ids):
        raise ValueError("Unfitted target embeddings are unavailable, not a reference fallback.")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32 or model.latent_dim != sources.latent_dim:
        raise ValueError("Frozen float32 dynamics and exact state axis are required.")
    device = parameter.device
    before = _model_digest(model)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            grid = axis_grid(sources.axis, steps_per_interval, device=device, dtype=torch.float32)
            for measure_id, index in zip(sources.measure_ids, sources.guide_indices, strict=True):
                state = sample_initial_particles(sources, (measure_id,), prefixes[-1], device=device,
                    dtype=torch.float32, seed=measure_seed(seed, measure_id, "source"))
                result = rollout(model, state, grid, context_provider=NoContextProvider(),
                    noise=sample_noise(state, grid, seed=measure_seed(seed, measure_id, "noise")))
                row = terminal_summary(result, prefixes, index, seed=seed)
                del result, state
                yield row
    finally:
        model.train(was_training)
        if _model_digest(model) != before:
            raise RuntimeError("Prediction changed frozen parameters or buffers; no seal is admissible.")


def full_catalog_probability(catalog: ForecastCatalog, modeled_indices: Sequence[int],
                             increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = tuple(modeled_indices)
    if any(type(i) is not int or i < 0 or i >= len(catalog.guides) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Distinct in-catalog modeled-guide indices are required.")
    increments = np.asarray(increments, dtype=np.float64)
    if increments.ndim != 2 or increments.shape[1] != len(indices) or not np.isfinite(increments).all():
        raise ValueError("Finite prefix-by-modeled-guide mass increments required.")
    logits = np.broadcast_to(np.log(catalog.source_frequency), (len(increments), len(catalog.guides))).copy()
    logits[:, indices] += increments
    log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    if np.any(probability <= 0) or not np.isfinite(probability).all():
        raise ValueError("Catalog-probability underflow is unresolved; do not manufacture zero forecasts.")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Complete-catalog normalization failed.")
    return log_probability, probability


def write_bank(destination: Path, sources: ForecastSources, forecasts: Iterator[GuideForecast], *,
               prefixes: Sequence[int], checkpoint_sha256: str, implementation_sha256: str,
               seed: int, steps_per_interval: int, maximum_output_bytes: int) -> dict:
    """Publish forecast arrays, without an assessment seal or outcome reads.

    Admission precedes consuming the lazy forecast iterator. Preserve a failed
    publication in place; no overwrite, automatic retry or cleanup is attempted.
    """
    prefixes = _prefixes(prefixes)
    _digest(checkpoint_sha256)
    _digest(implementation_sha256)
    measure_seed(seed, sources.measure_ids[0], "source")
    if type(steps_per_interval) is not int or steps_per_interval < 1:
        raise ValueError("Exact positive integration-step count is required.")
    n, dimension, largest = len(sources.measure_ids), sources.latent_dim, prefixes[-1]
    estimate = 4 * 1024**2 + n * (largest * (4 * dimension + 8) + len(prefixes) * (8 * (dimension + 4)))
    estimate += len(sources.catalog.guides) * (len(prefixes) * 16 + 64)
    if type(maximum_output_bytes) is not int or maximum_output_bytes < estimate:
        raise ValueError("Forecast array admission exceeds the explicit output budget.")
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    array_path = destination / "forecast.h5"
    increments = np.empty((len(prefixes), n), dtype=np.float64)
    stream = iter(forecasts)
    try:
        with h5py.File(array_path, "x") as handle:
            states = handle.create_dataset("terminal_state", (n, largest, dimension), dtype="<f4",
                                           chunks=(1, min(largest, 1024), dimension))
            log_weights = handle.create_dataset("terminal_log_weight", (n, largest), dtype="<f8",
                                                chunks=(1, min(largest, 1024)))
            features = handle.create_dataset("conditional_features", (len(prefixes), n, dimension + 1), dtype="<f8")
            ess = handle.create_dataset("ess_fraction", (len(prefixes), n), dtype="<f8")
            maximum = handle.create_dataset("max_weight_fraction", (len(prefixes), n), dtype="<f8")
            for row_index, (measure_id, guide_index) in enumerate(zip(sources.measure_ids, sources.guide_indices, strict=True)):
                try:
                    row = next(stream)
                except StopIteration as error:
                    raise ValueError("Forecast inventory is incomplete; no bank or seal.") from error
                if row.measure_id != measure_id or row.guide_index != guide_index:
                    raise ValueError("Forecast row identity/order differs from the bound source panel.")
                if type(row.seed) is not int or row.seed != seed or row.steps_per_interval != steps_per_interval:
                    raise ValueError("Forecast seed/integration identity differs from the publication contract.")
                arrays = (row.state, row.log_weight, row.mass_increment, row.conditional_features,
                          row.ess_fraction, row.max_weight_fraction)
                shapes = ((largest, dimension), (largest,), (len(prefixes),),
                          (len(prefixes), dimension + 1), (len(prefixes),), (len(prefixes),))
                if any(np.asarray(a).shape != shape or not np.isfinite(a).all() for a, shape in zip(arrays, shapes, strict=True)):
                    raise ValueError("Forecast shapes or finite-value requirements failed.")
                if not np.all(row.conditional_features[:, 0] == 1.0) and not np.allclose(
                    row.conditional_features[:, 0], 1.0, atol=1e-12, rtol=0
                ):
                    raise ValueError("Joint constant feature must equal one.")
                if np.any(row.ess_fraction <= 0) or np.any(row.ess_fraction > 1 + 1e-12):
                    raise ValueError("Invalid conditional effective-sample-size fraction.")
                states[row_index], log_weights[row_index] = row.state, row.log_weight
                features[:, row_index], ess[:, row_index] = row.conditional_features, row.ess_fraction
                maximum[:, row_index], increments[:, row_index] = row.max_weight_fraction, row.mass_increment
            if next(stream, None) is not None:
                raise ValueError("Undeclared extra forecast rows; no bank or seal.")
            log_probability, probability = full_catalog_probability(sources.catalog, sources.guide_indices, increments)
            handle.create_dataset("guide_index", data=np.asarray(sources.guide_indices, dtype="<i4"))
            handle.create_dataset("prefixes", data=np.asarray(prefixes, dtype="<i4"))
            handle.create_dataset("source_counts", data=np.asarray(sources.catalog.counts, dtype="<i8"))
            handle.create_dataset("source_frequency", data=sources.catalog.source_frequency)
            handle.create_dataset("mass_increment", data=increments)
            handle.create_dataset("catalog_log_probability", data=log_probability)
            handle.create_dataset("catalog_probability", data=probability)
            handle.flush()
    finally:
        if hasattr(stream, "close"):
            stream.close()
    result = dict(schema="credo.all4_full_ddr.forecast_bank.v1", status="arrays_written_not_sealed",
                  **sources.catalog.identity, seed=seed, prefixes=list(prefixes), latent_dim=dimension,
                  steps_per_interval=steps_per_interval, physical_interval_hours=40,
                  source_sampling="weighted_iid_with_replacement_at_maximum_prefix",
                  random_stream_identity="full-ddr-forecast-v1",
                  modeled_measure_ids=list(sources.measure_ids), modeled_guide_indices=list(sources.guide_indices),
                  checkpoint_sha256=checkpoint_sha256, implementation_sha256=implementation_sha256,
                  parameter_updates=0, endpoint_observations_read=False,
                  complement_policy="fixed_unnormalized_regularized_source_persistence",
                  artifacts={"forecast.h5": dict(sha256=sha_file(array_path), bytes=array_path.stat().st_size)},
                  maximum_output_bytes=maximum_output_bytes)
    if array_path.stat().st_size + len(json.dumps(result)) + 8192 > maximum_output_bytes:
        raise ValueError("Actual forecast publication exceeds output budget; no seal.")
    publish_json(destination / "BANK.json", result)
    return result


def seal_bank(destination: Path, *, input_check_receipt: Path, input_check_sha256: str) -> dict:
    """Seal only after an exact post-prediction input/parameter-check receipt.

    This receipt must be produced by the authorized lifecycle after closing its
    checked input lease. SHA strings or this API alone do not authorize outcomes.
    """
    destination, input_check_receipt = Path(destination), Path(input_check_receipt)
    bank_path = destination / "BANK.json"
    bank_sha = sha_file(bank_path)
    if sha_file(input_check_receipt) != _digest(input_check_sha256):
        raise ValueError("Post-prediction input-check receipt identity mismatch.")
    if bank_path.stat().st_size > 16 * 1024**2 or input_check_receipt.stat().st_size > 1024**2:
        raise ValueError("Oversized forecast metadata.")
    bank = json.loads(bank_path.read_text())
    check = json.loads(input_check_receipt.read_text())
    if bank.get("schema") != "credo.all4_full_ddr.forecast_bank.v1" or bank.get("status") != "arrays_written_not_sealed":
        raise ValueError("Expected an exact unsealed forecast bank.")
    required = dict(schema="credo.all4_full_ddr.forecast_input_check.v1", bank_sha256=bank_sha,
                    inputs_verified_after_prediction=True, parameters_unchanged=True, parameter_updates=0,
                    endpoint_observations_read=False)
    required.update({key: bank[key] for key in ("source_artifact_sha256", "support_sha256",
                                               "representation_sha256", "checkpoint_sha256")})
    if any(check.get(key) != value or type(check.get(key)) is not type(value) for key, value in required.items()):
        raise ValueError("Missing or mismatched post-prediction source/parameter custody checks.")
    if set(bank["artifacts"]) != {"forecast.h5"} or set(p.name for p in destination.iterdir()) != {"forecast.h5", "BANK.json"}:
        raise ValueError("Unexpected forecast publication inventory.")
    reference = bank["artifacts"]["forecast.h5"]
    if sha_file(destination / "forecast.h5") != reference["sha256"] or (destination / "forecast.h5").stat().st_size != reference["bytes"]:
        raise ValueError("Forecast arrays changed before seal.")
    result = dict(schema="credo.all4_full_ddr.forecast_seal.v1", status="sealed_forecast_bank",
                  bank_sha256=bank_sha, input_check_sha256=input_check_sha256,
                  parameter_updates=0, endpoint_observations_read=False)
    publish_json(destination / "COMPLETE.json", result)
    return result
