"""Tiny fabricated CPU forecast publication only; exactly zero optimizer calls."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from frontier.forecast import forecast_population
from frontier.forecast_storage import (ForecastStorageSpec, ForecastStorageLimits, ForecastWriter,
                                      ForecastReader, POOL, CATALOG_FIELDS, storage_estimate)
from frontier.evaluation import score_state_law
from frontier.selector_contract import Recipe
from frontier.native_inputs import FoldInputs
from test_forecast import source, model, binding, context, limits as forecast_limits, run


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def prepare(*, missing_pool=False, all_missing=False, family="F0", alpha=.25, spread=1.3):
    s, m = source(missing_pool=missing_pool, all_missing=all_missing), model(family)
    b, c, recipe = binding(m, s), context(m, s), Recipe(m.config.configuration_id, alpha, spread)
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=(5, 8))
    result = forecast_population(m, s, b, c, recipe, prefixes=(5, 8), limits=forecast_limits())
    return s, m, spec, result


def limits():
    return ForecastStorageLimits(128*1024**2, 128*1024**2, 2)


def emit(writer, result, *, reverse=False):
    for p in sorted(result.prefixes, reverse=reverse):
        output = result.prefixes[p]
        for key, law in output.guide_laws.items(): writer.emit(p, key, law)
        if output.pooled_control is not None: writer.emit(p, POOL, output.pooled_control)
        writer.write_catalog(p, **{field: getattr(output, field) for field in CATALOG_FIELDS})


def publish(path, spec, result, *, reverse=False):
    with ForecastWriter(path, spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
        emit(writer, result, reverse=reverse)
        return writer.finish(result.receipt)


def reader(path, spec, receipt, **kwargs):
    return ForecastReader(path, spec=spec, expected_complete_sha256=receipt["complete_sha256"],
        expected_manifest_sha256=receipt["manifest_sha256"], maximum_output_bytes=128*1024**2,
        maximum_working_bytes=128*1024**2, **kwargs)


@pytest.mark.parametrize("family", ["F0", "F2", "F5"])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_prefix_law_and_score_roundtrip(tmp_path, family, reverse):
    s, m, spec, result = prepare(family=family)
    path = tmp_path / "forecast"
    receipt = publish(path, spec, result, reverse=reverse)
    assert receipt["total_bytes"] == sum(p.stat().st_size for p in path.iterdir())
    assert {p.name for p in path.iterdir()} == {"raw_states.npy", "log_weights.npy", "log_mass.npy", "catalogs.npy", "MANIFEST.json", "COMPLETE.json"}
    assert np.load(path / "raw_states.npy", allow_pickle=False).shape == (5, 8, 48)
    with reader(path, spec, receipt) as loaded:
        for prefix, output in result.prefixes.items():
            for key in (*output.guide_laws, POOL):
                expected = output.pooled_control if key == POOL else output.guide_laws[key]
                actual = loaded.law(prefix, key)
                assert actual.log_mass_increment == expected.log_mass_increment
                for field in ("uncalibrated_states", "states", "log_weights"):
                    assert torch.equal(getattr(actual, field), getattr(expected, field))
                def score(law):
                    return score_state_law(law.states, law.log_weights.exp(), expected.states, expected.log_weights.exp(),
                                           None, None, include_source_diagnostics=False)
                assert score(actual) == score(expected)
            for field, value in loaded.catalog(prefix).items(): assert torch.equal(value, getattr(output, field))
            assert loaded.law(prefix, 4) is None and loaded.law(prefix, 5) is None
        assert all(not value.flags.writeable for value in loaded._maps.values())
    with pytest.raises(RuntimeError): loaded.law(5, 0)


def test_streaming_core_integration_and_no_whole_writable_mapping(tmp_path):
    s, m, spec, expected = prepare()
    recipe = Recipe(m.config.configuration_id, .25, 1.3)
    with patch("frontier.forecast_storage.np.memmap", side_effect=AssertionError("Writer must not map whole bank")):
        with ForecastWriter(tmp_path / "stream", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            output = forecast_population(m, s, binding(m, s), context(m, s), recipe, prefixes=(5, 8),
                limits=forecast_limits(maximum_output_bytes=128*1024**2), sink=writer)
            assert not hasattr(output, "prefixes")
            assert not (tmp_path / "stream" / "COMPLETE.json").exists()
            receipt = writer.finish(output.receipt)
    with reader(tmp_path / "stream", spec, receipt) as loaded:
        for p in spec.effective_prefixes:
            assert torch.equal(loaded.law(p, 0).states, expected.prefixes[p].guide_laws[0].states)
            assert torch.equal(loaded.catalog(p)["log_frequencies"], expected.prefixes[p].log_frequencies)


@pytest.mark.parametrize("all_missing", [False, True])
def test_absent_pool_and_missing_state_are_not_synthesized(tmp_path, all_missing):
    _, _, spec, result = prepare(missing_pool=True, all_missing=all_missing)
    receipt = publish(tmp_path / "missing", spec, result)
    with reader(tmp_path / "missing", spec, receipt) as loaded:
        assert loaded.law(5, POOL) is None
        assert loaded.law(5, 2) is None
        if all_missing:
            assert loaded._maps["raw_states"].shape == (0, 8, 48)
            assert loaded.law(8, 0) is None
            values = loaded.catalog(8)
            torch.testing.assert_close(values["log_frequencies"], values["log_source_frequencies"], atol=1e-15, rtol=0)
            assert torch.count_nonzero(values["log_mass_increments"]) == 0
        assert loaded.spec.document["pooled_control_unavailable_reason"] == "missing_positive_RNA_source_NTC_pool"


def test_caps_checked_before_directory_creation(tmp_path):
    _, _, spec, _ = prepare()
    estimate = storage_estimate(spec, limits())
    for bad in (replace(limits(), output_cap_bytes=estimate["output_upper_bound_bytes"]-1),
                replace(limits(), working_cap_bytes=estimate["maximum_working_bytes"]-1)):
        with pytest.raises(ValueError, match="cap"):
            ForecastWriter(tmp_path / "insufficient", spec=spec, producer_sha256="b"*64, limits=bad)
        assert not (tmp_path / "insufficient").exists()


@pytest.mark.parametrize("failure", ["duplicate", "unknown", "early_catalog", "invalid_weights", "wrong_calibration", "nonnested", "missing", "wrong_receipt", "no_final_checks", "partial_then_caught"])
def test_incomplete_or_invalid_emissions_never_complete(tmp_path, failure):
    _, _, spec, result = prepare()
    path = tmp_path / failure
    with pytest.raises((ValueError, RuntimeError)):
        with ForecastWriter(path, spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            law = result.prefixes[5].guide_laws[0]
            if failure == "duplicate": writer.emit(5, 0, law); writer.emit(5, 0, law)
            elif failure == "unknown": writer.emit(5, 4, law)
            elif failure == "early_catalog": writer.write_catalog(5, **{f: getattr(result.prefixes[5], f) for f in CATALOG_FIELDS})
            elif failure == "invalid_weights": writer.emit(5, 0, replace(law, log_weights=law.log_weights+1))
            elif failure == "wrong_calibration": writer.emit(5, 0, replace(law, states=law.states+1))
            elif failure == "nonnested":
                writer.emit(5, 0, law)
                long = result.prefixes[8].guide_laws[0]
                writer.emit(8, 0, replace(long, uncalibrated_states=long.uncalibrated_states+1, states=long.states+1))
            elif failure == "missing": writer.finish(result.receipt)
            elif failure == "partial_then_caught":
                try: writer.emit(5, 4, law)
                except ValueError: pass
                writer.emit(5, 0, law)
            else:
                emit(writer, result)
                receipt = dict(result.receipt)
                receipt["source_identity" if failure == "wrong_receipt" else "complete_prediction_checks_passed"] = "c"*64 if failure == "wrong_receipt" else False
                writer.finish(receipt)
    assert not (path / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError): ForecastWriter(path, spec=spec, producer_sha256="b"*64, limits=limits())


def test_catalog_cannot_drop_complement_or_change_calibration(tmp_path):
    _, _, spec, result = prepare()
    with pytest.raises(ValueError, match="catalog"):
        with ForecastWriter(tmp_path / "bad", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            for key, law in result.prefixes[5].guide_laws.items(): writer.emit(5, key, law)
            writer.emit(5, POOL, result.prefixes[5].pooled_control)
            values = {f: getattr(result.prefixes[5], f).clone() for f in CATALOG_FIELDS}
            values["log_mass_increments"][4] = .1
            writer.write_catalog(5, **values)


@pytest.mark.parametrize("file", ["raw_states.npy", "log_weights.npy", "log_mass.npy", "catalogs.npy"])
def test_writer_readback_detects_payload_tampering_before_completion(tmp_path, file):
    _, _, spec, result = prepare()
    with pytest.raises(ValueError):
        with ForecastWriter(tmp_path / "changed", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            emit(writer, result)
            with (tmp_path / "changed" / file).open("r+b") as stream:
                stream.seek(-1, 2); stream.write(b"\xff")
            writer.finish(result.receipt)
    assert not (tmp_path / "changed" / "COMPLETE.json").exists()


@pytest.mark.parametrize("failure", ["bytes", "symlink", "extra", "wrong_complete", "wrong_spec"])
def test_reader_rejects_changed_or_unbound_publication(tmp_path, failure):
    _, _, spec, result = prepare()
    path = tmp_path / "valid"
    receipt = publish(path, spec, result)
    if failure == "bytes":
        with (path / "raw_states.npy").open("r+b") as stream: stream.seek(-1, 2); stream.write(b"\xff")
    elif failure == "symlink":
        (path / "raw_states.npy").rename(tmp_path / "payload.npy")
        (path / "raw_states.npy").symlink_to(tmp_path / "payload.npy")
    elif failure == "extra": (path / "unexpected").touch()
    elif failure == "wrong_complete": receipt["complete_sha256"] = "f"*64
    else:
        _, _, spec, _ = prepare(alpha=.5)
    with pytest.raises((ValueError, OSError)): reader(path, spec, receipt)


def test_reader_lease_detects_postopen_change(tmp_path):
    _, _, spec, result = prepare()
    path = tmp_path / "lease"
    receipt = publish(path, spec, result)
    loaded = reader(path, spec, receipt)
    loaded.__enter__()
    with (path / "log_weights.npy").open("r+b") as stream: stream.seek(-1, 2); stream.write(b"\xff")
    with pytest.raises(ValueError): loaded.law(5, 0)
    with pytest.raises(ValueError): loaded.close()


def test_immutable_estimate_and_active_bindings(tmp_path):
    _, _, spec, result = prepare()
    with pytest.raises(ValueError, match="changed"):
        with ForecastWriter(tmp_path / "binding", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            with pytest.raises(TypeError): writer.estimate["maximum_working_bytes"] = 0
            writer.limits = replace(writer.limits, maximum_batch_laws=3)
            writer.assert_active()


def test_no_perlaw_json_reparse_and_batch_bound(tmp_path):
    _, _, spec, result = prepare()
    with ForecastWriter(tmp_path / "cached", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
        with patch.object(ForecastStorageSpec, "document", property(lambda self: (_ for _ in ()).throw(AssertionError("per-law metadata reparse")))):
            writer.assert_active()
            writer.emit(5, 0, result.prefixes[5].guide_laws[0])
        for p, output in result.prefixes.items():
            for key, law in output.guide_laws.items():
                if (p, key) != (5, 0): writer.emit(p, key, law)
            writer.emit(p, POOL, output.pooled_control)
            writer.write_catalog(p, **{f: getattr(output, f) for f in CATALOG_FIELDS})
        writer.finish(result.receipt)
    with pytest.raises(ValueError, match="bounded"):
        with ForecastWriter(tmp_path / "batch", spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            writer.emit_batch(5, result.prefixes[5].guide_laws)


def test_seed_specific_mixture_prefixes_preserved_for_later_ensemble(tmp_path):
    s = replace(source(), fold=FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4")))
    m = model("F2")
    b, c = replace(binding(m, s), accepted_updates=9216), replace(context(m, s), accepted_update=9216)
    recipe = Recipe("F2-V0", .25, 1.3)
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=(8, 16), ensemble_component_totals=(12,))
    result = forecast_population(m, s, b, c, recipe, prefixes=(8, 16), ensemble_component_totals=(12,), limits=forecast_limits())
    assert spec.effective_prefixes == (4, 8, 16)
    receipt = publish(tmp_path / "component", spec, result)
    with reader(tmp_path / "component", spec, receipt) as loaded:
        assert loaded.forecast_receipt == json.loads(json.dumps(result.receipt))
        for p in spec.effective_prefixes:
            assert torch.equal(loaded.law(p, 0).states, result.prefixes[p].guide_laws[0].states)
            assert torch.equal(loaded.law(p, 0).log_weights, result.prefixes[p].guide_laws[0].log_weights)
        # Corrected component proposals are prefix-specific, not a truncation
        # of the maximum prefix's normalized importance weights.
        assert not torch.equal(loaded.law(4, 0).log_weights, loaded.law(16, 0).log_weights[:4])


def test_reader_metadata_and_runtime_remain_bound(tmp_path):
    _, _, spec, result = prepare()
    receipt = publish(tmp_path / "runtime", spec, result)
    torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError, match="runtime"):
            reader(tmp_path / "runtime", spec, receipt)
    finally:
        torch.set_num_threads(2)
    with reader(tmp_path / "runtime", spec, receipt) as loaded:
        with patch.object(ForecastStorageSpec, "document", property(lambda self: (_ for _ in ()).throw(AssertionError("per-law metadata reparse")))):
            assert loaded.law(5, 0) is not None


def test_spec_requires_validated_factory_and_reader_cannot_rebind(tmp_path):
    _, _, spec, result = prepare()
    with pytest.raises(TypeError, match="factory"):
        ForecastStorageSpec(spec._encoded)
    receipt = publish(tmp_path / "spec", spec, result)
    loaded = reader(tmp_path / "spec", spec, receipt)
    loaded.__enter__()
    _, _, other, _ = prepare(alpha=.5)
    loaded.spec = other
    with pytest.raises(ValueError, match="binding changed"):
        loaded.law(5, 0)
    loaded.close(verify_hashes=False)


def test_root_replaced_during_hashing_cannot_publish_complete(tmp_path):
    import frontier.forecast_storage as storage
    _, _, spec, result = prepare()
    path, displaced = tmp_path / "root", tmp_path / "displaced"
    original = storage._digest_stream
    moved = False
    def replace_path(stream):
        nonlocal moved
        value = original(stream)
        if not moved:
            moved = True
            path.rename(displaced)
            path.mkdir()
        return value
    with pytest.raises(ValueError, match="root changed"):
        with ForecastWriter(path, spec=spec, producer_sha256="b"*64, limits=limits()) as writer:
            emit(writer, result)
            with patch.object(storage, "_digest_stream", side_effect=replace_path):
                writer.finish(result.receipt)
    assert not (path / "COMPLETE.json").exists()
    assert not (displaced / "COMPLETE.json").exists()
