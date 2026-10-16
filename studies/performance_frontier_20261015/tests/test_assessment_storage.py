"""Fabricated local publication/replay only; zero optimizer or model calls."""

from copy import deepcopy
import hashlib
import json
import os

import pytest

from frontier import assessment_storage as module
from frontier.assessment_storage import AssessmentWriter, verify_assessment
from frontier.forecast_storage import ForecastReader
from test_assessment import bounded_no_optimization, fixture, make_bank, open_view, score


MIB = 1024**2


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def writer(root, **changes):
    return AssessmentWriter(root, **(dict(producer_sha256="5"*64, output_cap_bytes=8*MIB,
        maximum_row_bytes=65536, maximum_summary_bytes=4*MIB) | changes))


def publish(fixture, root):
    with writer(root) as sink:
        result, _ = score(fixture, row_sink=sink)
        published = sink.finish(result, readers=(fixture.reader,))
    return result, published


def rehash_publication(root, *, transform_summary=None, transform_rows=None):
    """Deliberately malformed but rehashed fabricated records, not real output."""
    complete = json.loads((root/"COMPLETE.json").read_bytes())
    summary = json.loads((root/"SUMMARY.json").read_bytes())
    if transform_rows:
        rows = [json.loads(raw) for raw in (root/"SCORES.jsonl").read_bytes().splitlines()]
        transform_rows(rows)
        raw = b"".join(encoded(row) for row in rows)
        (root/"SCORES.jsonl").write_bytes(raw)
        summary["receipt"].update(rows_sha256=hashlib.sha256(raw).hexdigest(), rows_bytes=len(raw), emitted_rows=len(rows))
    if transform_summary: transform_summary(summary)
    (root/"SUMMARY.json").write_bytes(encoded(summary))
    for record in complete["files"]:
        raw = (root/record["relative_path"]).read_bytes()
        record.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    while True:
        raw = encoded(complete)
        total = len(raw)+sum(record["size_bytes"] for record in complete["files"])
        if complete["total_bytes"] == total: break
        complete["total_bytes"] = total
    (root/"COMPLETE.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_roundtrip_replays_all_reductions_and_closes_parent_before_complete(fixture, tmp_path, monkeypatch):
    root = tmp_path/"assessment"
    closed = []
    original = ForecastReader.close
    def close(self, *, verify_hashes=True):
        if self is fixture.reader and not self._closed:
            assert not (root/"COMPLETE.json").exists()
            assert verify_hashes is True
            closed.append(dict(self.publication_reference))
        return original(self, verify_hashes=verify_hashes)
    monkeypatch.setattr(ForecastReader, "close", close)
    result, publication = publish(fixture, root)
    assert closed == result["receipt"]["parent_publications"]
    assert fixture.reader._closed and not fixture.reader._entered
    assert {path.name for path in root.iterdir()} == module.FILES
    assert publication["total_bytes"] == sum(path.stat().st_size for path in root.iterdir())
    assert publication["parent_exit_hash_bytes"] == closed[0]["total_bytes"]
    monkeypatch.setattr(fixture.observation, "weighted_composition", lambda *_a, **_k: pytest.fail("No decoding in replay"))
    monkeypatch.setattr(ForecastReader, "__init__", lambda *_a, **_k: pytest.fail("No forecast reopening in replay"))
    replay = verify_assessment(root, expected_complete_sha256=publication["complete_sha256"], maxbytes=8*MIB)
    assert replay["summary"] == result
    assert replay["publication"] == publication
    assert publication["arithmetic_replayed"] is True
    assert publication["scientific_launch_authority"] is False


def test_exclusive_destination_and_no_incomplete_retry(fixture, tmp_path):
    root = tmp_path/"assessment"
    with pytest.raises(RuntimeError, match="without completion"):
        with writer(root): pass
    assert (root/"SCORES.jsonl").is_file()
    assert not (root/"COMPLETE.json").exists()
    with pytest.raises(FileExistsError): writer(root)


@pytest.mark.parametrize("change", ["traversal", "symlink"])
def test_unsafe_destination_rejected_without_writes(tmp_path, change):
    if change == "traversal":
        root = tmp_path/"nested"/".."/"bad"
    else:
        alias = tmp_path/"alias"; alias.symlink_to(tmp_path, target_is_directory=True)
        root = alias/"bad"
    with pytest.raises(ValueError): writer(root)
    assert not (tmp_path/"bad").exists()


@pytest.mark.parametrize("change", ["row-count", "row-hash", "summary-risk", "identity", "coverage", "derivation"])
def test_finish_reconciles_exact_result_and_saved_rows(fixture, tmp_path, change):
    result, rows = score(fixture)
    result = deepcopy(result)
    if change == "row-count": result["receipt"]["emitted_rows"] += 1
    elif change == "row-hash": result["receipt"]["rows_sha256"] = "0"*64
    elif change == "summary-risk": result["scores"][0]["scores"]["count_quadratic"] += .1
    elif change == "identity": result["scores"][0]["scores"]["identity"]["model_sha256"] = "0"*64
    elif change == "coverage": result["scores"][0]["scores"]["coverage"]["state_energy"]["available_weight"] = .5
    else:
        key = next(iter(result["receipt"]["prediction_derivations"]))
        result["receipt"]["prediction_derivations"][key] = "0"*64
    root = tmp_path/"assessment"
    with pytest.raises(ValueError):
        with writer(root) as sink:
            for row in rows: sink(row)
            sink.finish(result, readers=(fixture.reader,))
    assert not (root/"COMPLETE.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign"])
def test_complete_recipe_row_inventory_required(fixture, tmp_path, change):
    result, rows = score(fixture)
    if change == "missing": rows = rows[:-1]
    elif change == "duplicate": rows = rows + rows[:1]
    else:
        value = json.loads(rows[0]); value["row_id"] = "foreign-row"
        rows = [encoded(value)]+rows[1:]
    # Supply a matching stream receipt so inventory, rather than only hashing,
    # must detect the defect.
    result["receipt"].update(emitted_rows=len(rows), rows_bytes=sum(map(len, rows)),
        rows_sha256=hashlib.sha256(b"".join(rows)).hexdigest())
    root = tmp_path/"assessment"
    with pytest.raises(ValueError):
        with writer(root) as sink:
            for row in rows: sink(row)
            sink.finish(result, readers=(fixture.reader,))
    assert not (root/"COMPLETE.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "closed", "foreign"])
def test_only_exact_live_parent_readers_can_complete(fixture, tmp_path, change):
    result, rows = score(fixture)
    parents = (fixture.reader,)
    if change == "missing": parents = ()
    elif change == "duplicate": parents = (fixture.reader, fixture.reader)
    elif change == "closed": fixture.reader.close()
    else:
        entry = make_bank(fixture.bank_root/"different", fixture.source, offset=.5)
        other, _ = open_view(fixture.stack, fixture.bank_root, entry)
        parents = (other,)
    root = tmp_path/"assessment"
    with pytest.raises((ValueError, RuntimeError)):
        with writer(root) as sink:
            for row in rows: sink(row)
            sink.finish(result, readers=parents)
    assert not (root/"COMPLETE.json").exists()


def test_parent_exit_full_hash_failure_prevents_complete(fixture, tmp_path, monkeypatch):
    original = ForecastReader._lease_check
    checked = []
    def fail_hash(self, full_hash=False):
        if self is fixture.reader and full_hash:
            checked.append(True)
            raise ValueError("fabricated parent hash mismatch")
        return original(self, full_hash)
    root = tmp_path/"assessment"
    with writer(root) as sink:
        result, _ = score(fixture, row_sink=sink)
        monkeypatch.setattr(ForecastReader, "_lease_check", fail_hash)
        with pytest.raises(ValueError, match="parent hash mismatch"):
            sink.finish(result, readers=(fixture.reader,))
        with pytest.raises(RuntimeError): sink.finish(result, readers=(fixture.reader,))
    assert checked == [True]
    assert fixture.reader._closed
    assert (root/"SUMMARY.json").exists() and not (root/"COMPLETE.json").exists()


def test_real_parent_file_mutation_prevents_complete(fixture, tmp_path):
    result, rows = score(fixture)
    path = fixture.reader.root/"log_mass.npy"
    content = path.read_bytes(); path.write_bytes(content[:-1]+bytes([content[-1]^1]))
    root = tmp_path/"assessment"
    try:
        with pytest.raises(ValueError, match="Forecast file changed"):
            with writer(root) as sink:
                for row in rows: sink(row)
                sink.finish(result, readers=(fixture.reader,))
    finally:
        fixture.reader.close(verify_hashes=False)
    assert not (root/"COMPLETE.json").exists()


@pytest.mark.parametrize("change", ["row", "summary", "total"])
def test_explicit_caps_leave_incomplete_outputs(fixture, tmp_path, change):
    result, rows = score(fixture)
    cap = dict(maximum_row_bytes=1) if change == "row" else dict(maximum_summary_bytes=1)
    if change == "total":
        total = sum(map(len, rows))+len(encoded(result))
        cap = dict(output_cap_bytes=total, maximum_summary_bytes=len(encoded(result)), maximum_row_bytes=65536)
    root = tmp_path/"assessment"
    with pytest.raises(ValueError):
        with writer(root, **cap) as sink:
            for row in rows: sink(row)
            sink.finish(result, readers=(fixture.reader,))
    assert not (root/"COMPLETE.json").exists()


def test_noncanonical_or_multiline_row_is_terminal(fixture, tmp_path):
    result, rows = score(fixture)
    root = tmp_path/"assessment"
    with writer(root) as sink:
        with pytest.raises(ValueError): sink(rows[0]+rows[1])
        with pytest.raises(RuntimeError): sink(rows[0])
        with pytest.raises(RuntimeError): sink.finish(result, readers=(fixture.reader,))
    assert not (root/"COMPLETE.json").exists()


@pytest.mark.parametrize("change", ["summary-risk", "row-risk", "missing-row", "extra-file", "summary-symlink", "hash", "cap"])
def test_readback_checks_bytes_and_recomputes_even_rehashed_arithmetic(fixture, tmp_path, change):
    root = tmp_path/"assessment"
    _, published = publish(fixture, root)
    expected = published["complete_sha256"]
    if change == "summary-risk":
        def edit(result): result["scores"][0]["scores"]["count_quadratic"] += .1
        expected = rehash_publication(root, transform_summary=edit)
    elif change == "row-risk":
        def edit(rows): rows[0]["score"]["count_quadratic"] += .1
        expected = rehash_publication(root, transform_rows=edit)
    elif change == "missing-row": expected = rehash_publication(root, transform_rows=lambda rows: rows.pop())
    elif change == "extra-file": (root/"extra.json").write_bytes(b"{}\n")
    elif change == "summary-symlink":
        path = root/"SUMMARY.json"
        retained = tmp_path/"retained-summary.json"; retained.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(retained)
    elif change == "hash": expected = "0"*64
    with pytest.raises(ValueError):
        verify_assessment(root, expected_complete_sha256=expected,
                          maxbytes=published["total_bytes"]-1 if change == "cap" else 8*MIB)


def test_saved_row_path_replacement_before_finish_is_rejected(fixture, tmp_path):
    root = tmp_path/"assessment"
    with writer(root) as sink:
        result, _ = score(fixture, row_sink=sink)
        sink._stream.flush()
        replacement = tmp_path/"replacement.jsonl"
        replacement.write_bytes((root/"SCORES.jsonl").read_bytes())
        os.replace(replacement, root/"SCORES.jsonl")
        with pytest.raises(ValueError, match="row path changed"):
            sink.finish(result, readers=(fixture.reader,))
    assert not (root/"COMPLETE.json").exists()


@pytest.mark.parametrize("moment", ["after_link", "after_verification"])
@pytest.mark.parametrize("change", ["rows", "summary", "root"])
def test_writer_retains_original_output_leases_through_return(fixture, tmp_path, monkeypatch, moment, change):
    root = tmp_path/"assessment"
    def replace_identically():
        if change == "root":
            original = tmp_path/"original-assessment"
            root.rename(original)
            root.mkdir()
            for path in original.iterdir(): (root/path.name).write_bytes(path.read_bytes())
        else:
            path = root/("SCORES.jsonl" if change == "rows" else "SUMMARY.json")
            replacement = tmp_path/"replacement-file"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
    if moment == "after_link":
        original = os.link
        def link(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[:2] == (".COMPLETE.pending", "COMPLETE.json"):
                replace_identically()
            return value
        monkeypatch.setattr(module.os, "link", link)
    else:
        original = module.verify_assessment
        def verify(*args, **kwargs):
            value = original(*args, **kwargs)
            replace_identically()
            return value
        monkeypatch.setattr(module, "verify_assessment", verify)
    with pytest.raises(ValueError, match="(directory identity|output lease)"):
        publish(fixture, root)


def test_three_seed_ensemble_roundtrip_closes_all_exact_parents(tmp_path):
    from frontier.assessment import score_ensemble_bank
    from test_assessment import limits
    from test_ensemble_assessment import package
    with package(tmp_path) as p:
        root = tmp_path/"ensemble-assessment"
        with writer(root) as sink:
            result = score_ensemble_bank(p.ensemble, plan=p.plan, source=p.source, truth=p.truth,
                observation=p.observation, total=13, limits=limits(), row_sink=sink)
            publication = sink.finish(result, readers=p.readers)
        assert all(reader._closed for reader in p.readers)
        replay = verify_assessment(root, expected_complete_sha256=publication["complete_sha256"], maxbytes=8*MIB)
        assert replay["summary"] == result
        assert result["scores"][0]["scores"]["identity"]["seed"] is None
        assert result["receipt"]["predictor_kind"] == "three_seed_finite_measure_mixture"
        assert publication["parent_exit_hash_bytes"] == sum(r["total_bytes"] for r in result["receipt"]["parent_publications"])
