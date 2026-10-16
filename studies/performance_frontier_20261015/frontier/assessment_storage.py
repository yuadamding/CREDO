"""Completion-last publication of streamed assessment rows and reductions.

No forecasts or truth are read here. A caller supplies the scorer's complete
result and exact live parent ForecastReaders. Their full hash-on-close checks
must succeed before COMPLETE is published. A caller-written verification flag
cannot substitute for these reader operations. Failed attempts are retained,
not overwritten, resumed or retried.

JSONL is reread one bounded row at a time. Reconciliation retains only the
summary's fixed support inventory and scalar reduction terms, not particle,
RNA or whole-row payloads. Python working memory is proportional to the
explicitly bounded summary/support inventory; output caps are not measured
RSS guarantees. The verifier performs arithmetic replay, never neural replay,
outcome reopening, selector decisions or scientific acceptance.
"""

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from .cache_storage import _open_verified, _parse, _path, _stamp, _write_once
from .assessment import scientific_recipe_identity
from .evaluation import ScoreIdentity
from .fitting_provider import _hash, _sha
from .forecast_storage import ForecastReader
from .selection import CHANNELS
from .selector_contract import Recipe


SCHEMA = "credo.frontier.assessment-storage-complete.v1"
FILES = {"SCORES.jsonl", "SUMMARY.json", "COMPLETE.json"}
SCORE_SOURCES = {name+".py" for name in ("assessment", "evaluation", "measure_ops", "molecular",
    "observation", "calibration_view", "ensemble_view", "assessment_support", "forecast_seal")}


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError("Explicit positive integer bound required: " + name)


def _row(raw, maximum):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum or not raw.endswith(b"\n"):
        raise ValueError("One bounded canonical JSONL byte row required")
    value = _parse(raw)
    if _encode(value) != raw or set(value) != {"recipe", "recipe_identity", "prefix", "row_id", "score"}:
        raise ValueError("Exact canonical assessment row wrapper required")
    recipe = Recipe(**value["recipe"])
    _sha(value["recipe_identity"])
    if asdict(recipe) != value["recipe"] or not isinstance(value["row_id"], str) or not value["row_id"]:
        raise ValueError("Exact recipe and nonempty row identity required")
    score = value["score"]
    if not isinstance(score, dict) or score.get("arithmetic_only") is not False:
        raise ValueError("Source-bound score row required")
    identity = ScoreIdentity(**score["identity"])
    if (value["prefix"] != identity.particle_prefix or type(value["prefix"]) is not int
            or value["recipe_identity"] != identity.recipe_id or score.get("dtype") != "float64"
            or identity.recipe_id != scientific_recipe_identity(recipe)):
        raise ValueError("Row wrapper and score identity differ")
    return value


def _parents(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Exact nonempty parent_publications inventory required")
    expected = {"root", "complete_sha256", "manifest_sha256", "spec_sha256", "total_bytes"}
    roots, identities = set(), set()
    for record in value:
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("Exact parent forecast publication reference required")
        # Validate lexical paths; verification need not reopen historical banks.
        if not isinstance(record["root"], str) or not record["root"].startswith("/") or ".." in record["root"].split("/"):
            raise ValueError("Absolute nontraversing parent root required")
        for key in ("complete_sha256", "manifest_sha256", "spec_sha256"):
            _sha(record[key])
        _positive(record["total_bytes"], "parent total bytes")
        identity = _hash(record)
        if record["root"] in roots or identity in identities:
            raise ValueError("Duplicate parent publication")
        roots.add(record["root"]); identities.add(identity)
    return value


def _summary(value):
    if not isinstance(value, dict) or set(value) != {"scores", "receipt"}:
        raise ValueError("Exact scorer result with scores and receipt required")
    receipt, scores = value["receipt"], value["scores"]
    schemas = {"single_seed": "credo.frontier.streamed-calibration-assessment.v2",
               "three_seed_finite_measure_mixture": "credo.frontier.streamed-ensemble-assessment.v1"}
    if (not isinstance(receipt, dict)
            or receipt.get("predictor_kind") not in schemas
            or receipt.get("schema") != schemas.get(receipt.get("predictor_kind"))
            or receipt.get("status") != "numerical_complete_not_published"
            or receipt.get("optimizer_calls") != 0 or receipt.get("neural_forecast_calls") != 0
            or not isinstance(scores, list) or not scores
            or type(receipt.get("recipe_count")) is not int or receipt["recipe_count"] != len(scores)):
        raise ValueError("Complete unchanged numerical assessment receipt required")
    _parents(receipt.get("parent_publications"))
    for key in ("rows_sha256", "forecast_seal_sha256", "support_sha256", "derivation_sha256"):
        _sha(receipt[key])
    for key in ("emitted_rows", "rows_bytes", "prefix"):
        _positive(receipt[key], key)
    prepared = {}
    for entry in scores:
        if not isinstance(entry, dict) or set(entry) != {"recipe", "scores"}:
            raise ValueError("Exact recipe reduction entry required")
        recipe = Recipe(**entry["recipe"])
        key = _hash(asdict(recipe))
        reduction = entry["scores"]
        if key in prepared or not isinstance(reduction, dict):
            raise ValueError("Duplicate or invalid reduced recipe")
        identity = ScoreIdentity(**reduction["identity"])
        if (identity.particle_prefix != receipt["prefix"] or reduction.get("arithmetic_only") is not False
                or reduction.get("dtype") != "float64" or set(reduction.get("coverage", {})) != set(CHANNELS)
                or identity.recipe_id != scientific_recipe_identity(recipe)
                or identity.predictor_kind != receipt["predictor_kind"]):
            raise ValueError("Complete source-bound nine-channel reduction required")
        weights = {}
        for channel in CHANNELS:
            coverage = reduction["coverage"][channel]
            declared = coverage.get("fixed_weights")
            if not isinstance(declared, dict) or any(not isinstance(k, str) or not k for k in declared):
                raise ValueError("Explicit fixed channel support required")
            if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
                   for w in declared.values()) or (declared and abs(math.fsum(declared.values())-1.) > 1e-10):
                raise ValueError("Normalized finite nonnegative fixed channel weights required")
            weights[channel] = declared
        if reduction.get("reduction_sha256") != _hash(weights):
            raise ValueError("Fixed reduction support digest differs")
        required = set().union(*(set(w) for w in weights.values()))
        prepared[key] = dict(recipe=asdict(recipe), identity=asdict(identity), reduction=reduction,
            weights=weights, required=required, seen=set(),
            terms={c: [] for c in CHANNELS}, available={c: [] for c in CHANNELS},
            missing={c: {} for c in CHANNELS})
    if len({x["identity"]["recipe_id"] for x in prepared.values()}) != len(prepared):
        raise ValueError("Duplicate recipe identity")
    derivations = receipt.get("prediction_derivations")
    if not isinstance(derivations, dict) or set(derivations) != {x["identity"]["recipe_id"] for x in prepared.values()}:
        raise ValueError("Exact per-recipe prediction derivation inventory required")
    for digest in derivations.values(): _sha(digest)
    for item in prepared.values():
        label = "ensemble_derivation_sha256" if receipt["predictor_kind"] != "single_seed" else "derivation_sha256"
        if derivations[item["identity"]["recipe_id"]] != _hash({label: receipt["derivation_sha256"], "recipe": item["recipe"]}):
            raise ValueError("Recipe-specific prediction derivation digest differs")
    sources = receipt.get("scoring_implementation_sources")
    if not isinstance(sources, dict) or set(sources) != SCORE_SOURCES:
        raise ValueError("Bound scoring implementation source inventory required")
    for name, digest in sources.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Only flat local scoring module identities are permitted")
        _sha(digest)
    return receipt, prepared


def _bind_readers(readers, receipt, prepared):
    parents = receipt["parent_publications"]
    ensemble = receipt["predictor_kind"] == "three_seed_finite_measure_mixture"
    if (type(readers) is not tuple or len(readers) != len(parents) or len(readers) != (3 if ensemble else 1)
            or any(type(reader) is not ForecastReader for reader in readers)
            or len({id(reader) for reader in readers}) != len(readers)):
        raise ValueError("Exact distinct live parent ForecastReader tuple required")
    docs = []
    for reader, parent in zip(readers, parents):
        reader._access()
        if dict(reader.publication_reference) != parent:
            raise ValueError("Reader differs from exact required parent publication")
        docs.append(reader.spec.document)
    bindings = [doc["model_binding"] for doc in docs]
    if ensemble and [b["model_seed"] for b in bindings] != [0, 1, 2]:
        raise ValueError("Ordered three-seed ensemble parents required")
    first, b = docs[0], bindings[0]
    common = ("source_identity", "catalog_sha256", "feature_axis_sha256")
    if any(any(doc[k] != first[k] for k in common) or doc["model_binding"]["fold"] != b["fold"]
           or doc["model_binding"]["observation_binding"] != b["observation_binding"]
           or doc["model_binding"]["numerical_source_sha256"] != b["numerical_source_sha256"] for doc in docs):
        raise ValueError("Parent model/source/representation bindings differ")
    mixture = dict(schema="credo.frontier.fixed-three-seed-model-identity.v1",
        components=[dict(seed=value["model_seed"], model_sha256=value["model_artifact_sha256"]) for value in bindings],
        rule="equal_mixture_of_calibrated_finite_measures") if ensemble else None
    if receipt.get("ensemble_model_identity") != mixture:
        raise ValueError("Ensemble model descriptor differs from actual parent inventory")
    expected = dict(fold_id=b["fold"]["fold_id"], query_donor=first["source_reference"]["donor"],
        seed=None if ensemble else b["model_seed"], source_sha256=b["numerical_source_sha256"],
        representation_sha256=b["observation_binding"]["artifact_sha256"],
        model_sha256=_hash(mixture) if ensemble else b["model_artifact_sha256"],
        catalog_sha256=first["catalog_sha256"], gene_axis_sha256=first["feature_axis_sha256"])
    if (receipt.get("early_source_identity") != first["source_identity"]
            or receipt.get("observation_binding") != b["observation_binding"]):
        raise ValueError("Summary population/observation differs from parent source")
    for item in prepared.values():
        if any(item["identity"][key] != value for key, value in expected.items()):
            raise ValueError("Saved score identity differs from actual parent model")
        if item["recipe"]["configuration_id"] != b["configuration_id"]:
            raise ValueError("Saved recipe differs from actual parent configuration")


def _source_stamps(receipt):
    """Check only the scorer's explicitly named local implementation files."""
    result = {}
    for name, expected in receipt["scoring_implementation_sources"].items():
        path = _path(Path(__file__).parent/name)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024**2:
            raise ValueError("Bounded regular scoring implementation required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected or _stamp(path.lstat()) != _stamp(before):
            raise ValueError("Scoring implementation differs during publication")
        result[str(path)] = _stamp(before)
    return result


def _reconcile(stream, result, *, maximum_row_bytes, maximum_rows_bytes):
    receipt, recipes = _summary(result)
    digest, size, count = hashlib.sha256(), 0, 0
    stream.seek(0)
    while raw := stream.readline(maximum_row_bytes+1):
        size += len(raw)
        if size > maximum_rows_bytes:
            raise ValueError("Saved row bytes exceed explicit bound")
        row = _row(raw, maximum_row_bytes)
        key = _hash(row["recipe"])
        if key not in recipes:
            raise ValueError("Foreign recipe in saved assessment rows")
        item, name, score = recipes[key], row["row_id"], row["score"]
        if (row["recipe"] != item["recipe"] or score["identity"] != item["identity"]
                or name not in item["required"] or name in item["seen"]):
            raise ValueError("Foreign, duplicate or unexpected fixed score row")
        item["seen"].add(name)
        for channel in CHANNELS:
            if name not in item["weights"][channel]:
                continue
            if channel not in score:
                raise ValueError("Required fixed channel absent from saved row")
            value, reason = score[channel], score.get("unavailable", {}).get(channel)
            if value is None:
                if not isinstance(reason, str) or not reason:
                    raise ValueError("Unavailable score requires an explicit reason")
                item["missing"][channel][name] = reason
            else:
                if (reason is not None or isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("Finite nonnegative required score expected")
                weight = item["weights"][channel][name]
                item["terms"][channel].append(weight*value)
                item["available"][channel].append(weight)
        digest.update(raw); count += 1
    if (size != receipt["rows_bytes"] or count != receipt["emitted_rows"]
            or digest.hexdigest() != receipt["rows_sha256"]):
        raise ValueError("Saved row count/bytes/hash differ from numerical receipt")
    for item in recipes.values():
        if item["seen"] != item["required"]:
            raise ValueError("Incomplete fixed recipe/row inventory")
        unavailable = {}
        for channel in CHANNELS:
            weights, missing = item["weights"][channel], item["missing"][channel]
            coverage = dict(expected_rows=len(weights), available_rows=len(item["available"][channel]),
                available_weight=math.fsum(item["available"][channel]), missing_rows=missing,
                fixed_weights=weights)
            if coverage != item["reduction"]["coverage"][channel]:
                raise ValueError("Saved channel coverage does not reproduce summary")
            absent = not weights or any(weights[name] > 0 for name in missing)
            risk = None if absent else math.fsum(item["terms"][channel])
            if item["reduction"].get(channel) != risk:
                raise ValueError("Saved nine-channel arithmetic does not reproduce summary")
            if absent:
                unavailable[channel] = "no_declared_support" if not weights else "incomplete_fixed_support"
        if (item["reduction"].get("unavailable") != unavailable
                or item["reduction"].get("selection_available") is not (not unavailable)):
            raise ValueError("Saved unavailable/selection status differs")
    stream.seek(0)
    return dict(rows=count, rows_bytes=size, rows_sha256=digest.hexdigest(), recipes=len(recipes),
                nine_channel_reductions_recomputed=True)


class AssessmentWriter:
    """An exclusive new directory and callable byte sink; no retry or overwrite."""

    def __init__(self, root, *, producer_sha256, output_cap_bytes, maximum_row_bytes, maximum_summary_bytes):
        _sha(producer_sha256)
        for value, name in ((output_cap_bytes, "output cap"), (maximum_row_bytes, "row cap"),
                            (maximum_summary_bytes, "summary cap")):
            _positive(value, name)
        if maximum_row_bytes > 1024**2 or maximum_row_bytes > output_cap_bytes or maximum_summary_bytes > output_cap_bytes:
            raise ValueError("Explicit row/summary limits must fit the total cap")
        self.root = _path(root)
        self.producer_sha256, self.output_cap_bytes = producer_sha256, output_cap_bytes
        self.maximum_row_bytes, self.maximum_summary_bytes = maximum_row_bytes, maximum_summary_bytes
        self._root_fd = self._stream = None
        self._entered = self._finished = self._failed = self._closed = self._finish_attempted = False
        self._count = self._bytes = 0
        self._digest = hashlib.sha256()
        if os.path.lexists(self.root):
            raise FileExistsError("Assessment destination exists; no overwrite or resume")
        parent_fd = os.open(_path(self.root.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(self.root.name, 0o700, dir_fd=parent_fd)
            self._root_fd = os.open(self.root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            self._root_id = (os.fstat(self._root_fd).st_dev, os.fstat(self._root_fd).st_ino)
            fd = os.open("SCORES.jsonl", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._root_fd)
            self._stream = os.fdopen(fd, "w+b")
            os.fsync(parent_fd)
        except BaseException:
            self.close()
            raise
        finally:
            os.close(parent_fd)

    def _live(self):
        if self._closed or not self._entered or self._failed or self._finished:
            raise RuntimeError("Active unfinished assessment writer required")
        info = _path(self.root).lstat()
        descriptor = os.fstat(self._root_fd)
        if ((info.st_dev, info.st_ino) != self._root_id or _stamp(info) != _stamp(descriptor)
                or not stat.S_ISDIR(info.st_mode)):
            raise ValueError("Assessment directory identity changed")

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("Assessment writer cannot be entered twice")
        self._entered = True
        self._live()
        return self

    def __call__(self, raw):
        self._live()
        if self._finish_attempted:
            raise RuntimeError("No rows after finalization begins")
        try:
            _row(raw, self.maximum_row_bytes)
            if self._bytes+len(raw) > self.output_cap_bytes:
                raise ValueError("Assessment row bytes exceed total output cap")
            if self._stream.write(raw) != len(raw):
                raise OSError("Incomplete assessment row write")
            self._digest.update(raw); self._bytes += len(raw); self._count += 1
        except BaseException:
            self._failed = True
            raise

    def finish(self, result, *, readers):
        self._live()
        if self._finish_attempted:
            raise RuntimeError("Assessment finalization cannot be retried")
        self._finish_attempted = True
        try:
            metadata = _encode(result)
            if len(metadata) > self.maximum_summary_bytes or self._bytes+len(metadata) > self.output_cap_bytes:
                raise ValueError("Assessment summary exceeds explicit output allowance")
            result = _parse(metadata)  # Snapshot the caller's result, do not retain mutable aliases.
            receipt, prepared = _summary(result)
            sources = _source_stamps(receipt)
            parents = receipt["parent_publications"]
            _bind_readers(readers, receipt, prepared)
            self._stream.flush(); os.fsync(self._stream.fileno())
            row_stamp = _stamp(os.fstat(self._stream.fileno()))
            if row_stamp != _stamp(os.stat("SCORES.jsonl", dir_fd=self._root_fd, follow_symlinks=False)):
                raise ValueError("Assessment row path changed")
            if (self._bytes != receipt["rows_bytes"] or self._count != receipt["emitted_rows"]
                    or self._digest.hexdigest() != receipt["rows_sha256"]):
                raise ValueError("Sink and numerical row receipts differ")
            arithmetic = _reconcile(self._stream, result, maximum_row_bytes=self.maximum_row_bytes,
                                    maximum_rows_bytes=self.output_cap_bytes)
            summary_sha = hashlib.sha256(metadata).hexdigest()
            summary_stamp = _write_once(self._root_fd, "SUMMARY.json", (metadata,),
                                       expected_sha256=summary_sha, expected_bytes=len(metadata))
            completion = dict(schema=SCHEMA, status="complete", producer_sha256=self.producer_sha256,
                files=[dict(relative_path="SCORES.jsonl", sha256=receipt["rows_sha256"], size_bytes=self._bytes),
                       dict(relative_path="SUMMARY.json", sha256=summary_sha, size_bytes=len(metadata))],
                output_cap_bytes=self.output_cap_bytes, maximum_row_bytes=self.maximum_row_bytes,
                maximum_summary_bytes=self.maximum_summary_bytes, total_bytes=0,
                arithmetic=arithmetic, parent_publications=parents,
                parent_exit_policy="all_exact_live_ForecastReaders_closed_with_full_hash_verification_before_COMPLETE",
                parent_exit_hash_bytes=sum(p["total_bytes"] for p in parents), scientific_launch_authority=False)
            while True:
                raw_complete = _encode(completion)
                total = self._bytes+len(metadata)+len(raw_complete)
                if total == completion["total_bytes"]: break
                completion["total_bytes"] = total
            if total > self.output_cap_bytes:
                raise ValueError("Assessment completion metadata exceeds total output cap")
            # No caller flag/callback stands in for these actual source exits.
            for reader in readers:
                reader.close(verify_hashes=True)
            self._live()
            if _source_stamps(receipt) != sources:
                raise ValueError("Scoring implementation changed before completion")
            stamps = {"SCORES.jsonl": row_stamp, "SUMMARY.json": summary_stamp}
            if set(os.listdir(self._root_fd)) != set(stamps):
                raise ValueError("Unexpected assessment files before completion")
            for name, stamp in stamps.items():
                if _stamp(os.stat(name, dir_fd=self._root_fd, follow_symlinks=False)) != stamp:
                    raise ValueError("Assessment output changed before completion")
            complete_sha = hashlib.sha256(raw_complete).hexdigest()
            pending_stamp = _write_once(self._root_fd, ".COMPLETE.pending", (raw_complete,),
                                       expected_sha256=complete_sha, expected_bytes=len(raw_complete))
            os.link(".COMPLETE.pending", "COMPLETE.json", src_dir_fd=self._root_fd,
                    dst_dir_fd=self._root_fd, follow_symlinks=False)
            os.unlink(".COMPLETE.pending", dir_fd=self._root_fd)
            os.fsync(self._root_fd)
            complete_stamp = _stamp(os.stat("COMPLETE.json", dir_fd=self._root_fd, follow_symlinks=False))
            # Creating/removing the temporary hard link changes ctime only.
            if complete_stamp[:4]+complete_stamp[5:] != pending_stamp[:4]+pending_stamp[5:]:
                raise ValueError("Assessment completion path was replaced")
            stamps["COMPLETE.json"] = complete_stamp
            def retain_writer_leases():
                self._live()
                if set(os.listdir(self._root_fd)) != FILES:
                    raise ValueError("Assessment completion inventory changed")
                for name, stamp in stamps.items():
                    if _stamp(os.stat(name, dir_fd=self._root_fd, follow_symlinks=False)) != stamp:
                        raise ValueError("Assessment output lease changed at completion")
            retain_writer_leases()
            verified = verify_assessment(self.root, expected_complete_sha256=complete_sha, maxbytes=self.output_cap_bytes)
            retain_writer_leases()
            self._finished = True
            return verified["publication"]
        except BaseException:
            self._failed = True
            raise

    def close(self):
        if self._closed: return
        if self._stream is not None: self._stream.close()
        if self._root_fd is not None: os.close(self._root_fd)
        self._closed = True

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None: self._failed = True
        unfinished = not self._finished and not self._failed
        self.close()
        if unfinished:
            raise RuntimeError("Assessment exited without completion; incomplete files retained")


def verify_assessment(root, *, expected_complete_sha256, maxbytes):
    """Verify exact files and replay scalar reductions, never reopen outcomes."""
    _sha(expected_complete_sha256); _positive(maxbytes, "verification byte cap")
    root = _path(root)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leases = []
    try:
        root_stamp = _stamp(os.fstat(fd))
        if _stamp(root.lstat()) != root_stamp or set(os.listdir(fd)) != FILES:
            raise ValueError("Exact assessment three-file inventory required")
        complete_file = _open_verified(fd, "COMPLETE.json", expected_complete_sha256, maximum=min(maxbytes, 1024**2))
        leases.append(complete_file)
        raw_complete = complete_file.stream.read()
        complete = _parse(raw_complete)
        expected_keys = {"schema", "status", "producer_sha256", "files", "output_cap_bytes", "maximum_row_bytes",
            "maximum_summary_bytes", "total_bytes", "arithmetic", "parent_publications", "parent_exit_policy",
            "parent_exit_hash_bytes", "scientific_launch_authority"}
        if (set(complete) != expected_keys or _encode(complete) != raw_complete
                or complete["schema"] != SCHEMA or complete["status"] != "complete"
                or complete["scientific_launch_authority"] is not False
                or complete["parent_exit_policy"] != "all_exact_live_ForecastReaders_closed_with_full_hash_verification_before_COMPLETE"):
            raise ValueError("Exact completed assessment publication required")
        _sha(complete["producer_sha256"])
        for key in ("total_bytes", "output_cap_bytes", "maximum_row_bytes", "maximum_summary_bytes"):
            _positive(complete[key], key)
        if (complete["total_bytes"] > min(maxbytes, complete["output_cap_bytes"])
                or complete["maximum_row_bytes"] > min(1024**2, complete["output_cap_bytes"])
                or complete["maximum_summary_bytes"] > complete["output_cap_bytes"]):
            raise ValueError("Assessment publication exceeds verification bounds")
        parents = _parents(complete["parent_publications"])
        if complete["parent_exit_hash_bytes"] != sum(p["total_bytes"] for p in parents):
            raise ValueError("Parent exit byte inventory differs")
        records = complete["files"]
        if not isinstance(records, list) or len(records) != 2:
            raise ValueError("Exact row and summary file inventory required")
        for record, name in zip(records, ("SCORES.jsonl", "SUMMARY.json")):
            if not isinstance(record, dict) or set(record) != {"relative_path", "sha256", "size_bytes"} or record["relative_path"] != name:
                raise ValueError("Unexpected assessment file/path")
            _positive(record["size_bytes"], name)
            maximum = min(maxbytes, complete["maximum_summary_bytes"]) if name == "SUMMARY.json" else maxbytes
            leases.append(_open_verified(fd, name, record["sha256"], size=record["size_bytes"], maximum=maximum))
        raw_summary = leases[2].stream.read()
        summary = _parse(raw_summary)
        if _encode(summary) != raw_summary:
            raise ValueError("Canonical assessment summary required")
        if summary["receipt"].get("parent_publications") != parents:
            raise ValueError("Summary and completion parent inventory differ")
        arithmetic = _reconcile(leases[1].stream, summary, maximum_row_bytes=complete["maximum_row_bytes"], maximum_rows_bytes=maxbytes)
        if arithmetic != complete["arithmetic"] or sum(lease.size for lease in leases) != complete["total_bytes"]:
            raise ValueError("Completion arithmetic or total byte inventory differs")
        if set(os.listdir(fd)) != FILES or _stamp(root.lstat()) != root_stamp or _stamp(os.fstat(fd)) != root_stamp:
            raise ValueError("Assessment directory changed during verification")
        for lease in leases:
            if (_stamp(os.fstat(lease.stream.fileno())) != lease.stamp
                    or _stamp(os.stat(lease.name, dir_fd=fd, follow_symlinks=False)) != lease.stamp):
                raise ValueError("Assessment file changed during verification")
        return dict(summary=summary, publication=dict(root=str(root), complete_sha256=expected_complete_sha256,
            summary_sha256=records[1]["sha256"], rows_sha256=records[0]["sha256"], total_bytes=complete["total_bytes"],
            parent_exit_hash_bytes=complete["parent_exit_hash_bytes"], arithmetic_replayed=True,
            scientific_launch_authority=False))
    finally:
        for lease in leases: lease.stream.close()
        os.close(fd)
