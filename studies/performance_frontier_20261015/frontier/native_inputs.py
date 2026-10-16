"""Fold-native bridge to the retained GSE314342 prepared CSR reader.

Index construction reads four metadata tables, completion and inventory only.
It never opens cell tables, count payloads or historical model weights. New
pair/triple capabilities retain the prepared package's *whole* source shards;
historical row-half partitions and R48 checkpoints are not silently reused.
The old task-L donor split semantics are preserved, but its old prediction
seal is not a seal for this study.

Runtime handles reuse the exact retained PreparedShardReader implementation.
They are application guards, not a filesystem sandbox: an eventual worker
owner must separately enforce outcome custody and verify published manifests.
Metadata qualification here is not remote input or H100 qualification.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import types
from types import MappingProxyType

import numpy as np
import pandas as pd


COMPLETION_SHA256 = "77d4fab700449ac323ff750cef6946a3185c06653891c41123fb4c76a4d5f057"
INVENTORY_SHA256 = "9b6a046958ee587ad600dd3cae0cf4aaa6f719b38c436b25a6c8c5159f7dcc63"
DATASET_ROOT = "/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1"
ADAPTER_SHA256 = "6c34a12e984f5d5028cab64e16a014bbf3ed046475f0a4ef8634ef9c07be3bd9"
READER_SOURCE_HASHES = MappingProxyType({
    "errors.py": "f7450783f31f429c68bb03c94fe65c476bbf459d7835c3eb829b310bdc191c53",
    "canonical.py": "835abb437c71b0859c7396f115f74a3296c77354b730f9d607953297909c6a8d",
    "contracts/models.py": "cb1aae49413c74de4851c24c489f10ef22dae196665326123a90b1dadff4d235",
    "data/prepared_shards.py": "a18dcd3ac918a434697cfaf4c5a0c82716d0e47b8f544139acdb4cc99f3180d8",
})
DONORS = ("D1", "D2", "D3", "D4")
CONDITIONS = ("Stim8hr", "Stim48hr")
TABLES = ("samples_and_culture_branches.parquet", "features_and_source_positions.parquet",
          "guides.parquet", "shard_manifest.parquet")
_SEAL_TOKEN = object()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Exact lowercase SHA-256 required")
    return value


def _uri(value):
    if (not isinstance(value, str) or not value or PurePosixPath(value).is_absolute()
            or "\\" in value or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Nonempty canonical relative member path required")
    return value


def _member(root, name):
    root = Path(root).absolute()
    path = root
    # Reject symlinks in the root's ancestors too, not just the final filename.
    for parent in (*reversed(root.parents), root):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Package root must be a regular nonsymlink directory path")
    for part in PurePosixPath(_uri(name)).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink package members are forbidden")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Regular package member required")
    return path


@contextmanager
def _verified(root, name, expected_sha, *, size=None, maximum_bytes=None):
    path = _member(root, name)
    before = path.stat()
    if size is not None and before.st_size != size:
        raise ValueError("Artifact size mismatch")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise ValueError("Artifact exceeds bounded read budget")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Artifact changed while opening")
        digest = hashlib.sha256()
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != _sha(expected_sha):
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
        handle.seek(0)
        yield handle
        after = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError("Artifact changed during consumption")


def _load_reader(source_root):
    """Load only four hash-bound source files; never run historical __init__."""
    namespace = "_credo_frontier_prepared_a18dcd3ac918"
    payloads = {}
    for name, digest in READER_SOURCE_HASHES.items():
        with _verified(source_root, name, digest, maximum_bytes=2 * 1024 * 1024) as handle:
            payloads[name] = handle.read()
    # A private namespace avoids importing an arbitrary installed credo version.
    # Execute the verified bytes, not a second path open after verification.
    if namespace + ".data.prepared_shards" not in sys.modules:
        for suffix in ("", ".contracts", ".data"):
            package = types.ModuleType(namespace + suffix)
            package.__path__ = []
            sys.modules[package.__name__] = package
        for name in READER_SOURCE_HASHES:
            module_name = namespace + "." + name[:-3].replace("/", ".")
            module = types.ModuleType(module_name)
            module.__file__ = str(Path(source_root) / name)
            module.__package__ = module_name.rsplit(".", 1)[0]
            sys.modules[module_name] = module
            exec(compile(payloads[name], module.__file__, "exec"), module.__dict__)
    return sys.modules[namespace + ".data.prepared_shards"]


@dataclass(frozen=True)
class PackageBinding:
    completion_sha256: str = COMPLETION_SHA256
    inventory_sha256: str = INVENTORY_SHA256
    rna_features: int = 18129
    guide_categories: int = 26504
    synthetic_fixture: bool = False

    def __post_init__(self):
        _sha(self.completion_sha256)
        _sha(self.inventory_sha256)
        if any(type(n) is not int or n < 1 for n in (self.rna_features, self.guide_categories)):
            raise ValueError("Positive feature/catalog dimensions required")
        if not self.synthetic_fixture and (self.completion_sha256, self.inventory_sha256,
                                          self.rna_features, self.guide_categories) != (
                COMPLETION_SHA256, INVENTORY_SHA256, 18129, 26504):
            raise ValueError("Changed real-data package binding is not admitted")


@dataclass(frozen=True)
class FoldInputs:
    fold_id: str
    training_donors: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not re.fullmatch("[A-Za-z0-9_.-]+", self.fold_id):
            raise ValueError("Explicit safe fold identity required")
        if (type(self.training_donors) is not tuple or len(self.training_donors) not in (2, 3)
                or tuple(sorted(set(self.training_donors))) != self.training_donors
                or not set(self.training_donors).issubset(DONORS)):
            raise ValueError("A sorted two- or three-donor fitting subset is required")

    @property
    def identity(self):
        return _hash({"fold_id": self.fold_id, "training_donors": self.training_donors,
                      "task": "Stim8hr_to_Stim48hr", "Rest": "excluded"})


@dataclass(frozen=True)
class ForecastExpectation:
    """Owner-frozen complete output inventory, not an outcome-release boolean.

    forecast_ids must cover the declared stage's models, seeds, prefixes and
    capabilities. This primitive verifies that supplied inventory, but cannot
    infer whether a study owner omitted a scientifically required forecast.
    """
    fold_sha256: str
    query_access_sha256: str
    model_sha256s: tuple[str, ...]
    configuration_sha256: str
    source_sha256: str
    representation_sha256: str
    forecast_ids: tuple[str, ...]

    def __post_init__(self):
        for value in (self.fold_sha256, self.query_access_sha256, self.configuration_sha256,
                      self.source_sha256, self.representation_sha256, *self.model_sha256s):
            _sha(value)
        if (type(self.model_sha256s) is not tuple or not self.model_sha256s
                or len(set(self.model_sha256s)) != len(self.model_sha256s)
                or type(self.forecast_ids) is not tuple or not self.forecast_ids
                or len(set(self.forecast_ids)) != len(self.forecast_ids)
                or any(not isinstance(item, str) or not item for item in self.forecast_ids)):
            raise ValueError("Complete unique model and forecast identities required")

    def record(self):
        return {name: list(value) if isinstance(value, tuple) else value
                for name, value in self.__dict__.items()}


@dataclass(frozen=True, init=False)
class VerifiedForecastRelease:
    expectation: ForecastExpectation
    manifest_sha256: str

    def __init__(self, *, token, expectation, manifest_sha256):
        if token is not _SEAL_TOKEN:
            raise ValueError("Use verified complete forecast publication, not a release flag")
        object.__setattr__(self, "expectation", expectation)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)

    @classmethod
    def verify(cls, root, manifest_name, *, manifest_sha256, expectation):
        if not isinstance(expectation, ForecastExpectation):
            raise TypeError("Full frozen forecast expectation required")
        with _verified(root, manifest_name, manifest_sha256, maximum_bytes=32 * 1024 * 1024) as stream:
            record = json.load(stream)
        if (set(record) != {"schema", "status", "bindings", "forecasts"}
                or record["schema"] != "credo.frontier.forecast-seal.v1"
                or record["status"] != "complete" or record["bindings"] != expectation.record()):
            raise ValueError("Forecast seal is incomplete or belongs to changed bindings")
        rows = record["forecasts"]
        if (not isinstance(rows, list) or len(rows) != len(expectation.forecast_ids)
                or tuple(row.get("forecast_id") for row in rows) != expectation.forecast_ids):
            raise ValueError("Forecast inventory differs from the frozen complete inventory")
        paths = set()
        for row in rows:
            if set(row) != {"forecast_id", "artifacts"} or not row["artifacts"]:
                raise ValueError("Each forecast requires its published artifacts")
            for artifact in row["artifacts"]:
                if set(artifact) != {"relative_uri", "sha256", "size_bytes"}:
                    raise ValueError("Exact forecast artifact identity required")
                name = _uri(artifact["relative_uri"])
                if name in paths or name == manifest_name or type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 1:
                    raise ValueError("Duplicate, empty or self-referential forecast artifact")
                paths.add(name)
                with _verified(root, name, artifact["sha256"], size=artifact["size_bytes"]):
                    pass
        return cls(token=_SEAL_TOKEN, expectation=expectation, manifest_sha256=manifest_sha256)


class NativeInputIndex:
    """Metadata-only index; real payload reads require an explicit runtime handle."""

    def __init__(self, metadata_root, reader_source_root, *, binding=PackageBinding()):
        if not isinstance(binding, PackageBinding):
            raise TypeError("Explicit package binding required")
        self.binding, self.legacy = binding, _load_reader(reader_source_root)
        with _verified(metadata_root, "INPUT_DATA_COMPLETE.json", binding.completion_sha256,
                       maximum_bytes=1024 * 1024) as stream:
            complete = json.load(stream)
        with _verified(metadata_root, "SHA256SUMS", binding.inventory_sha256,
                       maximum_bytes=16 * 1024 * 1024) as stream:
            lines = stream.read().decode().splitlines()
        inventory = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            if _uri(name) in inventory:
                raise ValueError("Duplicate package inventory member")
            inventory[name] = _sha(digest)
        if complete["inventory_sha256"] != binding.inventory_sha256 or complete["RNA_features"] != binding.rna_features:
            raise ValueError("Completion disagrees with package binding")
        self.inventory = MappingProxyType(inventory)
        tables, references = {}, {}
        for short in TABLES:
            name = "source/" + short
            with _verified(metadata_root, name, inventory[name], maximum_bytes=64 * 1024 * 1024) as stream:
                tables[short] = pd.read_parquet(stream)
                references[name] = self._artifact(name, os.fstat(stream.fileno()).st_size)
        samples = tables[TABLES[0]]
        if not samples.source_id.is_unique:
            raise ValueError("Duplicate source identity")
        roles = {}
        for donor in DONORS:
            for condition in CONDITIONS:
                rows = samples[samples.donor_id.eq(donor) & samples.condition.eq(condition)]
                if len(rows) != 1:
                    raise ValueError("Exactly one prepared source per donor/condition required")
                roles[str(rows.iloc[0].source_id)] = (donor, condition)
        self.source_roles = MappingProxyType(roles)
        features = tables[TABLES[1]]
        canonical = None
        for source in roles:
            rows = features[features.source_id.eq(source)].sort_values("canonical_position")
            if (not np.array_equal(rows.canonical_position, np.arange(len(rows)))
                    or not np.array_equal(np.sort(rows.source_position), np.arange(len(rows)))
                    or not rows.feature_id.is_unique or rows.is_RNA.dtype.kind != "b"):
                raise ValueError("Feature positions must be an exact unique bijection")
            current = rows[["feature_id", "is_RNA"]].to_dict("records")
            if canonical is not None and current != canonical:
                raise ValueError("Source feature axes disagree")
            canonical = current
        self.feature_order_sha256 = _hash(canonical)
        self.rna_positions = tuple(index for index, feature in enumerate(canonical) if feature["is_RNA"])
        self.rna_feature_ids = tuple(canonical[index]["feature_id"] for index in self.rna_positions)
        self.rna_axis_sha256 = _hash(list(self.rna_feature_ids))
        self.n_features = len(canonical)
        if (len(self.rna_positions) != binding.rna_features
                or self.n_features - len(self.rna_positions) != complete["technical_features"]):
            raise ValueError("RNA/technical feature inventory mismatch")
        guides = tables[TABLES[2]]
        if len(guides) != binding.guide_categories or not guides.experimental_library_member.eq(True).all():
            raise ValueError("Complete declared experimental guide catalog required")
        catalog = pd.DataFrame({"guide_index": np.arange(len(guides)), "guide_id": guides.sgRNA,
                                "target_id": guides.primary_target_id, "is_control": guides.is_design_ntc})
        guide_binding = self.legacy.GuideCatalogBinding(
            artifact=references["source/guides.parquet"], ordered_catalog_sha256=_hash(catalog.to_dict("records")),
            guide_id_column="sgRNA", target_id_column="primary_target_id", control_column="is_design_ntc", index_column=None)
        catalog = self.legacy.canonical_guide_catalog(guides, guide_binding)
        self.guide_binding = guide_binding
        self._catalog = tuple(tuple(row) for row in catalog.itertuples(index=False, name=None))
        shards = []
        for row in tables[TABLES[3]].to_dict("records"):
            if row["source_id"] not in roles:
                continue  # Never construct a Rest capability.
            if (not row["exact_reload"] or inventory[row["path"]] != row["sha256"]
                    or inventory[row["cells_path"]] != row["cells_sha256"]):
                raise ValueError("Shard differs from the prepared inventory")
            shards.append(self.legacy.PreparedShard(
                source_id=row["source_id"], shard=row["shard"], rows=row["rows"], nnz=row["nnz"],
                counts=self._artifact(row["path"], row["bytes"]),
                cells=self._artifact(row["cells_path"], row["cell_table_bytes"])))
        self.shards = tuple(sorted(shards, key=lambda row: (row.source_id, row.shard)))
        if (set(row.source_id for row in shards) != set(roles)
                or len({(row.source_id, row.shard) for row in shards}) != len(shards)
                or len({ref.relative_uri for row in shards for ref in (row.counts, row.cells)}) != 2 * len(shards)):
            raise ValueError("Missing, duplicate or aliased source shards")
        self.identity = _hash({"binding": binding.__dict__, "metadata": {name: ref.model_dump(mode="json")
                                                                       for name, ref in references.items()},
                               "reader": dict(READER_SOURCE_HASHES)})

    def _artifact(self, name, size):
        return self.legacy.ArtifactRef(schema_id="prepared-package-member", schema_version=1,
                                      sha256=self.inventory[_uri(name)], size_bytes=size,
                                      media_type="application/octet-stream", relative_uri=name)

    def catalog(self):
        return pd.DataFrame(self._catalog, columns=["guide_index", "guide_id", "target_id", "is_control"])

    def access(self, fold, role, *, query_donor=None, release=None):
        if not isinstance(fold, FoldInputs):
            raise TypeError("Exact fold input specification required")
        condition_map = {"representation_fit": CONDITIONS, "baseline_fit": CONDITIONS,
                         "dynamics_source": ("Stim8hr",), "dynamics_supervision": ("Stim48hr",),
                         "query": ("Stim8hr",), "evaluation_truth": ("Stim48hr",)}
        if role not in condition_map:
            raise ValueError("Unknown input role")
        is_query = role in ("query", "evaluation_truth")
        if is_query:
            if query_donor not in DONORS or query_donor in fold.training_donors:
                raise ValueError("Query donor must be excluded from fitting donors")
            donors = (query_donor,)
        else:
            if query_donor is not None or release is not None:
                raise ValueError("Fitting access cannot carry a query or release")
            donors = fold.training_donors
        if role == "evaluation_truth":
            if not isinstance(release, VerifiedForecastRelease):
                raise TypeError("Endpoint truth requires a verified complete forecast release")
            query = self.access(fold, "query", query_donor=query_donor)
            if (release.expectation.fold_sha256 != fold.identity
                    or release.expectation.query_access_sha256 != query.identity()):
                raise ValueError("Forecast publication belongs to a different fold/query")
        elif release is not None:
            raise ValueError("Only evaluation truth consumes a release")
        sources = tuple(sorted(source for source, (donor, condition) in self.source_roles.items()
                               if donor in donors and condition in condition_map[role]))
        common = dict(package_completion_sha256=self.binding.completion_sha256,
                      package_inventory_sha256=self.binding.inventory_sha256,
                      parent_view_sha256=self.identity, amendment_sha256=fold.identity,
                      feature_order_sha256=self.feature_order_sha256, guide_catalog=self.guide_binding,
                      n_features=self.n_features, task_id="frontier-L-40h", source_ids=sources,
                      shards=tuple(row for row in self.shards if row.source_id in sources))
        if role == "evaluation_truth":
            return self.legacy.PreparedEvaluationAccess(**common, prediction_seal_sha256=release.manifest_sha256)
        return self.legacy.PreparedAccess(**common, role=role)

    def open(self, dataset_root, fold, role, *, query_donor=None, release=None, **limits):
        """Explicit runtime boundary; construction never reads payloads implicitly."""
        access = self.access(fold, role, query_donor=query_donor, release=release)
        for name, digest in (("INPUT_DATA_COMPLETE.json", self.binding.completion_sha256),
                             ("SHA256SUMS", self.binding.inventory_sha256)):
            with _verified(dataset_root, name, digest, maximum_bytes=16 * 1024 * 1024):
                pass
        return NativeHandle(self, Path(dataset_root), access, limits)


@dataclass(frozen=True)
class NativeBatch:
    rna_counts: object
    cells: pd.DataFrame
    row_ids: tuple[str, ...]
    rna_depth: np.ndarray
    positive_rna: np.ndarray


class NativeHandle:
    def __init__(self, index, root, access, limits):
        self.index, self.root, self.access = index, root, access
        caps = {"max_uncompressed_bytes": 1024 ** 3, "max_cached_bytes": 768 * 1024 ** 2,
                "max_output_bytes": 512 * 1024 ** 2, "max_batch_rows": 8192}
        if any(name not in caps or type(value) is not int or not 0 < value <= caps[name]
               for name, value in limits.items()):
            raise ValueError("Reader limits exceed admitted bounded sparse-working-set caps")
        self._reader = index.legacy.PreparedShardReader(root, access, **limits)
        if not self._reader.guide_catalog().equals(index.catalog()):
            raise ValueError("Runtime guide catalog differs from the frozen metadata catalog")
        self._publication_identity_sha256 = None
        self._publication_pid = None

    def _publication_members(self):
        """Exact admitted input paths; never enumerate another donor's payloads."""
        members = {"INPUT_DATA_COMPLETE.json": (self.index.binding.completion_sha256, None),
                   "SHA256SUMS": (self.index.binding.inventory_sha256, None)}
        for short in TABLES:
            name = "source/" + short
            members[name] = (self.index.inventory[name], None)
        for shard in self.access.shards:
            for reference in (shard.counts, shard.cells):
                members[reference.relative_uri] = (reference.sha256, reference.size_bytes)
        for source in self.access.source_ids:
            name = f"source/{source}/population_accounting.parquet"
            members[name] = (self.index.inventory[name], None)
        return members

    @staticmethod
    def _file_identity(path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Input publication member is not a regular file")
        return {name: int(getattr(info, name)) for name in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_gid", "st_mode", "st_nlink")}

    def capture_publication_identity(self):
        """Metadata-only, same-worker input lease, complementary to read hashes.

        The retained reader verifies each consumed payload's SHA-256. This
        snapshot additionally detects later changes before final publication,
        without a second full count-array scan. It does not assert that every
        member has been consumed, nor compare GPFS device numbers across nodes.
        """
        if self._publication_identity_sha256 is not None:
            raise ValueError("Input publication identity is captured once per worker handle")
        members = {}
        small = {"INPUT_DATA_COMPLETE.json", "SHA256SUMS", *("source/" + name for name in TABLES)}
        for name, (digest, size) in sorted(self._publication_members().items()):
            path = _member(self.root, name)
            identity = self._file_identity(path)
            if size is not None and identity["st_size"] != size:
                raise ValueError("Input publication member size differs from admitted source")
            if name in small:
                with _verified(self.root, name, digest, maximum_bytes=64 * 1024 ** 2):
                    pass
                if self._file_identity(path) != identity:
                    raise ValueError("Input index changed during publication capture")
            members[name] = {"sha256": digest, "file_identity": identity}
        snapshot = {"schema": "credo.frontier.native-input-lease.v1", "dataset_root": str(self.root),
                    "access_sha256": self.access.identity(), "metadata_index_sha256": self.index.identity,
                    "source_ids": list(self.access.source_ids), "members": members,
                    "payload_hashes_verified_on_consumption": True, "all_payloads_consumed_claimed": False}
        self._publication_identity_sha256, self._publication_pid = _hash(snapshot), os.getpid()
        return snapshot

    def verify_publication_identity(self, snapshot):
        """Check exact entered identities before accepting this worker's output."""
        if (self._publication_pid != os.getpid() or self._publication_identity_sha256 is None
                or _hash(snapshot) != self._publication_identity_sha256):
            raise ValueError("Exact same-worker captured input publication identity required")
        expected = self._publication_members()
        if set(snapshot["members"]) != set(expected):
            raise ValueError("Input lease member inventory differs from admitted role")
        for name, record in snapshot["members"].items():
            if (record["sha256"] != expected[name][0]
                    or self._file_identity(_member(self.root, name)) != record["file_identity"]):
                raise ValueError(f"Input publication changed before output acceptance: {name}")
        return {"schema": "credo.frontier.native-input-lease-exit.v1", "status": "unchanged",
                "input_lease_sha256": self._publication_identity_sha256,
                "access_sha256": self.access.identity(), "checked_members": len(expected),
                "additional_count_array_scans": 0, "same_worker_identity_check": True}

    def _metadata(self, cells):
        catalog = self.index.catalog()
        required = {"source_id", "row_in_shard", "cell_id", "guide_index", "donor_id", "condition", "RNA_UMIs"}
        if not required.issubset(cells.columns):
            raise ValueError("Cell table lacks exact row/guide/donor/condition/RNA support identity")
        indices = cells.guide_index.to_numpy()
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(catalog)):
            raise ValueError("Cell guide index outside complete catalog")
        for column, canonical in (("guide_id", "guide_id"), ("primary_target_id", "target_id"), ("is_control", "is_control")):
            if column in cells and not np.array_equal(cells[column], catalog[canonical].to_numpy()[indices]):
                raise ValueError("Cell guide identity differs from complete catalog")
        for source, block in cells.groupby("source_id", sort=False):
            if source not in self.access.source_ids:
                raise ValueError("Cell source outside admitted role")
            donor, condition = self.index.source_roles[source]
            if not block.donor_id.eq(donor).all() or not block.condition.eq(condition).all():
                raise ValueError("Cell donor/condition differs from source binding")
        if cells.RNA_UMIs.dtype.kind not in "iu" or (cells.RNA_UMIs < 0).any():
            raise ValueError("Invalid positive-RNA support metadata")

    def _batch(self, raw):
        if not len(raw.cells):
            return NativeBatch(raw.matrix[:, self.index.rna_positions], raw.cells, (), np.array([], dtype=np.uint64), np.array([], dtype=bool))
        self._metadata(raw.cells)
        rna = raw.matrix[:, self.index.rna_positions].tocsr()
        depth = np.asarray(rna.sum(axis=1, dtype=np.uint64)).ravel()
        if not np.array_equal(depth, raw.cells.RNA_UMIs):
            raise ValueError("Count payload disagrees with RNA support metadata")
        ids = tuple(json.dumps([source, cell], separators=(",", ":"))
                    for source, cell in zip(raw.cells.source_id, raw.cells.cell_id))
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source-scoped cell identity in minibatch")
        return NativeBatch(rna, raw.cells.copy(), ids, depth, depth > 0)

    def read_rows(self, addresses):
        """Addresses are (source_id, shard, row); authorization precedes payload I/O."""
        rows = [self.index.legacy.RowAddress(source_id=source, shard=shard, row=row)
                for source, shard, row in addresses]
        return self._batch(self._reader.read_rows(rows))

    def iterate_rows(self, **kwargs):
        for cursor, raw in self._reader.iterate_rows(**kwargs):
            yield cursor, self._batch(raw)

    def captured_counts(self, source_id):
        """Raw eligible capture denominator, not positive-RNA law weights."""
        if source_id not in self.access.source_ids:
            raise ValueError("Abundance source outside admitted role")
        name = f"source/{source_id}/population_accounting.parquet"
        with _verified(self.root, name, self.index.inventory[name], maximum_bytes=64 * 1024 * 1024) as stream:
            block = pd.read_parquet(stream).sort_values("guide_index").reset_index(drop=True)
        catalog = self.index.catalog()
        total = sum(row.rows for row in self.access.shards if row.source_id == source_id)
        if (len(block) != len(catalog) or not np.array_equal(block.guide_index, np.arange(len(catalog)))
                or not np.array_equal(block.guide_id, catalog.guide_id)
                or not block.source_id.eq(source_id).all() or block.observed.dtype.kind != "b" or not block.observed.all()
                or block.n_cells.dtype.kind not in "iu" or (block.n_cells < 0).any()
                or int(block.n_cells.sum()) != total or not block.captured_cell_denominator.eq(total).all()
                or not block.guide_universe_size.eq(len(catalog)).all()):
            raise ValueError("Incomplete or inconsistent complete-catalog captured denominator")
        expected = (block.n_cells.to_numpy(dtype=np.float64) + 0.5) / (total + 0.5 * len(catalog))
        if not np.allclose(block.jeffreys_frequency, expected, rtol=1e-13, atol=0):
            raise ValueError("Regularized frequency does not use complete catalog")
        return block.copy()

    def support_counts(self, source_id):
        """Metadata-only state support; never substitute this for captured counts."""
        if source_id not in self.access.source_ids:
            raise ValueError("Support source outside admitted role")
        result = self.index.catalog()
        raw = np.zeros(len(result), dtype=np.int64)
        positive = raw.copy()
        for shard in self.access.shards:
            if shard.source_id != source_id:
                continue
            cells = self._reader.metadata(source_id, shard.shard)
            self._metadata(cells)
            guides = cells.guide_index.to_numpy()
            raw += np.bincount(guides, minlength=len(result))
            positive += np.bincount(guides[cells.RNA_UMIs.to_numpy() > 0], minlength=len(result))
        result["captured_cells"] = raw
        result["positive_RNA_cells"] = positive
        result["source_id"] = source_id
        return result
