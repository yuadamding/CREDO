"""Exclusive seal for one owner-frozen fold/query batch of packed forecasts.

One forecast ID denotes one complete packed bank, including its explicit prefix
and capability inventory. The same bank is not repeated once per prefix: the
native release schema deliberately forbids duplicate artifact paths. All six
files (completion, manifest and four arrays) are referenced, never copied.
One additional small plan artifact preserves the readable configuration and
optional calibration/ensemble-derivation inventory, referenced once in the
first row. An ensemble binds its three existing calibrated parents; it never
adds duplicate bank paths or a newly materialized particle bank.

The caller freezes the complete required bank/spec inventory independently of
which banks succeeded, then supplies their exact admitted publication receipts.
This module cannot infer that a controller omitted a required model or prove
the timing of that freeze. It seals ONE fold/query batch, not the whole study.
The configuration binding is an ordered-entry configuration/recipe/prefix/context
digest, not a single-configuration hash; representation is the selected
observation artifact hash. File custody is not a claim of predictive validity.

No model, observation decoder, raw input handle, endpoint or optimizer is used.
Existing ForecastReader leases must remain open. Verification streams existing
bank bytes via the native release verifier; it does not copy a full bank into
RAM. An incomplete exclusive attempt is retained, with no retry or overwrite.
"""

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat

from .cache_compiler import native_source_input_identity
from .cache_storage import _json_bytes, _parse, _path, _stamp, _write_once
from .configuration import get_configuration
from .fitting_provider import _hash, _sha
from .forecast_storage import ARRAY_NAMES, ForecastReader, ForecastStorageSpec
from .native_inputs import (FoldInputs, ForecastExpectation, NativeInputIndex,
                            VerifiedForecastRelease, _uri, _verified)


SCHEMA = "credo.frontier.forecast-seal.v1"


@dataclass(frozen=True)
class ForecastSealEntry:
    forecast_id: str
    bank_relative_root: str
    spec: ForecastStorageSpec
    expected_complete_sha256: str
    expected_manifest_sha256: str
    derivation_manifest: bytes | None = None

    def __post_init__(self):
        if not isinstance(self.forecast_id, str) or not self.forecast_id:
            raise ValueError("Explicit complete-bank forecast identity required")
        _uri(self.bank_relative_root)
        if not isinstance(self.spec, ForecastStorageSpec):
            raise TypeError("Exact owner-frozen forecast storage specification required")
        self.spec.__post_init__()
        _sha(self.expected_complete_sha256); _sha(self.expected_manifest_sha256)
        if self.derivation_manifest is not None:
            if (type(self.derivation_manifest) is not bytes
                    or _json_bytes(_parse(self.derivation_manifest)) != self.derivation_manifest):
                raise ValueError("Immutable canonical calibration derivation bytes required")
            record = _parse(self.derivation_manifest)
            if (record.get("schema") != "credo.frontier.calibration-derivation.v1"
                    or record.get("status") != "source_only_inventory_bound"
                    or not isinstance(record.get("parent_publication"), dict)
                    or any(record["parent_publication"].get(k) != v for k, v in {
                        "complete_sha256": self.expected_complete_sha256,
                        "manifest_sha256": self.expected_manifest_sha256,
                        "spec_sha256": self.spec.identity}.items())):
                raise ValueError("Derivation must bind this exact admitted parent bank")


@dataclass(frozen=True)
class ForecastSealPlan:
    fold: FoldInputs
    query_donor: str
    query_access_sha256: str
    entries: tuple[ForecastSealEntry, ...]
    ensemble_derivations: tuple[bytes, ...] = ()

    def __post_init__(self):
        if not isinstance(self.fold, FoldInputs):
            raise TypeError("Exact fitting fold required")
        self.fold.__post_init__()
        _sha(self.query_access_sha256)
        if (self.query_donor not in ("D1", "D2", "D3", "D4")
                or self.query_donor in self.fold.training_donors
                or type(self.entries) is not tuple or not self.entries
                or not all(isinstance(e, ForecastSealEntry) for e in self.entries)):
            raise ValueError("Held-out query and complete nonempty ordered bank inventory required")
        for entry in self.entries: entry.__post_init__()
        for values in ((e.forecast_id for e in self.entries), (e.bank_relative_root for e in self.entries),
                       (e.spec.identity for e in self.entries)):
            values = tuple(values)
            if len(set(values)) != len(values):
                raise ValueError("Duplicate forecast, bank path or full specification in seal inventory")
        roots = [Path(e.bank_relative_root) for e in self.entries]
        if any(a in b.parents for a in roots for b in roots if a != b):
            raise ValueError("Nested bank paths cannot denote independent packed publications")
        documents = [e.spec.document for e in self.entries]
        reference = documents[0]
        common = ("source_identity", "source_reference", "catalog_sha256", "feature_axis_sha256", "synthetic_fixture")
        for document in documents:
            binding = document["model_binding"]
            if (binding["fold"] != {"fold_id": self.fold.fold_id, "training_donors": list(self.fold.training_donors)}
                    or document["source_reference"]["donor"] != self.query_donor
                    or document["source_reference"]["condition"] != "Stim8hr"
                    or document["source_reference"]["source_access_sha256"] != self.query_access_sha256
                    or any(document[k] != reference[k] for k in common)
                    or binding["observation_binding"] != reference["model_binding"]["observation_binding"]
                    or binding["numerical_source_sha256"] != reference["model_binding"]["numerical_source_sha256"]):
                raise ValueError("Banks must share the exact fold/query/source/representation/numerical source")
        _ensemble_inventory(self, documents)

    def configuration_inventory(self):
        rows = []
        for entry in self.entries:
            doc = entry.spec.document
            rows.append(dict(forecast_id=entry.forecast_id,
                configuration=asdict(get_configuration(doc["recipe"]["configuration_id"])),
                recipe=doc["recipe"], prefixes=doc["prefixes"],
                single_seed_prefixes=doc["single_seed_prefixes"],
                ensemble_component_totals=doc["ensemble_component_totals"],
                ensemble_component_prefixes=doc["ensemble_component_prefixes"], context=doc["context"],
                derivation_manifest=None if entry.derivation_manifest is None else _parse(entry.derivation_manifest)))
        inventory = dict(schema="credo.frontier.forecast-seal-configurations.v1", entries=rows)
        # Preserve the exact historical empty-plan configuration/hash semantics.
        if self.ensemble_derivations:
            inventory["ensemble_derivations"] = [_parse(raw) for raw in self.ensemble_derivations]
        return inventory

    @property
    def configuration_sha256(self):
        return _hash(self.configuration_inventory())

    @property
    def expectation(self):
        """Derive from the caller's frozen PLAN, never from available readers."""
        documents = [e.spec.document for e in self.entries]
        models = tuple(dict.fromkeys(d["model_binding"]["model_artifact_sha256"] for d in documents))
        binding = documents[0]["model_binding"]
        return ForecastExpectation(self.fold.identity, self.query_access_sha256, models,
            self.configuration_sha256, binding["numerical_source_sha256"],
            binding["observation_binding"]["artifact_sha256"], tuple(e.forecast_id for e in self.entries))

    def record(self):
        return dict(schema="credo.frontier.forecast-seal-plan.v1", fold=asdict(self.fold),
            query_donor=self.query_donor, query_access_sha256=self.query_access_sha256,
            expectation=self.expectation.record(), configuration_inventory=self.configuration_inventory(),
            entries=[dict(forecast_id=e.forecast_id, bank_relative_root=e.bank_relative_root,
                spec_sha256=e.spec.identity, complete_sha256=e.expected_complete_sha256,
                manifest_sha256=e.expected_manifest_sha256,
                model_binding=e.spec.document["model_binding"], source_identity=e.spec.document["source_identity"],
                source_reference=e.spec.document["source_reference"], catalog_sha256=e.spec.document["catalog_sha256"],
                feature_axis_sha256=e.spec.document["feature_axis_sha256"]) for e in self.entries],
            inventory_completeness_is_controller_obligation=True, whole_study_seal=False,
            scientific_launch_authority=False)


def freeze_calibration_derivation(view):
    """Bind a live source-only adapter's manifest, without evaluating a recipe."""
    from .calibration_view import CalibrationView
    if not isinstance(view, CalibrationView):
        raise TypeError("Live source-bound CalibrationView required")
    return _json_bytes(view.derivation_manifest)


def freeze_ensemble_derivation(view):
    """Freeze one live lazy outer ensemble without requesting laws or catalogs."""
    from .ensemble_view import EnsembleView
    if type(view) is not EnsembleView:
        raise TypeError("Exact live source-bound EnsembleView required")
    return _json_bytes(view.derivation_manifest)


def _ensemble_inventory(plan, documents):
    """Resolve canonical ensemble manifests to already declared parent entries.

    Live implementation/reader custody is checked separately at publication.
    This metadata-only check prevents incomplete, inner, foreign or duplicate
    component inventories from becoming an owner-frozen plan in the first place.
    """
    from .forecast import ensemble_prefix_plan
    if type(plan.ensemble_derivations) is not tuple:
        raise ValueError("Immutable ordered ensemble derivation tuple required")
    if not plan.ensemble_derivations:
        return ()
    parents = {}
    for i, entry in enumerate(plan.entries):
        if entry.derivation_manifest is not None:
            record = _parse(entry.derivation_manifest)
            identity = _hash(record)
            if identity in parents:
                raise ValueError("Duplicate calibration parent identity")
            parents[identity] = (i, record)
    seen, recipes, resolved = set(), set(), []
    for raw in plan.ensemble_derivations:
        if type(raw) is not bytes or _json_bytes(_parse(raw)) != raw:
            raise ValueError("Immutable canonical ensemble derivation bytes required")
        record = _parse(raw)
        if (not isinstance(record, dict)
                or record.get("schema") != "credo.frontier.lazy-ensemble-derivation.v1"
                or record.get("status") != "source_only_inventory_bound"):
            raise ValueError("Source-only lazy ensemble derivation required")
        identities = record.get("component_derivation_sha256")
        if (type(identities) is not list or len(identities) != 3
                or any(type(x) is not str or x not in parents for x in identities)
                or len(set(identities)) != 3):
            raise ValueError("All three distinct ensemble calibration parents must be declared entries")
        indices, components = zip(*(parents[identity] for identity in identities))
        docs = [documents[i] for i in indices]
        frozen = components[0].get("role", {}).get("frozen_outer_recipe")
        if (not isinstance(frozen, dict) or record.get("frozen_outer_recipe") != frozen
                or frozen.get("assessment_donor") != plan.query_donor
                or frozen.get("fitting_donors") != list(plan.fold.training_donors)
                or frozen.get("seeds") != [0, 1, 2] or frozen.get("updates") != 9216
                or record.get("seeds") != [0, 1, 2]
                or record.get("parent_publications") != [c["parent_publication"] for c in components]
                or record.get("model_bindings") != [d["model_binding"] for d in docs]):
            raise ValueError("Ensemble must bind one exact frozen outer recipe and ordered seed triplet")
        recipe = {k: frozen[k] for k in ("configuration_id", "alpha", "spread_scale")}
        if record.get("recipe") != recipe:
            raise ValueError("Ensemble recipe differs from frozen outer selection")
        totals = record.get("totals")
        if (type(totals) is not list or not totals
                or any(type(t) is not int for t in totals) or totals != sorted(set(totals))
                or (not docs[0]["synthetic_fixture"] and totals != [4096, 8192])):
            raise ValueError("Registered nonempty fixed-total ensemble prefix inventory required")
        counts = {str(t): [] for t in totals}
        for seed, (component, doc) in enumerate(zip(components, docs)):
            role, binding = component.get("role", {}), doc["model_binding"]
            if (role.get("phase") != "outer" or role.get("stage") != "outer"
                    or role.get("required_recipe_count") != 1
                    or role.get("frozen_outer_recipe") != frozen
                    or role.get("frozen_outer_recipe_sha256") != _hash(frozen)
                    or component.get("recipes") != [recipe]
                    or component.get("model_binding") != binding
                    or binding["model_seed"] != seed or binding["accepted_updates"] != 9216
                    or binding["configuration_id"] != recipe["configuration_id"]):
                raise ValueError("Ensemble parents must be the exact outer seed zero/one/two derivations")
            planned = ensemble_prefix_plan(seed, totals, doc["single_seed_prefixes"],
                                           synthetic_fixture=doc["synthetic_fixture"])
            if (not set(totals) <= set(doc["ensemble_component_totals"])
                    or any(doc["ensemble_component_prefixes"].get(str(t)) != planned[str(t)]
                           or planned[str(t)] not in component["permitted_prefixes"] for t in totals)):
                raise ValueError("Exact ensemble component prefixes must already exist in each parent")
            for t in totals: counts[str(t)].append(planned[str(t)])
        expected = dict(fold=docs[0]["model_binding"]["fold"],
            observation_binding=docs[0]["model_binding"]["observation_binding"],
            query_donor=plan.query_donor, support_sha256=components[0]["support_sha256"],
            component_counts=counts, **{k: docs[0][k] for k in (
                "source_identity", "catalog_sha256", "feature_axis_sha256", "supported_guide_indices",
                "unavailable_source_guide_indices", "pooled_control_identity", "synthetic_fixture")})
        if any(record.get(k) != value for k, value in expected.items()):
            raise ValueError("Ensemble source, support, representation or fixed-total allocation differs")
        identity, recipe_identity = _hash(record), _hash(frozen)
        if identity in seen or recipe_identity in recipes:
            raise ValueError("Duplicate ensemble identity or frozen outer recipe")
        seen.add(identity); recipes.add(recipe_identity); resolved.append(tuple(indices))
    return tuple(resolved)


def _ensemble_binding(plan, ensembles, readers, derivations, indices):
    if type(ensembles) is not tuple or len(ensembles) != len(plan.ensemble_derivations):
        raise ValueError("Every frozen ensemble requires exactly one live view in plan order")
    if len({id(view) for view in ensembles}) != len(ensembles):
        raise ValueError("Duplicate live ensemble view")
    for raw, view, parents in zip(plan.ensemble_derivations, ensembles, indices):
        if freeze_ensemble_derivation(view) != raw:
            raise ValueError("Live ensemble differs from the exact prebound derivation")
        for component, index in zip(view._components, parents):
            if component is not derivations[index] or component._reader is not readers[index]:
                raise ValueError("Ensemble requires the same live calibration views and parent readers")


def _derivation_binding(entry, view, reader):
    if entry.derivation_manifest is None:
        if view is not None:
            raise ValueError("Unplanned calibration view cannot be added during sealing")
        return
    if (view is None or freeze_calibration_derivation(view) != entry.derivation_manifest
            or view._reader is not reader
            or _parse(entry.derivation_manifest)["parent_publication"] != dict(reader.publication_reference)):
        raise ValueError("Exact frozen derivation and its same live parent reader required")


def _native_binding(plan, index):
    if not isinstance(index, NativeInputIndex):
        raise TypeError("Admitted metadata-only NativeInputIndex required")
    query = index.access(plan.fold, "query", query_donor=plan.query_donor)
    first = plan.entries[0].spec.document
    source = first["source_reference"]
    catalog = [dict(guide_id=row.guide_id, target_id=row.target_id, is_control=bool(row.is_control))
               for row in index.catalog().itertuples(index=False)]
    if (query.identity() != plan.query_access_sha256 or tuple(query.source_ids) != (source["source_id"],)
            or index.source_roles.get(source["source_id"]) != (plan.query_donor, "Stim8hr")
            or source["source_input_sha256"] != native_source_input_identity(index, source["source_id"])
            or first["feature_axis_sha256"] != index.rna_axis_sha256 or first["catalog"] != catalog
            or first["catalog_sha256"] != _hash(catalog)
            or first["synthetic_fixture"] != index.binding.synthetic_fixture):
        raise ValueError("Forecast source differs from the exact native early query/axis/catalog")


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_mode


def publish_forecast_seal(root, seal_name, *, plan, expectation, readers, native_index,
                          output_cap_bytes, derivations=None, ensembles=()):
    """Return the existing VerifiedForecastRelease after exclusive publication.

    output_cap_bytes charges the small new plan and seal (including JSON bytes),
    not already admitted banks. All existing bank bytes are streamed again by
    VerifiedForecastRelease.verify; this verification I/O is not new inference.
    The temporary name is the sole file removed, after its exclusive final link.
    A failed attempt remains present and is never retried automatically.
    """
    if not isinstance(plan, ForecastSealPlan) or not isinstance(expectation, ForecastExpectation):
        raise TypeError("Caller-frozen complete plan and native expectation required")
    plan.__post_init__(); expectation.__post_init__()
    if expectation.record() != plan.expectation.record():
        raise ValueError("Explicit expectation differs from the frozen plan, not successful-bank selection")
    if (type(readers) is not tuple or len(readers) != len(plan.entries)
            or not all(isinstance(r, ForecastReader) for r in readers)
            or len({id(r) for r in readers}) != len(readers)):
        raise ValueError("Every planned bank requires one unique live admitted reader in plan order")
    if derivations is None:
        derivations = (None,)*len(plan.entries)
    if type(derivations) is not tuple or len(derivations) != len(plan.entries):
        raise ValueError("One optional calibration view per frozen bank entry required")
    ensemble_indices = _ensemble_inventory(plan,
        [e.spec.document for e in plan.entries] if plan.ensemble_derivations else ())
    _ensemble_binding(plan, ensembles, readers, derivations, ensemble_indices)
    if (type(output_cap_bytes) is not int or output_cap_bytes < 1
            or _uri(seal_name) != Path(seal_name).name):
        raise ValueError("Explicit positive seal-byte cap and single canonical seal filename required")
    root = _path(root)
    _native_binding(plan, native_index)
    rows = []
    for entry, reader, derivation in zip(plan.entries, readers, derivations):
        reader.forecast_receipt  # Public access guard checks its active descriptor/path leases.
        _derivation_binding(entry, derivation, reader)
        bank = _path(root / entry.bank_relative_root)
        expected_publication = dict(root=str(bank), complete_sha256=entry.expected_complete_sha256,
            manifest_sha256=entry.expected_manifest_sha256, spec_sha256=entry.spec.identity)
        if (reader.root != bank or reader.spec.identity != entry.spec.identity
                or any(reader.publication_reference.get(k) != v for k, v in expected_publication.items())):
            raise ValueError("Missing, foreign, differently ordered or differently bound forecast bank")
        prefix = entry.bank_relative_root + "/"
        with _verified(root, prefix+"COMPLETE.json", entry.expected_complete_sha256,
                       maximum_bytes=32*1024**2) as stream:
            complete_bytes = stream.read(); complete = _parse(complete_bytes)
        with _verified(root, prefix+"MANIFEST.json", entry.expected_manifest_sha256,
                       size=complete["manifest_bytes"], maximum_bytes=128*1024**2) as stream:
            manifest_bytes = stream.read(); manifest = _parse(manifest_bytes)
        if (complete["spec_sha256"] != entry.spec.identity
                or complete["manifest_sha256"] != entry.expected_manifest_sha256
                or manifest["spec_sha256"] != entry.spec.identity
                or _json_bytes(manifest["spec"]) != entry.spec._encoded):
            raise ValueError("Exact bank COMPLETE/MANIFEST binding differs")
        artifacts = [dict(relative_uri=prefix+name, sha256=digest, size_bytes=len(raw)) for name, digest, raw in (
            ("COMPLETE.json", entry.expected_complete_sha256, complete_bytes),
            ("MANIFEST.json", entry.expected_manifest_sha256, manifest_bytes))]
        if tuple(r["relative_path"] for r in manifest["arrays"]) != tuple(n+".npy" for n in ARRAY_NAMES):
            raise ValueError("Exact packed forecast array inventory required")
        artifacts.extend(dict(relative_uri=prefix+r["relative_path"], sha256=r["file_sha256"],
                              size_bytes=r["file_bytes"]) for r in manifest["arrays"])
        if sum(a["size_bytes"] for a in artifacts) != reader.publication_reference["total_bytes"]:
            raise ValueError("Complete bank artifact bytes do not reconcile")
        rows.append(dict(forecast_id=entry.forecast_id, artifacts=artifacts))
    plan_name = seal_name + ".plan.json"
    plan_payload = _json_bytes(plan.record())
    plan_digest = hashlib.sha256(plan_payload).hexdigest()
    rows[0]["artifacts"].append(dict(relative_uri=plan_name, sha256=plan_digest, size_bytes=len(plan_payload)))
    payload = _json_bytes(dict(schema=SCHEMA, status="complete", bindings=expectation.record(), forecasts=rows))
    if len(payload)+len(plan_payload) > output_cap_bytes:
        raise ValueError("Plan plus seal exceeds its explicit output byte cap")
    pending = "." + seal_name + ".pending"
    if any(os.path.lexists(root / name) for name in (seal_name, pending, plan_name)):
        raise FileExistsError("Seal or incomplete attempt exists; no overwrite or retry")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        root_identity = _directory_identity(os.fstat(root_fd))
        plan_stamp = None
        def check():
            if (not stat.S_ISDIR(os.fstat(root_fd).st_mode)
                    or _directory_identity(_path(root).lstat()) != root_identity):
                raise ValueError("Seal root changed during publication")
            if plan_stamp is not None:
                try:
                    current_plan_stamp = _stamp(os.stat(plan_name, dir_fd=root_fd, follow_symlinks=False))
                except OSError as exc:
                    raise ValueError("Published seal plan disappeared during publication") from exc
                if current_plan_stamp != plan_stamp:
                    raise ValueError("Published seal plan changed during publication")
            for entry, reader, view in zip(plan.entries, readers, derivations):
                reader.forecast_receipt
                _derivation_binding(entry, view, reader)
            _ensemble_binding(plan, ensembles, readers, derivations, ensemble_indices)
        check()
        digest = hashlib.sha256(payload).hexdigest()
        plan_stamp = _write_once(root_fd, plan_name, (plan_payload,),
                                 expected_sha256=plan_digest, expected_bytes=len(plan_payload))
        _write_once(root_fd, pending, (payload,), expected_sha256=digest, expected_bytes=len(payload))
        # Verify the complete artifact set before publishing the final seal name.
        release = VerifiedForecastRelease.verify(root, pending, manifest_sha256=digest, expectation=expectation)
        check()
        os.link(pending, seal_name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
        os.unlink(pending, dir_fd=root_fd)
        os.fsync(root_fd)
        check()
        # The native token binds content, not filename. Verify the identical
        # exclusively linked final seal without rehashing every large bank twice.
        with _verified(root, plan_name, plan_digest, size=len(plan_payload),
                       maximum_bytes=output_cap_bytes) as plan_stream:
            with _verified(root, seal_name, digest, size=len(payload), maximum_bytes=output_cap_bytes) as stream:
                if plan_stream.read() != plan_payload or stream.read() != payload:
                    raise ValueError("Final plan/seal readback differs")
                check()
        check()
        return release
    finally:
        os.close(root_fd)
