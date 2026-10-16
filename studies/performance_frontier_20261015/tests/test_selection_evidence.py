"""Fabricated scalar metadata, NOT numerical results or particle simulations.

512/4096 are nominal protocol fields written from scratch. No smaller-particle
result is relabeled; no cells, model tensors, banks, forecasts, optimizer calls,
neural evaluation or scientific work are generated. Real scalar-file hashing
and the retained reduction verifier are exercised without mocking that verifier.
"""

from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier import selection_evidence as module
from frontier.assessment import scientific_recipe_identity
from frontier.assessment_storage import SCHEMA as STORAGE_SCHEMA, SCORE_SOURCES, _reconcile
from frontier.checkpoints import SCHEMA as CHECKPOINT_SCHEMA
from frontier.configuration import ALPHA_GRID, DONORS, FAMILIES, SPREAD_GRID, get_configuration
from frontier.evaluation import ScoreIdentity, reduce_scores
from frontier.fitting_provider import _hash
from frontier.forecast_storage import ARRAY_NAMES
from frontier.native_inputs import FoldInputs, ForecastExpectation
from frontier.planner import family_plan, finalist_plan, outer_plan, screen_plan
from frontier.selection import CHANNELS, UnavailableSelection
from frontier.selection_evidence import (BoundAssessmentEvidence, FABRICATED_SCOPE,
    SelectionEvidenceAdmission, SelectionMetadataFile, read_inner_assessment)
from frontier.selector_contract import Recipe


MIB = 1024**2


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def plain(value):
    return json.loads(encoded(value))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    raw = encoded(value)
    path.write_bytes(raw)
    return SelectionMetadataFile(str(path), hashlib.sha256(raw).hexdigest(), len(raw))


def stage(name):
    if name == "screen": return screen_plan()
    family = family_plan({d: tuple(f"{f}-V1" for f in FAMILIES) for d in DONORS})
    if name == "family": return family
    return finalist_plan(family, {d: ("F0-V1", "F2-V1", "F5-V1") for d in DONORS})


def package(tmp_path, *, name="screen", configuration=None, role_change=None,
            checkpoint_change=None, prefix_override=None, omitted_recipe=False,
            unavailable=False, changed_support=False, changed_input=False, reuse_group=None,
            group_change=None, selector="D1", training_pair=None):
    """Explicit metadata-only fixture with no bank or checkpoint tensor files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = stage(name)
    configuration = configuration or ("F0-V0" if name == "screen" else "F0-V1")
    task = next(t for t in plan.scores_for(selector) if t.key.configuration_id == configuration
                and (training_pair is None or t.key.training_donors==training_pair))
    fold = FoldInputs("subset-"+"-".join(task.key.training_donors), task.key.training_donors)
    prefix = prefix_override or (4096 if name == "finalist" else 512)
    observation = dict(artifact_sha256=digest("observation"), state_sha256=digest("observation-state"))
    roles = {k: digest(k) for k in ("representation_fit", "dynamics_source", "dynamics_supervision")}
    fit_input = _hash(dict(fold=fold.identity, source=roles["dynamics_source"], supervision=roles["dynamics_supervision"]))
    model_sha = digest("MODEL-METADATA-ONLY-"+configuration)
    source = digest("query-source"+task.validation_donor)
    query_access = digest("query-access"+task.validation_donor)
    binding = dict(fold=plain(asdict(fold)), observation_binding=observation,
        configuration_id=configuration, model_artifact_sha256=model_sha, model_state_sha256=digest("state-"+configuration),
        target_support_sha256=digest("target-support"), numerical_source_sha256=digest("producer"),
        model_seed=0, accepted_updates=task.updates)
    source_ref = dict(donor=task.validation_donor, condition="Stim8hr", source_id=task.validation_donor+"_Stim8hr",
        source_access_sha256=query_access, source_input_sha256=digest("source-input"))
    parent = dict(root=str(tmp_path/("bank-not-created-"+configuration)), complete_sha256=digest("bank-complete-"+configuration),
        manifest_sha256=digest("bank-manifest-"+configuration), spec_sha256=digest("bank-spec-"+configuration), total_bytes=60)
    recipes = [Recipe(configuration, a, s) for a in ALPHA_GRID for s in SPREAD_GRID]
    context = dict(configuration_id=configuration, training_donors=list(task.key.training_donors),
        model_seed=0, accepted_update=task.updates, purpose="forecast")
    role = dict(phase="inner", stage=name, selector=task.selector, task=plain(asdict(task)),
        stage_plan_sha256=_hash(asdict(plan)), eligible_configurations=list(plan.eligible_for(task.selector)),
        required_recipe_count=24, outer_outcome_selection_allowed=False)
    derivation = dict(schema="credo.frontier.calibration-derivation.v1", status="source_only_inventory_bound",
        parent_publication=parent, model_binding=binding, context=context, source_identity=source,
        catalog_sha256=digest("catalog"), feature_axis_sha256=digest("gene-axis"), role=role,
        recipes=[asdict(r) for r in recipes], permitted_prefixes=[prefix], synthetic_fixture=True,
        fixture_scope=FABRICATED_SCOPE)
    derivation_sha = _hash(derivation)
    entry = dict(forecast_id="fabricated-bank-"+configuration, bank_relative_root="bank-not-created-"+configuration,
        spec_sha256=parent["spec_sha256"], complete_sha256=parent["complete_sha256"], manifest_sha256=parent["manifest_sha256"],
        model_binding=binding, source_identity=source, source_reference=source_ref,
        catalog_sha256=derivation["catalog_sha256"], feature_axis_sha256=derivation["feature_axis_sha256"])
    config = dict(forecast_id="fabricated-bank-"+configuration, configuration=plain(asdict(get_configuration(configuration))),
        recipe=asdict(Recipe(configuration)), prefixes=[prefix], single_seed_prefixes=[prefix],
        ensemble_component_totals=[], ensemble_component_prefixes={}, context=context, derivation_manifest=derivation)
    if reuse_group is None:
        entries, configurations = [], []
        group_tasks = [t for t in plan.scores_for(task.selector)
            if t.key.training_donors == task.key.training_donors and t.validation_donor == task.validation_donor]
        for group_task in group_tasks:
            c = group_task.key.configuration_id
            gbinding = dict(binding, configuration_id=c, model_artifact_sha256=digest("MODEL-METADATA-ONLY-"+c),
                model_state_sha256=digest("state-"+c))
            gparent = dict(parent, root=str(tmp_path/("bank-not-created-"+c)), complete_sha256=digest("bank-complete-"+c),
                manifest_sha256=digest("bank-manifest-"+c), spec_sha256=digest("bank-spec-"+c))
            gcontext = dict(context, configuration_id=c)
            grole = dict(role, task=plain(asdict(group_task)))
            if role_change and c == configuration: role_change(grole)
            gderivation = dict(derivation, parent_publication=gparent, model_binding=gbinding,
                context=gcontext, role=grole, recipes=[asdict(Recipe(c,a,s)) for a in ALPHA_GRID for s in SPREAD_GRID])
            entries.append(dict(entry, forecast_id="fabricated-bank-"+c, bank_relative_root="bank-not-created-"+c,
                spec_sha256=gparent["spec_sha256"], complete_sha256=gparent["complete_sha256"],
                manifest_sha256=gparent["manifest_sha256"], model_binding=gbinding))
            configurations.append(dict(config, forecast_id="fabricated-bank-"+c,
                configuration=plain(asdict(get_configuration(c))), recipe=asdict(Recipe(c)),
                context=gcontext, derivation_manifest=gderivation))
        if group_change: group_change(entries, configurations)
        inventory = dict(schema="credo.frontier.forecast-seal-configurations.v1", entries=configurations)
        expectation = ForecastExpectation(fold.identity, query_access,
            tuple(dict.fromkeys(e["model_binding"]["model_artifact_sha256"] for e in entries)), _hash(inventory),
            binding["numerical_source_sha256"], observation["artifact_sha256"], tuple(e["forecast_id"] for e in entries))
        saved_plan = dict(schema="credo.frontier.forecast-seal-plan.v1", fold=plain(asdict(fold)),
            query_donor=task.validation_donor, query_access_sha256=query_access, expectation=expectation.record(),
            configuration_inventory=inventory, entries=entries, inventory_completeness_is_controller_obligation=True,
            whole_study_seal=False, scientific_launch_authority=False)
        plan_ref = write_json(tmp_path/"SEAL.json.plan.json", saved_plan)
        forecasts = []
        for pos,e in enumerate(entries):
            artifacts = [dict(relative_uri=e["bank_relative_root"]+"/"+n, sha256=h, size_bytes=10) for n,h in
                (("COMPLETE.json", e["complete_sha256"]), ("MANIFEST.json", e["manifest_sha256"]),
                 *((n+".npy", digest(n)) for n in ARRAY_NAMES))]
            if pos == 0:
                artifacts.append(dict(relative_uri="SEAL.json.plan.json", sha256=plan_ref.sha256, size_bytes=plan_ref.size_bytes))
            forecasts.append(dict(forecast_id=e["forecast_id"], artifacts=artifacts))
        seal_ref = write_json(tmp_path/"SEAL.json", dict(schema="credo.frontier.forecast-seal.v1", status="complete",
            bindings=expectation.record(), forecasts=forecasts))
    else:
        saved_plan = reuse_group.saved_plan
        plan_ref, seal_ref = reuse_group.admission.seal_plan, reuse_group.admission.seal
    selected = next(i for i,e in enumerate(saved_plan["entries"]) if e["model_binding"]["configuration_id"] == configuration)
    entry = saved_plan["entries"][selected]
    config = saved_plan["configuration_inventory"]["entries"][selected]
    derivation = config["derivation_manifest"]; derivation_sha = _hash(derivation)
    parent, binding = derivation["parent_publication"], entry["model_binding"]
    checkpoint_root = tmp_path/"checkpoint-metadata-only"; checkpoint_root.mkdir()
    checkpoint = dict(schema=CHECKPOINT_SCHEMA, identity=dict(trajectory_id="fabricated-metadata-only",
        fold_id=fold.fold_id, seed=0, input_sha256=fit_input,
        configuration_sha256=_hash(asdict(get_configuration(configuration))), source_sha256=binding["numerical_source_sha256"],
        native_roles=roles, training_owner=dict(observation_binding=observation)), checkpoint_sha256=model_sha,
        checkpoint_bytes=1, accepted_updates=task.updates, attempted_updates=task.updates, phase="joint",
        execution_scope=FABRICATED_SCOPE, milestone=True, claim_id="metadata-only", worker_id="metadata-only")
    if checkpoint_change: checkpoint_change(checkpoint)
    checkpoint_ref = write_json(checkpoint_root/"COMPLETE.json", checkpoint)
    rows, reduced = [], []
    for ri,r in enumerate(recipes[:-1] if omitted_recipe else recipes):
        sid = ScoreIdentity(fold.fold_id, task.validation_donor, scientific_recipe_identity(r), 0, prefix,
            binding["numerical_source_sha256"], observation["artifact_sha256"], model_sha,
            digest("different-input" if changed_input and ri == 1 else "shared-source-endpoint-support"),
            entry["catalog_sha256"], entry["feature_axis_sha256"])
        weights = {c: {"guide:0": .5, "guide:1": .5} for c in CHANNELS}
        weights["pooled_control_endpoint_mse"] = {"pooled_control": 1.}
        for c in ("count_ce_minus_entropy", "count_quadratic"): weights[c] = {"catalog": 1.}
        if changed_support and ri == 1: weights["state_energy"] = {"guide:0": .25, "guide:1": .75}
        scalar_rows = {}
        for key in ("catalog", "guide:0", "guide:1", "pooled_control"):
            score = dict(identity=asdict(sid), arithmetic_only=False, dtype="float64", unavailable={}, roundoff={},
                fixture_scope=FABRICATED_SCOPE, **{c: 1.+ri/100 for c in CHANNELS if key in weights[c]})
            if unavailable and ri == 1 and key == "guide:0":
                score["state_energy"] = None; score["unavailable"]["state_energy"] = "fabricated_missing_prediction"
            scalar_rows[key] = score
            rows.append(encoded(dict(recipe=asdict(r), recipe_identity=sid.recipe_id, prefix=prefix, row_id=key, score=score)))
        reduced.append(dict(recipe=asdict(r), scores=reduce_scores(scalar_rows, weights, identity=sid)))
    raw_rows = b"".join(rows)
    receipt = dict(schema="credo.frontier.streamed-calibration-assessment.v2", status="numerical_complete_not_published",
        predictor_kind="single_seed", optimizer_calls=0, neural_forecast_calls=0, recipe_count=len(reduced),
        parent_publications=[parent], rows_sha256=hashlib.sha256(raw_rows).hexdigest(), emitted_rows=len(rows),
        rows_bytes=len(raw_rows), prefix=prefix, forecast_seal_sha256=seal_ref.sha256,
        support_sha256=digest("common-support"), derivation_sha256=derivation_sha, early_source_identity=source,
        observation_binding=observation, ensemble_model_identity=None,
        prediction_derivations={scientific_recipe_identity(r): _hash(dict(derivation_sha256=derivation_sha, recipe=asdict(r)))
            for r in (recipes[:-1] if omitted_recipe else recipes)},
        scoring_implementation_sources={name:digest(name) for name in SCORE_SOURCES},
        runtime=dict(scope=FABRICATED_SCOPE, geometry_and_reduction_dtype="float64"),
        limits=dict(pair_block=128, gene_block=1024, row_block=128), fixture_scope=FABRICATED_SCOPE)
    summary = dict(scores=reduced, receipt=receipt)
    summary_raw = encoded(summary)
    assessment = tmp_path/"assessment-scalars"; assessment.mkdir()
    (assessment/"SCORES.jsonl").write_bytes(raw_rows)
    (assessment/"SUMMARY.json").write_bytes(summary_raw)
    arithmetic = _reconcile(io.BytesIO(raw_rows), summary, maximum_row_bytes=65536, maximum_rows_bytes=8*MIB)
    complete = dict(schema=STORAGE_SCHEMA, status="complete", producer_sha256=digest("fabricated-scalar-writer"),
        files=[dict(relative_path="SCORES.jsonl", sha256=receipt["rows_sha256"], size_bytes=len(raw_rows)),
               dict(relative_path="SUMMARY.json", sha256=hashlib.sha256(summary_raw).hexdigest(), size_bytes=len(summary_raw))],
        output_cap_bytes=8*MIB, maximum_row_bytes=65536, maximum_summary_bytes=4*MIB, total_bytes=0,
        arithmetic=arithmetic, parent_publications=[parent],
        parent_exit_policy="all_exact_live_ForecastReaders_closed_with_full_hash_verification_before_COMPLETE",
        parent_exit_hash_bytes=60, scientific_launch_authority=False)
    while True:
        total = len(raw_rows)+len(summary_raw)+len(encoded(complete))
        if total == complete["total_bytes"]: break
        complete["total_bytes"] = total
    completed = write_json(assessment/"COMPLETE.json", complete)
    admission = SelectionEvidenceAdmission(str(assessment), completed.sha256, 8*MIB,
        seal_ref, plan_ref, checkpoint_ref, 4*MIB, fabricated_scalar_metadata=True)
    return SimpleNamespace(admission=admission, plan=plan, task=task, parent=parent,
        root=tmp_path, prefix=prefix, fit_input=fit_input, saved_plan=saved_plan)


@pytest.mark.parametrize("name", ["screen", "family", "finalist"])
def test_saved_scalar_metadata_replays_into_24_fixed_inner_recipes(tmp_path, name):
    p = package(tmp_path, name=name)
    result = read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)
    assert len(result.evidence) == 24
    assert {e.context.scoring_particles for e in result.evidence} == {p.prefix}
    assert all(e.context.fitting_input_identity == p.fit_input for e in result.evidence)
    assert result.provenance["fabricated_scalar_metadata"] is True
    assert result.provenance["bank_or_outcome_or_checkpoint_tensor_reads"] == 0
    assert result.provenance["selection_performed"] is False
    assert not list(tmp_path.glob("bank-not-created-*"))
    assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.npy"))
    with pytest.raises(TypeError): result.evidence[0].risks["state_energy"] = 0.
    detached = result.provenance; detached["recipes"] = 0
    assert result.provenance["recipes"] == 24
    with pytest.raises(TypeError, match="read_inner_assessment"): BoundAssessmentEvidence()


def test_candidate_specific_custody_does_not_break_common_comparison_context(tmp_path):
    a = package(tmp_path/"incumbent", configuration="F0-V0")
    b = package(tmp_path/"candidate", configuration="F1-V0", reuse_group=a)
    ra = read_inner_assessment(a.admission, stage_plan=a.plan, task=a.task)
    rb = read_inner_assessment(b.admission, stage_plan=b.plan, task=b.task)
    assert ra.evidence[0].context == rb.evidence[0].context
    assert ra.evidence[0].coverage == rb.evidence[0].coverage
    assert ra.provenance["forecast_seal_sha256"] == rb.provenance["forecast_seal_sha256"]
    assert ra.provenance["model_artifact_sha256"] != rb.provenance["model_artifact_sha256"]


def test_distinct_group_republication_remains_explicitly_incompatible(tmp_path):
    a = package(tmp_path/"first")
    b = package(tmp_path/"second")
    ra = read_inner_assessment(a.admission, stage_plan=a.plan, task=a.task)
    rb = read_inner_assessment(b.admission, stage_plan=b.plan, task=b.task)
    assert ra.evidence[0].context.truth_identity != rb.evidence[0].context.truth_identity


@pytest.mark.parametrize("mutation", [
    lambda entries, configs: (entries.pop(), configs.pop()),
    lambda entries, configs: configs[-1]["derivation_manifest"]["role"].update(selector="D2"),
    lambda entries, configs: configs[-1]["derivation_manifest"]["recipes"].pop(),
    lambda entries, configs: entries[-1]["model_binding"].update(accepted_updates=3072),
])
def test_every_group_member_must_be_complete_and_allowed_before_scores(tmp_path, monkeypatch, mutation):
    p = package(tmp_path, group_change=mutation)
    monkeypatch.setattr(module, "verify_assessment", lambda *_a, **_k: pytest.fail("Invalid group opened scores"))
    with pytest.raises(ValueError, match="inventories|calibration role|calibration grid"):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


def test_outer_or_ineligible_task_rejected_before_any_file_operation(tmp_path, monkeypatch):
    p = package(tmp_path)
    monkeypatch.setattr(module, "_verified_file", lambda *_a, **_k: pytest.fail("File access before eligibility"))
    final = stage("finalist"); outer = outer_plan(final)
    with pytest.raises(ValueError, match="eligible inner task"):
        read_inner_assessment(p.admission, stage_plan=outer, task=outer.scores[0])
    with pytest.raises(ValueError, match="eligible inner task"):
        read_inner_assessment(p.admission, stage_plan=stage("family"), task=p.task)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(selector="D2"),
    lambda r: r.update(phase="outer"),
    lambda r: r["task"].update(validation_donor="D1"),
    lambda r: r.update(stage_plan_sha256="f"*64),
])
def test_wrong_saved_role_rejected_before_assessment_payload(tmp_path, monkeypatch, mutation):
    p = package(tmp_path, role_change=mutation)
    monkeypatch.setattr(module, "verify_assessment", lambda *_a, **_k: pytest.fail("Foreign score payload opened"))
    with pytest.raises(ValueError, match="calibration role"):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(attempted_updates=r["accepted_updates"]+1),
    lambda r: r.update(checkpoint_sha256="f"*64),
    lambda r: r["identity"].update(input_sha256="f"*64),
    lambda r: r["identity"].update(seed=1),
])
def test_checkpoint_metadata_mismatch_rejected_before_scores(tmp_path, monkeypatch, mutation):
    p = package(tmp_path, checkpoint_change=mutation)
    monkeypatch.setattr(module, "verify_assessment", lambda *_a, **_k: pytest.fail("Unbound score payload opened"))
    with pytest.raises(ValueError, match="Checkpoint"):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


def test_small_numeric_resolution_is_not_relabelled_as_registered(tmp_path, monkeypatch):
    p = package(tmp_path, prefix_override=7)
    monkeypatch.setattr(module, "verify_assessment", lambda *_a, **_k: pytest.fail("Wrong resolution opened"))
    with pytest.raises(ValueError, match="registered scoring prefix"):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


@pytest.mark.parametrize("changes, error", [
    (dict(omitted_recipe=True), "assessment differs"),
    (dict(unavailable=True), "required saved channel"),
    (dict(changed_support=True), "fixed context and channel support"),
    (dict(changed_input=True), "fixed context and channel support"),
])
def test_missing_or_changed_required_recipe_channel_support_fails_closed(tmp_path, changes, error):
    p = package(tmp_path, **changes)
    with pytest.raises((ValueError, UnavailableSelection), match=error):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


def test_fabricated_metadata_cannot_be_admitted_as_scientific(tmp_path):
    p = package(tmp_path)
    with pytest.raises(ValueError, match="calibration role|Checkpoint"):
        read_inner_assessment(replace(p.admission, fabricated_scalar_metadata=False), stage_plan=p.plan, task=p.task)


def test_exact_metadata_hash_and_symlink_checks_precede_scalar_scores(tmp_path, monkeypatch):
    p = package(tmp_path)
    monkeypatch.setattr(module, "verify_assessment", lambda *_a, **_k: pytest.fail("Unverified metadata accepted"))
    wrong = replace(p.admission, seal=replace(p.admission.seal, sha256="f"*64))
    with pytest.raises(ValueError, match="SHA-256"):
        read_inner_assessment(wrong, stage_plan=p.plan, task=p.task)
    original = Path(p.admission.checkpoint_complete.path)
    moved = original.with_name("original.json"); original.rename(moved); original.symlink_to(moved)
    with pytest.raises(ValueError, match="nonsymlink|symlink"):
        read_inner_assessment(p.admission, stage_plan=p.plan, task=p.task)


def test_aggregate_metadata_cap_is_explicit(tmp_path):
    p = package(tmp_path)
    total = sum(r.size_bytes for r in (p.admission.seal, p.admission.seal_plan, p.admission.checkpoint_complete))
    with pytest.raises(ValueError, match="Aggregate metadata"):
        replace(p.admission, maximum_metadata_bytes=total-1)
