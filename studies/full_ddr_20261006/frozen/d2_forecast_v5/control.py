"""Host-Python-3.6 exact-owner validator; no successor or endpoint access."""
import json
from pathlib import Path
import stat

import preparation_control as base
from d2_forecast_reuse import SOURCE_KEYS, validate_reuse, expected_reuse_seal


def seal_row(row):
    result = {key: row[key] for key in ("candidate_id", "seed", "storage", "path", "seal", "bank_identity")}
    if row["storage"] == "external_reused":
        result.update(reuse_seal_path=row["reuse_seal_path"], array_identity=row["array_identity"])
    return result


def checked_json(path, expected=None):
    identity, raw = base.checked_file(path, True)
    if expected is not None and identity != expected:
        raise RuntimeError("Bound recovery metadata identity differs")
    return json.loads(raw.decode())


def validate_qualification(report, checkpoint):
    trunk = "mlp" if checkpoint["candidate_id"] == "ddr_mlp128_e8" else "residual_mlp"
    expected = dict(status="passed", production_qualified=True, agreement_passed=True,
        scope="source_only_scalar_vs_production_not_endpoint_precision", atol=2e-5, rtol=1e-5,
        guides=16, compared_guides=16, particles=8192, steps=8, seed=checkpoint["seed"],
        parameter_updates=0, endpoint_observations_read=False)
    policy = dict(policy="architecture_fixed_v4_mlp_batch16_residual_single", trunk_kind=trunk,
        source_chunk_maximum_guides=16, dynamics_maximum_guides=16 if trunk == "mlp" else 1,
        precision="strict_fp32", automatic_fallback=False, diagnostic_only=False)
    if (any(report.get(key) != value or type(report.get(key)) is not type(value)
            for key, value in expected.items()) or report.get("numerical_policy") != policy):
        raise RuntimeError("Unchanged architecture-bound production qualification required")


def validate_result(plan, binding, root, owned):
    output = Path(plan["worker_output"])
    reference, raw = base.checked_file(output / "COMPLETE.json", True)
    complete = json.loads(raw.decode())
    if len(owned) != 1:
        raise RuntimeError("One exact D2 forecast worker required")
    expected = dict(status="completed_D2_frozen_forecasts", run_id=plan["run_id"],
        plan_sha256=base.core.sha(base.BASE / "PLAN.json"), pod_uid=owned[0]["metadata"]["uid"],
        implementation_sha256=plan["implementation_sha256"], handoff_sha256=plan["handoff_sha256"],
        input_lease_exit_verified=True, predictions_sealed=True, parameter_updates=0,
        query_deployment_performed=True, assessment_performed=False, D2_scores_computed=False,
        D1_access=False, D1_endpoint_access=False, LARRY_access=False, scientific_promotion=False,
        scientific_acceptance="not_assessed_forecasts_only", new_bank_count=6, reused_bank_count=3,
        all_model_qualification_before_bulk=True,
        external_reuse_manifest_sha256=plan["external_reuse_manifest_sha256"])
    if any(complete.get(k) != v or type(complete.get(k)) is not type(v) for k, v in expected.items()):
        raise RuntimeError("Forecast completion scope or identity differs")
    handoff = json.loads((base.BASE / "D2_HANDOFF.json").read_text())
    if complete.get("parent_completion_sha256") != handoff["parent"]["completion_sha256"]:
        raise RuntimeError("Fitting parent changed")
    rows = complete.get("banks", [])
    expected_pairs = [(r["candidate_id"], r["seed"]) for r in handoff["checkpoints"]]
    if [(r.get("candidate_id"), r.get("seed")) for r in rows] != expected_pairs or len(rows) != 9:
        raise RuntimeError("Nine exact distinct banks required")
    inventory, size = {}, 0
    for item in complete.get("files", []):
        name = item.get("path")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.as_posix() != name or name in inventory or name == "COMPLETE.json"):
            raise RuntimeError("Unsafe forecast result inventory")
        actual, _ = base.checked_file(output / relative)
        if actual != dict(sha256=item["sha256"], size_bytes=item["size_bytes"]):
            raise RuntimeError("Forecast result bytes differ")
        inventory[name] = actual
        size += actual["size_bytes"]
    names = set()
    for path in output.rglob("*"):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise RuntimeError("Unsafe output member")
        if stat.S_ISREG(info.st_mode):
            names.add(path.relative_to(output).as_posix())
    if (not inventory or names != set(inventory) | {"COMPLETE.json"}
            or complete.get("output_bytes") != size or size+reference["size_bytes"] > plan["maximum_new_output_bytes"]):
        raise RuntimeError("Complete output membership or forecast-phase budget differs")
    manifest_id, raw = base.checked_file(base.BASE / "EXTERNAL_REUSE.json", True)
    if manifest_id["sha256"] != plan["external_reuse_manifest_sha256"]:
        raise RuntimeError("External reuse authority differs")
    manifest = json.loads(raw.decode())
    references = checked_json(output / "REFERENCES.json", inventory["REFERENCES.json"])
    # This independently streams and hashes every reused external array and
    # verifies the failed producer's exact provenance. No originals are written.
    reused = validate_reuse(manifest, source_identity={key: references[key] for key in SOURCE_KEYS},
        measure_ids=references["modeled_measure_ids"], guide_indices=references["modeled_guide_indices"],
        checkpoints=handoff["checkpoints"], owner=307469,
        maximum_new_bytes=plan["maximum_new_output_bytes"])
    retained = reused["retained_bytes"]
    if (complete.get("retained_output_bytes") != retained
            or complete.get("combined_forecast_output_bytes") != retained+size
            or retained+size+reference["size_bytes"] > plan["maximum_output_bytes"]
            or retained+size+reference["size_bytes"] > plan["combined_phase_output_ceiling_bytes"]):
        raise RuntimeError("Retained plus new forecast budget or accounting differs")
    external = {(r["candidate_id"], r["seed"]): r for r in reused["banks"]}
    if sum(r.get("storage") == "external_reused" for r in rows) != 3 or sum(r.get("storage") == "new" for r in rows) != 6:
        raise RuntimeError("Exactly three reused and six new banks required")
    for row, checkpoint in zip(rows, handoff["checkpoints"]):
        label = row["candidate_id"]+"--seed-"+str(row["seed"])
        current = checked_json(output / (label+"--qualification.json"), inventory[label+"--qualification.json"])
        loader = checked_json(output / (label+"--loader.json"), inventory[label+"--loader.json"])
        if (row.get("checkpoint_sha256") != checkpoint["sha256"]
                or row.get("qualification") != current or row.get("loader") != loader
                or current.get("status") != "passed" or current.get("production_qualified") is not True
                or current.get("agreement_passed") is not True
                or current.get("atol") != 2e-5 or current.get("rtol") != 1e-5):
            raise RuntimeError("Frozen model or current production qualification differs")
        validate_qualification(current, checkpoint)
        pair = (row["candidate_id"], row["seed"])
        if pair in external:
            old = external[pair]
            seal_path = "reuse-seals/"+label+".json"
            if (row.get("storage") != "external_reused" or row.get("path") != old["path"]
                    or row.get("bank_identity") != old["bank_identity"]
                    or row.get("array_identity") != old["array_identity"]
                    or row.get("reuse_seal_path") != seal_path
                    or row.get("seal") != inventory.get(seal_path)
                    or row.get("reuse_seal") != inventory.get(seal_path)):
                raise RuntimeError("External bank identity or new reuse seal differs")
            seal = checked_json(output / seal_path, inventory[seal_path])
            if seal != expected_reuse_seal(manifest, reused, old):
                raise RuntimeError("External reuse seal lacks exact recovery provenance")
            bank = old["bank"]
        else:
            directory = "banks/"+label
            if (row.get("storage") != "new" or row.get("path") != directory
                    or row.get("seal") != inventory.get(directory+"/COMPLETE.json")
                    or row.get("bank_identity") != inventory.get(directory+"/BANK.json")):
                raise RuntimeError("New bank identity or seal differs")
            bank = checked_json(output / directory / "BANK.json", inventory[directory+"/BANK.json"])
            array = bank["artifacts"]["forecast.h5"]
            if (inventory[directory+"/forecast.h5"] != dict(sha256=array["sha256"], size_bytes=array["bytes"])
                    or bank.get("implementation_sha256") != plan["implementation_sha256"]):
                raise RuntimeError("New bank array or producer is not its sealed identity")
            bank_seal = checked_json(output / directory / "COMPLETE.json", inventory[directory+"/COMPLETE.json"])
            input_check = checked_json(output / (label+"--input-check.json"), inventory[label+"--input-check.json"])
            if (bank_seal.get("status") != "sealed_forecast_bank"
                    or bank_seal.get("bank_sha256") != row["bank_identity"]["sha256"]
                    or bank_seal.get("input_check_sha256") != inventory[label+"--input-check.json"]["sha256"]
                    or input_check.get("inputs_verified_after_prediction") is not True
                    or input_check.get("parameters_unchanged") is not True):
                raise RuntimeError("New bank post-lease seal is invalid")
        if (bank.get("modeled_measure_ids") != sorted(handoff["query_measure_ids"])
                or bank.get("prefixes") != [4096, 8192] or bank.get("steps_per_interval") != 8
                or bank.get("endpoint_observations_read") is not False or bank.get("parameter_updates") != 0
                or bank.get("checkpoint_sha256") != checkpoint["sha256"]):
            raise RuntimeError("Forecast row, precision or no-assessment contract differs")
    qualified = checked_json(output / "ALL_MODELS_QUALIFIED.json", inventory["ALL_MODELS_QUALIFIED.json"])
    expected_qualified = [dict(candidate_id=r["candidate_id"], seed=r["seed"],
        qualification=inventory[r["candidate_id"]+"--seed-"+str(r["seed"])+"--qualification.json"])
        for r in rows]
    if (qualified.get("status") != "all_nine_current_production_paths_qualified_before_new_bulk_arrays"
            or qualified.get("checkpoints") != expected_qualified):
        raise RuntimeError("All-model qualification barrier is incomplete")
    diagnostic = checked_json(output / "ddr_res256_e8--seed-0--legacy16-diagnostic.json",
        inventory["ddr_res256_e8--seed-0--legacy16-diagnostic.json"])
    if diagnostic.get("status") != "diagnostic_only" or diagnostic.get("production_qualified") is not False:
        raise RuntimeError("Legacy diagnostic cannot masquerade as production qualification")
    _, raw = base.checked_file(output / "PREDICTION_SEAL.json", True)
    seal = json.loads(raw.decode())
    if (seal.get("status") != "all_nine_forecasts_and_source_references_sealed"
            or seal.get("banks") != [seal_row(row) for row in rows]
            or seal.get("references") != inventory["REFERENCES.json"]
            or seal.get("endpoint_observations_read") is not False):
        raise RuntimeError("Common prediction-before-assessment barrier differs")
    _, raw = base.checked_file(root / "startup-validated.json", True)
    release = json.loads(raw.decode())
    if (release.get("uid") != binding["uid"] or release.get("startup_sha256") != inventory["STARTUP.json"]["sha256"]
            or release.get("actual_pod_audit_passed") is not True):
        raise RuntimeError("Actual-Pod startup release differs")
    if base.checked_file(output / "COMPLETE.json")[0] != reference:
        raise RuntimeError("Completion changed during validation")
    return True


if __name__ == "__main__":
    base.validate_result = validate_result
    base.main()
