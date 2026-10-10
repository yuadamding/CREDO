"""Metadata-only input/phase checks; no cohort arrays opened."""
# ruff: noqa: E402
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]
import pytest
from development_inputs import validate_contract, frozen_binding
from unittest.mock import patch


def contract():
    return json.loads((ROOT / "DEVELOPMENT_INPUTS.json").read_text())


def test_precise_development_contract_passes_without_numerical_access():
    validate_contract(contract())


@pytest.mark.parametrize("field,value", [("D1_access", True), ("LARRY_access", True),
                                        ("guide_count", 26503), ("source_frequency_pseudocount", 1.0)])
def test_scope_change_rejected(field, value):
    value_contract = contract()
    value_contract[field] = value
    with pytest.raises(ValueError):
        validate_contract(value_contract)


def test_wrong_phase_and_duplicate_population_rejected():
    value = contract()
    value["stores"][0]["source_ids"][0] = "D1_Stim48hr"
    with pytest.raises(ValueError):
        validate_contract(value)
    value = contract()
    value["stores"].append(copy.deepcopy(value["stores"][-1]))
    with pytest.raises(ValueError):
        validate_contract(value)


def test_complete_identity_is_checked_before_payload_fields_are_used():
    with patch("development_inputs.checked_metadata", side_effect=ValueError("SHA mismatch")):
        with pytest.raises(ValueError, match="SHA mismatch"):
            frozen_binding(contract(), contract()["stores"][0])
