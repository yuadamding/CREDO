"""The training handoff requires the corrected optimizer evidence."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_native_control import validate_adam_agreement  # noqa: E402


def evidence():
    check = dict(passed=True, atol=2e-5, rtol=2e-5)
    agreement = {name: dict(check) for name in ("loss", "gradients", "clipped", "norm", "Adam", "Adam_state")}
    for name in ("optimizer_initial_parameters", "optimizer_input_gradients"):
        agreement[name] = dict(passed=True, comparison="exact_tensor_equality", tensors=8)
    agreement["optimizer_initial_state"] = dict(passed=True, state="identical_empty_fresh_plain_Adam")
    return dict(profile="GPU_native_eager_foreach", numerical_agreement=agreement,
        independent_gradient_Adam_sensitivity=dict(status="descriptive_independent_gradient_sensitivity",
            affects_backend_acceptance=False, additional_optimizer_steps=0))


def test_valid_backend_evidence_and_large_descriptive_sensitivity():
    row = evidence()
    row["independent_gradient_Adam_sensitivity"]["maximum_absolute_parameter_difference"] = .002
    validate_adam_agreement(row)
    validate_adam_agreement(dict(profile="scalar_reference", numerical_agreement=dict(reference=True,
        optimizer_initial_state=dict(passed=True, state="identical_empty_fresh_plain_Adam"))))


@pytest.mark.parametrize("name", ["loss", "gradients", "clipped", "norm", "Adam", "Adam_state",
    "optimizer_initial_parameters", "optimizer_input_gradients", "optimizer_initial_state"])
def test_missing_or_failed_gate_cannot_start_training(name):
    row = evidence()
    del row["numerical_agreement"][name]
    with pytest.raises(RuntimeError):
        validate_adam_agreement(row)
    row = evidence()
    row["numerical_agreement"][name]["passed"] = False
    with pytest.raises(RuntimeError):
        validate_adam_agreement(row)


def test_approximate_input_tolerance_change_or_extra_step_rejected():
    for path, key, value in [
        (("numerical_agreement", "optimizer_input_gradients"), "comparison", "allclose"),
        (("numerical_agreement", "Adam"), "atol", .001),
        (("independent_gradient_Adam_sensitivity",), "additional_optimizer_steps", 1),
    ]:
        row = copy.deepcopy(evidence())
        target = row
        for component in path:
            target = target[component]
        target[key] = value
        with pytest.raises(RuntimeError):
            validate_adam_agreement(row)
