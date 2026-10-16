"""Logical-batch molecular derivatives for one shared control per donor.

This is a first-order replay primitive, not an optimizer or complete training
owner. Its full-gene compositions are current predictions collected without
graphs. Target and pooled-control branches can then replay once per microbatch,
including the control derivative accumulated from every targeting contrast.
"""

from dataclasses import dataclass
import math

import torch

from .molecular import _composition, transform_composition


def _coefficient(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative molecular coefficient required")
    return value


def _weights(value, size, reference):
    if (not isinstance(value, torch.Tensor) or value.shape != (size,)
            or value.dtype != reference.dtype or value.device != reference.device
            or value.requires_grad or not bool(torch.isfinite(value).all()) or bool((value < 0).any())
            or not torch.allclose(value.sum(), value.new_tensor(1), atol=2e-6 if value.dtype == torch.float32 else 1e-10, rtol=0)):
        raise ValueError("Fixed normalized logical reduction weights required")


@dataclass(frozen=True)
class MolecularAdjoint:
    target_compositions: torch.Tensor
    control_compositions: torch.Tensor
    target_derivatives: torch.Tensor
    control_derivatives: torch.Tensor
    objective: torch.Tensor
    target_mse: torch.Tensor
    control_mse: torch.Tensor
    effect_mse: torch.Tensor

    def replay_proxy(self, *, role, indices, current_compositions, atol, rtol):
        """Return a gradient proxy, NOT the displayed molecular loss.

        The training owner must cover each logical target and each control
        exactly once, check unchanged model/input identities, and discard all
        gradients after any failed replay. This function cannot prove that its
        caller performed that inventory reconciliation.
        """
        if role not in ("target", "control"):
            raise ValueError("Explicit target/control replay role required")
        _coefficient(atol)
        _coefficient(rtol)
        expected = self.target_compositions if role == "target" else self.control_compositions
        derivative = self.target_derivatives if role == "target" else self.control_derivatives
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64 or indices.ndim != 1
                or indices.device != expected.device or not indices.numel()
                or indices.unique().numel() != indices.numel()
                or bool((indices < 0).any()) or bool((indices >= len(expected)).any())):
            raise ValueError("Unique admitted logical replay indices required")
        _composition(current_compositions, ndim=2, name="replayed")
        if (current_compositions.dtype != expected.dtype or current_compositions.device != expected.device
                or current_compositions.shape != expected[indices].shape
                or not current_compositions.requires_grad
                or not torch.allclose(current_compositions.detach(), expected[indices], atol=atol, rtol=rtol)):
            raise ValueError("Molecular replay differs from the current first-pass composition")
        return (current_compositions * derivative[indices]).sum()


def molecular_adjoint(predicted_targets, observed_targets, predicted_controls, observed_controls,
                      *, target_donor_indices, target_weights, control_weights,
                      target_coefficient, control_coefficient, effect_coefficient):
    """Form exact first derivatives on the actual log1p(10000q) readout.

    ``target_weights`` reduce all targeting laws once; ``control_weights``
    reduce donor controls once, not in proportion to the number of contrasts.
    Pool guide-level control compositions using the declared fixed weights
    BEFORE calling this function. Replay derivatives pass through that pooling.
    """
    values = (predicted_targets, observed_targets, predicted_controls, observed_controls)
    for value in values:
        _composition(value, ndim=2, name="logical molecular")
    if (predicted_targets.shape != observed_targets.shape
            or predicted_controls.shape != observed_controls.shape
            or predicted_targets.shape[1] != predicted_controls.shape[1]
            or predicted_targets.dtype not in (torch.float32, torch.float64)
            or any(value.dtype != predicted_targets.dtype or value.device != predicted_targets.device for value in values)):
        raise ValueError("One full-gene axis and explicit FP32/FP64 device/dtype required")
    n, genes = predicted_targets.shape
    donors = len(predicted_controls)
    if (not isinstance(target_donor_indices, torch.Tensor) or target_donor_indices.dtype != torch.int64
            or target_donor_indices.shape != (n,) or target_donor_indices.device != predicted_targets.device
            or bool((target_donor_indices < 0).any()) or bool((target_donor_indices >= donors).any())):
        raise ValueError("Exact target-to-control donor routing required")
    _weights(target_weights, n, predicted_targets)
    _weights(control_weights, donors, predicted_targets)
    ct, cc, ce = map(_coefficient, (target_coefficient, control_coefficient, effect_coefficient))
    with torch.no_grad():
        qt, qc = predicted_targets.detach().clone(), predicted_controls.detach().clone()
        et = transform_composition(qt) - transform_composition(observed_targets)
        ec = transform_composition(qc) - transform_composition(observed_controls)
        contrast = et - ec[target_donor_indices]
        target_mse = (target_weights * et.square().mean(-1)).sum()
        control_mse = (control_weights * ec.square().mean(-1)).sum()
        effect_mse = (target_weights * contrast.square().mean(-1)).sum()
        target_derivative = (2 / genes) * target_weights[:, None] * (ct * et + ce * contrast)
        control_derivative = (2 / genes) * cc * control_weights[:, None] * ec
        control_derivative.index_add_(0, target_donor_indices,
                                      -(2 / genes) * ce * target_weights[:, None] * contrast)
        target_derivative *= 10000 / (1 + 10000 * qt)
        control_derivative *= 10000 / (1 + 10000 * qc)
        objective = ct * target_mse + cc * control_mse + ce * effect_mse
        if not all(bool(torch.isfinite(value).all()) for value in
                   (objective, target_derivative, control_derivative)):
            raise ValueError("Nonfinite molecular objective/derivative")
        return MolecularAdjoint(qt, qc, target_derivative, control_derivative, objective,
                                target_mse, control_mse, effect_mse)
