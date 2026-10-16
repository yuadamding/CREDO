import pytest
import torch

from frontier.molecular import transform_composition
from frontier.molecular_adjoint import molecular_adjoint


def fixture(dtype):
    generator = torch.Generator().manual_seed(173)
    x = torch.randn(7, 3, dtype=dtype, generator=generator)
    y = torch.randn(2, 3, dtype=dtype, generator=generator)
    parameter = torch.randn(3, 5, dtype=dtype, generator=generator, requires_grad=True)
    truth = torch.softmax(torch.randn(9, 5, dtype=dtype, generator=generator), -1)
    routing = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    a = torch.tensor([1, 2, 1, 3, 1, 2, 4], dtype=dtype) / 14
    b = torch.tensor([.5, .5], dtype=dtype)
    return x, y, parameter, truth[:7], truth[7:], routing, a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_replay_includes_shared_control_gradient(dtype, chunk):
    x, y, parameter, tt, tc, routing, a, b = fixture(dtype)
    qt, qc = torch.softmax(x @ parameter, -1), torch.softmax(y @ parameter, -1)
    et = transform_composition(qt) - transform_composition(tt)
    ec = transform_composition(qc) - transform_composition(tc)
    loss = .07 * (a * et.square().mean(-1)).sum() + .11 * (b * ec.square().mean(-1)).sum()
    loss = loss + .13 * (a * (et - ec[routing]).square().mean(-1)).sum()
    exact = torch.autograd.grad(loss, parameter)[0]
    plan = molecular_adjoint(qt, tt, qc, tc, target_donor_indices=routing, target_weights=a, control_weights=b,
                             target_coefficient=.07, control_coefficient=.11, effect_coefficient=.13)
    torch.testing.assert_close(plan.objective, loss.detach())
    assert not plan.target_derivatives.requires_grad and not plan.control_derivatives.requires_grad
    for start in range(0, 7, chunk):
        index = torch.arange(start, min(7, start + chunk))
        value = torch.softmax(x[index] @ parameter, -1)
        plan.replay_proxy(role="target", indices=index, current_compositions=value, atol=2e-6, rtol=2e-6).backward()
    for donor in range(2):
        index = torch.tensor([donor])
        value = torch.softmax(y[index] @ parameter, -1)
        plan.replay_proxy(role="control", indices=index, current_compositions=value, atol=2e-6, rtol=2e-6).backward()
    torch.testing.assert_close(parameter.grad, exact, atol=2e-6, rtol=2e-5)


def test_changed_replay_and_bad_donor_reject():
    x, y, parameter, tt, tc, routing, a, b = fixture(torch.float64)
    qt, qc = torch.softmax(x @ parameter, -1), torch.softmax(y @ parameter, -1)
    arguments = dict(target_donor_indices=routing, target_weights=a, control_weights=b,
                     target_coefficient=.1, control_coefficient=.1, effect_coefficient=.1)
    plan = molecular_adjoint(qt, tt, qc, tc, **arguments)
    with pytest.raises(ValueError, match="differs"):
        plan.replay_proxy(role="control", indices=torch.tensor([0]), current_compositions=qc[[1]], atol=0., rtol=0.)
    with pytest.raises(ValueError, match="donor routing"):
        molecular_adjoint(qt, tt, qc, tc, **{**arguments, "target_donor_indices": routing + 2})


def test_zero_coefficients_and_controls_are_not_replicated():
    x, y, parameter, tt, tc, routing, a, b = fixture(torch.float64)
    qt, qc = torch.softmax(x @ parameter, -1), torch.softmax(y @ parameter, -1)
    plan = molecular_adjoint(qt, tt, qc, tc, target_donor_indices=routing, target_weights=a, control_weights=b,
                             target_coefficient=0., control_coefficient=.2, effect_coefficient=0.)
    assert torch.count_nonzero(plan.target_derivatives) == 0
    expected = .2 * ((transform_composition(qc) - transform_composition(tc)).square().mean(-1) * b).sum()
    grad = torch.autograd.grad(expected, qc)[0]
    torch.testing.assert_close(plan.control_derivatives, grad)
