# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A circuit with no valid samples must not poison its rollout group.

`SqdDiagonalizer.diagonalize` returns +inf for an empty subspace, which is what an
approximate sampler produces when SVD truncation leaves no particle-number-valid
shots (observed with `sampler.kind=mps` at 38q: a 31538-gate circuit, valid fraction
0.000). Before this guard, that single value made the whole group's advantage NaN,
and the NaN reached Adam's moments, so the policy never recovered.
"""

import pytest

torch = pytest.importorskip("torch")

# Production imports must follow the optional dependency check.
from sqd_gqe.core.objective import GRPOObjective, GSPOObjective  # noqa: E402

OBJECTIVES = [GRPOObjective, GSPOObjective]
FINITE = [-479.1, -479.2, -479.0, -478.9]


def _loss_and_grad(cls, energies, log_probs, old_log_probs):
    log_probs = log_probs.clone().requires_grad_(True)
    context = {"gate_seqs": torch.zeros(log_probs.shape), "energies": energies,
               "old_log_probs": old_log_probs}
    loss = cls()(log_probs, context)
    loss.backward()
    return loss.detach(), log_probs.grad


def _reference_loss(cls, energies, log_probs, old_log_probs):
    """The clipped group-relative loss written out for an all-finite group."""
    adv = (energies.mean() - energies) / (energies.std() + 1e-8)
    if cls is GRPOObjective:
        ratio, adv = torch.exp(log_probs - old_log_probs), adv.unsqueeze(1)
    else:
        ratio = torch.exp((log_probs - old_log_probs).sum(dim=1) / log_probs.shape[1])
    clipped = torch.clamp(ratio, 0.8, 1.28)
    winner = -log_probs[torch.argmin(energies)].mean()
    return winner - min((clipped * adv).mean(), (ratio * adv).mean())


LOG_PROBS = torch.tensor([[-0.4, -0.5], [-0.1, -0.2], [-0.7, -0.8], [-0.3, -0.6]])
OLD_LOG_PROBS = LOG_PROBS + torch.tensor([[0.05, -0.02], [0.0, 0.03], [-0.04, 0.01], [0.02, 0.0]])


@pytest.mark.parametrize("cls", OBJECTIVES)
@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_one_bad_energy_leaves_the_loss_and_gradient_finite(cls, bad):
    energies = torch.tensor([*FINITE[:2], bad, FINITE[3]])
    loss, grad = _loss_and_grad(cls, energies, LOG_PROBS, OLD_LOG_PROBS)
    assert torch.isfinite(loss), f"{cls.__name__} gave loss {loss}"
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("cls", OBJECTIVES)
def test_the_bad_circuit_is_pushed_down_hardest(cls):
    """Lower energy is better, so a descent step must lower the failed circuit's
    log-probability more than any other circuit's."""
    energies = torch.tensor([*FINITE[:2], float("inf"), FINITE[3]])
    _, grad = _loss_and_grad(cls, energies, LOG_PROBS, LOG_PROBS)
    push = grad.sum(dim=1)                  # descent lowers log-probs where this is positive
    assert int(torch.argmax(push)) == 2 and push[2] > 0


@pytest.mark.parametrize("cls", OBJECTIVES)
def test_finite_groups_are_unchanged(cls):
    """The guard must not perturb the ordinary case."""
    energies = torch.tensor(FINITE)
    loss, _ = _loss_and_grad(cls, energies, LOG_PROBS, OLD_LOG_PROBS)
    assert loss == pytest.approx(float(_reference_loss(cls, energies, LOG_PROBS, OLD_LOG_PROBS)))


@pytest.mark.parametrize("cls", OBJECTIVES)
def test_nan_is_neutralised_before_winner_selection(cls):
    """The supervised winner term must not reinforce the circuit whose solve failed."""
    log_probs = torch.tensor([
        [-0.4, -0.5],
        [-0.1, -0.2],
        [-0.7, -0.8],
        [-0.3, -0.6],
    ])
    old_log_probs = log_probs.clone()
    energies = torch.tensor([*FINITE[:2], float("nan"), FINITE[3]])
    replacement = torch.tensor([*FINITE[:2], max(FINITE[:2] + FINITE[3:]) + 10.0, FINITE[3]])

    context = {"gate_seqs": torch.zeros((4, 2)), "energies": energies,
               "old_log_probs": old_log_probs}
    expected_context = {**context, "energies": replacement}
    assert cls()(log_probs, context) == pytest.approx(cls()(log_probs, expected_context))


@pytest.mark.parametrize("cls", OBJECTIVES)
def test_an_entirely_bad_group_has_zero_loss_and_zero_gradient(cls):
    log_probs = torch.randn(4, 3, requires_grad=True)
    context = {
        "gate_seqs": torch.zeros((4, 3)),
        "energies": torch.tensor([float("nan"), float("inf"), float("-inf"), float("nan")]),
        "old_log_probs": torch.zeros((4, 3)),
    }
    loss = cls()(log_probs, context)
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(log_probs.grad) == 0
