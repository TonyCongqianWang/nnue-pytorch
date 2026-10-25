"""The compiled loss must reproduce the eager pointwise chain."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import LossParams
from model.nnue import NNUE, calculate_sf_loss


def _make_inputs(device, batch, per_element_lambda=True):
    torch.manual_seed(1234)
    scorenet = (torch.randn(batch, 1, device=device) * 300).clamp(-3000, 3000)
    score = (torch.randn(batch, 1, device=device) * 200).clamp(-2000, 2000)
    outcome = torch.rand(batch, 1, device=device).round()
    if per_element_lambda:
        actual_lambda = torch.rand(batch, 1, device=device)
    else:
        actual_lambda = scorenet.new_tensor(0.65)
    return scorenet, score, outcome, actual_lambda


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("per_element_lambda", [True, False])
@pytest.mark.parametrize("endgame_reduction", [0.0, 0.1])
def test_compiled_loss_matches_eager(device, per_element_lambda, endgame_reduction):
    if device == "cuda":
        torch.cuda.set_device(0)
    params = LossParams(
        pow_exp_endgame_reduction=endgame_reduction, pow_exp_reference_error=0.12
    )
    params.pow_exp = 2.4340402395048404
    params.qp_asymmetry = 0.22866886710086187
    params.in_scaling = 295.6539508488627
    params.out_scaling = 379.98724077106635
    params.in_offset = 285.2706341467852
    params.out_offset = 289.1258344152218

    scorenet, score, outcome, actual_lambda = _make_inputs(
        torch.device(device), 4096, per_element_lambda
    )
    scorenet.requires_grad_()
    piece_count = torch.randint(2, 33, (4096,), device=device)

    eager = calculate_sf_loss(
        scorenet, score, outcome, params, actual_lambda, piece_count
    )
    (eager_grad,) = torch.autograd.grad(eager, scorenet)

    nnue = NNUE.__new__(NNUE)  # no heavy construction; only the loss is used
    nnue._sf_loss_fn = torch.compile(calculate_sf_loss)
    compiled = nnue._sf_loss_fn(
        scorenet, score, outcome, params, actual_lambda, piece_count
    )
    (compiled_grad,) = torch.autograd.grad(compiled, scorenet)

    exact = torch.equal(eager.detach(), compiled.detach())
    torch.testing.assert_close(eager.detach(), compiled.detach(), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(eager_grad, compiled_grad, rtol=1e-5, atol=1e-9)
    if not exact:
        diff = (eager.detach() - compiled.detach()).abs().max().item()
        print(f"[compiled loss] not bitwise identical; max abs diff {diff:.3e}")
