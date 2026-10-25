"""Check the behavioral contract of the optional endgame loss shape."""

import os
import sys
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import LossParams
from model.lambda_utils import LambdaController
from model.nnue import NNUE, calculate_sf_loss


def probability(score, params):
    return 0.5 * (
        1
        + ((score - params.in_offset) / params.in_scaling).sigmoid()
        - ((-score - params.in_offset) / params.in_scaling).sigmoid()
    )


def loss_and_gradient(params, count, residual):
    # Result-only labels let us prescribe a probability residual exactly.
    score = torch.tensor([[150.0]], dtype=torch.float64, requires_grad=True)
    target = probability(score, params).detach() - residual
    loss = calculate_sf_loss(
        score, torch.zeros_like(score), target, params, 0.0, torch.tensor([count])
    )
    (gradient,) = torch.autograd.grad(loss, score)
    return loss.detach(), gradient.detach()


def test_disabled_and_high_piece_count_preserve_existing_loss():
    baseline = LossParams()
    experimental = LossParams(pow_exp_endgame_reduction=0.1)
    # Include positions at and beyond the upper endpoint, and both residual signs.
    for residual in [-0.1, 0.01, 0.1, 0.5]:
        base_loss, base_grad = loss_and_gradient(baseline, 8, residual)
        for count in [24, 28, 32]:
            loss, grad = loss_and_gradient(experimental, count, residual)
            torch.testing.assert_close(loss, base_loss, rtol=0, atol=0)
            torch.testing.assert_close(grad, base_grad, rtol=0, atol=0)


@pytest.mark.parametrize("asymmetry", [0.0, 0.22866886710086187])
@pytest.mark.parametrize("residual_sign", [-1, 1])
@pytest.mark.parametrize("reference_error", [0.1, 0.12])
@pytest.mark.parametrize("reduction", [0.05, 0.1, 0.15])
def test_reference_gradient_is_preserved(
    asymmetry, residual_sign, reference_error, reduction
):
    residual = residual_sign * reference_error
    baseline = LossParams(qp_asymmetry=asymmetry)
    experimental = LossParams(
        qp_asymmetry=asymmetry,
        pow_exp_endgame_reduction=reduction,
        pow_exp_reference_error=reference_error,
    )
    _, base_grad = loss_and_gradient(baseline, 8, residual)
    for count in [2, 8, 12, 16, 20, 24, 32]:
        _, grad = loss_and_gradient(experimental, count, residual)
        torch.testing.assert_close(grad, base_grad, rtol=1e-12, atol=1e-15)


def test_small_errors_gain_pressure_and_outliers_lose_pressure_smoothly():
    baseline = LossParams()
    experimental = LossParams(pow_exp_endgame_reduction=0.1)
    for residual, increasing in [(0.01, False), (0.5, True)]:
        _, base_grad = loss_and_gradient(baseline, 8, residual)
        ratios = []
        for count in [2, 8, 12, 16, 20, 24, 32]:
            _, grad = loss_and_gradient(experimental, count, residual)
            ratios.append((grad / base_grad).item())
        assert ratios[0] == ratios[1]
        assert ratios[-2] == ratios[-1] == 1.0
        assert all(a <= b if increasing else a >= b for a, b in pairwise(ratios))
        expected_endgame_ratio = (abs(residual) / 0.1) ** -0.1
        assert ratios[0] == pytest.approx(expected_endgame_ratio, rel=1e-12)


def test_mixed_piece_counts_do_not_broadcast_across_samples_and_zero_is_finite():
    params = LossParams(pow_exp_endgame_reduction=0.1)
    score = torch.tensor([[0.0], [150.0], [-200.0]], requires_grad=True)
    teacher = torch.zeros_like(score)
    outcome = torch.tensor([[0.5], [0.5], [0.5]])
    counts = torch.tensor([8, 16, 32])
    loss = calculate_sf_loss(score, teacher, outcome, params, 0.7, counts)
    separate = torch.stack(
        [
            calculate_sf_loss(
                score[i : i + 1],
                teacher[i : i + 1],
                outcome[i : i + 1],
                params,
                0.7,
                counts[i : i + 1],
            )
            for i in range(3)
        ]
    ).mean()
    torch.testing.assert_close(loss, separate)
    (grad,) = torch.autograd.grad(loss, score)
    assert torch.isfinite(grad).all()
    assert grad[0].item() == 0.0


def test_loss_wrapper_passes_piece_counts():
    params = LossParams(pow_exp_endgame_reduction=0.1)
    nnue = NNUE.__new__(NNUE)
    torch.nn.Module.__init__(nnue)
    nnue.config = SimpleNamespace(loss_params=params)
    nnue.model = torch.nn.Identity()
    nnue.model.quantization = SimpleNamespace(nnue2score=1.0)
    nnue.lambda_scheduler = LambdaController()
    nnue.max_steps = 100
    counts = torch.tensor([8, 16, 32])
    score = torch.tensor([[0.0], [150.0], [-200.0]])
    outcome = torch.full_like(score, 0.5)
    batch = (None, None, None, None, outcome, torch.zeros_like(score), counts, None)
    actual = nnue.compute_loss_with_scorenet(score, batch, 0)
    expected = calculate_sf_loss(score, batch[5], outcome, params, 1.0, counts)
    torch.testing.assert_close(actual, expected)


def test_missing_piece_counts_fail_only_when_enabled():
    score = torch.zeros(3, 1)
    outcome = torch.full_like(score, 0.5)
    assert calculate_sf_loss(score, score, outcome, LossParams(), 0.7) == 0
    with pytest.raises(ValueError, match="piece_count is required"):
        calculate_sf_loss(
            score, score, outcome, LossParams(pow_exp_endgame_reduction=0.1), 0.7
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pow_exp_endgame_reduction": -0.1},
        {"pow_exp_endgame_reduction": float("nan")},
        {"pow_exp": 1.05, "pow_exp_endgame_reduction": 0.1},
        {"pow_exp_endgame_reduction": 0.1, "pow_exp_reference_error": 0.0},
        {"pow_exp_endgame_reduction": 0.1, "pow_exp_reference_error": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LossParams(**kwargs)
