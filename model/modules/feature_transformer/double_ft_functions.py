import torch

from .sparse_linear_functions import SparseLinearFunction


def double_feature_transform(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    psqt_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    max_ft_activation: float,
    l1_size: int,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    assert l1_size % 2 == 0

    wp = SparseLinearFunction.apply(white_indices, weight, bias, backend="torch")
    bp = SparseLinearFunction.apply(black_indices, weight, bias, backend="torch")

    # Split L1 features and the skip/residual path features
    w, w_skip = torch.split(wp, [l1_size, wp.shape[1] - l1_size], dim=1)
    b, b_skip = torch.split(bp, [l1_size, bp.shape[1] - l1_size], dim=1)

    # Combine perspectives for L1 (self-multiplied CReLU)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0_ = torch.clamp(l0_, 0.0, max_ft_activation)

    l0_s = torch.split(l0_, l1_size // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    l0_ = torch.cat(l0_s1, dim=1)

    # Combine perspectives for residual skip path (perspective subtraction)
    # us = 1.0 (White to move) -> w_skip - b_skip
    # us = 0.0 (Black to move) -> b_skip - w_skip
    residual_l0 = (us * w_skip + (1.0 - us) * b_skip) - (us * b_skip + (1.0 - us) * w_skip)

    return l0_, residual_l0
