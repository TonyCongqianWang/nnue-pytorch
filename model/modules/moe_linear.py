import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager
from .router import CODEBOOK_DIM, ieee_matmul_precision, rms_norm

try:
    from .grouped_linear import grouped_l1, grouped_l1_preferred
except (ImportError, OSError, RuntimeError):
    grouped_l1 = None


@dataclass
class ExpertParameters:
    """Coalesced affine parameters; hidden layers have separate activation biases."""

    weight: torch.Tensor
    bias: torch.Tensor

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]


class MoeLinear(nn.Module):
    """
    Mixture-of-Experts (MoE) Linear layer with low-rank adapter projection from codebook embeddings.
    Dynamically generates expert weights and biases on the fly or accesses coalesced tables.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        count: int,
        rank: int = 32,
        codebook_dim: int = CODEBOOK_DIM,
        quantization: QuantizationManager | None = None,
        layer_key: str | None = None,
        dual_bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias_features = out_features * (2 if dual_bias else 1)
        self.count = count
        self.rank = rank
        self.codebook_dim = codebook_dim
        self.quantization = quantization
        self.layer_key = layer_key
        self.is_coalesced = False

        self.adapter_down = nn.Parameter(torch.empty(rank, codebook_dim))
        self.adapter_up = nn.Parameter(torch.empty(out_features * in_features, rank))
        self.bias_adapter = nn.Parameter(torch.empty(self.bias_features, rank))

        self.register_buffer(
            "coalesced_weight",
            torch.zeros(count * out_features, in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "coalesced_bias",
            torch.zeros(count * self.bias_features, dtype=torch.float32),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound_cb = 1.0 / math.sqrt(self.codebook_dim) if self.codebook_dim > 0 else 0.0
        nn.init.uniform_(self.adapter_down, -bound_cb, bound_cb)

        # Each column is a complete layer basis. The unit-L2 bottleneck
        # preserves its variance when blending bases; no extra rank factor.
        bound_up = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.adapter_up, -bound_up, bound_up)

        bound_bias = 1.0 / math.sqrt(self.rank) if self.rank > 0 else 0.0
        nn.init.uniform_(self.bias_adapter, -bound_bias, bound_bias)

    def _bottleneck(self, codebook: torch.Tensor) -> torch.Tensor:
        if codebook.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"codebook dimension mismatch in MoeLinear: "
                f"adapter_down expects {self.codebook_dim}, got {codebook.shape[-1]}"
            )
        return rms_norm(F.linear(F.gelu(codebook), self.adapter_down), target_norm=1.0)

    def get_expert_weight_and_bias(self, c):
        with ieee_matmul_precision():
            h = self._bottleneck(c)
            return (
                F.linear(h, self.adapter_up).reshape(
                    self.out_features, self.in_features
                ),
                F.linear(h, self.bias_adapter),
            )

    def get_delta_weights(self, codebook):
        return F.linear(self._bottleneck(codebook), self.adapter_up).view(
            -1, self.out_features, self.in_features
        )

    def get_delta_biases(self, codebook):
        return F.linear(self._bottleneck(codebook), self.bias_adapter)

    def get_all_weights_and_biases(self, codebook=None):
        if self.is_coalesced:
            return self.coalesced_weight, self.coalesced_bias
        if codebook is None:
            raise ValueError(
                "codebook must be provided when MoeLinear is not coalesced"
            )
        with ieee_matmul_precision():
            h = self._bottleneck(codebook)
            return (
                F.linear(h, self.adapter_up)
                .reshape(self.count * self.out_features, self.in_features)
                .contiguous(),
                F.linear(h, self.bias_adapter).reshape(-1).contiguous(),
            )

    def forward(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        expert_indices: torch.Tensor,
        fake_quantize_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bias-free projection and selected activation biases separately."""
        flat_w, flat_b = self.get_all_weights_and_biases(codebook)

        if fake_quantize_weights:
            if self.quantization is None or self.layer_key is None:
                raise RuntimeError(
                    "self.quantization and self.layer_key are required to use fake quantize weights."
                )
            flat_w = self.quantization.clip_ls_weight(flat_w, self.layer_key)
            flat_w = self.quantization.fake_quantize_weights(
                flat_w, f"{self.layer_key}_weight"
            )
            flat_b = self.quantization.fake_quantize_weights(
                flat_b, f"{self.layer_key}_bias"
            )

        if (
            grouped_l1 is not None
            and x.is_cuda
            and torch.version.hip is None
            and x.dtype == flat_w.dtype == flat_b.dtype == torch.float32
            and not torch.is_autocast_enabled()
            and 128 <= self.in_features <= 4096
            and self.in_features % 128 == 0
            and 1 <= self.out_features <= 128
            and 1 <= self.count <= 256
            and grouped_l1_preferred(self.count)
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] == self.in_features
            and torch.cuda.get_device_capability(x.device)[0] >= 8
        ):
            projection = grouped_l1(x, flat_w, None, expert_indices, self.count)
        else:
            projection = self.select_output(F.linear(x, flat_w), expert_indices)
        bias = F.embedding(
            expert_indices.flatten(), flat_b.view(self.count, self.bias_features)
        )
        return projection, bias

    def select_output(
        self, stacked_output: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            expert_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = expert_indices.flatten() + idx_offset

        selected_output = reshaped_output[indices]
        return selected_output

    @torch.no_grad()
    def at_index(self, index: int, codebook=None) -> ExpertParameters:
        flat_w, flat_b = self.get_all_weights_and_biases(codebook)
        w = flat_w[index * self.out_features : (index + 1) * self.out_features]
        if self.quantization is not None and self.layer_key is not None:
            w = self.quantization.clip_ls_weight(w, self.layer_key)
        return ExpertParameters(
            w, flat_b[index * self.bias_features : (index + 1) * self.bias_features]
        )

    @torch.no_grad()
    def coalesce_weights(self, codebook: torch.Tensor) -> None:
        flat_w, flat_b = self.get_all_weights_and_biases(codebook)
        if self.quantization is not None and self.layer_key is not None:
            flat_w = self.quantization.clip_ls_weight(flat_w, self.layer_key)
        self.coalesced_weight.copy_(flat_w)
        self.coalesced_bias.copy_(flat_b)
        self.is_coalesced = True

    @torch.no_grad()
    def init_coalesced_storage(self) -> None:
        self.coalesced_weight.zero_()
        self.coalesced_bias.zero_()
        self.adapter_down.zero_()
        self.adapter_up.zero_()
        self.bias_adapter.zero_()
        self.is_coalesced = True

    @torch.no_grad()
    def permute_inputs(self, permutation: Sequence[int]) -> None:
        reshaped = self.adapter_up.view(self.out_features, self.in_features, self.rank)
        permuted = reshaped[:, permutation, :].contiguous()
        self.adapter_up.copy_(
            permuted.view(self.out_features * self.in_features, self.rank)
        )
        if self.is_coalesced:
            self.coalesced_weight.copy_(self.coalesced_weight[:, permutation])
