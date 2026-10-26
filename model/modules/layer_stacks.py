from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .moe_linear import MoeLinear
from .router import CODEBOOK_DIM


class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
        config: LayerStacksConfig,
        quantization: QuantizationManager,
        rank: int = 32,
        codebook_dim: int = CODEBOOK_DIM,
    ):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization
        self.rank = rank
        self.codebook_dim = codebook_dim

        self.l1 = MoeLinear(
            2 * self.L1 // 2,
            self.L2,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l1",
        )
        self.l2 = MoeLinear(
            self.L2 * 2,
            self.L3,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l2",
        )
        self.output = MoeLinear(
            self.L2 * 2 + self.L3 * 2,
            1,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_output",
        )

        with torch.no_grad():
            self.output.bias_adapter.zero_()

    def get_expert_deltas(self, codebook: torch.Tensor) -> list[torch.Tensor]:
        return [
            self.l1.get_delta_weights(codebook),
            self.l2.get_delta_weights(codebook),
            self.output.get_delta_weights(codebook),
        ]

    def forward_head(
        self,
        x,
        codebook,
        expert_indices,
        dyn_biases,
        fake_quantize_acts=True,
        fake_quantize_weights=True,
    ):
        """Counterfactual with the same complete route on both sides."""
        return self.forward(
            x,
            codebook,
            expert_indices,
            expert_indices,
            dyn_biases,
            dyn_biases,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def forward(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        e1: torch.Tensor,
        e2: torch.Tensor,
        dyn_biases_1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dyn_biases_2: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        def average(a, b, layer_key):
            value = (a + b) * 0.5
            if fake_quantize_acts:
                # C++ averages integer affine accumulators with truncation toward zero.
                # Round each operand first to remove floating-point matmul noise.
                scale = self.quantization.weight_scales_dict[f"{layer_key}_bias"]
                quantized = (
                    torch.trunc((torch.round(a * scale) + torch.round(b * scale)) * 0.5)
                    / scale
                )
                value = value + (quantized - value).detach()
            return value

        def activate(sqr, lin):
            sqr = sqr.square()
            if fake_quantize_acts:
                sqr = self.quantization.fake_quantize_ls_act(sqr)
                lin = self.quantization.fake_quantize_ls_act(lin)
            sqr = sqr * self.quantization.sqr_crelu_correction_factor
            return self.quantization.clip_ls_act(torch.cat([sqr, lin], dim=1))

        a = self.l1(x, codebook, e1, fake_quantize_weights)
        b = self.l1(x, codebook, e2, fake_quantize_weights)
        skip = average(a[:, -2:-1] - a[:, -1:], b[:, -2:-1] - b[:, -1:], "ls_l1")
        h1 = activate(
            average(a + dyn_biases_1[0], b + dyn_biases_2[0], "ls_l1"),
            average(a + dyn_biases_1[1], b + dyn_biases_2[1], "ls_l1"),
        )
        a = self.l2(h1, codebook, e1, fake_quantize_weights)
        b = self.l2(h1, codebook, e2, fake_quantize_weights)
        h2 = activate(
            average(a + dyn_biases_1[2], b + dyn_biases_2[2], "ls_l2"),
            average(a + dyn_biases_1[3], b + dyn_biases_2[3], "ls_l2"),
        )
        combined = torch.cat([h1, h2], dim=1)
        a = self.output(combined, codebook, e1, fake_quantize_weights)
        b = self.output(combined, codebook, e2, fake_quantize_weights)
        value = average(a, b, "ls_output") + skip
        return (
            self.quantization.fake_quantize_output(value)
            if fake_quantize_acts
            else value
        )

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
        codebook: torch.Tensor | None = None,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        if not self.l1.is_coalesced:
            if codebook is None:
                raise ValueError(
                    "codebook must be provided when layer stacks are not coalesced"
                )
            w1, b1 = self.l1.get_all_weights_and_biases(codebook)
            w2, b2 = self.l2.get_all_weights_and_biases(codebook)
            wo, bo = self.output.get_all_weights_and_biases(codebook)
            if self.quantization is not None:
                w1 = self.quantization.clip_ls_weight(w1, self.l1.layer_key)
                w2 = self.quantization.clip_ls_weight(w2, self.l2.layer_key)
                wo = self.quantization.clip_ls_weight(wo, self.output.layer_key)
        else:
            w1, b1 = self.l1.coalesced_weight, self.l1.coalesced_bias
            w2, b2 = self.l2.coalesced_weight, self.l2.coalesced_bias
            wo, bo = self.output.coalesced_weight, self.output.coalesced_bias

        for i in range(self.count):
            l1 = nn.Linear(self.l1.in_features, self.l1.out_features)
            l2 = nn.Linear(self.l2.in_features, self.l2.out_features)
            output = nn.Linear(self.output.in_features, self.output.out_features)

            l1.weight.copy_(
                w1[i * self.l1.out_features : (i + 1) * self.l1.out_features, :]
            )
            l1.bias.copy_(b1[i * self.l1.out_features : (i + 1) * self.l1.out_features])

            l2.weight.copy_(
                w2[i * self.l2.out_features : (i + 1) * self.l2.out_features, :]
            )
            l2.bias.copy_(b2[i * self.l2.out_features : (i + 1) * self.l2.out_features])

            output.weight.copy_(
                wo[i * self.output.out_features : (i + 1) * self.output.out_features, :]
            )
            output.bias.copy_(
                bo[i * self.output.out_features : (i + 1) * self.output.out_features]
            )

            yield (l1, l2, output)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook: torch.Tensor) -> None:
        self.l1.coalesce_weights(codebook)
        self.l2.coalesce_weights(codebook)
        self.output.coalesce_weights(codebook)
