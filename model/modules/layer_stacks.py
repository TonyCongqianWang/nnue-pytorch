from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .moe_linear import MoeLinear
from .router import CODEBOOK_DIM


class ExpertBank(nn.Module):
    def __init__(
        self,
        count: int,
        config: LayerStacksConfig,
        quantization: QuantizationManager,
        rank: int = 32,
        codebook_dim: int = CODEBOOK_DIM,
    ):
        super().__init__()

        if (
            config.L1 <= 0
            or config.L1 % 2
            or config.L2 < 4
            or config.L2 % 2
            or config.L3 < 2
            or config.L3 % 2
        ):
            raise ValueError(
                "MoE6 requires even widths and at least four total L2 neurons"
            )
        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization
        self.rank = rank
        self.codebook_dim = codebook_dim

        self.l1 = MoeLinear(
            2 * self.L1 // 2,
            self.L2 // 2,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l1",
        )
        self.l2 = MoeLinear(
            self.L2 * 2,
            self.L3 // 2,
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

    def get_expert_deltas(self, codebook):
        return [
            layer.get_delta_weights(codebook)
            for layer in (self.l1, self.l2, self.output)
        ]

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


class LayerStacks(ExpertBank):
    """Independent role banks; every branch consumes fixed global input ordering."""

    def __init__(self, count, config, quantization, rank=32, codebook_dim=CODEBOOK_DIM):
        super().__init__(count, config, quantization, rank, codebook_dim)
        self.opp = ExpertBank(count, config, quantization, rank, codebook_dim)

    def forward(
        self,
        x,
        codebook,
        e1,
        e2,
        dyn_biases_1,
        dyn_biases_2,
        fake_quantize_acts=True,
        fake_quantize_weights=True,
    ):
        cb1, cb2 = (None, None) if codebook is None else codebook

        def activate(sqr, lin):
            sqr = sqr.square()
            if fake_quantize_acts:
                sqr = self.quantization.fake_quantize_ls_act(sqr)
                lin = self.quantization.fake_quantize_ls_act(lin)
            return self.quantization.clip_ls_act(
                torch.cat(
                    [sqr * self.quantization.sqr_crelu_correction_factor, lin], dim=1
                )
            )

        a = self.l1(x, cb1, e1, fake_quantize_weights)
        b = self.opp.l1(x, cb2, e2, fake_quantize_weights)
        sa = a[:, -2:-1] - a[:, -1:]
        sb = b[:, -2:-1] - b[:, -1:]
        h1 = activate(
            torch.cat([a + dyn_biases_1[0], b + dyn_biases_2[0]], 1),
            torch.cat([a + dyn_biases_1[1], b + dyn_biases_2[1]], 1),
        )
        a = self.l2(h1, cb1, e1, fake_quantize_weights)
        b = self.opp.l2(h1, cb2, e2, fake_quantize_weights)
        h2 = activate(
            torch.cat([a + dyn_biases_1[2], b + dyn_biases_2[2]], 1),
            torch.cat([a + dyn_biases_1[3], b + dyn_biases_2[3]], 1),
        )
        combined = torch.cat([h1, h2], 1)
        a = self.output(combined, cb1, e1, fake_quantize_weights)
        b = self.opp.output(combined, cb2, e2, fake_quantize_weights)
        value = (a + sa + b + sb) * 0.5
        if fake_quantize_acts:
            # Readouts have scale 16384; skips have scale 32768. Average
            # complete predictions once in the common L1 reference scale.
            scale = self.quantization.weight_scales_dict["ls_l1_bias"]
            hard = (
                torch.trunc(
                    (
                        torch.round(a * scale)
                        + torch.round(sa * scale)
                        + torch.round(b * scale)
                        + torch.round(sb * scale)
                    )
                    * 0.5
                )
                / scale
            )
            value = value + (hard - value).detach()
            value = self.quantization.fake_quantize_output(value)
        return value

    @torch.no_grad()
    def get_coalesced_layer_stacks(self, codebook=None):
        cb1, cb2 = (None, None) if codebook is None else codebook
        yield from super().get_coalesced_layer_stacks(cb1)
        yield from self.opp.get_coalesced_layer_stacks(cb2)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook):
        super().coalesce_layer_stacks_inplace(codebook[0])
        self.opp.coalesce_layer_stacks_inplace(codebook[1])
