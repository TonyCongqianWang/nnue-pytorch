from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .moe_linear import ExpertParameters, MoeLinear
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
                "MoE7 requires even widths and at least four total L2 neurons"
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
            dual_bias=True,
        )
        self.l2 = MoeLinear(
            self.L2 * 2,
            self.L3 // 2,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l2",
            dual_bias=True,
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
            layer.get_delta_weights(cb)
            for layer, cb in zip(
                (self.l1, self.l2, self.output), (codebook[0], codebook[1], codebook[1])
            )
        ]

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self, codebook=None, orders=None
    ) -> Generator[tuple[ExpertParameters, ...], None, None]:
        layers = (self.l1, self.l2, self.output)
        codebooks = (
            (None, None, None)
            if codebook is None
            else (codebook[0], codebook[1], codebook[1])
        )
        parameters = [
            layer.get_all_weights_and_biases(cb) for layer, cb in zip(layers, codebooks)
        ]
        if orders is None:
            orders = (range(self.count), range(self.count))
        for i in range(self.count):
            result = []
            for pos, (layer, (weights, biases)) in enumerate(zip(layers, parameters)):
                expert = orders[0 if pos == 0 else 1][i]
                weight = weights[
                    expert * layer.out_features : (expert + 1) * layer.out_features
                ]
                if self.quantization is not None:
                    weight = self.quantization.clip_ls_weight(weight, layer.layer_key)
                result.append(
                    ExpertParameters(
                        weight,
                        biases[
                            expert * layer.bias_features : (expert + 1)
                            * layer.bias_features
                        ],
                    )
                )
            yield tuple(result)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook: torch.Tensor) -> None:
        self.l1.coalesce_weights(codebook[0])
        self.l2.coalesce_weights(codebook[1])
        self.output.coalesce_weights(codebook[1])


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
        cb1, cb2 = ((None, None), (None, None)) if codebook is None else codebook

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

        def project_pair(layer_a, layer_b, inputs, route):
            a, bias_a = layer_a(inputs, cb1[route], e1[route], fake_quantize_weights)
            b, bias_b = layer_b(inputs, cb2[route], e2[route], fake_quantize_weights)
            return a, b, bias_a, bias_b

        def hidden(a, b, bias_a, bias_b, offset):
            # Biases belong to activation paths; the shared raw projection
            # remains available to the bias-free FC0 skip.
            a_sqr, a_lin = bias_a.chunk(2, dim=1)
            b_sqr, b_lin = bias_b.chunk(2, dim=1)
            return activate(
                torch.cat(
                    [
                        a + (a_sqr + dyn_biases_1[offset]),
                        b + (b_sqr + dyn_biases_2[offset]),
                    ],
                    1,
                ),
                torch.cat(
                    [
                        a + (a_lin + dyn_biases_1[offset + 1]),
                        b + (b_lin + dyn_biases_2[offset + 1]),
                    ],
                    1,
                ),
            )

        a, b, bias_a, bias_b = project_pair(self.l1, self.opp.l1, x, 0)
        sa = a[:, -2:-1] - a[:, -1:]
        sb = b[:, -2:-1] - b[:, -1:]
        h1 = hidden(a, b, bias_a, bias_b, 0)
        a, b, bias_a, bias_b = project_pair(self.l2, self.opp.l2, h1, 1)
        h2 = hidden(a, b, bias_a, bias_b, 2)
        combined = torch.cat([h1, h2], 1)
        a, b, bias_a, bias_b = project_pair(self.output, self.opp.output, combined, 1)
        a, b = a + bias_a, b + bias_b
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
    def get_coalesced_layer_stacks(self, codebook=None, orders=None):
        cb1, cb2 = (None, None) if codebook is None else codebook
        orders = (None, None) if orders is None else orders
        yield from super().get_coalesced_layer_stacks(cb1, orders[0])
        yield from self.opp.get_coalesced_layer_stacks(cb2, orders[1])

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook):
        super().coalesce_layer_stacks_inplace(codebook[0])
        self.opp.coalesce_layer_stacks_inplace(codebook[1])
