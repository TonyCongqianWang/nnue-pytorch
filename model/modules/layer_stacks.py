
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

    def dense_layers(self):
        return (self.l2, self.output)


class LayerStacks(ExpertBank):
    """One full-width FC0 bank, followed by independent perspective tails."""

    def __init__(self, count, config, quantization, rank=32, codebook_dim=CODEBOOK_DIM):
        super().__init__(count, config, quantization, rank, codebook_dim)
        self.opp = ExpertBank(count, config, quantization, rank, codebook_dim)
        self.l1 = MoeLinear(
            config.L1,
            config.L2,
            2 * count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l1",
            dual_bias=True,
        )

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

        first, bias = self.l1(x, cb1[0], e1[0], fake_quantize_weights)
        a, b = first.chunk(2, dim=1)
        sqr, lin = bias.chunk(2, dim=1)
        sq_a, sq_b = sqr.chunk(2, dim=1)
        lin_a, lin_b = lin.chunk(2, dim=1)
        bias_a, bias_b = torch.cat([sq_a, lin_a], 1), torch.cat([sq_b, lin_b], 1)
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

    def dense_layers(self):
        return (self.l1, self.l2, self.output, self.opp.l2, self.opp.output)

    @torch.no_grad()
    def get_coalesced_layer_stacks(self, codebook=None, orders=None):
        cb1, cb2 = ((None, None), (None, None)) if codebook is None else codebook
        layers = self.dense_layers()
        parameters = [
            layer.get_all_weights_and_biases(cb)
            for layer, cb in zip(layers, (cb1[0], cb1[1], cb1[1], cb2[1], cb2[1]))
        ]
        if orders is None:
            orders = (range(self.l1.count), range(self.count), range(self.count))
        for i in range(self.l1.count):
            role, expert = divmod(i, self.count)
            result = []
            for pos in (0, 1 + role * 2, 2 + role * 2):
                layer = layers[pos]
                index = orders[0][i] if pos == 0 else orders[role + 1][expert]
                weights, biases = parameters[pos]
                weight = weights[
                    index * layer.out_features : (index + 1) * layer.out_features
                ]
                weight = self.quantization.clip_ls_weight(weight, layer.layer_key)
                result.append(
                    ExpertParameters(
                        weight,
                        biases[
                            index * layer.bias_features : (index + 1)
                            * layer.bias_features
                        ],
                    )
                )
            yield tuple(result)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook):
        self.l1.coalesce_weights(codebook[0][0])
        for bank, cb in ((self, codebook[0][1]), (self.opp, codebook[1][1])):
            bank.l2.coalesce_weights(cb)
            bank.output.coalesce_weights(cb)
