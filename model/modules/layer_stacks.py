import torch
from torch import nn

from .moe_linear import ExpertParameters, MoeLinear
from .router import (
    ADAPTER_RANK,
    CODEBOOK_DIM,
    FC0_ADAPTER_RANK,
    FC0_CODEBOOK_DIM,
    NUM_FC0_EXPERTS,
)


class LayerStacks(nn.Module):
    """One full-width expert per layer; FC1 and output share a tail ID."""

    def __init__(
        self,
        count,
        config,
        quantization,
        rank=ADAPTER_RANK,
        codebook_dim=CODEBOOK_DIM,
        fc0_count=NUM_FC0_EXPERTS,
        fc0_rank=FC0_ADAPTER_RANK,
        fc0_dim=FC0_CODEBOOK_DIM,
        fc0_orthogonal_up_init=False,
    ):
        super().__init__()
        if config.L2 < 4 or config.L2 % 2 or config.L3 != config.L2:
            raise ValueError("MoE8 requires equal even hidden widths, at least four")
        self.count, self.quantization = count, quantization
        self.l1 = MoeLinear(
            config.L1,
            config.L2,
            fc0_count,
            rank=fc0_rank,
            codebook_dim=fc0_dim,
            quantization=quantization,
            layer_key="ls_l1",
            dual_bias=True,
            factorized=True,
            orthogonal_up_init=fc0_orthogonal_up_init,
        )
        self.l2 = MoeLinear(
            2 * config.L2,
            config.L3,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l2",
            dual_bias=True,
        )
        self.output = MoeLinear(
            2 * (config.L2 + config.L3),
            1,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_output",
        )

    def dense_layers(self):
        return self.l1, self.l2, self.output

    def forward(
        self,
        x,
        codebook,
        fc0,
        tail,
        fake_quantize_acts=True,
        fake_quantize_weights=True,
    ):
        cb0, cbt = (None, None) if codebook is None else codebook

        def hidden(raw, bias):
            sq, lin = bias.chunk(2, dim=-1)
            sq = (raw + sq).square()
            lin = raw + lin
            if fake_quantize_acts:
                sq = self.quantization.fake_quantize_ls_act(sq)
                lin = self.quantization.fake_quantize_ls_act(lin)
            return self.quantization.clip_ls_act(
                torch.cat(
                    [sq * self.quantization.sqr_crelu_correction_factor, lin], dim=-1
                )
            )

        raw, bias = self.l1(x, cb0, fc0, fake_quantize_weights)
        a, b = raw.chunk(2, dim=-1)
        sa, sb = a[..., -2:-1] - a[..., -1:], b[..., -2:-1] - b[..., -1:]
        h1 = hidden(raw, bias)
        mixing = fc0.ndim == 2
        if mixing:
            h1 = 0.75 * h1[:, 0] + 0.25 * h1[:, 1]
            sa = 0.75 * sa[:, 0] + 0.25 * sa[:, 1]
            sb = 0.75 * sb[:, 0] + 0.25 * sb[:, 1]
        raw, bias = self.l2(h1, cbt, tail, fake_quantize_weights)
        h2 = hidden(raw, bias)
        if mixing:
            h1 = h1.unsqueeze(1).expand(-1, h2.size(1), -1)
        output, bias = self.output(
            torch.cat([h1, h2], dim=-1), cbt, tail, fake_quantize_weights
        )
        output = output + bias
        if mixing:
            top, second, third, fourth = output.unbind(1)
            output = 0.5 * top + 0.25 * second + 0.125 * (third + fourth)
        value = output + (sa + sb) * 0.5
        if fake_quantize_acts and not mixing:
            scale = self.quantization.weight_scales_dict["ls_l1_bias"]
            hard = (
                torch.trunc(
                    (
                        2 * torch.round(output * scale)
                        + torch.round(sa * scale)
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
        cb0, cbt = (None, None) if codebook is None else codebook
        if orders is None:
            orders = range(self.l1.count), range(self.count)
        for layers, cbs, order in [
            ((self.l1,), (cb0,), orders[0]),
            ((self.l2, self.output), (cbt, cbt), orders[1]),
        ]:
            params = [
                layer.get_all_weights_and_biases(cb) for layer, cb in zip(layers, cbs)
            ]
            for index in order:
                result = []
                for layer, (weights, biases) in zip(layers, params):
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
        for layer, cb in zip(
            self.dense_layers(), (codebook[0], codebook[1], codebook[1])
        ):
            layer.coalesce_weights(cb)
