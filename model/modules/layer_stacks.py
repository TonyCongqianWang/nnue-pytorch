import torch
import torch.nn.functional as F
from torch import nn

from .stacked_linear import FactorizedStackedLinear, StackedLinear


class ExpertBank(nn.Module):
    def __init__(self, count, config, quantization):
        super().__init__()
        self.count = count
        self.L1, self.L2, self.L3 = config.L1, config.L2, config.L3
        if self.L1 % 2 or self.L2 < 4 or self.L2 % 2 or self.L3 % 2:
            raise ValueError("Perspective experts require even widths and L2>=4")
        self.quantization = quantization
        self.l1 = FactorizedStackedLinear(
            self.L1, self.L2 // 2, count, quantization, "ls_l1"
        )
        self.l2 = StackedLinear(self.L2 * 2, self.L3 // 2, count, quantization, "ls_l2")
        self.output = StackedLinear(
            self.L2 * 2 + self.L3 * 2, 1, count, quantization, "ls_output"
        )
        self.output.linear.bias.data.zero_()
        for layer in self.layers():
            layer.register_buffer(
                "fallback_weight", torch.zeros(layer.out_features, layer.in_features)
            )
            layer.register_buffer("fallback_bias", torch.zeros(layer.out_features))

    def layers(self):
        return (self.l1, self.l2, self.output)

    def project(self, layer, x, ids, quantized):
        # Normal training uses upstream grouped-FC0 dispatch where supported.
        y = layer(x, ids.clamp_max(self.count - 1), quantized)
        weight, bias = layer.fallback_weight, layer.fallback_bias
        if quantized:
            weight = self.quantization.fake_quantize_weights(
                weight, layer.layer_key + "_weight"
            )
            bias = self.quantization.fake_quantize_weights(
                bias, layer.layer_key + "_bias"
            )
        if not self.training:
            y = torch.where(
                (ids == self.count).reshape(-1, 1), F.linear(x, weight, bias), y
            )
        return y

    def bias(self, layer, ids, quantized):
        bias = layer.linear.bias
        if hasattr(layer, "factorized_linear"):
            bias = bias + layer.factorized_linear.bias.repeat(self.count)
        bias = torch.cat([bias, layer.fallback_bias]).reshape(self.count + 1, -1)
        if quantized:
            bias = self.quantization.fake_quantize_weights(
                bias, layer.layer_key + "_bias"
            )
        return bias[ids]

    @torch.no_grad()
    def fallback_parameters(self, mask):
        """Reduce exposed coalesced affines on their current device.

        A dense mask avoids boolean indexing/nonzero, whose variable-sized
        result would synchronize CUDA with the host. Empty masks produce zero;
        the caller decides whether empty training history is permitted.
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
        mask = mask.to(device=self.l1.linear.weight.device, dtype=torch.bool)
        count = mask.sum().clamp_min(1)
        for layer in self.layers():
            weight = layer.linear.weight.reshape(
                self.count, layer.out_features, layer.in_features
            )
            bias = layer.linear.bias.reshape(self.count, layer.out_features)
            if hasattr(layer, "factorized_linear"):
                # Coalesce before averaging, as in ordinary expert export.
                weight = weight + layer.factorized_linear.weight
                bias = bias + layer.factorized_linear.bias
            yield (
                torch.where(mask[:, None, None], weight, 0).sum(0) / count,
                torch.where(mask[:, None], bias, 0).sum(0) / count,
            )

    @torch.no_grad()
    def export_layers(self, mask, loaded=False):
        all_layers = [
            [layer.at_index(i) for layer in self.layers()] for i in range(self.count)
        ]
        if not loaded and not any(mask):
            raise ValueError(
                "Cannot export a role with no training exposure; use explicit synthetic export"
            )
        parameters = (
            ((layer.fallback_weight, layer.fallback_bias) for layer in self.layers())
            if loaded
            else self.fallback_parameters(mask)
        )
        fallback = []
        for layer, (weight, bias) in zip(self.layers(), parameters):
            out = nn.Linear(
                layer.in_features,
                layer.out_features,
                device="cpu",
                dtype=layer.linear.weight.dtype,
            )
            out.weight.copy_(weight.cpu())
            out.bias.copy_(bias.cpu())
            fallback.append(out)
        yield from all_layers
        yield tuple(fallback)


class LayerStacks(ExpertBank):
    def __init__(self, count, config, quantization):
        super().__init__(count, config, quantization)
        self.opp = ExpertBank(count, config, quantization)

    def forward(self, x, indices, fake_quantize_acts=True, fake_quantize_weights=True):
        own, opp = indices
        q = self.quantization

        def activate(y):
            sq = y.square()
            if fake_quantize_acts:
                sq = q.fake_quantize_ls_act(sq)
                y = q.fake_quantize_ls_act(y)
            return q.clip_ls_act(torch.cat([sq * q.sqr_crelu_correction_factor, y], 1))

        a = self.project(self.l1, x, own, fake_quantize_weights)
        b = self.opp.project(self.opp.l1, x, opp, fake_quantize_weights)
        ra = a - self.bias(self.l1, own, fake_quantize_weights)
        rb = b - self.opp.bias(self.opp.l1, opp, fake_quantize_weights)
        skip = ra[:, -2:-1] - ra[:, -1:] + rb[:, -2:-1] - rb[:, -1:]
        h1 = activate(torch.cat([a, b], 1))
        a = self.project(self.l2, h1, own, fake_quantize_weights)
        b = self.opp.project(self.opp.l2, h1, opp, fake_quantize_weights)
        h2 = activate(torch.cat([a, b], 1))
        h = torch.cat([h1, h2], 1)
        a = self.project(self.output, h, own, fake_quantize_weights)
        b = self.opp.project(self.opp.output, h, opp, fake_quantize_weights)
        value = (a + b + skip) * 0.5
        if fake_quantize_acts:
            scale = q.weight_scales_dict["ls_l1_bias"]
            hard = (
                torch.trunc(
                    (
                        torch.round(a * scale)
                        + torch.round(b * scale)
                        + torch.round(skip * scale)
                    )
                    * 0.5
                )
                / scale
            )
            value = value + (hard - value).detach()
            value = q.fake_quantize_output(value)
        return value

    @torch.no_grad()
    def zero_virtual_weights(self):
        self.l1.zero_virtual_weights()
        self.opp.l1.zero_virtual_weights()

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self):
        self.l1.coalesce_weights()
        self.opp.l1.coalesce_weights()

    def get_coalesced_layer_stacks(self, masks, loaded=False):
        yield from self.export_layers(masks[0], loaded)
        yield from self.opp.export_layers(masks[1], loaded)
