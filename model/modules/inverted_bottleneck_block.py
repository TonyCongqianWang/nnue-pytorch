import torch
from torch import nn

from ..quantize import QuantizationManager
from .dual_activation import DualActivation
from .stacked_linear import StackedLinear


class InvertedBottleneckBlock(nn.Module):
    """
    Intermediate inverted bottleneck block for LayerStacks.

    Applies up-projection (res_dim -> expanded_dim), dual activation
    (squared + linear paths -> 2 * expanded_dim), and down-projection
    (2 * expanded_dim -> res_dim).
    """

    def __init__(
        self,
        res_dim: int,
        expanded_dim: int,
        count: int,
        quantization: QuantizationManager,
        layer_prefix: str,
    ):
        super().__init__()

        self.res_dim = res_dim
        self.expanded_dim = expanded_dim
        self.count = count
        self.quantization = quantization
        self.layer_prefix = layer_prefix

        self.up = StackedLinear(
            res_dim, expanded_dim, count, quantization, f"{layer_prefix}_up"
        )
        self.act = DualActivation(expanded_dim, quantization, f"{layer_prefix}_up")
        self.down = StackedLinear(
            2 * expanded_dim, res_dim, count, quantization, f"{layer_prefix}_down"
        )

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        up_out = self.up(x, ls_indices, fake_quantize_weights=fake_quantize_weights)
        act_out = self.act(
            up_out,
            fake_quantize_acts=fake_quantize_acts,
            fake_quantize_weights=fake_quantize_weights,
        )
        down_out = self.down(
            act_out, ls_indices, fake_quantize_weights=fake_quantize_weights
        )
        if fake_quantize_acts:
            down_out = self.quantization.fake_quantize_res_act(down_out)
        return down_out


class FinalInvertedBottleneckBlock(nn.Module):
    """
    Final inverted bottleneck block for LayerStacks.

    Applies up-projection (res_dim -> expanded_dim) and dual activation
    (expanded_dim -> 2 * expanded_dim), concatenated directly with the pre-processed
    residual stream (res_dim) into a fused output projection (res_dim + 2 * expanded_dim -> 1).
    """

    def __init__(
        self,
        res_dim: int,
        expanded_dim: int,
        count: int,
        quantization: QuantizationManager,
        layer_prefix: str,
    ):
        super().__init__()

        self.res_dim = res_dim
        self.expanded_dim = expanded_dim
        self.count = count
        self.quantization = quantization
        self.layer_prefix = layer_prefix

        self.up = StackedLinear(
            res_dim, expanded_dim, count, quantization, f"{layer_prefix}_up"
        )
        self.act = DualActivation(expanded_dim, quantization, f"{layer_prefix}_up")
        self.output = StackedLinear(
            res_dim + 2 * expanded_dim,
            1,
            count,
            quantization,
            "ls_output",
        )

        with torch.no_grad():
            self.output.linear.bias.zero_()

    def forward(
        self,
        res_stream: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        up_out = self.up(res_stream, ls_indices, fake_quantize_weights=fake_quantize_weights)
        act_out = self.act(
            up_out,
            fake_quantize_acts=fake_quantize_acts,
            fake_quantize_weights=fake_quantize_weights,
        )

        fused_input = torch.cat([res_stream, act_out], dim=1)
        l3c_ = self.output(fused_input, ls_indices, fake_quantize_weights=fake_quantize_weights)
        return l3c_
