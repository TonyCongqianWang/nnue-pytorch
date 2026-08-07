from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .inverted_bottleneck_block import (
    FinalInvertedBottleneckBlock,
    InvertedBottleneckBlock,
)
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class LayerStacks(nn.Module):
    def __init__(self, count: int, config: LayerStacksConfig, quantization: QuantizationManager):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.res_dim = config.res_dim
        self.expanded_dim = config.expanded_dim
        self.num_blocks = config.num_blocks
        self.quantization = quantization

        # Factorized linear for the first layer projecting to residual stream
        self.l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.res_dim, count, quantization, "ls_l1")

        # Intermediate inverted bottleneck blocks
        self.blocks = nn.ModuleList([
            InvertedBottleneckBlock(
                self.res_dim,
                self.expanded_dim,
                count,
                quantization,
                f"ls_b{i}",
            )
            for i in range(self.num_blocks - 1)
        ])

        # Final block up-projection, dual activation, and fused output
        last_block_idx = self.num_blocks - 1
        self.final_block = FinalInvertedBottleneckBlock(
            self.res_dim,
            self.expanded_dim,
            count,
            quantization,
            f"ls_b{last_block_idx}",
        )

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        res_stream = self.l1(x, ls_indices, fake_quantize_weights)

        # Process intermediate blocks
        for block in self.blocks:
            if fake_quantize_acts:
                res_stream = self.quantization.fake_quantize_res_act(res_stream)
            res_stream = self.quantization.clip_res_act(res_stream)

            down_out = block(
                res_stream,
                ls_indices,
                fake_quantize_acts=fake_quantize_acts,
                fake_quantize_weights=fake_quantize_weights,
            )
            res_stream = res_stream + down_out

        # Final block pre-processing and fused output
        if fake_quantize_acts:
            res_stream = self.quantization.fake_quantize_res_act(res_stream)
        res_stream = self.quantization.clip_res_act(res_stream)

        l3c_ = self.final_block(
            res_stream,
            ls_indices,
            fake_quantize_acts=fake_quantize_acts,
            fake_quantize_weights=fake_quantize_weights,
        )

        l3x_ = l3c_
        if fake_quantize_acts:
            l3x_ = self.quantization.fake_quantize_output(l3x_)

        assert l3x_.shape[1] == 1, f"Expected output shape (batch_size, 1), got {l3x_.shape}"
        return l3x_

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.l1.zero_virtual_weights()

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[list[nn.Linear], None, None]:
        for i in range(self.count):
            bucket_layers = [self.l1.at_index(i)]
            for block in self.blocks:
                bucket_layers.extend([block.up.at_index(i), block.down.at_index(i)])
            bucket_layers.extend([self.final_block.up.at_index(i), self.final_block.output.at_index(i)])
            yield bucket_layers
