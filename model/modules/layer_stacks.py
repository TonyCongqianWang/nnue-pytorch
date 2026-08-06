from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .stacked_linear import FactorizedStackedLinear, StackedLinear


class InvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        residual_dim: int,
        expanded_dim: int,
        count: int,
        quantization: QuantizationManager,
        is_final: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.expanded_dim = expanded_dim
        self.count = count
        self.is_final = is_final
        self.quantization = quantization
        self.layer_idx = layer_idx

        # Up projection (bias-free, separate biases are held for dual activation)
        self.fc_up = StackedLinear(
            residual_dim,
            expanded_dim,
            count,
            quantization,
            f"block_{layer_idx}_up",
            bias=False,
        )

        # Independent biases for dual activation branches
        self.bias_crelu = nn.Parameter(
            torch.empty(expanded_dim * count, dtype=torch.float32)
        )
        self.bias_sqr = nn.Parameter(
            torch.empty(expanded_dim * count, dtype=torch.float32)
        )

        if not is_final:
            self.fc_down = StackedLinear(
                2 * expanded_dim,
                residual_dim,
                count,
                quantization,
                f"block_{layer_idx}_down",
                bias=True,
            )
        else:
            self.fc_final = StackedLinear(
                residual_dim + 2 * expanded_dim,
                1,
                count,
                quantization,
                f"block_{layer_idx}_final",
                bias=True,
            )

        self._init_biases_uniformly()

    @torch.no_grad()
    def _init_biases_uniformly(self) -> None:
        import math

        sigma = math.sqrt(1 / self.residual_dim)
        nn.init.uniform_(self.bias_crelu[: self.expanded_dim], -sigma, sigma)
        nn.init.uniform_(self.bias_sqr[: self.expanded_dim], -sigma, sigma)

        self.bias_crelu.copy_(
            self.bias_crelu[: self.expanded_dim].repeat(self.count)
        )
        self.bias_sqr.copy_(self.bias_sqr[: self.expanded_dim].repeat(self.count))

    def select_bias(
        self, bias_param: torch.Tensor, ls_indices: torch.Tensor
    ) -> torch.Tensor:
        reshaped = bias_param.reshape(self.count, self.expanded_dim)
        return reshaped[ls_indices.flatten()]

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> torch.Tensor:
        # x is the residual skip path input of shape (batch_size, residual_dim)

        # 1. Up-projection
        up_out = self.fc_up(x, ls_indices, fake_quantize_weights)

        # Select dual activation biases for active bucket
        bias_crelu_selected = self.select_bias(self.bias_crelu, ls_indices)
        bias_sqr_selected = self.select_bias(self.bias_sqr, ls_indices)

        if fake_quantize_weights:
            bias_crelu_selected = self.quantization.fake_quantize_weights(
                bias_crelu_selected, f"block_{self.layer_idx}_up_bias_crelu"
            )
            bias_sqr_selected = self.quantization.fake_quantize_weights(
                bias_sqr_selected, f"block_{self.layer_idx}_up_bias_sqr"
            )

        # 2. Dual Activation
        # ClippedReLU branch
        x_crelu = up_out + bias_crelu_selected
        x_crelu = torch.clamp(x_crelu, 0.0, None)

        # SqrClippedReLU branch
        x_sqr = up_out + bias_sqr_selected
        x_sqr = torch.clamp(x_sqr, 0.0, None)
        x_sqr = torch.pow(x_sqr, 2.0)

        if fake_quantize_acts:
            x_crelu = self.quantization.fake_quantize_ls_act(x_crelu)
            x_sqr = self.quantization.fake_quantize_ls_act(x_sqr)

        x_sqr = x_sqr * self.quantization.sqr_crelu_correction_factor

        # Clip dual activation to i8 max range
        dual_act = torch.cat([x_sqr, x_crelu], dim=1)
        dual_act = self.quantization.clip_ls_act(dual_act)

        if not self.is_final:
            # Down-projection: 2 * expanded_dim -> residual_dim
            down_out = self.fc_down(dual_act, ls_indices, fake_quantize_weights)
            # Skip path addition
            out = x + down_out
            return out
        else:
            # Fused output: (residual_dim + 2 * expanded_dim) -> 1
            final_input = torch.cat([x, dual_act], dim=1)
            out = self.fc_final(final_input, ls_indices, fake_quantize_weights)
            return out


class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
        config: LayerStacksConfig,
        quantization: QuantizationManager,
    ):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.residual_dim = config.residual_dim
        self.expanded_dim = config.expanded_dim
        self.num_blocks = config.num_blocks
        self.quantization = quantization

        # Map L1 features to the residual skip path using two split projections
        self.l1_to_skip_a = FactorizedStackedLinear(
            self.L1 // 2,
            self.residual_dim // 2,
            count,
            quantization,
            "ls_l1_to_skip_a",
        )
        self.l1_to_skip_b = FactorizedStackedLinear(
            self.L1 // 2,
            self.residual_dim // 2,
            count,
            quantization,
            "ls_l1_to_skip_b",
        )

        # Stacked blocks
        blocks = []
        for i in range(self.num_blocks):
            is_final = i == self.num_blocks - 1
            blocks.append(
                InvertedBottleneckBlock(
                    self.residual_dim,
                    self.expanded_dim,
                    count,
                    quantization,
                    is_final=is_final,
                    layer_idx=i,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        residual_x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        # x is self-multiplied L1 feature of shape (batch_size, L1)
        # residual_x is initial residual accumulator skip path of shape (batch_size, residual_dim)

        half_l1 = self.L1 // 2
        half_us_them = half_l1 // 2

        # Extract features for sub-accumulator A and B
        x_a = torch.cat([x[:, 0:half_us_them], x[:, half_l1:half_l1 + half_us_them]], dim=1)
        x_b = torch.cat([x[:, half_us_them:half_l1], x[:, half_l1 + half_us_them:]], dim=1)

        # Initialize the skip path: accumulator skip path + split L1 projections
        skip_a = self.l1_to_skip_a(x_a, ls_indices, fake_quantize_weights)
        skip_b = self.l1_to_skip_b(x_b, ls_indices, fake_quantize_weights)
        skip_0 = residual_x + torch.cat([skip_a, skip_b], dim=1)

        # Pipeline the skip path through all blocks
        current_skip = skip_0
        for block in self.blocks:
            current_skip = self.quantization.clip_res_act(current_skip)
            current_skip = block(
                current_skip,
                ls_indices,
                fake_quantize_acts,
                fake_quantize_weights,
            )

        # Final score output: shape (batch_size, 1)
        out = current_skip
        if fake_quantize_acts:
            out = self.quantization.fake_quantize_output(out)

        assert (
            out.shape[1] == 1
        ), f"Expected output shape (batch_size, 1), got {out.shape}"
        return out

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.l1_to_skip_a.zero_virtual_weights()
        self.l1_to_skip_b.zero_virtual_weights()

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1_to_skip_a.coalesce_weights()
        self.l1_to_skip_b.coalesce_weights()

    @torch.no_grad()
    def get_coalesced_layers_for_bucket(self, bucket_idx: int) -> dict:
        """Returns all sub-layers for a given bucket index to help serialize.py"""
        layers = {}
        # 1. L1 to skip projection
        layers["l1_to_skip_a"] = self.l1_to_skip_a.at_index(bucket_idx)
        layers["l1_to_skip_b"] = self.l1_to_skip_b.at_index(bucket_idx)

        # 2. Block layers
        block_layers = []
        for block in self.blocks:
            b_data = {}
            # fc_up weights
            b_data["fc_up"] = block.fc_up.at_index(bucket_idx)
            # Fetch bias_crelu and bias_sqr for this bucket
            reshaped_crelu = block.bias_crelu.reshape(
                self.count, block.expanded_dim
            )
            reshaped_sqr = block.bias_sqr.reshape(
                self.count, block.expanded_dim
            )
            b_data["bias_crelu"] = reshaped_crelu[bucket_idx].clone()
            b_data["bias_sqr"] = reshaped_sqr[bucket_idx].clone()

            if not block.is_final:
                b_data["fc_down"] = block.fc_down.at_index(bucket_idx)
            else:
                b_data["fc_final"] = block.fc_final.at_index(bucket_idx)
            block_layers.append(b_data)

        layers["blocks"] = block_layers
        return layers
