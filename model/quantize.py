from dataclasses import dataclass
from typing import Callable, NotRequired, TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import NNUEModel


class WeightClippingConfig(TypedDict):
    params: list[torch.Tensor]
    min_weight: float
    max_weight: float
    virtual_params: NotRequired[torch.Tensor]


@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_hidden: float = 64.0
    weight_scale_out: float = 16.0
    ft_quantized_one: float = 255.0
    hidden_quantized_one: float = 127.0
    # New scale for PSQT branch: one_value = 600 * 2 = 1200
    psqt_scale: float = 2.0


class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.nnue2score = config.nnue2score
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.hidden_quantized_one = config.hidden_quantized_one
        self.ft_quantized_one = config.ft_quantized_one
        self.psqt_scale = config.psqt_scale

        self.max_hidden_weight = config.hidden_quantized_one / self.weight_scale_hidden
        self.max_threat_weight = config.ft_quantized_one / 512
        self.max_out_weight = (
            config.hidden_quantized_one * self.hidden_quantized_one
        ) / (self.nnue2score * self.weight_scale_out)

        # Calculate max weight for PSQ layer
        # Target Output Scale: nnue2score * weight_scale_out (9600)
        # Input Scale (PSQT): nnue2score * psqt_scale (1200)
        # Required Weight Scale: Target / Input = 9600 / 1200 = 8.0
        # Max Weight: 127 / 8.0 = 15.875
        self.psq_weight_scale = self.weight_scale_out / self.psqt_scale
        self.max_psq_weight = self.hidden_quantized_one / self.psq_weight_scale

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        return [
            {
                "params": [model.layer_stacks.l1.linear.weight],
                "min_weight": -self.max_hidden_weight,
                "max_weight": self.max_hidden_weight,
                "virtual_params": model.layer_stacks.l1.factorized_linear.weight,
            },
            {
                "params": [model.layer_stacks.l2.linear.weight],
                "min_weight": -self.max_hidden_weight,
                "max_weight": self.max_hidden_weight,
            },
            {
                "params": [model.layer_stacks.output.linear.weight],
                "min_weight": -self.max_out_weight,
                "max_weight": self.max_out_weight,
            },
            # Added PSQ Projection clipping with correct bounds
            {
                "params": [model.psq_proj.linear.weight],
                "min_weight": -self.max_psq_weight,
                "max_weight": self.max_psq_weight,
                "virtual_params": model.psq_proj.factorized_linear.weight,
            }
        ]

    def quantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.mul(self.ft_quantized_one).round().to(torch.int16)
        weight = weight.mul(self.ft_quantized_one).round().to(torch.int16)
        
        # PSQT quantized to int16 with scale 1200 (nnue2score * 2.0)
        psqt_weight = (
            psqt_weight.mul(self.nnue2score * self.psqt_scale)
            .round()
            .to(torch.int16)
        )

        callback(bias, weight, psqt_weight)

        return bias, weight, psqt_weight

    def dequantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.ft_quantized_one)
        weight = weight.divide(self.ft_quantized_one)
        psqt_weight = psqt_weight.divide(self.nnue2score * self.psqt_scale)

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        output_layer: bool = False,
        is_psq_layer: bool = False,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if is_psq_layer:
            # PSQ Layer logic (Scale ~8.0)
            kWeightScale = self.psq_weight_scale
            # Bias scale matches the output accumulator (9600)
            kBiasScale = self.weight_scale_out * self.nnue2score
        elif output_layer:
            # Standard Output Layer logic (Scale ~75.6)
            kWeightScale = (
                self.nnue2score * self.weight_scale_out / self.hidden_quantized_one
            )
            kBiasScale = self.weight_scale_out * self.nnue2score
        else:
            # Hidden Layer logic
            kWeightScale = self.weight_scale_hidden
            kBiasScale = self.weight_scale_hidden * self.hidden_quantized_one

        kMaxWeight = self.hidden_quantized_one / kWeightScale

        bias = bias.mul(kBiasScale).round().to(torch.int32)

        clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(
            torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        )

        weight = (
            weight.clamp(-kMaxWeight, kMaxWeight)
            .mul(kWeightScale)
            .round()
            .to(torch.int8)
        )

        callback(bias, weight, clipped, total_elements, clipped_max, kMaxWeight)

        return bias, weight

    def dequantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        output_layer: bool = False,
        is_psq_layer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if is_psq_layer:
            kWeightScale = self.psq_weight_scale
            kBiasScale = self.weight_scale_out * self.nnue2score
        elif output_layer:
            kWeightScale = (
                self.nnue2score * self.weight_scale_out / self.hidden_quantized_one
            )
            kBiasScale = self.weight_scale_out * self.nnue2score
        else:
            kWeightScale = self.weight_scale_hidden
            kBiasScale = self.weight_scale_hidden * self.hidden_quantized_one

        bias = bias.divide(kBiasScale)
        weight = weight.divide(kWeightScale)

        return bias, weight
