from dataclasses import dataclass
from typing import Callable, Optional, TypedDict, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .model import NNUEModel

class WeightClippingConfig(TypedDict):
    params: list[torch.Tensor]
    min_weight: Optional[float]
    max_weight: Optional[float]
    virtual_params: Optional[torch.Tensor]

@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_hidden: float = 64.0
    weight_scale_out: float = 16.0
    ft_quantized_one: float = 255.0
    hidden_quantized_one: float = 127.0
    threat_scale: float = 512.0

class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.nnue2score = config.nnue2score
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.hidden_quantized_one = config.hidden_quantized_one
        self.ft_quantized_one = config.ft_quantized_one

        self.max_hidden_weight = config.hidden_quantized_one / self.weight_scale_hidden
        self.max_threat_weight = config.ft_quantized_one / config.threat_scale
        
        # Max weight for the final output layer
        self.max_out_weight = (
            config.hidden_quantized_one * self.hidden_quantized_one
        ) / (self.nnue2score * self.weight_scale_out)

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        clipping_configs: list[WeightClippingConfig] = [
            # Int8 Quantized Hidden Weights
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
            # Int32 Quantized Output and PSQ
            {
                "params": [model.layer_stacks.output.linear.weight],
                "min_weight": -self.max_out_weight,
                "max_weight": self.max_out_weight,
                "virtual_params": None,
            },
            {
                "params": [model.psq_proj.linear.weight],
                "min_weight": -self.max_out_weight,
                "max_weight": self.max_out_weight,
                "virtual_params": model.psq_proj.factorized_linear.weight,
            }
        ]
        
        # Handle intermediate layers (L2, L3)
        if hasattr(model.layer_stacks, 'l2'):
             clipping_configs.append({
                "params": [model.layer_stacks.l2.linear.weight],
                "min_weight": -self.max_hidden_weight,
                "max_weight": self.max_hidden_weight,
                "virtual_params": None,
            })
            
        return clipping_configs

    def quantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.mul(self.ft_quantized_one).round().to(torch.int16)
        weight = weight.mul(self.ft_quantized_one).round().to(torch.int16)
        psqt_weight = (
            psqt_weight.mul(self.nnue2score * self.weight_scale_out)
            .round()
            .to(torch.int32)
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
        psqt_weight = psqt_weight.divide(self.nnue2score * self.weight_scale_out)

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        output_layer: bool = False,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kWeightScaleHidden = self.weight_scale_hidden
        kWeightScaleOut = (
            self.nnue2score * self.weight_scale_out / self.hidden_quantized_one
        )
        kWeightScale = kWeightScaleOut if output_layer else kWeightScaleHidden
        kBiasScaleOut = self.weight_scale_out * self.nnue2score
        kBiasScaleHidden = self.weight_scale_hidden * self.hidden_quantized_one
        kBiasScale = kBiasScaleOut if output_layer else kBiasScaleHidden
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kWeightScaleHidden = self.weight_scale_hidden
        kWeightScaleOut = (
            self.nnue2score * self.weight_scale_out / self.hidden_quantized_one
        )
        kWeightScale = kWeightScaleOut if output_layer else kWeightScaleHidden
        kBiasScaleOut = self.weight_scale_out * self.nnue2score
        kBiasScaleHidden = self.weight_scale_hidden * self.hidden_quantized_one
        kBiasScale = kBiasScaleOut if output_layer else kBiasScaleHidden

        bias = bias.divide(kBiasScale)
        weight = weight.divide(kWeightScale)

        return bias, weight
