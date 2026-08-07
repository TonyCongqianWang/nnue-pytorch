from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

if TYPE_CHECKING:
    from .model import NNUEModel

FAKE_QUANTIZE_EPS = 1e-5

class WeightClippingConfig(TypedDict):
    params: list[torch.Tensor]
    min_weight: float
    max_weight: float
    virtual_params: NotRequired[torch.Tensor]

def _safe_convert(value: torch.Tensor, target_dtype: torch.dtype):
    _info = torch.iinfo(target_dtype)
    # Symmetric range: [-max, max]
    min_val = -_info.max
    max_val = _info.max

    rounded_value = value.round()
    clamped_value = rounded_value.clamp(min_val, max_val)
    num_clipped = (rounded_value != clamped_value).sum()
    quantized_value = clamped_value.to(target_dtype)
    if num_clipped > 0:
        num_clipped_int = int(num_clipped.item())
        min = rounded_value.min().item()
        max = rounded_value.max().item()
        raise RuntimeError(f"Found {num_clipped_int} out of bounds values when converting to target dtype {target_dtype}. Min: {min}, max: {max}.")

    return quantized_value

def _fake_quantize_acts(value, act_scale):
    # Fake quantization with STE
    # Inference uses bitshift which is equivalent to rounding down (floor).
    # act_scale is in nnue-pytorch is `> 1`, inverted compared to normal literature.
    # will be slightly inaccurate unless all corrections factors are 1.0.
    value_hard = value.mul(act_scale).add(FAKE_QUANTIZE_EPS).floor().div(act_scale).detach()
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value

def _fake_quantize_weights(value, weight_scale):
    # Fake quantization with STE
    # In contrast to activations,
    # weights use rounding as they are
    # quantized during serialization.
    value_hard = value.mul(weight_scale).round().div(weight_scale).detach()
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value

@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    score_scale: float = 16.0
    weight_scale_l1: float = 64.0
    weight_scale_block_up: float = 128.0
    weight_scale_block_down: float = 64.0
    weight_scale_out: float = 128.0
    weight_quantized_max_hidden: float = 127.0 # i8 max
    ft_quantized_one: float = 256.0
    ft_quantized_max: float = 255.0 # limited to 255 for safe squaring within i16
    expanded_quantized_one: float = 128.0
    expanded_quantized_max: float = 127.0 # i8 max
    res_quantized_one: float = 128.0
    res_quantized_max: float = 32767.0 # i16 max

    # used to calculate correction factors
    inference_l0_division_factor: float = 512.0
    inference_l1_division_factor: float = 128.0
    inference_sqr_crelu_division_factor: float = 128.0


class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.nnue2score = config.nnue2score
        self.score_scale = config.score_scale
        self.weight_scale_l1 = config.weight_scale_l1
        self.weight_scale_block_up = config.weight_scale_block_up
        self.weight_scale_block_down = config.weight_scale_block_down
        self.weight_scale_out = config.weight_scale_out
        self.weight_quantized_max_hidden = config.weight_quantized_max_hidden
        self.expanded_quantized_one = config.expanded_quantized_one
        self.res_quantized_one = config.res_quantized_one
        self.ft_quantized_one = config.ft_quantized_one

        _i8 = torch.iinfo(torch.int8)
        self.min_threat_weight = -_i8.max / config.ft_quantized_one  # -127/256
        self.max_threat_weight = _i8.max / config.ft_quantized_one  # 127/256

        self.l0_correction_factor = config.ft_quantized_one ** 2 / config.inference_l0_division_factor / self.res_quantized_one
        l1_out_scale = config.weight_scale_l1 * (config.ft_quantized_one ** 2 / config.inference_l0_division_factor)
        self.l1_correction_factor = l1_out_scale / (config.inference_l1_division_factor * self.res_quantized_one)
        self.sqr_crelu_correction_factor = config.expanded_quantized_one / config.inference_sqr_crelu_division_factor
        self.max_ft_activation = config.ft_quantized_max / config.ft_quantized_one
        self.max_expanded_activation = config.expanded_quantized_max / config.expanded_quantized_one
        self.max_res_activation = config.res_quantized_max / config.res_quantized_one

        self.weight_scales_dict = {
            "ft_weight" : self.ft_quantized_one,
            "ft_bias" : self.ft_quantized_one,
            "ft_psqt_weight" : self.nnue2score * self.score_scale,
            "ls_l1_weight" : config.weight_scale_l1,
            "ls_l1_bias" : l1_out_scale,
            "ls_output_weight" : config.score_scale,
            "ls_output_bias" : config.score_scale * config.res_quantized_one,
        }

    def get_weight_scale(self, key: str) -> float:
        if key in self.weight_scales_dict:
            return self.weight_scales_dict[key]
        if key.endswith("_up_weight"):
            return self.config.weight_scale_block_up
        if key.endswith("_up_bias"):
            return self.config.weight_scale_block_up * self.config.res_quantized_one
        if key.endswith("_down_weight"):
            return self.config.weight_scale_block_down
        if key.endswith("_down_bias"):
            return self.config.weight_scale_block_down * self.config.expanded_quantized_one
        raise KeyError(f"Unknown quantization key: {key}")

    def clip_ft_act(self, preact):
        return torch.clamp(preact, 0.0, self.max_ft_activation)

    def clip_expanded_act(self, preact):
        return torch.clamp(preact, 0.0, self.max_expanded_activation)

    def clip_res_act(self, preact):
        return torch.clamp(preact, -self.max_res_activation, self.max_res_activation)

    def clip_ls_act(self, preact):
        return self.clip_expanded_act(preact)

    def fake_quantize_ft_act(self, preact):
        act_scale = self.config.ft_quantized_one
        return _fake_quantize_acts(preact, act_scale)

    def fake_quantize_expanded_act(self, preact):
        act_scale = self.config.expanded_quantized_one
        return _fake_quantize_acts(preact, act_scale)

    def fake_quantize_res_act(self, preact):
        act_scale = self.config.res_quantized_one
        return _fake_quantize_acts(preact, act_scale)

    def fake_quantize_ls_act(self, preact):
        return self.fake_quantize_expanded_act(preact)

    def fake_quantize_skip_act(self, preact):
        return preact

    def fake_quantize_output(self, preact: torch.Tensor) -> torch.Tensor:
        multiplier_int = int(self.config.nnue2score * self.config.score_scale)
        denominator_int = int(self.config.res_quantized_one * self.config.weight_scale_out)

        fwd_out_int = torch.round(preact * denominator_int).to(torch.int64)

        output_value_int = torch.div(
            fwd_out_int * multiplier_int,
            denominator_int,
            rounding_mode='trunc'
        )

        quantized_out = output_value_int.to(preact.dtype) / float(multiplier_int)

        return quantized_out.detach() + (preact - preact.detach())

    def fake_quantize_weights(self, tensor: torch.Tensor, key: str):
        weight_scale = self.get_weight_scale(key)
        return _fake_quantize_weights(tensor, weight_scale)

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        max_l1_w = self.weight_quantized_max_hidden / self.config.weight_scale_l1
        max_up_w = self.weight_quantized_max_hidden / self.config.weight_scale_block_up
        max_down_w = self.weight_quantized_max_hidden / self.config.weight_scale_block_down
        max_out_w = self.weight_quantized_max_hidden / self.config.score_scale

        configs: list[WeightClippingConfig] = [
            {
                "params": [model.layer_stacks.l1.linear.weight],
                "min_weight": -max_l1_w,
                "max_weight": max_l1_w,
                "virtual_params": model.layer_stacks.l1.factorized_linear.weight,
            }
        ]

        for block in model.layer_stacks.blocks:
            configs.append({
                "params": [block.up.linear.weight],
                "min_weight": -max_up_w,
                "max_weight": max_up_w,
            })
            configs.append({
                "params": [block.down.linear.weight],
                "min_weight": -max_down_w,
                "max_weight": max_down_w,
            })

        configs.append({
            "params": [model.layer_stacks.final_block.up.linear.weight],
            "min_weight": -max_up_w,
            "max_weight": max_up_w,
        })
        configs.append({
            "params": [model.layer_stacks.final_block.output.linear.weight],
            "min_weight": -max_out_w,
            "max_weight": max_out_w,
        })

        return configs

    def quantize_feature_transformer_weights(
        self,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        f_weight_export_dtype: torch.dtype = torch.int16,
        callback: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = weight.mul(self.weight_scales_dict["ft_weight"])
        weight = _safe_convert(weight, f_weight_export_dtype)
        psqt_weight = psqt_weight.mul(self.weight_scales_dict["ft_psqt_weight"])
        psqt_weight = _safe_convert(psqt_weight, torch.int32)

        if callback is not None:
            callback("ft_weight", weight)
            callback("psqt_weight", psqt_weight)

        return weight, psqt_weight

    def quantize_feature_transformer_bias(
        self,
        bias: torch.Tensor,
        callback: Callable | None = None,
    ) -> torch.Tensor:
        bias = bias.mul(self.weight_scales_dict["ft_bias"])
        bias = _safe_convert(bias, torch.int16)

        if callback is not None:
            callback("ft_bias", bias)

        return bias

    def dequantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.weight_scales_dict["ft_bias"])
        weight = weight.divide(self.weight_scales_dict["ft_weight"])
        psqt_weight = psqt_weight.divide(self.weight_scales_dict["ft_psqt_weight"])

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_key: str,
        callback: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = f"{layer_key}_weight"
        bias_key = f"{layer_key}_bias"

        bias = _safe_convert(bias.mul(self.get_weight_scale(bias_key)), torch.int32)
        weight = _safe_convert(weight.mul(self.get_weight_scale(weight_key)), torch.int8)

        if callback is not None:
            callback(bias_key, bias)
            callback(weight_key, weight)

        return bias, weight

    def dequantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = f"{layer_key}_weight"
        bias_key = f"{layer_key}_bias"

        bias = bias.divide(self.get_weight_scale(bias_key))
        weight = weight.divide(self.get_weight_scale(weight_key))

        return bias, weight

    def quantize_bias(
        self,
        bias: torch.Tensor,
        layer_key: str,
        callback: Callable | None = None,
    ) -> torch.Tensor:
        bias_key = f"{layer_key}_bias"
        bias = _safe_convert(bias.mul(self.get_weight_scale(bias_key)), torch.int32)

        if callback is not None:
            callback(bias_key, bias)

        return bias

    def dequantize_bias(
        self,
        bias: torch.Tensor,
        layer_key: str,
    ) -> torch.Tensor:
        bias_key = f"{layer_key}_bias"
        return bias.divide(self.get_weight_scale(bias_key))
