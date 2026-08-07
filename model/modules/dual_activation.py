import torch
from torch import nn

from ..quantize import QuantizationManager


class DualActivation(nn.Module):
    """
    Dual activation layer applying both squared activation and linear activation
    to a single preactivation input tensor.

    Includes a dedicated bias parameter for the squared activation path so that
    both activation branches can have independent biases.
    """

    def __init__(
        self,
        num_features: int,
        quantization: QuantizationManager,
        layer_key: str | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.quantization = quantization
        self.layer_key = layer_key
        self.sqr_bias = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        sqr_bias = self.sqr_bias
        if fake_quantize_weights and self.layer_key is not None:
            sqr_bias = self.quantization.fake_quantize_weights(
                sqr_bias, f"{self.layer_key}_bias"
            )

        sqr_act = torch.pow(x + sqr_bias, 2.0)
        if fake_quantize_acts:
            sqr_act = self.quantization.fake_quantize_expanded_act(sqr_act)

        sqr_act = sqr_act * self.quantization.sqr_crelu_correction_factor

        linear_act = x
        if fake_quantize_acts:
            linear_act = self.quantization.fake_quantize_expanded_act(linear_act)

        out = torch.cat([sqr_act, linear_act], dim=1)
        out = self.quantization.clip_expanded_act(out)
        return out
