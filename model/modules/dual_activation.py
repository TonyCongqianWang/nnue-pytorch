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

    def __init__(self, num_features: int, quantization: QuantizationManager):
        super().__init__()
        self.num_features = num_features
        self.quantization = quantization
        self.sqr_bias = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,
        fake_quantize_acts: bool = True,
    ) -> torch.Tensor:
        sqr_act = torch.pow(x + self.sqr_bias, 2.0)
        if fake_quantize_acts:
            sqr_act = self.quantization.fake_quantize_ls_act(sqr_act)

        sqr_act = sqr_act * self.quantization.sqr_crelu_correction_factor

        linear_act = x
        if fake_quantize_acts:
            linear_act = self.quantization.fake_quantize_ls_act(linear_act)

        out = torch.cat([sqr_act, linear_act], dim=1)
        out = self.quantization.clip_ls_act(out)
        return out
