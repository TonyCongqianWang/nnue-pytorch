import torch
from torch import nn

from .input_feature import InputFeature


class FullThreats(InputFeature):
    HASH = 0x2E6B9D04
    FEATURE_NAME = "Full_Threats"
    INPUT_FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128

    NUM_INPUTS = 59808
    NUM_REAL_FEATURES = 59808
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int, num_psqt_buckets: int = 8):
        super().__init__()

        self.num_outputs = num_outputs
        self.num_psqt_buckets = num_psqt_buckets
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self) -> None:
        L1 = self.num_outputs - self.num_psqt_buckets
        torch.nn.init.trunc_normal_(self.weight[:, L1:], mean=0.0, std=0.01, a=-0.03, b=0.03)

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        num_ft = self.num_outputs - self.num_psqt_buckets
        self.weight.data[:, :num_ft].clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
