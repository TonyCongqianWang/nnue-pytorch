import torch
from torch import nn

from .input_feature import InputFeature


class QK4(InputFeature):
    HASH = 0x41514B34
    FEATURE_NAME = "QK4"
    INPUT_FEATURE_NAME = "QK4"
    MAX_ACTIVE_FEATURES = 24
    EXPORT_WEIGHT_DTYPE = torch.int8

    # 64 buckets * 24 check ray directions * 4 contested states
    NUM_BUCKETS = 64
    NUM_RAYS = 24
    NUM_STATES = 4
    NUM_INPUTS = NUM_BUCKETS * NUM_RAYS * NUM_STATES  # 6,144
    NUM_REAL_FEATURES = NUM_INPUTS  # 6,144

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

