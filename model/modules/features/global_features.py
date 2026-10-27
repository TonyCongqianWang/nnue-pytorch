"""Global feature contract shared with Stockfish's experimental NNUE format.

Own and enemy supports have compact, independent tables. All inputs follow
the receiving perspective's horizontal PSQ orientation.
"""

import torch
from torch import nn

from .input_feature import InputFeature


class _GlobalFeature(InputFeature):
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(torch.empty(self.NUM_REAL_FEATURES, num_outputs))
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
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.detach().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.copy_(export_weight)


class OwnSupports(_GlobalFeature):
    HASH = 0x4F535031
    FEATURE_NAME = INPUT_FEATURE_NAME = "Own_Supports"
    NUM_INPUTS = NUM_REAL_FEATURES = 14952
    MAX_ACTIVE_FEATURES = 256

    def __init__(self, num_outputs: int):
        super().__init__(num_outputs * 3 // 4)


class EnemySupports(_GlobalFeature):
    HASH = 0x45535031
    FEATURE_NAME = INPUT_FEATURE_NAME = "Enemy_Supports"
    NUM_INPUTS = NUM_REAL_FEATURES = 14952
    MAX_ACTIVE_FEATURES = 256

    def __init__(self, num_outputs: int):
        super().__init__(num_outputs // 4)


class GlobalAttacks(_GlobalFeature):
    HASH = 0x47415432
    FEATURE_NAME = INPUT_FEATURE_NAME = "Global_Attacks"
    NUM_INPUTS = NUM_REAL_FEATURES = 29904
    MAX_ACTIVE_FEATURES = 256


class PawnFrontierPairs(_GlobalFeature):
    HASH = 0x50465032
    FEATURE_NAME = INPUT_FEATURE_NAME = "Pawn_FrontierPairs"
    NUM_INPUTS = NUM_REAL_FEATURES = 7 * 49 * 49
    MAX_ACTIVE_FEATURES = 7


class PawnFilePairs(_GlobalFeature):
    HASH = 0x50464332
    FEATURE_NAME = INPUT_FEATURE_NAME = "Pawn_FilePairs"
    NUM_INPUTS = NUM_REAL_FEATURES = 8 * 60
    # Also accommodates synthetic positions with more than 16 pawns.
    MAX_ACTIVE_FEATURES = 48
