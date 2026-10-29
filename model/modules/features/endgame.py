"""Sparse endgame piece squares; each piece color selects its own inventory.

Pawn categories: none/N/B/R/Q/NR/BR/NB. Non-pawn categories exclude none
and split by pawn presence. Mixed pairs require exactly one of each piece.
Squares use the receiving perspective's vertical and king-side horizontal flip.
"""

import torch
from torch import nn

from .input_feature import InputFeature


def endgame_category(counts):
    _, n, b, r, q = counts
    mask = bool(n) | (bool(b) << 1) | (bool(r) << 2) | (bool(q) << 3)
    if mask in (0, 1, 2, 4, 8):
        return {0: 0, 1: 1, 2: 2, 4: 3, 8: 4}[mask]
    if mask == 5 and n == r == 1:
        return 5
    if mask == 6 and b == r == 1:
        return 6
    if mask == 3 and n == b == 1:
        return 7
    return -1


def endgame_indices(board, perspective):
    flip = (0 if perspective else 56) ^ (7 if board.king(perspective) % 8 < 4 else 0)
    result = []
    for color in (perspective, not perspective):
        counts = [len(board.pieces(t, color)) for t in range(1, 6)]
        category = endgame_category(counts)
        if category < 0:
            continue
        for t in range(5):
            if t == 0:
                plane = category
            else:
                plane = (
                    8
                    + 10 * (not counts[0])
                    + (0, 0, 1, 2, 3, 4, 6, 8)[category]
                    + (category >= 5 and t == (2 if category == 7 else 3))
                )
            for sq in board.pieces(t + 1, color):
                result.append(plane * 128 + (color != perspective) * 64 + (sq ^ flip))
    return result


class Endgame(InputFeature):
    HASH = 0xE8D61408
    FEATURE_NAME = "Endgame"
    INPUT_FEATURE_NAME = "Endgame"
    NUM_INPUTS = 3584
    NUM_REAL_FEATURES = NUM_INPUTS
    MAX_ACTIVE_FEATURES = 30
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(torch.zeros(self.NUM_INPUTS, num_outputs))

    def merged_weight(self):
        return self.weight

    def coalesce(self):
        pass

    def zero_virtual_weights(self):
        pass

    def init_weights(self):
        pass

    @torch.no_grad()
    def get_export_weights(self):
        return self.weight.detach().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight):
        self.weight.copy_(export_weight)
