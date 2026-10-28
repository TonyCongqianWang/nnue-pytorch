"""Packed, nonredundant shared-base material parameterization.

Each class has 512 present buckets. Its row in bucket 1023 is the shared
base; the other 511 rows are corrections. Export folds base + correction.
"""

import torch
from torch import nn

from .input_feature import InputFeature

OFFSETS = [sum(b.bit_count() for b in range(bucket)) for bucket in range(1024)]
CLASSES = [t for b in range(1024) for t in range(10) if b & (1 << t)]


def composition_indices(board, perspective):
    counts = [
        len(board.pieces(t, color))
        for color in (perspective, not perspective)
        for t in range(1, 6)
    ]
    bucket = sum(1 << t for t, count in enumerate(counts) if count)
    row = OFFSETS[bucket]
    indices = []
    for count in counts:
        if count:
            indices.extend([row] * count)
            row += 1
    return bucket, counts, indices


class Composition(InputFeature):
    HASH = 0xC04D1024
    FEATURE_NAME = "Composition"
    INPUT_FEATURE_NAME = "Composition"
    NUM_INPUTS = 5120
    NUM_REAL_FEATURES = 5120
    MAX_ACTIVE_FEATURES = 30
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.register_buffer(
            "base_rows", torch.tensor(CLASSES) + OFFSETS[1023], persistent=False
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Zero material initially; existing spatial features remain trainable.
        self.weight.zero_()

    @torch.no_grad()
    def coalesce(self) -> None:
        # Bases and corrections remain independent; merged_weight folds them on demand.
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # No virtual parameters; the shared bases are real trainable rows.

    @torch.no_grad()
    def init_weights(self) -> None:
        pass  # Initialization is complete in reset_parameters.

    def merged_weight(self):
        return torch.cat(
            (
                self.weight[:-10] + self.weight[self.base_rows[:-10]],
                self.weight[-10:],
            )
        )

    @torch.no_grad()
    def get_export_weights(self):
        return self.merged_weight().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight):
        self.weight.copy_(export_weight)
        self.weight[:-10].sub_(export_weight[self.base_rows[:-10]])

    @torch.no_grad()
    def clip_weights(self, quantization):
        # Bounds apply to effective rows, not separately to base and corrections.
        low, high = quantization.get_ft_weight_bounds(self.EXPORT_WEIGHT_DTYPE)
        self.load_export_weights(self.merged_weight().clamp(low, high))
