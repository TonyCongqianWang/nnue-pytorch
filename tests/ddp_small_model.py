"""Small MoE used only by CPU DDP tests, including the real train.py entry point.

Keep native material/king context encoding and full LUT shapes, but reduce
their embedding widths and expert counts so validation is cheap on a CPU.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.nnue as nnue_module
from model.model import NNUEModel
from model.modules.layer_stacks import LayerStacks
from model.modules.router import MoERouter


class SmallNNUEModel(NNUEModel):
    def __init__(self, feature_name, config):
        super().__init__(feature_name, config)
        self.num_ls_buckets = 4
        self.router = MoERouter(
            num_fc0_experts=4, num_weight_experts=4, fc0_dim=8, dim=8
        )
        self.layer_stacks = LayerStacks(
            4,
            config,
            self.quantization,
            rank=4,
            codebook_dim=8,
            fc0_count=4,
            fc0_rank=4,
            fc0_dim=8,
        )


def install():
    nnue_module.NNUEModel = SmallNNUEModel


if __name__ == "__main__":
    install()
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "..", "train.py"), run_name="__main__"
    )
