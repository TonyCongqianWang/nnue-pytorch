"""Create explicit synthetic inference fixtures, without training or optimizer steps."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.utils.serialize import NNUEWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    m = NNUEModel("Full_Threats+PP_3Wide+K32H3^", ModelConfig())
    init_random_weights(m)
    with (args.directory / "random.nnue").open("wb") as f:
        f.write(NNUEWriter(m, verbose=False, synthetic=True).buf)
    m.training_exposure[0, ::3] = True
    m.training_exposure[1, 1::4] = True
    with (args.directory / "masked.nnue").open("wb") as f:
        f.write(NNUEWriter(m, verbose=False).buf)


if __name__ == "__main__":
    main()
