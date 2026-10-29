"""Isolated sparse-endgame coverage; baseline defaults and smoke stay unchanged."""

import io
import itertools
import os
import random
import subprocess
import sys

import chess
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import stream
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.endgame import Endgame, endgame_category, endgame_indices
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = DEFAULT_FEATURES + "+Endgame"


def positions():
    rng = random.Random(1729)
    boards = [chess.Board()]
    # Every category, pawn status, color, multiplicity rejection and mixed inventory.
    for counts in itertools.product(range(3), repeat=4):
        for pawns in (0, 2):
            while True:
                board = chess.Board(None)
                board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
                for t, count in enumerate((pawns, *counts), 1):
                    for _ in range(count):
                        squares = [
                            sq for sq in range(8, 56) if board.piece_at(sq) is None
                        ]
                        board.set_piece_at(
                            rng.choice(squares), chess.Piece(t, chess.WHITE)
                        )
                if board.is_valid():
                    break
            boards.extend((board, board.mirror()))
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        boards.append(board.copy())
    return boards


def test_categories():
    expected = {
        (0, 0, 0, 0): 0,
        (1, 0, 0, 0): 1,
        (0, 1, 0, 0): 2,
        (0, 0, 1, 0): 3,
        (0, 0, 0, 1): 4,
        (1, 0, 1, 0): 5,
        (0, 1, 1, 0): 6,
        (1, 1, 0, 0): 7,
    }
    for pawns in (0, 1, 8):
        for counts, cat in expected.items():
            assert endgame_category((pawns, *counts)) == cat
        assert endgame_category((pawns, 2, 0, 0, 0)) == 1
        assert endgame_category((pawns, 2, 0, 1, 0)) == -1
        assert endgame_category((pawns, 1, 1, 1, 0)) == -1


def test_native_python_engine_parity():
    boards = positions()
    # Exercise both horizontal orientations as well as perspective/color swaps.
    boards += [b.transform(chess.flip_horizontal) for b in boards]
    fens = [b.fen() for b in boards]
    batch = stream.get_sparse_batch_from_fens(
        "Endgame", fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    expected = []
    try:
        tensors = batch.contents.get_tensors("cpu")
        for i, board in enumerate(boards):
            for color, tensor in ((chess.WHITE, tensors[2]), (chess.BLACK, tensors[3])):
                ids = sorted(endgame_indices(board, color))
                assert sorted(tensor[i][tensor[i] >= 0].tolist()) == ids
                assert all(0 <= j < Endgame.NUM_INPUTS for j in ids)
                assert len(ids) <= 30
                expected.append(ids)
                assert ids == sorted(endgame_indices(board.mirror(), not color))
                assert ids == sorted(
                    endgame_indices(board.transform(chess.flip_horizontal), color)
                )
    finally:
        stream.destroy_sparse_batch(batch)
    executable = os.environ.get("ENDGAME_NATIVE_TEST")
    if executable:
        output = subprocess.check_output(
            [executable, "features"], input="\n".join(fens) + "\n", text=True
        )
        assert [
            sorted(map(int, line.split())) for line in output.splitlines()
        ] == expected


def test_forward_backward():
    feature = Endgame(1024)
    assert not torch.count_nonzero(feature.weight)
    feature.weight[[0, 64, 1024, 3583]].sum().backward()
    assert torch.count_nonzero(feature.weight.grad) == 4 * 1024
    torch.manual_seed(1)
    feature = Endgame(8)
    with torch.no_grad():
        feature.weight.uniform_(-0.02, 0.02)
    weight = feature.merged_weight()
    bias = torch.full((8,), 0.4, requires_grad=True)
    white = torch.tensor([[0, 64, -1], [1024, 3583, -1]], dtype=torch.int32)
    black = torch.tensor([[1, 65, -1], [0, -1, -1]], dtype=torch.int32)
    us = torch.tensor([[1.0], [0.0]])
    out = double_feature_transform(
        us, 1 - us, white, black, weight, bias, 1.0, 8, "torch"
    )
    sums = [
        torch.stack([weight[row[row >= 0]].sum(0) + bias for row in ids])
        for ids in (white, black)
    ]
    concat = us * torch.cat(sums, 1) + (1 - us) * torch.cat(sums[::-1], 1)
    chunks = concat.clamp(0, 1).chunk(4, dim=1)
    expected = torch.cat((chunks[0] * chunks[1], chunks[2] * chunks[3]), 1)
    torch.testing.assert_close(out, expected)
    for got, want in zip(
        torch.autograd.grad(out.sum(), (weight, bias), retain_graph=True),
        torch.autograd.grad(expected.sum(), (weight, bias)),
    ):
        torch.testing.assert_close(got, want)


def test_serialization_quantization_and_overhead():
    config = ModelConfig(L1=32)
    model = NNUEModel(FEATURES, config)
    feature = model.input.features[-1]
    with torch.no_grad():
        feature.weight.uniform_(-0.03, 0.03)
    model.clip_weights(True)
    first = NNUEWriter(model, description="endgame", verbose=False)
    restored = NNUEReader(io.BytesIO(first.buf), FEATURES, config).model
    second = NNUEWriter(restored, description="endgame", verbose=False)
    assert first.buf == second.buf
    baseline = NNUEModel(DEFAULT_FEATURES, config)
    base = NNUEWriter(baseline, description="endgame", verbose=False)
    assert len(first.buf) - len(base.buf) == 3584 * 32
    assert sum(p.numel() for p in feature.parameters()) == 3584 * 32
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(first.buf), DEFAULT_FEATURES, config)
    clone = Endgame(32)
    clone.load_state_dict(feature.state_dict())
    torch.testing.assert_close(clone.weight, feature.weight)


def test_independent_decay():
    from model.config import NNUEConfig
    from model.nnue import NNUE
    from model.optimizers import OptimizerConfig

    nnue = NNUE(
        NNUEConfig(
            features=FEATURES,
            model_config=ModelConfig(L1=8),
            optimizer_config=OptimizerConfig(
                optimizer_name="adamw", endgame_weight_decay=4e-5
            ),
        )
    )
    nnue.configure_optimizers()
    weight = nnue.model.input.features[-1].weight
    groups = nnue.optimizer_wrapper.optimizer.param_groups
    matched = [g for g in groups if any(p is weight for p in g["params"])]
    assert len(matched) == 1 and matched[0]["weight_decay"] == 4e-5
    params = [id(p) for g in groups for p in g["params"]]
    assert len(params) == len(set(params))


@pytest.mark.skipif(
    not os.environ.get("ENDGAME_ENGINE"), reason="matching engine required"
)
def test_engine_quantized_evaluation(tmp_path):
    from cross_check_eval import (
        eval_engine_batch,
        eval_model_batch,
        init_random_weights,
    )

    torch.manual_seed(42)
    config = ModelConfig()
    model = NNUEModel(FEATURES, config)
    init_random_weights(model)
    path = tmp_path / "endgame.nnue"
    path.write_bytes(NNUEWriter(model, verbose=False).buf)
    del model
    with path.open("rb") as f:
        model = NNUEReader(f, FEATURES, config).model
    boards = positions()
    boards += [b.transform(chess.flip_horizontal) for b in boards]
    fens = [b.fen() for b in boards if not b.is_check()]
    batch = stream.get_sparse_batch_from_fens(
        model.input_feature_name,
        fens,
        [0] * len(fens),
        [1] * len(fens),
        [0] * len(fens),
    )
    try:
        with torch.no_grad():
            python = eval_model_batch(model, batch, "cpu", True)
        engine = eval_engine_batch(os.environ["ENDGAME_ENGINE"], str(path), fens)
        assert len(engine) == len(fens)
        assert max(abs(a - b) for a, b in zip(python, engine)) < 1.0
    finally:
        stream.destroy_sparse_batch(batch)

    # Engine lane permutations must invert exactly during re-export.
    exported = tmp_path / "engine.nnue"
    subprocess.run(
        [os.environ["ENDGAME_ENGINE"]],
        input=f"uci\nsetoption name EvalFile value {path}\nposition startpos\neval\nexport_net {exported}\nquit\n",
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    assert exported.read_bytes() == path.read_bytes()
