"""Isolated experimental material coverage; baseline defaults stay unchanged."""

import io
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
from model.modules.features.composition import (
    CLASSES,
    OFFSETS,
    Composition,
    composition_indices,
)
from model.utils.serialize import NNUEReader, NNUEWriter

FEATURES = DEFAULT_FEATURES + "+Composition"


def positions():
    rng = random.Random(1729)
    result = [chess.Board(), chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    board = chess.Board()
    for _ in range(400):
        if board.is_game_over() or board.fullmove_number > 100:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        result.append(board.copy())
    # Legal placements cover every presence bucket, including rare compositions.
    for bucket in range(1024):
        while True:
            board = chess.Board(None)
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            for t in range(10):
                if bucket & (1 << t):
                    squares = [
                        sq
                        for sq in chess.SQUARES
                        if board.piece_at(sq) is None
                        and (t % 5 != 0 or 0 < chess.square_rank(sq) < 7)
                    ]
                    board.set_piece_at(
                        rng.choice(squares), chess.Piece(t % 5 + 1, t < 5)
                    )
            if board.is_valid():
                break
        assert composition_indices(board, chess.WHITE)[0] == bucket
        result.append(board)
    return result


def test_packing_all_buckets():
    row = 0
    for b in range(1024):
        assert OFFSETS[b] == row
        row += b.bit_count()
        swapped = ((b & 31) << 5) | (b >> 5)
        assert (((swapped & 31) << 5) | (swapped >> 5)) == b
    assert row == 5120
    assert len(CLASSES) == row
    assert all(CLASSES.count(t) == 512 for t in range(10))


def test_native_python_parity():
    boards = positions()
    fens = [b.fen() for b in boards]
    batch = stream.get_sparse_batch_from_fens(
        "Composition", fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    expected = []
    try:
        tensors = batch.contents.get_tensors("cpu")
        for i, board in enumerate(boards):
            for color, tensor in ((chess.WHITE, tensors[2]), (chess.BLACK, tensors[3])):
                bucket, counts, ids = composition_indices(board, color)
                assert tensor[i][tensor[i] >= 0].tolist() == ids
                assert sum(counts) == len(board.piece_map()) - 2
                expected.append([bucket, *ids])
            white = composition_indices(board, chess.WHITE)
            black = composition_indices(board, chess.BLACK)
            assert black[1] == white[1][5:] + white[1][:5]
            assert black[0] == ((white[0] & 31) << 5) | (white[0] >> 5)
            assert composition_indices(board.mirror(), chess.BLACK) == white
    finally:
        stream.destroy_sparse_batch(batch)
    # Optional independent Stockfish feature implementation, compiled by tests/composition.sh.
    engine_test = os.environ.get("COMPOSITION_NATIVE_TEST")
    if engine_test:
        output = subprocess.check_output(
            [engine_test, "features"], input="\n".join(fens) + "\n", text=True
        )
        assert [
            list(map(int, line.split())) for line in output.splitlines()
        ] == expected


def test_shared_base_and_backward():
    feature = Composition(1024)
    torch.manual_seed(42)
    with torch.no_grad():
        feature.weight.uniform_(-0.01, 0.01)
    effective = feature.merged_weight()
    row = OFFSETS[1]
    torch.testing.assert_close(
        effective[row], feature.weight[row] + feature.weight[OFFSETS[1023]]
    )
    # Duplicated IDs are counts; both the correction and shared base receive that gradient.
    effective[[row, row, row]].sum().backward()
    assert torch.equal(feature.weight.grad[row], torch.full((1024,), 3.0))
    assert torch.equal(feature.weight.grad[OFFSETS[1023]], torch.full((1024,), 3.0))
    assert torch.count_nonzero(feature.weight.grad) == 2048
    exported = feature.get_export_weights()
    restored = Composition(1024)
    restored.load_export_weights(exported)
    torch.testing.assert_close(restored.merged_weight(), exported)
    restored.load_state_dict(feature.state_dict())
    torch.testing.assert_close(restored.merged_weight(), effective)


def test_preclip_bilinear_and_gradient():
    torch.manual_seed(1)
    feature = Composition(8)
    with torch.no_grad():
        feature.weight.uniform_(-0.02, 0.02)
    weight = feature.merged_weight()
    bias = torch.full((8,), 0.4, requires_grad=True)
    white = torch.tensor([[0, 0, 0, -1], [3, 3, 4, -1]], dtype=torch.int32)
    black = torch.tensor([[1, 1, -1, -1], [0, 0, -1, -1]], dtype=torch.int32)
    us = torch.tensor([[1.0], [0.0]])
    them = 1 - us
    out = double_feature_transform(
        us, them, white, black, weight, bias, 1.0, 8, "torch"
    )
    sums = []
    for ids in (white, black):
        sums.append(torch.stack([weight[row[row >= 0]].sum(0) + bias for row in ids]))
    concat = us * torch.cat(sums, 1) + them * torch.cat(sums[::-1], 1)
    chunks = concat.clamp(0, 1).chunk(4, dim=1)
    expected = torch.cat((chunks[0] * chunks[1], chunks[2] * chunks[3]), 1)
    torch.testing.assert_close(out, expected)
    got_grad = torch.autograd.grad(out.sum(), (feature.weight, bias), retain_graph=True)
    want_grad = torch.autograd.grad(expected.sum(), (feature.weight, bias))
    for got, want in zip(got_grad, want_grad):
        torch.testing.assert_close(got, want)


def test_serialization_and_overhead():
    config = ModelConfig(L1=32)
    model = NNUEModel(FEATURES, config)
    feature = model.input.features[-1]
    with torch.no_grad():
        feature.weight.uniform_(-0.03, 0.03)
    model.clip_weights(True)
    first = NNUEWriter(model, description="composition", verbose=False)
    restored = NNUEReader(io.BytesIO(first.buf), FEATURES, config).model
    second = NNUEWriter(restored, description="composition", verbose=False)
    assert first.buf == second.buf
    baseline = NNUEModel(DEFAULT_FEATURES, config)
    base_bytes = NNUEWriter(baseline, description="composition", verbose=False).buf
    assert len(first.buf) - len(base_bytes) == 5120 * 32
    assert sum(p.numel() for p in feature.parameters()) == 5120 * 32
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(first.buf), DEFAULT_FEATURES, config)


def test_material_decay_group_is_independent():
    from model.config import NNUEConfig
    from model.nnue import NNUE
    from model.optimizers import OptimizerConfig

    nnue = NNUE(
        NNUEConfig(
            features=FEATURES,
            model_config=ModelConfig(L1=8),
            optimizer_config=OptimizerConfig(
                optimizer_name="adamw",
                composition_weight_decay=4e-5,
                ft_weight_decay=0.0,
            ),
        )
    )
    nnue.configure_optimizers()
    weight = nnue.model.input.features[-1].weight
    groups = nnue.optimizer_wrapper.optimizer.param_groups
    matched = [g for g in groups if any(p is weight for p in g["params"])]
    assert len(matched) == 1
    assert matched[0]["weight_decay"] == 4e-5
    assert len(matched[0]["params"]) == 1
    all_params = [id(p) for g in groups for p in g["params"]]
    assert len(all_params) == len(set(all_params))
