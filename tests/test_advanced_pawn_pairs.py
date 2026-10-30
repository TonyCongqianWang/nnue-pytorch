"""Independent advanced-pawn pair reference, symmetry and gradient coverage."""

import os
import random
import sys

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import stream
from model.modules.features.advanced_pawn_pairs import AdvancedPawnPairs

FENS = [
    chess.STARTING_FEN,
    "4k3/8/2P5/8/8/8/8/R3K3 w - - 0 1",
    "4k3/2P5/8/8/8/8/8/R3K3 w - - 0 1",
    "4k3/8/8/2P5/8/8/8/R3K3 w - - 0 1",
    "r3k3/8/8/8/8/2p5/8/4K3 b - - 0 1",
    "r3k3/8/8/8/8/8/2p5/4K3 b - - 0 1",
    "4k3/8/2P5/8/8/8/8/RN2K2r w - - 0 1",
    "4k3/8/2P5/8/8/8/8/RNB1K2r w - - 0 1",
    "4k3/8/2P5/8/8/8/8/4K3 w - - 0 1",
    "4k3/1P6/2P5/8/8/2p5/6p1/RN2K2r w - - 0 1",
]


def reference(fen, perspective):
    b = chess.Board(fen)
    partners = [
        (sq, pc)
        for sq, pc in b.piece_map().items()
        if pc.piece_type not in (chess.PAWN, chess.KING)
    ]
    if len(partners) > 3:
        return []
    # Upstream SFNNv19 mirrors kings on files a-d to files e-h.
    orient = 7 if chess.square_file(b.king(perspective)) < 4 else 0
    orient ^= 0 if perspective else 56
    result = []
    for color in (True, False):
        for pawn in b.pieces(chess.PAWN, color):
            relative_rank = (
                chess.square_rank(pawn) if color else 7 - chess.square_rank(pawn)
            )
            if relative_rank not in (5, 6):
                continue
            pawn_id = (
                int(color != perspective) * 16
                + (relative_rank - 5) * 8
                + ((pawn ^ orient) % 8)
            )
            for square, piece in partners:
                partner_id = (
                    int(piece.color != perspective) * 4
                    + piece.piece_type
                    - chess.KNIGHT
                )
                result.append((pawn_id * 8 + partner_id) * 64 + (square ^ orient))
    return sorted(result)


def native(fens):
    batch = stream.get_sparse_batch_from_fens(
        "Advanced_Pawn_Pairs", fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
        return tuple(
            [[int(i) for i in row if i >= 0] for row in tensor.tolist()]
            for tensor in tensors[2:4]
        )
    finally:
        stream.destroy_sparse_batch(batch)


def test_reference_and_symmetry():
    fens = FENS + [chess.Board(f).mirror().fen() for f in FENS]
    sides = native(fens)
    for i, fen in enumerate(fens):
        for p, color in enumerate((True, False)):
            actual = sorted(sides[p][i])
            assert actual == reference(fen, color)
            assert len(actual) == len(set(actual)) <= 48
            assert all(0 <= idx < 16384 for idx in actual)
    for i in range(len(FENS)):
        assert sorted(sides[0][i]) == sorted(sides[1][i + len(FENS)])
        assert sorted(sides[1][i]) == sorted(sides[0][i + len(FENS)])
    assert len(sides[0][6]) == 3
    assert not sides[0][7]  # Six non-pawns, counting kings.


def test_random_positions():
    rng = random.Random(301026)
    fens = []
    for _ in range(128):
        b = chess.Board(None)
        squares = rng.sample(range(64), 2)
        b.set_piece_at(squares[0], chess.Piece(chess.KING, True))
        b.set_piece_at(squares[1], chess.Piece(chess.KING, False))
        for _ in range(rng.randrange(5)):
            available = [s for s in range(64) if not b.piece_at(s)]
            b.set_piece_at(
                rng.choice(available),
                chess.Piece(rng.randrange(2, 6), rng.choice((True, False))),
            )
        for _ in range(rng.randrange(17)):
            available = [s for s in range(8, 56) if not b.piece_at(s)]
            b.set_piece_at(
                rng.choice(available),
                chess.Piece(chess.PAWN, rng.choice((True, False))),
            )
        fens.append(b.fen())
    sides = native(fens)
    for i, fen in enumerate(fens):
        for p, color in enumerate((True, False)):
            assert sorted(sides[p][i]) == reference(fen, color)


def test_weight_gradients():
    indices = reference(FENS[1], True)
    feature = AdvancedPawnPairs(8)
    feature.weight[indices].sum().backward()
    assert torch.all(feature.weight.grad[indices] == 1)
    assert torch.count_nonzero(feature.weight.grad) == len(indices) * 8
