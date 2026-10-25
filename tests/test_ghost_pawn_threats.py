"""Independent move/attack reference, exact native parity, and training coverage."""

import functools
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
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.ghost_pawn_threats import GhostPawnThreats

FENS = [
    chess.STARTING_FEN,
    "8/8/5k2/8/4P3/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/4PK2/8/8/8 w - - 0 1",
    "4r1k1/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "6k1/8/8/8/8/2b5/3P4/4K3 w - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/8/8/4N3/4P3/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
]


@functools.cache
def geometry(color, piece_type, square):
    board = chess.Board(None)
    board.set_piece_at(square, chess.Piece(piece_type, color))
    return int(board.attacks(square))


@functools.lru_cache(maxsize=1)
def offsets():
    table = {}
    total = 0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            for square in chess.SQUARES:
                table[color, piece_type, square] = total
                if piece_type != chess.PAWN or 8 <= square < 56:
                    total += geometry(color, piece_type, square).bit_count()
    assert total == 7504
    return table


def destinations(board, color):
    result = set()
    direction = 8 if color == chess.WHITE else -8
    for square in board.pieces(chess.PAWN, color):
        candidates = set()
        one = square + direction
        if board.piece_at(one) is None:
            candidates.add(one)
            two = one + direction
            if (
                chess.square_rank(square) == (1 if color else 6)
                and board.piece_at(two) is None
            ):
                candidates.add(two)
        candidates.update(board.attacks(square) & board.occupied_co[not color])
        candidates.intersection_update(board.pin(color, square))
        result.update(candidates)
        if (
            color == board.turn
            and board.ep_square is not None
            and board.ep_square in board.attacks(square)
        ):
            move = chess.Move(square, board.ep_square)
            if board.is_en_passant(move) and board.is_legal(move):
                result.add(board.ep_square)
    return result


def reference(board, perspective):
    orientation = (0 if chess.square_file(board.king(perspective)) < 4 else 7) ^ (
        0 if perspective else 56
    )
    features = []
    for color in (chess.WHITE, chess.BLACK):
        for target in destinations(board, color):
            for square, piece in board.piece_map().items():
                if target not in board.attacks(square):
                    continue
                source, to = square ^ orientation, target ^ orientation
                attacker_color = piece.color if perspective else not piece.color
                ghost_color = int(color != perspective)
                features.append(
                    ghost_color * 7504
                    + offsets()[attacker_color, piece.piece_type, source]
                    + (
                        geometry(attacker_color, piece.piece_type, source)
                        & ((1 << to) - 1)
                    ).bit_count()
                )
    return sorted(features)


def native(fens, features="Ghost_Pawn_Threats"):
    batch = stream.get_sparse_batch_from_fens(
        features, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
        return tuple(
            [[int(i) for i in row if i >= 0] for row in tensor.tolist()]
            for tensor in tensors[2:4]
        )
    finally:
        stream.destroy_sparse_batch(batch)


@pytest.mark.parametrize("fen", FENS)
def test_reference_and_mirror(fen):
    board = chess.Board(fen)
    white, black = native([fen, board.mirror().fen()])
    assert sorted(white[0]) == reference(board, chess.WHITE)
    assert sorted(black[0]) == reference(board, chess.BLACK)
    assert sorted(white[0]) == sorted(black[1])
    assert sorted(black[0]) == sorted(white[1])
    assert all(0 <= i < 15008 for i in white[0] + black[0])


def random_fens():
    rng = random.Random(20261025)
    board = chess.Board()
    result = list(FENS)
    for _ in range(200):
        if board.is_game_over():
            board.reset()
        board.push(rng.choice(list(board.legal_moves)))
        result.append(board.fen())
    return result


def test_random_reference():
    fens = random_fens()
    white, black = native(fens)
    for fen, w, b in zip(fens, white, black):
        board = chess.Board(fen)
        assert sorted(w) == reference(board, chess.WHITE), fen
        assert sorted(b) == reference(board, chess.BLACK), fen


def test_stockfish_exact_indices():
    probe = os.environ.get("STOCKFISH_GHOST_FEATURE_PROBE")
    if not probe:
        pytest.skip("Set STOCKFISH_GHOST_FEATURE_PROBE to the Stockfish feature probe")
    fens = random_fens()
    result = subprocess.run(
        [probe],
        input="\n".join(fens) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 2 * len(fens)
    white, black = native(fens)
    for index, (w, b) in enumerate(zip(white, black)):
        for perspective, expected in enumerate((w, b)):
            values = [int(i) for i in lines[2 * index + perspective].split()]
            assert values[0] == len(expected)
            assert values[1:] == sorted(expected), fens[index]


def test_ghost_weights_receive_gradients():
    torch.manual_seed(42)
    model = NNUEModel(feature_name=DEFAULT_FEATURES, config=ModelConfig(L1=64))
    batch = stream.get_sparse_batch_from_fens(
        model.input_feature_name, FENS[:3], [0] * 3, [1] * 3, [0] * 3
    )
    try:
        us, them, white, black, _, _, _count, _queen = batch.contents.get_tensors("cpu")
        model.forward_ft(us, them, white, black, False, False).sum().backward()
        ghost = next(f for f in model.input.features if isinstance(f, GhostPawnThreats))
        assert ghost.weight.grad is not None
        assert ghost.weight.grad.abs().sum() > 0
    finally:
        stream.destroy_sparse_batch(batch)
