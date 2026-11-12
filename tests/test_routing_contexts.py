import chess
import pytest

from data_loader import stream
from model.modules.router import (
    NUM_MATERIAL_STATES,
    decode_material_state,
    encode_material_state,
    make_router_index,
)


def get_context_indices(board):
    def material(color):
        own = board.occupied_co[color]
        t = min(15, chess.popcount(own) - 1)
        p = min(8, t, chess.popcount(board.pawns & own))
        b = board.bishops & own
        h = (
            0
            if t - p >= 1 and board.queens & own
            else 1
            if t - p >= 1 and board.rooks & own
            else 2
            if t - p >= 2 and b & chess.BB_LIGHT_SQUARES and b & chess.BB_DARK_SQUARES
            else 3
        )
        pawns = board.pawns & own
        wings = bool(pawns & 0x0707070707070707) + 2 * bool(pawns & 0xE0E0E0E0E0E0E0E0)
        return (
            encode_material_state(t, p, h, first_layer=True),
            encode_material_state(t, p, h),
            wings,
        )

    us = board.turn
    own = board.king(us)
    other = board.king(not us)
    rank = chess.square_rank(own) if us else 7 - chess.square_rank(own)
    theirs = chess.square_rank(other) if not us else 7 - chess.square_rank(other)
    f1, f2 = chess.square_file(own), chess.square_file(other)
    opposite = (f1 <= 2 and f2 >= 5) or (f2 <= 2 and f1 >= 5)
    king = 4 * (rank >= 2) + 2 * (rank > theirs) + opposite
    (a0, a, aw), (b0, b, bw) = material(us), material(not us)
    if aw == 3 or bw == 3:
        wings = 0 if aw == bw else 1 if aw == 3 else 2
    else:
        wings = 3 if aw and bw and aw != bw else 4
    return make_router_index(a0, b0), make_router_index(a, b, king, wings)


def test_material_layout_bijection():
    assert NUM_MATERIAL_STATES == 396
    for i in range(NUM_MATERIAL_STATES):
        assert encode_material_state(*decode_material_state(i)) == i
    for args in [(0, 0, 0), (1, 2, 3), (1, 0, 2)]:
        with pytest.raises(ValueError):
            encode_material_state(*args)


def native_contexts(fens):
    ptr = stream.get_sparse_batch_from_fens(
        "K32H3", fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        x = ptr.contents.get_tensors("cpu")
        return list(zip(x[-2].tolist(), x[-1].tolist()))
    finally:
        stream.destroy_sparse_batch(ptr)


@pytest.mark.parametrize(
    "king,enemy,expected",
    [
        ("h1", "g8", 0),
        ("h2", "g8", 2),
        ("c2", "g8", 3),
        ("d2", "h8", 2),
        ("h3", "g8", 6),
        ("h3", "g6", 4),
        ("h3", "g5", 4),
    ],
)
def test_king_state_native_parity(king, enemy, expected):
    b = chess.Board(None)
    b.turn = chess.WHITE
    b.set_piece_at(chess.SQUARE_NAMES.index(king), chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(
        chess.SQUARE_NAMES.index(enemy), chess.Piece(chess.KING, chess.BLACK)
    )
    cs = get_context_indices(b)
    assert cs[1] % 8 == expected
    assert native_contexts([b.fen(), b.mirror().fen()]) == [cs, cs]


@pytest.mark.parametrize(
    "ours,theirs,expected",
    [
        (["a3", "h3"], ["a6", "h6"], 0),
        (["a3", "h3"], ["h6"], 1),
        (["a3"], ["a6", "h6"], 2),
        (["a3"], ["h6"], 3),
        (["a3"], ["a6"], 4),
        (["d3"], ["h6"], 4),
        ([], ["h6"], 4),
    ],
)
def test_joint_pawn_wings(ours, theirs, expected):
    b = chess.Board("7k/8/8/8/8/8/8/4K3 w - - 0 1")
    for squares, color in ((ours, chess.WHITE), (theirs, chess.BLACK)):
        for square in squares:
            b.set_piece_at(
                chess.SQUARE_NAMES.index(square), chess.Piece(chess.PAWN, color)
            )
    cs = get_context_indices(b)
    assert cs[1] // 8 % 5 == expected
    assert native_contexts([b.fen(), b.mirror().fen()]) == [cs, cs]


def test_native_random_legal_positions_and_inversion():
    import random

    rng = random.Random(8)
    b = chess.Board()
    fens = []
    for _ in range(300):
        moves = list(b.legal_moves)
        if not moves:
            b = chess.Board()
            continue
        b.push(rng.choice(moves))
        fens.extend([b.fen(), b.mirror().fen()])
    expected = [get_context_indices(chess.Board(f)) for f in fens]
    assert native_contexts(fens) == expected
    assert all(expected[i] == expected[i + 1] for i in range(0, len(expected), 2))
