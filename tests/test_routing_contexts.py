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
        wide = (
            p >= 2
            and bool(pawns & 0x0707070707070707)
            and bool(pawns & 0xE0E0E0E0E0E0E0E0)
        )
        return encode_material_state(t, p, h, wide)

    us = board.turn
    own = board.king(us)
    other = board.king(not us)
    rank = chess.square_rank(own) if us else 7 - chess.square_rank(own)
    theirs = chess.square_rank(other) if not us else 7 - chess.square_rank(other)
    f1, f2 = chess.square_file(own), chess.square_file(other)
    opposite = (f1 <= 2 and f2 >= 5) or (f2 <= 2 and f1 >= 5)
    king = 4 * (rank >= 2) + 2 * (rank > theirs) + opposite
    a, b = material(us), material(not us)
    return make_router_index(a, b), make_router_index(a, b, king)


def test_material_layout_bijection():
    assert NUM_MATERIAL_STATES == 676
    for i in range(NUM_MATERIAL_STATES):
        assert encode_material_state(*decode_material_state(i)) == i
    for args in [(0, 0, 0, 0), (1, 1, 3, 1), (1, 0, 2, 0)]:
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
    "pawns,wide",
    [
        ([], False),
        (["d3", "e3"], False),
        (["a3", "e3"], False),
        (["c3", "f3"], True),
        (["a3", "h3"], True),
        (["d3"], False),
    ],
)
def test_outer_pawn_wings(pawns, wide):
    b = chess.Board("7k/8/8/8/8/8/8/4K3 w - - 0 1")
    for sq in pawns:
        b.set_piece_at(
            chess.SQUARE_NAMES.index(sq), chess.Piece(chess.PAWN, chess.WHITE)
        )
    cs = get_context_indices(b)
    assert bool(decode_material_state(cs[0] // 676)[3]) == wide
    assert native_contexts([b.fen()]) == [cs]


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
