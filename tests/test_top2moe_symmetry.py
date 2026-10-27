import chess
import torch

from model.modules.router import (
    NUM_MATERIAL_STATES,
    ROUTER_TABLE_SIZE,
    decode_material_state,
    decompose_router_index,
    encode_material_state,
    make_router_index,
)


def get_context_indices(board: chess.Board) -> tuple[int, int]:
    stm = board.turn
    non_stm = not stm

    def material_id(color):
        pieces = board.occupied_co[color]
        t = min(max(chess.popcount(pieces) - 1, 0), 15)
        p = min(chess.popcount(board.pawns & pieces), 8, t)
        bishops = board.bishops & pieces
        h = (
            0
            if t - p >= 1 and board.queens & pieces
            else 1
            if t - p >= 1 and board.rooks & pieces
            else 2
            if t - p >= 2
            and bishops & chess.BB_DARK_SQUARES
            and bishops & chess.BB_LIGHT_SQUARES
            else 3
        )
        return encode_material_state(t, p, h)

    mat_id_us = material_id(stm)
    # King features
    our_ksq = board.king(stm)
    enemy_ksq = board.king(non_stm)

    stm_flip = 0 if stm else 56
    our_ksq_oriented = our_ksq ^ stm_flip
    enemy_ksq_oriented = enemy_ksq ^ stm_flip

    our_castling = board.has_castling_rights(stm)
    enemy_castling = board.has_castling_rights(non_stm)

    if our_castling and our_ksq_oriented == chess.E1:
        our_king_bucket = 32
    else:
        r = our_ksq_oriented // 8
        f = our_ksq_oriented % 8
        our_king_bucket = r * 4 + (f if f < 4 else 7 - f)

    if enemy_castling and enemy_ksq_oriented == chess.E8:
        enemy_king_bucket = 64
    else:
        enemy_king_bucket = enemy_ksq_oriented

    c_us = make_router_index(mat_id_us, our_king_bucket, enemy_king_bucket)

    mat_id_them = material_id(non_stm)

    non_stm_flip = 56 if stm else 0
    their_ksq_oriented = enemy_ksq ^ non_stm_flip
    if enemy_castling and their_ksq_oriented == chess.E1:
        their_king_bucket = 32
    else:
        r = their_ksq_oriented // 8
        f = their_ksq_oriented % 8
        their_king_bucket = r * 4 + (f if f < 4 else 7 - f)

    our_ksq_oriented_them = our_ksq ^ non_stm_flip
    if our_castling and our_ksq_oriented_them == chess.E8:
        our_king_bucket_them = 64
    else:
        our_king_bucket_them = our_ksq_oriented_them

    c_them = make_router_index(mat_id_them, their_king_bucket, our_king_bucket_them)
    return c_us, c_them


def test_router_bijection():
    """Verify that material state and context indices decomposition is exact and bijective."""
    # 1. Material states bijection
    assert NUM_MATERIAL_STATES == 396
    for m in range(NUM_MATERIAL_STATES):
        t, p, h = decode_material_state(m)
        m_rec = encode_material_state(t, p, h)
        assert m == m_rec

    # 2. Context indexing bijection
    for m in [0, 100, 250, 395]:
        for k1 in [0, 16, 32]:
            for k2 in [0, 31, 64]:
                idx = make_router_index(m, k1, k2)
                assert 0 <= idx < ROUTER_TABLE_SIZE
                m_rec, k1_rec, k2_rec = decompose_router_index(torch.tensor([idx]))
                assert m_rec.item() == m
                assert k1_rec.item() == k1
                assert k2_rec.item() == k2
