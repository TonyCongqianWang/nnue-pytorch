"""Versioned, immutable handcrafted routing format (own material only)."""

import chess

LIVE_EXPERTS = 85
FALLBACK = 85
ROLE_EXPERTS = 86
INVALID = 255
ARCHITECTURE_HASH = 0x48503835
# fmt: off
NOMINAL_TO_COMPACT = (
    255, 255,   0, 255, 255, 255, 255, 255,   1, 255, 255, 255, 255,   2,   3, 255, 255, 255,  # band 0
      4,   5, 255, 255,   6,   7,   8,   9, 255, 255,  10,  11,  12,  13, 255, 255,  14,  15,  # band 1
     16,  17, 255,  18,  19,  20,  21,  22, 255,  23,  24,  25,  26,  27, 255,  28,  29,  30,  # band 2
     31,  32, 255,  33,  34,  35,  36,  37, 255,  38,  39,  40,  41,  42, 255,  43,  44,  45,  # band 3
     46,  47, 255,  48,  49,  50,  51,  52, 255,  53,  54,  55,  56,  57, 255,  58,  59,  60,  # band 4
    255, 255, 255,  61,  62,  63, 255, 255, 255,  64,  65,  66, 255, 255, 255,  67,  68,  69,  # band 5
    255, 255, 255,  70,  71,  72, 255, 255, 255,  73,  74,  75, 255, 255, 255,  76,  77,  78,  # band 6
    255, 255, 255,  79,  80,  81, 255, 255, 255,  82,  83,  84, 255, 255, 255, 255, 255, 255,  # band 7
)
# fmt: on


def bucket(board, color):
    p, n, b, r, q = [
        len(board.pieces(pt, color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    ]
    t = p + n + b + r + q
    if (
        p > 8
        or t > 15
        or sum(max(0, c - base) for c, base in zip((n, b, r, q), (2, 2, 2, 1))) > 8 - p
    ):
        return INVALID
    pawns = int(board.pieces(chess.PAWN, color))
    left = 0x0F0F0F0F0F0F0F0F
    layout = 2 if not pawns else 0 if pawns & left and pawns & ~left else 1
    heavy = 0 if q else 1 if r else 2
    return NOMINAL_TO_COMPACT[
        (((t // 2) * 3 + heavy) * 2 + (n + b + r + q >= 2)) * 3 + layout
    ]
