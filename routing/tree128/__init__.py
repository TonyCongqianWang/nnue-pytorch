"""Frozen tree128 semantic oracle. No fitted feature matrix at inference."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
IDENTITY = "c0ee18ceb89d0871c63f593ef4ff1a3f5d3d6e4d26652cbb35f5e89342c085f2"
TABLE_HASH = 0x5A7AB1E5 ^ int(IDENTITY[:8], 16)
DEFAULT_FACTORS = "tree-depth2:tree-depth4:tree-depth6"


def specification():
    data = (ROOT / "routing_tree.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != IDENTITY:
        raise ValueError("tree128 semantic identity mismatch")
    return json.loads(data)


SPEC = specification()
NODES = {n["node_id"]: n for n in SPEC["nodes"]}


def operand(d, raw):
    v = int(raw[d["raw"]])
    return int(v in d["values"]) if "values" in d else v


def condition(c, raw):
    if c["op"] == "raw_membership":
        return int(raw[c["raw_feature"]]) in c["values"]
    d = SPEC["derived_features"][c["feature"]]
    if d.get("op") == "bit_set":
        v = bool(int(raw[d["raw"]]) & d["bit"])
    elif d.get("op") == "membership":
        v = int(raw[d["raw"]]) in d["values"]
    else:
        a, b = operand(d["us"], raw), operand(d["them"], raw)
        v = {
            "sum": lambda: a + b,
            "difference": lambda: a - b,
            "absolute_difference": lambda: abs(a - b),
            "equal": lambda: a == b,
        }[d["relation"]]()
    return v <= c["threshold"]


def select(raw):
    if len(raw) != 8 or any(
        not 0 <= int(v) < f["size"] for v, f in zip(raw, SPEC["raw_features"])
    ):
        raise ValueError("tree128 context out of bounds")
    n = NODES[SPEC["root"]]
    while "leaf_id" not in n:
        n = NODES[n["yes"] if condition(n["condition"], raw) else n["no"]]
    return n["leaf_id"]


def ancestor_map(depth):
    """Canonical leaf -> ancestor at depth (early terminal leaves stay distinct)."""
    if not 1 <= depth <= SPEC["max_depth"]:
        raise ValueError("tree-depth must be in 1..10")
    result = [None] * 128

    def walk(node_id, path):
        node = NODES[node_id]
        path = path + [node_id]
        if "leaf_id" in node:
            result[node["leaf_id"]] = path[min(depth, len(path) - 1)]
        else:
            walk(node["yes"], path)
            walk(node["no"], path)

    walk(SPEC["root"], [])
    groups = {n: i for i, n in enumerate(sorted(set(result)))}
    return [groups[n] for n in result]


def context(board):
    """Independent python-chess extractor; native adapters never call this."""
    import chess

    def side(color):
        total = min(15, chess.popcount(board.occupied_co[color]) - 1)
        p, n, b, r, q = [
            len(board.pieces(pt, color))
            for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        ]
        p = min(8, total, p)
        relative = (
            0 if p == 0 else 1 if p <= total - p else 2 if p <= total - p + 2 else 3
        )
        bishops = int(board.pieces(chess.BISHOP, color))
        pair = bool(bishops & 0x55AA55AA55AA55AA) and bool(bishops & 0xAA55AA55AA55AA55)
        heavy = 2 * (0 if q else 1 if r else 2) + int(n + b + pair > r + 2 * q)
        pawns = int(board.pieces(chess.PAWN, color))
        wings = int(bool(pawns & 0x0F0F0F0F0F0F0F0F)) + 2 * int(
            bool(pawns & 0xF0F0F0F0F0F0F0F0)
        )
        rank = chess.square_rank(board.king(color)) ^ (0 if color else 7)
        return (total, relative, heavy), wings, rank

    us, uw, ur = side(board.turn)
    them, tw, tr = side(not board.turn)
    opposite = (chess.square_file(board.king(board.turn)) < 4) != (
        chess.square_file(board.king(not board.turn)) < 4
    )
    king = 4 * int(ur >= 2) + 2 * int(ur > tr) + int(opposite)
    wings = (
        0
        if uw == tw == 3
        else 1
        if uw == 3
        else 2
        if tw == 3
        else 3
        if uw and tw and uw != tw
        else 4
    )
    return (*us, *them, king, wings)
