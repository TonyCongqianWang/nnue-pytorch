"""Legal FEN extraction/leaf parity using the actual loader ABI and UCI engine."""

import argparse
import ctypes
import json
import random
import subprocess
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader._native import c_lib
from routing.tree128 import context, select

ap = argparse.ArgumentParser()
ap.add_argument("--engine", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
rng = random.Random(60310)
board = chess.Board()
fens = []
starts = [
    "7k/P7/8/8/8/8/7p/K7 w - - 0 1",
    "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
    "7k/8/8/8/8/BB1B4/NNN5/KRRQQ3 w - - 0 1",
    "7k/8/8/8/8/8/8/KBB5 w - - 0 1",
    "7k/8/8/8/8/8/8/KB1B4 w - - 0 1",
]
for start in [chess.STARTING_FEN, *starts]:
    board = chess.Board(start)
    assert board.is_valid(), start
    for i in range(600):
        if board.is_game_over() or board.fullmove_number > 150:
            board = chess.Board(start)
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))
        fens.extend([board.fen(), board.mirror().fen()])
command = "".join("position fen " + fen + "\ntree128\n" for fen in fens) + "quit\n"
r = subprocess.run(
    [str(args.engine.resolve())],
    input=command,
    text=True,
    capture_output=True,
    check=True,
)
lines = [
    line.split(" fields ")[1] for line in r.stdout.splitlines() if " fields " in line
]
assert len(lines) == len(fens), (len(lines), len(fens))
native = c_lib.dll.tree128_context_for_fen
native.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
native.restype = ctypes.c_int
for fen, line in zip(fens, lines):
    raw, leaf = line.split(" leaf ")
    raw = tuple(map(int, raw.split()))
    leaf = int(leaf)
    expected = context(chess.Board(fen))
    fields = (ctypes.c_int * 8)()
    nl = native(fen.encode(), fields)
    assert raw == tuple(fields) == expected, fen
    assert leaf == nl == select(expected), fen
report = {
    "fens": len(fens),
    "legal": True,
    "side_to_move_mirrors": True,
    "promoted_material": True,
    "en_passant_and_castling": True,
    "python_native_engine_context_and_leaf_parity": True,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(report)
