"""Inference-only integration validation, explicit synthetic weights and legal-move episodes."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch

from data_loader import stream
from model.config import ModelConfig
from model.perspective import NOMINAL_TO_COMPACT, bucket
from model.utils.serialize import NNUEReader


def episodes(seed=617):
    rng = random.Random(seed)
    result = []
    for _ in range(16):
        board = chess.Board()
        moves = []
        for _ in range(96):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        result.append((chess.STARTING_FEN, moves))
    result += [("7k/P7/8/8/8/8/7p/K7 w - - 0 1", [f"a7a8{p}"]) for p in "qrbn"]
    result += [
        ("7k/8/8/8/8/8/P7/K7 w - - 0 1", ["0000", "0000", "a2a4", "0000"]),
        ("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", ["e2e4", "d4e3"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]),
    ]
    return result


def check(probe, net):
    eps = episodes()
    fens = []
    for fen, moves in eps:
        b = chess.Board(fen)
        fens.append(fen)
        for move in moves:
            b.push(chess.Move.from_uci(move))
            fens.append(b.fen())
    # Test both side-to-move settings and color inversion on fresh reconstruction.
    roots = []
    for fen in fens[::7]:
        b = chess.Board(fen)
        for board in (b, b.mirror()):
            if not board.was_into_check():
                roots.append(board.fen())
    full = [*eps, *[(f, []) for f in roots]]
    text = "".join(f + "\n" + " ".join(m) + "\n" for f, m in full)
    p = subprocess.run(
        [probe, net], input=text, text=True, capture_output=True, check=True
    )
    lines = p.stdout.splitlines()
    accepted = [line for line in lines if line.startswith("root ")]
    assert len(accepted) == len(full), (len(accepted), len(full), p.stdout[-1000:])
    with open(net, "rb") as file:
        m = NNUEReader(file, "Full_Threats+PP_3Wide+K32H3^", ModelConfig()).model
    rootfens = [f for f, _ in full]
    batch = stream.get_sparse_batch_from_fens(
        m.input_feature_name,
        rootfens,
        [0] * len(full),
        [1] * len(full),
        [0] * len(full),
    )
    try:
        d = batch.contents.get_tensors("cpu")
        with torch.no_grad():
            scores = m(*d[:4], *d[-2:]) * 600
        for fen, line, score in zip(rootfens, accepted, scores.flatten()):
            _, own, oe, opp, pe, engine = map(str, line.split())
            b = chess.Board(fen)
            assert int(own) == bucket(b, b.turn)
            assert int(opp) == bucket(b, not b.turn)
            expected = m.calculate_buckets(
                torch.tensor([int(own)]), torch.tensor([int(opp)])
            )
            assert (int(oe), int(pe)) == (int(expected[0]), int(expected[1]))
            assert int(torch.round(score * 16).item() / 16) == int(engine), (
                fen,
                float(score),
                engine,
            )
    finally:
        stream.destroy_sparse_batch(batch)
    return {"roots": len(full), "probe": lines[-1]}


def material(probe):
    rows = []
    for p in range(10):
        for n in range(12):
            for b in range(12):
                for r, q in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 9)]:
                    for l in range(3):
                        if (p == 0) != (l == 2) or l == 0 and p < 2:
                            continue
                        rows.append((p, n, b, r, q, l))
    inp = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    result = subprocess.run(
        [probe, "--material"], input=inp, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    assert tuple(map(int, result[0].split())) == NOMINAL_TO_COMPACT
    seen = set()
    for row, res in zip(rows, result[1:]):
        p, n, b, r, q, l = row
        t = p + n + b + r + q
        valid = (
            p <= 8
            and t <= 15
            and sum(max(0, x - base) for x, base in zip((n, b, r, q), (2, 2, 2, 1)))
            <= 8 - p
        )
        expected = (
            NOMINAL_TO_COMPACT[
                (((t // 2) * 3 + (0 if q else 1 if r else 2)) * 2 + (t - p >= 2)) * 3
                + l
            ]
            if valid
            else 255
        )
        assert int(res) == expected, (row, res, expected)
        if expected != 255:
            seen.add(expected)
    assert len(seen) == 85
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe")
    p.add_argument("net")
    p.add_argument("--masked-net")
    args = p.parse_args()
    torch.set_num_threads(2)
    result = {
        "material_cases": material(args.probe),
        "synthetic": check(args.probe, args.net),
    }
    if args.masked_net:
        result["masked"] = check(args.probe, args.masked_net)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
