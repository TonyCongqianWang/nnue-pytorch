import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch
import tyro
from tyro.conf import OmitArgPrefixes

import data_loader
import model as M

# ─── Hardcoded test positions ──────────────────────────────────────────────────
# Selected to cover different king positions / bucket regions and side-to-move
# variations.  Positions in check are filtered out automatically (engine cannot
# evaluate them).
#
# FENs are chosen to catch the most common index-mapping bugs:
#   - Starting position: symmetric, eval ~0 → catches sign / offset errors
#   - Three rank-1 king squares (e1, g1, c1): exercises different file buckets
#   - Rank-8 king (g8): catches vertical flip of the king-bucket table
#   - Black to move: exercises the side-to-move perspective flip
#   - KQ vs K endgame: exercises PSQT columns with sparse material
#   - Pawn-heavy middlegame: exercises dense pawn feature activation
KNOWN_FENS = [
    # 1. Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. Open Ruy-López, white king on e1 (rank-1, middle files)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # 3. After kingside castling, white king on g1 (rank-1, h-side)
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7",
    # 4. After queenside castling, white king on c1 (rank-1, a-side)
    "2kr3r/ppp1qppp/2n2n2/3p4/3P4/2N1PN2/PPQ2PPP/2KR1B1R w - - 0 12",
    # 5. Black to move — perspective flip
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    # 6. White king on h8-area (rank-8 region) — catches vertical bucket flip
    "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
    # 7. KQ vs K — exercises PSQT columns
    "8/8/4k3/8/8/4K3/8/7Q w - - 0 1",
    # 8. Pawn-heavy middlegame
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]

# Quantization noise between fake-quantized Python eval and engine eval is
# typically 1-2 internal units.  A threshold of 2 is tight enough to catch
# any real index-mapping bug (which causes errors of 50-200+ units) while
# tolerating normal rounding.
DEFAULT_MAX_ERROR = 2

re_nnue_eval = re.compile(
    r"NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
)


@dataclass(frozen=True)
class SanityConfig:
    engine: str
    """Path to the Stockfish engine binary."""

    net: str
    """Path to the serialized .nnue net file."""

    checkpoint: str | None = None
    """Optional .ckpt checkpoint.  When provided it is used for the Python-side
    model evaluation instead of re-reading the .nnue file."""

    device: Literal["cuda", "mps", "cpu"] = "cpu"
    """Torch device for model evaluation."""

    max_error: int = DEFAULT_MAX_ERROR
    """Maximum allowed absolute error per FEN (internal units)."""


@dataclass(frozen=True)
class CliConfig:
    sanity_config: OmitArgPrefixes[SanityConfig]
    nnue_lightning_config: OmitArgPrefixes[M.NNUELightningConfig]


# ─── Model loading ─────────────────────────────────────────────────────────────


def _load_from_nnue(net_path: str, config: M.NNUELightningConfig) -> M.NNUEModel:
    with open(net_path, "rb") as f:
        reader = M.NNUEReader(f, config.features, config.model_config)
    return reader.model


def _load_from_checkpoint(ckpt_path: str, config: M.NNUELightningConfig) -> M.NNUEModel:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    nnue = M.NNUE(config=config)
    nnue.load_state_dict(checkpoint["state_dict"])
    return nnue.model


# ─── Evaluation helpers ────────────────────────────────────────────────────────


def eval_model(model: M.NNUEModel, fens: list[str], device: str) -> list[float]:
    """Return per-FEN evaluations (internal units, side-to-move positive)
    using fake-quantized forward pass to match engine precision."""
    feature_name = model.input_feature_name
    b = data_loader.get_sparse_batch_from_fens(
        feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    (us, them, white_indices, black_indices, _outcome, _score, piece_count) = (
        b.contents.get_tensors(device)
    )
    with torch.no_grad():
        evals = (
            model.forward(
                us,
                them,
                white_indices,
                black_indices,
                piece_count,
                fake_quantize_acts=True,
                fake_quantize_weights=True,
            )
            * model.quantization.nnue2score
        )
    data_loader.destroy_sparse_batch(b)
    return [v.item() for v in evals]


def eval_engine(engine_path: str, net_path: str, fens: list[str]) -> list[int]:
    """Return per-FEN NNUE evaluations (internal units) from the engine."""
    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    parts = ["uci", f"setoption name EvalFile value {net_path}"]
    for fen in fens:
        parts.extend([f"position fen {fen}", "eval"])
    parts.append("quit")
    out = engine.communicate(input="\n".join(parts))[0]

    evals = re.findall(re_nnue_eval, out)
    if len(evals) != len(fens):
        raise RuntimeError(
            f"Engine returned {len(evals)} eval(s) for {len(fens)} FEN(s).\n"
            f"Engine output:\n{out}"
        )
    return [int(v) for v in evals]


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = tyro.cli(CliConfig)
    cfg = args.sanity_config
    lcfg = args.nnue_lightning_config

    if cfg.checkpoint:
        model = _load_from_checkpoint(cfg.checkpoint, lcfg)
    else:
        model = _load_from_nnue(cfg.net, lcfg)
    model.to(cfg.device)
    model.eval()

    # Filter positions the engine cannot evaluate (king in check)
    fens = [f for f in KNOWN_FENS if not chess.Board(f).is_check()]

    W = 72
    print(f"\n{'='*W}")
    print(
        f"  FEN Sanity Check  -  {len(fens)} positions  "
        f"(max allowed error: {cfg.max_error} internal units)"
    )
    print(f"{'='*W}")

    py_evals = eval_model(model, fens, cfg.device)
    sf_evals = eval_engine(cfg.engine, cfg.net, fens)

    failures: list[tuple[str, float, int, float]] = []
    for i, (fen, py, sf) in enumerate(zip(fens, py_evals, sf_evals)):
        err = abs(py - sf)
        passed = err <= cfg.max_error
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] #{i + 1:2d}  py={py:+8.1f}  sf={sf:+8d}  err={err:6.1f}")
        print(f"        {fen}")
        if not passed:
            failures.append((fen, py, sf, err))

    print(f"{'='*W}")

    if failures:
        print(
            f"\nFAILED: {len(failures)} FEN(s) exceeded "
            f"the {cfg.max_error}-unit error threshold:\n"
        )
        for fen, py, sf, err in failures:
            print(f"    err={err:.1f}  py={py:+.1f}  sf={sf:+d}")
            print(f"    {fen}\n")
        print(
            "Hint: large systematic errors (>=50 units) typically indicate an\n"
            "index-mapping bug such as a mirrored king bucket table, wrong\n"
            "piece-type ordering, or a sign error in the orientation function."
        )
        sys.exit(1)
    else:
        print(f"\nPASSED: all {len(fens)} FENs within {cfg.max_error} internal units\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
