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
    # 2. Open Ruy-Lopez, white king on e1 (rank-1, middle files)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # 3. After kingside castling, white king on g1 (rank-1, h-side)
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 7",
    # 4. After queenside castling, white king on c1 (rank-1, a-side)
    "2kr3r/ppp1qppp/2n2n2/3p4/3P4/2N1PN2/PPQ2PPP/2KR1B1R w - - 0 12",
    # 5. Black to move - perspective flip
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    # 6. White king on h8-area (rank-8 region) - catches vertical bucket flip
    "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
    # 7. KQ vs K - exercises PSQT columns
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
    """Path for the .nnue file.  When --random is set, this file is written by the
    script and re-used across all random-net iterations."""

    checkpoint: str | None = None
    """Optional .ckpt checkpoint.  Ignored when --random is set."""

    device: Literal["cuda", "mps", "cpu"] = "cpu"
    """Torch device for model evaluation."""

    max_error: int = DEFAULT_MAX_ERROR
    """Maximum allowed absolute error per FEN (internal units)."""

    random: bool = False
    """When set, generate weight configurations in-script instead of loading from
    --net / --checkpoint.  The generated net is written to --net before each
    iteration so the engine can load it."""

    random_nets: int = 3
    """Number of random-weight nets to generate and test.
    Iterations 0..N-2 = uniform random different seeds (increasingly larger range).
    Iteration N-2 = positive saturation (slightly above max_ft_weight, then clipped).
    Iteration N-1 = negative saturation (slightly below min_fst_weight, then clipped)."""


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


# ─── Random net generation ─────────────────────────────────────────────────────

def _fill_ft_weights(model: M.NNUEModel, fill_value: float | None, seed: int, overshoot: float) -> None:
    """Fill all FT input feature weights then clip to the valid range.

    When fill_value is None, weights are drawn from a uniform distribution in
    [-overshoot * max_ft_weight, +overshoot * max_ft_weight].
    When fill_value is not None, all weights are set to that constant.

    After filling, model.clip_weights(include_input=True) is called so that the
    result is always within the serializable i8 range.  Passing weights that are
    slightly outside the bound intentionally exercises clip_weights correctness:
    if the implementation is wrong, _safe_convert in NNUEWriter will raise.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    q = model.quantization
    ft_bound = q.max_threat_weight * overshoot

    with torch.no_grad():
        for f in model.input.features:
            if fill_value is not None:
                f.weight.data.fill_(fill_value * overshoot)
                if hasattr(f, "virtual_weight"):
                    f.virtual_weight.data.fill_(fill_value * overshoot)
            else:
                f.weight.data.uniform_(-ft_bound, ft_bound, generator=rng)
                if hasattr(f, "virtual_weight"):
                    f.virtual_weight.data.uniform_(-ft_bound, ft_bound, generator=rng)

        # 1. l1_to_skip_a and l1_to_skip_b layers
        hw = q.max_hidden_weight[0] * overshoot
        if fill_value is not None:
            model.layer_stacks.l1_to_skip_a.linear.weight.data.fill_(fill_value * hw / abs(fill_value) if fill_value != 0 else 0.0)
            model.layer_stacks.l1_to_skip_a.linear.bias.data.fill_(fill_value * hw / abs(fill_value) if fill_value != 0 else 0.0)
            model.layer_stacks.l1_to_skip_b.linear.weight.data.fill_(fill_value * hw / abs(fill_value) if fill_value != 0 else 0.0)
            model.layer_stacks.l1_to_skip_b.linear.bias.data.fill_(fill_value * hw / abs(fill_value) if fill_value != 0 else 0.0)
        else:
            model.layer_stacks.l1_to_skip_a.linear.weight.data.uniform_(-hw, hw, generator=rng)
            model.layer_stacks.l1_to_skip_a.linear.bias.data.uniform_(-hw, hw, generator=rng)
            model.layer_stacks.l1_to_skip_b.linear.weight.data.uniform_(-hw, hw, generator=rng)
            model.layer_stacks.l1_to_skip_b.linear.bias.data.uniform_(-hw, hw, generator=rng)

        # 2. Block layers
        for block in model.layer_stacks.blocks:
            hw_up = q.max_hidden_weight[0] * overshoot
            hw_down = q.max_hidden_weight[1] * overshoot
            hw_final = q.max_hidden_weight[2] * overshoot

            if fill_value is not None:
                block.fc_up.linear.weight.data.fill_(fill_value * hw_up / abs(fill_value) if fill_value != 0 else 0.0)
                block.bias_crelu.data.fill_(fill_value * hw_up)
                block.bias_sqr.data.fill_(fill_value * hw_up)
                if not block.is_final:
                    block.fc_down.linear.weight.data.fill_(fill_value * hw_down / abs(fill_value) if fill_value != 0 else 0.0)
                    block.fc_down.linear.bias.data.fill_(fill_value * hw_down / abs(fill_value) if fill_value != 0 else 0.0)
                else:
                    block.fc_final.linear.weight.data.fill_(fill_value * hw_final / abs(fill_value) if fill_value != 0 else 0.0)
                    block.fc_final.linear.bias.data.fill_(fill_value * hw_final / abs(fill_value) if fill_value != 0 else 0.0)
            else:
                block.fc_up.linear.weight.data.uniform_(-hw_up, hw_up, generator=rng)
                block.bias_crelu.data.uniform_(-hw_up, hw_up, generator=rng)
                block.bias_sqr.data.uniform_(-hw_up, hw_up, generator=rng)
                if not block.is_final:
                    block.fc_down.linear.weight.data.uniform_(-hw_down, hw_down, generator=rng)
                    block.fc_down.linear.bias.data.uniform_(-hw_down, hw_down, generator=rng)
                else:
                    block.fc_final.linear.weight.data.uniform_(-hw_final, hw_final, generator=rng)
                    block.fc_final.linear.bias.data.uniform_(-hw_final, hw_final, generator=rng)

    # clip_weights enforces all bounds.  If the clipping logic is wrong,
    # NNUEWriter._safe_convert will raise RuntimeError and the test fails cleanly.
    model.clip_weights(include_input=True)


def _iter_label(i: int, total_nets: int) -> str:
    if i < total_nets - 2:
        return f"uniform random (seed={i})"
    if i == total_nets - 2:
        return "positive saturation (+1.5 x max_ft, clipped to +max_ft)"
    return "negative saturation (-1.5 x max_ft, clipped to -max_ft)"


def _generate_and_serialize(
    config: M.NNUELightningConfig,
    lcfg: M.NNUELightningConfig,
    net_path: str,
    iteration: int,
    total_nets: int,
    device: str,
) -> M.NNUEModel:
    """Build a fresh NNUEModel, fill weights per the iteration strategy, serialize."""
    nnue = M.NNUE(config=lcfg)
    model = nnue.model
    model.to(device)
    model.eval()

    if iteration == total_nets - 2:
        fill_value = 1.0   # positive saturation
        overshoot = 1.5   # 1.5 overshoot for testing
    elif iteration == total_nets - 1:
        fill_value = -1.0  # negative saturation
        overshoot = 1.5   # 1.5 overshoot for testing
    else:
        fill_value = None  # uniform random
        overshoot = 2.0 / total_nets * iteration + 0.1 # mix of large and small values, testing clipping.

    _fill_ft_weights(model, fill_value, seed=iteration, overshoot=overshoot)

    # Coalesce virtual weights into real weights before serialization
    model.input.coalesce()
    model.layer_stacks.coalesce_layer_stacks_inplace()

    writer = M.NNUEWriter(model, description=f"sanity_check_iter{iteration}", ft_compression="leb128")
    os.makedirs(os.path.dirname(os.path.abspath(net_path)), exist_ok=True)
    with open(net_path, "wb") as f:
        f.write(writer.buf)

    # Reload from the written file so Python eval uses the exact quantized weights
    return _load_from_nnue(net_path, lcfg)


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


# ─── Single-net test ──────────────────────────────────────────────────────────

def run_fen_check(
    model: M.NNUEModel,
    fens: list[str],
    net_path: str,
    engine_path: str,
    max_error: int,
    device: str,
    label: str,
) -> list[tuple[str, float, int, float]]:
    """Run FEN check for one net.  Returns list of (fen, py, sf, err) failures."""
    W = 72
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"  {len(fens)} FENs, max allowed error: {max_error} internal units")
    print(f"{'='*W}")

    py_evals = eval_model(model, fens, device)
    sf_evals = eval_engine(engine_path, net_path, fens)

    failures: list[tuple[str, float, int, float]] = []
    for i, (fen, py, sf) in enumerate(zip(fens, py_evals, sf_evals)):
        err = abs(py - sf)
        passed = err <= max_error
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] #{i + 1:2d}  py={py:+8.1f}  sf={sf:+8d}  err={err:6.1f}")
        print(f"        {fen}")
        if not passed:
            failures.append((fen, py, sf, err))

    print(f"{'='*W}")
    return failures


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = tyro.cli(CliConfig)
    cfg = args.sanity_config
    lcfg = args.nnue_lightning_config

    # Filter positions the engine cannot evaluate (king in check)
    fens = [f for f in KNOWN_FENS if not chess.Board(f).is_check()]

    all_failures: list[tuple[str, str, float, int, float]] = []  # (label, fen, py, sf, err)

    if cfg.random:
        n = max(cfg.random_nets, 1)
        for i in range(n):
            label = f"Random net #{i}: {_iter_label(i, n)}"
            model = _generate_and_serialize(lcfg, lcfg, cfg.net, i, n, cfg.device)
            model.to(cfg.device)
            model.eval()
            failures = run_fen_check(
                model, fens, cfg.net, cfg.engine, cfg.max_error, cfg.device, label
            )
            all_failures.extend((label, *f) for f in failures)
    else:
        if cfg.checkpoint:
            model = _load_from_checkpoint(cfg.checkpoint, lcfg)
        else:
            model = _load_from_nnue(cfg.net, lcfg)
        model.to(cfg.device)
        model.eval()
        failures = run_fen_check(
            model, fens, cfg.net, cfg.engine, cfg.max_error, cfg.device,
            "Trained net"
        )
        all_failures.extend(("Trained net", *f) for f in failures)

    if all_failures:
        print(f"\nFAILED: {len(all_failures)} FEN(s) exceeded "
              f"the {cfg.max_error}-unit error threshold:\n")
        for label, fen, py, sf, err in all_failures:
            print(f"  [{label}]  err={err:.1f}  py={py:+.1f}  sf={sf:+d}")
            print(f"    {fen}\n")
        print(
            "Hint: large systematic errors (>=50 units) typically indicate an\n"
            "index-mapping bug such as a mirrored king bucket table, wrong\n"
            "piece-type ordering, or a sign error in the orientation function.\n"
            "Errors only on saturation nets may indicate an accumulator overflow."
        )
        sys.exit(1)
    else:
        total = len(fens) * (cfg.random_nets if cfg.random else 1)
        print(f"\nPASSED: all {total} checks within {cfg.max_error} internal units\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
