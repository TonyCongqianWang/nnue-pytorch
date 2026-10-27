"""Independent global feature reference and training-free contract checks."""

import io
import json
import os
import random
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import stream
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.global_features import EnemySupports, OwnSupports
from model.utils.serialize import NNUEReader, NNUEWriter

FENS = [
    chess.STARTING_FEN,
    "3k4/8/3p4/2n5/2N5/3P4/8/3K4 w - - 0 1",
    "4k3/8/3p4/2n5/2N5/3P4/8/3K4 w - - 0 1",
    "3k4/8/3p4/2n5/2N5/3P4/8/4K3 w - - 0 1",
    "4k3/8/3p4/2n5/2N5/3P4/8/4K3 w - - 0 1",
    "4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/8/2p5/2pP4/2P5/2P5/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 8",
]
TARGETS = [
    [-1, 0, -1, 1, -1],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, 4],
]
BASES = [0, 0, 168, 1848, 4088, 7672]
SLOTS = [0, 84, 336, 560, 896, 1456]


def pseudo(pt, square, white=True):
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[white][square]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[square]
    diagonal = chess.BB_DIAG_ATTACKS[square][0]
    straight = chess.BB_RANK_ATTACKS[square][0] | chess.BB_FILE_ATTACKS[square][0]
    return (
        diagonal
        if pt == chess.BISHOP
        else straight
        if pt == chess.ROOK
        else diagonal | straight
    )


PREFIX = {pt: [] for pt in range(1, 6)}
for _pt in range(1, 6):
    _offset = 0
    for _square in range(64):
        PREFIX[_pt].append(_offset)
        if _pt != chess.PAWN or 8 <= _square < 56:
            _offset += pseudo(_pt, _square).bit_count()
    assert _offset == SLOTS[_pt]


def reference_relations(board, support, perspective, enemy=False):
    result = []
    for source, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        for target in board.attacks(source):
            other = board.piece_at(target)
            if other is None or other.piece_type == chess.KING:
                continue
            pt, target_pt = piece.piece_type, other.piece_type
            if (piece.color == other.color) != support or (
                pt == target_pt and source < target
            ):
                continue
            slot = TARGETS[pt - 1][target_pt - 1]
            if slot < 0:
                continue
            a, b, owner = source, target, piece.color
            if support and (owner != perspective) != enemy:
                continue
            reflected = not perspective
            if reflected:
                a ^= 56
                b ^= 56
            if chess.square_file(board.king(perspective)) < 4:
                a ^= 7
                b ^= 7
            color = owner == perspective
            if pt == target_pt and a < b:
                a, b = b, a
                if not support:
                    color = not color
            row = BASES[pt] + slot * SLOTS[pt] + PREFIX[pt][a]
            row += (pseudo(pt, a, color) & ((1 << b) - 1)).bit_count()
            if not support:
                row += 14952 * (not color)
            result.append(row)
    return sorted(result)


def reference_pawns(board, perspective):
    states, pairs = [], []
    for file in range(8):
        pawns = sorted(
            (s, p.color)
            for s, p in board.piece_map().items()
            if p.piece_type == chess.PAWN and chess.square_file(s) == file
        )
        w = max((chess.square_rank(s) for s, c in pawns if c), default=0)
        b = min((chess.square_rank(s) for s, c in pawns if not c), default=0)
        if not perspective:
            w, b = 7 - b if b else 0, 7 - w if w else 0
        states.append(7 * w + b)
        # Enumerating all legal endpoint pairs independently of the native formula
        # fixes the serialized order, including unused states.
        vocabulary = [(0, a, 0, b) for a in range(6) for b in range(a + 1, 6)]
        vocabulary += [(1, a, 1, b) for a in range(6) for b in range(a + 1, 6)]
        vocabulary += [(0, a, 1, b) for a in range(6) for b in range(6) if a != b]
        for first, second in pairwise(pawns):
            endpoints = []
            for square, color in (first, second):
                rank = chess.square_rank(square) - 1
                if not perspective:
                    color, rank = not color, 5 - rank
                endpoints.append((int(not color), rank))
            endpoints.sort()
            (c1, r1), (c2, r2) = endpoints
            oriented_file = (
                7 - file if chess.square_file(board.king(perspective)) < 4 else file
            )
            pairs.append(60 * oriented_file + vocabulary.index((c1, r1, c2, r2)))
    if chess.square_file(board.king(perspective)) < 4:
        states.reverse()
    return [2401 * f + 49 * states[f] + states[f + 1] for f in range(7)], sorted(pairs)


def native_ids(component, fens):
    batch = stream.get_sparse_batch_from_fens(
        component, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
        return [
            [sorted(row[row >= 0].tolist()) for row in tensor]
            for tensor in tensors[2:4]
        ]
    finally:
        stream.destroy_sparse_batch(batch)


def reference_position(fen):
    board = chess.Board(fen)
    out = {}
    for color, suffix in [(chess.WHITE, "w"), (chess.BLACK, "b")]:
        out[f"own_{suffix}"] = reference_relations(board, True, color)
        out[f"enemy_{suffix}"] = reference_relations(board, True, color, enemy=True)
        out[f"attack_{suffix}"] = reference_relations(board, False, color)
        front, pairs = reference_pawns(board, color)
        out[f"pawn_{suffix}"] = sorted(front + [16807 + i for i in pairs])
    return out


def check_features(fens):
    expected = [reference_position(fen) for fen in fens]
    components = [
        ("Own_Supports", "own"),
        ("Enemy_Supports", "enemy"),
        ("Global_Attacks", "attack"),
        ("Pawn_FrontierPairs", "front"),
        ("Pawn_FilePairs", "file"),
    ]
    for name, family in components:
        white, black = native_ids(name, fens)
        for i, fen in enumerate(fens):
            for suffix, actual in [("w", white[i]), ("b", black[i])]:
                if family in ("own", "enemy", "attack"):
                    want = expected[i][f"{family}_{suffix}"]
                else:
                    front, pairs = reference_pawns(chess.Board(fen), suffix == "w")
                    want = sorted(front if family == "front" else pairs)
                assert actual == want, (name, suffix, fen, actual, want)
    return expected


def test_native_features_and_perspectives():
    rng = random.Random(42)
    board = chess.Board()
    fens = FENS.copy()
    for ply in range(120):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if ply % 5 == 0:
            fens += [board.fen(), board.mirror().fen()]
    for fen in fens:
        assert reference_position(fen) == reference_position(
            chess.Board(fen).transform(chess.flip_horizontal).fen()
        )
    fens += [
        chess.Board(fen).transform(chess.flip_horizontal).fen() for fen in fens.copy()
    ]
    check_features(fens)


def test_support_tables_and_routing_gradients():
    own, enemy = OwnSupports(32), EnemySupports(32)
    assert own.weight.shape == (14952, 24)
    assert enemy.weight.shape == (14952, 8)
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=32))
    with torch.no_grad():
        for f in model.input.features:
            f.weight.zero_()
        model.input.bias.fill_(0.25)
    ids = torch.tensor([[7, 14952 + 9]], dtype=torch.int32)
    us, them = torch.ones(1, 1), torch.zeros(1, 1)
    out = model.input(us, them, ids, ids, False, False, backend="torch")
    # Own-support products precede enemy-support products in each perspective.
    out[:, :12].sum().backward()
    assert torch.count_nonzero(model.input.features[0].weight.grad[7]) == 24
    assert torch.count_nonzero(model.input.features[1].weight.grad) == 0
    model.zero_grad()
    out = model.input(us, them, ids, ids, False, False, backend="torch")
    out[:, 12:16].sum().backward()
    assert torch.count_nonzero(model.input.features[0].weight.grad) == 0
    assert torch.count_nonzero(model.input.features[1].weight.grad[9]) == 8


def test_export_round_trip_and_permutation():
    from ftperm import ft_permute_impl

    torch.manual_seed(42)
    config = ModelConfig(L1=32)
    model = NNUEModel(DEFAULT_FEATURES, config)
    with torch.no_grad():
        model.input.bias.copy_(torch.linspace(0.25, 0.75, 32))
    model.clip_weights(include_input=True)
    encoded = NNUEWriter(model, verbose=False).buf
    restored = NNUEReader(io.BytesIO(encoded), DEFAULT_FEATURES, config).model
    assert NNUEWriter(restored, verbose=False).buf == encoded
    support = restored.input.features[0]
    assert support.weight.shape == (14952, 24)
    batch = stream.get_sparse_batch_from_fens(
        DEFAULT_FEATURES.replace("^", ""),
        FENS,
        [0] * len(FENS),
        [1] * len(FENS),
        [0] * len(FENS),
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
    finally:
        stream.destroy_sparse_batch(batch)
    baseline = restored.forward_ft(*tensors[:4], True, True).detach()
    full_before = restored(*tensors[:4], *tensors[6:8]).detach()
    assert torch.count_nonzero(baseline) > 0
    original = support.weight.clone()
    permutation = torch.cat((torch.randperm(12), 12 + torch.randperm(4))).numpy()
    ft_permute_impl(restored, permutation)
    after = restored.forward_ft(*tensors[:4], True, True).detach()
    expected = baseline[:, list(permutation) + list(permutation + 16)]
    assert torch.equal(after, expected)
    torch.testing.assert_close(
        restored(*tensors[:4], *tensors[6:8]).detach(), full_before
    )
    ft_permute_impl(restored, permutation.argsort())
    assert torch.equal(original, support.weight)
    bad = list(range(16))
    bad[0], bad[15] = bad[15], bad[0]
    with pytest.raises(ValueError, match="preserve"):
        ft_permute_impl(restored, bad)


@pytest.mark.parametrize("version", [0x6A448AFA, 0x6A448AFB])
def test_old_format_rejected(version):
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=32))
    data = NNUEWriter(model, verbose=False).buf
    data[:4] = version.to_bytes(4, "little")
    with pytest.raises((ValueError, AssertionError)):
        NNUEReader(io.BytesIO(data), DEFAULT_FEATURES, ModelConfig(L1=32))


def test_engine_feature_parity(tmp_path):
    executable = os.environ.get("GLOBAL_NNUE_TEST")
    if executable is None:
        pytest.skip("Set GLOBAL_NNUE_TEST to the Stockfish global-nnue-test executable")
    positions = tmp_path / "positions.epd"
    positions.write_text("\n".join(FENS) + "\n")
    output = subprocess.check_output([executable, "--dump", str(positions)], text=True)
    assert [json.loads(line) for line in output.splitlines()] == check_features(FENS)


def compact_ft_reference(ft, us, them, white, black, quantize):
    from model.modules.feature_transformer.sparse_linear_functions import (
        SparseLinearFunction,
    )

    bias = ft.bias
    if quantize:
        bias = ft.quantization.fake_quantize_weights(bias, "ft_bias")

    def perspective(ids):
        offset, values = 0, []
        for f in ft.features:
            selected = torch.where(
                (ids >= offset) & (ids < offset + f.NUM_INPUTS), ids - offset, -1
            )
            weight = f.merged_weight()
            if quantize:
                weight = ft.quantization.fake_quantize_weights(weight, "ft_weight")
            values.append(
                SparseLinearFunction.apply(
                    selected, weight, weight.new_zeros(weight.shape[1]), "torch"
                )
            )
            offset += f.NUM_INPUTS
        own0, own1 = values[0].chunk(2, dim=1)
        enemy0, enemy1 = values[1].chunk(2, dim=1)
        return bias + sum(values[2:]) + torch.cat((own0, enemy0, own1, enemy1), dim=1)

    w, b = perspective(white), perspective(black)
    ordered = us * torch.cat((w, b), dim=1) + them * torch.cat((b, w), dim=1)
    a, b, c, d = ordered.clamp(0, ft.quantization.max_ft_activation).chunk(4, dim=1)
    out = torch.cat((a * b, c * d), dim=1)
    if quantize:
        out = ft.quantization.fake_quantize_ft_act(out)
    return out * ft.quantization.l0_correction_factor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantize", [False, True])
def test_fused_ft_matches_compact_reference(device, quantize):
    from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS

    if device == "cuda" and (not torch.cuda.is_available() or not _HAS_CUPY_KERNELS):
        pytest.skip("CUDA and CuPy required")
    torch.manual_seed(81)
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=32)).to(device)
    with torch.no_grad():
        model.input.bias.fill_(0.4)
    offsets, offset = [], 0
    for feature in model.input.features:
        offsets.append(offset + 7)
        offset += feature.NUM_INPUTS
    # Packed indices, unequal perspective lengths, and an empty row exercise
    # the fused kernel's padding contract without per-group sorting.
    white = torch.tensor(
        [offsets + [9, -1], offsets[::-1] + [11, -1], [-1] * 8],
        dtype=torch.int32,
        device=device,
    )
    black = white.flip(0).contiguous()
    us = torch.tensor([[1.0], [0.0], [1.0]], device=device)
    them = 1 - us
    backend = "fused" if device == "cuda" else "torch"
    actual = model.input(us, them, white, black, quantize, quantize, backend=backend)
    expected = compact_ft_reference(model.input, us, them, white, black, quantize)
    torch.testing.assert_close(actual, expected)
    params = list(model.input.parameters())
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * grad).sum(), params)
    expected_grad = torch.autograd.grad((expected * grad).sum(), params)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_composed_support_capacity_and_native_packing():
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=32))
    assert model.input.MAX_ACTIVE_FEATURES == 599
    batch = stream.get_sparse_batch_from_fens(
        DEFAULT_FEATURES.replace("^", ""),
        FENS,
        [0] * len(FENS),
        [1] * len(FENS),
        [0] * len(FENS),
    )
    try:
        assert batch.contents.max_active_features == model.input.MAX_ACTIVE_FEATURES
        tensors = batch.contents.get_tensors("cpu")
    finally:
        stream.destroy_sparse_batch(batch)
    offsets = []
    offset = 0
    for feature in model.input.features:
        offsets.append(offset)
        offset += feature.NUM_INPUTS
    # Compare the composed packed rows with independently extracted components.
    for side in (0, 1):
        expected = [[] for _ in FENS]
        for f, start in zip(model.input.features, offsets):
            rows = native_ids(f.INPUT_FEATURE_NAME, FENS)[side]
            for dest, row in zip(expected, rows):
                dest.extend(start + i for i in row)
        for row, want in zip(tensors[2 + side], expected):
            assert sorted(row[row >= 0].tolist()) == sorted(want)
            n = len(want)
            assert torch.all(row[:n] >= 0) and torch.all(row[n:] == -1)


def test_automatic_permutation_preserves_sections():
    import numpy as np

    from ftperm import find_perm_impl

    rng = np.random.default_rng(81)
    acts = rng.random((24, 32)) > 0.8
    perm = find_perm_impl(acts, "cpu", 32, support_boundary=12)
    assert sorted(perm[:12]) == list(range(12))
    assert sorted(perm[12:]) == list(range(12, 16))


if __name__ == "__main__":
    # Larger local/CI parity check without requiring a training dataset.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--engine-test", required=True)
    args = parser.parse_args()
    fens = Path(args.positions).read_text().splitlines()
    output = subprocess.check_output(
        [args.engine_test, "--dump", args.positions], text=True
    )
    engine = [json.loads(line) for line in output.splitlines()]
    expected = []
    for offset in range(0, len(fens), 256):
        expected.extend(check_features(fens[offset : offset + 256]))
    assert engine == expected
    print(f"PASS engine/loader/Python parity: {len(fens)} positions")
