"""CPU regressions for cascading contexts, K32H3, and perspective-relative concatenated expert inference."""

import chess
import pytest
import torch
import torch.nn.functional as F
from test_top2moe_symmetry import get_context_indices

from data_loader import stream
from model.modules.config import LayerStacksConfig
from model.modules.features.k32h3 import K32H3, InverseKingBuckets
from model.modules.layer_stacks import LayerStacks
from model.modules.router import decode_material_state, decompose_router_index
from model.quantize import QuantizationConfig, QuantizationManager


@pytest.mark.parametrize("heavy", ["q", "r", ""])
def test_native_heavy_features_and_context(heavy):
    # White king on e1 selects the highest king bucket: indices exceed 16 bits.
    board = chess.Board("k7/8/8/8/8/8/8/4K3 w - - 0 1")
    if heavy:
        board.set_piece_at(chess.B4, chess.Piece.from_symbol(heavy))
    fens = [board.fen(), board.mirror().fen()]
    batch = stream.get_sparse_batch_from_fens("K32H3", fens, [0, 0], [1, 1], [0, 0])
    try:
        tensors = batch.contents.get_tensors("cpu")
        for row, fen in enumerate(fens):
            position = chess.Board(fen)
            for color, indices in [
                (chess.WHITE, tensors[2]),
                (chess.BLACK, tensors[3]),
            ]:
                king = position.king(color)
                flip = (0 if color else 56) ^ (7 if chess.square_file(king) < 4 else 0)
                ksq = king ^ flip
                kb = InverseKingBuckets.index(ksq)
                h = (
                    0
                    if position.pieces(chess.QUEEN, not color)
                    else 1
                    if position.pieces(chess.ROOK, not color)
                    else 2
                )
                expected = sorted(
                    (kb * 3 + h) * 768
                    + ((piece.piece_type - 1) * 2 + (piece.color != color)) * 64
                    + (sq ^ flip)
                    for sq, piece in position.piece_map().items()
                )
                actual = indices[row][indices[row] >= 0].tolist()
                assert sorted(actual) == expected
                if color == position.turn:
                    assert max(actual) > 65535
            expected_contexts = get_context_indices(position)
            assert tensors[-2][row].item() == expected_contexts[0]
            assert tensors[-1][row].item() == expected_contexts[1]
    finally:
        stream.destroy_sparse_batch(batch)


@pytest.mark.parametrize(
    "pieces, category",
    [("qrbn", 0), ("rbb", 1), ("bb", 2), ("b", 3), ("nn", 3), ("", 3)],
)
def test_material_cascade(pieces, category):
    board = chess.Board("7k/8/8/8/8/8/8/7K b - - 0 1")
    for sq, piece in zip([chess.A4, chess.B4, chess.C4, chess.D4], pieces):
        board.set_piece_at(sq, chess.Piece.from_symbol(piece))
    batch = stream.get_sparse_batch_from_fens("K32H3", [board.fen()], [0], [1], [0])
    try:
        tensors = batch.contents.get_tensors("cpu")
        context = tensors[-2]
        mat, _, _ = decompose_router_index(context)
        assert decode_material_state(mat.item()) == (len(pieces), 0, category)
        assert context.item() == get_context_indices(board)[0]
    finally:
        stream.destroy_sparse_batch(batch)


def test_heavy_export_roundtrip():
    feature = K32H3(4)
    restored = K32H3(4)
    exported = feature.get_export_weights()
    assert exported.shape == (67584, 4)
    restored.load_export_weights(exported)
    torch.testing.assert_close(restored.get_export_weights(), exported, rtol=0, atol=0)


def make_stacks():
    torch.manual_seed(7)
    quantization = QuantizationManager(QuantizationConfig())
    stacks = LayerStacks(2, LayerStacksConfig(L1=4, L2=4, L3=4), quantization)
    for layer in (stacks.l1, stacks.l2, stacks.output):
        layer.is_coalesced = True
        layer.coalesced_weight.uniform_(-0.25, 0.5)
        layer.coalesced_bias.fill_(0.125)
    return stacks


def reference(stacks, x, e1, e2, biases1, biases2):
    def affine(layer, value, expert):
        w = layer.coalesced_weight.view(2, layer.out_features, layer.in_features)[
            expert
        ]
        b = layer.coalesced_bias.view(2, layer.out_features)[expert]
        return F.linear(value, w, b)

    def swapped(value):
        a_sq, b_sq, a_lin, b_lin = value.chunk(4, dim=1)
        return torch.cat([b_sq, a_sq, b_lin, a_lin], dim=1)

    def activation(a, b, a_sq, b_sq, a_lin, b_lin):
        return torch.cat(
            [(a + a_sq).square(), (b + b_sq).square(), a + a_lin, b + b_lin], dim=1
        ).clamp(0, 127 / 128)

    a = affine(stacks.l1, x, e1)
    b = affine(stacks.l1, torch.cat(x.chunk(2, dim=1)[::-1], dim=1), e2)
    skip = (a[:, -2:-1] - a[:, -1:] - b[:, -2:-1] + b[:, -1:]) / 2
    h1 = activation(a, b, biases1[0], biases2[0], biases1[1], biases2[1])
    a = affine(stacks.l2, h1, e1)
    b = affine(stacks.l2, swapped(h1), e2)
    h2 = activation(a, b, biases1[2], biases2[2], biases1[3], biases2[3])
    return (
        affine(stacks.output, torch.cat([h1, h2], dim=1), e1)
        - affine(stacks.output, torch.cat([swapped(h1), swapped(h2)], dim=1), e2)
    ) / 2 + skip


@pytest.mark.parametrize("quantized", [False, True])
def test_perspective_concatenation_and_duplicate_route(quantized):
    stacks = make_stacks()
    x = torch.tensor([[0.25, 0.5, 0.125, 0.75]])
    a, b = torch.tensor([0]), torch.tensor([1])
    ba = tuple(torch.full((1, 2), 0.025 * i) for i in range(4))
    bb = tuple(torch.full((1, 2), -0.05 * i) for i in range(4))
    args = {"fake_quantize_acts": quantized, "fake_quantize_weights": quantized}
    actual = stacks(x, None, a, b, ba, bb, **args)
    swapped_x = torch.cat(x.chunk(2, dim=1)[::-1], dim=1)
    torch.testing.assert_close(
        actual, -stacks(swapped_x, None, b, a, bb, ba, **args), rtol=0, atol=0
    )
    if not quantized:
        torch.testing.assert_close(actual, reference(stacks, x, 0, 1, ba, bb))
        same = stacks(x, None, a, a, ba, ba, **args)
        torch.testing.assert_close(same, reference(stacks, x, 0, 0, ba, ba))
        assert same.abs().max() > 1e-5  # Same route is not a symmetric input.
    symmetric = torch.cat([x[:, :2], x[:, :2]], dim=1)
    torch.testing.assert_close(
        stacks(symmetric, None, a, a, ba, ba, **args), torch.zeros(1, 1), rtol=0, atol=0
    )


def test_joint_hypernetwork_gradient_flow():
    torch.manual_seed(11)
    q = QuantizationManager(QuantizationConfig())
    stacks = LayerStacks(2, LayerStacksConfig(L1=4, L2=4, L3=4), q)
    codebook = torch.randn(2, 256, requires_grad=True)
    # Keep this derivative check inside the activation range, rather than
    # depending on random clipping at the narrower width.
    with torch.no_grad():
        for layer in (stacks.l1, stacks.l2):
            layer.adapter_up.mul_(0.1)
            layer.bias_adapter.zero_()
    x = torch.tensor([[0.125, 0.25, 0.375, 0.5]])
    ba = tuple(torch.full((1, 2), 0.25, requires_grad=True) for _ in range(4))
    bb = tuple(torch.full((1, 2), 0.125, requires_grad=True) for _ in range(4))
    value = stacks(
        x,
        codebook,
        torch.tensor([0]),
        torch.tensor([1]),
        ba,
        bb,
        fake_quantize_acts=False,
        fake_quantize_weights=False,
    )
    # A single derivative check, with no optimizer or training loop.
    value.sum().backward()
    assert (codebook.grad.norm(dim=1) > 0).all()
    for layer in (stacks.l1, stacks.l2, stacks.output):
        assert layer.adapter_up.grad.abs().sum() > 0
        assert layer.adapter_down.grad.abs().sum() > 0
    for bias in ba + bb:
        assert bias.grad.abs().sum() > 0


@pytest.mark.parametrize("a,b", [(-3, -2), (1, 2), (-2, 3), (3, 4)])
def test_integer_average_truncates_toward_zero(a, b):
    stacks = make_stacks()
    for layer in (stacks.l1, stacks.l2, stacks.output):
        layer.coalesced_weight.zero_()
        layer.coalesced_bias.zero_()
    scale = stacks.quantization.weight_scales_dict["ls_output_bias"]
    stacks.output.coalesced_bias.copy_(torch.tensor([a, b]) / scale)
    biases = tuple(torch.zeros(1, 2) for _ in range(4))
    actual = stacks(
        torch.zeros(1, 4), None, torch.tensor([0]), torch.tensor([1]), biases, biases
    )
    expected = stacks.quantization.fake_quantize_output(
        torch.tensor([[int((a - b) / 2) / scale]])
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_engine_accumulator_paths():
    """Optional engine integration; see docs/moe5.md for building the probe."""
    import os
    import random
    import subprocess

    probe = os.environ.get("MOE5_ENGINE_PROBE")
    net = os.environ.get("MOE5_TEST_NET")
    if not probe or not net:
        pytest.skip("Set MOE5_ENGINE_PROBE and MOE5_TEST_NET for engine integration")
    episodes = [
        ("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", ["e4d5"]),
        ("4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1", ["e4d5"]),
        ("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1", ["a7a8q", "e8f7", "a8b8", "h2h1r"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
    ]
    # Enough pieces to exercise the engine's hybrid king-update shortcut.
    for piece in ("q", "r"):
        board = chess.Board()
        board.remove_piece_at(chess.E2)
        board.remove_piece_at(chess.D8)
        if piece == "r":
            board.remove_piece_at(chess.A8)
            board.remove_piece_at(chess.H8)
        board.set_piece_at(chess.E2, chess.Piece.from_symbol(piece))
        episodes.append((board.fen(), ["e1e2"]))
    rng = random.Random(19)
    for _ in range(12):
        board = chess.Board()
        moves = []
        for _ in range(150):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(move.uci())
            board.push(move)
        episodes.append((chess.STARTING_FEN, moves))
    for fen, moves in episodes:
        board = chess.Board(fen)
        for move in moves:
            assert chess.Move.from_uci(move) in board.legal_moves
            board.push_uci(move)
    data = "".join(fen + "\n" + " ".join(moves) + "\n" for fen, moves in episodes)
    result = subprocess.run(
        [probe, net],
        input=data,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified 1547" in result.stdout


def test_default_moe5_shapes_and_mac_budget():
    stacks = LayerStacks(
        64, LayerStacksConfig(), QuantizationManager(QuantizationConfig())
    )
    assert (stacks.l1.in_features, stacks.l1.out_features) == (1024, 16)
    assert (stacks.l2.in_features, stacks.l2.out_features) == (64, 16)
    assert (stacks.output.in_features, stacks.output.out_features) == (128, 1)
    macs = 2 * sum(
        layer.in_features * layer.out_features
        for layer in (stacks.l1, stacks.l2, stacks.output)
    )
    assert macs == 35072  # Master: 34944; only the final scalar readout doubles.
