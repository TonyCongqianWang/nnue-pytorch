"""CPU regressions for cascading contexts, K32H3, and perspective-relative concatenated expert inference."""

import chess
import pytest
import torch
from test_routing_contexts import get_context_indices

from data_loader import stream
from model.modules.config import LayerStacksConfig
from model.modules.features.k32h3 import K32H3, InverseKingBuckets
from model.modules.layer_stacks import LayerStacks
from model.modules.router import decode_material_state, decompose_router_index
from model.quantize import QuantizationConfig, QuantizationManager


@pytest.mark.parametrize("heavy", ["q", "r", "", "Q", "R"])
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
                    2
                    if position.pieces(chess.QUEEN, not color)
                    else 1
                    if position.pieces(chess.ROOK, not color)
                    else 0
                )
                expected = sorted(
                    (h * 32 + kb) * 768
                    + ((piece.piece_type - 1) * 2 + (piece.color != color)) * 64
                    + (sq ^ flip)
                    for sq, piece in position.piece_map().items()
                )
                actual = indices[row][indices[row] >= 0].tolist()
                assert sorted(actual) == expected
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
    q = QuantizationManager(QuantizationConfig(), (2, 2))
    stacks = LayerStacks(2, LayerStacksConfig(L1=4, L2=4, L3=4), q, fc0_count=4)
    for layer in stacks.dense_layers():
        layer.is_coalesced = True
        layer.coalesced_weight.uniform_(-0.25, 0.5)
        layer.coalesced_bias.uniform_(-0.15, 0.25)
    return stacks


def reference(stacks, x, e1, e2, d1, d2):
    def project(layer, x, ids):
        w = layer.coalesced_weight.reshape(
            layer.count, layer.out_features, layer.in_features
        )[ids]
        b = layer.coalesced_bias.reshape(layer.count, layer.bias_features)[ids]
        return (w @ x.unsqueeze(-1)).squeeze(-1), b

    def activate(a, b, ba, bb, ds, dt):
        asq, alin = ba.chunk(2, 1)
        bsq, blin = bb.chunk(2, 1)
        return torch.cat(
            [
                (a + asq + ds[0]).square(),
                (b + bsq + dt[0]).square(),
                a + alin + ds[1],
                b + blin + dt[1],
            ],
            1,
        ).clamp(0, 127 / 128)

    raw, bias = project(stacks.l1, x, e1[0])
    a, b = raw.chunk(2, 1)
    sa, sb, la, lb = bias.chunk(4, 1)
    ba, bb = torch.cat([sa, la], 1), torch.cat([sb, lb], 1)
    skip = (a[:, -2:-1] - a[:, -1:] + b[:, -2:-1] - b[:, -1:]) * 0.5
    h1 = activate(a, b, ba, bb, d1[:2], d2[:2])
    a, ba = project(stacks.l2, h1, e1[1])
    b, bb = project(stacks.opp.l2, h1, e2[1])
    h2 = activate(a, b, ba, bb, d1[2:], d2[2:])
    h = torch.cat([h1, h2], 1)
    a, ba = project(stacks.output, h, e1[1])
    b, bb = project(stacks.opp.output, h, e2[1])
    return (a + ba + b + bb) * 0.5 + skip


def test_fixed_order_independent_full_readouts():
    s = make_stacks()
    x = torch.rand(3, 4)
    e1 = (torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1]))
    e2 = (torch.tensor([1, 0, 0]), torch.tensor([0, 0, 1]))
    d1 = tuple(torch.rand(3, 2) * 0.1 for _ in range(4))
    d2 = tuple(torch.rand(3, 2) * 0.1 for _ in range(4))
    torch.testing.assert_close(
        s(x, None, e1, e2, d1, d2, False, False), reference(s, x, e1, e2, d1, d2)
    )
    # Equal contexts/FT inputs do not force zero: role parameters are independent.
    d = tuple(torch.zeros(1, 2) for _ in range(4))
    for layer in s.dense_layers():
        layer.coalesced_weight.zero_()
        layer.coalesced_bias.zero_()
    for bank in (s, s.opp):
        bank.output.coalesced_bias.fill_(-0.25)
    assert (
        s(
            torch.zeros(1, 4),
            None,
            (torch.tensor([0]),) * 2,
            (torch.tensor([0]),) * 2,
            d,
            d,
        ).item()
        < 0
    )


@pytest.mark.parametrize(
    "oa,ob,sa,sb", [(-3, -2, 1, 0), (1, 2, -3, 0), (-2, 3, 0, -3), (3, 4, 1, 2)]
)
def test_common_scale_average(oa, ob, sa, sb):
    s = make_stacks()
    for layer in s.dense_layers():
        layer.coalesced_weight.zero_()
        layer.coalesced_bias.zero_()
    for bank, o, skip in ((s, oa, sa), (s.opp, ob, sb)):
        bank.output.coalesced_bias.fill_(o / 16384)
    w = s.l1.coalesced_weight.reshape(4, 4, 4)
    w[:, 0, 0], w[:, 2, 0] = sa / 32768, sb / 32768
    # Large, unequal activation biases must not enter either half of the skip.
    s.l1.coalesced_bias.uniform_(-0.5, 0.5)
    d = tuple(torch.zeros(1, 2) for _ in range(4))
    actual = s(
        torch.tensor([[1.0, 0, 0, 0]]),
        None,
        (torch.tensor([0]),) * 2,
        (torch.tensor([1]),) * 2,
        d,
        d,
    )
    expected = s.quantization.fake_quantize_output(
        torch.tensor([[int((2 * oa + 2 * ob + sa + sb) / 2) / 32768]])
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_engine_accumulator_paths():
    """Optional engine integration; see docs/moe7.md for building the probe."""
    import os
    import random
    import subprocess

    probe = os.environ.get("MOE7_ENGINE_PROBE")
    net = os.environ.get("MOE7_TEST_NET")
    if not probe or not net:
        pytest.skip("Set MOE7_ENGINE_PROBE and MOE7_TEST_NET for engine integration")
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


def test_default_moe7_shapes_and_mac_budget():
    stacks = LayerStacks(
        64, LayerStacksConfig(), QuantizationManager(QuantizationConfig())
    )
    assert (stacks.l1.in_features, stacks.l1.out_features) == (1024, 32)
    assert (stacks.l2.in_features, stacks.l2.out_features) == (64, 16)
    assert (stacks.output.in_features, stacks.output.out_features) == (128, 1)
    macs = sum(
        layer.in_features * layer.out_features for layer in stacks.dense_layers()
    )
    assert stacks.l1.bias_features == 64
    assert stacks.l2.bias_features == 32
    assert stacks.l1.count == 64
    assert not hasattr(stacks.opp, "l1")
    assert stacks.count == stacks.opp.count == 64
    assert all(layer.rank == 64 for layer in stacks.dense_layers())
    assert stacks.output.bias_features == 1
    assert stacks.quantization.weight_scales_dict["ls_l2_weight"] == 128
    assert macs == 35072  # Master: 34944; only the final scalar readout doubles.


def test_independent_generators_receive_gradients_without_optimizer():
    from model.modules.router import MoERouter

    torch.manual_seed(19)
    router = MoERouter(
        num_contexts=2,
        num_weight_experts=2,
        num_fc0_experts=4,
        num_bias_experts=2,
        hidden_sizes=(2, 2),
        num_biases=8,
    )
    stacks = LayerStacks(
        2,
        LayerStacksConfig(L1=4, L2=4, L3=4),
        QuantizationManager(QuantizationConfig(), (2, 2)),
        fc0_count=4,
    )
    a, b, d1, d2, loss, *_ = router(
        torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 1, 0])
    )
    value = stacks(torch.rand(4, 4), router.codebook, a, b, d1, d2, False, False)
    (value.square().sum() + loss).backward()
    for layer in stacks.dense_layers():
        assert layer.adapter_up.grad.abs().sum() > 0
    for head in router.bias_heads:
        up, down = head.up, head.down
        assert up.weight.grad.abs().sum() > 0
        assert (
            down.weight.grad.abs().sum() == 0
        )  # Zero output projection on the initial pass.
