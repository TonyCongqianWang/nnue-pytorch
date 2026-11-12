"""CPU regressions for cascading contexts, K32H3, and full-width expert inference."""

import chess
import pytest
import torch
from test_routing_contexts import get_context_indices

from data_loader import stream
from model import NNUE, ModelConfig, NNUEConfig
from model.modules.config import LayerStacksConfig
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.k32h3 import K32H3, InverseKingBuckets
from model.modules.layer_stacks import LayerStacks
from model.modules.moe_linear import MoeLinear
from model.modules.router import decode_material_state, decompose_router_index
from model.quantize import QuantizationConfig, QuantizationManager
from model.utils.load_model import load_model


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
        mat, _, _ = decompose_router_index(context, first_layer=True)
        assert decode_material_state(mat.item(), first_layer=True) == (
            len(pieces),
            0,
            category,
        )
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


def test_engine_accumulator_paths():
    """Optional engine integration; see docs/moe8.md for building the probe."""
    import os
    import random
    import subprocess

    probe = os.environ.get("NNUE_ENGINE_PROBE")
    net = os.environ.get("NNUE_TEST_NET")
    if not probe or not net:
        pytest.skip("Set NNUE_ENGINE_PROBE and NNUE_TEST_NET for engine integration")
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


def make_stacks():

    s = LayerStacks(
        3,
        LayerStacksConfig(L1=4, L2=4, L3=4),
        QuantizationManager(QuantizationConfig()),
        rank=4,
        codebook_dim=8,
        fc0_count=2,
        fc0_rank=3,
        fc0_dim=8,
    )
    for layer in s.dense_layers():
        layer.is_coalesced = True
        layer.coalesced_weight.uniform_(-0.25, 0.5)
        layer.coalesced_bias.uniform_(-0.2, 0.2)
    return s


def test_full_width_projection_and_bias_free_skip():
    torch.manual_seed(8)
    s = make_stacks()
    x = torch.rand(5, 4)
    f = torch.tensor([0, 1, 0, 1, 1])
    t = torch.tensor([0, 1, 2, 1, 0])

    def project(layer, z, index):
        w = layer.coalesced_weight.reshape(
            layer.count, layer.out_features, layer.in_features
        )[index]
        b = layer.coalesced_bias.reshape(layer.count, layer.bias_features)[index]
        return (w @ z.unsqueeze(-1)).squeeze(-1), b

    def activate(raw, bias):
        sq, lin = bias.chunk(2, 1)
        return torch.cat([(raw + sq).square(), raw + lin], 1).clamp(0, 127 / 128)

    raw, bias = project(s.l1, x, f)
    skip = (raw[:, 0:1] - raw[:, 1:2] + raw[:, 2:3] - raw[:, 3:4]) * 0.5
    h1 = activate(raw, bias)
    raw, bias = project(s.l2, h1, t)
    h2 = activate(raw, bias)
    out, bias = project(s.output, torch.cat([h1, h2], 1), t)
    torch.testing.assert_close(s(x, None, f, t, False, False), out + bias + skip)


@pytest.mark.parametrize(
    "output,sa,sb", [(1, -3, 0), (-2, 1, 0), (3, 2, 3), (-1, -2, -3), (2, 20, 6)]
)
def test_common_scale_rounding(output, sa, sb):
    s = make_stacks()
    for layer in s.dense_layers():
        layer.coalesced_weight.zero_()
        layer.coalesced_bias.zero_()
    s.output.coalesced_bias.fill_(output / 16384)
    w = s.l1.coalesced_weight.reshape(2, 4, 4)
    w[:, 0, 0] = sa / 256
    w[:, 2, 0] = sb / 256
    s.l1.coalesced_bias.uniform_(-1, 1)
    out = s(
        torch.tensor([[1 / 128, 0, 0, 0]]), None, torch.tensor([0]), torch.tensor([2])
    )
    combined = int((4 * output + sa + sb) / 2)
    expected = torch.tensor([[int(combined * 9600 / 32768) / 9600]])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_default_shapes_and_independent_router_sizes():

    s = LayerStacks(256, LayerStacksConfig(), QuantizationManager(QuantizationConfig()))
    assert [
        (l.count, l.in_features, l.out_features, l.rank, l.codebook_dim)
        for l in s.dense_layers()
    ] == [(64, 1024, 32, 64, 256), (256, 64, 32, 512, 2048), (256, 128, 1, 512, 2048)]
    assert sum(l.in_features * l.out_features for l in s.dense_layers()) == 34944
    assert not hasattr(s, "opp")


def test_generator_gradients_and_expert_biases_without_optimizer():
    from model.modules.router import MoERouter

    router = MoERouter(
        material_states=2, num_fc0_experts=2, num_weight_experts=3, fc0_dim=8, dim=8
    )
    s = LayerStacks(
        3,
        LayerStacksConfig(L1=4, L2=4, L3=4),
        QuantizationManager(QuantizationConfig()),
        rank=4,
        codebook_dim=8,
        fc0_count=2,
        fc0_rank=3,
        fc0_dim=8,
    )
    f, t, loss, *_ = router(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 9, 18, 31]))
    value = s(torch.rand(4, 4), router.codebook, f, t, False, False)
    (value.square().sum() + loss).backward()
    for layer in s.dense_layers():
        assert (
            layer.adapter_up.grad is not None and layer.adapter_up.grad.abs().sum() > 0
        )
        assert (
            layer.bias_adapter.grad is not None
            and layer.bias_adapter.grad.abs().sum() > 0
        )
        assert not hasattr(layer, "shared_bias")
    for layer, cb in zip(
        s.dense_layers(), (router.codebook[0], router.codebook[1], router.codebook[1])
    ):
        layer.bias_adapter.data.zero_()
        _, bias = layer.get_all_weights_and_biases(cb)
        assert torch.count_nonzero(bias) == 0


def test_factorized_initialization_and_all_coalescing_paths():
    torch.manual_seed(9)
    layer = MoeLinear(8, 4, 3, rank=4, codebook_dim=8, dual_bias=True, factorized=True)
    cb = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    assert torch.count_nonzero(layer.shared_weight) == 0
    assert torch.count_nonzero(layer.adapter_up) > 0
    assert torch.count_nonzero(layer.bias_adapter) == 0
    with torch.no_grad():
        layer.shared_weight.uniform_(-0.1, 0.1)
    weights, biases = layer.get_all_weights_and_biases(cb)
    expected = layer.get_delta_weights(cb) + layer.shared_weight
    torch.testing.assert_close(weights, expected.flatten(0, 1))
    assert torch.count_nonzero(biases) == 0
    for i in range(3):
        w, b = layer.get_expert_weight_and_bias(cb[i])
        torch.testing.assert_close(w, expected[i])
        assert torch.count_nonzero(b) == 0
    permutation = list(reversed(range(8)))
    layer.permute_inputs(permutation)
    torch.testing.assert_close(
        layer.get_all_weights_and_biases(cb)[0], weights[:, permutation]
    )
    layer.coalesce_weights(cb)
    torch.testing.assert_close(layer.coalesced_weight, weights[:, permutation])


def test_old_checkpoint_and_pickled_model_are_rejected(tmp_path):
    config = ModelConfig(L1=4, L2=4, L3=4)
    nnue = NNUE(NNUEConfig(model_config=config))
    assert all(
        not hasattr(layer, "shared_bias")
        for layer in nnue.model.layer_stacks.dense_layers()
    )
    state = nnue.state_dict()
    state["model.layer_stacks.l1.shared_bias"] = torch.zeros(8)
    with pytest.raises(RuntimeError, match="shared_bias"):
        nnue.load_state_dict(state)

    # Pickle bypasses constructors; reject legacy parameters rather than
    # quietly evaluating a model while ignoring its old shared contribution.
    nnue.model.layer_stacks.l1.register_parameter(
        "shared_bias", torch.nn.Parameter(torch.ones(8))
    )
    path = tmp_path / "old.pt"
    torch.save(nnue, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(str(path), DEFAULT_FEATURES, config)


def test_weighted_mix_uses_each_tail_with_its_own_activation():
    torch.manual_seed(8)
    s = LayerStacks(
        4,
        LayerStacksConfig(L1=4, L2=4, L3=4),
        QuantizationManager(QuantizationConfig()),
        rank=4,
        codebook_dim=8,
        fc0_count=2,
        fc0_rank=3,
        fc0_dim=8,
    )
    x = torch.rand(3, 4, requires_grad=True)
    fc0 = torch.tensor([[0, 1], [1, 0], [1, 0]])
    tail = torch.tensor([[0, 1, 2, 3], [3, 1, 2, 0], [1, 0, 2, 3]])
    cb0, cbt = torch.randn(2, 8), torch.randn(4, 8)

    def hidden(raw, bias):
        sq, lin = bias.chunk(2, -1)
        return torch.cat([(raw + sq).square(), raw + lin], -1).clamp(0, 127 / 128)

    h1 = skip = 0
    for ids, weight in zip(fc0.unbind(1), (0.75, 0.25)):
        raw, bias = s.l1(x, cb0, ids.contiguous())
        h1 = h1 + weight * hidden(raw, bias)
        skip = skip + weight * 0.5 * (
            raw[:, :1] - raw[:, 1:2] + raw[:, 2:3] - raw[:, 3:]
        )
    expected = skip
    for ids, weight in zip(tail.unbind(1), (0.5, 0.25, 0.125, 0.125)):
        ids = ids.contiguous()
        raw, bias = s.l2(h1, cbt, ids)
        out, bias = s.output(torch.cat([h1, hidden(raw, bias)], -1), cbt, ids)
        expected = expected + weight * (out + bias)

    actual = s(x, (cb0, cbt), fc0, tail, False, False)
    torch.testing.assert_close(actual, expected)
    parameters = (x, *s.parameters())
    for grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), parameters),
        torch.autograd.grad(expected.sum(), parameters),
    ):
        torch.testing.assert_close(grad, expected_grad)
