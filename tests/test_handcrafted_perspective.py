"""CPU-only validation; no training or optimizer steps."""

import io
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import pytest
import torch

from data_loader import stream
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.config import LayerStacksConfig
from model.modules.layer_stacks import LayerStacks
from model.perspective import FALLBACK, INVALID, NOMINAL_TO_COMPACT, bucket
from model.quantize import QuantizationConfig, QuantizationManager
from model.utils.serialize import NNUEReader, NNUEWriter

torch.set_num_threads(2)


def board_material(p, n, b, r, q, layout):
    board = chess.Board(None)
    board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    pawn_squares = [chess.square(f, rank) for rank in range(1, 7) for f in range(4)]
    if layout == 0:
        pawn_squares = [chess.A2, chess.E2] + [s for s in pawn_squares if s != chess.A2]
    for sq in pawn_squares[:p]:
        board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
    free = [sq for sq in range(64) if not board.piece_at(sq)]
    for pt, count in zip(
        (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN), (n, b, r, q)
    ):
        for _ in range(count):
            board.set_piece_at(free.pop(0), chess.Piece(pt, chess.WHITE))
    return board


def test_independent_feasibility_and_dense_ids():
    live = set()
    witnesses = {}
    for p in range(9):
        for n, b, r, q in itertools.product(range(16), repeat=4):
            t = p + n + b + r + q
            if (
                t > 15
                or sum(max(0, c - base) for c, base in zip((n, b, r, q), (2, 2, 2, 1)))
                > 8 - p
            ):
                continue
            for layout in [2] if p == 0 else [1] if p == 1 else [0, 1]:
                h = 0 if q else 1 if r else 2
                nominal = (((t // 2) * 3 + h) * 2 + (t - p >= 2)) * 3 + layout
                live.add(nominal)
                witnesses.setdefault(nominal, (p, n, b, r, q, layout))
    assert len(live) == 85
    assert [sum(x // 18 == band for x in live) for band in range(8)] == [
        4,
        12,
        15,
        15,
        15,
        9,
        9,
        6,
    ]
    assert tuple(i if i != 255 else 255 for i in NOMINAL_TO_COMPACT) == tuple(
        sorted(live).index(i) if i in live else 255 for i in range(144)
    )
    assert sorted(x for x in NOMINAL_TO_COMPACT if x != 255) == list(range(85))
    fens = []
    expected = []
    for nominal in sorted(live):
        board = board_material(*witnesses[nominal])
        assert bucket(board, chess.WHITE) == NOMINAL_TO_COMPACT[nominal]
        for position in (board, board.mirror()):
            for turn in (chess.WHITE, chess.BLACK):
                position.turn = turn
                fens.append(position.fen())
                expected.append((bucket(position, turn), bucket(position, not turn)))
    batch = stream.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+K32H3",
        fens,
        [0] * len(fens),
        [1] * len(fens),
        [0] * len(fens),
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
        assert list(zip(tensors[-2].tolist(), tensors[-1].tolist())) == expected
    finally:
        stream.destroy_sparse_batch(batch)


def test_unsupported_exact_material_even_with_live_projection():
    # Eight pawns and two queens fit a coarse live tuple but require a missing pawn.
    board = board_material(8, 0, 0, 0, 2, 0)
    assert bucket(board, chess.WHITE) == INVALID
    board = board_material(0, 12, 0, 0, 0, 2)
    assert bucket(board, chess.WHITE) == INVALID


def test_locality_precedence_and_pawn_halves():
    b = chess.Board()
    original = bucket(b, chess.WHITE)
    b.remove_piece_at(chess.D8)
    b.remove_piece_at(chess.A8)
    assert bucket(b, chess.WHITE) == original
    b.remove_piece_at(chess.A1)
    assert bucket(b, chess.WHITE) == original  # 15 -> 14 stays in band
    b.remove_piece_at(chess.B1)
    assert bucket(b, chess.WHITE) != original
    b.set_piece_at(chess.B1, chess.Piece(chess.KNIGHT, chess.WHITE))
    b.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    b.remove_piece_at(chess.D1)
    qgone = bucket(b, chess.WHITE)
    assert qgone != original
    b.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    assert bucket(b, chess.WHITE) == original
    x = board_material(2, 0, 0, 0, 0, 0)
    both = bucket(x, chess.WHITE)
    x.remove_piece_at(chess.E2)
    x.set_piece_at(chess.B2, chess.Piece(chess.PAWN, chess.WHITE))
    assert bucket(x, chess.WHITE) != both
    assert bucket(x.mirror(), chess.BLACK) == bucket(x, chess.WHITE)


def stacks():
    torch.manual_seed(72)
    return LayerStacks(
        85,
        LayerStacksConfig(L1=128, L2=32, L3=32),
        QuantizationManager(QuantizationConfig()),
    )


def test_full_input_branches_bias_free_skip_and_autograd():
    s = stacks()
    x = torch.randn(3, 128, requires_grad=True) * 0.1
    own = torch.tensor([0, 1, 84])
    opp = torch.tensor([84, 2, 0])

    def project(bank, layer, x, ids):
        w = layer.linear.weight.reshape(85, layer.out_features, layer.in_features)[ids]
        b = layer.linear.bias.reshape(85, -1)[ids]
        if hasattr(layer, "factorized_linear"):
            w = w + layer.factorized_linear.weight
            b = b + layer.factorized_linear.bias
        raw = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)
        return raw + b, raw

    def act(y):
        return torch.cat([y.square(), y], 1).clamp(0, 127 / 128)

    a, ra = project(s, s.l1, x, own)
    b, rb = project(s.opp, s.opp.l1, x, opp)
    h1 = act(torch.cat([a, b], 1))
    a, _ = project(s, s.l2, h1, own)
    b, _ = project(s.opp, s.opp.l2, h1, opp)
    h = torch.cat([h1, act(torch.cat([a, b], 1))], 1)
    a, _ = project(s, s.output, h, own)
    b, _ = project(s.opp, s.opp.output, h, opp)
    expected = (a + b + ra[:, -2:-1] - ra[:, -1:] + rb[:, -2:-1] - rb[:, -1:]) / 2
    actual = s(x, (own, opp), False, False)
    torch.testing.assert_close(actual, expected)
    params = [
        s.l1.linear.weight,
        s.opp.l1.linear.weight,
        s.l1.factorized_linear.bias,
        s.output.linear.weight,
        s.opp.output.linear.weight,
    ]
    ga = torch.autograd.grad(actual.sum(), params, retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), params)
    for a, b in zip(ga, ge):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_fallback_is_equal_parameter_mean_of_exposed_only():
    s = stacks()
    mask = torch.zeros(85, dtype=torch.bool)
    mask[[1, 84]] = True
    with torch.no_grad():
        s.l1.linear.weight[16:32].fill_(0.1)
        s.l1.linear.weight[-16:].fill_(0.3)
        s.l1.factorized_linear.weight.fill_(0.05)
    exported = list(s.export_layers(mask))
    torch.testing.assert_close(exported[-1][0].weight, torch.full((16, 128), 0.25))
    assert len(exported) == 86
    with pytest.raises(ValueError):
        list(s.export_layers(torch.zeros_like(mask)))


def small_model():
    return NNUEModel("K32H3^", ModelConfig(L1=128, L2=32, L3=32))


def test_serialization_masks_roundtrip_and_compatibility():
    m = small_model()
    m.clip_weights(True)
    with pytest.raises(ValueError, match="exposure"):
        NNUEWriter(m, verbose=False)
    m.training_exposure[0, [1, 84]] = True
    m.training_exposure[1, [0, 20]] = True
    blob = NNUEWriter(m, verbose=False).buf
    loaded = NNUEReader(io.BytesIO(blob), "K32H3^", ModelConfig(L1=128)).model
    assert NNUEWriter(loaded, verbose=False).buf == blob
    with pytest.raises(ValueError, match="inference-only"):
        loaded.train()
    a, b = loaded.calculate_buckets(
        torch.tensor([1, 0, 255]), torch.tensor([0, 84, 255])
    )
    assert a.tolist() == [1, FALLBACK, FALLBACK]
    assert b.tolist() == [0, FALLBACK, FALLBACK]
    damaged = bytearray(blob)
    damaged[4] ^= 1
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(damaged), "K32H3^", ModelConfig(L1=128))
    state = m.state_dict()
    state["architecture_tag"] += 1
    with pytest.raises(ValueError, match="Incompatible"):
        small_model().load_state_dict(state)


def test_training_exposure_validation_separation_and_checkpoint():
    m = small_model()
    m.record_exposure(torch.tensor([0, 84]), torch.tensor([1, 1]))
    assert m.training_exposure.sum() == 3
    clone = small_model()
    clone.load_state_dict(m.state_dict())
    assert torch.equal(clone.training_exposure, m.training_exposure)
    clone.eval()
    routes = clone.calculate_buckets(torch.tensor([2]), torch.tensor([4]))
    assert routes[0].item() == FALLBACK
    assert clone.training_exposure.sum() == 3
    clone.train()
    with pytest.raises(RuntimeError, match="Unsupported"):
        clone.calculate_buckets(torch.tensor([255]), torch.tensor([0]))


def test_ft_permutation_includes_both_role_fallbacks():
    import ftperm

    m = small_model()
    m.eval()
    m.zero_virtual_weights()
    for bank in (m.layer_stacks, m.layer_stacks.opp):
        bank.l1.fallback_weight.uniform_(-0.2, 0.2)
        bank.output.fallback_weight.uniform_(-0.2, 0.2)
    x = torch.rand(3, 128)
    ids = (torch.tensor([0, 85, 84]), torch.tensor([85, 84, 1]))
    before = m.layer_stacks(x, ids, True, True)
    perm = torch.randperm(64)
    full = torch.cat([perm, perm + 64])
    old = m.input.bias.clone()
    ftperm.ft_permute_impl(m, perm.numpy())
    torch.testing.assert_close(m.input.bias, old[full])
    after = m.layer_stacks(x[:, full], ids, True, True)
    torch.testing.assert_close(after, before, atol=1e-6, rtol=0)


def test_strict_h3_native_feature_blocks():
    for fen in ["7k/8/8/8/8/8/4R3/K7 w - - 0 1", "7k/8/8/8/8/8/4r3/KQ6 w - - 0 1"]:
        board = chess.Board(fen)
        batch = stream.get_sparse_batch_from_fens("K32H3", [fen], [0], [1], [0])
        try:
            data = batch.contents.get_tensors("cpu")
            for color, ids in [(chess.WHITE, data[2][0]), (chess.BLACK, data[3][0])]:
                block = (
                    2
                    if board.pieces(chess.QUEEN, not color)
                    else 1
                    if board.pieces(chess.ROOK, not color)
                    else 0
                )
                assert {int(x) // (32 * 768) for x in ids if x >= 0} == {block}
        finally:
            stream.destroy_sparse_batch(batch)


def _exposure_worker(rank, path):
    import os

    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    try:
        m = small_model()
        m.record_exposure(torch.tensor([rank]), torch.tensor([83 + rank]))
        assert m.training_exposure[0, :2].all() and m.training_exposure[1, 83:].all()
        assert int(m.training_exposure.sum()) == 4
    finally:
        dist.destroy_process_group()


def test_distributed_exposure_union_without_training(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _exposure_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        start_method="spawn",
    )


def test_optimizer_group_coverage_without_constructing_optimizer(monkeypatch):
    from model.config import NNUEConfig
    from model.nnue import NNUE
    from model.optimizers.config import OptimizerConfig

    class InspectOnly:
        def configure_optimizers(self, groups):
            return groups

    monkeypatch.setattr(
        OptimizerConfig, "get_optimizer_wrapper", lambda self: InspectOnly()
    )
    wrapper = NNUE(
        config=NNUEConfig(features="K32H3^", model_config=ModelConfig(L1=128))
    )
    groups = wrapper.configure_optimizers()
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in wrapper.model.parameters() if p.requires_grad}


def test_raw_skips_ignore_static_biases():
    s = stacks()
    with torch.no_grad():
        for p in s.parameters():
            p.zero_()
        for bank in (s, s.opp):
            bank.l1.linear.bias.view(85, -1)[:, -2] = 0.125
            bank.l1.linear.bias.view(85, -1)[:, -1] = 0.75
    value = s(
        torch.zeros(2, 128), (torch.tensor([0, 84]), torch.tensor([84, 0])), True, True
    )
    assert torch.equal(value, torch.zeros_like(value))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_export_non_cpu_experts_coalesces_on_cpu(dtype):
    # Fake meta tensors exercise mixed-device dispatch without CUDA hardware.
    # Before the fix, factorized at_index() tried CPU.add_(meta) and failed.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        bank = stacks().to(device="meta", dtype=dtype)
        for role in (bank, bank.opp):
            exported = list(role.export_layers([True] * 85))
            assert len(exported) == 86
            for layers in exported:
                for layer in layers:
                    assert layer.weight.device.type == "cpu"
                    assert layer.bias.device.type == "cpu"
                    assert layer.weight.dtype == dtype
                    assert layer.bias.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_export_cpu_coalesced_parameters_and_fallback_preserve_dtype(dtype):
    bank = stacks().to(dtype=dtype)
    with torch.no_grad():
        bank.l1.factorized_linear.weight.fill_(0.125)
        bank.l1.factorized_linear.bias.fill_(-0.25)
    mask = torch.zeros(85, dtype=torch.bool)
    mask[[1, 84]] = True
    exported = list(bank.export_layers(mask))
    for expert in (1, 84):
        layer = exported[expert][0]
        torch.testing.assert_close(
            layer.weight,
            bank.l1.linear.weight.view(85, 16, 128)[expert]
            + bank.l1.factorized_linear.weight,
        )
        torch.testing.assert_close(
            layer.bias,
            bank.l1.linear.bias.view(85, 16)[expert] + bank.l1.factorized_linear.bias,
        )
    for layer_index in range(3):
        fallback = exported[-1][layer_index]
        assert fallback.weight.dtype == dtype
        torch.testing.assert_close(
            fallback.weight,
            (exported[1][layer_index].weight + exported[84][layer_index].weight) / 2,
        )
        torch.testing.assert_close(
            fallback.bias,
            (exported[1][layer_index].bias + exported[84][layer_index].bias) / 2,
        )


@pytest.mark.parametrize("device_index", [0, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA inference required")
def test_cuda_validation_transition_and_cpu_export_without_training(
    device_index, monkeypatch
):
    if device_index >= torch.cuda.device_count():
        pytest.skip("CUDA device is unavailable")
    device = torch.device("cuda", device_index)
    model = small_model().to(device)
    model.clip_weights(True)
    model.record_exposure(
        torch.tensor([1, 84], device=device), torch.tensor([0, 20], device=device)
    )
    original = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        # This is the transition that failed on every rank in the supplied log.
        def forbidden(*args, **kwargs):
            raise AssertionError("Validation must remain on the model device")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.Tensor, "__bool__", forbidden)
            model.eval()
        ids = model.calculate_buckets(
            torch.tensor([1, 2, 255], device=device),
            torch.tensor([0, 84, 255], device=device),
        )
        value = model.layer_stacks(torch.rand(3, 128, device=device), ids, True, True)
        assert torch.isfinite(value).all()
        blob = NNUEWriter(model, verbose=False).buf
        loaded = NNUEReader(io.BytesIO(blob), "K32H3^", ModelConfig(L1=128)).model
        assert NNUEWriter(loaded, verbose=False).buf == blob
        model.train()
    assert model.training
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.detach().cpu(), original[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_validation_reduces_on_device_without_export_or_host_scalar_reads(
    monkeypatch, dtype
):
    model = small_model().to(dtype=dtype)
    model.record_exposure(torch.tensor([1, 7, 84]), torch.tensor([0, 7, 20]))
    masks = model.training_exposure.clone()
    expected = []
    for role, bank in enumerate((model.layer_stacks, model.layer_stacks.opp)):
        exported = list(bank.export_layers(masks[role]))
        expected.append([(x.weight.clone(), x.bias.clone()) for x in exported[-1]])
    original = {name: p.clone() for name, p in model.named_parameters()}

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Validation must not export, copy to CPU, or read host scalars"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        patch.setattr(torch.Tensor, "nonzero", forbidden)
        patch.setattr(torch, "nonzero", forbidden)
        for bank in (model.layer_stacks, model.layer_stacks.opp):
            patch.setattr(bank, "export_layers", forbidden)
            for layer in bank.layers():
                patch.setattr(layer, "at_index", forbidden)
        model.eval()
        model.train()
        model.eval()
    for bank, values in zip((model.layer_stacks, model.layer_stacks.opp), expected):
        for layer, (weight, bias) in zip(bank.layers(), values):
            torch.testing.assert_close(layer.fallback_weight, weight, rtol=0, atol=0)
            torch.testing.assert_close(layer.fallback_bias, bias, rtol=0, atol=0)
    assert torch.equal(model.exported_mask, masks)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_validation_non_cpu_reduction_stays_on_device(dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        model = small_model().to(device="meta", dtype=dtype)
        model.eval()
        for bank in (model.layer_stacks, model.layer_stacks.opp):
            for layer in bank.layers():
                assert layer.fallback_weight.device.type == "meta"
                assert layer.fallback_bias.device.type == "meta"
                assert layer.fallback_weight.dtype == dtype


def test_export_marker_restored_for_checkpoints_and_legacy_pickles():
    model = small_model()
    model.mark_export_loaded()
    clone = small_model()
    clone.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="inference-only"):
        clone.train()
    # Simulate a .pt object created before the Python cache/post-load hook existed.
    del model._export_loaded
    model._load_state_dict_post_hooks.clear()
    blob = io.BytesIO()
    torch.save(model, blob)
    blob.seek(0)
    restored = torch.load(blob, weights_only=False)
    with pytest.raises(ValueError, match="inference-only"):
        restored.train()
    restored.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("indices", [[1, 7, 84], list(range(85))])
def test_device_fallback_matches_independent_expert_average(dtype, indices):
    bank = stacks().to(dtype=dtype)
    with torch.no_grad():
        bank.l1.factorized_linear.weight.uniform_(-0.1, 0.1)
        bank.l1.factorized_linear.bias.uniform_(-0.1, 0.1)
    mask = torch.zeros(85, dtype=torch.bool)
    mask[indices] = True
    actual = list(bank.fallback_parameters(mask))
    for layer, (weight, bias) in zip(bank.layers(), actual):
        selected = [layer.at_index(i) for i in indices]
        torch.testing.assert_close(
            weight, torch.stack([x.weight for x in selected]).mean(0)
        )
        torch.testing.assert_close(
            bias, torch.stack([x.bias for x in selected]).mean(0)
        )
