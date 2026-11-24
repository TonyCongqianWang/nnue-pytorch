import ctypes
import io
from dataclasses import replace

import chess
import pytest
import torch

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.layer_stacks import LayerStacks, stack_count, stack_keys
from model.quantize import QuantizationConfig, QuantizationManager
from model.utils.serialize import NNUEReader, NNUEWriter
from routing.tree128 import IDENTITY, ancestor_map, context, select


def config(**kwargs):
    return ModelConfig(L1=32, L2=8, L3=8, ls_keys="tree128", **kwargs)


def test_stack_configuration_and_factor_coalescing():
    assert stack_count(stack_keys("tree128")) == 128
    with pytest.raises(ValueError):
        stack_keys("tree128:queen")
    c = config()
    q = QuantizationManager(QuantizationConfig())
    stacks = LayerStacks(128, c, q)
    assert stacks.count == stacks.raw_count == 128
    assert torch.equal(stacks.stack_remap, torch.arange(128))
    assert [len(set(ancestor_map(d))) for d in (2, 3)] == [4, 8]
    for layer in (stacks.l1, stacks.l2, stacks.output):
        for p in (*layer.group_factor_w, *layer.group_factor_b):
            p.data.uniform_(-0.01, 0.01)
    x = torch.rand(128, 32)
    ids = torch.arange(128)
    y = stacks(x, ids, False, False)
    y.sum().backward()
    assert torch.all(stacks.output.linear.bias.grad != 0)
    stacks.coalesce_layer_stacks_inplace()
    torch.testing.assert_close(stacks(x, ids, False, False), y, atol=2e-6, rtol=2e-6)
    state = stacks.state_dict()
    clone = LayerStacks(128, c, q)
    clone.load_state_dict(state)
    torch.testing.assert_close(clone(x, ids, False, False), y, atol=2e-6, rtol=2e-6)
    state["routing_identity"] = state["routing_identity"].clone()
    state["routing_identity"][0] ^= 1
    with pytest.raises(RuntimeError, match="routing identity"):
        clone.load_state_dict(state)
    for factors in ("queen", "tree-depth6"):
        with pytest.raises(ValueError):
            LayerStacks(128, config(ls_factors=factors), q)


def test_tree128_export_roundtrip_and_identity():
    c = config()
    model = NNUEModel("K32H3^", c)
    blob = bytes(NNUEWriter(model, description="tree128-test").buf)
    reader = NNUEReader(io.BytesIO(blob), "K32H3^", c)
    assert bytes(NNUEWriter(reader.model, description=reader.description).buf) == blob
    damaged = bytearray(blob)
    offset = blob.index(bytes.fromhex(IDENTITY))
    damaged[offset] ^= 1
    with pytest.raises(ValueError, match="routing identity"):
        NNUEReader(io.BytesIO(damaged), "K32H3^", c)
    with pytest.raises(ValueError):
        NNUEReader(
            io.BytesIO(blob),
            "K32H3^",
            replace(c, ls_keys="piececount:queen:pawnhalf:manyoffi"),
        )
    model.layer_stacks.routing_identity[0] ^= 1
    with pytest.raises(ValueError, match="routing identity"):
        NNUEWriter(model)
    whole_model = io.BytesIO()
    torch.save(model, whole_model)
    whole_model.seek(0)
    restored = torch.load(whole_model, weights_only=False)
    with pytest.raises(ValueError, match="routing identity"):
        restored.layer_stacks.validate_routing_identity()


def test_native_context_and_leaf():
    import random

    from data_loader._native import c_lib

    native = c_lib.dll.tree128_context_for_fen
    native.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    native.restype = ctypes.c_int
    rng = random.Random(128)
    board = chess.Board()
    cases = [
        chess.Board(f)
        for f in [
            "7k/8/8/8/8/8/8/K7 w - - 0 1",
            "7k/8/8/8/8/BB1B4/NNN5/KRRQQ3 w - - 0 1",
            "7k/8/8/8/8/8/PPPPPPPP/K7 w - - 0 1",
            "7k/8/8/8/8/8/P6p/K7 b - - 0 1",
            "7k/8/8/8/8/8/8/KB1B4 w - - 0 1",
            "7k/8/8/8/8/8/8/KBB5 w - - 0 1",
        ]
    ]
    for _ in range(2500):
        if board.is_game_over() or board.fullmove_number > 160:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        cases.extend([board.copy(), board.mirror()])
    for board in cases:
        fields = (ctypes.c_int * 8)()
        leaf = native(board.fen().encode(), fields)
        expected = context(board)
        assert tuple(fields) == expected, board.fen()
        assert leaf == select(expected), board.fen()

    from data_loader.stream import destroy_sparse_batch, get_sparse_batch_from_fens

    sample = cases[:64]
    expected = [select(context(b)) for b in sample]
    for packed in (False, True):
        batch = get_sparse_batch_from_fens(
            "K32H3",
            [b.fen() for b in sample],
            [0] * len(sample),
            [9999] * 2 + [20] * (len(sample) - 2),
            [0] * len(sample),
            ls_keys="tree128",
            prepack=packed,
        )
        try:
            assert batch.contents.num_tree_samples == 2
            assert batch.contents.get_tensors("cpu")[-1].tolist() == expected
        finally:
            destroy_sparse_batch(batch)
