"""Device/export checks use initialization and inference only, never training."""

import io

import chess
import pytest
import torch

import data_loader
from cross_check_eval import eval_model_batch, init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import ieee_matmul_precision
from model.utils.serialize import NNUEReader, NNUEWriter, prepare_for_export
from runtime import configure_runtime, resolve_device


def test_device_auto_fallback_and_explicit_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
    with pytest.warns(UserWarning, match="cross-device diagnostic"):
        assert resolve_device(metadata={"device_type": "cuda"}).type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("2")


def test_numeric_cuda_selection_respects_visible_ordinals(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert str(resolve_device("1")) == "cuda:1"
    assert selected == [torch.device("cuda:1")]
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("2")


def test_forced_tf32_override_cannot_bypass_sensitive_policy(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(RuntimeError, match="Unset TORCH_ALLOW"):
        configure_runtime(torch.device("cuda:0"))


def test_precision_guard_restores_policy_after_exception(monkeypatch):
    # Backend settings can be exercised without issuing a CUDA operation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError), ieee_matmul_precision():
            assert torch.get_float32_matmul_precision() == "highest"
            raise ValueError("exercise restoration")
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(original)


def test_prepared_checkpoint_matches_sorted_permuted_export():
    from cross_check_eval import create_permuted_model

    torch.manual_seed(14)
    config = ModelConfig(L1=32, L2=4, L3=4)
    model = NNUEModel(DEFAULT_FEATURES, config).eval()
    init_random_weights(model, build_router_lut=False)
    with torch.no_grad():
        model.layer_stacks.l1.shared_weight.uniform_(-0.9, 0.9)
        for bank in ("fc0", "tail"):
            counts = getattr(model.router, f"validation_counts_{bank}")
            counts.copy_(torch.randperm(len(counts)))
    prepare_for_export(model, rebuild_router=False)
    permuted = create_permuted_model(model)
    writer = NNUEWriter(permuted, verbose=False)
    restored = NNUEReader(io.BytesIO(writer.buf), DEFAULT_FEATURES, config).model.eval()
    fens = [chess.STARTING_FEN, "8/8/3k4/8/2K5/8/2P5/8 w - - 0 1"]
    batch = data_loader.get_sparse_batch_from_fens(
        model.input_feature_name, fens, [0] * 2, [1] * 2, [0] * 2
    )
    try:
        expected = eval_model_batch(model, batch, "cpu", True)
        actual = eval_model_batch(restored, batch, "cpu", True)
    finally:
        data_loader.destroy_sparse_batch(batch)
    assert actual == expected
