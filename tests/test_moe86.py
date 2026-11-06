"""MoE8.6 is a fresh architecture, with no shared-bias migration."""

import io

import pytest
import torch

from model import NNUE, ModelConfig, NNUEConfig
from model.modules.features import DEFAULT_FEATURES
from model.utils.load_model import load_model
from model.utils.serialize import NNUEReader, NNUEWriter


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


def test_old_network_hash_is_rejected():
    config = ModelConfig(L1=4, L2=4, L3=4)
    nnue = NNUE(NNUEConfig(model_config=config))
    nnue.model.router.lut_valid = True
    data = bytearray(NNUEWriter(nnue.model, verbose=False).buf)
    # The MoE8.5 router tag ended in 0x35; MoE8.6 ends in 0x36.
    data[4] ^= 0x35 ^ 0x36
    with pytest.raises(ValueError, match="Expected hash"):
        NNUEReader(io.BytesIO(data), DEFAULT_FEATURES, config)
