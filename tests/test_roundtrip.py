import io
import os
import subprocess
import sys
import tempfile

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter


def test_random_weights_roundtrip():
    torch.manual_seed(42)

    config = ModelConfig()
    model = NNUEModel(
        feature_name=DEFAULT_FEATURES,
        config=config,
        num_ls_buckets=32,
    )
    # Clip weights to quantization ranges
    model.clip_weights(include_input=True)

    # 1. Export initial model
    writer1 = NNUEWriter(model, description="Test Random Net", verbose=False)
    bytes1 = bytes(writer1.buf)
    assert len(bytes1) > 0

    # 2. Read back into model2
    reader1 = NNUEReader(io.BytesIO(bytes1), DEFAULT_FEATURES, config, num_ls_buckets=32)
    model2 = reader1.model

    # 3. Export model2
    writer2 = NNUEWriter(model2, description=reader1.description, verbose=False)
    bytes2 = bytes(writer2.buf)

    assert bytes1 == bytes2, f"Round-trip 1 mismatch: {len(bytes1)} vs {len(bytes2)} bytes"

    # 4. Read back into model3
    reader2 = NNUEReader(io.BytesIO(bytes2), DEFAULT_FEATURES, config, num_ls_buckets=32)
    model3 = reader2.model

    # 5. Export model3
    writer3 = NNUEWriter(model3, description=reader2.description, verbose=False)
    bytes3 = bytes(writer3.buf)

    assert bytes2 == bytes3, f"Round-trip 2 mismatch: {len(bytes2)} vs {len(bytes3)} bytes"


def test_stockfish_roundtrip_compatibility():
    stockfish_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Stockfish/src/stockfish")
    )
    if not os.path.exists(stockfish_path):
        pytest.skip(f"Stockfish binary not found at {stockfish_path}")

    torch.manual_seed(123)
    config = ModelConfig()
    model = NNUEModel(
        feature_name=DEFAULT_FEATURES,
        config=config,
        num_ls_buckets=32,
    )
    model.clip_weights(include_input=True)

    with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as tmp:
        net_path = tmp.name

    try:
        writer = NNUEWriter(model, description="Test PyTorch SF Parity Net", verbose=False)
        with open(net_path, "wb") as f:
            f.write(writer.buf)

        # Query Stockfish with UCI EvalFile
        cmd = [
            stockfish_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        uci_commands = f"uci\nsetoption name EvalFile value {net_path}\nisready\nposition startpos\neval\nquit\n"
        stdout, stderr = proc.communicate(input=uci_commands, timeout=30)
        assert proc.returncode == 0, f"Stockfish failed (code {proc.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        assert "NNUE evaluation" in stdout, f"Stockfish failed to evaluate with exported net:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    finally:
        if os.path.exists(net_path):
            os.remove(net_path)
