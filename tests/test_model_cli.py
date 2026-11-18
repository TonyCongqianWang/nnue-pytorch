"""Architecture CLI propagation and inference roundtrip; no optimizer steps."""

import argparse
import io
import shlex
import sys

import pytest
import torch
import tyro

from config import TrainingConfig
from cross_check_eval import CliConfig as CrossCheckCli
from cross_check_eval import init_random_weights
from model import ModelConfig, NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter
from serialize import CliConfig as SerializeCli

SMALL_ARGS = [
    "--l1=32",
    "--moe-fc0-experts=4",
    "--moe-tail-experts=6",
    "--moe-fc0-codebook-dim=8",
    "--moe-tail-codebook-dim=16",
    "--moe-fc0-rank=3",
    "--moe-tail-rank=5",
]


@pytest.mark.parametrize(
    "entry,required",
    [
        (TrainingConfig, ["unused.binpack"]),
        (SerializeCli, ["unused.ckpt", "unused.nnue"]),
        (CrossCheckCli, ["--engine=unused", "--data=unused"]),
    ],
)
def test_model_flags_are_shared_by_all_entry_points(entry, required):
    config = tyro.cli(entry, args=required + SMALL_ARGS).nnue_config.model_config
    model = NNUEModel(DEFAULT_FEATURES, config)
    fc0, fc1, output = model.layer_stacks.dense_layers()
    assert [
        (layer.count, layer.rank, layer.codebook_dim) for layer in (fc0, fc1, output)
    ] == [
        (4, 3, 8),
        (6, 5, 16),
        (6, 5, 16),
    ]
    assert model.router.codebooks["fc0"].shape == (4, 8)
    assert model.router.codebooks["tail"].shape == (6, 16)


def test_small_export_roundtrip_and_legacy_parser():
    parser = argparse.ArgumentParser()
    ModelConfig.add_model_args(parser)
    config = ModelConfig.get_model_config(parser.parse_args(SMALL_ARGS))
    model = NNUEModel(DEFAULT_FEATURES, config).eval()
    init_random_weights(model, build_router_lut=False)
    writer = NNUEWriter(model, verbose=False)
    restored = NNUEReader(io.BytesIO(writer.buf), DEFAULT_FEATURES, config).model.eval()
    inputs = torch.randint(0, 128, (12, 32)).float() / 128
    c0 = torch.arange(12)
    ct = c0 * 8
    before = model.layer_stacks(
        inputs, model.router.codebook, *model.router(c0, ct)[:2]
    )
    after = restored.layer_stacks(inputs, None, *restored.router(c0, ct)[:2])
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    wrong = ModelConfig(L1=32)
    with pytest.raises(ValueError, match="Expected hash"):
        NNUEReader(io.BytesIO(writer.buf), DEFAULT_FEATURES, wrong)


def test_pipeline_forwards_model_options_to_each_stage(monkeypatch, tmp_path):
    import test_training_pipeline_run as pipeline

    commands = []
    monkeypatch.setattr(pipeline, "run_command", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "-y",
            "--test-dir",
            str(tmp_path / "unused"),
            "--batch-size=32",
            "--epoch-size=256",
            *SMALL_ARGS,
        ],
    )
    pipeline.main()
    parsed = []
    for command in commands:
        args = shlex.split(command)
        for script, cli in [
            ("train.py", TrainingConfig),
            ("serialize.py", SerializeCli),
        ]:
            if script in args:
                result = tyro.cli(cli, args=args[args.index(script) + 1 :])
                parsed.append(result.nnue_config.model_config)
    assert len(parsed) == 8
    assert all(
        c.L1 == 32 and c.moe_fc0_experts == 4 and c.moe_tail_rank == 5 for c in parsed
    )
