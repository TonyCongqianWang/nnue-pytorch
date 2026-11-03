"""MoE8.5 checks use initialization/inference only; no training or optimizer steps."""

import csv
from types import SimpleNamespace

import torch
from torch import nn

from model.modules.moe_linear import MoeLinear
from model.modules.router import MoERouter
from trainer.callbacks import CheckpointManager, MoEDeadExpertReseeder
from trainer.engine import SimpleTrainer


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


class ValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        r = MoERouter(
            material_states=2, num_fc0_experts=2, num_weight_experts=3, fc0_dim=8, dim=8
        )
        self.model = nn.Module()
        self.model.router = r
        r.fc0_lut.copy_(torch.arange(4) % 2)
        r.lut.copy_(torch.arange(32) % 3)
        r.lut_valid = True

    def val_step(self, batch, epoch, step):
        c0, ct, loss = batch
        self.model.router(c0, ct)
        return {"val_loss": loss}


class FreshValidationStream:
    def __init__(self):
        self.consumed = 0

    def __len__(self):
        return 2

    def __iter__(self):
        for _ in range(2):
            i = self.consumed
            self.consumed += 1
            c = torch.tensor([i % 4, (i + 1) % 4])
            yield c, c * 8 + i % 8, torch.tensor(float(i))


def test_final_validation_consumes_fresh_batches_and_saves_counts(tmp_path):
    trainer = SimpleTrainer.__new__(SimpleTrainer)
    trainer.model = ValidationModel()
    trainer.callbacks = [MoEDeadExpertReseeder()]
    trainer.current_epoch = 599
    trainer.global_step = 1
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.default_root_dir = str(tmp_path)
    trainer.sanity_checking = False
    trainer.callback_metrics = {}
    trainer.loggers = []
    stream = FreshValidationStream()
    result = trainer.validate(stream, repeats=4, final=True)
    assert stream.consumed == trainer.num_val_batches == 8
    assert result["val_loss_epoch"] == 3.5
    r = trainer.model.model.router
    assert int(r.validation_counts_fc0.sum()) == 16
    assert int(r.validation_counts_tail.sum()) == 16
    assert int(r.validation_joint_counts.sum()) == 16
    assert not r.collect_lut_diagnostics and not r.collect_joint_diagnostics
    rows = list(csv.DictReader((tmp_path / "final-routing.csv").open()))
    assert sum(int(x["count"]) for x in rows if x["bank"] == "joint") == 16
    saved = []
    trainer.logger = SimpleNamespace(version=0)
    trainer.save_checkpoint = saved.append
    trainer.flush_checkpoints = lambda: None
    manager = CheckpointManager(dirpath=str(tmp_path))
    manager._last_save_epoch = trainer.current_epoch
    manager.on_train_end(trainer)
    assert saved == [str(tmp_path / "last.ckpt")]


def test_nonzero_rank_joins_final_count_collectives(monkeypatch):
    r = ValidationModel().model.router
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(router=r)),
        rank=1,
        world_size=2,
        validation_is_final=True,
        sanity_checking=False,
    )
    callback = MoEDeadExpertReseeder()
    callback.on_validation_epoch_start(trainer)
    assert r.collect_lut_diagnostics and r.collect_joint_diagnostics
    r.validation_counts_fc0.fill_(20_000_001)
    calls = []

    def reduce(t):
        calls.append(t)
        t.mul_(2)

    monkeypatch.setattr("trainer.callbacks.dist.all_reduce", reduce)
    callback.on_validation_epoch_end(trainer)
    assert len(calls) == 3
    assert int(r.validation_counts_fc0[0]) == 40_000_002
    assert not r.collect_lut_diagnostics and not r.collect_joint_diagnostics
