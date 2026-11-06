"""Final validation stream, global routing counts and checkpoint boundaries."""

import csv
from types import SimpleNamespace

import torch
from torch import nn

from model.modules.router import MoERouter
from trainer.callbacks import CheckpointManager, MoEDeadExpertReseeder
from trainer.engine import SimpleTrainer


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
