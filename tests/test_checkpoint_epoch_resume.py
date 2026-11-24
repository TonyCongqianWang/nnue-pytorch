import torch

from trainer.callbacks import Callback
from trainer.engine import SimpleTrainer


class Mean:
    def __init__(self):
        self.value = torch.tensor(0.0)

    def compute(self):
        return self.value


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.current_lambda = torch.ones(1)
        self.loss_metrics = {"train_loss_epoch": Mean()}

    def train_step(self, batch, epoch, step):
        loss = (self.weight - batch[0]).square().mean()
        self.loss_metrics["train_loss_epoch"].value = loss.detach()
        return {"loss": loss, "train_loss": loss.detach()}


def trainer(callbacks=()):
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return SimpleTrainer(
        model=model,
        optimizer=opt,
        schedulers=[torch.optim.lr_scheduler.StepLR(opt, 1)],
        max_epochs=1,
        check_val_every_n_epoch=1,
        gradient_clip_val=0,
        log_every_n_steps=1,
        default_root_dir=None,
        callbacks=callbacks,
        logger=None,
        device="cpu",
        rank=0,
        world_size=1,
        local_rank=0,
    )


def test_completed_epoch_checkpoint_does_not_repeat_steps(tmp_path):
    path = tmp_path / "epoch.ckpt"

    class Save(Callback):
        def on_train_epoch_end(self, t):
            t.save_checkpoint(path)

    first = trainer([Save()])
    first.fit([(torch.zeros(1),), (torch.zeros(1),)])
    saved = torch.load(path, weights_only=False)
    assert saved["epoch_finished"] is True
    assert saved["lr_schedulers"][0]["last_epoch"] == 1
    resumed = trainer()
    resumed.load_checkpoint(path)
    resumed.fit([(torch.zeros(1),), (torch.zeros(1),)])
    assert resumed.current_epoch == 1
    assert resumed.global_step == 2  # already reached max_epochs, no repeated epoch
    # Old epoch-end checkpoint format retains the same next-epoch semantics.
    del saved["epoch_finished"]
    torch.save(saved, path)
    resumed.load_checkpoint(path)
    assert resumed.current_epoch == 1


def test_interrupted_epoch_stays_on_that_epoch(tmp_path):
    path = tmp_path / "partial.ckpt"

    class Interrupt(Callback):
        def on_train_batch_end(self, t, *args):
            t.save_checkpoint(path)
            raise InterruptedError

    first = trainer([Interrupt()])
    try:
        first.fit([(torch.zeros(1),), (torch.zeros(1),)])
    except InterruptedError:
        pass
    resumed = trainer()
    saved = resumed.load_checkpoint(path)
    assert saved["epoch_finished"] is False
    assert resumed.current_epoch == 0
    assert resumed.global_step == 1
