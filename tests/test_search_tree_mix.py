import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_dataloader_read_balance import SKIP_ALL_OFF, _build_generator

from data_loader import SparseBatchDataset
from data_loader.config import DataloaderIOConfig
from train import parse_search_tree_schedule
from trainer.callbacks import SearchTreeMix


@pytest.mark.parametrize("tree_accept", [1.0, 0.1])
def test_search_tree_mix_updates_live(tmp_path, tree_accept):
    game_path = str(tmp_path / "games.binpack")
    tree_path = str(tmp_path / "tree.binpack")
    generator = _build_generator()
    subprocess.run([generator, game_path, "2048", "1"], check=True)
    subprocess.run([generator, tree_path, "2048", "2", "9999"], check=True)

    dataset = SparseBatchDataset(
        "K32H3",
        [game_path],
        256,
        num_workers=4,
        config=replace(SKIP_ALL_OFF, tree_sample_accept=tree_accept),
        io_config=DataloaderIOConfig(shuffle_buffer_entries=1024),
        search_tree_filenames=(tree_path,),
    )
    batches = iter(dataset)
    assert not (next(batches)[4] > 1).any()

    for fraction in (0.08, 0.15, 1.0, 0.0):
        dataset.set_search_tree_fraction(fraction)
        # Leave time for existing native batches to drain.
        for _ in range(16):
            next(batches)
        counts = [(next(batches)[4] > 1).sum().item() for _ in range(64)]
        assert sum(counts) / (64 * 256) == pytest.approx(fraction, abs=0.015)
        if fraction in (0.0, 1.0):
            assert all(count == 256 * fraction for count in counts)

    preskip, accepted, _ = batches.get_unique_stats()
    if tree_accept < 1.0:
        assert preskip > 2 * accepted
    else:
        assert preskip == pytest.approx(accepted, rel=0.05)

    with pytest.raises(ValueError):
        dataset.set_search_tree_fraction(float("nan"))


def test_search_tree_ramp_resumes_from_stage_origin():
    trainer = SimpleNamespace(global_step=1_000_000)
    dataset = Mock()
    callback = SearchTreeMix(dataset, 0.03, 0.15, 65536)
    callback.on_train_start(trainer)
    dataset.set_search_tree_fraction.assert_not_called()

    checkpoint = {}
    callback.on_save_checkpoint(trainer, checkpoint)
    trainer.global_step += 32768
    resumed = SearchTreeMix(dataset, 0.03, 0.15, 65536)
    resumed.on_load_checkpoint(trainer, checkpoint)
    resumed.on_train_start(trainer)
    dataset.set_search_tree_fraction.assert_called_once_with(pytest.approx(0.09))

    trainer.global_step += 32768
    resumed._update(trainer.global_step)
    dataset.set_search_tree_fraction.assert_called_with(0.15)


def test_search_tree_schedule_interpolates_and_resumes():
    knots = parse_search_tree_schedule("0:0,0.5:0.02,0.75:0.06,0.875:0.12,1:0.12")
    trainer = SimpleNamespace(global_step=0)
    dataset = Mock()
    callback = SearchTreeMix(dataset, 0.0, ramp_steps=800, knots=knots)
    callback.on_train_start(trainer)
    assert callback.target == 0.0
    for step, expected in [
        (200, 0.01),
        (400, 0.02),
        (600, 0.06),
        (650, 0.09),
        (700, 0.12),
        (900, 0.12),
    ]:
        callback._update(step)
        assert callback.target == pytest.approx(expected)

    checkpoint = {}
    callback.on_save_checkpoint(trainer, checkpoint)
    trainer.global_step = 650
    resumed = SearchTreeMix(dataset, 0.0, ramp_steps=800, knots=knots)
    resumed.on_load_checkpoint(trainer, checkpoint)
    resumed.on_train_start(trainer)
    assert resumed.target == pytest.approx(0.09)


@pytest.mark.parametrize("spec", ["", "0:0", "0:0,0:0.2,1:1", "0:0,1:nan", "0.1:0,1:1"])
def test_invalid_search_tree_schedule(spec):
    with pytest.raises(ValueError, match="search-tree-schedule|Search-tree schedule"):
        parse_search_tree_schedule(spec)
