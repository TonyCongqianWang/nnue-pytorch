import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_dataloader_read_balance import SKIP_ALL_OFF, _build_generator

from data_loader import SparseBatchDataset
from data_loader.config import DataloaderIOConfig
from trainer.callbacks import SearchTreeMix


def test_search_tree_mix_updates_live(tmp_path):
    game_path = str(tmp_path / "games.binpack")
    tree_path = str(tmp_path / "tree.binpack")
    generator = _build_generator()
    subprocess.run([generator, game_path, "2048", "1"], check=True)
    subprocess.run([generator, tree_path, "2048", "2", "9999"], check=True)

    # A zero legacy acceptance rate must have no effect on the separate source.
    dataset = SparseBatchDataset(
        "K32H3",
        [game_path],
        256,
        num_workers=4,
        config=replace(SKIP_ALL_OFF, tree_sample_accept=0.0),
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
