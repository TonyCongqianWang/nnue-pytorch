import pytest
import torch

from model.modules.router import (
    MOE_BANKS,
    MoERouter,
)


def test_hard_traffic_kl_counts_wins_even_with_nearly_uniform_soft_mass():
    router = MoERouter(material_states=2, fc0_dim=4, dim=4)
    codebook = torch.eye(4)
    balanced = codebook * 0.001
    collapsed = balanced[:1].repeat(4, 1).requires_grad_()
    _, _, _, uniform_loss = router.quantize(balanced, codebook, 4)
    _, _, _, collapsed_loss = router.quantize(collapsed, codebook, 4)
    torch.testing.assert_close(uniform_loss, torch.zeros(()))
    expected = -torch.tensor([3.85, 0.05, 0.05, 0.05]).log().mean()
    torch.testing.assert_close(collapsed_loss, expected)
    (gradient,) = torch.autograd.grad(collapsed_loss, collapsed)
    assert torch.isfinite(gradient).all()
    assert (gradient[:, 0] > 0).all()
    assert (gradient[:, 1:] < 0).all()


@pytest.mark.parametrize("name", MOE_BANKS)
def test_traffic_only_reseeding_per_bank(name):
    r = MoERouter(material_states=2, fc0_dim=8, dim=16)
    r.get_all_context_vectors = lambda: (_ for _ in ()).throw(
        AssertionError("No LUT rebuild for reseeding")
    )
    r.reseed_dead_experts_from_traffic()
    assert all(not parents for parents in r.last_reseed_parents.values())
    bank = next(b for b in r.banks() if b[0] == name)
    traffic = bank[2]
    traffic[0] = 0
    traffic[1] = 1
    r.reseed_dead_experts_from_traffic(temperature=1e-4)
    assert r.last_reseed_parents[name] == {0: 1}
    assert traffic[0] == traffic[1]
    assert not r.lut_valid


def test_role_losses_sum_and_traffic_updates_without_optimizer():
    r = MoERouter(material_states=2, fc0_dim=8, dim=16)
    # Function diagnostic only: no backward or parameter update.
    c = torch.tensor([0, 1])
    *_, loss, commitment, balance = r(c, c)
    torch.testing.assert_close(loss, r.loss_component_sum.sum())
    torch.testing.assert_close(commitment, r.loss_component_sum[::3].sum())
    torch.testing.assert_close(balance, r.loss_component_sum[1::3].sum())
    assert len(r.loss_component_sum) == 6
    assert r.loss_component_batches == 1
    assert all(
        torch.isclose(hard.sum(), torch.tensor(1.0)) for _, _, hard, _, _ in r.banks()
    )


def test_cached_lut_and_validation_counts():
    r = MoERouter(material_states=2, fc0_dim=8, dim=16)
    r.eval()
    r.collect_lut_diagnostics = True
    lut = tuple(t.clone() for t in r.get_lookup_table())
    r(torch.tensor([0, 1]), torch.tensor([1, 0]))
    r.get_lookup_table()
    assert r.lut_valid
    assert r.lut_margin_stats.shape == (2, 3)
    for name, _, _, _, _ in r.banks():
        assert getattr(r, "validation_counts_" + name).sum() == 2
    assert r.reseed_dead_experts_from_traffic() == ([], [])
    assert all(torch.equal(a, b) for a, b in zip(lut, r.get_lookup_table()))


def test_ddp_reseeding_synchronizes_both_banks(monkeypatch):
    from types import SimpleNamespace

    from trainer import callbacks

    router = MoERouter(material_states=2, fc0_dim=8, dim=16)
    state = {}
    for _, parameter, _, _, _ in router.banks():
        moment = torch.zeros_like(parameter)
        moment[0].fill_(3)
        state[parameter] = {"exp_avg": moment, "exp_avg_sq": moment.clone()}
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(router=router)),
        global_step=1024,
        world_size=2,
        rank=1,
        optimizer=SimpleNamespace(state=state),
    )
    monkeypatch.setattr(callbacks.dist, "is_available", lambda: True)
    monkeypatch.setattr(callbacks.dist, "is_initialized", lambda: True)
    broadcasts = []

    def broadcast(tensor, src):
        assert src == 0
        broadcasts.append(tensor)
        if tensor.dtype == torch.long:
            tensor[1] = 0  # Receive rank 0's parent map.

    monkeypatch.setattr(callbacks.dist, "broadcast", broadcast)
    router.lut_valid = True
    callbacks.MoEDeadExpertReseeder().on_train_batch_end(trainer)
    assert len(broadcasts) == 4
    assert not router.lut_valid
    for _, parameter, _, _, _ in router.banks():
        assert torch.equal(
            state[parameter]["exp_avg"][1], state[parameter]["exp_avg"][0]
        )
        assert torch.equal(
            state[parameter]["exp_avg_sq"][1], state[parameter]["exp_avg_sq"][0]
        )


def test_mixed_routes_preserve_winners_and_hard_traffic_kl():
    r = MoERouter(
        num_fc0_experts=2,
        num_weight_experts=4,
        material_states=2,
        fc0_dim=4,
        dim=4,
        tail_balance_kl_cost=0.002,
    )
    c = torch.tensor([0, 1, 3])
    fc0, tail, *_ = r(c, c)
    first = r.loss_component_sum.clone()
    r.loss_component_sum.zero_()
    fc0_k, tail_k, *_ = r(c, c, mix_experts=True)
    assert torch.equal(fc0, fc0_k[:, 0])
    assert torch.equal(tail, tail_k[:, 0])
    assert torch.equal(tail_k.sort(dim=1).values, torch.arange(4).expand(3, -1))
    assert int(r.runner_up_counts_fc0.sum()) == 3
    assert int(r.runner_up_counts_tail.sum()) == 9

    torch.testing.assert_close(r.loss_component_sum[:3], first[:3])
    for index, ids, count, cost in [(1, fc0, 2, 0.001), (4, tail, 4, 0.002)]:
        mean = torch.bincount(ids, minlength=count).float() / ids.numel()
        expected = -0.5 * cost * (0.95 * count * mean + 0.05).log().mean()
        torch.testing.assert_close(r.loss_component_sum[index], expected)
    r.eval()
    assert r(c, c, mix_experts=True)[1].shape == (3,)
