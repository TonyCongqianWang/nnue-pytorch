import pytest
import torch
import torch.nn.functional as F

from model.modules.router import (
    MOE_BANKS,
    MoERouter,
)


def test_balance_loss_bounds():
    """Verify load balancing loss is 0.0 for uniform distribution and K - 1 for collapsed distribution."""
    num_experts = 32

    # 1. Uniform case:
    # 32 batch items, each routed to a distinct expert with uniform soft probability
    distances = torch.zeros(32, num_experts)
    for i in range(num_experts):
        distances[i, i] = -10.0  # expert i is closest
    # Softmax probabilities will be concentrated on i, so dispatch fraction f_i = 1/32
    # and soft prob P_i = 1/32
    # Excess balance loss = K * sum_j(f_j * P_j) - 1.0 = 32 * 32 * (1/32) * (1/32) - 1.0 = 0.0
    probs = F.softmax(-distances, dim=-1)
    indices = torch.arange(num_experts)
    dispatch_mask = F.one_hot(indices, num_classes=num_experts).float()
    f = dispatch_mask.mean(dim=0)
    P = probs.mean(dim=0)
    balance_loss = num_experts * torch.sum(f * P) - 1.0
    assert torch.isclose(balance_loss, torch.tensor(0.0), atol=1e-3)

    # 2. Collapsed case: all items routed to expert 0
    collapsed_indices = torch.zeros(32, dtype=torch.long)
    collapsed_mask = F.one_hot(collapsed_indices, num_classes=num_experts).float()
    f_collapsed = collapsed_mask.mean(dim=0)
    P_collapsed = collapsed_mask.mean(dim=0)  # 100% prob on 0
    balance_loss_collapsed = num_experts * torch.sum(f_collapsed * P_collapsed) - 1.0
    assert torch.isclose(
        balance_loss_collapsed, torch.tensor(float(num_experts - 1)), atol=1e-3
    )


@pytest.mark.parametrize("name", MOE_BANKS)
def test_traffic_only_reseeding_per_bank(name):
    r = MoERouter(num_contexts=2)
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
    r = MoERouter(num_contexts=2)
    # Function diagnostic only: no backward or parameter update.
    c = torch.tensor([0, 1])
    *_, loss, commitment, balance = r(c, c)
    torch.testing.assert_close(loss, r.loss_component_sum.sum())
    torch.testing.assert_close(commitment, r.loss_component_sum[::3].sum())
    torch.testing.assert_close(balance, r.loss_component_sum[1::3].sum())
    assert len(r.loss_component_sum) == 15
    assert r.loss_component_batches == 1
    assert all(
        torch.isclose(hard.sum(), torch.tensor(1.0)) for _, _, hard, _, _ in r.banks()
    )


def test_cached_lut_and_validation_counts():
    r = MoERouter(num_contexts=2)
    r.eval()
    calls = []

    def contexts():
        calls.append(1)
        return r.codebook[0][0][:2].detach()

    r.get_all_context_vectors = contexts
    r.collect_lut_diagnostics = True
    lut = r.get_lookup_table().clone()
    r(torch.tensor([0, 1]), torch.tensor([1, 0]))
    r.get_lookup_table()
    assert len(calls) == 1
    assert r.lut_margin_stats.shape == (5, 3)
    for name, _, _, _, _ in r.banks():
        assert getattr(r, "validation_counts_" + name).sum() == 2
    assert r.reseed_dead_experts_from_traffic() == ([], [], [], [], [])
    assert torch.equal(lut, r.get_lookup_table())


def test_ddp_reseeding_synchronizes_all_five_banks(monkeypatch):
    from types import SimpleNamespace

    from trainer import callbacks

    router = MoERouter(num_contexts=2)
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
    router.lut_valid = router.coalesced_valid = True
    callbacks.MoEDeadExpertReseeder().on_train_batch_end(trainer)
    assert len(broadcasts) == 10
    assert not router.lut_valid and not router.coalesced_valid
    for _, parameter, _, _, _ in router.banks():
        assert torch.equal(
            state[parameter]["exp_avg"][1], state[parameter]["exp_avg"][0]
        )
        assert torch.equal(
            state[parameter]["exp_avg_sq"][1], state[parameter]["exp_avg_sq"][0]
        )
