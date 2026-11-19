import pytest
import torch

from model.modules.router import (
    MOE_BANKS,
    MoERouter,
    codebook_diversity_loss,
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


def test_commitment_is_dimension_independent_and_matches_2048_baseline():
    torch.manual_seed(17)
    router = MoERouter(material_states=2, fc0_dim=4, dim=4)
    context = torch.randn(5, 8, requires_grad=True)
    selected = torch.randn(5, 8, requires_grad=True)
    value = router.commitment_loss(context, selected)
    expanded_context = context.repeat_interleave(256, dim=1) / 16
    expanded_selected = selected.repeat_interleave(256, dim=1) / 16
    expected = 0.005 * (
        0.25
        * torch.nn.functional.mse_loss(expanded_context, expanded_selected.detach())
        + torch.nn.functional.mse_loss(expanded_selected, expanded_context.detach())
    )
    expanded = router.commitment_loss(expanded_context, expanded_selected)
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(value, expanded)
    for actual in (expected, expanded):
        for left, right in zip(
            torch.autograd.grad(value, (context, selected), retain_graph=True),
            torch.autograd.grad(actual, (context, selected), retain_graph=True),
        ):
            torch.testing.assert_close(left, right)


def test_shared_gradient_diagnostics_preserve_parameter_grads_and_traffic():
    router = MoERouter(4, 4, dim=8, material_states=2, shared=True, diversity_cost=0)
    router.eval()
    router.collect_lut_diagnostics = True
    router.collect_joint_diagnostics = True
    router(torch.arange(4), torch.tensor([0, 1, 2, 3]))
    counts = router.validation_counts_tail.clone()
    state = {k: v.clone() for k, v in router.state_dict().items()}
    before = {
        k: getattr(router, k).clone() for k in ("ema_traffic_tail", "ema_soft_tail")
    }
    with torch.inference_mode():
        diagnostics = router.gradient_diagnostics("tail", counts)
    assert diagnostics["sample_n"] == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics.values())
    assert diagnostics["commitment_encoder_grad"] > 0
    assert diagnostics["commitment_codebook_grad"] > 0
    assert router.loss_component_names == (
        "stack_commitment",
        "stack_balance",
        "stack_diversity",
    )
    assert all(p.grad is None for p in router.parameters())
    for key, tensor in router.state_dict().items():
        torch.testing.assert_close(tensor, state[key])
    for key, tensor in before.items():
        torch.testing.assert_close(getattr(router, key), tensor)


def test_weight_diversity_adds_to_bottleneck_diversity():
    from model.config import ModelConfig
    from model.modules.layer_stacks import LayerStacks
    from model.quantize import QuantizationConfig, QuantizationManager

    stacks = LayerStacks(
        4,
        ModelConfig(L1=8, L2=4, L3=4),
        QuantizationManager(QuantizationConfig()),
        codebook_dim=8,
        fc0_count=4,
        fc0_dim=8,
    )
    codebooks = (
        torch.randn(4, 8, requires_grad=True),
        torch.randn(4, 8, requires_grad=True),
    )
    costs = (0.004, 0.006)
    _, bottleneck = stacks.training_parameters(codebooks, 0.004, (0, 0))
    _, weights = stacks.training_parameters(codebooks, 0, costs)
    parameters, both = stacks.training_parameters(codebooks, 0.004, costs)
    torch.testing.assert_close(both, bottleneck + weights)
    for layer, vectors, params in zip(
        stacks.dense_layers(), (codebooks[0], codebooks[1], codebooks[1]), parameters
    ):
        for cached, fresh in zip(params, layer.get_all_weights_and_biases(vectors)):
            torch.testing.assert_close(cached, fresh)
    projection = stacks.l1.adapter_up
    bottleneck_gradient = torch.autograd.grad(
        bottleneck, projection, allow_unused=True, retain_graph=True
    )[0]
    assert bottleneck_gradient is None
    weight_gradient = torch.autograd.grad(weights, projection)[0]
    both_gradient = torch.autograd.grad(both, projection)[0]
    assert weight_gradient.abs().sum() > 0
    torch.testing.assert_close(both_gradient, weight_gradient)


def test_rank_power_target_attracts_top_expert_and_resists_collapse():
    router = MoERouter(
        64,
        64,
        dim=8,
        material_states=2,
        shared=True,
        balance_power=0.4658379531227075,
    )
    traffic = torch.arange(64, 0, -1, dtype=torch.float32)
    target = router.balance_target("tail", traffic)
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    torch.testing.assert_close(target[0], torch.tensor(4 / 64))
    assert target[-1] > 0.5 / 64
    assert not target.requires_grad
    # The same rank curve follows a different expert when the EMA order changes.
    reverse = router.balance_target("tail", traffic.flip(0))
    torch.testing.assert_close(reverse, target.flip(0))

    for top_share, expected_sign in ((1 / 64, -1), (0.2, 1)):
        hard = torch.full((64,), (1 - top_share) / 63)
        hard[0] = top_share
        logits = hard.log().requires_grad_()
        soft = logits.softmax(0)
        loss = router.traffic_balance_loss(soft, hard, 0.001, target)
        (gradient,) = torch.autograd.grad(loss, logits)
        assert expected_sign * gradient[0] > 0

    collapsed = torch.zeros(64)
    collapsed[0] = 1
    logits = torch.zeros(64, requires_grad=True)
    loss = router.traffic_balance_loss(logits.softmax(0), collapsed, 0.001, target)
    (gradient,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient[0] > 0
    assert gradient[1:].sum() < 0
    # Check recovery of a missing expert at both ends of the rank curve.
    for missing in (0, 63):
        hard = target.clone()
        hard[missing] = 0
        hard /= hard.sum()
        logits = torch.zeros(64, requires_grad=True)
        loss = router.traffic_balance_loss(logits.softmax(0), hard, 0.001, target)
        (gradient,) = torch.autograd.grad(loss, logits)
        assert gradient[missing] < 0


def test_codebook_cosine_diversity_penalizes_a_shared_direction():
    distinct = torch.eye(4)
    clustered = torch.nn.functional.normalize(
        torch.cat((0.01 * distinct, torch.ones(4, 1)), dim=1), dim=1
    ).requires_grad_()
    torch.testing.assert_close(codebook_diversity_loss(distinct), torch.zeros(()))
    loss = codebook_diversity_loss(clustered)
    assert loss > 0.99
    (gradient,) = torch.autograd.grad(loss, clustered)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_shared_codebook_is_the_adapter_for_every_layer():
    from model.config import ModelConfig
    from model.modules.layer_stacks import LayerStacks
    from model.quantize import QuantizationConfig, QuantizationManager

    router = MoERouter(64, 64, dim=64, context_dim=2048, shared=True, material_states=2)
    config = ModelConfig(
        L1=8,
        L2=4,
        L3=4,
        moe_shared_router=True,
        moe_fc0_experts=64,
        moe_tail_experts=64,
    )
    stacks = LayerStacks(64, config, QuantizationManager(QuantizationConfig()))
    codebook = router.codebook
    assert codebook[0] is codebook[1]
    assert router.encoders["tail"].sides[0][0].weight.shape[1] == 2048
    assert router.adapter_down.shape == (64, 2048)
    assert router.codebooks["tail"].shape == (64, 64)
    assert torch.linalg.matrix_rank(codebook[0]) == 64
    for layer in stacks.dense_layers():
        assert "adapter_down" not in dict(layer.named_parameters())
        weight, bias = layer.get_all_weights_and_biases(codebook[0])
        torch.testing.assert_close(
            weight.reshape(64, -1),
            torch.nn.functional.linear(codebook[0], layer.adapter_up),
        )
        torch.testing.assert_close(
            bias.reshape(64, -1),
            torch.nn.functional.linear(codebook[0], layer.bias_adapter),
        )
    _, loss = stacks.training_parameters(codebook, 0.008, (0, 0))
    torch.testing.assert_close(loss, 0.008 * codebook_diversity_loss(codebook[0]))
    (gradient,) = torch.autograd.grad(loss, router.codebooks["tail"])
    assert gradient.abs().sum() > 0 and gradient.isfinite().all()
    indices = torch.arange(4)
    vectors = router.codebook
    fc0, tail, *_ = router(indices, indices, codebook=vectors)
    score = stacks(torch.rand(4, 8), vectors, fc0, tail, False, False).sum()
    projection_gradient, codebook_gradient = torch.autograd.grad(
        score, (router.adapter_down, router.codebooks["tail"]), allow_unused=True
    )
    assert projection_gradient is None
    assert codebook_gradient.isfinite().all() and codebook_gradient.abs().sum() > 0


def test_context_generator_projects_before_codebook_selection():
    from model.modules.router import (
        center_and_rms_norm,
        decompose_router_index,
        rms_norm,
    )

    router = MoERouter(
        4, 4, dim=8, context_dim=32, shared=True, material_states=2, diversity_cost=0
    )
    indices = torch.tensor([0, 1, 41, 42])
    ours, theirs, geometry = decompose_router_index(indices, material_states=2)
    raw = (
        router.encoders["tail"](ours, theirs, router.tail_material_decode)
        + router.king(geometry % 8)
        + router.wings(geometry // 8)
    )
    expected = rms_norm(
        torch.nn.functional.linear(
            torch.nn.functional.gelu(center_and_rms_norm(raw)), router.adapter_down
        )
    )
    actual = router.get_context_vector(indices)
    assert raw.shape == (4, 32) and actual.shape == (4, 8)
    torch.testing.assert_close(actual, expected)
    ids, _, commitment, _ = router.quantize(actual, router.codebook[0], 4)
    distances = (actual[:, None] - router.codebook[0][None]).square().sum(-1)
    torch.testing.assert_close(ids, distances.argmin(-1))
    (gradient,) = torch.autograd.grad(commitment, router.adapter_down)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_shared_reseeding_message_uses_stack_label(capsys):
    from types import SimpleNamespace

    from trainer.callbacks import MoEDeadExpertReseeder

    router = MoERouter(4, 4, dim=8, context_dim=16, material_states=2, shared=True)
    router.ema_traffic_tail.fill_(1 / 3)
    router.ema_traffic_tail[0] = 0
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(router=router)),
        global_step=1024,
        world_size=1,
        rank=0,
        optimizer=None,
    )
    MoEDeadExpertReseeder().on_train_batch_end(trainer)
    message = capsys.readouterr().out
    assert "Reseeded 1 stack expert(s)" in message
    assert "tail expert" not in message
