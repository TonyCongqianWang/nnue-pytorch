from types import SimpleNamespace

import pytest
import torch

from model.modules.router import MoERouter, codebook_diversity_loss


def small_router(**kwargs):
    return MoERouter(
        4, 4, dim=8, context_dim=32, material_states=2, shared=True, **kwargs
    )


def test_commitment_reductions_and_stop_gradients():
    router = small_router()
    codebook = torch.eye(4, 8, requires_grad=True)
    indices = torch.tensor([0, 1, 1, 1])
    context = (
        codebook.detach()[indices]
        + torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    ).requires_grad_()
    encoder, centroid = router.commitment_losses(context, codebook, indices)
    torch.testing.assert_close(encoder, torch.tensor(0.0002 * 0.01))
    torch.testing.assert_close(centroid, torch.tensor(0.0001 * 0.01 * 2 / 4))
    torch.testing.assert_close(
        torch.autograd.grad(encoder, codebook, allow_unused=True)[0], None
    )
    assert torch.autograd.grad(centroid, context, allow_unused=True)[0] is None
    doubled = torch.cat((context[:1], context[1:].repeat(3, 1)))
    more_indices = torch.cat((indices[:1], indices[1:].repeat(3)))
    _, same_centroid = router.commitment_losses(doubled, codebook, more_indices)
    torch.testing.assert_close(centroid, same_centroid)
    expanded = router.commitment_losses(
        context.repeat_interleave(4, dim=1) / 2,
        codebook.repeat_interleave(4, dim=1) / 2,
        indices,
    )
    torch.testing.assert_close(encoder, expanded[0])
    torch.testing.assert_close(centroid, expanded[1])


def test_direct_feedback_is_bounded_and_needs_no_softmax(monkeypatch):
    router = small_router(balance_rate=0.1, balance_max_range=0.2)
    counts = torch.tensor([100.0, 0.0, 0.0, 0.0])
    before = router.routing_offset_tail.clone()
    router.update_traffic("tail", counts)
    assert router.routing_offset_tail[0] < before[0]
    assert (router.routing_offset_tail[1:] > before[1:]).all()
    assert (router.routing_offset_tail - before).abs().max() <= 0.100001
    # Inspect the range projection directly, without a training loop.
    router.routing_offset_tail.copy_(torch.tensor([-0.1, 0.1, 0.0, 0.0]))
    router.update_traffic("tail", counts)
    assert (
        router.routing_offset_tail.max() - router.routing_offset_tail.min() <= 0.200001
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "softmax",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("No balancing softmax")),
    )
    *_, total, encoder, centroid = router(torch.arange(4), torch.arange(4))
    torch.testing.assert_close(total, encoder + centroid)
    torch.testing.assert_close(total, router.loss_component_sum.sum())
    assert router.loss_component_names == (
        "stack_context_commitment",
        "stack_centroid_commitment",
    )


def test_power_target_and_controller_state_roundtrip():
    router = MoERouter(
        64,
        64,
        dim=8,
        context_dim=32,
        shared=True,
        material_states=2,
        balance_power=0.4658379531227075,
    )
    traffic = torch.arange(64, 0, -1.0)
    target = router.balance_target("tail", traffic)
    torch.testing.assert_close(target[0], torch.tensor(4 / 64))
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    router.update_traffic("tail", traffic)
    router.routing_diagnostics["tail"] = {"context_RMS": 0.001}
    other = MoERouter(
        64,
        64,
        dim=8,
        context_dim=32,
        shared=True,
        material_states=2,
        balance_power=0.4658379531227075,
    )
    other.load_state_dict(router.state_dict())
    torch.testing.assert_close(other.routing_offset_tail, router.routing_offset_tail)
    torch.testing.assert_close(other.ema_traffic_tail, router.ema_traffic_tail)
    assert other.routing_diagnostics == router.routing_diagnostics


def test_frozen_offsets_match_lut_and_unbiased_validation():
    router = small_router()
    router.routing_offset_tail.copy_(torch.tensor([0.3, -0.1, -0.1, -0.1]))
    router.eval()
    router.collect_lut_diagnostics = True
    contexts = torch.tensor([0, 1, 41, 159])
    before = router.routing_offset_tail.clone()
    ema = router.ema_traffic_tail.clone()
    router.get_lookup_table()
    expected = router.distances(
        router.get_context_vector(contexts), router.codebook[0], "tail"
    ).argmin(-1)
    first = contexts // 40
    full = contexts % 40
    fc0, tail, *_ = router(first, full)
    torch.testing.assert_close(tail, expected)
    torch.testing.assert_close(fc0, tail)
    assert router.validation_counts_tail.sum() == 4
    assert router.validation_unbiased_tail.sum() == 4
    torch.testing.assert_close(router.routing_offset_tail, before)
    torch.testing.assert_close(router.ema_traffic_tail, ema)
    orders = router.export_permutations()
    exported = router.export_lookup_tables(orders)[1]
    torch.testing.assert_close(
        exported[contexts], torch.argsort(orders["tail"])[expected]
    )
    router.gradient_diagnostics("tail", router.validation_counts_tail)
    assert all(p.grad is None for p in router.parameters())
    torch.testing.assert_close(router.routing_offset_tail, before)


def split_router(shared=True, collapsed=False):
    router = MoERouter(
        2, 2, fc0_dim=4, dim=4, context_dim=8, material_states=2, shared=shared
    )
    points = {}
    with torch.no_grad():
        for name, raw, traffic, offsets, cb in router.banks():
            raw.zero_()
            raw[0, 0] = 1
            raw[1, 3] = 1
            center = router.normalized_codebook(name)[0]
            axis = torch.tensor([0.0, 1.0, -1.0, 0.0]) / 2**0.5
            spread = torch.zeros(128) if collapsed else torch.linspace(-0.2, 0.2, 128)
            points[name] = torch.nn.functional.normalize(
                center[None] + spread[:, None] * axis, dim=1
            )
            traffic.copy_(torch.tensor([0.999, 0.001]))
            getattr(router, f"recent_contexts_{name}")[:128] = torch.arange(128)
            router.recent_context_sizes[name] = 128
    router.get_context_vector = lambda ids, bank="tail": points[bank][ids]
    return router


@pytest.mark.parametrize("shared", (False, True))
def test_geometric_splits_conserve_measured_mass_and_reset_drift(shared):
    from trainer.callbacks import MoEDeadExpertReseeder

    router = split_router(shared=shared)
    state = {}
    for _, raw, *_ in router.banks():
        state[raw] = {
            "exp_avg": torch.ones_like(raw) * 3,
            "exp_avg_sq": torch.arange(raw.numel(), dtype=raw.dtype).view_as(raw) + 1,
            "step": torch.tensor(123),
        }
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(router=router)),
        global_step=256,
        world_size=1,
        rank=0,
        optimizer=SimpleNamespace(state=state),
    )
    MoEDeadExpertReseeder().on_train_batch_end(trainer)
    for name, raw, traffic, *_ in router.banks():
        assert router.last_reseed_parents[name] == {1: 0}
        detail = router.last_reseed_details[name][1]
        ratio = detail["sample_child"] / (
            detail["sample_child"] + detail["sample_parent"]
        )
        torch.testing.assert_close(traffic.sum(), torch.tensor(1.0))
        torch.testing.assert_close(traffic[1], torch.tensor(ratio))
        assert state[raw]["exp_avg"].count_nonzero() == 0
        torch.testing.assert_close(
            state[raw]["exp_avg_sq"][1], state[raw]["exp_avg_sq"][0]
        )
        assert state[raw]["step"] == 123


def test_geometric_split_rejects_identical_contexts_without_fake_traffic():
    router = split_router(collapsed=True)
    raw = router.codebooks["tail"].detach().clone()
    traffic = router.ema_traffic_tail.clone()
    assert router.reseed_dead_experts_from_traffic() == ([],)
    assert router.reseed_history[-1]["rejected"]["no_spread"] == 1
    torch.testing.assert_close(router.codebooks["tail"], raw)
    torch.testing.assert_close(router.ema_traffic_tail, traffic)


def test_small_compiled_router_forward_and_backward_without_optimizer():
    router = small_router()
    compiled = torch.compile(router, backend="eager", fullgraph=True)
    *_, loss, _, _ = compiled(torch.arange(4), torch.arange(4))
    grads = torch.autograd.grad(loss, (router.adapter_down, router.codebooks["tail"]))
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in grads)


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

    router = MoERouter(4, 4, dim=8, context_dim=32, shared=True, material_states=2)
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
    ids, commitment, _ = router.quantize(
        actual, router.codebook[0], "tail", update=False
    )
    distances = (actual[:, None] - router.codebook[0][None]).square().sum(-1)
    torch.testing.assert_close(ids, distances.argmin(-1))
    (gradient,) = torch.autograd.grad(commitment, router.adapter_down)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_global_counts_normalize_ddp_centroid_gradients(monkeypatch):
    from model.modules import router as module

    router = small_router()
    global_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    contexts = torch.randn(6, 8)
    codebook = torch.randn(4, 8, requires_grad=True)
    counts, _ = router.assignment_counts(global_ids, 4)
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "all_reduce", lambda value: value.copy_(counts))
    gradients = []
    offsets = []
    for context, ids in zip(contexts.chunk(2), global_ids.chunk(2)):
        global_counts, world = router.assignment_counts(ids, 4, synchronize=True)
        _, loss = router.commitment_losses(context, codebook, ids, global_counts, world)
        gradients.append(torch.autograd.grad(loss, codebook)[0])
        replica = small_router()
        replica.update_traffic("tail", global_counts)
        offsets.append(replica.routing_offset_tail)
    _, reference = router.commitment_losses(contexts, codebook, global_ids)
    expected = torch.autograd.grad(reference, codebook)[0]
    torch.testing.assert_close(sum(gradients) / 2, expected)
    torch.testing.assert_close(*offsets)
