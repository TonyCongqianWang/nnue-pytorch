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
    target = router.balance_rank_target_tail
    torch.testing.assert_close(target[0], torch.tensor(4 / 64))
    torch.testing.assert_close(target.sum(), torch.tensor(1.0))
    router.update_traffic("tail", traffic)
    router.running_offset_tail.copy_(torch.linspace(-0.3, 0.3, 64))
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
    torch.testing.assert_close(other.running_offset_tail, router.running_offset_tail)
    torch.testing.assert_close(other.ema_traffic_tail, router.ema_traffic_tail)
    assert other.routing_diagnostics == router.routing_diagnostics


def test_frozen_offsets_match_lut_and_unbiased_validation():
    router = small_router()
    router.routing_offset_tail.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    router.running_offset_tail.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))
    router.eval()
    router.collect_lut_diagnostics = True
    contexts = torch.tensor([0, 1, 41, 159])
    before = router.routing_offset_tail.clone()
    ema = router.ema_traffic_tail.clone()
    running = router.running_offset_tail.clone()
    router.get_lookup_table()
    expected = router.distances(
        router.get_context_vector(contexts), router.codebook[0], "tail", inference=True
    ).argmin(-1)
    first = contexts // 40
    full = contexts % 40
    fc0, tail, *_ = router(first, full)
    torch.testing.assert_close(tail, expected)
    assert (tail == 1).all()
    assert (
        router.distances(
            router.get_context_vector(contexts), router.codebook[0], "tail"
        ).argmin(-1)
        == 0
    ).all()
    torch.testing.assert_close(router.running_offset_tail, running)
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
    for name in router.active_banks:
        getattr(router, f"routing_offset_{name}").fill_(0.3)
        getattr(router, f"running_offset_{name}").fill_(-0.7)
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
        torch.testing.assert_close(
            getattr(router, f"running_offset_{name}"),
            getattr(router, f"routing_offset_{name}"),
        )
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
    compiled = torch.compile(router, backend="eager")
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


def test_cold_start_does_not_credit_fictitious_uniform_traffic():
    router = MoERouter(
        64,
        64,
        dim=8,
        context_dim=32,
        shared=True,
        material_states=2,
        balance_power=0.4658379531227075,
    )
    counts = torch.zeros(64)
    counts[33] = 131072
    router.update_traffic("tail", counts)
    torch.testing.assert_close(router.ema_traffic_tail, counts / counts.sum())
    assert router.traffic_momentum("tail") == 0


def test_bias_corrected_traffic_momentum_matches_its_definition():
    router = small_router()
    numerator = torch.zeros(4)
    for step, counts in enumerate(
        (
            torch.tensor([100.0, 0, 0, 0]),
            torch.tensor([0.0, 50, 50, 0]),
            torch.tensor([0.0, 0, 0, 100]),
        ),
        1,
    ):
        numerator = 0.99 * numerator + 0.01 * counts / counts.sum()
        router.update_traffic("tail", counts)
        torch.testing.assert_close(
            router.ema_traffic_tail, numerator / (1 - 0.99**step)
        )
    router.balance_steps_tail.fill_(2049)
    torch.testing.assert_close(router.traffic_momentum("tail"), torch.tensor(0.99))
    other = small_router()
    other.load_state_dict(router.state_dict())
    torch.testing.assert_close(other.balance_steps_tail, router.balance_steps_tail)


def test_narrow_cell_split_can_compete_with_a_nearby_expert():
    router = small_router()
    with torch.no_grad():
        raw = router.codebooks["tail"]
        raw.zero_()
        raw[0, 0] = 1
        raw[1, 2] = 1
        raw[2, 0] = 1
        raw[2, 1] = 0.005
        raw[3, 3] = 1
        points = torch.zeros(128, 8)
        points[:, 0] = 1
        points[:, 1] = torch.linspace(-0.001, 0.001, 128)
        points = torch.nn.functional.normalize(points, dim=1)
        router.recent_contexts_tail[:128] = torch.arange(128)
        router.recent_context_sizes["tail"] = 128
        router.ema_traffic_tail.copy_(torch.tensor([0.9, 0.001, 0.05, 0.049]))
        router.get_context_vector = lambda ids, bank="tail": points[ids]
        router.reseed_dead_experts_from_traffic()
    assert router.last_reseed_parents["tail"] == {1: 0}
    detail = router.last_reseed_details["tail"][1]
    assert 0 < detail["separation_radians"] < 0.01
    assert min(detail["sample_parent"], detail["sample_child"]) >= 12


def test_early_diagnostics_leave_learning_state_untouched(capsys):
    from torch import nn

    from model.config import ModelConfig
    from model.modules.layer_stacks import LayerStacks
    from model.quantize import QuantizationConfig, QuantizationManager
    from trainer.callbacks import MoEDeadExpertReseeder

    config = ModelConfig(
        L1=8, L2=4, L3=4, moe_shared_router=True, moe_fc0_experts=4, moe_tail_experts=4
    )
    model = nn.Module()
    model.router = small_router()
    model.layer_stacks = LayerStacks(
        4,
        config,
        QuantizationManager(QuantizationConfig()),
        codebook_dim=8,
        fc0_count=4,
        fc0_dim=8,
    )
    model.adapter_diversity_cost = 0.016
    model.weight_diversity_costs = (0.002, 0.002)
    trainer = SimpleNamespace(
        model=SimpleNamespace(model=model), rank=0, global_step=0, optimizer=None
    )
    batch = (torch.zeros(4, 1),) * 6 + (torch.arange(4), torch.arange(4))
    parameters = [p.detach().clone() for p in model.parameters()]
    callback = MoEDeadExpertReseeder()
    callback.on_train_batch_start(trainer, batch, 0)
    assert model.router.balance_steps_tail == 0
    assert model.router.routing_offset_tail.count_nonzero() == 0
    assert model.router.loss_component_batches == 0
    assert model.layer_stacks.regularization_batches == 0
    assert all(p.grad is None for p in model.parameters())
    for before, after in zip(parameters, model.parameters()):
        torch.testing.assert_close(before, after)
    text = capsys.readouterr().out
    assert "Early routing step=0" in text
    assert "Early gradients step=0" in text
    assert not model.router.validation_context_samples


def test_projection_init_scale_and_router_optimizer_group():
    import math

    from model import NNUE, ModelConfig, NNUEConfig
    from model.optimizers import OptimizerConfig

    torch.manual_seed(91)
    router = small_router()
    torch.manual_seed(91)
    old_projection = torch.empty_like(router.adapter_down)
    torch.nn.init.uniform_(old_projection, -1 / math.sqrt(32), 1 / math.sqrt(32))
    torch.testing.assert_close(router.adapter_down, 4 * old_projection, rtol=0, atol=0)
    model_config = ModelConfig(
        L1=8,
        L2=4,
        L3=4,
        moe_shared_router=True,
        moe_fc0_experts=4,
        moe_tail_experts=4,
        moe_context_dim=32,
        moe_fc0_codebook_dim=8,
        moe_tail_codebook_dim=8,
    )
    nnue = NNUE(
        NNUEConfig(
            features="K32H3^",
            model_config=model_config,
            optimizer_config=OptimizerConfig(
                optimizer_name="adamw", dense_weight_decay=0.001
            ),
        )
    )
    optimizers, _ = nnue.configure_optimizers()
    groups = optimizers[0].param_groups
    projection = nnue.model.router.adapter_down
    matches = [g for g in groups if any(p is projection for p in g["params"])]
    assert len(matches) == 1 and matches[0]["weight_decay"] == 0
    embedding = next(nnue.model.router.encoders["tail"].parameters())
    assert any(p is embedding for p in matches[0]["params"])
    dense = nnue.model.layer_stacks.l1.adapter_up
    assert (
        next(g for g in groups if any(p is dense for p in g["params"]))["weight_decay"]
        == 0.001
    )


def test_quantile_solver_uses_fixed_rank_target_and_reduces_imbalance():
    from model.modules.quantile_routing import ITERATIONS, solve_biases

    torch.manual_seed(31)
    scores = torch.randn(2048, 8) * 0.3 + torch.linspace(0.5, -0.5, 8)
    ranked = torch.arange(1, 9).float().pow(-0.4658379531227075)
    ranked /= ranked.sum()
    result = solve_biases(
        scores,
        torch.zeros(8),
        ranked,
        torch.full_like(ranked, 1 / len(ranked)),
        0.25,
        0.25,
        0.01,
        4.0,
    )
    biases, after, before, target, _shrinkage, _normalization = result
    assert ITERATIONS == 4
    initial = before / before.sum()
    order = before.argsort(descending=True, stable=True)
    torch.testing.assert_close(target[order], 0.75 * initial[order] + 0.25 * ranked)
    assert (after / after.sum() - target).abs().sum() < (initial - target).abs().sum()
    assert biases.max() - biases.min() <= 4.000001
    assert after.sum() == len(scores)
    untouched = solve_biases(
        scores,
        torch.zeros(8),
        ranked,
        torch.full_like(ranked, 1 / len(ranked)),
        0.0,
        0.0,
        0.01,
        4.0,
    )
    torch.testing.assert_close(untouched[0], torch.zeros(8))
    torch.testing.assert_close(untouched[1], untouched[2])
    # All iterations are tensor operations, suitable for one root-GPU graph.
    compiled = torch.compile(solve_biases, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(
            scores,
            torch.zeros(8),
            ranked,
            torch.full_like(ranked, 1 / len(ranked)),
            0.25,
            0.25,
            0.01,
            4.0,
        ),
        result,
    )


def test_quantile_route_records_final_counts_and_uses_corrected_assignments(
    monkeypatch,
):
    router = small_router()
    monkeypatch.setattr(
        torch.nn.functional,
        "softmax",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No routing softmax")),
    )
    ids = torch.arange(64) % 4
    fc0, tail, total, context_loss, centroid_loss = router(ids, ids)
    counts = torch.bincount(tail, minlength=4).float()
    torch.testing.assert_close(router.batch_traffic_tail, counts / counts.sum())
    torch.testing.assert_close(router.ema_traffic_tail, counts / counts.sum())
    torch.testing.assert_close(fc0, tail)
    torch.testing.assert_close(total, context_loss + centroid_loss)
    torch.testing.assert_close(total, router.loss_component_sum.sum())
    assert router.balance_steps_tail == 1
    assert router.requested_traffic_tail.sum().isclose(torch.tensor(1.0))


def test_refined_quantiles_resolve_clustered_columns():
    from model.modules.quantile_routing import column_quantiles

    torch.manual_seed(8)
    values = torch.randn(2048, 4) * 1e-4 + torch.tensor([0.0, 1.0, -0.2, 2.0])
    values[:2] += torch.tensor([0.01, 0.05, 0.03, 0.02])
    fractions = torch.tensor([0.1, 0.3, 0.8, 0.99])
    expected = torch.stack(
        [torch.quantile(values[:, i], fractions[i]) for i in range(4)]
    )
    actual = column_quantiles(values, fractions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=4e-6)


def test_gather_solver_runs_only_on_root_and_broadcasts_global_result(monkeypatch):
    from model.modules import quantile_routing as module

    torch.manual_seed(4)
    parts = [torch.randn(16, 4), torch.randn(16, 4)]
    rank = [0]
    calls = []
    shared = {}
    original = module.solve_biases

    def solve(scores, *args):
        assert (
            rank[0] == 0 and scores.shape == (32, 4) and scores.dtype == torch.float32
        )
        calls.append("solve")
        return original(scores, *args)

    def gather(scores, gather_list, dst):
        assert dst == 0
        calls.append(("gather", rank[0]))
        if rank[0] == 0:
            for output, part in zip(gather_list, parts):
                output.copy_(part)
        else:
            assert gather_list is None

    def broadcast(result, src):
        assert src == 0
        calls.append(("broadcast", rank[0]))
        if rank[0] == 0:
            shared["result"] = result.clone()
        else:
            result.copy_(shared["result"])

    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank[0])
    monkeypatch.setattr(module.dist, "gather", gather)
    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    monkeypatch.setattr(module, "solve_biases", solve)
    target = torch.full((4,), 0.25)
    root, world = module.gather_and_solve(
        parts[0], torch.zeros(4), target, target, 0.25, 0.75, 0.01, 4.0
    )
    rank[0] = 1
    other, other_world = module.gather_and_solve(
        parts[1], torch.zeros(4), target, target, 0.25, 0.75, 0.01, 4.0
    )
    assert world == other_world == 2
    torch.testing.assert_close(root, other)
    assert calls == [
        ("gather", 0),
        "solve",
        ("broadcast", 0),
        ("gather", 1),
        ("broadcast", 1),
    ]


def test_constant_margin_quantiles_remain_finite():
    from model.modules.quantile_routing import column_quantiles, solve_biases

    values = torch.tensor([0.2, -0.1, 0.0, 2.0]).repeat(16, 1)
    quantiles = column_quantiles(values, torch.tensor([0.1, 0.5, 0.9, 1.0]))
    torch.testing.assert_close(quantiles, values[0], rtol=0, atol=1e-12)
    result = solve_biases(
        values,
        torch.zeros(4),
        torch.full((4,), 0.25),
        torch.full((4,), 0.25),
        0.25,
        0.75,
        0.01,
        4.0,
    )
    assert result.isfinite().all() and result[1].sum() == 16


def test_ema_recovery_weights_and_global_normalization():
    from model.modules.quantile_routing import routing_targets

    # The EMA changes recovery strength, not the batch ranking or target curve.
    initial = torch.tensor([0.7, 0.2, 0.1, 0.0])
    ranked = torch.tensor([0.4, 0.3, 0.2, 0.1])
    ema = torch.tensor([0.6, 0.3, 0.1, 0.0])
    target, alpha, normalization = routing_targets(
        initial, ranked, ema, 0.25, 0.75, 0.01
    )
    expected_alpha = 0.25 + 0.5 * (1 - (0.99 * ema + 0.01 / 4) / ranked).clamp(0, 1)
    raw = initial + expected_alpha * (ranked - initial)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(target, raw / raw.sum())
    torch.testing.assert_close(normalization, raw.sum())
    assert target.sum().isclose(torch.tensor(1.0)) and (target >= 0).all()
    assert alpha[0] == 0.25 and alpha[-1] > 0.73
    baseline, _, _ = routing_targets(initial, ranked, ema, 0.25, 0.25, 0.01)
    assert target[-1] > baseline[-1]
    # A stale starvation EMA also increases downward shrinkage after recovery.
    recovered = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, recovered_alpha, _ = routing_targets(recovered, ranked, ema, 0.25, 0.75, 0.01)
    assert recovered_alpha[-1] > 0.74
    # No backward path is used by the actual routing controller.
    assert not target.requires_grad


def test_running_offsets_use_batchnorm_style_mean_without_bias_correction():
    router = small_router(offset_ema_decay=0.9)
    assert router.running_offset_tail.count_nonzero() == 0
    with torch.no_grad():
        codebook = router.codebook[0]
        z = torch.nn.functional.normalize(torch.randn(64, 8), dim=1)
        router.quantize(z, codebook, "tail")
        first = router.routing_offset_tail.clone()
        torch.testing.assert_close(router.running_offset_tail, 0.1 * first)
        router.quantize(z.flip(0), codebook, "tail")
        second = router.routing_offset_tail.clone()
        torch.testing.assert_close(
            router.running_offset_tail, 0.9 * 0.1 * first + 0.1 * second
        )
