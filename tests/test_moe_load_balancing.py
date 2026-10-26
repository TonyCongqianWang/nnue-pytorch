import torch
import torch.distributed as dist
import torch.nn.functional as F

from model.modules.router import (
    NUM_WEIGHT_EXPERTS,
    ROUTER_TABLE_SIZE,
    DualVQCodebookRouter,
)
from trainer.callbacks import MoEDeadExpertReseeder


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


def test_dead_expert_reseeding():
    """Verify sustained low-traffic experts are reseeded without a context scan."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter()

    # Forcefully test reseeding with an extreme scenario:
    # Set all codebook entries to far away except expert 0
    with torch.no_grad():
        router.codebook_weights.data[1:] = 1000.0
        router.codebook_weights.data[0] = 0.0

    router.ema_traffic_weights.zero_()
    router.ema_traffic_weights[0] = 1.0

    def unexpected_context_scan():
        raise AssertionError("Reseeding must not enumerate contexts")

    router.get_all_context_vectors = unexpected_context_scan
    reseeded_w, reseeded_b = router.reseed_dead_experts_from_traffic()
    assert 0 in reseeded_w
    assert set(range(1, NUM_WEIGHT_EXPERTS)).issubset(set(reseeded_w))
    assert isinstance(reseeded_b, list)

    # Low-traffic experts should be reseeded near expert 0.
    for j in range(1, NUM_WEIGHT_EXPERTS):
        assert not torch.all(router.codebook_weights.data[j] == 1000.0), (
            f"Expert {j} was not reseeded!"
        )
    assert not router.lut_valid


def test_traffic_weighted_cell_splitting():
    """Verify EMA traffic tracking and traffic-weighted parent selection during cell splitting."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter()
    router.train()

    # Verify EMA buffer initialization
    assert router.ema_traffic_weights.shape == (NUM_WEIGHT_EXPERTS,)
    assert torch.isclose(router.ema_traffic_weights.sum(), torch.tensor(1.0), atol=1e-3)

    # Run forward step and verify EMA traffic update
    c_us = torch.zeros(32, dtype=torch.long)
    c_them = torch.zeros(32, dtype=torch.long)
    router(c_us, c_them)

    # Now make expert 31 dead while biasing EMA traffic heavily towards expert 5
    with torch.no_grad():
        router.codebook_weights.data[31] = 1000.0
        router.ema_traffic_weights.zero_()
        router.ema_traffic_weights[5] = 1.0

    # With greedy temperature (1e-4), expert 5 should be selected to split for dead expert 31
    reseeded_w, _ = router.reseed_dead_experts_from_traffic(temperature=1e-4)
    assert 31 in reseeded_w
    assert 5 in reseeded_w
    assert not torch.all(router.codebook_weights.data[31] == 1000.0)


def test_lut_cache_lifecycle():
    """Verify clean LUT cache invalidation, train-mode resets, and non-persistence."""
    router = DualVQCodebookRouter()
    assert not router.lut_valid

    # State dict should not contain ephemeral buffers lut and lut_valid
    sd = router.state_dict()
    assert "lut" not in sd
    assert "lut_valid" not in sd

    # update_lookup_table sets lut_valid = True
    lut = router.update_lookup_table()
    assert lut.shape[0] == ROUTER_TABLE_SIZE
    assert router.lut_valid

    # Switching to train mode invalidates the LUT cache
    router.train()
    assert not router.lut_valid


def test_dead_expert_callback_momentum_reset():
    """Verify MoEDeadExpertReseeder resets Adam optimizer momentum on dead and split parent rows."""
    from trainer.callbacks import MoEDeadExpertReseeder

    class DummyModel:
        def __init__(self, router):
            self.router = router

    class DummyNNUE:
        def __init__(self, router):
            self.model = DummyModel(router)

    class DummyTrainer:
        def __init__(self, nnue, optimizer):
            self.model = nnue
            self.optimizer = optimizer
            self.global_step = 1024
            self.rank = 0

    torch.manual_seed(42)
    router = DualVQCodebookRouter()
    # Only expert 31 is dead; set heavy traffic on expert 2 so it is selected as parent
    with torch.no_grad():
        router.codebook_weights.data[31] = -router.codebook_weights.data[2]
        router.ema_traffic_weights.fill_(0.02)
        router.ema_traffic_weights[31] = 0.0
        router.ema_traffic_weights[2] = 1.0

    opt = torch.optim.AdamW([router.codebook_weights, router.codebook_biases], lr=1e-3)
    # Simulate distinctive momentum buffers: expert 2 has momentum 2.5, expert 31 has 0.0
    exp_avg_w = torch.ones_like(router.codebook_weights)
    exp_avg_w[2] = 2.5
    exp_avg_w[31] = 0.0
    exp_avg_sq_w = torch.ones_like(router.codebook_weights) * 3.0
    exp_avg_sq_w[2] = 5.0
    exp_avg_sq_w[31] = 0.0

    opt.state[router.codebook_weights] = {
        "exp_avg": exp_avg_w,
        "exp_avg_sq": exp_avg_sq_w,
    }
    opt.state[router.codebook_biases] = {
        "exp_avg": torch.ones_like(router.codebook_biases),
        "exp_avg_sq": torch.ones_like(router.codebook_biases),
    }

    trainer = DummyTrainer(DummyNNUE(router), opt)
    callback = MoEDeadExpertReseeder(reseed_interval=1024, temperature=1e-4)
    callback.on_train_batch_end(trainer)

    # Expert 31 (reseeded child) should copy momentum from parent expert 2, parent retains 2.5
    state_w = opt.state[router.codebook_weights]
    assert torch.equal(state_w["exp_avg"][31], state_w["exp_avg"][2])
    assert torch.equal(state_w["exp_avg_sq"][31], state_w["exp_avg_sq"][2])
    assert state_w["exp_avg"][2].sum() == 2.5 * float(router.dim)
    assert state_w["exp_avg_sq"][2].sum() == 5.0 * float(router.dim)
    # An untouched active expert (e.g. expert 0) retains its momentum
    assert state_w["exp_avg"][0].sum() == float(router.dim)


def test_router_gradient_flow():
    """Verify that backpropagation produces non-zero gradients for router codebooks and embeddings."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter(balance_cost=0.02)
    router.train()

    c_us = torch.randint(0, 1000, (16,))
    c_them = torch.randint(0, 1000, (16,))

    (
        e1_w,
        e2_w,
        dyn_biases_1,
        dyn_biases_2,
        moe_loss,
        *_,
    ) = router(c_us, c_them)
    total_loss = (
        moe_loss
        + dyn_biases_1[0].sum()
        + dyn_biases_2[0].sum()
        + (e1_w.float().sum() + e2_w.float().sum()) * 0.0
    )
    total_loss.backward()

    assert router.codebook_weights.grad is not None
    assert router.codebook_biases.grad is not None
    assert router.bias_head_down.weight.grad is not None
    assert router.bias_head_up.weight.grad is not None
    assert torch.norm(router.codebook_weights.grad) > 0.0
    assert torch.norm(router.codebook_biases.grad) > 0.0
    assert torch.norm(router.bias_head_down.weight.grad) > 0.0
    assert torch.norm(router.bias_head_up.weight.grad) > 0.0


def test_dead_expert_callback_ddp_synchronization(monkeypatch):
    """Verify that under DDP, rank 0's reseeding decisions and weights are safely broadcast
    and optimizer momentum is reset on all ranks without collective mismatch."""

    class DummyModel:
        def __init__(self, router):
            self.router = router

    class DummyNNUE:
        def __init__(self, router):
            self.model = DummyModel(router)

    class DummyTrainer:
        def __init__(self, nnue, optimizer, rank, world_size=2):
            self.model = nnue
            self.optimizer = optimizer
            self.global_step = 1024
            self.rank = rank
            self.world_size = world_size

    # Setup Rank 0 with a dead expert 31
    torch.manual_seed(42)
    router_0 = DualVQCodebookRouter()
    with torch.no_grad():
        router_0.codebook_weights.data[31] = -router_0.codebook_weights.data[2]
        router_0.ema_traffic_weights.fill_(0.02)
        router_0.ema_traffic_weights[31] = 0.0
        router_0.ema_traffic_weights[2] = 1.0

    exp_avg_w = torch.ones_like(router_0.codebook_weights)
    exp_avg_w[2] = 2.5
    exp_avg_w[31] = 0.0
    exp_avg_sq_w = torch.ones_like(router_0.codebook_weights) * 3.0
    exp_avg_sq_w[2] = 5.0
    exp_avg_sq_w[31] = 0.0

    opt_0 = torch.optim.AdamW(
        [router_0.codebook_weights, router_0.codebook_biases], lr=1e-3
    )
    opt_0.state[router_0.codebook_weights] = {
        "exp_avg": exp_avg_w.clone(),
        "exp_avg_sq": exp_avg_sq_w.clone(),
    }

    # Setup Rank 1 (identical initial state, but no dead expert initially or different EMA)
    router_1 = DualVQCodebookRouter()
    with torch.no_grad():
        router_1.codebook_weights.copy_(router_0.codebook_weights)
        router_1.ema_traffic_weights.copy_(router_0.ema_traffic_weights)

    opt_1 = torch.optim.AdamW(
        [router_1.codebook_weights, router_1.codebook_biases], lr=1e-3
    )
    opt_1.state[router_1.codebook_weights] = {
        "exp_avg": exp_avg_w.clone(),
        "exp_avg_sq": exp_avg_sq_w.clone(),
    }

    trainer_0 = DummyTrainer(DummyNNUE(router_0), opt_0, rank=0, world_size=2)
    trainer_1 = DummyTrainer(DummyNNUE(router_1), opt_1, rank=1, world_size=2)

    broadcast_log = []
    shared_storage = {}

    def mock_broadcast(tensor, src=0):
        # Equal expert counts give the two parent maps identical shapes.
        # Collectives are matched by call order, not by tensor shape.
        ordinal = sum(rank == trainer_current_rank for rank, _ in broadcast_log)
        key = (ordinal, tensor.shape, tensor.dtype)
        if trainer_current_rank == src:
            shared_storage[key] = tensor.clone()
        else:
            tensor.copy_(shared_storage[key])
        broadcast_log.append((trainer_current_rank, key))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", mock_broadcast)

    callback = MoEDeadExpertReseeder(reseed_interval=1024, temperature=1e-4)

    # Rank 0 runs
    trainer_current_rank = 0
    callback.on_train_batch_end(trainer_0)

    # Rank 1 runs
    trainer_current_rank = 1
    callback.on_train_batch_end(trainer_1)

    # Both ranks must have invoked the exact same broadcast sequence
    calls_rank_0 = [k for r, k in broadcast_log if r == 0]
    calls_rank_1 = [k for r, k in broadcast_log if r == 1]
    assert len(calls_rank_0) == len(calls_rank_1)
    assert calls_rank_0 == calls_rank_1

    # Weights on Rank 1 must match Rank 0 exactly
    assert torch.equal(router_0.codebook_weights, router_1.codebook_weights)

    # Rank 1 optimizer momentum must be copied from parent expert 2 to child expert 31
    state_w_1 = opt_1.state[router_1.codebook_weights]
    state_w_0 = opt_0.state[router_0.codebook_weights]
    assert torch.equal(state_w_1["exp_avg"][31], state_w_1["exp_avg"][2])
    assert torch.equal(state_w_1["exp_avg_sq"][31], state_w_1["exp_avg_sq"][2])
    assert state_w_1["exp_avg"][2].sum() == 2.5 * float(router_1.dim)
    assert state_w_1["exp_avg"][0].sum() == float(router_1.dim)
    assert torch.equal(state_w_1["exp_avg"], state_w_0["exp_avg"])
    assert torch.equal(state_w_1["exp_avg_sq"], state_w_0["exp_avg_sq"])


def test_bias_reseeding_floor_scales_with_capacity():
    """A live 256-way bias cell must not inherit the 64-way absolute floor."""
    torch.manual_seed(7)
    router = DualVQCodebookRouter(num_contexts=256, dim=16)
    router.ema_traffic_biases[0] = 0.0004
    _, changed = router.reseed_dead_experts_from_traffic()
    assert not changed

    router.ema_traffic_biases[0] = 0.0001
    router.reseed_dead_experts_from_traffic()
    detail = router.last_reseed_details["biases"][0]
    assert detail["floor"] == 0.00025
    assert abs(detail["traffic"] - 0.0001) < 1e-9


def test_empty_lut_cells_with_healthy_traffic_do_not_trigger_reseeding():
    torch.manual_seed(7)
    router = DualVQCodebookRouter(num_contexts=2, dim=16)
    with torch.no_grad():
        router.codebook_biases[:] = router.codebook_biases[0].clone()
    router.get_all_context_vectors = lambda: router.codebook_b[:2].detach().clone()
    lut = router.get_lookup_table().clone()
    assert torch.all((lut >> 6) == 0)  # Experts 1..255 have no LUT contexts.
    original = router.codebook_biases.clone()

    def unexpected_context_scan():
        raise AssertionError("Healthy traffic must not trigger any LUT work")

    router.get_all_context_vectors = unexpected_context_scan
    assert router.reseed_dead_experts_from_traffic() == ([], [])
    assert router.lut_valid
    assert torch.equal(router.lut, lut)
    assert torch.equal(router.codebook_biases, original)


def test_lut_margin_diagnostics_reuse_one_cached_build():
    router = DualVQCodebookRouter(num_contexts=2, dim=16)
    router.collect_lut_diagnostics = True
    contexts = router.codebook_b[:2].detach().clone()
    calls = []

    def contexts_once():
        calls.append(1)
        return contexts

    router.get_all_context_vectors = contexts_once
    expected_w = torch.cdist(contexts, router.codebook_w).argmin(dim=1)
    expected_b = torch.cdist(contexts, router.codebook_b).argmin(dim=1)
    expected = expected_w | (expected_b << 6)
    assert torch.equal(router.get_lookup_table(), expected)
    assert router.lut_margin_valid
    assert router.lut_margin_samples == 1
    assert (router.lut_margin_stats >= 0).all()
    router.get_lookup_table()
    assert len(calls) == 1
    router.train()
    assert not router.lut_margin_valid


def test_bias_diversity_coefficient_is_independent():
    torch.manual_seed(7)
    router = DualVQCodebookRouter(dim=16)
    contexts = torch.tensor([0, 1, 2, 3])
    first = router(contexts, contexts)[4].detach()
    parts = router.loss_component_sum.clone()
    router.bias_diversity_cost *= 2
    second = router(contexts, contexts)[4].detach()
    # Only the bias diversity contribution changes; no optimization runs.
    torch.testing.assert_close(second - first, parts[5], atol=1e-8, rtol=1e-4)
    second_parts = router.loss_component_sum - parts
    torch.testing.assert_close(second_parts[:5], parts[:5])
    torch.testing.assert_close(second_parts[5], 2 * parts[5])
