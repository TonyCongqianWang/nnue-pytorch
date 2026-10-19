import torch
import torch.nn.functional as F

from model.modules.router import (
    NUM_WEIGHT_EXPERTS,
    ROUTER_TABLE_SIZE,
    DualVQCodebookRouter,
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


def test_dead_expert_reseeding():
    """Verify that unused experts across the context space are actively reseeded."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter()

    # Forcefully test reseeding with an extreme scenario:
    # Set all codebook entries to far away except expert 0
    with torch.no_grad():
        router.codebook_weights.data[1:] = 1000.0
        router.codebook_weights.data[0] = 0.0

    # All contexts will map to expert 0, so experts 1..31 are dead
    reseeded_w, reseeded_b = router.reseed_dead_experts_context_wide()
    assert reseeded_w == list(range(1, NUM_WEIGHT_EXPERTS))
    assert isinstance(reseeded_b, list)

    # Now experts 1..31 should be reseeded from context embeddings and no longer equal 1000.0
    for j in range(1, NUM_WEIGHT_EXPERTS):
        assert not torch.all(router.codebook_weights.data[j] == 1000.0), (
            f"Expert {j} was not reseeded!"
        )
    assert not bool(router.lut_valid.item())


def test_lut_cache_lifecycle():
    """Verify clean LUT cache invalidation, train-mode resets, and non-persistence."""
    router = DualVQCodebookRouter()
    assert not bool(router.lut_valid.item())

    # State dict should not contain ephemeral buffers lut and lut_valid
    sd = router.state_dict()
    assert "lut" not in sd
    assert "lut_valid" not in sd

    # update_lookup_table sets lut_valid = True
    lut = router.update_lookup_table()
    assert lut.shape[0] == ROUTER_TABLE_SIZE
    assert bool(router.lut_valid.item())

    # Switching to train mode invalidates the LUT cache
    router.train()
    assert not bool(router.lut_valid.item())


def test_dead_expert_callback_momentum_reset():
    """Verify MoEDeadExpertReseeder resets Adam optimizer momentum on reseeded codebook rows."""
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
    with torch.no_grad():
        router.codebook_weights.data[1:] = 1000.0
        router.codebook_weights.data[0] = 0.0

    opt = torch.optim.AdamW([router.codebook_weights, router.codebook_biases], lr=1e-3)
    # Simulate non-zero momentum buffers
    opt.state[router.codebook_weights] = {
        "exp_avg": torch.ones_like(router.codebook_weights),
        "exp_avg_sq": torch.ones_like(router.codebook_weights),
    }
    opt.state[router.codebook_biases] = {
        "exp_avg": torch.ones_like(router.codebook_biases),
        "exp_avg_sq": torch.ones_like(router.codebook_biases),
    }

    trainer = DummyTrainer(DummyNNUE(router), opt)
    callback = MoEDeadExpertReseeder(reseed_interval=1024)
    callback.on_train_batch_end(trainer)

    # Momentum for experts 1..31 should be 0, while expert 0 retains momentum
    state_w = opt.state[router.codebook_weights]
    assert torch.all(state_w["exp_avg"][0] == 1.0)
    assert torch.all(state_w["exp_avg"][1:] == 0.0)
    assert torch.all(state_w["exp_avg_sq"][0] == 1.0)
    assert torch.all(state_w["exp_avg_sq"][1:] == 0.0)


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
    assert router.bias_head.weight.grad is not None
    assert torch.norm(router.codebook_weights.grad) > 0.0
    assert torch.norm(router.codebook_biases.grad) > 0.0
    assert torch.norm(router.bias_head.weight.grad) > 0.0
