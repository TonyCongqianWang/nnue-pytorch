import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import VQCodebookRouter


def test_balance_loss_bounds():
    """Verify load balancing loss is 1.0 for uniform distribution and K for collapsed distribution."""
    num_experts = 32

    # 1. Uniform case:
    # 32 batch items, each routed to a distinct expert with uniform soft probability
    distances = torch.zeros(32, num_experts)
    for i in range(num_experts):
        distances[i, i] = -10.0  # expert i is closest
    # Softmax probabilities will be concentrated on i, so dispatch fraction f_i = 1/32
    # and soft prob P_i = 1/32
    # Load balancing loss = K * sum_j(f_j * P_j) = 32 * 32 * (1/32) * (1/32) = 1.0
    probs = F.softmax(-distances, dim=-1)
    indices = torch.arange(num_experts)
    dispatch_mask = F.one_hot(indices, num_classes=num_experts).float()
    f = dispatch_mask.mean(dim=0)
    P = probs.mean(dim=0)
    balance_loss = num_experts * torch.sum(f * P)
    assert torch.isclose(balance_loss, torch.tensor(1.0), atol=1e-3)

    # 2. Collapsed case: all items routed to expert 0
    collapsed_indices = torch.zeros(32, dtype=torch.long)
    collapsed_mask = F.one_hot(collapsed_indices, num_classes=num_experts).float()
    f_collapsed = collapsed_mask.mean(dim=0)
    P_collapsed = collapsed_mask.mean(dim=0)  # 100% prob on 0
    balance_loss_collapsed = num_experts * torch.sum(f_collapsed * P_collapsed)
    assert torch.isclose(
        balance_loss_collapsed, torch.tensor(float(num_experts)), atol=1e-3
    )


def test_dead_expert_reseeding():
    """Verify that unused experts in a batch are actively reseeded."""
    torch.manual_seed(42)
    router = VQCodebookRouter(num_experts=32, dim=32, reseed_dead_experts=True)

    # Fake input context vectors z: shape (64, 32)
    z = torch.randn(64, 32)

    # Pass through quantize with reseed_dead_experts enabled
    router.train()
    _ = router.quantize(z)

    # Forcefully test reseeding with an extreme scenario:
    # Set all codebook entries to far away except expert 0
    with torch.no_grad():
        router.codebook.data[1:] = 1000.0
        router.codebook.data[0] = z.mean(dim=0)

    # All items will choose expert 0, meaning experts 1..31 are dead
    router.train()
    _ = router.quantize(z)

    # Now experts 1..31 should be reseeded from z contexts and no longer equal 1000.0
    for j in range(1, 32):
        assert not torch.all(router.codebook.data[j] == 1000.0), (
            f"Expert {j} was not reseeded!"
        )


def test_router_gradient_flow():
    """Verify that backpropagation produces non-zero gradients for router codebook and embeddings."""
    torch.manual_seed(42)
    router = VQCodebookRouter(num_experts=32, dim=32, balance_cost=0.02)
    router.train()

    c_us = torch.randint(0, 1000, (16,))
    c_them = torch.randint(0, 1000, (16,))

    e1, e2, moe_loss, *_ = router(c_us, c_them)
    total_loss = moe_loss + (e1.sum() + e2.sum()) * 0.01
    total_loss.backward()

    assert router.codebook.grad is not None
    assert torch.norm(router.codebook.grad) > 0.0
    assert router.piece_embed.weight.grad is not None
    assert torch.norm(router.piece_embed.weight.grad) > 0.0


def test_no_zero_weights_trap():
    """Verify that zero_virtual_weights() on NNUEModel does NOT zero LowRankStackedLinear or router codebook."""
    config = ModelConfig()
    model = NNUEModel(DEFAULT_FEATURES, config)

    # Verify initial codebook, down_proj, and up_proj are non-zero
    assert torch.norm(model.router.codebook) > 0.0
    assert torch.norm(model.layer_stacks.l1.down_proj) > 0.0
    assert torch.norm(model.layer_stacks.l1.up_proj) > 0.0
    assert torch.norm(model.layer_stacks.l1.bias_adapter) > 0.0

    # Call zero_virtual_weights()
    model.zero_virtual_weights()

    # Codebook, down_proj, up_proj, and bias_adapter should still be non-zero!
    assert torch.norm(model.router.codebook) > 0.0
    assert torch.norm(model.layer_stacks.l1.down_proj) > 0.0
    assert torch.norm(model.layer_stacks.l1.up_proj) > 0.0
    assert torch.norm(model.layer_stacks.l1.bias_adapter) > 0.0
