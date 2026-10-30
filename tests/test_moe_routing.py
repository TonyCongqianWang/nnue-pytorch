import io

import pytest
import torch

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import *
from model.utils.serialize import NNUEReader, NNUEWriter


def test_material_bijection_exhaustive():
    """Verify that all 396 valid legal material states map bijectively to [0, 395]."""
    seen_ids = set()
    for m in range(NUM_MATERIAL_STATES):
        t, p, h = decode_material_state(m)
        assert 0 <= t <= 15
        assert 0 <= p <= min(8, t)
        assert 0 <= h < 4
        assert t - p >= (1 if h < 2 else 2 if h == 2 else 0)

        encoded_id = encode_material_state(t, p, h)
        assert encoded_id == m
        seen_ids.add(encoded_id)

    assert len(seen_ids) == 396

    # Verify illegal states raise ValueError
    with pytest.raises(ValueError):
        encode_material_state(0, 0, 0)  # Queen requires a non-pawn
    with pytest.raises(ValueError):
        encode_material_state(1, 0, 2)  # Bishop pair requires two non-pawns


def test_structured_king_geometry():
    """Verify StructuredKingEmbedding dimensions, lookups, and Chebyshev opposition."""
    emb = StructuredKingEmbedding(dim=256)

    # Test our king bucket 32 (e1 with castling)
    assert emb.our_rank[32].item() == 0
    assert emb.our_edge[32].item() == 3
    assert emb.our_castling[32].item() == 1

    # Test enemy king bucket 64 (e8 with castling)
    assert emb.enemy_rank[64].item() == 0  # distance to enemy home rank (rank 8)
    assert emb.enemy_edge[64].item() == 3
    assert emb.enemy_castling[64].item() == 1

    # Test delta_r table bounds
    assert emb.delta_r.min().item() == 0
    assert emb.delta_r.max().item() == 7
    assert emb.delta_r.shape == (33, 65)

    # Test delta_f table bounds
    assert emb.delta_f.min().item() == 0
    assert emb.delta_f.max().item() <= 7
    assert emb.delta_f.shape == (33, 65)

    # Test Chebyshev table bounds (mapped 2..7 -> 0..5)
    assert emb.chebyshev.min().item() == 0
    assert emb.chebyshev.max().item() == 5
    assert emb.chebyshev.shape == (33, 65)

    # Test forward
    kus = torch.tensor([0, 15, 32])
    kth = torch.tensor([0, 31, 64])
    z_k = emb(kus, kth)
    assert z_k.shape == (3, 256)


def test_material_fc0_and_four_perspective_fields():
    r = MoERouter(num_contexts=2)
    entries = torch.tensor(
        [
            63 | (63 << 6) | (59 << 16) | (62 << 22),
            2 | (3 << 6) | (61 << 16) | (6 << 22),
        ],
        dtype=torch.int64,
    )
    fc0 = torch.full((FC0_ROUTER_TABLE_SIZE,), 63, dtype=torch.uint8)
    r.load_lookup_table(
        entries.to(torch.uint32).numpy().tobytes(), fc0.numpy().tobytes()
    )
    d = torch.arange(2 * 64 * 64, dtype=torch.float32).reshape(2, 64, 64) / 32768
    r.load_coalesced_biases(d)
    r.eval()
    ew1, ew2, d1, d2, *_ = r(torch.tensor([0]), torch.tensor([1]))
    assert [x.item() for x in ew1] == [63, 63]
    assert [x.item() for x in ew2] == [63, 61]
    torch.testing.assert_close(torch.cat(d1, 1), d[0, 63:64])
    torch.testing.assert_close(torch.cat(d2, 1), d[1, 6:7])
    assert torch.equal(r.get_lookup_table(), entries)
    assert torch.equal(r.fc0_lut, fc0)


def test_fc0_material_pair_ignores_kings_and_preserves_role_order():
    r = MoERouter(dim=16)
    us = torch.tensor([make_router_index(100, k, (k * 2) % 65) for k in range(33)])
    them = torch.tensor(
        [make_router_index(200, 32 - k, (k * 3) % 65) for k in range(33)]
    )
    z = r.get_fc0_context_vector(us, them)
    torch.testing.assert_close(z, z[:1].expand_as(z), rtol=0, atol=0)
    assert not torch.allclose(z, r.get_fc0_context_vector(them, us))
    # Joint-material enumeration and the batched encoder must agree.
    all_z = r.get_all_fc0_context_vectors()
    torch.testing.assert_close(
        z[0], all_z[100 * NUM_MATERIAL_STATES + 200], rtol=0, atol=0
    )
    r.eval()
    r.update_lookup_table()
    a, b, *_ = r(us, them)
    assert torch.equal(a[0], b[0]) and a[0].unique().numel() == 1


def test_zero_bias_init_and_independent_codebooks():
    r = MoERouter(num_contexts=4)
    assert r.coalesced_biases.shape == (2, 64, 64)
    assert r.bias_rank == 64
    assert all(
        head.down.out_features == head.up.in_features == 64 for head in r.bias_heads
    )
    assert torch.count_nonzero(r.get_coalesced_bias_table()) == 0
    assert len(list(r.banks())) == 5
    assert len({p.data_ptr() for p in r.codebooks.values()}) == 5


def test_serialization_and_deserialization_roundtrip():
    m = NNUEModel(DEFAULT_FEATURES, ModelConfig())
    init_random_weights(m)
    writer = NNUEWriter(m, verbose=False)
    loaded = NNUEReader(io.BytesIO(writer.buf), DEFAULT_FEATURES, ModelConfig()).model
    torch.testing.assert_close(
        m.router.get_lookup_table(), loaded.router.get_lookup_table(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        m.router.get_coalesced_bias_table(),
        loaded.router.get_coalesced_bias_table(),
        rtol=0,
        atol=0,
    )
    assert torch.equal(m.router.fc0_lut, loaded.router.fc0_lut)
    for old, new in zip(
        m.layer_stacks.get_coalesced_layer_stacks(m.router.codebook),
        loaded.layer_stacks.get_coalesced_layer_stacks(),
    ):
        for a, b, key in zip(
            old, new, ("ls_l1",) if len(old) == 1 else ("ls_l2", "ls_output")
        ):
            ba, wa = m.quantization.quantize_fc_layer(a.bias, a.weight, key)
            bb, wb = loaded.quantization.quantize_fc_layer(b.bias, b.weight, key)
            assert torch.equal(ba, bb) and torch.equal(wa, wb)


def test_unit_l2_basis_initialization_preserves_layer_variance():
    """Generated weights should start at an ordinary layer's variance."""
    from model.modules.moe_linear import MoeLinear

    torch.manual_seed(7)
    router = MoERouter(num_weight_experts=4, num_bias_experts=4)
    for rank in (8, 32, 64):
        layer = MoeLinear(256, 64, 4, rank=rank)
        weights = layer.get_delta_weights(router.codebook[0][0])
        expected_rms = (1.0 / (3 * layer.in_features)) ** 0.5
        assert abs(weights.square().mean().sqrt().item() / expected_rms - 1) < 0.1


def test_export_traffic_sort_preserves_scores_and_training_ids():
    """Independently reorder all five banks and verify the remapped quantized model."""
    torch.manual_seed(811)
    config = ModelConfig(L1=4, L2=4, L3=4)
    m = NNUEModel(DEFAULT_FEATURES, config)
    m.layer_stacks.coalesce_layer_stacks_inplace(m.router.codebook)
    m.router.lut.random_(0, 1 << 32)
    m.router.lut.bitwise_and_(0x0FFF0FFF)
    m.router.fc0_lut.random_(0, 64)
    m.router.lut_valid = True
    m.router.load_coalesced_biases(torch.randn(2, 64, 8) * 0.05)
    m.eval()
    original_ids = m.router.lut.clone()
    original_fc0_ids = m.router.fc0_lut.clone()
    for spec in BANK_SPECS:
        counts = getattr(m.router, f"validation_counts_{spec.name}")
        # Different nontrivial orders for each bank, with ties.
        counts.copy_((torch.arange(len(counts)) * (spec.shift + 3)) % 11)
    state = {name: p.clone() for name, p in m.named_parameters()}
    orders = m.router.export_permutations()
    for spec in BANK_SPECS:
        counts = getattr(m.router, f"validation_counts_{spec.name}")
        assert orders[spec.name].tolist() == sorted(
            range(len(counts)), key=lambda i: (-counts[i].item(), i)
        )
    writer = NNUEWriter(m, description="traffic", verbose=False)
    corrupt = bytearray(writer.buf)
    corrupt[4] ^= 1
    with pytest.raises(ValueError, match="Expected"):
        NNUEReader(io.BytesIO(corrupt), DEFAULT_FEATURES, config)
    loaded = NNUEReader(io.BytesIO(writer.buf), DEFAULT_FEATURES, config).model.eval()
    assert torch.equal(original_ids, m.router.lut)
    assert torch.equal(original_fc0_ids, m.router.fc0_lut)
    assert all(torch.equal(state[name], p) for name, p in m.named_parameters())
    # Export to the same quantized format with identity order for the reference.
    for spec in BANK_SPECS:
        getattr(m.router, f"validation_counts_{spec.name}").zero_()
    plain = NNUEReader(
        io.BytesIO(NNUEWriter(m, description="traffic", verbose=False).buf),
        DEFAULT_FEATURES,
        config,
    ).model.eval()
    x = torch.rand(97, 4)
    u, t = torch.arange(97) * 2145, torch.arange(97).flip(0) * 2145

    def evaluate(model):
        a, b, da, db, *_ = model.router(u, t)
        return model.layer_stacks(x, None, a, b, da, db)

    torch.testing.assert_close(evaluate(loaded), evaluate(plain), rtol=0, atol=0)
    assert bytes(NNUEWriter(loaded, description="traffic", verbose=False).buf) == bytes(
        writer.buf
    )


def test_validation_traffic_survives_checkpoint_state_and_zero_fallback():
    r = MoERouter(num_contexts=2)
    for spec in BANK_SPECS:
        assert torch.equal(
            r.export_permutations()[spec.name],
            torch.arange(len(r.codebooks[spec.name])),
        )
        getattr(r, f"validation_counts_{spec.name}")[1] = 9
    restored = MoERouter(num_contexts=2)
    restored.load_state_dict(r.state_dict())
    assert all(restored.export_permutations()[spec.name][0] == 1 for spec in BANK_SPECS)
