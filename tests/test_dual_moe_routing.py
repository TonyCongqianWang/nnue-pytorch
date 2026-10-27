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


def test_four_byte_routing_and_bias_tables():
    r = DualVQCodebookRouter(num_contexts=2)
    entries = torch.tensor(
        [
            63 | (62 << 8) | (127 << 16) | (126 << 24),
            1 | (2 << 8) | (3 << 16) | (4 << 24),
        ],
        dtype=torch.int64,
    )
    r.load_lookup_table(entries.to(torch.uint32).numpy().tobytes())
    d = torch.arange(2 * 128 * 64, dtype=torch.float32).reshape(2, 128, 64) / 32768
    r.load_coalesced_biases(d)
    r.eval()
    ew1, ew2, d1, d2, *_ = r(torch.tensor([0]), torch.tensor([1]))
    assert ew1.item() == 63 and ew2.item() == 2
    torch.testing.assert_close(torch.cat(d1, 1), d[0, 127:128])
    torch.testing.assert_close(torch.cat(d2, 1), d[1, 4:5])
    assert torch.equal(r.get_lookup_table(), entries)


def test_zero_bias_init_and_independent_codebooks():
    r = DualVQCodebookRouter(num_contexts=4)
    assert r.coalesced_biases.shape == (2, 128, 64)
    assert torch.count_nonzero(r.get_coalesced_bias_table()) == 0
    assert len(list(r.banks())) == 4
    assert r.codebook[0].data_ptr() != r.codebook[1].data_ptr()


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
    for old, new in zip(
        m.layer_stacks.get_coalesced_layer_stacks(m.router.codebook),
        loaded.layer_stacks.get_coalesced_layer_stacks(),
    ):
        for a, b, key in zip(old, new, ("ls_l1", "ls_l2", "ls_output")):
            ba, wa = m.quantization.quantize_fc_layer(a.bias, a.weight, key)
            bb, wb = loaded.quantization.quantize_fc_layer(b.bias, b.weight, key)
            assert torch.equal(ba, bb) and torch.equal(wa, wb)


def test_unit_l2_basis_initialization_preserves_layer_variance():
    """Generated weights should start at an ordinary layer's variance."""
    from model.modules.moe_linear import MoeLinear

    torch.manual_seed(7)
    router = DualVQCodebookRouter(num_weight_experts=4, num_bias_experts=4)
    for rank in (8, 32):
        layer = MoeLinear(256, 64, 4, rank=rank)
        weights = layer.get_delta_weights(router.codebook_w)
        expected_rms = (1.0 / (3 * layer.in_features)) ** 0.5
        assert abs(weights.square().mean().sqrt().item() / expected_rms - 1) < 0.1
