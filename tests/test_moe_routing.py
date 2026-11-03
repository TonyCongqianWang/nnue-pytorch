import io

import pytest
import torch

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import MoERouter
from model.utils.serialize import (
    CountingExporter,
    NNUEReader,
    NNUEWriter,
    pack_lookup_table,
    unpack_lookup_table,
)


@pytest.mark.parametrize("bits", [6, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 65535, 65536, 65537])
def test_packed_ids_across_chunk_and_byte_boundaries(bits, count):
    x = torch.arange(count) % (1 << bits)
    data = pack_lookup_table(x, bits)
    assert len(data) == (count * bits + 7) // 8
    assert torch.equal(unpack_lookup_table(data, count, bits), x)
    if count:
        with pytest.raises(EOFError):
            unpack_lookup_table(data[:-1], count, bits)
    if count == 1 and bits % 8:
        corrupt = bytearray(data)
        corrupt[-1] |= 128
        with pytest.raises(ValueError):
            unpack_lookup_table(corrupt, count, bits)


def test_router_lut_matches_batch_selection_and_diagnostics():
    torch.manual_seed(8)
    r = MoERouter(
        material_states=4, num_fc0_experts=3, num_weight_experts=5, fc0_dim=8, dim=16
    ).eval()
    r.collect_lut_diagnostics = True
    fc0, tail = r.update_lookup_table(chunk_size=13)
    for name, table in [("fc0", fc0), ("tail", tail)]:
        z = r.get_context_vector(torch.arange(len(table)), name)
        cb = r.normalized_codebook(name)
        dist = (
            z.square().sum(-1, keepdim=True) + cb.square().sum(-1)[None] - 2 * z @ cb.T
        )
        assert torch.equal(table, dist.argmin(-1))
    f, t, *_ = r(torch.tensor([0, 15]), torch.tensor([0, 127]))
    assert torch.equal(f, fc0[[0, 15]]) and torch.equal(t, tail[[0, 127]])
    assert r.validation_counts_fc0.sum() == r.validation_counts_tail.sum() == 2
    assert len(r.codebooks) == 2 and r.lut_margin_stats.shape == (2, 3)


def make_model():
    torch.manual_seed(8)
    m = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=4, L2=4, L3=4))
    m.layer_stacks.coalesce_layer_stacks_inplace(m.router.codebook)
    m.router.fc0_lut.copy_(torch.arange(len(m.router.fc0_lut)) % 64)
    m.router.lut.copy_(torch.arange(len(m.router.lut)) % 256)
    m.router.lut_valid = True
    return m.eval()


def test_sorted_export_roundtrip_counting_and_packed_tables():
    m = make_model()
    state = {k: v.clone() for k, v in m.named_parameters()}
    m.router.validation_counts_fc0.copy_(torch.arange(64) % 7)
    m.router.validation_counts_tail.copy_(torch.arange(256) % 11)
    writer = NNUEWriter(m, description="MoE8", verbose=False)
    loaded = NNUEReader(
        io.BytesIO(writer.buf), DEFAULT_FEATURES, ModelConfig(L1=4, L2=4, L3=4)
    ).model.eval()
    assert bytes(NNUEWriter(loaded, description="MoE8", verbose=False).buf) == bytes(
        writer.buf
    )
    counter = CountingExporter()
    m.router.get_lookup_table = lambda: (_ for _ in ()).throw(
        AssertionError("No LUT generation for size counting")
    )
    NNUEWriter(m, description="MoE8", exporter=counter, verbose=False)
    assert counter.size == len(writer.buf)
    assert all(torch.equal(v, state[k]) for k, v in m.named_parameters())
    c0 = torch.arange(97) * 3
    c1 = c0 * 8 + torch.arange(97) % 8
    x = torch.rand(97, 4)
    f, t, *_ = loaded.router(c0, c1)
    actual = loaded.layer_stacks(x, None, f, t)
    m.router.validation_counts_fc0.zero_()
    m.router.validation_counts_tail.zero_()
    del m.router.get_lookup_table
    plain = NNUEReader(
        io.BytesIO(NNUEWriter(m, description="MoE8", verbose=False).buf),
        DEFAULT_FEATURES,
        ModelConfig(L1=4, L2=4, L3=4),
    ).model.eval()
    f, t, *_ = plain.router(c0, c1)
    torch.testing.assert_close(
        actual, plain.layer_stacks(x, None, f, t), rtol=0, atol=0
    )
    corrupt = bytearray(writer.buf)
    corrupt[4] ^= 1
    with pytest.raises(ValueError):
        NNUEReader(io.BytesIO(corrupt), DEFAULT_FEATURES, ModelConfig(L1=4, L2=4, L3=4))


def test_aggregate_histograms_are_default_and_count_all_experts(capsys):
    m = make_model()
    NNUEWriter(m, description="histograms")
    text = capsys.readouterr().out
    for layer in ["ls_l1", "ls_l2", "ls_output"]:
        assert text.count("all experts " + layer + "_weight:") == 1
        assert text.count("all experts " + layer + "_bias:") == 1
    assert "expert 0 ls_" not in text
    assert "FC0: 64 experts" in text and "TAIL: 256 experts" in text
    assert "Wide % u/o" in text and "+---" in text
    NNUEWriter(m, histogram_mode="none")
    assert capsys.readouterr().out == ""


def test_basis_initialization_preserves_variance():
    from model.modules.moe_linear import MoeLinear

    for rank in [64, 512]:
        torch.manual_seed(8)
        layer = MoeLinear(128, 32, 8, rank=rank, codebook_dim=64)
        cb = torch.nn.functional.normalize(torch.randn(8, 64), dim=1)
        weights = layer.get_delta_weights(cb)
        assert (
            abs(weights.square().mean().sqrt().item() / (1 / (3 * 128)) ** 0.5 - 1)
            < 0.1
        )
