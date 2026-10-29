# Handcrafted perspective experts

This is a fresh experiment directly on upstream SFNNv19, separate from MoE7.
No local training, optimizer steps, remote training or remote testing were run.
Stockfish engine: `2dca964965dbfc5b654f57377853b4282413c9c9`. Strength is unmeasured. The reference [exact comparison](https://github.com/xu-shawn/Stockfish/compare/a898a3954ba364cc8cb068143dc3a93c829e7407...xu-shawn:Stockfish:ea82de7004875af69c14a8365bb9e0d2c9b5c8e0)
changes selection, 32 to 192 full-width stacks, and the default net. RamRam was
already in its parent. Its diff cannot isolate routing strength or net strength.
RamRam is not included here.

## Routing format

Each role selects from its perspective's **own** material:

* T = non-king count; band = T // 2, eight bands 0–1 through 14–15.
* Heavy = Q if any own queen; else R if any own rook; else neither.
* Officers = own N+B+R+Q; flag is officers >= 2.
* Own pawns occupy both a–d/e–h file halves, exactly one half, or neither.

No global or opposing feature enters downstream selection. STM and OPP use
independent parameter banks, with the same immutable nominal-to-compact mapping.
Nominal ID is `(((band*3+heavy)*2+officer_flag)*3+layout)`, with heavy Q/R/neither
numbered 0/1/2 and layouts both/one/none numbered 0/1/2. Live nominal IDs in
ascending order get compact IDs 0..84; invalid nominal entries contain 255.

Promotion-aware enumeration of exact P,N,B,R,Q counts independently matches
coarse enumeration: **85 live tuples**, counts **[4,12,15,15,15,9,9,6]** by band.
The minimum required promotions are
`max(N-2,0)+max(B-2,0)+max(R-2,0)+max(Q-1,0) <= 8-P`, with P<=8 and T<=15.
These are material-support invariants, not a complete historical reachability
validator. Promoted and underpromoted material is retained. No feasible tuple is
removed because it is absent from a sample or impossible in a particular role pair.

## Architecture and quantization

L1=1024, L2=32, L3=32. Each role has FC0 1024→16 and sees the full fixed STM-first
input. FC0 activations are `[STM squared16, OPP squared16, STM linear16, OPP linear16]`.
Each FC1 consumes all 64 activations and produces 16 neurons, with the same
activation ordering. Each scalar readout consumes the full 128-value FC0/FC1
concatenation. The same deterministic role ID selects all three affine layers.

Training uses upstream additive shared-affine FC0 factorization independently
within each role; export coalesces it. FC1 and readouts are ordinary independent
affines. Hidden neurons have one static bias shared between their activation
paths. Each role's skip is the difference of its last two **pre-bias** FC0
projections; those neurons also remain in the activated paths.

Both readouts predict STM scores. Average complete predictions once, with no
antisymmetry constraint. Weight scales are upstream FC0=128, FC1=64, output=128;
hidden activation scale is 128. Skips and readouts both have integer scale 16384:

```
merged = trunc((out_stm + out_opp + skip_stm + skip_opp) / 2)
output16 = trunc(merged * 9600 / 16384)
engine_score = trunc(output16 / 16)
```

MoE7's extra factor of two reconciled different weight scales; it is not another
head average. Its doubled hidden scales, dual biases, codebooks, generators,
router losses and reseeding are absent.

The FT baseline remains Full_Threats + PP_3Wide + **strict opposing-heavy K32H3**:
opposing queen selects block 2, else opposing rook block 1, else block 0.
It retains upstream block-major storage and block-local engine indices. This
opposing-heavy input rule is independent of own-heavy downstream selection.

## Exposure and fallback

Every role stores 85 feasible experts and fallback index 85. The fallback is an
equal parameter average over that role's experts with positive training exposure,
after coalescing and before quantization. It has no independently trained parameters.
This is an averaged-parameter network, not an ensemble of 85 predictions.

A persistent 2×85 boolean training-exposure buffer is updated only on training
forwards with gradients enabled. DDP unions it across ranks before another buffer
broadcast can discard local observations. Validation never marks exposure.
Checkpoint validation derives the same fallback and mask as export using fixed-shape
masked reductions on the model's device. It creates no CPU export snapshots and
reads no device scalars on the host. The persisted export marker has a Python
cache restored at load time, avoiding scalar synchronization on mode changes.
File export uses the same reduction before copying parameters to CPU; regular
expert snapshots also explicitly preserve CPU placement and source dtype.
A role with no exposure cannot be exported without explicit synthetic mode. Positive exposure,
however small, retains the expert; absence from validation does not disable it.

Export appends two 11-byte little-bit-order masks. An unset role bit redirects
that role to fallback; IDs and all 85 parameter slots remain stable. Unsupported
exact material also selects fallback, even if its coarse nominal tuple is live.
The 255 sentinel is checked before any expert or exposure-mask access. Upstream
FEN admission rules are preserved: material rejected by `Position::set` produces
its existing UCI error/exit before NNUE. Fallback does not broaden FEN admission.
Trace
reports both compact and effective expert IDs and distinguishes fallback reasons.

## Compatibility and workflow

Architecture/checkpoint/loader tag is `0x48503835`; strict K32H3 feature hash is
`0x32B5E387`. The FC hash combines the new tag with layer hashes. Upstream and
MoE7 nets/checkpoints/loaders are incompatible and rejected. Stored ordering is
STM triples 0..85, then OPP triples 0..85, then 22 mask bytes. Readers preserve
stored fallback parameters and masks on byte-stable re-export. There is no
traffic-based expert permutation or learned routing table.

Start training fresh. `.nnue` files are inference artifacts; full continuation
requires a matching checkpoint with exposure history. Fresh synthetic exports
require `NNUEWriter(..., synthetic=True)` or serializer `--synthetic`.
Use an explicit compatible EvalFile; the upstream bundled default net is incompatible.
Do not profile this architecture using that default net.

The full recipe retains epochs 0..100, 100..560 and 560..600, batch 131072 for
the first two stages and 65536 only for the final stage. Dataset, loss and long
schedules are retained. Ordinary affine factorization uses the upstream recipe's
factorized weight decay 4e-5; no MoE-specific optimizer or routing options remain.
No recipe was submitted. A match against an unrelated existing net measures the
whole trained system; it cannot establish routing strength. Any controlled routing
ablation must keep this exact FT rule, data and training schedule fixed.

## CPU validation and measurements

See [results](handcrafted-perspective/results.md), [raw timings](handcrafted-perspective/inference-measurements.json)
and [traffic histograms](handcrafted-perspective/traffic-measurements.json).
Reproduce without training:

```
pytest tests/test_handcrafted_perspective.py tests/test_compact_transport.py tests/test_grouped_linear.py
python tests/prepare_handcrafted_nets.py /tmp/hp-nets
python tests/build_handcrafted_probe.py ENGINE/src BUILD.log /tmp/hp-probe
python tests/validate_handcrafted_engine.py /tmp/hp-probe /tmp/hp-nets/random.nnue --masked-net /tmp/hp-nets/masked.nnue
python cross_check_eval.py --net /tmp/hp-nets/random.nnue --data .pgo/small.binpack --engine ENGINE/src/stockfish --device cpu --max-error 1
```

The distributed exposure test needs local loopback sockets. CUDA coverage needs
another host; no CUDA or local training budget was assumed. The profiling and
benchmark harnesses live in the trainer, not the engine repository.
