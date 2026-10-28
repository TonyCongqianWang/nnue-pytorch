# MoE7: independent layer routing

MoE7 starts from upstream SFNNv19/K32H3 and retains its block-major feature
layout, block-local 16-bit engine indices, and accumulator/Finny implementation.
Its H3 rule is strictly perspective-relative: opposing queen selects block 2,
otherwise opposing rook selects block 1, otherwise block 0. Own heavy pieces do
not affect that perspective's PSQ block. The feature hash is `0x32B5E387`.

## Network

Widths remain L1=1024, L2=32 and L3=32. STM and OPP each compute 16 raw outputs
per hidden layer, in fixed global input order. Activated layout is
`[STM_squared16, OPP_squared16, STM_linear16, OPP_linear16]`. Both final readouts
consume the full 128-value FC0/FC1 activation vector and predict STM scores.

Each role independently routes 32 FC0 experts, 32 downstream experts (FC1 and
output together), and 64 dynamic-bias experts. All six codebooks share the
material/structured-king context encoder; each codebook has 256-dimensional
vectors. Weight and dynamic-bias generators retain unit-L2 rank-32 bottlenecks.
The material context remains non-king count, pawn count, and the cascading own
queen / rook / opposite-color bishop pair / other category, with castling-aware
king coordinates: 396 x 33 x 65 = 849,420 contexts.

Hidden weight experts have separate squared and linear static biases. Dynamic
biases add independent offsets for both paths at both hidden layers. Native
trainer projections omit bias; static and dynamic offsets are added at activation.
Each skip is the difference of the last two raw FC0 outputs, before all biases.

Scales are FC0=256, FC1=128 and output=128. Readouts have integer scale 16384;
skips have scale 32768. Reconcile scales and average complete predictions once:

```
merged = trunc((2*(output_stm + output_opp) + skip_stm + skip_opp)/2)
output = trunc(merged * 9600/32768)
```

The engine finally truncates `output/16`. MAC count is unchanged at 35,072.
FC0 and FC1 retain the fused shared-input traversal on AVX2/VNNI; other targets
use the existing single-bank kernels. An on-demand packed-matrix prototype was
bit-exact but failed the performance gate when routes changed, so it is not part
of production inference. There is no active-weight cache or inference copying.

## Routing, export and compatibility

Each uint32 LUT entry packs two 16-bit role words. STM uses bits 0..15 and OPP
bits 16..31. Within each word FC0 is bits 0..4, downstream 5..9 and biases 10..15.
Decode STM fields from the STM context and OPP fields from the OPP context.

After the FT, export stores 32 STM expert triples followed by 32 OPP triples.
A triple's FC0 comes from that bank's FC0 embedding; its FC1/output come from its
downstream embedding. Each hidden layer stores 32 int32 biases before its int8
weights; readouts store one bias. The uint32 LUT follows, then the two 64x64
int32 dynamic-bias tables. Raw downstream weight storage is 1,122,304 bytes,
half of MoE6.5; bias tables occupy 32 KiB and the LUT 3,397,680 bytes.

The router tag is `0x4D6F4537`, combined with FC0/downstream/bias expert counts.
Old networks/checkpoints and upstream's different H3 feature semantics are
incompatible. Start a fresh run.

Export sorts all six banks independently by descending rank-0 validation traffic,
with stable original-ID tie breaking. Small validation histograms persist in
checkpoints. FC0 and downstream components of stored triples can therefore have
different permutations. The exporter remaps every LUT field and moves each
layer's static biases with its weights. Training parameters, IDs and optimizer
state are untouched. Without validation traffic, export preserves identity order;
loaded `.nnue` files consequently re-export without another permutation. Verbose
export reports each new-to-training-ID map; routing profile cards use training IDs.

## Training and diagnostics

Dense weight decay is 1e-6; dynamic-bias generator decay is 1e-3. Static expert
bias adapters have zero decay. Existing commitment, balance, diversity and
softmax temperature settings are retained. Each weight bank contributes 1/4 of
its weighted losses; each bias bank contributes 1/2. This averages roles and the
two weight routers while retaining the previous total weight-loss budget.

All six banks receive EMA usage, validation usage, geometry and sampled LUT-margin
diagnostics; all 18 weighted loss terms are logged. Traffic-only reseeding uses
EMA decay .998, interval 1024 steps, and utilization floor .064/K. LUT occupancy
never independently triggers reseeding. DDP parent maps and optimizer moments
cover all six banks. Diagnostics add no per-batch host synchronization.

The full recipe uses batch 131072 for epochs 0..100 and 100..560, and 65536 only
for epochs 560..600. Its first stage starts fresh. Trace reports FC0/downstream/
bias IDs for both roles, actual evaluation, both original STM contexts, both
original OPP contexts, and exchanged contexts; counterfactuals decode role fields
again rather than transferring IDs between independent banks.

## CPU validation

Run `tests/test_moe7.py`, `tests/test_moe_routing.py`,
`tests/test_routing_contexts.py`, `tests/test_moe_load_balancing.py`, serialization,
FT permutation, compact transport and grouped-linear fallback tests. These checks
need no local training or optimizer steps. CUDA-specific coverage requires a
capable host.

Compile `tests/moe7_engine_probe.cpp` against a matching engine's objects except
`main.o`, using identical compiler flags. Set `MOE7_ENGINE_PROBE` and
`MOE7_TEST_NET` to enable the incremental/fresh/undo/trace integration test.
Use `cross_check_eval.py --random-weights --device cpu` for trainer/export/engine
agreement; validate AVX-VNNI and SSE execution and AVX512 compilation.
