# MoE7: material-only FC0 and perspective tails

MoE7 starts from upstream SFNNv19/K32H3 and retains its block-major feature
layout, block-local 16-bit engine indices, and accumulator/Finny implementation.
Its H3 rule is strictly perspective-relative: opposing queen selects block 2,
otherwise opposing rook selects block 1, otherwise block 0. Own heavy pieces do
not affect that perspective's PSQ block. The feature hash is `0x32B5E387`.

## Network

Widths remain L1=1024, L2=32 and L3=32. One full-width FC0 expert computes all
32 raw first-layer outputs with one conventional sparse traversal. Its two
16-neuron groups feed independent STM/OPP activation biases. FC1 retains two
16-neuron perspective projections. Activated layout at both layers is
`[STM_squared16, OPP_squared16, STM_linear16, OPP_linear16]`. Both final readouts
consume the full 128-value FC0/FC1 activation vector and predict STM scores.

One 64-expert FC0 bank routes on the **ordered pair of STM and OPP material
states only**. Each material state is non-king count, pawn count, and the
cascading own queen / rook / opposite-color bishop pair / other category.
There are 396 valid states per side, hence 156,816 joint contexts. Separate
STM/OPP material embedding tables preserve role ordering. King squares and
castling rights do not enter FC0 routing, even during training. The FT itself
retains its king-dependent features.

Each role separately routes 64 tail experts (FC1/output together) and 64 dynamic
bias experts. These four routers retain the material/structured-king encoder:
396 x 33 x 65 = 849,420 contexts per role, including castling-aware king
coordinates. Codebooks have dimension 256; all weight and bias generators retain
unit-L2 rank-64 bottlenecks.

Hidden weight experts have separate squared and linear static biases. Dynamic
biases add independent offsets for both activation paths at both hidden layers.
The FC0 static biases are stored as `[squared32, linear32]`. Native trainer
projections omit bias. The skip remains the average of the last-two-output
difference in each 16-neuron FC0 group, before all static and dynamic biases.

Scales remain FC0=256, FC1=128 and output=128. Readouts have integer scale 16384;
skips have scale 32768. Reconcile scales and average complete predictions once:

```
merged = trunc((2*(output_stm + output_opp) + skip_stm + skip_opp)/2)
output = trunc(merged * 9600/32768)
```

The engine finally truncates `output/16`. MAC count remains 35,072. FC0 now uses
the ordinary full-width sparse kernel on every target. FC1 retains its fused
pair traversal on AVX2/VNNI and single-bank fallbacks elsewhere. There is no
active-weight cache, inference copying, hysteresis or history-dependent routing.

## Routing, export and compatibility

The perspective table has 849,420 uint32 entries. STM uses bits 0..15 and OPP
bits 16..31. Within each role word tail is bits 0..5, biases 6..11, and bits
12..15 are reserved and exported as zero. Decode the STM fields from the STM
context and OPP fields from the OPP context.

A separate byte-aligned FC0 table has 156,816 uint8 entries (153.1 KiB):

```
index = material_stm * 396 + material_opp
fc0_expert = fc0_table[index]  # 0..63
```

After the FT, export stores 64 full-width FC0 experts, followed by 64 STM tails
and 64 OPP tails. Each FC0 record has its own input-layer hash and its static
biases/weights. Each tail record has the complete downstream hash and stores
FC1 followed by output parameters. The engine likewise keeps distinct FC0 and
tail arrays, so increasing tail counts never duplicates FC0 matrices. Each hidden
layer stores squared/linear int32 biases before int8 weights; readouts store one
bias. The uint32 perspective LUT follows, then the byte FC0 LUT, then the two
64x64 int32 dynamic-bias tables.
Raw downstream weight storage is 2,244,608 bytes; the larger FC0 expert bank
is intentional. The bias tables occupy 32 KiB. The architecture hash includes
full-width FC0 dimensions and the 64/64/64 expert counts, rejecting the earlier
MoE7 format. The name and router tag remain MoE7 (`0x4D6F4537`); start a fresh run.

All five banks sort independently by descending rank-0 validation traffic, with
stable original-ID tie breaking. Validation histograms persist in checkpoints.
Export remaps both LUTs, moves each layer's static biases with its weights, and
preserves training IDs and optimizer state. Without validation traffic, identity
order is retained; loaded `.nnue` files therefore re-export without another
permutation. Profile cards distinguish FC0 material pairs from perspective
contexts.

## Training and diagnostics

Dense weight decay remains 1e-6; the recipe also applies 1e-6 to virtual FT
weights. Dynamic-bias generator decay is 1e-3. Static
expert bias adapters have zero decay. Commitment, balance, diversity and softmax
temperature settings are unchanged. FC0 contributes 1/2 of the weight-router
loss budget, each tail 1/4, and each bias router 1/2 of the bias-router budget.

All five banks receive EMA usage, validation usage, geometry and sampled
LUT-margin diagnostics. All 15 weighted loss terms are logged. Margin sample
counts are bank-specific because the tables differ in size. Traffic-only
reseeding uses EMA decay .998, interval 1024 steps, and floor .064/K. LUT occupancy
never independently triggers reseeding. DDP parent maps and optimizer moments
cover all five banks. Diagnostics add no per-batch host synchronization.

The full recipe uses batch 131072 for epochs 0..100 and 100..560, and 65536 only
for epochs 560..600. Its first stage starts fresh. Trace reports one FC0 ID,
tail/bias IDs for both roles, actual evaluation, repeated STM/OPP contexts and
exchanged contexts. Counterfactuals recompute the material pair as well as role
fields rather than transferring IDs between independent banks.

## CPU validation

Run `tests/test_moe7.py`, `tests/test_moe_routing.py`,
`tests/test_routing_contexts.py`, `tests/test_moe_load_balancing.py`, serialization,
FT permutation, compact transport and grouped-linear fallback tests. These checks
need no local training or optimizer steps. CUDA-specific coverage requires a
capable host.

Compile `tests/moe7_engine_probe.cpp` against matching engine objects except
`main.o`, using identical compiler flags. Set `MOE7_ENGINE_PROBE` and
`MOE7_TEST_NET` to enable incremental/fresh/undo/trace integration tests.
Use `cross_check_eval.py --random-weights --device cpu` for trainer/export/engine
agreement; validate AVX-VNNI and SSE execution and AVX512 compilation.
