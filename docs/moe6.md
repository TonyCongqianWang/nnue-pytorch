# MoE6: independent STM/OPP routing

MoE6 retains K32H3 and the existing 849,420 material/king contexts. It replaces
MoE5's shared, perspective-swapped dense bank with independent STM and OPP banks.
Both banks predict in STM score coordinates; no odd-function symmetry is imposed.

## Network and inference

Total widths are L1=1024, L2=32, L3=32. Each role has 64 weight experts and
16 neurons in each hidden layer. Each expert is generated from a 256-dimensional
codebook through the existing unit-L2 rank-32 hypernetwork. The corrected dense
initialization, with expansion bound `1/sqrt(in_features)`, is retained.

Both FC0 branches consume the same `[STM FT, OPP FT]` input and sparse metadata.
Hidden activation layout is `[STM_squared, OPP_squared, STM_linear, OPP_linear]`.
Both FC1 branches consume the full 64-element FC0 activation vector. Both final
readouts consume all 128 activated values from FC0 and FC1. This costs 35,072
MACs, as in MoE5; master costs 34,944. There are no perspective swaps.

Each role has 128 bias experts producing 64 threshold values: FC0 squared16,
FC0 linear16, FC1 squared16, FC1 linear16. Bias generators have independent down
and up projections; up weights and offsets start at zero. Material and king
context embeddings remain shared. Dynamic-bias generator weight decay defaults
to 4e-4; the experiment uses 4e-5 for dense weights.

Weight scales are FC0=256, FC1=64, output=128. Bias scales are 32768, 8192,
and 16384, respectively. Dynamic thresholds use their corresponding hidden-layer
bias scales. FC0 activations shift by eight bits; FC1 by six bits.

For output accumulators `o_stm,o_opp` at scale 16384 and skip accumulators
`s_stm,s_opp` at scale 32768, merge once using wide integers:

```
merged = trunc((2*(o_stm + o_opp) + s_stm + s_opp)/2)
output = trunc(merged * (600*16)/32768)
```

Stockfish then truncates `output/16` to its internal evaluation unit. Skips retain
the existing last-two-FC0-neuron difference independently for each role. Their
signs are learned in STM coordinates. Trainer STE rounding follows this integer
path. Scale reconciliation and averaging are separate operations.

## Routing and serialization

Four independent codebooks route weights/biases for each role. Each uint32 LUT
entry is little-endian with byte0=STM weights, byte1=OPP weights, byte2=STM biases,
byte3=OPP biases. Weight IDs range 0..63; bias IDs 0..127. Unused byte bits are zero.
STM fields are read from the STM context, OPP fields from the opponent context.

The router tag is `0x4D6F4536` (MoE6), combined with per-role expert counts. Previous
nets and checkpoints are incompatible; start a fresh run. After the unchanged FT,
the file stores 64 STM expert triples, 64 OPP triples, 849,420 uint32 LUT entries,
and two 128x64 int32 bias tables in STM/OPP order. Expert weights remain coalesced
on disk. Bias tables occupy 64 KiB combined; the LUT occupies 3,397,680 bytes.

Role bank storage is explicit in the trainer: the existing `layer_stacks.l1/l2/output`
are STM; `layer_stacks.opp` contains OPP layers. `router.codebook` supplies the pair
of normalized weight codebooks. Coalescing, export/import and FT permutation cover
both banks.

## Losses, reseeding and diagnostics

Each bank has its own hard/soft usage EMA. Existing coefficients and temperature
are retained: commitment .005, weight balance .001, bias balance .004, weight
diversity .01, bias diversity .02, temperature .2. Each role contributes half its
weighted terms so splitting roles preserves overall auxiliary-loss strength.
Twelve weighted components are accumulated and logged at validation.

Reseeding remains traffic-only, with EMA decay .998, relative utilization floor
.064/K and checks every 1024 optimizer steps. A zero LUT census never independently
triggers reseeding. DDP broadcasts parent maps and affected codebooks for all four
banks and copies optimizer moments to reseeded children.

Rank-0 validation logging reports hard/soft training EMAs, validation hard usage,
codebook geometry and sampled LUT margins per bank. Validation counts use fixed-size
on-device scatter additions; there is no per-batch host synchronization or extra
LUT construction for diagnostics. Validation usage is rank-0 data, not a global
all-rank census.

Trace rows are actual STM/OPP contexts, both STM context, both OPP context, and
swapped contexts. Each row re-decodes the role-specific LUT fields; IDs are not
copied between unrelated role banks. Individual readouts depend on both branches.

## Validation without training

Run the MoE6/router/serialization/FT-permutation tests on CPU. Set
`MOE6_ENGINE_PROBE` and `MOE6_TEST_NET` to enable the accumulator/trace probe in
`tests/test_moe6.py`. Compile `tests/moe6_engine_probe.cpp` against a matching
Stockfish build's objects except main.o, with identical architecture flags.

Cross-check a randomly initialized exported net using `cross_check_eval.py
--random-weights --device cpu` and a compatible MoE6 engine. Cover AVX-VNNI and
SSE execution and AVX512 compilation. CUDA tests require a separate capable host;
local validation performs no training or optimizer steps.
