# MoE4 experiment

Historical architecture, preserved by the first commit on `moe5`. See
[MoE5](moe5.md) for the current dimensions, perspective ordering and checks.

The network uses 64 layer-stack experts and 256 bias experts. Each perspective's
uint16 router entry packs the weight ID in bits 0–5 and bias ID in bits 6–13;
bits 14–15 are zero. Downstream weights and biases remain coalesced on disk.

Material context retains non-king piece count T and pawn count P. A four-way
category replaces the independent queen and bishop-pair flags:

1. Queen present.
2. Otherwise rook present.
3. Otherwise bishops on both square colors.
4. Otherwise.

Enumerate T=0..15, P=0..min(8,T), category=0..3 in that order, excluding queen
or rook categories when T-P<1 and bishop-pair categories when T-P<2. This yields
396 states and 396*33*65 = 849,420 router contexts. The existing king context
and castling-right conventions are unchanged.

K32H3 combines 32 own-king buckets with opposing queen / opposing rook / neither.
It has 73,728 training inputs and 67,584 exported inputs after king-plane merging.
Stockfish PSQ indices must therefore be 32-bit, including the AVX-512 path.
Accumulator refreshes conservatively include opposing queen/rook captures and
promotions; the hybrid king update excludes captures of those pieces.

Both selected affine transforms consume the same input at every layer. Average
both affine contributions and corresponding dynamic biases before each squared
and linear activation. The output transforms consume concatenated shared hidden
activations; average their results and add the averaged first-layer skip.
Integer averages truncate toward zero, with wide additions in Stockfish and
matching straight-through fake quantization in the trainer. Existing weight
scales and the output conversion's scale-reconciliation factor of two stay intact.

Tracing reports complete weight/bias routes A and B, actual A+B, counterfactuals
A+A and B+B, and B+A as a symmetry check. These are combined-network evaluations,
not additive per-head contributions.

The router and PSQ hashes changed. Previous architecture networks/checkpoints
are incompatible; no automatic checkpoint migration is provided.

## Initialization and routing diagnostics

Each layer-stack rank column is a full layer basis initialized uniformly within
`+/-1/sqrt(in_features)`. The unit-L2 bottleneck already preserves the variance
of independent bases; an additional rank factor would shrink initial weights.
Only fresh initialization changes. Output bias remains zero, and normalization,
static/dynamic bias initialization, and quantization scales stay unchanged.

Weight/bias balancing costs are independently configurable with
`--moe-weight-balance-cost` (0.001) and `--moe-bias-balance-cost` (0.004).
Both balancing temperatures remain 0.2, with independent
`--moe-weight-temperature` and `--moe-bias-temperature` options. Commitment is
unchanged. Diversity coefficients are separately configurable using
`--moe-weight-diversity-cost` (0.01) and `--moe-bias-diversity-cost` (0.02).

`--moe-reseed-min-relative-utilization` replaces the old absolute-utilization
option. Its default 0.064 sets the traffic floor to 0.064/K: 0.001 for 64 weight
experts and 0.00025 for 256 bias experts. Reseeding uses hard-traffic EMA alone;
a momentarily empty LUT cell does not trigger it. Events log pre-split traffic
and floors. No context enumeration or LUT construction runs during reseeding.
`--moe-reseed-ema-decay` defaults to 0.998, giving roughly 1,373 zero-traffic steps
from uniform usage to the floor. Checks remain every 1,024 steps.

LUT construction stays lazy: validation/inference or export builds it once when
invalid and reuses the result until training or reseeding invalidates it. At
validation only, an existing necessary LUT build samples one in 256 contexts'
nearest/runner-up squared-distance gaps. Logging never requests another build;
export performs no margin top-k work. Validation also reports maximum pairwise
codebook cosine and the expert pair, plus sampled margin minimum, mean and
fraction at most 1e-6 when available. These are uniform context samples, not
training-frequency-weighted statistics.

Epoch metrics include all six weighted MoE contributions (commitment, balancing,
diversity, separately for weights and biases). All epoch loss accumulators reset
at each epoch boundary. Validation summaries print these contributions and
rank-0 training hard/soft traffic EMAs, effective usage, below-floor counts, and
five lowest-traffic experts. Loss contributions are averaged across ranks at the
epoch boundary. Traffic summaries are explicitly rank-local; no new
per-step collective, CPU synchronization, full-context sweep, or weight scan is
introduced. Existing assignment means and device-side loss sums are reused.
Validation does not update training traffic EMAs.

The expert count changes the network hash; earlier 64-bias networks/checkpoints
are incompatible with this iteration.

## CPU validation without training

Build the native loader from `data_loader/cpp`, then run:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_moe4.py tests/test_dual_moe_routing.py \
  tests/test_top2moe_symmetry.py tests/test_serialization_roundtrip.py \
  tests/test_ftperm.py tests/test_moe_load_balancing.py \
  tests/test_compact_transport.py
```

`cross_check_eval.py --random-weights --device cpu` checks random initialized
weights against the engine, including feature permutation and export/import,
without training. Supply a compatible Stockfish binary and binpack dataset.

`tests/moe4_engine_probe.cpp` checks complete accumulators, evaluation, tracing,
head-swap symmetry, lazy updates and undo against fresh reconstruction. Link it
against a compatible Stockfish build's object files, excluding `main.o` and
obsolete feature objects. Use the same architecture defines/compiler flags as
that build and link from the Stockfish source directory if embedding uses a
relative network path. The probe takes a compatible `.nnue` path as its argument.
Run its deterministic capture/promotion/castling/random-game suite with:

```sh
MOE4_ENGINE_PROBE=/absolute/path/to/probe \
MOE4_TEST_NET=/absolute/path/to/random.nnue \
python -m pytest tests/test_moe4.py::test_engine_accumulator_paths
```

GitHub CPU CI must pin the compatible Stockfish commit in both checkout steps.
Commit Stockfish first, then update those pins before committing the trainer.
