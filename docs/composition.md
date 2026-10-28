# Composition-conditioned material experiment

Enable `--features 'Full_Threats+PP_3Wide+K32H3^+Composition'` for training,
serialization, optimization, and cross-checking. The default feature set and
existing CI training jobs remain unchanged. Use the matching experimental
Stockfish revision pinned in `.github/workflows/composition.yml`.

## Parameters and export

The ten relative classes are own P/N/B/R/Q followed by enemy P/N/B/R/Q.
Presence forms a 10-bit bucket, shared between perspectives after normalization.
Rows are packed by bucket then class; absent classes have no row. Counts are
represented by repeated sparse IDs. This uses the existing native batch ABI and
sparse forward/backward kernels, before clipping and bilinear multiplication.

Each class occurs in 512 buckets. Its full-presence (1023) row is the shared
base. The other 511 rows are trainable corrections, with no correction parameter
for bucket 1023. The effective weight is base + correction in those buckets and
base alone in bucket 1023. This is a bijective parameterization of 5120 rows,
not 5130 redundant rows. Material starts at zero. Gradients from every visited
composition train the shared bases, including when a bucket's own correction has
never been sampled. `--composition-weight-decay` independently regularizes bases
and corrections; the recipe uses `4e-5`, encouraging rare corrections toward
zero/shared material. This value is experimental, not strength-tuned.

Export folds bases into corrections and quantizes the resulting rows to int8
with the baseline FT scale. Nothing extra is written for the shared base. The
feature hash distinguishes the format. At 1024 lanes the exact additional
trainable count is **5,242,880**, with **20 MiB** of float32 parameter storage and
**5 MiB** of additional uncompressed exported data. Optimizer states and gradients
need additional training memory. Accumulator width and dense layers are unchanged.
Material rows participate in the existing neuron permutation and serialization.

## Checks

Rebuild the loader after changing native sources. `pytest tests/test_composition.py`
checks all 1024 buckets, native/Python counts and feature parity, perspective
swaps, mirrored positions, shared-base gradients, repeated-ID backward behavior,
preclip/bilinear placement, quantized serialization, byte/parameter overhead,
format rejection, and independent decay grouping. Set `COMPOSITION_NATIVE_TEST`
to a Stockfish `tests/composition.sh` executable for three-way feature parity.

`scripts/test_composition.sh` runs native accumulator tests, these Python tests,
the full short training/resume/serialization/permutation pipeline at 128 lanes,
a 1024-lane synthetic-network cross-check, and a trained 128-lane cross-check
against a disposable matching scalar engine build (the baseline SIMD sparse
layer requires widths divisible by 256). The reduced smoke width is a memory
accommodation; the experimental recipe and engine architecture remain 1024.
The unchanged default training smoke still runs at 1024 lanes in baseline CI.

No full training, games, engine bench, or playing-strength claims are included.
See the Stockfish `tests/COMPOSITION.md` and nettest `COMPOSITION.md` for state,
cache, recipe and measured-validation details.

## Local validation results

Fresh bases were Stockfish `ff7cee4244ec96ea488c402669e3674c0574dce6` and
trainer `b26145006e54e6461cb4b51944ab05cc4850c580`.

- Native scalar and AVX2: 5379 accumulator comparisons per build, including the
  requested special moves, reverse lazy updates, undo, null moves, cache
  refreshes, and test counters for zero quiet material state updates.
- Six focused material tests pass. Parity covers all 1024 buckets across Python,
  the loader, and Stockfish, plus 400 legal-game positions. Shared-base backward
  coverage also uses the actual 1024-lane table.
- Existing relevant feature/serialization/optimizer/transport checks pass.
  A full pytest run reported 77 passed, 409 skipped (mostly accelerator/device
  tests), and one two-rank training failure by SIGKILL. The same test was killed
  on the freshly fetched, unmodified trainer default. The container has an 8 GiB
  memory limit. Socket-based tests require local network permissions here.
- Full-width random network versus AVX2 engine: 1024 positions, quantized error
  below 1 cp. Trainer export → engine export is byte-identical, including material.
- Full short training pipeline at 128 lanes: training, model resume, checkpoint
  resume, SWA, serialization, quantization, and neuron permutation pass. The
  trained 128-lane export versus a matching scalar engine also passes on 1024
  positions with quantized error below 1 cp. The full-width training smoke exceeded
  this container's memory limit, even with one loader worker; it is not reported
  as passing. Neither the default smoke width nor baseline CI was reduced.
- Actual complete export sizes (identical descriptions): 136,247,590 baseline,
  141,490,470 material bytes. Parameter counts: 143,350,848 baseline,
  148,593,728 material. Both deltas are exactly 5,242,880.
- Ruff, C++ formatting, shell syntax, and workflow YAML checks pass.

Synthetic warm-cache move-update measurements found quiet/king cases within
about 2% of baseline; two capture cases cost approximately 13% and 40% more.
Already-computed accumulator evaluation remained about 12 ns, with considerable
relative timing noise. These are narrow accumulator microbenchmarks, not search
throughput or playing-strength measurements. See Stockfish's timing harness and
results for details and the cold/stale king-cache qualification.
