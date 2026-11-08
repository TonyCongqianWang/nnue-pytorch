# MoE8.6: joint routing and expert-only biases

MoE8.6 retains upstream SFNNv19 block-major FT/Finny storage and the strict
opposing-side K32H3 rule: opposing queen, otherwise opposing rook, otherwise
neither. It is incompatible with earlier MoE checkpoints, .pt models and
network files, including MoE8.5. There is no migration or backward-compatible
loading path; training starts fresh. Its nettest recipe uses FT clipping
interval 1024 in every stage.

## Network

Widths are L1=1024, L2=L3=32. A 64-expert FC0 bank performs one conventional
full-width sparse projection. A 256-expert tail bank supplies the full-width
FC1 and scalar output projections. Both consume all relevant activations;
there are no perspective permutations, duplicate projections, or dynamic bias
banks. Each generated affine has expert-only biases; hidden layers retain
separate squared and linear bias paths. Native linear projections omit bias.

FC0 uses a rank-64 adapter and dimension-256 codebook. Tail adapters are rank
512 with dimension-2048 codebooks. Their normalized bottlenecks retain unit-L2
norm. Exports contain coalesced weights, with each bank independently sorted by
validation traffic and its LUT remapped accordingly.

FC0 has a shared weight matrix initialized to zero and a normally initialized
expert expansion. The base is coalesced into expert weights before QAT and
export, with no extra engine work. The base uses factorized weight decay
4e-5; generators retain dense decay 1e-6. All expert bias projections
initialize to zero. There are no shared downstream bias parameters or
optimizer entries. Bottleneck norms remain unchanged. FT
forward weights are rounded without clipping; downstream generated weights
retain their existing forward clamp.

Training, serialization and crosscheck share device resolution and a dense
matmul precision policy. `--device=auto` on export/crosscheck follows the
checkpoint backend when recorded, otherwise CUDA, MPS, then CPU. Numeric
device aliases select visible CUDA ordinals; explicit unavailable devices
fail. Training uses `--accelerator`/`--gpus` with the same backend resolver.
`--matmul-precision=auto` follows checkpoint metadata, or uses `high` for
quantized CUDA and `highest` otherwise. Routing and expert generation remain
inside IEEE/highest guards. A forced `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`
is rejected because it defeats those guards; matched nettest recipes unset it.

Export moves the model before FT projection, learned LUT generation, expert
coalescence and integer quantization; only byte packing moves integers to CPU.
Crosscheck uses the same preparation on the selected device. Checkpoint-Q
comparisons include export-time FT projection, explicitly announced in the
log. Maximum checkpoint/export error defaults to zero and is enforced on
every sampled position, rather than accepting the 99.7th percentile.
CPU random-net CI uses cyclic expert IDs to test every bank's codec without
a production learned-LUT sweep. CPU/CUDA learned-routing parity remains a
separate GPU diagnostic; matching precision is not a bit-identity guarantee
across hardware or software versions.

CPU DDP tests retain native material/king contexts, two-rank synchronization,
validation and checkpointing, using CLI-configured four-expert, eight-dimensional
routers and rank-four adapters. GPU kernel cases use representative shapes
instead of a full Cartesian sweep. No production architecture defaults are
reduced for testing.

The architecture can be reduced entirely through the shared model CLI flags:
`--l1`, `--l2`, `--l3`, `--moe-fc0-experts`, `--moe-tail-experts`,
`--moe-fc0-codebook-dim`, `--moe-tail-codebook-dim`, `--moe-fc0-rank`, and
`--moe-tail-rank`. The codebook dimensions also set the corresponding context
embedding widths. Pass identical model flags to training, serialization and
crosscheck. Production defaults remain 64/256 experts, dimensions 256/2048,
and ranks 64/512. Material/king context encoding is unchanged.

CPU pipeline CI uses the following small model for every stage, including
model/checkpoint resume, export, FT permutation and crosscheck:

```sh
--l1=256 --moe-fc0-experts=4 --moe-tail-experts=4 \
--moe-fc0-codebook-dim=8 --moe-tail-codebook-dim=8 \
--moe-fc0-rank=4 --moe-tail-rank=4
```

The matching Stockfish build uses `EXTRACXXFLAGS=-DNNUE_TEST_SMALL` and retains
32-neuron hidden layers for its SIMD paths. This is a compile-time CI preset;
ordinary builds retain the production architecture. Arbitrary smaller Python
models remain useful for trainer-only testing, but engine crosscheck requires
matching exported widths and expert counts. Adapter ranks/codebook widths
are coalesced away and do not affect the inference layout. The pipeline
test script forwards model flags to every subprocess and accepts separate
batch/epoch/validation/FT-optimization sizes, with small values selected in CI.

Tests are organized by current behavior (`test_moe_architecture.py`,
`test_final_validation.py`, `test_model_cli.py`) rather than historical MoE
version numbers. DDP tests invoke `train.py` directly, without a model factory
substitution.

FC0 scale is 512; FC1 and output scales are 128. Both bias-free FC0 differences
(raw outputs 14−15 and 30−31) contribute equally to the skip. Float inference
returns output + (skipA+skipB)/2. Integer inference uses
`merged = trunc((8*output + skipA + skipB)/2)`, then
`trunc(merged*9600/65536)`. Stockfish evaluation subsequently divides by 16.

## Routing context

Each side has 676 feasible `(T, P, H, wide)` material states: non-king count
0–15, pawn count 0–8, cascading own queen / own rook / opposite-color bishop
pair / other category, and pawn-wing occupancy. `wide` means at least one pawn
on a–c and one on f–h; central d/e pawns do not add a wing. Impossible states
are pruned, including wide states with fewer than two pawns.

FC0 routes on the ordered STM/OPP pair: `us*676 + opp`, 456,976 contexts.
Tail routing adds three king bits: bit 2 is our relative rank 3+, bit 1 is our
king strictly more advanced than theirs (both relative to their own home
rank), and bit 0 is clearly opposite flanks (a–c against f–h). A king on d/e
makes the opposite-flank flag false. Thus tail index is `pair*8 + king`, with
3,655,808 contexts. Castling rights are not included.

The loader's existing two int32 transport fields carry the FC0 pair and tail
index. They no longer describe separate perspectives. Both training encoders
use independent side embeddings; the tail additionally embeds the eight king
states. Full LUT construction is chunked to bound temporary memory.

## File and output

After coalesced FC0 and tail tensors, the file stores the FC0 LUT in six bits
per entry and the tail LUT in eight bits per entry. IDs are LSB-first; each
table starts on a byte boundary and final padding bits must be zero. Together
they occupy 3,998,540 bytes (3.81 MiB). Stockfish expands them once into two uint8
arrays (3.92 MiB) without inference-time unpacking.

Export defaults to `--histogram-mode aggregate`: one bias/weight histogram per
dense layer across all experts. `--histogram-mode all` prints each expert;
`--histogram-mode none` suppresses histograms and routing profiles. Legacy
`--histogram` requests all. Both bank profiles use matching ASCII tables;
aggregate mode omits long expert permutation lists. Upstream `--estimate-size`
and the counting exporter remain available and do not build LUTs.

Validation loss prints seven decimals. The recipe uses virtual FT decay 4e-6
and dense decay 1e-6, with no dynamic-bias settings. Only the last training
stage reduces the batch size to 65536.

## CPU validation

Run `pytest tests/test_moe_architecture.py tests/test_moe_routing.py
 tests/test_routing_contexts.py tests/test_moe_load_balancing.py -q`.
The optional incremental probe in `tests/moe_engine_probe.cpp` links against
a compatible engine's object files except main.o, with matching SIMD/compiler
flags. Set `NNUE_ENGINE_PROBE` and `NNUE_TEST_NET` to enable the engine test.
It compares forward/backward incremental updates, fresh reconstruction and
trace output across captures, promotions, castling and legal move sequences.
CUDA grouped-linear tests include all 512 experts, but require a GPU. No local
training is necessary for these checks.

## Final validation super-epoch

The final recipe stage sets `--final-validation-multiplier 4`. At the completed
final epoch, validation consumes four successive ordinary lengths from the
persistent input stream, without training or replaying cached batches. Earlier
stages and interrupted runs keep ordinary validation. The longer final pass
replaces the last ordinary validation when it is scheduled, and still runs if
the final epoch is not on the ordinary validation cadence.

Final router diagnostics collect exact int64 counts on every rank and reduce
only at the end. Ordinary validation remains rank-local for routing diagnostics.
`final-routing.csv` contains per-expert counts/fractions and nonzero joint
FC0/tail counts; the log gives their entropies and conditional tail entropy.
The final checkpoint is refreshed after aggregation so traffic-sorted export
uses the final global sample. No per-batch diagnostic collectives are added.
