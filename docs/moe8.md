# MoE8: joint routing and one full-width downstream network

MoE8 retains upstream SFNNv19 block-major FT/Finny storage and the strict
opposing-side K32H3 rule: opposing queen, otherwise opposing rook, otherwise
neither. It is incompatible with earlier MoE network files.

## Network

Widths are L1=1024, L2=L3=32. A 64-expert FC0 bank performs one conventional
full-width sparse projection. A 512-expert tail bank supplies the full-width
FC1 and scalar output projections. Both consume all relevant activations;
there are no perspective permutations, duplicate projections, or dynamic bias
banks. Each generated affine has a shared static bias; hidden layers retain
separate squared and linear bias paths. Native linear projections omit bias.

FC0 uses a rank-64 adapter and dimension-256 codebook. Tail adapters are rank
512 with dimension-2048 codebooks. Their normalized bottlenecks retain unit-L2
norm. Exports contain coalesced weights, with each bank independently sorted by
validation traffic and its LUT remapped accordingly.

FC0 scale is 256; FC1 and output scales are 128. Both bias-free FC0 differences
(raw outputs 14−15 and 30−31) contribute equally to the skip. Float inference
returns output + (skipA+skipB)/2. Integer inference uses
`merged = trunc((4*output + skipA + skipB)/2)`, then
`trunc(merged*9600/32768)`. Stockfish evaluation subsequently divides by 16.

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
per entry and the tail LUT in nine bits per entry. IDs are LSB-first; each
table starts on a byte boundary and final padding bits must be zero. Together
they occupy 4,455,516 bytes (4.25 MiB). Stockfish expands them once into uint8
and uint16 arrays (7.41 MiB) without inference-time unpacking.

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

Run `pytest tests/test_moe8.py tests/test_moe_routing.py
 tests/test_routing_contexts.py tests/test_moe_load_balancing.py -q`.
The optional incremental probe in `tests/moe8_engine_probe.cpp` links against
a compatible engine's object files except main.o, with matching SIMD/compiler
flags. Set `MOE8_ENGINE_PROBE` and `MOE8_TEST_NET` to enable the engine test.
It compares forward/backward incremental updates, fresh reconstruction and
trace output across captures, promotions, castling and legal move sequences.
CUDA grouped-linear tests include all 512 experts, but require a GPU. No local
training is necessary for these checks.
