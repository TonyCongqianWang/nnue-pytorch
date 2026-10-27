> Historical MoE5 architecture. The current experiment is documented in [moe6.md](moe6.md).

# MoE5: half-width perspective branches

MoE5 retains MoE4's K32H3 inputs, cascading material context, 64 weight experts,
256 bias experts, rank-32 code-to-weight bottleneck, quantization scales, and
traffic-only reseeding / loss settings. It changes downstream inference.

## Dimensions and coordinates

`--l1=1024 --l2=32 --l3=32` still denotes total widths. Each selected expert
now supplies 16 first-layer and 16 second-layer raw neurons. Dynamic bias
vectors have 64 elements: first squared 16, first linear 16, second squared 16,
second linear 16. Both routers retain their expert counts and LUT packing.

FT already returns `[STM, opponent]`. Branch A consumes this order and branch B
consumes `[opponent, STM]`. At each hidden layer, concatenate the affine neurons
before applying each activation kind, yielding:

```
[A_squared(16), B_squared(16), A_linear(16), B_linear(16)]
```

Both second-layer branches consume all 64 activations. B exchanges A/B within
each activation kind. This preserves cross-branch interaction. The output layer
consumes both hidden layers (128 activations); B exchanges the perspective groups
in each hidden layer before its own output transform.

A's scalar readout and first-layer skip are in STM coordinates; B's are in
opponent coordinates. Combine them as `(A - B)/2`, with integer differences widened
before averaging and truncation toward zero. The existing output conversion's
factor of two reconciles weight scales and is unchanged. There is no averaging
at hidden layers.

This shared-bank design is antisymmetric under exchanging FT perspective halves
and complete routes. It has no independent, perspective-even tempo term. Changing
only route IDs with FT inputs fixed has no prescribed symmetry. Identical routes
need not yield identical activations on asymmetric inputs; equal FT views and
equal complete routes do yield zero output.

## Arithmetic and storage

At the defaults, first-layer MACs are `2*1024*16 = 32768`, and second-layer MACs
are `2*64*16 = 2048`, matching master. The two scalar readouts cost 256 rather than
128 MACs. Total dense MACs are 35072 versus master's 34944. Weight tables are
approximately the footprint of 32 full-width experts; there is one shared bank,
not a duplicated weight bank for the two input orders.

Stockfish swaps the FT halves and sparse descriptors in stack buffers. Hidden
view exchanges account for the physical activation lane packing on each SIMD
backend. It retains existing activation kernels and serialized weight layouts.
Even identical expert IDs require evaluating both views; the old identical-input
shortcut is invalid here.

## Format, trace, and compatibility

The router tag is `MoE5` (`0x4D6F4535`), affine output sizes are halved, and the
serialized dynamic bias width is 64. Earlier networks and checkpoints are
incompatible; no automatic conversion is supplied.

Trace rows are ordered route assignments: `A / B` (actual), `A / A`, `B / B`,
and `B / A` (routes exchanged). They remain whole-network evaluations. The last
row is no longer labelled a symmetry check; perspective inversion must also
exchange the input views and negate the scalar.

## CPU checks without training

Build the native loader, then run:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_moe5.py tests/test_dual_moe_routing.py \
  tests/test_top2moe_symmetry.py tests/test_serialization_roundtrip.py \
  tests/test_ftperm.py tests/test_moe_load_balancing.py tests/test_compact_transport.py
```

`cross_check_eval.py --random-weights --device cpu` validates export/import and
feature permutation against a compatible engine without training. Use
`tests/moe5_engine_probe.cpp` for accumulator updates, undo, tracing and perspective
inversion. Link the probe against that engine's object files, excluding `main.o`,
with the same compiler/architecture flags. Enable its integration test with
`MOE5_ENGINE_PROBE=/path/to/probe MOE5_TEST_NET=/path/to/random.nnue`.

Commit the engine before pinning its SHA in both trainer CPU CI checkout steps,
then pin those compatible trainer and engine commits in nettest.
