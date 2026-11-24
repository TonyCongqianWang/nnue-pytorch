# Frozen tree128

This branch starts at trainer upstream/master `ef684e9aae2e894df0673b8cfd0062b123289fdd`.
Its matching engine starts at official Stockfish upstream/master
`3e8fde75412e7c47fc140cd82022a395bc681d92`; nettest starts at upstream/main
`6b75d7968ee9eb0a3cf98fd30db90cce37aedc28`. These were the fetched default
branch tips on 2026-11-24. No MoE neural parameters or architecture are used.

Use `--ls-keys tree128 --ls-remap-data ''`. There are 128 canonical raw leaf
IDs and initially 128 independent stored stacks with identity remapping. FT,
activations, skips, quantization and global FC0 factorization remain master's.
Do not use the Cartesian `ls-remap-data` nearest-neighbor heuristic for leaves.
Explicit remap tables and existing serialization/coalescing remain available.

## Factorization

`--ls-factors tree-depth2:tree-depth4:tree-depth6` shares additional weights and
biases among leaves with a common ancestor in the **original semantic tree**.
The three maps have 4, 16 and 59 groups. Leaves ending above a requested depth
retain their terminal node as their group. Group IDs are derived from sorted
semantic node IDs; they do not depend on machine dispatch layout. Every leaf
also keeps independent residual stack parameters. Factors start at zero and
use the existing optimizer, clipping, coalescing and export code.

When `tree128` is selected, the original default Cartesian factor string is
replaced by this ancestry setting. Other Cartesian factor names are rejected.
Use `--ls-factors ''` to disable group factors while retaining global FC0
factorization. The strict routing/count experiment is the unfactorized tree
versus the unfactorized 192-stack master control; ancestry factors are a
separate experimental arm. They are a reasonable structural prior, not a
measured strength improvement.

## Frozen specification and compatibility

`routing/tree128/routing_tree.json` is the semantic source, SHA256
`c0ee18ceb89d0871c63f593ef4ff1a3f5d3d6e4d26652cbb35f5e89342c085f2`.
Canonical IDs are the frozen leaf IDs, never the teacher class IDs. Lowering
artifacts are pinned separately and retain their original path preconditions.
Actual switches dispatch on one raw field or bounded pair key. Counts are
capped at 15; material-category counts remain unclipped; differences use signed
integers. Only needed context fields are extracted and cached on demand.

Generate both consumers from this checkout:

```sh
python -m routing.tree128.generate --engine /path/to/Stockfish
```

The generator verifies the bundled manifests. Exhaustive validation also
verifies the saved sklearn joblib and leaf map against the frozen manifest.
`implementation_hashes.json` records the generated code and shared extractor
hashes separately. Binary/small-switch/full-mixed layouts have one semantic
identity. `TREE128_LAYOUT=0/1/2` selects them at compile time; binary is the
conservative default pending repeatable gains with trained matching nets.

The network table/header compatibility hash incorporates the semantic digest,
and the table record also carries all 32 digest bytes immediately after its
marker, before raw/stored counts. Readers reject a wrong tree even if its
32-bit compatibility hash collides. Ordinary master network format is
unchanged. Checkpoints carry a persistent digest buffer and validate it before
loading; whole-object `.pt` loads and exports validate it too.

## Validation commands

Run commands from the trainer root, using its freshly built native library:

```sh
cmake -S data_loader/cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest -q tests/test_tree128.py tests/test_serialization_roundtrip.py tests/test_counting_exporter.py tests/test_compact_transport.py
python tools/validate_tree128.py --analysis /path/to/moe10_analysis --output /path/to/results/routing.json
python tools/validate_tree128_fens.py --engine /path/to/Stockfish/src/stockfish --output /path/to/results/fens.json
python tools/tree128_traffic.py --recipe /path/to/nettest/tree128.yaml --output /path/to/results/traffic.json /path/to/data.binpack
```

`cross_check_eval.py` retains the normal export/QAT/engine flow; pass
`--ls-keys tree128` consistently to training, serialization and crosscheck.
The engine's `tree128` diagnostic prints its semantic identity, eight fields and
canonical leaf ID. Engine incremental/undo and PGO utilities live under its
`tests/` directory.

## Traffic and experiment limits

Training writes `layer_stack_traffic.jsonl` in the run root each epoch: raw-leaf
counts from optimizer-consumed batches and a separately labeled source mix
from post-filter provider snapshots. Source snapshots include prefetch and
must not be interpreted as the exact epoch optimizer mixture. DDP sums both
statistics. Resumed epoch reports cover their process-local epoch segment.
Ply 9999 marks a search-tree sample, matching master's skip predicate.

The implementation also fixes a pre-existing epoch-boundary resume bug found
by the smoke test: checkpoints now record `epoch_finished`, completed epochs
resume at the next epoch, and epoch schedulers advance before saving. Old
checkpoints are treated as completed epochs, matching nettest's existing
interpretation. Mid-epoch checkpoints resume their unfinished epoch; exact
minibatch/sampler replay is not implemented.

95% fishpack / 5% search-tree is **router-analysis weighting**, not a guaranteed
training mixture. The recipe preserves upstream binpacks, filtering and tree
acceptance. Measure the resulting mixture over the full run before claiming
5%. The local fishpack-only diagnostics necessarily report zero search-tree
traffic.

Local acceptance results are in `validation/tree128/`. CUDA kernel/DDP
validation and long training/strength testing remain to be done on suitable
hardware. Validation networks are freshly randomized master networks; they
are not trained chess-strength candidates. No remote training or push is
part of this implementation.
