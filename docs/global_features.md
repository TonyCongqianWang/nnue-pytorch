# Experimental mirrored threats and routed support

Default features are
`Own_Supports+Enemy_Supports+Global_Attacks+Pawn_FrontierPairs+Pawn_FilePairs+K32Q2^`.
This architecture requires new training. Network version `0x6a448afc` rejects
previous experimental and upstream networks; incompatible checkpoints are not
migrated.

| Input | Training rows | Export rows | Columns at L1=1024 | Hash |
| --- | ---: | ---: | ---: | --- |
| Own_Supports | 14,952 | 14,952 | 768 | 0x4f535031 |
| Enemy_Supports | 14,952 | 14,952 | 256 | 0x45535031 |
| Global_Attacks | 29,904 | 29,904 | 1,024 | 0x47415432 |
| Pawn_FrontierPairs | 16,807 | 16,807 | 1,024 | 0x50465032 |
| Pawn_FilePairs | 480 | 480 | 1,024 | 0x50464332 |
| K32Q2^ | 49,152 | 45,056 | 1,024 | 0x32b5e284 |

The shared int16 bias precedes these int8 tables in order. Tables are serialized
individually at their actual widths, followed by the existing 32 downstream
stacks. L1=1024 and L2=L3=32 remain unchanged. FT quantization is symmetric
[-127,127]. Other L1 widths must be divisible by 16 when using routed supports.

Each perspective has two 512-lane product operands, giving 512 bilinear outputs
per perspective and 1024 downstream inputs. Each operand receives 384 own-support
lanes followed by 128 enemy-support lanes. Support tables split their columns
evenly between operands. PSQ, attacks and pawns contribute to every raw lane.
Support has no signed virtual rows: own/enemy tables learn independently, shared
across colors. Inference packs the sums into two 1024-wide owner-associated
support banks, plus two 1024-wide base banks. Support updates touch 1024 lanes;
attack updates can touch 2048 across the two perspectives.

Every spatial input follows the receiving king's PSQ horizontal mirror (files
a..d reflect). Black rank/color normalization matches PSQ. Enemy support uses
the receiving enemy king's orientation, not the supporting owner's king. Threat
target exclusions, deduplication and king exclusions are retained.

Frontier states use missing=0 and rank 2..7=1..6, with state `7*white+black`.
Boundary f indexes `2401*f+49*state[f]+state[f+1]`. Horizontal mirroring reverses
file order and swaps boundary states. Black normalization swaps colors and maps
nonzero ranks r to 7-r. File pairs select consecutive occupied pawn ranks,
regardless of color: 15 White-White, 15 Black-Black and 30 White-Black
distinct-rank states per file, in lexicographic order. Horizontal mirroring
replaces file f with 7-f. These families retain complete pawn placement.

Training zero-pads the compact support tables into their routed 1024-column
positions and concatenates them with the other tables. The existing fused CUDA
FT handles both perspectives and the paired activation; normal PyTorch fallback
remains available. Autograd routes gradients through padding/concatenation to
the compact parameters. Export remains compact and unchanged.

Own and enemy support share the previous 256-slot active-support allowance.
The default composed loader and Python feature metadata both reserve 599 sparse
index slots, matching the previous iteration. No per-group index sorting or
separate sparse FT calls are required. Aggregated-backward eligibility/tuning
is unchanged; the existing 288-slot cutoff still excludes this composition.

FT optimization permutes products only within the own-support (384) or
enemy-support (128) section, consistently across operands, perspectives,
support tables, PSQ/pawn/attack columns and downstream weights. Cross-section
permutations are rejected.

`tests/test_global_features.py` provides a python-chess reference, native-loader
and optional engine parity, routing gradients, quantized round trips and
permutation checks. Set `GLOBAL_NNUE_TEST` to the engine harness. Larger sets:

```sh
python tests/test_global_features.py --positions /path/to/positions.epd \
  --engine-test /path/to/Stockfish/src/global-nnue-test
```

CI retains full pytest discovery and the CPU training/resume/export/FT-optimize
smoke test, pinned to the compatible engine SHA. A separate compatibility
workflow runs generated/random-network checks without optimizer steps. Local
validation excludes training, bench, games and strength testing.
