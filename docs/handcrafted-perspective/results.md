# CPU validation and inference findings — 2026-10-29

These are inference measurements with synthetic weights and observational routing
samples. No trained handcrafted-perspective net exists from this work. No strength
measurement or controlled playing-speed comparison was made. No training or
optimizer steps ran locally, and no remote jobs were submitted.

## Validation

* 40 CPU unit/transport/grouped-fallback checks pass; 246 optional/GPU checks skip.
  The two-process Gloo exposure-union check passes separately with local loopback
  access. It exchanges metadata only and performs no training or optimizer steps.
* Independent exact P,N,B,R,Q enumeration and coarse-constraint enumeration give
  identical 85-live-tuple sets, with dense IDs and counts [4,12,15,15,15,9,9,6].
  Native trainer extraction matches Python on all live tuples, both colors and
  both side-to-move choices. Engine material extraction/mapping matches 18,144
  supported and unsupported exact-count cases, covering every live ID.
* Full-input branch and gradient reference, bias-free skips, exposed-only fallback
  averaging, persistent exposure state, optimizer parameter coverage without
  constructing an optimizer, mask dispatch, architecture rejection, and FT
  permutation including both role fallback matrices are checked on CPU.
* Synthetic and partially masked networks each pass 455 root comparisons and
  2,153 incremental/fresh/undo/actual-route trace checks. Episodes include captures,
  all four promotion types, en passant, castling, null moves and color inversion.
* AVX-VNNI and SSE each pass 512-position trainer/export/engine comparisons.
  A permuted masked net passes another 256 positions. Maximum differences are
  below one internal score unit, explained by final score truncation; the engine
  probe compares reconstructed integer scores exactly.
* Trainer re-export is byte-stable. Engine `export_net` reproduces both synthetic
  and masked source files byte-for-byte. AVX512 ICL compiles, without executing
  unsupported instructions. All builds and timings here are ordinary optimized
  builds, not PGO. The temporary search profiler explicitly loads a compatible
  synthetic network.

### Validation-device regression correction

The first remote CUDA validation exposed a missed device transition: ordinary
`at_index()` constructed a CPU layer, while factorized `at_index()` added shared
weights/biases still on the training device. CPU-only validation could not expose
that addition mismatch. Export snapshots now explicitly use CPU and retain the
source dtype; shared factors move to the snapshot's device/dtype before addition.
That correctness fix remains in the file-export path. Validation now avoids
CPU export entirely: both role means are computed by the same shared masked
reduction directly on the model device, and copied into device-resident fallback
buffers. Fixed shapes avoid boolean indexing/nonzero synchronization, and mode
changes use a Python cache of the persisted export marker rather than device
scalar reads. Empty fresh history and loaded inference artifacts retain their
existing behavior. File export copies the shared reduction's results to CPU.

Four new CPU regressions cover non-CPU fake/meta dispatch and numeric coalescing
and fallback means in float32/float64. The non-CPU fake regression reproduces the
old mixed-device failure without CUDA hardware. Two real-CUDA inference tests
cover validation transitions, fallback inference, export/re-export and returning
to training mode on devices 0 and 3, without optimizer steps; they skip locally
because CUDA is unavailable. Additional regressions forbid CPU copies, export
calls, tensor scalar reads and dynamic nonzero indexing during mode changes;
check that validation and export means match exactly on CPU; exercise non-CPU
fake/meta reductions; and restore the marker from state dicts and older `.pt`
objects. The CPU engine integration suite passes again.
Network/checkpoint formats and existing handcrafted checkpoint compatibility are
unchanged. No CUDA execution or training result is claimed by these local checks.

## Arithmetic and storage

The counts below use **85 live experts per role plus one derived fallback per
role**, not 144 allocated Cartesian-product experts. Quantized weights are int8
and biases int32. Shared FC0 factorization exists only in the trainer.

| Quantity | Value |
|---|---:|
| FC0 dense MACs, both roles | 32,768 |
| FC1 dense MACs, both roles | 2,048 |
| Readout dense MACs, both roles | 256 |
| Total dense MACs | 35,072 |
| Weights per role expert | 17,536 bytes |
| Biases per role expert | 132 bytes |
| All feasible experts, weights only | 2,981,120 bytes |
| All feasible experts plus fallbacks, weights only | 3,016,192 bytes |
| All parameters including fallbacks, before alignment | 3,038,896 bytes |
| Actual allocated expert banks, GCC `sizeof` | 3,071,232 bytes |
| Selected pair, raw parameters | 35,336 bytes |
| Selected pair, allocated objects | 35,712 bytes |

One expert object is 17,856 bytes with 64-byte-aligned matrices/biases. This is
slightly above the initial estimate because activation objects and inter-member
padding also occupy space. Routing adds a shared 144-byte constant table and
22 exposure-mask bytes; there is no large router LUT or active-weight cache.
A full nominal allocation would store 5,050,368 weight bytes before fallbacks;
structural pruning saves 2,069,248 weight bytes. The allocated fallback pair costs
35,712 bytes beyond the feasible banks.

Upstream's dense MAC count is 34,944 and MoE7's is 35,072. Including fallbacks,
this experiment stores 2.697× upstream's downstream weight bytes and 2.6875×
MoE7's, excluding MoE7 routing/bias metadata. More storage does not imply more
active MACs, but it increases the working set when routes change.

On the measured Intel i5-1245U, pinned logical CPU 0 shares a 48-KiB L1D and
1,280-KiB L2 with its sibling thread; L3 is 12 MiB. A selected pair can fit in L1D
by itself but competes with inputs, FT state, buffers and other engine data.
The complete 2.93-MiB expert allocation exceeds that core's L2 and fits within
shared L3 by itself. There are no measured hardware cache-miss counters here.

The 512 benchmark inputs contain a mean 177.51 nonzero four-value FC0 chunks.
The shared FC0 traversal therefore references approximately 22,721 weight bytes
per evaluation; FC1/readouts add 2,304, for about 25,025 executed MACs/weight bytes.
These are logical reads, not DRAM traffic or measured cache-line transfers.

## Matched direct-inference proxy

The reference uses separate calls to the two existing affine kernels. The
candidate uses fused direct two-bank FC0/FC1 kernels with shared NNZ discovery
and input loads. Neither packs or caches active weights. Weights are identical
deterministic quantized random values, loaded through the actual engine matrix
loader. Inputs are identical saved FT outputs of the compatible synthetic net
on legal-move episodes. They are **not trained-network activation statistics**.

Thirteen randomized paired rounds run pinned to logical CPU 0; the first is
excluded from summaries. Each case makes 250×512 downstream predictions and
includes material extraction, mask lookup, dispatch, activations, skips, head
averaging and scale conversion. All paired checksums agree. Lower ratios mean
less elapsed time for fused direct inference.

| Route pattern | Fused/separate median | Paired range | Fused ns/evaluation |
|---|---:|---:|---:|
| Fixed hot pair | 0.678 | 0.529–0.756 | 171.8 |
| Legal-game routes | 0.735 | 0.697–0.772 | 194.8 |
| Deterministic changing-route stress | 0.907 | 0.852–0.940 | 459.4 |

The changing-route stress accesses both entire banks, including fallback slots.
The higher absolute cost supports the cache-capacity concern even with unchanged
MACs. These figures compare fusion with separate calls for the **same architecture**;
they establish neither an upstream/MoE7 speedup nor playing strength. No packed
weight cache was reintroduced; the prior cache experiment remains a separate,
unsuccessful reference.

Input-file SHA256: `5197d1479d08ff609853542d7c395a5d09ec4d14d4e9eb0b2cc0734aff366c24`.
Synthetic net SHA256: `f3d8dd0c998555a33e31432e35d4afaf5150842377a6f482f3cfb5cb7864dc1f`.
Raw randomized timing rows and checksums are in `inference-measurements.json`.

## Routing traffic

Counts are observations, not uniform tuple enumeration. The legal-move proxy is
16 seeded random games of up to 96 plies plus explicit special-move episodes.
The dataset sample is the first 8,192 delivered positions from the available
`.pgo/small.binpack`, single-reader, without random/early filtering. Its order
is a loader sample, not a consecutive-search trajectory.

| Position source | Positions | Distinct role entries | Distinct pairs | Unused feasible STM / OPP |
|---|---:|---:|---:|---:|
| Legal-move proxy | 1,489 | 30 | 71 | 70 / 70 |
| Local dataset sample | 8,192 | 113 | 650 | 29 / 28 |
| Two depth-5 synthetic-net search traces | 459 NNUE calls | 12 | 16 | 79 / 79 |

Ten STM and fourteen OPP buckets have 1–9 observations in the dataset sample.
Full histograms and rare/unused ID lists are in `traffic-measurements.json`.
Sample absence never disables a feasible expert. Only persisted **training**
exposure controls the exported mask.

| Move class | Comparisons | STM ID changes | OPP ID changes | Pair changes | Color-normalized tuple changes |
|---|---:|---:|---:|---:|---:|
| Quiet nonpromotion | 1,287 | 687 | 687 | 687 | 0 |
| Capture, excluding promotion | 192 | 139 | 124 | 155 | 100 |
| Promotion | 7 | 2 | 6 | 6 | 4 |
| Null | 3 | 3 | 3 | 3 | 0 |
| Adjacent siblings, up to 16 legal children | 21,407 | 2,025 | 12 | 2,037 | — |

Quiet nonpromoting moves preserve each color's tuple by construction. Role
alternation can exchange different tuples between independent banks. Across
consecutive recorded legal positions, 849/1,488 pairs change, but 736 changes
are role alternation with unchanged color-normalized tuples. A changed ID thus
does not establish a new working-set entry.

To quantify reuse separately, simulated LRU caches keyed by `(role, compact ID)`
see 191 misses in 2,978 accesses with capacity four on legal episodes, versus
only 30 first-use entries in the whole sequence. On the short search traces,
there are 71 misses in 918 accesses at capacity four and 12 first-use entries.
These are **expert-entry reuse simulations**, not hardware cache measurements.
Matrices exceed one cache line and sparse reads need not fetch every column.

The trace-producing searches examined 331 and 660 nodes using the synthetic net;
only actual NNUE evaluations were logged. Their search trees are not comparable
to a different net's trees. Instrumentation was used only for traffic, not timings.
