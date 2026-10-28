#!/usr/bin/env bash
# Run from the trainer root with a matching, pinned Stockfish checkout.
set -euo pipefail
stockfish_root=${STOCKFISH_DIR:-$PWD/Stockfish}
output_root=${COMPOSITION_TEST_DIR:-/tmp/composition-smoke}
mkdir -p "$output_root"
COMPOSITION_TEST_OUT="$output_root/native" "$stockfish_root/tests/composition.sh" avx2
COMPOSITION_NATIVE_TEST="$output_root/native" pytest -q tests/test_composition.py
# Keep the original baseline CPU pipeline and its width unchanged. This separate
# smoke uses less memory, but runs all training/resume/export/permutation stages.
python -u tests/test_training_pipeline_run.py -y --device cpu --num_workers 1 \
  --l1 128 --features 'Full_Threats+PP_3Wide+K32H3^+Composition' \
  --test-dir "$output_root/training"
make -C "$stockfish_root/src" -j2 all ARCH=x86-64-avx2 EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF
python -u cross_check_eval.py --random-weights --device cpu --seed 42 --count 1024 \
  --features 'Full_Threats+PP_3Wide+K32H3^+Composition' --data .pgo/small.binpack \
  --engine "$stockfish_root/src/stockfish" --max-error 5
# Validate the trained 128-lane smoke too, in a disposable copy; the experiment's
# actual architecture remains 1024 lanes in the pinned source checkout.
engine128=$(mktemp -d "$output_root/engine128.XXXXXX")
cp -a "$stockfish_root/src/." "$engine128/"
python - "$engine128/nnue/nnue_architecture.h" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
assert 'constexpr IndexType L1 = 1024;' in s
p.write_text(s.replace('constexpr IndexType L1 = 1024;', 'constexpr IndexType L1 = 128;'))
PY
make -C "$engine128" objclean
make -C "$engine128" -j2 all ARCH=general-64 EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF
checkpoint_root="$output_root/training/training_logs/version_2/checkpoints"
python -u cross_check_eval.py --device cpu --count 1024 --l1 128 \
  --features 'Full_Threats+PP_3Wide+K32H3^+Composition' --data .pgo/small.binpack \
  --engine "$engine128/stockfish" --net "$checkpoint_root/last.nnue" \
  --checkpoint "$checkpoint_root/last.ckpt" --max-error 5
