"""Exhaustive independent semantic/sklearn/native validation, with traffic report.

Run from repository root: python tools/validate_tree128.py --analysis PATH --output report.json
The analysis directory is read-only; results and compiled probe go under --output.
"""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing.tree128 import IDENTITY, ROOT, SPEC


def predict(raw):
    """Vectorized frozen JSON oracle, independent of lowered rules/codegen."""
    nodes = {n["node_id"]: n for n in SPEC["nodes"]}
    out = np.empty(len(raw), np.int16)

    def operand(d, rows):
        v = raw[rows, d["raw"]].astype(np.int16)
        return np.isin(v, d["values"]).astype(np.int16) if "values" in d else v

    def walk(node_id, rows):
        if not len(rows):
            return
        n = nodes[node_id]
        if "leaf_id" in n:
            out[rows] = n["leaf_id"]
            return
        c = n["condition"]
        if c["op"] == "raw_membership":
            yes = np.isin(raw[rows, c["raw_feature"]], c["values"])
        else:
            d = SPEC["derived_features"][c["feature"]]
            if d.get("op") == "bit_set":
                v = (raw[rows, d["raw"]] & d["bit"]) != 0
            elif d.get("op") == "membership":
                v = np.isin(raw[rows, d["raw"]], d["values"])
            else:
                a, b = operand(d["us"], rows), operand(d["them"], rows)
                v = {
                    "sum": lambda: a + b,
                    "difference": lambda: a - b,
                    "absolute_difference": lambda: np.abs(a - b),
                    "equal": lambda: a == b,
                }[d["relation"]]()
            yes = v <= c["threshold"]
        walk(n["yes"], rows[yes])
        walk(n["no"], rows[~yes])

    walk(SPEC["root"], np.arange(len(raw)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.analysis / "architecture_scope/range_tree128"
    for name, digest in json.loads((ROOT / "artifact_hashes.json").read_text()).items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    sys.path.insert(0, str(args.analysis))
    from analyze_moe_lut import load_lut
    from refit_range_tree128 import range_features

    shared = args.output.parent / "tree128_probe.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "tests/tree128_native.cpp",
            "-o",
            str(shared),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(shared))
    lib.tree128_check_cartesian.restype = ctypes.c_int
    assert lib.tree128_check_cartesian() == 1
    fn = lib.tree128_select_many
    ptr = np.ctypeslib.ndpointer(dtype=np.int16, flags="C_CONTIGUOUS")
    fn.argtypes = [ptr, ctypes.c_int, ptr]
    print(
        "All 5,898,240 bounded contexts: reference/binary/small_switch/mixed exact parity",
        flush=True,
    )
    X, teacher, net_sha = load_lut(args.analysis / "nn-5ea7f7979fc4.nnue")
    saved = joblib.load(source / "candidate_tree128_ranges.joblib")
    model = saved["model"]
    canonical = {n: i for i, n in enumerate(SPEC["canonical_leaf_node_ids"])}
    result = np.empty(len(X), np.int16)
    for start in range(0, len(X), 16384):
        raw = np.ascontiguousarray(X[start : start + 16384], dtype=np.int16)
        native = np.empty((len(raw), 4), np.int16)
        fn(raw, len(raw), native)
        expected = predict(raw)
        assert np.all(native == expected[:, None]), start
        sklearn = np.array(
            [
                canonical[int(n)]
                for n in model.apply(range_features(raw, saved["package"])[0])
            ]
        )
        assert np.array_equal(expected, sklearn), start
        result[start : start + len(raw)] = expected
    cf = np.load(args.analysis / "fishpack_context_counts.npy")
    cs = np.load(args.analysis / "searchtree_context_counts.npy")
    w = 0.95 * cf / cf.sum() + 0.05 * cs / cs.sum()
    support = np.bincount(result, weights=w, minlength=128)
    expected_support = json.loads((ROOT / "leaf_map.json").read_text())["traffic"]
    np.testing.assert_allclose(support, expected_support, rtol=0, atol=1e-14)
    assert np.all(support > 0)
    labels = np.array(
        [
            int(model.classes_[model.tree_.value[n][0].argmax()])
            for n in SPEC["canonical_leaf_node_ids"]
        ]
    )
    agreement = float(w[labels[result] == teacher].sum())
    assert abs(agreement - 0.8789509773254394) < 1e-12
    report = {
        "semantic_sha256": IDENTITY,
        "teacher_sha256": net_sha,
        "lut_contexts": len(X),
        "bounded_contexts": 5898240,
        "weighted_teacher_agreement": agreement,
        "weighting": "analysis only: 95% fishpack / 5% search-tree; not measured training traffic",
        "leaves": [
            {
                "leaf": i,
                "weighted_support": float(support[i]),
                "fishpack": int(np.bincount(result, weights=cf, minlength=128)[i]),
                "searchtree": int(np.bincount(result, weights=cs, minlength=128)[i]),
            }
            for i in range(128)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"All {len(X):,} LUT contexts: JSON/sklearn/native exact canonical leaf parity; teacher agreement {agreement:.12%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
