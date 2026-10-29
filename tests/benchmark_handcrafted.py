"""Randomized paired inference proxy timings; a checksum mismatch is fatal."""

import argparse
import json
import os
import random
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("binary")
p.add_argument("inputs")
p.add_argument("output")
args = p.parse_args()
cpu = min(os.sched_getaffinity(0))
rng = random.Random(617)
proc = subprocess.Popen(
    ["taskset", "-c", str(cpu), args.binary, args.inputs],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)
sizes = proc.stdout.readline().strip()
rows = []
for repeat in range(13):
    cases = [(mode, kind) for kind in range(3) for mode in range(2)]
    rng.shuffle(cases)
    for mode, kind in cases:
        proc.stdin.write(f"{mode} {kind}\n")
        proc.stdin.flush()
        values = proc.stdout.readline().split()
        assert len(values) == 4, values
        rows.append(
            {
                "repeat": repeat,
                "mode": int(values[0]),
                "kind": int(values[1]),
                "ns": float(values[2]),
                "checksum": int(values[3]),
            }
        )
proc.stdin.close()
assert proc.wait() == 0
summary = {}
for kind in range(3):
    ratios = []
    for repeat in range(1, 13):
        fused = next(
            r
            for r in rows
            if r["kind"] == kind and r["mode"] == 0 and r["repeat"] == repeat
        )
        split = next(
            r
            for r in rows
            if r["kind"] == kind and r["mode"] == 1 and r["repeat"] == repeat
        )
        assert fused["checksum"] == split["checksum"], (fused, split)
        ratios.append(fused["ns"] / split["ns"])
    summary[kind] = {
        "fused_over_separate_median": statistics.median(ratios),
        "range": [min(ratios), max(ratios)],
        "fused_ns_median": statistics.median(
            r["ns"]
            for r in rows
            if r["kind"] == kind and r["mode"] == 0 and r["repeat"] > 0
        ),
    }
result = {"cpu": cpu, "sizes": sizes, "summary": summary, "rows": rows}
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
