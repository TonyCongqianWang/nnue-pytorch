"""Compact, matching ASCII profiles for the two joint-context expert banks."""

import sys

import numpy as np
import torch

from ..modules.router import KING_STATES, MOE_BANKS, TAIL_GEOMETRY_STATES


def _table(headers, rows, file):
    widths = [
        max(len(str(header)), max((len(str(row[i])) for row in rows), default=0))
        for i, header in enumerate(headers)
    ]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(border, file=file)
    print(
        "| " + " | ".join(str(v).center(w) for v, w in zip(headers, widths)) + " |",
        file=file,
    )
    print(border, file=file)
    for row in rows:
        print(
            "| " + " | ".join(str(v).rjust(w) for v, w in zip(row, widths)) + " |",
            file=file,
        )
    print(border, file=file)


@torch.no_grad()
def print_routing_profile(model, file=None):
    file = sys.stdout if file is None else file
    router = model.router
    tables = router.get_lookup_table()
    print("\nMoE10 routing profiles. Export sorts by validation traffic.", file=file)
    print("Heavy and wing shares are within each expert's LUT contexts.", file=file)
    for name, table in zip(MOE_BANKS, tables):
        material = getattr(router, f"{name}_material_decode").cpu().numpy()
        states = router.material_states[name]
        ids = table.cpu().numpy()
        n = len(router.codebooks[name])
        count = np.bincount(ids, minlength=n)
        sums = np.zeros((4, n))
        heavy = np.zeros((2, n, 4), dtype=np.int64)
        geometry = np.zeros((TAIL_GEOMETRY_STATES, n))
        for start in range(0, len(ids), 65536):
            chunk = ids[start : start + 65536]
            index = np.arange(start, start + len(chunk))
            pair = index if name == "fc0" else index // TAIL_GEOMETRY_STATES
            own, opp = material[pair // states], material[pair % states]
            for i, values in enumerate((own[:, 0], opp[:, 0], own[:, 1], opp[:, 1])):
                sums[i] += np.bincount(chunk, weights=values, minlength=n)
            for side, categories in enumerate((own[:, 2], opp[:, 2])):
                heavy[side] += np.bincount(
                    chunk * 4 + categories, minlength=n * 4
                ).reshape(n, 4)
            if name == "tail":
                group = index % TAIL_GEOMETRY_STATES
                geometry += np.bincount(
                    group * n + chunk, minlength=geometry.size
                ).reshape(geometry.shape)
        means = sums / np.maximum(count, 1)
        val = getattr(router, f"validation_counts_{name}").cpu().numpy()
        total_val = val.sum()
        headers = [
            "ID",
            "LUT %",
            "Val %",
            "T us/opp",
            "RelP us/opp" if name == "fc0" else "P us/opp",
            "Heavy % STM / OPP (Q/R/BP/Else)",
        ]
        if name == "tail":
            headers.append("Wing state % (0/1/2/3/4)")
            wings = geometry.reshape(-1, KING_STATES, n).sum(1)

        def shares(counts, total):
            return "/".join(f"{100 * value / total:.0f}" for value in counts)

        rows = []
        for expert in np.argsort(-val, kind="stable"):
            heavy_shares = (
                " : ".join(
                    shares(heavy[side, expert], count[expert]) for side in range(2)
                )
                if count[expert]
                else "-"
            )
            row = [
                expert,
                f"{100 * count[expert] / len(ids):.2f}",
                f"{100 * val[expert] / total_val:.2f}" if total_val else "-",
                f"{means[0, expert]:.1f}/{means[1, expert]:.1f}",
                f"{means[2, expert]:.1f}/{means[3, expert]:.1f}",
                heavy_shares,
            ]
            if name == "tail":
                row.append(
                    shares(wings[:, expert], count[expert]) if count[expert] else "-"
                )
            rows.append(row)

        print(f"\n{name.upper()}: {n} experts, {len(ids):,} contexts", file=file)
        _table(headers, rows, file)
