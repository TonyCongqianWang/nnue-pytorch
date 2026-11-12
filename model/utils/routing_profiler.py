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
    for name, table in zip(MOE_BANKS, tables):
        material = getattr(router, f"{name}_material_decode").cpu().numpy()
        states = router.material_states[name]
        ids = table.cpu().numpy()
        n = len(router.codebooks[name])
        count = np.bincount(ids, minlength=n)
        sums = np.zeros((4, n))
        geometry = np.zeros((TAIL_GEOMETRY_STATES, n))
        for start in range(0, len(ids), 65536):
            chunk = ids[start : start + 65536]
            index = np.arange(start, start + len(chunk))
            pair = index if name == "fc0" else index // TAIL_GEOMETRY_STATES
            own, opp = material[pair // states], material[pair % states]
            for i, values in enumerate((own[:, 0], opp[:, 0], own[:, 1], opp[:, 1])):
                sums[i] += np.bincount(chunk, weights=values, minlength=n)
            if name == "tail":
                group = index % TAIL_GEOMETRY_STATES
                geometry += np.bincount(
                    group * n + chunk, minlength=geometry.size
                ).reshape(geometry.shape)
        means = sums / np.maximum(count, 1)
        val = getattr(router, f"validation_counts_{name}").cpu().numpy()
        total_val = val.sum()
        rows = []
        for expert in np.argsort(-val, kind="stable"):
            wing = geometry.reshape(-1, KING_STATES, n).sum(1)[:, expert]
            rows.append(
                [
                    expert,
                    f"{100 * count[expert] / len(ids):.2f}",
                    f"{100 * val[expert] / total_val:.2f}" if total_val else "-",
                    f"{means[0, expert]:.1f}/{means[1, expert]:.1f}",
                    f"{means[2, expert]:.1f}/{means[3, expert]:.1f}",
                    "/".join(f"{100 * v / max(count[expert], 1):.0f}" for v in wing)
                    if name == "tail"
                    else "-",
                ]
            )
        print(f"\n{name.upper()}: {n} experts, {len(ids):,} contexts", file=file)
        _table(
            [
                "ID",
                "LUT %",
                "Val %",
                "T us/opp",
                "RelP us/opp" if name == "fc0" else "P us/opp",
                "Wing state % (0/1/2/3/4)",
            ],
            rows,
            file,
        )
