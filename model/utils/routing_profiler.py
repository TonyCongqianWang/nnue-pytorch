"""Compact, matching ASCII profiles for the two joint-context expert banks."""

import sys

import numpy as np
import torch

from ..modules.router import KING_STATES, MOE_BANKS


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
    material = router.material_decode_table.cpu().numpy()
    print(
        "\nMoE8 routing profiles: LUT-weighted context means; IDs are training IDs. Export sorts by validation traffic.",
        file=file,
    )
    for name, table in zip(MOE_BANKS, tables):
        ids = table.cpu().numpy()
        n = len(router.codebooks[name])
        count = np.bincount(ids, minlength=n)
        index = np.arange(len(ids), dtype=np.int64)
        pair = index if name == "fc0" else index // KING_STATES
        own = material[pair // router.material_states]
        opp = material[pair % router.material_states]
        val = getattr(router, f"validation_counts_{name}").cpu().numpy()
        total_val = val.sum()
        features = [own[:, 0], opp[:, 0], own[:, 1], opp[:, 1], own[:, 3], opp[:, 3]]
        means = [
            np.bincount(ids, weights=f, minlength=n) / np.maximum(count, 1)
            for f in features
        ]
        kings = []
        if name == "tail":
            kings = [
                np.bincount(ids, weights=((index % 8) >> bit) & 1, minlength=n)
                / np.maximum(count, 1)
                for bit in [2, 1, 0]
            ]
        rows = []
        for e in np.argsort(-val, kind="stable"):
            king = [f"{100 * x[e]:.1f}" for x in kings] if kings else ["-", "-", "-"]
            rows.append(
                [
                    e,
                    f"{100 * count[e] / len(ids):.2f}",
                    f"{100 * val[e] / total_val:.2f}" if total_val else "-",
                    f"{means[0][e]:.1f}/{means[1][e]:.1f}",
                    f"{means[2][e]:.1f}/{means[3][e]:.1f}",
                    f"{100 * means[4][e]:.0f}/{100 * means[5][e]:.0f}",
                    *king,
                ]
            )
        print(f"\n{name.upper()}: {n} experts, {len(ids):,} contexts", file=file)
        _table(
            [
                "ID",
                "LUT %",
                "Val %",
                "T us/opp",
                "P us/opp",
                "Wide % u/o",
                "Adv %",
                "Ahead %",
                "Opposite %",
            ],
            rows,
            file,
        )
