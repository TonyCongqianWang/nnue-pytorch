"""Measure post-filter loader traffic with recipe filtering parameters.

Run from the trainer root. This diagnostics uses K32H3 extraction for speed;
filtering and tree selection are identical to training. No optimizer is run.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader.config import DataloaderIOConfig, DataloaderSkipConfig
from data_loader.dataset import SparseBatchProvider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--positions", type=int, default=262144)
    ap.add_argument("binpacks", nargs="+")
    args = ap.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text())
    options = recipe["training"]["steps"][0]["run"]["other_options"]
    values = {k.replace("-", "_"): v for k, v in options.items()}
    skip = DataloaderSkipConfig(
        **{
            k: v
            for k, v in values.items()
            if k in DataloaderSkipConfig.__dataclass_fields__
        }
    )
    provider = SparseBatchProvider(
        "K32H3",
        args.binpacks,
        batch_size=4096,
        cyclic=True,
        num_workers=4,
        config=skip,
        io_config=DataloaderIOConfig(shuffle_buffer_entries=4096),
        device="cpu",
        use_pinned_memory=False,
        ls_keys="tree128",
    )
    counts = torch.zeros(128, dtype=torch.int64)
    total = 0
    while total < args.positions:
        batch = next(provider)
        counts += torch.bincount(batch[-1], minlength=128)
        total += len(batch[-1])
    delivered, tree = provider.get_source_traffic()
    report = {
        "binpacks": args.binpacks,
        "positions": total,
        "leaf_counts": counts.tolist(),
        "supported_leaves": int((counts > 0).sum()),
        "postfilter_tree_positions": tree,
        "postfilter_tree_fraction": tree / delivered,
        "filtering": skip.__dict__,
        "scope": "diagnostic sample after recipe filters; small shuffle window, no optimizer; not full recipe dataset mix",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{total} post-filter samples, {report['supported_leaves']}/128 leaves, tree fraction={report['postfilter_tree_fraction']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
