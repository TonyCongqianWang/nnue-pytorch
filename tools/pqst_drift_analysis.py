import os
import torch
import tyro
import itertools
from dataclasses import dataclass, field
from typing import Annotated, List, Dict

import model as M

@dataclass(frozen=True)
class CliConfig:
    files: Annotated[list[str], tyro.conf.Positional]
    """Arbitrary number of .nnue files to analyze."""

    psqt_size: int = 8
    """Number of output neurons allocated to psqt."""

    init_averages: int = 1
    """Number of times to initialize and average to find the expected origin."""

    nnue_lightning_config: tyro.conf.OmitArgPrefixes[M.NNUELightningConfig] = field(
        default_factory=M.NNUELightningConfig
    )

def extract_psqt_groups(nnue: M.NNUE, psqt_size: int) -> Dict[str, torch.Tensor]:
    input_module = nnue.model.input.features
    sub_keys = sorted([k for k in input_module._modules.keys() if k.isdigit()], key=int)

    groups = {}
    for k in sub_keys:
        w = getattr(input_module, k).weight.float()
        groups[k] = w[:, -psqt_size:]
    return groups

def spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1: return 1.0
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    if torch.norm(x_centered) == 0 or torch.norm(y_centered) == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(x_centered.unsqueeze(0), y_centered.unsqueeze(0)).item()

def print_table(title: str, headers: List[str], rows: List[List[str]]):
    print(f"\n>>> {title}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    fmt = " | ".join([f"{{:<{w}}}" for w in widths])
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*row))

def main():
    args = tyro.cli(CliConfig)
    if not args.files: return

    # 1. Compute expected origin per group
    print("Calculate expected init bias")
    group_init_sums = {}
    for _ in range(args.init_averages):
        dummy = M.NNUE(config=args.nnue_lightning_config, quantize_config=M.QuantizationConfig())
        groups = extract_psqt_groups(dummy, args.psqt_size)
        for k, v in groups.items():
            if k not in group_init_sums: group_init_sums[k] = v.detach()
            else: group_init_sums[k] += v

    expected_origins = {k: v / args.init_averages for k, v in group_init_sums.items()}
    ckpt_drifts = {}
    # 2. Extract and analyze each file
    print("Extract and analyze files")
    for file in args.files:
        if not file.endswith(".nnue"): continue
        with open(file, "rb") as f:
            nnue = M.NNUE(config=args.nnue_lightning_config, quantize_config=M.QuantizationConfig())
            reader = M.NNUEReader(f, args.nnue_lightning_config.features,
                                  config=args.nnue_lightning_config.model_config,
                                  quantize_config=M.QuantizationConfig())
            nnue.model = reader.model

        current_groups = extract_psqt_groups(nnue, args.psqt_size)
        drifts = {k: current_groups[k] - expected_origins[k] for k in current_groups}
        ckpt_drifts[file] = drifts

        mag_headers = ["Group", "Buck", "Init L2", "Drift L2", "Drift L1", "Drift Linf", "Sim Origin"]
        mag_rows = []

        # Track global stats per bucket
        for b in range(args.psqt_size):
            b_drift_all = torch.cat([drifts[k][:, b] for k in drifts], dim=0)
            b_orig_all = torch.cat([expected_origins[k][:, b] for k in expected_origins], dim=0)

            # Per-group breakdown
            for k in sorted(drifts.keys(), key=int):
                d_vec = drifts[k][:, b]
                o_vec = expected_origins[k][:, b]

                l2_orig = torch.norm(o_vec, p=2).item()
                l1 = torch.norm(d_vec, p=1).item()
                l2 = torch.norm(d_vec, p=2).item()
                linf = torch.norm(d_vec, p=float('inf')).item()
                sim = torch.nn.functional.cosine_similarity(d_vec.unsqueeze(0), o_vec.unsqueeze(0)).item() if torch.norm(d_vec) > 0 else 0.0
                mag_rows.append([k, b, f"{l2_orig:.4f}", f"{l2:.4f}", f"{l1:.4f}", f"{linf:.4f}", f"{sim:.4f}"])

            # Global row for this bucket
            g_l2_orig = torch.norm(b_orig_all, p=2).item()
            g_l1 = torch.norm(b_drift_all, p=1).item()
            g_l2 = torch.norm(b_drift_all, p=2).item()
            g_linf = torch.norm(b_drift_all, p=float('inf')).item()
            g_sim = torch.nn.functional.cosine_similarity(b_drift_all.unsqueeze(0), b_orig_all.unsqueeze(0)).item()
            mag_rows.append(["GLOBAL", b, f"{g_l2_orig:.4f}", f"{g_l2:.4f}", f"{g_l1:.4f}", f"{g_linf:.4f}", f"{g_sim:.4f}"])
            mag_rows.append(["-"*6] * len(mag_headers)) # Visual separator between buckets

        print_table(f"Magnitudes: {os.path.basename(file)}", mag_headers, mag_rows)

    # 3. Pairwise Directional Comparison
    if len(ckpt_drifts) > 1:
        for f1, f2 in itertools.combinations(list(ckpt_drifts.keys()), 2):
            headers = ["Group", "Buck", "Cosine", "Centered", "Spearman"]
            rows = []
            d1_map, d2_map = ckpt_drifts[f1], ckpt_drifts[f2]

            for b in range(args.psqt_size):
                v1_all = torch.cat([d1_map[k][:, b] for k in d1_map], dim=0)
                v2_all = torch.cat([d2_map[k][:, b] for k in d2_map], dim=0)

                for k in sorted(d1_map.keys(), key=int):
                    v1, v2 = d1_map[k][:, b], d2_map[k][:, b]
                    cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                    v1c, v2c = v1 - v1.mean(), v2 - v2.mean()
                    cen = torch.nn.functional.cosine_similarity(v1c.unsqueeze(0), v2c.unsqueeze(0)).item() if torch.norm(v1c) > 0 else 0.0
                    spr = spearman_rank_correlation(v1, v2)
                    rows.append([k, b, f"{cos:.4f}", f"{cen:.4f}", f"{spr:.4f}"])

                # Global row for pairwise
                g_cos = torch.nn.functional.cosine_similarity(v1_all.unsqueeze(0), v2_all.unsqueeze(0)).item()
                v1_all_c, v2_all_c = v1_all - v1_all.mean(), v2_all - v2_all.mean()
                g_cen = torch.nn.functional.cosine_similarity(v1_all_c.unsqueeze(0), v2_all_c.unsqueeze(0)).item()
                g_spr = spearman_rank_correlation(v1_all, v2_all)
                rows.append(["GLOBAL", b, f"{g_cos:.4f}", f"{g_cen:.4f}", f"{g_spr:.4f}"])
                rows.append(["-"*6] * len(headers))

            print_table(f"Pairwise: {os.path.basename(f1)} vs {os.path.basename(f2)}", headers, rows)

if __name__ == "__main__":
    main()