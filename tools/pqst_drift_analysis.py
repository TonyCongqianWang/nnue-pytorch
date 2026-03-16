import os
import torch
import tyro
import itertools
from dataclasses import dataclass, field
from typing import Annotated

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

def extract_psqt(nnue: M.NNUE, psqt_size: int) -> torch.Tensor:
    """Extracts the PSQT weights from the embedding layer."""
    w = nnue.model.input.weight.float()
    return w[:, -psqt_size:]

def spearman_rank_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Approximates Spearman's rho by computing the Pearson correlation of the ranks.
    """
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()

    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()

    if torch.norm(x_centered) == 0 or torch.norm(y_centered) == 0:
        return 0.0

    return torch.nn.functional.cosine_similarity(x_centered.unsqueeze(0), y_centered.unsqueeze(0)).item()

def main():
    args = tyro.cli(CliConfig)

    if not args.files:
        print("Error: No .nnue files provided.")
        return

    # 1. Compute the expected origin (biased mean of the initialization)
    init_sum = None
    for _ in range(args.init_averages):
        dummy_nnue = M.NNUE(
            config=args.nnue_lightning_config,
            quantize_config=M.QuantizationConfig(),
        )
        psqt = extract_psqt(dummy_nnue, args.psqt_size)
        if init_sum is None:
            init_sum = psqt
        else:
            init_sum += psqt

    expected_origin = init_sum / args.init_averages

    checkpoints_drift = {}

    # 2. Extract weights and compute drift magnitudes
    for file in args.files:
        if not file.endswith(".nnue"):
            print(f"Skipping {file}: Target is not a .nnue file.")
            continue

        with open(file, "rb") as f:
            nnue = M.NNUE(
                config=args.nnue_lightning_config,
                quantize_config=M.QuantizationConfig(),
            )
            reader = M.NNUEReader(
                f,
                args.nnue_lightning_config.features,
                config=args.nnue_lightning_config.model_config,
                quantize_config=M.QuantizationConfig(),
            )
            nnue.model = reader.model

        ckpt_psqt = extract_psqt(nnue, args.psqt_size)

        # Total drift from the expected origin
        drift = ckpt_psqt - expected_origin
        checkpoints_drift[file] = drift

        print(f"--- Drift Magnitudes for: {os.path.basename(file)} ---")
        l1_norm = torch.norm(drift, p=1).item()
        l2_norm = torch.norm(drift, p=2).item()
        linf_norm = torch.norm(drift, p=float('inf')).item()

        print(f"Overall L1:   {l1_norm:.4f}")
        print(f"Overall L2:   {l2_norm:.4f}")
        print(f"Overall Linf: {linf_norm:.4f}\n")

    # 3. Compute pairwise drift directions per bucket
    if len(checkpoints_drift) > 1:
        print("--- Pairwise Drift Directions (Per Bucket) ---")
        file_keys = list(checkpoints_drift.keys())

        for f1, f2 in itertools.combinations(file_keys, 2):
            print(f"\nComparing Drift: {os.path.basename(f1)} vs {os.path.basename(f2)}")
            drift1 = checkpoints_drift[f1]
            drift2 = checkpoints_drift[f2]

            print(f"{'Bucket':<8} | {'Cosine':<10} | {'Centered':<10} | {'Spearman':<10}")
            print("-" * 45)

            for bucket in range(args.psqt_size):
                v1 = drift1[:, bucket]
                v2 = drift2[:, bucket]

                if torch.norm(v1) == 0 or torch.norm(v2) == 0:
                    cos_sim, centered_cos, spearman = 0.0, 0.0, 0.0
                else:
                    cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

                    v1_centered = v1 - v1.mean()
                    v2_centered = v2 - v2.mean()
                    if torch.norm(v1_centered) == 0 or torch.norm(v2_centered) == 0:
                        centered_cos = 0.0
                    else:
                        centered_cos = torch.nn.functional.cosine_similarity(v1_centered.unsqueeze(0), v2_centered.unsqueeze(0)).item()

                    spearman = spearman_rank_correlation(v1, v2)

                print(f"Bucket {bucket:<1} | {cos_sim:>10.4f} | {centered_cos:>10.4f} | {spearman:>10.4f}")
    else:
        print("Requires at least two valid .nnue files to compute pairwise directions.")

if __name__ == "__main__":
    main()