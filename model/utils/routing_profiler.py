import sys

import numpy as np
import torch

from ..modules.router import decompose_router_index


def king_sq_name(k: int) -> str:
    """Returns human-readable oriented mirrored king square label."""
    if k == 32:
        return "e1[c]"
    r = k // 4
    f = k % 4
    file_map = ["a", "g", "c", "e"]
    return f"{file_map[f]}{r + 1}"


@torch.no_grad()
def print_routing_profile(model, file=None) -> None:
    """Prints the compact Expert Routing Profile card."""
    if file is None:
        file = sys.stdout

    router = getattr(model, "router", None)
    layer_stacks = getattr(model, "layer_stacks", None)
    if router is None:
        return

    lut = router.get_lookup_table()
    total_contexts = lut.numel()

    all_ctx = torch.arange(total_contexts, device=lut.device)
    p, q, k_us, _ = decompose_router_index(all_ctx)

    lut_np = lut.cpu().numpy()
    p_np = p.cpu().numpy() + 1  # 1..16
    q_np = q.cpu().numpy()  # 0 or 1
    k_us_np = k_us.cpu().numpy()  # 0..32

    # Get delta weights if layer_stacks has low-rank deltas
    delta_w_layers = []
    if layer_stacks is not None and hasattr(layer_stacks, "get_expert_deltas"):
        delta_w_layers = layer_stacks.get_expert_deltas(router.codebook)

    print("\nExpert Routing Profile (68,640 contexts):", file=file)
    print(
        f"{'Expert':^6} | {'Contexts (%)':^13} | {'PC (mean / mode)':^16} | {'Queen %':^7} | {'Top King Squares (>=10%)':^30} | {'||ΔW||_F':^8} | {'||c||':^5}",
        file=file,
    )
    print(
        f"{'-' * 6}-+-{'-' * 13}-+-{'-' * 16}-+-{'-' * 7}-+-{'-' * 30}-+-{'-' * 8}-+-{'-' * 5}",
        file=file,
    )

    for k in range(router.num_experts):
        mask = lut_np == k
        count = int(np.sum(mask))
        pct = (count / float(total_contexts)) * 100.0

        if count > 0:
            pc_subset = p_np[mask]
            pc_mean = float(np.mean(pc_subset))
            pc_mode = int(np.bincount(pc_subset).argmax())
            queen_pct = float(np.mean(q_np[mask])) * 100.0

            k_counts = np.bincount(k_us_np[mask], minlength=33)
            top_ks = np.argsort(-k_counts)
            top_squares = []
            for sq in top_ks:
                sq_pct = (k_counts[sq] / float(count)) * 100.0
                if sq_pct >= 10.0 and len(top_squares) < 3:
                    top_squares.append(f"{king_sq_name(int(sq))} ({sq_pct:.0f}%)")
            sq_str = ", ".join(top_squares) if top_squares else "-"
        else:
            pc_mean = 0.0
            pc_mode = 0
            queen_pct = 0.0
            sq_str = "-"

        # ||ΔW||_F
        if delta_w_layers:
            norm_sq = sum(
                float(torch.norm(dw[k], p="fro") ** 2) for dw in delta_w_layers
            )
            dw_norm = float(np.sqrt(norm_sq))
        else:
            dw_norm = 0.0

        # ||c||
        c_k = router.codebook[k]
        c_norm = float(torch.norm(c_k, p=2).item())

        print(
            f"  #{k:02d}  | {count:6,d} ({pct:4.1f}%) |   {pc_mean:4.1f} / {pc_mode:<2d}    |  {queen_pct:4.0f}%  | {sq_str:<30} |  {dw_norm:6.3f}  | {c_norm:4.2f}",
            file=file,
        )
    print(file=file)
