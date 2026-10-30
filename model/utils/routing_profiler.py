import sys

import numpy as np
import torch

from ..modules.router import (
    BANK_SPECS,
    NUM_WEIGHT_EXPERTS,
    decompose_router_index,
)


def king_sq_name(k: int) -> str:
    """Returns human-readable oriented mirrored king square label."""
    if k == 32:
        return "e1[c]"
    r = k // 4
    f = k % 4
    file_map = ["a", "b", "c", "d"]
    return f"{file_map[f]}{r + 1}"


def enemy_king_sq_name(sq: int) -> str:
    """Returns human-readable enemy king square label."""
    if sq == 64:
        return "e8[c]"
    return f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}"


@torch.no_grad()
def _print_tail_profile(model, layer_stacks, codebook, shift, role, file=None) -> None:
    """Prints the compact Expert Routing Profile card."""
    if file is None:
        file = sys.stdout

    router = model.router

    lut = router.get_lookup_table()
    total_contexts = lut.numel()

    all_ctx = torch.arange(total_contexts, device=lut.device)
    mat_id, k_us, k_them = decompose_router_index(all_ctx)
    mat_feats = router.material_decode_table[mat_id]

    lut_np = ((lut >> shift) & 31).cpu().numpy()  # Extract the five-bit tail ID
    t_np = mat_feats[:, 0].cpu().numpy()  # Non-king pieces: 0..15
    q_np = (mat_feats[:, 2] == 0).cpu().numpy()  # Queen-first heavy category
    k_us_np = k_us.cpu().numpy()  # 0..32
    k_them_np = k_them.cpu().numpy()  # 0..64

    layers = (layer_stacks.l2, layer_stacks.output)
    delta_w_layers = [layer.get_delta_weights(codebook) for layer in layers]
    is_coalesced = getattr(layers[0], "is_coalesced", False)
    coalesced_deltas = []
    if is_coalesced:
        for layer in layers:
            w = layer.coalesced_weight.reshape(
                NUM_WEIGHT_EXPERTS, layer.out_features, -1
            )
            w_mean = w.mean(dim=0, keepdim=True)
            coalesced_deltas.append(w - w_mean)

    w_exp = 6
    w_ctx = 14
    w_pc = 14
    w_q = 7
    w_kus = 34
    w_kth = 22
    w_dw = 8

    print(f"\n{role} Weight Routing Profile ({total_contexts:,} contexts):", file=file)
    header = (
        f"{'Expert':^{w_exp}} | {'Contexts (%)':^{w_ctx}} | {'PC (mean/mode)':^{w_pc}} | "
        f"{'Queen %':^{w_q}} | {'Top King (Our)':^{w_kus}} | {'Top King (Enemy)':^{w_kth}} | "
        f"{'||ΔW||_F':^{w_dw}}"
    )
    sep = (
        f"{'-' * w_exp}-+-{'-' * w_ctx}-+-{'-' * w_pc}-+-{'-' * w_q}-+-"
        f"{'-' * w_kus}-+-{'-' * w_kth}-+-{'-' * w_dw}"
    )
    print(header, file=file)
    print(sep, file=file)

    for k in range(NUM_WEIGHT_EXPERTS):
        mask = lut_np == k
        count = int(np.sum(mask))
        pct = (count / float(total_contexts)) * 100.0

        if count > 0:
            pc_subset = t_np[mask]
            pc_mean = float(np.mean(pc_subset))
            pc_mode = int(np.bincount(pc_subset).argmax())
            queen_pct = float(np.mean(q_np[mask])) * 100.0

            # Always print top 3 friendly king squares
            k_counts = np.bincount(k_us_np[mask], minlength=33)
            top_ks = np.argsort(-k_counts)[:3]
            top_squares = [
                f"{king_sq_name(int(sq))} ({(k_counts[sq] / count) * 100:.0f}%)"
                for sq in top_ks
                if k_counts[sq] > 0
            ]
            sq_str = ", ".join(top_squares) if top_squares else "-"

            # Top 2 enemy king squares
            kth_counts = np.bincount(k_them_np[mask], minlength=65)
            top_kth = np.argsort(-kth_counts)[:2]
            top_enemy_squares = [
                f"{enemy_king_sq_name(int(sq))} ({(kth_counts[sq] / count) * 100:.0f}%)"
                for sq in top_kth
                if kth_counts[sq] > 0
            ]
            enemy_sq_str = ", ".join(top_enemy_squares) if top_enemy_squares else "-"
        else:
            pc_mean = 0.0
            pc_mode = 0
            queen_pct = 0.0
            sq_str = "-"
            enemy_sq_str = "-"

        # ||ΔW||_F (calculated from low-rank deltas if available, or deviation from mean if coalesced)
        if (
            delta_w_layers
            and float(sum(torch.norm(dw).item() for dw in delta_w_layers)) > 0
        ):
            norm_sq = sum(
                float(torch.norm(dw[k], p="fro") ** 2) for dw in delta_w_layers
            )
            dw_norm = float(np.sqrt(norm_sq))
        elif coalesced_deltas:
            norm_sq = sum(
                float(torch.norm(cd[k], p="fro") ** 2) for cd in coalesced_deltas
            )
            dw_norm = float(np.sqrt(norm_sq))
        else:
            dw_norm = 0.0

        exp_str = f" #{k:02d}  "
        ctx_str = f"{count:6,d} ({pct:4.1f}%)"
        pc_str = f"  {pc_mean:4.1f} / {pc_mode:2d}   "
        q_str = f"  {queen_pct:3.0f}% "
        kus_str = f"{sq_str:<{w_kus}}"
        kth_str = f"{enemy_sq_str:<{w_kth}}"
        dw_str = f"  {dw_norm:5.2f} "

        print(
            f"{exp_str} | {ctx_str} | {pc_str} | {q_str} | {kus_str} | {kth_str} | {dw_str}",
            file=file,
        )
    print(file=file)


@torch.no_grad()
def print_routing_profile(model, file=None):
    file = sys.stdout if file is None else file
    router = model.router
    lut = router.get_lookup_table()
    print(
        f"\nMoE7 training-ID routing profile before export sorting ({lut.numel():,} contexts):",
        file=file,
    )
    for spec in BANK_SPECS:
        count = len(router.codebooks[spec.name])
        ids = (
            router.fc0_lut.long()
            if spec.kind == "fc0"
            else (lut >> spec.shift) & spec.mask
        )
        counts = torch.bincount(ids, minlength=count).cpu()
        mass = counts.float() / counts.sum()
        effective = (-(mass * mass.clamp_min(1e-12).log()).sum()).exp().item()
        print(
            f"{spec.name}: active={int((counts > 0).sum())}/{count}, effective={effective:.1f}",
            file=file,
        )
        if spec.kind == "fc0":
            mat = router.material_decode_table.cpu().numpy()
            pair = np.arange(len(ids))
            for expert in range(count):
                selected = pair[ids.cpu().numpy() == expert]
                if len(selected):
                    us, them = mat[selected // len(mat)], mat[selected % len(mat)]
                    print(
                        f"  #{expert:02d}: {len(selected)} material pairs; mean T/P STM={us[:, :2].mean(0)}, OPP={them[:, :2].mean(0)}",
                        file=file,
                    )
        elif not spec.bias:
            bank = model.layer_stacks if spec.role == 0 else model.layer_stacks.opp
            _print_tail_profile(
                model,
                bank,
                router.normalized_codebook(spec.name),
                spec.shift,
                spec.name,
                file,
            )
