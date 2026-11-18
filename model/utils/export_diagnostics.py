"""Small checkpoint diagnostics for the exported expert banks."""

import json
from pathlib import Path

import torch

from ..modules.router import HEAVY_CATEGORIES, MOE_BANKS


def _spectrum(matrix):
    """Rows are the items whose shared mean is removed."""
    matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    centered = matrix - matrix.mean(0)
    # The Gram matrix has the squared singular values and is small even for FC0.
    gram = (
        centered @ centered.T
        if len(centered) <= centered.shape[1]
        else centered.T @ centered
    )
    energy = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = energy.sum()
    mass = energy / total.clamp_min(1e-30)
    return {
        "shape": list(matrix.shape),
        "rms": float(matrix.square().mean().sqrt()),
        "effective_rank": float((-mass * mass.clamp_min(1e-30).log()).sum().exp())
        if total
        else 0.0,
        "top_one_share": float(mass[0]),
        "top_five_share": float(mass[:5].sum()),
        "centered_energy_fraction": float(
            total / matrix.square().sum().clamp_min(1e-30)
        ),
        "singular_energy": energy.tolist(),
    }


def _print_spectra(rows):
    print(
        f"{'Tensor / subset':<35} {'Rank':>7} {'RMS':>11} {'Centered RMS %':>15} {'Top 1 %':>9} {'Top 5 %':>9}"
    )
    for name, stats in rows:
        print(
            f"{name:<35} {stats['effective_rank']:7.2f} {stats['rms']:11.4g} "
            f"{100 * stats['centered_energy_fraction'] ** 0.5:15.2f} "
            f"{100 * stats['top_one_share']:9.2f} {100 * stats['top_five_share']:9.2f}"
        )


@torch.no_grad()
def write_export_diagnostics(model, path):
    router = model.router
    layers = model.layer_stacks
    if layers.l1.is_coalesced:
        raise ValueError("Export diagnostics require an uncoalesced checkpoint")

    report = {"projections": {}, "routers": {}, "generators": {}}
    names = ("FC0", "FC1", "readout")
    print("[Export diagnostics] Weight expansions centered along the adapter axis")
    if router.shared_routing:
        report["projections"]["shared context down"] = _spectrum(router.adapter_down)
    for name, layer in zip(names, layers.dense_layers()):
        if layer.is_direct:
            continue
        report["projections"][f"{name} up transpose"] = _spectrum(layer.adapter_up.T)
    _print_spectra(report["projections"].items())

    orders = {}
    for bank in router.active_banks:
        counts = getattr(router, f"validation_counts_{bank}").cpu()
        orders[bank] = torch.argsort(counts, descending=True, stable=True)
        n = len(counts)
        total = int(counts.sum())
        values = counts.double() / max(total, 1)
        zero = int((counts == 0).sum())
        minimum, median, maximum = (
            float(values.min()),
            float(values.quantile(0.5)),
            float(values.max()),
        )
        print(
            f"[Export traffic {router.bank_label(bank)}] positions={total:,} active={n - zero}/{n} "
            f"zero={zero} min/median/max={minimum:.6g}/{median:.6g}/{maximum:.6g}"
        )
        report["routers"][router.bank_label(bank)] = {
            "total": total,
            "counts": counts.tolist(),
            "export_to_training_ids": orders[bank].tolist(),
            "balance_power": router.balance_power,
            "balance_rank_targets": getattr(router, f"balance_rank_target_{bank}")
            .cpu()
            .tolist(),
            "commitment_cost": router.commitment_cost,
            "commitment_context_ratio": router.commitment_context_ratio,
            "zero": zero,
            "heavy_categories": HEAVY_CATEGORIES,
            "heavy_counts_stm_opp": getattr(router, f"validation_heavy_{bank}")
            .cpu()
            .tolist(),
        }
        if total:
            report["routers"][router.bank_label(bank)].update(
                bottom_decile_share=float(
                    values.sort().values[: max(1, (n + 9) // 10)].sum()
                ),
                below_one_percent_uniform=int((n * values < 0.01).sum()),
                below_ten_percent_uniform=int((n * values < 0.1).sum()),
            )

    print("[Export diagnostics] Shared adapter codebooks and centered expert weights")
    codebooks = dict(zip(MOE_BANKS, router.codebook))
    for bank in router.active_banks:
        vectors = codebooks[bank]
        report["routers"][router.bank_label(bank)]["codebook"] = _spectrum(vectors)
    _print_spectra(
        (
            f"{router.bank_label(bank)} codebook",
            report["routers"][router.bank_label(bank)]["codebook"],
        )
        for bank in router.active_banks
    )
    fc0_bank = "tail" if router.shared_routing else "fc0"
    for name, bank, layer in zip(
        names, (fc0_bank, "tail", "tail"), layers.dense_layers()
    ):
        vectors = codebooks[bank]
        weight, bias = layer.get_all_weights_and_biases(vectors)
        clipped = model.quantization.clip_ls_weight(weight, layer.layer_key)
        _, quantized = model.quantization.quantize_fc_layer(
            bias, clipped, layer.layer_key
        )
        matrices = (
            weight.reshape(layer.count, -1).cpu(),
            quantized.reshape(layer.count, -1).cpu(),
        )
        entries = {"adapter_router": router.bank_label(bank), "weights": {}}
        report["generators"][name] = entries
        rows = []
        traffic = report["routers"][router.bank_label(bank)]
        for label, k in (
            ("all", layer.count),
            ("top 25%", layer.count // 4),
            ("top 50%", layer.count // 2),
        ):
            if label != "all" and (k < 2 or not traffic["total"]):
                continue
            ids = orders[bank][:k]
            share = sum(traffic["counts"][i] for i in ids.tolist()) / max(
                traffic["total"], 1
            )
            stats = {
                kind: _spectrum(matrix[ids])
                for kind, matrix in zip(("float", "int8"), matrices)
            }
            entries["weights"][label] = {
                "expert_ids": ids.tolist(),
                "traffic_share": share,
                **stats,
            }
            print(
                f"{name} {label}: {k} experts, {100 * share:.2f}% of validation traffic"
            )
            rows.extend(
                (f"{name} {label} {kind}", value) for kind, value in stats.items()
            )
        _print_spectra(rows)

    for bank, stats in report["routers"].items():
        if stats["total"]:
            print(
                f"[Export low traffic {bank}] bottom decile={100 * stats['bottom_decile_share']:.3g}% "
                f"below 1% uniform={stats['below_one_percent_uniform']} "
                f"below 10% uniform={stats['below_ten_percent_uniform']}"
            )
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > 20_000_000:
        raise ValueError("Export diagnostics exceed 20 MB")
    Path(path).write_bytes(payload)
    print(f"Wrote {path} ({len(payload):,} bytes)")
