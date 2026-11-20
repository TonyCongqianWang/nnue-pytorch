"""Batch routing: gather FP32 scores once, solve on rank zero, broadcast biases."""

import torch
import torch.distributed as dist

ITERATIONS = 4
HISTOGRAM_BINS = 128


def column_quantiles(values, fractions):
    """Two-stage histograms resolve narrow expert margins without a full sort."""
    n, experts = values.shape
    bins = HISTOGRAM_BINS
    columns = torch.arange(experts, device=values.device)
    low = values.amin(0)
    width = (values.amax(0) - low).clamp_min(1e-12) / bins
    coarse = ((values - low) / width).long().clamp(0, bins - 1)
    keys = coarse + columns * bins
    histogram = (
        values.new_zeros(experts * bins)
        .scatter_add_(0, keys.flatten(), torch.ones_like(values).flatten())
        .view(experts, bins)
    )
    cumulative = histogram.cumsum(1)
    ranks = fractions * (n - 1) + 0.5
    selected = (cumulative >= ranks[:, None]).int().argmax(1)
    previous = torch.where(
        selected > 0, cumulative[columns, (selected - 1).clamp_min(0)], 0
    )
    base = low + selected * width
    fine_width = width / bins
    fine = ((values - base) / fine_width).long().clamp(0, bins - 1)
    fine_histogram = (
        values.new_zeros(experts * bins)
        .scatter_add_(
            0,
            (fine + columns * bins).flatten(),
            (coarse == selected[None]).to(values.dtype).flatten(),
        )
        .view(experts, bins)
    )
    fine_cumulative = fine_histogram.cumsum(1)
    residual = ranks - previous
    fine_selected = (fine_cumulative >= residual[:, None]).int().argmax(1)
    fine_previous = torch.where(
        fine_selected > 0, fine_cumulative[columns, (fine_selected - 1).clamp_min(0)], 0
    )
    fraction = (residual - fine_previous) / fine_histogram[
        columns, fine_selected
    ].clamp_min(1)
    return base + fine_width * (fine_selected + fraction)


def assignment_counts(scores, biases):
    winners = (scores + biases).argmax(1)
    return scores.new_zeros(scores.shape[1]).scatter_add_(
        0, winners, torch.ones_like(winners, dtype=scores.dtype)
    )


def routing_targets(initial, rank_target, ema, strength, max_strength, smoothing):
    """Detached KL-gradient weights: EMA sets strength, current traffic sets direction."""
    order = initial.argsort(descending=True, stable=True)
    power_target = torch.empty_like(rank_target)
    power_target[order] = rank_target
    smoothed = (1 - smoothing) * ema + smoothing / len(ema)
    # g / (1 + g), with g = max(q / smoothed - 1, 0).
    recovery = (1 - smoothed / power_target).clamp(0, 1)
    shrinkage = strength + (max_strength - strength) * recovery
    requested = initial + shrinkage * (power_target - initial)
    normalization = requested.sum()
    return requested / normalization, shrinkage, normalization


def solve_biases(
    scores,
    initial_biases,
    rank_target,
    ema,
    strength,
    max_strength,
    smoothing,
    max_range,
):
    """Four quantile sweeps with EMA-weighted targets held fixed during the solve."""
    before = assignment_counts(scores, initial_biases)
    initial = before / len(scores)
    if max_strength == 0:
        return torch.stack(
            (
                initial_biases,
                before,
                before,
                initial,
                torch.zeros_like(initial),
                torch.ones_like(initial),
            )
        )
    target, shrinkage, normalization = routing_targets(
        initial, rank_target, ema, strength, max_strength, smoothing
    )
    biases = initial_biases
    for _ in range(ITERATIONS):
        top = (scores + biases).topk(2, dim=1).values
        cutoff = (top[:, 0] + top[:, 1]) * 0.5
        biases = -column_quantiles(scores - cutoff[:, None], 1 - target)
        biases = biases - biases.mean()
        span = biases.max() - biases.min()
        biases = biases * (max_range / span.clamp_min(1e-12)).clamp(max=1)
    after = assignment_counts(scores, biases)
    return torch.stack(
        (biases, after, before, target, shrinkage, normalization.expand_as(target))
    )


# Lazy compilation is used only on the training GPU. CPU checks stay lightweight.
_compiled_solve = torch.compile(solve_biases, fullgraph=True)


@torch.compiler.disable
@torch.no_grad()
def gather_and_solve(
    scores, biases, rank_target, ema, strength, max_strength, smoothing, max_range
):
    """Equal local batches, as provided by the training pipeline; no host copies."""
    scores = scores.detach().float().contiguous()
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    experts = scores.shape[1]
    if world > 1:
        if rank == 0:
            global_scores = scores.new_empty((world * len(scores), experts))
            gathered = list(global_scores.view(world, *scores.shape).unbind(0))
        else:
            gathered = None
        dist.gather(scores, gather_list=gathered, dst=0)
    else:
        global_scores = scores
    if rank == 0:
        solver = _compiled_solve if scores.is_cuda else solve_biases
        result = solver(
            global_scores,
            biases,
            rank_target,
            ema,
            strength,
            max_strength,
            smoothing,
            max_range,
        )
    else:
        result = scores.new_empty((6, experts))
    if world > 1:
        dist.broadcast(result, src=0)
    return result, world
