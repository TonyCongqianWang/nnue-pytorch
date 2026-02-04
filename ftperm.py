"""

NOTE: This script uses CUDA and may require large amounts of VRAM. Decrease --count if encountering problems.

Example use:

1. Generate the activation matrix for some sample dataset.

python ftperm.py gather --data=data\fishpack32.binpack --net=networks\nn-5af11540bbfe.nnue --count=1000000 --features=HalfKAv2_hm --out ftact1m.npy

python ftperm.py gather --data=noob_master_leaf_static_d12_85M_0.binpack --net=nn-5af11540bbfe.nnue --count=10000 --features=HalfKAv2_hm --out ftact1m.npy

2. Find a permutation

python ftperm.py find_perm --data=ftact1m.npy --out=ftact.perm

3. Test the permutation against the baseline

python ftperm.py eval_perm --data=ftact1m.npy --perm=ftact.perm

4. Apply permutation and save
python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_perm=ftact.perm

----------------------------------------------------------------

OR do the whole process in one step

python serialize.py networks\nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_optimize --ft_optimize_data=data\fishpack32.binpack --ft_optimize_count=1000000

python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_optimize --ft_optimize_data=noob_master_leaf_static_d12_85M_0.binpack --ft_optimize_count=10000

"""

import argparse
import copy
from dataclasses import dataclass
import itertools
import time
from typing import Callable, Generator, TypeAlias

import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

import chess
import cupy as cp
import numpy as np
import numpy.typing as npt
import torch

import data_loader
import model as M
from model import (
    FeatureSet,
    NNUE,
    NNUEModel,
    NNUEReader,
    ModelConfig,
    QuantizationConfig,
)


"""

Algorithm by Daniel Monroe. Github @Ergodice.

"""

ZERO_BLOCK_SIZE = 4
VERBOSE = False


def batched(arr: npt.NDArray, batch_size: int) -> Generator[npt.NDArray, None, None]:
    """
    Utility generator that yields chunks of array `arr` of size `batch_size`
    Expects arr to be a numpy-like array
    """
    n_samples = arr.shape[0]
    idx = 0
    while idx < n_samples:
        yield arr[idx : min(idx + batch_size, n_samples)]
        idx += batch_size


def apply_swap(perm: npt.NDArray, i: int, j: int) -> None:
    """
    Swap `i`-th and `j`-th elements in the array `perm`.
    """
    perm[i], perm[j] = perm[j], perm[i]


def apply_rotate_right(perm: npt.NDArray, indices: tuple[int, ...]) -> None:
    """
    Rotates right the values in `perm` at selected indices `indices`.
    The rotation is performed as-if the selected indices were layed out in the order
    specified in the `indices` list.
    """
    values = [perm[i] for i in indices]
    new_values = [values[-1]] + values[:-1]
    for i, j in zip(indices, new_values):
        perm[i] = j


def get_swapped_zero_positive_count(
    actmat_flat: npt.NDArray[np.bool_], use_cupy: bool = True
) -> int:
    if use_cupy:
        actmat_flat = cp.asarray(actmat_flat, dtype=cp.int8)

    shape = actmat_flat.shape
    # Group into blocks that are processed at once during inference
    # actmat is a boolean matrix of shape (N, L1 // 2) with "True" meaning 0
    actmat_chunked = actmat_flat.reshape(
        (actmat_flat.shape[0], actmat_flat.shape[1] // ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
    )

    if use_cupy:
        # Calculate number of zeros in each block
        num_zeros = cp.sum(actmat_chunked, axis=2, keepdims=True)
        # Broadcast back to the same shape as actmat_chunked so it's easier to work with
        num_zeros = cp.tile(num_zeros, (1, 1, ZERO_BLOCK_SIZE))

        # Marks an element if all other elements in a block are zero.
        #
        # Example:
        #                                   b  i   k      b  i   k      b  i   k
        # slice                            [0, 13, :]    [0, 14, :]    [0, 15, :]
        # num_zeros           = [... [... [3, 3, 3, 3], [1, 1, 1, 1], [4, 4, 4, 4] ...] ...]
        # actmat_chunked      = [... [... [1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1] ...] ...]
        # rest_zero_indicator = [... [... [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1] ...] ...]
        #
        rest_zero_indicator = (
            (num_zeros - actmat_chunked == ZERO_BLOCK_SIZE - 1)
            .reshape(shape)
            .astype(cp.int8)
        )

        # Sum all possible pairs of elements in a single sample of actmat_flat and rest_zero_indicator.
        # Aggregate sum over the whole batch.
        # This tells us how much "good" a swap of i-th and j-th slices would do. It doesn't consider
        # how much "bad" it would do though, that will be accounted for later, for performance reasons.
        swapped_zero_count = cp.einsum(
            "bi,bj->ij", actmat_flat, rest_zero_indicator, dtype=int
        )

    else:
        # Same operation but with numpy
        num_zeros = np.sum(actmat_chunked, axis=2, keepdims=True)
        num_zeros = np.tile(num_zeros, (1, 1, ZERO_BLOCK_SIZE))

        rest_zero_indicator = (
            (num_zeros - actmat_chunked == ZERO_BLOCK_SIZE - 1)
            .reshape(shape)
            .astype(int)
        )

        swapped_zero_count = np.einsum("bi,bj->ij", actmat_flat, rest_zero_indicator)

    return swapped_zero_count


def get_swapped_zero_increase(
    actmat: npt.NDArray[np.bool_], use_cupy: bool = True
) -> npt.NDArray[np.int_]:
    n_neurons = actmat.shape[1]
    swapped_zero_count = 0

    # Process in batches since the arrays are too large
    # TODO: Find a good batch size. Try lowest as possible as VRAM is an issue on low end devices.
    BATCH_SIZE = 10000
    for actmat_batch in batched(actmat, BATCH_SIZE):
        swapped_zero_count += get_swapped_zero_positive_count(
            actmat_batch, use_cupy=use_cupy
        )

    # (L1/2) x (L1/2)
    if use_cupy:
        # Subtract from each i-th slice the positive value of the current i-th placement.
        # This is the place where we account for how much "bad" it would do.
        # It is done here because we process earlier in batches, but this operation is distributive,
        # so it needs to only be done once at the end.
        swapped_zero_increase = swapped_zero_count - cp.reshape(
            cp.diag(swapped_zero_count), (1, n_neurons)
        )
        swapped_zero_increase = cp.asnumpy(swapped_zero_increase)

    else:
        swapped_zero_increase = swapped_zero_count - np.reshape(
            np.diag(swapped_zero_count), (1, n_neurons)
        )

    return swapped_zero_increase


def get_score_change(
    actmat: npt.NDArray[np.bool_], use_cupy: bool = True
) -> npt.NDArray[np.int_]:
    # actmat is a boolean matrix of shape (N, L1) with "True" meaning 0

    n_neurons = actmat.shape[1]

    score_change = get_swapped_zero_increase(actmat, use_cupy)

    # Kill off swaps between neurons in the same block
    blocks = np.arange(n_neurons).reshape((n_neurons, 1)) // ZERO_BLOCK_SIZE
    same_block_killer = 1 - (blocks == blocks.T).astype(int)
    score_change = score_change * same_block_killer
    return score_change

@dataclass
class SwapResult:
    swaps: list[tuple[int, ...]]
    scores: list[float]
    score_change: float

SwapFunction: TypeAlias = Callable[[npt.NDArray[np.bool_], bool], SwapResult]

def evaluate_swaps_on_batch(
    actmat: npt.NDArray[np.bool_], 
    swaps: list[tuple[int, ...]], 
    use_cupy: bool = True
) -> list[float]:
    """
    Calculates the score change for a specific list of swaps/cycles on a given batch.
    """
    if not swaps:
        return []

    score_changes = get_score_change(actmat, use_cupy=use_cupy)
    
    if use_cupy:
        score_changes = cp.asnumpy(score_changes)

    cycle_len = len(swaps[0])
    scores = []

    if cycle_len == 2:
        for idx in swaps:
            i, j = idx
            # gain(i, j) = score_change[i, j] + score_change[j, i]
            gain = score_changes[i, j] + score_changes[j, i]
            scores.append(gain)
            
    elif cycle_len == 3:
        for idx in swaps:
            i, j, k = idx
            gain = score_changes[i, j] + score_changes[j, k] + score_changes[k, i]
            scores.append(gain)
            
    else:
        # Fallback for generic n-cycles
        for cycle in swaps:
            gain = 0
            for k in range(len(cycle)):
                u = cycle[k]
                v = cycle[(k + 1) % len(cycle)]
                gain += score_changes[u, v]
            scores.append(gain)

    return scores


def make_swaps_2(actmat: npt.NDArray[np.bool_], use_cupy: bool = True) -> SwapResult:
    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    score_change = get_score_change(actmat, use_cupy=use_cupy)
    if use_cupy:
        score_change = cp.asnumpy(score_change)
    
    score_change = score_change + score_change.T

    # Reshape and Transpose to (Nb, Nb, 4, 4)
    # 1. Reshape to (Nb, 4, Nb, 4)
    block_view = score_change.reshape(n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE)
    # 2. Transpose to (Nb, Nb, 4, 4)
    block_view_t = block_view.transpose(0, 2, 1, 3)
    # 3. Flatten inner dims to 16
    flat_local = block_view_t.reshape(n_blocks, n_blocks, -1)
    
    max_gains = np.max(flat_local, axis=2)
    max_indices = np.argmax(flat_local, axis=2)

    G = nx.Graph()
    G.add_nodes_from(range(n_blocks))
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            weight = max_gains[i, j]
            if weight > 1e-9:
                G.add_edge(i, j, weight=weight)

    matching = nx.max_weight_matching(G, maxcardinality=False)

    swaps = []
    scores = []
    total_score_change = 0.0

    for u, v in matching:
        gain = max_gains[u, v]
        total_score_change += gain
        
        flat_idx = max_indices[u, v]
        u_local = flat_idx // ZERO_BLOCK_SIZE
        v_local = flat_idx % ZERO_BLOCK_SIZE
        
        neuron_i = u * ZERO_BLOCK_SIZE + u_local
        neuron_j = v * ZERO_BLOCK_SIZE + v_local
        
        swaps.append((neuron_i, neuron_j))
        scores.append(gain)

    total_improvement = (
        total_score_change / n_samples / (n_neurons // ZERO_BLOCK_SIZE) * 100
    )
    
    return SwapResult(swaps, scores, total_improvement)


def make_swaps_3(actmat: npt.NDArray[np.bool_], use_cupy: bool = True) -> SwapResult:
    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    score_changes = get_score_change(actmat, use_cupy=use_cupy)
    score_changes = (
        score_changes[:, :, None]
        + score_changes[None, :, :]
        + (score_changes.T)[:, None, :]
    )
    if use_cupy:
        score_changes = cp.asnumpy(score_changes)

    # Reshape and Transpose to (Nb, Nb, Nb, 4, 4, 4)
    block_view = score_changes.reshape(
        n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE
    )
    block_view_t = block_view.transpose(0, 2, 4, 1, 3, 5)
    flat_local = block_view_t.reshape(n_blocks, n_blocks, n_blocks, -1)
    
    max_gains = np.max(flat_local, axis=3)
    max_indices_flat = np.argmax(flat_local, axis=3)

    candidates = []
    it = np.nditer(max_gains, flags=['multi_index'])
    for gain in it:
        if gain <= 1e-9: continue
        i, j, k = it.multi_index
        if i == j or j == k or i == k: continue
        if i > j or i > k: continue
        candidates.append((gain, i, j, k, max_indices_flat[i, j, k]))

    if not candidates:
        return SwapResult([], [], 0.0)

    # MIP Solver
    num_vars = len(candidates)
    c_obj = -np.array([cand[0] for cand in candidates])
    row_ind, col_ind, data = [], [], []
    
    for c_idx, (_, b1, b2, b3, _) in enumerate(candidates):
        for b in [b1, b2, b3]:
            row_ind.append(b)
            col_ind.append(c_idx)
            data.append(1)
            
    A_eq = coo_matrix((data, (row_ind, col_ind)), shape=(n_blocks, num_vars))
    constraints = LinearConstraint(A_eq, -np.inf, 1)
    bounds = Bounds(0, 1)
    integrality = np.ones(num_vars)
    
    # Hide scipy output
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
    
    if not res.success:
        return SwapResult([], [], 0.0)
        
    selected_indices = np.where(res.x > 0.5)[0]
    cycles = []
    scores = []
    total_score_change = 0.0
    
    for idx in selected_indices:
        gain, b_i, b_j, b_k, local_flat = candidates[idx]
        total_score_change += gain
        
        u_local = local_flat // 16
        rem = local_flat % 16
        v_local = rem // 4
        w_local = rem % 4
        
        neuron_i = b_i * ZERO_BLOCK_SIZE + u_local
        neuron_j = b_j * ZERO_BLOCK_SIZE + v_local
        neuron_k = b_k * ZERO_BLOCK_SIZE + w_local
        
        cycles.append((neuron_i, neuron_j, neuron_k))
        scores.append(gain)

    total_improvement = total_score_change / n_samples / (n_neurons // 4) * 100
    return SwapResult(cycles, scores, total_improvement)

def make_swaps_2_3(actmat: npt.NDArray[np.bool_], use_cupy: bool = True) -> SwapResult:
    """
    Jointly optimizes for both 2-swaps and 3-cycles using a single MIP formulation.
    This allows the solver to trade off between a 2-swap and a 3-cycle that might share a block.
    """
    start_time = time.time()
    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    # 1. Compute Base Score Matrix
    score_change = get_score_change(actmat, use_cupy=use_cupy)
    
    # --- Part A: Gather 2-Swap Candidates ---
    # Symmeterize for 2-swaps: gain(i,j) = S[i,j] + S[j,i]
    score_change_sym = score_change + score_change.T
    
    if use_cupy:
        score_change_np = cp.asnumpy(score_change)
        score_change_sym_np = cp.asnumpy(score_change_sym)
    else:
        score_change_np = score_change
        score_change_sym_np = score_change_sym

    # Reshape for blocks: (Nb, 4, Nb, 4) -> (Nb, Nb, 4, 4)
    # Note: We must transpose to align block dims
    block_view_2 = score_change_sym_np.reshape(n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE)
    block_view_2_t = block_view_2.transpose(0, 2, 1, 3) # (Nb, Nb, 4, 4)
    flat_local_2 = block_view_2_t.reshape(n_blocks, n_blocks, -1)
    
    max_gains_2 = np.max(flat_local_2, axis=2)
    max_indices_2 = np.argmax(flat_local_2, axis=2)

    candidates = [] # List of (gain, [blocks...], local_flat_index, type)

    # Collect 2-swap candidates (Upper triangle)
    # Using a slightly lower threshold for joint optimization to allow competition
    rows, cols = np.where(np.triu(max_gains_2, k=1) > 1e-9)
    for b_i, b_j in zip(rows, cols):
        gain = max_gains_2[b_i, b_j]
        candidates.append((gain, [b_i, b_j], max_indices_2[b_i, b_j], 2))

    # --- Part B: Gather 3-Cycle Candidates ---
    # Score for cycles: S[i,j] + S[j,k] + S[k,i]
    # We construct the tensor on the CPU (numpy) because it's 3D block view
    # Broadcast sum: (Nb, Nb, Nb)
    # We do this calculation on the block-maxed values to save memory/compute
    
    # First, max over local neurons for directed edges: Max(S[i,j]) for blocks B_i, B_j
    # shape: (Nb, Nb)
    block_directed_max = np.max(
        score_change_np.reshape(n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(n_blocks, n_blocks, -1),
        axis=2
    )
    
    # Also keep the indices: (Nb, Nb) -> value 0..15
    block_directed_argmax = np.argmax(
        score_change_np.reshape(n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(n_blocks, n_blocks, -1),
        axis=2
    )

    # 3-cycle gain for blocks i,j,k = Max(i->j) + Max(j->k) + Max(k->i)
    # Note: This is an UPPER BOUND approximation. The specific neurons maximizing i->j 
    # might not be the same starting neurons for k->i. 
    # HOWEVER, since we optimize disjoint blocks, and we are permuting *within* blocks,
    # we can pick the specific neuron permutation for the block to satisfy the cycle 
    # IF the cycle is disjoint.
    # Actually, for the "Find Permutation" task, we are moving specific neurons.
    # So we strictly need (u, v, w) such that u->v->w->u.
    # The previous exact solver did this correctly by tensor expansion.
    # Let's stick to the correct tensor expansion (it fits in memory for 3-cycles).
    
    score_changes_3 = (
        score_change_np[:, :, None]
        + score_change_np[None, :, :]
        + (score_change_np.T)[:, None, :]
    )
    
    block_view_3 = score_changes_3.reshape(
        n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE
    )
    block_view_3_t = block_view_3.transpose(0, 2, 4, 1, 3, 5) # (Nb, Nb, Nb, 4, 4, 4)
    flat_local_3 = block_view_3_t.reshape(n_blocks, n_blocks, n_blocks, -1)
    
    max_gains_3 = np.max(flat_local_3, axis=3)
    max_indices_3 = np.argmax(flat_local_3, axis=3)

    # Iterate 3-cycle candidates
    # We use iterator for efficiency
    it = np.nditer(max_gains_3, flags=['multi_index'])
    for gain in it:
        if gain <= 1e-9: continue
        i, j, k = it.multi_index
        
        # Constraints
        if i == j or j == k or i == k: continue # Distinct
        if i > j or i > k: continue # Canonical order
        
        candidates.append((gain, [i, j, k], max_indices_3[i, j, k], 3))

    if not candidates:
        return SwapResult([], [], 0.0)

    # --- Part C: Joint MIP Optimization ---
    
    num_vars = len(candidates)
    c_obj = -np.array([cand[0] for cand in candidates]) # Minimize negative gain
    
    row_ind, col_ind, data = [], [], []
    
    for c_idx, (_, blocks, _, _) in enumerate(candidates):
        for b in blocks:
            row_ind.append(b)
            col_ind.append(c_idx)
            data.append(1)
            
    A_eq = coo_matrix((data, (row_ind, col_ind)), shape=(n_blocks, num_vars))
    constraints = LinearConstraint(A_eq, -np.inf, 1) # Each block used at most once
    bounds = Bounds(0, 1)
    integrality = np.ones(num_vars)
    
    # Solve
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
    
    if not res.success:
        return SwapResult([], [], 0.0)
        
    # --- Part D: Reconstruct ---
    selected_indices = np.where(res.x > 0.5)[0]
    
    final_swaps = []
    final_scores = []
    total_gain = 0.0
    
    for idx in selected_indices:
        gain, blocks, local_flat, c_type = candidates[idx]
        total_gain += gain
        final_scores.append(gain)
        
        if c_type == 2:
            # Reconstruct 2-swap
            b_i, b_j = blocks
            # local_flat is index in 4x4=16
            u_local = local_flat // ZERO_BLOCK_SIZE
            v_local = local_flat % ZERO_BLOCK_SIZE
            
            n_i = b_i * ZERO_BLOCK_SIZE + u_local
            n_j = b_j * ZERO_BLOCK_SIZE + v_local
            final_swaps.append((n_i, n_j))
            
        elif c_type == 3:
            # Reconstruct 3-cycle
            b_i, b_j, b_k = blocks
            # local_flat is index in 4x4x4=64
            u_local = local_flat // 16
            rem = local_flat % 16
            v_local = rem // 4
            w_local = rem % 4
            
            n_i = b_i * ZERO_BLOCK_SIZE + u_local
            n_j = b_j * ZERO_BLOCK_SIZE + v_local
            n_k = b_k * ZERO_BLOCK_SIZE + w_local
            final_swaps.append((n_i, n_j, n_k))

    total_improvement = total_gain / n_samples / (n_neurons // 4) * 100
    
    return SwapResult(final_swaps, final_scores, total_improvement)

def get_best_cycles_n(
    actmat: npt.NDArray[np.bool_], n: int, use_cupy: bool = True
) -> SwapResult:
    """
    General solver for n-cycles (n >= 2).
    Uses iterative path extension to efficiently compute the gain of all block-cycles
    without exploding memory, then solves the packing problem using MIP.
    """
    print(f"Starting get_best_cycles_n (n={n})")
    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    # 1. Compute Base Score Matrix (Neurons x Neurons)
    score_change = get_score_change(actmat, use_cupy=use_cupy)
    
    # Move to numpy for complex indexing/looping
    if use_cupy:
        score_change = cp.asnumpy(score_change)

    # 2. Create Block Interaction View
    # Shape: (Nb, 4, Nb, 4)
    # block_view[i, u, j, v] = gain of moving neuron u (block i) to pos v (block j)
    block_view = score_change.reshape(
        n_blocks, ZERO_BLOCK_SIZE, n_blocks, ZERO_BLOCK_SIZE
    )
    
    # 3. Iteratively compute max path gains
    # We want to compute:
    # Path[b1, ..., bk, u_start, u_end] = max gain of path b1->...->bk
    # connecting neuron u_start (in b1) to u_end (in bk)
    
    # Start with length 1 paths (edges)
    # Shape: (Nb, Nb, 4, 4)
    # Transpose to: (Nb, Nb, 4, 4) -> (Nb, Nb, 4, 4) (already correct structure)
    current_paths = block_view.transpose(0, 2, 1, 3) 
    
    # To reconstruct the best cycle later, we need to store which neurons were used.
    # Because n is small, we can just re-evaluate the specific cycle candidates 
    # at the end, rather than storing a massive pointer tensor.
    
    # Iterate to extend paths up to length n-1
    # We stop at n-1 because the last step is closing the cycle (k -> start)
    for step in range(n - 2):
        print(f"  Extending paths to length {step + 2}...")
        
        # Current: (Nb_1, ..., Nb_k, u_start, u_curr)
        # Next Edge: (Nb_k, Nb_{k+1}, u_curr, u_next)
        # Target: (Nb_1, ..., Nb_{k+1}, u_start, u_next)
        
        # We perform a "Max-Plus" matrix multiplication over the `u_curr` dimension
        # and broadcasting over the block dimensions.
        
        # 1. Expand dims for broadcasting
        # Current: (..., Nb_k, 1,      u_start, u_curr, 1)
        # Next:    (..., 1,    Nb_k+1, 1,       u_curr, u_next)
        
        s_curr = current_paths.shape
        nb_dims = s_curr[:-2] # (Nb, ..., Nb)
        
        # Reshape for broadcast
        # A: (Nb..., Nb_last, 1, 4, 4, 1)
        A = current_paths.reshape(*nb_dims, 1, ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE, 1)
        
        # B: (1..., Nb_last, Nb_next, 1, 4, 4)
        # We use block_view: (Nb_last, 4, Nb_next, 4) -> (Nb_last, Nb_next, 4, 4)
        B_raw = block_view.transpose(0, 2, 1, 3)
        # Add singleton dims for the history blocks
        B = B_raw.reshape(*(1,) * (len(nb_dims) - 1), n_blocks, n_blocks, 1, ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
        
        # Sum: A + B
        # Shape: (Nb..., Nb_last, Nb_next, 4, 4, 4)
        # The axes are (..., u_start, u_curr, u_next)
        summed = A + B
        
        # Maximize over the connecting neuron `u_curr` (axis -2)
        # Result: (Nb..., Nb_last, Nb_next, u_start, u_next)
        current_paths = np.max(summed, axis=-2)
        
        # Check memory usage safety
        if current_paths.size > 200_000_000:
             print("Warning: Intermediate tensor large. Optimization might be slow or OOM.")

    # 4. Close the loop
    print("  Closing cycles...")
    # Current paths: (b1, ..., b_{n-1}, u_start, u_{n-1})
    # Closing edge: (b_{n-1}, b1, u_{n-1}, u_start)
    
    # A: (b1, ..., b_{n-1}, u_start, u_{n-1})
    # B: (b_{n-1}, b1, u_{n-1}, u_start)
    
    # We need to align b1 and b_{n-1}
    # A is (Nb, ..., Nb, 4, 4)
    # B is derived from block_view.transpose(0, 2, 1, 3) -> (Nb, Nb, 4, 4)
    
    # Let's use fancy indexing or a loop for the closing step to avoid constructing
    # the full NxN tensor if possible.
    # But for n=4, the tensor is (Nb, Nb, Nb, Nb). 268M elements. It fits.
    
    # Prepare A: (b1, ..., bn-1, u1, un)
    # We want final shape (b1, ..., bn-1) containing max cycle score
    
    # We can iterate over b1 to save memory if n is large
    candidates = []
    
    # Pre-transpose block view for fast access: (u_prev, u_start, b_prev, b_start)
    # Original block_view: (Nb_prev, u_prev, Nb_start, u_start)
    closing_edges = block_view.transpose(1, 3, 0, 2)
    
    # current_paths: (b1, b2, ..., bn-1, u1, un-1)
    # We iterate b1 to reduce memory pressure
    for b1 in range(n_blocks):
        # Slice paths starting at b1
        # shape: (b2, ..., bn-1, u1, un-1)
        paths_slice = current_paths[b1] 
        
        # Get closing edges for b1: connect bn-1 -> b1
        # shape: (u_prev, u_start, bn-1)
        # We select b_start = b1
        closer = closing_edges[:, :, :, b1] 
        
        # paths_slice axes: (b2..., bn-1, u1, un-1)
        # closer axes: (un-1, u1, bn-1)
        
        # We need to match un-1 (last neuron) and u1 (first neuron)
        # Let's move axes of closer to match paths_slice
        # closer: (bn-1, u1, un-1) -> transpose(2, 1, 0)
        closer_aligned = closer.transpose(2, 1, 0)
        
        # Broadcast closer to match intermediate blocks (b2...bn-2)
        # paths_slice has n-2 block dimensions.
        # closer_aligned needs to broadcast over the first n-3 dimensions.
        
        target_shape = paths_slice.shape # (Nb, ..., Nb, 4, 4)
        
        # Reshape closer to (1, ..., 1, Nb, 4, 4)
        # Number of intermediate dims = (n-1) - 1 - 1 = n-3
        # (Minus b1, minus bn-1)
        reshape_dims = (1,) * (n - 2 - 1) + (n_blocks, ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
        closer_view = closer_aligned.reshape(reshape_dims)
        
        # Total Cycle Score = Path + Closing Edge
        # Shape: (Nb..., Nb, 4, 4)
        total_scores = paths_slice + closer_view
        
        # Max over specific neurons (u1, un-1)
        # Shape: (Nb..., Nb)
        max_scores = np.max(total_scores, axis=(-2, -1))
        
        # Indices of best neurons (u1, un-1) flat index 0..15
        max_neuron_indices = np.argmax(total_scores.reshape(*max_scores.shape, -1), axis=-1)
        
        # Find candidates > 0
        # This gives us indices (b2, ..., bn-1)
        locs = np.argwhere(max_scores > 1e-9)
        
        for loc in locs:
            gain = max_scores[tuple(loc)]
            # loc is (b2, ..., bn-1)
            # Full cycle: (b1, *loc)
            cycle_blocks = (b1,) + tuple(loc)
            
            # Constraints: Distinct blocks
            if len(set(cycle_blocks)) != n:
                continue
            
            # Constraints: Canonical order (start with smallest index)
            # This handles cycle uniqueness (1,2,3) == (2,3,1)
            if b1 != min(cycle_blocks):
                continue

            # Retrieve neuron info to reconstruct exact cycle later
            # (We stored just the start/end interaction max, but we need intermediates?)
            # Actually, for MIP we just need the gain and the blocks.
            # We can re-calculate exact neurons only for the winning cycles.
            candidates.append((gain, cycle_blocks))

    print(f"  Found {len(candidates)} candidate {n}-cycles.")
    
    if not candidates:
        return SwapResult([], 0.0)

    # 5. Solve MIP
    # Maximize sum(gain * x) s.t. each block used at most once
    num_vars = len(candidates)
    c_obj = -np.array([c[0] for c in candidates])
    
    row_ind = []
    col_ind = []
    data = []
    
    for c_idx, (_, blocks) in enumerate(candidates):
        for b in blocks:
            row_ind.append(b)
            col_ind.append(c_idx)
            data.append(1)
            
    A_eq = coo_matrix((data, (row_ind, col_ind)), shape=(n_blocks, num_vars))
    constraints = LinearConstraint(A_eq, -np.inf, 1)
    bounds = Bounds(0, 1)
    integrality = np.ones(num_vars)
    
    print("  Solving MIP...")
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
    
    if not res.success:
        print("MIP Solver failed.")
        return SwapResult([], 0.0)
        
    # 6. Reconstruct Results
    selected_indices = np.where(res.x > 0.5)[0]
    final_cycles = []
    total_gain = 0.0
    
    # To find exact neurons for the selected blocks, we do a mini-search
    # This is cheap (only done for selected cycles)
    for idx in selected_indices:
        gain, blocks = candidates[idx]
        total_gain += gain
        
        # Re-resolve the specific neurons for this block tuple
        # We assume blocks is (b1, b2, ..., bn)
        # We want u1, u2, ..., un such that u_k in b_k
        # maximizing sum S[b_k, b_{k+1}, u_k, u_{k+1}]
        
        # Dynamic programming for specific cycle
        # dp[k, u_current] = max score to reach neuron u_current in block k
        # parent[k, u_current] = neuron in previous block
        
        # Initialization (Block 0)
        # We can't pick u0 independently of un.
        # So we just run the full exhaustive search for the specific tuple
        # Size: 4^n. For n=4, 256 checks. Trivial.
        
        best_cycle_neurons = None
        best_cycle_score = -1.0
        
        local_indices = [range(ZERO_BLOCK_SIZE) for _ in range(n)]
        for us in itertools.product(*local_indices):
            # us is (u1_local, u2_local, ...)
            current_score = 0
            for k in range(n):
                b_curr = blocks[k]
                b_next = blocks[(k + 1) % n]
                u_curr = us[k]
                u_next = us[(k + 1) % n]
                
                # Retrieve score from block_view
                # block_view shape: (Nb, u, Nb, v)
                s = block_view[b_curr, u_curr, b_next, u_next]
                current_score += s
            
            if current_score > best_cycle_score:
                best_cycle_score = current_score
                best_cycle_neurons = us
                
        # Convert local to global
        global_cycle = []
        for k in range(n):
            global_idx = blocks[k] * ZERO_BLOCK_SIZE + best_cycle_neurons[k]
            global_cycle.append(global_idx)
            
        final_cycles.append(tuple(global_cycle))
        if VERBOSE:
            print(f"Cycle {global_cycle} Gain: {gain}")

    total_improvement = total_gain / n_samples / (n_neurons // ZERO_BLOCK_SIZE) * 100
    return SwapResult(final_cycles, total_improvement)



def find_perm_impl(
    actmat: npt.NDArray[np.bool_], use_cupy: bool, L1: int
) -> npt.NDArray[np.int_]:
    # Flatten actmat
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))
    
    # Ensure master copy is on CPU to save VRAM
    if hasattr(actmat, "get"):
        actmat = actmat.get()
    
    actmat_orig = actmat 
    n_samples = actmat_orig.shape[0]
    n_neurons = actmat_orig.shape[1]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE
    
    # Metric tracking
    total_pct_improvement = 0.0
    perm = np.arange(L1 // 2)

    stages: list[SwapFunction] = [
        make_swaps_2_3,
    ]
    
    # Allow more fails for stochastic mini-batches
    stages_max_fails = [15]
    stage_id = 0
    num_fails = 0
    
    BATCH_SIZE = 2 ** 16
    
    # Weights for validation (0.5 means equal weight to finding and validating batch)
    W1 = 0.2
    W2 = 0.8

    start_time_global = time.time()

    for i in range(1000):
        # 1. Efficient Sampling
        # Pick 2 * BATCH_SIZE indices at once (indices for batch 1 and batch 2)
        # replace=False prevents overlap between batch1 and batch2
        if n_samples >= BATCH_SIZE * 2:
            current_indices = np.random.choice(n_samples, BATCH_SIZE * 2, replace=False)
            idx1 = current_indices[:BATCH_SIZE]
            idx2 = current_indices[BATCH_SIZE:]
        else:
            # Fallback for small datasets
            print("Warning: Dataset smaller than 2xBatch Size. Reusing data.")
            idx1 = np.arange(n_samples // 2)
            idx2 = np.arange(n_samples // 2, n_samples)

        # Slice on CPU, then transfer to GPU if needed
        batch1_cpu = actmat_orig[idx1, :][:, perm]
        batch2_cpu = actmat_orig[idx2, :][:, perm]

        if use_cupy:
            batch1 = cp.asarray(batch1_cpu, dtype=cp.int8)
            batch2 = cp.asarray(batch2_cpu, dtype=cp.int8)
        else:
            batch1 = batch1_cpu
            batch2 = batch2_cpu

        # 2. Find Candidates on Batch 1
        swap_fn = stages[stage_id]
        res1 = swap_fn(batch1, use_cupy)
        
        if not res1.swaps:
            # No candidates found
            accepted_swaps = []
            accepted_gain_raw = 0.0
        else:
            # 3. Evaluate Candidates on Batch 2
            scores2 = evaluate_swaps_on_batch(batch2, res1.swaps, use_cupy)
            
            accepted_swaps = []
            accepted_gain_raw = 0.0
            
            # 4. Filter based on Weighted Average
            for swap, s1, s2 in zip(res1.swaps, res1.scores, scores2):
                # Calculate weighted raw score
                weighted_raw = (W1 * s1) + (W2 * s2)
                
                if weighted_raw > 0:
                    accepted_swaps.append(swap)
                    accepted_gain_raw += weighted_raw

        # 5. Apply Accepted Swaps
        if accepted_swaps:
            for cycle in accepted_swaps:
                apply_rotate_right(perm, cycle)
            
            # NORMALIZE the gain to percentages
            # accepted_gain_raw is the weighted sum of zeros gained in a batch of size BATCH_SIZE
            # We normalize by (Batch Size * Blocks) to get the fraction of "perfect blocks" gained
            current_pct_gain = (accepted_gain_raw / BATCH_SIZE / n_blocks) * 100
            total_pct_improvement += current_pct_gain

            count_accepted = len(accepted_swaps)
            print(f"Iter {i+1} (Stage {stage_id}): Accepted {count_accepted}/{len(res1)} swaps.")
            print(f"  Iter improvement: {current_pct_gain:0.5f}%")
            print(f"  Total improvement: {total_pct_improvement:0.5f}%")
            print(f"  Time elapsed: {time.time() - start_time_global:0.3f}s")

            # Success! Reset fails and restart from Stage 0 (2-swaps)
            # This allows us to catch simple swaps created by the new configuration
            num_fails = 0
            stage_id = 0 
        else:
            num_fails += 1
            # Only print every fail if verbose, otherwise just status update
            print(f"Iter {i+1} (Stage {stage_id}): No improvement. Fails: {num_fails}/{stages_max_fails[stage_id]}")
            
            if num_fails > stages_max_fails[stage_id]:
                num_fails = 0
                stage_id += 1
                
                if stage_id >= len(stages):
                    print("No more improvement possible.")
                    break
                    
                print(f"Switching to stage {stage_id}")

    return perm


# -------------------------------------------------------------


def read_model(
    nnue_path: str,
    feature_set: FeatureSet,
    config: ModelConfig,
    quantize_config: QuantizationConfig,
) -> NNUEModel:
    with open(nnue_path, "rb") as f:
        reader = NNUEReader(f, feature_set, config, quantize_config)
        return reader.model


def make_fen_batch_provider(
    data_path: str, batch_size: int
) -> data_loader.FenBatchProvider:
    return data_loader.FenBatchProvider(
        data_path,
        True,
        4,  # some speedup and avoids StopIteration from fetch_next_fen_batch.
        batch_size,
        data_loader.DataloaderSkipConfig(
            random_fen_skipping=10,
        ),
    )


def filter_fens(fens: list[str]) -> list[str]:
    # We don't want fens where a king is in check, as these cannot be evaluated by the engine.
    filtered_fens = []
    for fen in fens:
        board = chess.Board(fen=fen)
        if not board.is_check():
            filtered_fens.append(fen)
    return filtered_fens


def quantize_ft(model: NNUEModel) -> None:
    model.input.weight.data = model.input.weight.data.mul(
        model.quantization.ft_quantized_one
    ).round()
    model.input.bias.data = model.input.bias.data.mul(
        model.quantization.ft_quantized_one
    ).round()


def forward_ft(
    model: NNUEModel,
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    white_values: torch.Tensor,
    black_indices: torch.Tensor,
    black_values: torch.Tensor,
    psqt_indices: torch.Tensor,
    layer_stack_indices: torch.Tensor,
) -> torch.Tensor:
    wp, bp = model.input(white_indices, white_values, black_indices, black_values)
    w, _ = torch.split(wp, model.L1, dim=1)
    b, _ = torch.split(bp, model.L1, dim=1)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0_ = torch.clamp(l0_, 0.0, model.quantization.ft_quantized_one)

    l0_s = torch.split(l0_, model.L1 // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    # We multiply by 255/512 because in the quantized network 1.0 is represented by 255
    # and we want to scale to 1.0=127, but a shift is faster than a division (in inference)
    l0_ = torch.cat(l0_s1, dim=1) * (1 / 512)

    return l0_.round()


def eval_ft(model: NNUEModel, batch: data_loader.SparseBatchPtr) -> torch.Tensor:
    with torch.no_grad():
        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch.contents.get_tensors("cuda")
        res = forward_ft(
            model,
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        )
        return res


def ft_permute_impl(model: NNUEModel, perm: npt.NDArray[np.int_]) -> None:
    permutation = list(perm)

    l1_size = model.layer_stacks.l1.linear.in_features
    if l1_size != len(permutation) * 2:
        raise Exception(
            f"Invalid permutation size. Expected {l1_size}. Got {len(permutation) * 2}."
        )

    # Both sides of the FT must use the same permutation.
    permutation.extend([x + l1_size // 2 for x in permutation])

    # Add identity permutation for PSQT weights
    ft_permutation = permutation + list(range(l1_size, model.input.num_outputs))

    # Apply the permutation in place.
    model.input.weight.data = model.input.weight.data[:, ft_permutation]
    model.input.bias.data = model.input.bias.data[ft_permutation]
    model.layer_stacks.l1.linear.weight.data = model.layer_stacks.l1.linear.weight.data[
        :, permutation
    ]


def ft_permute(model: NNUEModel, ft_perm_path: str) -> None:
    with open(ft_perm_path, "rb") as f:
        permutation = np.load(f)

    ft_permute_impl(model, permutation)


def gather_impl(model: NNUEModel, dataset: str, count: int) -> npt.NDArray[np.bool_]:
    ZERO_POINT = 0.0  # Vary this to check hypothetical forced larger truncation to zero
    BATCH_SIZE = 1000

    quantized_model = copy.deepcopy(model)
    quantize_ft(quantized_model)
    quantized_model.cuda()

    fen_batch_provider = make_fen_batch_provider(dataset, BATCH_SIZE)

    actmats = []

    done = 0
    print("Processed {} positions.".format(done))
    while done < count:
        fens = filter_fens(next(fen_batch_provider))

        b = data_loader.get_sparse_batch_from_fens(
            quantized_model.feature_set.name,
            fens,
            [0] * len(fens),
            [1] * len(fens),
            [0] * len(fens),
        )
        actmat = eval_ft(quantized_model, b).cpu()
        actmat = actmat <= ZERO_POINT
        actmats.append(actmat.numpy())
        data_loader.destroy_sparse_batch(b)

        done += len(fens)
        print("Processed {} positions.".format(done))

    return np.concatenate(actmats, axis=0)


def command_gather(args: argparse.Namespace) -> None:
    feature_set = M.get_feature_set_from_name(args.features)
    if args.checkpoint:
        nnue = NNUE.load_from_checkpoint(
            args.checkpoint,
            feature_set=feature_set,
            config=ModelConfig(L1=args.l1),
            quantize_config=QuantizationConfig(),
        )
        model = nnue.model
    else:
        model = read_model(
            args.net, feature_set, ModelConfig(L1=args.l1), QuantizationConfig()
        )

    model.eval()

    actmat = gather_impl(model, args.data, args.count)

    with open(args.out, "wb") as file:
        np.save(file, actmat)


def eval_act_mat(actmat: npt.NDArray[np.bool_]) -> float:
    actmat = actmat.reshape((actmat.shape[0], actmat.shape[1] // 4, 4))
    r = np.all(actmat, axis=2)
    return np.count_nonzero(r) / r.shape[0] / r.shape[1]


def eval_perm_impl(
    actmat: npt.NDArray[np.bool_], perm: npt.NDArray[np.int_] | None = None
) -> None:
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))

    actmat_eval = eval_act_mat(actmat)
    print(f"Combined zeros in base matrix: {actmat_eval * 100:0.6f}")

    if perm is not None:
        perm_act_mat = actmat[:, perm]
        perm_act_mat_eval = eval_act_mat(perm_act_mat)
        print(f"Combined zeros in perm matrix: {perm_act_mat_eval * 100:0.6f}")


def command_eval_perm(args: argparse.Namespace) -> None:
    with open(args.data, "rb") as file:
        actmat = np.load(file)

    if args.perm is not None:
        with open(args.perm, "rb") as file:
            perm = np.load(file)
    else:
        perm = None

    eval_perm_impl(actmat, perm)


def command_find_perm(args: argparse.Namespace) -> None:
    with open(args.data, "rb") as file:
        actmat = np.load(file)

    perm = find_perm_impl(actmat, args.use_cupy, args.l1)

    # perm = np.random.permutation([i for i in range(L1)])
    with open(args.out, "wb") as file:
        np.save(file, perm)


def ft_optimize(
    model: NNUEModel,
    dataset_path: str,
    count: int,
    actmat_save_path: str | None = None,
    perm_save_path: str | None = None,
    use_cupy: bool = True,
) -> None:
    print("Gathering activation data...")
    actmat = gather_impl(model, dataset_path, count)
    if actmat_save_path is not None:
        with open(actmat_save_path, "wb") as file:
            np.save(file, actmat)

    print("Finding permutation...")
    perm = find_perm_impl(actmat, use_cupy, model.L1)
    if perm_save_path is not None:
        with open(perm_save_path, "wb") as file:
            np.save(file, perm)

    print("Evaluating permutation...")
    eval_perm_impl(actmat, perm)

    print("Applying permutation...")
    ft_permute_impl(model, perm)


def set_cupy_device(device: int) -> None:
    if device is not None:
        cp.cuda.runtime.setDevice(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--no-cupy",
        action="store_false",
        dest="use_cupy",
        help="Disable CUPY usage if not enough GPU memory is available. This will use numpy instead, which is slower.",
    )
    parser.add_argument(
        "--device", type=int, default="0", help="Device to use for cupy"
    )
    subparsers = parser.add_subparsers()

    parser_gather = subparsers.add_parser("gather", help="a help")
    parser_gather.add_argument("--net", type=str, help="path to a .nnue net")
    parser_gather.add_argument(
        "--data", type=str, help="path to a .bin or .binpack dataset"
    )
    parser_gather.add_argument(
        "--checkpoint",
        type=str,
        help="Optional checkpoint (used instead of nnue for local eval)",
    )
    parser_gather.add_argument(
        "--count", type=int, default=1000, help="number of datapoints to process"
    )
    parser_gather.add_argument(
        "--out", type=str, help="Filename under which to save the resulting ft matrix"
    )
    parser_gather.add_argument("--l1", type=int, default=M.ModelConfig().L1)
    M.add_feature_args(parser_gather)
    parser_gather.set_defaults(func=command_gather)

    parser_find_perm = subparsers.add_parser("find_perm", help="a help")
    parser_find_perm.add_argument(
        "--data", type=str, help="path to the previously gathered ft activation data"
    )
    parser_find_perm.add_argument(
        "--out", type=str, help="path to where to save the permutation"
    )
    parser_find_perm.add_argument("--l1", type=int, default=M.ModelConfig().L1)
    parser_find_perm.set_defaults(func=command_find_perm)

    parser_eval_perm = subparsers.add_parser("eval_perm", help="a help")
    parser_eval_perm.add_argument(
        "--data", type=str, help="path to the previously gathered ft activation data"
    )
    parser_eval_perm.add_argument(
        "--perm", type=str, help="path to the previously generated perm file"
    )
    parser_eval_perm.set_defaults(func=command_eval_perm)

    args = parser.parse_args()

    if args.use_cupy:
        if args.device is not None:
            set_cupy_device(args.device)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
