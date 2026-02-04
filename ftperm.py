"""

NOTE: This script uses CUDA and may require large amounts of VRAM. Decrease --count if encountering problems.

Example use:

1. Generate the activation matrix for some sample dataset.

python ftperm.py gather -data="/kaggle/input/fishtest-data/fishpack32.binpack" --net=networks/nn-c288c895ea92.nnue --count=1000000 --features=Full_Threats --l1=1024 --out ftact1m.npy

2. Find a permutation

python ftperm.py find_perm --data=ftact1m.npy --out=ftact.perm --l1=1024

3. Test the permutation against the baseline

python ftperm.py eval_perm --data=ftact1m.npy --perm=ftact.perm

4. Apply permutation and save
python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=Full_Threats --ft_perm=ftact.perm

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
    Handles mixed cycle lengths (2-swaps and 3-cycles) in the same list.
    """
    if not swaps:
        return []

    # Precompute the score change matrix once
    score_changes = get_score_change(actmat, use_cupy=use_cupy)
    
    if use_cupy:
        score_changes = cp.asnumpy(score_changes)

    scores = []

    for idx in swaps:
        cycle_len = len(idx)
        
        if cycle_len == 2:
            # 2-swap: (i, j)
            i, j = idx
            # Gain is symmetric sum
            gain = score_changes[i, j] + score_changes[j, i]
            scores.append(gain)
            
        elif cycle_len == 3:
            # 3-cycle: (i, j, k)
            i, j, k = idx
            # Gain is sum of edges in the cycle i->j->k->i
            gain = score_changes[i, j] + score_changes[j, k] + score_changes[k, i]
            scores.append(gain)
            
        else:
            # Generic n-cycle fallback
            gain = 0
            for k in range(cycle_len):
                u = idx[k]
                v = idx[(k + 1) % cycle_len]
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
    # Note: This is an UPPER BOUND approximation.
    
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

def solve_dense_matching(score_matrix: npt.NDArray, size: int) -> list[tuple[int, int]]:
    """
    Solves Maximum Weight Matching for a dense symmetric matrix.
    Returns a list of pairs (i, j).
    """
    # Create graph from adjacency matrix
    # We only take the upper triangle to avoid duplicates and self-loops
    score_matrix = np.triu(score_matrix, k=1)
    
    # Get indices where score > 0
    rows, cols = np.nonzero(score_matrix)
    weights = score_matrix[rows, cols]
    
    G = nx.Graph()
    G.add_nodes_from(range(size))
    
    # Bulk add edges (much faster than iterating)
    edges = zip(rows, cols, weights)
    G.add_weighted_edges_from(edges)
    
    # Calculate matching
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return list(matching)

def hierarchical_initialization(
    actmat: npt.NDArray[np.bool_], use_cupy: bool = True,
) -> npt.NDArray[np.int_]:
    t0 = time.time()
    
    n_samples, n_neurons = actmat.shape
    
    # Ensure data is on the correct device for heavy matrix multiplication
    if use_cupy:
        actmat_gpu = cp.asarray(actmat, dtype=cp.float32)
    else:
        actmat_gpu = actmat.astype(np.float32)

    # --- Phase 1: Match Individual Neurons (1 -> 2) ---
    # Compute co-occurrence matrix: C[i, j] = count of samples where both i and j are zero (True)
    # Since inputs are boolean/1.0, dot product gives the count of shared 1s.
    scores_1 = actmat_gpu.T @ actmat_gpu
    
    if use_cupy:
        scores_1 = cp.asnumpy(scores_1)
        
    pairs = solve_dense_matching(scores_1, n_neurons)
    
    # Sort pairs internally for consistency and create a mapping
    pairs = [tuple(sorted(p)) for p in pairs]
    # Handle any unmatched neurons (shouldn't happen with even L1, but good for safety)
    matched_indices = set(itertools.chain(*pairs))
    unmatched = [i for i in range(n_neurons) if i not in matched_indices]
    # Naively pair up remaining
    for i in range(0, len(unmatched), 2):
        if i + 1 < len(unmatched):
            pairs.append((unmatched[i], unmatched[i+1]))
        else:
            # Odd number of neurons, leave last one hanging or handle gracefully
            # For NNUE L1 is usually even, so we ignore the odd case here.
            pass

    # --- Phase 2: Match Pairs (2 -> 4) ---
    n_pairs = len(pairs)
    
    # Create a "Pair Activation Matrix"
    # A pair is "active" (zero) only if BOTH constituents are zero.
    # We use logical AND.
    
    # We need to construct this matrix.
    # Indexing approach: actmat[:, [p[0] for p in pairs]] * actmat[:, [p[1] for p in pairs]]
    idx_left = [p[0] for p in pairs]
    idx_right = [p[1] for p in pairs]
    
    if use_cupy:
        idx_left_gpu = cp.array(idx_left)
        idx_right_gpu = cp.array(idx_right)
        pair_actmat = actmat_gpu[:, idx_left_gpu] * actmat_gpu[:, idx_right_gpu]
    else:
        pair_actmat = actmat_gpu[:, idx_left] * actmat_gpu[:, idx_right]

    # Compute co-occurrence for pairs
    scores_2 = pair_actmat.T @ pair_actmat
    
    if use_cupy:
        scores_2 = cp.asnumpy(scores_2)
        
    quad_matches = solve_dense_matching(scores_2, n_pairs)
    
    # --- Phase 3: Construct Permutation ---
    final_perm = []
    
    # Add matched quads
    matched_pair_indices = set()
    for p_idx_1, p_idx_2 in quad_matches:
        matched_pair_indices.add(p_idx_1)
        matched_pair_indices.add(p_idx_2)
        
        # Unpack the original neurons
        n1, n2 = pairs[p_idx_1]
        n3, n4 = pairs[p_idx_2]
        final_perm.extend([n1, n2, n3, n4])
        
    # Add unmatched pairs (if any)
    for i in range(n_pairs):
        if i not in matched_pair_indices:
            n1, n2 = pairs[i]
            final_perm.extend([n1, n2])
            
    # Add any remaining individual neurons (from odd sizing)
    final_perm.extend(unmatched)
    
    print(f"Hierarchical initialization done in {time.time() - t0:.2f}s")
    
    # Cast to standard numpy int array
    return np.array(final_perm, dtype=int)

import time
import numpy as np
import numpy.typing as npt

def prepare_data_split(
    actmat: npt.NDArray, 
    val_size: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Shuffles and splits the dataset into training and validation sets.
    Performs the split on CPU to conserve GPU memory.
    """
    n_samples = actmat.shape[0]
    
    # Generate shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - val_size))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create views (or copies depending on memory layout)
    train_data = actmat[train_indices]
    val_data = actmat[val_indices]
    
    print(f"Data Split: {len(train_data)} training, {len(val_data)} validation samples.")
    return train_data, val_data

def measure_validation_score(
    val_data: npt.NDArray, 
    perm: npt.NDArray, 
    use_cupy: bool, 
    n_neurons: int, 
    max_samples: int = 2 ** 18
) -> float:
    """
    Measures quality on the Held-Out Validation Set.
    """
    n_val = val_data.shape[0]
    if n_val == 0:
        return 0.0

    # If validation set is huge, we still subsample it for speed, 
    # but we strictly sample from val_data, never train_data.
    check_size = min(max_samples, n_val)
    
    # Deterministic slicing for validation stability (optional) 
    # or random sampling from the validation set:
    indices = np.random.choice(n_val, check_size, replace=False)
    sample_cpu = val_data[indices, :][:, perm]

    if use_cupy:
        import cupy as cp
        sample = cp.asarray(sample_cpu, dtype=cp.int8)
        score_mat = get_swapped_zero_positive_count(sample, use_cupy=True)
        # Metric: Trace / (N * D)
        quality = cp.trace(score_mat) / check_size / n_neurons * 100
        
        del sample, score_mat
        cp.get_default_memory_pool().free_all_blocks()
    else:
        score_mat = get_swapped_zero_positive_count(sample_cpu, use_cupy=False)
        quality = np.trace(score_mat) / check_size / n_neurons * 100
        
    return float(quality)

def get_training_batches(
    train_data: npt.NDArray, 
    perm: npt.NDArray, 
    batch_size: int, 
    use_cupy: bool
):
    """
    Samples strictly from the Training Data.
    """
    n_train = train_data.shape[0]
    
    # Check if we have enough data for 2 unique batches
    # If not, we allow overlap (replace=False will fail if we ask for more than n_train)
    needed = batch_size * 2
    
    if n_train >= needed:
        indices = np.random.choice(n_train, needed, replace=False)
        idx1 = indices[:batch_size]
        idx2 = indices[batch_size:]
    else:
        # Fallback: if training set is tiny, reuse data but warn once (handled in main loop)
        # We wrap around
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        idx1 = indices[:min(n_train, batch_size)]
        idx2 = indices[min(n_train, batch_size):] 
        # Note: Logic here implies idx2 might be empty or overlap if n_train is very small
        # For safety in very small datasets:
        if len(idx2) == 0: idx2 = idx1

    batch1_cpu = train_data[idx1, :][:, perm]
    batch2_cpu = train_data[idx2, :][:, perm]

    if use_cupy:
        import cupy as cp
        batch1 = cp.asarray(batch1_cpu, dtype=cp.int8)
        batch2 = cp.asarray(batch2_cpu, dtype=cp.int8)
    else:
        batch1 = batch1_cpu
        batch2 = batch2_cpu
        
    return batch1, batch2

def find_perm_impl(
    actmat: npt.NDArray[np.bool_], 
    use_cupy: bool, 
    L1: int,
    skip_init: bool = False,
    max_iters: int = 2000,
    log_steps: int = 50,
    validation_steps: int = 200,
    validation_set_size: float = 0.25
) -> npt.NDArray[np.int_]:
    
    # 1. Pre-processing
    # -----------------------------
    # Flatten/Reshape as per domain logic
    # Note: We reshape BEFORE splitting to ensure samples are consistent
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))
    
    if hasattr(actmat, "get"):
        actmat = actmat.get()
    
    n_neurons = actmat.shape[1]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE
    
    # 2. Train/Validation Split
    # -----------------------------
    train_data, val_data = prepare_data_split(actmat, validation_set_size)
    
    if train_data.shape[0] == 0:
        raise ValueError("Validation split resulted in 0 training samples. Decrease validation_set_size.")

    # 3. Initialization (On Training Data Only)
    # -----------------------------
    if not skip_init:
        print("Running hierarchical initialization on training data...")
        
        # --- Reintroduced INIT_BATCH_SIZE logic ---
        INIT_BATCH_SIZE = 2 ** 16
        if train_data.shape[0] > INIT_BATCH_SIZE:
            # Randomly sample a subset for the heavy initialization step
            init_indices = np.random.choice(train_data.shape[0], INIT_BATCH_SIZE, replace=False)
            init_data = train_data[init_indices]
        else:
            init_data = train_data
            
        perm = hierarchical_initialization(init_data, use_cupy=use_cupy)
    # ------------------------------------------
    
    if skip_init or len(perm) != n_neurons:
        perm = np.arange(n_neurons)
    if len(perm) != n_neurons:
        print(f"Warning: Init produced len {len(perm)}, expected {n_neurons}. Resetting.")

    # Initial Validation Score
    init_quality_0 = measure_validation_score(val_data, np.arange(n_neurons), use_cupy, n_neurons, max_samples=2 ** 20)
    print(f"Validation Quality Before Initialization: {init_quality_0:.4f}%")
    init_quality_1 = measure_validation_score(val_data, perm, use_cupy, n_neurons, max_samples=2 ** 20)
    print(f"Validation Quality After Initialization: {init_quality_1:.4f}%")

    best_perm = perm.copy()
    best_val_score = max(init_quality_0, init_quality_1)

    # 4. Setup Loop
    # -----------------------------
    stages: list[SwapFunction] = [
        make_swaps_2,
        make_swaps_2_3,
    ]
    stages_max_fails = [5, 15]
    stage_id = 0
    num_fails = 0
    
    BASE_BATCH_SIZE = 2 ** 10
    current_batch_size = BASE_BATCH_SIZE # Fixed: Was undefined 'BATCH_SIZE'
    W1 = 0.3

    if train_data.shape[0] < current_batch_size * 2:
        current_batch_size = train_data.shape[0] // 2
        if current_batch_size < 16:
            current_batch_size = train_data.shape[0]
    
    start_time_global = time.time()
    
    # 5. Optimization Loop
    # -----------------------------
    for i in range(max_iters):
        # --- Schedule & Batch Sizing ---
        if i in [int(frac * max_iters) for frac in [0.4, 0.8, 0.9, 0.97, 0.99]]:
            W1 /= 1.5
            current_batch_size *= 2
            
            # Check if Training Data supports this Batch Size
            if train_data.shape[0] < current_batch_size * 2:
                current_batch_size = train_data.shape[0] // 2
                if current_batch_size < 16:
                    current_batch_size = train_data.shape[0]

        # --- Get Training Batches ---
        batch1, batch2 = get_training_batches(train_data, perm, current_batch_size, use_cupy)

        # --- Search & Evaluate ---
        # 1. Find candidates on Batch 1
        swap_fn = stages[stage_id]
        res1 = swap_fn(batch1, use_cupy)
        
        accepted_swaps = []
        accepted_gain_raw = 0.0

        if res1.swaps:
            # 2. Verify candidates on Batch 2
            scores2 = evaluate_swaps_on_batch(batch2, res1.swaps, use_cupy)
            
            # 3. Weighted Filter
            for swap, s1, s2 in zip(res1.swaps, res1.scores, scores2):
                weighted_raw = (W1 * s1) + ((1 - W1) * s2)
                
                if weighted_raw > 0:
                    accepted_swaps.append(swap)
                    accepted_gain_raw += weighted_raw

        # --- Apply & Update ---
        if accepted_swaps:
            for cycle in accepted_swaps:
                apply_rotate_right(perm, cycle)
            
            # Gain calculation (on training batch)
            current_pct_gain = (accepted_gain_raw / current_batch_size / n_blocks) * 100

            count_accepted = len(accepted_swaps)
            count_candidates = len(res1.swaps)

            if log_steps > 0 and (i + 1) % log_steps == 0:
                print(f"Iter {i+1} / {max_iters} (Stage {stage_id}): Accepted {count_accepted}/{count_candidates}.")
                print(f"    Train improvement: {current_pct_gain:0.5f}%")

            num_fails = 0
            stage_id = 0 
        else:
            num_fails += 1
            max_fails = stages_max_fails[stage_id]
            if log_steps > 0 and (i + 1) % log_steps == 0:
                print(f"Iter {i+1} / {max_iters} (Stage {stage_id}): Fail {log_steps} / {max_fails}.")
            if num_fails > max_fails:
                print(f"Iter {i+1}: Stage {stage_id} max fails reached.")
                num_fails = 0
                stage_id += 1
                
                if stage_id >= len(stages):
                    print("No more improvement possible.")
                    break
                print(f"Switching to stage {stage_id}")


        # --- Periodic Validation (Indentation fixed: Runs every step check) ---
        if validation_steps > 0 and (i + 1) % validation_steps == 0 or i > 0.998 * max_iters:
            val_score = measure_validation_score(val_data, perm, use_cupy, n_neurons, max_samples=2 ** 20)
            elapsed = time.time() - start_time_global
            print(f"--- [Val] Iter {i + 1}: {val_score:.4f}% (Elapsed: {elapsed:.1f}s) ---")
            if val_score > best_val_score:
                print(f"--- [Val]     New best score: {val_score:.4f} from {best_val_score:.4f} ---")
                best_perm = perm.copy()
                best_val_score = val_score

    # Final Validation
    best_perm = best_perm if validation_steps > 0 else perm
    final_score = measure_validation_score(val_data, best_perm, use_cupy, n_neurons, max_samples=2 ** 20)
    print(f"Final Validation Quality: {final_score:.4f}%")
    
    return best_perm

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

def simple_eval(board: chess.Board) -> int:
    """
    Replicates the C++ simple_eval function.
    Calculates material imbalance from the perspective of the side to move.
    """
    PAWN_VALUE   = 208
    KNIGHT_VALUE = 781
    BISHOP_VALUE = 825
    ROOK_VALUE   = 1276
    QUEEN_VALUE  = 2538

    # Piece values map for non-pawn material calculation
    piece_values = {
        chess.KNIGHT: KNIGHT_VALUE,
        chess.BISHOP: BISHOP_VALUE,
        chess.ROOK:   ROOK_VALUE,
        chess.QUEEN:  QUEEN_VALUE,
    }

    # Counts for current side (c) and opponent (~c)
    us = board.turn
    them = not us

    # 1. Pawn Difference
    # (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
    pawn_score = (
        len(board.pieces(chess.PAWN, us)) - 
        len(board.pieces(chess.PAWN, them))
    ) * PAWN_VALUE

    # 2. Non-Pawn Material (pos.non_pawn_material(c))
    def non_pawn_material(color: chess.Color) -> int:
        mat = 0
        for piece_type, value in piece_values.items():
            mat += len(board.pieces(piece_type, color)) * value
        return mat

    us_non_pawn = non_pawn_material(us)
    them_non_pawn = non_pawn_material(them)

    # return PawnValue * (...) + pos.non_pawn_material(c) - pos.non_pawn_material(~c)
    return pawn_score + us_non_pawn - them_non_pawn


def filter_samples_impl(fens: list[str]) -> list[str]:
    """
    Filters FENs to keep only those that are NOT evaluated by the small net.
    Logic: use_smallnet returns true if abs(simple_eval) > 962.
    We want the inverse: keep if abs(simple_eval) <= 962.
    """
    THRESHOLD = 962
    filtered = []
    
    for fen in fens:
        board = chess.Board(fen)
        
        # Original filter: Check for check (engine limitation)
        if board.is_check():
            continue
            
        # New filter: Big Net logic
        score = simple_eval(board)
        if abs(score) <= THRESHOLD:
            filtered.append(fen)
            
    return filtered


def gather_impl(model: NNUEModel, dataset: str, count: int, filter_samples: bool) -> npt.NDArray[np.bool_]:
    ZERO_POINT = 0.0
    
    # Configuration
    GPU_BATCH_SIZE = 1024   # Optimal size for your GPU
    DISK_READ_SIZE = 16384  # Read large chunks to minimize I/O overhead
    log_steps = int(count / 20)
    
    quantized_model = copy.deepcopy(model)
    quantize_ft(quantized_model)
    quantized_model.cuda()

    # The provider reads raw lines from the file
    fen_batch_provider = make_fen_batch_provider(dataset, DISK_READ_SIZE)

    actmats = []
    fen_buffer = []  # The holding area

    old_done = 0
    done = 0
    print(f"Target count: {count}")
    if filter_samples:
        print("Sample filtering is enabled")
        num_filtered = 0
    while done < count:
        # 1. Fill the buffer until we have enough for a GPU batch
        #    (or until the dataset runs out)
        dataset_exhausted = False
        while len(fen_buffer) < GPU_BATCH_SIZE:
            try:
                raw_fens = next(fen_batch_provider)
                valid_fens = raw_fens
                if filter_samples:
                    valid_fens = filter_samples_impl(raw_fens)
                    num_filtered += len(raw_fens) - len(valid_fens)
                fen_buffer.extend(valid_fens)
            except StopIteration:
                dataset_exhausted = True
                break
        
        # 2. If buffer is empty after trying to fill, we are truly done
        if not fen_buffer:
            break

        # 3. Slice off exactly one batch (or the remainder if we are finishing)
        #    Don't exceed 'count' (global target)
        remaining_needed = count - done
        # We take the smaller of: GPU capacity, what's in buffer, or what's needed to finish
        current_batch_size = min(GPU_BATCH_SIZE, len(fen_buffer), remaining_needed)
        
        batch_fens = fen_buffer[:current_batch_size]
        
        # Remove used fens from buffer (keep the rest for next time)
        fen_buffer = fen_buffer[current_batch_size:]

        # 4. Process on GPU
        b = data_loader.get_sparse_batch_from_fens(
            quantized_model.feature_set.name,
            batch_fens,
            [0] * len(batch_fens),
            [1] * len(batch_fens),
            [0] * len(batch_fens),
        )
        
        with torch.no_grad():
            actmat = eval_ft(quantized_model, b).cpu()
        
        actmat = actmat <= ZERO_POINT
        actmats.append(actmat.numpy())
        
        data_loader.destroy_sparse_batch(b)

        done += len(batch_fens)
        if done > old_done + log_steps:
            old_done = done
            print(f"Processed {done}/{count} positions. (Buffer: {len(fen_buffer)})")
            if filter_samples:
                print(f"   Filtered Samples: {num_filtered}")

        if dataset_exhausted and not fen_buffer:
            print(f"Warning: Dataset exhausted before reaching target. Stopped at {done}.")
            break

    if not actmats:
        raise ValueError("No samples passed the filter criteria.")

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

    actmat = gather_impl(model, args.data, args.count, args.filter_samples)

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

    perm = find_perm_impl(actmat, args.use_cupy, args.l1, args.skip_init, args.max_iters)

    # perm = np.random.permutation([i for i in range(L1)])
    with open(args.out, "wb") as file:
        np.save(file, perm)


def ft_optimize(
    model: NNUEModel,
    dataset_path: str,
    count: int,
    actmat_save_path: str | None = None,
    perm_save_path: str | None = None,
    filter_samples: bool = True,
    skip_init: bool = False,
    max_iters: int = 6000,
    use_cupy: bool = True,
) -> None:
    print("Gathering activation data...")
    actmat = gather_impl(model, dataset_path, count, filter_samples)
    if actmat_save_path is not None:
        with open(actmat_save_path, "wb") as file:
            np.save(file, actmat)

    print("Finding permutation...")
    perm = find_perm_impl(actmat, use_cupy, model.L1, skip_init, max_iters)
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
        "--filter_samples", type=bool, default=True, help="Filter samples or not according to small_net deferral logic."
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
    parser_find_perm.add_argument("--max_iters", type=int, default=2000)
    parser_find_perm.add_argument(
        "--skip_init", type=bool, default=False, help="Skip init with hierachical matching."
    )
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
