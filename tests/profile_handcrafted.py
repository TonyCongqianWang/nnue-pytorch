"""Read-only traffic profiling and benchmark-input generation; no training."""

import argparse
import json
import sys
from collections import Counter, OrderedDict
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch
from validate_handcrafted_engine import episodes

import data_loader
from data_loader import stream
from model.config import ModelConfig
from model.perspective import bucket
from model.utils.serialize import NNUEReader


def pair(b):
    return (bucket(b, b.turn), bucket(b, not b.turn))


def comparisons(rows):
    result = {}
    for kind, items in rows.items():
        count = len(items)
        result[kind] = {
            "comparisons": count,
            "stm_changes": sum(a[0] != b[0] for a, b in items),
            "opp_changes": sum(a[1] != b[1] for a, b in items),
            "pair_changes": sum(a != b for a, b in items),
            "same_color_feature_changes": sum(a != (b[1], b[0]) for a, b in items)
            if kind != "siblings"
            else None,
        }
    return result


def traffic(fens):
    hist = [Counter(), Counter()]
    pairs = []
    colors = []
    turns = []
    for fen in fens:
        board = chess.Board(fen)
        ids = pair(board)
        colors.append(ids if board.turn else ids[::-1])
        turns.append(board.turn)
        pairs.append(ids)
        for role, b in enumerate(ids):
            hist[role][b] += 1
    changes = list(pairwise(pairs))
    lru = {}
    for capacity in [2, 4, 8, 16, 32, 64, 172]:
        cache = OrderedDict()
        misses = 0
        for ids in pairs:
            for role, expert in enumerate(ids):
                key = (role, expert)
                if key not in cache:
                    misses += 1
                    if len(cache) == capacity:
                        cache.popitem(last=False)
                else:
                    cache.pop(key)
                cache[key] = True
        lru[capacity] = misses
    sequence = comparisons({"adjacent": changes})["adjacent"]
    sequence.pop("same_color_feature_changes")
    sequence["color_normalized_pair_changes"] = sum(a != b for a, b in pairwise(colors))
    sequence["role_alternation_only"] = sum(
        pairs[i] != pairs[i - 1]
        and colors[i] == colors[i - 1]
        and turns[i] != turns[i - 1]
        for i in range(1, len(pairs))
    )
    return {
        "positions": len(fens),
        "simulated_expert_lru_misses": lru,
        "role_entry_accesses": len(pairs) * 2,
        "histograms": [dict(sorted(h.items())) for h in hist],
        "unused_feasible": [[i for i in range(85) if not h[i]] for h in hist],
        "rare_feasible_lt10": [[i for i in range(85) if 0 < h[i] < 10] for h in hist],
        "sequence_changes": sequence,
        "distinct_role_experts": sum(len(h) for h in hist),
        "distinct_pairs": len(set(pairs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--net")
    p.add_argument("--inputs")
    p.add_argument("--search-trace")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = {"quiet": [], "capture": [], "promotion": [], "null": [], "siblings": []}
    fens = []
    for fen, moves in episodes():
        b = chess.Board(fen)
        for text in moves:
            m = chess.Move.from_uci(text)
            kind = (
                "null"
                if not m
                else "promotion"
                if m.promotion
                else "capture"
                if b.is_capture(m)
                else "quiet"
            )
            old = pair(b)
            fens.append(b.fen())
            children = []
            for sibling in list(b.legal_moves)[:16]:
                b.push(sibling)
                children.append(pair(b))
                b.pop()
            rows["siblings"] += list(pairwise(children))
            b.push(m)
            rows[kind].append((old, pair(b)))
    result = {"legal_move_proxy": traffic(fens), "move_comparisons": comparisons(rows)}
    if args.data:
        provider = data_loader.FenBatchProvider(
            args.data,
            False,
            1,
            8192,
            data_loader.DataloaderSkipConfig(
                random_fen_skipping=0, early_fen_skipping=-1, soft_early_fen_skipping=-1
            ),
        )
        sample = next(iter(provider))
        result["dataset_sample"] = traffic(sample)
    if args.search_trace:
        result["short_search_trace"] = traffic(
            Path(args.search_trace).read_text().splitlines()
        )
    if args.inputs:
        with open(args.net, "rb") as file:
            m = NNUEReader(file, "Full_Threats+PP_3Wide+K32H3^", ModelConfig()).model
        selected = fens[:512]
        batch = stream.get_sparse_batch_from_fens(
            m.input_feature_name,
            selected,
            [0] * len(selected),
            [1] * len(selected),
            [0] * len(selected),
        )
        try:
            tensors = batch.contents.get_tensors("cpu")
            with torch.no_grad():
                values = torch.round(m.forward_ft(*tensors[:4], True, True) * 128).int()
            with open(args.inputs, "w") as f:
                for fen, row in zip(selected, values):
                    f.write(fen + "\n" + " ".join(map(str, row.tolist())) + "\n")
            nnz = (values.reshape(-1, 256, 4) != 0).any(-1).sum(-1).float()
            result["benchmark_inputs"] = {
                "positions": len(selected),
                "mean_nonzero_chunks4": nnz.mean().item(),
                "mean_fc0_weight_bytes": nnz.mean().item() * 128,
                "mean_executed_macs": nnz.mean().item() * 128 + 2304,
            }
        finally:
            stream.destroy_sparse_batch(batch)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
