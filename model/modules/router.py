import math
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .quantile_routing import ITERATIONS, gather_and_solve


@contextmanager
def ieee_matmul_precision():
    """Forces standard IEEE float32 precision (disabling TF32) for numerical sensitivity.

    Ensures discrete routing decisions (argmin) and codebook distance metrics
    are bit-identical between training (CUDA), table generation (CUDA/CPU),
    and Stockfish inference.
    """
    if not torch.cuda.is_available():
        yield
        return

    # Only the modern matmul-precision API: mixing it with the legacy
    # allow_tf32 knob is rejected by recent torch builds.
    prev_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)


NUM_MATERIAL_STATES = 468
HEAVY_CATEGORIES = ("Q-no", "Q-yes", "R-no", "R-yes", "Else-no", "Else-yes")
NUM_FC0_MATERIAL_STATES = 240
KING_STATES = 8
WING_STATES = 5
TAIL_GEOMETRY_STATES = KING_STATES * WING_STATES
FC0_ROUTER_TABLE_SIZE = NUM_FC0_MATERIAL_STATES**2
ROUTER_TABLE_SIZE = NUM_MATERIAL_STATES**2 * TAIL_GEOMETRY_STATES
NUM_FC0_EXPERTS = 64
NUM_WEIGHT_EXPERTS = 256
FC0_CODEBOOK_DIM = 64
CONTEXT_DIM = 2048
CODEBOOK_DIM = 64
ROUTER_HASH = 0x4D313200
SHARED_ROUTER_HASH = 0x4D313400
DEFAULT_CONTEXT_COMMITMENT_COST = 0.0002
DEFAULT_CENTROID_COMMITMENT_COST = 0.0001
DEFAULT_CODEBOOK_TARGET_NORM = 1.0
DEFAULT_MIN_RELATIVE_UTILIZATION = 0.064
MOE_BANKS = ("fc0", "tail")
MOE_LOSS_COMPONENTS = tuple(
    f"{bank}_{term}"
    for bank in MOE_BANKS
    for term in ("context_commitment", "centroid_commitment")
)


def rms_norm(
    x: torch.Tensor,
    target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projects vectors onto a hypersphere of radius target_norm (parameter-free RMSNorm)."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    target_rms = target_norm / math.sqrt(x.shape[-1])
    return x * (target_rms / rms)


def center_and_rms_norm(
    x: torch.Tensor,
    target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-centers vectors and projects them onto hypersphere of radius target_norm."""
    x_c = x - torch.mean(x, dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(x_c**2, dim=-1, keepdim=True) + eps)
    target_rms = target_norm / math.sqrt(x.shape[-1])
    return x_c * (target_rms / rms)


def codebook_diversity_loss(cb: torch.Tensor) -> torch.Tensor:
    """Penalizes pairwise cosine similarity between distinct codebook vectors (Frobenius distance to identity)."""
    k = cb.shape[0]
    gram = torch.matmul(cb, cb.t())
    off_diag = gram - torch.eye(k, device=cb.device, dtype=cb.dtype)
    return torch.sum(off_diag**2) / (k * (k - 1))


def get_router_hash(
    num_fc0_experts=NUM_FC0_EXPERTS, num_weight_experts=NUM_WEIGHT_EXPERTS, shared=False
):
    router_hash = SHARED_ROUTER_HASH if shared else ROUTER_HASH
    return router_hash ^ ((num_fc0_experts << 16) | num_weight_experts)


def relative_pawns(total, pawns):
    officers = total - pawns
    return (
        0
        if pawns == 0
        else 1
        if pawns <= officers
        else 2
        if pawns <= officers + 2
        else 3
    )


_MATERIAL_STATES = [
    (t, p, h)
    for t in range(16)
    for p in range(min(8, t) + 1)
    for h, minimum in enumerate((1, 3, 1, 3, 0, 1))
    if t - p >= minimum and (h != 4 or t == p)
]
_FC0_MATERIAL_STATES = sorted(
    {(t, relative_pawns(t, p), h) for t, p, h in _MATERIAL_STATES}
)
assert len(_MATERIAL_STATES) == NUM_MATERIAL_STATES
assert len(_FC0_MATERIAL_STATES) == NUM_FC0_MATERIAL_STATES
_MATERIAL_IDS = {state: i for i, state in enumerate(_MATERIAL_STATES)}
_FC0_MATERIAL_IDS = {state: i for i, state in enumerate(_FC0_MATERIAL_STATES)}


def encode_material_state(t, p, h, *, first_layer=False):
    if (t, p, h) not in _MATERIAL_IDS:
        raise ValueError(f"Illegal material state: {t, p, h}")
    if first_layer:
        return _FC0_MATERIAL_IDS[t, relative_pawns(t, p), h]
    return _MATERIAL_IDS[t, p, h]


def decode_material_state(index, *, first_layer=False):
    return (_FC0_MATERIAL_STATES if first_layer else _MATERIAL_STATES)[index]


def make_router_index(ours, theirs, king=None, wings=0):
    if king is None:
        return ours * NUM_FC0_MATERIAL_STATES + theirs
    return (
        (ours * NUM_MATERIAL_STATES + theirs) * WING_STATES + wings
    ) * KING_STATES + king


def decompose_router_index(index, *, first_layer=False, material_states=None):
    if material_states is None:
        material_states = (
            NUM_FC0_MATERIAL_STATES if first_layer else NUM_MATERIAL_STATES
        )
    pair = index if first_layer else index // TAIL_GEOMETRY_STATES
    return (
        pair // material_states,
        pair % material_states,
        torch.zeros_like(index) if first_layer else index % TAIL_GEOMETRY_STATES,
    )


class MaterialEncoder(nn.Module):
    def __init__(self, dim, pawn_states):
        super().__init__()
        self.sides = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Embedding(n, dim)
                        for n in (16, pawn_states, len(HEAVY_CATEGORIES))
                    ]
                )
                for _ in range(2)
            ]
        )
        bound = 1 / math.sqrt(dim)
        for p in self.parameters():
            nn.init.uniform_(p, -bound, bound)

    def forward(self, ours, theirs, table):
        def encode(side, ids):
            features = table[ids].unbind(-1)
            tables = self.sides[side]
            return (
                tables[0](features[0]) + tables[1](features[1]) + tables[2](features[2])
            )

        return encode(0, ours) + encode(1, theirs)


class MoERouter(nn.Module):
    """Full-context routing, with an optional independent material-only FC0 router."""

    def __init__(
        self,
        num_fc0_experts=NUM_FC0_EXPERTS,
        num_weight_experts=NUM_WEIGHT_EXPERTS,
        fc0_dim=FC0_CODEBOOK_DIM,
        dim=CODEBOOK_DIM,
        context_dim=None,
        material_states=None,
        context_commitment_cost=DEFAULT_CONTEXT_COMMITMENT_COST,
        centroid_commitment_cost=DEFAULT_CENTROID_COMMITMENT_COST,
        balance_power=0.0,
        balance_strength=0.25,
        balance_max_range=4.0,
        reseed_dead_experts=True,
        ema_decay=0.99,
        codebook_target_norm=1.0,
        shared=False,
    ):
        super().__init__()
        if not (
            1 < num_fc0_experts <= 64
            and 1 < num_weight_experts <= 256
            and (
                material_states is None
                or 1 <= material_states <= NUM_FC0_MATERIAL_STATES
            )
        ):
            raise ValueError(
                "MoE table counts must be first layer 2–64 and later layers 2–256"
            )
        self.shared = shared
        self.num_fc0_experts, self.num_weight_experts = (
            num_fc0_experts,
            num_weight_experts,
        )
        self.material_states = {
            "fc0": material_states or NUM_FC0_MATERIAL_STATES,
            "tail": material_states
            or (NUM_FC0_MATERIAL_STATES if shared else NUM_MATERIAL_STATES),
        }
        self.num_contexts = self.material_states["tail"] ** 2 * TAIL_GEOMETRY_STATES
        self.context_commitment_cost = context_commitment_cost
        self.centroid_commitment_cost = centroid_commitment_cost
        self.balance_power = balance_power
        self.balance_iterations = ITERATIONS
        self.balance_strength = balance_strength
        self.balance_max_range = balance_max_range
        self.reseed_dead_experts, self.ema_decay = reseed_dead_experts, ema_decay
        self.codebook_target_norm = codebook_target_norm
        context_dim = (context_dim or dim) if shared else dim
        self.adapter_down = (
            nn.Parameter(torch.empty(dim, context_dim)) if shared else None
        )
        if self.adapter_down is not None:
            # Output normalization largely cancels this scale. Under Adam,
            # larger parameters slow angular updates without a separate LR.
            bound = 4 / math.sqrt(context_dim)
            nn.init.uniform_(self.adapter_down, -bound, bound)
        self.encoders = nn.ModuleDict()
        if not shared:
            self.encoders["fc0"] = MaterialEncoder(fc0_dim, 4)
        self.encoders["tail"] = MaterialEncoder(context_dim, 4 if shared else 9)
        self.king = nn.Embedding(KING_STATES, context_dim)
        self.wings = nn.Embedding(WING_STATES, context_dim)
        for emb in (self.king, self.wings):
            nn.init.uniform_(
                emb.weight, -1 / math.sqrt(context_dim), 1 / math.sqrt(context_dim)
            )
        self.codebooks = nn.ParameterDict()
        if not shared:
            self.codebooks["fc0"] = nn.Parameter(torch.empty(num_fc0_experts, fc0_dim))
        self.codebooks["tail"] = nn.Parameter(torch.empty(num_weight_experts, dim))
        for parameter in self.codebooks.values():
            bound = 1 / math.sqrt(parameter.shape[1])
            nn.init.uniform_(parameter, -bound, bound)
        self.register_buffer(
            "fc0_material_decode", torch.tensor(_FC0_MATERIAL_STATES), persistent=False
        )
        self.register_buffer(
            "tail_material_decode",
            torch.tensor(_FC0_MATERIAL_STATES if shared else _MATERIAL_STATES),
            persistent=False,
        )
        for name, cb in self.codebooks.items():
            count = len(cb)
            ranked_target = torch.arange(1, count + 1, dtype=torch.float32).pow(
                -balance_power
            )
            self.register_buffer(
                f"balance_rank_target_{name}",
                ranked_target / ranked_target.sum(),
                persistent=False,
            )
            self.register_buffer(f"ema_traffic_{name}", torch.full((count,), 1 / count))
            self.register_buffer(
                f"balance_steps_{name}", torch.zeros((), dtype=torch.int64)
            )
            self.register_buffer(f"routing_offset_{name}", torch.zeros(count))
            self.register_buffer(f"batch_traffic_{name}", torch.zeros(count))
            self.register_buffer(f"provisional_traffic_{name}", torch.zeros(count))
            self.register_buffer(f"requested_traffic_{name}", torch.zeros(count))
            self.register_buffer(
                f"validation_counts_{name}", torch.zeros(count, dtype=torch.int64)
            )
            self.register_buffer(
                f"validation_unbiased_{name}", torch.zeros(count, dtype=torch.int64)
            )
            self.register_buffer(
                f"offset_adjustment_{name}", torch.zeros(count), persistent=False
            )
            self.register_buffer(
                f"recent_contexts_{name}",
                torch.zeros(2048, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"reseed_observed_{name}", torch.zeros(count), persistent=False
            )

            self.register_buffer(
                f"validation_heavy_{name}",
                torch.zeros(2, count, len(HEAVY_CATEGORIES), dtype=torch.int64),
            )
        for name, cb in self.codebooks.items():
            self.register_buffer(
                f"runner_up_counts_{name}",
                torch.zeros(len(cb), dtype=torch.int64),
                persistent=False,
            )
        self.register_buffer(
            "validation_joint_counts",
            torch.zeros(num_fc0_experts * num_weight_experts, dtype=torch.int64),
            persistent=False,
        )
        self.collect_joint_diagnostics = False
        self.register_buffer(
            "fc0_lut",
            torch.zeros(
                0 if shared else self.material_states["fc0"] ** 2, dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "lut", torch.zeros(self.num_contexts, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "loss_component_sum",
            torch.zeros(len(MOE_LOSS_COMPONENTS)),
            persistent=False,
        )
        self.register_buffer(
            "loss_component_batches", torch.zeros(()), persistent=False
        )
        self.register_buffer("lut_margin_stats", torch.zeros(2, 3), persistent=False)
        self.lut_valid = self.lut_margin_valid = self.collect_lut_diagnostics = False
        self.lut_margin_samples = [0, 0]
        self.last_reseed_details, self.last_reseed_parents = {}, {}
        self.validation_context_samples = {}
        self.recent_context_sizes = {name: 0 for name in self.active_banks}
        self.recent_context_cursors = {name: 0 for name in self.active_banks}
        self.reseed_history = []
        self.routing_diagnostics = {}

    def get_extra_state(self):
        return {
            "routing_diagnostics": self.routing_diagnostics,
            "reseed_history": self.reseed_history,
        }

    def set_extra_state(self, state):
        self.routing_diagnostics = state["routing_diagnostics"]
        self.reseed_history = state["reseed_history"]

    @property
    def shared_routing(self):
        return getattr(self, "shared", False)

    @property
    def active_banks(self):
        return ("tail",) if self.shared_routing else MOE_BANKS

    def bank_label(self, name):
        return "stack" if self.shared_routing else name

    @property
    def loss_component_names(self):
        return tuple(
            f"{self.bank_label(bank)}_{term}"
            for bank in self.active_banks
            for term in ("context_commitment", "centroid_commitment")
        )

    @staticmethod
    def assignment_counts(indices, count, synchronize=False):
        counts = torch.zeros(count, device=indices.device).scatter_add_(
            0, indices, torch.ones_like(indices, dtype=torch.float32)
        )
        world = 1
        if synchronize and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            world = dist.get_world_size()
        return counts, world

    def commitment_losses(self, context, codebook, indices, counts=None, world=1):
        selected = codebook[indices]
        if counts is None:
            counts, _ = self.assignment_counts(indices, len(codebook))
        context_error = (context - selected.detach()).square().sum(-1).mean()
        centroid_error = (selected - context.detach()).square().sum(-1)
        centroid_error = (centroid_error / counts[indices].clamp_min(1)).sum()
        return (
            self.context_commitment_cost * context_error,
            self.centroid_commitment_cost * world / len(codebook) * centroid_error,
        )

    def traffic_momentum(self, name):
        # Bias-corrected EMA: the first batch replaces the uniform placeholder.
        step = getattr(self, f"balance_steps_{name}").clamp_min(1)
        return (
            self.ema_decay
            * (1 - self.ema_decay ** (step - 1))
            / (1 - self.ema_decay**step).clamp_min(1e-12)
        )

    @torch.no_grad()
    def update_traffic(self, name, counts):
        """Record actual corrected global assignments, independently of the solver."""
        getattr(self, f"balance_steps_{name}").add_(1)
        current = counts / counts.sum().clamp_min(1)
        getattr(self, f"batch_traffic_{name}").copy_(current)
        getattr(self, f"ema_traffic_{name}").lerp_(
            current, 1 - self.traffic_momentum(name)
        )
        getattr(self, f"reseed_observed_{name}").add_(counts)

    @torch.no_grad()
    def remember_contexts(self, name, contexts):
        """A small, frequency-weighted ring sample, kept only on rank zero."""
        table = getattr(self, f"recent_contexts_{name}")
        n = min(32, len(contexts))
        sample = contexts[torch.randint(len(contexts), (n,), device=contexts.device)]
        cursor = self.recent_context_cursors[name]
        positions = (torch.arange(n, device=contexts.device) + cursor) % len(table)
        table[positions] = sample
        self.recent_context_cursors[name] = (cursor + n) % len(table)
        self.recent_context_sizes[name] = min(
            len(table), self.recent_context_sizes[name] + n
        )

    def distances(self, context, codebook, name, *, biased=True):
        distances = (
            context.square().sum(-1, keepdim=True)
            + codebook.square().sum(-1)[None]
            - 2 * context @ codebook.T
        )
        if biased:
            distances = distances - getattr(self, f"routing_offset_{name}")
        return distances

    @torch.enable_grad()
    def gradient_diagnostics(self, name, counts, *, model=None, optimizer=None):
        """Bounded validation-only diagnostics; never populate parameter .grad."""
        sample = self.validation_context_samples.get(name)
        if sample is None:
            return None
        with torch.inference_mode(False), ieee_matmul_precision():
            context = self.get_context_vector(sample.clone(), name)
            codebook = self.normalized_codebook(name)
            indices = self.distances(context, codebook, name).argmin(-1)
            context_loss, centroid_loss = self.commitment_losses(
                context, codebook, indices
            )
            parameters = list(self.encoders[name].parameters())
            groups = ["embedding"] * len(parameters)
            if name == "tail":
                parameters.extend((self.king.weight, self.wings.weight))
                groups.extend(("embedding", "embedding"))
            if self.shared_routing:
                parameters.append(self.adapter_down)
                groups.append("projection")
            parameters.append(self.codebooks[name])
            groups.append("codebook")
            scale = 1 / len(self.active_banks)
            objectives = {
                "context": context_loss * scale,
                "centroid": centroid_loss * scale,
            }
            if model is not None:
                objectives["cosine"] = (
                    model.adapter_diversity_cost
                    * scale
                    * codebook_diversity_loss(codebook)
                )
                gram = codebook.new_zeros(())
                for layer, bank, weight_cost in zip(
                    model.layer_stacks.dense_layers(),
                    ("tail" if self.shared_routing else "fc0", "tail", "tail"),
                    (
                        model.weight_diversity_costs[0],
                        model.weight_diversity_costs[1] / 2,
                        model.weight_diversity_costs[1] / 2,
                    ),
                ):
                    if bank == name and weight_cost:
                        weight, _ = layer.get_all_weights_and_biases(codebook)
                        gram = gram + weight_cost * layer.centered_diversity(
                            weight.reshape(layer.count, -1)
                        )
                if gram.requires_grad:
                    objectives["weight_gram"] = gram
            report = {
                "sample_n": len(sample),
                "distance_l2": float(
                    (context - codebook[indices]).square().sum(-1).mean().detach()
                ),
                "selected_cos": float(
                    F.cosine_similarity(context, codebook[indices]).mean().detach()
                ),
                "context_centered_RMS": float(
                    (context - context.mean(0)).square().mean().sqrt().detach()
                ),
            }
            for group in set(groups):
                ps = [p for p, key in zip(parameters, groups) if key == group]
                report[f"{group}_parameter_RMS"] = float(
                    (
                        sum(p.detach().square().sum() for p in ps)
                        / sum(p.numel() for p in ps)
                    ).sqrt()
                )
            for label, objective in objectives.items():
                gradients = torch.autograd.grad(
                    objective, parameters, retain_graph=True, allow_unused=True
                )
                for group in set(groups):
                    gs = [
                        g
                        for g, key in zip(gradients, groups)
                        if key == group and g is not None
                    ]
                    if gs:
                        energy = sum(g.square().sum() for g in gs)
                        report[f"{label}_{group}_norm"] = float(energy.sqrt())
                        report[f"{label}_{group}_RMS"] = float(
                            (energy / sum(g.numel() for g in gs)).sqrt()
                        )
            if optimizer is not None:
                for group in set(groups):
                    scales, epsilons = [], []
                    for parameter, key in zip(parameters, groups):
                        if key != group:
                            continue
                        state = optimizer.state.get(parameter, {})
                        if "exp_avg_sq" not in state:
                            continue
                        config = next(
                            g
                            for g in optimizer.param_groups
                            if any(p is parameter for p in g["params"])
                        )
                        step = state["step"]
                        rms = (
                            state["exp_avg_sq"] / (1 - config["betas"][1] ** step)
                        ).sqrt()
                        scales.append(rms.flatten())
                        epsilons.append(torch.full_like(rms.flatten(), config["eps"]))
                    if scales:
                        rms, eps = torch.cat(scales), torch.cat(epsilons)
                        report[f"adam_{group}_RMS"] = float(rms.square().mean().sqrt())
                        report[f"adam_{group}_below_epsilon"] = float(
                            (rms < eps).float().mean()
                        )
            self.routing_diagnostics[name] = report
            return report

    def normalized_codebook(self, name):
        source = "tail" if self.shared_routing else name
        # Coordinate centering would confine a 64-dimensional shared adapter
        # table to 63 directions before every layer's linear expansion.
        normalize = rms_norm if self.shared_routing else center_and_rms_norm
        return normalize(self.codebooks[source], self.codebook_target_norm)

    @property
    def codebook(self):
        tail = self.normalized_codebook("tail")
        return (tail if self.shared_routing else self.normalized_codebook("fc0"), tail)

    def banks(self, codebook=None):
        vectors = self.codebook if codebook is None else codebook
        for name, normalized in zip(
            self.active_banks, vectors[-len(self.active_banks) :]
        ):
            yield (
                name,
                self.codebooks[name],
                getattr(self, f"ema_traffic_{name}"),
                getattr(self, f"routing_offset_{name}"),
                normalized,
            )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.lut_valid = self.lut_margin_valid = False
        return self

    def get_context_vector(self, index, bank="tail"):
        ours, theirs, king = decompose_router_index(
            index, first_layer=bank == "fc0", material_states=self.material_states[bank]
        )
        raw = self.encoders[bank](
            ours, theirs, getattr(self, f"{bank}_material_decode")
        )
        if bank == "tail":
            raw = raw + self.king(king % KING_STATES) + self.wings(king // KING_STATES)
        normalized = center_and_rms_norm(raw, self.codebook_target_norm)
        if self.shared_routing:
            with ieee_matmul_precision():
                return rms_norm(
                    F.linear(F.gelu(normalized), self.adapter_down),
                    self.codebook_target_norm,
                )
        return normalized

    def quantize(self, z, codebook, name, *, top_k=1, update=True):
        with ieee_matmul_precision():
            # Dispatch has no backward path. Commitment below still trains both sides.
            scores = -self.distances(z, codebook, name, biased=False).detach()
            offsets = getattr(self, f"routing_offset_{name}")
            if update:
                result, world = gather_and_solve(
                    scores,
                    offsets,
                    getattr(self, f"balance_rank_target_{name}"),
                    self.balance_strength,
                    self.balance_max_range,
                )
                new_offsets, counts, before, target = result.unbind(0)
                getattr(self, f"offset_adjustment_{name}").copy_(new_offsets - offsets)
                offsets.copy_(new_offsets)
                getattr(self, f"provisional_traffic_{name}").copy_(
                    before / before.sum().clamp_min(1)
                )
                getattr(self, f"requested_traffic_{name}").copy_(target)
            else:
                world = 1
            biased = scores + offsets
            indices = biased.argmax(-1)
            if not update:
                counts, _ = self.assignment_counts(indices, len(codebook))
            if top_k > 1:
                alternatives = biased.scatter(1, indices[:, None], -float("inf"))
                ranked = torch.topk(alternatives, top_k - 1).indices
                routed = torch.cat((indices[:, None], ranked), dim=1)
            else:
                routed = indices
            context_loss, centroid_loss = self.commitment_losses(
                z, codebook, indices, counts, world
            )
            if update:
                self.update_traffic(name, counts)
            return routed, context_loss, centroid_loss

    @torch.no_grad()
    def reseed_dead_experts_from_traffic(
        self, temperature=0.5, min_relative_utilization=DEFAULT_MIN_RELATIVE_UTILIZATION
    ):
        """Fit proposed splits to a bounded context sample and verify their traffic."""
        self.last_reseed_parents, self.last_reseed_details = {}, {}
        modified_banks = []
        with ieee_matmul_precision():
            for name, raw, traffic, offsets, codebook in self.banks():
                floor = min_relative_utilization / len(raw)
                dead = (traffic < floor).nonzero().flatten().tolist()
                parents, details, modified = {}, {}, set()
                rejected = {"insufficient_sample": 0, "no_spread": 0, "empty_split": 0}
                size = self.recent_context_sizes[name]
                if dead and size:
                    contexts = self.get_context_vector(
                        getattr(self, f"recent_contexts_{name}")[:size], name
                    )
                    assignments = self.distances(contexts, codebook, name).argmin(-1)
                    for child in dead:
                        candidates = (
                            (traffic >= max(floor, 1 / len(raw))).nonzero().flatten()
                        )
                        available = torch.zeros(
                            len(raw), device=raw.device
                        ).scatter_add_(
                            0,
                            assignments,
                            torch.ones_like(assignments, dtype=torch.float32),
                        )
                        # A new child must establish traffic before becoming a donor.
                        if parents:
                            available[list(parents)] = 0
                        candidates = candidates[available[candidates] >= 8]
                        if not len(candidates):
                            rejected["insufficient_sample"] += 1
                            continue
                        logits = traffic[candidates].clamp_min(1e-12).log() / max(
                            temperature, 1e-4
                        )
                        donor = int(candidates[torch.multinomial(logits.softmax(0), 1)])
                        mask = (assignments == donor) | (assignments == child)
                        points = contexts[mask]
                        centered = points - points.mean(0)
                        direction = centered[centered.square().sum(-1).argmax()]
                        for _ in range(2):
                            direction = centered.T @ (
                                centered @ F.normalize(direction, dim=0)
                            )
                        if direction.norm() < 1e-8:
                            rejected["no_spread"] += 1
                            continue
                        projection = centered @ F.normalize(direction, dim=0)
                        side = projection > projection.median()
                        if not side.any() or side.all():
                            rejected["no_spread"] += 1
                            continue
                        centers = F.normalize(
                            torch.stack((points[~side].mean(0), points[side].mean(0))),
                            dim=1,
                        )
                        midpoint = F.normalize(centers.sum(0), dim=0)
                        axis = F.normalize(centers[1] - centers[0], dim=0)
                        # Follow the sampled spread, including very narrow cells.
                        # atan2 remains accurate when cosine rounds to one.
                        angle = torch.atan2(
                            (centers[1] - centers[0]).norm(),
                            (centers[1] + centers[0]).norm(),
                        ).clamp(max=0.25)
                        proposed = torch.stack(
                            (
                                midpoint * angle.cos() - axis * angle.sin(),
                                midpoint * angle.cos() + axis * angle.sin(),
                            )
                        )
                        pair = torch.tensor([donor, child], device=raw.device)
                        old_raw, old_offsets = raw[pair].clone(), offsets[pair].clone()
                        raw[pair] = proposed * raw[donor].norm().clamp_min(1e-6)
                        offsets[child] = offsets[donor]
                        cb = self.normalized_codebook(name)
                        trial = self.distances(contexts, cb, name)
                        winners = trial.argmin(-1)
                        n_parent = int(((winners == donor) & mask).sum())
                        n_child = int(((winners == child) & mask).sum())
                        minimum = max(2, int(len(points) * 0.1))
                        if min(
                            n_parent, n_child
                        ) < minimum or n_parent + n_child < 0.8 * len(points):
                            raw[pair], offsets[pair] = old_raw, old_offsets
                            rejected["empty_split"] += 1
                            continue
                        old_traffic = float(traffic[child])
                        mass = traffic[donor] + traffic[child]
                        share = n_child / (n_parent + n_child)
                        traffic[child], traffic[donor] = (
                            mass * share,
                            mass * (1 - share),
                        )
                        parents[child] = donor
                        details[child] = {
                            "traffic": old_traffic,
                            "floor": floor,
                            "sample_parent": n_parent,
                            "sample_child": n_child,
                            "child_share": share,
                            "separation_radians": float(2 * angle),
                        }
                        modified.update((child, donor))
                        assignments = winners
                elif dead:
                    rejected["insufficient_sample"] = len(dead)
                self.last_reseed_parents[name], self.last_reseed_details[name] = (
                    parents,
                    details,
                )
                observed = getattr(self, f"reseed_observed_{name}")
                previous = getattr(self, f"previous_reseed_children_{name}", [])
                recovery = observed / observed.sum().clamp_min(1)
                summary = {
                    "bank": self.bank_label(name),
                    "accepted": len(parents),
                    "rejected": rejected,
                    "previous_children": len(previous),
                    "previous_zero": int((observed[previous] == 0).sum())
                    if previous
                    else 0,
                    "previous_min_share": float(recovery[previous].min())
                    if previous
                    else 0.0,
                }
                self.reseed_history.append(summary)
                self.reseed_history = self.reseed_history[-256:]
                setattr(self, f"previous_reseed_children_{name}", list(parents))
                observed.zero_()
                modified_banks.append(sorted(modified))
        if any(modified_banks):
            self.lut_valid = self.lut_margin_valid = False
        return tuple(modified_banks)

    @torch.no_grad()
    def update_lookup_table(self, chunk_size=8192):
        # Generate the large tail table in bounded chunks.
        self.lut_margin_valid = False
        with ieee_matmul_precision():
            for name in self.active_banks:
                row = MOE_BANKS.index(name)
                table = self.fc0_lut if name == "fc0" else self.lut
                cb = self.normalized_codebook(name)
                margins = []
                for start in range(0, len(table), chunk_size):
                    index = torch.arange(
                        start, min(start + chunk_size, len(table)), device=table.device
                    )
                    z = self.get_context_vector(index, name)
                    dist = self.distances(z, cb, name)
                    table[start : start + len(index)] = dist.argmin(-1)
                    if self.collect_lut_diagnostics:
                        near = dist[::256].topk(2, largest=False).values
                        margins.append(near[:, 1] - near[:, 0])
                if margins:
                    gaps = torch.cat(margins)
                    self.lut_margin_stats[row].copy_(
                        torch.stack(
                            [gaps.min(), gaps.mean(), (gaps <= 1e-6).float().mean()]
                        )
                    )
                    self.lut_margin_samples[row] = len(gaps)
                    self.lut_margin_valid = True
        self.lut_valid = True
        return self.fc0_lut, self.lut

    def get_lookup_table(self):
        if not self.lut_valid:
            self.update_lookup_table()
        return self.fc0_lut, self.lut

    @torch.no_grad()
    def load_lookup_tables(self, fc0, tail):
        for source, target, count in [
            (fc0, self.fc0_lut, self.num_fc0_experts),
            (tail, self.lut, self.num_weight_experts),
        ]:
            source = torch.as_tensor(source, device=target.device, dtype=torch.long)
            if (
                source.shape != target.shape
                or torch.any(source < 0)
                or torch.any(source >= count)
            ):
                raise ValueError("Invalid MoE lookup table")
            target.copy_(source)
        self.lut_valid = True
        self.lut_margin_valid = False

    @torch.no_grad()
    def export_permutations(self):
        orders = {
            name: torch.argsort(
                getattr(self, f"validation_counts_{name}"), descending=True, stable=True
            )
            for name in self.active_banks
        }
        if self.shared_routing:
            orders["fc0"] = orders["tail"]
        return orders

    @torch.no_grad()
    def export_lookup_tables(self, orders):
        tables = self.get_lookup_table()
        return tuple(
            torch.argsort(orders[name])[table] for name, table in zip(MOE_BANKS, tables)
        )

    def forward(
        self,
        c_fc0,
        c_tail,
        fake_quantize=False,
        *,
        mix_experts=False,
        codebook=None,
    ):
        if self.shared_routing:
            # The loader provides the relative-pawn material pair in c_fc0
            # and the wing/king suffix in c_tail.
            c_tail = c_fc0 * TAIL_GEOMETRY_STATES + c_tail % TAIL_GEOMETRY_STATES
        if self.training:
            ids, parts = [], []
            bank_scale = 1.0 / len(self.active_banks)
            for name, _, _, _, cb in self.banks(codebook):
                context = c_fc0 if name == "fc0" else c_tail
                z = self.get_context_vector(context, name)
                top_k = (2 if name == "fc0" else 4) if mix_experts else 1
                index, context_loss, centroid_loss = self.quantize(
                    z, cb, name, top_k=top_k
                )
                if mix_experts:
                    with torch.no_grad():
                        alternates = index[:, 1:].flatten()
                        counts = getattr(self, f"runner_up_counts_{name}")
                        counts.scatter_add_(0, alternates, torch.ones_like(alternates))
                ids.append(index)
                parts.extend((context_loss * bank_scale, centroid_loss * bank_scale))
            parts = torch.stack(parts)
            with torch.no_grad():
                self.loss_component_sum[-len(parts) :].add_(parts.detach())
                self.loss_component_batches.add_(1)
            self.lut_valid = self.lut_margin_valid = False
            if self.shared_routing:
                ids = ids * 2
            return *ids, parts.sum(), parts[::2].sum(), parts[1::2].sum()
        fc0_table, tail_table = self.get_lookup_table()
        tail = tail_table[c_tail.long()]
        ids = (tail if self.shared_routing else fc0_table[c_fc0.long()], tail)
        active_ids = ids[-len(self.active_banks) :]
        if self.collect_lut_diagnostics:
            contexts = (c_tail,) if self.shared_routing else (c_fc0, c_tail)
            for name, context in zip(self.active_banks, contexts):
                if name not in self.validation_context_samples:
                    self.validation_context_samples[name] = (
                        context[:512].detach().clone()
                    )
            with ieee_matmul_precision():
                for name, context in zip(self.active_banks, contexts):
                    z = self.get_context_vector(context[:512], name)
                    cb = self.normalized_codebook(name)
                    unbiased = self.distances(z, cb, name, biased=False).argmin(-1)
                    counts = getattr(self, f"validation_unbiased_{name}")
                    counts.scatter_add_(0, unbiased, torch.ones_like(unbiased))
            for name, index in zip(self.active_banks, active_ids):
                count = getattr(self, f"validation_counts_{name}")
                count.scatter_add_(0, index, torch.ones_like(index, dtype=count.dtype))
        if self.collect_joint_diagnostics:
            if not self.shared_routing:
                pair = ids[0] * self.num_weight_experts + ids[1]
                self.validation_joint_counts.scatter_add_(
                    0, pair, torch.ones_like(pair)
                )
            # The full context encodes the heavy categories for both sides.
            ours, theirs, _ = decompose_router_index(
                c_tail, material_states=self.material_states["tail"]
            )
            for name, index in zip(self.active_banks, active_ids):
                counts = getattr(self, f"validation_heavy_{name}")
                for side, material in enumerate((ours, theirs)):
                    key = (
                        index * len(HEAVY_CATEGORIES)
                        + self.tail_material_decode[material, 2]
                    )
                    counts[side].view(-1).scatter_add_(0, key, torch.ones_like(key))
        zero = torch.zeros((), device=c_fc0.device)
        return *ids, zero, zero, zero
