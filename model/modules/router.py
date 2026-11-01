import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


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

    prev_precision = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


NUM_MATERIAL_STATES = 676
KING_STATES = 8
FC0_ROUTER_TABLE_SIZE = NUM_MATERIAL_STATES**2
ROUTER_TABLE_SIZE = FC0_ROUTER_TABLE_SIZE * KING_STATES
NUM_FC0_EXPERTS = 64
NUM_WEIGHT_EXPERTS = 512
FC0_CODEBOOK_DIM = 256
CODEBOOK_DIM = 2048
FC0_ADAPTER_RANK = 64
ADAPTER_RANK = 512
ROUTER_HASH = 0x4D6F4538
DEFAULT_CODEBOOK_TARGET_NORM = 1.0
DEFAULT_MIN_RELATIVE_UTILIZATION = 0.064
MOE_BANKS = ("fc0", "tail")
MOE_LOSS_COMPONENTS = tuple(
    f"{bank}_{term}"
    for bank in MOE_BANKS
    for term in ("commitment", "balance", "diversity")
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
    num_fc0_experts=NUM_FC0_EXPERTS, num_weight_experts=NUM_WEIGHT_EXPERTS
):
    return ROUTER_HASH ^ ((num_fc0_experts << 16) | num_weight_experts)


_MATERIAL_STATES = [
    (t, p, h, wide)
    for t in range(16)
    for p in range(min(8, t) + 1)
    for h in range(4)
    if t - p >= (1 if h < 2 else 2 if h == 2 else 0)
    for wide in range(2 if p >= 2 else 1)
]
assert len(_MATERIAL_STATES) == NUM_MATERIAL_STATES
_MATERIAL_IDS = {state: i for i, state in enumerate(_MATERIAL_STATES)}
_MATERIAL_DECODE_TABLE = torch.tensor(_MATERIAL_STATES, dtype=torch.long)


def encode_material_state(t, p, h, wide=False):
    try:
        return _MATERIAL_IDS[t, p, h, int(wide)]
    except KeyError:
        raise ValueError(f"Illegal material/layout state: {t, p, h, wide}") from None


def decode_material_state(index):
    return _MATERIAL_STATES[index]


def make_router_index(ours, theirs, king=None):
    pair = ours * NUM_MATERIAL_STATES + theirs
    return pair if king is None else pair * KING_STATES + king


def decompose_router_index(
    index, *, first_layer=False, material_states=NUM_MATERIAL_STATES
):
    pair = (
        index if first_layer else torch.div(index, KING_STATES, rounding_mode="floor")
    )
    return (
        torch.div(pair, material_states, rounding_mode="floor"),
        pair % material_states,
        torch.zeros_like(index) if first_layer else index % KING_STATES,
    )


class MaterialEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sides = nn.ModuleList(
            [
                nn.ModuleList([nn.Embedding(n, dim) for n in (16, 9, 4, 2)])
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
                tables[0](features[0])
                + tables[1](features[1])
                + tables[2](features[2])
                + tables[3](features[3])
            )

        return encode(0, ours) + encode(1, theirs)


class MoERouter(nn.Module):
    """Separate material-only FC0 and material/three-bit-king tail routers."""

    def __init__(
        self,
        num_fc0_experts=NUM_FC0_EXPERTS,
        num_weight_experts=NUM_WEIGHT_EXPERTS,
        fc0_dim=FC0_CODEBOOK_DIM,
        dim=CODEBOOK_DIM,
        material_states=NUM_MATERIAL_STATES,
        commitment_cost=0.005,
        balance_cost=0.001,
        temperature=0.2,
        diversity_cost=0.01,
        reseed_dead_experts=True,
        ema_decay=0.998,
        codebook_target_norm=1.0,
    ):
        super().__init__()
        if not (
            1 < num_fc0_experts <= 64
            and 1 < num_weight_experts <= 512
            and 1 <= material_states <= NUM_MATERIAL_STATES
        ):
            raise ValueError("MoE8 supports 64 FC0 and 512 tail experts")
        self.num_fc0_experts, self.num_weight_experts = (
            num_fc0_experts,
            num_weight_experts,
        )
        self.material_states = material_states
        self.num_contexts = material_states**2 * KING_STATES
        self.commitment_cost, self.balance_cost = commitment_cost, balance_cost
        self.temperature, self.diversity_cost = temperature, diversity_cost
        self.reseed_dead_experts, self.ema_decay = reseed_dead_experts, ema_decay
        self.codebook_target_norm = codebook_target_norm
        self.encoders = nn.ModuleDict(
            {"fc0": MaterialEncoder(fc0_dim), "tail": MaterialEncoder(dim)}
        )
        self.king = nn.Embedding(KING_STATES, dim)
        nn.init.uniform_(self.king.weight, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        self.codebooks = nn.ParameterDict(
            {
                "fc0": nn.Parameter(torch.empty(num_fc0_experts, fc0_dim)),
                "tail": nn.Parameter(torch.empty(num_weight_experts, dim)),
            }
        )
        for parameter in self.codebooks.values():
            bound = 1 / math.sqrt(parameter.shape[1])
            nn.init.uniform_(parameter, -bound, bound)
        self.register_buffer(
            "material_decode_table", _MATERIAL_DECODE_TABLE, persistent=False
        )
        for name, cb in self.codebooks.items():
            count = len(cb)
            self.register_buffer(
                f"ema_traffic_{name}", torch.full((count,), 1 / count), persistent=False
            )
            self.register_buffer(
                f"ema_soft_{name}", torch.full((count,), 1 / count), persistent=False
            )
            self.register_buffer(f"validation_counts_{name}", torch.zeros(count))
        self.register_buffer(
            "fc0_lut",
            torch.zeros(material_states**2, dtype=torch.long),
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

    def normalized_codebook(self, name):
        return center_and_rms_norm(self.codebooks[name], self.codebook_target_norm)

    @property
    def codebook(self):
        return tuple(self.normalized_codebook(name) for name in MOE_BANKS)

    def banks(self):
        for name in MOE_BANKS:
            yield (
                name,
                self.codebooks[name],
                getattr(self, f"ema_traffic_{name}"),
                getattr(self, f"ema_soft_{name}"),
                self.normalized_codebook(name),
            )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.lut_valid = self.lut_margin_valid = False
        return self

    def get_context_vector(self, index, bank="tail"):
        ours, theirs, king = decompose_router_index(
            index, first_layer=bank == "fc0", material_states=self.material_states
        )
        raw = self.encoders[bank](ours, theirs, self.material_decode_table)
        if bank == "tail":
            raw = raw + self.king(king)
        return center_and_rms_norm(raw, self.codebook_target_norm)

    def quantize(
        self,
        z: torch.Tensor,
        codebook: torch.Tensor,
        num_experts: int,
        temperature: float | None = None,
        traffic_ema: torch.Tensor | None = None,
        soft_ema: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with ieee_matmul_precision():
            z_sq = torch.sum(z**2, dim=-1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=-1).unsqueeze(0)
            dot = torch.matmul(z, codebook.t())
            distances = z_sq + c_sq - 2.0 * dot

            indices = torch.argmin(distances, dim=-1)
            z_q = F.embedding(indices, codebook)

            # Commitment loss
            vq_loss = 0.25 * F.mse_loss(z, z_q.detach()) + F.mse_loss(z_q, z.detach())
            commit_loss = self.commitment_cost * vq_loss

            # Load balancing auxiliary loss
            k_exp = num_experts
            one_hot = F.one_hot(indices, num_classes=k_exp).float()
            f = one_hot.mean(dim=0).detach()

            temperature = self.temperature if temperature is None else temperature
            scaled_neg_dist = -distances / max(temperature, 1e-4)
            probs = F.softmax(scaled_neg_dist, dim=-1)
            prob_mean = probs.mean(dim=0)

            balance_loss = k_exp * torch.sum(f * prob_mean) - 1.0
            if traffic_ema is not None and soft_ema is not None:
                with torch.no_grad():
                    traffic_ema.lerp_(f, 1.0 - self.ema_decay)
                    soft_ema.lerp_(prob_mean.detach(), 1.0 - self.ema_decay)

            # Straight-Through Estimator
            z_q_ste = z + (z_q - z).detach()

            return indices, z_q_ste, commit_loss, balance_loss

    @torch.no_grad()
    def reseed_dead_experts_from_traffic(
        self,
        temperature: float = 0.5,
        perturbation_scale: float = 0.02,
        min_relative_utilization: float = DEFAULT_MIN_RELATIVE_UTILIZATION,
    ) -> tuple[list[int], list[int]]:
        """Split active cells only after sustained low hard-traffic EMA.

        No context enumeration or LUT construction is needed. A temporarily
        empty Voronoi cell must not bypass the traffic EMA's grace period.
        """
        self.last_reseed_parents = {}
        self.last_reseed_details = {}
        reseeded = []
        for name, codebook, traffic, _, _ in self.banks():
            count = traffic.numel()
            floor = min_relative_utilization / count
            dead = torch.nonzero(traffic < floor, as_tuple=True)[0]
            self.last_reseed_details[name] = {
                idx: {"traffic": value, "floor": floor}
                for idx, value in zip(dead.tolist(), traffic[dead].cpu().tolist())
            }
            parents = {}
            modified = set()
            if dead.numel():
                # Prefer busy cells; fall back to any healthy cell. Never use
                # a dead child as a donor if all traffic estimates are zero.
                active = torch.nonzero(
                    (traffic >= 1.0 / count - 1e-5)
                    & (traffic >= floor)
                    & (traffic > 0),
                    as_tuple=True,
                )[0]
                if not active.numel():
                    active = torch.nonzero(
                        (traffic >= floor) & (traffic > 0), as_tuple=True
                    )[0]
                if active.numel():
                    logits = traffic[active].clamp_min(1e-8).log() / max(
                        temperature, 1e-4
                    )
                    probabilities = F.softmax(logits, dim=0)
                    for child in dead.tolist():
                        donor_pos = torch.multinomial(probabilities, 1).item()
                        donor = active[donor_pos].item()
                        direction = torch.randn_like(codebook[child])
                        offset = (
                            direction
                            / direction.norm().clamp_min(1e-8)
                            * perturbation_scale
                        )
                        original = codebook[donor].clone()
                        codebook[child].copy_(original + offset)
                        codebook[donor].copy_(original - offset)
                        traffic[donor] *= 0.5
                        traffic[child] = traffic[donor]
                        probabilities[donor_pos] *= 0.5
                        probabilities /= probabilities.sum().clamp_min(1e-8)
                        parents[child] = donor
                        modified.update((child, donor))
            self.last_reseed_parents[name] = parents
            reseeded.append(sorted(modified))

        if any(reseeded):
            self.lut_valid = False
            self.lut_margin_valid = False
        return tuple(reseeded)

    @torch.no_grad()
    def update_lookup_table(self, chunk_size=8192):
        # Never materialize the 3.65M x 2048 context tensor.
        self.lut_margin_valid = False
        with ieee_matmul_precision():
            for row, name in enumerate(MOE_BANKS):
                table = self.fc0_lut if name == "fc0" else self.lut
                cb = self.normalized_codebook(name)
                cb_sq = cb.square().sum(-1)[None]
                margins = []
                for start in range(0, len(table), chunk_size):
                    index = torch.arange(
                        start, min(start + chunk_size, len(table)), device=table.device
                    )
                    z = self.get_context_vector(index, name)
                    dist = z.square().sum(-1, keepdim=True) + cb_sq - 2 * z @ cb.T
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
                raise ValueError("Invalid MoE8 lookup table")
            target.copy_(source)
        self.lut_valid = True
        self.lut_margin_valid = False

    @torch.no_grad()
    def export_permutations(self):
        return {
            name: torch.argsort(
                getattr(self, f"validation_counts_{name}"), descending=True, stable=True
            )
            for name in MOE_BANKS
        }

    @torch.no_grad()
    def export_lookup_tables(self, orders):
        tables = self.get_lookup_table()
        return tuple(
            torch.argsort(orders[name])[table] for name, table in zip(MOE_BANKS, tables)
        )

    def forward(self, c_fc0, c_tail, fake_quantize=False):
        if self.training:
            ids, parts = [], []
            for name, _, hard, soft, cb in self.banks():
                context = c_fc0 if name == "fc0" else c_tail
                z = self.get_context_vector(context, name)
                index, _, commit, balance = self.quantize(
                    z, cb, len(cb), self.temperature, hard, soft
                )
                ids.append(index)
                parts.extend(
                    [
                        commit * 0.5,
                        balance * self.balance_cost * 0.5,
                        codebook_diversity_loss(cb) * self.diversity_cost * 0.5,
                    ]
                )
            parts = torch.stack(parts)
            with torch.no_grad():
                self.loss_component_sum.add_(parts.detach())
                self.loss_component_batches.add_(1)
            self.lut_valid = self.lut_margin_valid = False
            return *ids, parts.sum(), parts[::3].sum(), parts[1::3].sum()
        tables = self.get_lookup_table()
        ids = tuple(
            table[index.long()] for table, index in zip(tables, (c_fc0, c_tail))
        )
        if self.collect_lut_diagnostics:
            for name, index in zip(MOE_BANKS, ids):
                count = getattr(self, f"validation_counts_{name}")
                count.scatter_add_(0, index, torch.ones_like(index, dtype=count.dtype))
        zero = torch.zeros((), device=c_fc0.device)
        return *ids, zero, zero, zero
