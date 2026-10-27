import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager


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


NUM_MATERIAL_STATES = 396
NUM_OUR_KINGS = 33
NUM_ENEMY_KINGS = 65
ROUTER_TABLE_SIZE = NUM_MATERIAL_STATES * NUM_OUR_KINGS * NUM_ENEMY_KINGS  # 849,420
NUM_WEIGHT_EXPERTS = 64
NUM_BIAS_EXPERTS = 128
NUM_BIASES = 64
BIAS_RANK = 32
CODEBOOK_DIM = 256
ROUTER_HASH = 0x4D6F3635  # 'Mo65': dual activation biases and L2 scale 128
DEFAULT_CODEBOOK_TARGET_NORM = 1.0
# Weighted contributions, ordered by router, for inexpensive epoch accumulation.
MOE_BANKS = ("weights_stm", "biases_stm", "weights_opp", "biases_opp")
MOE_LOSS_COMPONENTS = tuple(
    f"{bank}_{term}"
    for bank in MOE_BANKS
    for term in ("commitment", "balance", "diversity")
)
DEFAULT_MIN_RELATIVE_UTILIZATION = 0.064


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
    num_weight_experts: int = NUM_WEIGHT_EXPERTS,
    num_bias_experts: int = NUM_BIAS_EXPERTS,
) -> int:
    """Combines the architecture tag and expert counts into a 32-bit router hash."""
    return ROUTER_HASH ^ ((num_bias_experts << 16) | num_weight_experts)


# Shared enumeration with the loader and Stockfish: queen, rook, bishop pair, else.
_MATERIAL_STATES = [
    (t, p, h)
    for t in range(16)
    for p in range(min(8, t) + 1)
    for h in range(4)
    if t - p >= (1 if h < 2 else 2 if h == 2 else 0)
]
assert len(_MATERIAL_STATES) == NUM_MATERIAL_STATES
_MATERIAL_ENCODE_TABLE = torch.full((16, 9, 4), -1, dtype=torch.long)
for _idx, (_t, _p, _h) in enumerate(_MATERIAL_STATES):
    _MATERIAL_ENCODE_TABLE[_t, _p, _h] = _idx
_MATERIAL_DECODE_TABLE = torch.tensor(_MATERIAL_STATES, dtype=torch.long)


def encode_material_state(t: int, p: int, h: int) -> int:
    if not (0 <= t < 16 and 0 <= p <= min(8, t) and 0 <= h < 4):
        raise ValueError(f"Illegal material state: T={t}, P={p}, H={h}")
    mat_id = int(_MATERIAL_ENCODE_TABLE[t, p, h].item())
    if mat_id < 0:
        raise ValueError(f"Illegal material state: T={t}, P={p}, H={h}")
    return mat_id


def decode_material_state(mat_id: int) -> tuple[int, int, int]:
    return _MATERIAL_STATES[mat_id]


def make_router_index(mat_id: int, our_king: int, enemy_king: int) -> int:
    """Computes the 1D context index in [0, 849419]."""
    return (mat_id * NUM_OUR_KINGS + our_king) * NUM_ENEMY_KINGS + enemy_king


def decompose_router_index(
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes context index c into (mat_id, our_king, enemy_king)."""
    enemy_king = c % NUM_ENEMY_KINGS
    rem1 = torch.div(c, NUM_ENEMY_KINGS, rounding_mode="floor")
    our_king = rem1 % NUM_OUR_KINGS
    mat_id = torch.div(rem1, NUM_OUR_KINGS, rounding_mode="floor")
    return mat_id, our_king, enemy_king


class StructuredKingEmbedding(nn.Module):
    """
    Structured spatial embedding for our king and enemy king.
    Encodes:
      - Symmetric single-king geometry:
        - Our king: distance to home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
        - Enemy king: distance to enemy home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
      - 3-part relative interaction:
        - Vertical distance delta_r (0..7)
        - Horizontal distance delta_f (0..7)
        - Chebyshev opposition distance (2..7 mapped to 0..5, saving impossible 0 and 1)
    """

    def __init__(self, dim: int = CODEBOOK_DIM):
        super().__init__()
        self.dim = dim

        self.embed_our_rank = nn.Embedding(8, dim)
        self.embed_our_edge = nn.Embedding(4, dim)
        self.embed_our_castling = nn.Embedding(2, dim)

        self.embed_enemy_rank = nn.Embedding(8, dim)
        self.embed_enemy_edge = nn.Embedding(4, dim)
        self.embed_enemy_castling = nn.Embedding(2, dim)

        self.embed_delta_r = nn.Embedding(8, dim)
        self.embed_delta_f = nn.Embedding(8, dim)
        self.embed_chebyshev = nn.Embedding(6, dim)

        # Precompute lookup tables
        our_rank = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        our_edge = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        our_castling = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        for k in range(NUM_OUR_KINGS):
            if k == 32:
                our_rank[k] = 0
                our_edge[k] = 3
                our_castling[k] = 1
            else:
                our_rank[k] = k // 4
                our_edge[k] = k % 4
                our_castling[k] = 0

        enemy_rank = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        enemy_edge = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        enemy_castling = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        for k in range(NUM_ENEMY_KINGS):
            if k == 64:
                enemy_rank[k] = 0
                enemy_edge[k] = 3
                enemy_castling[k] = 1
            else:
                r = k // 8
                f = k % 8
                enemy_rank[k] = 7 - r
                enemy_edge[k] = min(f, 7 - f)
                enemy_castling[k] = 0

        delta_r = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        delta_f = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        chebyshev = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        for k_us in range(NUM_OUR_KINGS):
            r_us = 0 if k_us == 32 else k_us // 4
            f_us = 3 if k_us == 32 else k_us % 4
            for k_them in range(NUM_ENEMY_KINGS):
                r_them = 7 if k_them == 64 else k_them // 8
                f_them = 4 if k_them == 64 else k_them % 8
                dr = abs(r_us - r_them)
                df = min(abs(f_us - f_them), abs((7 - f_us) - f_them))
                d_cheb = max(dr, df)
                delta_r[k_us, k_them] = dr
                delta_f[k_us, k_them] = df
                # Legal chess distance between kings is >= 2. Clamping to min 2 and subtracting 2 yields index 0..5.
                chebyshev[k_us, k_them] = max(2, d_cheb) - 2

        self.register_buffer("our_rank", our_rank)
        self.register_buffer("our_edge", our_edge)
        self.register_buffer("our_castling", our_castling)
        self.register_buffer("enemy_rank", enemy_rank)
        self.register_buffer("enemy_edge", enemy_edge)
        self.register_buffer("enemy_castling", enemy_castling)
        self.register_buffer("delta_r", delta_r)
        self.register_buffer("delta_f", delta_f)
        self.register_buffer("chebyshev", chebyshev)
        self.init_weights()

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.embed_our_rank.weight, -val, val)
        nn.init.uniform_(self.embed_our_edge.weight, -val, val)
        nn.init.uniform_(self.embed_our_castling.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_rank.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_edge.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_castling.weight, -val, val)
        nn.init.uniform_(self.embed_delta_r.weight, -val, val)
        nn.init.uniform_(self.embed_delta_f.weight, -val, val)
        nn.init.uniform_(self.embed_chebyshev.weight, -val, val)

    def forward(self, k_us: torch.Tensor, k_them: torch.Tensor) -> torch.Tensor:
        z_our = (
            self.embed_our_rank(self.our_rank[k_us])
            + self.embed_our_edge(self.our_edge[k_us])
            + self.embed_our_castling(self.our_castling[k_us])
        )
        z_enemy = (
            self.embed_enemy_rank(self.enemy_rank[k_them])
            + self.embed_enemy_edge(self.enemy_edge[k_them])
            + self.embed_enemy_castling(self.enemy_castling[k_them])
        )
        z_rel = (
            self.embed_delta_r(self.delta_r[k_us, k_them])
            + self.embed_delta_f(self.delta_f[k_us, k_them])
            + self.embed_chebyshev(self.chebyshev[k_us, k_them])
        )
        return z_our + z_enemy + z_rel


class DualVQCodebookRouter(nn.Module):
    """Four independent role-specific routers sharing material/king embeddings.

    LUT bytes: STM weights, OPP weights, STM biases, OPP biases.
    """

    def __init__(
        self,
        num_contexts: int = ROUTER_TABLE_SIZE,
        num_weight_experts: int = NUM_WEIGHT_EXPERTS,
        num_bias_experts: int = NUM_BIAS_EXPERTS,
        dim: int = CODEBOOK_DIM,
        bias_rank: int = BIAS_RANK,
        num_biases: int = NUM_BIASES,
        commitment_cost: float = 0.005,
        balance_cost: float = 0.001,
        temperature: float = 0.2,
        diversity_cost: float = 0.01,
        reseed_dead_experts: bool = True,
        ema_decay: float = 0.998,
        quantization: QuantizationManager | None = None,
        codebook_target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
        bias_balance_cost: float = 0.004,
        bias_temperature: float = 0.2,
        bias_diversity_cost: float = 0.02,
        hidden_sizes: tuple[int, int] = (16, 16),
    ):
        super().__init__()
        if not (1 <= num_weight_experts <= 64 and 1 <= num_bias_experts <= 128):
            raise ValueError("MoE6 supports 64 weights and 128 biases per role")
        self.num_contexts = num_contexts
        self.num_weight_experts = num_weight_experts
        self.num_bias_experts = num_bias_experts
        self.dim = dim
        self.bias_rank = bias_rank
        if num_biases != 2 * sum(hidden_sizes):
            raise ValueError(
                "Dynamic biases must match the two half-width hidden layers"
            )
        self.hidden_sizes = hidden_sizes
        self.num_biases = num_biases
        self.commitment_cost = commitment_cost
        self.balance_cost = balance_cost
        self.bias_balance_cost = bias_balance_cost
        self.temperature = temperature
        self.bias_temperature = bias_temperature
        self.diversity_cost = diversity_cost
        self.bias_diversity_cost = bias_diversity_cost
        self.reseed_dead_experts = reseed_dead_experts
        self.ema_decay = ema_decay
        self.quantization = quantization
        self.codebook_target_norm = codebook_target_norm

        # Material feature embedding tables
        self.embed_T = nn.Embedding(16, dim)
        self.embed_P = nn.Embedding(9, dim)
        self.embed_H = nn.Embedding(4, dim)

        # Structured spatial King embedding table
        self.king_embed = StructuredKingEmbedding(dim=dim)

        # STM weight codebook; the OPP bank is independent.
        self.codebook_weights = nn.Parameter(torch.empty(num_weight_experts, dim))

        # STM bias codebook: 128 experts in R^256.
        self.codebook_biases = nn.Parameter(torch.empty(num_bias_experts, dim))
        self.bias_head_down = nn.Linear(dim, bias_rank, bias=False)
        self.bias_head_up = nn.Linear(bias_rank, num_biases, bias=True)

        self.codebook_weights_opp = nn.Parameter(torch.empty(num_weight_experts, dim))
        self.codebook_biases_opp = nn.Parameter(torch.empty(num_bias_experts, dim))
        self.bias_head_down_opp = nn.Linear(dim, bias_rank, bias=False)
        self.bias_head_up_opp = nn.Linear(bias_rank, num_biases, bias=True)

        # Material decoding buffer on device
        self.register_buffer(
            "material_decode_table", _MATERIAL_DECODE_TABLE, persistent=False
        )

        for name, count in zip(MOE_BANKS, (num_weight_experts, num_bias_experts) * 2):
            self.register_buffer(
                f"ema_traffic_{name}",
                torch.full((count,), 1.0 / count),
                persistent=False,
            )
            self.register_buffer(
                f"ema_soft_{name}", torch.full((count,), 1.0 / count), persistent=False
            )
            self.register_buffer(
                f"validation_counts_{name}", torch.zeros(count), persistent=False
            )
        self.register_buffer("loss_component_sum", torch.zeros(12), persistent=False)
        self.register_buffer(
            "loss_component_batches", torch.zeros(()), persistent=False
        )

        # Ephemeral runtime cache for inference & Stockfish export
        self.register_buffer(
            "lut",
            torch.zeros(num_contexts, dtype=torch.int64),
            persistent=False,
        )
        self.lut_valid: bool = False
        self.collect_lut_diagnostics: bool = False
        self.lut_margin_valid: bool = False
        self.lut_margin_samples: int = 0
        self.register_buffer("lut_margin_stats", torch.zeros(4, 3), persistent=False)
        self.register_buffer(
            "coalesced_biases",
            torch.zeros(2, num_bias_experts, num_biases, dtype=torch.float32),
            persistent=False,
        )
        self.coalesced_valid: bool = False

        self.last_reseed_details: dict[str, dict[int, dict]] = {}
        self.last_reseed_parents: dict[str, dict[int, int]] = {}

        self.init_weights()

    @property
    def codebook_w(self) -> torch.Tensor:
        """Zero-mean RMS-normalized weight codebook (single source of truth)."""
        return center_and_rms_norm(self.codebook_weights, self.codebook_target_norm)

    @property
    def codebook_b(self) -> torch.Tensor:
        """Zero-mean RMS-normalized bias codebook (single source of truth)."""
        return center_and_rms_norm(self.codebook_biases, self.codebook_target_norm)

    @property
    def codebook(self):
        return self.codebook_w, center_and_rms_norm(
            self.codebook_weights_opp, self.codebook_target_norm
        )

    def banks(self):
        for name, parameter in zip(
            MOE_BANKS,
            (
                self.codebook_weights,
                self.codebook_biases,
                self.codebook_weights_opp,
                self.codebook_biases_opp,
            ),
        ):
            yield (
                name,
                parameter,
                getattr(self, f"ema_traffic_{name}"),
                getattr(self, f"ema_soft_{name}"),
                center_and_rms_norm(parameter, self.codebook_target_norm),
            )

    def bias_head(self, emb, role=0):
        down = self.bias_head_down if role == 0 else self.bias_head_down_opp
        up = self.bias_head_up if role == 0 else self.bias_head_up_opp
        return up(rms_norm(down(F.gelu(emb)), target_norm=1.0))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.lut_valid = False
            self.lut_margin_valid = False
            self.coalesced_valid = False
        return self

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.embed_T.weight, -val, val)
        nn.init.uniform_(self.embed_P.weight, -val, val)
        nn.init.uniform_(self.embed_H.weight, -val, val)
        nn.init.uniform_(self.codebook_weights, -val, val)
        nn.init.uniform_(self.codebook_biases, -val, val)

        nn.init.uniform_(self.codebook_weights_opp, -val, val)
        nn.init.uniform_(self.codebook_biases_opp, -val, val)
        for down, up in (
            (self.bias_head_down, self.bias_head_up),
            (self.bias_head_down_opp, self.bias_head_up_opp),
        ):
            nn.init.uniform_(down.weight, -val, val)
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)

    def get_context_vector(self, c: torch.Tensor) -> torch.Tensor:
        """Computes, mean-centers, and RMS-normalizes context vector for batch of context indices c."""
        mat_id, k_us, k_them = decompose_router_index(c)
        mat_feats = self.material_decode_table[mat_id]  # (B, 3)
        t = mat_feats[:, 0]
        p = mat_feats[:, 1]
        h = mat_feats[:, 2]

        z_mat = self.embed_T(t) + self.embed_P(p) + self.embed_H(h)
        z_king = self.king_embed(k_us, k_them)
        return center_and_rms_norm(z_mat + z_king, self.codebook_target_norm)

    def get_all_context_vectors(self) -> torch.Tensor:
        """Computes, mean-centers, and RMS-normalizes context vectors for all 849,420 contexts."""
        device = self.codebook_weights.device

        # Compute z_mat for all 396 material states
        mat_feats = self.material_decode_table  # (396, 3)
        z_mat = (
            self.embed_T(mat_feats[:, 0])
            + self.embed_P(mat_feats[:, 1])
            + self.embed_H(mat_feats[:, 2])
        )  # (396, dim)

        # Compute z_king for all 33 x 65 = 2,145 king pairs
        all_kus = torch.arange(NUM_OUR_KINGS, device=device).repeat_interleave(
            NUM_ENEMY_KINGS
        )
        all_kth = torch.arange(NUM_ENEMY_KINGS, device=device).repeat(NUM_OUR_KINGS)
        z_king = self.king_embed(all_kus, all_kth)  # (2145, dim)

        # Outer sum: (396, 2145, dim) -> (849420, dim)
        z_all = (z_mat.unsqueeze(1) + z_king.unsqueeze(0)).reshape(
            self.num_contexts, self.dim
        )
        return center_and_rms_norm(z_all, self.codebook_target_norm)

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

    def get_coalesced_bias_table(self) -> torch.Tensor:
        if self.coalesced_valid:
            return self.coalesced_biases
        with ieee_matmul_precision():
            raw_biases = torch.stack(
                [
                    self.bias_head(self.codebook_b),
                    self.bias_head(
                        center_and_rms_norm(
                            self.codebook_biases_opp, self.codebook_target_norm
                        ),
                        1,
                    ),
                ]
            )
            if self.quantization is not None:
                q_biases = self.quantization.quantize_dynamic_bias(raw_biases)
                self.coalesced_biases.copy_(
                    self.quantization.dequantize_dynamic_bias(q_biases)
                )
                self.coalesced_valid = True
                return self.coalesced_biases
            return raw_biases

    @torch.no_grad()
    def load_coalesced_biases(self, biases: torch.Tensor) -> None:
        self.coalesced_biases.copy_(
            biases.to(device=self.coalesced_biases.device, dtype=torch.float32)
        )
        self.coalesced_valid = True

    @torch.no_grad()
    def update_lookup_table(self):
        with ieee_matmul_precision():
            z_all = self.get_all_context_vectors()
            packed = torch.zeros(
                self.num_contexts, dtype=torch.int64, device=z_all.device
            )
            self.lut_margin_valid = False
            for row, (name, _, _, _, cb) in enumerate(self.banks()):
                shift = (0, 16, 8, 24)[row]
                margins = []
                for start in range(0, self.num_contexts, 65536):
                    z = z_all[start : start + 65536]
                    dist = (
                        z.square().sum(-1, keepdim=True)
                        + cb.square().sum(-1)[None]
                        - 2 * z @ cb.T
                    )
                    packed[start : start + len(z)] |= dist.argmin(-1).long() << shift
                    if self.collect_lut_diagnostics and len(cb) >= 2:
                        nearest = dist[::256].topk(2, largest=False).values
                        margins.append(nearest[:, 1] - nearest[:, 0])
                if margins:
                    gaps = torch.cat(margins)
                    self.lut_margin_stats[row].copy_(
                        torch.stack(
                            [gaps.min(), gaps.mean(), (gaps <= 1e-6).float().mean()]
                        )
                    )
                    self.lut_margin_samples = len(gaps)
                    self.lut_margin_valid = True
            self.lut.copy_(packed)
            self.lut_valid = True
            self.get_coalesced_bias_table()
            return self.lut

    def get_lookup_table(self) -> torch.Tensor:
        if not self.lut_valid:
            self.update_lookup_table()
        return self.lut

    @torch.no_grad()
    def load_lookup_table(self, table: torch.Tensor | bytes | bytearray) -> None:
        if isinstance(table, (bytes, bytearray)):
            table = torch.frombuffer(bytearray(table), dtype=torch.uint32).to(
                torch.int64
            )
        self.lut.copy_(table.to(device=self.lut.device, dtype=torch.int64))
        self.lut_valid = True
        self.lut_margin_valid = False
        for p in self.parameters():
            p.zero_()

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
            self.coalesced_valid = False
        return tuple(reseeded)

    def forward(self, c_us, c_them, fake_quantize=False):
        chunks = (self.hidden_sizes[0],) * 2 + (self.hidden_sizes[1],) * 2
        if self.training:
            zs = (self.get_context_vector(c_us), self.get_context_vector(c_them))
            indices, dynamic, parts = [], [], []
            for row, (name, _, hard, soft, cb) in enumerate(self.banks()):
                role, bias = divmod(row, 2)
                idx, zq, commit, balance = self.quantize(
                    zs[role],
                    cb,
                    len(cb),
                    self.bias_temperature if bias else self.temperature,
                    hard,
                    soft,
                )
                parts.extend(
                    [
                        commit * 0.5,
                        balance
                        * (self.bias_balance_cost if bias else self.balance_cost)
                        * 0.5,
                        codebook_diversity_loss(cb)
                        * (self.bias_diversity_cost if bias else self.diversity_cost)
                        * 0.5,
                    ]
                )
                if bias:
                    d = self.bias_head(zq, role)
                    if fake_quantize and self.quantization is not None:
                        d = self.quantization.fake_quantize_dynamic_bias(d)
                    dynamic.append(d.split(chunks, dim=1))
                else:
                    indices.append(idx)
            parts = torch.stack(parts)
            with torch.no_grad():
                self.loss_component_sum.add_(parts.detach())
                self.loss_component_batches.add_(1)
            self.lut_valid = self.lut_margin_valid = self.coalesced_valid = False
            return (
                indices[0],
                indices[1],
                dynamic[0],
                dynamic[1],
                parts.sum(),
                parts[::3].sum(),
                parts[1::3].sum(),
            )
        if not self.lut_valid:
            self.update_lookup_table()
        us, them = self.lut[c_us.long()], self.lut[c_them.long()]
        ew = (us & 255, (them >> 8) & 255)
        eb = ((us >> 16) & 255, (them >> 24) & 255)
        if self.collect_lut_diagnostics:
            for name, ids, count in zip(
                MOE_BANKS,
                (ew[0], eb[0], ew[1], eb[1]),
                (self.num_weight_experts, self.num_bias_experts) * 2,
            ):
                counts = getattr(self, f"validation_counts_{name}")
                counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
        table = self.get_coalesced_bias_table()
        zero = torch.zeros((), device=c_us.device)
        return (
            ew[0],
            ew[1],
            table[0, eb[0]].split(chunks, 1),
            table[1, eb[1]].split(chunks, 1),
            zero,
            zero,
            zero,
        )


VQCodebookRouter = DualVQCodebookRouter
