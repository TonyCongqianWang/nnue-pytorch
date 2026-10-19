import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_MATERIAL_STATES = 378
NUM_OUR_KINGS = 33
NUM_ENEMY_KINGS = 65
ROUTER_TABLE_SIZE = NUM_MATERIAL_STATES * NUM_OUR_KINGS * NUM_ENEMY_KINGS  # 810,810
NUM_WEIGHT_EXPERTS = 32
NUM_BIAS_EXPERTS = 512
NUM_BIASES = 128
BIAS_RANK = 32
ROUTER_HASH = 0x4475616C  # 'Dual'

# Precompute bijective mapping for the 378 legal material states
_MATERIAL_STATES: list[tuple[int, int, int, int]] = []
for _t in range(16):
    for _p in range(min(8, _t) + 1):
        for _q in range(2):
            if _q == 1 and _t - _p < 1:
                continue
            for _bp in range(2):
                if _bp == 1 and (_t - _p - _q < 2):
                    continue
                _MATERIAL_STATES.append((_t, _p, _q, _bp))

assert len(_MATERIAL_STATES) == NUM_MATERIAL_STATES

_MATERIAL_ENCODE_TABLE = torch.full((16, 9, 2, 2), -1, dtype=torch.long)
for _idx, (_t, _p, _q, _bp) in enumerate(_MATERIAL_STATES):
    _MATERIAL_ENCODE_TABLE[_t, _p, _q, _bp] = _idx

_MATERIAL_DECODE_TABLE = torch.tensor(_MATERIAL_STATES, dtype=torch.long)


def encode_material_state(t: int, p: int, q: int, bp: int) -> int:
    """Encodes (T, P, Q, BP) into material ID in [0, 377]."""
    mat_id = int(_MATERIAL_ENCODE_TABLE[t, p, q, bp].item())
    if mat_id < 0:
        raise ValueError(f"Illegal material state: T={t}, P={p}, Q={q}, BP={bp}")
    return mat_id


def decode_material_state(mat_id: int) -> tuple[int, int, int, int]:
    """Decodes material ID in [0, 377] into (T, P, Q, BP)."""
    return tuple(_MATERIAL_STATES[mat_id])  # type: ignore[return-value]


def make_router_index(mat_id: int, our_king: int, enemy_king: int) -> int:
    """Computes the 1D context index in [0, 810809]."""
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
      - Our king: distance to home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
      - Enemy king: distance to enemy home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
      - Chebyshev opposition distance: min Chebyshev distance between kings (0..7)
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

        self.embed_our_rank = nn.Embedding(8, dim)
        self.embed_our_edge = nn.Embedding(4, dim)
        self.embed_our_castling = nn.Embedding(2, dim)

        self.embed_enemy_rank = nn.Embedding(8, dim)
        self.embed_enemy_edge = nn.Embedding(4, dim)
        self.embed_enemy_castling = nn.Embedding(2, dim)

        self.embed_chebyshev = nn.Embedding(8, dim)

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

        chebyshev = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        for k_us in range(NUM_OUR_KINGS):
            r_us = 0 if k_us == 32 else k_us // 4
            f_us = 3 if k_us == 32 else k_us % 4
            for k_them in range(NUM_ENEMY_KINGS):
                r_them = 7 if k_them == 64 else k_them // 8
                f_them = 4 if k_them == 64 else k_them % 8
                dr = abs(r_us - r_them)
                df = min(abs(f_us - f_them), abs((7 - f_us) - f_them))
                chebyshev[k_us, k_them] = max(dr, df)

        self.register_buffer("our_rank", our_rank)
        self.register_buffer("our_edge", our_edge)
        self.register_buffer("our_castling", our_castling)
        self.register_buffer("enemy_rank", enemy_rank)
        self.register_buffer("enemy_edge", enemy_edge)
        self.register_buffer("enemy_castling", enemy_castling)
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
        nn.init.uniform_(self.embed_chebyshev.weight, -val, val)

    def forward(self, k_us: torch.Tensor, k_them: torch.Tensor) -> torch.Tensor:
        return (
            self.embed_our_rank(self.our_rank[k_us])
            + self.embed_our_edge(self.our_edge[k_us])
            + self.embed_our_castling(self.our_castling[k_us])
            + self.embed_enemy_rank(self.enemy_rank[k_them])
            + self.embed_enemy_edge(self.enemy_edge[k_them])
            + self.embed_enemy_castling(self.enemy_castling[k_them])
            + self.embed_chebyshev(self.chebyshev[k_us, k_them])
        )


class DualVQCodebookRouter(nn.Module):
    """
    Dual Discrete Vector Quantization (VQ) Codebook Router for Top-2 Downstream MoE:
    - Weight Router: K1 = 32 discrete experts (5 bits) selecting layer stack weights
    - Bias Router: K2 = 512 discrete experts (9 bits) selecting 128 dynamic threshold biases via rank-32 subspace
    - Single packed uint16 LUT of size 810,810 (1.55 MB)
    - 512 coalesced 128-element dynamic bias vectors (256 KB)
    """

    def __init__(
        self,
        num_contexts: int = ROUTER_TABLE_SIZE,
        num_weight_experts: int = NUM_WEIGHT_EXPERTS,
        num_bias_experts: int = NUM_BIAS_EXPERTS,
        dim: int = 32,
        bias_rank: int = BIAS_RANK,
        num_biases: int = NUM_BIASES,
        commitment_cost: float = 0.005,
        balance_cost: float = 0.001,
        temperature: float = 1.0,
        reseed_dead_experts: bool = True,
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_weight_experts = num_weight_experts
        self.num_bias_experts = num_bias_experts
        self.dim = dim
        self.bias_rank = bias_rank
        self.num_biases = num_biases
        self.commitment_cost = commitment_cost
        self.balance_cost = balance_cost
        self.temperature = temperature
        self.reseed_dead_experts = reseed_dead_experts

        # Material feature embedding tables
        self.embed_T = nn.Embedding(16, dim)
        self.embed_P = nn.Embedding(9, dim)
        self.embed_Q = nn.Embedding(2, dim)
        self.embed_BP = nn.Embedding(2, dim)

        # Structured spatial King embedding table
        self.king_embed = StructuredKingEmbedding(dim=dim)

        # Weight Router Codebook: 32 experts in R^32
        self.codebook_weights = nn.Parameter(torch.empty(num_weight_experts, dim))

        # Bias Router Codebook: 512 experts in R^32 (subspace rank 32)
        self.codebook_biases = nn.Parameter(torch.empty(num_bias_experts, bias_rank))
        self.bias_head = nn.Linear(bias_rank, num_biases)

        # Material decoding buffer on device
        self.register_buffer(
            "material_decode_table", _MATERIAL_DECODE_TABLE, persistent=False
        )

        # Ephemeral runtime cache for inference & Stockfish export
        self.register_buffer(
            "lut",
            torch.zeros(num_contexts, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "lut_valid",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "coalesced_biases",
            torch.zeros(num_bias_experts, num_biases, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "coalesced_valid",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )

        self.init_weights()

    @property
    def codebook(self) -> torch.Tensor:
        return self.codebook_weights

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.lut_valid.fill_(False)
            self.coalesced_valid.fill_(False)
        return self

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.embed_T.weight, -val, val)
        nn.init.uniform_(self.embed_P.weight, -val, val)
        nn.init.uniform_(self.embed_Q.weight, -val, val)
        nn.init.uniform_(self.embed_BP.weight, -val, val)
        nn.init.uniform_(self.codebook_weights, -val, val)

        val_bias = 1.0 / math.sqrt(self.bias_rank)
        nn.init.uniform_(self.codebook_biases, -val_bias, val_bias)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def get_context_vector(self, c: torch.Tensor) -> torch.Tensor:
        mat_id, k_us, k_them = decompose_router_index(c)
        mat_feats = self.material_decode_table[mat_id]  # (B, 4)
        t = mat_feats[:, 0]
        p = mat_feats[:, 1]
        q = mat_feats[:, 2]
        bp = mat_feats[:, 3]

        z_mat = self.embed_T(t) + self.embed_P(p) + self.embed_Q(q) + self.embed_BP(bp)
        z_king = self.king_embed(k_us, k_them)
        return z_mat + z_king

    def quantize(
        self, z: torch.Tensor, codebook: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        scaled_neg_dist = -distances / max(self.temperature, 1e-4)
        probs = F.softmax(scaled_neg_dist, dim=-1)
        prob_mean = probs.mean(dim=0)

        balance_loss = k_exp * torch.sum(f * prob_mean) - 1.0

        # Straight-Through Estimator
        z_q_ste = z + (z_q - z).detach()

        return indices, z_q_ste, commit_loss, balance_loss

    def get_coalesced_bias_table(self) -> torch.Tensor:
        if bool(self.coalesced_valid.item()):
            return self.coalesced_biases
        return self.bias_head(self.codebook_biases)

    @torch.no_grad()
    def load_coalesced_biases(self, biases: torch.Tensor) -> None:
        self.coalesced_biases.copy_(
            biases.to(device=self.coalesced_biases.device, dtype=torch.float32)
        )
        self.coalesced_valid.fill_(True)

    @torch.no_grad()
    def update_lookup_table(self) -> torch.Tensor:
        """Evaluates all 810,810 contexts and bit-packs (idx_w, idx_b) into uint16 lookup table."""
        device = self.codebook_weights.device

        # Compute z_mat for all 378 material states
        mat_feats = self.material_decode_table  # (378, 4)
        z_mat = (
            self.embed_T(mat_feats[:, 0])
            + self.embed_P(mat_feats[:, 1])
            + self.embed_Q(mat_feats[:, 2])
            + self.embed_BP(mat_feats[:, 3])
        )  # (378, 32)

        # Compute z_king for all 33 x 65 = 2,145 king pairs
        all_kus = torch.arange(NUM_OUR_KINGS, device=device).repeat_interleave(
            NUM_ENEMY_KINGS
        )
        all_kth = torch.arange(NUM_ENEMY_KINGS, device=device).repeat(NUM_OUR_KINGS)
        z_king = self.king_embed(all_kus, all_kth)  # (2145, 32)

        # Outer sum: (378, 2145, 32) -> (810810, 32)
        z_all = (z_mat.unsqueeze(1) + z_king.unsqueeze(0)).reshape(
            self.num_contexts, self.dim
        )

        cb_w = self.codebook_weights
        cb_b = self.codebook_biases
        cb_w_sq = torch.sum(cb_w**2, dim=-1).unsqueeze(0)
        cb_b_sq = torch.sum(cb_b**2, dim=-1).unsqueeze(0)

        chunk_size = 65536
        for i in range(0, self.num_contexts, chunk_size):
            z_chunk = z_all[i : i + chunk_size]
            z_sq = torch.sum(z_chunk**2, dim=-1, keepdim=True)

            # Weight routing (32 experts)
            dist_w = z_sq + cb_w_sq - 2.0 * torch.matmul(z_chunk, cb_w.t())
            idx_w = torch.argmin(dist_w, dim=-1)

            # Bias routing (512 experts)
            dist_b = z_sq + cb_b_sq - 2.0 * torch.matmul(z_chunk, cb_b.t())
            idx_b = torch.argmin(dist_b, dim=-1)

            # Bit packing: 5 bits weight | 9 bits bias
            packed = (idx_w & 0x1F) | ((idx_b & 0x1FF) << 5)
            self.lut[i : i + chunk_size] = packed

        self.lut_valid.fill_(True)
        return self.lut

    def get_lookup_table(self) -> torch.Tensor:
        if not bool(self.lut_valid.item()):
            self.update_lookup_table()
        return self.lut

    @torch.no_grad()
    def load_lookup_table(self, table: torch.Tensor | bytes | bytearray) -> None:
        if isinstance(table, (bytes, bytearray)):
            table = torch.frombuffer(bytearray(table), dtype=torch.uint16).to(
                torch.int32
            )
        self.lut.copy_(table.to(device=self.lut.device, dtype=torch.int32))
        self.lut_valid.fill_(True)
        self.codebook_weights.zero_()
        self.codebook_biases.zero_()
        self.bias_head.weight.zero_()
        self.bias_head.bias.zero_()
        self.embed_T.weight.zero_()
        self.embed_P.weight.zero_()
        self.embed_Q.weight.zero_()
        self.embed_BP.weight.zero_()

    @torch.no_grad()
    def reseed_dead_experts_context_wide(self) -> tuple[list[int], list[int]]:
        """Evaluates contexts across the board context space and reseeds dead experts for both codebooks."""
        device = self.codebook_weights.device
        mat_feats = self.material_decode_table
        z_mat = (
            self.embed_T(mat_feats[:, 0])
            + self.embed_P(mat_feats[:, 1])
            + self.embed_Q(mat_feats[:, 2])
            + self.embed_BP(mat_feats[:, 3])
        )
        all_kus = torch.arange(NUM_OUR_KINGS, device=device).repeat_interleave(
            NUM_ENEMY_KINGS
        )
        all_kth = torch.arange(NUM_ENEMY_KINGS, device=device).repeat(NUM_OUR_KINGS)
        z_king = self.king_embed(all_kus, all_kth)
        z_all = (z_mat.unsqueeze(1) + z_king.unsqueeze(0)).reshape(
            self.num_contexts, self.dim
        )

        cb_w = self.codebook_weights
        cb_b = self.codebook_biases
        cb_w_sq = torch.sum(cb_w**2, dim=-1).unsqueeze(0)
        cb_b_sq = torch.sum(cb_b**2, dim=-1).unsqueeze(0)

        chunk_size = 65536
        all_idx_w = []
        all_idx_b = []
        all_dist_w_min = []
        all_dist_b_min = []

        for i in range(0, self.num_contexts, chunk_size):
            z_chunk = z_all[i : i + chunk_size]
            z_sq = torch.sum(z_chunk**2, dim=-1, keepdim=True)

            dist_w = z_sq + cb_w_sq - 2.0 * torch.matmul(z_chunk, cb_w.t())
            min_w, idx_w = torch.min(dist_w, dim=-1)
            all_idx_w.append(idx_w)
            all_dist_w_min.append(min_w)

            dist_b = z_sq + cb_b_sq - 2.0 * torch.matmul(z_chunk, cb_b.t())
            min_b, idx_b = torch.min(dist_b, dim=-1)
            all_idx_b.append(idx_b)
            all_dist_b_min.append(min_b)

        idx_w = torch.cat(all_idx_w, dim=0)
        idx_b = torch.cat(all_idx_b, dim=0)
        dist_w_min = torch.cat(all_dist_w_min, dim=0)
        dist_b_min = torch.cat(all_dist_b_min, dim=0)

        counts_w = torch.bincount(idx_w, minlength=self.num_weight_experts)
        dead_w = torch.nonzero(counts_w == 0, as_tuple=True)[0]
        reseeded_w: list[int] = []
        if len(dead_w) > 0:
            k_worst = min(len(dead_w), self.num_contexts)
            _, worst_ctx_w = torch.topk(dist_w_min, k=k_worst)
            for i, dead_idx in enumerate(dead_w):
                ctx_idx = worst_ctx_w[i % k_worst]
                noise = torch.randn_like(self.codebook_weights[dead_idx]) * 0.01
                self.codebook_weights.data[dead_idx].copy_(
                    z_all[ctx_idx].detach() + noise
                )
            reseeded_w = dead_w.tolist()

        counts_b = torch.bincount(idx_b, minlength=self.num_bias_experts)
        dead_b = torch.nonzero(counts_b == 0, as_tuple=True)[0]
        reseeded_b: list[int] = []
        if len(dead_b) > 0:
            k_worst = min(len(dead_b), self.num_contexts)
            _, worst_ctx_b = torch.topk(dist_b_min, k=k_worst)
            for i, dead_idx in enumerate(dead_b):
                ctx_idx = worst_ctx_b[i % k_worst]
                noise = torch.randn_like(self.codebook_biases[dead_idx]) * 0.01
                self.codebook_biases.data[dead_idx].copy_(
                    z_all[ctx_idx].detach() + noise
                )
            reseeded_b = dead_b.tolist()

        if reseeded_w or reseeded_b:
            self.lut_valid.fill_(False)
            self.coalesced_valid.fill_(False)

        return reseeded_w, reseeded_b

    def forward(
        self, c_us: torch.Tensor, c_them: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.training:
            c_combined = torch.cat([c_us, c_them], dim=0)
            z = self.get_context_vector(c_combined)

            e_w, _, commit_loss_w, balance_loss_w = self.quantize(
                z, self.codebook_weights, self.num_weight_experts
            )
            _, z_q_ste_b, commit_loss_b, balance_loss_b = self.quantize(
                z, self.codebook_biases, self.num_bias_experts
            )

            dyn_bias = self.bias_head(z_q_ste_b)

            commit_loss = commit_loss_w + commit_loss_b
            balance_loss = balance_loss_w + balance_loss_b
            moe_loss = commit_loss + self.balance_cost * balance_loss

            self.lut_valid.fill_(False)
            self.coalesced_valid.fill_(False)

            half = c_us.shape[0]
            e1_w = e_w[:half]
            e2_w = e_w[half:]
            dyn_bias_1 = dyn_bias[:half]
            dyn_bias_2 = dyn_bias[half:]

            b1_sqr = dyn_bias_1[:, 0:32]
            b1_lin = dyn_bias_1[:, 32:64]
            b2_sqr = dyn_bias_1[:, 64:96]
            b2_lin = dyn_bias_1[:, 96:128]
            dyn_biases_1 = (b1_sqr, b1_lin, b2_sqr, b2_lin)

            b1_sqr_2 = dyn_bias_2[:, 0:32]
            b1_lin_2 = dyn_bias_2[:, 32:64]
            b2_sqr_2 = dyn_bias_2[:, 64:96]
            b2_lin_2 = dyn_bias_2[:, 96:128]
            dyn_biases_2 = (b1_sqr_2, b1_lin_2, b2_sqr_2, b2_lin_2)

            return (
                e1_w,
                e2_w,
                dyn_biases_1,
                dyn_biases_2,
                moe_loss,
                commit_loss,
                balance_loss,
            )
        else:
            if not bool(self.lut_valid.item()):
                self.update_lookup_table()
            packed_1 = self.lut[c_us.long()]
            packed_2 = self.lut[c_them.long()]

            e1_w = (packed_1 & 0x1F).long()
            e2_w = (packed_2 & 0x1F).long()
            e1_b = ((packed_1 >> 5) & 0x1FF).long()
            e2_b = ((packed_2 >> 5) & 0x1FF).long()

            coalesced = self.get_coalesced_bias_table()
            dyn_bias_1 = coalesced[e1_b]
            dyn_bias_2 = coalesced[e2_b]

            dyn_biases_1 = (
                dyn_bias_1[:, 0:32],
                dyn_bias_1[:, 32:64],
                dyn_bias_1[:, 64:96],
                dyn_bias_1[:, 96:128],
            )
            dyn_biases_2 = (
                dyn_bias_2[:, 0:32],
                dyn_bias_2[:, 32:64],
                dyn_bias_2[:, 64:96],
                dyn_bias_2[:, 96:128],
            )

            zero = torch.tensor(0.0, device=c_us.device)
            return e1_w, e2_w, dyn_biases_1, dyn_biases_2, zero, zero, zero


# Backward-compatible alias for any modules importing VQCodebookRouter
VQCodebookRouter = DualVQCodebookRouter
