import math

import torch
import torch.nn.functional as F
from torch import nn

ROUTER_TABLE_SIZE = 68640
NUM_EXPERTS = 32
ROUTER_HASH = 0x546F7032


def make_router_index(
    our_pieces: int, our_queen: int, our_king: int, enemy_king: int
) -> int:
    """Computes the 1D context index in [0, 68639]."""
    return ((our_pieces * 2 + our_queen) * 33 + our_king) * 65 + enemy_king


def decompose_router_index(
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes context index c into (our_pieces, our_queen, our_king, enemy_king)."""
    enemy_king = c % 65
    rem1 = torch.div(c, 65, rounding_mode="floor")
    our_king = rem1 % 33
    rem2 = torch.div(rem1, 33, rounding_mode="floor")
    our_queen = rem2 % 2
    our_pieces = torch.div(rem2, 2, rounding_mode="floor")
    return our_pieces, our_queen, our_king, enemy_king


class VQCodebookRouter(nn.Module):
    """
    Symmetric Discrete Vector Quantization (VQ) Codebook Router for Top-2 Downstream MoE.

    Evaluates context from perspective-relative board state:
      Context = ourPieceCount (16) x ourQueen (2) x ourKingPosition (33) x enemyKingPosition (65) = 68,640.

    During training:
      Learns embeddings for each component and 32 discrete codebook vectors in R^32.
      Quantizes context vectors using nearest neighbor with Straight-Through Estimator (STE)
      and commitment loss (no Gumbel-Softmax, no temperature annealing).

    During inference & Stockfish export:
      All 68,640 context keys are evaluated once into a 67 KB uint8 lookup table.
      Inference is zero-overhead 1-cycle array indexing: e1 = router[c_us], e2 = router[c_them].
    """

    def __init__(
        self,
        num_contexts: int = ROUTER_TABLE_SIZE,
        num_experts: int = NUM_EXPERTS,
        dim: int = 32,
        commitment_cost: float = 0.25,
        balance_cost: float = 0.02,
        temperature: float = 1.0,
        reseed_dead_experts: bool = True,
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_experts = num_experts
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.balance_cost = balance_cost
        self.temperature = temperature
        self.reseed_dead_experts = reseed_dead_experts

        # Context feature embedding tables
        self.piece_embed = nn.Embedding(16, dim)
        self.queen_embed = nn.Embedding(2, dim)
        self.our_king_embed = nn.Embedding(33, dim)
        self.enemy_king_embed = nn.Embedding(65, dim)

        # 32 discrete codebook vectors
        self.codebook = nn.Parameter(torch.empty(num_experts, dim))
        self.init_weights()

        # Cached discrete LUT for zero-overhead inference
        self.register_buffer("lut", torch.zeros(num_contexts, dtype=torch.uint8))
        self.register_buffer("lut_valid", torch.tensor(False, dtype=torch.bool))

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.piece_embed.weight, -val, val)
        nn.init.uniform_(self.queen_embed.weight, -val, val)
        nn.init.uniform_(self.our_king_embed.weight, -val, val)
        nn.init.uniform_(self.enemy_king_embed.weight, -val, val)
        nn.init.uniform_(self.codebook, -val, val)

    def get_context_vector(self, c: torch.Tensor) -> torch.Tensor:
        p, q, k_us, k_them = decompose_router_index(c)
        z = (
            self.piece_embed(p)
            + self.queen_embed(q)
            + self.our_king_embed(k_us)
            + self.enemy_king_embed(k_them)
        )
        return z

    @torch.no_grad()
    def _reseed_dead_experts(
        self, z: torch.Tensor, distances: torch.Tensor, f: torch.Tensor
    ) -> None:
        """Reseeds unassigned codebook vectors to the contexts with highest quantization error."""
        if z.shape[0] < self.num_experts * 2:
            return

        dead_mask = f == 0.0
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0:
            return

        dead_indices = torch.nonzero(dead_mask, as_tuple=True)[0]
        min_dists, _ = torch.min(distances, dim=-1)
        k_worst = min(num_dead, z.shape[0])
        _, worst_ctx_indices = torch.topk(min_dists, k=k_worst)

        for i, dead_idx in enumerate(dead_indices):
            ctx_idx = worst_ctx_indices[i % k_worst]
            noise = torch.randn_like(self.codebook[dead_idx]) * 0.01
            self.codebook.data[dead_idx].copy_(z[ctx_idx].detach() + noise)
        self.lut_valid.fill_(False)

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute squared Euclidean distances to codebook: ||z - c_j||^2
        # (B, D) vs (K, D)
        z_sq = torch.sum(z**2, dim=-1, keepdim=True)  # (B, 1)
        c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)  # (1, K)
        dot = torch.matmul(z, self.codebook.t())  # (B, K)
        distances = z_sq + c_sq - 2.0 * dot  # (B, K)

        indices = torch.argmin(distances, dim=-1)  # (B,)
        z_q = F.embedding(indices, self.codebook)  # (B, D)

        # Vector Quantization & Commitment loss
        commit_loss = self.commitment_cost * F.mse_loss(z, z_q.detach()) + F.mse_loss(
            z_q, z.detach()
        )

        # Load-Balancing Auxiliary Loss (Switch Transformer / GShard formulation)
        K = self.num_experts
        one_hot = F.one_hot(indices, num_classes=K).float()
        f = one_hot.mean(dim=0).detach()  # (K,) empirical dispatch fraction

        scaled_neg_dist = -distances / max(self.temperature, 1e-4)
        probs = F.softmax(scaled_neg_dist, dim=-1)  # (B, K) soft routing probabilities
        P = probs.mean(dim=0)  # (K,) batch-mean routing probability

        balance_loss = K * torch.sum(f * P)

        # Reseed dead experts if unassigned in this batch
        if self.training and self.reseed_dead_experts:
            self._reseed_dead_experts(z, distances, f)

        # Straight-Through Estimator (STE)
        z_q_ste = z + (z_q - z).detach()

        return indices, z_q_ste, commit_loss, balance_loss

    @torch.no_grad()
    def update_lookup_table(self) -> torch.Tensor:
        """Evaluates all 68,640 contexts and caches the discrete nearest codebook expert IDs."""
        device = self.codebook.device
        all_contexts = torch.arange(self.num_contexts, device=device, dtype=torch.int64)
        z = self.get_context_vector(all_contexts)
        z_sq = torch.sum(z**2, dim=-1, keepdim=True)
        c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)
        dot = torch.matmul(z, self.codebook.t())
        distances = z_sq + c_sq - 2.0 * dot
        indices = torch.argmin(distances, dim=-1).to(torch.uint8)

        # Ensure no expert has 0 contexts across the entire context space
        if self.reseed_dead_experts:
            counts = torch.bincount(indices.long(), minlength=self.num_experts)
            dead_indices = torch.nonzero(counts == 0, as_tuple=True)[0]
            if len(dead_indices) > 0:
                min_dists, _ = torch.min(distances, dim=-1)
                k_worst = min(len(dead_indices), z.shape[0])
                _, worst_contexts = torch.topk(min_dists, k=k_worst)
                for i, dead_idx in enumerate(dead_indices):
                    ctx_idx = worst_contexts[i % k_worst]
                    noise = torch.randn_like(self.codebook[dead_idx]) * 0.01
                    self.codebook.data[dead_idx].copy_(z[ctx_idx].detach() + noise)
                # Recompute lookup table with reseeded codebook
                c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)
                dot = torch.matmul(z, self.codebook.t())
                distances = z_sq + c_sq - 2.0 * dot
                indices = torch.argmin(distances, dim=-1).to(torch.uint8)

        self.lut.copy_(indices)
        self.lut_valid.fill_(True)
        return self.lut

    def get_lookup_table(self) -> torch.Tensor:
        if not bool(self.lut_valid.item()):
            self.update_lookup_table()
        return self.lut

    @torch.no_grad()
    def load_lookup_table(self, table: torch.Tensor | bytes | bytearray) -> None:
        if isinstance(table, (bytes, bytearray)):
            table = torch.frombuffer(bytearray(table), dtype=torch.uint8)
        self.lut.copy_(table.to(device=self.lut.device, dtype=torch.uint8))
        self.lut_valid.fill_(True)
        self.codebook.zero_()
        self.piece_embed.weight.zero_()
        self.queen_embed.weight.zero_()
        self.our_king_embed.weight.zero_()
        self.enemy_king_embed.weight.zero_()

    def forward(
        self, c_us: torch.Tensor, c_them: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            c_combined = torch.cat([c_us, c_them], dim=0)
            z = self.get_context_vector(c_combined)
            indices, _, commit_loss, balance_loss = self.quantize(z)

            half = c_us.shape[0]
            e1 = indices[:half]
            e2 = indices[half:]

            moe_loss = commit_loss + self.balance_cost * balance_loss
            self.lut_valid.fill_(False)
            return e1, e2, moe_loss, commit_loss, balance_loss
        else:
            if not bool(self.lut_valid.item()):
                self.update_lookup_table()
            e1 = self.lut[c_us.long()]
            e2 = self.lut[c_them.long()]
            zero = torch.tensor(0.0, device=c_us.device)
            return e1.long(), e2.long(), zero, zero, zero
