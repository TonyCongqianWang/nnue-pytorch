import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_CONTEXTS = 236196
NUM_EXPERTS = 1024
SUBSPACE_RANK = 32
BIAS_DIM = 128
ROUTER_HASH = 0x44425231  # "DBR1"


class DynamicBiasRouter(nn.Module):
    """
    Dynamic Bias Router using Discrete Vector Quantization (VQ) Codebook.

    Maps chess material configurations (236,196 contexts) into a discrete
    codebook of K = 1,024 experts. Each expert provides a coalesced dynamic
    bias vector in R^128 (32 L1-sqr, 32 L1-lin, 32 L2-sqr, 32 L2-lin)
    parameterized via a rank R = 32 subspace.
    """

    def __init__(
        self,
        num_contexts: int = NUM_CONTEXTS,
        num_experts: int = NUM_EXPERTS,
        dim: int = SUBSPACE_RANK,
        bias_dim: int = BIAS_DIM,
        commitment_cost: float = 0.005,
        balance_cost: float = 0.001,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_experts = num_experts
        self.dim = dim
        self.bias_dim = bias_dim
        self.commitment_cost = commitment_cost
        self.balance_cost = balance_cost
        self.temperature = temperature

        # Context feature embedding tables
        self.embed_p = nn.Embedding(9, dim)
        self.embed_n = nn.Embedding(3, dim)
        self.embed_b = nn.Embedding(3, dim)
        self.embed_r = nn.Embedding(3, dim)
        self.embed_q = nn.Embedding(2, dim)

        self.embed_p_opp = nn.Embedding(9, dim)
        self.embed_n_opp = nn.Embedding(3, dim)
        self.embed_b_opp = nn.Embedding(3, dim)
        self.embed_r_opp = nn.Embedding(3, dim)
        self.embed_q_opp = nn.Embedding(2, dim)

        self.embed_diff_nb = nn.Embedding(7, dim)
        self.embed_diff_r = nn.Embedding(5, dim)
        self.embed_diff_q = nn.Embedding(3, dim)
        self.embed_diff_p = nn.Embedding(9, dim)

        self.embed_joint_sig = nn.Embedding(2916, dim)

        # 1,024 discrete codebook centroids in R^32
        self.codebook = nn.Parameter(torch.empty(num_experts, dim))

        # Low-rank bias projection: maps R^32 -> R^129
        self.bias_proj = nn.Linear(dim, bias_dim, bias=True)

        # Ephemeral runtime cache (not saved in checkpoint)
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
            torch.zeros(num_experts, bias_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "coalesced_valid",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )

        self.init_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.lut_valid.fill_(False)
            self.coalesced_valid.fill_(False)
        return self

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        for emb in [
            self.embed_p,
            self.embed_n,
            self.embed_b,
            self.embed_r,
            self.embed_q,
            self.embed_p_opp,
            self.embed_n_opp,
            self.embed_b_opp,
            self.embed_r_opp,
            self.embed_q_opp,
            self.embed_diff_nb,
            self.embed_diff_r,
            self.embed_diff_q,
            self.embed_diff_p,
            self.embed_joint_sig,
        ]:
            nn.init.uniform_(emb.weight, -val, val)
        nn.init.uniform_(self.codebook, -val, val)

        # Initialize bias projection to zero so initial forward pass matches baseline exactly
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    @staticmethod
    def decode_material_context(c: torch.Tensor):
        mat_them = c % 486
        mat_us = c // 486

        def decode_side(idx):
            Q = idx % 2
            idx = idx // 2
            R = idx % 3
            idx = idx // 3
            B = idx % 3
            idx = idx // 3
            N = idx % 3
            idx = idx // 3
            P = idx
            return P, N, B, R, Q

        P_us, N_us, B_us, R_us, Q_us = decode_side(mat_us)
        P_them, N_them, B_them, R_them, Q_them = decode_side(mat_them)
        return (
            P_us,
            N_us,
            B_us,
            R_us,
            Q_us,
            P_them,
            N_them,
            B_them,
            R_them,
            Q_them,
        )

    def get_context_vector(self, c: torch.Tensor) -> torch.Tensor:
        (
            P_us,
            N_us,
            B_us,
            R_us,
            Q_us,
            P_them,
            N_them,
            B_them,
            R_them,
            Q_them,
        ) = self.decode_material_context(c)

        diff_nb = (N_us + B_us - N_them - B_them).clamp(-3, 3) + 3
        diff_r = (R_us - R_them).clamp(-2, 2) + 2
        diff_q = (Q_us - Q_them).clamp(-1, 1) + 1
        diff_p = (P_us - P_them).clamp(-4, 4) + 4

        sig_us = Q_us * 27 + R_us * 9 + B_us * 3 + N_us
        sig_them = Q_them * 27 + R_them * 9 + B_them * 3 + N_them
        joint_sig = sig_us * 54 + sig_them

        z = (
            self.embed_p(P_us)
            + self.embed_n(N_us)
            + self.embed_b(B_us)
            + self.embed_r(R_us)
            + self.embed_q(Q_us)
            + self.embed_p_opp(P_them)
            + self.embed_n_opp(N_them)
            + self.embed_b_opp(B_them)
            + self.embed_r_opp(R_them)
            + self.embed_q_opp(Q_them)
            + self.embed_diff_nb(diff_nb)
            + self.embed_diff_r(diff_r)
            + self.embed_diff_q(diff_q)
            + self.embed_diff_p(diff_p)
            + self.embed_joint_sig(joint_sig)
        )
        return z

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_sq = torch.sum(z**2, dim=-1, keepdim=True)  # (B, 1)
        c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)  # (1, K)
        dot = torch.matmul(z, self.codebook.t())  # (B, K)
        distances = z_sq + c_sq - 2.0 * dot  # (B, K)

        indices = torch.argmin(distances, dim=-1)  # (B,)
        z_q = F.embedding(indices, self.codebook)  # (B, D)

        # Commitment loss
        vq_loss = 0.25 * F.mse_loss(z, z_q.detach()) + F.mse_loss(z_q, z.detach())
        commit_loss = self.commitment_cost * vq_loss

        # Load balancing loss
        K = self.num_experts
        one_hot = F.one_hot(indices, num_classes=K).float()
        f = one_hot.mean(dim=0).detach()  # (K,)

        scaled_neg_dist = -distances / max(self.temperature, 1e-4)
        probs = F.softmax(scaled_neg_dist, dim=-1)  # (B, K)
        P = probs.mean(dim=0)  # (K,)

        balance_loss = K * torch.sum(f * P) - 1.0

        # Straight-Through Estimator
        z_q_ste = z + (z_q - z).detach()

        return indices, z_q_ste, commit_loss, balance_loss

    @torch.no_grad()
    def update_lookup_table(self) -> torch.Tensor:
        """Evaluates all 236,196 contexts into the discrete lookup table."""
        device = self.codebook.device
        all_contexts = torch.arange(self.num_contexts, device=device, dtype=torch.int64)

        chunk_size = 32768
        all_indices = []
        for i in range(0, self.num_contexts, chunk_size):
            ctx_chunk = all_contexts[i : i + chunk_size]
            z = self.get_context_vector(ctx_chunk)
            z_sq = torch.sum(z**2, dim=-1, keepdim=True)
            c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)
            dot = torch.matmul(z, self.codebook.t())
            distances = z_sq + c_sq - 2.0 * dot
            idx = torch.argmin(distances, dim=-1).to(torch.int32)
            all_indices.append(idx)

        indices = torch.cat(all_indices, dim=0)
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
            table = torch.frombuffer(bytearray(table), dtype=torch.int16).to(
                torch.int32
            )
        self.lut.copy_(table.to(device=self.lut.device, dtype=torch.int32))
        self.lut_valid.fill_(True)
        self.codebook.zero_()
        self.bias_proj.weight.zero_()
        self.bias_proj.bias.zero_()
        for emb in [
            self.embed_p,
            self.embed_n,
            self.embed_b,
            self.embed_r,
            self.embed_q,
            self.embed_p_opp,
            self.embed_n_opp,
            self.embed_b_opp,
            self.embed_r_opp,
            self.embed_q_opp,
            self.embed_diff_nb,
            self.embed_diff_r,
            self.embed_diff_q,
            self.embed_diff_p,
            self.embed_joint_sig,
        ]:
            emb.weight.zero_()

    def get_coalesced_bias_table(self) -> torch.Tensor:
        """Evaluates all K expert bias vectors into coalesced (K, 128) tensor."""
        if bool(self.coalesced_valid.item()):
            return self.coalesced_biases
        return self.bias_proj(self.codebook)

    @torch.no_grad()
    def load_coalesced_biases(self, biases: torch.Tensor) -> None:
        self.coalesced_biases.copy_(
            biases.to(device=self.coalesced_biases.device, dtype=torch.float32)
        )
        self.coalesced_valid.fill_(True)

    @torch.no_grad()
    def reseed_dead_experts_context_wide(self) -> list[int]:
        """Evaluates all 236,196 contexts and reseeds any dead expert."""
        device = self.codebook.device
        all_contexts = torch.arange(self.num_contexts, device=device, dtype=torch.int64)

        chunk_size = 32768
        all_indices = []
        all_min_dists = []
        for i in range(0, self.num_contexts, chunk_size):
            ctx_chunk = all_contexts[i : i + chunk_size]
            z = self.get_context_vector(ctx_chunk)
            z_sq = torch.sum(z**2, dim=-1, keepdim=True)
            c_sq = torch.sum(self.codebook**2, dim=-1).unsqueeze(0)
            dot = torch.matmul(z, self.codebook.t())
            distances = z_sq + c_sq - 2.0 * dot
            min_dist, idx = torch.min(distances, dim=-1)
            all_indices.append(idx)
            all_min_dists.append(min_dist)

        indices = torch.cat(all_indices, dim=0)
        min_dists = torch.cat(all_min_dists, dim=0)

        counts = torch.bincount(indices, minlength=self.num_experts)
        dead_indices = torch.nonzero(counts == 0, as_tuple=True)[0]
        if len(dead_indices) == 0:
            return []

        k_worst = min(len(dead_indices), self.num_contexts)
        _, worst_contexts = torch.topk(min_dists, k=k_worst)

        for i, dead_idx in enumerate(dead_indices):
            ctx_idx = worst_contexts[i % k_worst]
            ctx_tensor = torch.tensor([ctx_idx], device=device, dtype=torch.int64)
            z_target = self.get_context_vector(ctx_tensor).squeeze(0)
            noise = torch.randn_like(self.codebook[dead_idx]) * 0.01
            self.codebook.data[dead_idx].copy_(z_target.detach() + noise)

        self.lut_valid.fill_(False)
        return dead_indices.tolist()

    def forward(
        self, c_mat: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.training:
            z = self.get_context_vector(c_mat)
            indices, z_q_ste, commit_loss, balance_loss = self.quantize(z)
            dyn_bias = self.bias_proj(z_q_ste)
            self.lut_valid.fill_(False)
        else:
            if not bool(self.lut_valid.item()):
                self.update_lookup_table()
            indices = self.lut[c_mat.long()]
            coalesced = self.get_coalesced_bias_table()
            dyn_bias = coalesced[indices.long()]
            zero = torch.tensor(0.0, device=c_mat.device)
            commit_loss, balance_loss = zero, zero

        b1_sqr = dyn_bias[:, 0:32]
        b1_lin = dyn_bias[:, 32:64]
        b2_sqr = dyn_bias[:, 64:96]
        b2_lin = dyn_bias[:, 96:128]

        return b1_sqr, b1_lin, b2_sqr, b2_lin, commit_loss, balance_loss
