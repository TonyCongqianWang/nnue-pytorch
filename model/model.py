import warnings

import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
from .quantize import QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_ls_buckets = 32

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        # Dynamic Bias Router on exact material state space (236,196 contexts)
        psq_offset = 0
        self.psq_offset = None
        self.psq_span = None
        for f in self.input.features:
            if getattr(f, "NUM_PLANES", None) == 768 or f.INPUT_FEATURE_NAME in (
                "K32Q2",
                "HalfKAv2_hm",
            ):
                self.psq_offset = psq_offset
                self.psq_span = f.NUM_INPUTS
                break
            psq_offset += f.NUM_INPUTS

        if self.psq_offset is None:
            warnings.warn(
                "No piece-square feature set (such as K32Q2 or HalfKAv2_hm) found in input features. "
                "Dynamic bias routing will be disabled and will fall back to context 0. "
                "Real use and full training requires a piece-square feature set.",
                UserWarning,
                stacklevel=2,
            )

        from .modules import DynamicBiasRouter

        self.router = DynamicBiasRouter()

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights()

    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input:
            self.input.clip_weights(self.quantization)

        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    p_data_fp32.clamp_(min_weight, max_weight)

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.input.zero_virtual_weights()
        self.layer_stacks.zero_virtual_weights()

    def forward_ft(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> torch.Tensor:
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def compute_material_context(
        self, piece_square_indices: torch.Tensor
    ) -> torch.Tensor:
        if self.psq_offset is None or self.psq_span is None:
            return torch.zeros(
                piece_square_indices.shape[0],
                dtype=torch.long,
                device=piece_square_indices.device,
            )

        idx = piece_square_indices.long()
        rel = idx - self.psq_offset
        valid = (idx >= 0) & (rel >= 0) & (rel < self.psq_span)
        plane = torch.where(valid, rel % 768, torch.zeros_like(rel))
        piece = plane // 64

        n = idx.shape[0]
        counts = torch.zeros(n, 13, dtype=torch.long, device=idx.device)
        counts.scatter_add_(
            1,
            torch.where(valid, piece, torch.full_like(piece, 12)),
            torch.ones_like(piece),
        )

        P_us = counts[:, 0].clamp(max=8)
        P_them = counts[:, 1].clamp(max=8)
        N_us = counts[:, 2].clamp(max=2)
        N_them = counts[:, 3].clamp(max=2)
        B_us = counts[:, 4].clamp(max=2)
        B_them = counts[:, 5].clamp(max=2)
        R_us = counts[:, 6].clamp(max=2)
        R_them = counts[:, 7].clamp(max=2)
        Q_us = counts[:, 8].clamp(max=1)
        Q_them = counts[:, 9].clamp(max=1)

        mat_us = (((P_us * 3 + N_us) * 3 + B_us) * 3 + R_us) * 2 + Q_us
        mat_them = (((P_them * 3 + N_them) * 3 + B_them) * 3 + R_them) * 2 + Q_them
        c_mat = mat_us * 486 + mat_them
        return c_mat

    def calculate_buckets(self, piece_count: torch.Tensor, queen_bucket: torch.Tensor):
        pc_bucket = (piece_count - 1) // 4
        return pc_bucket * 4 + queen_bucket

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        queen_bucket: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        layer_stack_indices = self.calculate_buckets(piece_count, queen_bucket)

        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

        stm_indices = torch.where(us.view(-1, 1) > 0.5, white_indices, black_indices)
        c_mat = self.compute_material_context(stm_indices)
        b1_sqr, b1_lin, b2_sqr, b2_lin, commit_loss, balance_loss = self.router(c_mat)

        if self.training:
            self.commit_loss = commit_loss
            self.balance_loss = balance_loss
            self.moe_loss = commit_loss + self.router.balance_cost * balance_loss
        else:
            self.commit_loss = None
            self.balance_loss = None
            self.moe_loss = None

        dyn_biases = (b1_sqr, b1_lin, b2_sqr, b2_lin)
        x = self.layer_stacks(
            l0_,
            layer_stack_indices,
            fake_quantize_acts,
            fake_quantize_weights,
            dyn_biases,
        )

        return x
