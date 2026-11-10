import torch
from torch import nn

from .config import ModelConfig
from .modules import (
    ComposedFeatureTransformer,
    LayerStacks,
    MoERouter,
    get_feature_cls,
)
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
        hidden_sizes = (config.L2, config.L3)
        self.quantization = QuantizationManager(config.quantize_config, hidden_sizes)

        self.num_ls_buckets = config.moe_tail_experts
        self.tail_mean_balance_cost = config.moe_tail_mean_balance_cost

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.router = MoERouter(
            num_fc0_experts=config.moe_fc0_experts,
            num_weight_experts=config.moe_tail_experts,
            fc0_dim=config.moe_fc0_codebook_dim,
            dim=config.moe_tail_codebook_dim,
            balance_cost=config.moe_weight_balance_cost,
            diversity_cost=config.moe_weight_diversity_cost,
            temperature=config.moe_weight_temperature,
        )
        self.layer_stacks = LayerStacks(
            self.num_ls_buckets,
            config,
            self.quantization,
            rank=config.moe_tail_rank,
            codebook_dim=config.moe_tail_codebook_dim,
            fc0_count=config.moe_fc0_experts,
            fc0_rank=config.moe_fc0_rank,
            fc0_dim=config.moe_fc0_codebook_dim,
        )
        self.commit_loss: torch.Tensor | None = None

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights()

    def validate_architecture(self):
        if any(
            hasattr(layer, "shared_bias") for layer in self.layer_stacks.dense_layers()
        ):
            raise ValueError(
                "MoE8.6 requires expert-only biases; earlier .pt models are incompatible. "
                "Start a fresh MoE8.6 training pipeline."
            )

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

    def calculate_buckets(self, c_us: torch.Tensor, c_them: torch.Tensor):
        e1_w, e2_w, *_ = self.router(c_us, c_them)
        return e1_w, e2_w

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        c_us: torch.Tensor,
        c_them: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
        mix_experts: bool = False,
    ):
        (
            e1_w,
            e2_w,
            moe_loss,
            commit_loss,
            balance_loss,
        ) = self.router(
            c_us,
            c_them,
            fake_quantize=fake_quantize_weights,
            mix_experts=mix_experts,
            tail_mean_cost=self.tail_mean_balance_cost,
        )
        self.moe_loss = moe_loss
        self.commit_loss = commit_loss
        self.balance_loss = balance_loss

        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        x = self.layer_stacks(
            l0_,
            self.router.codebook,
            e1_w,
            e2_w,
            fake_quantize_acts,
            fake_quantize_weights,
        )

        return x
