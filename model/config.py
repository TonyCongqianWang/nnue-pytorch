from dataclasses import dataclass, field, fields
from typing import Annotated

import tyro
from tyro.conf import OmitArgPrefixes

from .modules import FeatureConfig, LayerStacksConfig
from .modules.router import (
    ADAPTER_RANK,
    CODEBOOK_DIM,
    FC0_ADAPTER_RANK,
    FC0_CODEBOOK_DIM,
    NUM_FC0_EXPERTS,
    NUM_WEIGHT_EXPERTS,
)
from .optimizers import OptimizerConfig
from .quantize import QuantizationConfig


# 3 layer fully connected network
@dataclass(kw_only=True)
class ModelConfig(LayerStacksConfig):
    moe_fc0_experts: int = NUM_FC0_EXPERTS
    """Number of FC0 experts (2–64)."""
    moe_tail_experts: int = NUM_WEIGHT_EXPERTS
    """Number of FC1/output experts (2–256)."""
    moe_fc0_codebook_dim: int = FC0_CODEBOOK_DIM
    """FC0 context embedding and codebook dimension."""
    moe_tail_codebook_dim: int = CODEBOOK_DIM
    """Tail context embedding and codebook dimension."""
    moe_fc0_rank: int = FC0_ADAPTER_RANK
    """FC0 hypernetwork adapter bottleneck width."""
    moe_tail_rank: int = ADAPTER_RANK
    """FC1/output hypernetwork adapter bottleneck width."""

    moe_mix_interval: int = 0
    """Apply weighted top-2 FC0 and top-4 tail training every N optimizer steps. Zero disables."""

    moe_fc0_bottleneck_diversity_cost: float = 0.0
    """Penalty on concentration of centered FC0 bottleneck variance."""
    moe_tail_bottleneck_diversity_cost: float = 0.0
    """Penalty on concentration of centered tail bottleneck variance."""
    moe_center_fc0_weights: bool = False
    """Center generated FC0 weights across experts before adding the shared matrix."""

    moe_fc0_orthogonal_up_init: bool = False
    """Initialize FC0 adapter-up columns orthogonally at the standard weight scale."""

    moe_fc0_balance_kl_cost: float = 0.0005
    """FC0 coefficient for KL from uniform to smoothed hard top-1 traffic."""
    moe_tail_balance_kl_cost: float = 0.001
    """Tail coefficient for KL from uniform to smoothed hard top-1 traffic."""
    moe_weight_diversity_cost: float = 0.01
    """Weight-router codebook diversity coefficient."""
    moe_weight_temperature: float = 0.2
    """Weight-router balancing softmax temperature."""

    def __post_init__(self):
        if not (2 <= self.moe_fc0_experts <= 64 and 2 <= self.moe_tail_experts <= 256):
            raise ValueError("MoE expert counts must be FC0 2–64 and tail 2–256")
        if (
            min(
                self.moe_fc0_balance_kl_cost,
                self.moe_tail_balance_kl_cost,
                self.moe_mix_interval,
                self.moe_fc0_bottleneck_diversity_cost,
                self.moe_tail_bottleneck_diversity_cost,
            )
            < 0
        ):
            raise ValueError(
                "MoE regularization costs and mixing interval must be nonnegative"
            )
        if self.moe_mix_interval and self.moe_tail_experts < 4:
            raise ValueError("MoE mixing requires at least four tail experts")
        if min(self.moe_fc0_codebook_dim, self.moe_tail_codebook_dim) < 2:
            raise ValueError("MoE codebook dimensions must be at least two")
        if min(self.moe_fc0_rank, self.moe_tail_rank) < 1:
            raise ValueError("MoE adapter ranks must be positive")

    @staticmethod
    def add_model_args(parser):
        # The legacy argparse tools share the same dimension/count flags as
        # the tyro-based training, serialization and crosscheck entry points.
        for f in fields(ModelConfig):
            if type(f.default) is int:
                parser.add_argument(
                    "--" + f.name.lower().replace("_", "-"),
                    dest=f.name,
                    type=int,
                    default=f.default,
                )

    @staticmethod
    def get_model_config(args) -> "ModelConfig":
        return ModelConfig(
            **{
                f.name: getattr(args, f.name)
                for f in fields(ModelConfig)
                if type(f.default) is int
            }
        )

    # Not omitting prefix on purpose.
    quantize_config: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass(kw_only=True)
class LambdaConfig:
    start_lambda: float | None = None
    """lambda to use at first epoch."""
    end_lambda: float | None = None
    """lambda to use at last epoch."""
    jitter_lambda_sample: float = 0.0
    """std of normal distributed per sample jitter to add to lambda (default=0.0, no jitter)."""
    jitter_lambda_batch: float = 0.0
    """std of normal distributed per batch jitter to add to lambda (default=0.0, no jitter)."""
    jitter_decay_lambda_batch: float = 0.0
    """decay of batch jitter (0.0 means full decay -> independent jitter per batch. 1.0 = no decay, not allowed)."""
    lambda_: Annotated[float, tyro.conf.arg(name="lambda")] = 1.0
    """1.0=train on evaluations, 0.0=train on game results, interpolates between (default=1.0)."""
    lambda_schedule_steps: int = -1
    """Number of steps for the lambda schedule (default=-1, disabled/fallback to max_steps)."""
    lambda_cycle_warmup_pct: float = 0.3
    """Warmup percentage of the cycle steps."""
    lambda_cycle_delta: float = 0.0
    """The maximum delta added to the linear lambda change during the peak of the cycle."""
    lambda_cycle_jitter: bool = False
    """If True, the lambda jitter (both batch and sample) will be scaled by the cosine factor (0 to 1) of the cycle."""

    def __post_init__(self):
        if (self.start_lambda is not None) != (self.end_lambda is not None):
            raise ValueError(
                "Either both or none of start_lambda and end_lambda must be specified."
            )
        if self.start_lambda is None:
            self.start_lambda = self.lambda_
        if self.end_lambda is None:
            self.end_lambda = self.lambda_

        if (
            self.jitter_decay_lambda_batch < 0.0
            or self.jitter_decay_lambda_batch >= 1.0
        ):
            raise ValueError(
                "jitter_decay_lambda_batch must be in the range [0.0, 1.0)."
            )
        if self.jitter_lambda_batch < 0.0 or self.jitter_lambda_sample < 0.0:
            raise ValueError(
                "jitter_lambda_batch and jitter_lambda_sample must be non-negative."
            )
        if self.lambda_schedule_steps >= 0:
            if self.lambda_schedule_steps == 0:
                raise ValueError("lambda_schedule_steps must be positive.")
            if self.lambda_cycle_warmup_pct < 0.0 or self.lambda_cycle_warmup_pct > 1.0:
                raise ValueError(
                    "lambda_cycle_warmup_pct must be in the range [0.0, 1.0]."
                )


# parameters needed for the definition of the loss
@dataclass(kw_only=True)
class LossParams:
    in_offset: float = 270
    """offset for conversion to win on input (default=270.0)"""
    out_offset: float = 270
    """offset for conversion to win on output (default=270.0)"""
    in_scaling: float = 340
    """scaling for conversion to win on input (default=340.0)"""
    out_scaling: float = 380
    """scaling for conversion to win on output (default=380.0)"""
    lambda_config: OmitArgPrefixes[LambdaConfig] = field(default_factory=LambdaConfig)
    """lambda configuration parameters"""
    pow_exp: float = 2.5
    """exponent of the power law used for the mean error (default=2.5)"""
    qp_asymmetry: float = 0.0
    """Adjust loss if q (prediction) > p (reference) (default=0.0)"""
    w1: float = 0.0
    """weight boost parameter 1 (default=0.0)"""
    w2: float = 0.5
    """weight boost parameter 2 (default=0.5)"""
    tb_remap_base: float = 4000.0
    """Tablebase score remapping base value (default=7000.0)"""
    tb_remap_scale: float = 4000.0
    """Tablebase score remapping scale value (default=20000.0)"""
    tb_remap_decay: float = 0.85
    """Tablebase score remapping decay parameter (default=0.8)"""


@dataclass(kw_only=True)
class NNUEConfig(FeatureConfig):
    use_fake_act_quantization: bool = True
    """Whether to use fake quantization with STE for activations during training."""
    use_fake_weight_quantization: bool = True
    """Whether to use fake quantization with STE for weights during training."""
    fake_quant_warmup_steps: int = 0
    """Train without fake quantization (activations and weights) for the first N steps."""
    init_seed: int = -1
    """Seed torch right before the model is built, so the initialization is reproducible; -1 leaves it unseeded."""

    model_config: OmitArgPrefixes[ModelConfig] = field(default_factory=ModelConfig)
    loss_params: OmitArgPrefixes[LossParams] = field(default_factory=LossParams)
    optimizer_config: OmitArgPrefixes[OptimizerConfig] = field(
        default_factory=OptimizerConfig
    )
