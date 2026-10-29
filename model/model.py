import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
from .perspective import ARCHITECTURE_HASH, FALLBACK, LIVE_EXPERTS
from .quantize import QuantizationConfig, QuantizationManager


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

        if config.quantize_config != QuantizationConfig():
            raise ValueError(
                "Handcrafted perspective format requires upstream quantization scales"
            )
        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_ls_buckets = LIVE_EXPERTS
        self.register_buffer(
            "architecture_tag", torch.tensor(ARCHITECTURE_HASH, dtype=torch.int64)
        )
        self.register_buffer(
            "training_exposure", torch.zeros(2, LIVE_EXPERTS, dtype=torch.bool)
        )
        self.register_buffer(
            "exported_mask", torch.ones(2, LIVE_EXPERTS, dtype=torch.bool)
        )
        self.register_buffer("loaded_export", torch.tensor(False))
        # Keep the persisted marker for checkpoint compatibility, but inspect it
        # only when loading. Mode changes must not read a CUDA scalar on the host.
        self._export_loaded = False
        self.register_load_state_dict_post_hook(self._restore_export_status)

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights()

    def __setstate__(self, state):
        # Older .pt objects have the persisted marker but not the Python cache.
        super().__setstate__(state)
        self._export_loaded = bool(self.loaded_export)
        if self._restore_export_status not in self._load_state_dict_post_hooks.values():
            self.register_load_state_dict_post_hook(self._restore_export_status)

    def _restore_export_status(self, module, incompatible_keys):
        self._export_loaded = bool(self.loaded_export)

    def mark_export_loaded(self):
        self.loaded_export.fill_(True)
        self._export_loaded = True

    def train(self, mode=True):
        if mode and self._export_loaded:
            raise ValueError(
                "Exported .nnue files are inference-only; resume a matching checkpoint"
            )
        super().train(mode)
        if not mode and not self._export_loaded:
            self.prepare_inference()
        return self

    @torch.no_grad()
    def prepare_inference(self):
        """Refresh fallback/masks on-device, without CPU export or scalar reads."""
        if self._export_loaded:
            return
        has_history = self.training_exposure.any()
        torch._assert_async(
            (self.training_exposure.any(dim=1).all() | ~has_history),
            "Cannot prepare a role with no training exposure",
        )
        for role, bank in enumerate((self.layer_stacks, self.layer_stacks.opp)):
            for layer, (weight, bias) in zip(
                bank.layers(), bank.fallback_parameters(self.training_exposure[role])
            ):
                layer.fallback_weight.copy_(
                    torch.where(has_history, weight, layer.fallback_weight)
                )
                layer.fallback_bias.copy_(
                    torch.where(has_history, bias, layer.fallback_bias)
                )
        # Fresh synthetic models retain all enabled IDs until they have history.
        self.exported_mask.copy_(
            torch.where(has_history, self.training_exposure, self.exported_mask)
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

    def calculate_buckets(self, stm_bucket, opp_bucket):
        result = []
        for role, ids in enumerate((stm_bucket, opp_bucket)):
            ids = ids.long().flatten()
            valid = (ids >= 0) & (ids < LIVE_EXPERTS)
            if self.training:
                torch._assert_async(
                    valid.all(), "Unsupported material in training batch"
                )
                result.append(ids)
            else:
                safe = ids.clamp(0, LIVE_EXPERTS - 1)
                exposed = self.exported_mask[role, safe]
                result.append(torch.where(valid & exposed, safe, FALLBACK))
        return tuple(result)

    @torch.no_grad()
    def record_exposure(self, own, opp):
        for role, ids in enumerate((own, opp)):
            valid = (ids >= 0) & (ids < LIVE_EXPERTS)
            torch._assert_async(valid.all(), "Unsupported material in training batch")
            self.training_exposure[role].scatter_(0, ids.long().flatten(), True)
        # Union before DDP can broadcast rank-zero buffers at the next forward.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mask = self.training_exposure.to(torch.uint8)
            torch.distributed.all_reduce(mask, op=torch.distributed.ReduceOp.MAX)
            self.training_exposure.copy_(mask.bool())

    def load_state_dict(self, state_dict, *args, **kwargs):
        if (
            "architecture_tag" not in state_dict
            or int(state_dict["architecture_tag"]) != ARCHITECTURE_HASH
        ):
            raise ValueError("Incompatible handcrafted-perspective checkpoint")
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        stm_bucket: torch.Tensor,
        opp_bucket: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        layer_stack_indices = self.calculate_buckets(stm_bucket, opp_bucket)
        if self.training and torch.is_grad_enabled():
            self.record_exposure(stm_bucket, opp_bucket)

        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        x = self.layer_stacks(
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights
        )

        return x
