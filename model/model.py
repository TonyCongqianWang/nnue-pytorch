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
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.num_psqt_buckets, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)
        self.input_clipping = self.quantization.generate_input_clipping_config(self)

        self.input.init_weights()


    @staticmethod
    @torch.no_grad()
    def _apply_weight_clipping(config) -> None:
        """Apply one clipping config list to its tensors.

        Supports an optional 'col_end' key to restrict clipping to a column slice
        (used to exclude PSQT columns from FT weight clipping for K32Q2).
        Supports 'virtual_params' to clip weight such that weight + expanded_virtual
        stays within [min_weight, max_weight], mirroring layer-stack factorized clipping.
        """
        for group in config:
            if "min_weight" not in group and "max_weight" not in group:
                continue
            min_w = group["min_weight"]
            max_w = group["max_weight"]
            col_end = group.get("col_end", None)

            for p in group["params"]:
                p_data = p.data
                # Restrict to the column slice that should be clipped (e.g. FT-only for K32Q2)
                target = p_data[:, :col_end] if col_end is not None else p_data

                if "virtual_params" in group:
                    virtual_params = group["virtual_params"]
                    xs = p_data.shape[0] // virtual_params.shape[0]
                    ys = p_data.shape[1] // virtual_params.shape[1]
                    expanded_virtual = virtual_params.repeat(xs, ys)
                    # Take the same column slice from the expanded virtual weight
                    ev = expanded_virtual[:, :col_end] if col_end is not None else expanded_virtual
                    # Clip weight so that weight + ev stays in [min_w, max_w]
                    eff_min = target.new_full(target.shape, min_w) - ev
                    eff_max = target.new_full(target.shape, max_w) - ev
                    target.clamp_(eff_min, eff_max)
                else:
                    target.clamp_(min_w, max_w)


    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input:
            self._apply_weight_clipping(self.input_clipping)
        self._apply_weight_clipping(self.weight_clipping)


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
        psqt_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def calculate_buckets(self, piece_count: torch.Tensor):
        psqt_indices = (piece_count - 1) // 4
        layer_stack_indices = psqt_indices

        return psqt_indices, layer_stack_indices


    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        fake_quantize_acts: bool=True,
        fake_quantize_weights: bool=True,
    ):
        psqt_indices, layer_stack_indices = self.calculate_buckets(piece_count)

        l0_, wpsqt, bpsqt = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights) + (wpsqt - bpsqt) * (us - 0.5)

        return x
