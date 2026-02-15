import torch
from torch import nn

from .config import ModelConfig
from .features import FeatureSet
from .modules import DoubleFeatureTransformer, LayerStacks
from .quantize import QuantizationConfig, QuantizationManager, LSQActivation


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        self.threat_features = config.threat_features
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = DoubleFeatureTransformer(
            feature_set.num_features, self.L1, self.num_psqt_buckets
        )
        self.feature_set = feature_set
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        
        # L0 Activation (between FT and LayerStacks)
        # Input is 2*L1 (White + Black)
        self.l0_activation = LSQActivation(2 * self.L1, num_groups=4, bits=8)

        self._init_layers()

    def _init_layers(self):
        self._zero_virtual_feature_weights()
        self._init_psqt()

    def _zero_virtual_feature_weights(self):
        weights = self.input.weight
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                weights[a:b, :] = 0.0
        self.input.weight = nn.Parameter(weights)

    def _init_psqt(self):
        input_weights = self.input.weight
        input_bias = self.input.bias
        scale = 1 / self.quantization.nnue2score

        with torch.no_grad():
            initial_values = self.feature_set.get_initial_psqt_features()
            new_weights = (
                torch.tensor(
                    initial_values,
                    device=input_weights.device,
                    dtype=input_weights.dtype,
                )
                * scale
            )
            for i in range(self.num_psqt_buckets):
                input_weights[:, self.L1 + i] = new_weights
                input_bias[self.L1 + i] = 0.0

        self.input.weight = nn.Parameter(input_weights)
        self.input.bias = nn.Parameter(input_bias)

    def clip_weights(self):
        pass

    def clip_threat_weights(self):
        pass

    def set_feature_set(self, new_feature_set: FeatureSet):
        if self.feature_set.name != new_feature_set.name:
            raise Exception("Not implemented")

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        white_values: torch.Tensor,
        black_indices: torch.Tensor,
        black_values: torch.Tensor,
        psqt_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, wpsqt = torch.split(wp, self.L1, dim=1)
        b, bpsqt = torch.split(bp, self.L1, dim=1)
        
        # Raw accumulation combination
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        
        # Quantize / Activate
        l0_quant = self.l0_activation(l0_)
        
        # Split and Mix
        # l0_quant is [Batch, 2*L1]
        # l0_mixed is [Batch, L1]
        l0_s = torch.chuk(l0_quant, 4, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_mixed = torch.cat(l0_s1, dim=1)

        # Gather PSQT values        
        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        
        x = self.layer_stacks(l0_mixed, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

        return x