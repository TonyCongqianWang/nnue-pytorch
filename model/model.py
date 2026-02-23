import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .features import FeatureSet
from .modules import DoubleFeatureTransformer, LayerStacks
from .quantize import QuantizationConfig, QuantizationManager

class SpoofedInputModule:
    """
    A dummy module that mimics the old combined DoubleFeatureTransformer 
    for read-only operations like binary network export.
    """
    def __init__(self, l1_module, psqt_module):
        self.l1 = l1_module
        self.psqt = psqt_module
        
    def __call__(self, white_indices, white_values, black_indices, black_values):
        """
        Mimics the forward pass of the original DoubleFeatureTransformer.
        Computes L1 and PSQT separately and concatenates them on the fly.
        """
        # 1. Compute both parts
        w_l1, b_l1 = self.l1(white_indices, white_values, black_indices, black_values)
        w_psqt, b_psqt = self.psqt(white_indices, white_values, black_indices, black_values)
        
        # 2. Concatenate along the feature dimension (dim=1)
        wp = torch.cat([w_l1, w_psqt], dim=1)
        bp = torch.cat([b_l1, b_psqt], dim=1)
        
        return wp, bp

    def parameters(self, recurse=True):
        """Spoofs the parameters iterator for the optimizer."""
        for p in self.l1.parameters(recurse=recurse):
            yield p
        for p in self.psqt.parameters(recurse=recurse):
            yield p

    @property
    def num_inputs(self):
        return self.l1.num_inputs

    @property
    def num_outputs(self):
        return self.l1.num_outputs + self.psqt.num_outputs

    @property
    def weight(self):
        return torch.cat([self.l1.weight, self.psqt.weight], dim=1)

    @property
    def bias(self):
        if self.l1.bias is None:
            return None
        return torch.cat([self.l1.bias, self.psqt.bias], dim=0)

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
        
        self.input_l1 = DoubleFeatureTransformer(feature_set.num_features, self.L1)
        self.input_psqt = DoubleFeatureTransformer(feature_set.num_features, self.num_psqt_buckets)
        
        self.input = SpoofedInputModule(self.input_l1, self.input_psqt)

        self.feature_set = feature_set
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self._init_layers()

    def _init_layers(self):
        self._zero_virtual_feature_weights()
        self._init_psqt()

    def _zero_virtual_feature_weights(self):
        """
        We zero all virtual feature weights because there's not need for them
        to be initialized; they only aid the training of correlated features.
        """
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                self.input_l1.weight[a:b, :].zero_()
                self.input_psqt.weight[a:b, :].zero_()

    def _init_psqt(self):   
        input_weights = self.input_psqt.weight
        input_bias = self.input_psqt.bias
        # 1.0 / kPonanzaConstant
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

            # Bias doesn't matter because it cancels out during
            # inference during perspective averaging. We set it to 0
            # just for the sake of it. It might still diverge away from 0
            # due to gradient imprecision but it won't change anything.
            for i in range(self.num_psqt_buckets):
                input_weights[:, i] = new_weights
                input_bias[i] = 0.0
                
    def replace_with_quantized_weights(self, target: str):
        """
        Applies fake quantization to specific modules. 
        Valid targets: "psqt", "l1", "feature_transformer", "layer_stacks", "all"
        """
        with torch.no_grad():
            # --- Feature Transformer (L1 + PSQT) ---
            if target in ["psqt", "l1", "feature_transformer", "all"]:
                bias = self.input_l1.bias if self.input_l1.bias is not None else torch.zeros(self.L1, device=self.input_l1.weight.device)
                
                # Quantize -> Dequantize
                q_b, q_w, q_psqt = self.quantization.quantize_feature_transformer(
                    bias, self.input_l1.weight, self.input_psqt.weight
                )
                dq_b, dq_w, dq_psqt = self.quantization.dequantize_feature_transformer(q_b, q_w, q_psqt)
                
                if target in ["psqt", "feature_transformer", "all"]:
                    self.input_psqt.weight.copy_(dq_psqt.to(self.input_psqt.weight.dtype))
                
                if target in ["l1", "feature_transformer", "all"]:
                    self.input_l1.weight.copy_(dq_w.to(self.input_l1.weight.dtype))
                    if self.input_l1.bias is not None:
                        self.input_l1.bias.copy_(dq_b.to(self.input_l1.bias.dtype))
                        
            # --- Layer Stacks ---
            if target in ["layer_stacks", "all"]:
                layers = [
                    (self.layer_stacks.l1.linear, False),
                    (self.layer_stacks.l2.linear, False),
                    (self.layer_stacks.output.linear, True)
                ]
                
                for layer, is_output in layers:
                    q_b, q_w = self.quantization.quantize_fc_layer(
                        layer.bias, layer.weight, output_layer=is_output
                    )
                    dq_b, dq_w = self.quantization.dequantize_fc_layer(
                        q_b, q_w, output_layer=is_output
                    )
                    
                    layer.bias.copy_(dq_b.to(layer.bias.dtype))
                    layer.weight.copy_(dq_w.to(layer.weight.dtype))

    def set_learnable_modules(self, config: dict[str, bool]):
        """
        Dynamically freezes/unfreezes parts of the network.
        Example config: {"psqt": False, "l1": True, "layer_stacks": True}
        """
        # 1. Update requires_grad for specified modules
        if "psqt" in config:
            for p in self.input_psqt.parameters():
                p.requires_grad = config["psqt"]
                
        if "l1" in config:
            for p in self.input_l1.parameters():
                p.requires_grad = config["l1"]
                
        if "layer_stacks" in config:
            for p in self.layer_stacks.parameters():
                p.requires_grad = config["layer_stacks"]

    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
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

    def clip_threat_weights(self):
        if self.feature_set.name.startswith("Full_Threats"):
            # Point to input_l1
            p = self.input_l1.weight[0:self.threat_features]
            min_weight = -128 / 255
            max_weight = 127 / 255
            p.data.clamp_(min_weight, max_weight)

    def set_feature_set(self, new_feature_set: FeatureSet):
        """
        This method attempts to convert the model from using the self.feature_set
        to new_feature_set. Currently only works for adding virtual features.
        """
        if self.feature_set.name == new_feature_set.name:
            return

        # TODO: Implement this for more complicated conversions.
        #       Currently we support only a single feature block.
        if len(self.feature_set.features) > 1:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

        # Currently we only support conversion for feature sets with
        # one feature block each so we'll dig the feature blocks directly
        # and forget about the set.
        old_feature_block = self.feature_set.features[0]
        new_feature_block = new_feature_set.features[0]

        # next(iter(new_feature_block.factors)) is the way to get the
        # first item in a OrderedDict. (the ordered dict being str : int
        # mapping of the factor name to its size).
        # It is our new_feature_factor_name.
        # For example old_feature_block.name == "HalfKP"
        # and new_feature_factor_name == "HalfKP^"
        # We assume here that the "^" denotes factorized feature block
        # and we would like feature block implementers to follow this convention.
        # So if our current feature_set matches the first factor in the new_feature_set
        # we only have to add the virtual feature on top of the already existing real ones.
        if old_feature_block.name == next(iter(new_feature_block.factors)):
            # We can just extend with zeros since it's unfactorized -> factorized
            self.input.expand_input_layer(new_feature_block.num_virtual_features)
            self.feature_set = new_feature_set
        else:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )
    
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
    ) -> torch.Tensor:

        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)        
        wpsqt, bpsqt = self.input_psqt(white_indices, white_values, black_indices, black_values)
        
        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        psqt_out = (wpsqt - bpsqt) * (us - 0.5)

        if getattr(self, "train_psqt_only", False):
            return psqt_out

        w, b = self.input_l1(white_indices, white_values, black_indices, black_values)

        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))        
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        l0_s = torch.split(l0_, self.L1 // 2, dim=1)
        
        # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
        # and it's more efficient to divide by 128 instead.
        scale = 127 / 128
        l0_s1 = (l0_s[0] * l0_s[1] * scale, l0_s[2] * l0_s[3] * scale)

        l0_ = torch.cat(l0_s1, dim=1)
        x = self.layer_stacks(l0_, layer_stack_indices) + psqt_out

        return x
