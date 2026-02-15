import math
import torch
from torch import nn
from .functions import SparseLinearFunction
from ...quantize import LSQEmbeddingParams

class BaseFeatureTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs_l1, num_outputs_psqt):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs_l1 = num_outputs_l1
        self.num_outputs_psqt = num_outputs_psqt
        self.total_outputs = num_outputs_l1 + num_outputs_psqt

        sigma = math.sqrt(1 / num_inputs)

        self.weight = nn.Parameter(
            torch.rand(num_inputs, self.total_outputs, dtype=torch.float32) * (2 * sigma)
            - sigma
        )
        self.bias = nn.Parameter(
            torch.rand(self.total_outputs, dtype=torch.float32) * (2 * sigma) - sigma
        )
        
        self.quant_l1 = LSQEmbeddingParams(num_outputs_l1, bits=16)
        self.quant_psqt = LSQEmbeddingParams(num_outputs_psqt, bits=32)

    def get_quantized_weights_and_bias(self):
        w_l1 = self.weight[:, :self.num_outputs_l1]
        w_psqt = self.weight[:, self.num_outputs_l1:]
        
        wq_l1 = self.quant_l1(w_l1)
        wq_psqt = self.quant_psqt(w_psqt)
        
        w_q = torch.cat([wq_l1, wq_psqt], dim=1)
        return w_q, self.bias

class FeatureTransformer(BaseFeatureTransformer):
    def forward(self, feature_indices, feature_values):
        w_q, b = self.get_quantized_weights_and_bias()
        return SparseLinearFunction.apply(
            feature_indices, feature_values, w_q, b
        )

class DoubleFeatureTransformer(BaseFeatureTransformer):
    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        w_q, b = self.get_quantized_weights_and_bias()
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                w_q,
                b,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                w_q,
                b,
            ),
        )