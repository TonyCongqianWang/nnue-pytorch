import torch
from torch import nn
import torch.nn.functional as F
from ..quantize import LSQLinear

class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        
        self.linear = LSQLinear(in_features, out_features * count, bias=True)
        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        init_weight = self.linear.weight[0 : self.out_features, :]
        init_bias = self.linear.bias[0 : self.out_features]

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        # Optimization: gather weights then bmm
        w_all = self.linear.quantized_weight
        b_all = self.linear.bias
        
        w_reshaped = w_all.view(self.count, self.out_features, self.in_features)
        b_reshaped = b_all.view(self.count, self.out_features)
        
        indices = ls_indices.flatten().long()
        
        # [Batch, Out, In]
        w_selected = w_reshaped.index_select(0, indices)
        # [Batch, Out]
        b_selected = b_reshaped.index_select(0, indices)
        
        # [Batch, In, 1]
        x_uns = x.unsqueeze(2)
        # [Batch, Out]
        out = torch.bmm(w_selected, x_uns).squeeze(2)
        
        return out + b_selected

    @torch.no_grad()
    def at_index(self, index: int) -> LSQLinear:
        layer = LSQLinear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])
        layer.log_alpha_w.copy_(self.linear.log_alpha_w[begin:end])
        layer.w_init_done.copy_(self.linear.w_init_done)

        return layer


class FactorizedStackedLinear(StackedLinear):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__(in_features, out_features, count)
        self.factorized_linear = LSQLinear(in_features, out_features, bias=True)

        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        out_stacked = super().forward(x, ls_indices)
        out_factorized = self.factorized_linear(x)
        return out_stacked + out_factorized

    @torch.no_grad()
    def at_index(self, index: int) -> LSQLinear:
        layer = super().at_index(index)
        layer.weight.add_(self.factorized_linear.weight)
        layer.bias.add_(self.factorized_linear.bias)
        return layer

    @torch.no_grad()
    def coalesce_weights(self) -> None:
        w_fact = self.factorized_linear.weight
        b_fact = self.factorized_linear.bias
        
        for i in range(self.count):
            begin = i * self.out_features
            end = (i + 1) * self.out_features

            self.linear.weight[begin:end, :].add_(w_fact)
            self.linear.bias[begin:end].add_(b_fact)

        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()