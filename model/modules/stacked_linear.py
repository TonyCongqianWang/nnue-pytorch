import torch
from torch import nn
import torch.nn.functional as F

class StackedLinear(nn.Module):
    """
    Applies a linear transformation y = xA^T + b.
    The weights A and bias b are selected based on an index (stack_idx).
    
    Implementation using direct gathering and BMM
    rather than masking or looping.
    """
    def __init__(self, in_features, out_features, num_stacks, bias=True):
        super(StackedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_stacks = num_stacks
        
        # Shape: (Num_Stacks, Out, In) to match Linear standard (y = xA^T)
        self.weight = nn.Parameter(torch.Tensor(num_stacks, out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_stacks, out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, stack_idx):
        """
        input: (Batch, In_Features)
        stack_idx: (Batch,) of integers in [0, num_stacks-1]
        """
        # 1. Gather the weights for this batch: (Batch, Out, In)
        # We index the 'stacks' dimension (dim 0)
        batch_weights = self.weight[stack_idx] 
        
        # 2. Batch Matrix Multiplication
        # Input needs to be (Batch, In, 1) to multiply with (Batch, Out, In)
        # OR (Batch, 1, In) to multiply with (Batch, In, Out)
        
        # Calculation: y = x @ W.T
        # x: (Batch, 1, In)
        # W: (Batch, Out, In) -> Transpose to (Batch, In, Out)
        # Result: (Batch, 1, Out)
        
        x_unsqueezed = input.unsqueeze(1) # (B, 1, In)
        w_transposed = batch_weights.permute(0, 2, 1) # (B, In, Out)
        
        output = torch.bmm(x_unsqueezed, w_transposed).squeeze(1) # (B, Out)
        
        if self.bias is not None:
            batch_bias = self.bias[stack_idx] # (Batch, Out)
            output += batch_bias
            
        return output


class FactorizedStackedLinear(StackedLinear):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__(in_features, out_features, count)

        self.factorized_linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        merged_weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        merged_bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)

        stacked_output = F.linear(x, merged_weight, merged_bias)

        return self.select_output(stacked_output, ls_indices)

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = super().at_index(index)

        layer.weight.add_(self.factorized_linear.weight)
        layer.bias.add_(self.factorized_linear.bias)

        return layer

    @torch.no_grad()
    def coalesce_weights(self) -> None:
        for i in range(self.count):
            begin = i * self.out_features
            end = (i + 1) * self.out_features

            self.linear.weight[begin:end, :].add_(self.factorized_linear.weight)
            self.linear.bias[begin:end].add_(self.factorized_linear.bias)

        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()
