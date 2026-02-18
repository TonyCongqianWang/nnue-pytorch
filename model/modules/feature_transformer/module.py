import math
import torch
from torch import nn
import torch.nn.functional as F

from .functions import SparseLinearFunction

def expand_2d_bucketed_scales(
    scale: torch.Tensor, 
    in_boundaries: torch.Tensor, 
    out_boundaries: torch.Tensor, 
    num_inputs: int,
    output_size: int
) -> torch.Tensor:
    """
    Expands a (num_in_buckets, num_out_buckets) scale tensor 
    into a full (num_inputs, output_size) dense tensor.
    """
    full_scale = torch.empty((num_inputs, output_size), dtype=scale.dtype, device=scale.device)
    
    in_start = 0
    for i, in_bound in enumerate(in_boundaries):
        in_end = min(in_bound.item(), num_inputs)
        if in_start >= in_end:
            continue
            
        out_start = 0
        for j, out_bound in enumerate(out_boundaries):
            out_end = min(out_bound.item(), output_size)
            if out_start >= out_end:
                continue
                
            # Fill the rectangular region with the single scalar value
            full_scale[in_start:in_end, out_start:out_end] = scale[i, j]
            
            out_start = out_end
        in_start = in_end
        
    return full_scale

class SmartSlice:
    def __init__(self, proxy, key):
        self.proxy = proxy
        self.key = key

    def zero_(self):
        # tanh(0) = 0, so we can zero the master parameter directly
        with torch.no_grad():
            self.proxy.outer._weight_param[self.key].zero_()
        return self

    def fill_(self, value):
        with torch.no_grad():
            target_a = self.proxy._get_scale_slice(self.key)
            inverse_vals = self.proxy._inverse_op(value, target_a)
            self.proxy.outer._weight_param[self.key].copy_(inverse_vals)
        return self

    @property
    def shape(self):
        return self.proxy.outer._weight_param[self.key].shape

    def __repr__(self):
        return repr(self.proxy.data[self.key])

    def __getattr__(self, name):
        # forwarding all other attributes to the original
        return getattr(self.proxy.data[self.key], name)


class WeightProxy:
    def __init__(self, outer, a):
        self.outer = outer
        self.a = a

    @property
    def data(self):
        return self.a * torch.tanh(self.outer._weight_param.data / self.a)

    @data.setter
    def data(self, new_val):
        # Full assignment: use the full 'a' tensor
        self.outer._weight_param.data = self._inverse_op(new_val, self.a)

    # --- Helper to get the scale corresponding to a slice ---
    def _get_scale_slice(self, key):
        return self.a.expand_as(self.outer._weight_param)[key]

    def _inverse_op(self, w, scale):
        # w: value to set (tensor or scalar)
        # scale: corresponding 'a' values (must broadcast to w)
        ratio = (w / scale).clamp(-0.995, 0.995)
        return scale * 0.5 * torch.log((1 + ratio) / (1 - ratio))

    # --- Metadata Delegation (Fast) ---
    @property
    def device(self): return self.outer._weight_param.device
    @property
    def dtype(self): return self.outer._weight_param.dtype
    @property
    def shape(self): return self.outer._weight_param.shape
    def size(self, *args, **kwargs): return self.outer._weight_param.size(*args, **kwargs)
    def numel(self): return self.outer._weight_param.numel()
    def __len__(self): return len(self.outer._weight_param)

    def __getitem__(self, key):
        return SmartSlice(self, key)

    def __setitem__(self, key, value):
        target_a = self._get_scale_slice(key)
        inverse_val = self._inverse_op(value, target_a)
        self.outer._weight_param.data[key] = inverse_val

    def __repr__(self):
        return repr(self.data)

    def __getattr__(self, name):
        # forwarding all other attributes to the original
        return getattr(self.data, name)


class BaseFeatureTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs, out_scale_config=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if out_scale_config is None:
            out_scale_config = dict(
                scale=torch.tensor([[1.0]]),
                in_boundaries=torch.tensor([num_inputs]),
                out_boundaries=torch.tensor([num_outputs]),
            )
        out_scale = expand_2d_bucketed_scales(**out_scale_config, num_inputs=num_inputs, output_size=num_outputs)
        
        self.register_buffer('scale', out_scale_config['scale']) 
        self.register_buffer('in_boundaries', out_scale_config['in_boundaries'])
        self.register_buffer('out_boundaries', out_scale_config['out_boundaries'])
        self.register_buffer('out_scale', out_scale)

        self._weight_param = nn.Parameter(torch.empty((num_inputs, num_outputs), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(num_outputs, dtype=torch.float32))
        
        self.reset_parameters()

    @property
    def weight(self):
        return WeightProxy(self, self.out_scale)

    @weight.setter
    def weight(self, value):
        if isinstance(value, nn.Parameter):
            self._weight_param.data = WeightProxy(self, self.out_scale)._inverse_tensor(value.data)
        else:
            self._weight_param.data = WeightProxy(self, self.out_scale)._inverse_tensor(value)

    def reset_parameters(self):
        sigma = math.sqrt(1 / self.num_inputs)
        with torch.no_grad():
            initial_w = torch.zeros_like(self._weight_param).uniform_(-sigma, sigma)
            self.weight.data = initial_w
            self.bias.uniform_(-sigma, sigma)

    def expand_input_layer(self, additional_features):
        assert additional_features >= 0
        if additional_features == 0:
            return

        with torch.no_grad():
            new_weight_param = F.pad(
                self._weight_param.data, 
                (0, 0, 0, additional_features), 
                value=0
            )
            self._weight_param = nn.Parameter(new_weight_param)
            self.num_inputs += additional_features


class FeatureTransformer(BaseFeatureTransformer):
    def forward(self, feature_indices, feature_values):
        return SparseLinearFunction.apply(
            feature_indices, feature_values, self._weight_param,
            self.scale, self.in_boundaries, self.out_boundaries, self.bias
        )


class DoubleFeatureTransformer(BaseFeatureTransformer):
    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        w = self._weight_param
        a = self.scale
        ib = self.in_boundaries
        ob = self.out_boundaries
        b = self.bias
        return (
            SparseLinearFunction.apply(
                feature_indices_0, feature_values_0, w, a, ib, ob, b,
            ),
            SparseLinearFunction.apply(
                feature_indices_1, feature_values_1, w, a, ib, ob, b,
            ),
        )