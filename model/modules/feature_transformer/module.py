import math
import torch
from torch import nn
import torch.nn.functional as F

from .functions import SparseLinearFunction

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
    def __init__(self, num_inputs, num_outputs, out_scale=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if out_scale is None:
            out_scale = torch.full((1, num_outputs), 16.0, dtype=torch.float32)
        else:
            out_scale = out_scale.view(1, -1)
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
            feature_indices, feature_values, self._weight_param, self.out_scale, self.bias
        )


class DoubleFeatureTransformer(BaseFeatureTransformer):
    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        w = self._weight_param
        a = self.out_scale
        b = self.bias
        return (
            SparseLinearFunction.apply(
                feature_indices_0, feature_values_0, w, a, b,
            ),
            SparseLinearFunction.apply(
                feature_indices_1, feature_values_1, w, a, b,
            ),
        )