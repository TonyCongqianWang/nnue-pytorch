import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import TypedDict, NotRequired, Tuple

# --- Configuration ---

@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_hidden: float = 64.0
    weight_scale_out: float = 16.0
    ft_quantized_one: float = 256.0
    hidden_quantized_one: float = 128.0


# --- Helper Functions ---

def grad_scale(x, scale_factor):
    y = x
    y_grad = x * scale_factor
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    return (x.round() - x).detach() + x

def get_quantization_params(log_alpha, bits):
    q_max = (2 ** (bits - 1)) - 1
    q_min = -q_max
    
    alpha_round = round_pass(log_alpha)
    effective_range = 2 ** alpha_round
    # Using q_max+1 (128) as the divisor allows exact power-of-two alignment
    scale = effective_range / (q_max + 1)
    
    return scale, q_min, q_max

def compute_requantization_factors(scale_in, scale_weight, scale_out):
    """
    Computes int32 multiplier and right shift for:
    output = (input * weight * multiplier) >> shift
    """
    real_factor = (scale_in * scale_weight) / scale_out
    
    if real_factor == 0:
        return 0, 0

    shift = 0
    while real_factor < (1 << 30):
        real_factor *= 2
        shift += 1
        
    while real_factor >= (1 << 31):
        real_factor /= 2
        shift -= 1
        
    multiplier = int(round(real_factor))
    return multiplier, shift


# --- Autograd Function ---

class SymmetricLSQPoT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, log_alpha, bits):
        scale, q_min, q_max = get_quantization_params(log_alpha, bits)
        
        x_div = input / scale
        x_div_clipped = x_div.clamp(q_min, q_max)
        x_quant = round_pass(x_div_clipped)
        
        x_fake = x_quant * scale
        
        ctx.save_for_backward(x_div, scale)
        ctx.q_params = (q_min, q_max)
        return x_fake

    @staticmethod
    def backward(ctx, grad_output):
        x_div, scale = ctx.saved_tensors
        q_min, q_max = ctx.q_params
        
        mask_lo = (x_div < q_min)
        mask_hi = (x_div > q_max)
        mask_in = ~(mask_lo | mask_hi)
        
        grad_input = grad_output * mask_in.float()
        
        grad_scale_elem = torch.zeros_like(grad_output)
        grad_scale_elem[mask_lo] = grad_output[mask_lo] * q_min
        grad_scale_elem[mask_hi] = grad_output[mask_hi] * q_max
        
        quant_error = (x_div[mask_in].round() - x_div[mask_in])
        grad_scale_elem[mask_in] = grad_output[mask_in] * quant_error
        
        dims = list(range(grad_scale_elem.ndim))
        keep_dims = [i for i, s in enumerate(scale.shape) if s > 1]
        sum_dims = [d for d in dims if d not in keep_dims]
        
        grad_scale_val = grad_scale_elem.sum(dim=sum_dims, keepdim=True)
        grad_log_alpha = grad_scale_val * scale * math.log(2)
        
        return grad_input, grad_log_alpha, None


# --- Modules ---

class LSQActivation(nn.Module):
    def __init__(self, in_features, num_groups=1, bits=8, init_range=128.0):
        super().__init__()
        self.bits = bits
        self.num_groups = num_groups
        self.in_features = in_features
        
        if in_features > 0 and in_features % num_groups != 0:
             raise Exception(f"num_groups {num_groups} must divide in_features {in_features}")
        
        init_exp = math.log2(init_range)
        self.log_alpha = nn.Parameter(torch.full((1, num_groups, 1), init_exp))
        self.register_buffer('init_done', torch.tensor(False))

    def forward(self, x):
        B, C = x.shape
        group_size = C // self.num_groups
        x_grouped = x.view(B, self.num_groups, group_size)
        
        if not self.init_done:
            with torch.no_grad():
                max_val = x_grouped.abs().amax(dim=(0, 2), keepdim=True)
                max_val = max_val.clamp(min=1e-5)
                self.log_alpha.data = torch.log2(max_val)
                self.init_done.fill_(True)
        
        num_elements = B * group_size
        q_max = (2 ** (self.bits - 1)) - 1
        g_factor = 1.0 / math.sqrt(num_elements * q_max) if num_elements > 0 else 1.0
        
        log_alpha_scaled = grad_scale(self.log_alpha, g_factor)
        
        x_quant = SymmetricLSQPoT.apply(x_grouped, log_alpha_scaled, self.bits)
        
        return x_quant.view(B, C)

    def get_scales(self):
        scale, _, _ = get_quantization_params(self.log_alpha, self.bits)
        return scale.view(-1)


class LSQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.log_alpha_w = nn.Parameter(torch.zeros(out_features, 1))
        self.register_buffer('w_init_done', torch.tensor(False))

    def forward(self, x):
        w_quant = self.quantized_weight
        return F.linear(x, w_quant, self.bias)

    @property
    def quantized_weight(self):
        if not self.w_init_done:
            with torch.no_grad():
                w_abs = self.weight.abs().view(self.out_features, -1)
                max_val = w_abs.amax(dim=1, keepdim=True)
                max_val = max_val.clamp(min=1e-5)
                self.log_alpha_w.data = torch.log2(max_val)
                self.w_init_done.fill_(True)

        q_max = (2 ** (self.bits - 1)) - 1
        g_factor = 1.0 / math.sqrt(self.weight.numel() * q_max)
        
        log_alpha_w_scaled = grad_scale(self.log_alpha_w, g_factor)
        return SymmetricLSQPoT.apply(self.weight, log_alpha_w_scaled, self.bits)

    def get_weight_scales(self):
        scale, _, _ = get_quantization_params(self.log_alpha_w, self.bits)
        return scale.view(-1)


class LSQEmbeddingParams(nn.Module):
    def __init__(self, num_outputs, bits=16):
        super().__init__()
        self.bits = bits
        self.log_alpha = nn.Parameter(torch.zeros(1, num_outputs))
        self.register_buffer('init_done', torch.tensor(False))

    def forward(self, weight):
        if not self.init_done:
            with torch.no_grad():
                max_val = weight.abs().amax(dim=0, keepdim=True)
                max_val = max_val.clamp(min=1e-5)
                self.log_alpha.data = torch.log2(max_val)
                self.init_done.fill_(True)

        q_max = (2 ** (self.bits - 1)) - 1
        g_factor = 1.0 / math.sqrt(weight.numel() * q_max)
        
        log_alpha_scaled = grad_scale(self.log_alpha, g_factor)
        return SymmetricLSQPoT.apply(weight, log_alpha_scaled, self.bits)
        
    def get_scales(self):
        scale, _, _ = get_quantization_params(self.log_alpha, self.bits)
        return scale.view(-1)

# Backwards compatibility dummy
class QuantizationManager:
    def __init__(self, config):
        pass
    def generate_weight_clipping_config(self, model):
        return []