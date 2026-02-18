import cupy as cp
import torch


def _find_nearest_divisor(value: int, target: int) -> int:
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache: dict[int, int] = dict()


def _get_num_threads_for_forward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache: dict[int, int] = dict()


def _get_num_threads_for_backward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_backward_cache[output_size]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)

    return f


_sparse_input_linear_forward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(max_active_indices: int, output_size: int):
    """
    @param: max_active_indices
        The maximum number of indices that are non-zero
        for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.
    """
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_indices, output_size, num_threads, "fused_tanh")
    if key not in _sparse_input_linear_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void sparse_input_linear_forward(
    const int32_t* const input_indices,
    const float* const input_values,
    const float* const weight_param, /* theta */
    const float* const scale,        /* a */
    const float* const bias,
          float* const output
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float* const output_slice        = output + block_idx * {output_size} + slice_offset;
    const float* const bias_slice          = bias                               + slice_offset;
    const float* const scale_slice         = scale                              + slice_offset;
          float* shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const input_index_row     = input_indices + block_idx * {max_active_indices};
    const float* const input_value_row     = input_values  + block_idx * {max_active_indices};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_indices}; ++k)
    {{
        const int32_t input_index = input_index_row[k];
        const float   input_value = input_value_row[k];
        if (input_index != -1)
        {{
            /* Access raw theta parameter */
            const float* const weight_slice = weight_param + input_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                /* Fuse Reparameterization: w = a * tanh(theta / a) */
                float a = scale_slice[s];
                float theta = weight_slice[s];
                float w = a * tanhf(theta / a);
                
                shared_output_slice[s] += w * input_value;
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
            ),
            "sparse_input_linear_forward",
        )
        kernel.compile()
        _sparse_input_linear_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_forward_kernel_cache[key]


_sparse_input_linear_backward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(max_active_indices: int, output_size: int):
    num_threads = _get_num_threads_for_backward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_indices, output_size, num_threads, "fused_tanh")
    if key not in _sparse_input_linear_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
void sparse_input_linear_backward(
    const int32_t* const input_indices,
    const float* const input_values,
    const float* const weight_param, /* theta */
    const float* const scale,        /* a */
          float* const weight_grad,
          float* const bias_grad,
    const float* const output_grad
) {{
    __shared__
          float          shared_output_grad[{output_size}];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * {output_thread_slice_size};

    const float* const output_grad_slice        = output_grad + block_idx * {output_size} + slice_offset;
          float* const bias_grad_slice          = bias_grad                               + slice_offset;
    const float* const scale_slice              = scale                                   + slice_offset;
          float* shared_output_grad_slice = shared_output_grad                      + slice_offset;

    const int32_t* const input_index_row          = input_indices + block_idx * {max_active_indices};
    const float* const input_value_row          = input_values  + block_idx * {max_active_indices};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_grad_slice[s] = output_grad_slice[s];
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        const float sog = shared_output_grad_slice[s];
        if (sog != 0.0f)
        {{
            atomicAdd(&bias_grad_slice[s], sog);
        }}
    }}

    for (uint32_t k = 0; k < {max_active_indices}; ++k)
    {{
        const int32_t input_index = input_index_row[k];
        const float   input_value = input_value_row[k];
        if (input_index != -1)
        {{
            float* const weight_grad_slice  = weight_grad  + input_index * {output_size} + slice_offset;
            const float* const weight_param_slice = weight_param + input_index * {output_size} + slice_offset;
            
            #pragma unroll
            for (int s = 0; s < {output_thread_slice_size}; ++s)
            {{
                const float sog = shared_output_grad_slice[s];
                if (sog != 0.0f)
                {{
                    /* Fuse Derivative: grad_theta = grad_w * (1 - tanh^2(theta/a)) */
                    float a = scale_slice[s];
                    float theta = weight_param_slice[s];
                    float t = tanhf(theta / a);
                    float d_tanh = 1.0f - t * t;
                    
                    atomicAdd(&weight_grad_slice[s], sog * input_value * d_tanh);
                }}
            }}
        }} else break;
    }}
}}
""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
            ),
            "sparse_input_linear_backward",
        )
        kernel.compile()
        _sparse_input_linear_backward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_backward_kernel_cache[key]