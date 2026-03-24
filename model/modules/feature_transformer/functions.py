import torch
from torch import autograd

from .kernel import (
    make_sparse_input_linear_forward_kernel,
    make_sparse_input_linear_backward_kernel,
    make_sparse_input_linear_backward_sparse_kernel,
)


class SparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias, use_sparse_grad=False):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)
        ctx.use_sparse_grad = use_sparse_grad

        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        kernel = make_sparse_input_linear_forward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr(),
            ),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]
        num_inputs = weight.shape[0]

        weight_grad_dense = torch.zeros(
            num_inputs, output_size, dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_sparse_input_linear_backward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad_dense.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
            ),
        )

        if getattr(ctx, 'use_sparse_grad', False):
            # get sparsified grad tensor
            unique_indices = torch.unique(feature_indices)
            valid_indices = unique_indices[unique_indices != -1]
            sparse_values = weight_grad_dense[valid_indices]

            weight_grad = torch.sparse_coo_tensor(
                valid_indices.unsqueeze(0).to(torch.int64),
                sparse_values,
                size=(num_inputs, output_size),
                device=device,
                requires_grad=False
            )
        else:
            weight_grad = weight_grad_dense

        return None, None, weight_grad, bias_grad, None
