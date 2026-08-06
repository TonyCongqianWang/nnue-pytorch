import operator
import struct
from collections.abc import Sequence
from functools import reduce
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from numba import njit
from torch import nn

from ..config import ModelConfig
from ..model import NNUEModel
from ..quantize import _safe_convert


def ascii_hist(name, x, bins=7):
    start, end = int(x.min()), int(x.max())
    if start >= end - bins:
        start -= (bins + 1) // 2
        end += bins // 2
    edges = np.linspace(start, end + 1, bins + 1).astype(int)
    edges = np.unique(edges)
    N, X = np.histogram(x, bins=edges)
    width = 50
    nmax = N.max()

    print(name)
    for xi, n in zip(X, N):
        bar = "#" * int(n * 1.0 * width / nmax)
        xi = f"{xi: <8.4g}".ljust(10)
        print(f"{xi}| {bar}")


def get_histogram_callback(hist_title: str, verbose: bool):
    if not verbose:
        return None

    def histogram_callback(
        hist_subtitle: str,
        values: torch.Tensor,
    ):
        total_elements = values.numel()
        hist_desc = [hist_title, hist_subtitle]
        hist_desc = " ".join(filter(None, hist_desc))

        if total_elements == 0:
            print(f"Layer '{hist_desc}' is empty.")
            return

        print("-" * 15)

        min_value = values.min().item()
        num_argmin = int((values == min_value).sum().item())
        max_value = values.max().item()
        num_argmax = int((values == max_value).sum().item())

        ascii_hist(f"{hist_desc}: ", values.detach().cpu().numpy())
        print(f"Number of elements: {total_elements}")
        print(f"Minimum value in layer is {min_value}, occurring {num_argmin} times.")
        print(f"Maximum value in layer is {max_value}, occurring {num_argmax} times.")
        print("-" * 15)

    return histogram_callback


@njit
def encode_leb_128_array(arr: npt.NDArray) -> list:
    res = []
    for v in arr:
        while True:
            byte = v & 0x7F
            v = v >> 7
            if (v == 0 and byte & 0x40 == 0) or (v == -1 and byte & 0x40 != 0):
                res.append(byte)
                break
            res.append(byte | 0x80)
    return res


@njit
def decode_leb_128_array(arr: bytes, n: int) -> npt.NDArray:
    ints = np.zeros(n)
    k = 0
    for i in range(n):
        r = 0
        shift = 0
        while True:
            byte = arr[k]
            k = k + 1
            r |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                ints[i] = r if (byte & 0x40) == 0 else r | ~((1 << shift) - 1)
                break
    return ints


# hardcoded for now
VERSION = 0x6A448AFA
DEFAULT_DESCRIPTION = "Network trained with the Inverted Bottleneck ResNet nnue-pytorch trainer."


class NNUEWriter:
    """
    All values are stored in little endian.
    """

    def __init__(
        self,
        model: NNUEModel,
        description: str | None = None,
        ft_compression: str = "none",
        verbose: bool = True,
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION

        self.buf = bytearray()
        self.verbose = verbose

        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash, description)
        self.int32(model.feature_hash ^ (model.L1 * 2))  # Feature transformer hash
        self.write_feature_transformer(model, ft_compression)

        layer_stacks = model.layer_stacks
        for bucket in range(model.num_ls_buckets):
            layers_dict = layer_stacks.get_coalesced_layers_for_bucket(bucket)
            self.int32(fc_hash)  # FC layers hash

            # 1. Write l1_to_skip_a and l1_to_skip_b layers
            self.write_fc_layer(
                model,
                layers_dict["l1_to_skip_a"],
                "ls_l1_to_skip_a",
                f"bucket {bucket} a",
            )
            self.write_fc_layer(
                model,
                layers_dict["l1_to_skip_b"],
                "ls_l1_to_skip_b",
                f"bucket {bucket} b",
            )

            # 2. Write blocks
            for i, b_data in enumerate(layers_dict["blocks"]):
                # Write fc_up biases: crelu first, then sqr
                self.write_fc_bias(
                    model,
                    b_data["bias_crelu"],
                    f"block_{i}_up_bias_crelu",
                    f"bucket {bucket} block {i} bias_crelu",
                )
                self.write_fc_bias(
                    model,
                    b_data["bias_sqr"],
                    f"block_{i}_up_bias_sqr",
                    f"bucket {bucket} block {i} bias_sqr",
                )

                # Write fc_up weights (bias-free layer)
                self.write_fc_weight(
                    model,
                    b_data["fc_up"].weight.data,
                    f"block_{i}_up",
                    f"bucket {bucket} block {i} up",
                )

                if "fc_down" in b_data:
                    self.write_fc_layer(
                        model,
                        b_data["fc_down"],
                        f"block_{i}_down",
                        f"bucket {bucket} block {i} down",
                    )
                else:
                    self.write_fc_layer(
                        model,
                        b_data["fc_final"],
                        f"block_{i}_final",
                        f"bucket {bucket} block {i} final",
                    )

    @staticmethod
    def fc_hash(model: NNUEModel) -> int:
        # InputSlice hash
        prev_hash = 0xEC42E90D
        prev_hash ^= model.L1 * 2

        # 1. l1_to_skip_a hash
        layer_hash = 0xCC03DAE4
        layer_hash += model.residual_dim // 2
        layer_hash ^= prev_hash >> 1
        layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
        prev_hash = layer_hash

        # 2. l1_to_skip_b hash
        layer_hash = 0xCC03DAE4
        layer_hash += model.residual_dim // 2
        layer_hash ^= prev_hash >> 1
        layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
        prev_hash = layer_hash

        # 2. Block hashes
        for i in range(model.num_blocks):
            is_final = i == model.num_blocks - 1
            # fc_up
            layer_hash = 0xCC03DAE4
            layer_hash += model.expanded_dim
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash

            if not is_final:
                # fc_down
                layer_hash = 0xCC03DAE4
                layer_hash += model.residual_dim
                layer_hash ^= prev_hash >> 1
                layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
                prev_hash = layer_hash
            else:
                # fc_final
                layer_hash = 0xCC03DAE4
                layer_hash += 1
                layer_hash ^= prev_hash >> 1
                layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
                prev_hash = layer_hash

        return prev_hash

    def write_header(
        self, model: NNUEModel, fc_hash: int, description: str
    ) -> None:
        self.int32(VERSION)  # version
        self.int32(
            fc_hash ^ model.feature_hash ^ (model.L1 * 2)
        )  # halfkp network hash
        encoded_description = description.encode("utf-8")
        self.int32(len(encoded_description))  # Network definition
        self.buf.extend(encoded_description)

    def write_leb_128_array(self, arr: npt.NDArray) -> None:
        buf = encode_leb_128_array(arr)
        self.int32(len(buf))
        self.buf.extend(buf)

    def write_tensor(self, arr: torch.Tensor, compression="none") -> None:
        arr = arr.detach().flatten().cpu().numpy()
        if compression == "none":
            self.buf.extend(arr.tobytes())
        elif compression == "leb128":
            self.buf.extend(b"COMPRESSED_LEB128")
            self.write_leb_128_array(arr)
        else:
            raise ValueError("Invalid compression method.")

    def write_feature_transformer(
        self, model: NNUEModel, ft_compression: str
    ) -> None:
        layer = model.input

        # biases for both L1 and skip path
        bias = layer.bias.data

        # Get export weights
        export_weight = layer.get_export_weights()
        weight = export_weight[:, : model.L1]
        residual_weight = export_weight[:, model.L1 :]

        # biases are exported as i16s
        biases = model.quantization.quantize_feature_transformer_bias(
            bias, get_histogram_callback("ft_bias", self.verbose)
        )

        self.write_tensor(biases, ft_compression)

        # Weights stored as [num_features][outputs]
        offset = 0
        for f in layer.features:
            n = f.NUM_REAL_FEATURES
            f_export_dtype = f.EXPORT_WEIGHT_DTYPE

            ft_histogram_callback = get_histogram_callback(
                f.FEATURE_NAME, self.verbose
            )
            segment_weight = weight[offset : offset + n]
            segment_residual_weight = residual_weight[offset : offset + n]

            # Quantize weights (both use ft_weight scale factor)
            segment_weight = (
                model.quantization.quantize_feature_transformer_weights(
                    segment_weight, f_export_dtype, ft_histogram_callback
                )
            )
            segment_residual_weight = (
                model.quantization.quantize_feature_transformer_weights(
                    segment_residual_weight,
                    f_export_dtype,
                    ft_histogram_callback,
                )
            )

            segment_compression = (
                ft_compression if f_export_dtype != torch.int8 else "none"
            )
            offset += n

            self.write_tensor(segment_weight, segment_compression)
            self.write_tensor(segment_residual_weight, segment_compression)

    def write_fc_layer(
        self,
        model: NNUEModel,
        layer: nn.Linear,
        layer_key: str | None,
        desc: str,
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        bias = layer.bias.data
        weight = layer.weight.data

        if layer_key is None:
            raise RuntimeError("layer_key required for quantization.")

        bias, weight = model.quantization.quantize_fc_layer(
            bias, weight, layer_key, get_histogram_callback(desc, self.verbose)
        )

        # FC inputs are padded to 32 elements by spec.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, : weight.shape[1]] = weight
            weight = new_w

        self.write_tensor(bias, "none")
        # Weights stored as [outputs][inputs], so we can flatten
        self.write_tensor(weight, "none")

    def write_fc_bias(
        self,
        model: NNUEModel,
        bias: torch.Tensor,
        bias_key: str,
        desc: str,
    ) -> None:
        quantized_bias = _safe_convert(
            bias.mul(model.quantization.weight_scales_dict[bias_key]),
            torch.int32,
        )
        callback = get_histogram_callback(desc, self.verbose)
        if callback is not None:
            callback(bias_key, quantized_bias)
        self.write_tensor(quantized_bias, "none")

    def write_fc_weight(
        self,
        model: NNUEModel,
        weight: torch.Tensor,
        layer_key: str,
        desc: str,
    ) -> None:
        weight_key = f"{layer_key}_weight"
        quantized_weight = _safe_convert(
            weight.mul(model.quantization.weight_scales_dict[weight_key]),
            torch.int8,
        )
        callback = get_histogram_callback(desc, self.verbose)
        if callback is not None:
            callback(weight_key, quantized_weight)

        # Pad to 32 elements by spec
        num_input = quantized_weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(
                quantized_weight.shape[0], num_input, dtype=torch.int8
            )
            new_w[:, : quantized_weight.shape[1]] = quantized_weight
            quantized_weight = new_w

        self.write_tensor(quantized_weight, "none")

    def int32(self, v: int) -> None:
        self.buf.extend(struct.pack("<I", v))


class NNUEReader:
    def __init__(
        self,
        f: BinaryIO,
        feature_name: str,
        config: ModelConfig,
    ):
        self.f = f
        self.feature_name = feature_name
        self.model = NNUEModel(feature_name, config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        self.read_header(self.model.feature_hash, fc_hash)
        self.read_int32(
            self.model.feature_hash ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.model.zero_virtual_weights()

        self.read_feature_transformer(self.model.input, self.model.residual_dim)

        num_ls_buckets = self.model.num_ls_buckets
        for b in range(num_ls_buckets):
            self.read_int32(fc_hash)  # FC layers hash
            layers_dict = self.model.layer_stacks.get_coalesced_layers_for_bucket(
                b
            )

            # 1. Read l1_to_skip_a and l1_to_skip_b layers
            self.read_fc_layer(
                layers_dict["l1_to_skip_a"].weight.data,
                layers_dict["l1_to_skip_a"].bias.data,
                "ls_l1_to_skip_a",
            )
            a_out = self.model.layer_stacks.l1_to_skip_a.out_features
            self.model.layer_stacks.l1_to_skip_a.linear.weight.data[
                b * a_out : (b + 1) * a_out
            ].copy_(layers_dict["l1_to_skip_a"].weight.data)
            self.model.layer_stacks.l1_to_skip_a.linear.bias.data[
                b * a_out : (b + 1) * a_out
            ].copy_(layers_dict["l1_to_skip_a"].bias.data)

            self.read_fc_layer(
                layers_dict["l1_to_skip_b"].weight.data,
                layers_dict["l1_to_skip_b"].bias.data,
                "ls_l1_to_skip_b",
            )
            b_out = self.model.layer_stacks.l1_to_skip_b.out_features
            self.model.layer_stacks.l1_to_skip_b.linear.weight.data[
                b * b_out : (b + 1) * b_out
            ].copy_(layers_dict["l1_to_skip_b"].weight.data)
            self.model.layer_stacks.l1_to_skip_b.linear.bias.data[
                b * b_out : (b + 1) * b_out
            ].copy_(layers_dict["l1_to_skip_b"].bias.data)

            # 2. Read blocks
            for i, b_data in enumerate(layers_dict["blocks"]):
                crelu_bias = self.read_fc_bias(
                    f"block_{i}_up_bias_crelu", b_data["bias_crelu"].shape
                )
                sqr_bias = self.read_fc_bias(
                    f"block_{i}_up_bias_sqr", b_data["bias_sqr"].shape
                )

                block = self.model.layer_stacks.blocks[i]
                block.bias_crelu.data.reshape(
                    num_ls_buckets, block.expanded_dim
                )[b].copy_(crelu_bias)
                block.bias_sqr.data.reshape(
                    num_ls_buckets, block.expanded_dim
                )[b].copy_(sqr_bias)

                self.read_fc_weight(b_data["fc_up"].weight.data, f"block_{i}_up")
                up_out = block.fc_up.out_features
                block.fc_up.linear.weight.data[
                    b * up_out : (b + 1) * up_out
                ].copy_(b_data["fc_up"].weight.data)

                if "fc_down" in b_data:
                    self.read_fc_layer(
                        b_data["fc_down"].weight.data,
                        b_data["fc_down"].bias.data,
                        f"block_{i}_down",
                    )
                    down_out = block.fc_down.out_features
                    block.fc_down.linear.weight.data[
                        b * down_out : (b + 1) * down_out
                    ].copy_(b_data["fc_down"].weight.data)
                    block.fc_down.linear.bias.data[
                        b * down_out : (b + 1) * down_out
                    ].copy_(b_data["fc_down"].bias.data)
                else:
                    self.read_fc_layer(
                        b_data["fc_final"].weight.data,
                        b_data["fc_final"].bias.data,
                        f"block_{i}_final",
                    )
                    final_out = block.fc_final.out_features
                    block.fc_final.linear.weight.data[
                        b * final_out : (b + 1) * final_out
                    ].copy_(b_data["fc_final"].weight.data)
                    block.fc_final.linear.bias.data[
                        b * final_out : (b + 1) * final_out
                    ].copy_(b_data["fc_final"].bias.data)

    def read_header(self, feature_hash: int, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        self.read_int32(fc_hash ^ feature_hash ^ (self.config.L1 * 2))
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def read_leb_128_array(
        self, dtype: npt.DTypeLike, shape: Sequence[int]
    ) -> torch.Tensor:
        len_bytes = self.read_int32()
        d = self.f.read(len_bytes)
        if len(d) != len_bytes:
            raise EOFError("Unexpected end of file when reading compressed data.")

        res = torch.tensor(
            decode_leb_128_array(d, reduce(operator.mul, shape, 1)),
            dtype=torch.float32,
        )
        res = res.reshape(shape)
        return res

    def peek(self, length: int = 1) -> bytes:
        pos = self.f.tell()
        data = self.f.read(length)
        self.f.seek(pos)
        return data

    def determine_compression(self) -> str:
        leb128_magic = b"COMPRESSED_LEB128"
        if self.peek(len(leb128_magic)) == leb128_magic:
            self.f.read(len(leb128_magic))  # actually advance the file pointer
            return "leb128"
        else:
            return "none"

    def tensor(self, dtype: npt.DTypeLike, shape: Sequence[int]) -> torch.Tensor:
        compression = self.determine_compression()

        if compression == "none":
            d = np.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
            d = torch.from_numpy(d.astype(np.float32))
            d = d.reshape(shape)
            return d
        elif compression == "leb128":
            return self.read_leb_128_array(dtype, shape)
        else:
            raise ValueError("Invalid compression method.")

    def read_feature_transformer(self, layer, residual_dim: int) -> None:
        num_outputs = layer.num_outputs
        L1 = num_outputs - residual_dim

        bias = self.tensor(np.int16, [L1 + residual_dim])
        segments = []
        segments_residual = []

        for feature in layer.features:
            dtype = (
                np.int8
                if feature.EXPORT_WEIGHT_DTYPE == torch.int8
                else np.int16
            )
            s = self.tensor(dtype, [feature.NUM_REAL_FEATURES, L1])
            segments.append(s)
            s_residual = self.tensor(
                dtype, [feature.NUM_REAL_FEATURES, residual_dim]
            )
            segments_residual.append(s_residual)

        weight = torch.cat(segments, dim=0)
        residual_weight = torch.cat(segments_residual, dim=0)

        # Dequantize FT parameters
        bias = bias.divide(
            self.model.quantization.weight_scales_dict["ft_bias"]
        ).to(torch.float32)
        weight = weight.divide(
            self.model.quantization.weight_scales_dict["ft_weight"]
        ).to(torch.float32)
        residual_weight = residual_weight.divide(
            self.model.quantization.weight_scales_dict["ft_weight"]
        ).to(torch.float32)

        # Load weights back
        layer.bias.data.copy_(bias)
        export_weight = torch.cat([weight, residual_weight], dim=1)
        layer.load_export_weights(export_weight)

    def read_fc_layer(
        self,
        layer_weight_t: torch.Tensor,
        layer_bias_t: torch.Tensor,
        layer_key: str,
    ) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer_weight_t.shape
        padded_shape = (
            non_padded_shape[0],
            ((non_padded_shape[1] + 31) // 32) * 32,
        )

        bias = self.tensor(np.int32, layer_bias_t.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, layer_key
        )

        layer_bias = bias.to(torch.float32)
        # Strip padding.
        layer_weight = weight[
            : non_padded_shape[0], : non_padded_shape[1]
        ].to(torch.float32)

        layer_bias_t.data.copy_(layer_bias)
        layer_weight_t.data.copy_(layer_weight)

    def read_fc_bias(self, bias_key: str, shape) -> torch.Tensor:
        bias = self.tensor(np.int32, shape)
        bias = bias.divide(self.model.quantization.weight_scales_dict[bias_key])
        return bias.to(torch.float32)

    def read_fc_weight(
        self, layer_weight_t: torch.Tensor, layer_key: str
    ) -> None:
        non_padded_shape = layer_weight_t.shape
        padded_shape = (
            non_padded_shape[0],
            ((non_padded_shape[1] + 31) // 32) * 32,
        )
        weight = self.tensor(np.int8, padded_shape)
        weight_key = f"{layer_key}_weight"
        weight = weight.divide(
            self.model.quantization.weight_scales_dict[weight_key]
        )
        layer_weight = weight[
            : non_padded_shape[0], : non_padded_shape[1]
        ].to(torch.float32)
        layer_weight_t.data.copy_(layer_weight)

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise ValueError(f"Expected: {expected:x}, got: {v:x}")
        return v
