import operator
import struct
from collections.abc import Sequence
from functools import reduce
from typing import BinaryIO

import numpy as np
import numpy.typing as npt
import torch

from ..config import ModelConfig
from ..model import NNUEModel
from ..modules.moe_linear import ExpertParameters


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


from ..modules.router import FC0_ROUTER_TABLE_SIZE, ROUTER_TABLE_SIZE, get_router_hash


def pack_lookup_table(values, bits):
    """LSB-first IDs, byte-aligned per table; unused final bits are zero."""
    values = torch.as_tensor(values).detach().cpu().numpy().astype(np.uint16)
    if np.any(values >= (1 << bits)):
        raise ValueError("Expert ID exceeds packed width")
    chunks = []
    for start in range(0, len(values), 65536):
        block = values[start : start + 65536]
        binary = ((block[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(
            np.uint8
        )
        chunks.append(np.packbits(binary.ravel(), bitorder="little").tobytes())
    return b"".join(chunks)


def unpack_lookup_table(data, count, bits):
    expected = (count * bits + 7) // 8
    if len(data) != expected:
        raise EOFError(f"Expected {expected} packed LUT bytes, got {len(data)}")
    used = count * bits % 8
    if used and data[-1] >> used:
        raise ValueError("Nonzero packed LUT padding")
    binary = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[
        : count * bits
    ]
    values = (
        binary.reshape(count, bits).astype(np.uint16)
        * (1 << np.arange(bits, dtype=np.uint16))
    ).sum(1, dtype=np.uint16)
    return torch.from_numpy(values.astype(np.int64))


# hardcoded for now
VERSION = 0x6A448AFA
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."


class BufferExporter:
    """The default exporter — an in-memory buffer."""

    def __init__(self):
        self.buf = bytearray()

    def extend(self, data) -> None:
        self.buf.extend(data)


class CountingExporter:
    """A counting exporter — tallies the bytes written without producing output."""

    def __init__(self):
        self.size = 0

    def extend(self, data) -> None:
        self.size += len(data)


class NNUEWriter:
    """
    All values are stored in little endian.
    """

    def __init__(
        self,
        model,
        description=None,
        verbose=None,
        exporter=None,
        histogram_mode="aggregate",
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION
        if histogram_mode not in ("none", "aggregate", "all"):
            raise ValueError("histogram_mode must be none, aggregate or all")
        self.histogram_mode = (
            ("all" if verbose else "none") if verbose is not None else histogram_mode
        )
        self.verbose = self.histogram_mode != "none"
        self.exporter = BufferExporter() if exporter is None else exporter
        self._layer_histograms = {}
        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash, description)
        # Counting must not build multi-million-context LUTs or coalesce weights.
        if isinstance(self.exporter, CountingExporter):
            self.exporter.size += self._payload_size(model)
            return
        orders = model.router.export_permutations()
        fc0_table, tail_table = model.router.export_lookup_tables(orders)
        bank_orders = tuple(orders[name].tolist() for name in ("fc0", "tail"))
        self.int32(model.feature_hash ^ (model.L1 * 2))
        self.write_feature_transformer(model)
        for bucket, layers in enumerate(
            model.layer_stacks.get_coalesced_layer_stacks(
                model.router.codebook, bank_orders
            )
        ):
            first = len(layers) == 1
            self.int32(self.fc0_hash(model) if first else fc_hash)
            keys = ("ls_l1",) if first else ("ls_l2", "ls_output")
            bank = "FC0" if first else "tail"
            expert = bucket if first else bucket - model.layer_stacks.l1.count
            for layer, key in zip(layers, keys):
                self.write_fc_layer(model, layer, key, f"{bank} expert {expert}")
        self.exporter.extend(pack_lookup_table(fc0_table, 6))
        self.exporter.extend(pack_lookup_table(tail_table, 9))
        if self.histogram_mode == "aggregate":
            for name, values in self._layer_histograms.items():
                get_histogram_callback("all experts", True)(name, torch.cat(values))
        if self.verbose:
            from .routing_profiler import print_routing_profile

            print_routing_profile(model)
            if self.histogram_mode == "all":
                for name, order in orders.items():
                    print(f"Export {name}, new -> training IDs: {order.tolist()}")

    @staticmethod
    def _payload_size(model):
        size = 4 + model.L1 * 2
        size += sum(
            f.NUM_REAL_FEATURES
            * model.L1
            * torch.empty((), dtype=f.EXPORT_WEIGHT_DTYPE).element_size()
            for f in model.input.features
        )

        def layer_size(layer):
            return layer.bias_features * 4 + layer.out_features * (
                (layer.in_features + 31) // 32 * 32
            )

        stacks = model.layer_stacks
        size += stacks.l1.count * (4 + layer_size(stacks.l1))
        size += stacks.count * (4 + layer_size(stacks.l2) + layer_size(stacks.output))
        size += (model.router.fc0_lut.numel() * 6 + 7) // 8
        size += (model.router.lut.numel() * 9 + 7) // 8
        return size

    def _layer_histogram_callback(self, desc):
        if self.histogram_mode == "none":
            return None
        if self.histogram_mode == "all":
            return get_histogram_callback(desc, True)

        def collect(name, values):
            self._layer_histograms.setdefault(name, []).append(
                values.detach().flatten().cpu()
            )

        return collect

    @property
    def buf(self) -> bytearray:
        """The default exporter's in-memory buffer (kept for compatibility)."""
        return self.exporter.buf

    @staticmethod
    def layer_hash(layers, prev_hash):
        for layer in layers:
            layer_hash = (
                (0xCC03DAE4 + layer.out_features)
                ^ (prev_hash >> 1)
                ^ ((prev_hash << 31) & 0xFFFFFFFF)
            )
            if layer.out_features != 1:
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash
        return prev_hash

    @staticmethod
    def fc0_hash(model):
        return NNUEWriter.layer_hash(
            (model.layer_stacks.l1,), 0xEC42E90D ^ (model.L1 * 2)
        )

    @staticmethod
    def fc_hash(model):
        return NNUEWriter.layer_hash(
            (model.layer_stacks.l2, model.layer_stacks.output),
            NNUEWriter.fc0_hash(model),
        )

    def write_header(self, model: NNUEModel, fc_hash: int, description: str) -> None:
        self.int32(VERSION)  # version
        router_hash = get_router_hash(
            model.router.num_fc0_experts,
            model.router.num_weight_experts,
        )
        net_hash = fc_hash ^ model.feature_hash ^ (model.L1 * 2) ^ router_hash
        self.int32(net_hash)  # halfkp network hash
        encoded_description = description.encode("utf-8")
        self.int32(len(encoded_description))  # Network definition
        self.exporter.extend(encoded_description)

    def write_tensor(self, arr: torch.Tensor) -> None:
        arr = arr.detach().flatten().cpu().numpy()
        self.exporter.extend(arr.tobytes())

    def write_feature_transformer(self, model: NNUEModel) -> None:
        layer = model.input

        bias = layer.bias.data[: model.L1]

        # Get export weights (coalesced + remapped 12→11 piece types)
        weight = layer.get_export_weights()

        # biases are exported as i16s
        biases = model.quantization.quantize_feature_transformer_bias(
            bias, get_histogram_callback("", self.verbose)
        )

        self.write_tensor(biases)

        # Weights stored as [num_features][outputs]
        offset = 0
        for f in layer.features:
            n = f.NUM_REAL_FEATURES
            f_export_dtype = f.EXPORT_WEIGHT_DTYPE

            ft_histogram_callback = get_histogram_callback(f.FEATURE_NAME, self.verbose)
            segment_weight = weight[offset : offset + n]
            segment_weight = model.quantization.quantize_feature_transformer_weights(
                segment_weight, f_export_dtype, ft_histogram_callback
            )
            offset += n

            self.write_tensor(segment_weight)

    def write_fc_layer(
        self,
        model: NNUEModel,
        layer: ExpertParameters,
        layer_key: str | None,
        desc: str,
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        bias = layer.bias.data
        weight = layer.weight.data

        if layer_key is None:
            raise RuntimeError("layer_key required for quantization.")

        bias, weight = model.quantization.quantize_fc_layer(
            bias, weight, layer_key, self._layer_histogram_callback(desc)
        )

        # FC inputs are padded to 32 elements by spec.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, : weight.shape[1]] = weight
            weight = new_w

        self.write_tensor(bias)
        # Weights stored as [outputs][inputs], so we can flatten
        self.write_tensor(weight)

    def int32(self, v: int) -> None:
        self.exporter.extend(struct.pack("<I", v))


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

        self.read_feature_transformer(self.model.input)

        stacks = self.model.layer_stacks
        for layer in stacks.dense_layers():
            layer.init_coalesced_storage()
        groups = (
            ((stacks.l1,), stacks.l1.count, NNUEWriter.fc0_hash(self.model)),
            ((stacks.l2, stacks.output), stacks.count, fc_hash),
        )
        for layers, count, expected_hash in groups:
            for index in range(count):
                self.read_int32(expected_hash)
                for layer in layers:
                    self.read_fc_layer(
                        layer.coalesced_weight[
                            index * layer.out_features : (index + 1)
                            * layer.out_features
                        ],
                        layer.coalesced_bias[
                            index * layer.bias_features : (index + 1)
                            * layer.bias_features
                        ],
                        layer.layer_key,
                    )

        fc0_bytes = (FC0_ROUTER_TABLE_SIZE * 6 + 7) // 8
        tail_bytes = (ROUTER_TABLE_SIZE * 9 + 7) // 8
        self.model.router.load_lookup_tables(
            unpack_lookup_table(self.f.read(fc0_bytes), FC0_ROUTER_TABLE_SIZE, 6),
            unpack_lookup_table(self.f.read(tail_bytes), ROUTER_TABLE_SIZE, 9),
        )
        if self.f.read(1):
            raise ValueError("Unexpected trailing network data")

    def read_header(self, feature_hash: int, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        header_hash = self.read_int32()
        router_hash = get_router_hash(
            self.model.router.num_fc0_experts,
            self.model.router.num_weight_experts,
        )
        expected_hash = fc_hash ^ feature_hash ^ (self.config.L1 * 2) ^ router_hash
        if header_hash != expected_hash:
            raise ValueError(f"Expected hash {expected_hash:x}, got {header_hash:x}")
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def tensor(self, dtype: npt.DTypeLike, shape: Sequence[int]) -> torch.Tensor:
        count = reduce(operator.mul, shape, 1)
        itemsize = np.dtype(dtype).itemsize
        raw = self.f.read(count * itemsize)
        if len(raw) != count * itemsize:
            raise EOFError(
                f"Unexpected end of file: expected {count * itemsize} bytes, got {len(raw)}"
            )
        d = np.frombuffer(raw, dtype=dtype)
        d = torch.from_numpy(d.astype(np.float32))
        d = d.reshape(shape)
        return d

    def read_feature_transformer(self, layer) -> None:
        L1 = layer.num_outputs

        bias = self.tensor(np.int16, [L1])
        segments = []

        for feature in layer.features:
            dtype = np.int8 if feature.EXPORT_WEIGHT_DTYPE == torch.int8 else np.int16
            s = self.tensor(dtype, [feature.NUM_REAL_FEATURES, L1])
            segments.append(s)

        weight = torch.cat(segments, dim=0)

        bias, weight = self.model.quantization.dequantize_feature_transformer(
            bias, weight
        )

        layer.bias.data = bias.to(torch.float32)
        layer.load_export_weights(weight.to(torch.float32))

    def read_fc_layer(
        self,
        layer_weight_t: torch.Tensor,
        layer_bias_t: torch.Tensor,
        layer_key: str,
    ) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer_weight_t.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        bias = self.tensor(np.int32, layer_bias_t.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, layer_key
        )

        layer_bias = bias.to(torch.float32)
        # Strip padding.
        layer_weight = weight[: non_padded_shape[0], : non_padded_shape[1]].to(
            torch.float32
        )

        layer_bias_t.data.copy_(layer_bias)
        layer_weight_t.data.copy_(layer_weight)

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise ValueError(f"Expected: {expected:x}, got: {v:x}")
        return v
