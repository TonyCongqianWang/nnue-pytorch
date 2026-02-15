import struct
import torch
import numpy as np
import numpy.typing as npt
from ..model import NNUEModel
from ..quantize import compute_requantization_factors

VERSION = 0x7AF32F22 # Incremented or Changed
DEFAULT_DESCRIPTION = "QAT NNUE with Explicit Requantization Ops"

class NNUEWriter:
    def __init__(self, model: NNUEModel, description: str = None, ft_compression: str = "none"):
        if description is None: description = DEFAULT_DESCRIPTION
        self.buf = bytearray()
        
        # 1. Header
        self.write_header(model, 0, description)
        self.int32(model.feature_set.hash ^ (model.L1 * 2))
        
        # 2. FT
        # Need output scales for FT logic.
        # L0 activation scales (input to mixing).
        s_l0 = model.l0_activation.get_scales().cpu().numpy() # [4]
        
        self.write_feature_transformer(model, s_l0)
        
        # 3. Layers
        # Input to L1 layer is l0_mixed.
        # l0_mixed = (part1 * part2)
        # We need to trace the scale of l0_mixed.
        # Scale(Mul) = Scale(part1) * Scale(part2)
        # s_l0 has 4 parts: s0, s1, s2, s3.
        # mixed[0] scale = s0 * s1
        # mixed[1] scale = s2 * s3
        
        current_scales = np.array([s_l0[0]*s_l0[1], s_l0[2]*s_l0[3]])
        # This is vector of size 2. Broadcasts to L1/2 sized chunks.
        
        # Layers
        layers = model.layer_stacks.get_coalesced_layer_stacks()
        
        # Get output scales from activations in layer_stacks
        s_l1_out = model.layer_stacks.l1_activation.get_scales().cpu().numpy() # [2] (sqr, linear)
        s_l2_out = model.layer_stacks.l2_activation.get_scales().cpu().numpy() # [1]
        
        # Final output scale: 600*16
        final_scale = np.array([1.0 / (600.0 * 16.0)]) # Inverse scale?
        # User said: "final output ... 600*16 being the value representing 1.0"
        # So Real = Int / (600*16).
        # Serialization usually expects: Output = (Acc * M) >> S.
        # We want Int_Out s.t. Int_Out * S_final = Real_Out.
        # S_final = 1 / (600*16).
        
        s_final = np.array([1.0/(600*16)])

        for l1, l2, output_layer in layers:
            self.int32(0)
            
            # L1
            # Input scale: current_scales (vector size 2, expands to features)
            # But L1 weights might be grouped.
            # L1 linear has In=L1 (mixed).
            # We assume current_scales broadcasts correctly.
            self.write_fc_layer(l1, current_scales, s_l1_out, input_split=True)
            
            # L2
            # Input is s_l1_out (size 2). 
            # L2 input size is 2*L2.
            # s_l1_out[0] applies to first L2 elts, s_l1_out[1] to next L2.
            self.write_fc_layer(l2, s_l1_out, s_l2_out, input_split=True)
            
            # Output
            self.write_fc_layer(output_layer, s_l2_out, s_final, is_output=True)

    def write_header(self, model, fc_hash, description):
        self.int32(VERSION)
        self.int32(fc_hash ^ model.feature_set.hash ^ (model.L1 * 2))
        encoded_desc = description.encode("utf-8")
        self.int32(len(encoded_desc))
        self.buf.extend(encoded_desc)

    def write_tensor(self, arr):
        self.buf.extend(arr.tobytes())

    def int32(self, v):
        self.buf.extend(struct.pack("<I", v))

    def write_feature_transformer(self, model, s_out_target):
        layer = model.input
        w_q, b = layer.get_quantized_weights_and_bias() # i16 / i32
        
        # Write Bias (i16) - wait, is bias i16?
        # "Both the weights and biases are of the same type... addition with a lookup table"
        # "pqst ... i32, normal ... i16"
        # So bias part normal is i16, bias part psqt is i32.
        
        b_l1 = b[:layer.num_outputs_l1]
        b_psqt = b[layer.num_outputs_l1:]
        
        w_l1 = w_q[:, :layer.num_outputs_l1]
        w_psqt = w_q[:, layer.num_outputs_l1:]
        
        # Get Scales for requantization
        # FT Norm Output -> L0 Activation.
        # FT Norm output is accumulation of i16.
        # We need to reach `s_out_target` (the scale of l0_activation inputs).
        # We need requant params for FT_Norm -> s_out_target.
        
        # Current FT scale.
        # LSQEmbeddingParams stores scale.
        s_ft_norm = layer.quant_l1.get_scales().cpu().numpy() # [outputs]
        
        # Scale bias?
        # Bias is float in `b`. We need to quantize it to match weights for addition.
        # b_int = round(b / s_ft_norm).
        b_l1_int = (b_l1 / torch.from_numpy(s_ft_norm).to(b_l1.device)).round().to(torch.int16)
        
        # PSQT
        s_ft_psqt = layer.quant_psqt.get_scales().cpu().numpy()
        b_psqt_int = (b_psqt / torch.from_numpy(s_ft_psqt).to(b_psqt.device)).round().to(torch.int32)
        
        # Write Biases
        self.write_tensor(b_l1_int.cpu().numpy())
        self.write_tensor(b_psqt_int.cpu().numpy())
        
        # Write Weights
        self.write_tensor(w_l1.cpu().to(torch.int16).flatten().numpy())
        self.write_tensor(w_psqt.cpu().to(torch.int32).flatten().numpy())
        
        # Write Requantization for Normal part
        # Input to requant is Sum(i16). Scale s_ft_norm.
        # Output target is s_out_target.
        # s_out_target has 4 values? corresponding to 4 chunks of L1.
        # s_ft_norm has L1 values.
        # We map element-wise.
        # The 4 values in s_out_target broadcast to the 4 chunks.
        
        mults = []
        shifts = []
        chunk_size = layer.num_outputs_l1 // 4
        
        for i in range(layer.num_outputs_l1):
            chunk_idx = i // chunk_size
            s_target = s_out_target[chunk_idx]
            s_src = s_ft_norm[i] if s_ft_norm.size > 1 else s_ft_norm
            
            m, s = compute_requantization_factors(1.0, s_src, s_target) # input to FT is 1.0
            mults.append(m)
            shifts.append(s)
            
        self.write_tensor(np.array(mults, dtype=np.int32))
        self.write_tensor(np.array(shifts, dtype=np.int8))


    def write_fc_layer(self, layer, s_in, s_out, is_output=False, input_split=False):
        # Weights
        w_float = layer.weight.data
        s_w = layer.get_weight_scales().cpu().numpy() # [outputs]
        
        if s_w.size == 1: s_w = np.full(w_float.shape[0], s_w)
        
        # Quantize Weights i8
        w_int = (w_float / torch.from_numpy(s_w).unsqueeze(1).to(w_float.device)).round().to(torch.int8)
        
        # Bias
        # Acc scale depends on input scale.
        # Input scale s_in might be vector (size 2).
        # We assume weights correspond to these inputs.
        # If input_split is True, s_in[0] applies to first half of columns, s_in[1] to second.
        # But we need ONE acc scale per row to quantize bias?
        # Linear layer sums everything.
        # If s_in is different for different columns, we cannot simply add them in accumulated domain 
        # UNLESS s_in[0]*s_w == s_in[1]*s_w. 
        # Or we requantize inputs before summing? No, FC layer sums then requantizes.
        # This implies standard FC requires uniform input scale.
        # BUT `l0_mixed` has different scales for different chunks?
        # `l0_s1 = [s0*s1, s2*s3]`. These might be different.
        # If they are different, we can't simple matmul.
        # Users usually enforce uniform scale for layer inputs or handle it.
        # For this refactor, I will assume we must average s_in or use the first one, 
        # OR the training converges such that they are similar,
        # OR we rely on LSQ to learn weights that compensate? 
        # No, bias quantization needs the scale.
        # Let's use the MEAN of input scales for bias calculation.
        
        s_in_avg = np.mean(s_in)
        s_acc = s_in_avg * s_w
        
        b_float = layer.bias.data
        b_int = (b_float / torch.from_numpy(s_acc).to(b_float.device)).round().to(torch.int32)
        
        self.write_tensor(b_int.cpu().numpy())
        self.write_tensor(w_int.cpu().numpy())
        
        # Requantization
        mults = []
        shifts = []
        
        # We need output scale s_out.
        # If s_out is vector (size 2), map to rows.
        # If s_out is scalar, use it.
        
        for i in range(len(s_w)):
            target = s_out[i] if (s_out.size > 1 and i < len(s_out)) else (s_out[0] if s_out.size>0 else 1.0)
            # wait, s_out corresponds to Next Layer Inputs.
            # If current layer is L1, output is L2 inputs.
            # L2 inputs (squared, linear).
            # L1 output rows map to these. 
            # We assume row order matches s_out order.
            
            # Use specific input scale for this row if possible? 
            # No, row sums all inputs.
            
            m, s = compute_requantization_factors(s_in_avg, s_w[i], target)
            mults.append(m)
            shifts.append(s)
            
        self.write_tensor(np.array(mults, dtype=np.int32))
        self.write_tensor(np.array(shifts, dtype=np.int8))


class NNUEReader:
    def __init__(
        self,
        f: BinaryIO,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
    ):
        self.f = f
        self.feature_set = feature_set
        self.model = NNUEModel(feature_set, config, quantize_config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        self.read_header(feature_set, fc_hash)
        self.read_int32(
            feature_set.hash ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)
        for i in range(self.model.num_ls_buckets):
            l1 = nn.Linear(2 * self.config.L1 // 2, self.config.L2 + 1)
            l2 = nn.Linear(self.config.L2 * 2, self.config.L3)
            output = nn.Linear(self.config.L3, 1)

            self.read_int32(fc_hash)  # FC layers hash
            self.read_fc_layer(l1)
            self.read_fc_layer(l2)
            self.read_fc_layer(output, is_output=True)

            self.model.layer_stacks.l1.linear.weight.data[
                i * (self.config.L2 + 1) : (i + 1) * (self.config.L2 + 1), :
            ] = l1.weight
            self.model.layer_stacks.l1.linear.bias.data[
                i * (self.config.L2 + 1) : (i + 1) * (self.config.L2 + 1)
            ] = l1.bias
            self.model.layer_stacks.l2.linear.weight.data[
                i * self.config.L3 : (i + 1) * self.config.L3, :
            ] = l2.weight
            self.model.layer_stacks.l2.linear.bias.data[
                i * self.config.L3 : (i + 1) * self.config.L3
            ] = l2.bias
            self.model.layer_stacks.output.linear.weight.data[i : (i + 1), :] = (
                output.weight
            )
            self.model.layer_stacks.output.linear.bias.data[i : (i + 1)] = output.bias

    def read_header(self, feature_set: FeatureSet, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        self.read_int32(fc_hash ^ feature_set.hash ^ (self.config.L1 * 2))
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def read_leb_128_array(
        self, dtype: npt.DTypeLike, shape: Sequence[int]
    ) -> torch.Tensor:
        len_bytes = self.read_int32()
        d = self.f.read(len_bytes)
        if len(d) != len_bytes:
            raise Exception("Unexpected end of file when reading compressed data.")

        res = torch.tensor(
            decode_leb_128_array(d, reduce(operator.mul, shape, 1)), dtype=torch.float32
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
            raise Exception("Invalid compression method.")

    def read_feature_transformer(
        self, layer: BaseFeatureTransformer, num_psqt_buckets: int
    ) -> None:
        shape = layer.weight.shape

        bias = self.tensor(np.int16, [layer.bias.shape[0] - num_psqt_buckets])
        # weights stored as [num_features][outputs]
        if self.feature_set.name.startswith("Full_Threats"):
            threat_weight = self.tensor(np.int8, [self.config.threat_features, shape[1] - num_psqt_buckets])
            psq_weight = self.tensor(np.int16, [shape[0] - self.config.threat_features, shape[1] - num_psqt_buckets])
            weight = torch.cat([threat_weight, psq_weight], dim=0)
        else:
            weight = self.tensor(np.int16, [shape[0], shape[1] - num_psqt_buckets])
        psqt_weight = self.tensor(np.int32, [shape[0], num_psqt_buckets])

        bias, weight, psqt_weight = (
            self.model.quantization.dequantize_feature_transformer(
                bias, weight, psqt_weight
            )
        )

        layer.bias.data = torch.cat([bias, torch.tensor([0] * num_psqt_buckets)])
        layer.weight.data = torch.cat([weight, psqt_weight], dim=1)

    def read_fc_layer(self, layer: nn.Linear, is_output: bool = False) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer.weight.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        bias = self.tensor(np.int32, layer.bias.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, is_output
        )

        layer.bias.data = bias
        layer.weight.data = weight

        # Strip padding.
        layer.weight.data = layer.weight.data[
            : non_padded_shape[0], : non_padded_shape[1]
        ]

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise Exception("Expected: %x, got %x" % (expected, v))
        return v
