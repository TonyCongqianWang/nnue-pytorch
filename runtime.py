"""Shared device and precision policy for training, export and crosscheck."""

import os
import warnings
from typing import Literal

import torch

MatmulPrecision = Literal["auto", "highest", "high"]


def resolve_device(requested="auto", metadata=None):
    requested = str(requested)
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested == "auto":
        preferred = (metadata or {}).get("device_type")
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
        if preferred and not available.get(preferred, False):
            warnings.warn(
                f"Checkpoint backend {preferred} is unavailable; auto is falling back. "
                "This is a cross-device diagnostic, not matched-device parity.",
                stacklevel=2,
            )
        requested = (
            preferred
            if preferred and available.get(preferred, False)
            else next(name for name, present in available.items() if present)
        )
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {device} is unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested device {device} is unavailable")
        torch.cuda.set_device(device)
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested MPS device is unavailable")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device {device}")
    return device


def configure_runtime(
    device, precision: MatmulPrecision = "auto", *, metadata=None, quantized=True
):
    # A forced cuBLAS override defeats the local IEEE guards around routing
    # and expert generation, even when the global dense policy is "highest".
    if (
        device.type == "cuda"
        and os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1"
    ):
        raise RuntimeError(
            "Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE before starting Python; "
            "it bypasses the IEEE guards for routing and expert generation."
        )
    if precision == "auto":
        legacy_override = os.environ.get("NNUE_TF32")
        precision = (metadata or {}).get("matmul_precision")
        if precision is None:
            precision = (
                "high"
                if device.type == "cuda" and quantized and legacy_override != "0"
                else "highest"
            )
    if device.type != "cuda" or os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        precision = "highest"
    torch.set_float32_matmul_precision(precision)
    result = {
        "device_type": device.type,
        "device": str(device),
        "matmul_precision": precision,
        "sensitive_matmul_precision": "highest",
        "dtype": "float32",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["device_capability"] = torch.cuda.get_device_capability(device)
    print(f"Runtime: {result}", flush=True)
    return result
