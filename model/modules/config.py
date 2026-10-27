from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Total raw second-layer width across both MoE6 roles."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Total raw third-layer width across both MoE6 roles."""
