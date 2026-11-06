from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Raw FC0 output width."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Raw FC1 output width (must match FC0)."""
