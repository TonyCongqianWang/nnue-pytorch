from dataclasses import dataclass
from typing import Annotated

import tyro


@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    res_dim: Annotated[int, tyro.conf.arg(name="res-dim")] = 32
    """Dimension of residual stream."""
    expanded_dim: Annotated[int, tyro.conf.arg(name="expanded-dim")] = 64
    """Expanded dimension inside bottleneck blocks."""
    num_blocks: Annotated[int, tyro.conf.arg(name="num-blocks")] = 2
    """Number of inverted bottleneck blocks."""

