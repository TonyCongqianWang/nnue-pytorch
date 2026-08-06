from dataclasses import dataclass
from typing import Annotated

import tyro


# Residual network config
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1536
    """Size of first hidden layer."""
    residual_dim: Annotated[int, tyro.conf.arg(name="residual-dim")] = 32
    """Size of the residual skip path."""
    expanded_dim: Annotated[int, tyro.conf.arg(name="expanded-dim")] = 64
    """Expanded dimension of inverted bottleneck blocks."""
    num_blocks: Annotated[int, tyro.conf.arg(name="num-blocks")] = 2
    """Number of inverted bottleneck blocks."""
