from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
    ft_dynamic_bias_dim: Annotated[int, tyro.conf.arg(name="ft-dynamic-bias-dim")] = 256
    """Size of dynamic bias accumulator extension."""

    @property
    def ft_outputs(self) -> int:
        return self.L1 + self.ft_dynamic_bias_dim
