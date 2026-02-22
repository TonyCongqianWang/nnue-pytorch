from dataclasses import dataclass


# 3 layer fully connected network
@dataclass
class ModelConfig:
    threat_features: int = 60144
    L1: int = 1024
    L2: int = 31
    L3: int = 32


# parameters needed for the definition of the loss
@dataclass
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5


import itertools

def get_default_qat_training_schedule(phase_durations: tuple[int, int]) -> dict[int, dict]:
    """
    Takes the duration (in epochs) of the first 3 phases and computes their start epochs.
    Example: durations (15, 15, 15) -> phase starts: [0, 15, 30, 45]
    """
    # itertools.accumulate computes the running total (e.g., [15, 15+15, 15+15+15])
    phase_starts = [0] + list(itertools.accumulate(phase_durations))
    
    return {
        # Phase 1: Train PSQT only
        phase_starts[0]: {
            "train_psqt_only": True,
            "learnable": {"psqt": True, "l1": False, "layer_stacks": False},
        },
        # Phase 2: Quantize PSQT, unfreeze the rest, bump LR
        phase_starts[1]: {
            "quantize": "psqt", 
            "learnable": {"psqt": False, "l1": True, "layer_stacks": True},
            "reset_optimizer": True,
            "reset_scheduler": True,
        },
        # Phase 3: Quantize full FeatureTransformer, train LayerStacks
        phase_starts[2]: {
            "quantize": "feature_transformer", 
            "learnable": {"psqt": False, "l1": False, "layer_stacks": True},
            "reset_optimizer": True,
        },
    }
    