from .full_threats import FullThreats


class GhostPawnThreats(FullThreats):
    """Unfactorized int8 weights for attacks on candidate pawn destinations.

    The two ghost colors each own 7,504 geometric (attacker, from, to) edges.
    Kings are included. Orientation follows Full_Threats.
    """

    HASH = 0x71C9E5A3
    FEATURE_NAME = "Ghost_Pawn_Threats"
    INPUT_FEATURE_NAME = "Ghost_Pawn_Threats"
    NUM_INPUTS = 15008
    NUM_REAL_FEATURES = 15008
    MAX_ACTIVE_FEATURES = 2048
