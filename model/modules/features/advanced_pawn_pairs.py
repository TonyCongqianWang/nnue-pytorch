from .full_threats import FullThreats


class AdvancedPawnPairs(FullThreats):
    """Sixth/seventh-rank pawns paired with non-pawn, non-king pieces.

    Both pawn and partner colors are represented in each perspective. Enabled
    only with at most five non-pawns on the board, including both kings.
    """

    HASH = 0x8D36B4E1
    FEATURE_NAME = "Advanced_Pawn_Pairs"
    INPUT_FEATURE_NAME = "Advanced_Pawn_Pairs"
    NUM_INPUTS = 16384
    NUM_REAL_FEATURES = 16384
    MAX_ACTIVE_FEATURES = 48
