# Features

This doc is pretty much a copy-paste from [here](https://github.com/glinscott/nnue-pytorch/pull/30)

A feature transformer is defined over a `feature_set`. A `feature_set` is a list of `feature_block`s. Each `feature_block` contains a name, hash, num_real_features, num_virtual_features, num_features, and knows some stuff about how to manipulate its indices. The number of real features and virtual features is used to distinguish factorizers, because they have to be handled differently on serialization to .nnue. A `feature_block` consists of `factors` which are an `OrderedDict` of `(str, int)` pairs where the first is the name of the factor and the second is its size. The first factor is assumed to be real, all the following ones are assumed to be virtual. `feature_set` concatenates these blocks and exposes similar operations. A `feature_block` knows how to coalesce its feature to real features. A `feature_block` is identified by it's name, for example `HalfKP` is a block, as is `HalfKP^` which denotes a factorized `HalfKP` (this is just a convention, but is strict and is used in some places for conversion ability discovery). A `feature_set` name is `'+'.join(...)` of names of its blocks.

From now on the feature set used for learning and serialization/deserialization has to be specified explicitly as a program argument. The new argument `--features=...` takes a name of the feature set. For example `--features="HalfKP^"` or `--features="HalfKP"` or some imaginary `--features="HalfKP^+FancyFeatures+MoreFancyFeatures^"`. This argument is present in both train.py and serialize.py.

The current semantics are as follows:

1. When training a new net from scratch - `--features` specifies the feature set to use for learning. The feature transformer weights are initialized normally for the real features and zero initialized for the virtual features.
2. When resuming training from a .pt model - `--features` specifies the feature set to use for learning. If the feature set specified doesn't match the feature set from the .pt model a conversion is attempted. Right now only a conversion of feature set with a single block from non-factorized to factorized is supported. The factorized block must have the non-factorized features as the first factor. The virtual feature weights are initialized to zero.
3. When converting .ckpt to .nnue - `--features` specifies the features as stored in the .ckpt file. The user must pass the correct feature set through `--features` because it can't be inferred from the .ckpt. If the features from `--features` and the saved model don't match it'll likely stack trace on some dimension mismatch.
4. When converting .pt to .nnue - `--features` is ignored, the `feature_set` from the saved model is used, the weights are coalesced when writing the .nnue file.
5. When converting .nnue to .pt - `--features` specifies the features in the .nnue file. The resulting .pt model has the same feature_set. Note that when resuming training this model can be converted to a compatible feature_set, see point 2.

## Advanced pawn pairs

The default experimental set is
`Full_Threats+PP_3Wide+Advanced_Pawn_Pairs+K32H3^`.
`Advanced_Pawn_Pairs` pairs actual pawns on their own sixth or seventh rank
with every knight, bishop, rook, and queen of either color. Features are active
only with at most five non-pawns total, counting both kings. Kings determine
orientation, but are not pair partners. Both perspectives receive all pairs.

The 16,384 unfactorized rows encode pawn color (2), advanced pawn square (16),
partner color/type (8), and partner square (64). Orientation follows SFNNv19:
reflect files when the perspective king is on files a-d, and flip ranks/colors
for Black. There are at most 48 active rows per perspective (16 pawns times
three non-king partners). At width 1,024, the exported int8 table occupies 16 MiB.
It is serialized between `PP_3Wide` and `K32H3^`; the feature hash is `0x8D36B4E1`.
The material condition limits active work, not parameter storage.

Validation: `pytest -q tests/test_advanced_pawn_pairs.py
 tests/test_serialization_roundtrip.py` checks independent index enumeration,
color symmetry, the material cutoff, gradient flow, and serialization. The
engine's `tests/advanced_pawn_pairs.sh`, run from `src/` after an AVX2 build,
checks nonzero incremental accumulators against fresh full accumulators.
