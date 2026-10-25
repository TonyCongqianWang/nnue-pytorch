# Network training

For reproducible training and engine testing, follow the upstream guides:

- [Run a training workflow locally](https://github.com/vondele/nettest#local-execution).
- [Configure training stages and restarts](https://github.com/vondele/nettest#recipe-description).
- [Select trainer revisions, engines, and datasets](https://github.com/vondele/nettest#external-tools-and-data).
- [Run a workflow in CI](https://github.com/vondele/nettest#execution-in-the-ci-environment).

## Hardware and datasets

GPU training requires at least 16 GB of system RAM and 8 GB of GPU VRAM.
8 GB of VRAM is tight; more gives room for larger batches and networks.

The complete dataset set in the
[threats training recipe](https://github.com/vondele/nettest/blob/master/threats.yaml)
requires 800+ GB of storage, plus space for checkpoints, caches, and outputs.
Training can also use a single small binpack, which requires much less storage
but produces weaker networks. Use fast storage for data loading and allow
additional CPU and memory resources for engine testing.

## Experimental piece-count-dependent loss exponent

Pass `--pow-exp-endgame-reduction 0.1` to reduce the loss exponent by a small
amount in endgames. The default is zero, preserving the existing objective.
Keep the tuned `--pow-exp` value as the base exponent. With a base of 2.43404,
the experimental exponent is:

| Total pieces, including kings | Exponent |
|---|---:|
| 24 or more | 2.43404 |
| 20 | 2.41842 |
| 16 | 2.38404 |
| 12 | 2.34967 |
| 8 or fewer | 2.33404 |

The transition is `u = clamp((24 - pieces) / 16, 0, 1)`, followed by
`exponent = base - reduction * u*u*(3 - 2*u)`. Piece count is already supplied
by the data loader; no extra position decoding or network outputs are needed.

To avoid increasing the overall emphasis on endgames just by lowering the
power, the per-position loss is normalized as
`(base / exponent) * reference_error**(base - exponent) * abs(error)**exponent`.
This preserves the original gradient magnitude at the reference probability
residual, set by `--pow-exp-reference-error` (default 0.1), before the existing
asymmetry and sample weights. With a reduction of 0.1, a residual of 0.01 gets
about 26% more gradient in the lowest-piece positions, while a residual of 0.5
gets about 15% less. These ratios refer to probability residual gradients at
the same prediction; score conversion, asymmetry, and sample weighting still apply.

This changes the relative emphasis on small and large errors. It does not
establish that endgame draw prediction or playing strength improves. Validation
losses from different exponent settings are different objectives and should
not be compared directly as measures of model quality. Check outcome errors
and teacher agreement separately for drawn and decisive endgames.

Existing workflow recipes may pin an older trainer. Select a trainer revision
containing this option before adding it to a recipe. No engine or network-file
format changes are needed.

## Local trainer development

Use `run_docker.sh` to develop and test this checkout. See
[Developing and testing with Docker](development.md) for checks, the training
pipeline test, evaluation cross-checks, and logging.
