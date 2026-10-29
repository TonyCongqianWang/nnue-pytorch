"""Run synthetic export and engine integration checks in one environment, without training."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_src", type=Path)
    parser.add_argument("build_log", type=Path)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="handcrafted-inference-") as directory:
        root = Path(directory)
        subprocess.run(
            [sys.executable, str(scripts / "prepare_handcrafted_nets.py"), str(root)],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(scripts / "build_handcrafted_probe.py"),
                str(args.engine_src),
                str(args.build_log),
                str(root / "probe"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(scripts / "validate_handcrafted_engine.py"),
                str(root / "probe"),
                str(root / "random.nnue"),
                "--masked-net",
                str(root / "masked.nnue"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
