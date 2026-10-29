"""Build a probe using the exact flags/objects recorded by a normal engine make build."""

import argparse
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_src", type=Path)
    parser.add_argument("build_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    lines = args.build_log.read_text().splitlines()
    flags = shlex.split(next(l for l in lines if l.startswith("g++ ") and " -c " in l))[
        1:
    ]
    flags = flags[: flags.index("-c")]
    source = (
        args.source.resolve()
        if args.source
        else Path(__file__).with_name("handcrafted_engine_probe.cpp").resolve()
    )
    obj = args.output.with_suffix(".o")
    subprocess.run(
        ["g++", *flags, "-I.", "-c", str(source), "-o", str(obj)],
        cwd=args.engine_src,
        check=True,
    )
    link = shlex.split(next(l for l in reversed(lines) if l.startswith("g++ -o ")))
    link[2] = str(args.output)
    link[link.index("main.o")] = str(obj)
    subprocess.run(link, cwd=args.engine_src, check=True)


if __name__ == "__main__":
    main()
