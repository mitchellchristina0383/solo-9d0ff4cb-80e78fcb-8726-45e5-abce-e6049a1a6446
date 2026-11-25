"""Generate reproducible synthetic calibration data with known ground truth."""

from __future__ import annotations

import argparse
import random
import sys


def generate(path: str, rows: int, *, w: float = 1.7, b: float = -0.35,
             noise: float = 0.25, seed: int = 7, header: bool = True,
             extra_blank: bool = False) -> None:
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if header:
            fh.write("measured,reference\n")
        for i in range(rows):
            x = rng.uniform(-10.0, 10.0)
            y = w * x + b + rng.gauss(0.0, noise)
            fh.write(f"{x:.6f},{y:.6f}\n")
            if extra_blank and i % 97 == 96:
                fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("-n", "--rows", type=int, default=20_000)
    p.add_argument("--w", type=float, default=1.7)
    p.add_argument("--b", type=float, default=-0.35)
    p.add_argument("--noise", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-header", action="store_true")
    p.add_argument("--blank-lines", action="store_true")
    args = p.parse_args(argv)
    if args.rows < 1:
        print("rows must be >= 1", file=sys.stderr)
        return 2
    generate(args.output, args.rows, w=args.w, b=args.b, noise=args.noise,
             seed=args.seed, header=not args.no_header,
             extra_blank=args.blank_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
