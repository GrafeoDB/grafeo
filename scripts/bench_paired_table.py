#!/usr/bin/env python3
"""Tables of a paired benchmark run (see bench-paired.sh).

Reads the Criterion output of each round, `<runs>/<bench>-<side>-<round>.txt`,
and prints per benchmark case each side's fastest round, the change of the
second side against the first, and whether the two sides' rounds overlap.
The fastest round is the one least disturbed by the rest of the machine;
when the sides' rounds overlap, the change is within the noise.

Usage:
    python scripts/bench_paired_table.py <runs> <base> <head> <bench>...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

UNITS = {"ps": 1e-3, "ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}
CASE = re.compile(
    r"^(\S[^\n]*?)\s*\n?\s*time:\s+\[[\d.]+ \S+ ([\d.]+) (\S+)", re.MULTILINE
)


def medians(path: Path) -> dict[str, float]:
    """The middle estimate of each case in one Criterion output, in ns."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        match.group(1).strip(): float(match.group(2)) * UNITS.get(match.group(3), 1e3)
        for match in CASE.finditer(text)
    }


def show(nanoseconds: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if nanoseconds >= scale:
            return f"{nanoseconds / scale:.2f} {unit}"
    return f"{nanoseconds:.1f} ns"


def main() -> None:
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    runs, base, head, benches = (
        Path(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4:],
    )
    for bench in benches:
        rounds: dict[str, dict[str, list[float]]] = {}
        for side in (base, head):
            for path in sorted(runs.glob(f"{bench}-{side}-*.txt")):
                for case, value in medians(path).items():
                    rounds.setdefault(case, {}).setdefault(side, []).append(value)
        print(f"\n### {bench}\n")
        print(f"| case | {base} | {head} | change | rounds overlap |")
        print("| --- | --- | --- | --- | --- |")
        for case in sorted(rounds):
            sides = rounds[case]
            if base not in sides or head not in sides:
                continue
            first, second = sides[base], sides[head]
            change = (min(second) - min(first)) / min(first) * 100
            overlap = not (max(second) < min(first) or min(second) > max(first))
            print(
                f"| {case} | {show(min(first))} | {show(min(second))} | {change:+.1f}% "
                f"| {'yes (noise)' if overlap else 'no'} |"
            )


if __name__ == "__main__":
    main()
