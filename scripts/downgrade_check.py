"""Check that a released Grafeo refuses, cleanly, the files a newer release writes.

A release that writes something an older one cannot read (a new catalog record kind, a new
section) needs the older release to refuse such a file without changing it: an error that
names what it cannot read, never a crash, and never an open that reads less than the file
holds. This script runs with the older release's wheel and opens a copy of each fixture of
`crates/grafeo-engine/tests/fixtures/downgrade/<newer>/` (a `.grafeo` file and, when there is
one, its sidecar WAL directory `<name>.wal/`), each in a process of its own, so a crash is
seen as one. It checks that each open is refused with a message holding `--refusal`, and that
the copy, file and WAL, is byte for byte what it was. Run it from the repository root:

    uv run --no-project --python 3.13 --with grafeo==0.6.2 \\
        python scripts/downgrade_check.py 0.6.3 --release 0.6.2

`--refusal` defaults to the text 0.6.2 gives for a catalog record of a kind it does not know.
The exit code is 0 when every fixture is refused cleanly, 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURES = Path("crates/grafeo-engine/tests/fixtures/downgrade")

#: What 0.6.2 says of a catalog record of a kind a newer release wrote.
REFUSAL = "a newer release wrote it"

#: The exit codes of the child that opens one copy.
REFUSED = 3
OPENED = 4


def fixtures(newer: str) -> list[Path]:
    """The database files of the fixtures of release `newer`, sorted."""
    return sorted((FIXTURES / newer).glob("*.grafeo"))


def sidecar(path: Path) -> Path:
    """The sidecar WAL directory of the database file `path`."""
    return path.with_name(path.name + ".wal")


def contents(path: Path) -> dict[str, bytes]:
    """Every byte of the database at `path`: the file and each file of its sidecar WAL, by
    their path relative to the file's directory."""
    found = {path.name: path.read_bytes()}
    wal = sidecar(path)
    if wal.is_dir():
        for file in sorted(wal.rglob("*")):
            if file.is_file():
                found[file.relative_to(path.parent).as_posix()] = file.read_bytes()
    return found


def copy_database(path: Path, into: Path) -> Path:
    """A copy of the database at `path` (its file and its sidecar WAL) in `into`."""
    copy = into / path.name
    shutil.copy2(path, copy)
    if sidecar(path).is_dir():
        shutil.copytree(sidecar(path), sidecar(copy))
    return copy


def open_in_child(path: Path) -> subprocess.CompletedProcess[str]:
    """Opens the database at `path` with this interpreter's grafeo, in a process of its own."""
    return subprocess.run(
        [sys.executable, __file__, "--open", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def child(path: str) -> int:
    """The child's open: `REFUSED` with the error's text on stdout, `OPENED` otherwise."""
    import grafeo

    try:
        db = grafeo.GrafeoDB(path)
    except Exception as error:  # noqa: BLE001 (any refusal is the outcome being checked)
        print(error)
        return REFUSED
    db.close()
    return OPENED


def check(fixture: Path, refusal: str, opener=open_in_child) -> list[str]:
    """What is wrong with `fixture`'s open by the older release: nothing when it is refused
    with `refusal` in its message and the copy it opened is unchanged."""
    problems = []
    with tempfile.TemporaryDirectory() as scratch:
        copy = copy_database(fixture, Path(scratch))
        before = contents(copy)
        opened = opener(copy)
        if opened.returncode == OPENED:
            problems.append(f"{fixture.name}: opened, not refused")
        elif opened.returncode != REFUSED:
            problems.append(
                f"{fixture.name}: the open ended with code {opened.returncode}, not a refusal:"
                f" {opened.stderr.strip()[-2000:]}"
            )
        elif refusal not in opened.stdout:
            problems.append(
                f"{fixture.name}: refused with {opened.stdout.strip()!r}, without {refusal!r}"
            )
        after = contents(copy)
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                problems.append(f"{fixture.name}: the open changed {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("newer", nargs="?", help="the release whose fixtures to open")
    parser.add_argument(
        "--release", help="the release this interpreter's grafeo must be"
    )
    parser.add_argument("--refusal", default=REFUSAL, help="text the refusal must hold")
    parser.add_argument("--open", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.open:
        return child(arguments.open)
    if not arguments.newer:
        parser.error("name the release whose fixtures to open")
    import grafeo

    if arguments.release and grafeo.__version__ != arguments.release:
        print(
            f"grafeo {grafeo.__version__} is installed, not {arguments.release}",
            file=sys.stderr,
        )
        return 1
    found = fixtures(arguments.newer)
    if not found:
        print(f"no fixtures in {FIXTURES / arguments.newer}", file=sys.stderr)
        return 1
    problems = [
        problem for fixture in found for problem in check(fixture, arguments.refusal)
    ]
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        names = ", ".join(fixture.name for fixture in found)
        print(f"grafeo {grafeo.__version__} refuses {names} cleanly")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
