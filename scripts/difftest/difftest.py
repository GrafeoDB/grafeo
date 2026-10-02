"""Differential test: run one query corpus on two builds of Grafeo and compare them.

Every difference between two builds is an intended change or a regression. Before a
release, run the corpus on the previous release (from PyPI) and on the release branch,
and review every difference. The reviewed differences of a release are listed in
`scripts/difftest/reviewed/<version>.txt`, one per line: the case key, the fingerprint of
the reviewed result and why it changed. The gate then fails on a difference nobody
reviewed, on a reviewed result that changed since, and on a reviewed difference that is
gone (a fix that was undone). The corpus and its fixtures are in `corpus.py`.

Run from the repository root (needs uv; `candidate` also needs maturin and Rust):

    # The grafeo this interpreter imports, for example a `maturin develop` build
    python scripts/difftest/difftest.py run target/difftest/dev.json

    # A published release, installed from PyPI into its own environment
    python scripts/difftest/difftest.py baseline 0.5.43 target/difftest/0.5.43.json

    # A commit, built as a release wheel in a clean worktree
    python scripts/difftest/difftest.py candidate HEAD target/difftest/head.json

    # The cases whose results differ (with --reviewed: fail on anything not reviewed)
    python scripts/difftest/difftest.py compare OLD.json NEW.json --reviewed FILE

    # The cases whose GQL and Cypher results differ within one run
    python scripts/difftest/difftest.py parity RESULTS.json

    # The release gate: baseline, candidate and compare
    python scripts/difftest/difftest.py gate 0.5.43 --reviewed scripts/difftest/reviewed/0.5.44.txt

Work files (environments, the worktree, wheels, results) go to `target/difftest/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "target" / "difftest"
# The Python wheels are abi3 wheels for Python 3.12 and later.
PYTHON = "3.12"
ERROR_LENGTH = 200

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Running the corpus
# ---------------------------------------------------------------------------


def normalize(value):
    """A JSON form of a returned value that compares equal across builds."""
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): normalize(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "NaN"}
        if math.isinf(value):
            return {"float": "Infinity" if value > 0 else "-Infinity"}
        return round(value, 9)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    for attribute in ("to_dict", "as_dict"):
        if hasattr(value, attribute):
            return {
                "type": type(value).__name__,
                **normalize(getattr(value, attribute)()),
            }
    return {"repr": repr(value)}


def error_text(error: Exception) -> str:
    lines = str(error).splitlines()
    return lines[0][:ERROR_LENGTH] if lines else type(error).__name__


def run_case(db, language: str, query: str) -> dict:
    """The result of one query: its columns and rows, or the first line of its error."""
    execute = {"gql": db.execute, "cypher": db.execute_cypher}[language]
    try:
        result = execute(query)
        records = [dict(row) for row in result]
    except Exception as error:  # noqa: BLE001 (a failing query is a result too)
        return {"error": error_text(error)}
    columns = list(getattr(result, "columns", None) or (records[0] if records else []))
    rows = [[normalize(value) for value in record.values()] for record in records]
    return {"columns": columns, "rows": rows}


def run_corpus(out: Path) -> None:
    # Imported here, so that comparing results works without grafeo installed.
    import grafeo
    from corpus import CASES, FIXTURES

    databases = {}
    results = {}
    for case in CASES:
        if case.fixture not in databases:
            databases[case.fixture] = FIXTURES[case.fixture](grafeo)
        for language in case.languages:
            result = run_case(databases[case.fixture], language, case.query)
            results[f"{case.id}|{language}"] = {
                "query": case.query,
                "ordered": case.ordered,
                **result,
            }
    build_info = getattr(grafeo, "build_info", None)
    meta = {
        "version": grafeo.__version__,
        "commit": build_info()["commit"] if build_info else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"meta": meta, "results": results}, indent=1, sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"ran {len(results)} cases on grafeo {describe(meta)}, results in {out}")


# ---------------------------------------------------------------------------
# Comparing results
# ---------------------------------------------------------------------------


def describe(meta: dict) -> str:
    commit = meta.get("commit")
    return (
        f"{meta.get('version')} ({commit[:8]})" if commit else f"{meta.get('version')}"
    )


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(result: dict):
    """What two results must share to count as equal: rows in order when the case is
    ordered, otherwise as a multiset; for a failed query, its error."""
    if "error" in result:
        return ["error", result["error"]]
    rows = result["rows"]
    if not result.get("ordered"):
        rows = sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
    return ["rows", result.get("columns"), rows]


def canonical_text(result: dict) -> str:
    """The canonical form as JSON text: unlike Python values, it tells 1, 1.0
    and true apart, so a value that changes type counts as a difference."""
    return json.dumps(canonical(result), sort_keys=True)


def fingerprint(result: dict) -> str:
    return hashlib.sha256(canonical_text(result).encode("utf-8")).hexdigest()[:10]


def case_order(key: str):
    case_id, _, language = key.partition("|")
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", case_id)
    if match:
        return (match.group(1), int(match.group(2)), language)
    return (case_id, 0, language)


def differing(old: dict, new: dict) -> list[str]:
    """The case keys of both runs whose results differ, in corpus order."""
    common = old.keys() & new.keys()
    keys = [
        key for key in common if canonical_text(old[key]) != canonical_text(new[key])
    ]
    return sorted(keys, key=case_order)


def read_reviewed(path: Path) -> dict[str, tuple[str, str]]:
    """The reviewed differences: case key -> (fingerprint, reason)."""
    reviewed = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3 or "|" not in parts[0]:
            raise SystemExit(
                f"{path}:{number}: expected 'ID|language fingerprint reason'"
            )
        key, digest, reason = parts
        if key in reviewed:
            raise SystemExit(f"{path}:{number}: {key} is listed twice")
        reviewed[key] = (digest, reason)
    return reviewed


def summary(result: dict, limit: int = 6) -> str:
    if "error" in result:
        return "ERROR " + result["error"]
    rows = result["rows"]
    shown = json.dumps(rows[:limit], sort_keys=True)
    more = f" ... ({len(rows)} rows)" if len(rows) > limit else f" ({len(rows)} rows)"
    return f"columns {result.get('columns')} {shown}{more}"


def compare(old_path: Path, new_path: Path, reviewed_path: Path | None = None) -> int:
    old, new = load(old_path), load(new_path)
    old_results, new_results = old["results"], new["results"]
    print(f"old: grafeo {describe(old['meta'])}, new: grafeo {describe(new['meta'])}")
    reviewed = read_reviewed(reviewed_path) if reviewed_path else {}
    keys = differing(old_results, new_results)
    failures = 0
    for key in keys:
        digest = fingerprint(new_results[key])
        if not reviewed_path:
            state = "changed"
        elif key not in reviewed:
            state = "NOT REVIEWED"
        elif reviewed[key][0] != digest:
            state = "CHANGED SINCE THE REVIEW"
        else:
            state = "reviewed"
        failures += state not in ("changed", "reviewed")
        print(f"=== {key} [{state}]: {new_results[key]['query']}")
        print("  old:", summary(old_results[key]))
        print("  new:", summary(new_results[key]))
        if key in reviewed:
            print("  why:", reviewed[key][1])
        elif reviewed_path:
            print(f"  to review: {key} {digest} <why it changed>")
    only = sorted(old_results.keys() ^ new_results.keys(), key=case_order)
    if only:
        print(f"{len(only)} cases ran in one build only: {', '.join(only)}")
    print(
        f"{len(keys)} differences in {len(old_results.keys() & new_results.keys())} cases"
    )
    if not reviewed_path:
        return 0
    gone = sorted(set(reviewed) - set(keys), key=case_order)
    for key in gone:
        print(f"=== {key} [NO LONGER DIFFERENT]: reviewed as '{reviewed[key][1]}'")
    failures += len(gone)
    print(f"{failures} of them need attention" if failures else "all reviewed")
    return 1 if failures else 0


def parity(path: Path) -> int:
    """The cases whose GQL and Cypher runs disagree: one fails and the other does not,
    or both succeed with different rows. Both failing counts as agreeing (the error
    texts differ by language)."""
    results = load(path)["results"]
    mismatches = []
    for key in sorted(results, key=case_order):
        case_id, _, language = key.partition("|")
        other = f"{case_id}|cypher"
        if language != "gql" or other not in results:
            continue
        gql, cypher = results[key], results[other]
        if "error" in gql and "error" in cypher:
            continue
        # Rows only: the column names may differ by language.
        gql_rows = json.dumps(canonical(gql)[2:], sort_keys=True)
        cypher_rows = json.dumps(canonical(cypher)[2:], sort_keys=True)
        if gql_rows != cypher_rows:
            mismatches.append(case_id)
            print(f"=== {case_id}: {gql['query']}")
            print("  gql:   ", summary(gql))
            print("  cypher:", summary(cypher))
    print(f"{len(mismatches)} cases where GQL and Cypher disagree")
    return 1 if mismatches else 0


# ---------------------------------------------------------------------------
# Builds to compare
# ---------------------------------------------------------------------------


def tool(*command: str, cwd: Path = ROOT, env: dict | None = None) -> str:
    done = subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def venv_python(venv: Path) -> Path:
    windows = venv / "Scripts" / "python.exe"
    return windows if windows.exists() else venv / "bin" / "python"


def environment(name: str, requirement: str, reinstall: bool) -> Path:
    """A uv environment in target/difftest with `requirement` installed."""
    venv = WORK / f"venv-{name}"
    if not venv_python(venv).exists():
        tool("uv", "venv", str(venv), "--python", PYTHON, "--quiet")
    python = venv_python(venv)
    command = ["uv", "pip", "install", "--quiet", "--python", str(python), requirement]
    tool(*command, *(["--reinstall"] if reinstall else []))
    return python


def run_in(python: Path, out: Path) -> None:
    subprocess.run(
        [str(python), str(Path(__file__).resolve()), "run", str(out)], check=True
    )


def baseline(version: str, out: Path) -> None:
    run_in(environment(version, f"grafeo=={version}", reinstall=False), out)


def build_wheel(ref: str) -> Path:
    """A release wheel of `ref`, built in a clean worktree of this repository (the main
    tree may hold a module from `maturin develop` that the wheel build trips over)."""
    commit = tool("git", "rev-parse", "--verify", f"{ref}^{{commit}}")
    worktree = WORK / "worktree"
    if (worktree / ".git").exists():
        tool("git", "-C", str(worktree), "checkout", "--detach", "--force", commit)
    else:
        tool("git", "worktree", "add", "--detach", str(worktree), commit)
    wheels = WORK / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    maturin = shutil.which("maturin")
    command = [maturin] if maturin else ["uvx", "maturin"]
    env = {**os.environ, "CARGO_TARGET_DIR": str(WORK / "cargo")}
    print(f"building a release wheel of {ref} ({commit[:8]})")
    subprocess.run(
        [*command, "build", "--release", "--out", str(wheels)],
        cwd=worktree / "crates" / "bindings" / "python",
        env=env,
        check=True,
    )
    built = list(wheels.glob("*.whl"))
    if len(built) != 1:
        raise SystemExit(f"expected one wheel in {wheels}, found {len(built)}")
    return built[0]


def candidate(ref: str, out: Path) -> None:
    run_in(environment("candidate", str(build_wheel(ref)), reinstall=True), out)


def gate(version: str, ref: str, reviewed: Path | None) -> int:
    old = WORK / f"results-{version}.json"
    new = WORK / "results-candidate.json"
    baseline(version, old)
    candidate(ref, new)
    return compare(old, new, reviewed)


def main(argv: list[str] | None = None) -> int:
    # Line by line, so that this script's lines stay in order with its subprocesses'.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "run", help="run the corpus with the grafeo installed here"
    )
    command.add_argument("out", type=Path)
    command = commands.add_parser(
        "baseline", help="run the corpus on a release from PyPI"
    )
    command.add_argument("version")
    command.add_argument("out", type=Path)
    command = commands.add_parser(
        "candidate", help="run the corpus on a release build of a commit"
    )
    command.add_argument("ref")
    command.add_argument("out", type=Path)
    command = commands.add_parser("compare", help="list the cases whose results differ")
    command.add_argument("old", type=Path)
    command.add_argument("new", type=Path)
    command.add_argument("--reviewed", type=Path)
    command = commands.add_parser(
        "parity", help="list cases where GQL and Cypher disagree"
    )
    command.add_argument("results", type=Path)
    command = commands.add_parser(
        "gate", help="baseline, candidate and compare in one go"
    )
    command.add_argument(
        "version", help="the release to compare against, for example 0.5.43"
    )
    command.add_argument("--ref", default="HEAD")
    command.add_argument("--reviewed", type=Path)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_corpus(args.out)
    elif args.command == "baseline":
        baseline(args.version, args.out)
    elif args.command == "candidate":
        candidate(args.ref, args.out)
    elif args.command == "compare":
        return compare(args.old, args.new, args.reviewed)
    elif args.command == "parity":
        return parity(args.results)
    elif args.command == "gate":
        return gate(args.version, args.ref, args.reviewed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
