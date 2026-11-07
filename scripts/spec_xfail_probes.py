"""Checks how a spec runner handles expected failures, with the probes of
tests/spec-probes/xfail.

Runs the runner once per probe directory, with GRAFEO_SPEC_ROOT naming it, and
once on a copy of the runner_error probe placed outside spec-probes, and checks
how each run ends:

- assertion: passes, and its summary line says `skipped=0 xfailed=2`;
- xpass: fails, and the output says XPASS with the case's spec and issue;
- runner_error: fails with the runner's ordinary error, counting no xfail;
- parse_skip_and_xfail, parse_xfail_without_issue: fail, naming the case;
- runner_error outside spec-probes: fails, naming the file (the `probe` key is
  accepted only under a spec-probes directory).

Every runner prints one summary line, `gtest summary: skipped=S xfailed=X`.

The runner must be built first, as scripts/ci-spec.ps1 builds it (the native
library copied next to the go, csharp and dart runners). The Rust harness is
tested in process (crates/grafeo-spec-tests), so `rust` is not a runner here.

Usage: python scripts/spec_xfail_probes.py <python|node|go|csharp|dart>
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "tests" / "spec-probes" / "xfail"
SUMMARY = re.compile(r"gtest summary: skipped=(\d+) xfailed=(\d+)")

# How the ordinary, non-assertion error each runner raises for `probe: runner_error`
# shows in its output. Dart prints a StateError as "Bad state: <message>".
RUNNER_ERRORS = {
    "python": "RuntimeError",
    "node": "TypeError",
    "go": "panic",
    "csharp": "InvalidOperationException",
    "dart": "Bad state",
}

# The spec and issue of the xpass probe, which its XPASS report must name.
XPASS_SPEC = "harness probe: XPASS names this spec"
XPASS_ISSUE = "fix in 0.6.2"


@dataclass(frozen=True)
class Run:
    """A runner's command: its working directory, arguments and extra environment."""

    cwd: Path
    args: list[str]
    env: dict[str, str]


def command(runner: str) -> Run:
    """The command scripts/ci-spec.ps1 runs for `runner`, from its directory."""
    windows = sys.platform == "win32"
    runners = ROOT / "tests" / "spec" / "runners"
    if runner == "python":
        return Run(ROOT, [sys.executable, "-m", "pytest", "tests/spec/", "-v"], {})
    if runner == "node":
        vitest = "vitest.cmd" if windows else "vitest"
        binary = (
            ROOT / "crates" / "bindings" / "node" / "node_modules" / ".bin" / vitest
        )
        args = [str(binary), "run", "--dir", "tests/spec/runners/node"]
        return Run(ROOT, [*args, "spec-runner.test.mjs"], {})
    if runner == "go":
        here = runners / "go"
        env = {"CGO_ENABLED": "1"}
        if sys.platform.startswith("linux"):
            env |= {
                "CGO_LDFLAGS": "-L. -lgrafeo_c -lm -ldl -lpthread",
                "LD_LIBRARY_PATH": ".",
            }
        elif sys.platform == "darwin":
            flags = "-L. -lgrafeo_c -lm -ldl -lpthread -framework Security"
            env |= {"CGO_LDFLAGS": flags, "DYLD_LIBRARY_PATH": "."}
        else:
            flags = "-L. -lgrafeo_c -lws2_32 -lbcrypt -lntdll -luserenv"
            env |= {"CGO_LDFLAGS": flags}
        args = ["go", "test", "-count=1", "-run", "TestSpec", "-timeout", "120s", "-v"]
        return Run(here, args, env)
    if runner == "csharp":
        here = runners / "csharp"
        path = f"{here}{os.pathsep}{os.environ.get('PATH', '')}"
        # Detailed console output, so the summary line of a passing run shows.
        logger = "console;verbosity=detailed"
        args = ["dotnet", "test", "--verbosity", "minimal", "--logger", logger]
        return Run(here, args, {"LD_LIBRARY_PATH": str(here), "PATH": path})
    if runner == "dart":
        here = runners / "dart"
        library = ROOT / "crates" / "bindings" / "dart"
        args = ["dart", "test", "spec_runner_test.dart"]
        return Run(here, args, {"LD_LIBRARY_PATH": str(library)})
    raise SystemExit(f"unknown runner {runner!r}: one of {', '.join(RUNNER_ERRORS)}")


def summaries(output: str) -> list[tuple[int, int]]:
    """Every `(skipped, xfailed)` pair the output's summary lines report."""
    return [(int(s), int(x)) for s, x in SUMMARY.findall(output)]


def judge(probe: str, runner: str, code: int, output: str) -> list[str]:
    """What is wrong with how the run of `probe` ended; empty when it ended right."""
    problems = []
    counted = summaries(output)
    if probe == "assertion":
        if code != 0:
            problems.append(f"exit status {code}, expected 0")
        if counted != [(0, 2)]:
            problems.append(
                f"summary {counted}, expected one line: skipped=0 xfailed=2"
            )
        return problems
    if code == 0:
        problems.append("exit status 0, expected a failed run")
    if any(xfailed for _, xfailed in counted):
        problems.append(f"summary {counted} counts an xfail")
    expected = {
        "xpass": ["XPASS", XPASS_SPEC, XPASS_ISSUE],
        "runner_error": [RUNNER_ERRORS[runner]],
        "parse_skip_and_xfail": ["probe_skip_and_xfail"],
        "parse_xfail_without_issue": ["probe_xfail_without_issue"],
        # A case the runner cannot run (no dataset, no query) fails as a fault, xfail or not.
        "runner_fault": ["probe_fault_missing_dataset", "probe_fault_no_query"],
        # The parser's own refusal of the key, which names `probe:` and `spec-probes`, not a
        # runner that ran the probe and failed with its ordinary error.
        "runner_error_outside": ["runner_error.gtest", "`probe:`", "`spec-probes`"],
    }[probe]
    problems += [
        f"the output does not say {text!r}" for text in expected if text not in output
    ]
    if probe == "runner_error_outside" and "XPASS" in output:
        problems.append(
            "the case ran: the probe key must be a parse error outside spec-probes"
        )
    return problems


def run(runner: str, spec_root: Path) -> tuple[int, str]:
    """Runs `runner` on the .gtest files under `spec_root`: its exit status and output."""
    how = command(runner)
    env = os.environ | how.env | {"GRAFEO_SPEC_ROOT": str(spec_root)}
    result = subprocess.run(
        how.args,
        cwd=how.cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode, result.stdout + result.stderr


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    runner = argv[0]
    command(runner)
    probes = sorted(path.name for path in PROBES.iterdir() if path.is_dir())
    expected = sorted(
        [
            "assertion",
            "parse_skip_and_xfail",
            "parse_xfail_without_issue",
            "runner_error",
            "runner_fault",
            "xpass",
        ]
    )
    if probes != expected:
        print(f"probe directories {probes}, expected {expected}", file=sys.stderr)
        return 1
    failed = 0
    with tempfile.TemporaryDirectory() as scratch:
        outside = Path(scratch) / "spec-copy"
        outside.mkdir()
        shutil.copy(PROBES / "runner_error" / "runner_error.gtest", outside)
        cases = [(name, PROBES / name) for name in probes]
        cases.append(("runner_error_outside", outside))
        for probe, spec_root in cases:
            code, output = run(runner, spec_root)
            problems = judge(probe, runner, code, output)
            print(f"{runner} {probe}: {'ok' if not problems else 'FAILED'}")
            for problem in problems:
                print(f"  {problem}")
            if problems:
                failed += 1
                print("  output, last 40 lines:")
                for line in output.splitlines()[-40:]:
                    print(f"    {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
