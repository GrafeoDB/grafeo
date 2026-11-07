"""Tests for scripts/spec_xfail_probes.py: how it judges a runner's output, and the
probe fixtures it runs (no runner needed).

Run: uv run --with pytest python -m pytest scripts/tests
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "spec_xfail_probes.py"
spec = importlib.util.spec_from_file_location("spec_xfail_probes", SCRIPT)
probes = importlib.util.module_from_spec(spec)
sys.modules["spec_xfail_probes"] = probes
spec.loader.exec_module(probes)

RUNNERS = sorted(probes.RUNNER_ERRORS)


def test_the_summary_line_is_read_wherever_it_stands():
    output = "PASSED x\ngtest summary: skipped=0 xfailed=2\n== 3 passed ==\n"
    assert probes.summaries(output) == [(0, 2)]
    assert probes.summaries("no line here") == []


@pytest.mark.parametrize("runner", RUNNERS)
def test_the_assertion_probe_passes_with_two_xfails(runner):
    good = "gtest summary: skipped=0 xfailed=2\n"
    assert probes.judge("assertion", runner, 0, good) == []
    assert probes.judge("assertion", runner, 1, good) != []
    assert (
        probes.judge("assertion", runner, 0, "gtest summary: skipped=0 xfailed=1") != []
    )
    assert probes.judge("assertion", runner, 0, "all passed") != []
    twice = good + good
    assert probes.judge("assertion", runner, 0, twice) != [], "one summary line"


@pytest.mark.parametrize("runner", RUNNERS)
def test_an_xpass_fails_the_run_and_names_its_spec_and_issue(runner):
    said = f"XPASS probe_xpass: {probes.XPASS_SPEC} ({probes.XPASS_ISSUE})"
    assert probes.judge("xpass", runner, 1, said) == []
    assert probes.judge("xpass", runner, 0, said) != []
    assert probes.judge("xpass", runner, 1, "XPASS probe_xpass") != []
    counted = said + "\ngtest summary: skipped=0 xfailed=1"
    assert probes.judge("xpass", runner, 1, counted) != []


@pytest.mark.parametrize("runner", RUNNERS)
def test_a_runner_error_fails_the_run_with_the_runners_own_error(runner):
    error = probes.RUNNER_ERRORS[runner]
    assert probes.judge("runner_error", runner, 1, f"{error}: probe") == []
    assert probes.judge("runner_error", runner, 0, f"{error}: probe") != []
    absorbed = f"{error}: probe\ngtest summary: skipped=0 xfailed=1"
    assert probes.judge("runner_error", runner, 1, absorbed) != []
    others = [e for r, e in probes.RUNNER_ERRORS.items() if r != runner and e != error]
    assert probes.judge("runner_error", runner, 1, " ".join(others)) != []


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize(
    ("probe", "named"),
    [
        ("parse_skip_and_xfail", "probe_skip_and_xfail"),
        ("parse_xfail_without_issue", "probe_xfail_without_issue"),
        ("runner_error_outside", "runner_error.gtest"),
    ],
)
def test_a_parse_refusal_fails_the_run_and_names_the_case_or_file(runner, probe, named):
    assert probes.judge(probe, runner, 1, f"parse error: {named}") == []
    assert probes.judge(probe, runner, 0, f"parse error: {named}") != []
    assert probes.judge(probe, runner, 1, "parse error") != []


def test_a_probe_key_outside_spec_probes_must_not_run_the_case():
    output = "runner_error.gtest: XPASS"
    assert probes.judge("runner_error_outside", "python", 1, output) != []


def test_every_runner_has_a_command_and_an_unknown_one_is_refused():
    for runner in RUNNERS:
        assert probes.command(runner).args
    with pytest.raises(SystemExit):
        probes.command("rust")


def gtest_cases(text: str) -> dict[str, str]:
    """The case blocks of a .gtest text, by name."""
    blocks = text.split("\n  - name: ")[1:]
    return {block.split("\n", 1)[0]: block for block in blocks}


def test_the_fixtures_hold_the_cases_the_judge_expects():
    root = probes.PROBES
    found = {path.parent.name: path for path in root.glob("*/*.gtest")}
    assert sorted(found) == sorted(
        [
            "assertion",
            "parse_skip_and_xfail",
            "parse_xfail_without_issue",
            "runner_error",
            "xpass",
        ]
    )
    texts = {name: path.read_bytes().decode() for name, path in found.items()}
    assert all("\r" not in text for text in texts.values())
    assert sorted(gtest_cases(texts["assertion"])) == [
        "probe_xfail_on_a_row_mismatch",
        "probe_xfail_on_a_statement_error",
    ]
    assert all("xfail:" in case for case in gtest_cases(texts["assertion"]).values())
    xpass = gtest_cases(texts["xpass"])["probe_xpass"]
    assert probes.XPASS_SPEC in xpass and probes.XPASS_ISSUE in xpass
    runner_error = gtest_cases(texts["runner_error"])["probe_runner_error"]
    assert "probe: runner_error" in runner_error and "xfail:" in runner_error
    both = gtest_cases(texts["parse_skip_and_xfail"])["probe_skip_and_xfail"]
    assert "skip:" in both and "xfail:" in both
    no_issue = gtest_cases(texts["parse_xfail_without_issue"])[
        "probe_xfail_without_issue"
    ]
    assert "spec:" in no_issue and "issue:" not in no_issue
    # Only the runner_error probe uses the probe-only key.
    keyed = [name for name, text in texts.items() if "\n    probe: " in text]
    assert keyed == ["runner_error"]
