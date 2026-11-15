"""Tests for scripts/downgrade_check.py.

The checker's own tests run without a grafeo release: a stand-in open shows that it fails
when an open is not refused, crashes, is refused with other words, or changes a byte. The
release test runs the checker with 0.6.2's wheel from PyPI on 0.6.3's fixtures, through uv;
it skips without uv, without the fixtures, or when uv cannot fetch the wheel.

Run: uv run --with pytest python -m pytest scripts/tests/test_downgrade_check.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "downgrade_check.py"

spec = importlib.util.spec_from_file_location("downgrade_check", SCRIPT)
downgrade_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downgrade_check)


@pytest.fixture
def fixture(tmp_path):
    """A stand-in fixture: a database file and a sidecar WAL with one segment."""
    path = tmp_path / "fixture" / "standalone-wal.grafeo"
    path.parent.mkdir()
    path.write_bytes(b"GRAFEO Alix")
    wal = downgrade_check.sidecar(path)
    wal.mkdir()
    (wal / "segment-0").write_bytes(b"Gus")
    return path


def opener(returncode, stdout="", change=None):
    """A stand-in open that ends with `returncode` and `stdout`, after `change(path)`."""

    def run(path):
        if change is not None:
            change(path)
        return subprocess.CompletedProcess([], returncode, stdout, "")

    return run


def test_a_clean_refusal_passes(fixture):
    refused = opener(downgrade_check.REFUSED, "kind 12 (a newer release wrote it)")
    assert downgrade_check.check(fixture, "a newer release wrote it", refused) == []


@pytest.mark.parametrize(
    ("ended", "wrong"),
    [
        (opener(downgrade_check.OPENED), "opened, not refused"),
        (opener(-11), "not a refusal"),
        (opener(downgrade_check.REFUSED, "a file error"), "without"),
        (
            opener(
                downgrade_check.REFUSED,
                "a newer release wrote it",
                lambda path: path.write_bytes(b"GRAFEO Vincent"),
            ),
            "changed standalone-wal.grafeo",
        ),
        (
            opener(
                downgrade_check.REFUSED,
                "a newer release wrote it",
                lambda path: (downgrade_check.sidecar(path) / "segment-0").unlink(),
            ),
            "changed standalone-wal.grafeo.wal/segment-0",
        ),
        (
            opener(
                downgrade_check.REFUSED,
                "a newer release wrote it",
                lambda path: (downgrade_check.sidecar(path) / "segment-1").write_bytes(
                    b""
                ),
            ),
            "changed standalone-wal.grafeo.wal/segment-1",
        ),
    ],
)
def test_the_check_fails_on_anything_but_a_clean_refusal(fixture, ended, wrong):
    problems = downgrade_check.check(fixture, "a newer release wrote it", ended)
    assert any(wrong in problem for problem in problems), problems


def test_the_check_leaves_the_fixture_alone(fixture):
    before = downgrade_check.contents(fixture)
    downgrade_check.check(
        fixture,
        "a newer release wrote it",
        opener(downgrade_check.REFUSED, "", lambda path: path.write_bytes(b"")),
    )
    assert downgrade_check.contents(fixture) == before


def test_0_6_2_refuses_the_files_0_6_3_writes_cleanly():
    """0.6.2 refuses each 0.6.3 fixture (a standalone type a graph type lists, in a catalog
    section and only in the WAL) with an error, and leaves file and WAL byte-equal (#764)."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    if not list((ROOT / downgrade_check.FIXTURES / "0.6.3").glob("*.grafeo")):
        pytest.skip("no 0.6.3 downgrade fixtures")
    command = [
        uv,
        "run",
        "--no-project",
        "--python",
        "3.13",
        "--with",
        "grafeo==0.6.2",
        "python",
        str(SCRIPT),
        "0.6.3",
        "--release",
        "0.6.2",
    ]
    ran = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False
    )
    unfetched = ("No solution found", "Failed to fetch", "Failed to download")
    if ran.returncode != 0 and any(text in ran.stderr for text in unfetched):
        pytest.skip(f"uv could not fetch the 0.6.2 wheel: {ran.stderr.strip()[-300:]}")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "refuses" in ran.stdout
