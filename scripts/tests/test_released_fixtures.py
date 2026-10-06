"""Tests for scripts/released_fixtures.py: its arguments (no grafeo release needed).

Run: uv run --with pytest python -m pytest scripts/tests
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "released_fixtures.py"
VERSION = "0.0.3"


@pytest.fixture
def script(monkeypatch, tmp_path):
    """The script, loaded with a stand-in `grafeo` module, run from `tmp_path`.

    Opening a database fails the test: these tests only check which mode the
    arguments select.
    """
    grafeo = types.ModuleType("grafeo")
    grafeo.__version__ = VERSION

    def no_database(*_args, **_kwargs):
        raise AssertionError("the script opened a database")

    grafeo.GrafeoDB = no_database
    monkeypatch.setitem(sys.modules, "grafeo", grafeo)
    spec = importlib.util.spec_from_file_location("released_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    return module


@pytest.fixture
def committed(tmp_path):
    """A committed fixture of the version, which a regeneration would delete."""
    fixture = (
        tmp_path
        / "crates/grafeo-engine/tests/fixtures/released"
        / VERSION
        / "closed.grafeo"
    )
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"Alix")
    return fixture


@pytest.mark.parametrize(
    "arguments",
    [
        ["--unflushed"],
        ["--unflushed", "unflushed.grafeo", "extra"],
        ["--flushed", "unflushed.grafeo"],
        ["unflushed.grafeo"],
    ],
)
def test_malformed_arguments_exit_with_usage_and_touch_nothing(
    script, committed, monkeypatch, capsys, arguments
):
    monkeypatch.setattr(sys, "argv", ["released_fixtures.py", *arguments])
    with pytest.raises(SystemExit) as exited:
        script.main()
    assert exited.value.code == 2
    assert "usage" in capsys.readouterr().err
    assert committed.read_bytes() == b"Alix", "the committed fixture is kept"


def test_unflushed_with_a_path_writes_only_that_database(
    script, committed, monkeypatch
):
    written = []
    monkeypatch.setattr(
        script,
        "write",
        lambda path, *, close_at_end: written.append((path, close_at_end)),
    )
    monkeypatch.setattr(
        sys, "argv", ["released_fixtures.py", "--unflushed", "unflushed.grafeo"]
    )
    script.main()
    assert written == [(Path("unflushed.grafeo"), False)]
    assert committed.read_bytes() == b"Alix", "the committed fixture is kept"
