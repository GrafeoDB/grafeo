"""Shared fixtures for the dataset checks. They run against the grafeo binding built from this repository."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

# A missing binding fails the run loudly and never skips it: CI must not pass without the engine.
try:
    import grafeo  # noqa: F401
except ImportError as err:
    raise pytest.UsageError(
        "datasets/tests need the grafeo Python binding built from this repository "
        f"(maturin develop -m crates/bindings/python/Cargo.toml); import failed: {err}"
    ) from err

TESTS = Path(__file__).resolve().parent
DATASETS = TESTS.parent
BUILD = DATASETS / "tools" / "build.mjs"


@pytest.fixture
def db():
    """A fresh in-memory database."""
    return grafeo.GrafeoDB()


@pytest.fixture(scope="session")
def derived():
    """The folder holding the generated Silicon Valley files."""
    return DATASETS / "silicon-valley" / "derived"


@pytest.fixture(scope="session")
def fixtures():
    """The folder holding the test fixtures."""
    return TESTS / "fixtures"


def load_lines(db, path, method="execute"):
    """Run each non-empty line of `path` as one statement through `db.<method>`.

    Raises AssertionError naming the file, line number and statement start when the engine refuses one.
    """
    run = getattr(db, method)
    count = 0
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    # One statement per line, split on LF only: a description may hold other line breaks in escaped form.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            run(line)
        except Exception as err:
            raise AssertionError(
                f"{Path(path).name}:{number}: {err}\n  {line[:160]}"
            ) from err
        count += 1
    return count


def scalar(db, query):
    """The first column of the single row `query` returns."""
    rows = list(db.execute(query))
    assert len(rows) == 1, f"expected one row from {query!r}, got {len(rows)}"
    return next(iter(rows[0].values()))


@pytest.fixture(scope="session")
def tricky(tmp_path_factory, fixtures):
    """The tricky source built to a temp folder: (source dict, output folder). No vectors are written."""
    source_path = fixtures / "tricky-source.json"
    out = tmp_path_factory.mktemp("tricky")
    if shutil.which("node") is None:
        pytest.fail("node is needed to build the tricky fixture")
    command = ["node", str(BUILD), "--source", str(source_path), "--out", str(out)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        pytest.fail(f"{' '.join(command)} failed: {err.stderr}")
    return json.loads(source_path.read_text(encoding="utf-8")), out
