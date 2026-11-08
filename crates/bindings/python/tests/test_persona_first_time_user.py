"""Persona: a first-time user copies the README's examples verbatim.

Every self-contained Python example of the project README and the Python
README (one that imports grafeo itself) runs as written, in a fresh process
and an empty directory, and prints what its comments promise. The API
reference blocks assume a database from the quick start; the lines a reader
copies from them (the transaction and schema examples) run after it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import grafeo  # noqa: F401

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

ROOT = Path(__file__).resolve().parents[4]
READMES = {
    "project": ROOT / "README.md",
    "python": ROOT / "crates" / "bindings" / "python" / "README.md",
}


def python_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def self_contained():
    found = []
    for name, path in READMES.items():
        for index, block in enumerate(python_blocks(path)):
            code = [
                line for line in block.splitlines() if line.strip() and not line.startswith("#")
            ]
            if code and re.match(r"(import grafeo|from grafeo import)", code[0]):
                marks = (
                    [pytest.mark.requires(*sorted(needs))] if (needs := features_of(block)) else []
                )
                found.append(pytest.param(block, id=f"{name}-{index}", marks=marks))
    return found


# What a block calls that a build may leave out, and the feature it needs.
FEATURE_CALLS = {
    "create_vector_index": "vector-index",
    "vector_search": "vector-index",
    "mmr_search": "vector-index",
    "create_text_index": "text-index",
    "text_search": "text-index",
    "hybrid_search": "hybrid-search",
    ".algorithms": "algos",
    "begin_transaction_with_cdc": "cdc",
    "execute_sparql": "sparql",
    "CREATE VECTOR INDEX": "vector-index",
    "CREATE TEXT INDEX": "text-index",
}

# The procedures a `CALL grafeo.<name>` reaches that a build may leave out: the search ones need
# their index, and every name not listed here or in NO_FEATURE is a graph algorithm.
PROCEDURE_FEATURES = {
    "search.vector": "vector-index",
    "search.mmr": "vector-index",
    "search.text": "text-index",
}
# The procedures every build registers: the catalog, the listing and the CSV loads.
NO_FEATURE = {"labels", "relationshipTypes", "propertyKeys", "procedures"}


def procedure_feature(name):
    if name in NO_FEATURE or name.startswith("load_"):
        return None
    return PROCEDURE_FEATURES.get(name, "algos")


def features_of(block):
    called = {
        procedure_feature(name) for name in re.findall(r"CALL\s+grafeo\.([A-Za-z_.]+)", block)
    }
    return {feature for call, feature in FEATURE_CALLS.items() if call in block} | (called - {None})


def run(code, cwd):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_persona_first_time_user_the_readmes_have_examples():
    assert len(self_contained()) >= 3


@pytest.mark.parametrize("block", self_contained())
def test_persona_first_time_user_runs_a_readme_example(block, tmp_path):
    done = run(block, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]


QUICK_START = """
from grafeo import GrafeoDB
db = GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix', age: 30})")
db.execute("INSERT (:Person {name: 'Gus', age: 25})")
"""


@pytest.mark.requires("cdc")
def test_persona_first_time_user_follows_the_transaction_example(tmp_path):
    # The Python README's transaction block, after the quick start.
    block = next(b for b in python_blocks(READMES["python"]) if "begin_transaction()" in b)
    done = run(QUICK_START + block, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]


def test_persona_first_time_user_follows_the_schema_example(tmp_path):
    # The Python README's graphs-and-schemas block, after the quick start.
    block = next(b for b in python_blocks(READMES["python"]) if "set_schema(" in b)
    done = run(QUICK_START + block, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]


def test_persona_first_time_user_opens_a_database_twice_in_one_process(tmp_path):
    # The second handle on a path fails while the first is open: an error
    # that says so, not an internal error about another process.
    code = (
        "import grafeo\n"
        "first = grafeo.GrafeoDB('people')\n"
        "try:\n"
        "    grafeo.GrafeoDB.open('people')\n"
        "except grafeo.GrafeoError as error:\n"
        "    print(error.gql_status, error)\n"
        "first.close()\n"
        "grafeo.GrafeoDB.open('people').close()\n"
        "print('reopened')\n"
    )
    done = run(code, tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "reopened" in done.stdout
    assert "Internal error" not in done.stdout, done.stdout
    assert "locked" in done.stdout, done.stdout
