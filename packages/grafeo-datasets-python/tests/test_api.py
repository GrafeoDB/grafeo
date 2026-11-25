"""Tests for grafeo_datasets.silicon_valley. Run after `node datasets/tools/stage.mjs`."""

import importlib.util
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from grafeo_datasets import silicon_valley as sv

PACKAGE = Path(__file__).resolve().parent.parent
GRAPH = PACKAGE.parent.parent / "datasets" / "silicon-valley" / "derived" / "graph.json"
TWICE = (
    "The silicon-valley dataset is already in this database, fully or in part "
    "(a node with key 'anton' exists)."
)


def test_nodes_and_edges_match_graph_json():
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    assert sv.nodes() == graph["nodes"]
    assert sv.edges() == graph["edges"]
    assert sv.nodes() is not sv.nodes()
    assert len(sv.timeline()) > 0


def test_variants_match_the_derived_files():
    derived = GRAPH.parent
    assert sv.rdfs() == (derived / "silicon-valley.rdfs.ttl").read_text(
        encoding="utf-8"
    )
    assert sv.owl() == (derived / "silicon-valley.owl.ttl").read_text(encoding="utf-8")
    lpg = sv.lpg_import()
    assert lpg == json.loads(
        (derived / "silicon-valley.lpg.json").read_text(encoding="utf-8")
    )
    assert len(lpg["nodes"]) == len(sv.nodes())
    rdf = sv.rdf_import()
    assert rdf == json.loads(
        (derived / "silicon-valley.rdf.json").read_text(encoding="utf-8")
    )


def test_vectors_shape():
    vs = sv.vectors()
    assert vs.dimension == 384
    assert vs.model == "Xenova/all-MiniLM-L6-v2"
    assert set(vs.by_key) == {n["key"] for n in sv.nodes()}
    for key, vec in vs.by_key.items():
        assert len(vec) == 384, key
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-3), key


def test_scripts_are_text():
    assert sv.gql().lstrip().startswith("INSERT")
    assert "richard-hendricks" in sv.cypher()
    assert "https://docs.grafeo.dev/datasets/silicon-valley/" in sv.turtle()


def grafeo_db():
    grafeo = pytest.importorskip("grafeo")
    return grafeo.GrafeoDB()


def lines(text):
    return [line for line in text.split("\n") if line.strip()]


def test_load_counts_statements():
    db = grafeo_db()
    assert sv.load(db) == len(lines(sv.gql()))
    count = db.execute("MATCH (n) RETURN count(n) AS c").scalar()
    assert count == len(sv.nodes())


def test_load_twice_refuses():
    db = grafeo_db()
    sv.load(db)
    with pytest.raises(ValueError) as error:
        sv.load(db)
    assert str(error.value) == TWICE
    assert db.execute("MATCH (n) RETURN count(n) AS c").scalar() == len(sv.nodes())


def test_load_after_a_partial_load_refuses():
    db = grafeo_db()
    # A load that failed after its first statement leaves the first node behind.
    db.execute(lines(sv.gql())[0])
    with pytest.raises(ValueError) as error:
        sv.load(db)
    assert str(error.value) == TWICE
    assert db.execute("MATCH (n) RETURN count(n) AS c").scalar() == 1
    with pytest.raises(ValueError):
        sv.load(db, vectors=True)


def test_load_with_vectors_searches():
    db = grafeo_db()
    n = sv.load(db, vectors=True)
    assert n > len(lines(sv.gql()))
    query = sv.vectors().by_key["middle-out"]
    hits = db.vector_search("Technology", "embedding", query, 2)
    assert len(hits) == 2
    first = db.execute(
        f"MATCH (n) WHERE id(n) = {hits[0][0]} RETURN n.key AS k"
    ).scalar()
    assert first == "middle-out"


def test_wheel_contains_exactly_the_expected_files(tmp_path):
    # Against an installed wheel (a release check in a fresh venv) there is nothing to build.
    if not (PACKAGE / "pyproject.toml").is_file() or not importlib.util.find_spec(
        "build"
    ):
        pytest.skip("needs the package source and the build module")
    command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)]
    # Without network, build in the current environment when it has the build tools.
    if importlib.util.find_spec("hatchling") and importlib.util.find_spec("build"):
        command.append("--no-isolation")
    try:
        subprocess.run(command, cwd=PACKAGE, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"{' '.join(command)} failed:"
            + chr(10)
            + f"{error.stderr}"
            + chr(10)
            + f"{error.stdout}"
        )
    (wheel,) = tmp_path.glob("grafeo_datasets-*.whl")
    names = {
        n
        for n in zipfile.ZipFile(wheel).namelist()
        if "/dist-info/" not in n and ".dist-info/" not in n
    }
    data = "grafeo_datasets/data/silicon_valley/"
    assert names == {
        "grafeo_datasets/__init__.py",
        "grafeo_datasets/silicon_valley.py",
        "grafeo_datasets/py.typed",
        data + "graph.json",
        data + "timeline.json",
        data + "silicon-valley.gql",
        data + "silicon-valley.cypher",
        data + "silicon-valley.ttl",
        data + "silicon-valley.vectors.gql",
        data + "vectors.f32",
        data + "vectors.json",
    }
