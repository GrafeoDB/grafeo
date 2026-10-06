"""k-core decomposition returns core numbers, the same through db.algorithms and CALL."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

# (edges, expected core number per node); each edge is created in one direction only.
SHAPES = {
    "triangle": ([("a", "b"), ("b", "c"), ("c", "a")], {"a": 2, "b": 2, "c": 2}),
    "triangle_with_pendant": (
        [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")],
        {"a": 2, "b": 2, "c": 2, "d": 1},
    ),
    "star": ([("h", "x"), ("h", "y"), ("h", "z")], {"h": 1, "x": 1, "y": 1, "z": 1}),
    "k4": (
        [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
        {"a": 3, "b": 3, "c": 3, "d": 3},
    ),
}


def build(edges):
    db = grafeo.GrafeoDB()
    names = sorted({name for edge in edges for name in edge})
    ids = {name: db.create_node(["N"], {"name": name}).id for name in names}
    for source, target in edges:
        db.create_edge(ids[source], ids[target], "R", {})
    return db, {node: name for name, node in ids.items()}


@pytest.mark.parametrize("shape", SHAPES)
def test_kcore_returns_core_numbers(shape):
    edges, expected = SHAPES[shape]
    db, name = build(edges)
    for _ in range(3):
        result = db.algorithms.kcore()
        assert set(result) == {"core_numbers", "max_core"}
        assert {name[node]: core for node, core in result["core_numbers"].items()} == expected
        assert result["max_core"] == max(expected.values())


# `CALL` is a GQL statement: a build without GQL (such as the analytics profile) skips it.
@pytest.mark.gql
@pytest.mark.skipif("gql" not in grafeo.features(), reason="grafeo built without gql feature")
@pytest.mark.parametrize("shape", SHAPES)
def test_call_kcore_yields_the_same_core_numbers(shape):
    edges, expected = SHAPES[shape]
    db, name = build(edges)
    for _ in range(3):
        rows = db.execute("CALL grafeo.kcore() YIELD node_id, core_number")
        assert {name[row["node_id"]]: row["core_number"] for row in rows} == expected


def test_kcore_with_k_returns_the_k_core():
    edges, _ = SHAPES["triangle_with_pendant"]
    db, name = build(edges)
    assert sorted(name[node] for node in db.algorithms.kcore(k=2)) == ["a", "b", "c"]
    assert sorted(name[node] for node in db.algorithms.kcore(k=1)) == ["a", "b", "c", "d"]
    assert db.algorithms.kcore(k=3) == []


def test_kcore_on_an_empty_database():
    assert grafeo.GrafeoDB().algorithms.kcore() == {"core_numbers": {}, "max_core": 0}
