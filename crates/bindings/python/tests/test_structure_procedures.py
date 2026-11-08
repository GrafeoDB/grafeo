"""The bridge's ktruss, subgraph_isomorphism_count and total_triangles answer as CALL does.

The engine test `structure_procedures.rs` checks the answers against networkx and brute forces;
these check that the Python methods give the same answers in their documented shapes:
`ktruss()` a dict of `max_truss` and `truss_numbers` keyed by `(source, target)`, `ktruss(k)` a
list of `(source, target)`, `subgraph_isomorphism_count(pattern_edges, pattern_nodes)` an int with
the pattern as a list of pairs, and a malformed pattern refused as CALL refuses its string form.
"""

import itertools

import grafeo
import pytest

pytestmark = [
    pytest.mark.skipif(
        not hasattr(grafeo.GrafeoDB(), "algorithms"),
        reason="grafeo built without algos feature",
    ),
    pytest.mark.skipif("gql" not in grafeo.features(), reason="grafeo built without gql feature"),
]

INVALID_PARAMETER_VALUE = "22023"


def build(nodes, edges):
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["V"]).id for _ in range(nodes)]
    for u, v in edges:
        db.create_edge(ids[u], ids[v], "E")
    return db, ids


def clique(n):
    return list(itertools.combinations(range(n), 2))


# A clique of five, two 4-cliques sharing a node, a tree, and a triangle with a tail written in
# both directions, with a parallel edge and self-loops.
GRAPHS = [
    (5, clique(5)),
    (7, clique(4) + [(u + 3, v + 3) for u, v in clique(4)]),
    (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    (4, [(0, 1), (1, 0), (1, 2), (1, 2), (2, 0), (2, 2), (3, 3), (2, 3)]),
    (3, []),
]


def call_rows(db, query):
    return [list(row.values()) for row in db.execute(query)]


@pytest.mark.parametrize(("nodes", "edges"), GRAPHS)
def test_ktruss_is_the_call_decomposition(nodes, edges):
    db, _ = build(nodes, edges)
    found = db.algorithms.ktruss()
    rows = call_rows(db, "CALL grafeo.ktruss() YIELD source, target, truss_number")
    assert found["truss_numbers"] == {(s, t): k for s, t, k in rows}
    assert found["max_truss"] == max((k for _, _, k in rows), default=0)
    assert all(s < t for s, t in found["truss_numbers"])


@pytest.mark.parametrize(("nodes", "edges"), GRAPHS)
def test_ktruss_at_k_is_the_call_list(nodes, edges):
    db, _ = build(nodes, edges)
    largest = db.algorithms.ktruss()["max_truss"]
    for k in range(largest + 2):
        rows = call_rows(db, f"CALL grafeo.ktruss({k}) YIELD source, target")
        assert db.algorithms.ktruss(k) == [(s, t) for s, t in rows], k


def test_ktruss_of_a_clique_of_five_is_five_everywhere():
    db, _ = build(5, clique(5))
    found = db.algorithms.ktruss()
    assert found["max_truss"] == 5
    assert sorted(set(found["truss_numbers"].values())) == [5]
    assert len(found["truss_numbers"]) == 10


@pytest.mark.parametrize(("nodes", "edges"), GRAPHS)
def test_total_triangles_agrees_with_call_and_the_per_node_count(nodes, edges):
    db, _ = build(nodes, edges)
    total = db.algorithms.total_triangles()
    (row,) = call_rows(db, "CALL grafeo.total_triangles() YIELD total_triangles")
    assert total == row[0]
    assert total == sum(db.algorithms.triangle_count().values()) // 3


PATTERNS = [
    ([(0, 1)], 2, "0-1"),
    ([(0, 1), (1, 2)], 3, "0-1,1-2"),
    ([(0, 1), (1, 2), (2, 0)], 3, "0-1,1-2,2-0"),
    ([(0, 1), (1, 2), (2, 3), (3, 0)], 4, "0-1,1-2,2-3,3-0"),
    ([(0, 1), (0, 2), (0, 3)], 4, "0-1,0-2,0-3"),
    ([(0, 1)], 3, "0-1"),
    ([], 2, ""),
]


@pytest.mark.parametrize(("nodes", "edges"), GRAPHS)
def test_subgraph_isomorphism_count_is_the_call_count(nodes, edges):
    db, _ = build(nodes, edges)
    for pattern, pattern_nodes, text in PATTERNS:
        (row,) = call_rows(
            db,
            f"CALL grafeo.subgraph_isomorphism('{text}', {pattern_nodes}) YIELD count",
        )
        assert db.algorithms.subgraph_isomorphism_count(pattern, pattern_nodes) == row[0], text


def test_the_triangle_pattern_counts_six_per_triangle():
    db, _ = build(7, clique(4) + [(u + 3, v + 3) for u, v in clique(4)])
    triangle = [(0, 1), (1, 2), (2, 0)]
    count = db.algorithms.subgraph_isomorphism_count(triangle, 3)
    assert count == 6 * db.algorithms.total_triangles() == 48


def test_a_path_of_three_on_a_triangle_counts_six():
    db, _ = build(3, [(0, 1), (1, 2), (2, 0)])
    assert db.algorithms.subgraph_isomorphism_count([(0, 1), (1, 2)], 3) == 6


@pytest.mark.parametrize(
    ("pattern", "pattern_nodes"),
    [([(0, 3)], 3), ([(0, 1), (1, 5)], 3), ([(-1, 2)], 3), ([(0, 1)], 0)],
)
def test_a_malformed_pattern_is_refused_as_call_refuses_it(pattern, pattern_nodes):
    db, _ = build(3, [(0, 1), (1, 2), (2, 0)])
    with pytest.raises(grafeo.GrafeoError) as bridge:
        db.algorithms.subgraph_isomorphism_count(pattern, pattern_nodes)
    text = ",".join(f"{u}-{v}" for u, v in pattern)
    with pytest.raises(grafeo.GrafeoError) as call:
        db.execute(f"CALL grafeo.subgraph_isomorphism('{text}', {pattern_nodes}) YIELD count")
    assert bridge.value.gql_status == call.value.gql_status == INVALID_PARAMETER_VALUE
    assert "pattern_edges" in str(bridge.value), str(bridge.value)
