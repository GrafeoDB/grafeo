"""`topological_sort()` keeps returning None on a graph with a cycle, as documented, while CALL
raises 22000 naming the cycle; `is_dag()` is the check to make first (the ruling of 2026-11-06)."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

CYCLIC = [
    (4, [(0, 1), (1, 2), (2, 0), (2, 3)]),
    (3, [(0, 1), (2, 2)]),
    (5, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)]),
    (4, [(1, 2), (2, 3), (3, 1), (3, 0)]),
    (40, [(i, (i + 1) % 40) for i in range(40)]),
]


def build(nodes, edges):
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["V"]).id for _ in range(nodes)]
    for u, v in edges:
        db.create_edge(ids[u], ids[v], "E")
    return db, ids


@pytest.mark.parametrize(("nodes", "edges"), CYCLIC)
def test_a_cycle_gives_none_and_is_no_dag(nodes, edges):
    db, _ = build(nodes, edges)
    assert db.algorithms.topological_sort() is None
    assert db.algorithms.is_dag() is False
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute("CALL grafeo.topological_sort()")
    assert raised.value.gql_status == "22000"
    assert "the graph has a cycle: " in str(raised.value)


def test_a_dag_gives_its_order():
    db, ids = build(4, [(2, 0), (0, 1), (3, 1)])
    assert db.algorithms.is_dag() is True
    assert db.algorithms.topological_sort() == [ids[2], ids[0], ids[3], ids[1]]
