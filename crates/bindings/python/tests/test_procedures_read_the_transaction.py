"""Every graph procedure reads what the statement around it may see.

Inside a transaction a procedure reads that transaction's snapshot and its own
writes; outside, it reads the committed graph. Each result is compared with the
same procedure on a fresh database committed to the expected state, so a
procedure added later is covered too, and must name its arguments in
``arguments_of``. ``test_snapshot_isolation.py`` covers plain queries.
"""

import json

import grafeo
import pytest

# The procedures are listed at import: in a build without algos or gql the list stays empty, so
# the module is collected and skipped rather than failing.
RUNS = hasattr(grafeo.GrafeoDB(), "algorithms") and "gql" in grafeo.features()

pytestmark = [
    pytest.mark.skipif(
        not hasattr(grafeo.GrafeoDB(), "algorithms"),
        reason="grafeo built without algos feature",
    ),
    pytest.mark.skipif("gql" not in grafeo.features(), reason="grafeo built without gql feature"),
]

# Alix -> Gus -> Vincent -> Mia
BEFORE = [
    "INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})",
    "INSERT (:Person {name: 'Vincent'}), (:Person {name: 'Mia'})",
    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS {w: 1}]->(b)",
    "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'}) INSERT (a)-[:KNOWS {w: 1}]->(b)",
    "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Mia'}) INSERT (a)-[:KNOWS {w: 1}]->(b)",
]
# Adds a node and two edges, and removes the edge that held the chain together. The new edges
# close a triangle (Vincent, Mia, Butch) that has no directed cycle, so the triangle counts see the
# change and a topological order still exists.
CHANGE = [
    "INSERT (:Person {name: 'Butch'})",
    "MATCH (:Person {name: 'Gus'})-[e:KNOWS]->(:Person {name: 'Vincent'}) DELETE e",
    "MATCH (a:Person {name: 'Mia'}), (b:Person {name: 'Butch'}) INSERT (a)-[:KNOWS {w: 1}]->(b)",
    "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Butch'}) "
    "INSERT (a)-[:KNOWS {w: 1}]->(b)",
]
EDGES_BEFORE = [("Alix", "Gus"), ("Gus", "Vincent"), ("Vincent", "Mia")]
EDGES_AFTER = [("Alix", "Gus"), ("Vincent", "Mia"), ("Mia", "Butch"), ("Vincent", "Butch")]

# These list the catalog: the change adds no label, property or edge type.
CATALOG = {"grafeo.labels", "grafeo.propertyKeys", "grafeo.relationshipTypes"}


def database(*statements):
    db = grafeo.GrafeoDB()
    for statement in statements:
        db.execute(statement)
    return db


def ids_by_name(reader):
    rows = reader.execute("MATCH (p:Person) RETURN p.name AS name, id(p) AS node")
    return {row["name"]: row["node"] for row in rows}


IDS = ids_by_name(database(*BEFORE)) if RUNS else {}


def arguments_of(name):
    alix, mia = IDS["Alix"], IDS["Mia"]
    return {
        "grafeo.bfs": f"{alix}",
        "grafeo.dfs": f"{alix}",
        "grafeo.dijkstra": f"{alix}",
        "grafeo.bellman_ford": f"{alix}",
        "grafeo.sssp": "'Alix'",
        "grafeo.max_flow": f"{alix}, {mia}",
        "grafeo.min_cost_max_flow": f"{alix}, {mia}",
        # Its pattern is required: a triangle.
        "grafeo.subgraph_isomorphism": "'0-1,1-2,2-0', 3",
    }.get(name, "")


PROCEDURES = (
    sorted(
        row["name"]
        for row in database().execute("CALL grafeo.procedures()")
        # Search needs an index, and reads no adjacency.
        if not row["name"].startswith("grafeo.search.")
    )
    if RUNS
    else []
)


def result_of(reader, name):
    rows = reader.execute(f"CALL {name}({arguments_of(name)})")
    return sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)


def assert_reads(reader, name, statements, edges):
    """``reader`` sees what a fresh database committed to ``statements`` shows."""
    got = result_of(reader, name)
    if name == "grafeo.topological_sort":
        # Several orders are right, so check that this one is.
        ids = ids_by_name(database(*statements))
        order = {json.loads(row)["node_id"]: json.loads(row)["order"] for row in got}
        assert set(order) == set(ids.values())
        assert all(order[ids[a]] < order[ids[b]] for a, b in edges)
    else:
        assert got == result_of(database(*statements), name)


def test_procedures_were_found():
    assert "grafeo.pagerank" in PROCEDURES
    assert "grafeo.connected_components" in PROCEDURES


@pytest.mark.parametrize("name", sorted(set(PROCEDURES) - CATALOG))
def test_the_change_shows_in_the_result(name):
    """Without this, the tests below would pass for a procedure that reads nothing."""
    before = result_of(database(*BEFORE), name)
    assert before != result_of(database(*BEFORE, *CHANGE), name)


@pytest.mark.parametrize("name", PROCEDURES)
def test_a_transaction_reads_its_own_writes(name):
    db = database(*BEFORE)
    result_of(db, name)
    tx = db.begin_transaction()
    try:
        for statement in CHANGE:
            tx.execute(statement)
        assert_reads(tx, name, [*BEFORE, *CHANGE], EDGES_AFTER)
    finally:
        tx.rollback()


@pytest.mark.parametrize("name", PROCEDURES)
def test_another_session_reads_the_committed_graph(name):
    db = database(*BEFORE)
    result_of(db, name)
    tx = db.begin_transaction()
    try:
        for statement in CHANGE:
            tx.execute(statement)
        assert_reads(db, name, BEFORE, EDGES_BEFORE)
    finally:
        tx.rollback()


@pytest.mark.parametrize("name", PROCEDURES)
def test_a_rollback_leaves_the_committed_graph(name):
    db = database(*BEFORE)
    result_of(db, name)
    tx = db.begin_transaction()
    for statement in CHANGE:
        tx.execute(statement)
    tx.rollback()
    assert_reads(db, name, BEFORE, EDGES_BEFORE)


@pytest.mark.parametrize("name", PROCEDURES)
def test_an_open_transaction_does_not_read_a_later_commit(name):
    db = database(*BEFORE)
    reader = db.begin_transaction()
    try:
        assert_reads(reader, name, BEFORE, EDGES_BEFORE)
        for statement in CHANGE:
            db.execute(statement)
        assert_reads(reader, name, BEFORE, EDGES_BEFORE)
    finally:
        reader.rollback()
    assert_reads(db, name, [*BEFORE, *CHANGE], EDGES_AFTER)
