"""batch_create_edges and batch nodes with several labels: one transaction per batch."""

import pytest
from grafeo import GrafeoDB


def values(target, query):
    return [list(row.values()) for row in target.execute(query)]


def people(db):
    return db.batch_create_nodes_with_props(
        "Person", [{"name": "Alix"}, {"name": "Gus"}, {"name": "Vincent"}]
    )


def test_batch_nodes_take_a_list_of_labels():
    db = GrafeoDB()
    ids = db.batch_create_nodes_with_props(["Graph", "File"], [{"id": "f1"}, {"id": "f2"}])
    assert len(ids) == 2
    assert values(db, "MATCH (n:Graph:File) RETURN count(n)") == [[2]]


def test_batch_edges_carry_their_type_and_properties():
    db = GrafeoDB()
    alix, gus, vincent = people(db)
    ids = db.batch_create_edges([(alix, gus, "KNOWS", {"since": 2020}), (gus, vincent, "LIKES")])
    assert len(ids) == 2
    assert values(
        db, "MATCH (a)-[r]->(b) RETURN a.name, type(r), r.since, b.name ORDER BY a.name"
    ) == [
        ["Alix", "KNOWS", 2020, "Gus"],
        ["Gus", "LIKES", None, "Vincent"],
    ]


def test_a_failing_batch_of_edges_creates_none():
    db = GrafeoDB()
    alix, gus, _ = people(db)
    with pytest.raises(Exception, match="does not exist"):
        db.batch_create_edges([(alix, gus, "KNOWS"), (alix, 999, "KNOWS")])
    assert values(db, "MATCH ()-[r]->() RETURN count(r)") == [[0]]
    with pytest.raises(TypeError):
        db.batch_create_edges([(alix, gus)])


def test_a_graph_handle_batches_into_its_graph():
    db = GrafeoDB()
    db.create_graph("model")
    model = db.graph("model")
    a, b = model.batch_create_nodes_with_props(
        ["Graph", "Component"], [{"id": "ac::a"}, {"id": "ac::b"}]
    )
    model.batch_create_edges([(a, b, "USES")])
    assert values(model, "MATCH (:Graph:Component)-[r:USES]->() RETURN count(r)") == [[1]]
    assert values(db, "MATCH ()-[r]->() RETURN count(r)") == [[0]]
