"""upsert_nodes and upsert_edges: create or update by key, and report every row."""

import pytest
from grafeo import GrafeoDB


def values(db, query):
    return [list(row.values()) for row in db.execute(query)]


@pytest.fixture
def db():
    db = GrafeoDB()
    db.create_property_index("id")
    return db


def test_nodes_are_created_then_updated_by_key(db):
    result = db.upsert_nodes(
        ["Graph", "File"],
        [{"id": "f1", "size": 3}, {"size": 4}, {"id": "f1", "lang": "rs"}],
    )
    assert result == {"created": 1, "updated": 1, "skipped": 1, "skipped_rows": [1]}
    assert values(db, "MATCH (n:Graph:File) RETURN n.id, n.size, n.lang") == [["f1", 3, "rs"]]

    db.upsert_nodes(["Graph", "File"], [{"id": "f1", "size": 5}], replace=True)
    assert values(db, "MATCH (n:File) RETURN n.size, n.lang") == [[5, None]]


def test_edges_connect_existing_nodes_only(db):
    db.upsert_nodes(["File"], [{"id": "f1"}, {"id": "f2"}])
    result = db.upsert_edges(
        "Graph:USES",
        [
            {"src": "f1", "dst": "f2", "id": "u1", "w": 1},
            {"src": "f1", "dst": "missing", "id": "u2"},
            {"src": "f1", "dst": "f2", "id": "u1", "w": 5},
        ],
    )
    assert result == {"created": 1, "updated": 1, "skipped": 1, "skipped_rows": [1]}
    assert values(db, "MATCH ()-[r]->() RETURN type(r), r.id, r.w") == [["Graph:USES", "u1", 5]]


def test_edge_options(db):
    db.upsert_nodes(["File"], [{"id": "f1"}, {"id": "f2"}])
    db.execute("INSERT (:Other {id: 'f2'})")
    result = db.upsert_edges(
        "CALLS",
        [{"from": "f1", "to": "f2", "rid": "c1", "w": 1}],
        key="rid",
        endpoint_labels=["File"],
        src_field="from",
        dst_field="to",
    )
    assert result["created"] == 1
    db.upsert_edges(
        "CALLS",
        [{"from": "f1", "to": "f2", "rid": "c1"}],
        key="rid",
        endpoint_labels=["File"],
        src_field="from",
        dst_field="to",
        replace=True,
    )
    assert values(db, "MATCH (:File)-[r:CALLS]->(d:File) RETURN r.rid, r.w") == [["c1", None]]


def test_a_graph_handle_upserts_into_its_graph():
    db = GrafeoDB()
    db.create_graph("model")
    model = db.graph("model")
    assert model.upsert_nodes(["Component"], [{"id": "c1"}, {"id": "c2"}])["created"] == 2
    assert model.upsert_edges("USES", [{"src": "c1", "dst": "c2", "id": "u1"}])["created"] == 1
    assert values(model, "MATCH (a)-[r]->(b) RETURN a.id, r.id, b.id") == [["c1", "u1", "c2"]]
    assert values(db, "MATCH (n) RETURN count(n)") == [[0]]


def test_a_failed_upsert_writes_nothing(db):
    db.execute("CREATE CONSTRAINT file_path FOR (n:File) ON (n.path) UNIQUE")
    with pytest.raises(Exception, match="UNIQUE"):
        db.upsert_nodes(["File"], [{"id": "f1", "path": "/a"}, {"id": "f2", "path": "/a"}])
    assert values(db, "MATCH (n:File) RETURN count(n)") == [[0]]
