"""to_memory() and reopening a file keep indexes and constraints."""

import pytest
from grafeo import GrafeoDB


def build(db):
    db.create_property_index("id")
    db.execute("CREATE CONSTRAINT file_id FOR (n:Graph) ON (n.id) UNIQUE")
    db.execute("INSERT (:Graph:File {id: 'f0'})")
    db.create_graph("model")
    model = db.graph("model")
    model.create_property_index("id")
    model.execute("INSERT (:Graph {id: 'm0'})")


def assert_built(db):
    assert db.has_property_index("id")
    assert db.graph("model").has_property_index("id")
    assert len(db.graph("model").find_nodes_by_property("id", "m0")) == 1
    with pytest.raises(Exception, match="(?i)unique"):
        db.execute("INSERT (:Graph:File {id: 'f0'})")


@pytest.mark.parametrize("persistent", [False, True])
def test_to_memory_keeps_indexes_and_constraints(tmp_path, persistent):
    source = GrafeoDB(str(tmp_path / "source.grafeo")) if persistent else GrafeoDB()
    build(source)
    copy = source.to_memory()
    assert_built(copy)
    copy.execute("INSERT (:Graph:File {id: 'f1'})")
    assert list(source.find_nodes_by_property("id", "f1")) == []
    source.close()


def test_reopening_a_file_keeps_indexes_and_constraints(tmp_path):
    path = str(tmp_path / "db.grafeo")
    db = GrafeoDB(path)
    build(db)
    db.close()
    db = GrafeoDB(path)
    assert_built(db)
    db.close()


def test_to_memory_keeps_vector_and_text_indexes():
    db = GrafeoDB()
    db.create_node(["Doc"], {"emb": [1.0, 0.0, 0.0], "body": "rust graph database"})
    db.create_node(["Doc"], {"emb": [0.0, 2.0, 0.0], "body": "python web framework"})
    db.create_vector_index("Doc", "emb", dimensions=3, metric="euclidean")
    db.create_text_index("Doc", "body")
    copy = db.to_memory()
    query = [2.0, 0.0, 0.0]
    assert copy.vector_search("Doc", "emb", query, 2) == db.vector_search("Doc", "emb", query, 2)
    assert len(copy.text_search("Doc", "body", "graph", 10)) == 1
