"""schema() lists only the labels, edge types and property keys still in use.

A label whose nodes were all deleted or relabelled, an edge type with no edge
left, and a property key no element holds are left out: LLM integrations put
schema() into prompts, and a label that matches nothing misleads them.
CALL db.labels() agrees (tests/spec/lpg/gql/index_and_graph_type_ddl.gtest).
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


def names(entries):
    return sorted(entry["name"] if isinstance(entry, dict) else entry for entry in entries)


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Gone {nick: 'x'})-[:WENT]->(:Stay {name: 'Gus'})")
    db.execute("INSERT (:Draft {title: 'Canals'})")
    return db


def test_a_label_and_edge_type_with_no_element_left_are_left_out(db):
    db.execute("MATCH (n:Gone) DETACH DELETE n")
    schema = db.schema()
    assert names(schema["labels"]) == ["Draft", "Stay"]
    assert names(schema["edge_types"]) == []


def test_a_relabelled_label_is_left_out(db):
    db.execute("MATCH (d:Draft) REMOVE d:Draft SET d:Published")
    assert names(db.schema()["labels"]) == ["Gone", "Published", "Stay"]


def test_a_removed_property_key_is_left_out(db):
    db.execute("MATCH (n:Gone) DETACH DELETE n")
    db.execute("MATCH (s:Stay) REMOVE s.name SET s.city = 'Paris'")
    assert sorted(db.schema()["property_keys"]) == ["city", "title"]


def test_counts_of_the_labels_in_use():
    # Control: labels in use keep their counts.
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'}), (:City {name: 'Paris'})")
    labels = {entry["name"]: entry["count"] for entry in db.schema()["labels"]}
    assert labels == {"Person": 2, "City": 1}
