"""schema() gives each label and edge type its properties and value types.

Every label and edge type lists the properties its elements hold, each with
the sorted types of its values (nulls add none), and every edge type lists
the source and target label pairs it connects with a count. A node with
several labels counts in each pair; an end without a label is in none. An
LLM integration puts this into a prompt, so it is plain dicts and lists.
"""

import json

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


def by_name(entries, name):
    return next(entry for entry in entries if entry["name"] == name)


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (alix:Person:Employee {name: 'Alix', age: 30}), "
        "(gus:Person {name: 'Gus', age: '31'}), "
        "(acme:Company:Org {name: 'Acme'}), ({note: 'loose'}), "
        "(alix)-[:WORKS_AT {since: 2020}]->(acme), (gus)-[:WORKS_AT {since: 2.5e0}]->(acme)"
    )
    db.execute("MATCH (n {note: 'loose'}), (g:Person {name: 'Gus'}) INSERT (g)-[:LINK]->(n)")
    return db


def test_a_label_lists_its_properties_with_their_types(db):
    person = by_name(db.schema()["labels"], "Person")
    assert person["count"] == 2
    assert person["properties"] == [
        {"name": "age", "types": ["INT64", "STRING"]},
        {"name": "name", "types": ["STRING"]},
    ]


def test_an_edge_type_lists_every_label_pair_it_connects(db):
    works_at = by_name(db.schema()["edge_types"], "WORKS_AT")
    assert works_at["count"] == 2
    assert works_at["endpoints"] == [
        {"source": "Employee", "target": "Company", "count": 1},
        {"source": "Employee", "target": "Org", "count": 1},
        {"source": "Person", "target": "Company", "count": 2},
        {"source": "Person", "target": "Org", "count": 2},
    ]
    assert works_at["properties"] == [{"name": "since", "types": ["FLOAT64", "INT64"]}]


def test_an_edge_to_an_unlabelled_node_is_in_no_pair(db):
    link = by_name(db.schema()["edge_types"], "LINK")
    assert link["count"] == 1
    assert link["endpoints"] == []
    assert link["properties"] == []


def test_the_summary_is_plain_json(db):
    # Dicts, lists, strings and ints only: json.dumps takes it as it is,
    # and reading it back gives the same summary.
    schema = db.schema()
    assert json.loads(json.dumps(schema)) == schema


def test_the_summary_agrees_with_the_db_procedures(db):
    schema = db.schema()

    def column(query):
        return [next(iter(row.values())) for row in db.execute(query).to_list()]

    assert column("CALL db.labels() YIELD label RETURN label") == [
        label["name"] for label in schema["labels"]
    ]
    assert column("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType") == [
        edge_type["name"] for edge_type in schema["edge_types"]
    ]
    assert column("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey") == list(
        schema["property_keys"]
    )
