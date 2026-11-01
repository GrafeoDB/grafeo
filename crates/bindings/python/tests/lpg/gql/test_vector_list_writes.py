"""A Python list written to a vector-indexed property is an embedding (#587).

The binding makes a list of only floats a VECTOR, but a list with an integer
(`[1, 0, 0]`, what many embedding tools give for a sparse vector) stays a
LIST, and a GQL list literal is a LIST too: both were stored where the index
could not see them, with no error. A numeric list of the indexed dimension is
now a VECTOR the index finds; one of another dimension fails; a list on a
property without a vector index stays a LIST.
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

WRONG_SIZE = "has a vector index of 3 dimensions, got a vector of 2"


@pytest.fixture
def db():
    """A 3-dimensional cosine index on :Doc(emb), with one VECTOR document."""
    db = grafeo.GrafeoDB()
    db.create_node(["Doc"], {"name": "anchor", "emb": grafeo.vector([0.0, 0.0, 1.0])})
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    return db


def found(db, query):
    """The names of the documents a search for `query` finds at distance 0."""
    names = []
    for node_id, distance in db.vector_search("Doc", "emb", query, k=100):
        if distance < 1e-6:
            names.append(db.get_node(node_id).properties()["name"])
    return sorted(names)


def is_list(db, name):
    rows = list(
        db.execute("MATCH (d {name: $name}) RETURN d.emb IS TYPED LIST AS l", {"name": name})
    )
    return rows[0]["l"]


@pytest.mark.parametrize(
    "emb",
    [[1, 0, 0], [1, 0.0, 0], [1.0, 0.0, 0.0]],
    ids=["integers", "mixed", "floats"],
)
def test_create_node_with_a_list_is_indexed(db, emb):
    db.create_node(["Doc"], {"name": "Alix", "emb": emb})
    assert found(db, [1.0, 0.0, 0.0]) == ["Alix"]
    assert is_list(db, "Alix") is False


def test_set_node_property_with_an_integer_list_is_indexed(db):
    node = db.create_node(["Doc"], {"name": "Gus"})
    db.set_node_property(node.id, "emb", [0, 1, 0])
    assert found(db, [0.0, 1.0, 0.0]) == ["Gus"]
    assert is_list(db, "Gus") is False


def test_a_list_parameter_and_a_literal_are_indexed(db):
    db.execute("INSERT (:Doc {name: 'Vincent', emb: $emb})", {"emb": [1, 1, 0]})
    db.execute("INSERT (:Doc {name: 'Jules', emb: [0.0, 1.0, 1.0]})")
    assert found(db, [1.0, 1.0, 0.0]) == ["Vincent"]
    assert found(db, [0.0, 1.0, 1.0]) == ["Jules"]


@pytest.mark.parametrize("emb", [[1, 0], [1.0, 0.0]], ids=["integers", "floats"])
def test_a_list_of_another_dimension_fails(db, emb):
    with pytest.raises(Exception, match=WRONG_SIZE):
        db.create_node(["Doc"], {"name": "Mia", "emb": emb})
    with pytest.raises(Exception, match=WRONG_SIZE):
        db.execute("INSERT (:Doc {name: 'Mia', emb: $emb})", {"emb": emb})
    assert list(db.execute("MATCH (d:Doc {name: 'Mia'}) RETURN d")) == []


def test_a_list_on_an_unindexed_property_stays_a_list(db):
    db.create_node(["Draft"], {"name": "Butch", "emb": [1, 0, 0]})
    db.create_node(["Doc"], {"name": "Django", "tags": [1, 0, 0]})
    assert is_list(db, "Butch") is True
    rows = list(db.execute("MATCH (d {name: 'Django'}) RETURN d.tags IS TYPED LIST AS l"))
    assert rows[0]["l"] is True
