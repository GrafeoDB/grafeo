"""set_graph, drop_graph and set_schema make the checks a query makes (#245).

set_graph to a graph that does not exist, or set_schema to a missing schema,
raised nothing and left the session in a graph or schema that was not there;
dropping the active graph left it selected.
"""

import pytest
from grafeo import GrafeoDB, GrafeoError


def count(db):
    return next(iter(db.execute("MATCH (n) RETURN count(n) AS c")))["c"]


def test_set_graph_to_a_missing_graph_raises_and_keeps_the_graph():
    db = GrafeoDB()
    db.create_graph("trips")
    db.set_graph("trips")
    with pytest.raises(GrafeoError, match="Graph 'missing' does not exist"):
        db.set_graph("missing")
    assert db.current_graph() == "trips"


def test_set_schema_to_a_missing_schema_raises_and_keeps_the_schema():
    db = GrafeoDB()
    with pytest.raises(GrafeoError, match="Schema 'missing' does not exist"):
        db.set_schema("missing")
    assert db.current_schema() is None


def test_dropping_the_active_graph_returns_to_the_default_graph():
    db = GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.create_graph("trips")
    db.set_graph("trips")
    db.execute("INSERT (:Stop {city: 'Prague'}), (:Stop {city: 'Berlin'})")
    assert count(db) == 2
    assert db.drop_graph("trips") is True
    assert db.current_graph() is None
    assert count(db) == 1, "queries read the default graph again"
    assert db.drop_graph("trips") is False, "dropping a missing graph reports it"
