"""A read that fails raises; an absent node or edge reads as None.

A read of a graph that was dropped is a failing read: a graph handle raises
``GrafeoError`` instead of answering None.
"""

import grafeo
import pytest
from grafeo import GrafeoDB


def test_an_absent_node_or_edge_reads_as_none():
    db = GrafeoDB()
    mia = db.create_node(["Person"], {"name": "Mia"})
    db.delete_node(mia.id)
    assert db.get_node(mia.id) is None
    assert db.get_node_labels(mia.id) is None
    assert db.get_node(424242) is None
    assert db.get_edge(424242) is None
    assert db.node_count == 0


def test_a_graph_handle_raises_once_its_graph_is_dropped():
    db = GrafeoDB()
    db.create_graph("trips")
    trips = db.graph("trips")
    berlin = trips.create_node(["City"], {"name": "Berlin"})
    assert trips.get_node(424242) is None, "absent, not an error"
    assert trips.get_node(berlin.id).properties()["name"] == "Berlin"
    db.execute("DROP GRAPH trips")
    with pytest.raises(grafeo.GrafeoError) as raised:
        trips.get_node(berlin.id)
    assert len(raised.value.gql_status) == 5
