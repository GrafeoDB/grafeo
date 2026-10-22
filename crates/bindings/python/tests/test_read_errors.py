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


def test_the_graph_bridges_read_the_selected_graph():
    db = GrafeoDB()
    db.create_node(["Person"], {"name": "Alix"})
    db.create_graph("trips")
    trips = db.graph("trips")
    berlin = trips.create_node(["City"], {"name": "Berlin"})
    paris = trips.create_node(["City"], {"name": "Paris"})
    trips.create_edge(berlin.id, paris.id, "ROUTE")
    assert db.as_networkx().number_of_nodes == 1
    db.set_graph("trips")
    adapter = db.as_networkx()
    assert (adapter.number_of_nodes, adapter.number_of_edges) == (2, 1)
    assert sorted(adapter.nodes()) == sorted([berlin.id, paris.id])
    assert db.as_solvor().graph_stats()["nodes"] == 2
    assert sorted(id for id, _ in db.get_nodes_by_label("City")) == sorted([berlin.id, paris.id])
    db.reset_graph()
    assert db.as_networkx().number_of_nodes == 1
