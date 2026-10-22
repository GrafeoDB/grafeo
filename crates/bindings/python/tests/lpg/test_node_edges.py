"""get_node_edges: a node's edges as (neighbor_id, edge_id) pairs without a query (#402)."""

import pytest
from grafeo import GrafeoDB


@pytest.fixture
def graph():
    db = GrafeoDB()
    alix = db.create_node(["Person"], {"name": "Alix"}).id
    gus = db.create_node(["Person"], {"name": "Gus"}).id
    vincent = db.create_node(["Person"], {"name": "Vincent"}).id
    edges = {
        "knows_gus": db.create_edge(alix, gus, "KNOWS").id,
        "knows_vincent": db.create_edge(alix, vincent, "KNOWS").id,
        "works_with": db.create_edge(gus, alix, "WORKS_WITH").id,
        "note": db.create_edge(alix, alix, "NOTE").id,
    }
    return db, alix, gus, vincent, edges


def test_outgoing_is_the_default(graph):
    db, alix, gus, vincent, e = graph
    assert sorted(db.get_node_edges(alix)) == sorted(
        [(gus, e["knows_gus"]), (vincent, e["knows_vincent"]), (alix, e["note"])]
    )
    assert all(isinstance(pair, tuple) for pair in db.get_node_edges(alix))


def test_directions(graph):
    db, alix, gus, vincent, e = graph
    assert sorted(db.get_node_edges(alix, direction="in")) == sorted(
        [(gus, e["works_with"]), (alix, e["note"])]
    )
    # Both directions list the self-loop once.
    assert sorted(db.get_node_edges(alix, direction="both")) == sorted(
        [
            (gus, e["knows_gus"]),
            (vincent, e["knows_vincent"]),
            (gus, e["works_with"]),
            (alix, e["note"]),
        ]
    )
    assert db.get_node_edges(vincent, direction="out") == []
    with pytest.raises(ValueError, match="direction"):
        db.get_node_edges(alix, direction="sideways")


def test_edge_types(graph):
    db, alix, gus, vincent, e = graph
    assert sorted(db.get_node_edges(alix, "both", ["KNOWS"])) == sorted(
        [(gus, e["knows_gus"]), (vincent, e["knows_vincent"])]
    )
    assert sorted(db.get_node_edges(alix, "both", edge_types=("WORKS_WITH", "NOTE"))) == sorted(
        [(gus, e["works_with"]), (alix, e["note"])]
    )
    assert db.get_node_edges(alix, edge_types="NOTE") == [(alix, e["note"])]
    assert db.get_node_edges(alix, edge_types=[]) == db.get_node_edges(alix)


def test_deleted_edges_and_missing_nodes(graph):
    db, alix, gus, vincent, e = graph
    db.delete_edge(e["knows_gus"])
    assert db.get_node_edges(gus, "in") == []
    db.detach_delete_node(vincent)
    assert db.get_node_edges(alix, edge_types=["KNOWS"]) == []
    assert db.get_node_edges(88888, "both") == []


def test_a_graph_handle_lists_the_edges_of_its_graph():
    db = GrafeoDB()
    db.create_graph("trips")
    trips = db.graph("trips")
    prague = trips.create_node(["City"], {"name": "Prague"}).id
    berlin = trips.create_node(["City"], {"name": "Berlin"}).id
    route = trips.create_edge(prague, berlin, "ROUTE").id
    assert trips.get_node_edges(prague) == [(berlin, route)]
    assert trips.get_node_edges(berlin, "in", ["ROUTE"]) == [(prague, route)]
    assert db.get_node_edges(prague, "both") == []

    db.drop_graph("trips")
    with pytest.raises(Exception, match="trips"):
        trips.get_node_edges(prague)
