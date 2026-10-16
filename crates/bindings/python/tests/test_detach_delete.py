"""A node deleted with its edges is one change (#465)."""

from grafeo import GrafeoDB


def people(db):
    alix = db.create_node(["Person"], {"name": "Alix"})
    gus = db.create_node(["Person"], {"name": "Gus"})
    vincent = db.create_node(["Person"], {"name": "Vincent"})
    first = db.create_edge(alix.id, gus.id, "KNOWS")
    db.create_edge(alix.id, vincent.id, "KNOWS")
    second = db.create_edge(vincent.id, gus.id, "KNOWS")
    return gus, (first, second)


def test_detach_delete_node_removes_the_node_with_its_edges():
    db = GrafeoDB()
    gus, _ = people(db)
    assert db.detach_delete_node(gus.id) is True
    assert (db.node_count, db.edge_count) == (2, 1)
    assert db.detach_delete_node(gus.id) is False


def test_the_edge_events_name_the_node():
    db = GrafeoDB(cdc=True)
    gus, edges = people(db)
    db.detach_delete_node(gus.id)
    for edge in edges:
        delete = db.edge_history(edge.id)[-1]
        assert delete["kind"] == "delete"
        assert delete["deleted_with"] == gus.id
    assert db.node_history(gus.id)[-1]["deleted_with"] is None
