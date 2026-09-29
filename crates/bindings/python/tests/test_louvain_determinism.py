"""Louvain returns the same communities, numbered the same way, on every call."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)


def tied_moves_graph():
    """30 nodes where many Louvain moves gain the same modularity."""
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["N"], {"id": f"n{i}"}).id for i in range(30)]
    for i in range(30):
        for j in (i + 1, i + 3, i * 7 % 30):
            if j < 30 and j != i:
                db.create_edge(ids[i], ids[j], "USES", {})
    return db


def partition(db):
    groups = {}
    for node, community in db.algorithms.louvain()["communities"].items():
        groups.setdefault(community, set()).add(node)
    return frozenset(frozenset(group) for group in groups.values())


def test_louvain_is_deterministic():
    db = tied_moves_graph()
    assert len({partition(db) for _ in range(20)}) == 1


def test_louvain_community_ids_are_canonical():
    db = tied_moves_graph()
    first = db.algorithms.louvain()
    assert all(db.algorithms.louvain() == first for _ in range(5))

    # Communities are numbered 0, 1, 2, ... by their smallest node id.
    smallest = {}
    for node, community in first["communities"].items():
        smallest[community] = min(smallest.get(community, node), node)
    by_smallest_node = [c for c, _ in sorted(smallest.items(), key=lambda item: item[1])]
    assert by_smallest_node == list(range(len(smallest)))
    assert first["num_communities"] == len(smallest)


def test_louvain_numbers_interleaved_communities_by_smallest_node():
    # Two triangles whose members alternate in creation order, joined by one edge.
    db = grafeo.GrafeoDB()
    names = ["Alix", "Gus", "Vincent", "Jules", "Mia", "Butch"]
    ids = {name: db.create_node(["Person"], {"name": name}).id for name in names}
    for a, b in [
        ("Alix", "Vincent"),
        ("Vincent", "Mia"),
        ("Mia", "Alix"),
        ("Gus", "Jules"),
        ("Jules", "Butch"),
        ("Butch", "Gus"),
        ("Mia", "Butch"),
    ]:
        db.create_edge(ids[a], ids[b], "KNOWS", {})

    communities = db.algorithms.louvain()["communities"]
    assert {name: communities[node] for name, node in ids.items()} == {
        "Alix": 0,
        "Gus": 1,
        "Vincent": 0,
        "Jules": 1,
        "Mia": 0,
        "Butch": 1,
    }
