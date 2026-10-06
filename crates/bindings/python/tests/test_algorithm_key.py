"""Algorithm results keyed by a node property, and run in its order (#566 `key=`)."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

# Two triangles joined by a bridge, with a tail: every algorithm has something
# to say (communities, an articulation point, cores, triangles).
EDGES = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3), (5, 6)]
NODES = 7


def key(i):
    return f"n{i:02d}"


def build(reverse=False):
    """The graph, nodes and edges inserted in order or both reversed, each node
    with its key in `id`. Returns the database and its node id to key map."""
    db = grafeo.GrafeoDB()
    order = range(NODES - 1, -1, -1) if reverse else range(NODES)
    ids = {}
    for i in order:
        ids[i] = db.create_node(["Graph"], {"id": key(i)}).id
    for source, target in reversed(EDGES) if reverse else EDGES:
        db.create_edge(ids[source], ids[target], "REL", {})
    return db, {node: key(i) for i, node in ids.items()}


# Every method with key=, and how its result is shaped: a dict per node, a list
# of nodes, or a dict with an inner per-node map.
PER_NODE = [
    ("pagerank", {}),
    ("pagerank", {"directed": False}),
    ("degree_centrality", {}),
    ("degree_centrality", {"normalized": True}),
    ("betweenness_centrality", {}),
    ("closeness_centrality", {}),
    ("label_propagation", {}),
    ("connected_components", {}),
    ("triangle_count", {}),
    ("local_clustering_coefficient", {}),
]


@pytest.mark.parametrize(("name", "kwargs"), PER_NODE)
def test_key_maps_a_per_node_result(name, kwargs):
    """With keys that sort like the node ids, key= gives the plain result with
    each node id replaced by its key, in key order."""
    db, keys = build()
    method = getattr(db.algorithms, name)
    plain = method(**kwargs)
    keyed = method(key="id", **kwargs)
    assert keyed == {keys[node]: value for node, value in plain.items()}
    assert list(keyed) == sorted(keyed)


def test_key_maps_louvain():
    db, keys = build()
    plain = db.algorithms.louvain()
    keyed = db.algorithms.louvain(key="id")
    assert keyed["communities"] == {keys[n]: c for n, c in plain["communities"].items()}
    assert keyed["modularity"] == plain["modularity"]
    assert keyed["num_communities"] == plain["num_communities"]


def test_key_maps_kcore_and_articulation_points():
    db, keys = build()
    plain = db.algorithms.kcore()
    keyed = db.algorithms.kcore(key="id")
    assert keyed["core_numbers"] == {keys[n]: c for n, c in plain["core_numbers"].items()}
    assert keyed["max_core"] == plain["max_core"]
    assert db.algorithms.kcore(2, key="id") == [keys[n] for n in db.algorithms.kcore(2)]
    points = db.algorithms.articulation_points(key="id")
    assert points == [keys[n] for n in db.algorithms.articulation_points()]
    assert points == sorted(points) and points


@pytest.mark.parametrize("directed", [True, False])
def test_pagerank_by_key_is_the_same_for_any_insertion_order(directed):
    forward, _ = build()
    backward, _ = build(reverse=True)
    assert forward.algorithms.pagerank(directed=directed, key="id") == backward.algorithms.pagerank(
        directed=directed, key="id"
    )


def test_communities_by_key_are_the_same_for_any_insertion_order():
    forward, _ = build()
    backward, _ = build(reverse=True)
    assert forward.algorithms.louvain(key="id") == backward.algorithms.louvain(key="id")
    assert forward.algorithms.label_propagation(key="id") == backward.algorithms.label_propagation(
        key="id"
    )


def test_a_node_without_the_key_raises():
    db, _ = build()
    lone = db.create_node(["Graph"], {}).id
    with pytest.raises(grafeo.GrafeoError, match=rf"node {lone} has no value for the key 'id'"):
        db.algorithms.pagerank(key="id")


def test_a_repeated_key_raises():
    db, _ = build()
    db.create_node(["Graph"], {"id": key(3)})
    with pytest.raises(grafeo.GrafeoError, match=r"have the same value for the key 'id': \"n03\""):
        db.algorithms.louvain(key="id")


def test_only_the_nodes_in_scope_need_a_key():
    """A projection's nodes need the key; nodes outside it do not."""
    db, keys = build()
    db.create_node(["Model"], {})
    db.create_projection("extraction", node_labels=["Graph"])
    keyed = db.algorithms.pagerank(projection="extraction", key="id")
    assert set(keyed) == set(keys.values())
    with pytest.raises(grafeo.GrafeoError):
        db.algorithms.pagerank(key="id")
