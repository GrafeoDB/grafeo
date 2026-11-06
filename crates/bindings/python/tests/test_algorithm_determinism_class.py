"""Every algorithm bridge method gives the same result for the same graph (#592), keyed on the
bridge's own methods so a new one fails until it is covered.

- Two calls, and a second database built the same way, give equal results (floats to the bit).
- A result keyed by node (a dict) comes back in ascending node-id order.
- The topological order is the smallest by node id (Kahn's algorithm with a min-heap, as
  networkx's `lexicographical_topological_sort`), the same whatever order the edges were added in,
  and a graph with a cycle has none.

0.6.1's `topological_sort()` gave a different order on every call; `test_algorithm_determinism.py`
lists its methods by hand and had none for it.
"""

import heapq
import random

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

NODES = 80


def edges_of(seed=5):
    """A seeded DAG from lower to higher index, with distinct weights."""
    rng = random.Random(seed)
    found = [(u, v) for u in range(NODES) for v in range(u + 1, NODES) if rng.random() < 0.08]
    return [(u, v, i + 1) for i, (u, v) in enumerate(found)]


def build(edges_reversed=False):
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["V"], {"name": f"v{i}"}).id for i in range(NODES)]
    edges = edges_of()
    for u, v, w in reversed(edges) if edges_reversed else edges:
        db.create_edge(ids[u], ids[v], "E", {"w": w, "c": 1 + w % 5, "k": 1 + w % 4})
    return db, ids


def calls(ids):
    s, t = ids[0], ids[-1]
    return {
        "articulation_points": lambda a: a.articulation_points(),
        "astar": lambda a: a.astar(s, t, weight="w"),
        "bellman_ford": lambda a: a.bellman_ford(s, weight="w"),
        "betweenness_centrality": lambda a: a.betweenness_centrality(),
        "bfs": lambda a: a.bfs(s),
        "bfs_layers": lambda a: a.bfs_layers(s),
        "bridges": lambda a: a.bridges(),
        "closeness_centrality": lambda a: a.closeness_centrality(),
        "clustering_coefficient": lambda a: a.clustering_coefficient(),
        "connected_component_count": lambda a: a.connected_component_count(),
        "connected_components": lambda a: a.connected_components(),
        "degree_centrality": lambda a: a.degree_centrality(),
        "dfs": lambda a: a.dfs(s),
        "dfs_all": lambda a: a.dfs_all(),
        "dijkstra": lambda a: a.dijkstra(s, weight="w"),
        "floyd_warshall": lambda a: a.floyd_warshall(weight="w"),
        "global_clustering_coefficient": lambda a: a.global_clustering_coefficient(),
        "is_dag": lambda a: a.is_dag(),
        "kcore": lambda a: a.kcore(),
        "kruskal": lambda a: a.kruskal(weight="w"),
        "label_propagation": lambda a: a.label_propagation(),
        "local_clustering_coefficient": lambda a: a.local_clustering_coefficient(),
        "louvain": lambda a: a.louvain(),
        "max_flow": lambda a: a.max_flow(s, t, capacity="c"),
        "min_cost_max_flow": lambda a: a.min_cost_max_flow(s, t, capacity="c", cost="k"),
        "pagerank": lambda a: a.pagerank(),
        "prim": lambda a: a.prim(weight="w"),
        "sssp": lambda a: a.sssp("v0", weight_attr="w"),
        "strongly_connected_components": lambda a: a.strongly_connected_components(),
        "topological_sort": lambda a: a.topological_sort(),
        "total_triangles": lambda a: a.total_triangles(),
        "triangle_count": lambda a: a.triangle_count(),
    }


def bridge_methods():
    algorithms = grafeo.GrafeoDB().algorithms
    return sorted(
        name
        for name in dir(algorithms)
        if not name.startswith("_") and callable(getattr(algorithms, name))
    )


def comparable(value):
    """A result as text, dict order included: equal text is an equal result, floats to the bit."""
    return repr(value)


def test_every_bridge_method_has_a_call():
    assert sorted(calls([0, 1])) == bridge_methods()


@pytest.mark.parametrize("method", sorted(calls([0, 1])))
def test_a_method_answers_the_same_for_the_same_graph(method):
    db, ids = build()
    other, other_ids = build()
    assert ids == other_ids
    once = comparable(calls(ids)[method](db.algorithms))
    assert comparable(calls(ids)[method](db.algorithms)) == once, f"{method}: a second call"
    assert comparable(calls(ids)[method](other.algorithms)) == once, f"{method}: a second database"


@pytest.mark.parametrize("method", sorted(calls([0, 1])))
def test_a_result_keyed_by_node_comes_in_node_order(method):
    db, ids = build()
    result = calls(ids)[method](db.algorithms)
    if isinstance(result, dict) and result and all(isinstance(k, int) for k in result):
        keys = list(result)
        assert keys == sorted(keys), method


def smallest_order(edges):
    """Kahn's algorithm, the ready node with the smallest index first."""
    indegree = [0] * NODES
    out = [[] for _ in range(NODES)]
    for u, v, _ in edges:
        out[u].append(v)
        indegree[v] += 1
    ready = [v for v in range(NODES) if indegree[v] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        for v in out[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(ready, v)
    return order


@pytest.mark.parametrize("edges_reversed", [False, True])
def test_the_topological_order_is_the_smallest_by_node_id(edges_reversed):
    db, ids = build(edges_reversed)
    index = {node: i for i, node in enumerate(ids)}
    assert [index[n] for n in db.algorithms.topological_sort()] == smallest_order(edges_of())


def test_a_cycle_has_no_topological_order():
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["V"]).id for _ in range(3)]
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        db.create_edge(ids[u], ids[v], "E")
    assert db.algorithms.topological_sort() is None
