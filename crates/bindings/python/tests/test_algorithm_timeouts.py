"""Every algorithm bridge method stops at the database's query timeout, as `execute` does.

`db.algorithms.<method>()` enters the database's deadline (`query_timeout`) when it is called, and
the algorithm reads the clock on its first unit of work and every 1,024th after it. A fast machine
can reach the first read within a microsecond, so the graph makes every method do more than 1,024
units: a chain `v0 -> v1 -> ... -> v1999` under the random edges lets each method that starts at
`v0` reach every node and the last one (a breadth-first search does about 10,000 units, a shortest
path to the last node about 2,900). The second read then comes long after a microsecond, and every
method raises the timeout on any machine; the same exception as `execute` past its timeout, after
which the database answers again. One realistic case runs betweenness (hundreds of milliseconds
here) under a 1 ms timeout. 0.6.1 had no deadline in the bridge: a method ran to its end whatever
the timeout.
"""

import math
import random

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

NODES = 2_000
# The graph every method runs to its end on: Floyd-Warshall alone is cubic in the nodes.
SMALL = 120


def build(query_timeout, nodes=NODES):
    """A DAG of `nodes` nodes, about three edges per node from lower to higher index in a seeded
    order, and a chain from each node to the next, each edge with a weight `w`, a capacity `c` and
    a cost `k`, all positive; node names `v0`, `v1`, ..."""
    rng = random.Random(17)
    db = grafeo.GrafeoDB(query_timeout=query_timeout)
    ids = [db.create_node(["V"], {"name": f"v{i}"}).id for i in range(nodes)]
    for _ in range(3 * nodes):
        u, v = rng.randrange(nodes), rng.randrange(nodes)
        if u != v:
            props = {"w": rng.randint(1, 9), "c": rng.randint(1, 5), "k": rng.randint(1, 4)}
            db.create_edge(ids[min(u, v)], ids[max(u, v)], "E", props)
    # Without it `v0` reaches 18 nodes in 35 units of a breadth-first search, under the clock's
    # second read, and the last node not at all.
    for u, v in zip(ids, ids[1:], strict=False):
        db.create_edge(u, v, "E", {"w": 9, "c": 5, "k": 4})
    return db, ids


def calls(ids):
    """One call of each bridge method, by name."""
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
        "ktruss": lambda a: a.ktruss(),
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
        "subgraph_isomorphism_count": lambda a: a.subgraph_isomorphism_count(
            [(0, 1), (1, 2), (2, 0)], 3
        ),
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


@pytest.fixture(scope="module")
def timed_out():
    """The graph under a 1 microsecond timeout, and the error `execute` raises past it."""
    db, ids = build(1e-6)
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute("MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*)")
    return db, ids, raised.value


def test_every_bridge_method_has_a_call():
    assert sorted(calls([0, 1])) == bridge_methods()


@pytest.mark.parametrize("method", sorted(calls([0, 1])))
def test_a_bridge_method_stops_at_the_timeout(method, timed_out):
    db, ids, reference = timed_out
    with pytest.raises(grafeo.GrafeoError) as raised:
        calls(ids)[method](db.algorithms)
    assert type(raised.value) is type(reference), method
    assert raised.value.gql_status == reference.gql_status, (method, str(raised.value))
    assert "timeout" in str(raised.value).lower(), (method, str(raised.value))
    # The database answers the next statement once it may take more than a microsecond: no
    # statement meets a 1 microsecond deadline, whose clock starts before the setup of its run.
    answer = db.execute("RETURN 1 AS one", timeout=math.inf)
    assert [list(r.values()) for r in answer] == [[1]]


def test_betweenness_stops_at_a_one_millisecond_timeout():
    db, _ = build(0.001)
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.algorithms.betweenness_centrality()
    assert "timeout" in str(raised.value).lower(), str(raised.value)


def test_without_a_timeout_every_method_answers():
    db, ids = build(None, SMALL)
    for method, call in calls(ids).items():
        try:
            call(db.algorithms)
        except grafeo.GrafeoError as error:
            pytest.fail(f"{method} without a timeout: {error}")
