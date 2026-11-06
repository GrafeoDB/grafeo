"""Every algorithm bridge method stops at the database's query timeout, as `execute` does.

`db.algorithms.<method>()` enters the database's deadline (`query_timeout`) when it is called, and
the algorithm reads the clock on its first unit of work. With a timeout of 1 microsecond, the
snapshot and the setup before that unit always outlast it, so every method raises the timeout at
its first poll, on any machine; the same exception as `execute` past its timeout, after which the
database answers again. One realistic case runs betweenness (hundreds of milliseconds here) under
a 1 ms timeout. 0.6.1 had no deadline in the bridge: a method ran to its end whatever the timeout.
"""

import random

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

NODES = 2_000


def build(query_timeout):
    """A DAG of 2,000 nodes, ~6,000 edges from lower to higher index in a seeded order, each with a
    weight `w`, a capacity `c` and a cost `k`, all positive; node names `v0`, `v1`, ..."""
    rng = random.Random(17)
    db = grafeo.GrafeoDB(query_timeout=query_timeout)
    ids = [db.create_node(["V"], {"name": f"v{i}"}).id for i in range(NODES)]
    for _ in range(3 * NODES):
        u, v = rng.randrange(NODES), rng.randrange(NODES)
        if u != v:
            props = {"w": rng.randint(1, 9), "c": rng.randint(1, 5), "k": rng.randint(1, 4)}
            db.create_edge(ids[min(u, v)], ids[max(u, v)], "E", props)
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
    # The database answers the next statement.
    assert [list(r.values()) for r in db.execute("RETURN 1 AS one")] == [[1]]


def test_betweenness_stops_at_a_one_millisecond_timeout():
    db, _ = build(0.001)
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.algorithms.betweenness_centrality()
    assert "timeout" in str(raised.value).lower(), str(raised.value)


def test_without_a_timeout_every_method_answers():
    db, ids = build(None)
    for method, call in calls(ids).items():
        try:
            call(db.algorithms)
        except grafeo.GrafeoError as error:
            pytest.fail(f"{method} without a timeout: {error}")
