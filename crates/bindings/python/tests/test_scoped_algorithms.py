"""Graph algorithms on the selected graph or a projection, and undirected PageRank (#566)."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)


def path_d_a_b_c():
    """The path d - a - b - c, created as a -> b, b -> c, a -> d."""
    db = grafeo.GrafeoDB()
    ids = {n: db.create_node(["Graph"], {"n": n}).id for n in "abcd"}
    for source, target in [("a", "b"), ("b", "c"), ("a", "d")]:
        db.create_edge(ids[source], ids[target], "R", {})
    return db, {node: n for n, node in ids.items()}


def test_undirected_pagerank_ranks_by_connectivity_not_direction():
    db, name = path_d_a_b_c()
    scores = {name[node]: s for node, s in db.algorithms.pagerank(directed=False).items()}
    assert abs(scores["a"] - scores["b"]) < 1e-12
    assert abs(scores["c"] - scores["d"]) < 1e-12
    assert scores["a"] > scores["c"]


def test_directed_pagerank_stays_the_default():
    db, _ = path_d_a_b_c()
    assert db.algorithms.pagerank() == db.algorithms.pagerank(directed=True)
    assert db.algorithms.pagerank() != db.algorithms.pagerank(directed=False)


def test_undirected_pagerank_counts_parallel_edges_once():
    def scores(extra_edge):
        db = grafeo.GrafeoDB()
        ids = [db.create_node(["Graph"], {"n": n}).id for n in range(3)]
        db.create_edge(ids[0], ids[1], "CONTAINS", {})
        db.create_edge(ids[1], ids[2], "CALLS", {})
        if extra_edge:
            db.create_edge(ids[0], ids[1], "IMPORTS", {})
            db.create_edge(ids[1], ids[0], "USES", {})
        name = {node: n for n, node in enumerate(ids)}
        return {name[node]: s for node, s in db.algorithms.pagerank(directed=False).items()}

    assert scores(extra_edge=True) == scores(extra_edge=False)


def test_networkx_undirected_pagerank_matches():
    db, _ = path_d_a_b_c()
    assert db.as_networkx(directed=False).pagerank() == db.algorithms.pagerank(directed=False)
    assert db.as_networkx(directed=True).pagerank() == db.algorithms.pagerank()


def test_call_undirected_pagerank_matches():
    db, _ = path_d_a_b_c()
    rows = db.execute("CALL grafeo.pagerank({directed: false}) YIELD node_id, score")
    assert {row["node_id"]: row["score"] for row in rows} == db.algorithms.pagerank(directed=False)


# ---------------------------------------------------------------------------
# Scope: projections, the selected graph and graph handles
# ---------------------------------------------------------------------------

ALGORITHMS = ["pagerank", "louvain", "kcore", "articulation_points", "degree_centrality"]


def two_namespaces():
    """An extraction graph (two triangles of Graph nodes, joined) next to a Model namespace."""
    db = grafeo.GrafeoDB()
    graph = [db.create_node(["Graph"], {"id": f"g{i}"}).id for i in range(6)]
    model = [db.create_node(["Model"], {"id": f"m{i}"}).id for i in range(3)]
    for a, b in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]:
        db.create_edge(graph[a], graph[b], "REL", {})
    db.create_edge(model[0], model[1], "SERVES", {})
    db.create_edge(model[1], graph[0], "DESCRIBES", {})
    db.create_projection("extraction", node_labels=["Graph"])
    return db, set(graph)


def scored_nodes(name, result):
    if name == "louvain":
        return set(result["communities"])
    if name == "kcore":
        return set(result["core_numbers"])
    return set(result)


@pytest.mark.parametrize("name", ALGORITHMS)
def test_a_projection_scores_only_its_nodes(name):
    db, graph_nodes = two_namespaces()
    nodes = scored_nodes(name, getattr(db.algorithms, name)(projection="extraction"))
    if name == "articulation_points":
        assert nodes <= graph_nodes
    else:
        assert nodes == graph_nodes


def test_articulation_points_of_a_projection_ignore_other_nodes():
    # Inside the projection the edge g2 - g3 joins the triangles: g2 and g3 cut it.
    db, _ = two_namespaces()
    ids = {r["i"]: r["k"] for r in db.execute("MATCH (n:Graph) RETURN id(n) AS i, n.id AS k")}
    points = db.algorithms.articulation_points(projection="extraction")
    assert {ids[node] for node in points} == {"g2", "g3"}


def test_an_unknown_projection_raises():
    db, _ = two_namespaces()
    with pytest.raises(grafeo.GrafeoError, match="Projection 'nope' does not exist"):
        db.algorithms.pagerank(projection="nope")


def test_algorithms_follow_set_graph_like_call_and_a_graph_handle():
    db = grafeo.GrafeoDB()
    db.create_node(["Person"], {"name": "Alix"})
    db.create_graph("g")
    handle = db.graph("g")
    gus = handle.create_node(["Person"], {"name": "Gus"}).id
    vincent = handle.create_node(["Person"], {"name": "Vincent"}).id
    handle.create_edge(gus, vincent, "KNOWS", {})
    assert set(handle.algorithms.pagerank()) == {gus, vincent}
    db.set_graph("g")
    via_call = {row["node_id"] for row in db.execute("CALL grafeo.pagerank() YIELD node_id")}
    assert set(db.algorithms.pagerank()) == via_call == {gus, vincent}
    db.reset_graph()
    assert len(db.algorithms.pagerank()) == 1


def test_a_selected_graph_dropped_elsewhere_raises():
    db = grafeo.GrafeoDB()
    db.create_node(["Person"], {"name": "Alix"})
    db.create_graph("g")
    db.set_graph("g")
    # Dropped by a transaction: the database's own selection still names g.
    with db.begin_transaction() as tx:
        tx.execute("DROP GRAPH g")
        tx.commit()
    assert db.current_graph() == "g"
    with pytest.raises(grafeo.GrafeoError, match="Graph 'g' does not exist"):
        db.algorithms.pagerank()
    with pytest.raises(grafeo.GrafeoError, match="Graph 'g' does not exist"):
        db.create_projection("p")


def test_a_projection_wins_over_a_graph_handle():
    db, graph_nodes = two_namespaces()
    db.create_graph("other")
    assert set(db.graph("other").algorithms.pagerank(projection="extraction")) == graph_nodes


def test_a_projection_keeps_its_graph_after_set_graph():
    db, graph_nodes = two_namespaces()
    db.create_graph("other")
    db.set_graph("other")
    assert set(db.algorithms.pagerank(projection="extraction")) == graph_nodes


def test_create_projection_follows_set_graph():
    db = grafeo.GrafeoDB()
    db.create_node(["Graph"], {"id": "default"})
    db.create_graph("g")
    handle = db.graph("g")
    inside = {handle.create_node(["Graph"], {"id": f"g{i}"}).id for i in range(2)}
    db.set_graph("g")
    assert db.create_projection("p", node_labels=["Graph"]) is True
    assert db.create_projection("p", node_labels=["Graph"]) is False
    db.reset_graph()
    assert set(db.algorithms.pagerank(projection="p")) == inside


def test_call_with_a_projection_matches_the_python_api():
    db, graph_nodes = two_namespaces()
    query = "CALL grafeo.pagerank({projection: 'extraction', directed: false}) YIELD node_id, score"
    rows = {row["node_id"]: row["score"] for row in db.execute(query)}
    assert set(rows) == graph_nodes
    assert rows == db.algorithms.pagerank(projection="extraction", directed=False)
