"""Louvain merges communities level by level, not only by moving single nodes."""

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)


def path(n):
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["P"], {"i": i}).id for i in range(n)]
    for a, b in zip(ids, ids[1:], strict=False):
        db.create_edge(a, b, "NEXT", {})
    return db


def test_louvain_aggregates_communities_on_a_path():
    # One level of local moving pairs neighbours up: 500 communities at modularity 0.5.
    result = path(1000).algorithms.louvain()
    assert result["modularity"] > 0.9
    assert 10 <= result["num_communities"] <= 100


def test_louvain_keeps_two_joined_cliques_apart():
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["N"], {"i": i}).id for i in range(10)]
    for clique in (ids[:5], ids[5:]):
        for position, a in enumerate(clique):
            for b in clique[position + 1 :]:
                db.create_edge(a, b, "E", {})
    db.create_edge(ids[4], ids[5], "E", {})

    communities = db.algorithms.louvain()["communities"]
    assert [communities[node] for node in ids] == [0] * 5 + [1] * 5


def test_louvain_stays_deterministic_across_levels():
    assert path(1000).algorithms.louvain() == path(1000).algorithms.louvain()


def test_louvain_call_matches_the_python_api():
    db = path(200)
    expected = db.algorithms.louvain()
    rows = list(db.execute("CALL grafeo.louvain() YIELD node_id, community_id, modularity"))
    assert {row["node_id"]: row["community_id"] for row in rows} == expected["communities"]
    assert {row["modularity"] for row in rows} == {expected["modularity"]}


def test_louvain_resolution_changes_the_communities():
    db = path(200)
    coarse = db.algorithms.louvain(resolution=0.01)
    fine = db.algorithms.louvain(resolution=4.0)
    assert (
        coarse["num_communities"]
        < db.algorithms.louvain()["num_communities"]
        < fine["num_communities"]
    )
