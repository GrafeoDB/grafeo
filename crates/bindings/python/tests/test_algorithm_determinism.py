"""Graph algorithm results come back in node-id order, the same in every process (#592)."""

import json
import os
import subprocess
import sys

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

# Builds one graph and prints every result as ordered (key, value) lists.
SCRIPT = r"""
import json
import grafeo

db = grafeo.GrafeoDB()
ids = [db.create_node(["N"], {"i": i}).id for i in range(60)]
for i in range(60):
    for j in (i + 1, i + 7, (i * 13) % 60):
        if j < 60 and j != i:
            db.create_edge(ids[i], ids[j], "E", {})
a = db.algorithms
clustering = a.clustering_coefficient()
out = {
    "pagerank": list(a.pagerank().items()),
    "pagerank_undirected": list(a.pagerank(directed=False).items()),
    "betweenness": list(a.betweenness_centrality().items()),
    "closeness": list(a.closeness_centrality().items()),
    "degree": [[k, sorted(v.items())] for k, v in a.degree_centrality().items()],
    "label_propagation": list(a.label_propagation().items()),
    "louvain": list(a.louvain()["communities"].items()),
    "kcore": list(a.kcore()["core_numbers"].items()),
    "components": list(a.connected_components().items()),
    "triangles": list(a.triangle_count().items()),
    "local_clustering": list(a.local_clustering_coefficient().items()),
    "clustering": list(clustering["coefficients"].items()),
    "global_clustering": a.global_clustering_coefficient(),
    "articulation_points": a.articulation_points(),
    "bridges": a.bridges(),
}
print(json.dumps(out))
"""


def results(seed):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def two_processes():
    return results("1"), results("2")


def test_results_are_identical_in_other_processes(two_processes):
    first, second = two_processes
    for name in first:
        assert first[name] == second[name], name


def test_dicts_and_lists_come_in_node_id_order(two_processes):
    first, _ = two_processes
    for name, value in first.items():
        if name == "global_clustering":
            continue
        keys = [item[0] if isinstance(item, list) else item for item in value]
        assert keys == sorted(keys), name


def test_communities_are_numbered_by_their_smallest_node():
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["N"], {}).id for _ in range(6)]
    for a, b in [(0, 2), (2, 4), (4, 0), (1, 3), (3, 5), (5, 1)]:
        db.create_edge(ids[a], ids[b], "E", {})
    for _ in range(5):
        labels = db.algorithms.label_propagation()
        assert [labels[node] for node in ids] == [0, 1, 0, 1, 0, 1]
