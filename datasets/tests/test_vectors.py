"""The vector statements load, and a query vector finds the node the fixture expects."""

import json

from conftest import load_lines


def key_of(db, node_id):
    rows = list(db.execute(f"MATCH (n) WHERE id(n) = {node_id} RETURN n.key AS k"))
    return rows[0]["k"]


def test_vectors_load_and_index(db, derived, fixtures):
    load_lines(db, derived / "silicon-valley.vectors.gql", "execute")
    queries = json.loads((fixtures / "vector-queries.json").read_text(encoding="utf-8"))
    assert queries
    graph = json.loads((derived / "graph.json").read_text(encoding="utf-8"))
    labels = sorted({n["label"] for n in graph["nodes"]})
    assert labels
    for query in queries:
        best = None
        for label in labels:
            hits = db.vector_search(label, "embedding", query["vector"], 1)
            assert hits, f"{query['text']!r}: no hits from the {label} index"
            if best is None or hits[0][1] < best[1]:
                best = hits[0]
        key = key_of(db, best[0])
        assert key in query["expect"], (
            f"{query['text']!r}: got {key}, expected one of {query['expect']}"
        )
