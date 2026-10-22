"""Nodes and edges in results carry their values (#268, #335), and search ties
come back in one order (#619).

#268: `RETURN a, r, b` gave the edge as a bare integer; an edge now comes
back as a dict naming its type, its source and its target by node id, the
stored ones whichever way the pattern reads it. #335: `RETURN n ORDER BY
n.title LIMIT 1` gave an integer instead of the node. #619: documents with
the same score came back in a different order on each call; the lowest node
ids come first, also after a reopen, as the reporter's persistent store had
it.
"""

import pytest
from grafeo import GrafeoDB


@pytest.fixture
def knows():
    db = GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})")
    return db


@pytest.mark.parametrize(
    ("language", "query"),
    [
        ("gql", "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN a, r, b"),
        ("gql", "MATCH (b)<-[r:KNOWS]-(a:Person {name: 'Alix'}) RETURN a, r, b"),
        ("cypher", "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN a, r, b"),
        ("cypher", "MATCH (b)<-[r:KNOWS]-(a:Person {name: 'Alix'}) RETURN a, r, b"),
    ],
)
def test_an_edge_names_its_type_and_endpoints(knows, language, query):
    run = knows.execute if language == "gql" else knows.execute_cypher
    rows = list(run(query))
    assert len(rows) == 1
    a, r, b = rows[0]["a"], rows[0]["r"], rows[0]["b"]
    assert (a["name"], a["_labels"]) == ("Alix", ["Person"])
    assert (b["name"], b["_labels"]) == ("Gus", ["Person"])
    assert (r["_type"], r["since"]) == ("KNOWS", 2020)
    assert (r["_source"], r["_target"]) == (a["_id"], b["_id"]), "the stored direction"


@pytest.mark.parametrize("language", ["gql", "cypher"])
def test_order_by_limit_returns_the_node(language):
    db = GrafeoDB()
    db.execute(
        "INSERT (:Article {title: 'A2'}), (:Article {title: 'A1'}), (:Article {title: 'A3'})"
    )
    run = db.execute if language == "gql" else db.execute_cypher
    rows = list(run("MATCH (n:Article) RETURN n ORDER BY n.title LIMIT 2"))
    assert [(row["n"]["title"], row["n"]["_labels"]) for row in rows] == [
        ("A1", ["Article"]),
        ("A2", ["Article"]),
    ]


def tied_documents(db):
    """Three documents that score the same for "canals", and one that does not match."""
    for title in ["a", "b", "c"]:
        db.execute(f"INSERT (:Doc {{t: '{title}', body: 'graph canals'}})")
    db.execute("INSERT (:Doc {t: 'd', body: 'bridges'})")
    db.create_text_index("Doc", "body")


def titles(db, ids):
    return [
        next(iter(db.execute(f"MATCH (d:Doc) WHERE id(d) = {node_id} RETURN d.t AS t")))["t"]
        for node_id in ids
    ]


def test_tied_text_scores_keep_the_lowest_node_ids(tmp_path):
    path = str(tmp_path / "docs.grafeo")
    db = GrafeoDB(path)
    tied_documents(db)
    first = [node_id for node_id, _ in db.text_search("Doc", "body", "canals", 2)]
    assert titles(db, first) == ["a", "b"]
    for _ in range(10):
        assert [node_id for node_id, _ in db.text_search("Doc", "body", "canals", 2)] == first
    db.close()

    for reopen in (GrafeoDB, GrafeoDB.open_read_only):
        db = reopen(path)
        for _ in range(10):
            assert [node_id for node_id, _ in db.text_search("Doc", "body", "canals", 2)] == first
        db.close()
