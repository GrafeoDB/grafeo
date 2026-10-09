"""Nodes and edges stay nodes and edges inside lists, maps and paths.

A node or edge in a list or map literal, a whole path and what `startNode` and
`endNode` return come back as node and edge dicts (`_id`, `_labels` or `_type`,
the properties), as `RETURN n` gives them (openCypher; ISO/IEC 39075 list,
record and path values hold node and edge references). They used to come back
as bare IDs. A path keeps its dict shape, `{"nodes": [...], "edges": [...]}`.
"""

import grafeo
import pytest

HAS_CYPHER = hasattr(grafeo.GrafeoDB(), "execute_cypher")

LANGUAGES = [
    pytest.param(False, id="gql"),
    pytest.param(
        True,
        id="cypher",
        marks=pytest.mark.skipif(not HAS_CYPHER, reason="grafeo built without cypher feature"),
    ),
]

ALIX = (["Person"], {"name": "Alix", "age": 19})
GUS = (["Person"], {"name": "Gus", "age": 88})
KNOWS = ("KNOWS", {"w": 3})


def run(db, query, cypher):
    return db.execute_cypher(query) if cypher else db.execute(query)


def rows(db, query, cypher):
    return [dict(row) for row in run(db, query, cypher)]


def node(value):
    """The labels and properties of a node dict."""
    assert isinstance(value, dict), f"expected a node, got {value!r}"
    return (value["_labels"], {k: v for k, v in value.items() if not k.startswith("_")})


def edge(value):
    """The type and properties of an edge dict."""
    assert isinstance(value, dict), f"expected an edge, got {value!r}"
    return (value["_type"], {k: v for k, v in value.items() if not k.startswith("_")})


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 19})-[:KNOWS {w: 3}]->(:Person {name: 'Gus', age: 88})"
    )
    return db


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_list_literal_returns_its_node_and_edge(db, cypher):
    query = "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN [a, r, 3] AS l"
    [row] = rows(db, query, cypher)
    first, second, number = row["l"]
    assert node(first) == ALIX
    assert edge(second) == KNOWS
    assert number == 3


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_map_literal_returns_its_node(db, cypher):
    query = "MATCH (a:Person {name: 'Gus'}) WITH {msg: a, t: 19} AS x RETURN x, x.msg.name AS n"
    [row] = rows(db, query, cypher)
    assert node(row["x"]["msg"]) == GUS
    assert row["x"]["t"] == 19
    assert row["n"] == "Gus"


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_returned_path_holds_its_nodes_and_edges(db, cypher):
    query = (
        "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() "
        "RETURN p, nodes(p) AS ns, relationships(p) AS rs"
    )
    [row] = rows(db, query, cypher)
    path = row["p"]
    assert set(path) == {"nodes", "edges"}, "the path keeps its dict shape"
    assert [node(n) for n in path["nodes"]] == [ALIX, GUS]
    assert [edge(e) for e in path["edges"]] == [KNOWS]
    assert path["nodes"] == row["ns"], "the same nodes as nodes(p)"
    assert path["edges"] == row["rs"], "the same edges as relationships(p)"


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_startnode_and_endnode_return_nodes(db, cypher):
    query = "MATCH ()-[r:KNOWS]->() RETURN startNode(r) AS s, endNode(r) AS e"
    [row] = rows(db, query, cypher)
    assert node(row["s"]) == ALIX
    assert node(row["e"]) == GUS


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_the_result_lists_the_nodes_and_edges_inside_values(db, cypher):
    result = run(db, "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() RETURN p", cypher)
    assert sorted(n.properties()["name"] for n in result.nodes()) == ["Alix", "Gus"]
    assert [e.edge_type for e in result.edges()] == ["KNOWS"]
    result = run(db, "MATCH (a:Person {name: 'Gus'}) RETURN {k: [a]} AS m", cypher)
    assert [n.properties()["name"] for n in result.nodes()] == ["Gus"]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_size_of_a_string_counts_characters(db, cypher):
    [row] = rows(db, "RETURN size('\U0001f337') AS tulip, size('Plzeň') AS city", cypher)
    assert row == {"tulip": 1, "city": 5}
