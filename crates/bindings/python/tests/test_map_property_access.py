"""Dotted access into map-valued properties: `n.meta.route` reads like `n.meta['route']`."""

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


def rows(db, query, cypher):
    result = db.execute_cypher(query) if cypher else db.execute(query)
    return [dict(row) for row in result]


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:A {id: 'a', meta: {route: 'directory', score: 0.5, nested: {level: 2}}})")
    db.execute("INSERT (:A {id: 'b', meta: {route: 'llm', score: 0.9}})")
    db.execute("INSERT (:A {id: 'c', meta: 'plain text'})")
    return db


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_reads_a_map_key(db, cypher):
    query = "MATCH (n:A) WHERE n.meta.route = 'directory' RETURN n.id AS id, n.meta.score AS score"
    assert rows(db, query, cypher) == [{"id": "a", "score": 0.5}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_to_a_missing_key_is_null(db, cypher):
    query = "MATCH (n:A) WHERE n.id = 'a' RETURN n.meta.missing AS m"
    assert rows(db, query, cypher) == [{"m": None}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_chains_into_nested_maps(db, cypher):
    query = "MATCH (n:A) WHERE n.meta.nested.level = 2 RETURN n.id AS id, n.meta.nested.level AS l"
    assert rows(db, query, cypher) == [{"id": "a", "l": 2}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_on_a_non_map_is_null_like_subscript(db, cypher):
    query = (
        "MATCH (n:A) WHERE n.id = 'c' RETURN n.meta.route AS dotted, n.meta['route'] AS subscript"
    )
    assert rows(db, query, cypher) == [{"dotted": None, "subscript": None}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_column_without_alias_is_named_as_written(db, cypher):
    query = "MATCH (n:A) WHERE n.id = 'a' RETURN n.meta.score, n.meta.nested.level"
    assert rows(db, query, cypher) == [{"n.meta.score": 0.5, "n.meta.nested.level": 2}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_on_a_node_expression_reads_the_node(db, cypher):
    db.execute("MATCH (a:A {id: 'a'}), (b:A {id: 'b'}) INSERT (a)-[:R]->(b)")
    query = "MATCH (:A)-[r:R]->(:A) RETURN startNode(r).id AS s, endNode(r).id AS e"
    assert rows(db, query, cypher) == [{"s": "a", "e": "b"}]


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_dotted_access_on_a_string_expression_is_an_error(db, cypher):
    with pytest.raises(Exception, match="not a map value"):
        rows(db, "MATCH (n:A) RETURN toUpper(n.id).route", cypher)
