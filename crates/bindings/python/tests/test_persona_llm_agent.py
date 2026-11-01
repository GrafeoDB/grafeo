"""Persona: an LLM agent drives Grafeo, as an MCP server or grafeo-langchain does.

It reads schema() into its prompt, generates GQL from it, and retries a query
that fails, guided by the error alone. It needs schema() to match the data,
each error to name what to change (with a GQLSTATUS to branch on), and a
malformed generated query never to take the host down: the database answers
the next query.
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (a:Person {name: 'Alix', age: 30})-[:KNOWS {since: 2020}]->"
        "(g:Person {name: 'Gus', age: 25}), "
        "(a)-[:LIVES_IN]->(:City {name: 'Amsterdam', population: 900000}), "
        "(g)-[:LIVES_IN]->(:City {name: 'Berlin'})"
    )
    return db


def answers(db):
    """The database still runs a query after a failed one."""
    return db.execute("MATCH (p:Person) RETURN count(p) AS c").to_list() == [{"c": 2}]


# ---------------------------------------------------------------------------
# schema() in the prompt
# ---------------------------------------------------------------------------


def test_persona_llm_agent_schema_matches_the_data(db):
    schema = db.schema()
    assert schema["mode"] == "lpg"
    assert {entry["name"]: entry["count"] for entry in schema["labels"]} == {
        "Person": 2,
        "City": 2,
    }
    assert {entry["name"]: entry["count"] for entry in schema["edge_types"]} == {
        "KNOWS": 1,
        "LIVES_IN": 2,
    }
    assert sorted(schema["property_keys"]) == ["age", "name", "population", "since"]


def test_persona_llm_agent_queries_built_from_the_schema_agree_with_it(db):
    schema = db.schema()
    for entry in schema["labels"]:
        query = f"MATCH (n:{entry['name']}) RETURN count(n) AS c"
        assert db.execute(query).to_list() == [{"c": entry["count"]}], query
    for entry in schema["edge_types"]:
        query = f"MATCH ()-[r:{entry['name']}]->() RETURN count(r) AS c"
        assert db.execute(query).to_list() == [{"c": entry["count"]}], query
    for key in schema["property_keys"]:
        query = f"MATCH (n) WHERE n.{key} IS NOT NULL RETURN count(n) AS c"
        nodes = db.execute(query).to_list()[0]["c"]
        query = f"MATCH ()-[r]->() WHERE r.{key} IS NOT NULL RETURN count(r) AS c"
        edges = db.execute(query).to_list()[0]["c"]
        assert nodes + edges > 0, f"{key} is listed but no element holds it"


def test_persona_llm_agent_the_procedure_listing_names_its_columns(db):
    listing = db.execute("CALL grafeo.procedures()").to_list()
    pagerank = next(row for row in listing if row["name"] == "grafeo.pagerank")
    columns = [column.strip() for column in pagerank["output_columns"].split(",")]
    query = f"CALL grafeo.pagerank() YIELD {', '.join(columns)} RETURN {', '.join(columns)}"
    rows = db.execute(query).to_list()
    assert len(rows) == 4
    assert set(rows[0]) == set(columns)


# ---------------------------------------------------------------------------
# Errors a retry can act on
# ---------------------------------------------------------------------------

# (generated query, params, words the error must hold, GQLSTATUS, the retry)
RETRIES = [
    (
        "MATCH (p:Person) RETURN p.name; MATCH (c:City) RETURN c.name",
        None,
        ["multiple statements", "run them separately"],
        "42001",
        "MATCH (p:Person) RETURN p.name",
    ),
    (
        "MATCH (p:Person) WHERE p.name = $name RETURN p.age",
        None,
        ["Missing parameter: $name"],
        None,
        "MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age",
    ),
    (
        "MATCH (p:Person) RETURN p.name AS name, p.age AS name",
        None,
        ["duplicate column name 'name'", "distinct aliases"],
        None,
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age",
    ),
    (
        "MATCH (p:Person) RETURN p.name LIMIT -1",
        None,
        ["must be non-negative"],
        None,
        "MATCH (p:Person) RETURN p.name LIMIT 1",
    ),
    (
        "MATCH (p:Person) RETURN p.name LIMIT 'ten'",
        None,
        ["must be an integer"],
        None,
        "MATCH (p:Person) RETURN p.name LIMIT 10",
    ),
    (
        "MATCH (p:Person) RETURN unknown_fn(p.name)",
        None,
        ["Unknown function 'unknown_fn'"],
        None,
        "MATCH (p:Person) RETURN upper(p.name)",
    ),
    (
        "CALL grafeo.pagerank() YIELD nodeId, score RETURN nodeId",
        None,
        ["'nodeId'", "it has: node_id, score"],
        None,
        "CALL grafeo.pagerank() YIELD node_id, score RETURN node_id",
    ),
    (
        "CALL grafeo.page_rank()",
        None,
        ["unknown procedure 'grafeo.page_rank'", "grafeo.procedures()"],
        None,
        "CALL grafeo.pagerank()",
    ),
    (
        "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC NULLS",
        None,
        ["Expected FIRST or LAST after NULLS"],
        "42001",
        "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC NULLS LAST",
    ),
    (
        "MATCH (p:Person) RETURN p.name ORDER BY",
        None,
        ["Expected expression"],
        "42001",
        "MATCH (p:Person) RETURN p.name ORDER BY p.name",
    ),
    (
        "MATCH (p:Person RETURN p.name",
        None,
        ["Expected RParen"],
        "42001",
        "MATCH (p:Person) RETURN p.name",
    ),
    (
        "RETURN 1 / 0 AS x",
        None,
        ["division by zero"],
        "22012",
        "RETURN 1 / 1 AS x",
    ),
    (
        "RETURN 9223372036854775807 + 1 AS x",
        None,
        ["out of range"],
        "22003",
        "RETURN 9223372036854775806 + 1 AS x",
    ),
]


@pytest.mark.parametrize("query, params, words, status, retry", RETRIES)
def test_persona_llm_agent_retries_from_the_error(db, query, params, words, status, retry):
    with pytest.raises(grafeo.GrafeoError) as caught:
        db.execute(query, params) if params else db.execute(query)
    message = str(caught.value)
    for word in words:
        assert word in message, f"{query}: {message}"
    if status is not None:
        assert caught.value.gql_status == status, message
    assert message.startswith("GRAFEO-"), message
    db.execute(retry)
    assert answers(db)


def test_persona_llm_agent_deletes_with_the_advice_of_the_error(db):
    with pytest.raises(grafeo.GrafeoError, match="Use DETACH DELETE"):
        db.execute("MATCH (c:City {name: 'Berlin'})<-[:LIVES_IN]-(p) DELETE p")
    assert answers(db), "the failed delete removed nothing"
    db.execute("MATCH (c:City {name: 'Berlin'}) DETACH DELETE c")
    assert db.execute("MATCH (c:City) RETURN count(c) AS c").to_list() == [{"c": 1}]


# ---------------------------------------------------------------------------
# Malformed generated text never takes the host down
# ---------------------------------------------------------------------------

MALFORMED = [
    "",
    "   ",
    "-- just a comment",
    "```gql\nMATCH (p:Person) RETURN p.name\n```",
    "​MATCH (p:Person) RETURN p.name",
    "MATCH (p:Person) RETURN p.name" + chr(0),
    "MATCH (p:Person) RETURN 'unterminated",
    "MATCH (p:Person) RETURN p.name AS `na`me`",
    "SELECT name FROM Person WHERE age > 20",
    "MATCH (p:Person) WHERE p.name == 'Alix' RETURN p.age",
    "MATCH (p:Person) WHERE p.age > 20 OR RETURN p",
    "MATCH (p:Person) RETURN p {.*}",
    "RETURN " + "(" * 5000 + "1" + ")" * 5000,
    "RETURN " + " + ".join(["1"] * 20000),
    "MATCH " + "-".join(["(a)"] * 3000) + " RETURN a",
    "RETURN [" + ",".join(["["] * 3000) + "]",
]


@pytest.mark.parametrize("query", MALFORMED, ids=range(len(MALFORMED)))
def test_persona_llm_agent_a_malformed_query_is_an_error_and_the_database_answers(db, query):
    with pytest.raises(grafeo.GrafeoError) as caught:
        db.execute(query)
    assert str(caught.value).startswith("GRAFEO-"), str(caught.value)[:200]
    assert caught.value.gql_status, str(caught.value)[:200]
    assert answers(db)


def test_persona_llm_agent_a_huge_literal_is_fine(db):
    big = "x" * 1_000_000
    assert db.execute(f"RETURN '{big}' AS big").to_list() == [{"big": big}]
    assert answers(db)


def test_persona_llm_agent_a_trailing_semicolon_is_fine(db):
    assert db.execute("MATCH (p:Person) RETURN p.name AS n ORDER BY n;").to_list() == [
        {"n": "Alix"},
        {"n": "Gus"},
    ]
