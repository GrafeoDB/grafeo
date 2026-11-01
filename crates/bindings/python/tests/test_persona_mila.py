"""Persona: Mila moves a Python app from Neo4j and Cypher to GQL.

She starts from the "Coming from Cypher" page (docs/user-guide/gql/coming-from-cypher.md),
keeps her Cypher habits in GQL, and runs a shadow comparison: each query in
both languages over the same data, where the rows must agree except where the
page says GQL differs (walks against trails, exact decimals, sum() of no
values, zoned against local datetimes). Her app is shaped like a code graph:
every node `:Graph` plus a kind, selectors with an OR of labels and EXISTS.
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
    """Alix, Gus and Vincent know each other in a cycle; Alix likes a message."""
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (a:Person {name: 'Alix', age: 30, city: 'Amsterdam', score: 0.77, hobbies: "
        "['chess', 'cycling']})"
        "-[:KNOWS {since: 2020}]->(g:Person {name: 'Gus', age: 25, city: 'Berlin', score: 0.5, "
        "hobbies: []}), "
        "(g)-[:KNOWS {since: 2021}]->(v:Person {name: 'Vincent', age: 41, city: 'Paris', score: "
        "0.9}), "
        "(v)-[:KNOWS {since: 2022}]->(a), "
        "(m:Message {text: 'hi'})<-[:LIKES {creationDate: 2024}]-(a)"
    )
    return db


def rows(result):
    return [dict(row) for row in result]


def values(result):
    """The rows as sorted tuples of their values, column names aside."""
    return sorted(tuple(row.values()) for row in rows(result))


# ---------------------------------------------------------------------------
# Path mode: GQL walks, Cypher trails
# ---------------------------------------------------------------------------


def test_persona_mila_trail_gives_the_rows_cypher_gives(db):
    cypher = db.execute_cypher(
        "MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..2]-(b) RETURN b.name AS b"
    )
    trail = db.execute("MATCH TRAIL (a:Person {name: 'Alix'})-[:KNOWS]-{1,2}(b) RETURN b.name AS b")
    assert values(trail) == values(cypher) == [("Gus",), ("Gus",), ("Vincent",), ("Vincent",)]


def test_persona_mila_a_walk_may_come_back_along_its_edge(db):
    walk = db.execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS]-{1,2}(b) RETURN b.name AS b")
    # The trail's four rows, and Alix twice: out and back along one edge.
    assert values(walk) == [("Alix",), ("Alix",), ("Gus",), ("Gus",), ("Vincent",), ("Vincent",)]


def test_persona_mila_the_cypher_quantifier_runs_in_gql(db):
    want = [("Alix",), ("Gus",), ("Vincent",)]
    for pattern in ["-[:KNOWS*1..3]->", "-[:KNOWS]->{1,3}", "-[:KNOWS*]->", "-[:KNOWS]->{1,}"]:
        result = db.execute(
            f"MATCH (a:Person {{name: 'Alix'}}){pattern}(b) RETURN DISTINCT b.name AS b"
        )
        assert values(result) == want, pattern


def test_persona_mila_a_path_length_in_both(db):
    gql = db.execute(
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,2}(b) RETURN length(p) AS l, b.name AS "
        "b ORDER BY l"
    )
    cypher = db.execute_cypher(
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b) RETURN length(p) AS l, b.name AS "
        "b ORDER BY l"
    )
    assert rows(gql) == rows(cypher) == [{"l": 1, "b": "Gus"}, {"l": 2, "b": "Vincent"}]


# ---------------------------------------------------------------------------
# Numbers with a point
# ---------------------------------------------------------------------------


def test_persona_mila_a_number_with_a_point_is_exact_in_gql(db):
    assert rows(db.execute("RETURN 0.1 + 0.2 = 0.3 AS exact")) == [{"exact": True}]
    assert rows(db.execute_cypher("RETURN 0.1 + 0.2 = 0.3 AS exact")) == [{"exact": False}]


def test_persona_mila_a_float_property_keeps_the_same_rows_in_both(db):
    for cutoff in ["0.77", "0.5", "0.9", "0.1"]:
        query = f"MATCH (n:Person) WHERE n.score > {cutoff} RETURN n.name AS n"
        assert values(db.execute(query)) == values(db.execute_cypher(query)), cutoff
        query = f"MATCH (n:Person) WHERE n.score >= {cutoff} RETURN n.name AS n"
        assert values(db.execute(query)) == values(db.execute_cypher(query)), cutoff


def test_persona_mila_writes_a_float_in_gql(db):
    row = rows(db.execute("RETURN 0.77F AS f, 0.77e0 AS e, $p AS p", {"p": 0.77}))[0]
    assert row == {"f": 0.77, "e": 0.77, "p": 0.77}
    assert all(isinstance(value, float) for value in row.values())


# ---------------------------------------------------------------------------
# WITH, UNWIND and LET; comments; reserved words
# ---------------------------------------------------------------------------


def test_persona_mila_with_and_let_give_the_same_rows(db):
    with_rows = db.execute(
        "MATCH (n:Person) WITH n, n.age * 2 AS doubled RETURN n.name AS n, doubled ORDER BY n"
    )
    let_rows = db.execute(
        "MATCH (n:Person) LET doubled = n.age * 2 RETURN n.name AS n, doubled ORDER BY n"
    )
    cypher = db.execute_cypher(
        "MATCH (n:Person) WITH n, n.age * 2 AS doubled RETURN n.name AS n, doubled ORDER BY n"
    )
    want = [
        {"n": "Alix", "doubled": 60},
        {"n": "Gus", "doubled": 50},
        {"n": "Vincent", "doubled": 82},
    ]
    assert rows(with_rows) == rows(let_rows) == rows(cypher) == want


def test_persona_mila_unwind_and_for_give_the_same_rows(db):
    want = [("Alix", "chess"), ("Alix", "cycling")]
    assert values(db.execute("MATCH (p:Person) FOR h IN p.hobbies RETURN p.name AS p, h")) == want
    assert (
        values(db.execute("MATCH (p:Person) UNWIND p.hobbies AS h RETURN p.name AS p, h")) == want
    )
    assert (
        values(db.execute_cypher("MATCH (p:Person) UNWIND p.hobbies AS h RETURN p.name AS p, h"))
        == want
    )


def test_persona_mila_a_list_comprehension_runs_in_gql(db):
    assert rows(db.execute("RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 2] AS l")) == [{"l": [4, 6]}]


def test_persona_mila_a_double_dash_comment_in_gql(db):
    query = (
        "-- People in Amsterdam\n"
        "MATCH (p:Person {city: 'Amsterdam'}) /* any age */\n"
        "RETURN p.name AS n"
    )
    assert rows(db.execute(query)) == [{"n": "Alix"}]


def test_persona_mila_keeps_her_double_slash_comments(db):
    # Her Cypher habit: `//` starts a line comment in GQL too.
    query = (
        "// People in Amsterdam\n"
        "MATCH (p:Person {city: 'Amsterdam'}) // any age\n"
        "RETURN p.name AS n"
    )
    assert rows(db.execute(query)) == [{"n": "Alix"}]
    assert rows(db.execute_cypher(query)) == [{"n": "Alix"}]


def test_persona_mila_quotes_a_reserved_word_in_backticks(db):
    assert rows(db.execute("MATCH (p:Person) RETURN count(p) AS `exists`")) == [{"exists": 3}]
    assert rows(
        db.execute(
            "MATCH (:Message)<-[`like`:LIKES]-(p:Person) RETURN p.name AS n, `like`.creationDate "
            "AS d"
        )
    ) == [{"n": "Alix", "d": 2024}]
    # Cypher reserves neither.
    assert rows(db.execute_cypher("MATCH (p:Person) RETURN count(p) AS exists")) == [{"exists": 3}]


# ---------------------------------------------------------------------------
# Aggregates and datetimes
# ---------------------------------------------------------------------------


def test_persona_mila_sum_of_no_values(db):
    assert rows(db.execute("UNWIND [] AS x RETURN sum(x) AS total")) == [{"total": None}]
    assert rows(db.execute_cypher("UNWIND [] AS x RETURN sum(x) AS total")) == [{"total": 0}]
    assert rows(db.execute("MATCH (p:Person {name: 'Nobody'}) RETURN sum(p.age) AS total")) == [
        {"total": None}
    ]
    assert rows(
        db.execute_cypher("MATCH (p:Person {name: 'Nobody'}) RETURN sum(p.age) AS total")
    ) == [{"total": 0}]


def test_persona_mila_zoned_against_local_datetimes(db):
    gql = db.execute(
        "RETURN zoned_datetime('2024-03-15T10:00:00+01:00') = "
        "local_datetime('2024-03-15T09:00:00') AS eq"
    )
    cypher = db.execute_cypher(
        "RETURN datetime('2024-03-15T10:00:00+01:00') = localdatetime('2024-03-15T09:00:00') AS eq"
    )
    assert rows(gql) == [{"eq": True}]
    assert rows(cypher) == [{"eq": False}]


# ---------------------------------------------------------------------------
# Shadow comparison: the same query in both languages
# ---------------------------------------------------------------------------

SHADOW = [
    "MATCH (p:Person) WHERE p.age > 26 RETURN p.name AS n, p.age AS a",
    "MATCH (p:Person) WHERE p.name STARTS WITH 'A' OR p.city = 'Paris' RETURN p.name AS n",
    "MATCH (p:Person) WHERE p.name IN ['Alix', 'Gus'] RETURN p.name AS n",
    "MATCH (p:Person) WHERE p.city IS NOT NULL RETURN count(*) AS c",
    "MATCH (p:Person) RETURN p.city AS city, count(*) AS c, avg(p.age) AS a",
    "MATCH (p:Person) RETURN p.name AS n ORDER BY p.age DESC SKIP 1 LIMIT 1",
    "MATCH (a:Person)-[k:KNOWS]->(b:Person) WHERE k.since >= 2021 RETURN a.name AS a, b.name AS b",
    (
        "MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (a)-[:LIKES]->(m:Message) RETURN a.name "
        "AS a, m.text AS t"
    ),
    (
        "MATCH (a:Person {name: 'Gus'}) OPTIONAL MATCH (a)-[:LIKES]->(m:Message) RETURN a.name AS "
        "a, m.text AS t"
    ),
    "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:LIKES]->(:Message) } RETURN p.name AS n",
    "MATCH (p:Person) WHERE NOT EXISTS { MATCH (p)-[:LIKES]->() } RETURN p.name AS n",
    (
        "MATCH (p:Person) RETURN CASE WHEN p.age > 30 THEN 'senior' ELSE 'junior' END AS c, "
        "count(*) AS k"
    ),
    "MATCH (p:Person) RETURN toUpper(p.name) AS u, size(p.hobbies) AS s",
    "MATCH (a)-[r]->(b) RETURN type(r) AS t, count(*) AS c",
    "MATCH (p:Person) RETURN collect(p.name) AS names",
    "MATCH (p:Person) RETURN min(p.age) AS lo, max(p.age) AS hi, count(DISTINCT p.city) AS cities",
    "MATCH (p:Person) WHERE p:Person AND (p.age < 26 OR p.age > 40) RETURN p.name AS n",
]


@pytest.mark.parametrize("query", SHADOW)
def test_persona_mila_shadow_comparison(db, query):
    # A result is read once: take its rows, then compare.
    gql, cypher = rows(db.execute(query)), rows(db.execute_cypher(query))
    assert gql, query
    if "collect(" in query:
        assert sorted(gql[0]["names"]) == sorted(cypher[0]["names"]) == ["Alix", "Gus", "Vincent"]
    else:
        assert sorted(tuple(row.values()) for row in gql) == sorted(
            tuple(row.values()) for row in cypher
        ), query


def test_persona_mila_shadow_comparison_with_parameters(db):
    query = "MATCH (p:Person) WHERE p.name IN $names AND p.age >= $min RETURN p.name AS n"
    params = {"names": ["Alix", "Gus", "Vincent"], "min": 26}
    assert (
        values(db.execute(query, params))
        == values(db.execute_cypher(query, params))
        == [("Alix",), ("Vincent",)]
    )


# ---------------------------------------------------------------------------
# Her app: selectors over a code graph
# ---------------------------------------------------------------------------


@pytest.fixture
def code_graph():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (f:Graph:File {id: 'f1', active: true})-[:CONTAINS]->(m1:Graph:Method {id: 'm1', "
        "active: true}), "
        "(f)-[:CONTAINS]->(m2:Graph:Method {id: 'm2', active: true}), "
        "(pk:Graph:Package {id: 'p1', active: true})-[:CONTAINS]->(f), "
        "(m1)-[:USES]->(m2), (m2)-[:DEPRECATED_BY]->(m3:Graph:Method {id: 'm3', active: false}), "
        "(t:Graph:Test {id: 't1', active: true})-[:COVERS]->(m1), "
        "(f2:Graph:File {id: 'f2', active: false})"
    )
    return db


SELECTOR = (
    "MATCH (n:Graph) WHERE (n:Method OR n:File) AND n.active "
    "AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
    "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } "
    "AND NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() } RETURN n.id AS id"
)


def test_persona_mila_her_selector_gives_the_same_rows_in_both(code_graph):
    # m1: uses m2, contained in f1, not deprecated; f1: contains, in p1.
    # m2: deprecated; m3 and f2: inactive.
    assert (
        values(code_graph.execute(SELECTOR))
        == values(code_graph.execute_cypher(SELECTOR))
        == [("f1",), ("m1",)]
    )


def test_persona_mila_her_selector_equals_the_union_of_its_labels(code_graph):
    union = (
        "MATCH (n:Method) WHERE n.active AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
        "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } AND NOT EXISTS { MATCH "
        "(n)-[:DEPRECATED_BY]->() } "
        "RETURN n.id AS id UNION "
        "MATCH (n:File) WHERE n.active AND EXISTS { MATCH (n)-[:USES|CONTAINS]->() } "
        "AND EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() } AND NOT EXISTS { MATCH "
        "(n)-[:DEPRECATED_BY]->() } "
        "RETURN n.id AS id"
    )
    assert values(code_graph.execute(union)) == values(code_graph.execute(SELECTOR))


def test_persona_mila_her_multi_label_patterns(code_graph):
    cypher = code_graph.execute_cypher("MATCH (n:Graph:Method) RETURN n.id AS id")
    gql = code_graph.execute("MATCH (n IS Graph & Method) RETURN n.id AS id")
    assert values(cypher) == values(gql) == [("m1",), ("m2",), ("m3",)]


def test_persona_mila_upserts_her_nodes_and_reads_them_back(code_graph):
    summary = code_graph.upsert_nodes(
        ["Graph", "Method"],
        [{"id": "m1", "active": False}, {"id": "m4", "active": True}],
    )
    assert summary["created"] == 1
    assert summary["updated"] == 1
    got = code_graph.execute("MATCH (n:Graph:Method) RETURN n.id AS id, n.active AS a ORDER BY id")
    assert rows(got) == [
        {"id": "m1", "a": False},
        {"id": "m2", "a": True},
        {"id": "m3", "a": False},
        {"id": "m4", "a": True},
    ]
    # The direct API gives (node id, properties) pairs.
    assert sorted(props["id"] for _, props in code_graph.get_nodes_by_label("Method")) == [
        "m1",
        "m2",
        "m3",
        "m4",
    ]
