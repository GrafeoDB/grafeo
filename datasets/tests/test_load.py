"""The generated GQL and Cypher load into the engine and rebuild the graph."""

import json

import rdflib

from conftest import load_lines, scalar


def graph(derived):
    return json.loads((derived / "graph.json").read_text(encoding="utf-8"))


def test_gql_loads(db, derived):
    load_lines(db, derived / "silicon-valley.gql", "execute")
    g = graph(derived)
    assert scalar(db, "MATCH (n) RETURN count(n) AS c") == len(g["nodes"])
    assert scalar(db, "MATCH ()-[e]->() RETURN count(e) AS c") == len(g["edges"])


def test_cypher_loads(db, derived):
    load_lines(db, derived / "silicon-valley.cypher", "execute_cypher")
    g = graph(derived)
    assert scalar(db, "MATCH (n) RETURN count(n) AS c") == len(g["nodes"])
    assert scalar(db, "MATCH ()-[e]->() RETURN count(e) AS c") == len(g["edges"])


def description_query(key):
    return f"MATCH (n {{key: '{key}'}}) RETURN n.description AS d"


def test_tricky_text_round_trips(db, tricky):
    source, out = tricky
    node = source["nodes"][0]
    expected = node["description"]
    # The fixture must really hold what the test is about.
    for part in ("'", '"', chr(92), chr(10), chr(9), chr(0xE9)):
        assert part in expected

    load_lines(db, out / "silicon-valley.gql", "execute")
    other = type(db)()
    load_lines(other, out / "silicon-valley.cypher", "execute_cypher")
    for engine in (db, other):
        rows = list(engine.execute(description_query(node["key"])))
        assert [r["d"].encode("utf-8") for r in rows] == [expected.encode("utf-8")]
        names = [
            r["n"]
            for r in engine.execute("MATCH (n {key: 'the-palapa'}) RETURN n.name AS n")
        ]
        assert names == ["Erlich's palapa"]
        person = list(
            engine.execute(
                f"MATCH (n {{key: '{node['key']}'}}) RETURN n.name AS n, n.title AS t"
            )
        )
        assert [(r["n"], r["t"]) for r in person] == [
            (node["name"], node["props"]["title"])
        ]

    g = rdflib.Graph()
    g.parse(out / "silicon-valley.ttl", format="turtle")
    uri = rdflib.URIRef(
        "https://docs.grafeo.dev/datasets/silicon-valley/id#" + node["key"]
    )
    literal = g.value(uri, rdflib.URIRef("https://schema.org/description"))
    assert str(literal).encode("utf-8") == expected.encode("utf-8")
    for predicate, want in (
        ("https://schema.org/name", node["name"]),
        (
            "https://docs.grafeo.dev/datasets/silicon-valley/vocab#title",
            node["props"]["title"],
        ),
    ):
        got = g.value(uri, rdflib.URIRef(predicate))
        assert str(got).encode("utf-8") == want.encode("utf-8"), predicate
    palapa = g.value(
        rdflib.URIRef("https://docs.grafeo.dev/datasets/silicon-valley/id#the-palapa"),
        rdflib.URIRef("https://schema.org/name"),
    )
    assert str(palapa) == "Erlich's palapa"


def test_tricky_month_dates(db, tricky):
    _, out = tricky
    load_lines(db, out / "silicon-valley.gql", "execute")
    rows = list(
        db.execute(
            "MATCH (:Person)-[e:WORKS_AT]->(:Company) "
            "WHERE e.from < e.to AND e.from_precision = 'month' AND e.to_precision IS NULL "
            "RETURN count(e) AS c"
        )
    )
    assert rows[0]["c"] == 1


def weissman(engine):
    rows = list(
        engine.execute("MATCH (n {key: 'middle-out'}) RETURN n.weissman_score AS v")
    )
    assert len(rows) == 1
    return rows[0]["v"]


def test_float_props_have_one_type_in_gql_and_cypher(db, derived):
    other = type(db)()
    load_lines(db, derived / "silicon-valley.gql", "execute")
    load_lines(other, derived / "silicon-valley.cypher", "execute_cypher")
    for value in (weissman(db), weissman(other)):
        assert type(value) is float
        assert value == 5.2
