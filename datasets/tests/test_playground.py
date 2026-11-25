"""The queries the playground runs on this dataset (it bundles silicon-valley.gql). Renaming a label, edge
type or property they use breaks the playground; adding nodes, edges or properties does not."""

import pytest

from conftest import load_lines

QUERIES = {
    "sample": "MATCH (p:Person)-[r]->(c:Company {key: 'pied-piper'}) RETURN p, r, c",
    "who works where": "MATCH (p:Person)-[w:WORKS_AT]->(c:Company) RETURN p.name AS person, c.name AS company, w.role",
    "founders": "MATCH (p:Person)-[:FOUNDED]->(c:Company) RETURN p.name AS founder, c.name AS company",
    "who built what": "MATCH (p:Person)-[:BUILT]->(t) RETURN p.name AS builder, t.name AS built",
}


@pytest.fixture(scope="module")
def loaded(derived):
    import grafeo

    db = grafeo.GrafeoDB()
    load_lines(db, derived / "silicon-valley.gql", "execute")
    return db


@pytest.mark.parametrize("name", QUERIES)
def test_playground_query_returns_rows_in_gql_and_cypher(loaded, name):
    gql = list(loaded.execute(QUERIES[name]))
    cypher = list(loaded.execute_cypher(QUERIES[name]))
    assert len(gql) > 0, name
    assert len(cypher) == len(gql), name


def test_playground_smoke_finds_pied_piper(loaded):
    rows = list(
        loaded.execute("MATCH (c:Company {key: 'pied-piper'}) RETURN c.name AS name")
    )
    assert [r["name"] for r in rows] == ["Pied Piper"]


def test_people_have_titles(loaded):
    """The GraphQL sample asks for `Person { name title }`: some people must have a title."""
    rows = list(
        loaded.execute(
            "MATCH (p:Person) WHERE p.title IS NOT NULL RETURN count(p) AS c"
        )
    )
    assert rows[0]["c"] > 0


def test_playground_rdf_terms_are_in_the_turtle(derived):
    """The playground also bundles silicon-valley.ttl; its SPARQL questions use these terms."""
    import rdflib

    g = rdflib.Graph().parse(derived / "silicon-valley.ttl", format="turtle")
    sv = rdflib.Namespace("https://docs.grafeo.dev/datasets/silicon-valley/vocab#")
    svr = rdflib.Namespace("https://docs.grafeo.dev/datasets/silicon-valley/id#")
    schema = rdflib.Namespace("https://schema.org/")
    assert (svr["pied-piper"], rdflib.RDF.type, schema.Organization) in g
    assert (svr["pied-piper"], schema.name, rdflib.Literal("Pied Piper")) in g
    people = set(g.subjects(rdflib.RDF.type, schema.Person))
    assert people and all(g.value(p, schema.name) is not None for p in people)
    for link in (sv.worksAt, sv.founded):
        assert any(s in people for s in g.subjects(link, None)), link
