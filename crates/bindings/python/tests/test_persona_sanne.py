"""Persona: Sanne, a linked-data engineer, works in SPARQL from Python.

She loads data with INSERT DATA (the docs' way for the bindings), in the
default graph and named graphs, queries with OPTIONAL, MINUS, NOT EXISTS,
GRAPH and FROM NAMED, aggregates, follows property paths, builds triples with
CONSTRUCT and exports them, and maintains the data with updates. Her slips: a
forgotten PREFIX, Turtle's @prefix in a query, RDF-star out of habit, a
variable in INSERT DATA, LIMIT before ORDER BY.
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

P = "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nPREFIX ex: <http://example.org/>\n"
EX = "http://example.org/"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.execute_sparql(
        P
        + """INSERT DATA {
  ex:alix a foaf:Person ; foaf:name "Alix" ; foaf:age 30 ; foaf:knows ex:gus , ex:vincent .
  ex:gus a foaf:Person ; foaf:name "Gus" ; foaf:age 25 .
  ex:vincent a foaf:Person ; foaf:name "Vincent"@en ; foaf:knows ex:mia .
  ex:mia a foaf:Person ; foaf:name "Mia" ; foaf:mbox <mailto:mia@example.org> .
  GRAPH ex:travel { ex:alix ex:visited ex:paris . ex:gus ex:visited ex:prague , ex:paris . }
  GRAPH ex:work { ex:alix ex:worksAt ex:grafeo . }
}"""
    )
    return db


def sparql(db, query):
    return db.execute_sparql(P + query).to_list()


def text(term):
    """A term's lexical form or IRI, None for unbound."""
    return None if term is None else str(term.value if hasattr(term, "value") else term)


def column(db, query, name):
    return [text(row[name]) for row in sparql(db, query)]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_persona_sanne_selects_with_a_type_and_a_property_list(db):
    names = column(db, "SELECT ?n WHERE { ?p a foaf:Person ; foaf:name ?n } ORDER BY ?n", "n")
    assert names == ["Alix", "Gus", "Mia", "Vincent"]


def test_persona_sanne_keeps_literal_kinds(db):
    rows = sparql(db, "SELECT ?n ?age WHERE { ?p foaf:name ?n OPTIONAL { ?p foaf:age ?age } }")
    by_name = {text(row["n"]): row for row in rows}
    assert text(by_name["Alix"]["age"].datatype) == XSD_INTEGER
    assert text(by_name["Alix"]["age"]) == "30"
    assert by_name["Mia"]["age"] is None
    assert by_name["Vincent"]["n"].language == "en"


def test_persona_sanne_optional_minus_and_not_exists(db):
    assert column(
        db, "SELECT ?n WHERE { ?p foaf:name ?n MINUS { ?p foaf:age ?a } } ORDER BY ?n", "n"
    ) == [
        "Mia",
        "Vincent",
    ]
    assert column(
        db,
        "SELECT ?n WHERE { ?p foaf:name ?n FILTER NOT EXISTS { ?p foaf:knows ?x } } ORDER BY ?n",
        "n",
    ) == ["Gus", "Mia"]
    rows = sparql(
        db,
        "SELECT ?n (COUNT(?f) AS ?friends) WHERE { ?p foaf:name ?n OPTIONAL { ?p foaf:knows ?f } } "
        "GROUP BY ?n ORDER BY DESC(?friends) ?n",
    )
    assert [(text(row["n"]), text(row["friends"])) for row in rows] == [
        ("Alix", "2"),
        ("Vincent", "1"),
        ("Gus", "0"),
        ("Mia", "0"),
    ]


def test_persona_sanne_language_tags_in_filters(db):
    assert column(db, "SELECT ?n WHERE { ?p foaf:name ?n FILTER(LANG(?n) = 'en') }", "n") == [
        "Vincent"
    ]
    assert column(db, "SELECT ?n WHERE { ?p foaf:name ?n FILTER(?n = 'Vincent') }", "n") == []
    assert column(db, "SELECT ?n WHERE { ?p foaf:name ?n FILTER(STR(?n) = 'Vincent') }", "n") == [
        "Vincent"
    ]


def test_persona_sanne_named_graphs(db):
    rows = sparql(
        db,
        "SELECT ?g (COUNT(*) AS ?c) WHERE { GRAPH ?g { ?s ?p ?o } } GROUP BY ?g ORDER BY ?g",
    )
    assert [(text(row["g"]), text(row["c"])) for row in rows] == [
        (EX + "travel", "3"),
        (EX + "work", "1"),
    ]
    # The default graph holds none of the named graphs' triples.
    assert sparql(db, "SELECT ?s WHERE { ?s ex:visited ?c }") == []
    visits = sparql(
        db,
        "SELECT ?s ?city FROM NAMED ex:travel WHERE { GRAPH ?g { ?s ex:visited ?city } } "
        "ORDER BY ?s ?city",
    )
    assert [(text(row["s"]), text(row["city"])) for row in visits] == [
        (EX + "alix", EX + "paris"),
        (EX + "gus", EX + "paris"),
        (EX + "gus", EX + "prague"),
    ]


def test_persona_sanne_follows_a_property_path(db):
    assert column(db, "SELECT ?x WHERE { ex:alix foaf:knows+ ?x } ORDER BY ?x", "x") == [
        EX + "gus",
        EX + "mia",
        EX + "vincent",
    ]
    assert sparql(db, "ASK { ex:alix foaf:knows/foaf:knows ex:mia }") == [{"boolean": True}]
    assert sparql(db, "ASK { ex:gus foaf:knows ex:alix }") == [{"boolean": False}]


def test_persona_sanne_constructs_and_exports(db):
    result = db.execute_sparql(P + "CONSTRUCT { ?a foaf:knownBy ?b } WHERE { ?b foaf:knows ?a }")
    lines = sorted(line for line in result.to_ntriples().splitlines() if line)
    knownby = "<http://xmlns.com/foaf/0.1/knownBy>"
    assert lines == [
        f"<{EX}gus> {knownby} <{EX}alix> .",
        f"<{EX}mia> {knownby} <{EX}vincent> .",
        f"<{EX}vincent> {knownby} <{EX}alix> .",
    ]
    turtle = db.execute_sparql(
        P + "CONSTRUCT { ?a foaf:knownBy ?b } WHERE { ?b foaf:knows ?a }"
    ).to_turtle()
    assert f"<{EX}mia> {knownby} <{EX}vincent> ." in turtle


def test_persona_sanne_describes_a_resource(db):
    rows = sparql(db, "DESCRIBE ex:mia")
    predicates = sorted(text(row["predicate"]) for row in rows)
    assert predicates == [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://xmlns.com/foaf/0.1/mbox",
        "http://xmlns.com/foaf/0.1/name",
    ]


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


def test_persona_sanne_updates_ages_in_place(db):
    sparql(
        db,
        "DELETE { ?p foaf:age ?a } INSERT { ?p foaf:age ?b } "
        "WHERE { ?p foaf:age ?a BIND(?a + 1 AS ?b) }",
    )
    rows = sparql(db, "SELECT ?n ?a WHERE { ?p foaf:name ?n ; foaf:age ?a } ORDER BY ?n")
    assert [(text(row["n"]), text(row["a"])) for row in rows] == [("Alix", "31"), ("Gus", "26")]


def test_persona_sanne_deletes_copies_and_clears_graphs(db):
    sparql(db, "DELETE WHERE { ?p foaf:mbox ?m }")
    assert sparql(db, "ASK { ?p foaf:mbox ?m }") == [{"boolean": False}]
    sparql(db, "COPY ex:travel TO ex:backup")
    sparql(db, "CLEAR GRAPH ex:travel")
    count = "SELECT (COUNT(*) AS ?c) WHERE { GRAPH ex:%s { ?s ?p ?o } }"
    assert text(sparql(db, count % "backup")[0]["c"]) == "3"
    assert text(sparql(db, count % "travel")[0]["c"]) == "0"
    sparql(db, "DROP SILENT GRAPH ex:nothere")
    with pytest.raises(grafeo.GrafeoError, match="does not exist"):
        sparql(db, "DROP GRAPH ex:nothere")


# ---------------------------------------------------------------------------
# Her slips
# ---------------------------------------------------------------------------


def test_persona_sanne_forgets_a_prefix(db):
    with pytest.raises(
        grafeo.GrafeoError, match=r"the prefix 'foaf:' is not declared \(add PREFIX foaf:"
    ):
        db.execute_sparql("SELECT ?n WHERE { ?p foaf:name ?n }")
    # The predefined prefixes need no declaration.
    rows = db.execute_sparql(
        "PREFIX ex: <http://example.org/> "
        'SELECT ?p WHERE { ?p <http://xmlns.com/foaf/0.1/age> "30"^^xsd:integer }'
    ).to_list()
    assert [text(row["p"]) for row in rows] == [EX + "alix"]


def test_persona_sanne_writes_rdf_star_in_a_query(db):
    with pytest.raises(grafeo.GrafeoError, match="SPARQL 1.2") as caught:
        sparql(db, "SELECT ?s WHERE { << ex:alix foaf:knows ex:gus >> ex:since ?s }")
    assert caught.value.gql_status == "42001"


def test_persona_sanne_writes_rdf_star_in_insert_data(db):
    # As in a query: the error names SPARQL 1.2, not a missing brace.
    with pytest.raises(grafeo.GrafeoError, match="SPARQL 1.2") as caught:
        sparql(db, "INSERT DATA { << ex:alix foaf:knows ex:gus >> ex:since 2020 }")
    assert caught.value.gql_status == "42001"


def test_persona_sanne_puts_a_variable_in_insert_data(db):
    # INSERT DATA takes no variables: a syntax error (SPARQL 1.1 Update,
    # QuadData), not an internal error.
    with pytest.raises(grafeo.GrafeoError, match=r"'\?x'") as caught:
        sparql(db, "INSERT DATA { ?x foaf:name 'X' }")
    assert caught.value.gql_status == "42001", str(caught.value)
    assert sparql(db, "ASK { ?p foaf:name 'X' }") == [{"boolean": False}]


def test_persona_sanne_puts_limit_before_order_by(db):
    with pytest.raises(grafeo.GrafeoError, match="ORDER") as caught:
        sparql(db, "SELECT ?n WHERE { ?p foaf:name ?n } LIMIT 2 ORDER BY ?n")
    assert caught.value.gql_status == "42001"
