"""Turtle checks, with rdflib."""

import rdflib

SV = "https://docs.grafeo.dev/datasets/silicon-valley/vocab#"
SVR = "https://docs.grafeo.dev/datasets/silicon-valley/id#"
SCHEMA = "https://schema.org/"

PREFIXES = f"""
PREFIX sv: <{SV}>
PREFIX svr: <{SVR}>
PREFIX schema: <{SCHEMA}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def test_turtle_parses_and_answers(derived):
    g = rdflib.Graph()
    g.parse(derived / "silicon-valley.ttl", format="turtle")
    # Richard's Hooli job has only an end date, so `sv:from` is optional. Rows are ordered by the
    # start date; a job without one sorts first (unbound sorts before bound), which puts Hooli first.
    rows = list(
        g.query(
            PREFIXES
            + """
            SELECT ?company ?from WHERE {
                ?e a sv:Employment ; sv:person svr:richard-hendricks ; sv:company ?company .
                OPTIONAL { ?e sv:from ?from }
            } ORDER BY ?from
            """
        )
    )
    # Richard's source has four jobs: Hooli, then three stints at Pied Piper (CEO, CTO, CEO).
    companies = [str(r.company).removeprefix(SVR) for r in rows]
    assert companies == ["hooli", "pied-piper", "pied-piper", "pied-piper"]
    starts = [str(r["from"]) for r in rows if r["from"] is not None]
    assert starts == ["2014-03-01", "2016-04-01", "2016-05-01"]


def literal_text(value):
    """The lexical form of a SPARQL result value from rdflib or Grafeo."""
    return None if value is None else str(getattr(value, "value", value))


def turtle_db(db, derived):
    db.load_rdf_text(
        (derived / "silicon-valley.ttl").read_text(encoding="utf-8"), "turtle"
    )
    return db


def test_turtle_loads_into_grafeo(db, derived):
    turtle_db(db, derived)
    rows = list(
        db.execute_sparql(
            PREFIXES
            + """
            SELECT ?company ?from WHERE {
                ?e a sv:Employment ; sv:person svr:richard-hendricks ; sv:company ?company .
                OPTIONAL { ?e sv:from ?from }
            } ORDER BY ?from
            """
        )
    )
    companies = [literal_text(r["company"]).removeprefix(SVR) for r in rows]
    assert companies == ["hooli", "pied-piper", "pied-piper", "pied-piper"]
    starts = [literal_text(r["from"]) for r in rows if r["from"] is not None]
    assert starts == ["2014-03-01", "2016-04-01", "2016-05-01"]


def test_month_dates_carry_a_precision(derived):
    g = rdflib.Graph()
    g.parse(derived / "silicon-valley.ttl", format="turtle")
    rows = list(
        g.query(
            PREFIXES
            + """
            SELECT ?from ?precision ?to ?toPrecision WHERE {
                ?b a sv:Base ; sv:company svr:pied-piper ; sv:place svr:hacker-hostel ;
                   sv:from ?from ; sv:to ?to .
                OPTIONAL { ?b sv:fromPrecision ?precision }
                OPTIONAL { ?b sv:toPrecision ?toPrecision }
            }
            """
        )
    )
    assert [(str(r["from"]), str(r["precision"])) for r in rows] == [
        ("2014-03-01", "month")
    ]
    assert rows[0]["from"].datatype == rdflib.XSD.date
    assert str(rows[0]["toPrecision"]) == "month"
    # A day date has no precision triple, and no month-only datatype is left anywhere.
    assert all(
        o.datatype != rdflib.XSD.gYearMonth
        for o in g.objects()
        if isinstance(o, rdflib.Literal)
    )


RANGE_QUERY = (
    PREFIXES
    + """
    SELECT ?place ?city WHERE {
        ?b a sv:Base ; sv:company svr:pied-piper ; sv:place ?place ; sv:from ?from .
        ?place sv:locatedIn ?city .
        FILTER(?from < "2014-09-01"^^xsd:date)
    }
    """
)


def test_date_range_filter_finds_the_palo_alto_base(db, derived):
    g = rdflib.Graph()
    g.parse(derived / "silicon-valley.ttl", format="turtle")
    turtle_db(db, derived)
    for engine, rows in (
        ("rdflib", list(g.query(RANGE_QUERY))),
        ("grafeo", list(db.execute_sparql(RANGE_QUERY))),
    ):
        found = [
            (
                literal_text(r["place"]).removeprefix(SVR),
                literal_text(r["city"]).removeprefix(SVR),
            )
            for r in rows
        ]
        assert found == [("hacker-hostel", "palo-alto")], engine
