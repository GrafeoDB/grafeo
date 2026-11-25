"""The RDFS, OWL and JSON variants hold the same data as the plain files, and load into the engine."""

import datetime
import json

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from conftest import load_lines
from test_rdf import PREFIXES, SVR

SV = "https://docs.grafeo.dev/datasets/silicon-valley/vocab#"
ONTOLOGY = rdflib.URIRef(SV[:-1])
EMPLOYMENT = (
    PREFIXES
    + """
    SELECT ?company WHERE {
        ?e a sv:Employment ; sv:person svr:richard-hendricks ; sv:company ?company .
    }
    """
)


def turtle_graph(path):
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    return g


def is_vocabulary(term):
    """A term only the vocabulary blocks introduce: the ontology, an sv: or schema: term, or a blank node."""
    return (
        isinstance(term, rdflib.BNode)
        or term == ONTOLOGY
        or str(term).startswith((SV, "https://schema.org/"))
    )


def test_rdfs_and_owl_are_the_data_plus_a_vocabulary(derived):
    data = turtle_graph(derived / "silicon-valley.ttl")
    for name, kinds in (
        ("rdfs", {RDFS.Class, RDF.Property}),
        ("owl", {OWL.Class, OWL.ObjectProperty}),
    ):
        variant = turtle_graph(derived / f"silicon-valley.{name}.ttl")
        assert set(data) <= set(variant), name
        extra = set(variant) - set(data)
        assert extra and all(is_vocabulary(s) for s, _, _ in extra), name
        declared = {
            s
            for s, _, o in variant.triples((None, RDF.type, None))
            if o in kinds | {OWL.DatatypeProperty}
        }
        used = {p for _, p, _ in data if str(p).startswith(SV)}
        assert used <= declared, (
            f"{name}: undeclared {sorted(map(str, used - declared))}"
        )


def test_every_stated_domain_and_range_holds_in_the_data(derived):
    """RDFS states a domain or range only when the data never contradicts it."""
    g = turtle_graph(derived / "silicon-valley.rdfs.ttl")
    for prop, cls in g.subject_objects(RDFS.domain):
        for s in g.subjects(prop, None):
            assert (s, RDF.type, cls) in g, (prop, s)
    for prop, rng in g.subject_objects(RDFS.range):
        for o in g.objects(None, prop):
            if isinstance(o, rdflib.Literal):
                assert o.datatype == rng or (
                    o.datatype is None and str(rng).endswith("#string")
                ), (prop, o)
            else:
                assert (o, RDF.type, rng) in g, (prop, o)


def test_rdfs_and_owl_load_into_grafeo(db, derived):
    for name in ("rdfs", "owl"):
        engine = type(db)()
        engine.load_rdf_text(
            (derived / f"silicon-valley.{name}.ttl").read_text(encoding="utf-8"),
            "turtle",
        )
        companies = sorted(
            str(r["company"]).removeprefix(SVR)
            for r in engine.execute_sparql(EMPLOYMENT)
        )
        assert companies == ["hooli", "pied-piper", "pied-piper", "pied-piper"], name
        classes = list(
            engine.execute_sparql(
                PREFIXES
                + "SELECT ?c WHERE { ?c a <http://www.w3.org/2002/07/owl#Class> }"
            )
        )
        assert (len(classes) > 0) == (name == "owl")


def ntriples(payload):
    """The `importRdf` payload as N-Triples text."""

    def term(value):
        if isinstance(value, str):
            return f"<{value}>"
        literal = rdflib.Literal(value["value"], datatype=value.get("datatype"))
        return literal.n3()

    return (
        "\n".join(
            f"<{t['subject']}> <{t['predicate']}> {term(t['object'])} ."
            for t in payload["triples"]
        )
        + "\n"
    )


def test_rdf_payload_is_the_turtle_data(db, derived):
    payload = json.loads(
        (derived / "silicon-valley.rdf.json").read_text(encoding="utf-8")
    )
    from_payload = rdflib.Graph().parse(data=ntriples(payload), format="nt")
    assert set(from_payload) == set(turtle_graph(derived / "silicon-valley.ttl"))
    db.load_rdf_text(ntriples(payload), "ntriples")
    companies = sorted(
        str(r["company"]).removeprefix(SVR) for r in db.execute_sparql(EMPLOYMENT)
    )
    assert companies == ["hooli", "pied-piper", "pied-piper", "pied-piper"]


def value(v):
    """A payload value as the binding takes it: `{"$date": day}` is a date."""
    return datetime.date.fromisoformat(v["$date"]) if isinstance(v, dict) else v


def dump(db):
    nodes = sorted(
        (sorted(r["l"]), sorted(r["p"].items()))
        for r in db.execute("MATCH (n) RETURN labels(n) AS l, properties(n) AS p")
    )
    edges = sorted(
        (r["s"], r["t"], r["o"], sorted(r["p"].items()))
        for r in db.execute(
            "MATCH (a)-[e]->(b) RETURN a.key AS s, type(e) AS t, b.key AS o, properties(e) AS p"
        )
    )
    return nodes, edges


def test_lpg_payload_is_the_gql_graph(db, derived):
    payload = json.loads(
        (derived / "silicon-valley.lpg.json").read_text(encoding="utf-8")
    )
    ids = [
        db.create_node(
            n["labels"], {k: value(v) for k, v in n["properties"].items()}
        ).id
        for n in payload["nodes"]
    ]
    for e in payload["edges"]:
        db.create_edge(
            ids[e["source"]],
            ids[e["target"]],
            e["type"],
            {k: value(v) for k, v in e["properties"].items()},
        )
    gql = type(db)()
    load_lines(gql, derived / "silicon-valley.gql", "execute")
    assert dump(db) == dump(gql)
