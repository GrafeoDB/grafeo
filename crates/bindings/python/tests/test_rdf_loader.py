"""The RDF loader from Python (0.6.1): db.load_rdf(path, format=None, graph=None) and
db.load_rdf_text(text, format, graph=None).

Turtle and N-Triples by extension, a str or a pathlib path, a named format over the
extension, a bad format name (ValueError), an extension it cannot tell, a line that does
not parse (nothing loaded), duplicates counted once, a missing file, and a persistent
database closed and reopened. Text loads as its file does; a file or text loads into a
named graph as INSERT DATA into it does, and the default graph stays empty; an empty
graph name is refused; SPARQL LOAD names the loader. Each load is checked against
SPARQL. Files are written to pytest's tmp_path.
"""

import pytest

grafeo = pytest.importorskip("grafeo")

if not hasattr(grafeo.GrafeoDB, "load_rdf"):
    pytest.skip("this build has no triple store", allow_module_level=True)

TURTLE = """@prefix ex: <http://ex.org/> .
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

ex:alix a foaf:Person ;
    foaf:name "Alix" , "Alix"@nl ;
    ex:age 30 ;
    ex:likes ( ex:jazz ex:chess ) ;
    ex:address [ ex:city "Amsterdam" ] .
ex:gus foaf:name "Gus" .
"""
# a type, two names, an age, the list (its link and two list nodes of two triples
# each), the address (its link and the city) and Gus's name
TURTLE_TRIPLES = 12
NTRIPLES = """# people of Amsterdam
<http://ex.org/alix> <http://ex.org/name> "Alix" .

<http://ex.org/alix> <http://ex.org/city> "Amsterdam"@en .
<http://ex.org/gus> <http://ex.org/age> "25"^^<http://www.w3.org/2001/XMLSchema#integer> .
"""


def count(db):
    return int(db.execute_sparql("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }").to_list()[0]["n"])


def names(db):
    rows = db.execute_sparql(
        "SELECT ?name WHERE { ?p <http://xmlns.com/foaf/0.1/name> ?name }"
    ).to_list()
    return sorted(str(row["name"]) for row in rows)


def test_turtle_by_extension_as_str_and_as_path(tmp_path):
    file = tmp_path / "people.ttl"
    file.write_text(TURTLE, encoding="utf-8")
    for path in (str(file), file):
        db = grafeo.GrafeoDB()
        assert db.load_rdf(path) == TURTLE_TRIPLES
        assert count(db) == TURTLE_TRIPLES
        assert names(db) == ["Alix", "Alix", "Gus"]
        city = db.execute_sparql(
            "SELECT ?c WHERE { <http://ex.org/alix> <http://ex.org/address> ?a . "
            "?a <http://ex.org/city> ?c }"
        ).to_list()
        assert [str(row["c"]) for row in city] == ["Amsterdam"]


def test_ntriples_by_extension(tmp_path):
    file = tmp_path / "people.NT"
    file.write_text(NTRIPLES, encoding="utf-8")
    db = grafeo.GrafeoDB()
    assert db.load_rdf(file) == 3
    age = db.execute_sparql(
        "SELECT ?a WHERE { <http://ex.org/gus> <http://ex.org/age> ?a }"
    ).to_list()
    assert int(age[0]["a"]) == 25


@pytest.mark.parametrize("name", ["turtle", "TTL", "ttl"])
def test_a_named_format_wins_over_the_extension(tmp_path, name):
    file = tmp_path / "people.data"
    file.write_text(TURTLE, encoding="utf-8")
    db = grafeo.GrafeoDB()
    assert db.load_rdf(file, format=name) == TURTLE_TRIPLES


def test_ntriples_by_name(tmp_path):
    file = tmp_path / "people.txt"
    file.write_text(NTRIPLES, encoding="utf-8")
    for name in ("ntriples", "n-triples", "nt"):
        db = grafeo.GrafeoDB()
        assert db.load_rdf(str(file), format=name) == 3


def test_a_bad_format_name_is_a_value_error(tmp_path):
    file = tmp_path / "people.ttl"
    file.write_text(TURTLE, encoding="utf-8")
    db = grafeo.GrafeoDB()
    with pytest.raises(ValueError, match="rdfxml"):
        db.load_rdf(file, format="rdfxml")
    assert count(db) == 0


def test_an_extension_it_cannot_tell(tmp_path):
    file = tmp_path / "people.rdf"
    file.write_text(NTRIPLES, encoding="utf-8")
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="name the format"):
        db.load_rdf(file)
    assert count(db) == 0


def test_a_line_that_does_not_parse_loads_nothing(tmp_path):
    file = tmp_path / "bad.nt"
    file.write_text(
        '<http://ex.org/alix> <http://ex.org/city> "Amsterdam" .\n'
        "<http://ex.org/gus> <http://ex.org/city> Berlin .\n",
        encoding="utf-8",
    )
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="2"):
        db.load_rdf(file)
    assert count(db) == 0


def test_duplicates_count_once(tmp_path):
    file = tmp_path / "cities.nt"
    line = '<http://ex.org/mia> <http://ex.org/city> "Prague" .\n'
    file.write_text(line + line, encoding="utf-8")
    db = grafeo.GrafeoDB()
    assert db.load_rdf(file) == 1
    assert db.load_rdf(file) == 0
    assert count(db) == 1


def test_a_missing_file(tmp_path):
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="cannot read"):
        db.load_rdf(tmp_path / "missing.ttl")


def test_a_persistent_database_keeps_the_load(tmp_path):
    file = tmp_path / "people.ttl"
    file.write_text(TURTLE, encoding="utf-8")
    path = str(tmp_path / "people.grafeo")
    db = grafeo.GrafeoDB.open(path)
    assert db.load_rdf(file) == TURTLE_TRIPLES
    db.close()
    db = grafeo.GrafeoDB.open(path)
    assert count(db) == TURTLE_TRIPLES
    db.close()


# ---------------------------------------------------------------------------
# load_rdf_text(text, format, graph=None) and load_rdf(path, graph=...)
# ---------------------------------------------------------------------------

PEOPLE = "http://ex.org/graphs/people"


def count_in(db, graph):
    query = f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"
    return int(db.execute_sparql(query).to_list()[0]["n"])


def test_text_loads_as_its_file_does():
    db = grafeo.GrafeoDB()
    assert db.load_rdf_text(TURTLE, "turtle") == TURTLE_TRIPLES
    assert names(db) == ["Alix", "Alix", "Gus"]
    db = grafeo.GrafeoDB()
    assert db.load_rdf_text(NTRIPLES, "nt") == 3
    assert count(db) == 3


def test_text_with_a_bad_format_name_is_a_value_error():
    db = grafeo.GrafeoDB()
    with pytest.raises(ValueError, match="jsonld"):
        db.load_rdf_text(TURTLE, "jsonld")
    assert count(db) == 0


def test_text_with_a_line_that_does_not_parse_loads_nothing():
    db = grafeo.GrafeoDB()
    bad = '<http://ex.org/alix> <http://ex.org/city> "Amsterdam" .\n<http://ex.org/gus> .\n'
    with pytest.raises(Exception, match="the text is no N-Triples"):
        db.load_rdf_text(bad, "ntriples")
    assert count(db) == 0


def test_a_file_into_a_named_graph(tmp_path):
    file = tmp_path / "people.ttl"
    file.write_text(TURTLE, encoding="utf-8")
    db = grafeo.GrafeoDB()
    assert db.load_rdf(file, graph=PEOPLE) == TURTLE_TRIPLES
    assert count_in(db, PEOPLE) == TURTLE_TRIPLES
    assert count(db) == 0, "the default graph stays empty"
    # Triples the graph holds are nothing new (a Turtle file's blank nodes
    # are new ones on each load, so this file has none).
    plain = tmp_path / "people.nt"
    plain.write_text(NTRIPLES, encoding="utf-8")
    assert db.load_rdf(plain, graph=PEOPLE) == 3
    assert db.load_rdf(plain, graph=PEOPLE) == 0
    assert count_in(db, PEOPLE) == TURTLE_TRIPLES + 3
    rows = db.execute_sparql(
        f"SELECT ?name WHERE {{ GRAPH <{PEOPLE}> {{ <http://ex.org/gus> "
        "<http://xmlns.com/foaf/0.1/name> ?name } }"
    ).to_list()
    assert [str(row["name"]) for row in rows] == ["Gus"]


def test_text_into_a_named_graph_holds_what_insert_data_holds():
    db = grafeo.GrafeoDB()
    assert db.load_rdf_text(NTRIPLES, "ntriples", graph=PEOPLE) == 3
    other = grafeo.GrafeoDB()
    body = "\n".join(line for line in NTRIPLES.splitlines() if not line.startswith("#"))
    other.execute_sparql(f"INSERT DATA {{ GRAPH <{PEOPLE}> {{ {body} }} }}")
    query = f"SELECT ?s ?p ?o WHERE {{ GRAPH <{PEOPLE}> {{ ?s ?p ?o }} }} ORDER BY ?s ?p ?o"
    assert [str(r) for r in db.execute_sparql(query).to_list()] == [
        str(r) for r in other.execute_sparql(query).to_list()
    ]
    assert count(db) == 0


def test_an_empty_graph_name_is_refused():
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="empty name"):
        db.load_rdf_text(NTRIPLES, "ntriples", graph="")
    assert count(db) == 0


def test_a_persistent_database_keeps_a_named_graph_load(tmp_path):
    path = str(tmp_path / "people.grafeo")
    db = grafeo.GrafeoDB.open(path)
    assert db.load_rdf_text(TURTLE, "turtle", graph=PEOPLE) == TURTLE_TRIPLES
    db.close()
    db = grafeo.GrafeoDB.open(path)
    assert count_in(db, PEOPLE) == TURTLE_TRIPLES
    assert count(db) == 0
    db.close()


def test_sparql_load_points_to_the_loader():
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="load_rdf_text"):
        db.execute_sparql("LOAD <file:///data/people.ttl>")
