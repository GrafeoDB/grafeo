"""SPARQL results hold exact RDF terms: grafeo.IRI, grafeo.Literal and
grafeo.BlankNode, each a str of its text that compares and hashes as an RDF
term (#464)."""

import datetime
import decimal
import pickle
import subprocess
import sys
import textwrap

import pytest
from grafeo import IRI, BlankNode, GrafeoDB, Literal

XSD = "http://www.w3.org/2001/XMLSchema#"
RDF_LANG_STRING = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"
EX = "http://example.org/"


@pytest.fixture
def fruit(db):
    db.execute_sparql(
        f"""PREFIX xsd: <{XSD}>
        INSERT DATA {{
            <{EX}apple> <{EX}label> "Apple"@en, "Pomme"@fr ;
                        <{EX}price> "1.50"^^xsd:decimal ;
                        <{EX}picked> "2026-01-15"^^xsd:date ;
                        <{EX}note> "fresh" ;
                        <{EX}grower> _:farm .
        }}"""
    )
    return db


def objects(db, predicate):
    result = db.execute_sparql(
        f"SELECT ?o WHERE {{ <{EX}apple> <{EX}{predicate}> ?o }} ORDER BY ?o"
    )
    return [row["o"] for row in result]


def test_iris_and_blank_nodes_are_their_own_types(fruit):
    subjects = [
        row["s"]
        for row in fruit.execute_sparql(f"SELECT DISTINCT ?s WHERE {{ ?s <{EX}label> ?o }}")
    ]
    assert subjects == [IRI(f"{EX}apple")]
    assert type(subjects[0]) is IRI
    (grower,) = objects(fruit, "grower")
    assert type(grower) is BlankNode
    assert not grower.startswith("_:"), "a blank node's text is its label"


def test_literals_keep_their_language_and_datatype(fruit):
    labels = objects(fruit, "label")
    assert [str(label) for label in labels] == ["Apple", "Pomme"]
    assert [(type(label), label.language) for label in labels] == [(Literal, "en"), (Literal, "fr")]
    assert {label.datatype for label in labels} == {IRI(RDF_LANG_STRING)}
    (price,) = objects(fruit, "price")
    assert (str(price), price.language, str(price.datatype)) == ("1.50", None, f"{XSD}decimal")
    assert price.datatype == IRI(f"{XSD}decimal")
    (note,) = objects(fruit, "note")
    assert (str(note), note.language, str(note.datatype)) == ("fresh", None, f"{XSD}string")


def test_a_literal_reads_as_the_python_value_of_its_type(fruit):
    (price,) = objects(fruit, "price")
    assert price.to_python() == decimal.Decimal("1.50")
    (picked,) = objects(fruit, "picked")
    assert picked.to_python() == datetime.date(2026, 1, 15)
    (count,) = [
        row["n"] for row in fruit.execute_sparql("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }")
    ]
    assert (count, count.to_python()) == (Literal("6", datatype=f"{XSD}integer"), 6)
    assert Literal("INF", datatype=f"{XSD}double").to_python() == float("inf")
    assert Literal("1", datatype=f"{XSD}boolean").to_python() is True
    with pytest.raises(ValueError):
        Literal("abc", datatype=f"{XSD}integer").to_python()


def test_ask_is_one_boolean_and_construct_a_graph(fruit):
    ask = fruit.execute_sparql(f'ASK {{ ?s <{EX}label> "Pomme"@fr }}')
    assert (ask.kind, ask.boolean) == ("boolean", True)
    assert fruit.execute_sparql(f'ASK {{ ?s <{EX}label> "Pomme"@en }}').boolean is False
    select = fruit.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
    assert (select.kind, select.boolean) == ("rows", None)
    graph = fruit.execute_sparql(f"CONSTRUCT {{ ?s <{EX}label> ?o }} WHERE {{ ?s <{EX}label> ?o }}")
    assert graph.kind == "graph"
    assert sorted(graph.to_ntriples().splitlines()) == [
        f'<{EX}apple> <{EX}label> "Apple"@en .',
        f'<{EX}apple> <{EX}label> "Pomme"@fr .',
    ]


def test_to_ntriples_writes_every_kind_of_term_exactly(fruit):
    graph = fruit.execute_sparql(f"CONSTRUCT {{ <{EX}apple> ?p ?o }} WHERE {{ <{EX}apple> ?p ?o }}")
    lines = sorted(graph.to_ntriples().splitlines())
    assert f'<{EX}apple> <{EX}price> "1.50"^^<{XSD}decimal> .' in lines
    assert f'<{EX}apple> <{EX}note> "fresh" .' in lines
    assert any(line.startswith(f"<{EX}apple> <{EX}grower> _:") for line in lines), lines


def test_a_term_round_trips_as_a_property():
    db = GrafeoDB()
    terms = [
        IRI(f"{EX}Kraków"),
        BlankNode("b0"),
        Literal("Pomme", language="fr"),
        Literal("2026-01-15", datatype=f"{XSD}date"),
        Literal("Apple"),
    ]
    for n, term in enumerate(terms):
        db.create_node(["Concept"], {"n": n, "term": term})
    back = [row["c.term"] for row in db.execute("MATCH (c:Concept) RETURN c.term ORDER BY c.n")]
    assert back == terms
    assert [type(term) for term in back] == [IRI, BlankNode, Literal, Literal, Literal]
    assert [(term.language, str(term.datatype)) for term in back[2:]] == [
        ("fr", RDF_LANG_STRING),
        (None, f"{XSD}date"),
        (None, f"{XSD}string"),
    ]


def test_terms_equal_only_terms_of_the_same_kind_text_language_and_datatype():
    assert IRI(f"{EX}a") == IRI(f"{EX}a")
    assert IRI(f"{EX}a") != IRI(f"{EX}b")
    assert BlankNode("b0") == BlankNode("b0")
    assert Literal("Pomme", language="fr") == Literal("Pomme", language="fr")
    # Another kind, language or datatype is another term, with the same text
    assert IRI(f"{EX}a") != Literal(f"{EX}a")
    assert Literal(f"{EX}a") != IRI(f"{EX}a")
    assert IRI("b0") != BlankNode("b0")
    assert BlankNode("b0") != Literal("b0")
    assert Literal("Pomme", language="fr") != Literal("Pomme", language="en")
    assert Literal("Pomme", language="fr") != Literal("Pomme")
    assert Literal("1", datatype=f"{XSD}integer") != Literal("1", datatype=f"{XSD}decimal")
    assert Literal("1", datatype=f"{XSD}integer") != Literal("1")


def test_a_term_never_equals_a_plain_string():
    iri = IRI(f"{EX}a")
    assert iri != f"{EX}a"
    assert f"{EX}a" != iri, "str on the left asks the term first"
    assert not (iri == f"{EX}a")
    assert not (f"{EX}a" == iri)
    assert Literal("Apple") != "Apple"
    assert BlankNode("b0") != "b0"
    assert iri != 1
    assert iri is not None
    # The text is str(term), and the str methods still work
    assert str(iri) == f"{EX}a"
    assert type(str(iri)) is str
    assert iri.startswith("http://") and iri.endswith("/a")
    assert iri.removeprefix(EX) == "a"
    assert f"<{iri}>" == f"<{EX}a>"
    assert isinstance(iri, str)


def test_terms_hash_as_they_compare():
    pairs = [
        (IRI(f"{EX}a"), IRI(f"{EX}a")),
        (BlankNode("b0"), BlankNode("b0")),
        (Literal("Pomme", language="fr"), Literal("Pomme", language="fr")),
        (Literal("Pomme", language="FR"), Literal("Pomme", language="fr")),
        (Literal("Apple"), Literal("Apple", datatype=f"{XSD}string")),
        (Literal("1.50", datatype=f"{XSD}decimal"), Literal("1.50", datatype=f"{XSD}decimal")),
    ]
    for left, right in pairs:
        assert left == right
        assert hash(left) == hash(right), (left, right)
    # Terms of one text but other kinds stay apart in sets and dicts
    same_text = {IRI("x"), BlankNode("x"), Literal("x"), Literal("x", language="nl"), "x"}
    assert len(same_text) == 5
    labels = {Literal("Pomme", language="fr"): "fr", Literal("Apple", language="en"): "en"}
    assert labels[Literal("Pomme", language="FR")] == "fr"
    assert "Pomme" not in labels, "a plain string is no key for a term"
    assert {IRI(f"{EX}a"), IRI(f"{EX}a"), Literal(f"{EX}a")} == {IRI(f"{EX}a"), Literal(f"{EX}a")}


def test_query_results_compare_with_terms(fruit):
    subjects = {
        row["s"] for row in fruit.execute_sparql(f"SELECT ?s WHERE {{ ?s <{EX}label> ?o }}")
    }
    assert subjects == {IRI(f"{EX}apple")}
    assert f"{EX}apple" not in subjects
    labels = set(objects(fruit, "label"))
    assert labels == {Literal("Apple", language="en"), Literal("Pomme", language="fr")}
    assert {str(label) for label in labels} == {"Apple", "Pomme"}
    (note,) = objects(fruit, "note")
    assert note == Literal("fresh") == Literal("fresh", datatype=f"{XSD}string")


def test_terms_pickle_and_stay_immutable():
    terms = [
        IRI(f"{EX}a"),
        BlankNode("b0"),
        Literal("Pomme", language="fr"),
        Literal("1.50", datatype=f"{XSD}decimal"),
        Literal("Apple"),
    ]
    for term in terms:
        back = pickle.loads(pickle.dumps(term))
        assert (type(back), back, hash(back)) == (type(term), term, hash(term))
    literal = Literal("Pomme", language="fr")
    with pytest.raises(AttributeError):
        literal.language = "en"
    with pytest.raises(AttributeError):
        literal.datatype = IRI(f"{XSD}string")
    assert literal == Literal("Pomme", language="fr")


def test_a_language_tag_is_lowercase_and_compares_case_insensitively(fruit):
    tagged = Literal("Pomme", language="FR")
    assert tagged.language == "fr"
    assert tagged == Literal("Pomme", language="fr")
    assert repr(tagged) == "Literal('Pomme', language='fr')"
    assert tagged.n3() == '"Pomme"@fr'
    assert Literal("Kraków", language="pl-PL").language == "pl-pl"
    (label,) = [label for label in objects(fruit, "label") if label.language == "fr"]
    assert label == tagged
    # A term passed to a query carries the lowercase tag, and matches
    db = GrafeoDB()
    db.create_node(["Concept"], {"n": 1, "term": Literal("Pomme", language="fr")})
    db.create_node(["Concept"], {"n": 2, "term": tagged})
    (passed,) = [row["t"] for row in db.execute("RETURN $t AS t", {"t": tagged})]
    assert (passed, passed.language) == (Literal("Pomme", language="fr"), "fr")
    matched = db.execute(
        "MATCH (c:Concept) WHERE c.term = $t RETURN c.n ORDER BY c.n",
        {"t": Literal("Pomme", language="Fr")},
    )
    assert [row["c.n"] for row in matched] == [1, 2]
    # A stored term keeps the lowercase tag
    (back,) = [row["c.term"] for row in db.execute("MATCH (c:Concept {n: 2}) RETURN c.term")]
    assert (back, back.language) == (Literal("Pomme", language="fr"), "fr")


def test_a_literal_of_xsd_string_is_the_simple_literal(fruit):
    typed = Literal("fresh", datatype=f"{XSD}string")
    simple = Literal("fresh")
    assert typed == simple and hash(typed) == hash(simple)
    assert (typed.language, str(typed.datatype)) == (None, f"{XSD}string")
    assert repr(typed) == "Literal('fresh')"
    assert typed.n3() == '"fresh"'
    assert Literal("fresh", datatype=IRI(f"{XSD}string")) == simple
    (note,) = objects(fruit, "note")
    assert note == typed
    # Passed to a query, it matches the stored simple literal and comes back
    # as one
    db = GrafeoDB()
    db.create_node(["Concept"], {"n": 1, "term": simple})
    matched = db.execute("MATCH (c:Concept) WHERE c.term = $t RETURN c.n", {"t": typed})
    assert [row["c.n"] for row in matched] == [1]
    (passed,) = [row["t"] for row in db.execute("RETURN $t AS t", {"t": typed})]
    assert (passed, repr(passed)) == (simple, "Literal('fresh')")


def test_a_term_python_cannot_build_raises_instead_of_being_a_value(tmp_path):
    """A term the conversion cannot build (here grafeo.terms.Literal fails,
    as for a kind the binding does not know) raises from every read of the
    result, never comes back as an exception object in the row."""
    script = textwrap.dedent(
        f"""
        import grafeo
        import grafeo.terms

        class Broken(grafeo.terms.Literal):
            def __new__(cls, *args, **kwargs):
                raise ValueError("no literal today")

        grafeo.terms.Literal = Broken
        db = grafeo.GrafeoDB()
        db.execute_sparql('INSERT DATA {{ <{EX}apple> <{EX}note> "fresh" }}')
        result = db.execute_sparql("SELECT ?o WHERE {{ ?s ?p ?o }}")
        reads = {{
            "index": lambda: result[0],
            "iterate": lambda: list(result),
            "to_list": lambda: result.to_list(),
            "scalar": lambda: result.scalar(),
        }}
        for name, read in reads.items():
            try:
                value = read()
            except ValueError as error:
                assert "no literal today" in str(error), (name, error)
            else:
                raise SystemExit(f"{{name}} returned {{value!r}}")
        print("ok")
        """
    )
    path = tmp_path / "broken_terms.py"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120, check=False
    )
    assert (run.returncode, run.stdout.strip()) == (0, "ok"), run.stderr


def test_a_literal_with_a_language_has_no_other_datatype():
    with pytest.raises(ValueError):
        Literal("Pomme", language="fr", datatype=f"{XSD}string")
    assert repr(Literal("Pomme", language="fr")) == "Literal('Pomme', language='fr')"
    assert repr(IRI(f"{EX}a")) == f"IRI('{EX}a')"


PEOPLE = """
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    INSERT DATA {
        ex:alix  a foaf:Person ; foaf:name "Alix"  ; ex:city "Amsterdam"@en ; ex:age 30 .
        ex:gus   a foaf:Person ; foaf:name "Gus"   ; ex:city "Berlin"@en    ; ex:age 35 .
        ex:jules a foaf:Person ; foaf:name "Jules" ; ex:city "Amsterdam"@en ; ex:age 42 .
    }
"""
PEOPLE_QUERY = """
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?name ?city ?age
    WHERE { ?person a foaf:Person ; foaf:name ?name ; ex:city ?city ; ex:age ?age }
    ORDER BY ?name
"""


def test_terms_export_to_arrow_as_native_values_or_text(db):
    pa = pytest.importorskip("pyarrow")
    db.execute_sparql(PEOPLE)
    table = db.execute_sparql(PEOPLE_QUERY).to_arrow()
    assert table.schema.field("age").type == pa.int64()
    assert table.schema.field("city").type == pa.string()
    assert table.column("age").to_pylist() == [30, 35, 42]
    # A tagged literal exported as text keeps its lexical form only
    assert table.column("city").to_pylist() == ["Amsterdam", "Berlin", "Amsterdam"]
    assert table.column("person").to_pylist() == [
        "http://example.org/alix",
        "http://example.org/gus",
        "http://example.org/jules",
    ]


def test_terms_export_to_dataframes_as_native_values(db):
    pd = pytest.importorskip("pandas")
    db.execute_sparql(PEOPLE)
    frame = db.execute_sparql(PEOPLE_QUERY).to_pandas()
    assert pd.api.types.is_integer_dtype(frame["age"])
    assert frame["age"].mean() == pytest.approx(107 / 3)
    assert [type(city) for city in frame["city"]] == [str, str, str]


def test_the_duckdb_tutorial_averages_sparql_numbers(db):
    """docs/tutorials/examples/duckdb-integration.md: SPARQL to Arrow to DuckDB."""
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    db.execute_sparql(PEOPLE)
    people_table = db.execute_sparql(PEOPLE_QUERY).to_arrow()  # noqa: F841 (read by DuckDB)
    stats = duckdb.sql(
        "SELECT city, COUNT(*) AS count, AVG(age) AS avg_age FROM people_table "
        "GROUP BY city ORDER BY count DESC"
    ).fetchall()
    assert stats == [("Amsterdam", 2, 36.0), ("Berlin", 1, 35.0)]
