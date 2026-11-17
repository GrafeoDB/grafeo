"""Tests for scripts/convert_w3c_sparql.py on a small W3C-shaped suite.

Each case of the suite checks one thing the converter must keep from the W3C
files: prefixed Turtle data, literal lexical forms, duplicate rows, column
order, ASK and empty results, TSV cells, and the tests it must leave out.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_w3c_sparql as converter  # noqa: E402

MANIFEST = """\
@prefix :      <manifest#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix mf:    <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
@prefix qt:    <http://www.w3.org/2001/sw/DataAccess/tests/test-query#> .
@prefix ut:    <http://www.w3.org/2009/sparql/tests/test-update#> .
@prefix dawgt: <http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#> .

<> mf:entries ( :select-star :ask :empty :tsv :named :update :withdrawn ) .

:select-star a mf:QueryEvaluationTest ;
    mf:name "Select (star) with duplicates" ;
    mf:action [ qt:query <star.rq> ; qt:data <data.ttl> ] ;
    mf:result <star.srx> .

:ask a mf:QueryEvaluationTest ;
    mf:action [ qt:query <ask.rq> ; qt:data <data.ttl> ] ;
    mf:result <ask.srx> .

:empty a mf:QueryEvaluationTest ;
    mf:action [ qt:query <empty.rq> ; qt:data <data.ttl> ] ;
    mf:result <empty.srx> .

:tsv a mf:QueryEvaluationTest ;
    mf:action [ qt:query <tsv.rq> ; qt:data <data.ttl> ] ;
    mf:result <tsv.tsv> .

:named a mf:QueryEvaluationTest ;
    mf:action [ qt:query <star.rq> ; qt:graphData <data.ttl> ] ;
    mf:result <star.srx> .

:update a mf:UpdateEvaluationTest ;
    mf:action [ ut:request <star.rq> ] .

:withdrawn a mf:QueryEvaluationTest ;
    dawgt:approval dawgt:Withdrawn ;
    mf:action [ qt:query <ask.rq> ; qt:data <data.ttl> ] ;
    mf:result <ask.srx> .
"""

DATA = """\
@prefix ex:  <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:alix ex:knows ex:gus ;
    ex:score 3.21E4 ;
    ex:rate "1.0"^^xsd:decimal ;
    ex:source <> ;
    ex:address [ ex:city "Amsterdam" ; ex:next [ ex:city "Utrecht" ] ] .
"""

# The header lists the variables out of query order, and one row twice.
STAR_SRX = """\
<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="o"/><variable name="s"/></head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/alix</uri></binding>
      <binding name="o"><literal datatype="http://www.w3.org/2001/XMLSchema#double">3.21E4</literal></binding>
    </result>
    <result>
      <binding name="s"><uri>http://example.org/alix</uri></binding>
      <binding name="o"><literal datatype="http://www.w3.org/2001/XMLSchema#double">3.21E4</literal></binding>
    </result>
  </results>
</sparql>
"""

ASK_SRX = """\
<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head/>
  <boolean>true</boolean>
</sparql>
"""

EMPTY_SRX = """\
<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="x"/></head>
  <results/>
</sparql>
"""

TSV = '?s\t?o\n<http://example.org/alix>\t"4,4"\n<http://example.org/gus>\t2.2\n'

FILES = {
    "manifest.ttl": MANIFEST,
    "data.ttl": DATA,
    "star.rq": "PREFIX ex: <http://example.org/>\nSELECT * WHERE { ?s ex:score ?o }",
    "star.srx": STAR_SRX,
    "ask.rq": "ASK { ?s ?p ?o }",
    "ask.srx": ASK_SRX,
    "empty.rq": "SELECT ?x WHERE { ?x <http://example.org/none> ?y }",
    "empty.srx": EMPTY_SRX,
    "tsv.rq": "SELECT ?s ?o WHERE { ?s ?p ?o }",
    "tsv.tsv": TSV,
}


def convert(tmp_path: Path) -> converter.Converted:
    suite = tmp_path / "fixture"
    suite.mkdir(parents=True, exist_ok=True)
    for name, text in FILES.items():
        (suite / name).write_text(text, encoding="utf-8", newline="\n")
    return converter.convert_manifest(suite / "manifest.ttl")


def case(converted: converter.Converted, name: str) -> converter.Case:
    return next(c for c in converted.cases if c.name == name)


def test_cases_follow_the_manifest_order_and_left_out_tests_name_a_reason(tmp_path):
    converted = convert(tmp_path)
    assert [c.name for c in converted.cases] == ["select_star", "ask", "empty", "tsv"]
    assert converted.left_out == [
        ("named", "named graphs (qt:graphData)"),
        ("update", "UpdateEvaluationTest: not a query evaluation test"),
        ("withdrawn", "withdrawn"),
    ]


def test_data_becomes_full_iri_ntriples_with_lexical_forms_kept(tmp_path):
    data = case(convert(tmp_path), "select_star").data
    plain = [line for line in data if "_:" not in line and "source" not in line]
    assert plain == [
        "<http://example.org/alix> <http://example.org/knows> <http://example.org/gus> .",
        '<http://example.org/alix> <http://example.org/rate> "1.0"^^<http://www.w3.org/2001/XMLSchema#decimal> .',
        '<http://example.org/alix> <http://example.org/score> "3.21E4"^^<http://www.w3.org/2001/XMLSchema#double> .',
    ]


def test_relative_iris_resolve_against_the_published_suite(tmp_path):
    data = case(convert(tmp_path), "select_star").data
    published = (
        "<http://www.w3.org/2009/sparql/docs/tests/data-sparql11/fixture/data.ttl>"
    )
    assert (
        f"<http://example.org/alix> <http://example.org/source> {published} ." in data
    )


def test_conversion_is_the_same_on_every_run_and_machine(tmp_path):
    first, second = (
        converter.generate_gtest(convert(tmp_path / run), "fixture", "Fixture")
        for run in ("first", "second")
    )
    assert first == second
    assert "file:" not in first
    data = case(convert(tmp_path / "third"), "select_star").data
    labels = {word for line in data for word in line.split() if word.startswith("_:")}
    assert labels == {"_:b0", "_:b1"}


def test_select_rows_keep_duplicates_lexical_forms_and_query_column_order(tmp_path):
    expect = case(convert(tmp_path), "select_star").expect
    row = "  - ['<http://example.org/alix>', '\"3.21E4\"^^<http://www.w3.org/2001/XMLSchema#double>']"
    assert expect == ["terms:", row, row]


def test_ask_and_empty_results(tmp_path):
    converted = convert(tmp_path)
    assert case(converted, "ask").expect == ["rows:", "  - [true]"]
    assert case(converted, "empty").expect == ["empty: true"]


def test_tsv_cells_keep_commas_and_bare_numbers_are_typed(tmp_path):
    assert case(convert(tmp_path), "tsv").expect == [
        "terms:",
        "  - ['<http://example.org/alix>', '\"4,4\"']",
        "  - ['<http://example.org/gus>', '\"2.2\"^^<http://www.w3.org/2001/XMLSchema#decimal>']",
    ]


def test_generated_file_lists_left_out_tests_and_has_no_placeholder(tmp_path):
    text = converter.generate_gtest(convert(tmp_path), "fixture", "Fixture")
    assert "#   named: named graphs (qt:graphData)" in text
    assert "count: 1" not in text
    assert text.count("  - name: ") == 4
    assert "  # Select (star) with duplicates\n  - name: select_star\n" in text


def test_the_same_graph_written_in_another_order_gives_the_same_lines(tmp_path):
    # rdflib labels blank nodes in parse order, so only canonical labels make these agree
    prefix = "@prefix ex: <http://example.org/> .\n"
    written = {
        "first.ttl": prefix + '_:x ex:city "Amsterdam" . _:y ex:city "Utrecht" .\n',
        "second.ttl": prefix + '_:q ex:city "Utrecht" . _:p ex:city "Amsterdam" .\n',
    }
    for name, text in written.items():
        (tmp_path / name).write_text(text, encoding="utf-8", newline="\n")
    first, second = (converter.ntriples([tmp_path / name]) for name in written)
    assert first == second
