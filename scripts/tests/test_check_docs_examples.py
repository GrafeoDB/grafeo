"""The docs example runner: block extraction, statement splitting, checking."""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_docs_examples import (  # noqa: E402
    MarkerError,
    check_statement,
    extract_blocks,
    main,
    query_kind,
    run_python_page,
    split_statements,
)

try:
    import grafeo
except ImportError:  # the policy job has no wheel
    grafeo = None

needs_grafeo = pytest.mark.skipif(
    grafeo is None or not hasattr(grafeo.GrafeoDB, "explain_sql"),
    reason="grafeo with the EXPLAIN entry points is not installed",
)
FIXTURES = Path(__file__).parent / "fixtures" / "docs_examples"


def test_finds_indented_fences():
    text = '=== "Python"\n\n    ```python\n    x = 1\n    ```\n'
    assert [b.code for b in extract_blocks(text)] == ["x = 1"]


def test_finds_fences_in_admonitions_and_tabs_with_tabs():
    text = "!!! example\n\n\t```gql\n\tMATCH (n)\n\tRETURN n\n\t```\n"
    block = extract_blocks(text)[0]
    assert block.code == "MATCH (n)\nRETURN n"
    assert block.lang == "gql"
    assert block.line == 3


def test_crlf_page():
    assert extract_blocks("```python\r\nx = 1\r\n```\r\n")[0].code == "x = 1"


def test_skip_needs_reason():
    with pytest.raises(MarkerError):
        extract_blocks("<!-- docs-check: skip -->\n```python\nx\n```\n")


def test_skip_marks_the_next_block_only():
    text = "<!-- docs-check: skip needs a server -->\n```python\nx\n```\n\n```python\ny\n```\n"
    first, second = extract_blocks(text)
    assert first.skip == "needs a server"
    assert second.skip is None


def test_all_skipped_page_is_reported(tmp_path, capsys):
    shutil.copy(FIXTURES / "all_skipped.md", tmp_path / "all_skipped.md")
    report = tmp_path / "r.json"
    assert main([str(tmp_path), "--report", str(report)]) == 1
    page = json.loads(report.read_text(encoding="utf-8"))["pages"][0]
    assert page["ran"] == 0
    assert len(page["skipped"]) == 1
    assert "ran 0" in capsys.readouterr().out


def test_split_statements_shares_sparql_prefix():
    code = "PREFIX ex: <http://e/>\nSELECT * WHERE { ?s ex:p ?o }\n\nSELECT * WHERE { ?s ex:q ?o }"
    parts = split_statements(code, "sparql")
    assert len(parts) == 2
    assert all(s.startswith("PREFIX ex:") for s in parts)


def test_split_on_semicolons_and_comments():
    code = "-- all\nMATCH (n) RETURN n;\nMATCH (m) RETURN m"
    assert split_statements(code, "gql") == ["MATCH (n) RETURN n", "MATCH (m) RETURN m"]


def test_gremlin_comments_dropped():
    assert split_statements("// all people\ng.V().hasLabel('Person')", "gremlin") == [
        "g.V().hasLabel('Person')"
    ]


def test_query_kind_by_page():
    assert query_kind("model-and-query/gql/patterns.md", "sql") == "gql"
    assert query_kind("model-and-query/sql-pgq/index.md", "sql") == "sql"
    assert query_kind("languages/rust/index.md", "rust") is None
    assert query_kind("a.md", "cypher") == "cypher"


@needs_grafeo
def test_syntax_error_fails_missing_state_passes():
    db = grafeo.GrafeoDB()
    assert check_statement(db, "gql", "MATCH (n:Person RETURN n") is not None
    assert check_statement(db, "gql", "USE GRAPH missing MATCH (n) RETURN n") is None
    assert check_statement(db, "gql", "MATCH (n:Person) RETURN n") is None


@needs_grafeo
def test_refused_statements_run_for_real():
    db = grafeo.GrafeoDB()
    assert check_statement(db, "gql", "CREATE NODE TYPE Person (name STRING)") is None
    assert check_statement(db, "gql", "START TRANSACTION") is None
    assert check_statement(db, "gql", "CREATE NODE TYPE (") is not None


@needs_grafeo
def test_each_language_is_checked():
    db = grafeo.GrafeoDB()
    assert check_statement(db, "cypher", "MATCH (n RETURN n") is not None
    assert check_statement(db, "sparql", "SELECT * WHERE { ?s ?p ?o }") is None
    assert check_statement(db, "gremlin", "g.V((") is not None


def test_hidden_setup_runs_first():
    text = (
        "```python\nassert base == 40\n```\n\n<!-- docs-check: setup\nbase = 40\n-->\n"
    )
    assert run_python_page(extract_blocks(text)) == []


def test_python_error_carries_page_line():
    text = "intro\n\n```python\nx = 1\nraise ValueError('boom')\n```\n"
    errors = run_python_page(extract_blocks(text))
    assert len(errors) == 1
    assert errors[0].startswith("line 5:")
    assert "boom" in errors[0]


def test_blocks_share_state_in_order():
    text = "```python\nx = 1\n```\n\n```python\nassert x == 1\n```\n"
    assert run_python_page(extract_blocks(text)) == []


def test_main_exit_codes(tmp_path):
    shutil.copy(FIXTURES / "ok.md", tmp_path / "ok.md")
    assert main([str(tmp_path)]) == 0
    shutil.copy(FIXTURES / "bad_python.md", tmp_path / "bad_python.md")
    assert main([str(tmp_path)]) == 1
    assert main([str(tmp_path), "--section", "ok.md"]) == 0


@needs_grafeo
def test_query_page_reports_the_bad_block(tmp_path):
    shutil.copy(FIXTURES / "queries.md", tmp_path / "queries.md")
    report = tmp_path / "r.json"
    assert main([str(tmp_path), "--report", str(report)]) == 1
    page = json.loads(report.read_text(encoding="utf-8"))["pages"][0]
    assert page["checked"] == 2
    assert len(page["failures"]) == 1


def test_sparql_semicolon_is_predicate_list_syntax():
    code = "SELECT ?name WHERE {\n  ?p a :Person ;\n      :name ?name .\n}"
    assert split_statements(code, "sparql") == [code]


def test_semicolon_inside_a_string_does_not_split():
    code = "MATCH (n) WHERE n.name = 'a;b' RETURN n"
    assert split_statements(code, "gql") == [code]
    assert split_statements(code, "cypher") == [code]


def test_comment_only_fragments_are_dropped():
    code = "CREATE GRAPH shared TYPED social.social_network;     -- qualified reference\nMATCH (n) RETURN n"
    assert split_statements(code, "gql") == [
        "CREATE GRAPH shared TYPED social.social_network",
        "MATCH (n) RETURN n",
    ]
    assert split_statements("// c\nMATCH (n) RETURN n // after", "cypher") == [
        "MATCH (n) RETURN n // after"
    ]
    assert split_statements("// c\nMATCH (n) RETURN n", "gql") == ["MATCH (n) RETURN n"]
    assert split_statements("-- only a comment", "gql") == []
    assert split_statements("MATCH (a)--(b) RETURN a", "gql") == [
        "MATCH (a)--(b) RETURN a"
    ]


def test_sparql_hash_comment_lines_not_iri_fragments():
    code = "# all\nSELECT * WHERE { ?s <http://e/#x> ?o }"
    assert split_statements(code, "sparql") == [
        "SELECT * WHERE { ?s <http://e/#x> ?o }"
    ]


@needs_grafeo
def test_leading_explain_is_stripped():
    db = grafeo.GrafeoDB()
    assert check_statement(db, "sparql", "EXPLAIN SELECT * WHERE { ?s ?p ?o }") is None
    assert (
        check_statement(
            db,
            "sql",
            "EXPLAIN ANALYZE SELECT * FROM GRAPH_TABLE (g MATCH (n) COLUMNS (n.x AS x))",
        )
        is None
    )
    assert check_statement(db, "sparql", "EXPLAIN SELECT * WHERE { ?s ?p") is not None
    assert check_statement(db, "gql", "PROFILE MATCH (n) RETURN n") is None


@needs_grafeo
def test_q002_passes_only_for_missing_state():
    db = grafeo.GrafeoDB()
    assert check_statement(db, "gql", "USE GRAPH missing MATCH (n) RETURN n") is None
    assert check_statement(db, "gql", "MATCH (n) RETURN m") is not None
    assert check_statement(db, "gql", "MATCH (n) RETURN foo(n)") is not None
    assert check_statement(db, "gql", "MATCH (n) RETURN abs(1, 2)") is not None


def test_page_with_only_skipped_checkable_blocks_fails(tmp_path):
    shutil.copy(FIXTURES / "skipped_python_and_text.md", tmp_path / "p.md")
    report = tmp_path / "r.json"
    assert main([str(tmp_path), "--report", str(report)]) == 1
    page = json.loads(report.read_text(encoding="utf-8"))["pages"][0]
    assert "checkable" in page["failures"][0]["error"]


def test_skipped_python_beside_a_running_block_passes(tmp_path):
    (tmp_path / "p.md").write_bytes(
        b"```python\nx = 1\n```\n\n<!-- docs-check: skip server -->\n```python\nconnect()\n```\n"
    )
    assert main([str(tmp_path)]) == 0
