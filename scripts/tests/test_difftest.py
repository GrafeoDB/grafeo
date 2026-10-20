"""Tests for scripts/difftest (the comparison and the corpus; no grafeo build needed).

Run: uv run --with pytest python -m pytest scripts/tests
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "difftest"))

import corpus
import difftest


def rows(*values, ordered=False, columns=("x",)):
    return {
        "query": "q",
        "ordered": ordered,
        "columns": list(columns),
        "rows": [list(v) for v in values],
    }


def error(text):
    return {"query": "q", "ordered": False, "error": text}


def write_results(path: Path, results: dict, version="0.0.0") -> Path:
    payload = {"meta": {"version": version, "commit": None}, "results": results}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- normalize ---------------------------------------------------------------


def test_normalize_makes_floats_comparable():
    assert difftest.normalize(0.1 + 0.2) == 0.3
    assert difftest.normalize(float("nan")) == {"float": "NaN"}
    assert difftest.normalize(float("-inf")) == {"float": "-Infinity"}
    assert difftest.normalize(True) is True


def test_normalize_sorts_map_keys_and_keeps_list_order():
    value = {"b": [3, 1], "a": (2.5, None)}
    assert difftest.normalize(value) == {"a": [2.5, None], "b": [3, 1]}
    assert list(difftest.normalize(value)) == ["a", "b"]


def test_normalize_uses_an_entities_dict_form():
    class Node:
        def to_dict(self):
            return {"name": "Alix", "id": 0}

    assert difftest.normalize(Node()) == {"type": "Node", "id": 0, "name": "Alix"}


# --- canonical and differing --------------------------------------------------


def test_unordered_rows_compare_as_a_multiset():
    assert difftest.canonical(rows([1], [2])) == difftest.canonical(rows([2], [1]))
    assert difftest.canonical(rows([1], [1])) != difftest.canonical(rows([1]))


def test_ordered_rows_compare_in_order():
    first = rows([1], [2], ordered=True)
    second = rows([2], [1], ordered=True)
    assert difftest.canonical(first) != difftest.canonical(second)


def test_a_value_that_changes_type_differs():
    old = {"A1|gql": rows([1]), "A2|gql": rows([1]), "A3|gql": rows([True])}
    new = {"A1|gql": rows([1.0]), "A2|gql": rows([True]), "A3|gql": rows([True])}
    assert difftest.differing(old, new) == ["A1|gql", "A2|gql"]


def test_differing_lists_changed_cases_in_corpus_order():
    old = {
        "A10|gql": rows([1]),
        "A2|gql": rows([1]),
        "B1|cypher": rows([1]),
        "C1|gql": rows([1]),
    }
    new = {
        "A10|gql": rows([2]),
        "A2|gql": error("boom"),
        "B1|cypher": rows([1]),
        "D1|gql": rows(),
    }
    assert difftest.differing(old, new) == ["A2|gql", "A10|gql"]


# --- the reviewed list --------------------------------------------------------


def test_read_reviewed_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "reviewed.txt"
    path.write_text(
        "# 0.5.44 against 0.5.43\n\nA1|gql 0123456789 Edges keep their values (#482)\n",
        encoding="utf-8",
    )
    assert difftest.read_reviewed(path) == {
        "A1|gql": ("0123456789", "Edges keep their values (#482)")
    }


@pytest.mark.parametrize(
    "text",
    [
        "A1|gql 0123456789\n",  # no reason
        "A1 0123456789 no language\n",
        "A1|gql 0123456789 one\nA1|gql 0123456789 two\n",  # listed twice
    ],
)
def test_read_reviewed_rejects_malformed_lines(tmp_path, text):
    path = tmp_path / "reviewed.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        difftest.read_reviewed(path)


def gate_files(tmp_path, reviewed_lines):
    old = write_results(
        tmp_path / "old.json", {"A1|gql": rows([1]), "A2|gql": rows([5])}
    )
    new = write_results(
        tmp_path / "new.json", {"A1|gql": rows([2]), "A2|gql": rows([5])}
    )
    reviewed = tmp_path / "reviewed.txt"
    reviewed.write_text(
        "".join(f"{line}\n" for line in reviewed_lines), encoding="utf-8"
    )
    return old, new, reviewed


def test_compare_passes_when_every_difference_is_reviewed(tmp_path, capsys):
    digest = difftest.fingerprint(rows([2]))
    old, new, reviewed = gate_files(tmp_path, [f"A1|gql {digest} A1 now returns 2"])
    assert difftest.compare(old, new, reviewed) == 0
    assert "all reviewed" in capsys.readouterr().out


def test_compare_fails_on_a_difference_nobody_reviewed(tmp_path, capsys):
    old, new, reviewed = gate_files(tmp_path, [])
    assert difftest.compare(old, new, reviewed) == 1
    out = capsys.readouterr().out
    assert "[NOT REVIEWED]" in out
    assert f"to review: A1|gql {difftest.fingerprint(rows([2]))}" in out


def test_compare_fails_when_a_reviewed_result_changed_since(tmp_path, capsys):
    digest = difftest.fingerprint(rows([3]))
    old, new, reviewed = gate_files(
        tmp_path, [f"A1|gql {digest} A1 returned 3 when reviewed"]
    )
    assert difftest.compare(old, new, reviewed) == 1
    assert "[CHANGED SINCE THE REVIEW]" in capsys.readouterr().out


def test_compare_fails_when_a_reviewed_difference_is_gone(tmp_path, capsys):
    digest = difftest.fingerprint(rows([2]))
    old, new, reviewed = gate_files(
        tmp_path,
        [f"A1|gql {digest} A1 now returns 2", f"A2|gql {digest} A2 changed once"],
    )
    assert difftest.compare(old, new, reviewed) == 1
    assert "A2|gql [NO LONGER DIFFERENT]" in capsys.readouterr().out


def test_a_varying_case_passes_whether_or_not_it_differs(tmp_path, capsys):
    """A `~` line: the old build's result varies, so the case may also not differ."""
    digest = difftest.fingerprint(rows([2]))
    varies = difftest.fingerprint(rows([5]))
    old, new, reviewed = gate_files(
        tmp_path,
        [
            f"A1|gql ~{digest} A1 now returns 2",
            f"A2|gql ~{varies} A2 varies in the old build",
        ],
    )
    assert difftest.compare(old, new, reviewed) == 0
    out = capsys.readouterr().out
    assert "=== A1|gql [reviewed]" in out
    assert "A2|gql" not in out
    assert "all reviewed" in out


@pytest.mark.parametrize("key", ["A1|gql", "A2|gql"])  # A1 differs, A2 does not
def test_a_varying_case_fails_when_its_new_result_changed(tmp_path, capsys, key):
    """A `~` line still checks the new result, also when the old build agrees."""
    reviewed_a1 = f"A1|gql {difftest.fingerprint(rows([2]))} A1 now returns 2"
    varying = f"{key} ~{difftest.fingerprint(rows([3]))} returned 3 when reviewed"
    lines = [varying] if key == "A1|gql" else [reviewed_a1, varying]
    old, new, reviewed = gate_files(tmp_path, lines)
    assert difftest.compare(old, new, reviewed) == 1
    assert f"=== {key} [CHANGED SINCE THE REVIEW]" in capsys.readouterr().out


def test_compare_without_a_reviewed_list_only_reports(tmp_path, capsys):
    old, new, _ = gate_files(tmp_path, [])
    assert difftest.compare(old, new) == 0
    out = capsys.readouterr().out
    assert "=== A1|gql [changed]" in out
    assert "REVIEW" not in out


# --- parity --------------------------------------------------------------------


def test_parity_reports_languages_that_disagree(tmp_path, capsys):
    results = {
        "A1|gql": rows([1], columns=("n",)),
        "A1|cypher": rows([1], columns=("a.n",)),  # column names may differ
        "A2|gql": rows([1]),
        "A2|cypher": rows([2]),
        "A3|gql": error("gql error"),
        "A3|cypher": error("cypher error"),  # both fail: they agree
        "A4|gql": rows([1]),
        "A4|cypher": error("unsupported"),
        "A5|gql": rows([1]),  # GQL only
    }
    path = write_results(tmp_path / "results.json", results)
    assert difftest.parity(path) == 1
    out = capsys.readouterr().out
    assert "=== A2:" in out and "=== A4:" in out
    assert "=== A1:" not in out and "=== A3:" not in out
    assert "2 cases where GQL and Cypher disagree" in out


# --- the corpus ------------------------------------------------------------------


def test_case_ids_are_unique():
    ids = [case.id for case in corpus.CASES]
    assert len(ids) == len(set(ids))


def test_cases_name_known_languages_and_fixtures():
    for case in corpus.CASES:
        assert case.languages and set(case.languages) <= corpus.LANGUAGES, case.id
        assert case.fixture in corpus.FIXTURES, case.id
        assert re.fullmatch(r"[A-Z]+\d+", case.id), case.id


WRITES = re.compile(r"\b(INSERT|CREATE|SET|REMOVE|DELETE|MERGE|DROP)\b", re.IGNORECASE)
# A quoted string, with backslash escapes: query text inside one is a value.
STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def writes_outside_strings(query: str) -> bool:
    return WRITES.search(STRING_LITERAL.sub("''", query)) is not None


def test_a_write_keyword_counts_only_outside_a_string():
    assert writes_outside_strings("MATCH (n) DETACH DELETE n")
    assert writes_outside_strings("CALL p('x') YIELD v SET v.a = 1")
    assert not writes_outside_strings(
        r"CALL p('it\'s\' DETACH DELETE n') YIELD v RETURN v"
    )
    assert not writes_outside_strings('RETURN "CREATE (n)" AS text')


def test_only_the_write_fixtures_have_writing_cases():
    # Each fixture is built once and shared by its cases, so a write would change
    # what later cases see. The write fixtures (`corpus.WRITE_FIXTURES`) are the
    # exception: databases of their own, whose cases run in corpus order and read
    # what the writes before them left. A write keyword inside a string literal is
    # a value (BZ2 passes query text as an argument that must run nothing).
    for case in corpus.CASES:
        if case.fixture in corpus.WRITE_FIXTURES:
            continue
        assert not writes_outside_strings(case.query), case.id


def test_a_write_fixture_serves_only_its_sections():
    # A write fixture's cases read what the writes before them left, so a case of
    # another section on it would see those writes: each one is used by the
    # sections it names only.
    for case in corpus.CASES:
        sections = corpus.WRITE_FIXTURES.get(case.fixture)
        if sections is not None:
            section = re.match(r"[A-Z]+", case.id).group()
            assert section in sections, (case.id, case.fixture)
    for fixture in corpus.WRITE_FIXTURES:
        assert fixture in corpus.FIXTURES, fixture
