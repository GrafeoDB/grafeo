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

import corpus  # noqa: E402
import difftest  # noqa: E402


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


def test_cases_do_not_write():
    # Each fixture is built once and shared by its cases, so a write would change
    # what later cases see.
    writes = re.compile(
        r"\b(INSERT|CREATE|SET|REMOVE|DELETE|MERGE|DROP)\b", re.IGNORECASE
    )
    for case in corpus.CASES:
        assert not writes.search(case.query), case.id
