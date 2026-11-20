"""Tests for scripts/convert_opencypher_tck.py on scenarios shaped like the TCK's."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_opencypher_tck as converter  # noqa: E402
import opencypher_gherkin as gherkin  # noqa: E402

HEADER = "#\n# Copyright (c) Fixture\n# Licensed under the Apache License, Version 2.0\n#\n\n"


def scenario(steps: str) -> gherkin.Scenario:
    text = "Feature: F\n\n  Scenario: [1] Fixture\n" + steps
    return gherkin.parse_feature(text).scenarios[0]


def cases(steps: str, graphs: dict | None = None) -> list[converter.Case]:
    return converter.convert_scenario(scenario(steps), graphs or {})


RESULT = '''\
    Given an empty graph
    And having executed:
      """
      CREATE (:A {name: 'Alix'})
      """
    When executing query:
      """
      MATCH (n) RETURN n
      """
    Then the result should be, in any order:
      | n                   |
      | (:A {name: 'Alix'}) |
    And no side effects
'''


def test_a_result_becomes_a_values_table_and_no_side_effects_an_empty_map():
    (case,) = cases(RESULT)
    assert case.name == "s1_fixture"
    assert case.setup == ["CREATE (:A {name: 'Alix'})"]
    assert case.query == "MATCH (n) RETURN n"
    assert case.expect == [
        "values: |",
        "  | n |",
        "  | (:A {name: 'Alix'}) |",
        "side_effects: {}",
    ]


def test_order_and_list_order_flags():
    steps = RESULT.replace(
        "in any order:", "in order (ignoring element order for lists):"
    )
    (case,) = cases(steps)
    assert case.expect[:2] == ["ordered: true", "unordered_lists: true"]


def test_side_effects_become_counts():
    steps = '''\
    Given an empty graph
    When executing query:
      """
      CREATE (:A)
      """
    Then the result should be empty
    And the side effects should be:
      | +nodes  | 1 |
      | +labels | 1 |
'''
    (case,) = cases(steps)
    assert case.expect == [
        "empty: true",
        "side_effects:",
        "  +nodes: 1",
        "  +labels: 1",
    ]


@pytest.mark.parametrize(
    ("error", "status", "comments"),
    [
        ("a TypeError should be raised at compile time: InvalidArgumentType", "42", 1),
        ("a TypeError should be raised at runtime: InvalidArgumentType", "22", 1),
        ("a SyntaxError should be raised at compile time: UndefinedVariable", "42", 1),
        (
            "a ConstraintVerificationFailed should be raised at runtime: DeleteConnectedNode",
            "G1",
            1,
        ),
        ("a EntityNotFound should be raised at runtime: DeletedEntityAccess", "22", 2),
        ("a TypeError should be raised at any time: *", "22", 1),
    ],
)
def test_errors_map_to_their_gqlstatus_class(error, status, comments):
    (case,) = cases(
        f'    When executing query:\n      """\n      RETURN x\n      """\n    Then {error}\n'
    )
    assert case.expect == [f'gql_status: "{status}"', "side_effects: {}"]
    assert len(case.comments) == comments and case.comments[0].startswith("TCK: ")


def test_parameters_keep_their_types():
    steps = '''\
    And parameters are:
      | s | 'Gus'        |
      | i | 1            |
      | f | 1.5          |
      | l | [1, 'a']     |
      | n | null         |
    When executing query:
      """
      RETURN $s
      """
    Then the result should be empty
'''
    (case,) = cases(steps)
    assert case.params == [
        ("s", '"Gus"'),
        ("i", "1"),
        ("f", "1.5"),
        ("l", '[1, "a"]'),
        ("n", "null"),
    ]


def test_a_procedure_becomes_a_gql_procedure_over_its_table():
    steps = '''\
    And there exists a procedure test.my.proc(in :: INTEGER?) :: (out :: STRING?):
      | in | out   |
      | 1  | 'one' |
    When executing query:
      """
      CALL test.my.proc(1)
      """
    Then the result should be empty
'''
    (case,) = cases(steps)
    assert case.setup_gql == [
        "CREATE PROCEDURE `test.my.proc`(in INT64) RETURNS (out STRING) AS "
        "{ UNWIND [{p0: 1, p1: 'one'}] AS r "
        "FILTER (r.p0 = $in OR (r.p0 IS NULL AND $in IS NULL)) RETURN r.p1 AS out }"
    ]


def test_a_void_procedure_finishes_without_columns():
    steps = "    And there exists a procedure test.doNothing() :: ():\n      |\n"
    steps += (
        '    When executing query:\n      """\n      CALL test.doNothing()\n      """\n'
    )
    steps += "    Then the result should be empty\n"
    (case,) = cases(steps)
    assert case.setup_gql == [
        "CREATE PROCEDURE `test.doNothing`() RETURNS () AS { UNWIND [] AS r FINISH }"
    ]


def test_a_control_query_is_a_second_case_after_the_query():
    steps = '''\
    Given the tree graph
    When executing query:
      """
      CREATE (:B)
      """
    Then the result should be empty
    When executing control query:
      """
      MATCH (n:B) RETURN count(n) AS c
      """
    Then the result should be, in any order:
      | c |
      | 1 |
'''
    first, control = cases(steps, {"tree": "CREATE (:T)"})
    assert first.setup == ["CREATE (:T)"] and first.query == "CREATE (:B)"
    assert control.name == "s1_fixture_control"
    assert control.setup == ["CREATE (:T)", "CREATE (:B)"]
    assert control.query.startswith("MATCH (n:B)")


def test_an_unknown_step_or_a_bad_cell_stops_the_conversion():
    with pytest.raises(ValueError):
        cases("    When something new happens\n")
    with pytest.raises(ValueError):
        cases(RESULT.replace("(:A {name: 'Alix'}) |", "(:A {name: 'Alix' |"))


FEATURE = (
    HEADER
    + "Feature: Fixture1 - Fixture\n\n  Scenario: [1] First\n"
    + RESULT
    + "\n  @ignore\n  Scenario: [2] Second\n"
    + RESULT
    + "\n  Scenario: [3] Third\n"
    + RESULT
)


def convert(tmp_path: Path, known: dict, text: str = FEATURE) -> str | None:
    features = tmp_path / "features"
    path = features / "clauses" / "Fixture1.feature"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return converter.convert_feature(path, features, {}, known, "abc123")


def test_a_file_keeps_the_licence_and_names_what_it_left_out(tmp_path):
    known = {"s3_third": {"reason": "out-of-scope", "note": "why"}}
    text = convert(tmp_path, known)
    assert text.startswith(
        "#\n# Copyright (c) Fixture\n# Licensed under the Apache License"
    )
    assert (
        "# Source: https://github.com/opencypher/openCypher at tag 2024.3 (abc123)"
        in text
    )
    assert "#   [2] Second: tagged @ignore upstream" in text
    assert "#   [3] Third: out-of-scope: why" in text
    assert "  values_dialect: tck" in text
    assert text.count("  - name: ") == 1


def test_a_known_failure_with_an_issue_is_a_strict_xfail(tmp_path):
    text = convert(tmp_path, {"s1_first": {"issue": "fix in 0.6.4"}})
    xfail = '    xfail:\n      spec: "openCypher TCK 2024.3 Fixture1: [1] First"\n      issue: fix in 0.6.4\n'
    assert "  - name: s1_first\n" + xfail in text


def test_a_known_failure_can_pass_in_runners_that_cannot_see_it(tmp_path):
    known = {"s1_first": {"issue": "fix in 0.6.4", "passes_in": ["node", "wasm"]}}
    text = convert(tmp_path, known)
    assert "      issue: fix in 0.6.4\n      passes_in: [node, wasm]\n" in text


def test_stale_or_incomplete_known_failures_stop_the_conversion(tmp_path):
    for known in ({"s9_missing": {"issue": "#1"}}, {"s1_first": {"note": "no issue"}}):
        with pytest.raises(ValueError):
            convert(tmp_path, known)


def test_a_feature_without_scenarios_gives_no_file(tmp_path):
    assert convert(tmp_path, {}, HEADER + "Feature: Fixture1 - Empty\n") is None


def temporal_feature(query: str) -> str:
    steps = RESULT.replace("MATCH (n) RETURN n", query)
    return HEADER + "Feature: Fixture1 - Fixture\n\n  Scenario: [1] First\n" + steps


@pytest.mark.parametrize(
    ("query", "marked"),
    [
        ("RETURN date('1984-10-11') AS n", True),
        ("RETURN datetime.fromepoch(1, 2) AS n", True),
        ("MATCH (n) RETURN n", False),
        ("CREATE ({tagline: 'There comes a time...'})", False),
    ],
)
def test_a_case_that_builds_a_temporal_carries_the_runner_gap(tmp_path, query, marked):
    text = convert(tmp_path, {}, temporal_feature(query))
    assert ("    runner_gap:\n      runners: [wasm]\n" in text) is marked


ERROR_FEATURE = (
    HEADER
    + "Feature: Fixture1 - Fixture\n\n  Scenario: [1] First\n"
    + "    Given any graph\n    When executing query:\n"
    + '      """\n      MATCH (n) RETURN n SKIP $s\n      """\n'
    + "    Then a SyntaxError should be raised at compile time: NegativeIntegerArgument\n"
)


def test_a_gql_class_replaces_the_tck_class_and_records_the_disagreement(tmp_path):
    known = {
        "s1_first": {"gql_class": "22", "note": "a parameter is checked at run time"}
    }
    text = convert(tmp_path, known, ERROR_FEATURE)
    assert '      gql_status: "22"\n' in text
    assert (
        "  # GQL class 22, the TCK expects 42: a parameter is checked at run time\n"
        in text
    )
    assert "xfail" not in text


def test_a_gql_class_can_still_be_a_known_failure(tmp_path):
    known = {"s1_first": {"gql_class": "22", "note": "why", "issue": "fix in 0.6.5"}}
    text = convert(tmp_path, known, ERROR_FEATURE)
    assert '      gql_status: "22"\n' in text and "      issue: fix in 0.6.5\n" in text


def test_a_gql_class_needs_an_error_case_and_a_note(tmp_path):
    with pytest.raises(ValueError):
        convert(tmp_path, {"s1_first": {"gql_class": "22", "note": "why"}})
    with pytest.raises(ValueError):
        convert(tmp_path, {"s1_first": {"gql_class": "22"}}, ERROR_FEATURE)


def test_a_table_without_rows_carries_no_gap(tmp_path):
    # Every binding names the columns of a result without rows, so no runner
    # skips such a case.
    row = "      | (:A {name: 'Alix'}) |\n"
    text = convert(
        tmp_path, {}, temporal_feature("MATCH (n) RETURN n").replace(row, "")
    )
    assert "runner_gap" not in text
    assert "      values: |\n        | n |\n" in text
    # A temporal case keeps the temporal gap
    temporal = temporal_feature("RETURN date('1984-10-11') AS n").replace(row, "")
    assert "runners: [wasm]" in convert(tmp_path, {}, temporal)


def test_a_listed_value_is_replaced_in_the_query_and_the_rows(monkeypatch):
    digest = converter.hashlib.sha256(b"'Vincent'").hexdigest()
    monkeypatch.setattr(converter, "NEUTRAL_VALUES", {digest: "'Grafeo'"})
    (case,) = cases(
        RESULT.replace(
            "MATCH (n) RETURN n", "MATCH (n) SET n.name = 'Vincent' RETURN n"
        ).replace("{name: 'Alix'}) |\n    And", "{name: 'Vincent'}) |\n    And")
    )
    assert "'Vincent'" not in case.query + "\n".join(case.expect)
    assert "SET n.name = 'Grafeo'" in case.query
    assert "(:A {name: 'Grafeo'})" in "\n".join(case.expect)
    assert "CREATE (:A {name: 'Alix'})" in case.setup[0]
    assert "a product name in the TCK's data is replaced by 'Grafeo'" in case.comments


def test_the_neutral_values_are_keyed_by_hash():
    for digest, value in converter.NEUTRAL_VALUES.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest) and value == "'Grafeo'"
