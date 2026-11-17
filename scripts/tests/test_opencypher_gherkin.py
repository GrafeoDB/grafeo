"""Tests for scripts/opencypher_gherkin.py on a feature shaped like the TCK's."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opencypher_gherkin as gherkin  # noqa: E402

FEATURE = '''\
#
# Copyright (c) Fixture
#

#encoding: utf-8

Feature: Fixture1 - A fixture

  Background:
    Given an empty graph

  Scenario: [1] Plain scenario
    And having executed:
      """
      CREATE (:A {name: 'Alix'})
      """
    When executing query:
      """
      MATCH (n)
        RETURN n
      """
    Then the result should be, in any order:
      | n                     |
      | (:A {name: 'Alix'})   |
      | 'a \\| b'             |
    And no side effects

  @ignore
  Scenario Outline: [2] Outline <what>
    When executing query:
      """
      RETURN <value> AS v
      """
    Then the result should be, in any order:
      | v       |
      | <value> |
    And no side effects

    Examples:
      | what   | value |
      | int    | 1     |
      | string | 'a'   |
'''


@pytest.fixture
def feature() -> gherkin.Feature:
    return gherkin.parse_feature(FEATURE)


def test_scenarios_in_file_order_with_outlines_expanded(feature):
    assert feature.name == "Fixture1 - A fixture"
    assert [s.name for s in feature.scenarios] == [
        "[1] Plain scenario",
        "[2] Outline int #1",
        "[2] Outline string #2",
    ]


def test_the_background_comes_first_in_every_scenario(feature):
    assert all(s.steps[0].text == "an empty graph" for s in feature.scenarios)


def test_docstrings_lose_their_common_indent_and_keep_their_own(feature):
    query = feature.scenarios[0].steps[2]
    assert query.text == "executing query:"
    assert query.docstring == "MATCH (n)\n  RETURN n"


def test_tables_split_on_unescaped_pipes(feature):
    table = feature.scenarios[0].steps[3].table
    assert table == [["n"], ["(:A {name: 'Alix'})"], ["'a | b'"]]


def test_outline_values_fill_texts_docstrings_and_tables(feature):
    first, second = feature.scenarios[1:]
    assert first.steps[1].docstring == "RETURN 1 AS v"
    assert second.steps[1].docstring == "RETURN 'a' AS v"
    assert second.steps[2].table == [["v"], ["'a'"]]


def test_tags_stay_on_the_scenario_they_precede(feature):
    assert [s.tags for s in feature.scenarios] == [[], ["@ignore"], ["@ignore"]]


def test_a_line_that_is_no_step_is_refused():
    with pytest.raises(ValueError):
        gherkin.parse_feature("Feature: F\n  Scenario: S\n    Whenever this\n")


def test_a_cell_undoes_gherkin_s_backslash_escapes_and_keeps_newline_escapes() -> None:
    # Gherkin escapes a backslash as \ and a pipe as \|; \n stays for the value parser
    row = gherkin.table_row(r"| 'a\\b' | 'x\|y' | 'Foo\nFoo' | '\'' |")
    assert row == [r"'a\b'", "'x|y'", r"'Foo\nFoo'", r"'\''"]
