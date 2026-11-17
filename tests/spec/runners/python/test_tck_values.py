"""The TCK value notation parser: every kind of cell, and the cells it must refuse."""

import json
import math
from pathlib import Path as FilePath

import grafeo
import pytest
from comparator import temporals_as_text
from parser import _parse_line_based, _parse_with_yaml, param_value
from tck_values import Float, Node, Path, Relationship, matches, parse, to_neutral


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", 42),
        ("-7", -7),
        ("1.0", Float(1.0)),
        ("1e-3", Float(0.001)),
        ("-Inf", Float(-math.inf)),
        ("'Alix'", "Alix"),
        ("'it\\'s'", "it's"),
        ("'a\\\\b'", "a\\b"),
        ("''", ""),
        ("true", True),
        ("false", False),
        ("null", None),
        ("[]", []),
        ("[1, 'a', [null]]", [1, "a", [None]]),
        ("{}", {}),
        ("{name: 'Gus', `odd key`: 1.5}", {"name": "Gus", "odd key": Float(1.5)}),
    ],
)
def test_scalars_and_containers(text, value):
    assert parse(text) == value


def test_an_integer_and_a_float_stay_apart():
    # Python's 1 == 1.0, so a float cell is a Float and never equals the integer
    assert parse("1") != parse("1.0")
    assert (
        parse("true") is True and parse("1") == 1 and not isinstance(parse("1"), bool)
    )


def test_nan_is_a_float():
    value = parse("NaN")
    assert isinstance(value, Float) and math.isnan(value.value)


def test_nodes_and_relationships():
    assert parse("()") == Node(frozenset())
    assert parse("(:A:B {p: 1})") == Node(frozenset({"A", "B"}), {"p": 1})
    assert parse("(:B:A {p: 1})") == parse("(:A:B {p: 1})")
    assert parse("[:KNOWS]") == Relationship("KNOWS")
    assert parse("[:T {since: 2020}]") == Relationship("T", {"since": 2020})


def test_paths_keep_order_and_direction():
    path = parse("<(:A)-[:T]->(:B)<-[:U]-()>")
    assert isinstance(path, Path)
    assert [sorted(n.labels) for n in path.nodes] == [["A"], ["B"], []]
    assert [r.type for r in path.relationships] == ["T", "U"]
    assert path.forward == [True, False]
    assert parse("<()>") == Path([Node(frozenset())], [], [])


@pytest.mark.parametrize(
    "text", ["'open", "[1, 2", "1 2", "(:A", "{k 1}", "", "<(:A)-[:T]-(:B)>"]
)
def test_malformed_cells_are_refused(text):
    with pytest.raises(ValueError):
        parse(text)


CASE_FILE = json.loads(
    (FilePath(__file__).resolve().parents[1] / "values_cases.json").read_text(
        encoding="utf-8"
    )
)


def native(actual, db):
    """A neutral value as the binding returns it: a node, a relationship and a path as
    dicts, a ``cypher`` value from the engine itself."""
    if actual is None or isinstance(actual, bool):
        return actual
    if isinstance(actual, list):
        return [native(a, db) for a in actual]
    ((kind, inner),) = actual.items()
    if kind == "int":
        return int(inner)
    if kind == "float":
        return {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}.get(
            inner, None
        ) or float(inner)
    if kind == "string":
        return inner
    if kind == "map":
        return {k: native(v, db) for k, v in inner.items()}
    if kind == "node":
        props = {k: native(v, db) for k, v in inner["properties"].items()}
        return {"_id": inner["id"], "_labels": inner["labels"], **props}
    if kind == "relationship":
        props = {k: native(v, db) for k, v in inner["properties"].items()}
        ends = {"_source": inner["source"], "_target": inner["target"]}
        return {"_id": inner["id"], "_type": inner["type"], **ends, **props}
    if kind == "path":
        nodes = [native({"node": n}, db) for n in inner["nodes"]]
        edges = [native({"relationship": r}, db) for r in inner["relationships"]]
        return {"nodes": nodes, "edges": edges}
    if kind == "cypher":
        return list(db.execute_cypher(f"RETURN {inner} AS v"))[0]["v"]
    raise ValueError(f"unknown neutral kind {kind}")


@pytest.mark.parametrize(
    "case",
    CASE_FILE["cases"],
    ids=lambda c: f"{c['cell']} vs {json.dumps(c['actual'])}",
)
def test_the_shared_value_cases(case):
    db = grafeo.GrafeoDB()
    value = native(case["actual"], db)
    if case.get("dialect") == "tck":
        value = temporals_as_text(db, value)
    verdict = matches(
        parse(case["cell"]),
        to_neutral(value),
        unordered_lists=case.get("unordered_lists", False),
    )
    # A known limitation, or a gap of this binding, gives the opposite verdict until fixed
    gap = "limitation" in case or "python" in case.get("runner_gaps", {})
    assert verdict is (case["match"] != gap)


@pytest.mark.parametrize("trip", CASE_FILE["round_trips"], ids=lambda t: t["cypher"])
def test_the_shared_round_trips(trip):
    db = grafeo.GrafeoDB()
    value = native({"cypher": trip["cypher"]}, db)
    holds = temporals_as_text(db, value) == trip["text"]
    # A strict expected failure where the gap names this binding
    assert holds is ("python" not in trip["runner_gaps"])


@pytest.mark.parametrize("case", CASE_FILE["params"], ids=lambda c: c["param"])
def test_the_shared_param_cases(case):
    value = param_value(case["param"])
    assert (
        (value is None) if case["value"] is None else value == case["value"]["string"]
    )


GAP_FILE = """meta:
  language: cypher
  values_dialect: tck

tests:
  - name: one
    runner_gap:
      runners: [python, dart]
      issue: fix in 0.6.4
      reason: "temporals"
    query: RETURN 1 AS n
    expect:
      values: |
        | n |
        | 1 |
      side_effects:
        +nodes: 2
"""


@pytest.mark.parametrize("parse_file", [_parse_with_yaml, _parse_line_based])
def test_the_tck_keys_parse_in_both_parsers(parse_file):
    case = parse_file(GAP_FILE, FilePath("x.gtest")).tests[0]
    assert case.runner_gap.runners == ["python", "dart"]
    assert (case.runner_gap.issue, case.runner_gap.reason) == (
        "fix in 0.6.4",
        "temporals",
    )
    assert case.expect.values == "| n |\n| 1 |"
    assert case.expect.side_effects == [("+nodes", 2)]


@pytest.mark.parametrize("parse_file", [_parse_with_yaml, _parse_line_based])
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("python, dart", "python, cobol"),
        ("      issue: fix in 0.6.4\n", ""),
        ("+nodes: 2", "+edges: 2"),
    ],
)
def test_wrong_tck_keys_are_refused_in_both_parsers(parse_file, old, new):
    with pytest.raises(ValueError):
        parse_file(GAP_FILE.replace(old, new), FilePath("x.gtest"))
