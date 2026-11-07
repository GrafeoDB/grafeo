"""Query warnings in Python: `result.warnings` and `warnings.warn` (0.6.2).

A deprecated form returns its rows with a warning on `result.warnings` (code, kind, message), and
the binding raises each through `warnings.warn` with `DeprecationWarning` for a deprecation, from
the caller's line. Under `-W error` the warning raises after the statement has run, and a write
outside a transaction has then committed (documented). The codes and the forms across every
binding are in `tests/spec/common/warnings.gtest`; this file holds what is Python's own.
"""

import subprocess
import sys
import warnings

import grafeo
import pytest

ZONED = "ZONED DATETIME '2024-01-01T10:00:00+01:00'"


def test_a_result_lists_its_warnings():
    db = grafeo.GrafeoDB()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = db.execute(f"RETURN {ZONED} AS z, [1] + [2] AS l")
    found = [(w.code, w.kind) for w in result.warnings]
    assert found == [("01X01", "deprecation"), ("01X02", "deprecation")]
    for warning in result.warnings:
        assert "0.7.0" in warning.message, warning.message


def test_a_warning_is_raised_as_a_deprecation_warning_from_the_caller():
    db = grafeo.GrafeoDB()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = db.execute(f"RETURN {ZONED} AS v")
    assert [w.category for w in caught] == [DeprecationWarning]
    assert str(caught[0].message) == result.warnings[0].message
    assert caught[0].filename == __file__, caught[0].filename


def test_each_warning_is_raised_once():
    db = grafeo.GrafeoDB()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db.execute(f"RETURN {ZONED} AS a, {ZONED} AS b, [1] + [2] AS c")
    assert len(caught) == 2, [str(w.message) for w in caught]


def test_no_warning_is_an_empty_list_and_nothing_raised():
    db = grafeo.GrafeoDB()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = db.execute("RETURN [1] || [2] AS v")
    assert result.warnings == []
    assert caught == []


def test_a_warning_is_frozen():
    db = grafeo.GrafeoDB()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warning = db.execute(f"RETURN {ZONED} AS v").warnings[0]
    with pytest.raises((AttributeError, TypeError)):
        warning.code = "00000"


SCRIPT = f"""
import grafeo
db = grafeo.GrafeoDB()
try:
    db.execute("INSERT (:T {{z: {ZONED}}})")
    print("no raise")
except DeprecationWarning as warning:
    print("raised:", warning)
count = list(db.execute("MATCH (n:T) RETURN count(n) AS n"))[0]["n"]
print("count:", count)
"""


def test_w_error_raises_after_the_write_has_committed():
    out = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "raised:" in out, out
    assert "0.7.0" in out, out
    assert "count: 1" in out, out
