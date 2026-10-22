"""Result comparison logic for .gtest spec tests.

Mirrors the assertion helpers in ``grafeo-spec-tests/src/lib.rs`` so that the
Python runner validates results identically to the Rust runner.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Value to canonical string
# ---------------------------------------------------------------------------


def value_to_string(val: Any) -> str:
    """Convert a Python value (as returned by the grafeo bindings) to its
    canonical string representation for comparison.

    This must match the Rust ``value_to_string`` in
    ``grafeo-spec-tests/src/lib.rs``.
    """
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        # Rust's Display for f64 drops the ".0" suffix for whole numbers:
        #   format!("{}", 15.0_f64)  -> "15"
        #   format!("{}", 15.5_f64)  -> "15.5"
        # Python's str(15.0) produces "15.0", so we need to strip the
        # trailing ".0" when the value is integral.
        if val == int(val) and abs(val) < 2**53:
            return str(int(val))
        return str(val)
    if isinstance(val, list):
        inner = ", ".join(value_to_string(v) for v in val)
        return f"[{inner}]"
    if isinstance(val, dict):
        # Duration is returned as {months, days, nanos} from the C FFI.
        # Convert to ISO 8601 duration format to match the Rust runner.
        if set(val.keys()) == {"months", "days", "nanos"}:
            return _duration_to_iso(val["months"], val["days"], val["nanos"])
        entries = sorted(f"{k}: {value_to_string(v)}" for k, v in val.items())
        return "{" + ", ".join(entries) + "}"
    if isinstance(val, bytes):
        return f"bytes[{len(val)}]"
    # datetime, date, time: use str() which gives ISO format
    return str(val)


def _duration_to_iso(total_months: int, days: int, nanos: int) -> str:
    """Convert a duration from {months, days, nanos} to ISO 8601 format.

    Matches the Rust Display impl for Duration which produces P1Y2M3DT4H5M6S.
    """
    years, months = divmod(total_months, 12)
    hours, rem = divmod(nanos, 3_600_000_000_000)
    minutes, rem = divmod(rem, 60_000_000_000)
    seconds, sub_nanos = divmod(rem, 1_000_000_000)

    parts = ["P"]
    if years:
        parts.append(f"{years}Y")
    if months:
        parts.append(f"{months}M")
    if days:
        parts.append(f"{days}D")

    time_parts = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds or sub_nanos:
        if sub_nanos:
            # Format as decimal seconds
            frac = f"{sub_nanos:09d}".rstrip("0")
            time_parts.append(f"{seconds}.{frac}S")
        else:
            time_parts.append(f"{seconds}S")

    if time_parts:
        parts.append("T")
        parts.extend(time_parts)

    result = "".join(parts)
    return result if result != "P" else "P0D"


# ---------------------------------------------------------------------------
# Result -> rows of strings
# ---------------------------------------------------------------------------


def result_to_rows(result, columns: Optional[List[str]] = None) -> List[List[str]]:
    """Convert a grafeo QueryResult into rows of canonical strings.

    Each row is obtained by iterating the result (which yields dicts keyed by
    column name), then converting each value via ``value_to_string``.

    If *columns* is provided, values are extracted in that order.  Otherwise
    the column order from ``result.columns`` is used.
    """
    cols = columns or list(result.columns)
    rows: List[List[str]] = []

    # result[i] returns a dict for row i
    for i in range(len(result)):
        row_dict: Dict[str, Any] = result[i]
        row: List[str] = []
        for col in cols:
            row.append(value_to_string(row_dict.get(col)))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_rows_sorted(
    result,
    expected: List[List[str]],
    columns: Optional[List[str]] = None,
) -> None:
    """Assert that result rows match *expected* after sorting both sides."""
    actual = result_to_rows(result, columns)
    actual_sorted = sorted(actual)
    expected_sorted = sorted(expected)

    assert len(actual_sorted) == len(expected_sorted), (
        f"Row count mismatch: got {len(actual_sorted)} rows, "
        f"expected {len(expected_sorted)}\n"
        f"Actual:   {actual_sorted}\n"
        f"Expected: {expected_sorted}"
    )

    for i, (act, exp) in enumerate(zip(actual_sorted, expected_sorted)):
        assert len(act) == len(exp), (
            f"Column count mismatch at sorted row {i}: "
            f"got {len(act)} cols, expected {len(exp)}\n"
            f"Actual row:   {act}\n"
            f"Expected row: {exp}"
        )
        for j, (a, e) in enumerate(zip(act, exp)):
            assert a == e, (
                f"Mismatch at sorted row {i}, col {j}: "
                f"got '{a}', expected '{e}'\n"
                f"Full actual row:   {act}\n"
                f"Full expected row: {exp}"
            )


def assert_rows_ordered(
    result,
    expected: List[List[str]],
    columns: Optional[List[str]] = None,
) -> None:
    """Assert that result rows match *expected* in exact order."""
    actual = result_to_rows(result, columns)

    assert len(actual) == len(expected), (
        f"Row count mismatch: got {len(actual)} rows, "
        f"expected {len(expected)}\n"
        f"Actual:   {actual}\n"
        f"Expected: {expected}"
    )

    for i, (act, exp) in enumerate(zip(actual, expected)):
        assert len(act) == len(exp), (
            f"Column count mismatch at row {i}: "
            f"got {len(act)} cols, expected {len(exp)}\n"
            f"Actual row:   {act}\n"
            f"Expected row: {exp}"
        )
        for j, (a, e) in enumerate(zip(act, exp)):
            assert a == e, (
                f"Mismatch at row {i}, col {j}: "
                f"got '{a}', expected '{e}'\n"
                f"Full actual row:   {act}\n"
                f"Full expected row: {exp}"
            )


def assert_count(result, expected_count: int) -> None:
    """Assert that the result contains exactly *expected_count* rows."""
    actual = len(result)
    assert actual == expected_count, (
        f"Row count mismatch: got {actual}, expected {expected_count}"
    )


def assert_empty(result) -> None:
    """Assert that the result contains zero rows."""
    actual = len(result)
    assert actual == 0, f"Expected empty result, got {actual} row(s)"


def assert_error(exc: Exception, expected_substr: str) -> None:
    """Assert that the exception message contains *expected_substr*."""
    msg = str(exc)
    assert expected_substr in msg, f"Error '{msg}' does not contain '{expected_substr}'"


def assert_columns(result, expected_columns: List[str]) -> None:
    """Assert that result column names match *expected_columns* exactly."""
    actual = list(result.columns)
    assert actual == expected_columns, (
        f"Column mismatch: got {actual}, expected {expected_columns}"
    )


def assert_rows_with_precision(
    result,
    expected: List[List[str]],
    precision: int,
    columns: Optional[List[str]] = None,
) -> None:
    """Assert rows match with floating-point tolerance.

    Cells that parse as float on both sides are compared within
    ``10**(-precision)`` tolerance. All other cells use exact string match.
    """
    actual = result_to_rows(result, columns)
    tolerance = 10 ** (-precision)

    assert len(actual) == len(expected), (
        f"Row count mismatch: got {len(actual)}, expected {len(expected)}"
    )

    for i, (act_row, exp_row) in enumerate(zip(actual, expected)):
        assert len(act_row) == len(exp_row), (
            f"Column count mismatch at row {i}: "
            f"got {len(act_row)}, expected {len(exp_row)}"
        )
        for j, (a, e) in enumerate(zip(act_row, exp_row)):
            try:
                af, ef = float(a), float(e)
                assert abs(af - ef) < tolerance, (
                    f"Float mismatch at row {i}, col {j}: "
                    f"got {af}, expected {ef} (tolerance {tolerance})"
                )
            except (ValueError, TypeError):
                assert a == e, (
                    f"Mismatch at row {i}, col {j}: got '{a}', expected '{e}'"
                )


def assert_hash(
    result,
    expected_hash: str,
    columns: Optional[List[str]] = None,
) -> None:
    """Assert that the MD5 hash of sorted, pipe-delimited rows matches.

    Mirrors ``assert_hash`` in the Rust runner.
    """
    rows = result_to_rows(result, columns)
    rows.sort()

    hasher = hashlib.md5()
    for row in rows:
        hasher.update("|".join(row).encode())
        hasher.update(b"\n")

    actual_hash = hasher.hexdigest()
    assert actual_hash == expected_hash, (
        f"Hash mismatch: got '{actual_hash}', expected '{expected_hash}'\nRows: {rows}"
    )
