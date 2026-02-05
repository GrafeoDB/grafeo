"""Test utilities and helpers.

This module provides common utilities for testing.
"""

import pytest
from functools import wraps
from typing import Callable, Any


# Check if grafeo is available
try:
    import grafeo
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


def skip_if_unavailable(feature_name: str = "grafeo"):
    """Decorator to skip tests if a feature is not available.

    Args:
        feature_name: Name of the feature to check

    Usage:
        @skip_if_unavailable("grafeo")
        def test_something(self, db):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if feature_name == "grafeo" and not GRAFEO_AVAILABLE:
                pytest.skip(f"{feature_name} not installed")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assert_row_count(result, expected: int, message: str = None):
    """Assert that a query result has expected number of rows.

    Args:
        result: Query result (iterable)
        expected: Expected row count
        message: Optional assertion message
    """
    rows = list(result)
    msg = message or f"Expected {expected} rows, got {len(rows)}"
    assert len(rows) == expected, msg
    return rows


def assert_contains_values(rows: list, key: str, expected_values: set, message: str = None):
    """Assert that rows contain expected values for a key.

    Args:
        rows: List of row dictionaries
        key: Key to check in each row
        expected_values: Set of expected values
        message: Optional assertion message
    """
    actual = {r.get(key) for r in rows}
    msg = message or f"Expected {expected_values}, got {actual}"
    assert expected_values.issubset(actual), msg


def extract_values(rows: list, key: str) -> list:
    """Extract values for a key from all rows.

    Args:
        rows: List of row dictionaries
        key: Key to extract

    Returns:
        List of values
    """
    return [r.get(key) for r in rows]


def get_first_value(result, key: str = None) -> Any:
    """Get the first value from a query result.

    Args:
        result: Query result (iterable)
        key: Optional key to extract from first row

    Returns:
        First row or first row's key value
    """
    rows = list(result)
    if not rows:
        return None
    if key:
        return rows[0].get(key)
    return rows[0]


class QueryResult:
    """Wrapper for query results with helpful assertion methods."""

    def __init__(self, result):
        self.rows = list(result)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def count(self) -> int:
        return len(self.rows)

    def values(self, key: str) -> list:
        """Extract values for a key from all rows."""
        return [r.get(key) for r in self.rows]

    def first(self, key: str = None) -> Any:
        """Get first row or first row's key value."""
        if not self.rows:
            return None
        if key:
            return self.rows[0].get(key)
        return self.rows[0]

    def assert_count(self, expected: int, message: str = None):
        """Assert row count."""
        msg = message or f"Expected {expected} rows, got {len(self.rows)}"
        assert len(self.rows) == expected, msg
        return self

    def assert_contains(self, key: str, *values):
        """Assert that rows contain expected values for a key."""
        actual = set(self.values(key))
        expected = set(values)
        assert expected.issubset(actual), f"Expected {expected} in {actual}"
        return self

    def assert_not_contains(self, key: str, *values):
        """Assert that rows do not contain specified values for a key."""
        actual = set(self.values(key))
        forbidden = set(values)
        overlap = actual & forbidden
        assert not overlap, f"Found unexpected values {overlap}"
        return self


def format_query_value(value: Any) -> str:
    """Format a Python value for use in a query string.

    Args:
        value: Value to format

    Returns:
        Formatted string suitable for query
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Escape single quotes
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        items = [format_query_value(v) for v in value]
        return f"[{', '.join(items)}]"
    else:
        return repr(value)
