"""pytest plugin that discovers and runs .gtest spec files through the Python bindings.

Drop this file (or the containing ``runners/python`` package) on the pytest
search path and every ``.gtest`` file under ``tests/spec/`` becomes a set of
pytest test items.

Each test case gets a **fresh** ``GrafeoDB()`` instance, loads the declared
dataset, runs setup queries, executes the query, and asserts the expected
result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the runner modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comparator import (
    assert_columns,
    assert_count,
    assert_empty,
    assert_error,
    assert_hash,
    assert_rows_ordered,
    assert_rows_sorted,
    assert_rows_with_precision,
)
from parser import GtestFile, TestCase, coerce_params, parse_gtest_file

# ---------------------------------------------------------------------------
# Grafeo availability
# ---------------------------------------------------------------------------

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

# Languages that require a separate execute method
_LANGUAGE_METHODS: dict[str, str] = {
    "gql": "execute",
    "cypher": "execute_cypher",
    "gremlin": "execute_gremlin",
    "graphql": "execute_graphql",
    "sparql": "execute_sparql",
    "sql-pgq": "execute_sql",
    "sql_pgq": "execute_sql",
}

# Capabilities declared by this runner (not compiled features, but runner-level support)
_RUNNER_CAPABILITIES = {"int64-safe"}

# Cached set of compiled feature flags from db.info()["features"]
_compiled_features: set | None = None


def _get_compiled_features() -> set:
    """Return the set of features compiled into the grafeo binary."""
    global _compiled_features
    if _compiled_features is None:
        db = grafeo.GrafeoDB()
        info = db.info()
        _compiled_features = set(info.get("features", []))
        db.close()
    return _compiled_features


# Repo root: tests/spec/runners/python/conftest.py -> five .parent calls
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


def pytest_collect_file(parent, file_path):
    """Discover .gtest files for collection."""
    if file_path.suffix == ".gtest":
        return GtestFileCollector.from_parent(parent, path=file_path)
    return None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class GtestFileCollector(pytest.File):
    """Collects test items from a single .gtest file."""

    def collect(self):
        try:
            gtest_file = parse_gtest_file(self.path)
        except Exception as exc:
            raise pytest.UsageError(f"Failed to parse {self.path}: {exc}") from exc

        for tc in gtest_file.tests:
            if tc.variants:
                # Rosetta: one item per variant language
                for lang, query in tc.variants.items():
                    item_name = f"{tc.name}[{lang}]"
                    yield GtestItem.from_parent(
                        self,
                        name=item_name,
                        gtest_file=gtest_file,
                        test_case=tc,
                        variant_lang=lang,
                        variant_query=query,
                    )
            else:
                yield GtestItem.from_parent(
                    self,
                    name=tc.name,
                    gtest_file=gtest_file,
                    test_case=tc,
                    variant_lang=None,
                    variant_query=None,
                )


# ---------------------------------------------------------------------------
# Test item
# ---------------------------------------------------------------------------


class GtestItem(pytest.Item):
    """A single test case from a .gtest file."""

    def __init__(
        self,
        name: str,
        parent: GtestFileCollector,
        gtest_file: GtestFile,
        test_case: TestCase,
        variant_lang: str | None,
        variant_query: str | None,
    ):
        super().__init__(name, parent)
        self.gtest_file = gtest_file
        self.test_case = test_case
        self.variant_lang = variant_lang
        self.variant_query = variant_query

    def runtest(self):
        tc = self.test_case
        meta = self.gtest_file.meta

        # Skip if grafeo is not importable
        if not GRAFEO_AVAILABLE:
            pytest.skip("grafeo Python package not installed")

        # Skip if test has a skip field
        if tc.skip is not None:
            pytest.skip(f"skipped in .gtest: {tc.skip}")

        # Determine language: per-test override > rosetta variant > file meta
        language = self.variant_lang or tc.language or meta.language or "gql"

        # Check requires: skip if the compiled build lacks the feature
        all_requires = list(meta.requires) + list(tc.requires)
        if language not in ("gql", "") and language not in all_requires:
            all_requires.append(language)

        features = _get_compiled_features()
        for req in all_requires:
            key = req.replace("_", "-")
            if key == "gql":
                continue
            # Compound language keys: "graphql-rdf" requires both "graphql" and "rdf"
            if key == "graphql-rdf":
                if "graphql" not in features or "rdf" not in features:
                    pytest.skip(f"grafeo build does not include '{key}' feature")
            elif key not in features and key not in _RUNNER_CAPABILITIES:
                pytest.skip(f"grafeo build does not include '{key}' feature")

        # Fresh database per test
        db = grafeo.GrafeoDB()

        # Load dataset (per-test override takes priority over file-level)
        effective_dataset = tc.dataset or meta.dataset
        if effective_dataset and effective_dataset != "empty":
            _load_dataset(db, effective_dataset)

        # Run setup queries in the file's declared language
        setup_language = meta.language or "gql"
        for setup_q in tc.setup:
            _execute(db, setup_language, setup_q)

        # Determine query / statements
        query = self.variant_query or tc.query
        statements = tc.statements

        if statements:
            queries = list(statements)
        elif query or tc.expect.error is not None:
            queries = [query or ""]
        else:
            pytest.fail(f"No query or statements in test '{tc.name}' in {self.path}")
            return  # unreachable, but keeps linters happy

        expect = tc.expect

        # Type the params (every statement gets them)
        params = coerce_params(tc.params)

        # Error tests
        if expect.error is not None:
            self._run_error_test(db, language, queries, expect.error, params)
            return

        # Execute all-but-last as fire-and-forget (with params, so
        # INSERT/SET statements can reference $param variables).
        for q in queries[:-1]:
            _execute(db, language, q, params)

        # Last query: capture result (with params if present)
        result = _execute(db, language, queries[-1], params)

        # Column assertion (checked before value assertions)
        if expect.columns:
            assert_columns(result, expect.columns)

        # Value assertions
        if expect.empty:
            assert_empty(result)
        elif expect.count is not None:
            assert_count(result, expect.count)
        elif expect.hash is not None:
            assert_hash(result, expect.hash)
        elif expect.rows:
            if expect.precision is not None:
                assert_rows_with_precision(result, expect.rows, expect.precision)
            elif expect.ordered:
                assert_rows_ordered(result, expect.rows)
            else:
                assert_rows_sorted(result, expect.rows)
        # If none of the above, the test just checks the query does not error

    def _run_error_test(
        self,
        db,
        language: str,
        queries: list[str],
        expected_substr: str,
        params=None,
    ) -> None:
        """Execute queries expecting the last one to raise an error."""
        for q in queries[:-1]:
            _execute(db, language, q, params)

        # Last query should fail (with params if present)
        try:
            _execute(db, language, queries[-1], params)
        # Any error the binding raises counts (a GrafeoError, or a TypeError
        # or ValueError from converting a parameter): the message decides.
        except Exception as exc:  # noqa: BLE001
            assert_error(exc, expected_substr)
        else:
            pytest.fail(
                f"Expected error containing '{expected_substr}' but query succeeded"
            )

    def repr_failure(self, excinfo):
        """Provide a readable failure message."""
        if isinstance(excinfo.value, AssertionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, None, f"{self.path.name}::{self.name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(db, dataset_name: str) -> None:
    """Load a .setup dataset file into the database using GQL."""
    dataset_path = _REPO_ROOT / "tests" / "spec" / "datasets" / f"{dataset_name}.setup"
    if not dataset_path.exists():
        pytest.fail(f"Dataset file not found: {dataset_path}")

    content = dataset_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        db.execute(trimmed)


def _execute(db, language: str, query: str, params=None):
    """Execute a query in the specified language, returning the QueryResult."""
    method_name = _LANGUAGE_METHODS.get(language)
    if method_name is not None:
        method = getattr(db, method_name, None)
        if method is None:
            pytest.skip(
                f"grafeo build does not support language '{language}' "
                f"(no {method_name} method)"
            )
        if params is not None:
            return method(query, params)
        return method(query)
    # Fall back to generic execute_language for non-standard languages
    method = getattr(db, "execute_language", None)
    if method is None:
        pytest.skip(f"grafeo build does not support language '{language}'")
    if params is not None:
        return method(language, query, params)
    return method(language, query)
