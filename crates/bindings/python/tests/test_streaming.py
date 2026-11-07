"""Tests for db.execute_lazy() cursor-based query streaming.

Focus:
- Iterator protocol yields the same rows as db.execute()
- Early break drops the stream cleanly (subsequent queries still work)
- Columns are exposed before iteration starts
- Empty results behave like normal empty iterators
- Non-streamable queries (mutations, ORDER BY, session cmds, EXPLAIN) raise
- GIL is released during chunk pulls: other Python threads make progress
"""

import threading
import time

import pytest

pytestmark = pytest.mark.gql


@pytest.fixture
def people_db(db):
    """Fresh in-memory DB seeded with five Person nodes."""
    for name, age in (
        ("Alix", 32),
        ("Gus", 28),
        ("Vincent", 45),
        ("Jules", 40),
        ("Mia", 24),
    ):
        db.create_node(["Person"], {"name": name, "age": age})
    return db


def _rows_as_dict_set(result):
    """Convert an iterable of dicts into a frozenset of sorted tuples for set comparison."""
    return frozenset(tuple(sorted(row.items())) for row in result)


def test_streaming_matches_materialized(people_db):
    query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
    materialized = list(people_db.execute(query))
    streamed = list(people_db.execute_lazy(query))

    assert len(materialized) == len(streamed) == 5
    assert _rows_as_dict_set(materialized) == _rows_as_dict_set(streamed)


def test_streaming_yields_dicts_with_column_keys(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5
    assert all(isinstance(row, dict) for row in rows)
    assert all("p.name" in row for row in rows)


def test_stream_columns_exposed_before_iteration(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
    assert stream.columns == ["name", "age"]


def test_streaming_filter(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name"))
    names = {row["name"] for row in rows}
    assert names == {"Alix", "Vincent", "Jules"}


def test_streaming_empty_result(people_db):
    rows = list(people_db.execute_lazy("MATCH (p:Person) WHERE p.age > 999 RETURN p.name"))
    assert rows == []


def test_streaming_early_break(people_db):
    # Pull one row, break, then make sure subsequent queries still work.
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    first = next(iter(stream))
    assert "p.name" in first
    del stream

    # Subsequent query on the same DB works.
    rows = list(people_db.execute("MATCH (p:Person) RETURN p.name"))
    assert len(rows) == 5


def test_streaming_rejects_mutation(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("INSERT (:Person {name: 'Butch'})")
    msg = str(exc_info.value).lower()
    assert "mutat" in msg or "cannot be streamed" in msg or "execute() instead" in msg


def test_streaming_rejects_order_by(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("MATCH (p:Person) RETURN p.name AS n ORDER BY n")
    msg = str(exc_info.value).lower()
    assert "push" in msg or "cannot be streamed" in msg


def test_streaming_rejects_session_command(db):
    with pytest.raises(Exception) as exc_info:
        db.execute_lazy("SESSION SET GRAPH analytics")
    assert "session" in str(exc_info.value).lower()


def test_streaming_rejects_explain(people_db):
    with pytest.raises(Exception) as exc_info:
        people_db.execute_lazy("EXPLAIN MATCH (p:Person) RETURN p.name")
    msg = str(exc_info.value).lower()
    assert "explain" in msg or "cannot be streamed" in msg


def test_streaming_repr(people_db):
    stream = people_db.execute_lazy("MATCH (p:Person) RETURN p.name")
    assert "ResultStream" in repr(stream)


def test_concurrent_iteration_does_not_deadlock(people_db):
    """Two threads iterating separate streams must both complete without
    deadlocking. This is the reliably-observable consequence of GIL release
    in __next__: if GIL release were missing, the Rust mutex taken by a
    running __next__ on one thread could stall the other thread indefinitely
    once pyo3 tried to reacquire Python state.

    A stricter "timing-based" GIL-release test (comparing parallel vs serial
    iteration time) is intentionally avoided: on an in-memory store each row
    pull is sub-microsecond, so the per-row detach window is shorter than
    OS thread-switch granularity, and the signal is lost in scheduling noise.
    """
    row_count = 2_000
    for i in range(row_count):
        people_db.create_node(["Widget"], {"index": i})

    results = {"a": 0, "b": 0}
    errors: list[Exception] = []

    def iterate(key: str) -> None:
        try:
            for _row in people_db.execute_lazy("MATCH (w:Widget) RETURN w.index"):
                results[key] += 1
        except Exception as exc:  # noqa: BLE001 - want any failure here
            errors.append(exc)

    threads = [
        threading.Thread(target=iterate, args=("a",), daemon=True),
        threading.Thread(target=iterate, args=("b",), daemon=True),
    ]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    elapsed = time.monotonic() - start

    assert not errors, f"concurrent iteration failed: {errors}"
    assert all(not t.is_alive() for t in threads), "threads did not finish (deadlock?)"
    assert results == {"a": row_count, "b": row_count}, f"incomplete iteration: {results}"
    # Sanity cap: two threads of 2k rows should finish in far under 30s;
    # blowing past that points at GIL starvation even without strict timing.
    assert elapsed < 20.0, f"concurrent iteration took {elapsed:.2f}s"


def test_a_lazy_stream_of_a_temporary_session_reads_every_row(people_db):
    """The session goes when the expression ends; outside a transaction its stream reads on."""
    rows = list(people_db.session().execute_lazy("MATCH (p:Person) RETURN p.name"))
    assert sorted(row["p.name"] for row in rows) == ["Alix", "Gus", "Jules", "Mia", "Vincent"]


def test_a_lazy_stream_reads_on_after_its_session_closes(people_db):
    """Closing the session ends its statements, not a stream it opened outside a transaction."""
    session = people_db.session()
    stream = session.execute_lazy("MATCH (p:Person) RETURN p.name")
    first = next(iter(stream))
    session.close()
    rest = list(stream)
    names = sorted([first["p.name"], *(row["p.name"] for row in rest)])
    assert names == ["Alix", "Gus", "Jules", "Mia", "Vincent"]
