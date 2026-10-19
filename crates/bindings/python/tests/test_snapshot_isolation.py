"""Snapshot isolation of values and deletes through the Python binding (#412).

The reproduction of issue #412 and its read-modify-write counter. On today's
store values are written in place and a delete takes effect at the deleting
transaction's start, so these fail: each is a strict ``xfail`` that fails as
soon as it passes, and then moves out of ``xfail``. The engine's full set of
guarantees is ``crates/grafeo-engine/tests/snapshot_isolation.rs``.
"""

import contextlib
import threading

import grafeo
import pytest
from grafeo import GrafeoDB

IN_PLACE = "#412: values are written in place, every reader sees an open transaction's write"
DELETE_AT_START = (
    "#412: a delete takes effect at the deleting transaction's start, not at its commit"
)


def only(result, column):
    """The value of ``column`` in the one row of ``result``."""
    rows = list(result)
    if len(rows) != 1:
        raise LookupError(f"expected one row, read {rows}")
    return rows[0][column]


def people():
    """Alix (30) knows Gus (40), committed."""
    db = GrafeoDB()
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 30})-[:KNOWS]->(:Person {name: 'Gus', age: 40})"
    )
    return db


def age(reader, name):
    return only(reader.execute(f"MATCH (p:Person {{name: '{name}'}}) RETURN p.age AS a"), "a")


def test_a_rollback_restores_the_value():
    db = people()
    tx = db.begin_transaction()
    tx.execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 99")
    tx.rollback()
    assert age(db, "Alix") == 30


def test_a_transaction_reads_its_own_write():
    db = people()
    tx = db.begin_transaction()
    tx.execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 99")
    assert age(tx, "Alix") == 99
    tx.rollback()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=IN_PLACE)
def test_another_session_does_not_read_an_uncommitted_value():
    db = people()
    tx = db.begin_transaction()
    tx.execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 99")
    try:
        assert age(db, "Alix") == 30
    finally:
        tx.rollback()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=IN_PLACE)
def test_a_transaction_reads_the_same_value_twice():
    db = people()
    tx = db.begin_transaction()
    first = age(tx, "Gus")
    db.execute("MATCH (n:Person {name: 'Gus'}) SET n.age = 41")
    second = age(tx, "Gus")
    tx.commit()
    assert (first, second) == (40, 40)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=DELETE_AT_START)
def test_another_session_still_counts_an_edge_an_open_transaction_deletes():
    db = people()
    tx = db.begin_transaction()
    tx.execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() DELETE r")
    try:
        assert only(db.execute("MATCH (:Person)-[r:KNOWS]->() RETURN count(r) AS c"), "c") == 1
    finally:
        tx.rollback()


def counter():
    db = GrafeoDB()
    db.execute("INSERT (:Counter {n: 0})")
    return db


def read_counter(reader):
    return only(reader.execute("MATCH (c:Counter) RETURN c.n AS n"), "n")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=IN_PLACE)
def test_an_aborted_increment_is_not_counted():
    db = counter()
    first = db.begin_transaction()
    first.execute("MATCH (c:Counter) SET c.n = 1")
    second = db.begin_transaction()
    seen = read_counter(second)
    first.rollback()
    second.execute(f"MATCH (c:Counter) SET c.n = {seen + 1}")
    second.commit()
    assert read_counter(db) == 1


def increment(db):
    """Reads the counter and writes it plus one in a transaction, retried on a conflict."""
    while True:
        tx = db.begin_transaction()
        try:
            n = read_counter(tx)
            tx.execute(f"MATCH (c:Counter) SET c.n = {n + 1}")
            tx.commit()
            return
        except grafeo.GrafeoError as error:
            if not error.is_retryable:
                raise
            if tx.is_active:
                with contextlib.suppress(grafeo.GrafeoError):
                    tx.rollback()


STEP_TIMEOUT = 10.0  # seconds a thread waits for a step of the other one


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=IN_PLACE)
def test_concurrent_increments_lose_and_add_nothing():
    """Two threads increment one counter in a fixed order of their steps, so
    the outcome does not depend on how the threads are scheduled (a free race
    rarely interleaves under the GIL and passed by chance). In each round:

    1. the writer begins, reads the counter and writes it plus one;
    2. the reader begins and reads the counter;
    3. the writer gives its attempt up (rolls back, as a conflict does);
    4. the reader writes what it read plus one, and commits;
    5. the writer increments again.

    Each round commits two increments. Under snapshot isolation the reader
    reads the committed value, so the counter ends at twice the rounds.
    Today's store writes the writer's value in place and the reader reads it,
    so each round adds three."""
    rounds = 25
    db = counter()
    events = ("written", "read", "given_up", "committed", "incremented")
    steps = [{name: threading.Event() for name in events} for _ in range(rounds)]
    errors = []

    def wait(step, name):
        if not step[name].wait(STEP_TIMEOUT):
            raise TimeoutError(f"the other thread did not reach `{name}`")

    def fail(error):
        errors.append(error)
        for step in steps:
            for event in step.values():
                event.set()

    def writer():
        try:
            for step in steps:
                tx = db.begin_transaction()
                n = read_counter(tx)
                tx.execute(f"MATCH (c:Counter) SET c.n = {n + 1}")
                step["written"].set()
                wait(step, "read")
                tx.rollback()
                step["given_up"].set()
                wait(step, "committed")
                increment(db)
                step["incremented"].set()
        except Exception as error:  # noqa: BLE001 (raised below)
            fail(error)

    def reader():
        try:
            for step in steps:
                wait(step, "written")
                tx = db.begin_transaction()
                n = read_counter(tx)
                step["read"].set()
                wait(step, "given_up")
                tx.execute(f"MATCH (c:Counter) SET c.n = {n + 1}")
                tx.commit()
                step["committed"].set()
                wait(step, "incremented")
        except Exception as error:  # noqa: BLE001 (raised below)
            fail(error)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    assert read_counter(db) == 2 * rounds
