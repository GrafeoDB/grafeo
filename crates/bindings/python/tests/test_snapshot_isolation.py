"""Snapshot isolation of values and deletes through the Python binding (#412).

The reproduction of issue #412 and its read-modify-write counter. On today's
store values are written in place and a delete takes effect at the deleting
transaction's start, so these fail: each is a strict ``xfail`` that fails as
soon as it passes, and then moves out of ``xfail``. The engine's full set of
guarantees is ``crates/grafeo-engine/tests/snapshot_isolation.rs``.
"""

import contextlib
import sys
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=IN_PLACE)
def test_concurrent_increments_lose_and_add_nothing():
    """Four threads increment one counter 100 times each, in twenty rounds:
    a single round of today's store can end at 400 by chance, twenty in a row
    do not. A query holds the GIL, so the threads switch between statements
    only with a short switch interval."""
    finals = []
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            db = counter()
            errors = []

            def run(db=db, errors=errors):
                try:
                    for _ in range(100):
                        increment(db)
                except Exception as error:  # noqa: BLE001 (raised below)
                    errors.append(error)

            threads = [threading.Thread(target=run) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            finals.append(read_counter(db))
    finally:
        sys.setswitchinterval(interval)
    assert finals == [400] * 20
