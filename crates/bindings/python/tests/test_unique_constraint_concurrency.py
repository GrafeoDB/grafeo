"""A UNIQUE value written by two concurrent transactions commits once.

The #400 reporter's insert-if-absent from eight Python threads, and the
errors a second writer of a key gets: the retryable write conflict
(GRAFEO-T001) while the first writer's transaction is open, the graph type
violation (G2000) once it committed. The engine's tests of every write path
are ``crates/grafeo-engine/tests/unique_constraint_claims.rs``.
"""

import threading

import pytest
from grafeo import GrafeoDB, GrafeoError

CONSTRAINT = "CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE"


def keyed():
    db = GrafeoDB()
    db.execute(CONSTRAINT)
    return db


def count(db, key):
    rows = list(db.execute(f"MATCH (m:Memory {{key: '{key}'}}) RETURN count(m) AS c"))
    return rows[0]["c"]


def test_a_second_writer_gets_a_conflict_then_a_violation():
    db = keyed()
    holder = db.begin_transaction()
    holder.execute("INSERT (:Memory {key: 'ssi'})")
    with pytest.raises(GrafeoError) as raised:
        db.execute("INSERT (:Memory {key: 'ssi'})")
    assert raised.value.error_code == "GRAFEO-T001"
    assert count(db, "ssi") == 0
    holder.commit()
    with pytest.raises(GrafeoError) as raised:
        db.execute("INSERT (:Memory {key: 'ssi'})")
    assert raised.value.gql_status == "G2000"
    assert count(db, "ssi") == 1


def test_a_rollback_frees_the_key():
    db = keyed()
    holder = db.begin_transaction()
    holder.execute("INSERT (:Memory {key: 'ssi'})")
    holder.rollback()
    db.execute("INSERT (:Memory {key: 'ssi'})")
    assert count(db, "ssi") == 1


def test_eight_threads_insert_if_absent_commit_once():
    db = keyed()
    barrier = threading.Barrier(8)
    results = [None] * 8

    def insert_if_absent(slot):
        barrier.wait()
        tx = db.begin_transaction()
        try:
            found = list(tx.execute("MATCH (m:Memory {key: 'ssi'}) RETURN m.key AS k"))
            if found:
                tx.rollback()
                results[slot] = "found"
                return
            tx.execute("INSERT (:Memory {key: 'ssi'})")
            tx.commit()
            results[slot] = "committed"
        except GrafeoError as error:
            results[slot] = (error.error_code, error.gql_status)

    threads = [threading.Thread(target=insert_if_absent, args=(slot,)) for slot in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count("committed") == 1, results
    for result in results:
        assert (
            result in ("committed", "found") or result[0] == "GRAFEO-T001" or result[1] == "G2000"
        ), results
    assert count(db, "ssi") == 1
