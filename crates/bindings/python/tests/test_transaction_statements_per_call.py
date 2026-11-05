"""A transaction statement through `db.execute` is refused, as the call has no lasting session.

`START TRANSACTION` used to open a transaction that ended with the call, the next write committed
at once, and a `ROLLBACK` found nothing to undo. Use `begin_transaction()`.
"""

import grafeo
import pytest


@pytest.mark.parametrize(
    "statement",
    ["START TRANSACTION", "START TRANSACTION READ ONLY", "COMMIT", "ROLLBACK"],
)
def test_a_transaction_statement_on_a_call_is_refused(statement):
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError, match="begin_transaction") as caught:
        db.execute(statement)
    assert caught.value.gql_status.startswith("25")


def test_the_refused_calls_leave_the_writes_around_them_alone():
    db = grafeo.GrafeoDB()
    with pytest.raises(grafeo.GrafeoError):
        db.execute("START TRANSACTION")
    db.execute("INSERT (:Person {name: 'Alix'})")
    with pytest.raises(grafeo.GrafeoError):
        db.execute("ROLLBACK")
    assert [row["p.name"] for row in db.execute("MATCH (p:Person) RETURN p.name")] == ["Alix"]


def test_begin_transaction_is_where_the_error_points():
    db = grafeo.GrafeoDB()
    tx = db.begin_transaction()
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.rollback()
    assert list(db.execute("MATCH (p:Person) RETURN p.name")) == []
    tx = db.begin_transaction()
    tx.execute("INSERT (:Person {name: 'Gus'})")
    tx.commit()
    assert [row["p.name"] for row in db.execute("MATCH (p:Person) RETURN p.name")] == ["Gus"]


def test_a_statement_that_only_looks_like_one_runs():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:COMMIT {word: 'ROLLBACK'})")
    assert len(list(db.execute("MATCH (n:COMMIT) RETURN n.word"))) == 1
