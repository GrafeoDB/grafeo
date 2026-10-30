"""A commit that fails ends its transaction, and the caller can tell.

When another transaction committed a write to the same node first, the
conflict shows at this transaction's commit. The commit fails with the
conflict (a GQLSTATUS of class 40); after it the transaction is no longer
active, and a rollback in the caller's except block does not raise over the
conflict. is_active said True and rollback raised "No active transaction".
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


def losing_transaction():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Account {id: 'alix', balance: 100})")
    first = db.begin_transaction()
    second = db.begin_transaction()
    first.execute("MATCH (a:Account) SET a.balance = 1")
    first.commit()
    second.execute("MATCH (a:Account) SET a.balance = 2")
    return db, second


def test_a_commit_that_loses_a_conflict_fails_with_class_40():
    db, second = losing_transaction()
    with pytest.raises(grafeo.GrafeoError, match="Write-write conflict") as caught:
        second.commit()
    assert caught.value.gql_status.startswith("40")
    assert db.execute("MATCH (a:Account) RETURN a.balance AS b").to_list() == [{"b": 1}]


def test_after_a_failed_commit_the_transaction_is_not_active():
    _, second = losing_transaction()
    with pytest.raises(grafeo.GrafeoError):
        second.commit()
    assert second.is_active is False


def test_a_rollback_after_a_failed_commit_does_not_hide_the_conflict():
    _, second = losing_transaction()
    try:
        second.commit()
    except grafeo.GrafeoError as error:
        conflict = error
        second.rollback()
    assert "Write-write conflict" in str(conflict)
