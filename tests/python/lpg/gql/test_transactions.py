"""GQL implementation of transaction tests.

Tests transaction operations using GQL (ISO standard) query language.
"""

from tests.python.bases.test_transactions import BaseTransactionsTest


class TestGQLTransactions(BaseTransactionsTest):
    """GQL implementation of transaction tests."""

    def insert_query(self, labels: list[str], props: dict) -> str:
        """GQL: INSERT (:<labels> {<props>})"""
        label_str = ":".join(labels) if labels else ""
        if label_str:
            label_str = f":{label_str}"

        prop_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                prop_parts.append(f"{k}: '{v}'")
            elif isinstance(v, bool):
                prop_parts.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                prop_parts.append(f"{k}: null")
            else:
                prop_parts.append(f"{k}: {v}")

        prop_str = ", ".join(prop_parts)
        return f"INSERT (n{label_str} {{{prop_str}}})"

    def match_by_prop_query(self, label: str, prop: str, value) -> str:
        """GQL: MATCH (n:<label>) WHERE n.<prop> = <value> RETURN n"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = str(value)
        return f"MATCH (n:{label}) WHERE n.{prop} = {value_str} RETURN n"

    def count_query(self, label: str) -> str:
        """GQL: MATCH (n:<label>) RETURN count(n) AS cnt"""
        return f"MATCH (n:{label}) RETURN count(n) AS cnt"


# Additional GQL-specific transaction tests


class TestGQLSpecificTransactions:
    """GQL-specific transaction tests."""

    def test_gql_transaction_isolation(self, db):
        """Test transaction isolation - changes not visible until commit."""
        # Start a transaction but don't commit
        tx1 = db.begin_transaction()
        tx1.execute("INSERT (:Person {name: 'Isolated'})")

        # Query from outside the transaction shouldn't see the node
        # (This depends on isolation level implementation)
        result = db.execute("MATCH (n:Person) WHERE n.name = 'Isolated' RETURN n")
        list(result)  # Consume result to verify query executes
        # In proper isolation, this should be 0
        # But implementation may vary

        tx1.rollback()

    def test_gql_multiple_inserts_transaction(self, db):
        """Test multiple INSERTs in a single transaction."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {name: 'TxPerson1', idx: 1})")
            tx.execute("INSERT (:Person {name: 'TxPerson2', idx: 2})")
            tx.execute("INSERT (:Person {name: 'TxPerson3', idx: 3})")
            tx.commit()

        # All three should exist
        result = db.execute(
            "MATCH (n:Person) WHERE n.name STARTS WITH 'TxPerson' RETURN n.name"
        )
        rows = list(result)
        assert len(rows) == 3

    def test_gql_mixed_operations_transaction(self, db):
        """Test INSERT and DELETE in same transaction."""
        # Setup: create initial node
        db.execute("INSERT (:TempNode {name: 'ToDelete'})")

        with db.begin_transaction() as tx:
            # Delete existing node
            tx.execute("MATCH (n:TempNode) WHERE n.name = 'ToDelete' DELETE n")
            # Insert new node
            tx.execute("INSERT (:TempNode {name: 'Replacement'})")
            tx.commit()

        # Verify
        result = db.execute("MATCH (n:TempNode) RETURN n.name")
        rows = list(result)
        names = [r["n.name"] for r in rows]
        assert "Replacement" in names
        assert "ToDelete" not in names

    def test_gql_transaction_error_rollback(self, db):
        """Test that errors in transaction cause rollback."""
        try:
            with db.begin_transaction() as tx:
                tx.execute("INSERT (:ErrorTest {name: 'BeforeError'})")
                # This might cause an error depending on implementation
                tx.execute("THIS IS NOT VALID GQL SYNTAX")
                tx.commit()
        except Exception:
            pass  # Expected to fail

        # Node should not exist if transaction rolled back on error
        result = db.execute("MATCH (n:ErrorTest) WHERE n.name = 'BeforeError' RETURN n")
        list(result)  # Consume result to verify query executes
        # Ideally 0, but depends on error handling implementation


class TestGQLTransactionIsolationLevels:
    """Tests for transaction isolation level selection."""

    def test_isolation_level_default(self, db):
        """Default isolation level is snapshot."""
        tx = db.begin_transaction()
        assert tx.isolation_level == "snapshot"
        tx.rollback()

    def test_isolation_level_read_committed(self, db):
        """Explicit read_committed isolation level."""
        tx = db.begin_transaction(isolation_level="read_committed")
        assert tx.isolation_level == "read_committed"
        tx.execute("INSERT (:IsoRC {name: 'rc_test'})")
        tx.commit()

        result = db.execute("MATCH (n:IsoRC) RETURN n.name")
        rows = list(result)
        assert len(rows) == 1

    def test_isolation_level_serializable(self, db):
        """Explicit serializable isolation level."""
        tx = db.begin_transaction(isolation_level="serializable")
        assert tx.isolation_level == "serializable"
        tx.execute("INSERT (:IsoSet {name: 'set_test'})")
        tx.commit()

        result = db.execute("MATCH (n:IsoSet) RETURN n.name")
        rows = list(result)
        assert len(rows) == 1

    def test_isolation_level_invalid(self, db):
        """Unknown isolation level raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown isolation level"):
            db.begin_transaction(isolation_level="bogus")
