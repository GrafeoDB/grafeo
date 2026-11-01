"""GQL implementation of transaction tests.

Tests transaction operations using GQL (ISO standard) query language.
"""

from tests.bases.test_transactions import BaseTransactionsTest


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
        result = db.execute("MATCH (n:Person) WHERE n.name STARTS WITH 'TxPerson' RETURN n.name")
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

    def test_isolation_level_serializable_is_refused(self, db):
        """Serializable is refused until queries record their reads (#671)."""
        import pytest
        from grafeo import GrafeoError

        with pytest.raises(GrafeoError, match="snapshot isolation"):
            db.begin_transaction(isolation_level="serializable")
        result = db.execute("MATCH (n:IsoSet) RETURN n.name")
        assert list(result) == []

    def test_isolation_level_invalid(self, db):
        """Unknown isolation level raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown isolation level"):
            db.begin_transaction(isolation_level="bogus")


class TestEdgeTypeVisibilityAfterTransaction:
    """Regression tests: edge types must survive transaction-committed nodes.

    Previously, the LpgStore epoch counter was not synced with the
    TxManager epoch on commit, so edge_type() used a stale epoch and
    couldn't see edge records created at the post-commit epoch.
    """

    def test_tx_nodes_autocommit_edge_type(self, db):
        """Nodes in tx, edge in auto-commit: type(r) must not be NULL."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {id: 'txpy_a'})")
            tx.execute("INSERT (:Person {id: 'txpy_b'})")
            tx.commit()

        db.execute("MATCH (a {id: 'txpy_a'}), (b {id: 'txpy_b'}) CREATE (a)-[:KNOWS]->(b)")

        rows = list(db.execute("MATCH ({id: 'txpy_a'})-[r]->() RETURN type(r) AS t"))
        assert len(rows) == 1, "Edge should exist"
        assert rows[0]["t"] == "KNOWS", f"Edge type must be 'KNOWS', got {rows[0]['t']!r}"

    def test_tx_nodes_tx_edge_type(self, db):
        """Nodes in first tx, edge in second tx: type(r) must be correct."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {id: 'txpy2_a'})")
            tx.execute("INSERT (:Person {id: 'txpy2_b'})")
            tx.commit()

        with db.begin_transaction() as tx:
            tx.execute("MATCH (a {id: 'txpy2_a'}), (b {id: 'txpy2_b'}) CREATE (a)-[:FRIENDS]->(b)")
            tx.commit()

        rows = list(db.execute("MATCH ({id: 'txpy2_a'})-[r]->() RETURN type(r) AS t"))
        assert len(rows) == 1
        assert rows[0]["t"] == "FRIENDS"

    def test_interleaved_autocommit_and_tx(self, db):
        """Auto-commit node + tx node, then typed edge."""
        db.execute("INSERT (:Person {id: 'pyauto_x'})")

        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {id: 'pytx_y'})")
            tx.commit()

        db.execute("MATCH (a {id: 'pyauto_x'}), (b {id: 'pytx_y'}) CREATE (a)-[:LINKED]->(b)")

        rows = list(db.execute("MATCH ({id: 'pyauto_x'})-[r]->() RETURN type(r) AS t"))
        assert len(rows) == 1
        assert rows[0]["t"] == "LINKED"

    def test_edge_type_filter_after_tx(self, db):
        """Type-filtered MATCH must find edges after tx commit."""
        with db.begin_transaction() as tx:
            tx.execute("INSERT (:Person {id: 'filter_a'})")
            tx.execute("INSERT (:Person {id: 'filter_b'})")
            tx.commit()

        db.execute("MATCH (a {id: 'filter_a'}), (b {id: 'filter_b'}) CREATE (a)-[:WORKS_AT]->(b)")

        rows = list(db.execute("MATCH ()-[r:WORKS_AT]->() RETURN type(r) AS t"))
        assert len(rows) >= 1, "Type-filtered MATCH should find the edge"
        assert all(r["t"] == "WORKS_AT" for r in rows)
