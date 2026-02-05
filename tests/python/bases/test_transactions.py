"""Base class for transaction tests.

This module defines test logic for transaction commit, rollback, and isolation.
"""

from abc import ABC, abstractmethod
import pytest


class BaseTransactionsTest(ABC):
    """Abstract base class for transaction tests."""

    def execute_query(self, db, query):
        """Execute a query using the appropriate language parser.

        Override in subclasses that need a specific parser (e.g., Cypher).
        Default uses GQL parser via db.execute().
        """
        return db.execute(query)

    def execute_in_tx(self, tx, query):
        """Execute a query within a transaction.

        Override in subclasses that need a specific parser (e.g., Gremlin).
        Default uses tx.execute() (GQL parser).
        """
        return tx.execute(query)

    @abstractmethod
    def insert_query(self, labels: list[str], props: dict) -> str:
        """Return query to insert a node.

        Args:
            labels: Node labels
            props: Node properties

        Returns:
            Language-specific INSERT/CREATE query
        """
        raise NotImplementedError

    @abstractmethod
    def match_by_prop_query(self, label: str, prop: str, value) -> str:
        """Return query to match node by property.

        Args:
            label: Node label
            prop: Property name
            value: Property value

        Returns:
            Query that returns matching nodes
        """
        raise NotImplementedError

    @abstractmethod
    def count_query(self, label: str) -> str:
        """Return query to count nodes by label.

        Returns:
            Query that returns count as 'cnt'
        """
        raise NotImplementedError

    # ===== Test Methods =====

    def test_transaction_commit(self, db):
        """Test that committed transaction persists data."""
        with db.begin_transaction() as tx:
            query = self.insert_query(["Person"], {"name": "CommitTest"})
            self.execute_in_tx(tx, query)
            tx.commit()

        # Data should be visible after commit
        match_query = self.match_by_prop_query("Person", "name", "CommitTest")
        result = self.execute_query(db, match_query)
        rows = list(result)
        assert len(rows) == 1

    def test_transaction_auto_commit(self, db):
        """Test that transactions auto-commit on context exit."""
        with db.begin_transaction() as tx:
            query = self.insert_query(["Person"], {"name": "AutoCommitTest"})
            self.execute_in_tx(tx, query)
            # No explicit commit - should auto-commit

        # Data should be visible
        match_query = self.match_by_prop_query("Person", "name", "AutoCommitTest")
        result = self.execute_query(db, match_query)
        rows = list(result)
        assert len(rows) == 1

    def test_transaction_rollback(self, db):
        """Test that rollback discards changes."""
        # Verify database is empty for this label/prop combo
        match_query = self.match_by_prop_query("Person", "name", "RollbackTest")
        result = self.execute_query(db, match_query)
        assert len(list(result)) == 0

        # Create node and rollback
        with db.begin_transaction() as tx:
            query = self.insert_query(["Person"], {"name": "RollbackTest"})
            self.execute_in_tx(tx, query)
            tx.rollback()

        # Data should NOT be visible after rollback
        result = self.execute_query(db, match_query)
        rows = list(result)
        assert len(rows) == 0, f"Expected 0 rows after rollback, got {len(rows)}"

    def test_transaction_is_active(self, db):
        """Test transaction is_active property."""
        tx = db.begin_transaction()
        assert tx.is_active is True

        tx.commit()
        assert tx.is_active is False

    def test_multiple_operations_in_transaction(self, db):
        """Test multiple operations in a single transaction."""
        with db.begin_transaction() as tx:
            # Create multiple nodes
            self.execute_in_tx(tx, self.insert_query(["Person"], {"name": "Multi1", "idx": 1}))
            self.execute_in_tx(tx, self.insert_query(["Person"], {"name": "Multi2", "idx": 2}))
            self.execute_in_tx(tx, self.insert_query(["Person"], {"name": "Multi3", "idx": 3}))
            tx.commit()

        # All nodes should exist
        count_query = self.count_query("Person")
        result = self.execute_query(db, count_query)
        rows = list(result)
        # Get count value - column name varies by language (cnt, count, etc)
        # If count_query returns actual rows instead of a count, use len(rows)
        if len(rows) == 0:
            count_value = 0
        elif rows[0].get("cnt") is not None:
            count_value = rows[0].get("cnt")
        elif rows[0].get("count") is not None:
            count_value = rows[0].get("count")
        elif len(rows[0]) > 0:
            first_value = list(rows[0].values())[0]
            # If it's a number, use it as count; otherwise count the rows
            count_value = first_value if isinstance(first_value, (int, float)) else len(rows)
        else:
            count_value = len(rows)
        assert count_value >= 3
