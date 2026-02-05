"""Cypher implementation of transaction tests.

Tests transaction commit, rollback, and isolation using Cypher syntax.
"""

from tests.python.bases.test_transactions import BaseTransactionsTest


class TestCypherTransactions(BaseTransactionsTest):
    """Cypher implementation of transaction tests."""

    def execute_query(self, db, query):
        """Execute query using Cypher parser."""
        return db.execute_cypher(query)

    def insert_query(self, labels: list[str], props: dict) -> str:
        """Return Cypher CREATE query."""
        label_str = ":".join(labels)
        props_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                props_parts.append(f"{k}: '{v}'")
            else:
                props_parts.append(f"{k}: {v}")
        props_str = ", ".join(props_parts)
        return f"CREATE (n:{label_str} {{{props_str}}}) RETURN n"

    def match_by_prop_query(self, label: str, prop: str, value) -> str:
        """Return Cypher MATCH query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"MATCH (n:{label}) WHERE n.{prop} = {val} RETURN n"

    def count_query(self, label: str) -> str:
        """Return Cypher COUNT query."""
        return f"MATCH (n:{label}) RETURN count(n) AS cnt"
