"""Gremlin implementation of transaction tests.

Tests transaction commit, rollback, and isolation using Gremlin syntax.
"""

import pytest
from tests.python.bases.test_transactions import BaseTransactionsTest


def execute_gremlin(db, query: str):
    """Execute Gremlin query, skip if not supported."""
    try:
        return db.execute_gremlin(query)
    except AttributeError:
        pytest.skip("Gremlin support not available in this build")
    except NotImplementedError:
        pytest.skip("Gremlin not implemented")


class TestGremlinTransactions(BaseTransactionsTest):
    """Gremlin implementation of transaction tests.

    Note: Uses begin_transaction() with Gremlin queries.
    """

    def insert_query(self, labels: list[str], props: dict) -> str:
        """Return Gremlin addV query."""
        label = labels[0] if labels else "Vertex"
        prop_parts = []
        for k, v in props.items():
            if isinstance(v, str):
                prop_parts.append(f".property('{k}', '{v}')")
            else:
                prop_parts.append(f".property('{k}', {v})")
        props_str = "".join(prop_parts)
        return f"g.addV('{label}'){props_str}"

    def match_by_prop_query(self, label: str, prop: str, value) -> str:
        """Return Gremlin match query."""
        val = f"'{value}'" if isinstance(value, str) else value
        return f"g.V().hasLabel('{label}').has('{prop}', {val})"

    def count_query(self, label: str) -> str:
        """Return Gremlin count query."""
        return f"g.V().hasLabel('{label}').count()"

    def execute_in_tx(self, tx, query: str):
        """Execute query in transaction context."""
        try:
            return tx.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")

    def execute_query(self, db, query: str):
        """Execute query on database."""
        return execute_gremlin(db, query)
