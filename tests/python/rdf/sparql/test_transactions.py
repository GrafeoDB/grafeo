"""SPARQL implementation of transaction tests.

Tests transaction semantics with SPARQL operations.
Note: Uses Python API for transaction control with SPARQL queries inside.
"""

import pytest
from tests.python.bases.test_transactions import BaseTransactionsTest

# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE,
    reason="Grafeo Python bindings not installed"
)


@pytest.fixture
def db():
    """Create a fresh database instance for each test."""
    return GrafeoDB()


class TestSPARQLTransactions(BaseTransactionsTest):
    """SPARQL implementation of transaction tests.

    Note: Transactions are controlled via Python API.
    SPARQL queries and updates are executed within transactions.
    """

    def execute_query(self, db, query):
        """Execute a SPARQL query using db.execute_sparql()."""
        return db.execute_sparql(query)

    def execute_in_tx(self, tx, query):
        """Execute a SPARQL query within a transaction."""
        return tx.execute_sparql(query)

    def insert_query(self, labels: list, props: dict) -> str:
        """Build SPARQL INSERT DATA statement.

        Note: SPARQL doesn't have labels like LPG.
        We use rdf:type for classification and properties as triples.
        """
        label = labels[0] if labels else "Resource"
        name = props.get("name", "unnamed")
        uri = f"<http://example.org/{label.lower()}/{name}>"

        triples = []
        triples.append(f'{uri} a <http://example.org/{label}>')

        for k, v in props.items():
            if isinstance(v, str):
                triples.append(f'{uri} <http://example.org/{k}> "{v}"')
            else:
                triples.append(f'{uri} <http://example.org/{k}> {v}')

        return f"INSERT DATA {{ {' . '.join(triples)} }}"

    def match_by_prop_query(self, label: str, prop: str, value) -> str:
        """Build SPARQL SELECT to find resource by property."""
        if isinstance(value, str):
            val = f'"{value}"'
        else:
            val = str(value)

        return f"""
            SELECT ?s WHERE {{
                ?s a <http://example.org/{label}> .
                ?s <http://example.org/{prop}> {val} .
            }}
        """

    def count_query(self, label: str) -> str:
        """Build SPARQL COUNT query."""
        return f"""
            SELECT (COUNT(?s) AS ?cnt) WHERE {{
                ?s a <http://example.org/{label}> .
            }}
        """

    def test_transaction_rollback(self, db):
        """Test that rollback discards changes."""
        super().test_transaction_rollback(db)


class TestSPARQLTransactionVerification:
    """Tests that verify transaction behavior with SPARQL.

    These tests use the Python API (create_node, execute) which operates
    on the LPG store, not the RDF store. They verify that transaction
    semantics work correctly with the LPG store.
    """

    def setup_method(self):
        """Create a fresh database."""
        self.db = GrafeoDB()

    def test_transaction_commit_on_success(self):
        """Verify transaction commits on success."""
        initial_count = len(list(self.db.execute("MATCH (n) RETURN n")))

        with self.db.begin_transaction() as tx:
            tx.execute("INSERT (:Test {name: 'committed'})")

        # Count should be incremented
        final_count = len(list(self.db.execute("MATCH (n) RETURN n")))
        assert final_count == initial_count + 1

    def test_transaction_rollback_on_error(self):
        """Verify transaction rollback on error."""
        initial_count = len(list(self.db.execute("MATCH (n) RETURN n")))

        try:
            with self.db.begin_transaction() as tx:
                # Create a node
                tx.execute("INSERT (:Test {name: 'temp'})")
                # Simulate an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Count should be unchanged
        final_count = len(list(self.db.execute("MATCH (n) RETURN n")))
        assert final_count == initial_count
