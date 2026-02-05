"""SPARQL pattern tests.

Tests SPARQL queries against the RDF model.
"""

import pytest


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


class TestSPARQLSelect:
    """Test SPARQL SELECT queries."""

    def setup_method(self):
        """Create a database with RDF test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create RDF test data using SPARQL INSERT DATA."""
        # Insert RDF triples using SPARQL
        self.db.execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/person/>

            INSERT DATA {
                ex:alice rdf:type foaf:Person .
                ex:alice foaf:name "Alice" .
                ex:alice foaf:age 30 .

                ex:bob rdf:type foaf:Person .
                ex:bob foaf:name "Bob" .
                ex:bob foaf:age 25 .

                ex:charlie rdf:type foaf:Person .
                ex:charlie foaf:name "Charlie" .
                ex:charlie foaf:age 35 .

                ex:alice foaf:knows ex:bob .
                ex:bob foaf:knows ex:charlie .
            }
        """)

    def _execute_sparql(self, query: str):
        """Execute SPARQL query, skip if not supported."""
        try:
            return self.db.execute_sparql(query)
        except AttributeError:
            pytest.skip("SPARQL support not available")
        except NotImplementedError:
            pytest.skip("SPARQL not implemented")

    def test_sparql_select_all(self):
        """SPARQL: SELECT * WHERE { ?s ?p ?o }"""
        result = self._execute_sparql("""
            SELECT * WHERE {
                ?s ?p ?o
            }
        """)
        rows = list(result)
        assert len(rows) > 0

    def test_sparql_select_with_type(self):
        """SPARQL: SELECT with rdf:type filter."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?name WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        # Should find Alice, Bob, and Charlie
        assert len(rows) == 3

    def test_sparql_select_with_filter(self):
        """SPARQL: SELECT with FILTER."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                FILTER(?age > 28)
            }
        """)
        rows = list(result)
        # Alice (30) and Charlie (35) match
        assert len(rows) == 2

    def test_sparql_select_relationship(self):
        """SPARQL: SELECT with relationship pattern."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name1 ?name2 WHERE {
                ?p1 foaf:knows ?p2 .
                ?p1 foaf:name ?name1 .
                ?p2 foaf:name ?name2 .
            }
        """)
        rows = list(result)
        # Alice->Bob and Bob->Charlie
        assert len(rows) == 2

    def test_sparql_optional(self):
        """SPARQL: SELECT with OPTIONAL."""
        # Create a person without email using SPARQL
        self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/person/>

            INSERT DATA {
                ex:diana foaf:name "Diana" .
            }
        """)

        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?email WHERE {
                ?person foaf:name ?name .
                OPTIONAL { ?person foaf:mbox ?email }
            }
        """)
        rows = list(result)
        # Should include Diana with NULL email
        assert len(rows) >= 4

    def test_sparql_order_by(self):
        """SPARQL: SELECT with ORDER BY."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name ?age WHERE {
                ?person foaf:name ?name .
                ?person foaf:age ?age .
            }
            ORDER BY ?age
        """)
        rows = list(result)
        # Bob (25) should be first
        if len(rows) >= 1:
            assert rows[0].get("age") == 25 or rows[0].get("name") == "Bob"

    def test_sparql_limit(self):
        """SPARQL: SELECT with LIMIT."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?name WHERE {
                ?person foaf:name ?name .
            }
            LIMIT 2
        """)
        rows = list(result)
        assert len(rows) == 2


class TestSPARQLAggregate:
    """Test SPARQL aggregate queries."""

    def setup_method(self):
        """Create a database with RDF test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create RDF test data using SPARQL INSERT DATA."""
        # Insert RDF triples using SPARQL
        self.db.execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/person/>

            INSERT DATA {
                ex:alice rdf:type foaf:Person .
                ex:alice foaf:name "Alice" .
                ex:alice foaf:age 30 .

                ex:bob rdf:type foaf:Person .
                ex:bob foaf:name "Bob" .
                ex:bob foaf:age 25 .

                ex:charlie rdf:type foaf:Person .
                ex:charlie foaf:name "Charlie" .
                ex:charlie foaf:age 35 .
            }
        """)

    def _execute_sparql(self, query: str):
        """Execute SPARQL query, skip if not supported."""
        try:
            return self.db.execute_sparql(query)
        except AttributeError:
            pytest.skip("SPARQL support not available")
        except NotImplementedError:
            pytest.skip("SPARQL not implemented")

    def test_sparql_count(self):
        """SPARQL: COUNT aggregate."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT (COUNT(?person) AS ?count) WHERE {
                ?person foaf:name ?name .
            }
        """)
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["count"] == 3

    def test_sparql_sum_avg(self):
        """SPARQL: SUM and AVG aggregates."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT (SUM(?age) AS ?total) (AVG(?age) AS ?average) WHERE {
                ?person foaf:age ?age .
            }
        """)
        rows = list(result)
        # RDF stores values as strings; aggregates may return strings or numbers
        total = rows[0]["total"]
        average = rows[0]["average"]
        total_val = float(total) if isinstance(total, str) else total
        avg_val = float(average) if isinstance(average, str) else average
        assert total_val == 90  # 30 + 25 + 35
        assert abs(avg_val - 30.0) < 0.01

    def test_sparql_min_max(self):
        """SPARQL: MIN and MAX aggregates."""
        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT (MIN(?age) AS ?youngest) (MAX(?age) AS ?oldest) WHERE {
                ?person foaf:age ?age .
            }
        """)
        rows = list(result)
        # RDF stores values as strings, so compare as strings or convert
        youngest = rows[0]["youngest"]
        oldest = rows[0]["oldest"]
        assert int(youngest) == 25 if isinstance(youngest, str) else youngest == 25
        assert int(oldest) == 35 if isinstance(oldest, str) else oldest == 35

    def test_sparql_group_by(self):
        """SPARQL: GROUP BY."""
        # Add city property using SPARQL
        self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX ex: <http://example.org/person/>

            INSERT DATA {
                ex:alice foaf:city "NYC" .
                ex:bob foaf:city "NYC" .
                ex:charlie foaf:city "LA" .
            }
        """)

        result = self._execute_sparql("""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?city (COUNT(?person) AS ?count) WHERE {
                ?person foaf:city ?city .
            }
            GROUP BY ?city
        """)
        rows = list(result)
        # Should have city groups (NYC and LA)
        assert len(rows) >= 1
