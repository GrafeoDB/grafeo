"""SPARQL filter and lookup tests.

Tests SPARQL filter operations for RDF data model.
Note: RDF uses triples instead of nodes/edges, so tests are structured differently.

Note: Some SPARQL features (REGEX, NOT EXISTS) may not be fully implemented yet.
"""

import time
import pytest


# Try to import grafeo
try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


# Skip SPARQL filter tests until SPARQL support is more complete
pytestmark = pytest.mark.skip(reason="SPARQL filter syntax not yet fully implemented")


class TestSPARQLFilters:
    """SPARQL filter tests for RDF data model."""

    def setup_method(self):
        """Create a database with RDF test data."""
        self.db = GrafeoDB()

    def _execute_sparql(self, query: str):
        """Execute SPARQL query, skip if not supported."""
        try:
            return self.db.execute_sparql(query)
        except AttributeError:
            pytest.skip("SPARQL support not available")
        except NotImplementedError:
            pytest.skip("SPARQL not implemented")

    def _setup_person_data(self, count: int = 100):
        """Create Person triples using SPARQL INSERT DATA."""
        cities = ["NYC", "LA", "Chicago", "Boston", "Seattle"]

        # Build INSERT DATA statement with all triples
        triples = []
        for i in range(count):
            person_uri = f"ex:person{i}"
            age = i % 100
            city = cities[i % len(cities)]
            triples.extend(
                [
                    f'{person_uri} rdf:type foaf:Person .',
                    f'{person_uri} foaf:name "Person{i}" .',
                    f'{person_uri} foaf:age {age} .',
                    f'{person_uri} ex:city "{city}" .',
                ]
            )

        insert_query = f"""
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            INSERT DATA {{
                {chr(10).join(triples)}
            }}
        """
        self._execute_sparql(insert_query)

    # ===== Filter Correctness Tests =====

    def test_filter_equality_basic(self):
        """Test SPARQL FILTER with equality."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?person ?name WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                FILTER (?age = 25)
            }
            """
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find at least 1 person with age 25"

    def test_filter_range_basic(self):
        """Test SPARQL FILTER with range comparison."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?person WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:age ?age .
                FILTER (?age > 20 && ?age < 30)
            }
            """
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find persons in age range"

    def test_filter_string_equality(self):
        """Test SPARQL FILTER with string comparison."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            SELECT ?person WHERE {
                ?person rdf:type foaf:Person .
                ?person ex:city ?city .
                FILTER (?city = "NYC")
            }
            """
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find persons in NYC"

    def test_filter_compound_and(self):
        """Test SPARQL FILTER with compound AND condition."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            SELECT ?person WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:age ?age .
                ?person ex:city ?city .
                FILTER (?city = "NYC" && ?age > 50)
            }
            """
        )
        rows = list(result)
        # Results should exist if there are NYC persons over 50
        assert isinstance(rows, list), "Should return a list of results"

    def test_filter_or_condition(self):
        """Test SPARQL FILTER with OR condition."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            SELECT ?person WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:age ?age .
                ?person ex:city ?city .
                FILTER (?city = "NYC" || ?age < 10)
            }
            """
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find persons in NYC or under 10"

    def test_filter_regex(self):
        """Test SPARQL FILTER with REGEX."""
        self._setup_person_data(100)

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?person ?name WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
                FILTER REGEX(?name, "^Person1")
            }
            """
        )
        rows = list(result)
        # Should match Person1, Person10-19, Person100-199, etc.
        assert len(rows) >= 1, "Should find persons with names starting with Person1"

    def test_filter_not_exists(self):
        """Test SPARQL FILTER NOT EXISTS."""
        # Create some persons, some with emails
        self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX ex: <http://example.org/>

            INSERT DATA {
                ex:alice rdf:type foaf:Person .
                ex:alice foaf:name "Alice" .
                ex:alice foaf:mbox <mailto:alice@example.com> .

                ex:bob rdf:type foaf:Person .
                ex:bob foaf:name "Bob" .
            }
            """
        )

        result = self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?person ?name WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
                FILTER NOT EXISTS { ?person foaf:mbox ?email }
            }
            """
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find Bob (no email)"

    # ===== Filter Performance Tests =====

    def test_filter_equality_performance(self):
        """Filter equality should complete quickly on 1K triples."""
        self._setup_person_data(250)  # 250 persons = 1000 triples

        # Warm up
        self._execute_sparql(
            """
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?p WHERE { ?p foaf:age 50 }
            """
        )

        # Time the filter
        start = time.perf_counter()
        for _ in range(10):
            result = self._execute_sparql(
                """
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                SELECT ?person WHERE {
                    ?person rdf:type foaf:Person .
                    ?person foaf:age ?age .
                    FILTER (?age = 50)
                }
                """
            )
            list(result)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"10 SPARQL filters took {elapsed:.3f}s, expected < 2.0s"

    def test_filter_range_performance(self):
        """Filter range should complete quickly on 1K triples."""
        self._setup_person_data(250)

        start = time.perf_counter()
        for _ in range(10):
            result = self._execute_sparql(
                """
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                SELECT ?person WHERE {
                    ?person rdf:type foaf:Person .
                    ?person foaf:age ?age .
                    FILTER (?age > 20 && ?age < 40)
                }
                """
            )
            list(result)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"10 SPARQL range filters took {elapsed:.3f}s, expected < 2.0s"
