"""
Tests for both LPG (Labeled Property Graph) and RDF graph models.

Run with: pytest tests/python/test_graph_models.py -v
"""

import pytest


# Try to import grafeo
try:
    from grafeo import GrafeoDB
    GRAPHOS_AVAILABLE = True
except ImportError:
    GRAPHOS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAPHOS_AVAILABLE,
    reason="Grafeo Python bindings not installed"
)


class TestLPGModel:
    """Test Labeled Property Graph (LPG) model operations."""

    def setup_method(self):
        """Create a fresh LPG database."""
        # Default model is LPG
        self.db = GrafeoDB()

    # ===== Node Operations =====

    def test_lpg_create_node_single_label(self):
        """LPG: Create node with single label."""
        node = self.db.create_node(["Person"], {"name": "Alice"})
        assert node is not None
        assert "Person" in node.labels
        assert len(node.labels) == 1

    def test_lpg_create_node_multiple_labels(self):
        """LPG: Create node with multiple labels."""
        node = self.db.create_node(
            ["Person", "Employee", "Developer"],
            {"name": "Bob"}
        )
        assert "Person" in node.labels
        assert "Employee" in node.labels
        assert "Developer" in node.labels
        assert len(node.labels) == 3

    def test_lpg_node_properties(self):
        """LPG: Node properties of various types."""
        node = self.db.create_node(["Data"], {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None,
        })

        assert node.properties()["string_val"] == "hello"
        assert node.properties()["int_val"] == 42
        assert abs(node.properties()["float_val"] - 3.14) < 0.01
        assert node.properties()["bool_val"] is True

    def test_lpg_node_update_property(self):
        """LPG: Update node property via query."""
        node = self.db.create_node(["Person"], {"name": "Charlie", "age": 25})

        # Update via SET
        self.db.execute(
            f"MATCH (n:Person) WHERE n.name = 'Charlie' SET n.age = 26"
        )

        # Verify update
        result = self.db.execute(
            "MATCH (n:Person) WHERE n.name = 'Charlie' RETURN n.age"
        )
        rows = list(result)
        assert rows[0]["n.age"] == 26

    def test_lpg_node_add_label(self):
        """LPG: Add label to existing node."""
        node = self.db.create_node(["Person"], {"name": "Diana"})

        # Add label via SET
        self.db.execute(
            "MATCH (n:Person) WHERE n.name = 'Diana' SET n:Employee"
        )

        # Verify label was added
        result = self.db.execute(
            "MATCH (n:Person:Employee) WHERE n.name = 'Diana' RETURN n"
        )
        rows = list(result)
        assert len(rows) == 1

    # ===== Edge Operations =====

    def test_lpg_create_edge(self):
        """LPG: Create edge between nodes."""
        alice = self.db.create_node(["Person"], {"name": "Alice"})
        bob = self.db.create_node(["Person"], {"name": "Bob"})

        edge = self.db.create_edge(
            alice.id, bob.id, "KNOWS",
            {"since": 2020, "strength": 0.8}
        )

        assert edge is not None
        assert edge.edge_type == "KNOWS"
        assert edge.source_id == alice.id
        assert edge.target_id == bob.id
        assert edge.properties()["since"] == 2020

    def test_lpg_edge_directions(self):
        """LPG: Test directed edge semantics."""
        a = self.db.create_node(["Node"], {"id": "a"})
        b = self.db.create_node(["Node"], {"id": "b"})

        # Create directed edge a -> b
        self.db.create_edge(a.id, b.id, "POINTS_TO", {})

        # Query outgoing from a
        result = self.db.execute(
            "MATCH (a:Node {id: 'a'})-[:POINTS_TO]->(b) RETURN b.id"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["b.id"] == "b"

        # Query incoming to a (should be empty)
        result = self.db.execute(
            "MATCH (a:Node {id: 'a'})<-[:POINTS_TO]-(b) RETURN b.id"
        )
        rows = list(result)
        assert len(rows) == 0

    def test_lpg_multiple_edge_types(self):
        """LPG: Multiple edge types between same nodes."""
        alice = self.db.create_node(["Person"], {"name": "Alice"})
        company = self.db.create_node(["Company"], {"name": "Acme"})

        # Alice works at and owns shares in the company
        self.db.create_edge(alice.id, company.id, "WORKS_AT", {"role": "Engineer"})
        self.db.create_edge(alice.id, company.id, "OWNS_SHARES", {"amount": 100})

        # Query both relationships
        result = self.db.execute(
            "MATCH (p:Person)-[r]->(c:Company) "
            "RETURN type(r) AS rel_type"
        )
        rows = list(result)
        types = [r["rel_type"] for r in rows]
        assert "WORKS_AT" in types
        assert "OWNS_SHARES" in types

    # ===== Pattern Matching =====

    def test_lpg_variable_length_path(self):
        """LPG: Variable length path matching."""
        # Create a chain: a -> b -> c -> d
        a = self.db.create_node(["Node"], {"name": "a"})
        b = self.db.create_node(["Node"], {"name": "b"})
        c = self.db.create_node(["Node"], {"name": "c"})
        d = self.db.create_node(["Node"], {"name": "d"})

        self.db.create_edge(a.id, b.id, "NEXT", {})
        self.db.create_edge(b.id, c.id, "NEXT", {})
        self.db.create_edge(c.id, d.id, "NEXT", {})

        # Find all nodes reachable from a in 1-3 hops
        result = self.db.execute(
            "MATCH (start:Node {name: 'a'})-[:NEXT*1..3]->(end:Node) "
            "RETURN end.name"
        )
        rows = list(result)
        names = [r["end.name"] for r in rows]
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_lpg_shortest_path(self):
        """LPG: Shortest path query."""
        # Create a graph with multiple paths
        a = self.db.create_node(["Node"], {"name": "a"})
        b = self.db.create_node(["Node"], {"name": "b"})
        c = self.db.create_node(["Node"], {"name": "c"})
        d = self.db.create_node(["Node"], {"name": "d"})

        # Direct path: a -> d
        self.db.create_edge(a.id, d.id, "DIRECT", {})

        # Longer path: a -> b -> c -> d
        self.db.create_edge(a.id, b.id, "STEP", {})
        self.db.create_edge(b.id, c.id, "STEP", {})
        self.db.create_edge(c.id, d.id, "STEP", {})

        result = self.db.execute(
            "MATCH p = shortestPath((a:Node {name: 'a'})-[*]-(d:Node {name: 'd'})) "
            "RETURN length(p) AS path_length"
        )
        rows = list(result)
        assert rows[0]["path_length"] == 1  # Direct path is shortest


class TestRDFModel:
    """
    Test RDF (Resource Description Framework) model operations.

    Note: RDF model may be feature-gated. Tests will be skipped if not available.
    """

    @pytest.fixture(autouse=True)
    def setup_rdf_db(self):
        """Try to create an RDF database."""
        try:
            # RDF mode might be configured differently
            # This is a placeholder - actual API may vary
            self.db = GrafeoDB()  # May need: GrafeoDB(model="rdf")
            self.rdf_available = True
        except Exception:
            self.rdf_available = False
            pytest.skip("RDF model not available")

    def test_rdf_triple_insert(self):
        """RDF: Insert a triple (subject, predicate, object)."""
        # In RDF, everything is a triple
        # subject: resource being described
        # predicate: property/relationship
        # object: value or another resource

        # Create nodes for subject and object
        alice = self.db.create_node(["Resource"], {
            "uri": "http://example.org/person/alice",
            "type": "Person"
        })
        knows_bob = self.db.create_node(["Resource"], {
            "uri": "http://example.org/person/bob",
            "type": "Person"
        })

        # Create the triple as an edge
        self.db.create_edge(
            alice.id, knows_bob.id,
            "http://xmlns.com/foaf/0.1/knows",  # FOAF vocabulary
            {}
        )

        # Query the triple
        result = self.db.execute(
            "MATCH (s)-[p]->(o) "
            "WHERE s.uri = 'http://example.org/person/alice' "
            "RETURN s.uri, type(p) AS predicate, o.uri"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0]["predicate"] == "http://xmlns.com/foaf/0.1/knows"

    def test_rdf_literal_values(self):
        """RDF: Store literal values (strings, numbers, dates)."""
        # RDF literals are typed values
        resource = self.db.create_node(["Resource"], {
            "uri": "http://example.org/book/1",
            "http://purl.org/dc/elements/1.1/title": "The Great Gatsby",
            "http://purl.org/dc/elements/1.1/date": "1925-04-10",
            "http://example.org/rating": 4.5,
        })

        result = self.db.execute(
            "MATCH (r:Resource) "
            "WHERE r.uri = 'http://example.org/book/1' "
            "RETURN r.`http://purl.org/dc/elements/1.1/title` AS title"
        )
        rows = list(result)
        assert rows[0]["title"] == "The Great Gatsby"

    def test_rdf_blank_nodes(self):
        """RDF: Blank nodes (anonymous resources)."""
        # Blank nodes are nodes without URIs
        blank = self.db.create_node(["BlankNode"], {
            "type": "Address",
            "street": "123 Main St",
            "city": "NYC"
        })

        person = self.db.create_node(["Resource"], {
            "uri": "http://example.org/person/alice"
        })

        self.db.create_edge(
            person.id, blank.id,
            "http://xmlns.com/foaf/0.1/address",
            {}
        )

        result = self.db.execute(
            "MATCH (p:Resource)-[:`http://xmlns.com/foaf/0.1/address`]->(addr:BlankNode) "
            "RETURN addr.city"
        )
        rows = list(result)
        assert len(rows) == 1

    def test_rdf_reification(self):
        """RDF: Reification (statements about statements)."""
        # Create the base statement
        alice = self.db.create_node(["Resource"], {
            "uri": "http://example.org/alice"
        })
        bob = self.db.create_node(["Resource"], {
            "uri": "http://example.org/bob"
        })
        edge = self.db.create_edge(
            alice.id, bob.id,
            "http://xmlns.com/foaf/0.1/knows",
            {}
        )

        # Create a reified statement (statement about the statement)
        statement = self.db.create_node(["Statement"], {
            "subject_uri": "http://example.org/alice",
            "predicate_uri": "http://xmlns.com/foaf/0.1/knows",
            "object_uri": "http://example.org/bob",
            "confidence": 0.9,
            "source": "social_network_analysis"
        })

        assert statement is not None
        assert statement.properties()["confidence"] == 0.9

    def test_rdf_sparql_like_query(self):
        """RDF: SPARQL-like query patterns."""
        # Create RDF-like data
        alice = self.db.create_node(["Resource"], {
            "uri": "http://example.org/alice",
            "rdf:type": "http://xmlns.com/foaf/0.1/Person",
            "foaf:name": "Alice",
            "foaf:age": 30
        })

        bob = self.db.create_node(["Resource"], {
            "uri": "http://example.org/bob",
            "rdf:type": "http://xmlns.com/foaf/0.1/Person",
            "foaf:name": "Bob",
            "foaf:age": 25
        })

        self.db.create_edge(alice.id, bob.id, "foaf:knows", {})

        # SPARQL-like: SELECT ?name WHERE { ?person rdf:type foaf:Person . ?person foaf:name ?name }
        result = self.db.execute(
            "MATCH (p:Resource) "
            "WHERE p.`rdf:type` = 'http://xmlns.com/foaf/0.1/Person' "
            "RETURN p.`foaf:name` AS name"
        )
        rows = list(result)
        names = [r["name"] for r in rows]
        assert "Alice" in names
        assert "Bob" in names


class TestModelInteroperability:
    """Test interoperability between LPG and RDF representations."""

    def setup_method(self):
        """Create a database."""
        self.db = GrafeoDB()

    def test_lpg_to_rdf_mapping(self):
        """Test that LPG data can be queried with RDF-like patterns."""
        # Create LPG data
        alice = self.db.create_node(["Person"], {
            "name": "Alice",
            "age": 30
        })
        bob = self.db.create_node(["Person"], {
            "name": "Bob",
            "age": 25
        })
        self.db.create_edge(alice.id, bob.id, "KNOWS", {})

        # Query with RDF-like thinking:
        # ?alice a Person ; name "Alice" ; knows ?bob .
        result = self.db.execute(
            "MATCH (alice:Person {name: 'Alice'})-[:KNOWS]->(bob:Person) "
            "RETURN alice.name, bob.name"
        )
        rows = list(result)
        assert len(rows) == 1

    def test_rdf_vocabulary_in_lpg(self):
        """Test using RDF vocabularies in LPG model."""
        # Use FOAF vocabulary as property names
        person = self.db.create_node(["foaf:Person"], {
            "foaf:name": "Alice Smith",
            "foaf:mbox": "alice@example.org",
            "foaf:birthday": "1990-01-15"
        })

        result = self.db.execute(
            "MATCH (p:`foaf:Person`) RETURN p.`foaf:name` AS name"
        )
        rows = list(result)
        assert rows[0]["name"] == "Alice Smith"


class TestGraphModelPerformance:
    """Performance tests for different graph models."""

    def test_lpg_bulk_insert_performance(self):
        """LPG: Measure bulk insert performance."""
        db = GrafeoDB()

        import time
        start = time.perf_counter()

        for i in range(1000):
            db.create_node(["Person"], {
                "name": f"Person{i}",
                "age": 20 + i % 60,
                "city": ["NYC", "LA", "Chicago"][i % 3]
            })

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        stats = db.stats()
        assert stats.node_count == 1000

        # Should complete in reasonable time
        assert elapsed_ms < 10000  # 10 seconds max

    def test_lpg_query_performance(self):
        """LPG: Measure query performance."""
        db = GrafeoDB()

        # Setup
        node_ids = []
        for i in range(500):
            node = db.create_node(["Person"], {
                "name": f"Person{i}",
                "age": 20 + i % 60
            })
            node_ids.append(node.id)

        # Create edges (sparse graph)
        for i in range(0, len(node_ids) - 1, 2):
            db.create_edge(node_ids[i], node_ids[i + 1], "KNOWS", {})

        # Benchmark query
        import time
        start = time.perf_counter()

        for _ in range(10):
            result = db.execute(
                "MATCH (p:Person) WHERE p.age > 50 RETURN count(p)"
            )
            list(result)

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        # 10 queries should complete quickly
        assert elapsed_ms < 5000  # 5 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
