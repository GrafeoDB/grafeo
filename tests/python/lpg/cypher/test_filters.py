"""Cypher implementation of filter and direct lookup tests.

Tests filter operations and direct lookup APIs with Cypher for setup/verification.
"""

import pytest
from tests.python.bases.test_filters import BaseFilterAndLookupTest


class TestCypherFilters(BaseFilterAndLookupTest):
    """Cypher implementation of filter and lookup tests."""

    def create_person_nodes(self, db, count: int = 1000) -> list:
        """Create Person nodes using direct API (faster than queries)."""
        cities = ["NYC", "LA", "Chicago", "Boston", "Seattle"]
        node_ids = []

        for i in range(count):
            node = db.create_node(
                ["Person"],
                {
                    "name": f"Person{i}",
                    "age": i % 100,  # Ages 0-99, cycling
                    "city": cities[i % len(cities)],
                },
            )
            node_ids.append(node.id)

        return node_ids

    def filter_by_age_equals(self, db, age: int) -> list:
        """Filter using Cypher WHERE clause."""
        result = db.execute_cypher(f"MATCH (p:Person) WHERE p.age = {age} RETURN p")
        return list(result)

    def filter_by_age_range(self, db, min_age: int, max_age: int) -> list:
        """Filter using Cypher range comparison."""
        result = db.execute_cypher(
            f"MATCH (p:Person) WHERE p.age > {min_age} AND p.age < {max_age} RETURN p"
        )
        return list(result)

    def filter_by_city(self, db, city: str) -> list:
        """Filter using Cypher string comparison."""
        result = db.execute_cypher(f"MATCH (p:Person) WHERE p.city = '{city}' RETURN p")
        return list(result)

    def filter_compound_and(self, db, city: str, min_age: int) -> list:
        """Filter using compound Cypher WHERE clause."""
        result = db.execute_cypher(
            f"MATCH (p:Person) WHERE p.city = '{city}' AND p.age > {min_age} RETURN p"
        )
        return list(result)


class TestCypherFilterVerification:
    """Additional Cypher-specific filter tests with verification."""

    def test_filter_with_relationship(self, db):
        """Test filtering with relationship pattern."""
        alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
        bob = db.create_node(["Person"], {"name": "Bob", "age": 25})
        charlie = db.create_node(["Person"], {"name": "Charlie", "age": 35})

        db.create_edge(alice.id, bob.id, "KNOWS", {})
        db.create_edge(alice.id, charlie.id, "KNOWS", {})

        result = db.execute_cypher(
            "MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(friend:Person) "
            "WHERE friend.age > 30 "
            "RETURN friend.name"
        )
        friends = list(result)
        assert len(friends) == 1, "Should find 1 friend over 30"

    def test_filter_or_condition(self, db):
        """Test OR filter condition."""
        db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "age": 25, "city": "LA"})
        db.create_node(["Person"], {"name": "Charlie", "age": 35, "city": "Chicago"})

        result = db.execute_cypher(
            "MATCH (p:Person) WHERE p.city = 'NYC' OR p.age < 30 RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 2, "Should find Alice (NYC) and Bob (age < 30)"

    def test_filter_starts_with(self, db):
        """Test Cypher STARTS WITH filter."""
        db.create_node(["Person"], {"name": "Alice"})
        db.create_node(["Person"], {"name": "Albert"})
        db.create_node(["Person"], {"name": "Bob"})

        result = db.execute_cypher(
            "MATCH (p:Person) WHERE p.name STARTS WITH 'Al' RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 2, "Should find Alice and Albert"

    def test_filter_contains(self, db):
        """Test Cypher CONTAINS filter."""
        db.create_node(["Person"], {"name": "Alice"})
        db.create_node(["Person"], {"name": "Malice"})
        db.create_node(["Person"], {"name": "Bob"})

        result = db.execute_cypher(
            "MATCH (p:Person) WHERE p.name CONTAINS 'lic' RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 2, "Should find Alice and Malice"
