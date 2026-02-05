"""GQL implementation of filter and direct lookup tests.

Tests filter operations and direct lookup APIs with GQL for setup/verification.
"""

from tests.python.bases.test_filters import BaseFilterAndLookupTest


class TestGQLFilters(BaseFilterAndLookupTest):
    """GQL implementation of filter and lookup tests."""

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
        """Filter using GQL WHERE clause."""
        result = db.execute(f"MATCH (p:Person) WHERE p.age = {age} RETURN p")
        return list(result)

    def filter_by_age_range(self, db, min_age: int, max_age: int) -> list:
        """Filter using GQL range comparison."""
        result = db.execute(
            f"MATCH (p:Person) WHERE p.age > {min_age} AND p.age < {max_age} RETURN p"
        )
        return list(result)

    def filter_by_city(self, db, city: str) -> list:
        """Filter using GQL string comparison."""
        result = db.execute(f"MATCH (p:Person) WHERE p.city = '{city}' RETURN p")
        return list(result)

    def filter_compound_and(self, db, city: str, min_age: int) -> list:
        """Filter using compound GQL WHERE clause."""
        result = db.execute(
            f"MATCH (p:Person) WHERE p.city = '{city}' AND p.age > {min_age} RETURN p"
        )
        return list(result)


class TestGQLFilterVerification:
    """Additional GQL-specific filter tests with verification."""

    def test_filter_with_relationship(self, db):
        """Test filtering with relationship pattern."""
        # Create people and relationships
        alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
        bob = db.create_node(["Person"], {"name": "Bob", "age": 25})
        charlie = db.create_node(["Person"], {"name": "Charlie", "age": 35})

        db.create_edge(alice.id, bob.id, "KNOWS", {})
        db.create_edge(alice.id, charlie.id, "KNOWS", {})

        # Filter friends of Alice who are over 30
        result = db.execute(
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

        # Filter by city = NYC OR age < 30
        result = db.execute(
            "MATCH (p:Person) WHERE p.city = 'NYC' OR p.age < 30 RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 2, "Should find Alice (NYC) and Bob (age < 30)"

    def test_filter_inequality(self, db):
        """Test inequality filter condition."""
        db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "age": 25, "city": "LA"})

        # Filter by city <> NYC (not equal)
        result = db.execute(
            "MATCH (p:Person) WHERE p.city <> 'NYC' RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 1, "Should find only Bob (not in NYC)"

    def test_filter_less_than(self, db):
        """Test less than filter condition."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})

        # Filter by age < 28
        result = db.execute(
            "MATCH (p:Person) WHERE p.age < 28 RETURN p.name"
        )
        matches = list(result)
        assert len(matches) == 1, "Should find only Bob (age 25)"
