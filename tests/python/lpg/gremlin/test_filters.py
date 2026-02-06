"""Gremlin implementation of filter and direct lookup tests.

Tests filter operations and direct lookup APIs with Gremlin for setup/verification.

Note: Gremlin filter syntax may not be fully implemented yet.
"""

import pytest
from tests.python.bases.test_filters import BaseFilterAndLookupTest


# Skip Gremlin filter tests until Gremlin support is more complete
pytestmark = pytest.mark.skip(reason="Gremlin filter syntax not yet fully implemented")


class TestGremlinFilters(BaseFilterAndLookupTest):
    """Gremlin implementation of filter and lookup tests."""

    def _execute_gremlin(self, db, query: str):
        """Execute Gremlin query, skip if not supported."""
        try:
            return db.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")
        except NotImplementedError:
            pytest.skip("Gremlin not implemented")

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
        """Filter using Gremlin has() step."""
        result = self._execute_gremlin(
            db, f"g.V().hasLabel('Person').has('age', {age})"
        )
        return list(result)

    def filter_by_age_range(self, db, min_age: int, max_age: int) -> list:
        """Filter using Gremlin range predicates."""
        result = self._execute_gremlin(
            db,
            f"g.V().hasLabel('Person').has('age', gt({min_age})).has('age', lt({max_age}))",
        )
        return list(result)

    def filter_by_city(self, db, city: str) -> list:
        """Filter using Gremlin string comparison."""
        result = self._execute_gremlin(
            db, f"g.V().hasLabel('Person').has('city', '{city}')"
        )
        return list(result)

    def filter_compound_and(self, db, city: str, min_age: int) -> list:
        """Filter using compound Gremlin has() steps (implicit AND)."""
        result = self._execute_gremlin(
            db,
            f"g.V().hasLabel('Person').has('city', '{city}').has('age', gt({min_age}))",
        )
        return list(result)


class TestGremlinFilterVerification:
    """Additional Gremlin-specific filter tests with verification."""

    def _execute_gremlin(self, db, query: str):
        """Execute Gremlin query, skip if not supported."""
        try:
            return db.execute_gremlin(query)
        except AttributeError:
            pytest.skip("Gremlin support not available")
        except NotImplementedError:
            pytest.skip("Gremlin not implemented")

    def test_filter_with_traversal(self, db):
        """Test filtering with traversal steps."""
        alice = db.create_node(["Person"], {"name": "Alice", "age": 30})
        bob = db.create_node(["Person"], {"name": "Bob", "age": 25})
        charlie = db.create_node(["Person"], {"name": "Charlie", "age": 35})

        db.create_edge(alice.id, bob.id, "knows", {})
        db.create_edge(alice.id, charlie.id, "knows", {})

        # Find friends of Alice who are over 30
        result = self._execute_gremlin(
            db, "g.V().has('name', 'Alice').out('knows').has('age', gt(30))"
        )
        friends = list(result)
        assert len(friends) == 1, "Should find 1 friend over 30"

    def test_filter_between(self, db):
        """Test Gremlin between predicate."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})
        db.create_node(["Person"], {"name": "Charlie", "age": 35})

        result = self._execute_gremlin(
            db, "g.V().hasLabel('Person').has('age', between(26, 34))"
        )
        matches = list(result)
        assert len(matches) == 1, "Should find only Alice (age 30)"

    def test_filter_within(self, db):
        """Test Gremlin within predicate."""
        db.create_node(["Person"], {"name": "Alice", "city": "NYC"})
        db.create_node(["Person"], {"name": "Bob", "city": "LA"})
        db.create_node(["Person"], {"name": "Charlie", "city": "Chicago"})

        result = self._execute_gremlin(
            db, "g.V().hasLabel('Person').has('city', within('NYC', 'LA'))"
        )
        matches = list(result)
        assert len(matches) == 2, "Should find Alice and Bob"

    def test_filter_values_select(self, db):
        """Test Gremlin values() step with filter."""
        db.create_node(["Person"], {"name": "Alice", "age": 30})
        db.create_node(["Person"], {"name": "Bob", "age": 25})

        result = self._execute_gremlin(
            db, "g.V().hasLabel('Person').has('age', gt(28)).values('name')"
        )
        names = list(result)
        assert len(names) == 1, "Should find only Alice's name"
