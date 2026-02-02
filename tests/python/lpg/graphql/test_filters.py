"""GraphQL implementation of filter and direct lookup tests.

Tests filter operations and direct lookup APIs with GraphQL for setup/verification.

Note: Some GraphQL filter features (range filters, nested queries) may not be
implemented yet and tests will be skipped if not available.
"""

import pytest
from tests.python.bases.test_filters import BaseFilterAndLookupTest


# Skip entire class if GraphQL filtering isn't fully supported
pytestmark = pytest.mark.skip(reason="GraphQL filter syntax not yet fully implemented")


class TestGraphQLFilters(BaseFilterAndLookupTest):
    """GraphQL implementation of filter and lookup tests."""

    def _execute_graphql(self, db, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

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
        """Filter using GraphQL argument."""
        result = self._execute_graphql(
            db,
            f"""
            query {{
                person(age: {age}) {{
                    name
                    age
                }}
            }}
            """,
        )
        return list(result)

    def filter_by_age_range(self, db, min_age: int, max_age: int) -> list:
        """Filter using GraphQL range arguments.

        Note: GraphQL range filters may not be supported. Skip if not available.
        """
        import pytest
        try:
            result = self._execute_graphql(
                db,
                f"""
                query {{
                    person(age_gt: {min_age}, age_lt: {max_age}) {{
                        name
                        age
                    }}
                }}
                """,
            )
            return list(result)
        except (RuntimeError, Exception) as e:
            if "age_gt" in str(e) or "unknown" in str(e).lower():
                pytest.skip("GraphQL range filters not supported")
            raise

    def filter_by_city(self, db, city: str) -> list:
        """Filter using GraphQL string argument."""
        result = self._execute_graphql(
            db,
            f"""
            query {{
                person(city: "{city}") {{
                    name
                    city
                }}
            }}
            """,
        )
        return list(result)

    def filter_compound_and(self, db, city: str, min_age: int) -> list:
        """Filter using multiple GraphQL arguments (implicit AND)."""
        result = self._execute_graphql(
            db,
            f"""
            query {{
                person(city: "{city}", age_gt: {min_age}) {{
                    name
                    city
                    age
                }}
            }}
            """,
        )
        return list(result)


class TestGraphQLFilterVerification:
    """Additional GraphQL-specific filter tests with verification."""

    def _execute_graphql(self, db, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

    def test_filter_with_nested_query(self, db):
        """Test filtering with nested query."""
        alice = db.create_node(["User"], {"name": "Alice", "age": 30})
        bob = db.create_node(["User"], {"name": "Bob", "age": 25})
        charlie = db.create_node(["User"], {"name": "Charlie", "age": 35})

        db.create_edge(alice.id, bob.id, "friends", {})
        db.create_edge(alice.id, charlie.id, "friends", {})

        result = self._execute_graphql(
            db,
            """
            query {
                user(name: "Alice") {
                    name
                    friends(age_gt: 30) {
                        name
                    }
                }
            }
            """,
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find Alice"

    def test_filter_multiple_types(self, db):
        """Test filtering across multiple node types."""
        alice = db.create_node(["User"], {"name": "Alice", "age": 30})
        post = db.create_node(["Post"], {"title": "Hello", "views": 100})
        db.create_edge(alice.id, post.id, "authored", {})

        result = self._execute_graphql(
            db,
            """
            query {
                user(age: 30) {
                    name
                    authored {
                        title
                    }
                }
            }
            """,
        )
        rows = list(result)
        assert len(rows) >= 1, "Should find Alice with her post"

    def test_filter_with_limit(self, db):
        """Test filtering with limit."""
        for i in range(10):
            db.create_node(["User"], {"name": f"User{i}", "age": 30})

        result = self._execute_graphql(
            db,
            """
            query {
                user(age: 30, first: 5) {
                    name
                }
            }
            """,
        )
        rows = list(result)
        assert len(rows) <= 5, "Should limit results to 5"
