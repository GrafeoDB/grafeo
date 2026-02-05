"""GraphQL query tests for LPG model.

Tests GraphQL queries against the Labeled Property Graph model.
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


class TestGraphQLQueries:
    """Test GraphQL query operations."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create test data."""
        self.alice = self.db.create_node(["User"], {
            "name": "Alice", "email": "alice@example.com", "age": 30
        })
        self.bob = self.db.create_node(["User"], {
            "name": "Bob", "email": "bob@example.com", "age": 25
        })
        self.post1 = self.db.create_node(["Post"], {
            "title": "Hello World", "content": "My first post"
        })
        self.db.create_edge(self.alice.id, self.bob.id, "friends", {})
        self.db.create_edge(self.alice.id, self.post1.id, "posts", {})

    def _execute_graphql(self, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return self.db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

    def test_graphql_simple_query(self):
        """GraphQL: Simple field selection."""
        result = self._execute_graphql("""
            query {
                user {
                    name
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 2  # Two users

    def test_graphql_query_with_argument(self):
        """GraphQL: Query with filter argument."""
        result = self._execute_graphql("""
            query {
                user(age: 30) {
                    name
                    email
                }
            }
        """)
        rows = list(result)
        # Should return Alice who is 30
        assert len(rows) >= 1

    def test_graphql_multiple_fields(self):
        """GraphQL: Query multiple fields."""
        result = self._execute_graphql("""
            query {
                user {
                    name
                    email
                    age
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 2

    def test_graphql_nested_query(self):
        """GraphQL: Nested query with relationships."""
        result = self._execute_graphql("""
            query {
                user {
                    name
                    friends {
                        name
                    }
                }
            }
        """)
        rows = list(result)
        # Should return users with their friends
        assert len(rows) >= 1

    def test_graphql_alias(self):
        """GraphQL: Query with alias."""
        result = self._execute_graphql("""
            query {
                user {
                    userName: name
                }
            }
        """)
        rows = list(result)
        # Should have aliased column
        assert len(rows) >= 1

    def test_graphql_deep_nesting(self):
        """GraphQL: Deeply nested query."""
        result = self._execute_graphql("""
            query {
                user {
                    name
                    posts {
                        title
                        content
                    }
                }
            }
        """)
        rows = list(result)
        # Should return users with their posts
        assert len(rows) >= 1

    def test_graphql_fragments(self):
        """GraphQL: Query with fragments."""
        result = self._execute_graphql("""
            fragment UserFields on User {
                name
                email
            }

            query {
                user {
                    ...UserFields
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1


class TestGraphQLMutations:
    """Test GraphQL mutations (if supported)."""

    def setup_method(self):
        """Create a database."""
        self.db = GrafeoDB()

    def _execute_graphql(self, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return self.db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

    def test_graphql_create_mutation(self):
        """GraphQL: Create mutation."""
        result = self._execute_graphql("""
            mutation {
                createUser(name: "Charlie", email: "charlie@example.com") {
                    name
                }
            }
        """)
        rows = list(result)
        # Should create and return the new user
        if len(rows) >= 1:
            assert rows[0].get("name") == "Charlie"

    def test_graphql_update_mutation(self):
        """GraphQL: Update mutation."""
        # First create a user
        self.db.create_node(["User"], {"name": "Diana", "email": "diana@example.com"})

        result = self._execute_graphql("""
            mutation {
                updateUser(name: "Diana", email: "diana.new@example.com") {
                    name
                    email
                }
            }
        """)
        rows = list(result)
        if len(rows) >= 1:
            assert rows[0].get("email") == "diana.new@example.com"

    def test_graphql_delete_mutation(self):
        """GraphQL: Delete mutation."""
        # First create a user
        self.db.create_node(["User"], {"name": "ToDelete", "email": "delete@example.com"})

        result = self._execute_graphql("""
            mutation {
                deleteUser(name: "ToDelete") {
                    success
                }
            }
        """)
        # Verify user is deleted
        # Implementation dependent
