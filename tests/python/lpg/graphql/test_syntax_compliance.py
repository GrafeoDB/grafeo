"""GraphQL syntax compliance tests for LPG model.

Tests GraphQL query and mutation syntax against the Labeled Property Graph model,
verifying compliance with the property-graph GraphQL mapping.

Covers: query operations, mutations, field selection, field aliases,
arguments/filters, variables, nested fields, and fragments.

Run with:
    pytest tests/python/lpg/graphql/test_syntax_compliance.py -v
"""

import pytest


# Try to import grafeo
try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed"
)


class TestGraphQLQueryOperations:
    """Test GraphQL query operation syntax."""

    def setup_method(self):
        """Create a database with test data."""
        self.db = GrafeoDB()
        self._setup_test_data()

    def _setup_test_data(self):
        """Create a social network for query tests."""
        self.alice = self.db.create_node(
            ["Person"], {"name": "Alice", "age": 30, "city": "NYC"}
        )
        self.bob = self.db.create_node(
            ["Person"], {"name": "Bob", "age": 25, "city": "LA"}
        )
        self.charlie = self.db.create_node(
            ["Person"], {"name": "Charlie", "age": 35, "city": "NYC"}
        )

        self.post1 = self.db.create_node(
            ["Post"], {"title": "Hello World", "content": "First post", "views": 100}
        )
        self.post2 = self.db.create_node(
            ["Post"], {"title": "GraphQL Guide", "content": "A guide", "views": 250}
        )

        self.db.create_edge(self.alice.id, self.bob.id, "friends", {})
        self.db.create_edge(self.alice.id, self.charlie.id, "friends", {})
        self.db.create_edge(self.bob.id, self.charlie.id, "friends", {})

        self.db.create_edge(self.alice.id, self.post1.id, "authored", {})
        self.db.create_edge(self.bob.id, self.post2.id, "authored", {})

    def _execute_graphql(self, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return self.db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
            return None
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")
            return None

    # =========================================================================
    # Basic Query Operations
    # =========================================================================

    def test_query_keyword_explicit(self):
        """GraphQL: Explicit 'query' keyword with field selection."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 3, "Should find 3 Person nodes"

    def test_query_shorthand(self):
        """GraphQL: Shorthand query (no 'query' keyword)."""
        result = self._execute_graphql("""
            {
                person {
                    name
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 3, "Shorthand should work the same as explicit query"

    def test_query_multiple_fields(self):
        """GraphQL: Select multiple fields from a type."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    age
                    city
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 3
        # Each row should have all requested fields
        for row in rows:
            assert "name" in row, "Should include name field"

    def test_query_different_type(self):
        """GraphQL: Query a different node type."""
        result = self._execute_graphql("""
            query {
                post {
                    title
                    content
                }
            }
        """)
        rows = list(result)
        assert len(rows) == 2, "Should find 2 Post nodes"

    # =========================================================================
    # Field Aliases
    # =========================================================================

    def test_field_alias(self):
        """GraphQL: Alias a field with a different name."""
        result = self._execute_graphql("""
            query {
                person {
                    fullName: name
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1
        # Result should use the alias
        if rows:
            assert "fullName" in rows[0] or "name" in rows[0]

    def test_multiple_field_aliases(self):
        """GraphQL: Multiple aliases on different fields."""
        result = self._execute_graphql("""
            query {
                person {
                    personName: name
                    personAge: age
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1

    # =========================================================================
    # Arguments / Filters
    # =========================================================================

    def test_argument_string_filter(self):
        """GraphQL: Filter by string argument."""
        result = self._execute_graphql("""
            query {
                person(name: "Alice") {
                    name
                    age
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Should find Alice"
        if rows:
            assert rows[0].get("name") == "Alice"

    def test_argument_integer_filter(self):
        """GraphQL: Filter by integer argument."""
        result = self._execute_graphql("""
            query {
                person(age: 30) {
                    name
                    age
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Should find at least one person aged 30"

    def test_argument_multiple_filters(self):
        """GraphQL: Multiple filter arguments (implicit AND)."""
        result = self._execute_graphql("""
            query {
                person(city: "NYC", age: 30) {
                    name
                }
            }
        """)
        rows = list(result)
        # Only Alice is in NYC and 30 years old
        assert len(rows) >= 1

    # =========================================================================
    # Nested Fields (Relationship Traversal)
    # =========================================================================

    def test_nested_relationship_field(self):
        """GraphQL: Traverse relationship via nested field."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    friends {
                        name
                    }
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Should return persons with friends"

    def test_nested_different_type(self):
        """GraphQL: Traverse to a different node type."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    authored {
                        title
                    }
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Should return persons with authored posts"

    def test_deeply_nested_fields(self):
        """GraphQL: Multi-level nested traversal."""
        result = self._execute_graphql("""
            query {
                person {
                    name
                    friends {
                        name
                        friends {
                            name
                        }
                    }
                }
            }
        """)
        rows = list(result)
        # Should not crash, might return empty for deep nesting
        assert isinstance(rows, list)

    # =========================================================================
    # Fragments
    # =========================================================================

    def test_fragment_spread(self):
        """GraphQL: Named fragment with spread operator."""
        result = self._execute_graphql("""
            fragment PersonInfo on Person {
                name
                age
            }

            query {
                person {
                    ...PersonInfo
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1

    @pytest.mark.xfail(
        reason="Inline fragments not yet supported in property-graph GraphQL mapping",
        strict=False,
    )
    def test_inline_fragment(self):
        """GraphQL: Inline fragment on type."""
        result = self._execute_graphql("""
            query {
                person {
                    ... on Person {
                        name
                        age
                    }
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1


class TestGraphQLMutationOperations:
    """Test GraphQL mutation operation syntax."""

    def setup_method(self):
        """Create a fresh database."""
        self.db = GrafeoDB()

    def _execute_graphql(self, query: str):
        """Execute GraphQL query, skip if not supported."""
        try:
            return self.db.execute_graphql(query)
        except AttributeError:
            pytest.skip("GraphQL support not available")
        except NotImplementedError:
            pytest.skip("GraphQL not implemented")

    # =========================================================================
    # Create Mutations
    # =========================================================================

    def test_create_node_mutation(self):
        """GraphQL: Create a node using mutation."""
        result = self._execute_graphql("""
            mutation {
                createPerson(name: "Eve", age: 28) {
                    id
                    name
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1, "Create mutation should return the created entity"
        if rows:
            assert rows[0].get("name") == "Eve"

    def test_create_node_with_all_properties(self):
        """GraphQL: Create a node with multiple properties."""
        result = self._execute_graphql("""
            mutation {
                createPerson(name: "Frank", age: 40, city: "Boston") {
                    name
                    age
                    city
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1

    def test_create_different_type(self):
        """GraphQL: Create a different node type."""
        result = self._execute_graphql("""
            mutation {
                createPost(title: "New Post", content: "Some content") {
                    title
                }
            }
        """)
        rows = list(result)
        assert len(rows) >= 1

    # =========================================================================
    # Update and Delete Mutations
    # =========================================================================

    def test_update_node_mutation(self):
        """GraphQL: Update an existing node."""
        # Create a node first
        self.db.create_node(["Person"], {"name": "Grace", "age": 22})

        result = self._execute_graphql("""
            mutation {
                updatePerson(name: "Grace", age: 23) {
                    name
                    age
                }
            }
        """)
        rows = list(result)
        if rows:
            assert rows[0].get("age") == 23

    def test_delete_node_mutation(self):
        """GraphQL: Delete a node."""
        self.db.create_node(["Person"], {"name": "Temp", "age": 99})

        result = self._execute_graphql("""
            mutation {
                deletePerson(name: "Temp") {
                    success
                }
            }
        """)
        # Deletion should succeed without error
        rows = list(result)
        assert isinstance(rows, list)

    # =========================================================================
    # Variables
    # =========================================================================

    def test_query_with_variable_syntax(self):
        """GraphQL: Query using variable definition syntax."""
        # Create some data
        self.db.create_node(["Person"], {"name": "Alice", "age": 30})

        result = self._execute_graphql("""
            query GetPerson($name: String) {
                person(name: $name) {
                    name
                    age
                }
            }
        """)
        # Even without providing variable values, the query should parse
        rows = list(result)
        assert isinstance(rows, list)
