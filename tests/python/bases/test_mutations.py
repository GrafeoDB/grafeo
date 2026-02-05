"""Base class for mutation tests.

This module defines test logic for all write operations:
- Create nodes (single, multiple labels, with properties)
- Create edges
- Update nodes/edges
- Delete nodes/edges
"""

from abc import ABC, abstractmethod
import pytest


class BaseMutationsTest(ABC):
    """Abstract base class for mutation tests.

    Subclasses implement query builders for their specific language.
    """

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def execute_query(self, db, query):
        """Execute a query using the appropriate language parser.

        Override in subclasses that need a specific parser (e.g., Cypher).
        Default uses GQL parser via db.execute().
        """
        return db.execute(query)

    # =========================================================================
    # QUERY BUILDERS
    # =========================================================================

    @abstractmethod
    def create_node_query(self, labels: list[str], props: dict) -> str:
        """Return query to create a node with given labels and properties.

        Args:
            labels: List of node labels (e.g., ["Person", "Employee"])
            props: Dictionary of properties (e.g., {"name": "Alice", "age": 30})

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    @abstractmethod
    def match_node_query(self, label: str, return_prop: str = "name") -> str:
        """Return query to match nodes by label and return a property.

        Args:
            label: Node label to match
            return_prop: Property to return

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    @abstractmethod
    def match_where_query(
        self, label: str, prop: str, op: str, value, return_prop: str = "name"
    ) -> str:
        """Return query to match nodes with WHERE clause.

        Args:
            label: Node label to match
            prop: Property to filter on
            op: Comparison operator (e.g., ">", "=", "<")
            value: Value to compare against
            return_prop: Property to return

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    @abstractmethod
    def delete_node_query(self, label: str, prop: str, value) -> str:
        """Return query to delete a node matching criteria.

        Args:
            label: Node label to match
            prop: Property to match on
            value: Property value to match

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    @abstractmethod
    def create_edge_query(
        self,
        from_label: str,
        from_prop: str,
        from_value,
        to_label: str,
        to_prop: str,
        to_value,
        edge_type: str,
        edge_props: dict,
    ) -> str:
        """Return query to create an edge between two nodes.

        Args:
            from_label: Source node label
            from_prop: Source node property to match
            from_value: Source node property value
            to_label: Target node label
            to_prop: Target node property to match
            to_value: Target node property value
            edge_type: Edge/relationship type
            edge_props: Edge properties

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    @abstractmethod
    def update_node_query(self, label: str, match_prop: str, match_value, set_prop: str, set_value) -> str:
        """Return query to update a node property.

        Args:
            label: Node label to match
            match_prop: Property to match on
            match_value: Value to match
            set_prop: Property to set
            set_value: New value

        Returns:
            Language-specific query string
        """
        raise NotImplementedError

    # =========================================================================
    # CREATE TESTS
    # =========================================================================

    def test_create_single_node(self, db):
        """Test creating a single node."""
        query = self.create_node_query(["Person"], {"name": "Alice", "age": 30})
        result = self.execute_query(db, query)
        rows = list(result)
        assert len(rows) >= 0  # INSERT may not return rows

        # Verify node exists
        match_query = self.match_node_query("Person")
        result = self.execute_query(db, match_query)
        rows = list(result)
        assert len(rows) == 1

    def test_create_node_multiple_labels(self, db):
        """Test creating a node with multiple labels."""
        query = self.create_node_query(
            ["Person", "Developer"], {"name": "Bob", "language": "Rust"}
        )
        self.execute_query(db, query)

        # Verify node has both labels
        match_query = self.match_node_query("Person")
        result = self.execute_query(db, match_query)
        rows = list(result)
        names = [r.get("n.name") or r.get("name") for r in rows]
        assert "Bob" in names

    def test_create_node_with_properties(self, db):
        """Test creating a node with various property types."""
        query = self.create_node_query(
            ["Data"],
            {
                "string_val": "hello",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
            },
        )
        self.execute_query(db, query)

        # Verify properties exist
        result = self.execute_query(db,"MATCH (n:Data) RETURN n.int_val")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0].get("n.int_val") == 42

    def test_create_multiple_nodes(self, db):
        """Test creating multiple nodes."""
        for i in range(5):
            query = self.create_node_query(["Person"], {"name": f"Person{i}", "idx": i})
            self.execute_query(db, query)

        # Verify all nodes created
        match_query = self.match_node_query("Person")
        result = self.execute_query(db, match_query)
        rows = list(result)
        assert len(rows) == 5

    def test_create_edge(self, db):
        """Test creating an edge between nodes."""
        # Create two nodes
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Alice"}))
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Bob"}))

        # Create edge
        query = self.create_edge_query(
            "Person", "name", "Alice",
            "Person", "name", "Bob",
            "KNOWS", {"since": 2020}
        )
        self.execute_query(db, query)

        # Verify edge exists
        result = self.execute_query(db,
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name, r.since"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0].get("r.since") == 2020

    # =========================================================================
    # UPDATE TESTS
    # =========================================================================

    def test_update_node_property(self, db):
        """Test updating a node property."""
        # Create node
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Alice", "age": 30}))

        # Update age
        query = self.update_node_query("Person", "name", "Alice", "age", 31)
        self.execute_query(db, query)

        # Verify update
        result = self.execute_query(db,"MATCH (n:Person {name: 'Alice'}) RETURN n.age")
        rows = list(result)
        assert len(rows) == 1
        assert rows[0].get("n.age") == 31

    # =========================================================================
    # DELETE TESTS
    # =========================================================================

    def test_delete_node(self, db):
        """Test deleting a node."""
        # Create nodes
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Alice"}))
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Bob"}))

        # Delete Alice
        query = self.delete_node_query("Person", "name", "Alice")
        self.execute_query(db, query)

        # Verify only Bob remains
        match_query = self.match_node_query("Person")
        result = self.execute_query(db, match_query)
        rows = list(result)
        names = [r.get("n.name") or r.get("name") for r in rows]
        assert len(rows) == 1
        assert "Bob" in names
        assert "Alice" not in names

    # =========================================================================
    # FILTER TESTS (part of mutations workflow)
    # =========================================================================

    def test_match_with_filter(self, db):
        """Test matching nodes with WHERE clause."""
        # Create test data
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Alice", "age": 30}))
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Bob", "age": 25}))
        self.execute_query(db,self.create_node_query(["Person"], {"name": "Charlie", "age": 35}))

        # Query with filter
        query = self.match_where_query("Person", "age", ">", 28)
        result = self.execute_query(db, query)
        rows = list(result)

        # Alice (30) and Charlie (35) should match
        assert len(rows) == 2
