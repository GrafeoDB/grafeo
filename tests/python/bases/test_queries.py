"""Base class for query tests.

This module defines test logic for all read operations:
- Pattern matching (MATCH, WHERE, relationships)
- Paths (variable-length, shortest path)
- Aggregations (COUNT, SUM, AVG, MIN, MAX, GROUP BY)
"""

from abc import ABC, abstractmethod
import pytest


class BaseQueriesTest(ABC):
    """Abstract base class for query tests.

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
    # SETUP METHODS
    # =========================================================================

    @abstractmethod
    def setup_pattern_graph(self, db):
        """Set up test data for pattern tests.

        Should create a graph with:
        - Person nodes: Alice (30, NYC), Bob (25, LA), Charlie (35, NYC)
        - Company nodes: Acme Corp, Globex Inc
        - KNOWS edges between persons
        - WORKS_AT edges from persons to companies

        Returns:
            dict with node references if needed
        """
        raise NotImplementedError

    @abstractmethod
    def setup_chain_graph(self, db):
        """Set up a chain graph: a -> b -> c -> d.

        Returns:
            dict with node IDs
        """
        raise NotImplementedError

    @abstractmethod
    def setup_multi_path_graph(self, db):
        """Set up a graph with multiple paths.

        Creates:
        - Direct path: a -> d
        - Longer path: a -> b -> c -> d

        Returns:
            dict with node IDs
        """
        raise NotImplementedError

    @abstractmethod
    def setup_aggregation_data(self, db):
        """Set up test data for aggregation tests.

        Should create Person nodes with name, age, and city properties:
        - Alice (30, NYC)
        - Bob (25, LA)
        - Charlie (35, NYC)
        """
        raise NotImplementedError

    # =========================================================================
    # PATTERN QUERIES
    # =========================================================================

    @abstractmethod
    def match_label_query(self, label: str, return_prop: str = "name") -> str:
        """Return query to match nodes by label."""
        raise NotImplementedError

    @abstractmethod
    def match_where_query(
        self, label: str, prop: str, op: str, value, return_prop: str = "name"
    ) -> str:
        """Return query to match nodes with WHERE clause."""
        raise NotImplementedError

    @abstractmethod
    def match_and_query(
        self,
        label: str,
        prop1: str,
        op1: str,
        value1,
        prop2: str,
        op2: str,
        value2,
        return_prop: str = "name",
    ) -> str:
        """Return query to match nodes with AND in WHERE clause."""
        raise NotImplementedError

    @abstractmethod
    def match_relationship_query(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        return_from: str = "name",
        return_to: str = "name",
    ) -> str:
        """Return query to match relationship pattern."""
        raise NotImplementedError

    @abstractmethod
    def match_relationship_with_props_query(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        rel_prop: str,
        op: str,
        value,
    ) -> str:
        """Return query to match relationship with property filter."""
        raise NotImplementedError

    @abstractmethod
    def match_multi_hop_query(
        self,
        start_label: str,
        rel_type: str,
        end_label: str,
    ) -> str:
        """Return query for 2-hop path pattern."""
        raise NotImplementedError

    # =========================================================================
    # PATH QUERIES
    # =========================================================================

    @abstractmethod
    def variable_length_path_query(
        self,
        start_label: str,
        start_prop: str,
        start_value,
        rel_type: str,
        end_label: str,
        min_hops: int,
        max_hops: int,
    ) -> str:
        """Return query for variable-length path."""
        raise NotImplementedError

    @abstractmethod
    def shortest_path_query(
        self,
        start_label: str,
        start_prop: str,
        start_value,
        end_label: str,
        end_prop: str,
        end_value,
    ) -> str:
        """Return query for shortest path."""
        raise NotImplementedError

    # =========================================================================
    # AGGREGATION QUERIES
    # =========================================================================

    @abstractmethod
    def count_query(self, label: str) -> str:
        """Return query for COUNT aggregation. Returns count as 'cnt'."""
        raise NotImplementedError

    @abstractmethod
    def count_distinct_query(self, label: str, prop: str) -> str:
        """Return query for COUNT DISTINCT."""
        raise NotImplementedError

    @abstractmethod
    def sum_avg_query(self, label: str, prop: str) -> str:
        """Return query for SUM and AVG. Returns 'total' and 'average'."""
        raise NotImplementedError

    @abstractmethod
    def min_max_query(self, label: str, prop: str) -> str:
        """Return query for MIN and MAX. Returns 'minimum' and 'maximum'."""
        raise NotImplementedError

    @abstractmethod
    def group_by_query(self, label: str, group_prop: str) -> str:
        """Return query for GROUP BY with count."""
        raise NotImplementedError

    # =========================================================================
    # PATTERN TESTS
    # =========================================================================

    def test_simple_match(self, db):
        """Test simple node match by label."""
        self.setup_pattern_graph(db)

        query = self.match_label_query("Person")
        result = self.execute_query(db, query)
        rows = list(result)
        assert len(rows) == 3

    def test_match_with_where(self, db):
        """Test MATCH with WHERE clause."""
        self.setup_pattern_graph(db)

        query = self.match_where_query("Person", "age", ">", 28)
        result = self.execute_query(db, query)
        rows = list(result)

        names = [
            r.get("n.name") or r.get("p.name") or r.get("name") for r in rows
        ]
        assert len(rows) == 2
        assert "Alice" in names
        assert "Charlie" in names
        assert "Bob" not in names

    def test_match_with_and(self, db):
        """Test MATCH with AND in WHERE clause."""
        self.setup_pattern_graph(db)

        query = self.match_and_query(
            "Person", "city", "=", "NYC", "age", ">", 25
        )
        result = self.execute_query(db, query)
        rows = list(result)

        names = [
            r.get("n.name") or r.get("p.name") or r.get("name") for r in rows
        ]
        assert "Alice" in names
        assert "Charlie" in names

    def test_match_relationship(self, db):
        """Test matching relationship patterns."""
        self.setup_pattern_graph(db)

        query = self.match_relationship_query("Person", "KNOWS", "Person")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 3

    def test_match_relationship_with_properties(self, db):
        """Test matching relationship with property filter."""
        self.setup_pattern_graph(db)

        query = self.match_relationship_with_props_query(
            "Person", "KNOWS", "Person", "since", ">=", 2020
        )
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) >= 2

    def test_match_multi_hop(self, db):
        """Test multi-hop path pattern."""
        self.setup_pattern_graph(db)

        query = self.match_multi_hop_query("Person", "KNOWS", "Person")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) >= 1

    def test_match_heterogeneous(self, db):
        """Test matching across different node types."""
        self.setup_pattern_graph(db)

        query = self.match_relationship_query("Person", "WORKS_AT", "Company")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 3

    # =========================================================================
    # PATH TESTS
    # =========================================================================

    def test_variable_length_path(self, db):
        """Test variable-length path matching."""
        self.setup_chain_graph(db)

        query = self.variable_length_path_query(
            "Node", "name", "a", "NEXT", "Node", 1, 3
        )
        result = self.execute_query(db, query)
        rows = list(result)

        names = [
            r.get("end.name") or r.get("e.name") or r.get("name")
            for r in rows
        ]
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_shortest_path(self, db):
        """Test shortest path query."""
        self.setup_multi_path_graph(db)

        query = self.shortest_path_query(
            "Node", "name", "a", "Node", "name", "d"
        )
        result = self.execute_query(db, query)
        rows = list(result)

        if len(rows) > 0:
            path_length = (
                rows[0].get("path_length")
                or rows[0].get("length")
                or rows[0].get("len")
            )
            assert path_length == 1

    def test_path_with_exact_length(self, db):
        """Test path with exact length."""
        self.setup_chain_graph(db)

        query = self.variable_length_path_query(
            "Node", "name", "a", "NEXT", "Node", 2, 2
        )
        result = self.execute_query(db, query)
        rows = list(result)

        names = [
            r.get("end.name") or r.get("e.name") or r.get("name")
            for r in rows
        ]
        assert "c" in names
        assert "b" not in names
        assert "d" not in names

    # =========================================================================
    # AGGREGATION TESTS
    # =========================================================================

    def test_count(self, db):
        """Test COUNT aggregation."""
        self.setup_aggregation_data(db)

        query = self.count_query("Person")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 1
        assert rows[0]["cnt"] == 3

    def test_count_distinct(self, db):
        """Test COUNT DISTINCT."""
        self.setup_aggregation_data(db)

        query = self.count_distinct_query("Person", "city")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 1
        count = rows[0].get("cities") or rows[0].get("cnt") or list(rows[0].values())[0]
        assert count == 2  # NYC and LA

    def test_sum_avg(self, db):
        """Test SUM and AVG aggregations."""
        self.setup_aggregation_data(db)

        query = self.sum_avg_query("Person", "age")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 1
        total = rows[0].get("total") or rows[0].get("total_age")
        assert total == 90  # 30 + 25 + 35
        average = rows[0].get("average") or rows[0].get("avg_age")
        assert abs(average - 30.0) < 0.01

    def test_min_max(self, db):
        """Test MIN and MAX aggregations."""
        self.setup_aggregation_data(db)

        query = self.min_max_query("Person", "age")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 1
        minimum = rows[0].get("minimum") or rows[0].get("youngest")
        maximum = rows[0].get("maximum") or rows[0].get("oldest")
        assert minimum == 25  # Bob
        assert maximum == 35  # Charlie

    def test_group_by(self, db):
        """Test GROUP BY."""
        self.setup_aggregation_data(db)

        query = self.group_by_query("Person", "city")
        result = self.execute_query(db, query)
        rows = list(result)

        assert len(rows) == 2
        city_counts = {
            r.get("p.city") or r.get("city"): r.get("cnt") or r.get("count")
            for r in rows
        }
        assert city_counts.get("NYC") == 2
        assert city_counts.get("LA") == 1
