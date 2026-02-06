"""Base class for filter and direct lookup API tests.

This module defines test logic for:
- Filter operations (equality, range, compound)
- Direct lookup APIs (get_node, get_edge, get_neighbors)
- Filter performance regression tests
"""

from abc import ABC, abstractmethod
import time


class BaseFilterAndLookupTest(ABC):
    """Abstract base class for filter and direct lookup tests.

    Subclasses must implement query-language-specific methods for:
    - Creating test data
    - Running filter queries
    """

    # ===== Abstract Methods =====

    @abstractmethod
    def create_person_nodes(self, db, count: int = 1000) -> list:
        """Create Person nodes with age, name, and city properties.

        Args:
            db: Database instance
            count: Number of nodes to create

        Returns:
            List of created node IDs
        """
        raise NotImplementedError

    @abstractmethod
    def filter_by_age_equals(self, db, age: int) -> list:
        """Filter nodes where age equals a specific value.

        Args:
            db: Database instance
            age: Age value to filter by

        Returns:
            List of matching results
        """
        raise NotImplementedError

    @abstractmethod
    def filter_by_age_range(self, db, min_age: int, max_age: int) -> list:
        """Filter nodes where age is within a range.

        Args:
            db: Database instance
            min_age: Minimum age (exclusive)
            max_age: Maximum age (exclusive)

        Returns:
            List of matching results
        """
        raise NotImplementedError

    @abstractmethod
    def filter_by_city(self, db, city: str) -> list:
        """Filter nodes by city property.

        Args:
            db: Database instance
            city: City name to filter by

        Returns:
            List of matching results
        """
        raise NotImplementedError

    @abstractmethod
    def filter_compound_and(self, db, city: str, min_age: int) -> list:
        """Filter nodes by city AND age > min_age.

        Args:
            db: Database instance
            city: City name
            min_age: Minimum age (exclusive)

        Returns:
            List of matching results
        """
        raise NotImplementedError

    # ===== Direct Lookup API Tests =====

    def test_get_node_by_id(self, db):
        """Test direct node lookup by ID."""
        # Create a node
        node_ids = self.create_person_nodes(db, count=1)
        node_id = node_ids[0]

        # Direct lookup
        node = db.get_node(node_id)
        assert node is not None, "get_node should return the node"
        assert node.id == node_id, "Node ID should match"
        assert "Person" in node.labels, "Node should have Person label"

    def test_get_node_nonexistent(self, db):
        """Test get_node returns None for nonexistent ID."""
        result = db.get_node(999999999)
        assert result is None, "get_node should return None for nonexistent node"

    def test_get_edge_by_id(self, db):
        """Test direct edge lookup by ID."""
        # Create nodes and edge
        node_ids = self.create_person_nodes(db, count=2)
        edge = db.create_edge(node_ids[0], node_ids[1], "KNOWS", {"since": 2020})

        # Direct lookup
        retrieved = db.get_edge(edge.id)
        assert retrieved is not None, "get_edge should return the edge"
        assert retrieved.id == edge.id, "Edge ID should match"
        assert retrieved.edge_type == "KNOWS", "Edge type should match"

    def test_get_edge_nonexistent(self, db):
        """Test get_edge returns None for nonexistent ID."""
        result = db.get_edge(999999999)
        assert result is None, "get_edge should return None for nonexistent edge"

    # ===== Filter Correctness Tests =====

    def test_filter_equality_basic(self, db):
        """Test basic equality filter."""
        self.create_person_nodes(db, count=100)

        # Filter by age = 25 (100 nodes with age 0-99, so 1 match)
        results = self.filter_by_age_equals(db, 25)
        assert len(results) >= 1, "Should find at least 1 node with age 25"

    def test_filter_equality_no_match(self, db):
        """Test equality filter with no matches."""
        self.create_person_nodes(db, count=100)

        # Filter by age = 999 (no node has this age)
        results = self.filter_by_age_equals(db, 999)
        assert len(results) == 0, "Should find no nodes with age 999"

    def test_filter_range_basic(self, db):
        """Test basic range filter."""
        self.create_person_nodes(db, count=100)

        # Filter by age in (20, 30) exclusive
        results = self.filter_by_age_range(db, 20, 30)
        # Should match ages 21-29 (9 values out of 100)
        assert len(results) >= 1, "Should find nodes in age range"

    def test_filter_string_equality(self, db):
        """Test string equality filter."""
        self.create_person_nodes(db, count=100)

        # Filter by city = "NYC"
        results = self.filter_by_city(db, "NYC")
        assert len(results) >= 1, "Should find nodes in NYC"

    def test_filter_compound_and(self, db):
        """Test compound AND filter."""
        self.create_person_nodes(db, count=100)

        # Filter by city = "NYC" AND age > 50
        results = self.filter_compound_and(db, "NYC", 50)
        # Results should be subset of both individual filters
        nyc_results = self.filter_by_city(db, "NYC")
        assert len(results) <= len(nyc_results), (
            "AND filter should be subset of single filter"
        )

    # ===== Filter Performance Tests =====

    def test_filter_equality_performance(self, db):
        """Filter equality should complete quickly on 1K nodes.

        This is a regression test for the filter optimization that
        uses direct property access instead of loading all properties.
        """
        # Create 1000 nodes
        self.create_person_nodes(db, count=1000)

        # Warm up
        self.filter_by_age_equals(db, 50)

        # Time the filter
        start = time.perf_counter()
        for _ in range(10):  # Run 10 times for more stable measurement
            self.filter_by_age_equals(db, 50)
        elapsed = time.perf_counter() - start

        # Should complete 10 filters on 1K nodes in under 1 second
        assert elapsed < 1.0, (
            f"10 equality filters took {elapsed:.3f}s, expected < 1.0s"
        )

    def test_filter_range_performance(self, db):
        """Filter range should complete quickly on 1K nodes."""
        self.create_person_nodes(db, count=1000)

        # Warm up
        self.filter_by_age_range(db, 20, 40)

        # Time the filter
        start = time.perf_counter()
        for _ in range(10):
            self.filter_by_age_range(db, 20, 40)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"10 range filters took {elapsed:.3f}s, expected < 1.0s"

    def test_direct_lookup_performance(self, db):
        """Direct lookup should be very fast."""
        node_ids = self.create_person_nodes(db, count=1000)

        # Time 1000 direct lookups
        start = time.perf_counter()
        for node_id in node_ids:
            node = db.get_node(node_id)
            assert node is not None
        elapsed = time.perf_counter() - start

        # 1000 lookups should complete in under 0.5 seconds
        assert elapsed < 0.5, (
            f"1000 direct lookups took {elapsed:.3f}s, expected < 0.5s"
        )
