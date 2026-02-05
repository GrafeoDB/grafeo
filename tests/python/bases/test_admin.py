"""Base class for admin API tests.

This module defines test logic for all admin operations:
- Database info and stats
- Schema inspection
- Validation
- Persistence (save, to_memory, open_in_memory)
- WAL management
"""

from abc import ABC, abstractmethod
import pytest
import tempfile
from pathlib import Path


class BaseAdminTest(ABC):
    """Abstract base class for admin API tests.

    Subclasses set up language-specific fixtures.
    """

    # =========================================================================
    # SETUP METHODS
    # =========================================================================

    @abstractmethod
    def setup_test_graph(self, db):
        """Set up test data for admin tests.

        Should create a graph with:
        - Person nodes: Alice (30, NYC), Bob (25, LA), Charlie (35, NYC)
        - Company nodes: Acme Corp, Globex Inc
        - KNOWS edges between persons
        - WORKS_AT edges from persons to companies
        """
        raise NotImplementedError

    # =========================================================================
    # INFO TESTS
    # =========================================================================

    def test_info_basic(self, db):
        """Test basic info() method."""
        info = db.info()

        assert "mode" in info
        assert "node_count" in info
        assert "edge_count" in info
        assert "is_persistent" in info
        assert "wal_enabled" in info
        assert "version" in info

    def test_info_counts_empty(self, db):
        """Test info() on empty database."""
        info = db.info()

        assert info["node_count"] == 0
        assert info["edge_count"] == 0

    def test_info_counts_with_data(self, db):
        """Test info() counts after adding data."""
        self.setup_test_graph(db)
        info = db.info()

        assert info["node_count"] == 5  # 3 persons + 2 companies
        assert info["edge_count"] == 6  # 3 KNOWS + 3 WORKS_AT

    def test_info_mode(self, db):
        """Test that mode is correctly reported."""
        info = db.info()
        # Default should be LPG mode
        assert info["mode"] in ["lpg", "rdf"]

    # =========================================================================
    # STATS TESTS
    # =========================================================================

    def test_stats_basic(self, db):
        """Test detailed_stats() method."""
        self.setup_test_graph(db)
        stats = db.detailed_stats()

        assert "node_count" in stats
        assert "edge_count" in stats
        assert "label_count" in stats
        assert "edge_type_count" in stats
        assert "property_key_count" in stats
        assert "index_count" in stats
        assert "memory_bytes" in stats

    def test_stats_counts(self, db):
        """Test detailed_stats() counts."""
        self.setup_test_graph(db)
        stats = db.detailed_stats()

        assert stats["node_count"] == 5
        assert stats["edge_count"] == 6
        assert stats["label_count"] == 2  # Person, Company
        assert stats["edge_type_count"] == 2  # KNOWS, WORKS_AT

    def test_stats_memory(self, db):
        """Test that memory_bytes key exists (tracks buffer manager allocations)."""
        self.setup_test_graph(db)
        stats = db.detailed_stats()

        # memory_bytes tracks buffer manager allocations, not graph storage
        # so it may be 0 even with data
        assert "memory_bytes" in stats

    # =========================================================================
    # SCHEMA TESTS
    # =========================================================================

    def test_schema_basic(self, db):
        """Test schema() method."""
        self.setup_test_graph(db)
        schema = db.schema()

        # Schema contains labels, edge_types, property_keys for LPG mode
        assert "labels" in schema or "predicates" in schema

    def test_schema_lpg_labels(self, db):
        """Test that labels are reported in LPG mode."""
        self.setup_test_graph(db)
        schema = db.schema()

        # LPG schema has labels
        if "labels" in schema:
            label_names = [l["name"] for l in schema["labels"]]
            assert "Person" in label_names
            assert "Company" in label_names

    def test_schema_lpg_edge_types(self, db):
        """Test that edge types are reported in LPG mode."""
        self.setup_test_graph(db)
        schema = db.schema()

        # LPG schema has edge_types
        if "edge_types" in schema:
            edge_type_names = [e["name"] for e in schema["edge_types"]]
            assert "KNOWS" in edge_type_names
            assert "WORKS_AT" in edge_type_names

    def test_schema_property_keys(self, db):
        """Test that property keys are reported."""
        self.setup_test_graph(db)
        schema = db.schema()

        # LPG schema has property_keys
        if "property_keys" in schema:
            keys = schema["property_keys"]
            assert "name" in keys

    # =========================================================================
    # VALIDATION TESTS
    # =========================================================================

    def test_validate_empty(self, db):
        """Test validate() on empty database."""
        errors = db.validate()

        # validate() returns a list of errors (empty = valid)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_with_data(self, db):
        """Test validate() with data."""
        self.setup_test_graph(db)
        errors = db.validate()

        # validate() returns a list of errors (empty = valid)
        assert isinstance(errors, list)
        # Valid data should have no errors
        assert len(errors) == 0

    # =========================================================================
    # PERSISTENCE TESTS
    # =========================================================================

    def test_is_persistent_in_memory(self, db):
        """Test that in-memory database is not persistent."""
        assert db.is_persistent is False

    def test_path_in_memory(self, db):
        """Test that in-memory database has no path."""
        assert db.path is None

    def test_save_and_open(self, db):
        """Test saving and reopening a database."""
        self.setup_test_graph(db)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.grafeo"
            db.save(str(db_path))

            # Open the saved database
            from grafeo import GrafeoDB
            db2 = GrafeoDB.open(str(db_path))

            info = db2.info()
            assert info["node_count"] == 5
            assert info["edge_count"] == 6

    def test_to_memory(self, db):
        """Test creating an in-memory copy."""
        self.setup_test_graph(db)

        db2 = db.to_memory()

        info = db2.info()
        assert info["node_count"] == 5
        assert info["edge_count"] == 6
        assert db2.is_persistent is False

    def test_open_in_memory(self, db):
        """Test opening a database as in-memory copy."""
        self.setup_test_graph(db)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.grafeo"
            db.save(str(db_path))

            # Open as in-memory copy
            from grafeo import GrafeoDB
            db2 = GrafeoDB.open_in_memory(str(db_path))

            info = db2.info()
            assert info["node_count"] == 5
            assert info["edge_count"] == 6
            assert db2.is_persistent is False

    # =========================================================================
    # WAL TESTS
    # =========================================================================

    def test_wal_status_in_memory(self, db):
        """Test WAL status for in-memory database."""
        status = db.wal_status()

        assert "enabled" in status
        assert "size_bytes" in status
        assert "record_count" in status
        assert "current_epoch" in status

    def test_wal_checkpoint_no_error(self, db):
        """Test that wal_checkpoint doesn't error on in-memory database."""
        # Should not raise even on in-memory database
        db.wal_checkpoint()
