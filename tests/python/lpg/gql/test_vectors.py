"""Tests for vector support in Python bindings.

Covers: list[float] → Vector conversion, vector() function,
cosine_similarity/distance functions, and create_vector_index().
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed"
)


@pytest.fixture
def db():
    """Create a fresh in-memory database with vector test data."""
    db = grafeo.GrafeoDB()
    return db


class TestVectorConversion:
    """Test Python list[float] → Value::Vector auto-conversion."""

    def test_list_float_stored_as_vector(self, db):
        """A Python list of floats should be stored as Value::Vector."""
        node = db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        result = db.execute(
            f"MATCH (n:Doc) WHERE id(n) = {node.id} RETURN n.embedding AS emb"
        )
        rows = list(result)
        assert len(rows) == 1
        emb = rows[0]["emb"]
        assert emb is not None
        assert isinstance(emb, list)
        assert len(emb) == 3
        assert abs(emb[0] - 1.0) < 0.001
        assert abs(emb[1]) < 0.001
        assert abs(emb[2]) < 0.001

    def test_vector_roundtrip(self, db):
        """Vectors should survive store → retrieve roundtrip."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        db.create_node(["Vec"], {"data": original})
        result = db.execute("MATCH (n:Vec) RETURN n.data AS d")
        rows = list(result)
        retrieved = rows[0]["d"]
        assert len(retrieved) == len(original)
        for orig, ret in zip(original, retrieved):
            assert abs(orig - ret) < 0.001

    def test_empty_list_stays_list(self, db):
        """An empty list should remain a List, not become a Vector."""
        db.create_node(["Empty"], {"items": []})
        result = db.execute("MATCH (n:Empty) RETURN n.items AS items")
        rows = list(result)
        # Empty list should be retrievable (may be None or empty list)
        assert len(rows) == 1

    def test_mixed_list_stays_list(self, db):
        """A list with mixed types should remain a List."""
        db.create_node(["Mixed"], {"data": [1, "two", 3.0]})
        result = db.execute("MATCH (n:Mixed) RETURN n.data AS d")
        rows = list(result)
        assert len(rows) == 1


class TestGqlVectorFunction:
    """Test the vector() function in GQL queries."""

    def test_vector_literal_insertion(self, db):
        """INSERT with vector() should store a proper Vector value."""
        db.execute("INSERT (:Doc {id: 'a', embedding: vector([1.0, 0.0, 0.0])})")
        result = db.execute("MATCH (n:Doc) WHERE n.id = 'a' RETURN n.embedding AS emb")
        rows = list(result)
        assert len(rows) == 1
        emb = rows[0]["emb"]
        assert emb is not None
        assert len(emb) == 3
        assert abs(emb[0] - 1.0) < 0.001

    def test_vector_in_with_return(self, db):
        """vector() should work in WITH/RETURN expressions."""
        db.create_node(["Dummy"], {"x": 1})
        result = db.execute("""
            MATCH (n:Dummy)
            WITH vector([0.5, 0.5, 0.0]) AS v
            RETURN v
        """)
        rows = list(result)
        assert len(rows) == 1
        v = rows[0]["v"]
        assert v is not None
        assert len(v) == 3
        assert abs(v[0] - 0.5) < 0.001


class TestCosineSimilarity:
    """Test cosine_similarity() and other distance functions."""

    def setup_vectors(self, db):
        """Insert test vectors."""
        db.create_node(["Doc"], {"id": "a", "embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"id": "b", "embedding": [0.9, 0.1, 0.0]})
        db.create_node(["Doc"], {"id": "c", "embedding": [0.0, 1.0, 0.0]})

    def test_cosine_similarity_identical(self, db):
        """Cosine similarity of identical vectors should be ~1.0."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc) WHERE n.id = 'a'
            RETURN cosine_similarity(n.embedding, vector([1.0, 0.0, 0.0])) AS s
        """)
        rows = list(result)
        assert len(rows) == 1
        s = rows[0]["s"]
        assert s is not None
        assert abs(s - 1.0) < 0.01

    def test_cosine_similarity_orthogonal(self, db):
        """Cosine similarity of orthogonal vectors should be ~0.0."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc) WHERE n.id = 'c'
            RETURN cosine_similarity(n.embedding, vector([1.0, 0.0, 0.0])) AS s
        """)
        rows = list(result)
        assert len(rows) == 1
        s = rows[0]["s"]
        assert s is not None
        assert abs(s) < 0.01

    def test_cosine_similarity_ordering(self, db):
        """Cosine similarity should rank vectors correctly."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc)
            WITH n, cosine_similarity(n.embedding, vector([1.0, 0.0, 0.0])) AS s
            RETURN n.id AS id, s
            ORDER BY s DESC
        """)
        rows = list(result)
        assert len(rows) == 3
        # 'a' should be first (most similar), 'c' last
        assert rows[0]["id"] == "a"
        assert rows[2]["id"] == "c"

    def test_euclidean_distance(self, db):
        """Euclidean distance between identical vectors should be ~0."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc) WHERE n.id = 'a'
            RETURN euclidean_distance(n.embedding, vector([1.0, 0.0, 0.0])) AS d
        """)
        rows = list(result)
        d = rows[0]["d"]
        assert d is not None
        assert abs(d) < 0.01

    def test_dot_product(self, db):
        """Dot product of unit vector with itself should be 1.0."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc) WHERE n.id = 'a'
            RETURN dot_product(n.embedding, vector([1.0, 0.0, 0.0])) AS dp
        """)
        rows = list(result)
        dp = rows[0]["dp"]
        assert dp is not None
        assert abs(dp - 1.0) < 0.01

    def test_manhattan_distance(self, db):
        """Manhattan distance between identical vectors should be ~0."""
        self.setup_vectors(db)
        result = db.execute("""
            MATCH (n:Doc) WHERE n.id = 'a'
            RETURN manhattan_distance(n.embedding, vector([1.0, 0.0, 0.0])) AS d
        """)
        rows = list(result)
        d = rows[0]["d"]
        assert d is not None
        assert abs(d) < 0.01


class TestVectorFunction:
    """Test the grafeo.vector() Python function."""

    def test_vector_function_exists(self):
        """grafeo.vector() should be available."""
        assert hasattr(grafeo, "vector")

    def test_vector_function_returns_list(self):
        """grafeo.vector() should return a list of floats."""
        vec = grafeo.vector([1.0, 2.0, 3.0])
        assert isinstance(vec, list)
        assert len(vec) == 3

    def test_vector_function_empty_raises(self):
        """grafeo.vector([]) should raise ValueError."""
        with pytest.raises(ValueError):
            grafeo.vector([])

    def test_vector_as_property(self, db):
        """grafeo.vector() result should be usable as a property value."""
        vec = grafeo.vector([0.5, 0.5, 0.0])
        db.create_node(["Test"], {"data": vec})
        result = db.execute("MATCH (n:Test) RETURN n.data AS d")
        rows = list(result)
        assert rows[0]["d"] is not None
        assert len(rows[0]["d"]) == 3


class TestCreateVectorIndex:
    """Test create_vector_index() method."""

    def test_create_vector_index_basic(self, db):
        """Basic vector index creation should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"embedding": [0.0, 1.0, 0.0]})
        db.create_vector_index("Doc", "embedding")

    def test_create_vector_index_with_metric(self, db):
        """Vector index with explicit metric should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_vector_index("Doc", "embedding", metric="euclidean")

    def test_create_vector_index_with_dimensions(self, db):
        """Vector index with explicit dimensions should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_vector_index("Doc", "embedding", dimensions=3)

    def test_create_vector_index_no_vectors_fails(self, db):
        """Creating index on nodes without vectors should fail."""
        db.create_node(["Doc"], {"name": "no embedding"})
        with pytest.raises(RuntimeError, match="No vector properties"):
            db.create_vector_index("Doc", "embedding")

    def test_create_vector_index_dimension_mismatch_fails(self, db):
        """Dimension mismatch should fail."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            db.create_vector_index("Doc", "embedding", dimensions=5)

    def test_create_vector_index_invalid_metric_fails(self, db):
        """Invalid metric should fail."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        with pytest.raises(RuntimeError, match="Unknown distance metric"):
            db.create_vector_index("Doc", "embedding", metric="invalid")

    def test_schema_ddl_error_message(self, db):
        """CREATE VECTOR INDEX via execute() should give helpful error."""
        with pytest.raises(RuntimeError, match="create_vector_index"):
            db.execute("CREATE VECTOR INDEX idx ON :Doc(embedding)")

    def test_create_vector_index_with_m(self, db):
        """Vector index with custom m parameter should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"embedding": [0.0, 1.0, 0.0]})
        db.create_vector_index("Doc", "embedding", m=32)

    def test_create_vector_index_with_ef_construction(self, db):
        """Vector index with custom ef_construction should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"embedding": [0.0, 1.0, 0.0]})
        db.create_vector_index("Doc", "embedding", ef_construction=200)

    def test_create_vector_index_all_tuning_params(self, db):
        """Vector index with all tuning parameters should succeed."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"embedding": [0.0, 1.0, 0.0]})
        db.create_vector_index(
            "Doc", "embedding", metric="cosine", m=32, ef_construction=256
        )


class TestVectorSearch:
    """Test vector_search() method."""

    def setup_data(self, db):
        """Insert test vectors and build index."""
        db.create_node(["Doc"], {"id": "a", "embedding": [1.0, 0.0, 0.0]})
        db.create_node(["Doc"], {"id": "b", "embedding": [0.9, 0.1, 0.0]})
        db.create_node(["Doc"], {"id": "c", "embedding": [0.0, 1.0, 0.0]})
        db.create_vector_index("Doc", "embedding", metric="cosine")

    def test_vector_search_basic(self, db):
        """Basic vector search should return k results."""
        self.setup_data(db)
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        # Each result is (node_id, distance)
        for node_id, distance in results:
            assert isinstance(node_id, int)
            assert isinstance(distance, float)

    def test_vector_search_ordering(self, db):
        """Results should be sorted by distance ascending (closest first)."""
        self.setup_data(db)
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=3)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_vector_search_closest_first(self, db):
        """The closest vector should be first."""
        self.setup_data(db)
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=3)
        # First result should be the identical vector (distance ~0)
        _, first_dist = results[0]
        assert first_dist < 0.01
        # Last result should be the orthogonal vector (distance ~1.0)
        _, last_dist = results[2]
        assert last_dist > 0.5

    def test_vector_search_k_limits(self, db):
        """k should limit the number of results."""
        self.setup_data(db)
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=2)
        assert len(results) == 2

    def test_vector_search_with_ef(self, db):
        """Explicit ef parameter should work."""
        self.setup_data(db)
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=3, ef=200)
        assert len(results) == 3
        # Higher ef should still return correct ordering
        _, first_dist = results[0]
        assert first_dist < 0.01

    def test_vector_search_no_index_fails(self, db):
        """Searching without an index should fail."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        with pytest.raises(RuntimeError, match="No vector index"):
            db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=1)

    def test_vector_search_euclidean(self, db):
        """Vector search with euclidean metric should work."""
        db.create_node(["Vec"], {"data": [1.0, 0.0, 0.0]})
        db.create_node(["Vec"], {"data": [0.0, 1.0, 0.0]})
        db.create_vector_index("Vec", "data", metric="euclidean")
        results = db.vector_search("Vec", "data", [1.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        # Identical vector has distance ~0
        _, first_dist = results[0]
        assert first_dist < 0.01


class TestBatchCreateNodes:
    """Test batch_create_nodes() method."""

    def test_batch_create_nodes_basic(self, db):
        """Batch insert should create multiple nodes."""
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ids = db.batch_create_nodes("Doc", "embedding", vectors)
        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique IDs

    def test_batch_create_nodes_properties_stored(self, db):
        """Batch-inserted vectors should be retrievable."""
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        ids = db.batch_create_nodes("Vec", "data", vectors)
        node = db.get_node(ids[0])
        assert node is not None
        assert "Vec" in node.labels
        # Vector property should exist
        emb = node.properties.get("data")
        assert emb is not None
        assert len(emb) == 2

    def test_batch_create_nodes_indexable(self, db):
        """Batch-inserted nodes should be indexable and searchable."""
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        db.batch_create_nodes("Doc", "embedding", vectors)
        db.create_vector_index("Doc", "embedding", metric="cosine")
        results = db.vector_search("Doc", "embedding", [1.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        # Closest should be nearly 0 distance
        _, first_dist = results[0]
        assert first_dist < 0.01

    def test_batch_create_nodes_empty(self, db):
        """Batch insert with empty list should return empty."""
        ids = db.batch_create_nodes("Doc", "embedding", [])
        assert len(ids) == 0


class TestBatchVectorSearch:
    """Test batch_vector_search() method."""

    def setup_data(self, db):
        """Insert test vectors and build index."""
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        db.batch_create_nodes("Doc", "embedding", vectors)
        db.create_vector_index("Doc", "embedding", metric="cosine")

    def test_batch_vector_search_basic(self, db):
        """Batch search should return results for each query."""
        self.setup_data(db)
        queries = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        results = db.batch_vector_search("Doc", "embedding", queries, k=2)
        assert len(results) == 2
        for result in results:
            assert len(result) == 2
            for node_id, distance in result:
                assert isinstance(node_id, int)
                assert isinstance(distance, float)

    def test_batch_vector_search_with_ef(self, db):
        """Batch search with explicit ef should work."""
        self.setup_data(db)
        queries = [[1.0, 0.0, 0.0]]
        results = db.batch_vector_search("Doc", "embedding", queries, k=3, ef=200)
        assert len(results) == 1
        assert len(results[0]) == 3

    def test_batch_vector_search_closest_correct(self, db):
        """Each query's closest result should be the matching vector."""
        self.setup_data(db)
        queries = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        results = db.batch_vector_search("Doc", "embedding", queries, k=1)
        assert len(results) == 3
        for result in results:
            assert len(result) == 1
            _, dist = result[0]
            assert dist < 0.01  # Each query matches its vector exactly

    def test_batch_vector_search_no_index_fails(self, db):
        """Batch searching without an index should fail."""
        db.create_node(["Doc"], {"embedding": [1.0, 0.0, 0.0]})
        with pytest.raises(RuntimeError, match="No vector index"):
            db.batch_vector_search("Doc", "embedding", [[1.0, 0.0, 0.0]], k=1)
