"""GQL hybrid search integration tests."""

import pytest

try:
    from grafeo import GrafeoDB

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def hybrid_db(db):
    """Database with both text and vector indexes."""
    db.create_node(
        ["Doc"],
        {"content": "Rust graph database engine", "emb": [1.0, 0.0, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Python machine learning", "emb": [0.0, 1.0, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Rust systems programming", "emb": [0.9, 0.1, 0.0]},
    )
    db.create_node(
        ["Doc"],
        {"content": "Graph neural network", "emb": [0.5, 0.5, 0.0]},
    )

    db.create_text_index("Doc", "content")
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    return db


class TestHybridSearch:
    def test_hybrid_search_basic(self, hybrid_db):
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="Rust graph",
            k=4,
            query_vector=[1.0, 0.0, 0.0],
        )
        assert len(results) > 0

    def test_hybrid_search_text_only(self, hybrid_db):
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="Rust",
            k=4,
        )
        assert len(results) > 0

    def test_hybrid_search_no_text_matches(self, hybrid_db):
        # Even with no text matches, vector search may contribute
        results = hybrid_db.hybrid_search(
            "Doc",
            text_property="content",
            vector_property="emb",
            query_text="nonexistentxyzquery",
            k=4,
            query_vector=[0.0, 0.0, 0.0],
        )
        # Should not error
        assert isinstance(results, list)


@pytest.fixture
def city_db(db):
    """Notes from Amsterdam and Berlin, with text and vector indexes.

    For "canals" the text index ranks Alix, Vincent, Jules; for [1, 0] the
    vector index ranks Gus, Mia, Vincent, Jules. Alix (a text match only) and
    Gus (a vector match only) live in Amsterdam.
    """
    for owner, city, rank, text, emb in [
        ("Alix", "Amsterdam", 3, "Alix rides along canals, canals and canals", [0.0, 1.0]),
        ("Gus", "Amsterdam", 19, "Gus buys museum tickets", [1.0, 0.0]),
        ("Vincent", "Berlin", 88, "Vincent paints canals", [0.8, 0.6]),
        ("Mia", "Berlin", 3, "Mia dances in Berlin clubs", [0.95, 0.31]),
        ("Jules", "Berlin", 19, "Jules swims past the old canals of Berlin at dawn", [0.6, 0.8]),
    ]:
        db.create_node(
            ["Doc"],
            {"owner": owner, "city": city, "rank": rank, "text": text, "emb": emb},
        )
    db.create_text_index("Doc", "text")
    db.create_vector_index("Doc", "emb", dimensions=2, metric="cosine")
    return db


def _owners(db, results):
    return [db.get_node(node_id).properties()["owner"] for node_id, _ in results]


class TestHybridSearchFilters:
    def test_filters_narrow_the_text_and_the_vector_search(self, city_db):
        unfiltered = _owners(
            city_db,
            city_db.hybrid_search("Doc", "text", "emb", "canals", 10, query_vector=[1.0, 0.0]),
        )
        assert "Alix" in unfiltered and "Gus" in unfiltered, unfiltered

        berlin = city_db.hybrid_search(
            "Doc", "text", "emb", "canals", 10, query_vector=[1.0, 0.0], filters={"city": "Berlin"}
        )
        assert sorted(_owners(city_db, berlin)) == ["Jules", "Mia", "Vincent"]

    def test_k_counts_the_matching_nodes(self, city_db):
        top_two = city_db.hybrid_search("Doc", "text", "emb", "canals", 2, query_vector=[1.0, 0.0])
        assert _owners(city_db, top_two) == ["Vincent", "Jules"]

        amsterdam = city_db.hybrid_search(
            "Doc",
            "text",
            "emb",
            "canals",
            2,
            query_vector=[1.0, 0.0],
            filters={"city": "Amsterdam"},
        )
        assert _owners(city_db, amsterdam) == ["Alix", "Gus"]

    def test_operator_filters_and_weighted_fusion(self, city_db):
        results = city_db.hybrid_search(
            "Doc",
            "text",
            "emb",
            "canals",
            10,
            query_vector=[1.0, 0.0],
            fusion="weighted",
            filters={"rank": {"$gt": 3}},
        )
        assert _owners(city_db, results) == ["Vincent", "Gus", "Jules"]

    def test_a_filter_no_node_matches_finds_nothing(self, city_db):
        assert (
            city_db.hybrid_search(
                "Doc",
                "text",
                "emb",
                "canals",
                10,
                query_vector=[1.0, 0.0],
                filters={"city": "Paris"},
            )
            == []
        )
