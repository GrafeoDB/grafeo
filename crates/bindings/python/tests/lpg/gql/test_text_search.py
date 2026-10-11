"""GQL text search integration tests."""

import pytest

try:
    from grafeo import GrafeoDB, GrafeoError

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False


@pytest.fixture
def db():
    if not GRAFEO_AVAILABLE:
        pytest.skip("grafeo not installed")
    return GrafeoDB()


@pytest.fixture
def text_db(db):
    """Database with text-indexed articles."""
    db.create_node(["Article"], {"title": "Rust graph database engine"})
    db.create_node(["Article"], {"title": "Python machine learning"})
    db.create_node(["Article"], {"title": "Rust systems programming"})
    db.create_text_index("Article", "title")
    return db


class TestTextSearch:
    def test_text_search_basic(self, text_db):
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 2

    def test_text_search_no_matches(self, text_db):
        results = text_db.text_search("Article", "title", "nonexistentxyz", k=10)
        assert len(results) == 0

    def test_text_search_after_mutation(self, text_db):
        text_db.create_node(["Article"], {"title": "Rust web framework"})
        results = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(results) >= 3

    def test_drop_and_rebuild_text_index(self, text_db):
        # Search works
        r1 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r1) > 0

        # Drop
        text_db.drop_text_index("Article", "title")

        # Rebuild
        text_db.rebuild_text_index("Article", "title")

        # Search works again
        r2 = text_db.text_search("Article", "title", "Rust", k=10)
        assert len(r2) > 0

    def test_text_search_no_index_error(self, db):
        db.create_node(["Article"], {"title": "test"})
        with pytest.raises(Exception, match=r".+"):
            db.text_search("Article", "title", "test", k=10)


@pytest.fixture
def notes_db(db):
    """Notes in Chinese and Russian, with a city and a rank."""
    for owner, city, rank, body in [
        ("Alix", "Berlin", 3, "阿利克斯住在柏林"),
        ("Gus", "Amsterdam", 19, "古斯住在阿姆斯特丹"),
        ("Vincent", "Berlin", 88, "Винсент живёт в Берлине"),
        ("Mia", "Prague", 19, "Мия и Жюль в Праге"),
    ]:
        db.create_node(["Note"], {"owner": owner, "city": city, "rank": rank, "body": body})
    return db


def _owners(db, results):
    return sorted(db.get_node(node_id).properties()["owner"] for node_id, _ in results)


class TestTextIndexOptions:
    def test_the_cjk_bigram_tokenizer_finds_words_inside_sentences(self, notes_db):
        notes_db.create_text_index(
            "Note", "body", tokenizer="cjk_bigram", stop_words=["住在", "и"], k1=0.3, b=0.19
        )
        assert _owners(notes_db, notes_db.text_search("Note", "body", "柏林", 10)) == ["Alix"]
        assert notes_db.text_search("Note", "body", "住在", 10) == [], "a stop word"
        assert notes_db.text_search("Note", "body", "и", 10) == [], "a stop word"
        assert _owners(notes_db, notes_db.text_search("Note", "body", "БЕРЛИНЕ", 10)) == ["Vincent"]

    def test_the_default_tokenizer_keeps_a_chinese_sentence_whole(self, notes_db):
        notes_db.create_text_index("Note", "body")
        assert notes_db.text_search("Note", "body", "柏林", 10) == []

    def test_options_out_of_range_raise(self, notes_db):
        for kwargs in [{"k1": -0.3}, {"b": 1.88}, {"tokenizer": "jieba"}]:
            with pytest.raises(GrafeoError, match="GRAFEO-V001"):
                notes_db.create_text_index("Note", "body", **kwargs)

    def test_rebuild_keeps_the_options(self, notes_db):
        notes_db.create_text_index("Note", "body", tokenizer="cjk_bigram")
        notes_db.rebuild_text_index("Note", "body")
        assert _owners(notes_db, notes_db.text_search("Note", "body", "柏林", 10)) == ["Alix"]


class TestTextSearchFilters:
    def test_filters_keep_the_matching_nodes_with_their_scores(self, notes_db):
        notes_db.create_text_index("Note", "body", tokenizer="cjk_bigram")
        everything = dict(notes_db.text_search("Note", "body", "住在", 10))
        assert len(everything) == 2, "Alix and Gus"
        berlin = notes_db.text_search("Note", "body", "住在", 10, filters={"city": "Berlin"})
        assert _owners(notes_db, berlin) == ["Alix"]
        assert dict(berlin).items() <= everything.items(), "the scores of the whole index"
        ranked = notes_db.text_search("Note", "body", "住在", 1, filters={"rank": {"$gt": 3}})
        assert _owners(notes_db, ranked) == ["Gus"]
