"""Persona: Ravi builds retrieval for a RAG pipeline in a Jupyter notebook.

He stores documents with embeddings from a model (numpy arrays, float32 or
float64), creates a vector and a text index, searches by top-k, by a
similarity threshold in a query, with filters, in batches, hybrid and with
MMR, exports to pandas and arrow, and compares two embedding runs with time
travel. His slips: a query vector of the wrong size, a NaN from a broken
model, searching before creating the index, a run without history kept.
"""

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

TOPICS = ["graphs", "canals", "cycling"]


def topic_vector(topic, dtype=np.float32):
    vector = np.zeros(4, dtype=dtype)
    vector[TOPICS.index(topic)] = 1.0
    return vector


def make_db(**options):
    """Twelve notes, four per topic, each embedding near its topic's axis."""
    rng = np.random.default_rng(7)
    db = grafeo.GrafeoDB(**options)
    for i in range(12):
        topic = TOPICS[i % 3]
        embedding = topic_vector(topic) + rng.normal(0, 0.05, 4).astype(np.float32)
        db.execute(
            "INSERT (:Doc {title: $title, topic: $topic, body: $body, emb: $emb})",
            {
                "title": f"d{i:02}",
                "topic": topic,
                "body": f"a note about {topic}",
                "emb": embedding,
            },
        )
    return db


@pytest.fixture
def db():
    db = make_db()
    db.create_vector_index("Doc", "emb", dimensions=4, metric="cosine")
    db.create_text_index("Doc", "body")
    return db


def topics_of(db, hits):
    """The topic of each (node id, score) hit, in order."""
    return [db.get_node(node_id).properties()["topic"] for node_id, _ in hits]


# ---------------------------------------------------------------------------
# Embeddings in and out
# ---------------------------------------------------------------------------


def test_persona_ravi_numpy_embeddings_are_stored_as_vectors(db):
    pytest.importorskip("pyarrow")
    row = db.execute("MATCH (d:Doc {title: 'd00'}) RETURN d.emb AS e").to_list()[0]
    assert len(row["e"]) == 4
    assert row["e"][0] == pytest.approx(1.0, abs=0.2)
    schema = db.execute("MATCH (d:Doc) RETURN d.title AS t, d.emb AS e LIMIT 1").to_arrow().schema
    assert str(schema.field("e").type) == "fixed_size_list<item: float not null>[4]"


def test_persona_ravi_exports_to_pandas_and_arrow(db):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = db.execute(
        "MATCH (d:Doc) RETURN d.title AS title, d.topic AS topic, d.emb AS emb ORDER BY title"
    ).to_pandas()
    assert list(frame.columns) == ["title", "topic", "emb"]
    assert len(frame) == 12
    assert frame["topic"].value_counts().to_dict() == {"graphs": 4, "canals": 4, "cycling": 4}
    nodes = db.nodes_to_pandas()
    assert {"_id", "_labels", "title", "topic", "body", "emb"} <= set(nodes.columns)
    assert len(nodes) == 12
    table = db.execute("MATCH (d:Doc) RETURN d.title AS title ORDER BY title").to_arrow()
    assert table.num_rows == 12
    assert table.column("title").to_pylist()[:2] == ["d00", "d01"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_persona_ravi_top_k_finds_the_topic_from_any_array_kind(db):
    for topic in TOPICS:
        want = db.vector_search("Doc", "emb", topic_vector(topic), 4)
        assert topics_of(db, want) == [topic] * 4
        assert db.vector_search("Doc", "emb", topic_vector(topic, np.float64), 4) == want
        assert db.vector_search("Doc", "emb", topic_vector(topic).tolist(), 4) == want
        as_ints = [int(x) for x in topic_vector(topic)]
        assert db.vector_search("Doc", "emb", as_ints, 4) == want


def test_persona_ravi_a_threshold_and_top_k_in_a_query(db):
    query = topic_vector("canals")
    rows = db.execute(
        "MATCH (d:Doc) WHERE cosine_similarity(d.emb, $q) > 0.9 "
        "RETURN d.title AS t, d.topic AS topic, cosine_similarity(d.emb, $q) AS s "
        "ORDER BY s DESC LIMIT 3",
        {"q": query},
    ).to_list()
    assert [row["topic"] for row in rows] == ["canals"] * 3
    scores = [row["s"] for row in rows]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0.9 for score in scores)
    # The same three as the index's top three.
    hits = db.vector_search("Doc", "emb", query, 3)
    titles = [db.get_node(node_id).properties()["title"] for node_id, _ in hits]
    assert [row["t"] for row in rows] == titles


def test_persona_ravi_a_threshold_nobody_passes_gives_no_rows(db):
    rows = db.execute(
        "MATCH (d:Doc) WHERE cosine_similarity(d.emb, $q) > 0.999999 RETURN d.title AS t LIMIT 3",
        {"q": [0.5, 0.5, 0.5, 0.5]},
    ).to_list()
    assert rows == []


def test_persona_ravi_filters_keep_one_topic(db):
    hits = db.vector_search("Doc", "emb", topic_vector("graphs"), 3, filters={"topic": "canals"})
    assert topics_of(db, hits) == ["canals"] * 3


def test_persona_ravi_a_batch_gives_each_queries_hits(db):
    queries = np.stack([topic_vector(topic) for topic in TOPICS])
    batch = db.batch_vector_search("Doc", "emb", queries, 2)
    assert batch == [db.vector_search("Doc", "emb", query, 2) for query in queries]


def test_persona_ravi_hybrid_search(db):
    hits = db.hybrid_search("Doc", "body", "emb", "canals", 3, query_vector=topic_vector("canals"))
    assert topics_of(db, hits) == ["canals"] * 3
    filtered = db.hybrid_search(
        "Doc",
        "body",
        "emb",
        "note",
        3,
        query_vector=topic_vector("canals"),
        filters={"topic": "cycling"},
    )
    assert topics_of(db, filtered) == ["cycling"] * 3


def test_persona_ravi_mmr_starts_at_the_nearest_and_spreads(db):
    query = topic_vector("graphs")
    nearest = db.vector_search("Doc", "emb", query, 1)[0][0]
    hits = db.mmr_search("Doc", "emb", query, 3, fetch_k=12, lambda_mult=0.3)
    assert len(hits) == 3
    assert hits[0][0] == nearest
    assert len({node_id for node_id, _ in hits}) == 3
    assert len(set(topics_of(db, hits))) > 1, "a low lambda trades relevance for diversity"
    # A lambda of 1 is plain relevance: the top-k.
    relevant = db.mmr_search("Doc", "emb", query, 3, fetch_k=12, lambda_mult=1.0)
    assert [node_id for node_id, _ in relevant] == [
        node_id for node_id, _ in db.vector_search("Doc", "emb", query, 3)
    ]


# ---------------------------------------------------------------------------
# His slips
# ---------------------------------------------------------------------------


def test_persona_ravi_searches_before_creating_the_index():
    db = make_db()
    with pytest.raises(grafeo.GrafeoError, match="no vector index on :Doc\\(emb\\); create one"):
        db.vector_search("Doc", "emb", topic_vector("graphs"), 3)


def test_persona_ravi_a_query_vector_of_the_wrong_size(db):
    with pytest.raises(
        grafeo.GrafeoError, match="3 dimensions; the index on :Doc\\(emb\\) expects 4"
    ):
        db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 3)


def test_persona_ravi_a_nan_from_a_broken_model(db):
    with pytest.raises(grafeo.GrafeoError, match="NaN at position 0"):
        db.vector_search("Doc", "emb", [float("nan"), 0.0, 0.0, 0.0], 3)


def test_persona_ravi_k_of_zero_gives_nothing(db):
    assert db.vector_search("Doc", "emb", topic_vector("graphs"), 0) == []


# ---------------------------------------------------------------------------
# Comparing two embedding runs
# ---------------------------------------------------------------------------


def test_persona_ravi_compares_two_embedding_runs():
    db = make_db(history_retention="forever")
    db.create_vector_index("Doc", "emb", dimensions=4, metric="cosine")
    first_run = db.current_epoch()
    # The second run moves every graphs note onto the canals axis.
    db.execute("MATCH (d:Doc {topic: 'graphs'}) SET d.emb = $v", {"v": topic_vector("canals")})
    query = (
        "MATCH (d:Doc) RETURN d.topic AS topic "
        "ORDER BY cosine_similarity(d.emb, vector([1.0, 0.0, 0.0, 0.0])) DESC LIMIT 4"
    )
    then = db.execute_at_epoch(query, first_run).to_list()
    assert [row["topic"] for row in then] == ["graphs"] * 4
    assert db.as_of(first_run).execute(query).to_list() == then
    assert db.execute_as_of(query, first_run).to_list() == then
    now = db.execute(query).to_list()
    assert [row["topic"] for row in now] != ["graphs"] * 4
    # The index answers for the current run.
    assert topics_of(db, db.vector_search("Doc", "emb", topic_vector("graphs"), 1)) != ["graphs"]


def test_persona_ravi_forgets_to_keep_history():
    db = make_db()
    first_run = db.current_epoch()
    db.execute("MATCH (d:Doc) SET d.emb = $v", {"v": topic_vector("canals")})
    db.execute("MATCH (d:Doc) SET d.topic = 'moved'")
    with pytest.raises(grafeo.GrafeoError, match="not retained"):
        db.execute_at_epoch("MATCH (d:Doc) RETURN d.topic AS t", first_run)
