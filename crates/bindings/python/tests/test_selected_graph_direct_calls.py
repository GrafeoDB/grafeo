"""Vector and text indexes and searches follow the selected graph (#677).

After ``set_graph("archive")``, queries read and write archive, and the
docstring of ``set_graph`` says the direct API follows it too. Creating a
vector or text index and searching must then act on archive, not on the
default graph. The default graph holds Amsterdam notes and archive Berlin
notes, with node ids that do not overlap, so each result tells which graph it
came from. The controls, on the default graph without a selection, pass
today.
"""

import pytest
from grafeo import GrafeoDB, GrafeoError

ARCHIVE = "archive"


DEFAULT_DOCS = [
    ([1.0, 0.0, 0.0], "Amsterdam canal notes by Alix"),
    ([0.9, 0.1, 0.0], "Amsterdam museum notes by Gus"),
]
ARCHIVE_DOCS = [
    ([0.0, 0.0, 1.0], "Berlin wall notes by Vincent"),
    ([0.0, 0.1, 0.9], "Berlin park notes by Jules"),
    ([0.1, 0.0, 0.9], "Berlin station notes by Mia"),
]
ARCHIVE_VECTOR_INDEX = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3 METRIC 'cosine'"


@pytest.fixture
def docs():
    db = GrafeoDB()
    default_docs = {db.create_node(["Doc"], {"emb": e, "body": b}).id for e, b in DEFAULT_DOCS}
    db.create_graph(ARCHIVE)
    archive = db.graph(ARCHIVE)
    # Two shelves first, so the archive's document ids are not the default
    # graph's, whether each graph counts ids on its own or not.
    for _ in range(2):
        archive.create_node(["Shelf"], {"city": "Paris"})
    archive_docs = {archive.create_node(["Doc"], {"emb": e, "body": b}).id for e, b in ARCHIVE_DOCS}
    assert default_docs.isdisjoint(archive_docs)
    return db, default_docs, archive_docs


def ids(results):
    return {node_id for node_id, _ in results}


def test_control_vector_index_and_search_on_the_default_graph(docs):
    db, default_docs, _ = docs
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    assert ids(db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 5)) == default_docs


def test_control_text_index_and_search_on_the_default_graph(docs):
    db, default_docs, _ = docs
    db.create_text_index("Doc", "body")
    assert ids(db.text_search("Doc", "body", "notes", 5)) == default_docs


def test_vector_index_and_search_follow_the_selected_graph(docs):
    db, default_docs, archive_docs = docs
    db.set_graph(ARCHIVE)
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    found = ids(db.vector_search("Doc", "emb", [0.0, 0.0, 1.0], 5))
    assert not found & default_docs, f"found the default graph's documents: {found}"
    assert found and found <= archive_docs, f"found {found}, expected among {archive_docs}"

    db.reset_graph()
    with pytest.raises(GrafeoError, match="there is no vector index"):
        db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 5)


def test_text_index_and_search_follow_the_selected_graph(docs):
    db, default_docs, archive_docs = docs
    db.set_graph(ARCHIVE)
    db.create_text_index("Doc", "body")
    found = ids(db.text_search("Doc", "body", "notes", 5))
    assert not found & default_docs, f"found the default graph's documents: {found}"
    assert found and found <= archive_docs, f"found {found}, expected among {archive_docs}"

    db.reset_graph()
    with pytest.raises(GrafeoError, match="there is no text index"):
        db.text_search("Doc", "body", "notes", 5)


def test_drop_vector_index_follows_the_selected_graph(docs):
    db, default_docs, _ = docs
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    db.graph(ARCHIVE).execute(ARCHIVE_VECTOR_INDEX)
    db.set_graph(ARCHIVE)
    assert db.drop_vector_index("Doc", "emb")
    db.reset_graph()
    assert ids(db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 5)) == default_docs
