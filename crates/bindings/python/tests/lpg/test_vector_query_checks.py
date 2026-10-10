"""A query vector the index cannot measure is an error, not a crash (#593).

A query vector of another size than the index, or with NaN or an infinity,
raises `grafeo.GrafeoError` (code GRAFEO-V001) naming the index and both
sizes, or the value, from every search call; it used to raise a
`PanicException`, which `except Exception` does not catch, or return NaN
distances. The database answers as before afterwards. Removing a vector
keeps the others findable (#600).
"""

import math

import grafeo
import pytest

SHORT = ([0.3, 0.19], "the query vector has 2 dimensions; the index on :Doc(emb) expects 3")
LONG = (
    [0.3, 0.19, 0.88, 0.3],
    "the query vector has 4 dimensions; the index on :Doc(emb) expects 3",
)
NAN = ([0.9, math.nan, 0.0], "the query vector has NaN at position 1")
INFINITY = ([0.9, math.inf, 0.0], "the query vector has inf at position 1")
CASES = [SHORT, LONG, NAN, INFINITY]
IDS = ["short", "long", "nan", "infinity"]


@pytest.fixture
def docs():
    db = grafeo.GrafeoDB()
    for vector, city in [
        ([1.0, 0.0, 0.0], "Amsterdam"),
        ([0.0, 1.0, 0.0], "Berlin"),
        ([0.0, 0.0, 1.0], "Paris"),
        ([0.5, 0.5, 0.0], "Prague"),
    ]:
        db.create_node(["Doc"], {"emb": vector, "text": f"graph notes from {city}"})
    db.create_vector_index("Doc", "emb", dimensions=3)
    db.create_text_index("Doc", "text")
    return db


def assert_still_answers(db):
    hits = db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 1)
    assert len(hits) == 1
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)


def assert_refused(call, message):
    with pytest.raises(grafeo.GrafeoError) as caught:
        call()
    assert caught.value.error_code == "GRAFEO-V001", str(caught.value)
    assert message in str(caught.value)


@pytest.mark.parametrize(("query", "message"), CASES, ids=IDS)
def test_every_search_call_refuses_a_query_vector_the_index_cannot_measure(docs, query, message):
    assert_refused(lambda: docs.vector_search("Doc", "emb", query, 2), message)
    assert_refused(
        lambda: docs.vector_search(
            "Doc", "emb", query, 2, filters={"text": "graph notes from Paris"}
        ),
        message,
    )
    assert_refused(
        lambda: docs.batch_vector_search("Doc", "emb", [[1.0, 0.0, 0.0], query], 2),
        message,
    )
    assert_refused(lambda: docs.mmr_search("Doc", "emb", query, 2), message)
    assert_refused(
        lambda: docs.hybrid_search("Doc", "text", "emb", "graph", 2, query_vector=query),
        message,
    )
    assert_still_answers(docs)


@pytest.mark.parametrize(("query", "message"), [SHORT, LONG], ids=["short", "long"])
def test_search_procedures_refuse_a_query_vector_of_another_size(docs, query, message):
    literal = "[" + ", ".join(str(value) for value in query) + "]"
    assert_refused(
        lambda: docs.execute(f"CALL grafeo.search.vector('Doc', 'emb', {literal}, 2)"),
        message,
    )
    assert_refused(
        lambda: docs.execute(f"CALL grafeo.search.mmr('Doc', 'emb', {literal}, 2, 3, 0.5)"),
        message,
    )
    assert_still_answers(docs)


@pytest.mark.parametrize(("query", "message"), CASES, ids=IDS)
def test_a_vector_predicate_on_an_indexed_property_refuses_it(docs, query, message):
    assert_refused(
        lambda: docs.execute(
            "MATCH (d:Doc) WHERE cosine_similarity(d.emb, $q) > 0.1 RETURN d.text",
            {"q": query},
        ),
        message,
    )
    assert_still_answers(docs)


def test_an_indexed_property_refuses_nan(docs):
    assert_refused(
        lambda: docs.create_node(["Doc"], {"emb": [0.9, math.nan, 0.0]}),
        "property 'emb' on :Doc has a vector index, which cannot measure NaN (at position 1)",
    )
    # Without an index, the vector is a value like any other.
    docs.create_node(["Draft"], {"emb": [0.9, math.nan, 0.0]})
    assert_still_answers(docs)


def test_vector_index_arguments_are_invalid_values(docs):
    assert_refused(
        lambda: docs.create_vector_index("Doc", "other", dimensions=0),
        "a vector index needs at least 1 dimension",
    )
    assert_refused(
        lambda: docs.create_vector_index("Doc", "other", dimensions=3, metric="hamming"),
        "Unknown distance metric 'hamming'",
    )


def test_removing_a_vector_keeps_the_others_findable():
    db = grafeo.GrafeoDB()
    ids = [db.create_node(["Item"], {"embedding": [1.0, i / 100, 0.0, 0.0]}).id for i in range(8)]
    db.create_vector_index("Item", "embedding", dimensions=4, metric="euclidean")
    db.remove_node_property(ids[3], "embedding")
    found = sorted(
        node for node, _ in db.vector_search("Item", "embedding", [1.0, 0.0, 0.0, 0.0], 8)
    )
    assert found == sorted(ids[:3] + ids[4:])
