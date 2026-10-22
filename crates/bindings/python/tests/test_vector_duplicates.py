"""Many identical vectors are all found (#391 follow-up).

More exact duplicates than an HNSW neighbor list holds (33 at the default
m = 16) crowded each other, and every vector inserted after them, out of the
index's graph, so a search silently missed them. Duplicates now share one
graph node: every vector is found, the duplicates at their distance and in id
order, counting toward k.
"""

import pytest
from grafeo import GrafeoDB

DUPLICATES = 40
DISTANT = 5


@pytest.fixture
def docs():
    db = GrafeoDB()
    duplicates = [db.create_node(["Doc"], {"emb": [1.0, 0.0, 0.0]}).id for _ in range(DUPLICATES)]
    distant = [db.create_node(["Doc"], {"emb": [0.0, 1.0, i * 0.1]}).id for i in range(DISTANT)]
    db.create_vector_index("Doc", "emb", dimensions=3, metric="cosine")
    return db, sorted(duplicates), sorted(distant)


def ids(results):
    return [node_id for node_id, _ in results]


def test_every_identical_vector_is_found(docs):
    db, duplicates, distant = docs
    found = ids(db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], DUPLICATES + DISTANT))
    assert found[:DUPLICATES] == duplicates
    assert sorted(found[DUPLICATES:]) == distant


def test_duplicates_count_toward_k_in_id_order(docs):
    db, duplicates, _ = docs
    assert ids(db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], 3)) == duplicates[:3]


def test_a_deleted_duplicate_leaves_the_others_found(docs):
    db, duplicates, distant = docs
    db.delete_node(duplicates[0])
    found = ids(db.vector_search("Doc", "emb", [1.0, 0.0, 0.0], DUPLICATES + DISTANT))
    assert found[: DUPLICATES - 1] == duplicates[1:]
    assert sorted(found[DUPLICATES - 1 :]) == distant
