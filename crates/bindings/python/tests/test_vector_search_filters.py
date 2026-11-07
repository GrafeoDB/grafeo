"""A filtered vector search finds the nearest nodes that match the filters, on every index.

`vector_search(..., filters=...)` searches among the nodes the filters allow. A quantized index
used to take the nearest nodes of the whole index first and drop the ones the filters reject, so
filters that allow only far nodes found fewer than k, or none. Each quantization and metric is
checked here against a brute force over the allowed vectors, with filters that allow no node, the
farthest node, the farthest k, the farthest 3k, every other node, every node, and only exact copies
of other vectors. The index has more vectors than a quantizer needs to train.
"""

from __future__ import annotations

import math

import pytest
from grafeo import GrafeoDB

pytestmark = pytest.mark.requires("vector-index")

DIMENSIONS = 8
ORIGINALS = 1200
COPIES = 3
K = 5
QUERY = [0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5]
METRICS = ["cosine", "euclidean", "dot_product", "manhattan"]
QUANTIZATIONS = [None, "scalar", "binary", "product"]
# The filters, by name: each shape is a boolean property of the nodes it allows.
SHAPES = ["none", "far1", "far_k", "far_3k", "half", "every", "copy"]
# Quantizations whose search with a beam over the whole index finds the nearest allowed nodes:
# product codes choose k of the candidates before the exact rescoring reorders them.
EXACT = {None, "scalar", "binary"}


def distance(metric: str, a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "cosine":
        return 1.0 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    if metric == "euclidean":
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    if metric == "dot_product":
        return -dot
    if metric == "manhattan":
        return sum(abs(x - y) for x, y in zip(a, b, strict=True))
    raise AssertionError(metric)


def vectors() -> list[list[float]]:
    """A fixed spread from a linear congruential sequence in [-1, 1), then copies of the first."""
    state = 0x2545F4914F6CDD1D
    found = []
    for _ in range(ORIGINALS):
        vector = []
        for _ in range(DIMENSIONS):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            vector.append((state >> 48) / 32768.0 - 1.0)
        found.append(vector)
    return found + [list(v) for v in found[:COPIES]]


@pytest.fixture(
    scope="module",
    params=[(q, m) for q in QUANTIZATIONS for m in METRICS],
    ids=lambda param: f"{param[0] or 'full'}-{param[1]}",
)
def index(request):
    quantization, metric = request.param
    points = vectors()
    by_far = sorted(range(ORIGINALS), key=lambda i: (-distance(metric, QUERY, points[i]), i))
    shapes = {
        "none": set(),
        "far1": set(by_far[:1]),
        "far_k": set(by_far[:K]),
        "far_3k": set(by_far[: 3 * K]),
        "half": set(range(0, len(points), 2)),
        "every": set(range(len(points))),
        "copy": set(range(ORIGINALS, ORIGINALS + COPIES)),
    }
    db = GrafeoDB()
    ids = []
    for i, vector in enumerate(points):
        properties = {"i": i, "emb": vector}
        properties.update({shape: True for shape, held in shapes.items() if i in held})
        ids.append(db.create_node(["Doc"], properties).id)
    db.create_vector_index(
        "Doc", "emb", dimensions=DIMENSIONS, metric=metric, quantization=quantization
    )
    allowed = {shape: {ids[i] for i in held} for shape, held in shapes.items()}
    vector_of = dict(zip(ids, points, strict=True))
    return db, quantization, metric, allowed, vector_of


def check(found, allowed, vector_of, metric, exact):
    """What is wrong with `found`, a search among `allowed`: [] when nothing is."""
    wrong = []
    nearest = sorted(distance(metric, QUERY, vector_of[i]) for i in allowed)[:K]
    ids = [node_id for node_id, _ in found]
    if len(ids) != len(nearest):
        wrong.append(f"{len(ids)} results {ids}, want {len(nearest)}")
    if not set(ids) <= allowed:
        wrong.append(f"{sorted(set(ids) - allowed)} not allowed")
    if len(set(ids)) != len(ids):
        wrong.append(f"a node twice in {ids}")
    distances = [d for _, d in found]
    if distances != sorted(distances):
        wrong.append(f"distances out of order: {distances}")
    for node_id, reported in found:
        if node_id in vector_of:
            want = distance(metric, QUERY, vector_of[node_id])
            if not math.isclose(reported, want, rel_tol=1e-4, abs_tol=1e-4):
                wrong.append(f"node {node_id} at {reported}, want {want}")
    if exact and len(ids) == len(nearest) and set(ids) <= allowed:
        got = sorted(distance(metric, QUERY, vector_of[i]) for i in ids)
        if not all(
            math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)
            for a, b in zip(got, nearest, strict=True)
        ):
            wrong.append(f"the nearest allowed are at {nearest}, found {got}")
    return wrong


@pytest.mark.parametrize("shape", SHAPES)
def test_a_filtered_search_finds_the_nearest_allowed_nodes(index, shape):
    db, quantization, metric, allowed, vector_of = index
    filters = {"i": -1} if shape == "none" else {shape: True}
    wrong = {}
    for beam, ef in [("the index's beam", None), ("a beam over the index", len(vector_of))]:
        found = db.vector_search("Doc", "emb", QUERY, K, ef=ef, filters=filters)
        exact = ef is not None and quantization in EXACT
        problems = check(found, allowed[shape], vector_of, metric, exact)
        if problems:
            wrong[beam] = problems
    assert not wrong, f"{quantization} {metric} {shape}: {wrong}"
