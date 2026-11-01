"""Persona: Deriva selects code elements from a code graph.

Every node is `:Graph` plus one kind label (Method, File, Class, Test,
Package, Environment), with an `active` flag and edges between kinds:
CONTAINS, USES, COVERS, IN_ENV and DEPRECATED_BY. Deriva's selectors are
one MATCH with an OR of labels, each with its own EXISTS, some negated,
behind `n.active`. Each selector gives the same nodes as the hand-written
UNION of one query per label, in GQL and Cypher, and as a reference
computed in Python from the edges themselves.
"""

import random

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")

KINDS = {"Method": 120, "File": 60, "Class": 40, "Test": 50, "Package": 10, "Environment": 4}
RETURN = " RETURN n.id AS id"
NOT_DEPRECATED = "NOT EXISTS { MATCH (n)-[:DEPRECATED_BY]->() }"
CONTAINS = "EXISTS { MATCH (n)-[:CONTAINS]->() }"
USES_OR_CONTAINS = "EXISTS { MATCH (n)-[:USES|CONTAINS]->() }"
CONTAINED_OR_COVERED = "EXISTS { MATCH (n)<-[:CONTAINS|COVERS]-() }"
IN_ENV = "EXISTS { MATCH (n)-[:IN_ENV]->(:Environment) }"
THREE = f"{USES_OR_CONTAINS} AND {CONTAINED_OR_COVERED} AND {NOT_DEPRECATED}"


@pytest.fixture(scope="module")
def graph():
    """The database, each node's kind, code and active flag, and the edges."""
    rng = random.Random(6)
    db = grafeo.GrafeoDB()
    nodes = {}
    ids = {}
    for kind, count in KINDS.items():
        rows = [{"id": f"{kind.lower()}-{i}", "active": rng.random() < 0.7} for i in range(count)]
        ids[kind] = db.batch_create_nodes_with_props(["Graph", kind], rows)
        for node_id, row in zip(ids[kind], rows, strict=True):
            nodes[node_id] = (kind, row["id"], row["active"])
    edges = []

    def link(src_kind, dst_kind, edge_type, per_source, probability):
        for src in ids[src_kind]:
            if rng.random() < probability:
                for _ in range(per_source):
                    edges.append((src, rng.choice(ids[dst_kind]), edge_type))

    link("File", "Method", "CONTAINS", 2, 0.8)
    link("Package", "File", "CONTAINS", 4, 1.0)
    link("Method", "Method", "USES", 2, 0.6)
    link("Test", "Method", "COVERS", 2, 0.9)
    link("Class", "Environment", "IN_ENV", 1, 0.5)
    link("Method", "Method", "DEPRECATED_BY", 1, 0.1)
    link("File", "File", "DEPRECATED_BY", 1, 0.1)
    db.batch_create_edges(edges)
    return db, nodes, edges


def ids_of(db, language, query):
    run = db.execute if language == "gql" else db.execute_cypher
    return sorted(row["id"] for row in run(query).to_list())


def outgoing(edges, types):
    return {src for src, _, edge_type in edges if edge_type in types}


def incoming(edges, types):
    return {dst for _, dst, edge_type in edges if edge_type in types}


def negated_and_plain(kind, active, node, edges):
    method = kind == "Method" and node not in outgoing(edges, {"DEPRECATED_BY"})
    file = kind == "File" and node in outgoing(edges, {"CONTAINS"})
    return active and (method or file)


def three_exists(kind, active, node, edges):
    return (
        active
        and kind in ("Method", "File")
        and node in outgoing(edges, {"USES", "CONTAINS"})
        and node in incoming(edges, {"CONTAINS", "COVERS"})
        and node not in outgoing(edges, {"DEPRECATED_BY"})
    )


def cheap_branch(kind, active, node, edges):
    return active and ((kind == "Class" and node in outgoing(edges, {"IN_ENV"})) or kind == "Test")


def covered_methods(kind, active, node, edges):
    return active and kind == "Method" and node in incoming(edges, {"COVERS"})


# (name, selector, the UNION of one query per label, the reference)
SELECTORS = [
    (
        "negated_and_plain_exists_per_label",
        "MATCH (n:Graph) WHERE n.active "
        f"AND ((n:Method AND {NOT_DEPRECATED}) OR (n:File AND {CONTAINS})){RETURN}",
        f"MATCH (n:Method) WHERE n.active AND {NOT_DEPRECATED}{RETURN} "
        f"UNION MATCH (n:File) WHERE n.active AND {CONTAINS}{RETURN}",
        negated_and_plain,
    ),
    (
        "three_exists_over_an_or_of_labels",
        f"MATCH (n:Graph) WHERE (n:Method OR n:File) AND n.active AND {THREE}{RETURN}",
        f"MATCH (n:Method) WHERE n.active AND {THREE}{RETURN} "
        f"UNION MATCH (n:File) WHERE n.active AND {THREE}{RETURN}",
        three_exists,
    ),
    (
        "a_label_with_a_cheap_branch",
        f"MATCH (n:Graph) WHERE n.active AND ((n:Class AND {IN_ENV}) OR n:Test){RETURN}",
        f"MATCH (n:Class) WHERE n.active AND {IN_ENV}{RETURN} "
        f"UNION MATCH (n:Test) WHERE n.active{RETURN}",
        cheap_branch,
    ),
    (
        "multi_label_patterns",
        "MATCH (n:Graph:Method) WHERE n.active "
        f"AND EXISTS {{ MATCH (n)<-[:COVERS]-(:Graph:Test) }}{RETURN}",
        f"MATCH (n:Method) WHERE n.active AND EXISTS {{ MATCH (n)<-[:COVERS]-(t:Test) }}{RETURN}",
        covered_methods,
    ),
]


@pytest.mark.parametrize("language", ["gql", "cypher"])
@pytest.mark.parametrize(
    ("name", "selector", "union", "reference"), SELECTORS, ids=[s[0] for s in SELECTORS]
)
def test_persona_deriva_selector_matches_the_union(
    graph, language, name, selector, union, reference
):
    db, nodes, edges = graph
    expected = sorted(
        code for node, (kind, code, active) in nodes.items() if reference(kind, active, node, edges)
    )
    assert expected, f"{name}: the data selects nothing, the check would be vacuous"
    assert ids_of(db, language, union) == expected, f"{name}: the UNION"
    assert ids_of(db, language, selector) == expected, f"{name}: the selector"


def test_persona_deriva_selector_takes_its_ids_as_a_parameter(graph):
    # Deriva narrows a selector to a batch of ids: the same rows as the
    # selector filtered in Python.
    db, nodes, edges = graph
    _, selector, _, reference = SELECTORS[0]
    wanted = sorted(code for _, code, _ in nodes.values())[::3]
    narrowed = selector.replace(RETURN, f" AND n.id IN $ids{RETURN}")
    rows = sorted(row["id"] for row in db.execute(narrowed, {"ids": wanted}).to_list())
    expected = sorted(
        code
        for node, (kind, code, active) in nodes.items()
        if code in set(wanted) and reference(kind, active, node, edges)
    )
    assert expected
    assert rows == expected
