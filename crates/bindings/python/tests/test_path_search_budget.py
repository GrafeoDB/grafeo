"""A path search over its memory budget raises GrafeoError instead of aborting.

Reported downstream (Deriva, 2026-10-09): on a code graph of about 1,500
nodes (a CONTAINS tree with IMPORTS and CALLS edges in cycles),
`MATCH (repo)-[*]->(f:File) ... RETURN count(f)` printed "memory allocation
of 47244640256 bytes failed" and ended the Python process. A search that
would hold more paths than its budget now raises an error that names what to
do: an upper bound, DISTINCT, or a shortest path search. A path longer than
the stack is deep no longer overflows it either.

An edge pattern also names one type: `-[:Graph:CONTAINS]->` raises a syntax
error that names :`Graph:CONTAINS` and `:Graph|CONTAINS`, where it matched
the edges of type Graph or CONTAINS.
"""

import random

import grafeo
import pytest

HAS_CYPHER = hasattr(grafeo.GrafeoDB(), "execute_cypher")

LANGUAGES = [
    pytest.param(False, id="gql"),
    pytest.param(
        True,
        id="cypher",
        marks=pytest.mark.skipif(not HAS_CYPHER, reason="grafeo built without cypher feature"),
    ),
]


def run(db, query, cypher):
    return db.execute_cypher(query) if cypher else db.execute(query)


@pytest.fixture(scope="module")
def code_graph():
    """1,485 nodes of the reported shape, picked from a fixed sequence."""
    rng = random.Random(3)
    db = grafeo.GrafeoDB()
    repo = db.create_node(["Repository"], {"repoName": "deriva"}).id
    directories, level = [], [repo]
    for _ in range(3):
        nxt = []
        for parent in level:
            for _ in range(4):
                directory = db.create_node(["Directory"]).id
                db.create_edge(parent, directory, "CONTAINS")
                directories.append(directory)
                nxt.append(directory)
        level = nxt
    files = []
    for i in range(500):
        file = db.create_node(["File"], {"filePath": f"f{i}.py"}).id
        db.create_edge(rng.choice(directories), file, "CONTAINS")
        files.append(file)
    methods = []
    for _ in range(900):
        method = db.create_node(["Method"]).id
        db.create_edge(rng.choice(files), method, "CONTAINS")
        methods.append(method)
    for file in files:
        for target in rng.sample(files, 3):
            db.create_edge(file, target, "IMPORTS")
    for method in methods:
        for target in rng.sample(methods, 2):
            db.create_edge(method, target, "CALLS")
        db.create_edge(method, rng.choice(files), "USES")
    return db


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_an_unbounded_pattern_over_cycles_raises(code_graph, cypher):
    query = "MATCH (repo:Repository)-[*]->(f:File) WHERE repo.repoName = 'deriva' RETURN count(f)"
    with pytest.raises(grafeo.GrafeoError) as raised:
        run(code_graph, query, cypher)
    message = str(raised.value)
    for advice in ("upper bound", "DISTINCT", "shortest"):
        assert advice in message, message
    # The database still answers, and the distinct targets come at once
    distinct = query.replace("count(f)", "count(DISTINCT f)")
    assert run(code_graph, distinct, cypher).scalar() == 500


def test_a_path_longer_than_the_stack_is_deep():
    # 0.5.44 overflowed the stack on this one, which ends the process
    db = grafeo.GrafeoDB()
    length = 100_000
    ids = [db.create_node(["Step"], {"i": i}).id for i in range(length + 1)]
    for a, b in zip(ids, ids[1:], strict=False):
        db.create_edge(a, b, "NEXT")
    query = f"MATCH p = (a:Step {{i: 0}})-[:NEXT*{length}..{length}]->(b) RETURN length(p)"
    assert db.execute(query).scalar() == length


@pytest.mark.parametrize("cypher", LANGUAGES)
def test_a_second_colon_in_an_edge_type_raises(cypher):
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (:Repository)-[:`Graph:CONTAINS`]->(:Directory)-[:`Graph:CONTAINS`]->(:File)"
    )
    with pytest.raises(grafeo.GrafeoError, match="Graph:CONTAINS") as raised:
        run(db, "MATCH (r:Repository)-[:Graph:CONTAINS*]->(f:File) RETURN count(f)", cypher)
    assert ":Graph|CONTAINS" in str(raised.value)
    query = "MATCH (r:Repository)-[:`Graph:CONTAINS`*]->(f:File) RETURN count(f)"
    assert run(db, query, cypher).scalar() == 1
