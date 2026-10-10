"""Parameters in CALL arguments run exactly like the same literals.

A parameter that a procedure argument reads used to be ignored: optional
arguments ran with their defaults (a wrong answer without an error), required
ones failed as "required".
"""

import grafeo
import pytest

pytestmark = [
    pytest.mark.skipif(
        not hasattr(grafeo.GrafeoDB(), "algorithms"),
        reason="grafeo built without algos feature",
    ),
    pytest.mark.skipif("gql" not in grafeo.features(), reason="grafeo built without gql feature"),
]

CHAIN = (
    "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})"
    "-[:KNOWS]->(:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Mia'}), "
    "(a)-[:KNOWS]->(:Person {name: 'Jules'})"
)
PAGERANK = "CALL grafeo.pagerank({}) YIELD node_id, score RETURN node_id, score ORDER BY node_id"


def chain():
    db = grafeo.GrafeoDB()
    db.execute(CHAIN)
    return db


def scores(result):
    return [(row["node_id"], row["score"]) for row in result]


def test_positional_parameters_run_like_the_same_literals():
    db = chain()
    literal = scores(db.execute(PAGERANK.format("0.5, 1, 0.0001")))
    parameters = scores(db.execute(PAGERANK.format("$d, $m, $t"), {"d": 0.5, "m": 1, "t": 0.0001}))
    assert parameters == literal
    assert literal != scores(db.execute(PAGERANK.format(""))), "the arguments change the scores"


def test_a_map_parameter_names_the_arguments():
    db = chain()
    literal = scores(db.execute(PAGERANK.format("{damping: 0.5, max_iterations: 1}")))
    config = {"damping": 0.5, "max_iterations": 1}
    assert scores(db.execute(PAGERANK.format("$config"), {"config": config})) == literal


def test_a_required_argument_from_a_parameter_runs():
    db = chain()
    alix = db.execute("MATCH (p:Person {name: 'Alix'}) RETURN id(p) AS id").scalar()
    rows = db.execute(
        "CALL grafeo.bfs($s) YIELD node_id, depth RETURN node_id, depth ORDER BY node_id",
        {"s": alix},
    )
    assert [row["depth"] for row in rows] == [0, 1, 2, 3, 1]


def test_a_missing_parameter_is_an_error():
    db = chain()
    with pytest.raises(grafeo.GrafeoError, match=r"Missing parameter: \$d"):
        db.execute("CALL grafeo.pagerank($d) YIELD score RETURN score")


def test_an_argument_of_the_wrong_type_is_an_error():
    db = chain()
    with pytest.raises(grafeo.GrafeoError, match="Argument 'damping' of grafeo.pagerank"):
        db.execute("CALL grafeo.pagerank($d) YIELD score RETURN score", {"d": "high"})


@pytest.mark.skipif("cypher" not in grafeo.features(), reason="grafeo built without cypher")
def test_cypher_parameters_run_like_the_same_literals():
    db = chain()
    literal = scores(db.execute_cypher(PAGERANK.format("0.5, 1, 0.0001")))
    parameters = scores(
        db.execute_cypher(PAGERANK.format("$d, $m, $t"), {"d": 0.5, "m": 1, "t": 0.0001})
    )
    assert parameters == literal


@pytest.mark.skipif("cypher" not in grafeo.features(), reason="grafeo built without cypher")
def test_cypher_argument_read_from_a_row_is_an_error():
    db = chain()
    with pytest.raises(grafeo.GrafeoError, match="must be a constant"):
        db.execute_cypher(
            "MATCH (p:Person {name: 'Alix'}) WITH id(p) AS s "
            "CALL grafeo.bfs(s) YIELD node_id RETURN node_id"
        )
