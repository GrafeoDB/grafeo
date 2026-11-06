"""The algorithm bridge refuses an edge value it cannot use, as CALL does.

A string or boolean weight is 22G12; NaN, an infinity and a negative weight where the algorithm
needs a non-negative one are 22023; the message names the edge and the property. A missing weight
is the algorithm's default. 0.6.1 read `'5'`, `true` and `[5]` as 1.0 and ran on NaN.
"""

import math

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

WRONG_KIND = "22G12"
INVALID_PARAMETER_VALUE = "22023"


def graph():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (a:P {name: 'Alix'}), (b:P {name: 'Gus'}), (c:P {name: 'Vincent'}), "
        "(d:P {name: 'Mia'}), (a)-[:E {first: true, w: 3, c: 5, k: 1}]->(b), "
        "(b)-[:E {w: 4, c: 5, k: 1}]->(c), (a)-[:E {w: 10, c: 2, k: 4}]->(c), "
        "(c)-[:E {w: 2, c: 6, k: 1}]->(d)"
    )
    ids = {
        row["name"]: row["id"]
        for row in db.execute("MATCH (n:P) RETURN n.name AS name, id(n) AS id")
    }
    return db, ids["Alix"], ids["Mia"]


# Bridge function, how it runs with the property `w` probed, and whether a negative is allowed.
CALLS = [
    ("dijkstra", lambda al, a, d: al.dijkstra(a, weight="w"), False),
    ("sssp", lambda al, a, d: al.sssp("Alix", weight_attr="w"), False),
    ("bellman_ford", lambda al, a, d: al.bellman_ford(a, weight="w"), True),
    ("floyd_warshall", lambda al, a, d: al.floyd_warshall(weight="w"), True),
    ("kruskal", lambda al, a, d: al.kruskal(weight="w"), True),
    ("prim", lambda al, a, d: al.prim(weight="w"), True),
    ("max_flow", lambda al, a, d: al.max_flow(a, d, capacity="w"), False),
    (
        "min_cost_capacity",
        lambda al, a, d: al.min_cost_max_flow(a, d, capacity="w", cost="k"),
        False,
    ),
    ("min_cost_cost", lambda al, a, d: al.min_cost_max_flow(a, d, capacity="c", cost="w"), True),
]


def probe(value):
    db, a, d = graph()
    db.execute("MATCH ()-[e:E {first: true}]->() SET e.w = $x", {"x": value})
    return db, a, d


@pytest.mark.parametrize(("name", "call", "negative"), CALLS)
def test_a_weight_of_another_kind_is_refused(name, call, negative):
    for value in ["5", "x", True, [5]]:
        db, a, d = probe(value)
        with pytest.raises(grafeo.GrafeoError) as raised:
            call(db.algorithms, a, d)
        assert raised.value.gql_status == WRONG_KIND, (name, value, str(raised.value))
        assert "edge" in str(raised.value), str(raised.value)


@pytest.mark.parametrize(("name", "call", "negative"), CALLS)
def test_a_weight_outside_the_domain_is_refused(name, call, negative):
    for value in [math.nan, math.inf, -math.inf]:
        db, a, d = probe(value)
        with pytest.raises(grafeo.GrafeoError) as raised:
            call(db.algorithms, a, d)
        assert raised.value.gql_status == INVALID_PARAMETER_VALUE, (name, value)
    db, a, d = probe(-2)
    if negative:
        call(db.algorithms, a, d)
    else:
        with pytest.raises(grafeo.GrafeoError) as raised:
            call(db.algorithms, a, d)
        assert raised.value.gql_status == INVALID_PARAMETER_VALUE, name


@pytest.mark.parametrize(("name", "call", "negative"), CALLS)
def test_signed_zero_and_whole_numbers_run(name, call, negative):
    db, a, d = probe(-0.0)
    zero_db, za, zd = probe(0.0)
    assert call(db.algorithms, a, d) == call(zero_db.algorithms, za, zd), name
    db, a, d = probe(3)
    float_db, fa, fd = probe(3.0)
    assert call(db.algorithms, a, d) == call(float_db.algorithms, fa, fd), name


def test_min_cost_max_flow_refuses_a_negative_cycle():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (s:N {i: 0}), (a:N {i: 1}), (b:N {i: 2}), (t:N {i: 3}), "
        "(s)-[:E {c: 5, k: 1}]->(a), (a)-[:E {c: 5, k: -3}]->(b), "
        "(b)-[:E {c: 5, k: -3}]->(a), (b)-[:E {c: 5, k: 1}]->(t)"
    )
    ids = {row["i"]: row["id"] for row in db.execute("MATCH (n:N) RETURN n.i AS i, id(n) AS id")}
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.algorithms.min_cost_max_flow(ids[0], ids[3], "c", "k")
    assert raised.value.gql_status == INVALID_PARAMETER_VALUE
