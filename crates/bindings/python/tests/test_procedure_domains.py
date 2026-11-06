"""The algorithm bridge refuses a value outside its parameter's domain as CALL does (22023).

`db.algorithms.pagerank(damping=2.0)` and the other bridge functions run the same check as
`CALL grafeo.pagerank(2.0)`, so neither way in runs an algorithm on a meaningless value. A bridge
function holds no default of its own: its keyword defaults to None, which runs as the catalogue's
default (`CALL grafeo.procedures()`). 0.6.1 ran them and returned NaN or meaningless scores.
"""

import inspect
import math

import grafeo
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(grafeo.GrafeoDB(), "algorithms"),
    reason="grafeo built without algos feature",
)

INVALID_PARAMETER_VALUE = "22023"

# Bridge function, keyword, kind, domain: the decided domains of the procedures the bridge reaches.
DOMAINS = [
    ("pagerank", "damping", "float", "unit"),
    ("pagerank", "max_iterations", "int", "non_negative"),
    ("pagerank", "tolerance", "float", "non_negative"),
    ("louvain", "resolution", "float", "non_negative"),
    ("label_propagation", "max_iterations", "int", "non_negative"),
    ("kcore", "k", "int", "non_negative"),
]


def probes(kind, domain):
    if kind == "int":
        return [-1], [0, 1]
    outside = [-1e-9, -1.0, math.nan, math.inf, -math.inf]
    inside = [0.0, 1.0]
    if domain == "unit":
        outside += [1.0 + 1e-9, 2.0]
        inside.append(0.5)
    return outside, inside


def graph():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'})"
        "-[:KNOWS]->(c:Person {name: 'Vincent'})-[:KNOWS]->(a), "
        "(:Person {name: 'Jules'})-[:KNOWS]->(:Person {name: 'Mia'})"
    )
    return db


def catalogue():
    """`{procedure: {parameter: default}}` from `CALL grafeo.procedures()`."""
    db = grafeo.GrafeoDB()
    found = {}
    for row in db.execute(
        "CALL grafeo.procedures() YIELD name, parameters RETURN name, parameters"
    ):
        name, parameters = list(row.values())
        defaults = {}
        for part in parameters.split("), "):
            if "default=" in part:
                key = part.split(" (")[0]
                defaults[key] = part.split("default=")[1].rstrip(")")
        found[name.removeprefix("grafeo.")] = defaults
    return found


@pytest.mark.parametrize(("function", "keyword", "kind", "domain"), DOMAINS)
def test_a_value_outside_the_domain_is_refused(function, keyword, kind, domain):
    db = graph()
    run = getattr(db.algorithms, function)
    outside, inside = probes(kind, domain)
    for value in outside:
        with pytest.raises(grafeo.GrafeoError) as raised:
            run(**{keyword: value})
        assert raised.value.gql_status == INVALID_PARAMETER_VALUE, (function, keyword, value)
        assert keyword in str(raised.value), str(raised.value)
    for value in inside:
        run(**{keyword: value})


@pytest.mark.parametrize(("function", "keyword", "kind", "domain"), DOMAINS)
def test_the_bridge_defaults_are_the_catalogues(function, keyword, kind, domain):
    """The keyword defaults to None, and None runs as the catalogue's default given explicitly."""
    signature = inspect.signature(getattr(grafeo.GrafeoDB().algorithms, function))
    assert signature.parameters[keyword].default is None, (function, keyword, signature)
    listed = catalogue()[function].get(keyword)
    if listed is None:
        return
    db = graph()
    run = getattr(db.algorithms, function)
    value = int(listed) if kind == "int" else float(listed)
    assert repr(run(**{keyword: None})) == repr(run(**{keyword: value})), (function, keyword)


def test_the_bridge_and_call_refuse_the_same_damping():
    db = graph()
    for damping in (2.0, -1.0, math.nan):
        with pytest.raises(grafeo.GrafeoError) as bridge:
            db.algorithms.pagerank(damping=damping)
        with pytest.raises(grafeo.GrafeoError) as call:
            db.execute("CALL grafeo.pagerank($d) YIELD score", {"d": damping})
        assert bridge.value.gql_status == call.value.gql_status == INVALID_PARAMETER_VALUE


def test_louvain_at_resolution_zero_keeps_each_component_whole():
    db = grafeo.GrafeoDB()
    db.execute(
        "INSERT (a:P {name: 'Alix'})-[:K]->(b:P {name: 'Gus'})"
        "-[:K]->(c:P {name: 'Vincent'})-[:K]->(a), "
        "(:P {name: 'Jules'})-[:K]->(:P {name: 'Mia'})-[:K]->(:P {name: 'Butch'})"
    )
    result = db.algorithms.louvain(resolution=0.0)
    communities = result["communities"]
    assert len(set(communities.values())) == 2, communities
