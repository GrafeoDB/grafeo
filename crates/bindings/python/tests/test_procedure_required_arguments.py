"""A procedure's required argument is required, through CALL and through the bridge (22023).

A CALL that omits a required argument or passes null for it, positionally or in the map form, is
refused with 22023 naming the procedure and the argument; `CALL grafeo.subgraph_isomorphism()`
without its pattern counted the empty pattern, `{count: 1}`, as a success. The bridge function of
the same name refuses a None for it the same way (a value the bridge passes counts as passed), and
no bridge function gives a required argument a default: a default makes the parameter optional, so
the catalogue would say so.
"""

import inspect
import re

import grafeo
import pytest

# The lists below are built at import from the catalogue and the bridge: in a build without
# either they stay empty, so the module is collected and skipped rather than failing.
HAS_ALGORITHMS = hasattr(grafeo.GrafeoDB(), "algorithms")
HAS_GQL = "gql" in grafeo.features()
RUNS = HAS_ALGORITHMS and HAS_GQL

pytestmark = [
    pytest.mark.skipif(not HAS_ALGORITHMS, reason="grafeo built without algos feature"),
    pytest.mark.skipif(not HAS_GQL, reason="grafeo built without gql feature"),
]

INVALID_PARAMETER_VALUE = "22023"

# A value of each type that every domain accepts, as GQL text and as a Python value.
SAMPLES = {
    "Integer": ("1", 1),
    "Float": ("0.5", 0.5),
    "String": ("'Doc'", "Doc"),
    "Boolean": ("true", True),
    "List": ("[1.0, 0.0]", [1.0, 0.0]),
}


def graph():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Doc {name: 'Alix'})-[:E {w: 1}]->(:Doc {name: 'Gus'})")
    node = list(db.execute("MATCH (d:Doc {name: 'Alix'}) RETURN id(d) AS id"))[0]["id"]
    return db, node


def catalogue():
    """`[(procedure, [(parameter, type, required)])]`, each in its order, from
    `grafeo.procedures()`: a required parameter is listed as `name (Type)`, an optional one with
    its default or `optional`."""
    db = grafeo.GrafeoDB()
    found = []
    for row in db.execute(
        "CALL grafeo.procedures() YIELD name, parameters RETURN name, parameters"
    ):
        name, parameters = list(row.values())
        parsed = [
            (parameter, kind, not rest)
            for parameter, kind, rest in re.findall(r"(\w+) \((\w+)((?:, [^)]*)?)\)", parameters)
        ]
        found.append((name, parsed))
    return found


def text(kind, node):
    return str(node) if kind == "NodeId" else SAMPLES[kind][0]


def value(kind, node):
    return node if kind == "NodeId" else SAMPLES[kind][1]


def calls_without(procedure, parameters, missing, node):
    names = [p for p, _, _ in parameters]
    at = names.index(missing)
    last = max(i for i, (_, _, required) in enumerate(parameters) if required)
    before = [text(kind, node) for _, kind, _ in parameters[:at]]
    with_null = [
        "null" if p == missing else text(kind, node) for p, kind, _ in parameters[: last + 1]
    ]
    others = [f"{p}: {text(kind, node)}" for p, kind, required in parameters if required]
    others = [o for o in others if not o.startswith(f"{missing}:")]
    return [
        f"CALL {procedure}({', '.join(before)})",
        f"CALL {procedure}({', '.join(with_null)})",
        f"CALL {procedure}({{{', '.join(others)}}})",
        f"CALL {procedure}({{{', '.join([*others, f'{missing}: null'])}}})",
    ]


REQUIRED = [
    (procedure, parameters, parameter)
    for procedure, parameters in (catalogue() if RUNS else [])
    for parameter, _, required in parameters
    if required
]


def test_the_catalogue_lists_required_parameters():
    found = {f"{procedure}.{parameter}" for procedure, _, parameter in REQUIRED}
    expected = [
        "grafeo.subgraph_isomorphism.pattern_edges",
        "grafeo.bfs.start",
        "grafeo.max_flow.sink",
    ]
    if "vector-index" in grafeo.features():
        expected.append("grafeo.search.vector.label")
    for known in expected:
        assert known in found, (known, sorted(found))


@pytest.mark.parametrize(
    ("procedure", "parameters", "missing"),
    REQUIRED,
    ids=[f"{procedure}.{parameter}" for procedure, _, parameter in REQUIRED],
)
def test_a_call_without_a_required_argument_is_refused(procedure, parameters, missing):
    db, node = graph()
    for query in calls_without(procedure, parameters, missing, node):
        with pytest.raises(grafeo.GrafeoError) as raised:
            db.execute(query)
        assert raised.value.gql_status == INVALID_PARAMETER_VALUE, (query, str(raised.value))
        assert procedure in str(raised.value) and missing in str(raised.value), (
            query,
            str(raised.value),
        )


def bridge_of(procedure):
    """The bridge function of the same name."""
    return procedure.removeprefix("grafeo.")


WITH_BRIDGE = [row for row in REQUIRED if hasattr(grafeo.GrafeoDB().algorithms, bridge_of(row[0]))]


@pytest.mark.parametrize(
    ("procedure", "parameters", "missing"),
    WITH_BRIDGE,
    ids=[f"{procedure}.{parameter}" for procedure, _, parameter in WITH_BRIDGE],
)
def test_the_bridge_refuses_a_none_for_a_required_argument(procedure, parameters, missing):
    db, node = graph()
    function = getattr(db.algorithms, bridge_of(procedure))
    signature = inspect.signature(function)
    assert missing in signature.parameters, (bridge_of(procedure), missing, signature)
    kinds = {p: kind for p, kind, _ in parameters}
    arguments = {
        p: value(kinds[p], node)
        for p, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty and p in kinds
    }
    arguments[missing] = None
    with pytest.raises(grafeo.GrafeoError) as raised:
        function(**arguments)
    assert raised.value.gql_status == INVALID_PARAMETER_VALUE, str(raised.value)
    assert missing in str(raised.value), str(raised.value)


def test_no_bridge_gives_a_required_argument_a_default():
    algorithms = grafeo.GrafeoDB().algorithms
    defaulted = []
    for procedure, _, missing in WITH_BRIDGE:
        signature = inspect.signature(getattr(algorithms, bridge_of(procedure)))
        parameter = signature.parameters.get(missing)
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            defaulted.append((bridge_of(procedure), missing, parameter.default))
    assert not defaulted, defaulted


# ---------------------------------------------------------------------------------------------
# Node ids through the bridge, and null for an optional argument
# ---------------------------------------------------------------------------------------------

LARGEST_ID = 2**63 - 1


def outcome(run):
    """`("ok", result)` or `("error", gql_status)`."""
    try:
        return ("ok", run())
    except grafeo.GrafeoError as error:
        return ("error", error.gql_status)


def test_a_node_id_beyond_the_largest_is_refused():
    """A node id above i64::MAX (u64::MAX is the invalid-id sentinel) names no node: 22023."""
    db, _ = graph()
    for start in (LARGEST_ID + 1, 2**64 - 1):
        with pytest.raises(grafeo.GrafeoError) as raised:
            db.algorithms.bfs(start)
        assert raised.value.gql_status == INVALID_PARAMETER_VALUE, (start, str(raised.value))
        assert "start" in str(raised.value), str(raised.value)


def test_the_largest_node_id_passes_the_check_as_call_does():
    db, _ = graph()
    bridge = outcome(lambda: sorted(db.algorithms.bfs(LARGEST_ID)))
    call = outcome(
        lambda: sorted(row["node_id"] for row in db.execute(f"CALL grafeo.bfs({LARGEST_ID})"))
    )
    assert bridge != ("error", INVALID_PARAMETER_VALUE), "refused by the argument check"
    assert bridge == call


def test_a_none_for_a_required_node_id_says_it_got_null():
    db, _ = graph()
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.algorithms.bfs(None)
    assert raised.value.gql_status == INVALID_PARAMETER_VALUE, str(raised.value)
    assert str(raised.value).endswith(", got null"), str(raised.value)


# One optional parameter of each type (no procedure has an optional list): procedure, bridge,
# the required node-id arguments it needs, the parameter.
OPTIONAL = [
    ("grafeo.pagerank", "pagerank", (), "damping"),
    ("grafeo.pagerank", "pagerank", (), "max_iterations"),
    ("grafeo.degree_centrality", "degree_centrality", (), "normalized"),
    ("grafeo.dijkstra", "dijkstra", ("source",), "weight"),
    ("grafeo.prim", "prim", (), "start"),
]


@pytest.mark.parametrize(
    ("procedure", "bridge", "required", "optional"),
    OPTIONAL,
    ids=[f"{procedure}.{optional}" for procedure, _, _, optional in OPTIONAL],
)
def test_null_for_an_optional_argument_is_its_default(procedure, bridge, required, optional):
    db, node = graph()
    given = dict.fromkeys(required, node)
    entries = [f"{name}: {value}" for name, value in given.items()]

    def rows(query):
        return sorted(repr(sorted(row.items())) for row in db.execute(query))

    omitted = rows(f"CALL {procedure}({{{', '.join(entries)}}})")
    assert rows(f"CALL {procedure}({{{', '.join([*entries, f'{optional}: null'])}}})") == omitted
    first = next(parameters for name, parameters in catalogue() if name == procedure)[0][0]
    if first == optional:
        assert rows(f"CALL {procedure}(null)") == rows(f"CALL {procedure}()"), "positional null"
    function = getattr(db.algorithms, bridge)
    assert repr(function(**given, **{optional: None})) == repr(function(**given))


# ---------------------------------------------------------------------------------------------
# Every optional bridge parameter: None is the procedure's declared default
# ---------------------------------------------------------------------------------------------

# Bridge keywords that are no procedure parameter.
NOT_PARAMETERS = {"projection", "key"}

# The bridge's optional parameters the decision named (user, 2026-11-06), as a guard that the
# class below finds them.
NAMED = {
    ("pagerank", "damping"),
    ("pagerank", "max_iterations"),
    ("pagerank", "tolerance"),
    ("pagerank", "directed"),
    ("degree_centrality", "normalized"),
    ("betweenness_centrality", "normalized"),
    ("closeness_centrality", "wf_improved"),
    ("label_propagation", "max_iterations"),
    ("louvain", "resolution"),
    ("clustering_coefficient", "parallel"),
}


def declared():
    """`{procedure: {parameter: (type, default or None)}}` from `grafeo.procedures()`."""
    db = grafeo.GrafeoDB()
    found = {}
    for row in db.execute(
        "CALL grafeo.procedures() YIELD name, parameters RETURN name, parameters"
    ):
        name, parameters = list(row.values())
        found[name.removeprefix("grafeo.")] = {
            parameter: (kind, default or None)
            for parameter, kind, default in re.findall(
                r"(\w+) \((\w+)(?:, default=([^)]*))?(?:, optional)?\)", parameters
            )
        }
    return found


DECLARED = declared() if RUNS else {}


def optional_bridge_parameters():
    """`(bridge, parameter)` for every optional parameter of a bridge function whose procedure has
    the same name and lists the parameter."""
    algorithms = grafeo.GrafeoDB().algorithms
    procedures = DECLARED
    found = []
    for name in sorted(dir(algorithms)):
        function = getattr(algorithms, name)
        if name.startswith("_") or not callable(function) or name not in procedures:
            continue
        for parameter, spec in inspect.signature(function).parameters.items():
            if (
                parameter not in NOT_PARAMETERS
                and spec.default is not inspect.Parameter.empty
                and parameter in procedures[name]
            ):
                found.append((name, parameter))
    return found


OPTIONAL_BRIDGE = optional_bridge_parameters() if RUNS else []


def test_the_class_finds_every_named_optional_parameter():
    assert NAMED <= set(OPTIONAL_BRIDGE), sorted(NAMED - set(OPTIONAL_BRIDGE))


@pytest.mark.parametrize(
    ("bridge", "parameter"),
    OPTIONAL_BRIDGE,
    ids=[f"{bridge}.{parameter}" for bridge, parameter in OPTIONAL_BRIDGE],
)
def test_an_optional_bridge_parameter_defaults_to_none_and_none_is_the_default(bridge, parameter):
    """The bridge holds no default of its own: its default is None, and None runs as leaving the
    argument out (the procedure's declared default)."""
    db, node = graph()
    function = getattr(db.algorithms, bridge)
    signature = inspect.signature(function)
    assert signature.parameters[parameter].default is None, (bridge, parameter, signature)
    kinds = DECLARED[bridge]
    required = {
        name: value(kinds[name][0], node)
        for name, spec in signature.parameters.items()
        if spec.default is inspect.Parameter.empty and name in kinds
    }
    assert repr(function(**required, **{parameter: None})) == repr(function(**required))


def typed(text):
    """A default as the docs or the catalogue write it, as a value: a bool, a number or a string."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(lowered)
    except ValueError:
        return text.strip()


DOCUMENTED = [
    (bridge, parameter)
    for bridge, parameter in OPTIONAL_BRIDGE
    if DECLARED[bridge][parameter][1] is not None
]


@pytest.mark.parametrize(
    ("bridge", "parameter"),
    DOCUMENTED,
    ids=[f"{bridge}.{parameter}" for bridge, parameter in DOCUMENTED],
)
def test_the_docstring_states_the_declared_default(bridge, parameter):
    """A parameter the catalogue gives a default states that default in its docstring line
    (`(default: X)`), so the docs say what None means."""
    doc = getattr(grafeo.GrafeoDB().algorithms, bridge).__doc__ or ""
    stated = re.search(rf"^\s*{parameter}: .*?\(default: ([^,)]+)", doc, re.MULTILINE)
    assert stated, (bridge, parameter, doc)
    assert typed(stated.group(1)) == typed(DECLARED[bridge][parameter][1]), (
        bridge,
        parameter,
        stated.group(1),
        DECLARED[bridge][parameter][1],
    )
