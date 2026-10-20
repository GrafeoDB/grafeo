"""`QueryResult.profile`: the operator tree of a `PROFILE` statement (#461).

It holds the tree the text cell of `PROFILE` shows, as dicts: `total_time_ms`
and `root`, each operator with `operator`, `detail`, `rows`, `calls`,
`self_time_ms`, `time_ms` and `children`. Other statements have no profile.
"""

OPERATOR_KEYS = {"operator", "detail", "rows", "calls", "self_time_ms", "time_ms", "children"}


def _people(db):
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 30}), (:Person {name: 'Gus', age: 25}), "
        "(:Person {name: 'Vincent', age: 35})"
    )


def _operators(node, depth=0):
    yield depth, node
    for child in node["children"]:
        yield from _operators(child, depth + 1)


def test_profile_holds_the_tree_of_the_text(db):
    _people(db)
    result = db.execute("PROFILE MATCH (p:Person) WHERE p.age > 28 RETURN p.name")
    profile = result.profile
    assert set(profile) == {"root", "total_time_ms"}
    assert profile["root"]["rows"] == 2
    assert profile["total_time_ms"] <= result.execution_time_ms

    text = result.scalar()
    lines = [line for line in text.splitlines() if line and not line.startswith("Total time")]
    operators = list(_operators(profile["root"]))
    assert len(lines) == len(operators)
    for (depth, operator), line in zip(operators, lines, strict=True):
        assert set(operator) == OPERATOR_KEYS
        indent = "  " * depth
        prefix = f"{indent}{operator['operator']} ({operator['detail']})  rows={operator['rows']}  "
        assert line.startswith(prefix), (line, prefix)
        assert operator["time_ms"] >= operator["self_time_ms"] >= 0


def test_other_statements_have_no_profile(db):
    _people(db)
    assert db.execute("MATCH (p:Person) RETURN p.name").profile is None
    assert db.execute("EXPLAIN MATCH (p:Person) RETURN p.name").profile is None
