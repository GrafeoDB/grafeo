"""A range on a key holding lists of temporal values answers as the filter it stands for.

A comparison of a key with a list parameter (`n.k > $v`) is read as a range on the key. Inside two
lists, a zoned against a local datetime, and a time with an offset against one without, compare by
the statement's rule: not at all in Cypher, and at the session time zone in GQL. The range read
compared them by their instants, so it returned rows the filter rejects. Each form is checked here
against the same comparison written `(...) = true`, which the range read does not take.
"""

from __future__ import annotations

import datetime as dt

import pytest
from grafeo import GrafeoDB

# Stored values written as query literals: Python passes an aware datetime as a local one at UTC.
STORED = {
    "cypher": [
        "datetime('2023-10-05T10:00:00Z')",
        "datetime('2023-10-05T12:00:00+02:00')",
        "localdatetime('2023-10-05T10:00:00')",
        "localdatetime('2023-10-05T11:00:00')",
        "time('12:00:00+01:00')",
        "time('10:00:00Z')",
        "localtime('12:00:00')",
        "localtime('10:00:00')",
    ],
    "gql": [
        "ZONED_DATETIME('2023-10-05T10:00:00Z')",
        "ZONED_DATETIME('2023-10-05T12:00:00+02:00')",
        "LOCAL_DATETIME('2023-10-05T10:00:00')",
        "LOCAL_DATETIME('2023-10-05T11:00:00')",
        "ZONED_TIME('12:00:00+01:00')",
        "ZONED_TIME('10:00:00Z')",
        "LOCAL_TIME('12:00:00')",
        "LOCAL_TIME('10:00:00')",
    ],
}
# The language and its session time zone.
SESSIONS = [("cypher", None), ("gql", "+02:00"), ("gql", "-05:00"), ("gql", None)]
BOUNDS = [
    dt.datetime(2023, 10, 5, 9, 0, 0),
    dt.datetime(2023, 10, 5, 10, 0, 0),
    dt.time(9, 0),
    dt.time(12, 0),
]
OPERATORS = [">", ">=", "<", "<="]


def shapes(value):
    """`value` inside lists: alone, first, last and nested."""
    return [f"[{value}]", f"[{value}, 1]", f"[1, {value}]", f"[[{value}]]"]


def bound_shapes(value):
    return [[value], [value, 1], [1, value], [[value]]]


@pytest.fixture(params=SESSIONS, ids=lambda s: f"{s[0]}-{s[1] or 'no-zone'}")
def database(request):
    language, zone = request.param
    db = GrafeoDB()
    run = db.execute_cypher if language == "cypher" else db.execute
    if zone:
        db.execute(f"SESSION SET TIME ZONE '{zone}'")
    create = "CREATE" if language == "cypher" else "INSERT"
    nodes = []
    for value in STORED[language]:
        for key in shapes(value):
            nodes.append(f"(:L {{row: {len(nodes)}, k: {key}}})")
    nodes.append("(:L {row: -1})")
    run(f"{create} " + ", ".join(nodes))
    return run, language, zone


def rows(run, query, bound):
    return sorted(row["r"] for row in run(query, {"v": bound}))


def test_a_list_range_on_temporals_answers_as_the_filter(database):
    run, language, zone = database
    wrong = []
    ranged = 0
    for value in BOUNDS:
        for bound in bound_shapes(value):
            for op in OPERATORS:
                query = f"MATCH (n:L) WHERE n.k {op} $v RETURN n.row AS r"
                filtered = f"MATCH (n:L) WHERE (n.k {op} $v) = true RETURN n.row AS r"
                plan = str(list(run(f"EXPLAIN {query}", {"v": bound})))
                ranged += "[range: k]" in plan
                found, wanted = rows(run, query, bound), rows(run, filtered, bound)
                if found != wanted:
                    wrong.append(f"{op} {bound!r}: range {found}, filter {wanted}")
    assert ranged, "no form took the range read"
    assert not wrong, f"{language} at {zone}: {len(wrong)} wrong:\n" + "\n".join(wrong)
