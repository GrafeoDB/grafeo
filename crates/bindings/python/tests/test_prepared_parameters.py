"""A statement called again with other values answers for those values.

The engine optimizes a statement with values once, with its parameters in,
and fills a copy of that plan with each call's values: every call through
the binding's parameter path must answer as the statement with its values
written as literals answers.
"""

import grafeo
import pytest

PEOPLE = [("Alix", 30), ("Gus", 25), ("Vincent", 41), ("Mia", 25)]


def people():
    db = grafeo.GrafeoDB()
    for name, age in PEOPLE:
        db.execute("INSERT (:Person {name: $name, age: $age})", {"name": name, "age": age})
    return db


def test_gql_lookups_take_each_calls_value():
    db = people()
    query = "MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age"
    for name, age in [("Gus", 25), ("Alix", 30), ("Nobody", None), ("Gus", 25)]:
        rows = [row["age"] for row in db.execute(query, {"name": name})]
        assert rows == ([] if age is None else [age]), name


def test_gql_skip_and_limit_take_each_calls_values():
    db = people()
    query = "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP $skip LIMIT $limit"
    names = sorted(name for name, _ in PEOPLE)
    for skip, limit in [(0, 1), (1, 2), (3, 5), (0, 4)]:
        rows = [row["name"] for row in db.execute(query, {"skip": skip, "limit": limit})]
        assert rows == names[skip : skip + limit], (skip, limit)


def test_gql_a_value_of_another_type_answers_as_its_literal():
    db = people()
    query = "MATCH (p:Person) WHERE p.age = $age RETURN p.name AS name"
    for value, literal in [(25, "25"), ("25", "'25'"), (25, "25"), (True, "true")]:
        got = sorted(row["name"] for row in db.execute(query, {"age": value}))
        expected = sorted(row["name"] for row in db.execute(query.replace("$age", literal)))
        assert got == expected, value


def test_cypher_merge_takes_each_calls_values():
    db = people()
    if not hasattr(db, "execute_cypher"):
        pytest.skip("grafeo built without cypher")
    merge = (
        "MATCH (a:Person {name: $a}), (b:Person {name: $b}) "
        "MERGE (a)-[:LIKES]->(b) RETURN a.name AS a, b.name AS b"
    )
    count = "MATCH (:Person)-[r:LIKES]->(:Person) RETURN count(r) AS n"
    for a, b, total in [("Alix", "Gus", 1), ("Gus", "Mia", 2), ("Alix", "Gus", 2)]:
        rows = [(row["a"], row["b"]) for row in db.execute_cypher(merge, {"a": a, "b": b})]
        assert rows == [(a, b)]
        assert [row["n"] for row in db.execute(count)] == [total]
