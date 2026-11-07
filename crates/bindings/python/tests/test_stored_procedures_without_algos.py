"""Stored procedures need GQL only, not the algorithm library: in every build that has GQL,
including the lpg profile without `algos`, CREATE PROCEDURE then CALL runs the body, with its
arguments and over the graph.
"""

import pytest

grafeo = pytest.importorskip("grafeo")

pytestmark = pytest.mark.requires("gql")


def test_a_stored_procedure_runs_its_body():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
    db.execute("CREATE PROCEDURE echo(text STRING) RETURNS (v STRING) AS { RETURN $text AS v }")
    assert [row["v"] for row in db.execute("CALL echo('Vincent') YIELD v RETURN v")] == ["Vincent"]
    db.execute(
        "CREATE PROCEDURE people() RETURNS (name STRING) AS "
        "{ MATCH (p:Person) RETURN p.name AS name }"
    )
    names = sorted(row["name"] for row in db.execute("CALL people() YIELD name RETURN name"))
    assert names == ["Alix", "Gus"]
