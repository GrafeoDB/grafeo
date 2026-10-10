"""A statement that ends with a write and no RETURN has no result (#580).

A GQL statement of one INSERT returned the last node it created, in a column
such as `_anon_0`. It now has no columns and no rows, like a Cypher CREATE,
through every way Python runs a statement, and its counters still say what
it wrote. After such a statement, the one after NEXT reads one empty row. A
DELETE of a variable nothing binds fails; it deleted every node.
"""

import pytest
from grafeo import GrafeoDB


def in_a_transaction(db):
    with db.begin_transaction() as tx:
        result = tx.execute("INSERT (:City {name: 'Amsterdam'})")
        tx.commit()
    return result


@pytest.mark.parametrize(
    "run",
    [
        lambda db: db.execute("INSERT (:City {name: 'Amsterdam'})"),
        lambda db: db.execute("CREATE (:City {name: 'Amsterdam'})"),
        lambda db: db.execute("INSERT (:City {name: $name})", {"name": "Amsterdam"}),
        lambda db: db.execute_language("gql", "INSERT (:City {name: 'Amsterdam'})"),
        lambda db: db.execute_cypher("CREATE (:City {name: 'Amsterdam'})"),
        in_a_transaction,
    ],
    ids=["execute", "create", "parameters", "execute_language", "cypher", "transaction"],
)
def test_a_lone_insert_has_no_result(run):
    db = GrafeoDB()
    result = run(db)
    assert result.columns == []
    assert len(result) == 0
    assert list(result) == []
    assert result.counters["nodes_created"] == 1
    assert result.counters["properties_set"] == 1
    names = db.execute("MATCH (c:City) RETURN c.name AS name")
    assert [row["name"] for row in names] == ["Amsterdam"]


def test_a_lone_insert_after_next_reads_the_rows_before_it():
    db = GrafeoDB()
    db.execute("INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'})")
    result = db.execute("MATCH (c:City) RETURN c.name AS name NEXT INSERT (:Stop {city: name})")
    assert result.columns == []
    assert len(result) == 0
    stops = db.execute("MATCH (s:Stop) RETURN s.city AS city ORDER BY city")
    assert [row["city"] for row in stops] == ["Amsterdam", "Berlin"]


def test_next_after_a_write_without_a_result_reads_one_empty_row():
    db = GrafeoDB()
    db.execute("INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'})")
    db.execute("MATCH (c:City) SET c.seen = true NEXT INSERT (:Stop {city: 'Paris'})")
    stops = db.execute("MATCH (s:Stop) RETURN count(s) AS n")
    assert [row["n"] for row in stops] == [1]


@pytest.mark.parametrize("query", ["DETACH DELETE c", "DELETE c"])
def test_a_delete_of_an_unbound_variable_deletes_nothing(query):
    db = GrafeoDB()
    db.execute("INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'})")
    with pytest.raises(Exception, match="Undefined variable 'c'"):
        db.execute(query)
    cities = db.execute("MATCH (c:City) RETURN count(c) AS n")
    assert [row["n"] for row in cities] == [2]
