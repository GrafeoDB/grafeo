"""Every Grafeo error carries its GQLSTATUS (#639).

`GrafeoError.gql_status` is the five-character ISO GQL status of the condition,
next to the Grafeo `error_code`: `22003` for a numeric value out of range,
`22012` for a division by zero, `42001` for a syntax error, and so on.
"""

import grafeo
import pytest


@pytest.mark.parametrize(
    ("query", "status", "message"),
    [
        ("RETURN 1 / 0 AS x", "22012", "division by zero: 1 / 0"),
        ("RETURN 1 % 0 AS x", "22012", "division by zero: 1 % 0"),
        (
            "RETURN 9223372036854775807 + 1 AS x",
            "22003",
            "numeric value out of range: 9223372036854775807 + 1",
        ),
        (
            "UNWIND [9223372036854775807, 1] AS x RETURN sum(x) AS s",
            "22003",
            "numeric value out of range: sum 9223372036854775808",
        ),
    ],
)
def test_numeric_errors_carry_their_status(db, query, status, message):
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute(query)
    error = raised.value
    assert error.gql_status == status
    assert error.error_code == "GRAFEO-Q006"
    assert message in str(error)


def test_the_status_reaches_cypher_too(db):
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute_cypher("RETURN 10 / 0 AS x")
    assert raised.value.gql_status == "22012"


def test_a_syntax_error_carries_its_status(db):
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.execute("MATCH (n RETURN n")
    assert raised.value.gql_status == "42001"


def test_a_missing_node_carries_no_data(db):
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.set_node_property(999999, "key", "value")
    assert raised.value.error_code == "GRAFEO-V002"
    assert raised.value.gql_status == "02000"


def test_a_write_after_close_carries_an_invalid_transaction_state(tmp_path):
    db = grafeo.GrafeoDB(str(tmp_path / "prague.grafeo"))
    db.close()
    with pytest.raises(grafeo.DatabaseClosedError) as raised:
        db.execute("INSERT (:Person {name: 'Alix'})")
    assert raised.value.gql_status == "25000"
