"""A Python int past 64 bits is an exact DECIMAL, up to 38 digits.

An int that fits 64 bits stays INT64. One past it, but within 38 digits,
reads as an exact DECIMAL with scale 0 (it used to become a float and lose
its last digits). Past 38 digits it is an error naming the parameter or
property and the value. The rule is the same in every binding and in LOAD
DATA.
"""

from decimal import Decimal

import pytest
from grafeo import GrafeoDB

WIDE = [2**63, 2**64, -(2**63) - 1, 10**38 - 1, -(10**38) + 1]


@pytest.fixture
def db():
    return GrafeoDB()


def test_an_int_that_fits_64_bits_stays_int64(db):
    for value in [2**63 - 1, -(2**63)]:
        row = next(iter(db.execute("RETURN $x AS x, $x IS TYPED INT64 AS int", {"x": value})))
        assert (row["x"], row["int"]) == (value, True)
        assert type(row["x"]) is int


@pytest.mark.parametrize("value", WIDE)
def test_a_parameter_past_64_bits_is_an_exact_decimal(db, value):
    row = next(iter(db.execute("RETURN $x AS x, $x IS TYPED DECIMAL AS decimal", {"x": value})))
    assert row["decimal"] is True
    assert row["x"] == Decimal(value)
    assert type(row["x"]) is Decimal


def test_a_wide_int_inside_a_list_and_a_map_is_exact(db):
    rows = list(
        db.execute(
            "UNWIND $xs AS v RETURN v IS TYPED DECIMAL AS decimal, v AS v",
            {"xs": [3, 2**64]},
        )
    )
    assert [(row["decimal"], row["v"]) for row in rows] == [(False, 3), (True, Decimal(2**64))]
    row = next(iter(db.execute("RETURN $m.big AS big", {"m": {"big": 10**38 - 1}})))
    assert row["big"] == Decimal(10**38 - 1)


@pytest.mark.parametrize("value", [10**38, -(10**38), 2**200])
def test_an_int_past_38_digits_is_an_error_naming_the_parameter(db, value):
    with pytest.raises(ValueError, match=rf"parameter 'x'.*{value}.*38 digits"):
        db.execute("RETURN $x AS x", {"x": value})


def test_an_int_past_38_digits_is_an_error_naming_the_property(db):
    with pytest.raises(ValueError, match=rf"property 'v'.*{10**38}.*38 digits"):
        db.create_node(["N"], {"v": 10**38})
    assert db.node_count == 0


def test_a_property_past_64_bits_is_stored_exactly(db, tmp_path):
    path = str(tmp_path / "wide.grafeo")
    db = GrafeoDB(path)
    db.create_node(["N"], {"v": 2**64})
    db.close()
    db = GrafeoDB(path)
    row = next(iter(db.execute("MATCH (n:N) RETURN n.v AS v, n.v IS TYPED DECIMAL AS decimal")))
    assert (row["v"], row["decimal"]) == (Decimal(2**64), True)
    db.close()


def test_arithmetic_over_int64_and_decimal_rows_stays_exact(db):
    for value in [5, 2**64, -(2**63)]:
        db.execute("INSERT (:N {v: $v})", {"v": value})
    rows = list(db.execute("MATCH (n:N) RETURN n.v + 1 AS next ORDER BY next"))
    assert [row["next"] for row in rows] == [-(2**63) + 1, 6, Decimal(2**64 + 1)]
    total = next(iter(db.execute("MATCH (n:N) RETURN sum(n.v) AS total")))["total"]
    assert total == Decimal(5 + 2**64 - 2**63)
    assert type(total) is Decimal
