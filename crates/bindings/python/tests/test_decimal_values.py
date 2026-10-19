"""Exact decimals cross the Python binding as decimal.Decimal, both ways (#610)."""

from decimal import Decimal

import grafeo
import pytest


def test_a_decimal_parameter_comes_back_as_a_decimal_with_its_scale():
    db = grafeo.GrafeoDB()
    rows = db.execute("RETURN $a + $b AS total", {"a": Decimal("0.10"), "b": Decimal("0.2")})
    (total,) = [row["total"] for row in rows]
    assert isinstance(total, Decimal)
    assert str(total) == "0.30"


def test_a_stored_decimal_property_reads_back_exactly():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Item {name: 'Mia', price: $p})", {"p": Decimal("19.99")})
    rows = db.execute("MATCH (n:Item) RETURN n.price AS price")
    assert [row["price"] for row in rows] == [Decimal("19.99")]


@pytest.mark.parametrize(
    "value, text",
    [
        (Decimal("1E+3"), "1000"),
        (Decimal("-0.001"), "-0.001"),
        (
            Decimal("12345678901234567890.123456789012345678"),
            "12345678901234567890.123456789012345678",
        ),
    ],
)
def test_decimals_in_exponent_form_or_with_many_digits_keep_their_value(value, text):
    db = grafeo.GrafeoDB()
    rows = db.execute("RETURN $x AS v", {"x": value})
    (back,) = [row["v"] for row in rows]
    assert isinstance(back, Decimal)
    assert str(back) == text


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("1" * 39)])
def test_a_decimal_without_an_exact_value_is_refused(value):
    db = grafeo.GrafeoDB()
    with pytest.raises((TypeError, ValueError)):
        db.execute("RETURN $x AS v", {"x": value})


def test_a_float_parameter_stays_a_float():
    db = grafeo.GrafeoDB()
    rows = db.execute("RETURN $x AS v", {"x": 1.5})
    (back,) = [row["v"] for row in rows]
    assert isinstance(back, float)


def _prices():
    db = grafeo.GrafeoDB()
    return db.execute(
        "UNWIND [$a, $b] AS price RETURN price",
        {"a": Decimal("1.5"), "b": Decimal("-2.25")},
    )


def test_to_arrow_gives_a_decimal128_column():
    pa = pytest.importorskip("pyarrow")
    result = _prices()
    if not hasattr(result, "to_arrow"):
        pytest.skip("grafeo built without arrow-export")
    table = result.to_arrow()
    column = table.column("price")
    assert pa.types.is_decimal(column.type)
    assert column.type.scale == 2
    assert column.to_pylist() == [Decimal("1.50"), Decimal("-2.25")]


def test_to_pandas_and_to_polars_keep_decimals():
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    frame = _prices().to_pandas()
    assert list(frame["price"]) == [Decimal("1.5"), Decimal("-2.25")]
    assert isinstance(frame["price"][0], Decimal)
    polars_frame = _prices().to_polars()
    assert isinstance(polars_frame.schema["price"], pl.Decimal)
    assert polars_frame["price"].to_list() == [Decimal("1.50"), Decimal("-2.25")]
    del pd
