"""Signed integers and negative literals cross the Python binding as int, float
and decimal.Decimal, and keep their type through a compact and a reopen (#160,
#301): the sum of signed integers is an int, never a float."""

from decimal import Decimal

from grafeo import GrafeoDB


def write(db):
    db.execute("FOR i IN range(-1000, 999) INSERT (:N {v: i})")
    db.execute("INSERT (:Location {a: (-6.248), b: 0 - 6.248, c: -6, d: -33.8688e0})")


def check(db):
    (row,) = list(db.execute("MATCH (n:N) RETURN sum(n.v) AS s, min(n.v) AS lo, count(n) AS c"))
    assert row == {"s": -1000, "lo": -1000, "c": 2000}
    assert type(row["s"]) is int
    (exact,) = list(db.execute("MATCH (n:N) WHERE n.v = -7 RETURN n.v AS v"))
    assert exact == {"v": -7}
    (location,) = list(
        db.execute("MATCH (l:Location) RETURN l.a AS a, l.b AS b, l.c AS c, l.d AS d")
    )
    assert location == {"a": Decimal("-6.248"), "b": Decimal("-6.248"), "c": -6, "d": -33.8688}
    assert type(location["a"]) is Decimal
    assert type(location["c"]) is int
    assert type(location["d"]) is float


def test_signed_values_in_memory_and_after_compact():
    db = GrafeoDB()
    write(db)
    check(db)
    db.compact()
    check(db)


def test_signed_values_after_compact_and_reopen(tmp_path):
    path = str(tmp_path / "signed.grafeo")
    db = GrafeoDB(path)
    write(db)
    db.compact()
    db.close()
    db = GrafeoDB(path)
    check(db)
    db.execute("CREATE INDEX n_v FOR (n:N) ON (n.v)")
    check(db)
    db.close()


def test_a_negative_parameter_keeps_its_type():
    db = GrafeoDB()
    db.execute(
        "INSERT (:P {i: $i, f: $f, d: $d})",
        {"i": -9223372036854775808, "f": -0.5, "d": Decimal("-250.75")},
    )
    (row,) = list(db.execute("MATCH (p:P) RETURN p.i AS i, p.f AS f, p.d AS d"))
    assert row == {"i": -9223372036854775808, "f": -0.5, "d": Decimal("-250.75")}
    assert [type(row[key]) for key in ("i", "f", "d")] == [int, float, Decimal]
