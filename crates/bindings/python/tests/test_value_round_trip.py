"""Bytes, times and dates make the round trip from Python into Grafeo and back.

`bytes` and `bytearray` were read as lists of ints on the way in (they are
sequences, and the list case came first), so a query returned `[71, 0, 255]`.
A time with a UTC offset came back as a naive `datetime.time`, and a
`datetime.time` or `datetime.date` could not be passed in at all.
"""

import datetime

import grafeo
import pytest

DATA = b"G\x00\xff"
PLUS_ONE = datetime.timezone(datetime.timedelta(hours=1))
MINUS_HALF = datetime.timezone(-datetime.timedelta(hours=5, minutes=30))


def values(db, query, params=None):
    return [row["v"] for row in db.execute(query, params or {})]


@pytest.mark.parametrize("value", [DATA, bytearray(DATA)], ids=["bytes", "bytearray"])
def test_bytes_come_back_as_bytes(value):
    db = grafeo.GrafeoDB()
    db.create_node(["Blob"], {"b": value})
    assert values(db, "MATCH (n:Blob) RETURN n.b AS v") == [DATA]
    assert values(db, "RETURN $x AS v", {"x": value}) == [DATA]


def test_bytes_inside_lists_and_maps():
    db = grafeo.GrafeoDB()
    assert values(db, "RETURN $x AS v", {"x": [DATA, b""]}) == [[DATA, b""]]
    assert values(db, "RETURN $x AS v", {"x": {"k": DATA}}) == [{"k": DATA}]
    assert values(db, "UNWIND $x AS v RETURN v", {"x": [DATA, b"\x01"]}) == [DATA, b"\x01"]


def test_a_list_of_ints_stays_a_list():
    db = grafeo.GrafeoDB()
    assert values(db, "RETURN $x AS v", {"x": [71, 0, 255]}) == [[71, 0, 255]]


def test_a_time_with_an_offset_keeps_it():
    db = grafeo.GrafeoDB()
    (t,) = values(db, "RETURN time('10:00:00+01:00') AS v")
    assert t == datetime.time(10, 0, tzinfo=PLUS_ONE)
    assert t.utcoffset() == datetime.timedelta(hours=1)
    (naive,) = values(db, "RETURN time('10:00:00') AS v")
    assert naive == datetime.time(10, 0)
    assert naive.tzinfo is None


@pytest.mark.parametrize(
    "value, text",
    [
        (datetime.time(10, 0, tzinfo=PLUS_ONE), "10:00:00+01:00"),
        (datetime.time(23, 59, 58, 250_000, tzinfo=MINUS_HALF), "23:59:58.25-05:30"),
        (datetime.time(7, 5), "07:05:00"),
    ],
    ids=["plus-one", "minus-five-thirty-with-microseconds", "naive"],
)
def test_a_time_parameter_makes_the_round_trip(value, text):
    db = grafeo.GrafeoDB()
    db.create_node(["T"], {"t": value})
    assert values(db, "MATCH (n:T) RETURN n.t AS v") == [value]
    assert values(db, "MATCH (n:T) RETURN toString(n.t) AS v") == [text]
    assert values(db, "RETURN $x AS v", {"x": value}) == [value]


def test_a_date_parameter_makes_the_round_trip():
    db = grafeo.GrafeoDB()
    day = datetime.date(2024, 2, 29)
    db.create_node(["D"], {"d": day})
    assert values(db, "MATCH (n:D) RETURN n.d AS v") == [day]
    assert values(db, "MATCH (n:D) WHERE n.d = date('2024-02-29') RETURN n.d AS v") == [day]
    assert values(db, "RETURN $x AS v", {"x": day}) == [day]


def test_a_datetime_is_still_a_timestamp_not_a_date():
    db = grafeo.GrafeoDB()
    moment = datetime.datetime(2024, 3, 1, 9, 30, tzinfo=datetime.UTC)
    (back,) = values(db, "RETURN $x AS v", {"x": moment})
    assert isinstance(back, datetime.datetime)
    assert back.replace(tzinfo=datetime.UTC) == moment
