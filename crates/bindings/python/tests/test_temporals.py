"""Temporal values round-trip without loss: a value comes back as the Python
type that holds it exactly, or as a Grafeo type (`grafeo.Time`, `grafeo.Date`,
`grafeo.Timestamp`, `grafeo.ZonedDateTime`, experimental) where the Python
type cannot, and passed back as a parameter it is the same value.

Before 0.6.4 a time lost its nanoseconds, an aware datetime parameter lost its
offset (it was stored as the UTC instant), a `timedelta` parameter was
refused, and a date outside years 1 to 9999 came back as `None`.
"""

from datetime import date, datetime, time, timedelta, timezone

import grafeo
import pytest
from grafeo import GrafeoDB


@pytest.fixture
def db():
    return GrafeoDB()


def value(db, expression, params=None):
    return db.execute_cypher(f"RETURN {expression} AS v", params or {})[0]["v"]


def text_of(db, v):
    """The engine's text of `v` passed back as a parameter."""
    return value(db, "toString($v)", {"v": v})


@pytest.mark.parametrize(
    ("expression", "text"),
    [
        ("localtime('12:31:14.645876123')", "12:31:14.645876123"),
        ("datetime('1984-10-11T12:31:14.645876+01:00')", "1984-10-11T12:31:14.645876+01:00"),
        ("time('12:31:14.645876123+01:30')", "12:31:14.645876123+01:30"),
    ],
)
def test_a_returned_temporal_passes_back_unchanged(db, expression, text):
    assert text_of(db, value(db, expression)) == text


def test_a_time_with_nanoseconds_is_a_grafeo_time(db):
    v = value(db, "localtime('12:31:14.645876123')")
    assert isinstance(v, grafeo.Time)
    assert (v.hour, v.minute, v.second, v.nanosecond, v.offset_seconds) == (
        12,
        31,
        14,
        645876123,
        None,
    )
    assert str(v) == "12:31:14.645876123"
    assert repr(v) == "grafeo.Time('12:31:14.645876123')"
    assert v.to_native() == time(12, 31, 14, 645876)
    assert v == grafeo.Time("12:31:14.645876123")
    assert hash(v) == hash(grafeo.Time("12:31:14.645876123"))
    assert v != grafeo.Time("12:31:14.645876124")


def test_an_exact_time_stays_a_python_time(db):
    v = value(db, "localtime('12:31:14.645876')")
    assert type(v) is time
    assert v == time(12, 31, 14, 645876)


def test_an_aware_datetime_parameter_keeps_its_offset(db):
    plus_one = timezone(timedelta(hours=1))
    v = datetime(1984, 10, 11, 12, 31, 14, 645876, tzinfo=plus_one)
    assert text_of(db, v) == "1984-10-11T12:31:14.645876+01:00"
    back = value(db, "$v", {"v": v})
    assert back == v
    assert back.utcoffset() == timedelta(hours=1)


def test_a_naive_datetime_parameter_is_utc_as_before(db):
    v = datetime(1984, 10, 11, 12, 31, 14, 645876)
    back = value(db, "$v", {"v": v})
    assert back == v
    assert back.tzinfo is None


def test_a_timedelta_parameter_is_a_duration(db):
    assert text_of(db, timedelta(days=1, hours=2, microseconds=5)) == value(
        db, "toString(duration('P1DT2H0.000005S'))"
    )
    assert value(db, "$v = duration('P1DT2H')", {"v": timedelta(days=1, hours=2)}) is True
    # One sign for both parts: an hour back, not a day back and 23 hours on.
    assert text_of(db, timedelta(hours=-1)) == value(db, "toString(duration('PT-1H'))")


def test_a_date_outside_python_years_is_a_grafeo_date(db):
    v = value(db, "date('9999-12-31') + duration('P1D')")
    assert isinstance(v, grafeo.Date)
    assert (v.year, v.month, v.day) == (10000, 1, 1)
    assert text_of(db, v) == value(db, "toString(date('9999-12-31') + duration('P1D'))")
    with pytest.raises(ValueError):
        v.to_native()
    assert value(db, "date('2024-02-29')") == date(2024, 2, 29)


def test_a_zoned_datetime_outside_python_years_is_a_grafeo_zoned_datetime(db):
    expression = "datetime('9999-12-31T23:00:00+01:00') + duration('PT2H')"
    v = value(db, expression)
    assert isinstance(v, grafeo.ZonedDateTime)
    assert text_of(db, v) == value(db, f"toString({expression})")


def test_a_timestamp_outside_python_years_is_a_grafeo_timestamp(db):
    expression = "localdatetime('9999-12-31T23:00:00') + duration('PT2H')"
    v = value(db, expression)
    assert isinstance(v, grafeo.Timestamp)
    assert text_of(db, v) == value(db, f"toString({expression})")


@pytest.mark.parametrize(
    ("expression", "native"),
    [
        ("duration('P3D')", timedelta(days=3)),
        ("duration('P1DT2H0.000005S')", timedelta(days=1, hours=2, microseconds=5)),
        ("duration('PT-1H')", timedelta(hours=-1)),
        ("duration('P-1DT-2H')", -timedelta(days=1, hours=2)),
    ],
)
def test_a_duration_a_timedelta_holds_is_a_timedelta(db, expression, native):
    v = value(db, expression)
    assert type(v) is timedelta
    assert v == native
    assert text_of(db, v) == value(db, f"toString({expression})")


@pytest.mark.parametrize(
    "text",
    [
        "P1Y2M10D",  # months
        "PT0.000000001S",  # nanoseconds
        "P1DT-1H",  # a time part of the other sign: a timedelta makes it 23 hours
        "PT25H",  # a time part of a day or more: a timedelta makes it P1DT1H
    ],
)
def test_a_duration_a_timedelta_cannot_hold_is_a_grafeo_duration(db, text):
    v = value(db, f"duration('{text}')")
    assert isinstance(v, grafeo.Duration)
    assert text_of(db, v) == value(db, f"toString(duration('{text}'))")
    assert v == grafeo.Duration(text)
    assert (v.months, v.days, v.nanos) == (
        grafeo.Duration(text).months,
        grafeo.Duration(text).days,
        grafeo.Duration(text).nanos,
    )


def test_a_grafeo_duration_with_months_has_no_native_form(db):
    v = value(db, "duration('P1Y2M10D')")
    assert (v.months, v.days, v.nanos) == (14, 10, 0)
    with pytest.raises(ValueError):
        v.to_native()


AMSTERDAM_1936 = timezone(timedelta(minutes=19, seconds=32))  # local mean time before 1937


@pytest.mark.parametrize(
    "v",
    [
        datetime(1936, 5, 1, 12, 0, tzinfo=AMSTERDAM_1936),
        time(12, 0, tzinfo=AMSTERDAM_1936),
        datetime(2024, 1, 15, 12, 0, tzinfo=timezone(timedelta(hours=1, microseconds=1))),
    ],
)
def test_an_offset_that_is_not_whole_minutes_is_refused_with_22009(db, v):
    with pytest.raises(grafeo.GrafeoError) as raised:
        value(db, "$v", {"v": v})
    assert raised.value.gql_status == "22009"


def test_a_whole_minute_offset_is_kept(db):
    v = time(12, 0, tzinfo=timezone(timedelta(hours=5, minutes=45)))
    assert value(db, "$v", {"v": v}) == v
    assert text_of(db, v) == "12:00:00+05:45"


@pytest.mark.parametrize(
    "v",
    [timedelta.max, timedelta.min, timedelta(days=-1, microseconds=1)],
    ids=["max", "min", "a microsecond less than a day back"],
)
def test_a_timedelta_at_the_edges_passes_back_unchanged(db, v):
    back = value(db, "$v", {"v": v})
    assert type(back) is timedelta
    assert back == v


def test_the_largest_timedelta_is_the_engine_duration_of_its_parts(db):
    assert text_of(db, timedelta.max) == value(
        db, "toString(duration('P999999999DT23H59M59.999999S'))"
    )


def test_equal_temporals_hash_alike_whatever_their_offset(db):
    one = grafeo.Time("13:00:00.000000001+00:00")
    other = grafeo.Time("14:00:00.000000001+01:00")
    assert one == other
    assert len({one, other}) == 1


def test_a_datetime_in_a_daylight_saving_gap_keeps_the_instant_python_gives_it(db):
    from zoneinfo import ZoneInfo

    # 02:30 on 2024-03-31 does not exist in Amsterdam: Python reads it at the
    # offset before the change, +01:00.
    v = datetime(2024, 3, 31, 2, 30, tzinfo=ZoneInfo("Europe/Amsterdam"))
    assert v.utcoffset() == timedelta(hours=1)
    back = value(db, "$v", {"v": v})
    # A time in a gap never equals one in another zone (PEP 495), so compare
    # the instants.
    assert back.timestamp() == v.timestamp()
    assert back.utcoffset() == timedelta(hours=1)


@pytest.mark.parametrize(
    "expression",
    [
        "date('0001-01-01') - duration('P1D')",
        "localdatetime('0001-01-01T00:00:00') - duration('PT1H')",
    ],
)
def test_a_value_before_year_1_passes_back_unchanged(db, expression):
    v = value(db, expression)
    assert isinstance(v, (grafeo.Date, grafeo.Timestamp))
    assert text_of(db, v) == value(db, f"toString({expression})")


def test_a_negative_sub_microsecond_duration_is_a_grafeo_duration(db):
    v = value(db, "duration('PT-0.000000001S')")
    assert isinstance(v, grafeo.Duration)
    assert v.nanos == -1
    assert text_of(db, v) == value(db, "toString(duration('PT-0.000000001S'))")


def test_a_property_write_refuses_an_offset_that_is_not_whole_minutes(db):
    node = db.create_node(["Person"], {"name": "Alix"})
    with pytest.raises(grafeo.GrafeoError) as raised:
        db.set_node_property(node.id, "since", time(12, 0, tzinfo=AMSTERDAM_1936))
    assert raised.value.gql_status == "22009"
