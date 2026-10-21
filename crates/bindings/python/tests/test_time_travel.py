"""Time travel: reads of the database as it was at an earlier epoch or time.

`history_retention` keeps the history; `as_of` gives a read-only view at a
point, `execute_as_of` runs one query there and `epoch_as_of` names the epoch.
A point is an int (an epoch), an aware datetime or a timedelta (that long
ago); a bare number is never a time, and a naive datetime is refused.
"""

import datetime
import time

import grafeo
import pytest


def people(view):
    return sorted(row["name"] for row in view.execute("MATCH (p:Person) RETURN p.name AS name"))


@pytest.fixture
def db():
    db = grafeo.GrafeoDB(history_retention="forever")
    db.execute("INSERT (:Person {name: 'Alix'})")
    return db


def test_as_of_an_epoch_reads_the_state_then(db):
    then = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Amsterdam'")
    with db.as_of(then) as view:
        assert view.epoch == then
        assert people(view) == ["Alix"]
        alix = view.execute("MATCH (p:Person {name: 'Alix'}) RETURN p.city AS city")
        assert [row["city"] for row in alix] == [None]
    assert people(db) == ["Alix", "Gus"]


def test_the_view_refuses_a_write(db):
    with db.as_of(db.current_epoch()) as view:
        with pytest.raises(grafeo.GrafeoError):
            view.execute("INSERT (:Person {name: 'Mia'})")
    assert people(db) == ["Alix"]


def test_direct_reads_through_the_view(db):
    alix = db.execute("MATCH (p:Person) RETURN id(p) AS id").scalar()
    then = db.current_epoch()
    db.execute("MATCH (p:Person) SET p.name = 'Alix B.'")
    with db.as_of(then) as view:
        assert view.get_node(alix).properties()["name"] == "Alix"
        assert view.get_node_property(alix, "name") == "Alix"
        assert view.get_node(alix + 1000) is None
        assert view.get_edge(0) is None


def test_execute_as_of_with_params(db):
    then = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    result = db.execute_as_of(
        "MATCH (p:Person) WHERE p.name <> $not RETURN count(p) AS n", then, {"not": "Vincent"}
    )
    assert result.scalar() == 1


def test_a_datetime_point_reads_the_last_commit_at_or_before_it(db):
    # Python's clock can lag the engine's by a millisecond on Windows: a time
    # after the history began.
    time.sleep(0.01)
    before_gus = datetime.datetime.now(datetime.UTC)
    # Commit times have the resolution of a millisecond: Gus commits after it.
    time.sleep(0.01)
    db.execute("INSERT (:Person {name: 'Gus'})")
    assert db.epoch_as_of(before_gus) <= db.current_epoch()
    with db.as_of(before_gus) as view:
        assert people(view) == ["Alix"]
    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=1)
    assert people(db.as_of(later)) == ["Alix", "Gus"]


def test_a_datetime_in_another_zone_names_the_same_instant(db):
    time.sleep(0.01)  # a time after the history began
    now = datetime.datetime.now(datetime.UTC)
    amsterdam = now.astimezone(datetime.timezone(datetime.timedelta(hours=2)))
    assert db.epoch_as_of(now) == db.epoch_as_of(amsterdam)


def test_a_timedelta_point_is_that_long_ago(db):
    assert db.epoch_as_of(datetime.timedelta(0)) == db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    assert people(db.as_of(datetime.timedelta(0))) == ["Alix", "Gus"]


def test_a_time_before_the_history_is_not_retained():
    db = grafeo.GrafeoDB(history_retention=datetime.timedelta(hours=1))
    db.execute("INSERT (:Person {name: 'Alix'})")
    with pytest.raises(grafeo.GrafeoError, match="not retained"):
        db.as_of(datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC))


def test_a_later_epoch_reads_now(db):
    assert db.epoch_as_of(db.current_epoch() + 1000) == db.current_epoch()


def test_a_naive_datetime_is_refused(db):
    with pytest.raises(ValueError, match="timezone"):
        db.as_of(datetime.datetime(2026, 3, 1, 12, 0))


@pytest.mark.parametrize("point", [1.5, "3", True, None, [3]])
def test_other_points_are_a_type_error(db, point):
    with pytest.raises(TypeError):
        db.epoch_as_of(point)


@pytest.mark.parametrize("point", [-1, -datetime.timedelta(seconds=1)])
def test_a_negative_point_is_a_value_error(db, point):
    with pytest.raises(ValueError):
        db.epoch_as_of(point)


def test_a_datetime_reads_at_its_exact_millisecond():
    db = grafeo.GrafeoDB(history_retention=datetime.timedelta(hours=1))
    point = datetime.datetime(2000, 1, 1, 0, 0, 0, 339999, tzinfo=datetime.UTC)
    with pytest.raises(grafeo.GrafeoError, match=r"a read at 2000-01-01T00:00:00\.339000Z"):
        db.epoch_as_of(point)
    ago = datetime.timedelta(days=-1, seconds=86399, microseconds=999999)
    with pytest.raises(ValueError, match="not negative"):
        db.epoch_as_of(ago)


def test_without_history_a_time_is_not_retained():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})")
    with pytest.raises(grafeo.GrafeoError, match="needs history"):
        db.as_of(datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC))
    # The current epoch reads now, also without history.
    assert people(db.as_of(db.current_epoch())) == ["Alix"]


def test_a_closed_view_raises(db):
    view = db.as_of(db.current_epoch())
    view.close()
    view.close()
    with pytest.raises(ValueError, match="closed"):
        view.execute("MATCH (n) RETURN n")
    assert "closed" in repr(view)


@pytest.mark.parametrize("retention", ["off", "forever", "FOREVER", datetime.timedelta(hours=1)])
def test_history_retention_values(retention):
    grafeo.GrafeoDB(history_retention=retention)


@pytest.mark.parametrize(
    ("retention", "error"),
    [("always", ValueError), (-datetime.timedelta(hours=1), ValueError), (3600, TypeError)],
)
def test_history_retention_refuses_other_values(retention, error):
    with pytest.raises(error):
        grafeo.GrafeoDB(history_retention=retention)


def test_the_retention_is_stored_in_the_file(tmp_path):
    path = str(tmp_path / "history.grafeo")
    db = grafeo.GrafeoDB(path, history_retention="forever")
    db.execute("INSERT (:Person {name: 'Alix'})")
    then = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()
    db = grafeo.GrafeoDB(path)
    assert people(db.as_of(then)) == ["Alix"]
    db.close()
