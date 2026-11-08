"""The memory settings from Python (#751): `GrafeoDB(memory_limit=..., query_memory_limit=...)`,
read back through `memory_usage()`. The default limit is enforced; a write past the limit raises
GRAFEO-S001 naming the memory limit and changes nothing; the query limit defaults to the global
one and never exceeds it.
"""

import warnings

import pytest

grafeo = pytest.importorskip("grafeo")

MIB = 1024 * 1024


def fill(db):
    """Writes 1 MiB statements until one is refused; returns the refusal."""
    for _ in range(10_000):
        try:
            db.execute("UNWIND range(1, 1024) AS i INSERT (:Fill {text: '" + "x" * 1000 + "'})")
        except grafeo.GrafeoError as error:
            return error
    pytest.fail("the limit never bound")


def count(db, label):
    return len(list(db.execute(f"MATCH (n:{label}) RETURN n")))


def test_the_settings_read_back():
    db = grafeo.GrafeoDB(memory_limit=64 * MIB, query_memory_limit=8 * MIB)
    usage = db.memory_usage()
    assert usage["limit_bytes"] == 64 * MIB
    assert usage["query_limit_bytes"] == 8 * MIB


def test_the_defaults():
    usage = grafeo.GrafeoDB().memory_usage()
    assert usage["limit_bytes"] > 0, "the default is enforced, so it is a number"
    assert usage["query_limit_bytes"] == usage["limit_bytes"]


def test_the_query_limit_never_exceeds_the_global_one():
    usage = grafeo.GrafeoDB(memory_limit=32 * MIB, query_memory_limit=256 * MIB).memory_usage()
    assert usage["query_limit_bytes"] == 32 * MIB


def test_a_write_past_the_limit_raises_and_changes_nothing():
    db = grafeo.GrafeoDB(memory_limit=64 * MIB)
    error = fill(db)
    assert error.error_code == "GRAFEO-S001", str(error)
    assert "memory limit" in str(error), str(error)
    held = count(db, "Fill")
    assert held > 0, "writes up to the limit succeeded"
    with pytest.raises(grafeo.GrafeoError):
        db.execute("UNWIND range(1, 1024) AS i INSERT (:Fill {text: '" + "x" * 1000 + "'})")
    assert count(db, "Fill") == held, "the refused write changed nothing"
    assert count(db, "Fill") > 0, "reads work at the limit"


@pytest.mark.parametrize("keywords", [{"memory_limit": -1}, {"query_memory_limit": -1}])
def test_a_negative_limit_is_refused(keywords):
    with pytest.raises((ValueError, OverflowError, grafeo.GrafeoError)):
        grafeo.GrafeoDB(**keywords)


def test_the_spill_settings_read_back(tmp_path):
    """A database file has a spill directory, so its spill settings read back."""
    db = grafeo.GrafeoDB(str(tmp_path / "zero.grafeo"), spill_limit=0)
    assert db.memory_usage()["spill"]["limit_bytes"] == 0, "zero: nothing spills"
    db.close()
    db = grafeo.GrafeoDB(
        str(tmp_path / "unlimited.grafeo"), spill_limit=16 * MIB, unlimited_spill=True
    )
    assert db.memory_usage()["spill"]["limit_bytes"] is None, "unlimited"
    db.close()


def test_an_in_memory_database_spills_nothing_whatever_the_settings():
    usage = grafeo.GrafeoDB(unlimited_spill=True).memory_usage()
    assert usage["spill"]["limit_bytes"] == 0


# Distinct 1 KiB strings: data that stays past the limit once reloaded (a repeated string and
# dense integers load back as small compressed columns).
FILLER = (
    "UNWIND range(1, 20000) AS i INSERT (:Fill {i: i, text: CAST(i AS STRING) || '"
    + "x" * 1000
    + "'})"
)


def test_an_open_over_the_limit_warns_and_reads(tmp_path):
    path = str(tmp_path / "over.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute(FILLER)
    db.close()

    with pytest.warns(UserWarning, match="MEMORY_LIMIT_PASSED"):
        tight = grafeo.GrafeoDB(path, memory_limit=MIB)
    assert count(tight, "Fill") == 20000, "reads work over the limit"
    tight.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roomy = grafeo.GrafeoDB(path)
    roomy.close()
    assert not [w for w in caught if "MEMORY_LIMIT_PASSED" in str(w.message)]
