"""The CDC setting is stored with the database (#679): `enable_cdc()` and `disable_cdc()` persist
it, so `cdc_enabled` after a close and a reopen is what it was, and they raise when it cannot be
stored (a read-only open). An in-memory database toggles as before.
"""

import pytest

grafeo = pytest.importorskip("grafeo")

pytestmark = pytest.mark.requires("cdc")


def test_the_setting_survives_a_reopen_both_ways(tmp_path):
    path = str(tmp_path / "amsterdam.grafeo")
    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is False
    db.enable_cdc()
    db.close()

    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is True
    db.disable_cdc()
    db.close()

    db = grafeo.GrafeoDB(path)
    assert db.cdc_enabled is False
    db.close()


def test_a_read_only_open_raises_and_changes_nothing(tmp_path):
    path = str(tmp_path / "berlin.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()

    reader = grafeo.GrafeoDB.open_read_only(path)
    with pytest.raises(grafeo.GrafeoError):
        reader.enable_cdc()
    assert reader.cdc_enabled is False
    reader.close()


def test_an_in_memory_database_toggles():
    db = grafeo.GrafeoDB()
    db.enable_cdc()
    assert db.cdc_enabled is True
    db.disable_cdc()
    assert db.cdc_enabled is False
