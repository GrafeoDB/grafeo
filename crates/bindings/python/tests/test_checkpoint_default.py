"""Automatic checkpoints are on by default, and the WAL's segment size is a setting.

`checkpoint_wal_size=` is the WAL size past the last checkpoint at which the
database checkpoints by itself: 256 MiB by default, `None` turns it off (the
WAL then grows until close or a manual checkpoint). `wal_segment_size=` is
where the WAL rotates, 64 MiB by default. A segment size outside 1 MiB to
1 GiB and a checkpoint size below 1 MiB are refused, naming the setting and
its bounds.
"""

import math
import time
from pathlib import Path

import grafeo
import pytest

MIB = 1 << 20


def write(db, size=64 * 1024):
    db.execute("INSERT (:Blob {b: $b})", {"b": "x" * size})


def wal_bytes(db):
    return db.wal_status()["size_bytes"]


def wait_until(what, condition):
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, f"never: {what}"
        time.sleep(0.02)


def segments(path):
    return sorted(Path(str(path) + ".wal").glob("wal_*.log"))


def test_the_default_is_256_mib_and_no_interval(tmp_path):
    db = grafeo.GrafeoDB(str(tmp_path / "amsterdam.grafeo"))
    status = db.checkpoint_status()
    assert status["wal_bytes"] == 256 * MIB
    assert status["interval_secs"] is None
    db.close()


def test_none_turns_automatic_checkpoints_off(tmp_path):
    db = grafeo.GrafeoDB(str(tmp_path / "berlin.grafeo"), checkpoint_wal_size=None)
    assert db.checkpoint_status()["wal_bytes"] is None
    while wal_bytes(db) < 2 * MIB:
        write(db)
    before = wal_bytes(db)
    time.sleep(0.5)
    assert wal_bytes(db) == before, "nothing checkpoints by itself"
    db.wal_checkpoint()
    assert wal_bytes(db) == 0
    db.close()


def test_a_checkpoint_size_checkpoints_by_itself(tmp_path):
    db = grafeo.GrafeoDB(str(tmp_path / "lisbon.grafeo"), checkpoint_wal_size=MIB)
    assert db.checkpoint_status()["wal_bytes"] == MIB
    # As many writes as the WAL lacks, counted first: the automatic checkpoint may shrink the WAL
    # between two of them, and a loop on its size would then go on writing.
    for _ in range(math.ceil((MIB + 64 * 1024 - wal_bytes(db)) / (64 * 1024))):
        write(db)
    wait_until("the automatic checkpoint", lambda: wal_bytes(db) < MIB)
    assert db.checkpoint_status()["failures"] == 0
    db.close()


def test_the_segment_size_sets_where_the_wal_rotates(tmp_path):
    path = tmp_path / "porto.grafeo"
    db = grafeo.GrafeoDB(str(path), checkpoint_wal_size=None, wal_segment_size=MIB)
    for _ in range(48):
        write(db)
    files = segments(path)
    assert len(files) >= 3, files
    for sealed in files[:-1]:
        size = sealed.stat().st_size
        assert MIB // 2 < size <= MIB + 128 * 1024, (sealed.name, size)
    db.close()


@pytest.mark.parametrize(
    ("setting", "value", "names"),
    [
        ("wal_segment_size", MIB - 1, ["wal_segment_size", "1 MiB", "1 GiB"]),
        ("wal_segment_size", (1 << 30) + 1, ["wal_segment_size", "1 MiB", "1 GiB"]),
        ("checkpoint_wal_size", MIB - 1, ["checkpoint_wal_size", "1 MiB"]),
        ("checkpoint_wal_size", 0, ["checkpoint_wal_size", "1 MiB"]),
    ],
)
def test_sizes_out_of_bounds_are_refused_naming_their_bounds(tmp_path, setting, value, names):
    path = tmp_path / "antwerp.grafeo"
    with pytest.raises(grafeo.GrafeoError) as raised:
        grafeo.GrafeoDB(str(path), **{setting: value})
    for name in names:
        assert name in str(raised.value), str(raised.value)
    assert not path.exists()
