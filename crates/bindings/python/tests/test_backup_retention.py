"""The WAL a backup cursor keeps is bounded, and the cursor can be released.

`wal_retention_limit=` caps the WAL a checkpoint keeps beyond itself for the
backup cursor (1 GiB by default). Once incremental backups stop and the
checkpoints pass the limit, the next incremental backup fails with
GRAFEO-S005, naming the limit, and a full backup starts a new chain.
`release_backup_cursor()` drops the cursor on purpose: it returns whether one
was released, and the next incremental backup fails with GRAFEO-S005 until a
full backup.
"""

import grafeo
import pytest

LIMIT = 256 * 1024


def write_batch(db, start):
    db.execute(
        f"UNWIND range({start}, {start + 99}) AS i INSERT (:Person {{i: i, note: '{'x' * 300}'}})"
    )


def chain_broken(db, backups):
    with pytest.raises(grafeo.GrafeoError) as error:
        db.backup_incremental(backups)
    assert error.value.error_code == "GRAFEO-S005", str(error.value)
    return str(error.value)


def test_a_stopped_backup_is_cut_at_the_limit_and_a_full_backup_restarts(tmp_path):
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(str(tmp_path / "db.grafeo"), wal_retention_limit=LIMIT)
    db.backup_full(backups)
    for round_ in range(30):
        write_batch(db, round_ * 100)
        db.wal_checkpoint()
    message = chain_broken(db, backups)
    assert str(LIMIT) in message, message
    db.backup_full(backups)
    write_batch(db, 3_000)
    db.backup_incremental(backups)
    db.close()


def test_the_cursor_is_released_once_and_asks_for_a_full_backup(tmp_path):
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(str(tmp_path / "db.grafeo"))
    write_batch(db, 0)
    db.backup_full(backups)
    write_batch(db, 100)
    assert db.release_backup_cursor() is True
    assert db.release_backup_cursor() is False
    assert "cursor" in chain_broken(db, backups)
    db.backup_full(backups)
    write_batch(db, 200)
    db.backup_incremental(backups)
    db.close()
