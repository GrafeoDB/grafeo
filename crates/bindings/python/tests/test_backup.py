"""Full and incremental backups and a restore to an epoch.

A full backup is an image of the database at an epoch, written while commits go
on; the incremental backups after it hold the WAL after that epoch, so a restore
is the image and those segments.
"""

import json
import os
import shutil
import threading

import grafeo
import pytest


def people(db):
    return sorted(row["name"] for row in db.execute("MATCH (p:Person) RETURN p.name AS name"))


def test_full_and_incremental_backups_restore_to_each_epoch(tmp_path):
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(str(tmp_path / "amsterdam.grafeo"))
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.backup_full(backups)
    full_epoch = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.backup_incremental(backups)
    latest = db.current_epoch()
    db.close()

    at_full = str(tmp_path / "full.grafeo")
    grafeo.GrafeoDB.restore_to_epoch(backups, full_epoch, at_full)
    restored = grafeo.GrafeoDB(at_full)
    assert people(restored) == ["Alix"]
    restored.close()

    at_latest = str(tmp_path / "latest.grafeo")
    grafeo.GrafeoDB.restore_to_epoch(backups, latest, at_latest)
    restored = grafeo.GrafeoDB(at_latest)
    assert people(restored) == ["Alix", "Gus"]
    restored.close()


def test_an_incremental_backup_works_across_a_close(tmp_path):
    """The chain survives restarts: close() keeps the WAL the next incremental reads."""
    path = str(tmp_path / "prague.grafeo")
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.backup_full(backups)
    full_epoch = db.current_epoch()
    db.close()

    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()

    db = grafeo.GrafeoDB(path)
    db.backup_incremental(backups)
    latest = db.current_epoch()
    db.close()

    for epoch, expected in ((full_epoch, ["Alix"]), (latest, ["Alix", "Gus"])):
        restored_path = str(tmp_path / f"restored-{epoch}.grafeo")
        grafeo.GrafeoDB.restore_to_epoch(backups, epoch, restored_path)
        restored = grafeo.GrafeoDB(restored_path)
        assert people(restored) == expected
        restored.close()


WRITERS = 3
# Commits each writer makes before the backup starts, and at least in all.
BEFORE_THE_BACKUP = 5
AT_LEAST = 20


def committed(db):
    """The (writer, seq) of every Person and of every City, each sorted."""
    return tuple(
        sorted(
            (row["w"], row["s"])
            for row in db.execute(f"MATCH (n:{label}) RETURN n.writer AS w, n.seq AS s")
        )
        for label in ("Person", "City")
    )


def test_a_full_backup_during_concurrent_writers_holds_whole_transactions(tmp_path):
    """Writers commit a Person and a City per transaction, and roll one back
    each time, while a full backup is taken: its image holds every transaction
    whole, each writer's commits from the first on, at least the ones before the
    backup started, and nothing rolled back. The incremental backup after them
    restores every commit, opened for writing and read-only."""
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(str(tmp_path / "paris.grafeo"))
    stop = threading.Event()
    counts = [0] * WRITERS
    started = [threading.Event() for _ in range(WRITERS)]
    errors = []

    def write(writer):
        try:
            seq = 0
            while seq < AT_LEAST or not stop.is_set():
                with db.begin_transaction() as tx:
                    tx.execute(f"INSERT (:Person {{writer: {writer}, seq: {seq}}})")
                    tx.execute(f"INSERT (:City {{writer: {writer}, seq: {seq}}})")
                    tx.commit()
                with db.begin_transaction() as tx:
                    tx.execute(f"INSERT (:Person {{writer: {writer}, seq: -1}})")
                    tx.rollback()
                seq += 1
                counts[writer] = seq
                if seq == BEFORE_THE_BACKUP:
                    started[writer].set()
        except Exception as error:  # noqa: BLE001 - reported by the main thread
            errors.append(error)
            started[writer].set()

    threads = [threading.Thread(target=write, args=(writer,)) for writer in range(WRITERS)]
    for thread in threads:
        thread.start()
    for event in started:
        assert event.wait(timeout=60)
    before = list(counts)
    db.backup_full(backups)
    stop.set()
    for thread in threads:
        thread.join()
    assert errors == []
    total = list(counts)

    image = tmp_path / "image.grafeo"
    shutil.copy(tmp_path / "backups" / "backup_full_0000.grafeo", image)
    copy = grafeo.GrafeoDB.open_read_only(str(image))
    people, cities = committed(copy)
    copy.close()
    assert people == cities, "every transaction is whole"
    for writer in range(WRITERS):
        seqs = [seq for (w, seq) in people if w == writer]
        assert seqs == list(range(len(seqs))), f"writer {writer} from its first commit on"
        assert before[writer] <= len(seqs) <= total[writer]

    db.backup_incremental(backups)
    latest = db.current_epoch()
    db.close()
    restored = str(tmp_path / "restored.grafeo")
    grafeo.GrafeoDB.restore_to_epoch(backups, latest, restored)
    expected = sorted((writer, seq) for writer in range(WRITERS) for seq in range(total[writer]))
    for open_restored in (grafeo.GrafeoDB, grafeo.GrafeoDB.open_read_only):
        db = open_restored(restored)
        assert committed(db) == (expected, expected)
        db.close()


def test_a_read_only_database_takes_a_full_backup_of_what_it_shows(tmp_path):
    """A database opened read-only takes a full backup of the state it shows, in
    the directory of the writer's backups: a restore to each backup's epoch has
    exactly the people committed by then."""
    path = str(tmp_path / "ghent.grafeo")
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.backup_full(backups)
    alix = db.current_epoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()

    db = grafeo.GrafeoDB.open_read_only(path)
    assert people(db) == ["Alix", "Gus"]
    db.backup_full(backups)
    gus = db.current_epoch()
    db.close()

    for epoch, expected in ((alix, ["Alix"]), (gus, ["Alix", "Gus"])):
        restored = str(tmp_path / f"restored-{epoch}.grafeo")
        grafeo.GrafeoDB.restore_to_epoch(backups, epoch, restored)
        db = grafeo.GrafeoDB.open_read_only(restored)
        assert people(db) == expected
        db.close()


def listing(directory):
    """Every file of `directory` with its bytes, sorted by name."""
    files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                files.append((name, f.read()))
    return files


def test_an_incremental_backup_with_nothing_new_is_a_no_op(tmp_path):
    """Nothing written since the last backup: the call succeeds and writes nothing."""
    path = str(tmp_path / "lisbon.grafeo")
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.backup_full(backups)
    before = listing(backups)
    db.backup_incremental(backups)
    db.backup_incremental(backups)
    assert listing(backups) == before
    db.execute("INSERT (:Person {name: 'Gus'})")
    db.backup_incremental(backups)
    latest = db.current_epoch()
    db.close()
    restored = str(tmp_path / "restored.grafeo")
    grafeo.GrafeoDB.restore_to_epoch(backups, latest, restored)
    db = grafeo.GrafeoDB(restored)
    assert people(db) == ["Alix", "Gus"]
    db.close()


def test_a_removed_wal_segment_breaks_the_chain(tmp_path):
    """The WAL segment the next incremental backup starts at is gone: the error has
    its own code (GRAFEO-S005), is not retryable, says to take a new full backup,
    and writes nothing; a new full backup starts a chain that works."""
    path = str(tmp_path / "vienna.grafeo")
    backups = str(tmp_path / "backups")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.backup_full(backups)
    db.execute("INSERT (:Person {name: 'Gus'})")
    wal = path + ".wal"
    with open(os.path.join(wal, "backup_cursor.meta"), encoding="utf-8") as f:
        start = json.load(f)["wal_lsn"]
    os.remove(os.path.join(wal, f"wal_{start:020}.log"))
    before = listing(backups)

    with pytest.raises(grafeo.GrafeoError) as raised:
        db.backup_incremental(backups)
    assert raised.value.error_code == "GRAFEO-S005", str(raised.value)
    assert raised.value.is_retryable is False
    assert "full backup" in str(raised.value).lower()
    assert listing(backups) == before

    db.backup_full(backups)
    db.execute("INSERT (:Person {name: 'Vincent'})")
    db.backup_incremental(backups)
    latest = db.current_epoch()
    db.close()
    restored = str(tmp_path / "restored.grafeo")
    grafeo.GrafeoDB.restore_to_epoch(backups, latest, restored)
    db = grafeo.GrafeoDB(restored)
    assert people(db) == ["Alix", "Gus", "Vincent"]
    db.close()
