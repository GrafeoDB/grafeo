"""What compact() reports: the checkpoint, the versions dropped and the rows
whose property values it compressed, in memory and in a database file."""

from grafeo import GrafeoDB


def people(db):
    db.execute("UNWIND range(0, 999) AS i INSERT (:Person {i: i, age: i % 90})")


def total_age(db):
    return [row["total"] for row in db.execute("MATCH (p:Person) RETURN sum(p.age) AS total")]


def test_compact_compresses_an_in_memory_database():
    db = GrafeoDB()
    people(db)
    before = total_age(db)
    report = db.compact()
    assert set(report) == {
        "checkpointed",
        "versions_collected",
        "rows_compressed",
        "duration_ms",
    }
    assert report["checkpointed"] is False
    assert report["rows_compressed"] == 1000
    assert total_age(db) == before
    assert db.compact()["rows_compressed"] == 0, "nothing is decoded any more"


def test_compact_compresses_a_database_file(tmp_path):
    path = str(tmp_path / "prague.grafeo")
    db = GrafeoDB(path)
    people(db)
    report = db.compact()
    assert report["checkpointed"] is True
    assert report["rows_compressed"] == 1000
    db.execute("MATCH (p:Person {i: 3}) SET p.age = 41")
    db.close()
    db = GrafeoDB(path)
    ages = [row["age"] for row in db.execute("MATCH (p:Person {i: 3}) RETURN p.age AS age")]
    assert ages == [41]
    db.close()
