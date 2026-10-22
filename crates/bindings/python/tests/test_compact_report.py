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


# The reproduction of #221, in Cypher as reported: compact() of a graph with
# two edge types failed with "duplicate edge type".
SEVERAL_EDGE_TYPES = [
    "CREATE (:A {name: 'a'})",
    "CREATE (:B {name: 'b'})",
    "CREATE (:C {name: 'c'})",
    "MATCH (a:A), (b:B) CREATE (a)-[:CALLS]->(b)",
    "MATCH (a:A), (c:C) CREATE (a)-[:USES_TYPE]->(c)",
]


def edges_from_a(db):
    rows = db.execute_cypher("MATCH (:A)-[r]->(x) RETURN type(r) AS t, x.name AS n ORDER BY t")
    return [(row["t"], row["n"]) for row in rows]


def test_compact_with_several_edge_types_in_memory():
    db = GrafeoDB()
    for statement in SEVERAL_EDGE_TYPES:
        db.execute_cypher(statement)
    db.compact()
    assert edges_from_a(db) == [("CALLS", "b"), ("USES_TYPE", "c")]


def test_compact_with_several_edge_types_survives_a_reopen(tmp_path):
    path = str(tmp_path / "calls.grafeo")
    db = GrafeoDB(path)
    for statement in SEVERAL_EDGE_TYPES:
        db.execute_cypher(statement)
    db.compact()
    db.execute_cypher("MATCH (a:A), (c:C) CREATE (a)-[:IMPORTS]->(c)")
    db.close()
    db = GrafeoDB(path)
    assert edges_from_a(db) == [("CALLS", "b"), ("IMPORTS", "c"), ("USES_TYPE", "c")]
    db.close()
