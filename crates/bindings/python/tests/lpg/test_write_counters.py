"""result.counters: what a query's writes changed."""

from grafeo import GrafeoDB


def test_counters_report_what_the_writes_changed():
    db = GrafeoDB()
    result = db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
    assert result.counters == {
        "nodes_created": 2,
        "nodes_deleted": 0,
        "edges_created": 1,
        "edges_deleted": 0,
        "properties_set": 2,
        "labels_added": 2,
        "labels_removed": 0,
    }


def test_merge_counts_only_what_it_creates():
    db = GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})")
    result = db.execute(
        "UNWIND $names AS name MERGE (p:Person {name: name}) SET p.seen = true",
        {"names": ["Alix", "Vincent", "Vincent"]},
    )
    assert result.counters["nodes_created"] == 1
    assert result.counters["properties_set"] == 4  # Vincent's name, three seen flags


def test_reads_and_deletes():
    db = GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
    assert not any(db.execute("MATCH (p:Person) RETURN p.name").counters.values())
    deleted = db.execute("MATCH (p:Person {name: 'Alix'}) DETACH DELETE p").counters
    assert (deleted["nodes_deleted"], deleted["edges_deleted"]) == (1, 1)
