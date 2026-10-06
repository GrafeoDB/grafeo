"""A closed database takes no more writes.

Once `close()` of a persistent database starts, every write raises
`DatabaseClosedError` (a `GrafeoError` with code `GRAFEO-T007`): before 0.6.0
such a write reported success and was lost. Reads still work, and an
in-memory database, which has nothing to persist, keeps taking writes.
"""

import grafeo
import pytest


def names(db):
    return [
        row["name"] for row in db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
    ]


def test_a_write_after_close_raises_database_closed_error(tmp_path):
    path = str(tmp_path / "amsterdam.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()

    with pytest.raises(grafeo.DatabaseClosedError) as raised:
        db.execute("INSERT (:Person {name: 'Gus'})")
    error = raised.value
    assert isinstance(error, grafeo.GrafeoError)
    assert isinstance(error, RuntimeError)
    assert error.error_code == "GRAFEO-T007"
    assert error.is_retryable is False
    assert "database is closed" in str(error)


def test_every_kind_of_write_after_close_raises_and_changes_nothing(tmp_path):
    path = str(tmp_path / "berlin.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()

    with pytest.raises(grafeo.DatabaseClosedError):
        db.create_node(["Person"], {"name": "Gus"})
    with pytest.raises(grafeo.DatabaseClosedError):
        db.execute("CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE")
    assert names(db) == ["Alix"], "reads still work and show no refused write"

    reopened = grafeo.GrafeoDB(path)
    assert names(reopened) == ["Alix"]
    reopened.close()


def test_an_in_memory_database_still_takes_writes_after_close():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()
    db.execute("INSERT (:Person {name: 'Gus'})")
    assert names(db) == ["Alix", "Gus"]


def test_a_sparql_update_after_close_raises_and_is_not_in_the_file(tmp_path):
    path = str(tmp_path / "prague.grafeo")
    db = grafeo.GrafeoDB(path)
    if not hasattr(db, "execute_sparql"):
        db.close()
        pytest.skip("this build has no SPARQL")
    db.execute_sparql('INSERT DATA { <http://ex.org/alix> <http://ex.org/city> "Amsterdam" . }')
    db.close()

    with pytest.raises(grafeo.DatabaseClosedError):
        db.execute_sparql('INSERT DATA { <http://ex.org/gus> <http://ex.org/city> "Berlin" . }')
    query = "SELECT ?s WHERE { ?s ?p ?o }"
    assert len(db.execute_sparql(query)) == 1, "queries still run"

    reopened = grafeo.GrafeoDB(path)
    assert len(reopened.execute_sparql(query)) == 1, "the file holds Alix's triple only"
    reopened.close()


def test_graph_index_and_persist_calls_after_close_raise(tmp_path):
    path = str(tmp_path / "paris.grafeo")
    db = grafeo.GrafeoDB(path)
    db.create_graph("berlin")
    db.close()

    for call in (
        lambda: db.create_graph("prague"),
        lambda: db.drop_graph("berlin"),
        lambda: db.create_property_index("name"),
        lambda: db.drop_property_index("name"),
        db.wal_checkpoint,
        lambda: db.save(str(tmp_path / "copy.grafeo")),
    ):
        with pytest.raises(grafeo.DatabaseClosedError):
            call()
    assert not (tmp_path / "copy.grafeo").exists(), "nothing was saved"

    reopened = grafeo.GrafeoDB(path)
    assert reopened.list_graphs() == ["berlin"]
    assert not reopened.has_property_index("name")
    reopened.close()
