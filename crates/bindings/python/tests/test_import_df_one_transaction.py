"""`import_df` is one transaction: a row that fails leaves nothing of the import.

Each row used to commit on its own, so a failing row left the rows before it
in the database, half an import.
"""

import pytest
from grafeo import GrafeoDB, GrafeoError

pd = pytest.importorskip("pandas")


def names(db):
    return sorted(row["n"] for row in db.execute("MATCH (p:Person) RETURN p.name AS n"))


def test_a_failing_node_row_imports_nothing():
    db = GrafeoDB()
    db.execute("CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE")
    people = pd.DataFrame({"name": ["Alix", "Gus", "Alix"]})
    with pytest.raises(GrafeoError):
        db.import_df(people, mode="nodes", label="Person")
    assert names(db) == []
    assert db.import_df(pd.DataFrame({"name": ["Alix", "Gus"]}), mode="nodes", label="Person") == 2
    assert names(db) == ["Alix", "Gus"]


def test_a_failing_edge_row_imports_nothing():
    db = GrafeoDB()
    db.import_df(pd.DataFrame({"name": ["Alix", "Gus"]}), mode="nodes", label="Person")
    ids = [row["id"] for row in db.execute("MATCH (p:Person) RETURN id(p) AS id ORDER BY p.name")]
    edges = pd.DataFrame({"source": [ids[0], ids[0]], "target": [ids[1], 999_999]})
    with pytest.raises(GrafeoError):
        db.import_df(edges, mode="edges", edge_type="KNOWS")
    assert list(db.execute("MATCH ()-[k:KNOWS]->() RETURN count(k) AS c"))[0]["c"] == 0
