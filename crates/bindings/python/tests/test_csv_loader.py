"""The native CSV loader from Python (0.6.1): db.load_nodes_csv and db.load_edges_csv.

Typed columns, the report as a dict, ends found by key, a missing end (error or skip),
a bad value naming its row and column with nothing loaded, and the same rows as the
batch API on the same file. Files are written to pytest's tmp_path.
"""

import datetime

import pytest

grafeo = pytest.importorskip("grafeo")

PEOPLE = (
    "id,name,score,active,born,junk\n"
    "1,Alix,9.5,true,1990-05-01,x\n"
    "2,Gus,7,false,1985-12-24,y\n"
    "3,Vincent,8.25,true,2001-02-03,z\n"
)
TYPES = {
    "id": "int64",
    "name": "string",
    "score": "float64",
    "active": "bool",
    "born": "date",
    "junk": "skip",
}
KNOWS = "src,dst,since\n1,2,2019\n2,3,2021\n3,1,2020\n"
KNOWS_TYPES = {"src": "int64", "dst": "int64", "since": "int64"}


@pytest.fixture
def files(tmp_path):
    people = tmp_path / "people.csv"
    people.write_text(PEOPLE, encoding="utf-8")
    knows = tmp_path / "knows.csv"
    knows.write_text(KNOWS, encoding="utf-8")
    return people, knows


def test_nodes_with_types(files):
    people, _ = files
    db = grafeo.GrafeoDB()
    report = db.load_nodes_csv(str(people), ["Person"], key="id", types=TYPES)
    assert report == {"rows": 3, "created": 3, "skipped": 0}
    rows = [
        tuple(r.values())
        for r in db.execute(
            "MATCH (n:Person) RETURN n.id, n.name, n.score, n.active, n.born, n.junk ORDER BY n.id"
        )
    ]
    assert rows[0] == (1, "Alix", 9.5, True, datetime.date(1990, 5, 1), None)
    assert rows[1][2] == 7.0 and isinstance(rows[1][2], float)


def test_a_path_object_and_one_label_as_text(files):
    people, _ = files
    db = grafeo.GrafeoDB()
    assert db.load_nodes_csv(people, "Person")["created"] == 3
    # Without types every column is text.
    assert list(db.execute("MATCH (n:Person {name: 'Gus'}) RETURN n.id"))[0]["n.id"] == "2"


def test_edges_by_key(files):
    people, knows = files
    db = grafeo.GrafeoDB()
    db.load_nodes_csv(str(people), ["Person"], key="id", types=TYPES)
    report = db.load_edges_csv(
        str(knows),
        "KNOWS",
        source=("Person", "id", "src"),
        target=("Person", "id", "dst"),
        types=KNOWS_TYPES,
    )
    assert report == {"rows": 3, "created": 3, "skipped": 0}
    names = [
        tuple(r.values())
        for r in db.execute(
            "MATCH (a)-[r:KNOWS]->(b) RETURN a.name, b.name, r.since ORDER BY r.since"
        )
    ]
    assert names == [
        ("Alix", "Gus", 2019),
        ("Vincent", "Alix", 2020),
        ("Gus", "Vincent", 2021),
    ]


def test_a_missing_end(tmp_path, files):
    people, _ = files
    knows = tmp_path / "broken.csv"
    knows.write_text("src,dst\n1,2\n2,77\n", encoding="utf-8")
    db = grafeo.GrafeoDB()
    db.load_nodes_csv(str(people), ["Person"], key="id", types=TYPES)
    ends = {
        "source": ("Person", "id", "src"),
        "target": ("Person", "id", "dst"),
        "types": {"src": "int64", "dst": "int64"},
    }
    with pytest.raises(Exception, match="77"):
        db.load_edges_csv(str(knows), "KNOWS", **ends)
    assert list(db.execute("MATCH ()-[r]->() RETURN count(r) AS c"))[0]["c"] == 0
    report = db.load_edges_csv(str(knows), "KNOWS", missing="skip", **ends)
    assert report == {"rows": 2, "created": 1, "skipped": 1}


def test_a_bad_value_loads_nothing(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,score\n1,9.5\n2,lots\n", encoding="utf-8")
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="score") as error:
        db.load_nodes_csv(str(path), ["Person"], types={"id": "int64", "score": "float64"})
    assert "lots" in str(error.value)
    assert list(db.execute("MATCH (n) RETURN count(n) AS c"))[0]["c"] == 0


def test_an_unknown_type_name(files):
    people, _ = files
    db = grafeo.GrafeoDB()
    with pytest.raises(Exception, match="decimal128"):
        db.load_nodes_csv(str(people), ["Person"], types={"id": "decimal128"})


def test_the_same_rows_as_the_batch_api(files):
    """The loader and batch_create_nodes_with_props give the same nodes for the same file."""
    people, _ = files
    loaded = grafeo.GrafeoDB()
    loaded.load_nodes_csv(str(people), ["Person"], key="id", types=TYPES)
    batched = grafeo.GrafeoDB()
    batched.batch_create_nodes_with_props(
        "Person",
        [
            {
                "id": 1,
                "name": "Alix",
                "score": 9.5,
                "active": True,
                "born": datetime.date(1990, 5, 1),
            },
            {
                "id": 2,
                "name": "Gus",
                "score": 7.0,
                "active": False,
                "born": datetime.date(1985, 12, 24),
            },
            {
                "id": 3,
                "name": "Vincent",
                "score": 8.25,
                "active": True,
                "born": datetime.date(2001, 2, 3),
            },
        ],
    )
    query = (
        "MATCH (n:Person) RETURN n.id, n.name, n.score, n.active, n.born, "
        "size(keys(n)) AS k ORDER BY n.id"
    )
    assert [tuple(r.values()) for r in loaded.execute(query)] == [
        tuple(r.values()) for r in batched.execute(query)
    ]


def test_type_names_in_any_case(files):
    people, _ = files
    db = grafeo.GrafeoDB()
    mixed = {"id": "Int64", "score": "FLOAT64", "active": "Bool", "born": "date"}
    assert db.load_nodes_csv(str(people), ["Person"], key="id", types=mixed)["created"] == 3
    row = next(iter(db.execute("MATCH (n:Person {id: 1}) RETURN n.score, n.active")))
    assert tuple(row.values()) == (9.5, True)
