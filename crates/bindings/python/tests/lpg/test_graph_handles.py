"""db.graph(name): work in one named graph without switching the database's graph."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from grafeo import GrafeoDB, GraphHandle


def ids(target):
    return sorted(row["id"] for row in target.execute("MATCH (n) RETURN n.id AS id"))


@pytest.fixture
def db():
    db = GrafeoDB()
    db.create_graph("extraction")
    db.create_graph("model")
    return db


def test_handles_interleave_without_switching(db):
    extraction, model = db.graph("extraction"), db.graph("model")
    assert isinstance(extraction, GraphHandle)
    assert extraction.name == "extraction"

    file = extraction.create_node(["File"], {"id": "file::a"})
    model.create_node(["Component"], {"id": "ac::a"})
    extraction.set_node_property(file.id, "size", 3)
    model.execute("INSERT (:Component {id: 'ac::b'})")

    assert ids(extraction) == ["file::a"]
    assert ids(model) == ["ac::a", "ac::b"]
    assert ids(db) == []
    assert db.current_graph() is None
    assert extraction.get_node(file.id).properties()["size"] == 3


def test_handles_are_safe_across_threads(db):
    extraction, model = db.graph("extraction"), db.graph("model")

    def write(handle, prefix):
        for i in range(200):
            handle.create_node(["N"], {"id": f"{prefix}{i}"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(write, h, p)
            for h, p in [(extraction, "g"), (model, "m"), (extraction, "h"), (model, "n")]
        ]
        for future in futures:
            future.result()

    assert len(ids(extraction)) == 400 and {i[0] for i in ids(extraction)} == {"g", "h"}
    assert len(ids(model)) == 400 and {i[0] for i in ids(model)} == {"m", "n"}


def test_a_missing_graph_raises_instead_of_falling_back(db):
    with pytest.raises(Exception, match="'nowhere' does not exist"):
        db.graph("nowhere")
    model = db.graph("model")
    db.drop_graph("model")
    with pytest.raises(Exception, match="does not exist"):
        model.create_node(["Stray"], {"id": "x"})
    assert ids(db) == []


def test_a_handle_ignores_set_graph(db):
    db.set_graph("extraction")
    model = db.graph("model")
    model.create_node(["Component"], {"id": "ac::a"})
    assert ids(model) == ["ac::a"]
    assert ids(db) == []
    assert db.current_graph() == "extraction"


def test_direct_api_and_indexes_are_per_graph(db):
    model, extraction = db.graph("model"), db.graph("extraction")
    model.create_property_index("id")
    component = model.create_node(["Component"], {"id": "x"})
    extraction.create_node(["File"], {"id": "x"})
    service = model.create_node(["Service"], {"id": "y"})
    edge = model.create_edge(component.id, service.id, "SERVES", {"since": 2020})

    assert model.has_property_index("id")
    assert not extraction.has_property_index("id")
    assert model.find_nodes_by_property("id", "x") == [component.id]
    assert model.get_edge(edge.id).edge_type == "SERVES"
    assert model.add_node_label(component.id, "Critical")
    assert model.remove_node_property(component.id, "id")
    assert model.delete_edge(edge.id)
    assert model.delete_node(service.id)
    assert len(model.batch_create_nodes_with_props("Component", [{"id": "c"}, {"id": "d"}])) == 2

    rows = [
        (row["id"], sorted(row["labels"]))
        for row in model.execute("MATCH (n) RETURN n.id AS id, labels(n) AS labels")
    ]
    assert sorted(rows, key=str) == sorted(
        [(None, ["Component", "Critical"]), ("c", ["Component"]), ("d", ["Component"])], key=str
    )


def test_a_transaction_on_a_handle_groups_writes(db):
    model = db.graph("model")
    with model.begin_transaction() as tx:
        tx.execute("INSERT (:Component {id: 'ac::a'})")
        tx.rollback()
    assert ids(model) == []

    with model.begin_transaction() as tx:
        tx.execute("INSERT (:Component {id: 'ac::b'})")
        tx.commit()
    assert ids(model) == ["ac::b"]
    assert ids(db) == []


def test_cypher_runs_in_the_handle_graph(db):
    model = db.graph("model")
    model.execute_cypher("CREATE (:Component {id: 'ac::a'})")
    assert [row["id"] for row in model.execute_cypher("MATCH (n) RETURN n.id AS id")] == ["ac::a"]
    assert ids(db) == []
