"""Bulk paths a code-graph client (Deriva) needs from Python.

- `update_nodes(labels, rows, key="id")` sets properties on existing nodes
  only, matched by `key` and all of `labels`, in one call: a row's keys are
  merged into the node's (the others stay), a row whose key matches no node
  is skipped and creates nothing, a row without the key is skipped. It
  returns `{"updated", "skipped", "skipped_rows"}`, as `upsert_nodes` does,
  and gives the same nodes as `UNWIND $rows ... MATCH ... SET`.
- `get_properties_batch(ids, keys)` reads several properties of many nodes
  in one call, without pyarrow, as `get_property_batch(ids, key)` reads one:
  a dict of one list per key, aligned with `ids`, None where a node lacks
  the property or no node has the id.
- `MATCH (n) RETURN n` gives each node with the labels and properties
  `get_nodes_by_label` gives (its speed is in scripts/bench/deriva_selectors.py).
- The `arrow` extra (`grafeo[arrow]`) installs pyarrow for `nodes_to_arrow`
  and its siblings, so a downstream package can declare it.
"""

import importlib.metadata

import pytest

try:
    import grafeo

    GRAFEO_AVAILABLE = True
except ImportError:
    GRAFEO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GRAFEO_AVAILABLE, reason="Grafeo Python bindings not installed")


@pytest.fixture
def db():
    db = grafeo.GrafeoDB()
    db.batch_create_nodes_with_props(
        ["Graph", "Method"],
        [{"id": f"m{i}", "name": f"Method{i}", "lines": i * 10, "active": True} for i in range(5)],
    )
    db.batch_create_nodes_with_props(["Graph", "File"], [{"id": "f0", "name": "File0"}])
    return db


def node_props(db, node_id):
    row = db.execute(
        "MATCH (n:Graph {id: $id}) RETURN properties(n) AS p", {"id": node_id}
    ).to_list()
    assert len(row) == 1, f"{node_id}: {row}"
    return row[0]["p"]


def count(db, query="MATCH (n) RETURN count(n) AS c"):
    return db.execute(query).to_list()[0]["c"]


# ---------------------------------------------------------------------------
# update_nodes
# ---------------------------------------------------------------------------


def test_update_nodes_merges_partial_rows_into_existing_nodes(db):
    summary = db.update_nodes(
        ["Graph", "Method"],
        [{"id": "m1", "lines": 11}, {"id": "m2", "active": False, "owner": "Alix"}],
    )
    assert summary["updated"] == 2
    assert summary["skipped"] == 0
    assert node_props(db, "m1") == {"id": "m1", "name": "Method1", "lines": 11, "active": True}
    assert node_props(db, "m2") == {
        "id": "m2",
        "name": "Method2",
        "lines": 20,
        "active": False,
        "owner": "Alix",
    }
    assert node_props(db, "m3")["lines"] == 30


def test_update_nodes_skips_missing_ids_and_creates_nothing(db):
    before = count(db)
    summary = db.update_nodes(
        ["Graph", "Method"],
        [
            {"id": "m0", "lines": 1},
            {"id": "nope", "lines": 2},
            {"lines": 3},
            {"id": "m4", "lines": 4},
        ],
    )
    assert summary["updated"] == 2
    assert summary["skipped"] == 2
    assert sorted(summary["skipped_rows"]) == [1, 2]
    assert count(db) == before
    assert count(db, "MATCH (n {id: 'nope'}) RETURN count(n) AS c") == 0
    assert node_props(db, "m0")["lines"] == 1
    assert node_props(db, "m4")["lines"] == 4


def test_update_nodes_matches_every_label(db):
    # f0 has the key but not the Method label: skipped, unchanged.
    summary = db.update_nodes(["Graph", "Method"], [{"id": "f0", "name": "changed"}])
    assert (summary["updated"], summary["skipped"]) == (0, 1)
    assert node_props(db, "f0")["name"] == "File0"
    summary = db.update_nodes(["Graph"], [{"id": "f0", "name": "changed"}])
    assert summary["updated"] == 1
    assert node_props(db, "f0")["name"] == "changed"


def test_update_nodes_with_another_key(db):
    summary = db.update_nodes(["Method"], [{"name": "Method3", "lines": 333}], key="name")
    assert summary["updated"] == 1
    assert node_props(db, "m3")["lines"] == 333


def test_update_nodes_gives_what_the_unwind_query_gives(db):
    rows = [{"id": f"m{i}", "lines": 1000 + i, "tag": f"t{i % 2}"} for i in range(0, 5, 2)]
    other = grafeo.GrafeoDB()
    other.batch_create_nodes_with_props(
        ["Graph", "Method"],
        [{"id": f"m{i}", "name": f"Method{i}", "lines": i * 10, "active": True} for i in range(5)],
    )
    other.batch_create_nodes_with_props(["Graph", "File"], [{"id": "f0", "name": "File0"}])
    db.update_nodes(["Graph", "Method"], rows)
    other.execute(
        "UNWIND $rows AS r MATCH (n:Graph:Method {id: r.id}) SET n.lines = r.lines, n.tag = r.tag",
        {"rows": rows},
    )
    query = "MATCH (n:Graph) RETURN n.id AS id, properties(n) AS p ORDER BY id"
    assert db.execute(query).to_list() == other.execute(query).to_list()


def test_update_nodes_of_no_rows_is_a_no_op(db):
    assert db.update_nodes(["Method"], [])["updated"] == 0


# ---------------------------------------------------------------------------
# get_properties_batch
# ---------------------------------------------------------------------------


def method_ids(db):
    return [node_id for node_id, _ in db.get_nodes_by_label("Method")]


def test_get_properties_batch_reads_columns_aligned_with_the_ids(db):
    ids = sorted(method_ids(db))
    columns = db.get_properties_batch(ids, ["id", "lines", "active"])
    assert set(columns) == {"id", "lines", "active"}
    assert all(len(values) == len(ids) for values in columns.values())
    for position, node_id in enumerate(ids):
        props = db.get_node(node_id).properties()
        assert columns["id"][position] == props["id"]
        assert columns["lines"][position] == props["lines"]
        assert columns["active"][position] is True


def test_get_properties_batch_gives_none_for_missing_properties_and_nodes(db):
    file_id = next(node_id for node_id, _ in db.get_nodes_by_label("File"))
    method_id = method_ids(db)[0]
    columns = db.get_properties_batch([method_id, file_id, 999_999], ["lines", "name"])
    assert columns["lines"][1] is None
    assert columns["name"][1] == "File0"
    assert columns["lines"][2] is None
    assert columns["name"][2] is None
    assert columns["lines"][0] is not None


def test_get_properties_batch_keeps_order_and_repeats(db):
    ids = method_ids(db)
    picked = [ids[3], ids[0], ids[3]]
    columns = db.get_properties_batch(picked, ["id"])
    assert columns["id"][0] == columns["id"][2]
    assert columns["id"][0] != columns["id"][1]


def test_get_properties_batch_of_no_ids_or_no_keys(db):
    assert db.get_properties_batch([], ["id", "lines"]) == {"id": [], "lines": []}
    assert db.get_properties_batch(method_ids(db), []) == {}


# ---------------------------------------------------------------------------
# Whole nodes through the query path
# ---------------------------------------------------------------------------


def test_return_n_gives_the_nodes_get_nodes_by_label_gives(db):
    # A node in a result row is a dict: `_id`, `_labels` and its properties.
    direct = dict(db.get_nodes_by_label("Method"))
    rows = db.execute("MATCH (n:Method) RETURN n").to_list()
    assert len(rows) == len(direct) == 5
    for row in rows:
        node = dict(row["n"])
        node_id = node.pop("_id")
        assert sorted(node.pop("_labels")) == ["Graph", "Method"]
        assert node == direct[node_id]


# ---------------------------------------------------------------------------
# The arrow extra
# ---------------------------------------------------------------------------


def test_the_arrow_extra_is_declared_and_brings_pyarrow():
    # Parsed, not compared as text: maturin writes `pyarrow>=14 ; extra == 'arrow'`.
    from packaging.requirements import Requirement

    metadata = importlib.metadata.metadata("grafeo")
    assert "arrow" in (metadata.get_all("Provides-Extra") or [])
    requires = [Requirement(r) for r in metadata.get_all("Requires-Dist") or []]
    arrow = [
        r
        for r in requires
        if r.name == "pyarrow" and r.marker is not None and r.marker.evaluate({"extra": "arrow"})
    ]
    assert arrow, [str(r) for r in requires]
    # Not pulled in without the extra.
    assert not arrow[0].marker.evaluate({"extra": ""})


def test_nodes_to_arrow_with_the_extra(db):
    pa = pytest.importorskip("pyarrow")
    table = db.nodes_to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 6
