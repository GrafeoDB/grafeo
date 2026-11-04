"""A schema statement belongs to its transaction, as a data write does.

A rollback undoes it, in memory and on disk, also back to a savepoint; another
session does not see it before the commit; a commit keeps it, also after a
reopen. The catalog is read with SHOW, or a CALL for procedures.
"""

import json

import grafeo
import pytest

INDEXES = "SHOW INDEXES"
NODE_TYPES = "SHOW NODE TYPES"
EDGE_TYPES = "SHOW EDGE TYPES"
GRAPH_TYPES = "SHOW GRAPH TYPES"
CONSTRAINTS = "SHOW CONSTRAINTS"
GRAPHS = "SHOW GRAPHS"
SCHEMAS = "SHOW SCHEMAS"
PROCEDURE = "CALL p1()"

HASH_INDEX = "CREATE INDEX i1 FOR (n:Person) ON (n.age)"
BTREE_INDEX = "CREATE INDEX i1 FOR (n:Person) ON (n.age) USING BTREE"
TEXT_INDEX = "CREATE INDEX i1 FOR (n:Person) ON (n.name) USING TEXT"
VECTOR_INDEX = "CREATE VECTOR INDEX i1 ON :Person(embedding) DIMENSION 2 METRIC 'cosine'"
NODE_TYPE = "CREATE NODE TYPE City (name STRING)"
EDGE_TYPE = "CREATE EDGE TYPE ROAD (km INTEGER)"
GRAPH_TYPE = "CREATE GRAPH TYPE atlas { (:Town {size INT64})-[:ROAD]->(:Town) }"
UNIQUE = "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.email) UNIQUE"
STORED_PROCEDURE = "CREATE PROCEDURE p1() RETURNS (v INTEGER) AS { RETURN 1 AS v }"

# name: (method, statement before the transaction, statement in it, what reads the catalog)
KINDS = {
    "create_graph": ("execute", None, "CREATE GRAPH g2", GRAPHS),
    "drop_graph": ("execute", "CREATE GRAPH g2", "DROP GRAPH g2", GRAPHS),
    "create_schema": ("execute", None, "CREATE SCHEMA s2", SCHEMAS),
    "drop_schema": ("execute", "CREATE SCHEMA s2", "DROP SCHEMA s2", SCHEMAS),
    "create_hash_index": ("execute", None, HASH_INDEX, INDEXES),
    "create_btree_index": ("execute", None, BTREE_INDEX, INDEXES),
    "create_text_index": ("execute", None, TEXT_INDEX, INDEXES),
    "create_vector_index": ("execute", None, VECTOR_INDEX, INDEXES),
    "drop_hash_index": ("execute", HASH_INDEX, "DROP INDEX i1", INDEXES),
    "drop_btree_index": ("execute", BTREE_INDEX, "DROP INDEX i1", INDEXES),
    "drop_text_index": ("execute", TEXT_INDEX, "DROP INDEX i1", INDEXES),
    "drop_vector_index": ("execute", VECTOR_INDEX, "DROP INDEX i1", INDEXES),
    "create_node_type": (
        "execute",
        None,
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        NODE_TYPES,
    ),
    "alter_node_type": (
        "execute",
        NODE_TYPE,
        "ALTER NODE TYPE City ADD PROPERTY size STRING",
        NODE_TYPES,
    ),
    "drop_node_type": ("execute", NODE_TYPE, "DROP NODE TYPE City", NODE_TYPES),
    "create_edge_type": ("execute", None, EDGE_TYPE, EDGE_TYPES),
    "alter_edge_type": (
        "execute",
        EDGE_TYPE,
        "ALTER EDGE TYPE ROAD ADD PROPERTY name STRING",
        EDGE_TYPES,
    ),
    "drop_edge_type": ("execute", EDGE_TYPE, "DROP EDGE TYPE ROAD", EDGE_TYPES),
    "create_graph_type": ("execute", None, GRAPH_TYPE, GRAPH_TYPES),
    "alter_graph_type": (
        "execute",
        GRAPH_TYPE,
        "ALTER GRAPH TYPE atlas ADD EDGE TYPE RAIL",
        GRAPH_TYPES,
    ),
    "drop_graph_type": ("execute", GRAPH_TYPE, "DROP GRAPH TYPE atlas", GRAPH_TYPES),
    "create_not_null_constraint": (
        "execute",
        None,
        "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.name) NOT NULL",
        CONSTRAINTS,
    ),
    "create_unique_constraint": ("execute", None, UNIQUE, CONSTRAINTS),
    "drop_constraint": ("execute", UNIQUE, "DROP CONSTRAINT c1", CONSTRAINTS),
    "create_procedure": ("execute", None, STORED_PROCEDURE, PROCEDURE),
    "drop_procedure": ("execute", STORED_PROCEDURE, "DROP PROCEDURE p1", PROCEDURE),
    "cypher_create_index": ("execute_cypher", None, HASH_INDEX, INDEXES),
    "cypher_drop_index": ("execute_cypher", HASH_INDEX, "DROP INDEX i1", INDEXES),
    "cypher_create_not_null_constraint": (
        "execute_cypher",
        None,
        "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.name IS NOT NULL",
        CONSTRAINTS,
    ),
    "cypher_create_unique_constraint": (
        "execute_cypher",
        None,
        "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.email IS UNIQUE",
        CONSTRAINTS,
    ),
    "cypher_drop_constraint": (
        "execute_cypher",
        UNIQUE,
        "DROP CONSTRAINT c1",
        CONSTRAINTS,
    ),
}


def prepared(kind, path=None):
    db = grafeo.GrafeoDB(str(path)) if path else grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix', age: 30, embedding: vector([1.0, 0.0])})")
    before = KINDS[kind][1]
    if before:
        db.execute(before)
    return db


def catalog(db, query):
    """The catalog's rows, or that the procedure is not there; any other failure is the test's."""
    try:
        return sorted(json.dumps(row, sort_keys=True, default=str) for row in db.execute(query))
    except grafeo.GrafeoError as error:
        if query != PROCEDURE or "unknown procedure 'p1'" not in str(error):
            raise
        return "no such procedure"


def run_in(tx, kind):
    method, _, statement, _ = KINDS[kind]
    getattr(tx, method)(statement)


def catalogs(kind):
    """The catalog read before and after the same statement outside a transaction."""
    db = prepared(kind)
    query = KINDS[kind][3]
    before = catalog(db, query)
    run_in(db, kind)
    after = catalog(db, query)
    db.close()
    assert before != after, "the catalog read does not show this statement"
    return before, after


@pytest.mark.parametrize("kind", KINDS)
def test_a_rollback_undoes_it(kind):
    before, _ = catalogs(kind)
    db = prepared(kind)
    tx = db.begin_transaction()
    run_in(tx, kind)
    tx.rollback()
    assert catalog(db, KINDS[kind][3]) == before


@pytest.mark.parametrize("kind", KINDS)
def test_a_rollback_to_a_savepoint_undoes_it(kind):
    before, _ = catalogs(kind)
    db = prepared(kind)
    tx = db.begin_transaction()
    tx.savepoint("s")
    run_in(tx, kind)
    tx.rollback_to_savepoint("s")
    tx.commit()
    assert catalog(db, KINDS[kind][3]) == before


@pytest.mark.parametrize("kind", KINDS)
def test_another_session_does_not_see_it_before_the_commit(kind):
    before, _ = catalogs(kind)
    db = prepared(kind)
    tx = db.begin_transaction()
    run_in(tx, kind)
    try:
        assert catalog(db, KINDS[kind][3]) == before
    finally:
        tx.rollback()


@pytest.mark.parametrize("kind", KINDS)
def test_a_commit_keeps_it(kind):
    _, after = catalogs(kind)
    db = prepared(kind)
    tx = db.begin_transaction()
    run_in(tx, kind)
    tx.commit()
    assert catalog(db, KINDS[kind][3]) == after


@pytest.mark.parametrize("kind", KINDS)
def test_a_rollback_holds_after_a_reopen(kind, tmp_path):
    before, _ = catalogs(kind)
    path = tmp_path / "rolled_back.grafeo"
    db = prepared(kind, path)
    tx = db.begin_transaction()
    run_in(tx, kind)
    tx.rollback()
    db.close()
    db = grafeo.GrafeoDB(str(path))
    assert catalog(db, KINDS[kind][3]) == before
    db.close()


@pytest.mark.parametrize("kind", KINDS)
def test_a_commit_holds_after_a_reopen(kind, tmp_path):
    _, after = catalogs(kind)
    path = tmp_path / "committed.grafeo"
    db = prepared(kind, path)
    tx = db.begin_transaction()
    run_in(tx, kind)
    tx.commit()
    db.close()
    db = grafeo.GrafeoDB(str(path))
    assert catalog(db, KINDS[kind][3]) == after
    db.close()


# ---------------------------------------------------------------------------
# Data beside the schema statement
# ---------------------------------------------------------------------------

REFUSING = {
    "not_null_node_type": (
        "execute",
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "INSERT (:City {size: 3})",
    ),
    "not_null_constraint": (
        "execute",
        "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.email) NOT NULL",
        "INSERT (:Person {name: 'Gus'})",
    ),
    "unique_constraint": (
        "execute",
        "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.age) UNIQUE",
        "INSERT (:Person {name: 'Gus', age: 30})",
    ),
    "cypher_not_null_constraint": (
        "execute_cypher",
        "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.email IS NOT NULL",
        "INSERT (:Person {name: 'Gus'})",
    ),
    "cypher_unique_constraint": (
        "execute_cypher",
        "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.age IS UNIQUE",
        "INSERT (:Person {name: 'Gus', age: 30})",
    ),
}


def people():
    db = grafeo.GrafeoDB()
    db.execute("INSERT (:Person {name: 'Alix', age: 30, email: 'alix@example.org'})")
    return db


@pytest.mark.parametrize("kind", REFUSING)
def test_a_write_in_the_transaction_reads_the_new_schema(kind):
    method, statement, write = REFUSING[kind]
    db = people()
    tx = db.begin_transaction()
    getattr(tx, method)(statement)
    with pytest.raises(grafeo.GrafeoError):
        tx.execute(write)
    tx.rollback()


@pytest.mark.parametrize("kind", REFUSING)
def test_a_rolled_back_constraint_refuses_no_later_write(kind):
    method, statement, write = REFUSING[kind]
    db = people()
    tx = db.begin_transaction()
    getattr(tx, method)(statement)
    tx.rollback()
    db.execute(write)


@pytest.mark.parametrize("kind", REFUSING)
def test_a_committed_constraint_refuses_a_later_write(kind):
    method, statement, write = REFUSING[kind]
    db = people()
    tx = db.begin_transaction()
    getattr(tx, method)(statement)
    tx.commit()
    with pytest.raises(grafeo.GrafeoError):
        db.execute(write)


def test_a_rollback_undoes_the_schema_and_the_data_together():
    db = people()
    tx = db.begin_transaction()
    tx.execute(NODE_TYPE)
    tx.execute("INSERT (:City {name: 'Amsterdam'})")
    tx.rollback()
    assert list(db.execute(NODE_TYPES)) == []
    assert list(db.execute("MATCH (c:City) RETURN c.name AS name")) == []


def test_a_rollback_to_a_savepoint_keeps_what_came_before_it():
    db = people()
    tx = db.begin_transaction()
    tx.execute("CREATE NODE TYPE City (name STRING)")
    tx.savepoint("s")
    tx.execute("CREATE NODE TYPE Town (name STRING)")
    tx.rollback_to_savepoint("s")
    tx.commit()
    names = [row["name"] for row in db.execute(NODE_TYPES)]
    assert names == ["City"]


def test_an_index_made_in_the_transaction_serves_its_own_writes():
    db = people()
    tx = db.begin_transaction()
    tx.execute(HASH_INDEX)
    tx.execute("INSERT (:Person {name: 'Gus', age: 40})")
    rows = list(tx.execute("MATCH (n:Person {age: 40}) RETURN n.name AS name"))
    tx.rollback()
    assert rows == [{"name": "Gus"}]


def test_a_name_is_free_again_after_a_rollback():
    db = people()
    tx = db.begin_transaction()
    tx.execute(HASH_INDEX)
    tx.rollback()
    db.execute(HASH_INDEX)
    assert len(list(db.execute(INDEXES))) == 1


def test_a_rolled_back_drop_leaves_a_working_index():
    db = people()
    db.execute(HASH_INDEX)
    tx = db.begin_transaction()
    tx.execute("DROP INDEX i1")
    tx.rollback()
    db.execute("INSERT (:Person {name: 'Gus', age: 40})")
    rows = list(db.execute("MATCH (n:Person {age: 40}) RETURN n.name AS name"))
    assert rows == [{"name": "Gus"}]


def test_a_rolled_back_drop_of_a_graph_keeps_its_data():
    db = people()
    db.execute("CREATE GRAPH g2")
    tx = db.begin_transaction()
    tx.execute("USE GRAPH g2")
    tx.execute("INSERT (:Person {name: 'Vincent'})")
    tx.commit()
    tx = db.begin_transaction()
    tx.execute("DROP GRAPH g2")
    tx.rollback()
    tx = db.begin_transaction()
    tx.execute("USE GRAPH g2")
    rows = list(tx.execute("MATCH (n:Person) RETURN n.name AS name"))
    tx.rollback()
    assert rows == [{"name": "Vincent"}]


def test_a_type_created_and_dropped_in_one_transaction_leaves_nothing():
    db = people()
    tx = db.begin_transaction()
    tx.execute(NODE_TYPE)
    tx.execute("DROP NODE TYPE City")
    tx.commit()
    assert list(db.execute(NODE_TYPES)) == []


def test_gp18_a_rollback_undoes_the_schema_and_the_data_together():
    db = grafeo.GrafeoDB()
    tx = db.begin_transaction()
    tx.execute("CREATE NODE TYPE City (name STRING NOT NULL)")
    tx.execute("INSERT (:City {name: 'Amsterdam'})")
    tx.rollback()
    assert list(db.execute("SHOW NODE TYPES")) == []
    assert list(db.execute("MATCH (c:City) RETURN c.name")) == []
    # No rule stayed: a city with no name is no error.
    db.execute("INSERT (:City {size: 3})")


def test_gp18_a_commit_keeps_the_schema_and_the_data_together():
    db = grafeo.GrafeoDB()
    tx = db.begin_transaction()
    tx.execute("CREATE NODE TYPE City (name STRING NOT NULL)")
    tx.execute("INSERT (:City {name: 'Amsterdam'})")
    tx.commit()
    assert len(list(db.execute("SHOW NODE TYPES"))) == 1
    assert [row["c.name"] for row in db.execute("MATCH (c:City) RETURN c.name")] == ["Amsterdam"]
    with pytest.raises(Exception, match="GRAFEO-V001"):
        db.execute("INSERT (:City {size: 3})")


def test_the_issues_rolled_back_rule_refuses_no_later_write(tmp_path):
    """A draft in a transaction, rolled back, leaves no rule: the way it was first met."""
    path = str(tmp_path / "rule.grafeo")
    db = grafeo.GrafeoDB(path)
    db.create_node(["File"], {"id": "a", "must": 1})
    tx = db.begin_transaction()
    tx.execute_cypher("CREATE CONSTRAINT c3 FOR (n:File) REQUIRE n.must IS NOT NULL", {})
    tx.rollback()
    db.create_node(["File"], {"id": "b"})
    db.close()
    db = grafeo.GrafeoDB(path)
    assert list(db.execute("SHOW CONSTRAINTS")) == []
    db.create_node(["File"], {"id": "c"})
    assert len(list(db.execute("MATCH (f:File) RETURN f.id"))) == 3
    db.close()


def test_the_issues_rolled_back_drop_graph_loses_no_data(tmp_path):
    """A rolled-back `DROP GRAPH` keeps the graph and its data, also after a reopen."""
    path = str(tmp_path / "graphs.grafeo")
    db = grafeo.GrafeoDB(path)
    db.execute("CREATE GRAPH g2")
    tx = db.begin_transaction()
    tx.execute("USE GRAPH g2")
    tx.execute("INSERT (:Person {name: 'Vincent'})")
    tx.commit()
    tx = db.begin_transaction()
    tx.execute("DROP GRAPH g2")
    tx.rollback()

    def people(database):
        with database.begin_transaction() as reader:
            reader.execute("USE GRAPH g2")
            return [row["p.name"] for row in reader.execute("MATCH (p:Person) RETURN p.name")]

    assert len(list(db.execute("SHOW GRAPHS"))) == 1
    assert people(db) == ["Vincent"]
    db.close()
    db = grafeo.GrafeoDB(path)
    assert people(db) == ["Vincent"]
    db.close()
