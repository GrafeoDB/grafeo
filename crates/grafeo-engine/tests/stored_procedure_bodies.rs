//! A stored procedure's body runs as the same statement run directly: its
//! writes are checked against the schema and the constraints, and a body
//! focused on a graph with `USE GRAPH` writes that graph. The body was
//! planned without the constraint validator, so a CALL stored what the
//! statement run directly refused, and without the graph its `USE` names,
//! so the write landed in the session's graph.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main stored_procedure_bodies::
//! ```

#![cfg(all(feature = "gql", feature = "algos"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The single value `query` returns.
fn single(db: &GrafeoDB, query: &str) -> Value {
    db.execute(query).unwrap().rows()[0][0].clone()
}

/// The names of the cities of the graph `query` reads, sorted.
fn cities(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut names: Vec<String> = db
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("a city without a name: {other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// A body that leaves out a property its node type requires is refused,
/// as the same insert run directly is, and stores nothing.
#[test]
fn a_body_that_breaks_a_not_null_property_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    db.execute(
        "CREATE PROCEDURE nameless() RETURNS (n INTEGER) AS { \
         INSERT (:Person {city: 'Paris'}) RETURN 1 AS n }",
    )
    .unwrap();

    let direct = db
        .execute("INSERT (:Person {city: 'Paris'}) RETURN 1 AS n")
        .unwrap_err()
        .to_string();
    let called = db
        .execute("CALL nameless() YIELD n RETURN n")
        .expect_err("the body breaks the node type")
        .to_string();
    assert!(
        called.contains("missing required property 'name' on :Person"),
        "the call fails as the direct insert does ({direct}): {called}"
    );
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(0)
    );
}

/// A body that writes a value a UNIQUE constraint already holds is refused,
/// as the same insert run directly is.
#[test]
fn a_body_that_breaks_a_unique_constraint_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT FOR (p:Person) ON (p.email) UNIQUE")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Alix', email: 'alix@example.org'})")
        .unwrap();
    db.execute(
        "CREATE PROCEDURE duplicate() RETURNS (n INTEGER) AS { \
         INSERT (:Person {name: 'Gus', email: 'alix@example.org'}) RETURN 1 AS n }",
    )
    .unwrap();

    db.execute("INSERT (:Person {name: 'Gus', email: 'alix@example.org'})")
        .expect_err("the direct insert breaks the constraint");
    let called = db
        .execute("CALL duplicate() YIELD n RETURN n")
        .expect_err("the body breaks the constraint")
        .to_string();
    assert!(called.contains("UNIQUE constraint violation"), "{called}");
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(1),
        "only Alix"
    );
}

/// A body focused on the graph `paris` writes `paris`, not the session's
/// graph, as the same statement run directly does.
#[test]
fn a_body_focused_on_a_graph_writes_that_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute("USE GRAPH paris INSERT (:City {name: 'Direct'})")
        .unwrap();
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute("CALL to_paris() YIELD n RETURN n").unwrap();

    assert_eq!(
        cities(&db, "MATCH (c:City) RETURN c.name"),
        Vec::<String>::new(),
        "the default graph holds no city"
    );
    assert_eq!(
        cities(&db, "USE GRAPH paris MATCH (c:City) RETURN c.name"),
        vec!["Called", "Direct"]
    );
}

/// A body without a `USE` writes the graph of the session that calls it,
/// in a transaction or not.
#[test]
fn a_body_writes_the_session_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH berlin").unwrap();
    db.execute(
        "CREATE PROCEDURE add_city() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();

    let session = db.session();
    session.execute("USE GRAPH berlin").unwrap();
    session.execute("CALL add_city() YIELD n RETURN n").unwrap();
    let mut session = db.session();
    session.execute("USE GRAPH berlin").unwrap();
    session.begin_transaction().unwrap();
    session.execute("CALL add_city() YIELD n RETURN n").unwrap();
    session.commit().unwrap();

    assert_eq!(
        cities(&db, "MATCH (c:City) RETURN c.name"),
        Vec::<String>::new()
    );
    assert_eq!(
        cities(&db, "USE GRAPH berlin MATCH (c:City) RETURN c.name"),
        vec!["Called", "Called"]
    );
}

/// The single value the call `query` returns in `session`.
fn single_in(session: &grafeo_engine::Session, query: &str) -> Value {
    session.execute(query).unwrap().rows()[0][0].clone()
}

/// A body focused on a graph bound to a closed graph type is checked
/// against that type, as the same insert run directly is.
#[test]
fn a_body_that_breaks_a_closed_graph_type_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE City (name STRING)").unwrap();
    db.execute("CREATE GRAPH TYPE travel (NODE TYPE City)")
        .unwrap();
    db.execute("CREATE GRAPH trips TYPED travel").unwrap();
    db.execute(
        "CREATE PROCEDURE add_person() RETURNS (n INTEGER) AS { \
         USE GRAPH trips INSERT (:Person {name: 'Alix'}) RETURN 1 AS n }",
    )
    .unwrap();

    let direct = db
        .execute("USE GRAPH trips INSERT (:Person {name: 'Alix'}) RETURN 1 AS n")
        .expect_err("the graph type has no node type Person")
        .to_string();
    let called = db
        .execute("CALL add_person() YIELD n RETURN n")
        .expect_err("the body breaks the graph type")
        .to_string();
    assert_eq!(called, direct);
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(0),
        "nothing in the default graph"
    );
    assert_eq!(
        single(&db, "USE GRAPH trips MATCH (p:Person) RETURN count(p)"),
        Value::Int64(0)
    );
}

/// A body that writes a property larger than the database allows is
/// refused, as the same insert run directly is.
#[test]
fn a_body_that_writes_a_too_large_property_is_refused() {
    let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_max_property_size(64))
        .unwrap();
    let long = "Amsterdam".repeat(10);
    db.execute(&format!(
        "CREATE PROCEDURE long_name() RETURNS (n INTEGER) AS {{ \
         INSERT (:City {{name: '{long}'}}) RETURN 1 AS n }}"
    ))
    .unwrap();

    let direct = db
        .execute(&format!("INSERT (:City {{name: '{long}'}}) RETURN 1 AS n"))
        .expect_err("the name is too large")
        .to_string();
    let called = db
        .execute("CALL long_name() YIELD n RETURN n")
        .expect_err("the body writes a too large name")
        .to_string();
    assert!(called.contains("exceeds maximum size"), "{called}");
    assert_eq!(called, direct);
    assert_eq!(
        single(&db, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(0)
    );
}

/// A body focused on a graph reads that graph: its matches, its name and
/// its counts are the ones the statement run directly sees.
#[test]
fn a_body_focused_on_a_graph_reads_that_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute("USE GRAPH paris INSERT (:City {name: 'Paris'}), (:City {name: 'Lyon'})")
        .unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    db.execute(
        "CREATE PROCEDURE paris_cities() RETURNS (n INTEGER) AS { \
         USE GRAPH paris MATCH (c:City) RETURN count(c) AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE paris_name() RETURNS (g STRING) AS { \
         USE GRAPH paris RETURN CURRENT_GRAPH AS g }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE paris_info() RETURNS (i ANY) AS { \
         USE GRAPH paris RETURN info() AS i }",
    )
    .unwrap();

    assert_eq!(
        single(&db, "CALL paris_cities() YIELD n RETURN n"),
        Value::Int64(2)
    );
    assert_eq!(
        single(&db, "CALL paris_name() YIELD g RETURN g"),
        single(&db, "USE GRAPH paris RETURN CURRENT_GRAPH AS g")
    );
    let info = single(&db, "CALL paris_info() YIELD i RETURN i");
    assert_eq!(info, single(&db, "USE GRAPH paris RETURN info() AS i"));
    let Value::Map(info) = info else {
        panic!("info() is a map: {info:?}");
    };
    assert_eq!(
        info.get(&grafeo_common::types::PropertyKey::from("node_count")),
        Some(&Value::Int64(2)),
        "the counts of paris"
    );
}

/// A body focused on a graph that does not exist is refused as the same
/// statement run directly is, and writes nothing anywhere.
#[test]
fn a_body_focused_on_a_missing_graph_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE to_prague() RETURNS (n INTEGER) AS { \
         USE GRAPH prague INSERT (:City {name: 'Prague'}) RETURN 1 AS n }",
    )
    .unwrap();

    let direct = db
        .execute("USE GRAPH prague INSERT (:City {name: 'Prague'}) RETURN 1 AS n")
        .expect_err("there is no graph prague")
        .to_string();
    let called = db
        .execute("CALL to_prague() YIELD n RETURN n")
        .expect_err("the body names a missing graph")
        .to_string();
    assert!(called.contains("Graph 'prague' does not exist"), "{called}");
    assert!(direct.contains("Graph 'prague' does not exist"), "{direct}");
    assert_eq!(
        single(&db, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(0)
    );
}

/// A body focused on a graph the user has no grant for is refused, as the
/// same statement run directly is; a read grant does not let it write.
#[test]
fn a_body_focused_on_a_graph_without_a_grant_is_refused() {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = GrafeoDB::new_in_memory();
    for graph in ["berlin", "paris"] {
        db.execute(&format!("CREATE GRAPH {graph}")).unwrap();
    }
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE paris_cities() RETURNS (n INTEGER) AS { \
         USE GRAPH paris MATCH (c:City) RETURN count(c) AS n }",
    )
    .unwrap();

    let no_grant = db.session_with_identity(
        Identity::new("gus", [Role::ReadWrite])
            .with_grants([Grant::new("berlin", Role::ReadWrite)]),
    );
    no_grant.execute("USE GRAPH berlin").unwrap();
    let direct = no_grant
        .execute("USE GRAPH paris INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect_err("gus has no grant for paris")
        .to_string();
    for call in [
        "CALL to_paris() YIELD n RETURN n",
        "CALL paris_cities() YIELD n RETURN n",
    ] {
        let called = no_grant
            .execute(call)
            .expect_err("the body names a graph without a grant")
            .to_string();
        assert!(called.contains("permission denied"), "{called}");
        assert_eq!(called, direct);
    }

    let read_grant = db.session_with_identity(
        Identity::new("vincent", [Role::ReadWrite])
            .with_grants([Grant::new("paris", Role::ReadOnly)]),
    );
    assert_eq!(
        single_in(&read_grant, "CALL paris_cities() YIELD n RETURN n"),
        Value::Int64(0),
        "a read grant lets the body read"
    );
    let direct = read_grant
        .execute("USE GRAPH paris INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect_err("vincent may only read paris")
        .to_string();
    let called = read_grant
        .execute("CALL to_paris() YIELD n RETURN n")
        .expect_err("the body writes a graph with a read grant")
        .to_string();
    assert!(called.contains("permission denied"), "{called}");
    assert_eq!(called, direct);

    for graph in [
        "MATCH (c:City)",
        "USE GRAPH paris MATCH (c:City)",
        "USE GRAPH berlin MATCH (c:City)",
    ] {
        assert_eq!(
            single(&db, &format!("{graph} RETURN count(c)")),
            Value::Int64(0),
            "{graph}"
        );
    }
}

/// A read-only session, by its role or its transaction, may not call a
/// body that writes, as it may not run the write directly.
#[test]
fn a_read_only_session_may_not_call_a_writing_body() {
    use grafeo_engine::auth::Role;

    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE add_city() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();

    let reader = db.session_with_role(Role::ReadOnly);
    let direct = reader
        .execute("USE GRAPH paris INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect_err("a reader may not write")
        .to_string();
    for call in [
        "CALL to_paris() YIELD n RETURN n",
        "CALL add_city() YIELD n RETURN n",
    ] {
        let called = reader.execute(call).expect_err("a reader may not write");
        assert_eq!(called.to_string(), direct);
    }

    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let direct = session
        .execute("USE GRAPH paris INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect_err("a read-only transaction may not write")
        .to_string();
    for call in [
        "CALL to_paris() YIELD n RETURN n",
        "CALL add_city() YIELD n RETURN n",
    ] {
        let called = session
            .execute(call)
            .expect_err("a read-only transaction may not write");
        assert_eq!(called.to_string(), direct);
    }
    session.execute("ROLLBACK").unwrap();

    for graph in ["MATCH (c:City)", "USE GRAPH paris MATCH (c:City)"] {
        assert_eq!(
            single(&db, &format!("{graph} RETURN count(c)")),
            Value::Int64(0),
            "{graph}"
        );
    }
}

/// A body run in an explicit transaction writes in it: the transaction
/// sees the body's writes in the graph its `USE` names and in its own, and
/// its rollback leaves nothing in either.
#[test]
fn a_body_in_a_transaction_that_rolls_back_leaves_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE add_city() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Here'}) RETURN 1 AS n }",
    )
    .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CALL to_paris() YIELD n RETURN n").unwrap();
    session.execute("CALL add_city() YIELD n RETURN n").unwrap();
    assert_eq!(
        single_in(&session, "USE GRAPH paris MATCH (c:City) RETURN count(c)"),
        Value::Int64(1),
        "the transaction sees the body's write in paris"
    );
    assert_eq!(
        single_in(&session, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(1),
        "and the one in its own graph"
    );
    assert_eq!(
        single(&db, "USE GRAPH paris MATCH (c:City) RETURN count(c)"),
        Value::Int64(0),
        "another session does not see the open transaction's write"
    );
    session.rollback().unwrap();

    for graph in ["MATCH (c:City)", "USE GRAPH paris MATCH (c:City)"] {
        assert_eq!(
            single(&db, &format!("{graph} RETURN count(c)")),
            Value::Int64(0),
            "{graph}"
        );
    }
}

/// A body is GQL, so it compares a zoned datetime with a local one as GQL
/// does, at the session time zone, whatever the language of the call.
#[cfg(feature = "cypher")]
#[test]
fn a_body_compares_datetimes_as_gql_from_a_cypher_call() {
    let db = GrafeoDB::new_in_memory();
    let comparison = "zoned_datetime('2026-03-01T10:00:00+01:00') = \
                      local_datetime('2026-03-01T10:00:00')";
    db.execute(&format!(
        "CREATE PROCEDURE same_instant() RETURNS (same BOOLEAN) AS {{ \
         RETURN {comparison} AS same }}"
    ))
    .unwrap();

    let session = db.session();
    session.set_time_zone("+01:00").unwrap();
    let direct = single_in(&session, &format!("RETURN {comparison} AS same"));
    assert_eq!(
        direct,
        Value::Bool(true),
        "GQL reads the local one at +01:00"
    );
    assert_eq!(
        single_in(&session, "CALL same_instant() YIELD same RETURN same"),
        direct
    );
    let from_cypher = session
        .execute_cypher("CALL same_instant() YIELD same RETURN same")
        .unwrap()
        .rows()[0][0]
        .clone();
    assert_eq!(from_cypher, direct);
}

/// A body that calls another stored procedure: the inner body runs on the
/// graph of the body that calls it, unless its own `USE` names another,
/// and is checked against the schema as any body is.
#[test]
fn a_nested_call_runs_as_its_body_run_directly() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH berlin").unwrap();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    db.execute(
        "CREATE PROCEDURE add_city() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Inner'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Paris'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE nameless() RETURNS (n INTEGER) AS { \
         INSERT (:Person {city: 'Prague'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE in_berlin() RETURNS (n INTEGER) AS { \
         USE GRAPH berlin CALL add_city() YIELD n RETURN n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE via_paris() RETURNS (n INTEGER) AS { \
         CALL to_paris() YIELD n RETURN n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE via_nameless() RETURNS (n INTEGER) AS { \
         CALL nameless() YIELD n RETURN n }",
    )
    .unwrap();

    db.execute("CALL in_berlin() YIELD n RETURN n").unwrap();
    db.execute("CALL via_paris() YIELD n RETURN n").unwrap();
    let called = db
        .execute("CALL via_nameless() YIELD n RETURN n")
        .expect_err("the inner body breaks the node type")
        .to_string();
    assert!(
        called.contains("missing required property 'name' on :Person"),
        "{called}"
    );

    assert_eq!(
        cities(&db, "MATCH (c:City) RETURN c.name"),
        Vec::<String>::new(),
        "the default graph holds no city"
    );
    assert_eq!(
        cities(&db, "USE GRAPH berlin MATCH (c:City) RETURN c.name"),
        vec!["Inner"]
    );
    assert_eq!(
        cities(&db, "USE GRAPH paris MATCH (c:City) RETURN c.name"),
        vec!["Paris"]
    );
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(0)
    );
}

/// Creates the graphs `berlin` and `paris` and two procedures: `to_paris`,
/// whose body writes `paris` by its `USE`, and `add_city`, whose body has
/// no `USE` and writes the graph of the statement that calls it.
fn berlin_and_paris(db: &GrafeoDB) {
    for graph in ["berlin", "paris"] {
        db.execute(&format!("CREATE GRAPH {graph}")).unwrap();
    }
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE add_city() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Here'}) RETURN 1 AS n }",
    )
    .unwrap();
}

/// A call whose body writes another graph by its `USE` needs a write grant
/// for that graph only, as the same statement run directly does; a body
/// without a `USE` writes the caller's graph and needs one for it.
#[test]
fn a_body_that_writes_another_graph_needs_no_write_grant_for_the_callers() {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let db = GrafeoDB::new_in_memory();
    berlin_and_paris(&db);
    db.execute(
        "CREATE PROCEDURE via_paris() RETURNS (n INTEGER) AS { \
         CALL to_paris() YIELD n RETURN n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE via_add_city() RETURNS (n INTEGER) AS { \
         CALL add_city() YIELD n RETURN n }",
    )
    .unwrap();

    let gus = db.session_with_identity(Identity::new("gus", [Role::ReadWrite]).with_grants([
        Grant::new("berlin", Role::ReadOnly),
        Grant::new("paris", Role::ReadWrite),
    ]));
    gus.execute("USE GRAPH berlin").unwrap();
    gus.execute("USE GRAPH paris INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect("gus may write paris");
    gus.execute("CALL to_paris() YIELD n RETURN n")
        .expect("the body writes paris only");
    gus.execute("CALL via_paris() YIELD n RETURN n")
        .expect("the nested body writes paris only");

    let direct = gus
        .execute("INSERT (:City {name: 'Direct'}) RETURN 1 AS n")
        .expect_err("gus may only read berlin")
        .to_string();
    assert!(
        direct.contains("no write grant for graph 'berlin'"),
        "{direct}"
    );
    for call in [
        "CALL add_city() YIELD n RETURN n",
        "CALL via_add_city() YIELD n RETURN n",
    ] {
        let called = gus
            .execute(call)
            .expect_err("the body writes berlin")
            .to_string();
        assert_eq!(called, direct, "{call}");
    }

    assert_eq!(
        cities(&db, "USE GRAPH paris MATCH (c:City) RETURN c.name"),
        vec!["Called", "Called", "Direct"]
    );
    assert_eq!(
        single(&db, "USE GRAPH berlin MATCH (c:City) RETURN count(c)"),
        Value::Int64(0)
    );
}

/// A statement focused on a graph by its `USE` calls a body without one
/// on that graph.
#[test]
fn a_call_after_a_use_prefix_runs_on_that_graph() {
    let db = GrafeoDB::new_in_memory();
    berlin_and_paris(&db);
    db.execute("USE GRAPH berlin CALL add_city() YIELD n RETURN n")
        .unwrap();

    assert_eq!(
        cities(&db, "USE GRAPH berlin MATCH (c:City) RETURN c.name"),
        vec!["Here"]
    );
    assert_eq!(
        single(&db, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(0),
        "the session's graph stays as it was"
    );
}

/// In a session schema, the graph a body's `USE` names is the schema's,
/// as for the same statement run directly.
#[test]
fn a_body_focused_on_a_graph_in_the_session_schema_writes_that_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH paris").unwrap();
    db.execute(
        "CREATE PROCEDURE to_paris() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Called'}) RETURN 1 AS n }",
    )
    .unwrap();
    let travel = db.session();
    travel.execute("CREATE SCHEMA travel").unwrap();
    travel.execute("SESSION SET SCHEMA travel").unwrap();
    travel.execute("CREATE GRAPH paris").unwrap();

    travel.execute("CALL to_paris() YIELD n RETURN n").unwrap();

    assert_eq!(
        single_in(&travel, "USE GRAPH paris MATCH (c:City) RETURN count(c)"),
        Value::Int64(1),
        "travel's paris holds the city"
    );
    assert_eq!(
        single(&db, "USE GRAPH paris MATCH (c:City) RETURN count(c)"),
        Value::Int64(0),
        "the paris of the default schema holds none"
    );
}

/// In an explicit transaction, a statement that fails after it wrote its
/// own graph and the body it calls wrote another undoes both writes, and
/// the transaction goes on.
#[test]
fn a_failed_statement_undoes_the_write_of_its_body_in_another_graph() {
    let db = GrafeoDB::new_in_memory();
    berlin_and_paris(&db);
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    // Writes paris, then breaks the node type.
    db.execute(
        "CREATE PROCEDURE paris_then_fail() RETURNS (n INTEGER) AS { \
         USE GRAPH paris INSERT (:City {name: 'Doomed'}) \
         WITH 1 AS one INSERT (:Person {city: 'Paris'}) RETURN 1 AS n }",
    )
    .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CALL to_paris() YIELD n RETURN n").unwrap();
    let error = session
        .execute(
            "INSERT (:City {name: 'Doomed'}) \
             WITH 1 AS one CALL paris_then_fail() YIELD n RETURN n",
        )
        .expect_err("the body breaks the node type after its write")
        .to_string();
    assert!(
        error.contains("missing required property 'name' on :Person"),
        "{error}"
    );
    assert_eq!(
        single_in(&session, "USE GRAPH paris MATCH (c:City) RETURN count(c)"),
        Value::Int64(1),
        "the failed statement's call left nothing in paris"
    );
    assert_eq!(
        single_in(&session, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(0),
        "nor in the session's graph"
    );
    session
        .execute("INSERT (:City {name: 'After'})")
        .expect("the transaction goes on");
    session.commit().unwrap();

    assert_eq!(
        cities(&db, "USE GRAPH paris MATCH (c:City) RETURN c.name"),
        vec!["Called"]
    );
    assert_eq!(cities(&db, "MATCH (c:City) RETURN c.name"), vec!["After"]);
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(0)
    );
}

/// A UNIQUE constraint holds in one transaction across a direct insert and
/// the insert of a body.
#[test]
fn a_unique_constraint_holds_across_a_direct_and_a_body_insert() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT FOR (p:Person) ON (p.email) UNIQUE")
        .unwrap();
    db.execute(
        "CREATE PROCEDURE add_gus() RETURNS (n INTEGER) AS { \
         INSERT (:Person {name: 'Gus', email: 'mia@example.org'}) RETURN 1 AS n }",
    )
    .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Mia', email: 'mia@example.org'})")
        .unwrap();
    let error = session
        .execute("CALL add_gus() YIELD n RETURN n")
        .expect_err("the body's email is taken in the transaction")
        .to_string();
    assert!(error.contains("UNIQUE constraint violation"), "{error}");
    session.commit().unwrap();

    let names: Vec<Value> = db
        .execute("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    assert_eq!(names, vec![Value::from("Mia")]);
}

/// A procedure that calls itself, or two that call each other, fail with
/// an error that names the depth, and write nothing.
#[test]
fn a_procedure_that_calls_itself_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE again() RETURNS (n INTEGER) AS { \
         CALL again() YIELD n RETURN n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE ping() RETURNS (n INTEGER) AS { \
         INSERT (:City {name: 'Ping'}) WITH 1 AS one CALL pong() YIELD n RETURN n }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE pong() RETURNS (n INTEGER) AS { \
         CALL ping() YIELD n RETURN n }",
    )
    .unwrap();

    for call in [
        "CALL again() YIELD n RETURN n",
        "CALL ping() YIELD n RETURN n",
    ] {
        let error = db
            .execute(call)
            .expect_err("the calls never end")
            .to_string();
        assert!(error.contains("more than 16 calls deep"), "{call}: {error}");
    }
    assert_eq!(
        single(&db, "MATCH (c:City) RETURN count(c)"),
        Value::Int64(0),
        "the failed call's inserts are undone"
    );
}

/// A call outside a transaction whose body writes another graph by its
/// `USE`, beside a write of the statement itself, commits once: every
/// change is at one epoch, each naming its graph.
#[cfg(feature = "cdc")]
#[test]
fn an_auto_commit_call_with_a_focused_body_commits_once() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    berlin_and_paris(&db);
    let before = db.current_epoch();
    db.execute("INSERT (:City {name: 'Here'}) WITH 1 AS one CALL to_paris() YIELD n RETURN n")
        .unwrap();
    let after = db.current_epoch();
    assert_eq!(after.as_u64(), before.as_u64() + 1, "one commit");

    let events = db
        .changes_between(EpochId::new(before.as_u64() + 1), after)
        .unwrap();
    let mut graphs: Vec<Option<String>> = events.iter().map(|e| e.graph.clone()).collect();
    graphs.sort();
    assert_eq!(graphs, vec![None, Some("paris".to_string())], "{events:?}");
    assert!(
        events.iter().all(|event| event.epoch == after),
        "{events:?}"
    );
}

/// One transaction writes two graphs, its own directly and through a body
/// without a `USE`, and another through a body's `USE`: its commit reports
/// each change under its graph.
#[cfg(feature = "cdc")]
#[test]
fn a_transaction_that_writes_two_graphs_reports_each_under_its_graph() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    berlin_and_paris(&db);
    let before = db.current_epoch();
    let mut session = db.session();
    session.use_graph("berlin");
    session.begin_transaction().unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session.execute("CALL to_paris() YIELD n RETURN n").unwrap();
    session.execute("CALL add_city() YIELD n RETURN n").unwrap();
    session.commit().unwrap();

    let events = db
        .changes_between(EpochId::new(before.as_u64() + 1), db.current_epoch())
        .unwrap();
    let mut graphs: Vec<Option<String>> = events.iter().map(|e| e.graph.clone()).collect();
    graphs.sort();
    assert_eq!(
        graphs,
        vec![
            Some("berlin".to_string()),
            Some("berlin".to_string()),
            Some("paris".to_string())
        ],
        "{events:?}"
    );
}

/// A transaction that writes two graphs, one through a body's `USE`,
/// survives a crash: the reopen replays both from the WAL. The writes run
/// in a child process that exits without `close()`.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_engine::GrafeoDB;

    /// Tells the child process where its database is.
    const PATH_VAR: &str = "GRAFEO_STORED_PROCEDURE_BODIES_CRASH_PATH";

    /// The writes: graphs, procedures, and one transaction over two graphs.
    fn writes(db: &GrafeoDB) {
        super::berlin_and_paris(db);
        let mut session = db.session();
        session.use_graph("berlin");
        session.begin_transaction().unwrap();
        session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
        session.execute("CALL to_paris() YIELD n RETURN n").unwrap();
        session.commit().unwrap();
    }

    /// Child-process entry: runs the writes on a new database and exits
    /// without `close()`; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        writes(&db);
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    #[test]
    fn a_transaction_over_two_graphs_survives_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paris.grafeo");
        let status =
            child_process::run(grafeo_common::child_test!("crash_child").env(PATH_VAR, &path))
                .unwrap();
        assert!(status.success(), "the child process failed");
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        assert!(
            PathBuf::from(sidecar).exists(),
            "the crash left the WAL for the reopen to replay"
        );

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            super::cities(&db, "USE GRAPH berlin MATCH (c:City) RETURN c.name"),
            vec!["Berlin"]
        );
        assert_eq!(
            super::cities(&db, "USE GRAPH paris MATCH (c:City) RETURN c.name"),
            vec!["Called"]
        );
        db.close().unwrap();
    }
}
