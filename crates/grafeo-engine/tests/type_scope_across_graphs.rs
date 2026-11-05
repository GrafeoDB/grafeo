//! A graph type's rules apply only in the graphs bound to it (#730).
//!
//! Tests first, from the issue: after a graph type and a graph typed by it, writes with the same
//! label in other graphs were changed or refused by that type's rules. Other graphs now store what
//! is written, the typed graph still applies them, an empty closed graph type allows no nodes,
//! `CREATE PROPERTY GRAPH` behaves the same, and rows already stored are not changed.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main type_scope_across_graphs::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const WITH_DEFAULTS: &str = "CREATE GRAPH TYPE t (NODE TYPE Person (name STRING NOT NULL DEFAULT 'Mia', age FLOAT64, city STRING DEFAULT 'Berlin'))";
const REQUIRED: &str = "CREATE GRAPH TYPE t (NODE TYPE Person (name STRING NOT NULL, age INT64))";

fn typed_with(db: &GrafeoDB, graph_type: &str) -> Session {
    let session = db.session();
    session.execute(graph_type).unwrap();
    session.execute("CREATE GRAPH typed TYPED t").unwrap();
    session.execute("CREATE GRAPH other").unwrap();
    session
}

fn rows(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session.execute(query).unwrap().rows().to_vec()
}

const PEOPLE: &str = "MATCH (p:Person) RETURN p.age, p.name, p.city";

fn string(text: &str) -> Value {
    Value::from(text)
}

/// The issue's example: in the default graph and in an untyped one, what is written is stored.
#[test]
fn other_graphs_store_what_is_written() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, WITH_DEFAULTS);
    for graph in ["default", "other"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        session.execute("INSERT (:Person {age: 2})").unwrap();
        assert_eq!(
            rows(&session, PEOPLE),
            [[Value::Int64(2), Value::Null, Value::Null]],
            "{graph}: no default, no cast"
        );
    }
}

/// The typed graph still applies them: the defaults, the cast and the rule.
#[test]
fn the_typed_graph_still_applies_them() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, WITH_DEFAULTS);
    session.execute("USE GRAPH typed").unwrap();
    session.execute("INSERT (:Person {age: 2})").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Float64(2.0), string("Mia"), string("Berlin")]]
    );
    let error = session
        .execute("INSERT (:Person {name: 7})")
        .expect_err("a name of another type");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");

    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    session.execute("USE GRAPH typed").unwrap();
    let error = session
        .execute("INSERT (:Person {age: 6})")
        .expect_err("no required name");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
    session
        .execute("INSERT (:Person {name: 'Gus', age: 6})")
        .unwrap();
    assert_eq!(rows(&session, PEOPLE).len(), 1);
}

/// What a type would refuse passes elsewhere: a missing property, a value of another type, a
/// `SET` and a `REMOVE`, in GQL and in Cypher.
#[test]
fn what_a_type_refuses_passes_elsewhere() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    for graph in ["default", "other"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        for (language, statement) in [
            ("GQL", "INSERT (:Person {age: 2.5})"),
            ("GQL", "INSERT (:Person {name: 7, age: 3})"),
            ("GQL", "MATCH (p:Person {age: 3}) SET p.age = 'text'"),
            ("GQL", "MATCH (p:Person) WHERE p.name = 7 REMOVE p.name"),
            ("Cypher", "CREATE (:Person {age: 9.5})"),
            ("Cypher", "MERGE (:Person {name: 11})"),
        ] {
            let outcome = if language == "GQL" {
                session.execute(statement)
            } else {
                session.execute_cypher(statement)
            };
            outcome.unwrap_or_else(|error| panic!("{graph}, {language}: {statement}: {error}"));
        }
    }
}

/// Edge types the same way.
#[test]
fn an_edge_types_rules_stay_in_its_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE GRAPH TYPE t (NODE TYPE Stop (name STRING), \
             EDGE TYPE ROAD (km INTEGER NOT NULL, toll FLOAT64 DEFAULT 1.5))",
        )
        .unwrap();
    session.execute("CREATE GRAPH typed TYPED t").unwrap();
    session
        .execute("INSERT (:Stop)-[:ROAD {km: 'far'}]->(:Stop)")
        .unwrap_or_else(|error| panic!("the default graph: {error}"));
    assert_eq!(
        rows(&session, "MATCH ()-[r:ROAD]->() RETURN r.km, r.toll"),
        [[string("far"), Value::Null]]
    );
    session.execute("USE GRAPH typed").unwrap();
    assert!(
        session
            .execute("INSERT (:Stop)-[:ROAD {km: 'far'}]->(:Stop)")
            .is_err(),
        "the typed graph"
    );
    session
        .execute("INSERT (:Stop)-[:ROAD {km: 3}]->(:Stop)")
        .unwrap();
    assert_eq!(
        rows(&session, "MATCH ()-[r:ROAD]->() RETURN r.km, r.toll"),
        [[Value::Int64(3), Value::Float64(1.5)]]
    );
}

/// The direct API and a transaction take the same scope as a statement.
#[test]
fn the_direct_api_and_a_transaction_take_the_same_scope() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    db.create_node_with_props(&["Person"], [("age", Value::Float64(2.5))])
        .unwrap_or_else(|error| panic!("the direct API: {error}"));
    assert_eq!(
        rows(&session, "MATCH (p:Person) RETURN p.age"),
        [[Value::Float64(2.5)]]
    );

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Person {age: 'in a transaction'})")
        .unwrap();
    writer.execute("USE GRAPH typed").unwrap();
    assert!(
        writer.execute("INSERT (:Person {age: 6})").is_err(),
        "the typed graph, in the same transaction"
    );
    writer.execute("USE GRAPH default").unwrap();
    writer.commit().unwrap();
    assert_eq!(rows(&session, "MATCH (p:Person) RETURN p.age").len(), 2);
}

/// Rows already stored are not changed: not by the type, and not by binding a graph to it.
#[test]
fn rows_already_stored_are_not_changed() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {age: 1})").unwrap();
    session.execute("CREATE GRAPH other").unwrap();
    session.execute("USE GRAPH other").unwrap();
    session.execute("INSERT (:Person {age: 'one'})").unwrap();
    session.execute("USE GRAPH default").unwrap();
    session.execute(WITH_DEFAULTS).unwrap();
    session.execute("CREATE GRAPH typed TYPED t").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Int64(1), Value::Null, Value::Null]]
    );
    session.execute("USE GRAPH other").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[string("one"), Value::Null, Value::Null]]
    );
    session
        .execute("MATCH (p:Person) SET p.seen = true")
        .unwrap_or_else(|error| panic!("a write beside the rule: {error}"));
    assert_eq!(
        rows(&session, "MATCH (p:Person) RETURN p.age, p.name, p.seen"),
        [[string("one"), Value::Null, Value::Bool(true)]]
    );
}

/// It holds after a reopen, on both sides.
#[test]
fn the_scope_holds_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scope.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = typed_with(&db, WITH_DEFAULTS);
    session.execute("INSERT (:Person {age: 2})").unwrap();
    drop(session);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Int64(2), Value::Null, Value::Null]]
    );
    session.execute("INSERT (:Person {age: 3})").unwrap();
    assert_eq!(
        rows(
            &session,
            "MATCH (p:Person) WHERE p.name IS NOT NULL RETURN p.age"
        )
        .len(),
        0
    );
    session.execute("USE GRAPH typed").unwrap();
    session.execute("INSERT (:Person {age: 2})").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Float64(2.0), string("Mia"), string("Berlin")]]
    );
    drop(session);
    db.close().unwrap();
}

const CRASH_PATH_VAR: &str = "GRAFEO_TYPE_SCOPE_PATH";

/// Child-process entry: writes on both sides, then the process dies with no close.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(CRASH_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let session = typed_with(&db, WITH_DEFAULTS);
    session.execute("INSERT (:Person {age: 2})").unwrap();
    session.execute("USE GRAPH typed").unwrap();
    session.execute("INSERT (:Person {age: 2})").unwrap();
    std::process::exit(0);
}

/// Replay reads the writes as they ran: no default or cast in the graphs with no type.
#[test]
fn replay_keeps_each_graphs_rows_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scope-crash.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("crash_child").env(CRASH_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Int64(2), Value::Null, Value::Null]]
    );
    session.execute("USE GRAPH typed").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Float64(2.0), string("Mia"), string("Berlin")]]
    );
    drop(session);
    db.close().unwrap();
}

/// A closed graph type with nothing in it allows no nodes, of any label or none: the empty list
/// is no open type. Other graphs are not touched by it, and an open type allows what it likes.
#[test]
fn an_empty_closed_graph_type_allows_no_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE GRAPH TYPE nothing {node_types: [], edge_types: []}")
        .unwrap();
    session
        .execute("CREATE GRAPH closed TYPED nothing")
        .unwrap();
    session.execute("USE GRAPH closed").unwrap();
    for statement in ["INSERT (:Person {name: 'Gus'})", "INSERT ({name: 'Gus'})"] {
        let error = session
            .execute(statement)
            .expect_err(&format!("{statement}: accepted"));
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            "G2000",
            "{statement}: {error}"
        );
    }
    assert_eq!(rows(&session, "MATCH (n) RETURN n.name").len(), 0);
    session.execute("USE GRAPH default").unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .unwrap_or_else(|error| panic!("the default graph: {error}"));

    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE GRAPH TYPE anything {node_types: [], edge_types: [], open: true}")
        .unwrap();
    session.execute("CREATE GRAPH free TYPED anything").unwrap();
    session.execute("USE GRAPH free").unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus'})-[:KNOWS]->({name: 'Mia'})")
        .unwrap_or_else(|error| panic!("an open type: {error}"));
}

/// A graph type's closed list stays in its graph too: a label it does not name is free elsewhere.
#[test]
fn a_closed_types_list_stays_in_its_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    session
        .execute("INSERT (:Robot {serial: 1})")
        .unwrap_or_else(|error| panic!("the default graph: {error}"));
    session.execute("USE GRAPH typed").unwrap();
    let error = session
        .execute("INSERT (:Robot {serial: 1})")
        .expect_err("a label the type does not name");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
}

#[cfg(feature = "sql-pgq")]
mod property_graphs {
    use super::*;

    const SOCIAL: &str = "CREATE PROPERTY GRAPH Social NODE TABLES (Person (id BIGINT PRIMARY KEY, name VARCHAR(40)))";

    /// A property graph's types check its own writes, and nobody else's.
    #[test]
    fn a_property_graphs_types_stay_in_it() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {age: 1})").unwrap();
        session.execute_sql(SOCIAL).unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus'})")
            .unwrap_or_else(|error| panic!("the default graph: {error}"));
        session
            .execute("INSERT (:Person {name: 7})")
            .unwrap_or_else(|error| panic!("the default graph, another type: {error}"));
        session.execute("CREATE GRAPH other").unwrap();
        session.execute("USE GRAPH other").unwrap();
        session
            .execute("INSERT (:Person {name: 'Mia'})")
            .unwrap_or_else(|error| panic!("an untyped graph: {error}"));
        session.execute("USE GRAPH Social").unwrap();
        assert!(
            session.execute("INSERT (:Person {name: 'Hans'})").is_err(),
            "its own write, with no id"
        );
        session
            .execute("INSERT (:Person {id: 1, name: 'Hans'})")
            .unwrap_or_else(|error| panic!("its own write: {error}"));
        assert_eq!(rows(&session, "MATCH (p:Person) RETURN p.id").len(), 1);
    }

    /// Rows that were there first stay as they were.
    #[test]
    fn a_property_graph_leaves_stored_rows_alone() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 7})").unwrap();
        session
            .execute_sql(SOCIAL)
            .unwrap_or_else(|error| panic!("another graph's rows: {error}"));
        assert_eq!(
            rows(&session, "MATCH (p:Person) RETURN p.name"),
            [[Value::Int64(7)]]
        );
    }
}

// ---------------------------------------------------------------------------
// A type of the database that a graph type names, and dropping
// ---------------------------------------------------------------------------

/// A standalone node type is a rule of the database until a graph type names it: from then on it
/// applies in the graphs bound to that graph type, not everywhere.
#[test]
fn a_standalone_type_named_by_a_graph_type_stops_applying_everywhere() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "no graph type names it yet: it applies"
    );
    session
        .execute("CREATE GRAPH TYPE t (NODE TYPE Person)")
        .unwrap();
    session.execute("CREATE GRAPH typed TYPED t").unwrap();
    session
        .execute("INSERT (:Person {age: 1})")
        .unwrap_or_else(|error| panic!("the default graph: {error}"));
    session.execute("USE GRAPH typed").unwrap();
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "the typed graph"
    );
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
}

/// A type no graph type names keeps working as it does: the control for the test above.
#[test]
fn a_standalone_type_no_graph_type_names_still_applies() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .unwrap();
    session
        .execute("CREATE GRAPH TYPE t (NODE TYPE Person (name STRING))")
        .unwrap();
    session.execute("CREATE GRAPH typed TYPED t").unwrap();
    assert!(session.execute("INSERT (:City {size: 3})").is_err());
}

/// A graph type a graph is bound to cannot be dropped (ISO/IEC 39075 12.7, Syntax Rule 6), and
/// everything stays. What becomes of its types once no graph type lists them waits on a decision.
#[test]
fn a_bound_graph_type_cannot_be_dropped() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, WITH_DEFAULTS);
    session.execute("INSERT (:Person {age: 2})").unwrap();
    assert!(
        session.execute("DROP GRAPH TYPE t").is_err(),
        "a graph is bound to it"
    );
    session.execute("USE GRAPH typed").unwrap();
    session.execute("INSERT (:Person {age: 2})").unwrap();
    assert_eq!(
        rows(&session, PEOPLE),
        [[Value::Float64(2.0), string("Mia"), string("Berlin")]]
    );
}

/// Two schemas, one type name: a listing is of the type in its own schema. In the schema whose
/// type a graph type lists it stops applying, and the other schema's stays a rule by label.
#[test]
fn a_listing_is_of_the_type_in_its_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE SCHEMA s1",
        "CREATE SCHEMA s2",
        "SESSION SET SCHEMA s1",
        "CREATE NODE TYPE Customer (name STRING NOT NULL)",
        "CREATE GRAPH TYPE shop (NODE TYPE Customer)",
        "SESSION SET SCHEMA s2",
        "CREATE NODE TYPE Customer (name STRING NOT NULL)",
    ] {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    assert!(
        session.execute("INSERT (:Customer {age: 1})").is_err(),
        "s2's type: no graph type lists it"
    );
    session.execute("SESSION SET SCHEMA s1").unwrap();
    session
        .execute("INSERT (:Customer {age: 1})")
        .unwrap_or_else(|error| panic!("s1's type is listed: {error}"));
}

// ---------------------------------------------------------------------------
// Open types, two graph types, named constraints, listing changes and #729 by scope
// ---------------------------------------------------------------------------

fn standalone_person(session: &Session) {
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
}

/// An open bound type applies the types it lists and lets other labels through; a graph bound to
/// it is one that applies them, and other graphs are not.
#[test]
fn an_open_bound_type_applies_what_it_lists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    standalone_person(&session);
    session
        .execute("CREATE GRAPH TYPE o {node_types: [Person], edge_types: [], open: true}")
        .unwrap();
    session.execute("CREATE GRAPH free TYPED o").unwrap();
    session.execute("USE GRAPH free").unwrap();
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "a listed type"
    );
    session
        .execute("INSERT (:Robot {serial: 1})")
        .unwrap_or_else(|error| panic!("another label, an open type: {error}"));
    session.execute("USE GRAPH default").unwrap();
    session
        .execute("INSERT (:Person {age: 1})")
        .unwrap_or_else(|error| panic!("the default graph: {error}"));
}

/// A type two graph types list applies in the graphs of each, and elsewhere in neither.
#[test]
fn a_type_two_graph_types_list_applies_in_both() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    standalone_person(&session);
    for statement in [
        "CREATE GRAPH TYPE first_type (NODE TYPE Person)",
        "CREATE GRAPH TYPE second_type (NODE TYPE Person)",
        "CREATE GRAPH one TYPED first_type",
        "CREATE GRAPH two TYPED second_type",
        "CREATE GRAPH neither",
    ] {
        session.execute(statement).unwrap();
    }
    for graph in ["one", "two"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        assert!(
            session.execute("INSERT (:Person {age: 1})").is_err(),
            "{graph}"
        );
    }
    for graph in ["neither", "default"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        session
            .execute("INSERT (:Person {age: 1})")
            .unwrap_or_else(|error| panic!("{graph}: {error}"));
    }
}

/// A named constraint on a listed label follows the type: it holds in the bound graph, and is no
/// rule in the others.
#[test]
fn a_named_constraint_on_a_listed_label_follows_the_type() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    session
        .execute("CREATE CONSTRAINT one_name FOR (p:Person) ON (p.name) UNIQUE")
        .unwrap();
    for _ in 0..2 {
        session
            .execute("INSERT (:Person {name: 'Gus', age: 1})")
            .unwrap_or_else(|error| panic!("the default graph: {error}"));
    }
    session.execute("USE GRAPH typed").unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 1})")
        .unwrap();
    let error = session
        .execute("INSERT (:Person {name: 'Gus', age: 2})")
        .expect_err("the constraint, in the bound graph");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
}

/// A constraint on a label no graph type lists stays a rule by label, in every graph.
#[test]
fn a_named_constraint_on_an_unlisted_label_stays_everywhere() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    session
        .execute("CREATE CONSTRAINT one_id FOR (f:File) ON (f.id) UNIQUE")
        .unwrap();
    for graph in ["default", "other"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        session.execute("INSERT (:File {id: 'a'})").unwrap();
        assert!(
            session.execute("INSERT (:File {id: 'a'})").is_err(),
            "{graph}"
        );
    }
}

/// A standalone type a new graph type lists stops applying elsewhere at once, with no graph bound
/// yet, and the statement checks nothing as it removes a rule.
#[test]
fn a_new_listing_removes_the_rule_at_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    standalone_person(&session);
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "no graph type yet"
    );
    session
        .execute("CREATE GRAPH TYPE t (NODE TYPE Person)")
        .unwrap();
    session
        .execute("INSERT (:Person {age: 1})")
        .unwrap_or_else(|error| panic!("listed, and no graph bound: {error}"));
}

/// `ALTER GRAPH TYPE ... ADD NODE TYPE` lists a type at once: elsewhere it stops, and in the bound
/// graph it is checked whole under the new type.
#[test]
fn adding_a_type_to_a_graph_type_moves_its_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "CREATE NODE TYPE Town (name STRING)",
        "CREATE GRAPH TYPE atlas (NODE TYPE Town)",
        "CREATE GRAPH maps TYPED atlas",
    ] {
        session.execute(statement).unwrap();
    }
    assert!(
        session.execute("INSERT (:City {size: 3})").is_err(),
        "not listed yet"
    );
    session
        .execute("ALTER GRAPH TYPE atlas ADD NODE TYPE City")
        .unwrap_or_else(|error| panic!("{error}"));
    session
        .execute("INSERT (:City {size: 3})")
        .unwrap_or_else(|error| panic!("listed now, the default graph: {error}"));
    session.execute("USE GRAPH maps").unwrap();
    assert!(
        session.execute("INSERT (:City {size: 3})").is_err(),
        "the bound graph"
    );
    session.execute("INSERT (:City {name: 'Paris'})").unwrap();
}

/// A closed type emptied over a graph with nothing on that side refuses the next write.
#[test]
fn an_emptied_closed_type_refuses_the_next_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE EDGE TYPE ROAD (km INTEGER)",
        "CREATE GRAPH TYPE atlas (NODE TYPE City, EDGE TYPE ROAD)",
        "CREATE GRAPH maps TYPED atlas",
    ] {
        session.execute(statement).unwrap();
    }
    session
        .execute("ALTER GRAPH TYPE atlas DROP NODE TYPE City")
        .unwrap();
    session.execute("USE GRAPH maps").unwrap();
    let error = session
        .execute("INSERT (:City {name: 'Paris'})")
        .expect_err("no node type left");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
}

/// The last edge type gone, a closed type takes nodes and no edges.
#[test]
fn a_closed_type_with_no_edge_types_refuses_edges() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE GRAPH TYPE atlas (NODE TYPE City)",
        "CREATE GRAPH maps TYPED atlas",
        "USE GRAPH maps",
    ] {
        session.execute(statement).unwrap();
    }
    session.execute("INSERT (:City {name: 'Paris'})").unwrap();
    let error = session
        .execute("INSERT (:City {name: 'Lyon'})-[:ROAD]->(:City {name: 'Nice'})")
        .expect_err("no edge type listed");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
}

/// #729 by scope: a listed type's check reads the graphs bound to it, and another graph's rows
/// neither stop it nor change.
#[test]
fn a_listed_types_check_reads_its_own_graphs_only() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE GRAPH TYPE atlas (NODE TYPE City)",
        "CREATE GRAPH maps TYPED atlas",
        "INSERT (:City {size: 3})",
    ] {
        session.execute(statement).unwrap();
    }
    // The default graph's city has no name and no `zone`, and does not matter.
    session
        .execute("ALTER NODE TYPE City ADD PROPERTY zone STRING NOT NULL DEFAULT 'A'")
        .unwrap_or_else(|error| panic!("only the default graph's row breaks it: {error}"));
    assert_eq!(
        rows(&session, "MATCH (c:City) RETURN c.size, c.zone"),
        [[Value::Int64(3), Value::Null]],
        "the row is not touched"
    );
    // A row in the bound graph without it does matter.
    session.execute("USE GRAPH maps").unwrap();
    session
        .execute("INSERT (:City {name: 'Paris', zone: 'B'})")
        .unwrap();
    session.execute("USE GRAPH default").unwrap();
    let error = session
        .execute("ALTER NODE TYPE City ADD PROPERTY code STRING NOT NULL")
        .expect_err("the bound graph's city has no code");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
}

/// A type no graph type lists keeps its check over every graph, as #729 has it.
#[test]
fn an_unlisted_types_check_reads_every_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE GRAPH other",
        "USE GRAPH other",
        "INSERT (:City {size: 3})",
        "USE GRAPH default",
    ] {
        session.execute(statement).unwrap();
    }
    let error = session
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .expect_err("a rule by label, and another graph breaks it");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
}

/// A named constraint on a listed label reads the bound graph only.
#[test]
fn a_constraint_on_a_listed_label_checks_the_bound_graph_only() {
    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    for _ in 0..2 {
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    }
    session
        .execute("CREATE CONSTRAINT one_name FOR (p:Person) ON (p.name) UNIQUE")
        .unwrap_or_else(|error| panic!("the duplicates are in another graph: {error}"));

    let db = GrafeoDB::new_in_memory();
    let session = typed_with(&db, REQUIRED);
    session.execute("USE GRAPH typed").unwrap();
    for _ in 0..2 {
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    }
    let error = session
        .execute("CREATE CONSTRAINT one_name FOR (p:Person) ON (p.name) UNIQUE")
        .expect_err("the duplicates are in the bound graph");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
}

/// A rule of each kind after a reopen: a listed type's stays in its graph, an unlisted type's
/// stays everywhere, and a standalone type once listed stays listed.
#[test]
fn both_kinds_of_type_hold_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kinds.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "CREATE NODE TYPE Person (name STRING NOT NULL)",
        "CREATE GRAPH TYPE t (NODE TYPE Person)",
        "CREATE GRAPH typed TYPED t",
    ] {
        session.execute(statement).unwrap();
    }
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session
        .execute("INSERT (:Person {age: 1})")
        .unwrap_or_else(|error| panic!("a listed type, elsewhere: {error}"));
    assert!(
        session.execute("INSERT (:City {size: 1})").is_err(),
        "an unlisted type"
    );
    session.execute("USE GRAPH typed").unwrap();
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "a listed type, at home"
    );
    assert!(
        session.execute("INSERT (:City {size: 1})").is_err(),
        "not in the closed list"
    );
    drop(session);
    db.close().unwrap();
}

/// The same in a crash, with the writes of each kind replayed.
const KINDS_PATH_VAR: &str = "GRAFEO_TYPE_SCOPE_KINDS_PATH";

#[test]
fn kinds_crash_child() {
    let Some(path) = std::env::var_os(KINDS_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "CREATE NODE TYPE Person (name STRING DEFAULT 'Mia', age INT64)",
        "CREATE GRAPH TYPE t (NODE TYPE Person)",
        "CREATE GRAPH typed TYPED t",
        "INSERT (:Person {age: 1})",
        "USE GRAPH typed",
        "INSERT (:Person {age: 2})",
    ] {
        session.execute(statement).unwrap();
    }
    std::process::exit(0);
}

#[test]
fn both_kinds_of_type_hold_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kinds-crash.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("kinds_crash_child").env(KINDS_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(
        rows(&session, "MATCH (p:Person) RETURN p.age, p.name"),
        [[Value::Int64(1), Value::Null]]
    );
    assert!(
        session.execute("INSERT (:City {size: 1})").is_err(),
        "an unlisted type"
    );
    session.execute("USE GRAPH typed").unwrap();
    assert_eq!(
        rows(&session, "MATCH (p:Person) RETURN p.age, p.name"),
        [[Value::Int64(2), string("Mia")]]
    );
    drop(session);
    db.close().unwrap();
}

/// Cypher's `ALTER CURRENT GRAPH TYPE` declares types for the session's graph, not through a graph
/// that is bound to them, so they stay rules by label: in the default graph and in a named one,
/// and a graph bound to its own type answers from that type instead.
#[test]
fn a_cypher_current_graph_type_stays_a_rule_by_label() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Kept {name STRING}) }")
        .unwrap();
    let session = db.session();
    for statement in [
        "CREATE GRAPH TYPE own (NODE TYPE Other (name INT64))",
        "CREATE GRAPH plain",
        "CREATE GRAPH bound TYPED own",
    ] {
        session.execute(statement).unwrap();
    }
    for graph in ["default", "plain"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        assert!(
            session.execute("INSERT (:Kept {name: 7})").is_err(),
            "{graph}: a value of another type"
        );
        session
            .execute("INSERT (:Kept {name: 'Gus'})")
            .unwrap_or_else(|error| panic!("{graph}: {error}"));
    }
    session.execute("USE GRAPH bound").unwrap();
    assert!(
        session.execute("INSERT (:Kept {name: 'Gus'})").is_err(),
        "a label its own closed type does not list"
    );
    session.execute("INSERT (:Other {name: 7})").unwrap();
}

// ---------------------------------------------------------------------------
// A drop takes the types only it listed
// ---------------------------------------------------------------------------

const ROUTES: &str =
    "CREATE GRAPH TYPE routes { (:City {name STRING NOT NULL})-[:ROUTE {km INT64}]->(:City) }";

/// Whether the rows of a `SHOW` statement name `name`.
fn shows(session: &Session, statement: &str, name: &str) -> bool {
    rows(session, statement)
        .iter()
        .any(|row| row.first() == Some(&string(name)))
}

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// The drop's status message, which names what went.
fn dropped(session: &Session, statement: &str) -> String {
    session
        .execute(statement)
        .unwrap_or_else(|error| panic!("{statement}: {error}"))
        .status_message
        .unwrap_or_else(|| panic!("{statement}: no status message"))
}

/// The types a graph type declared go with it, and the status names them. Its own NOT NULL is no
/// named constraint and stops nothing. Rows are not touched, and a name that went is free again.
#[test]
fn a_drop_takes_the_types_only_it_listed_and_names_them() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[ROUTES, "INSERT (:City {population: 'many'})"]);
    assert_eq!(
        dropped(&session, "DROP GRAPH TYPE routes"),
        "Dropped graph type 'routes', and its types City, ROUTE"
    );
    assert!(!shows(&session, "SHOW GRAPH TYPES", "routes"));
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    assert!(!shows(&session, "SHOW EDGE TYPES", "ROUTE"));
    assert_eq!(
        rows(&session, "MATCH (c:City) RETURN c.population"),
        [[string("many")]],
        "the rows stay"
    );
    // No rule by label is left behind, and the name can be declared again.
    session
        .execute("INSERT (:City {population: 'few'})")
        .unwrap();
    session
        .execute("CREATE NODE TYPE City (population STRING)")
        .unwrap_or_else(|error| panic!("the name is free again: {error}"));
}

/// A standalone type it named by reference goes too, once only it listed the type: no rule by
/// label comes back.
#[test]
fn a_standalone_type_named_by_reference_goes_too() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE EDGE TYPE ROAD (km INTEGER NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, EDGE TYPE ROAD)",
        ],
    );
    let status = dropped(&session, "DROP GRAPH TYPE atlas");
    for name in ["City", "ROAD"] {
        assert!(status.contains(name), "{name} in: {status}");
    }
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    assert!(!shows(&session, "SHOW EDGE TYPES", "ROAD"));
    session
        .execute("INSERT (:City {size: 1})-[:ROAD]->(:City)")
        .unwrap_or_else(|error| panic!("no rule by label is left: {error}"));
    assert_eq!(rows(&session, "MATCH (c:City) RETURN c.size").len(), 2);
}

/// A type another graph type lists stays, with no name in the status, and its graphs keep their
/// rule. Dropping the last graph type that lists it takes it.
#[test]
fn a_type_another_graph_type_lists_stays() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE a (NODE TYPE Person)",
            "CREATE GRAPH TYPE b (NODE TYPE Person)",
            "CREATE GRAPH circle TYPED b",
        ],
    );
    assert_eq!(
        dropped(&session, "DROP GRAPH TYPE a"),
        "Dropped graph type 'a'",
        "still listed"
    );
    assert!(shows(&session, "SHOW NODE TYPES", "Person"));
    session.execute("USE GRAPH circle").unwrap();
    assert!(
        session.execute("INSERT (:Person {age: 1})").is_err(),
        "the other graph type's rule"
    );
    session.execute("USE GRAPH default").unwrap();
    session
        .execute("INSERT (:Person {age: 1})")
        .unwrap_or_else(|error| panic!("still no rule by label: {error}"));
    session.execute("DROP GRAPH circle").unwrap();
    let status = dropped(&session, "DROP GRAPH TYPE b");
    assert!(status.contains("Person"), "the last listing: {status}");
    assert!(!shows(&session, "SHOW NODE TYPES", "Person"));
}

/// A type it did not list is none of its business: it stays a rule by label.
#[test]
fn a_type_it_did_not_list_stays_a_rule_by_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE NODE TYPE Museum (name STRING NOT NULL)", ROUTES],
    );
    let status = dropped(&session, "DROP GRAPH TYPE routes");
    assert!(!status.contains("Museum"), "{status}");
    assert!(shows(&session, "SHOW NODE TYPES", "Museum"));
    assert!(session.execute("INSERT (:Museum {size: 1})").is_err());
}

/// A named constraint on a type that would go refuses the statement and names the constraint,
/// and nothing has gone. Once the constraint is dropped, so can the graph type be.
#[test]
fn a_named_constraint_on_such_a_type_refuses_the_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
            "CREATE CONSTRAINT city_names FOR (n:City) ON (n.name) UNIQUE",
        ],
    );
    let error = session
        .execute("DROP GRAPH TYPE atlas")
        .expect_err("a named constraint sits on the type");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(
        error.to_string().contains(
            "node type 'City' goes with its graph type, and constraint 'city_names' is on it: \
             drop the constraint first"
        ),
        "{error}"
    );
    assert!(shows(&session, "SHOW GRAPH TYPES", "atlas"));
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    // Still whole: a graph can be typed by it.
    run_all(
        &session,
        &["CREATE GRAPH maps TYPED atlas", "DROP GRAPH maps"],
    );
    run_all(&session, &["DROP CONSTRAINT city_names"]);
    let status = dropped(&session, "DROP GRAPH TYPE atlas");
    assert!(status.contains("City"), "{status}");
}

/// The refusal is about types that would go: a constraint on a type another graph type lists, or
/// on a label no graph type lists, stops nothing and stays.
#[test]
fn a_named_constraint_elsewhere_does_not_refuse_the_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE a (NODE TYPE Person)",
            "CREATE GRAPH TYPE b (NODE TYPE Person)",
            "CREATE CONSTRAINT person_names FOR (n:Person) ON (n.name) UNIQUE",
            "CREATE CONSTRAINT file_ids FOR (n:File) ON (n.id) UNIQUE",
        ],
    );
    session
        .execute("DROP GRAPH TYPE a")
        .unwrap_or_else(|error| panic!("another graph type lists Person: {error}"));
    let constraints = rows(&session, "SHOW CONSTRAINTS");
    assert_eq!(constraints.len(), 2, "{constraints:?}");
}

/// `IF EXISTS` does the same, and over nothing does nothing.
#[test]
fn if_exists_takes_the_same_types() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[ROUTES]);
    let status = dropped(&session, "DROP GRAPH TYPE IF EXISTS routes");
    assert!(status.contains("City"), "{status}");
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    session.execute("DROP GRAPH TYPE IF EXISTS routes").unwrap();
}

/// `ALTER GRAPH TYPE ... DROP NODE TYPE` of a type no other graph type lists is the same drop: the
/// type goes, and a named constraint on it refuses the statement.
#[test]
fn an_altered_drop_takes_a_type_only_it_listed() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Town (name STRING NOT NULL)",
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, NODE TYPE Town, NODE TYPE Person)",
            "CREATE GRAPH TYPE people (NODE TYPE Person)",
            "CREATE CONSTRAINT town_names FOR (n:Town) ON (n.name) UNIQUE",
        ],
    );
    assert_eq!(
        dropped(&session, "ALTER GRAPH TYPE atlas DROP NODE TYPE City"),
        "Altered graph type 'atlas', and its types City"
    );
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    let error = session
        .execute("ALTER GRAPH TYPE atlas DROP NODE TYPE Town")
        .expect_err("a named constraint sits on the type");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(error.to_string().contains("town_names"), "{error}");
    assert!(shows(&session, "SHOW NODE TYPES", "Town"));
    assert_eq!(
        dropped(&session, "ALTER GRAPH TYPE atlas DROP NODE TYPE Person"),
        "Altered graph type 'atlas'",
        "another graph type lists it"
    );
    assert!(shows(&session, "SHOW NODE TYPES", "Person"));
}

/// Edge types go the same way, and have no named constraints to stop them.
#[test]
fn an_altered_drop_takes_an_edge_type_only_it_listed() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE ROAD (km INTEGER NOT NULL)",
            "CREATE EDGE TYPE RAIL (km INTEGER NOT NULL)",
            "CREATE GRAPH TYPE atlas (EDGE TYPE ROAD, EDGE TYPE RAIL)",
            "CREATE GRAPH TYPE lines (EDGE TYPE RAIL)",
        ],
    );
    assert_eq!(
        dropped(&session, "ALTER GRAPH TYPE atlas DROP EDGE TYPE ROAD"),
        "Altered graph type 'atlas', and its types ROAD"
    );
    assert!(!shows(&session, "SHOW EDGE TYPES", "ROAD"));
    assert_eq!(
        dropped(&session, "ALTER GRAPH TYPE atlas DROP EDGE TYPE RAIL"),
        "Altered graph type 'atlas'"
    );
    assert!(shows(&session, "SHOW EDGE TYPES", "RAIL"));
    session
        .execute("INSERT (:City)-[:ROAD]->(:City)")
        .unwrap_or_else(|error| panic!("no rule by label is left: {error}"));
}

/// Cypher's current graph type is rules by label, not a listing: dropping from it takes no type,
/// and the rule stands.
#[test]
fn a_cypher_drop_from_the_current_graph_type_takes_no_type() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Kept {name STRING}) }")
        .unwrap();
    db.execute_cypher("ALTER CURRENT GRAPH TYPE DROP { (:Kept {name STRING}) }")
        .unwrap();
    let session = db.session();
    assert!(shows(&session, "SHOW NODE TYPES", "Kept"));
    assert!(session.execute("INSERT (:Kept {name: 7})").is_err());
}

/// A replace drops the old graph type first (ISO/IEC 39075 12.6, General Rule 2), so what it stops
/// listing goes as in a drop, and the status names it.
#[test]
fn a_replace_takes_what_it_stops_listing() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
        ],
    );
    assert_eq!(
        dropped(
            &session,
            "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Person)",
        ),
        "Created graph type 'atlas', and its types City"
    );
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    assert!(shows(&session, "SHOW NODE TYPES", "Person"));
    session
        .execute("INSERT (:City {size: 1})")
        .unwrap_or_else(|error| panic!("no rule by label is left: {error}"));
}

/// What it still lists stays, and what another graph type lists stays, with neither named.
#[test]
fn a_replace_keeps_what_is_still_listed() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, NODE TYPE Person)",
            "CREATE GRAPH TYPE people (NODE TYPE Person)",
        ],
    );
    let status = dropped(
        &session,
        "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE City)",
    );
    assert!(!status.contains("City"), "{status}");
    assert!(!status.contains("Person"), "{status}");
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    assert!(shows(&session, "SHOW NODE TYPES", "Person"));
}

/// A named constraint on a type the replace stops listing refuses the statement and names the
/// constraint, and the old graph type stands. On a type it still lists, it stops nothing.
#[test]
fn a_named_constraint_refuses_a_replace_that_stops_listing_its_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, NODE TYPE Person)",
            "CREATE CONSTRAINT city_names FOR (n:City) ON (n.name) UNIQUE",
            "CREATE CONSTRAINT person_names FOR (n:Person) ON (n.name) UNIQUE",
        ],
    );
    let error = session
        .execute("CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Person)")
        .expect_err("a named constraint sits on the type it stops listing");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(error.to_string().contains("city_names"), "{error}");
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    // The old listing stands: both labels are the type's, in a graph bound to it.
    run_all(
        &session,
        &["CREATE GRAPH maps TYPED atlas", "USE GRAPH maps"],
    );
    run_all(
        &session,
        &[
            "INSERT (:City {name: 'Paris'}), (:Person {name: 'Gus'})",
            "USE GRAPH default",
            "DROP GRAPH maps",
            "DROP CONSTRAINT city_names",
        ],
    );
    let status = dropped(
        &session,
        "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Person)",
    );
    assert!(status.contains("City"), "{status}");
    assert!(
        rows(&session, "SHOW CONSTRAINTS").len() == 1,
        "the constraint on what it still lists stays"
    );
}

/// The old graph type is gone first, so a replace may declare a name it held differently.
#[test]
fn a_replace_may_declare_its_own_type_again() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING NOT NULL))",
            "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE City (size INT64))",
            "CREATE GRAPH maps TYPED atlas",
            "USE GRAPH maps",
        ],
    );
    session
        .execute("INSERT (:City {size: 3})")
        .unwrap_or_else(|error| panic!("the new declaration: {error}"));
    assert!(session.execute("INSERT (:City {name: 'Paris'})").is_err());
}

/// A rollback of a replace brings back the old listing and the type it stopped listing.
#[test]
fn a_rolled_back_replace_takes_nothing() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Person (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
        ],
    );
    session.begin_transaction().unwrap();
    session
        .execute("CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Person)")
        .unwrap();
    session.rollback().unwrap();
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    run_all(
        &session,
        &["CREATE GRAPH maps TYPED atlas", "USE GRAPH maps"],
    );
    session.execute("INSERT (:City {name: 'Paris'})").unwrap();
    assert!(session.execute("INSERT (:Person {name: 'Gus'})").is_err());
}

/// A rollback brings back the graph type and its types, with their rules.
#[test]
fn a_rolled_back_drop_takes_nothing() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[ROUTES]);
    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH TYPE routes").unwrap();
    assert!(
        !shows(&session, "SHOW NODE TYPES", "City"),
        "gone inside the transaction"
    );
    session.rollback().unwrap();
    assert!(shows(&session, "SHOW GRAPH TYPES", "routes"));
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    run_all(
        &session,
        &["CREATE GRAPH atlas TYPED routes", "USE GRAPH atlas"],
    );
    assert!(session.execute("INSERT (:City {size: 1})").is_err());
}

/// A rollback to a savepoint brings them back, what came before it stays, and the commit then
/// keeps the graph type and its types whole.
#[test]
fn a_drop_rolled_back_to_a_savepoint_takes_nothing() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[ROUTES]);
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Museum {name: 'Rijks'})").unwrap();
    session.savepoint("s").unwrap();
    session.execute("DROP GRAPH TYPE routes").unwrap();
    session.rollback_to_savepoint("s").unwrap();
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    session.commit().unwrap();
    assert!(shows(&session, "SHOW GRAPH TYPES", "routes"));
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    assert!(shows(&session, "SHOW EDGE TYPES", "ROUTE"));
    assert_eq!(rows(&session, "MATCH (m:Museum) RETURN m.name").len(), 1);
}

/// A drop before a savepoint stays when a later one rolls back, and the commit makes it stand.
#[test]
fn a_committed_drop_in_a_transaction_takes_the_types() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[ROUTES]);
    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH TYPE routes").unwrap();
    session.savepoint("s").unwrap();
    session.execute("INSERT (:Museum {name: 'Rijks'})").unwrap();
    session.rollback_to_savepoint("s").unwrap();
    session.commit().unwrap();
    assert!(!shows(&session, "SHOW GRAPH TYPES", "routes"));
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    assert!(!shows(&session, "SHOW EDGE TYPES", "ROUTE"));
}

/// A refused drop in a transaction takes nothing, and the transaction goes on.
#[test]
fn a_refused_drop_leaves_the_transaction_going() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
            "CREATE CONSTRAINT city_names FOR (n:City) ON (n.name) UNIQUE",
        ],
    );
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Museum {name: 'Rijks'})").unwrap();
    assert!(session.execute("DROP GRAPH TYPE atlas").is_err());
    session.execute("INSERT (:Museum {name: 'Prado'})").unwrap();
    session.commit().unwrap();
    assert!(shows(&session, "SHOW GRAPH TYPES", "atlas"));
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    assert_eq!(rows(&session, "MATCH (m:Museum) RETURN m.name").len(), 2);
}

/// After a reopen they are still gone, and no rule by label came back.
#[test]
fn a_drop_holds_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drop.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
            "DROP GRAPH TYPE atlas",
        ],
    );
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(!shows(&session, "SHOW NODE TYPES", "City"));
    session.execute("INSERT (:City {size: 1})").unwrap();
    drop(session);
    db.close().unwrap();
}

const DROP_PATH_VAR: &str = "GRAFEO_TYPE_SCOPE_DROP_PATH";
const DROP_SCENARIO_VAR: &str = "GRAFEO_TYPE_SCOPE_DROP_SCENARIO";

fn crash_with_drop(path: &std::path::Path, scenario: &str) {
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("drop_crash_child")
            .env(DROP_PATH_VAR, path)
            .env(DROP_SCENARIO_VAR, scenario),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
}

#[test]
fn drop_crash_child() {
    let Some(path) = std::env::var_os(DROP_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let mut session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE NODE TYPE Museum (name STRING NOT NULL)",
            "CREATE EDGE TYPE ROAD (km INTEGER NOT NULL)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, EDGE TYPE ROAD)",
        ],
    );
    match std::env::var(DROP_SCENARIO_VAR).unwrap().as_str() {
        "statement" => run_all(&session, &["DROP GRAPH TYPE atlas"]),
        "transaction" => {
            session.begin_transaction().unwrap();
            run_all(
                &session,
                &["DROP GRAPH TYPE atlas", "INSERT (:Town {size: 1})"],
            );
            session.commit().unwrap();
        }
        "rolled back" => {
            session.begin_transaction().unwrap();
            run_all(&session, &["DROP GRAPH TYPE atlas"]);
            session.rollback().unwrap();
        }
        "open" => {
            session.begin_transaction().unwrap();
            run_all(&session, &["DROP GRAPH TYPE atlas"]);
        }
        other => panic!("no scenario {other}"),
    }
    std::process::exit(0);
}

/// The same from the log alone: the statement replays as one, in a transaction too, and a type it
/// did not list stays.
#[test]
fn a_drop_holds_after_a_crash() {
    for scenario in ["statement", "transaction"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drop-crash.grafeo");
        crash_with_drop(&path, scenario);
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        assert!(!shows(&session, "SHOW GRAPH TYPES", "atlas"), "{scenario}");
        assert!(!shows(&session, "SHOW NODE TYPES", "City"), "{scenario}");
        assert!(!shows(&session, "SHOW EDGE TYPES", "ROAD"), "{scenario}");
        assert!(shows(&session, "SHOW NODE TYPES", "Museum"), "{scenario}");
        session.execute("INSERT (:City {size: 1})").unwrap();
        assert!(session.execute("INSERT (:Museum {size: 1})").is_err());
        drop(session);
        db.close().unwrap();
    }
}

/// A drop rolled back, or still open at the crash, takes nothing after it.
#[test]
fn a_drop_not_committed_takes_nothing_after_a_crash() {
    for scenario in ["rolled back", "open"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("drop-crash.grafeo");
        crash_with_drop(&path, scenario);
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        assert!(shows(&session, "SHOW GRAPH TYPES", "atlas"), "{scenario}");
        assert!(shows(&session, "SHOW NODE TYPES", "City"), "{scenario}");
        assert!(shows(&session, "SHOW EDGE TYPES", "ROAD"), "{scenario}");
        drop(session);
        db.close().unwrap();
    }
}

/// The drop of the graph type and of its types is one group of the log: cut in its middle by a
/// crash, all of it is gone, not the graph type alone.
#[test]
fn a_torn_drop_takes_nothing() {
    use grafeo_storage::file::v3::header::FileHeaderV3;
    use grafeo_storage::wal::{SEGMENT_HEADER_BYTES, ScanOptions, WalScan, list_wal_directory};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drop-torn.grafeo");
    crash_with_drop(&path, "statement");
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    let wal = std::path::PathBuf::from(wal);
    let page = std::fs::read(&path).unwrap();
    let database_id = FileHeaderV3::decode(&page[..4096]).unwrap().database_id;
    let (first_lsn, segment) = list_wal_directory(&wal).unwrap().segments.pop().unwrap();
    let mut scan = WalScan::open(&wal, ScanOptions::new(database_id, first_lsn)).unwrap();
    let mut last = None;
    while let Some(group) = scan.next_group().unwrap() {
        last = Some((group.start_lsn(), group.end_lsn()));
    }
    drop(scan);
    let (start, end) = last.expect("the drop's group");
    let cut = SEGMENT_HEADER_BYTES as u64 + (start - first_lsn) + (end - start) / 2;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&segment)
        .unwrap()
        .set_len(cut)
        .unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(shows(&session, "SHOW GRAPH TYPES", "atlas"));
    assert!(shows(&session, "SHOW NODE TYPES", "City"));
    assert!(shows(&session, "SHOW EDGE TYPES", "ROAD"));
    drop(session);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// A second declaration of a name
// ---------------------------------------------------------------------------

/// 0.6.1 kept the last declaration, so a graph of the first type lost its required `name`: a
/// second, different declaration is refused, leaves nothing, and the first keeps its rules.
#[test]
fn a_second_different_declaration_of_a_name_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH TYPE a (NODE TYPE Person (name STRING NOT NULL))"],
    );
    let error = session
        .execute("CREATE GRAPH TYPE b (NODE TYPE Person (age INT64))")
        .expect_err("Person is declared differently");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(
        error.to_string().contains(
            "node type 'Person' is already declared differently: a name stands for one type \
             across graph types, so declare it the same way or use another name"
        ),
        "{error}"
    );
    assert!(!shows(&session, "SHOW GRAPH TYPES", "b"));
    run_all(&session, &["CREATE GRAPH ga TYPED a", "USE GRAPH ga"]);
    assert!(session.execute("INSERT (:Person {age: 1})").is_err());
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
}

/// The same over a standalone type, and for an edge type.
#[test]
fn a_second_declaration_over_a_standalone_or_edge_type_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE EDGE TYPE ROAD (km INT64)",
        ],
    );
    for statement in [
        "CREATE GRAPH TYPE b (NODE TYPE City (size INT64))",
        "CREATE GRAPH TYPE c { (:Town)-[:ROAD {km STRING}]->(:Town) }",
    ] {
        session
            .execute(statement)
            .expect_err("a different declaration");
    }
    assert!(!shows(&session, "SHOW GRAPH TYPES", "b"));
    assert!(!shows(&session, "SHOW GRAPH TYPES", "c"));
    assert!(
        !shows(&session, "SHOW NODE TYPES", "Town"),
        "a refused statement leaves nothing"
    );
}

/// The same declaration twice is one type, and both graph types list it: only the last drop takes
/// it.
#[test]
fn the_same_declaration_twice_is_one_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE a (NODE TYPE Person (name STRING NOT NULL))",
            "CREATE GRAPH TYPE b (NODE TYPE Person (name STRING NOT NULL))",
            "CREATE GRAPH ga TYPED a",
            "CREATE GRAPH gb TYPED b",
        ],
    );
    for graph in ["ga", "gb"] {
        session.execute(&format!("USE GRAPH {graph}")).unwrap();
        assert!(
            session.execute("INSERT (:Person {age: 1})").is_err(),
            "{graph}"
        );
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    }
    run_all(
        &session,
        &["USE GRAPH default", "DROP GRAPH ga", "DROP GRAPH gb"],
    );
    assert_eq!(
        dropped(&session, "DROP GRAPH TYPE a"),
        "Dropped graph type 'a'"
    );
    assert_eq!(
        dropped(&session, "DROP GRAPH TYPE b"),
        "Dropped graph type 'b', and its types Person"
    );
}

/// What makes a node type's declaration another one: each of these against `Person`.
#[test]
fn every_difference_of_a_node_type_is_refused() {
    for (what, declaration) in [
        (
            "another property",
            "Person (name STRING, age INT64, city STRING)",
        ),
        ("a property less", "Person (name STRING)"),
        ("another type", "Person (name STRING, age STRING)"),
        ("nullability", "Person (name STRING NOT NULL, age INT64)"),
        ("a default", "Person (name STRING, age INT64 DEFAULT 3)"),
        (
            "a key label",
            "Person KEY (PersonKey) (name STRING, age INT64)",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(
            &session,
            &["CREATE GRAPH TYPE a (NODE TYPE Person (name STRING, age INT64))"],
        );
        let statement = format!("CREATE GRAPH TYPE b (NODE TYPE {declaration})");
        let error = session
            .execute(&statement)
            .expect_err(&format!("{what}: {statement}"));
        assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{what}: {error}");
        assert!(!shows(&session, "SHOW GRAPH TYPES", "b"), "{what}");
    }
}

/// A graph type cannot name a parent, so a type with one is another declaration than its own
/// properties alone.
#[test]
fn a_type_with_a_parent_is_not_its_properties_alone() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE Being (born INT64)",
            "CREATE NODE TYPE Person EXTENDS Being (name STRING, age INT64)",
        ],
    );
    let error = session
        .execute("CREATE GRAPH TYPE b (NODE TYPE Person (name STRING, age INT64))")
        .expect_err("Person has a parent");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    session
        .execute("CREATE GRAPH TYPE c (NODE TYPE Person)")
        .unwrap_or_else(|error| panic!("a bare reference is that type: {error}"));
}

/// Property order is no difference, and neither is a bare reference.
#[test]
fn the_same_type_written_otherwise_is_one_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE a (NODE TYPE Person (name STRING NOT NULL, age INT64 DEFAULT 3))",
            "CREATE GRAPH TYPE b (NODE TYPE Person (age INT64 DEFAULT 3, name STRING NOT NULL))",
            "CREATE GRAPH TYPE c (NODE TYPE Person)",
        ],
    );
    assert_eq!(
        rows(&session, "SHOW NODE TYPES")
            .iter()
            .filter(|row| row.first() == Some(&string("Person")))
            .count(),
        1
    );
}

/// An edge type's endpoints: a pattern may add them to one that named none, and then they are part
/// of what it is. Dropping or changing them is another declaration.
#[test]
fn an_edge_types_endpoints_count_once_it_has_them() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE KNOWS (since INT64)",
            "CREATE GRAPH TYPE a { (:Person)-[:KNOWS {since INT64}]->(:Person) }",
        ],
    );
    assert_eq!(
        rows(&session, "SHOW EDGE TYPES")
            .into_iter()
            .find(|row| row.first() == Some(&string("KNOWS")))
            .expect("KNOWS")[2..4],
        [string("Person"), string("Person")],
        "the pattern added them"
    );
    for statement in [
        "CREATE GRAPH TYPE b (EDGE TYPE KNOWS (since INT64))",
        "CREATE GRAPH TYPE c { (:Town)-[:KNOWS {since INT64}]->(:Person) }",
    ] {
        let error = session
            .execute(statement)
            .expect_err(&format!("the endpoints differ: {statement}"));
        assert!(error.to_string().contains("KNOWS"), "{error}");
    }
}

/// A replace may change a type only it lists, not one another graph type lists too.
#[test]
fn a_replace_cannot_change_a_type_another_graph_type_lists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE a (NODE TYPE Person (name STRING NOT NULL))",
            "CREATE GRAPH TYPE b (NODE TYPE Person (name STRING NOT NULL))",
        ],
    );
    let error = session
        .execute("CREATE OR REPLACE GRAPH TYPE a (NODE TYPE Person (age INT64))")
        .expect_err("b lists Person as it is");
    assert!(
        error.to_string().contains("declared differently"),
        "{error}"
    );
    run_all(&session, &["CREATE GRAPH gb TYPED b", "USE GRAPH gb"]);
    assert!(session.execute("INSERT (:Person {age: 1})").is_err());
}

// ---------------------------------------------------------------------------
// The warning in validate()
// ---------------------------------------------------------------------------

/// The messages of the warnings about rules a graph no longer has.
fn not_bound(db: &GrafeoDB) -> Vec<String> {
    let result = db.validate().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    result
        .warnings
        .into_iter()
        .filter(|warning| warning.code == "GRAPH_TYPE_NOT_BOUND")
        .map(|warning| warning.message)
        .collect()
}

fn with_statements(statements: &[&str]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, statements);
    drop(session);
    db
}

/// The `context` says what kind of type, and a warning leaves the database valid.
#[test]
fn the_warning_gives_the_kind_of_type_in_its_context() {
    let db = with_statements(&[
        "CREATE GRAPH TYPE atlas { (:City {name STRING})-[:ROAD {km INT64}]->(:City) }",
        "INSERT (:City {size: 1})-[:ROAD {km: 'far'}]->(:City {size: 2})",
    ]);
    let result = db.validate().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let mut contexts: Vec<String> = result
        .warnings
        .into_iter()
        .filter(|warning| warning.code == "GRAPH_TYPE_NOT_BOUND")
        .filter_map(|warning| warning.context)
        .collect();
    contexts.sort_unstable();
    assert_eq!(contexts, ["edge_type:ROAD", "node_type:City"]);
}

/// Nodes of a listed label in the default graph: a warning naming the type, the graph type and the
/// graph.
#[test]
fn a_listed_label_with_nodes_in_the_default_graph_warns() {
    let db = with_statements(&[
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "CREATE GRAPH TYPE atlas (NODE TYPE City)",
        "INSERT (:City {size: 1})",
    ]);
    let warnings = not_bound(&db);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    for part in ["City", "atlas", "default"] {
        assert!(warnings[0].contains(part), "{part} in {warnings:?}");
    }
}

/// In a named graph not bound to it: the warning names that graph.
#[test]
fn a_listed_label_with_nodes_in_an_unbound_graph_warns() {
    let db = with_statements(&[
        "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
        "CREATE GRAPH plain",
        "USE GRAPH plain",
        "INSERT (:City {size: 1})",
    ]);
    let warnings = not_bound(&db);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("plain"), "{warnings:?}");
}

/// Edge types the same: a listed edge type with edges outside its graphs.
#[test]
fn a_listed_edge_type_with_edges_in_an_unbound_graph_warns() {
    for (where_, statements) in [
        (
            "default",
            &[
                "CREATE GRAPH TYPE atlas (EDGE TYPE ROAD (km INT64))",
                "INSERT (:A)-[:ROAD {km: 'far'}]->(:B)",
            ][..],
        ),
        (
            "plain",
            &[
                "CREATE GRAPH TYPE atlas (EDGE TYPE ROAD (km INT64))",
                "CREATE GRAPH plain",
                "USE GRAPH plain",
                "INSERT (:A)-[:ROAD {km: 'far'}]->(:B)",
            ][..],
        ),
    ] {
        let warnings = not_bound(&with_statements(statements));
        assert_eq!(warnings.len(), 1, "{where_}: {warnings:?}");
        for part in ["ROAD", "atlas", where_] {
            assert!(warnings[0].contains(part), "{part} in {warnings:?}");
        }
    }
}

/// One warning for a type, however many graphs hold it and however many graph types list it, and
/// one for each type.
#[test]
fn one_warning_for_each_type() {
    let db = with_statements(&[
        "CREATE GRAPH TYPE a (NODE TYPE City (name STRING), NODE TYPE Town (name STRING))",
        "CREATE GRAPH TYPE b (NODE TYPE City (name STRING))",
        "CREATE GRAPH plain",
        "INSERT (:City {size: 1}), (:Town {size: 2})",
        "USE GRAPH plain",
        "INSERT (:City {size: 3})",
    ]);
    let warnings = not_bound(&db);
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert_eq!(warnings.iter().filter(|w| w.contains("City")).count(), 1);
    assert_eq!(warnings.iter().filter(|w| w.contains("Town")).count(), 1);
}

/// Nothing to warn of: nodes only in the bound graph, or in a graph of another graph type that
/// lists the same type; a type no graph type lists; a listed label with no nodes.
#[test]
fn no_warning_where_no_rule_is_lost() {
    for (what, statements) in [
        (
            "only in the bound graph",
            &[
                "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
                "CREATE GRAPH maps TYPED atlas",
                "USE GRAPH maps",
                "INSERT (:City {name: 'Paris'})",
            ][..],
        ),
        (
            "in a graph of another type that lists it",
            &[
                "CREATE GRAPH TYPE a (NODE TYPE City (name STRING))",
                "CREATE GRAPH TYPE b (NODE TYPE City (name STRING))",
                "CREATE GRAPH maps TYPED b",
                "USE GRAPH maps",
                "INSERT (:City {name: 'Paris'})",
            ][..],
        ),
        (
            "a type no graph type lists",
            &[
                "CREATE NODE TYPE City (name STRING NOT NULL)",
                "INSERT (:City {name: 'Paris'})",
            ][..],
        ),
        (
            "a listed label with no nodes",
            &[
                "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
                "INSERT (:Town {name: 'Paris'})",
            ][..],
        ),
    ] {
        let warnings = not_bound(&with_statements(statements));
        assert!(warnings.is_empty(), "{what}: {warnings:?}");
    }
}

/// It reads the state now: the nodes deleted, or the type taken away, and the warning is gone.
#[test]
fn the_warning_goes_with_its_cause() {
    let db = with_statements(&[
        "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
        "INSERT (:City {size: 1})",
    ]);
    assert_eq!(not_bound(&db).len(), 1);
    db.execute("MATCH (c:City) DELETE c").unwrap();
    assert!(not_bound(&db).is_empty(), "the nodes are deleted");
    db.execute("INSERT (:City {size: 2})").unwrap();
    assert_eq!(not_bound(&db).len(), 1);
    db.execute("DROP GRAPH TYPE atlas").unwrap();
    assert!(not_bound(&db).is_empty(), "the graph type is dropped");
}

/// Cypher's current graph type is rules by label, which hold everywhere: nothing is lost.
#[test]
fn no_warning_for_the_cypher_current_graph_type() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Kept {name STRING}) }")
        .unwrap();
    db.execute("INSERT (:Kept {name: 'Gus'})").unwrap();
    assert!(not_bound(&db).is_empty(), "rules by label lose nothing");
}

/// The warning stands after a reopen, and a warning alone does not make a database invalid.
#[test]
fn the_warning_stands_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("warnings.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
            "INSERT (:City {size: 1})",
        ],
    );
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(not_bound(&db).len(), 1);
    db.close().unwrap();
}

/// A file of an earlier release, which held these rules in its default graph, warns of each listed
/// type with elements there, and of no other type.
#[test]
fn a_released_file_warns_of_the_rules_it_lost() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/released/0.6.0/closed.grafeo"
        ),
        &path,
    )
    .unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let warnings = not_bound(&db);
    db.close().unwrap();
    for (name, want) in [
        ("City", 1),
        ("Museum", 1),
        ("IN_CITY", 1),
        ("ROUTE", 0),
        ("Person", 0),
    ] {
        let found = warnings.iter().filter(|w| w.contains(name)).count();
        assert_eq!(found, want, "{name} in {warnings:?}");
    }
}

// ---------------------------------------------------------------------------
// Bare patterns, and endpoints a pattern adds
// ---------------------------------------------------------------------------

/// A bare `(:City)` over a declared `City` refers to it: its properties stay, and they are the
/// rules of a graph bound to the graph type.
#[test]
fn a_bare_pattern_over_a_declared_type_keeps_its_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "CREATE GRAPH TYPE atlas { (:City)-[:ROAD]->(:City) }",
            "CREATE GRAPH maps TYPED atlas",
            "USE GRAPH maps",
        ],
    );
    let city = rows(&session, "SHOW NODE TYPES")
        .into_iter()
        .find(|row| row.first() == Some(&string("City")))
        .expect("City");
    assert!(
        format!("{:?}", city[1]).contains("name"),
        "the declared properties: {city:?}"
    );
    assert!(
        session
            .execute("INSERT (:City {size: 1})-[:ROAD]->(:City {name: 'Paris'})")
            .is_err(),
        "the required name"
    );
    session
        .execute("INSERT (:City {name: 'Prague'})-[:ROAD]->(:City {name: 'Paris'})")
        .unwrap_or_else(|error| panic!("what it declares: {error}"));
}

/// A bare `(:City)` over a type another graph type declared is that type too, no second
/// declaration.
#[test]
fn a_bare_pattern_over_another_graph_types_type_is_that_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE a (NODE TYPE City (name STRING NOT NULL))",
            "CREATE GRAPH TYPE b { (:City)-[:ROAD]->(:City) }",
            "CREATE GRAPH maps TYPED b",
            "USE GRAPH maps",
        ],
    );
    assert!(
        session
            .execute("INSERT (:City {size: 1})-[:ROAD]->(:City {name: 'Paris'})")
            .is_err(),
        "the required name of a's City"
    );
}

/// A bare `(:Town)` with no such type defines an empty one: it exists, with no properties.
#[test]
fn a_bare_pattern_with_no_such_type_defines_an_empty_one() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH TYPE atlas { (:Town)-[:ROAD]->(:Town) }"],
    );
    let town = rows(&session, "SHOW NODE TYPES")
        .into_iter()
        .find(|row| row.first() == Some(&string("Town")))
        .expect("Town");
    assert_eq!(town[1], string(""), "no properties: {town:?}");
}

/// Endpoints a pattern adds: allowed when nothing else uses the edge type, which then has them.
/// The rows of a graph of it obey them.
#[test]
fn a_pattern_may_add_endpoints_to_an_edge_type_nothing_uses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE KNOWS (since INT64)",
            "CREATE GRAPH TYPE people { (:Person)-[:KNOWS {since INT64}]->(:Person), (:Town) }",
            "CREATE GRAPH circle TYPED people",
            "USE GRAPH circle",
        ],
    );
    assert!(
        session
            .execute("INSERT (:Town)-[:KNOWS {since: 1}]->(:Person)")
            .is_err(),
        "the endpoints it added"
    );
    session
        .execute("INSERT (:Person)-[:KNOWS {since: 1}]->(:Person)")
        .unwrap();
}

/// Declared in the same statement: the pattern is both its declaration and its endpoints. (The
/// grammar takes no `EDGE TYPE` clause and a pattern in one body.)
#[test]
fn a_pattern_declares_a_new_edge_type_with_its_endpoints() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH TYPE people { (:Person)-[:KNOWS {since INT64}]->(:Person) }"],
    );
    let knows = row_of_type(&session, "SHOW EDGE TYPES", "KNOWS");
    assert_eq!(knows[1], string("since INT64"), "{knows:?}");
    assert_eq!(
        knows[2..4],
        [string("Person"), string("Person")],
        "{knows:?}"
    );
}

/// Otherwise it is a second, different declaration: another graph type lists the edge type with no
/// endpoints, and a graph bound to that type must keep what it allows.
#[test]
fn a_pattern_cannot_add_endpoints_to_an_edge_type_another_graph_type_uses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE open_roads (NODE TYPE Town (name STRING), NODE TYPE City (name STRING), \
             EDGE TYPE ROAD (km INT64))",
            "CREATE GRAPH maps TYPED open_roads",
        ],
    );
    let error = session
        .execute("CREATE GRAPH TYPE strict { (:City)-[:ROAD {km INT64}]->(:City) }")
        .expect_err("open_roads uses ROAD with no endpoints");
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(
        error.to_string().contains("declared differently"),
        "{error}"
    );
    assert!(!shows(&session, "SHOW GRAPH TYPES", "strict"));
    // The first type's graph still takes an edge the refused endpoints would have stopped.
    run_all(
        &session,
        &[
            "USE GRAPH maps",
            "INSERT (:Town {name: 'Delft'})-[:ROAD {km: 3}]->(:City {name: 'Paris'})",
        ],
    );
    assert_eq!(
        rows(&session, "MATCH (:Town)-[r:ROAD]->(:City) RETURN r.km"),
        [[Value::Int64(3)]]
    );
}

/// `rows` of `SHOW NODE TYPES` or `SHOW EDGE TYPES` for one name.
fn row_of_type(session: &Session, statement: &str, name: &str) -> Vec<Value> {
    rows(session, statement)
        .into_iter()
        .find(|row| row.first() == Some(&string(name)))
        .unwrap_or_else(|| panic!("no {name} in {statement}"))
}

/// A pattern adds endpoints to a standalone edge type nothing lists: the stored type keeps its
/// properties and gains them. 0.6.1 stored the pattern's empty properties over it.
#[test]
fn a_pattern_keeps_the_properties_of_what_it_names() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE NODE TYPE A (name STRING NOT NULL)",
            "CREATE EDGE TYPE E (km INT64 NOT NULL)",
        ],
    );
    let node_before = row_of_type(&session, "SHOW NODE TYPES", "A");
    let edge_before = row_of_type(&session, "SHOW EDGE TYPES", "E");
    run_all(&session, &["CREATE GRAPH TYPE t { (:A)-[:E]->(:B) }"]);
    assert_eq!(
        row_of_type(&session, "SHOW NODE TYPES", "A"),
        node_before,
        "the node type"
    );
    let edge = row_of_type(&session, "SHOW EDGE TYPES", "E");
    assert_eq!(edge[1], edge_before[1], "the edge type's properties");
    assert_eq!(
        edge[2..4],
        [string("A"), string("B")],
        "the endpoints added"
    );
}

/// Same endpoints and no properties written: it only names the type.
#[test]
fn a_pattern_with_the_same_endpoints_only_names_the_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE EDGE TYPE E CONNECTING (A) TO (B) (km INT64)"],
    );
    let before = row_of_type(&session, "SHOW EDGE TYPES", "E");
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE one { (:A)-[:E]->(:B) }",
            "CREATE GRAPH TYPE two { (:A)-[:E]->(:B) }",
        ],
    );
    assert_eq!(row_of_type(&session, "SHOW EDGE TYPES", "E"), before);
}

/// The keyword form with nothing stated only names the type, whatever its endpoints.
#[test]
fn the_keyword_form_with_nothing_stated_only_names_the_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE EDGE TYPE E CONNECTING (A) TO (B) (km INT64)"],
    );
    let before = row_of_type(&session, "SHOW EDGE TYPES", "E");
    run_all(&session, &["CREATE GRAPH TYPE t (EDGE TYPE E ())"]);
    assert_eq!(row_of_type(&session, "SHOW EDGE TYPES", "E"), before);
}

/// A replace may change an edge type only it lists: a pattern with no properties keeps them and
/// takes the new endpoints.
#[test]
fn a_replace_may_change_the_endpoints_of_an_edge_type_only_it_lists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE E (km INT64)",
            "CREATE GRAPH TYPE t { (:A)-[:E]->(:B) }",
            "CREATE OR REPLACE GRAPH TYPE t { (:C)-[:E]->(:D) }",
        ],
    );
    let edge = row_of_type(&session, "SHOW EDGE TYPES", "E");
    assert_eq!(edge[1], string("km INT64"), "{edge:?}");
    assert_eq!(edge[2..4], [string("C"), string("D")], "{edge:?}");
}

/// Different endpoints from the ones it has are another declaration, also with no properties
/// written: before, they were ignored while the graph's writes followed the old ones.
#[test]
fn different_endpoints_are_refused_without_properties_too() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE EDGE TYPE E CONNECTING (A) TO (B) (km INT64)"],
    );
    let before = row_of_type(&session, "SHOW EDGE TYPES", "E");
    let error = session
        .execute("CREATE GRAPH TYPE t { (:C)-[:E]->(:D) }")
        .expect_err("other endpoints");
    assert!(
        error.to_string().contains("declared differently"),
        "{error}"
    );
    assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{error}");
    assert!(!shows(&session, "SHOW GRAPH TYPES", "t"));
    assert!(!shows(&session, "SHOW NODE TYPES", "C"), "leaves nothing");
    assert_eq!(row_of_type(&session, "SHOW EDGE TYPES", "E"), before);
}

/// Cypher's current graph type holds rules by label in every graph, so it cannot add endpoints to
/// a standalone edge type.
#[test]
fn the_current_graph_type_cannot_add_endpoints_to_a_standalone_edge_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["CREATE EDGE TYPE E (km INT64)"]);
    let before = row_of_type(&session, "SHOW EDGE TYPES", "E");
    let error = db
        .execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:A)-[:E]->(:B) }")
        .expect_err("rules by label in every graph");
    assert!(
        error.to_string().contains("declared differently"),
        "{error}"
    );
    assert_eq!(row_of_type(&session, "SHOW EDGE TYPES", "E"), before);
}

/// One reading of "nothing else uses it": edges of a standalone type in an unbound graph
/// do not stop a pattern from adding endpoints, and that graph's rules do not change through it:
/// listing moved the rule off it, and the endpoints reach only the graph type's graphs.
#[test]
fn endpoints_a_pattern_adds_do_not_reach_an_unbound_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE E (km INT64)",
            "CREATE GRAPH plain",
            "USE GRAPH plain",
            "INSERT (:Z)-[:E {km: 1}]->(:Z)",
            "USE GRAPH default",
            "CREATE GRAPH TYPE t { (:A {name STRING})-[:E]->(:B {name STRING}) }",
            "USE GRAPH plain",
        ],
    );
    session
        .execute("INSERT (:Y)-[:E {km: 'far'}]->(:Y)")
        .unwrap_or_else(|error| panic!("no rule of t in plain: {error}"));
    assert_eq!(rows(&session, "MATCH ()-[e:E]->() RETURN e.km").len(), 2);
    run_all(
        &session,
        &[
            "USE GRAPH default",
            "CREATE GRAPH bound TYPED t",
            "USE GRAPH bound",
        ],
    );
    session
        .execute("INSERT (:A {name: 'x'})-[:E {km: 1}]->(:B {name: 'y'})")
        .unwrap_or_else(|error| panic!("the endpoints it added: {error}"));
    assert!(
        session
            .execute("INSERT (:B {name: 'x'})-[:E {km: 1}]->(:A {name: 'y'})")
            .is_err(),
        "the other way round is no endpoint of E"
    );
}

// ---------------------------------------------------------------------------
// Two patterns for one edge type in a statement
// ---------------------------------------------------------------------------

/// A refusal of several patterns names the type, says it is not supported yet, and leaves nothing.
fn assert_two_patterns_refused(session: &Session, error: &grafeo_common::utils::error::Error) {
    assert_eq!(GqlStatus::from(error).as_str(), "42000", "{error}");
    assert!(error.to_string().contains('E'), "{error}");
    assert!(error.to_string().contains("not supported yet"), "{error}");
    assert!(!shows(session, "SHOW GRAPH TYPES", "t"));
    assert!(!shows(session, "SHOW EDGE TYPES", "E"));
}

/// Differing on one side: the type takes both sources, its properties stand, a write from each
/// passes, and a write from a source not stated is refused.
#[test]
fn two_patterns_differing_on_one_side_merge() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE t { (:A)-[:E {km INT64}]->(:B), (:C)-[:E {km INT64}]->(:B), (:D) }",
            "CREATE GRAPH g TYPED t",
            "USE GRAPH g",
        ],
    );
    for statement in [
        "INSERT (:A)-[:E {km: 1}]->(:B)",
        "INSERT (:C)-[:E {km: 2}]->(:B)",
    ] {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    assert_eq!(rows(&session, "MATCH ()-[e:E]->() RETURN e.km").len(), 2);
    let error = session
        .execute("INSERT (:D)-[:E {km: 3}]->(:B)")
        .expect_err("a source not stated");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
    assert!(
        session
            .execute("INSERT (:A)-[:E {km: 'far'}]->(:B)")
            .is_err(),
        "the first pattern's km stands"
    );
}

/// Differing on both sides would allow what neither pattern states: refused, leaving nothing.
#[test]
fn two_patterns_differing_on_both_sides_are_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute("CREATE GRAPH TYPE t { (:A)-[:E]->(:B), (:C)-[:E]->(:D) }")
        .expect_err("both sides differ");
    assert_two_patterns_refused(&session, &error);
    assert!(!shows(&session, "SHOW NODE TYPES", "A"));
}

/// Differing properties are two declarations in one statement: refused, leaving nothing.
#[test]
fn two_patterns_with_different_properties_are_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute("CREATE GRAPH TYPE t { (:A)-[:E {km INT64}]->(:B), (:C)-[:E {km STRING}]->(:B) }")
        .expect_err("km is two types");
    assert_two_patterns_refused(&session, &error);
}

/// The same pattern twice is one type with those endpoints.
#[test]
fn identical_patterns_are_one_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH TYPE t { (:A)-[:E {km INT64}]->(:B), (:A)-[:E {km INT64}]->(:B) }"],
    );
    let edge = row_of_type(&session, "SHOW EDGE TYPES", "E");
    assert_eq!(edge[1], string("km INT64"), "{edge:?}");
    assert_eq!(edge[2..4], [string("A"), string("B")], "{edge:?}");
}

/// The case the rule is for: one edge type from two kinds of source, and a third is refused.
#[test]
fn two_sources_of_one_edge_type_are_accepted_and_a_third_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE t { (:Post)-[:HC]->(:Person), (:Comment)-[:HC]->(:Person) }",
            "CREATE GRAPH g TYPED t",
            "USE GRAPH g",
            "INSERT (:Post)-[:HC]->(:Person)",
            "INSERT (:Comment)-[:HC]->(:Person)",
        ],
    );
    let error = session
        .execute("INSERT (:Person)-[:HC]->(:Person)")
        .expect_err("Person is no source of HC");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
}

/// An undirected pattern and a directed one inside it, in either order: A and B on both sides.
#[test]
fn a_pattern_inside_another_is_one_type() {
    for patterns in [
        "(:A)-[:E]-(:B), (:A)-[:E]->(:B)",
        "(:A)-[:E]->(:B), (:A)-[:E]-(:B)",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(
            &session,
            &[
                &format!("CREATE GRAPH TYPE t {{ {patterns}, (:C) }}"),
                "CREATE GRAPH g TYPED t",
                "USE GRAPH g",
                "INSERT (:A)-[:E]->(:B)",
                "INSERT (:B)-[:E]->(:A)",
            ],
        );
        assert!(
            session.execute("INSERT (:C)-[:E]->(:A)").is_err(),
            "{patterns}: C is on neither side"
        );
    }
}

/// Each pattern is checked against what is merged so far: the third would make the lists accept an
/// edge no pattern states, C to D.
#[test]
fn a_third_pattern_that_would_allow_what_none_states_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute("CREATE GRAPH TYPE t { (:A)-[:E]->(:B), (:C)-[:E]->(:B), (:A)-[:E]->(:D) }")
        .expect_err("the lists would accept C to D");
    assert_two_patterns_refused(&session, &error);
}

/// `{km INT64}` against no properties is a difference as written. Pending a decision on its reading:
/// if a bare pattern only names the type, this one becomes the merge.
#[test]
fn a_pattern_with_properties_and_one_without_are_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute("CREATE GRAPH TYPE t { (:A)-[:E {km INT64}]->(:B), (:C)-[:E]->(:B) }")
        .expect_err("km against none");
    assert_two_patterns_refused(&session, &error);
}

/// The merged type goes through the rule for a pattern over an existing type as one declaration.
#[test]
fn merged_patterns_over_an_existing_edge_type_are_one_declaration() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE E (km INT64)",
            "CREATE GRAPH TYPE t { (:A)-[:E]->(:B), (:C)-[:E]->(:B) }",
            "CREATE GRAPH g TYPED t",
            "USE GRAPH g",
            "INSERT (:A)-[:E {km: 1}]->(:B)",
            "INSERT (:C)-[:E {km: 2}]->(:B)",
        ],
    );
    assert!(session.execute("INSERT (:B)-[:E {km: 3}]->(:B)").is_err());

    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE EDGE TYPE E (km INT64)",
            "CREATE GRAPH TYPE listed (EDGE TYPE E ())",
        ],
    );
    let error = session
        .execute("CREATE GRAPH TYPE t { (:A)-[:E]->(:B), (:C)-[:E]->(:B) }")
        .expect_err("another graph type lists E");
    assert!(
        error.to_string().contains("declared differently"),
        "{error}"
    );
    assert!(!shows(&session, "SHOW GRAPH TYPES", "t"));
}

/// Keyword-form repeats go through the same check.
#[test]
fn keyword_repeats_of_an_edge_type_are_checked_the_same_way() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["CREATE GRAPH TYPE same (EDGE TYPE E (x INT64), EDGE TYPE E (x INT64))"],
    );
    assert_eq!(
        row_of_type(&session, "SHOW EDGE TYPES", "E")[1],
        string("x INT64")
    );
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute("CREATE GRAPH TYPE t (EDGE TYPE E (x INT64), EDGE TYPE E (x STRING))")
        .expect_err("x is two types");
    assert_two_patterns_refused(&session, &error);
}

/// Cypher's `ALTER CURRENT GRAPH TYPE SET` takes the same path.
#[test]
fn the_current_graph_type_refuses_patterns_on_both_sides_too() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:A)-[:E]->(:B), (:C)-[:E]->(:D) }")
        .expect_err("both sides differ");
    assert!(error.to_string().contains("not supported yet"), "{error}");
    let session = db.session();
    assert!(!shows(&session, "SHOW EDGE TYPES", "E"));
}
