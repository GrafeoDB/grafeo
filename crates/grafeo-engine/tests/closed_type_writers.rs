//! What a closed graph type guarantees holds whichever way a write comes in (typed-graph design, H1
//! to H3).
//!
//! A node on a closed graph has at least one declared label (ISO/IEC 39075:2024 4.13.2.1), every
//! property declared by one of its labels, and every edge between endpoints its type allows. Removing
//! a label can break each of these, so `REMOVE` is refused where it would; every other writer either
//! checks the type or says it cannot write here. Open and untyped graphs change nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main closed_type_writers::
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::BatchEdge;
use grafeo_engine::loader::{EdgeLoad, EndKey, NodeLoad};
use grafeo_engine::session::Session;

/// `Far` alone declares `k` among the first two; `Free` declares what any test's node holds.
const TYPES: &[&str] = &[
    "CREATE NODE TYPE Far (k INTEGER, name STRING)",
    "CREATE NODE TYPE Near (name STRING)",
    "CREATE NODE TYPE Free (id STRING, k INTEGER, name STRING)",
    "CREATE EDGE TYPE R CONNECTING (Far) TO (Near)",
];

/// The kinds of graph a removal is checked on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Closed,
    Open,
    Untyped,
}

/// A graph `g` of the kind, current for the database and for the session returned.
fn graph(kind: Kind) -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in TYPES {
        session.execute(statement).unwrap();
    }
    let statements: &[&str] = match kind {
        Kind::Closed => &[
            "CREATE GRAPH TYPE atlas (NODE TYPE Far, NODE TYPE Near, NODE TYPE Free, EDGE TYPE R)",
            "CREATE GRAPH g TYPED atlas",
        ],
        Kind::Open => &[
            "CREATE GRAPH TYPE atlas { node_types: [Far, Near, Free], edge_types: [R], open: true }",
            "CREATE GRAPH g TYPED atlas",
        ],
        Kind::Untyped => &["CREATE GRAPH g"],
    };
    for statement in statements {
        session.execute(statement).unwrap();
    }
    session.execute("USE GRAPH g").unwrap();
    db.set_current_graph(Some("g")).unwrap();
    (db, session)
}

/// Every node as its name and sorted labels, sorted.
fn nodes(session: &Session) -> Vec<String> {
    let result = session
        .execute("MATCH (x) RETURN x.name, labels(x)")
        .unwrap();
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| {
            let mut labels: Vec<String> = match &row[1] {
                Value::List(items) => items
                    .iter()
                    .map(|item| item.as_str().unwrap().to_string())
                    .collect(),
                other => panic!("labels: {other:?}"),
            };
            labels.sort();
            let name = row[0].as_str().unwrap_or("").to_string();
            format!("{name}:{}", labels.join(","))
        })
        .collect();
    rows.sort();
    rows
}

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// The GQLSTATUS of an error.
fn status_of_error(error: &grafeo_common::utils::error::Error) -> String {
    grafeo_common::utils::gqlstatus::GqlStatus::from(error)
        .as_str()
        .to_string()
}

fn status(error: &grafeo_common::utils::error::Error) -> String {
    status_of_error(error)
}

fn id_of(session: &Session, name: &str) -> NodeId {
    let found = session
        .execute(&format!("MATCH (x {{name: '{name}'}}) RETURN id(x)"))
        .unwrap();
    NodeId::new(u64::try_from(found.rows()[0][0].as_int64().unwrap()).unwrap())
}

/// The ways a label is removed: GQL, Cypher and each direct call.
#[derive(Clone, Copy, Debug)]
enum Way {
    Gql,
    Cypher,
    Session,
    Database,
    Handle,
}

const WAYS: [Way; 5] = [
    Way::Gql,
    Way::Cypher,
    Way::Session,
    Way::Database,
    Way::Handle,
];

/// Removes `label` from the node named `name`, the way given. `Ok(false)` when it had no such label.
fn remove(
    way: Way,
    db: &GrafeoDB,
    session: &Session,
    name: &str,
    label: &str,
) -> Result<bool, String> {
    let id = id_of(session, name);
    let text = format!("MATCH (x {{name: '{name}'}}) REMOVE x:{label}");
    match way {
        Way::Gql => session
            .execute(&text)
            .map(|_| true)
            .map_err(|e| e.to_string()),
        Way::Cypher => session
            .execute_cypher(&text)
            .map(|_| true)
            .map_err(|e| e.to_string()),
        Way::Session => session
            .remove_node_label(id, label)
            .map_err(|e| e.to_string()),
        Way::Database => db.remove_node_label(id, label).map_err(|e| e.to_string()),
        Way::Handle => db
            .graph("g")
            .unwrap()
            .remove_node_label(id, label)
            .map_err(|e| e.to_string()),
    }
}

/// On a closed graph the removal is refused and nothing moves; on an open or untyped one, it goes.
fn refused_only_where_closed(data: &[&str], name: &str, label: &str, after: &[&str]) {
    for way in WAYS {
        for kind in [Kind::Closed, Kind::Open, Kind::Untyped] {
            let (db, session) = graph(kind);
            run_all(&session, data);
            let before = nodes(&session);
            let outcome = remove(way, &db, &session, name, label);
            if kind == Kind::Closed {
                assert!(
                    outcome.is_err(),
                    "{way:?}, {data:?}: removing {label} from {name} leaves {:?}",
                    nodes(&session)
                );
                assert_eq!(nodes(&session), before, "{way:?}: nothing moved");
            } else {
                outcome.unwrap_or_else(|error| panic!("{kind:?}, {way:?}: {error}"));
                assert_eq!(nodes(&session), after, "{kind:?}, {way:?}");
            }
        }
    }
}

/// A node with no label is what a create refuses.
#[test]
fn the_last_label_stays_on_a_closed_graph() {
    refused_only_where_closed(
        &["INSERT (:Free {name: 'Alix'})"],
        "Alix",
        "Free",
        &["Alix:"],
    );
}

/// The label that declares a held property: after it goes, nothing declares `k`.
#[test]
fn a_label_that_declares_a_held_property_stays_on_a_closed_graph() {
    refused_only_where_closed(
        &["INSERT (:Far:Near {name: 'Alix', k: 1})"],
        "Alix",
        "Far",
        &["Alix:Near"],
    );
}

/// An edge's source or target no longer of a type its edge type connects. The rule is the edge type's,
/// so the label stays wherever a create would check it: on a closed graph, an open one, and one with
/// no graph type at all. Another remaining label that fits lets it go.
#[test]
fn an_endpoints_label_stays_wherever_its_rule_applies() {
    for way in WAYS {
        for kind in [Kind::Closed, Kind::Open, Kind::Untyped] {
            let (db, session) = graph(kind);
            run_all(
                &session,
                &[
                    "INSERT (:Far:Free {name: 'Alix'})-[:R]->(:Near:Free {name: 'Gus'})",
                    "INSERT (:Far:Near {name: 'Vincent'})-[:R]->(:Far:Near {name: 'Jules'})",
                ],
            );
            let before = nodes(&session);
            for (name, label) in [("Alix", "Far"), ("Gus", "Near")] {
                assert!(
                    remove(way, &db, &session, name, label).is_err(),
                    "{kind:?}, {way:?}: {name} is the edge's end through {label}"
                );
                assert_eq!(nodes(&session), before, "{kind:?}, {way:?}");
            }
            // Vincent is only ever a source, Jules only ever a target: the other label is free.
            remove(way, &db, &session, "Vincent", "Near").unwrap();
            remove(way, &db, &session, "Jules", "Far").unwrap();
        }
    }
}

/// A node with no edge of a constrained type, on any graph, loses the label as before.
#[test]
fn a_label_no_edge_needs_goes_wherever_nothing_is_closed() {
    for kind in [Kind::Open, Kind::Untyped] {
        let (_db, session) = graph(kind);
        run_all(
            &session,
            &[
                "INSERT (:Far {name: 'Alix'})",
                "MATCH (x {name: 'Alix'}) REMOVE x:Far",
            ],
        );
        assert_eq!(nodes(&session), vec!["Alix:"], "{kind:?}");
    }
}

/// The check is on the state the statement leaves (ISO/IEC 39075:2024 13.4), so the order of its
/// items does not matter: the property goes with the label that declared it, written either way.
#[test]
fn order_in_one_statement_does_not_matter() {
    for statement in [
        "MATCH (x {name: 'Alix'}) REMOVE x:Far, x.k",
        "MATCH (x {name: 'Alix'}) REMOVE x.k, x:Far",
    ] {
        let (_db, session) = graph(Kind::Closed);
        session
            .execute("INSERT (:Far:Near {name: 'Alix', k: 1})")
            .unwrap();
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
        assert_eq!(nodes(&session), vec!["Alix:Near"], "{statement}");
        let held = session.execute("MATCH (x:Near) RETURN x.k").unwrap();
        assert_eq!(held.rows()[0][0], Value::Null, "{statement}");
    }
}

/// Each statement is checked on what it leaves: taking the label first fails, though the next would
/// have mended it, in the order the two are written within one query or apart.
#[test]
fn two_statements_are_checked_each() {
    for statements in [
        &["MATCH (x {name: 'Alix'}) REMOVE x:Free SET x:Near"][..],
        &[
            "MATCH (x {name: 'Alix'}) REMOVE x:Free",
            "MATCH (x {name: 'Alix'}) SET x:Near",
        ][..],
    ] {
        let (_db, session) = graph(Kind::Closed);
        session.execute("INSERT (:Free {name: 'Alix'})").unwrap();
        let error = session.execute(statements[0]).expect_err(statements[0]);
        assert_eq!(status(&error), "G2000", "{statements:?}: {error}");
        assert_eq!(nodes(&session), vec!["Alix:Free"], "{statements:?}");
    }
    let (_db, session) = graph(Kind::Closed);
    session
        .execute("INSERT (:Far:Near {name: 'Alix', k: 1})")
        .unwrap();
    assert!(
        session
            .execute("MATCH (x {name: 'Alix'}) REMOVE x:Far")
            .is_err()
    );
    session
        .execute("MATCH (x {name: 'Alix'}) REMOVE x.k")
        .unwrap();
    session
        .execute("MATCH (x {name: 'Alix'}) REMOVE x:Far")
        .unwrap();
    assert_eq!(nodes(&session), vec!["Alix:Near"]);
}

/// The refusals name what the type says: a graph type violation.
#[test]
fn a_refused_removal_is_a_graph_type_violation() {
    for (data, statement) in [
        (
            "INSERT (:Free {name: 'Alix'})",
            "MATCH (x {name: 'Alix'}) REMOVE x:Free",
        ),
        (
            "INSERT (:Far:Near {name: 'Alix', k: 1})",
            "MATCH (x {name: 'Alix'}) REMOVE x:Far",
        ),
        (
            "INSERT (:Far:Free {name: 'Alix'})-[:R]->(:Near {name: 'Gus'})",
            "MATCH (x {name: 'Alix'}) REMOVE x:Far",
        ),
    ] {
        let (_db, session) = graph(Kind::Closed);
        session.execute(data).unwrap();
        let error = session.execute(statement).expect_err(statement);
        assert_eq!(status(&error), "G2000", "{statement}: {error}");
    }
}

/// `REMOVE` and a new edge that needs the label cannot both commit: the second gets a write conflict,
/// whichever commits first, and the rule holds.
#[test]
fn a_removal_and_a_new_edge_conflict_in_either_order() {
    for removal_first in [true, false] {
        let (db, setup) = graph(Kind::Untyped);
        setup
            .execute("INSERT (:Far:Free {name: 'Alix'}), (:Near {name: 'Gus'})")
            .unwrap();
        let mut removing = db.session();
        removing.execute("USE GRAPH g").unwrap();
        let mut linking = db.session();
        linking.execute("USE GRAPH g").unwrap();
        removing.begin_transaction().unwrap();
        linking.begin_transaction().unwrap();
        removing
            .execute("MATCH (x {name: 'Alix'}) REMOVE x:Far")
            .unwrap();
        // A statement may meet the conflict at once; then the removal stands.
        let linked =
            linking.execute("MATCH (x {name: 'Alix'}), (y {name: 'Gus'}) INSERT (x)-[:R]->(y)");
        let (first, second, kept) = if removal_first {
            (&mut removing, &mut linking, "removal")
        } else {
            (&mut linking, &mut removing, "edge")
        };
        if let Err(error) = linked {
            assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
            removing.commit().unwrap();
            linking.rollback().unwrap();
        } else {
            first
                .commit()
                .unwrap_or_else(|error| panic!("{kept} commits first: {error}"));
            let error = second.commit().expect_err("the second conflicts");
            assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
        }
        let edges = setup
            .execute("MATCH (x)-[:R]->(y) RETURN x.name")
            .unwrap()
            .row_count();
        let labels = nodes(&setup);
        if edges == 1 {
            assert!(labels.contains(&"Alix:Far,Free".to_string()), "{labels:?}");
        } else {
            assert!(labels.contains(&"Alix:Free".to_string()), "{labels:?}");
        }
    }
}

/// A label nothing depends on goes on a closed graph too.
#[test]
fn a_label_nothing_needs_goes_on_a_closed_graph() {
    for way in WAYS {
        let (db, session) = graph(Kind::Closed);
        run_all(
            &session,
            &[
                "INSERT (:Far:Free {name: 'Alix', k: 1})",
                "INSERT (:Far:Near {name: 'Gus', k: 2})",
            ],
        );
        remove(way, &db, &session, "Alix", "Free").unwrap();
        remove(way, &db, &session, "Gus", "Near").unwrap();
        assert_eq!(nodes(&session), vec!["Alix:Far", "Gus:Far"], "{way:?}");
    }
}

/// A label the node does not have changes nothing and is no error, on any graph.
#[test]
fn a_label_it_does_not_have_changes_nothing() {
    let (_db, session) = graph(Kind::Closed);
    run_all(
        &session,
        &[
            "INSERT (:Free {name: 'Alix'})",
            "MATCH (x {name: 'Alix'}) REMOVE x:Far",
        ],
    );
    assert_eq!(nodes(&session), vec!["Alix:Free"]);
}

/// In a transaction, a refused removal leaves the node as it was, and a rollback takes back one that
/// went.
#[test]
fn a_transaction_keeps_the_labels_it_had() {
    let (_db, mut session) = graph(Kind::Closed);
    run_all(
        &session,
        &[
            "INSERT (:Free {name: 'Alix'})",
            "INSERT (:Far:Free {name: 'Gus', k: 1})",
        ],
    );
    session.begin_transaction().unwrap();
    assert!(
        session
            .execute("MATCH (x {name: 'Alix'}) REMOVE x:Free")
            .is_err()
    );
    assert_eq!(nodes(&session), vec!["Alix:Free", "Gus:Far,Free"]);
    session.rollback().unwrap();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (x {name: 'Gus'}) REMOVE x:Free")
        .unwrap();
    assert_eq!(nodes(&session), vec!["Alix:Free", "Gus:Far"]);
    session.rollback().unwrap();
    assert_eq!(nodes(&session), vec!["Alix:Free", "Gus:Far,Free"]);
}

/// Two labels going at once is the check of what remains, not of each in turn.
#[test]
fn removing_two_labels_checks_what_remains() {
    let (_db, session) = graph(Kind::Closed);
    session
        .execute("INSERT (:Far:Free {name: 'Alix'})")
        .unwrap();
    assert!(
        session
            .execute("MATCH (x {name: 'Alix'}) REMOVE x:Far, x:Free")
            .is_err()
    );
    assert_eq!(nodes(&session), vec!["Alix:Far,Free"]);
}

/// A property on a node after another statement took its label's place is still checked: a statement
/// that adds the other label first can remove the first.
#[test]
fn a_label_added_first_lets_the_other_go() {
    let (_db, session) = graph(Kind::Closed);
    run_all(
        &session,
        &[
            "INSERT (:Free {name: 'Alix'})",
            "MATCH (x {name: 'Alix'}) SET x:Near",
            "MATCH (x {name: 'Alix'}) REMOVE x:Free",
        ],
    );
    assert_eq!(nodes(&session), vec!["Alix:Near"]);
}

// ---------------------------------------------------------------------------
// The other writers
// ---------------------------------------------------------------------------

fn tsv(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("edges.tsv");
    std::fs::write(&path, "1\t2\n2\t3\n").unwrap();
    path
}

fn count_nodes(session: &Session) -> usize {
    session.execute("MATCH (x) RETURN x").unwrap().row_count()
}

/// Each of the three import calls on one file's rows, each on a fresh database.
fn imports(
    dir: &std::path::Path,
    edge_type: &str,
) -> Vec<(
    &'static str,
    Result<(usize, usize), grafeo_common::utils::error::Error>,
    Session,
)> {
    (0..3)
        .map(|at| {
            let (db, session) = graph(Kind::Closed);
            let (call, outcome) = imports_one(&db, dir, edge_type, at);
            (call, outcome, session)
        })
        .collect()
}

/// A closed type that does not declare `_Imported` refuses with `G2000`, before anything is written.
#[test]
fn an_import_is_refused_by_a_closed_type_without_its_label() {
    let dir = tempfile::tempdir().unwrap();
    for (call, outcome, session) in imports(dir.path(), "R") {
        let error = outcome.expect_err(call);
        assert_eq!(status_of_error(&error), "G2000", "{call}: {error}");
        assert_eq!(count_nodes(&session), 0, "{call}");
    }
}

/// A closed type that declares it, and an edge type with no endpoint rule, imports as ever. An edge
/// type it does not declare, or whose endpoints `_Imported` does not fit, is refused.
#[test]
fn an_import_goes_on_where_a_closed_type_declares_its_shape() {
    let declaring = |edge_type_rule: &str| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for statement in [
            "CREATE NODE TYPE _Imported",
            "CREATE NODE TYPE Near (name STRING)",
            edge_type_rule,
            "CREATE EDGE TYPE OTHER",
            "CREATE GRAPH TYPE imported (NODE TYPE _Imported, NODE TYPE Near, EDGE TYPE LINK)",
            "CREATE GRAPH g TYPED imported",
            "USE GRAPH g",
        ] {
            session.execute(statement).unwrap();
        }
        db.set_current_graph(Some("g")).unwrap();
        (db, session)
    };
    for at in 0..3 {
        let dir = tempfile::tempdir().unwrap();
        let (db, session) = declaring("CREATE EDGE TYPE LINK");
        let (call, outcome) = imports_one(&db, dir.path(), "LINK", at);
        outcome.unwrap_or_else(|error| panic!("{call}: {error}"));
        assert_eq!(count_nodes(&session), 3, "{call}");

        let (db, session) = declaring("CREATE EDGE TYPE LINK");
        let (call, outcome) = imports_one(&db, dir.path(), "OTHER", at);
        assert_eq!(
            status_of_error(&outcome.expect_err(call)),
            "G2000",
            "{call}: undeclared type"
        );
        assert_eq!(count_nodes(&session), 0, "{call}");

        let (db, session) = declaring("CREATE EDGE TYPE LINK CONNECTING (Near) TO (Near)");
        let (call, outcome) = imports_one(&db, dir.path(), "LINK", at);
        assert!(outcome.is_err(), "{call}: endpoints it does not fit");
        assert_eq!(count_nodes(&session), 0, "{call}");
    }
}

/// One of the three calls, on a fresh database each time.
fn imports_one(
    db: &GrafeoDB,
    dir: &std::path::Path,
    edge_type: &str,
    at: usize,
) -> (
    &'static str,
    Result<(usize, usize), grafeo_common::utils::error::Error>,
) {
    let mtx = dir.join("edges.mtx");
    std::fs::write(
        &mtx,
        "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n",
    )
    .unwrap();
    match at {
        0 => ("import_tsv", db.import_tsv(tsv(dir), edge_type, true)),
        1 => (
            "import_tsv_str",
            db.import_tsv_str("1\t2\n2\t3\n", edge_type, true),
        ),
        _ => ("import_mmio", db.import_mmio(&mtx, edge_type)),
    }
}

/// On an open or untyped graph an import goes on as it did, through all three calls.
#[test]
fn an_import_goes_on_where_nothing_is_closed() {
    for kind in [Kind::Untyped, Kind::Open] {
        for at in 0..3 {
            let dir = tempfile::tempdir().unwrap();
            let (db, session) = graph(kind);
            let (call, outcome) = imports_one(&db, dir.path(), "LINK", at);
            outcome.unwrap_or_else(|error| panic!("{kind:?}, {call}: {error}"));
            assert_eq!(count_nodes(&session), 3, "{kind:?}, {call}");
        }
    }
}

/// The direct calls check the type: an undeclared label, property or edge type is refused, and what
/// is declared goes. Guards of what holds today.
#[test]
fn the_direct_calls_check_a_closed_type() {
    let (db, session) = graph(Kind::Closed);
    assert!(db.create_node(&["Undeclared"]).is_err());
    assert!(db.create_node(&[]).is_err(), "no label");
    assert!(
        db.create_node_with_props(&["Far"], [("undeclared", Value::Int64(1))])
            .is_err()
    );
    let far = db
        .create_node_with_props(&["Far"], [("k", Value::Int64(1))])
        .unwrap();
    let near = db.create_node(&["Near"]).unwrap();
    assert!(
        db.set_node_property(far, "undeclared", Value::Int64(1))
            .is_err()
    );
    assert!(db.add_node_label(far, "Undeclared").is_err());
    assert!(db.create_edge(far, near, "Undeclared").is_err());
    assert!(
        db.create_edge(near, far, "R").is_err(),
        "the wrong way round"
    );
    db.create_edge(far, near, "R").unwrap();
    assert_eq!(count_nodes(&session), 2);
}

/// The batch calls too.
#[test]
fn the_batch_calls_check_a_closed_type() {
    let (db, session) = graph(Kind::Closed);
    assert!(
        db.batch_create_nodes("Undeclared", "v", vec![vec![1.0_f32]])
            .is_err()
    );
    assert!(
        db.batch_create_nodes("Far", "undeclared", vec![vec![1.0_f32]])
            .is_err()
    );
    assert_eq!(count_nodes(&session), 0);
    let far = db.create_node(&["Far"]).unwrap();
    let near = db.create_node(&["Near"]).unwrap();
    let edge = |edge_type: &str, src: NodeId, dst: NodeId| BatchEdge {
        src,
        dst,
        edge_type: edge_type.to_string(),
        properties: HashMap::<PropertyKey, Value>::new(),
    };
    assert!(
        db.batch_create_edges(vec![edge("Undeclared", far, near)])
            .is_err()
    );
    assert!(db.batch_create_edges(vec![edge("R", near, far)]).is_err());
    db.batch_create_edges(vec![edge("R", far, near)]).unwrap();
}

/// The CSV loader, called and as a procedure, checks a closed type.
#[test]
fn the_csv_loader_checks_a_closed_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.csv");
    std::fs::write(&path, "id,name\n1,Alix\n2,Gus\n").unwrap();
    let load = |labels: &[&str]| NodeLoad {
        labels: labels.iter().map(ToString::to_string).collect(),
        ..NodeLoad::default()
    };
    let (db, session) = graph(Kind::Closed);
    assert!(
        db.load_nodes_csv(&path, &load(&["Undeclared"])).is_err(),
        "an undeclared label"
    );
    assert!(db.load_nodes_csv(&path, &load(&[])).is_err(), "no label");
    assert!(
        db.load_nodes_csv(&path, &load(&["Far"])).is_err(),
        "columns no label declares"
    );
    assert_eq!(count_nodes(&session), 0);
    db.load_nodes_csv(&path, &load(&["Free"])).unwrap();
    assert_eq!(count_nodes(&session), 2);

    let edges = dir.path().join("edges.csv");
    std::fs::write(&edges, "a,b\n1,2\n").unwrap();
    let end = |column: &str| EndKey {
        label: "Free".to_string(),
        property: "id".to_string(),
        column: column.to_string(),
    };
    let load_edges = |edge_type: &str| EdgeLoad {
        edge_type: edge_type.to_string(),
        source: end("a"),
        target: end("b"),
        ..EdgeLoad::default()
    };
    assert!(
        db.load_edges_csv(&edges, &load_edges("Undeclared"))
            .is_err()
    );
    assert!(
        db.load_edges_csv(&edges, &load_edges("R")).is_err(),
        "ends its type does not connect"
    );
    assert_eq!(
        session
            .execute("MATCH ()-[r]->() RETURN r")
            .unwrap()
            .row_count(),
        0
    );
}

/// Gremlin writes through the same check: an undeclared label, property and edge type are refused.
#[test]
fn gremlin_writes_check_a_closed_type() {
    let (_db, session) = graph(Kind::Closed);
    for query in [
        "g.addV('Undeclared')",
        "g.addV('Far').property('undeclared', 1)",
    ] {
        assert!(session.execute_gremlin(query).is_err(), "{query}");
        assert_eq!(count_nodes(&session), 0, "{query}");
    }
    session
        .execute_gremlin("g.addV('Far').property('name', 'Alix')")
        .unwrap();
    session
        .execute_gremlin("g.addV('Near').property('name', 'Gus')")
        .unwrap();
    let id = |name: &str| id_of(&session, name).as_u64();
    let (alix, gus) = (id("Alix"), id("Gus"));
    for query in [
        format!("g.V({alix}).addE('Undeclared').to(g.V({gus}))"),
        format!("g.V({gus}).addE('R').to(g.V({alix}))"),
    ] {
        assert!(session.execute_gremlin(&query).is_err(), "{query}");
    }
    let edges = session.execute("MATCH ()-[r]->() RETURN r").unwrap();
    assert_eq!(edges.row_count(), 0);
}

/// GraphQL mutations too.
#[test]
fn graphql_writes_check_a_closed_type() {
    let (_db, session) = graph(Kind::Closed);
    for query in [
        r#"mutation { createUndeclared(name: "Alix") { name } }"#,
        r#"mutation { createFar(undeclared: "Alix") { name } }"#,
    ] {
        assert!(session.execute_graphql(query).is_err(), "{query}");
        assert_eq!(count_nodes(&session), 0, "{query}");
    }
    session
        .execute_graphql(r#"mutation { createFar(name: "Alix") { name } }"#)
        .unwrap();
    assert_eq!(nodes(&session), vec!["Alix:Far"]);
}

/// The two procedures that write check it as the calls do. No other procedure writes.
#[test]
fn the_writing_procedures_check_a_closed_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.csv");
    std::fs::write(&path, "id,name\n1,Alix\n2,Gus\n").unwrap();
    let path = path.display().to_string().replace(chr_backslash(), "/");
    let (_db, session) = graph(Kind::Closed);
    for labels in ["Undeclared", "Far"] {
        let call = format!("CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['{labels}']}})");
        assert!(session.execute(&call).is_err(), "{call}");
        assert_eq!(count_nodes(&session), 0, "{call}");
    }
    session
        .execute(&format!(
            "CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Free']}})"
        ))
        .unwrap();
    assert_eq!(count_nodes(&session), 2);
    let edges = dir.path().join("edges.csv");
    std::fs::write(&edges, "a,b\n1,2\n").unwrap();
    let edges = edges.display().to_string().replace(chr_backslash(), "/");
    let call = format!(
        "CALL grafeo.load_edges_csv({{path: '{edges}', type: 'Undeclared', \
         source: {{label: 'Free', property: 'id', column: 'a'}}, \
         target: {{label: 'Free', property: 'id', column: 'b'}}}})"
    );
    assert!(session.execute(&call).is_err(), "{call}");
}

fn chr_backslash() -> char {
    char::from(92_u8)
}

/// Replay installs what was committed: a closed graph comes back as it was.
#[test]
fn a_reopen_keeps_a_closed_graph_as_committed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        for statement in TYPES {
            session.execute(statement).unwrap();
        }
        run_all(
            &session,
            &[
                "CREATE GRAPH TYPE atlas (NODE TYPE Far, NODE TYPE Near, NODE TYPE Free, EDGE TYPE R)",
                "CREATE GRAPH g TYPED atlas",
                "USE GRAPH g",
                "INSERT (:Far:Free {name: 'Alix', k: 1})",
                "MATCH (x {name: 'Alix'}) REMOVE x:Free",
            ],
        );
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("USE GRAPH g").unwrap();
    assert_eq!(nodes(&session), vec!["Alix:Far"]);
    assert!(
        session
            .execute("MATCH (x {name: 'Alix'}) REMOVE x:Far")
            .is_err()
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// One graph type body, two forms
// ---------------------------------------------------------------------------

/// A body is `NODE TYPE` and `EDGE TYPE` items in parentheses, or patterns in braces. The two do not
/// mix, and the error says so at the item.
#[test]
fn a_graph_type_body_does_not_mix_its_two_forms() {
    for body in [
        "(NODE TYPE A, NODE TYPE B, EDGE TYPE X, (:A)-[:Y]->(:B))",
        "(EDGE TYPE X, (:A)-[:Y]->(:B))",
        "{ (:A)-[:Y]->(:B), EDGE TYPE X }",
        "{ (:A)-[:Y]->(:B), NODE TYPE C }",
        "{ EDGE TYPE X, (:A)-[:Y]->(:B) }",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for statement in [
            "CREATE NODE TYPE A",
            "CREATE NODE TYPE B",
            "CREATE NODE TYPE C",
            "CREATE EDGE TYPE X",
        ] {
            session.execute(statement).unwrap();
        }
        let statement = format!("CREATE GRAPH TYPE mixed {body}");
        assert!(session.execute(&statement).is_err(), "{statement}");
    }
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE GRAPH TYPE patterns { (:A)-[:Y]->(:B), (:C) }")
        .unwrap();
}
