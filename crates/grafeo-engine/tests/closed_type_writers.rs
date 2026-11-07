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
use std::path::{Path, PathBuf};

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
        #[cfg(feature = "cypher")]
        Way::Cypher => session
            .execute_cypher(&text)
            .map(|_| true)
            .map_err(|e| e.to_string()),
        // With no Cypher to write it in, the same statement in GQL.
        #[cfg(not(feature = "cypher"))]
        Way::Cypher => session
            .execute(&text)
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
#[cfg(feature = "gremlin")]
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
#[cfg(feature = "graphql")]
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

/// A closed graph that breaks its type, in a file from 0.6.1, which let it happen
/// (`scripts/broken_closed_type_fixture.py`). A copy, opened; `g` current.
fn broken_file() -> (tempfile::TempDir, GrafeoDB, Session) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.grafeo");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/closed_type_broken/0.6.1.grafeo"),
        &path,
    )
    .unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("USE GRAPH g").unwrap();
    (dir, db, session)
}

/// Nothing is checked until a write: the file opens, and every node reads as it is.
#[test]
fn a_file_that_breaks_its_type_opens_and_reads_as_it_is() {
    let (_dir, db, session) = broken_file();
    assert_eq!(
        nodes(&session),
        vec![
            "Alix:Far,Free",
            "Gus:",
            "Jules:Near",
            "Mia:Gone",
            "Vincent:Free"
        ]
    );
    let read = session
        .execute("MATCH (x {name: 'Alix'}) RETURN x.extra")
        .unwrap();
    assert_eq!(read.rows()[0][0], Value::Int64(1));
    let read = session
        .execute("MATCH (x {name: 'Mia'}) RETURN x.anything")
        .unwrap();
    assert_eq!(read.rows()[0][0], Value::Int64(7));
    let edges = session
        .execute("MATCH ({name: 'Vincent'})-[:R]->({name: 'Jules'}) RETURN 1")
        .unwrap();
    assert_eq!(edges.row_count(), 1);
    db.close().unwrap();
}

/// A removal on a node that already breaks the type is refused even where the label it takes is not
/// the cause, since what the statement leaves does not fit. The node stays readable, and it can be
/// mended first.
#[test]
fn a_node_that_already_breaks_the_type_is_refused_a_removal_until_it_is_mended() {
    let (_dir, db, session) = broken_file();
    let error = session
        .execute("MATCH (x {name: 'Alix'}) REMOVE x:Free")
        .expect_err("what is left does not fit");
    assert_eq!(status(&error), "G2000", "{error}");
    assert!(nodes(&session).contains(&"Alix:Far,Free".to_string()));
    run_all(
        &session,
        &[
            "MATCH (x {name: 'Alix'}) REMOVE x.extra",
            "MATCH (x {name: 'Alix'}) REMOVE x:Free",
        ],
    );
    assert!(nodes(&session).contains(&"Alix:Far".to_string()));
    db.close().unwrap();
}

/// A write that does not touch what is broken goes on, and a reopen keeps the file as it was.
#[test]
fn other_writes_go_on_in_a_file_that_breaks_its_type() {
    let (dir, db, session) = broken_file();
    run_all(&session, &["INSERT (:Near {name: 'Hans'})"]);
    db.close().unwrap();
    let db = GrafeoDB::open(dir.path().join("broken.grafeo")).unwrap();
    let session = db.session();
    session.execute("USE GRAPH g").unwrap();
    assert!(nodes(&session).contains(&"Hans:Near".to_string()));
    assert!(nodes(&session).contains(&"Gus:".to_string()));
    db.close().unwrap();
}

/// A listed name with no definition, from an older file: making it again is checked as a change of
/// the type is. Fewer properties than the data holds are refused, and enough are accepted.
#[test]
fn making_a_dropped_type_again_is_checked_against_the_data() {
    let (_dir, db, session) = broken_file();
    let error = session
        .execute("CREATE NODE TYPE Gone (name STRING)")
        .expect_err("`anything` is held");
    assert_eq!(status(&error), "G2000", "{error}");
    run_all(
        &session,
        &["CREATE NODE TYPE Gone (name STRING, anything INTEGER)"],
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// A change of a type that the data on a closed graph would break
// ---------------------------------------------------------------------------

/// A closed graph whose node and edge hold what `Held`, `Only` and `Link` declare.
const HELD_TYPES: &[&str] = &[
    "CREATE NODE TYPE Held (name STRING, extra INTEGER, k INTEGER)",
    "CREATE NODE TYPE Other (k INTEGER)",
    "CREATE NODE TYPE Only (x INTEGER)",
    "CREATE EDGE TYPE Link (w INTEGER, z INTEGER)",
    "CREATE EDGE TYPE Lone (w INTEGER)",
];
const HELD_DATA: &str = "INSERT (:Held {name: 'Alix', extra: 1})-[:Link {w: 1}]->(:Only {x: 1})-[:Lone {w: 2}]->(:Other)";

fn held(kind: Kind) -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, HELD_TYPES);
    let graph_statements: &[&str] = match kind {
        Kind::Closed => &[
            "CREATE GRAPH TYPE held (NODE TYPE Held, NODE TYPE Other, NODE TYPE Only, EDGE TYPE Link, EDGE TYPE Lone)",
            "CREATE GRAPH g TYPED held",
        ],
        Kind::Open => &[
            "CREATE GRAPH TYPE held { node_types: [Held, Other, Only], edge_types: [Link, Lone], open: true }",
            "CREATE GRAPH g TYPED held",
        ],
        Kind::Untyped => &["CREATE GRAPH g"],
    };
    run_all(&session, graph_statements);
    run_all(&session, &["USE GRAPH g", HELD_DATA]);
    (db, session)
}

/// The changes that would leave a held value undeclared, in each form.
const BREAKING: &[&str] = &[
    "ALTER NODE TYPE Held DROP PROPERTY extra",
    "ALTER NODE TYPE Only DROP PROPERTY x",
    "ALTER EDGE TYPE Link DROP PROPERTY w",
    "ALTER EDGE TYPE Lone DROP PROPERTY w",
    "CREATE OR REPLACE NODE TYPE Held (name STRING)",
    "CREATE OR REPLACE NODE TYPE Only",
    "CREATE OR REPLACE EDGE TYPE Link (z INTEGER)",
    "CREATE OR REPLACE EDGE TYPE Lone",
];

/// While the data on a closed graph breaks it, the change is refused with `G2000`, and the type is as
/// it was: a value it declared can still be written.
#[test]
fn a_type_change_the_data_breaks_is_refused() {
    for statement in BREAKING {
        let (_db, session) = held(Kind::Closed);
        let error = session.execute(statement).expect_err(statement);
        assert_eq!(status(&error), "G2000", "{statement}: {error}");
        run_all(
            &session,
            &[
                "INSERT (:Held {name: 'Gus', extra: 2})",
                "INSERT (:Only {x: 2})",
                "MATCH (a:Held {name: 'Gus'}), (b:Only {x: 2}) INSERT (a)-[:Link {w: 3}]->(b), (b)-[:Lone {w: 4}]->(a)",
            ],
        );
    }
}

/// Run from any graph: what counts is the graphs bound to the type, not the current one.
#[test]
fn a_type_change_is_refused_from_another_graph() {
    for statement in BREAKING {
        let (_db, session) = held(Kind::Closed);
        run_all(&session, &["USE GRAPH default"]);
        let error = session.execute(statement).expect_err(statement);
        assert_eq!(status(&error), "G2000", "{statement}: {error}");
    }
}

/// The ways of dropping a type that are refused already.
#[test]
fn dropping_from_a_bound_graph_type_stays_refused() {
    for statement in [
        "ALTER GRAPH TYPE held DROP NODE TYPE Held",
        "ALTER GRAPH TYPE held DROP EDGE TYPE Link",
    ] {
        let (_db, session) = held(Kind::Closed);
        let error = session.execute(statement).expect_err(statement);
        assert_eq!(status(&error), "G2000", "{statement}: {error}");
    }
    for statement in [
        "CREATE OR REPLACE GRAPH TYPE held (NODE TYPE Other)",
        "CREATE OR REPLACE GRAPH TYPE held (NODE TYPE Other, NODE TYPE Only, EDGE TYPE Lone)",
    ] {
        let (_db, session) = held(Kind::Closed);
        assert!(session.execute(statement).is_err(), "{statement}");
    }
}

/// A type a closed graph type lists, with a graph bound to it, stays while the graph does, whether or
/// not the graph holds anything of it: the graph type would name what no longer exists.
#[test]
fn dropping_a_listed_type_is_refused_while_a_closed_graph_is_bound() {
    for statement in [
        "DROP NODE TYPE Held",
        "DROP NODE TYPE Other",
        "DROP EDGE TYPE Link",
        "DROP EDGE TYPE Lone",
    ] {
        let (_db, session) = held(Kind::Closed);
        let error = session.execute(statement).expect_err(statement).to_string();
        assert!(error.contains("closed graph type"), "{statement}: {error}");
        let read = session
            .execute("MATCH (a:Held)-[r:Link]->() RETURN a.extra, r.w")
            .unwrap();
        assert_eq!(read.row_count(), 1, "{statement}: nothing moved");
        run_all(&session, &["INSERT (:Held {name: 'Gus', extra: 2})"]);
    }
    // With no data at all.
    for statement in ["DROP NODE TYPE Held", "DROP EDGE TYPE Link"] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, HELD_TYPES);
        run_all(
            &session,
            &[
                "CREATE GRAPH TYPE held (NODE TYPE Held, NODE TYPE Other, NODE TYPE Only, EDGE TYPE Link, EDGE TYPE Lone)",
                "CREATE GRAPH g TYPED held",
            ],
        );
        let error = session.execute(statement).expect_err(statement).to_string();
        assert!(
            error.contains("closed graph type"),
            "{statement}, with no data: {error}"
        );
    }
}

/// An open type's, an unbound type's and an unlisted type's definitions go as they did.
#[test]
fn dropping_a_type_goes_where_nothing_closed_is_bound_to_it() {
    for kind in [Kind::Open, Kind::Untyped] {
        for statement in ["DROP NODE TYPE Held", "DROP EDGE TYPE Link"] {
            let (_db, session) = held(kind);
            session
                .execute(statement)
                .unwrap_or_else(|error| panic!("{kind:?}, {statement}: {error}"));
        }
    }
    // A closed type no graph is bound to.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, HELD_TYPES);
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE held (NODE TYPE Held, EDGE TYPE Link)",
            "DROP NODE TYPE Held",
            "DROP EDGE TYPE Link",
        ],
    );
    // A type the bound closed type does not list.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, HELD_TYPES);
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE held (NODE TYPE Held, EDGE TYPE Link)",
            "CREATE GRAPH g TYPED held",
            "DROP NODE TYPE Other",
            "DROP EDGE TYPE Lone",
        ],
    );
}

/// On an open type or an untyped graph, the same changes go, and the values stay readable.
#[test]
fn a_type_change_goes_on_where_nothing_is_closed() {
    for kind in [Kind::Open, Kind::Untyped] {
        for statement in BREAKING {
            let (_db, session) = held(kind);
            session
                .execute(statement)
                .unwrap_or_else(|error| panic!("{kind:?}, {statement}: {error}"));
            let read = session.execute("MATCH (a:Held) RETURN a.extra").unwrap();
            assert_eq!(read.rows()[0][0], Value::Int64(1), "{kind:?}, {statement}");
        }
    }
}

/// A property no node holds goes, and one another label declares goes: the data still fits.
#[test]
fn a_type_change_the_data_does_not_break_goes() {
    let (_db, session) = held(Kind::Closed);
    run_all(&session, &["ALTER NODE TYPE Held DROP PROPERTY k"]);
    let (_db, session) = held(Kind::Closed);
    run_all(&session, &["ALTER EDGE TYPE Link DROP PROPERTY z"]);
    let (_db, session) = held(Kind::Closed);
    run_all(
        &session,
        &[
            "INSERT (:Held:Other {name: 'Gus', k: 7})",
            "ALTER NODE TYPE Held DROP PROPERTY k",
        ],
    );
    let read = session
        .execute("MATCH (a:Other {k: 7}) RETURN a.name")
        .unwrap();
    assert_eq!(read.rows()[0][0], Value::from("Gus"));
    // Once the last holder lets it go, it can be dropped.
    let (_db, session) = held(Kind::Closed);
    run_all(
        &session,
        &[
            "MATCH (a:Held) REMOVE a.extra",
            "ALTER NODE TYPE Held DROP PROPERTY extra",
        ],
    );
}

/// A transaction that wrote the property, and commits after another's change of the type, does not
/// commit a value the type no longer declares.
#[test]
fn a_write_in_progress_fails_after_the_type_lost_its_property() {
    let (db, session) = held(Kind::Closed);
    run_all(&session, &["MATCH (a:Held) REMOVE a.extra"]);
    let mut writer = db.session();
    writer.execute("USE GRAPH g").unwrap();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Held {name: 'Gus', extra: 2})")
        .unwrap();
    // Decided (#728 review): the change goes through, as the writer's value is not data yet, and the
    // writer's commit checks its writes against the type as it stands then.
    session
        .execute("ALTER NODE TYPE Held DROP PROPERTY extra")
        .unwrap();
    assert!(writer.commit().is_err(), "the value is no longer declared");
    let read = session
        .execute("MATCH (a:Held {name: 'Gus'}) RETURN a")
        .unwrap();
    assert_eq!(read.row_count(), 0);
}

/// A change inside a transaction that rolls back changes nothing, and one refused there leaves the
/// transaction's type as it was.
#[test]
fn a_type_change_in_a_transaction_that_rolls_back_changes_nothing() {
    let (_db, mut session) = held(Kind::Closed);
    session.begin_transaction().unwrap();
    assert!(
        session
            .execute("ALTER NODE TYPE Held DROP PROPERTY extra")
            .is_err()
    );
    session.rollback().unwrap();
    run_all(&session, &["INSERT (:Held {name: 'Gus', extra: 2})"]);

    let (_db, mut session) = held(Kind::Closed);
    session.begin_transaction().unwrap();
    run_all(&session, &["ALTER NODE TYPE Held DROP PROPERTY k"]);
    session.rollback().unwrap();
    run_all(&session, &["INSERT (:Held {name: 'Gus', k: 2})"]);
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
