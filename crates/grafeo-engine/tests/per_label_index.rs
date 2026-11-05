//! A property index per label: `CREATE INDEX ... FOR (n:L) ON (n.k)` and a rule's index hold only the
//! nodes that have the label, in some version a reader can see.
//!
//! Tests first, from the ruling. The main risk is a label changing under an index: `SET n:L`,
//! `REMOVE n:L`, an undo of either, collection, and an index on a label no node has yet. Whatever
//! happens to an entry, the rows are those of a plan that reads no index, now and at an older view:
//! a stale entry may cost memory and reads, never a row. A lookup uses a per-label index only of a
//! label every result must have, then a key-wide one, then the scan.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main per_label_index::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const INDEX: &str = "CREATE INDEX by_k FOR (n:L) ON (n.k)";

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

fn with_history() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Forever))
        .unwrap()
}

fn sorted(session: &Session, query: &str) -> Vec<String> {
    let result = session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn names(wanted: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = wanted
        .iter()
        .map(|name| format!("{:?}", vec![Value::from(*name)]))
        .collect();
    rows.sort();
    rows
}

/// Every way of asking for `k = value` on `label`: a constant, a pattern after a clause, a key from
/// the rows.
fn lookups(label: &str, value: i64) -> [String; 4] {
    [
        format!("MATCH (n:{label} {{k: {value}}}) RETURN n.name"),
        format!("MATCH (n:{label}) WHERE n.k = {value} RETURN n.name"),
        format!("UNWIND [{value}] AS x MATCH (n:{label}) WHERE n.k = x RETURN n.name"),
        format!("UNWIND [1] AS y MATCH (n:{label}) WHERE n.k = {value} RETURN n.name"),
    ]
}

/// The rows of each lookup, with the usual plan and one that reads no index, are `wanted`.
fn assert_found(db: &GrafeoDB, label: &str, value: i64, wanted: &[&str], at: Option<EpochId>) {
    let usual = db.session();
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    if let Some(epoch) = at {
        usual.set_viewing_epoch(epoch);
        scan.set_viewing_epoch(epoch);
    }
    let wanted = names(wanted);
    for query in lookups(label, value) {
        assert_eq!(
            sorted(&scan, &query),
            wanted,
            "no index, at {at:?}: {query}"
        );
        assert_eq!(sorted(&usual, &query), wanted, "usual, at {at:?}: {query}");
    }
}

/// Whether a key from the rows finds `pattern`'s nodes through an index on `k`.
fn reads_index(session: &Session, pattern: &str) -> bool {
    plan_of(
        session,
        &format!("UNWIND [1] AS x MATCH (n{pattern}) WHERE n.k = x RETURN n.k"),
    )
    .contains("[index: k]")
}

fn plan_of(session: &Session, query: &str) -> String {
    session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
        .as_str()
        .unwrap()
        .to_string()
}

fn index_bytes(db: &GrafeoDB) -> usize {
    db.memory_usage().indexes.property_index_bytes
}

fn node_of(db: &GrafeoDB, name: &str) -> NodeId {
    let result = db
        .execute(&format!("MATCH (n {{name: '{name}'}}) RETURN id(n)"))
        .unwrap();
    match result.rows()[0][0] {
        Value::Int64(id) => NodeId::new(id.cast_unsigned()),
        ref other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// What the index holds
// ---------------------------------------------------------------------------

/// The reason for this work: writes of another label with the key pay nothing for the index.
#[test]
fn an_index_holds_only_its_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX]);
    let empty = index_bytes(&db);
    run_all(
        &session,
        &["UNWIND range(1, 2000) AS i INSERT (:Other {k: i})"],
    );
    assert_eq!(index_bytes(&db), empty, "another label's writes");
    run_all(
        &session,
        &["MATCH (n:Other) WHERE n.k < 1000 SET n.k = n.k + 5000"],
    );
    assert_eq!(index_bytes(&db), empty, "another label's updates");
    run_all(&session, &["UNWIND range(1, 2000) AS i INSERT (:L {k: i})"]);
    assert!(index_bytes(&db) > empty, "its own label's writes");
}

/// A build reads the label's rows, with what they held in older versions.
#[test]
fn a_build_finds_what_stood_before_it() {
    let db = with_history();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:L {name: 'Alix', k: 1}), (:Other {name: 'Gus', k: 1})",
            "INSERT (:L:M {name: 'Vincent', k: 1})",
        ],
    );
    let before = db.current_epoch();
    run_all(&session, &["MATCH (n {name: 'Alix'}) SET n.k = 2"]);
    run_all(&session, &[INDEX]);
    assert!(reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Vincent"], None);
    assert_found(&db, "L", 2, &["Alix"], None);
    assert_found(&db, "L", 1, &["Alix", "Vincent"], Some(before));
    assert_found(&db, "Other", 1, &["Gus"], None);
}

// ---------------------------------------------------------------------------
// A label that changes
// ---------------------------------------------------------------------------

#[test]
fn a_label_added_brings_the_node_into_the_index() {
    let db = with_history();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"]);
    let before = db.current_epoch();
    assert_found(&db, "L", 1, &[], None);
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "L", 1, &[], Some(before));
    assert_found(&db, "Other", 1, &["Alix"], None);
    // It is kept up from then on.
    run_all(&session, &["MATCH (n:L) SET n.k = 2"]);
    assert_found(&db, "L", 2, &["Alix"], None);
    assert_found(&db, "L", 1, &[], None);
}

/// A label added enters the values the node holds in older versions too: a reader between them
/// sees the node with the label and the older value.
#[test]
fn a_label_added_enters_the_values_of_older_versions_a_reader_can_see() {
    let db = with_history();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"]);
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    let with_label = db.current_epoch();
    run_all(&session, &["MATCH (n:L) SET n.k = 2"]);
    assert_found(&db, "L", 1, &["Alix"], Some(with_label));
    assert_found(&db, "L", 2, &[], Some(with_label));
    assert_found(&db, "L", 2, &["Alix"], None);

    // The value changed first, the label after: the older value is no one's with the label.
    let db = with_history();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Gus', k: 1})"]);
    run_all(&session, &["MATCH (n:Other) SET n.k = 2"]);
    let value_changed = db.current_epoch();
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "L", 2, &["Gus"], None);
    assert_found(&db, "L", 2, &[], Some(value_changed));
}

#[test]
fn a_label_removed_leaves_the_rows_and_stays_for_older_snapshots() {
    let db = with_history();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
    let with_label = db.current_epoch();
    run_all(&session, &["MATCH (n:L) REMOVE n:L"]);
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "L", 1, &["Alix"], Some(with_label));
    assert_found(&db, "Other", 1, &["Alix"], None);
    // Its values move on with no label, and the older reader's stay.
    run_all(&session, &["MATCH (n:Other) SET n.k = 2"]);
    assert_found(&db, "L", 2, &[], None);
    assert_found(&db, "L", 1, &["Alix"], Some(with_label));
}

/// Added, removed and added again, each with a reader in between.
#[test]
fn a_label_that_comes_and_goes_answers_at_each_view() {
    let db = with_history();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"]);
    let none = db.current_epoch();
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    let first = db.current_epoch();
    run_all(&session, &["MATCH (n:L) REMOVE n:L"]);
    let gone = db.current_epoch();
    run_all(&session, &["MATCH (n:Other) SET n.k = 2"]);
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    assert_found(&db, "L", 1, &[], Some(none));
    assert_found(&db, "L", 1, &["Alix"], Some(first));
    assert_found(&db, "L", 1, &[], Some(gone));
    assert_found(&db, "L", 2, &["Alix"], None);
    assert_found(&db, "L", 1, &[], None);
}

/// The direct API's label calls keep the index as a statement does.
#[test]
fn the_direct_api_keeps_the_index_when_a_label_changes() {
    let db = GrafeoDB::new_in_memory();
    run_all(
        &db.session(),
        &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"],
    );
    let id = node_of(&db, "Alix");
    assert!(db.add_node_label(id, "L").unwrap());
    assert_found(&db, "L", 1, &["Alix"], None);
    assert!(db.remove_node_label(id, "L").unwrap());
    assert_found(&db, "L", 1, &[], None);
    db.create_node_with_props(
        &["L"],
        [("name", Value::from("Gus")), ("k", Value::Int64(1))],
    )
    .unwrap();
    db.batch_create_nodes_with_props(
        "L",
        vec![std::collections::HashMap::from([
            ("name".into(), Value::from("Mia")),
            ("k".into(), Value::Int64(1)),
        ])],
    )
    .unwrap();
    assert_found(&db, "L", 1, &["Gus", "Mia"], None);
}

/// A node's several labels: an index of each holds it for as long as it has that label.
#[test]
fn a_node_of_two_labels_sits_in_the_index_of_each() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "CREATE INDEX by_k_m FOR (n:M) ON (n.k)",
            "INSERT (:L:M {name: 'Alix', k: 1})",
        ],
    );
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "M", 1, &["Alix"], None);
    run_all(&session, &["MATCH (n:M) REMOVE n:L"]);
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "M", 1, &["Alix"], None);
    run_all(&session, &["MATCH (n:M) SET n.k = 2"]);
    assert_found(&db, "M", 2, &["Alix"], None);
    assert_found(&db, "L", 2, &[], None);
}

/// A deleted node, and a value removed: gone at the latest view, there at an older one.
#[test]
fn a_delete_and_a_removed_value_keep_older_readers() {
    let db = with_history();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "INSERT (:L {name: 'Alix', k: 1}), (:L {name: 'Gus', k: 1})",
        ],
    );
    let before = db.current_epoch();
    run_all(
        &session,
        &[
            "MATCH (n {name: 'Alix'}) DELETE n",
            "MATCH (n {name: 'Gus'}) REMOVE n.k",
        ],
    );
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "L", 1, &["Alix", "Gus"], Some(before));
}

// ---------------------------------------------------------------------------
// An undo
// ---------------------------------------------------------------------------

/// An undone add leaves no entry behind: after the undo the row has no chain, so collection would
/// never visit it. Bytes back where they stood is what a stale entry cannot do.
#[test]
fn an_undone_label_add_leaves_no_entry() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "UNWIND range(1, 2000) AS i INSERT (:Other {name: 'n' + toString(i), k: i})",
        ],
    );
    let before = index_bytes(&db);
    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    assert!(index_bytes(&db) > before, "the transaction's add");
    session.rollback().unwrap();
    assert_eq!(index_bytes(&db), before, "after the undo");
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "Other", 1, &["n1"], None);
}

#[test]
fn an_undone_label_remove_keeps_the_rows() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n:Other) REMOVE n:L"]);
    session.rollback().unwrap();
    assert_found(&db, "L", 1, &["Alix"], None);
    // Still kept up.
    run_all(&session, &["MATCH (n:L) SET n.k = 2"]);
    assert_found(&db, "L", 2, &["Alix"], None);
    assert_found(&db, "L", 1, &[], None);
}

/// Back to a savepoint, an add of the label in the same transaction as a value: the rows of the
/// commit are those of what stayed.
#[test]
fn a_savepoint_takes_back_a_label_and_keeps_the_rest() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"]);
    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n:Other) SET n.k = 2"]);
    session.savepoint("before").unwrap();
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    session.rollback_to_savepoint("before").unwrap();
    session.commit().unwrap();
    assert_found(&db, "L", 2, &[], None);
    assert_found(&db, "Other", 2, &["Alix"], None);
}

/// In one transaction: the label added and the value changed, then all undone; and the reads of the
/// transaction itself see its own label through the index.
#[test]
fn a_transaction_reads_its_own_label_changes() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "INSERT (:Other {name: 'Alix', k: 1}), (:L {name: 'Gus', k: 1})",
        ],
    );
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "MATCH (n {name: 'Alix'}) SET n:L",
            "MATCH (n {name: 'Gus'}) REMOVE n:L",
        ],
    );
    assert!(reads_index(&session, ":L"));
    for query in lookups("L", 1) {
        assert_eq!(sorted(&session, &query), names(&["Alix"]), "{query}");
    }
    // Others see what was committed.
    assert_found(&db, "L", 1, &["Gus"], None);
    session.rollback().unwrap();
    assert_found(&db, "L", 1, &["Gus"], None);
}

/// A second session's view while a transaction holds an uncommitted add of the label.
#[test]
fn an_open_transactions_label_is_seen_by_no_one_else() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"]);
    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    assert_found(&db, "L", 1, &[], None);
    session.commit().unwrap();
    assert_found(&db, "L", 1, &["Alix"], None);
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// With no reader left, a removed label's entries go: stale ones would cost memory for good.
#[test]
fn collection_drops_the_entries_of_a_label_no_reader_sees() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX]);
    let empty = index_bytes(&db);
    run_all(
        &session,
        &["UNWIND range(1, 2000) AS i INSERT (:L:Other {name: 'n' + toString(i), k: i})"],
    );
    assert!(index_bytes(&db) > empty);
    run_all(&session, &["MATCH (n:Other) REMOVE n:L"]);
    assert_found(&db, "L", 1, &[], None);
    db.gc();
    assert_eq!(index_bytes(&db), empty, "nothing left to read");
    assert_found(&db, "L", 1, &[], None);
    assert_found(&db, "Other", 1, &["n1"], None);
}

/// Collection keeps what an open reader can still see.
#[test]
fn collection_keeps_a_removed_label_for_an_open_snapshot() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(
        sorted(&reader, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
        names(&["Alix"])
    );
    run_all(&session, &["MATCH (n:Other) REMOVE n:L"]);
    db.gc();
    for query in lookups("L", 1) {
        assert_eq!(sorted(&reader, &query), names(&["Alix"]), "{query}");
    }
    reader.commit().unwrap();
    db.gc();
    assert_found(&db, "L", 1, &[], None);
}

/// A label removed and added back, then collected: the node stays in the index.
#[test]
fn collection_keeps_a_label_added_back() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
    run_all(&session, &["MATCH (n:Other) REMOVE n:L"]);
    run_all(&session, &["MATCH (n:Other) SET n:L"]);
    db.gc();
    assert_found(&db, "L", 1, &["Alix"], None);
    run_all(&session, &["MATCH (n:L) SET n.k = 2"]);
    db.gc();
    assert_found(&db, "L", 2, &["Alix"], None);
    assert_found(&db, "L", 1, &[], None);
}

/// An older value of a node that keeps its label is collected as before.
#[test]
fn collection_drops_older_values_and_keeps_the_newest() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L {name: 'Alix', k: 0})"]);
    let one = index_bytes(&db);
    for value in 1..200 {
        run_all(&session, &[&format!("MATCH (n:L) SET n.k = {value}")]);
    }
    db.gc();
    assert_eq!(index_bytes(&db), one, "one value left");
    assert_found(&db, "L", 199, &["Alix"], None);
}

// ---------------------------------------------------------------------------
// A label no node has yet
// ---------------------------------------------------------------------------

fn label_names(db: &GrafeoDB) -> Vec<String> {
    let mut labels: Vec<String> = db
        .execute("CALL db.labels() YIELD label RETURN label")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].as_str().unwrap().to_string())
        .collect();
    labels.sort();
    labels
}

/// The index does not make the label seen in listings or statistics, and catches what comes.
#[test]
fn an_index_on_a_label_no_node_has_stays_out_of_sight_and_catches_later_writes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["INSERT (:Other {name: 'Alix', k: 1})"]);
    let labels = label_names(&db);
    let schema = format!("{:?}", db.schema().unwrap());
    run_all(&session, &[INDEX]);
    assert_eq!(label_names(&db), labels, "CALL db.labels()");
    assert_eq!(format!("{:?}", db.schema().unwrap()), schema, "schema()");
    assert_eq!(
        sorted(&session, "SHOW INDEXES").len(),
        1,
        "the index itself is listed"
    );
    run_all(&session, &["INSERT (:L {name: 'Gus', k: 1})"]);
    assert_found(&db, "L", 1, &["Gus"], None);
    assert!(label_names(&db).contains(&"L".to_string()));
}

/// The same across a reopen: the label comes back with no node, and still catches a write.
#[test]
fn an_index_on_a_label_no_node_has_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty-label.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    run_all(
        &db.session(),
        &[INDEX, "INSERT (:Other {name: 'Alix', k: 1})"],
    );
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(!label_names(&db).contains(&"L".to_string()));
    run_all(&session, &["INSERT (:L {name: 'Gus', k: 1})"]);
    assert!(reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Gus"], None);
    assert_found(&db, "Other", 1, &["Alix"], None);
    drop(session);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Which lookups use it
// ---------------------------------------------------------------------------

/// A label every result must have uses the index; anything else reads a key-wide one or scans, and
/// the rows are the same.
#[test]
fn a_lookup_uses_the_index_only_of_a_label_every_result_must_have() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "INSERT (:L {name: 'Alix', k: 1}), (:Other {name: 'Gus', k: 1}), (:L:M {name: 'Mia', k: 1}), (:M {name: 'Jules', k: 1})",
        ],
    );
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    for (condition, uses, rows) in [
        ("(n:L)", true, vec!["Alix", "Mia"]),
        ("(n:L:M)", true, vec!["Mia"]),
        ("(n:M:L)", true, vec!["Mia"]),
        ("(n:Other)", false, vec!["Gus"]),
        ("(n:M)", false, vec!["Jules", "Mia"]),
        ("(n)", false, vec!["Alix", "Gus", "Jules", "Mia"]),
        ("(n:L|Other)", false, vec!["Alix", "Gus", "Mia"]),
        ("(n:!L)", false, vec!["Gus", "Jules"]),
        ("(n:L|M)", false, vec!["Alix", "Jules", "Mia"]),
    ] {
        let query = format!("UNWIND [1] AS x MATCH {condition} WHERE n.k = x RETURN n.name");
        let plan = plan_of(&session, &query);
        assert_eq!(plan.contains("[index: k]"), uses, "{query}\n{plan}");
        assert_eq!(sorted(&session, &query), names(&rows), "{query}");
        assert_eq!(sorted(&scan, &query), names(&rows), "no index: {query}");
    }
    for (condition, rows) in [
        ("n:L OR n:Other", vec!["Alix", "Gus", "Mia"]),
        ("NOT n:L", vec!["Gus", "Jules"]),
        ("n:L", vec!["Alix", "Mia"]),
    ] {
        let query = format!("MATCH (n) WHERE ({condition}) AND n.k = 1 RETURN n.name");
        assert_eq!(sorted(&session, &query), names(&rows), "{query}");
        assert_eq!(sorted(&scan, &query), names(&rows), "no index: {query}");
    }
}

/// A far end with the label keeps the index, and one with none falls back.
#[test]
fn a_far_end_keeps_the_index_when_its_label_is_required() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "INSERT (a:A {name: 'Alix'})-[:R]->(:L {name: 'Gus', k: 1})",
            "MATCH (a:A) INSERT (a)-[:R]->(:Other {name: 'Mia', k: 1})",
        ],
    );
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    for (far, rows) in [
        (":L", vec!["Gus"]),
        (":Other", vec!["Mia"]),
        ("", vec!["Gus", "Mia"]),
    ] {
        for query in [
            format!("MATCH (:A)-[:R]->(n{far} {{k: 1}}) RETURN n.name"),
            format!("UNWIND [1] AS x MATCH (:A)-[:R]->(n{far}) WHERE n.k = x RETURN n.name"),
            format!("MATCH (n{far})<-[:R]-(:A) WHERE n.k = 1 RETURN n.name"),
        ] {
            assert_eq!(sorted(&session, &query), names(&rows), "{query}");
            assert_eq!(sorted(&scan, &query), names(&rows), "no index: {query}");
        }
    }
}

/// A key-wide index from the API serves any lookup, and a per-label one its own first.
#[test]
fn the_apis_key_wide_index_stays_key_wide() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:L {name: 'Alix', k: 1}), (:Other {name: 'Gus', k: 1})"],
    );
    for pattern in [":L", ":Other", ""] {
        assert!(reads_index(&session, pattern), "{pattern}");
    }
    assert!(db.has_property_index("k"));
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "Other", 1, &["Gus"], None);
}

/// `has_property_index` is true for any index on the key, so a released script keeps its answer;
/// `drop_property_index` takes every index on the key in the current graph.
#[test]
fn the_apis_answers_cover_a_per_label_index() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert!(!db.has_property_index("k"));
    run_all(&session, &[INDEX, "CREATE INDEX by_k_m FOR (n:M) ON (n.k)"]);
    assert!(db.has_property_index("k"));
    run_all(&session, &["INSERT (:L {name: 'Alix', k: 1})"]);
    assert!(db.drop_property_index("k").unwrap());
    assert!(!db.has_property_index("k"));
    assert!(!reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Alix"], None);
    assert!(!db.drop_property_index("k").unwrap());
}

/// `SHOW INDEXES` lists the index with its label; a drop takes only its own.
#[test]
fn drop_index_takes_its_own_label_only() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "CREATE INDEX by_k_m FOR (n:M) ON (n.k)",
            "INSERT (:L {name: 'Alix', k: 1}), (:M {name: 'Gus', k: 1})",
        ],
    );
    run_all(&session, &["DROP INDEX by_k"]);
    assert!(!reads_index(&session, ":L"));
    assert!(reads_index(&session, ":M"));
    assert!(db.has_property_index("k"));
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "M", 1, &["Gus"], None);
    // The name is free again, also for another label.
    run_all(&session, &["CREATE INDEX by_k FOR (n:L) ON (n.name)"]);
}

/// A graph's index is its own.
#[test]
fn an_index_serves_only_its_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH other",
            "USE GRAPH other",
            INDEX,
            "INSERT (:L {name: 'Alix', k: 1})",
            "USE GRAPH default",
            "INSERT (:L {name: 'Gus', k: 1})",
        ],
    );
    assert!(!reads_index(&session, ":L"), "the default graph");
    assert_eq!(
        sorted(&session, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
        names(&["Gus"])
    );
    run_all(&session, &["USE GRAPH other"]);
    assert!(reads_index(&session, ":L"), "its graph");
    assert_eq!(
        sorted(&session, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
        names(&["Alix"])
    );
}

// ---------------------------------------------------------------------------
// Claims: per graph, label and key
// ---------------------------------------------------------------------------

#[test]
fn claims_are_per_graph_label_and_key() {
    let both = |first: &str, second: &str, setup: &[&str], conflict: bool| {
        let db = GrafeoDB::new_in_memory();
        run_all(&db.session(), setup);
        let (mut a, mut b) = (db.session(), db.session());
        for statement in setup.iter().filter(|s| s.starts_with("USE GRAPH")) {
            run_all(&a, &[statement]);
        }
        a.begin_transaction().unwrap();
        b.begin_transaction().unwrap();
        run_all(&a, &[first]);
        let second_result = b.execute(second);
        assert_eq!(second_result.is_err(), conflict, "{first}, then {second}");
        if let Err(error) = second_result {
            assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
            b.rollback().unwrap();
            a.commit().unwrap();
        } else {
            a.commit().unwrap();
            b.commit().unwrap();
        }
    };
    both(INDEX, "CREATE INDEX o FOR (n:Other) ON (n.k)", &[], false);
    both(INDEX, "CREATE INDEX o FOR (n:L) ON (n.other)", &[], false);
    both(INDEX, "CREATE INDEX o FOR (n:L) ON (n.k)", &[], true);
    both(
        "CREATE CONSTRAINT a FOR (n:L) ON (n.k) UNIQUE",
        "CREATE CONSTRAINT b FOR (n:Other) ON (n.k) UNIQUE",
        &[],
        false,
    );
    both(
        "CREATE CONSTRAINT a FOR (n:L) ON (n.k) UNIQUE",
        "CREATE CONSTRAINT b FOR (n:L) ON (n.k, n.p) NODE KEY",
        &[],
        true,
    );
}

/// One transaction's index on a label that another's write fills: the committed index holds both.
#[test]
fn an_index_committed_past_an_open_writer_holds_its_rows() {
    let db = GrafeoDB::new_in_memory();
    let (mut writer, mut schema) = (db.session(), db.session());
    writer.begin_transaction().unwrap();
    run_all(&writer, &["INSERT (:L {name: 'Alix', k: 1})"]);
    schema.begin_transaction().unwrap();
    run_all(&schema, &[INDEX]);
    let commit = schema.commit();
    writer.commit().unwrap();
    assert!(commit.is_ok() || commit.unwrap_err().to_string().contains("GRAFEO-T001"));
    assert_found(&db, "L", 1, &["Alix"], None);
}

// ---------------------------------------------------------------------------
// A reopen, a crash, a copy
// ---------------------------------------------------------------------------

fn run_child(child: &mut std::process::Command) {
    let status = grafeo_common::testing::child_process::run(child).unwrap();
    assert!(status.success(), "the child process failed");
}

/// What `label_changes` leaves, on any database that had it committed.
fn assert_after_label_changes(path: &Path) {
    let db = GrafeoDB::open(path).unwrap();
    let session = db.session();
    assert!(reads_index(&session, ":L"), "the index is per-label still");
    assert!(!reads_index(&session, ":Other"), "only its label");
    assert!(db.has_property_index("k"));
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "L", 2, &[], None);
    assert_found(&db, "Other", 2, &["Gus"], None);
    assert_found(&db, "L", 3, &["Mia"], None);
    assert_found(&db, "M", 4, &["Jules"], None);
    run_all(&session, &["INSERT (:L {name: 'Vincent', k: 9})"]);
    assert_found(&db, "L", 9, &["Vincent"], None);
    assert_found(&db, "Other", 9, &[], None);
    drop(session);
    db.close().unwrap();
}

fn label_changes(session: &Session) {
    run_all(
        session,
        &[
            INDEX,
            "INSERT (:Other {name: 'Alix', k: 1})",
            "MATCH (n {name: 'Alix'}) SET n:L",
            "INSERT (:L:Other {name: 'Gus', k: 2})",
            "MATCH (n {name: 'Gus'}) REMOVE n:L",
            "INSERT (:L {name: 'Mia', k: 3})",
            "INSERT (:M {name: 'Jules', k: 4})",
        ],
    );
}

#[test]
fn a_reopen_keeps_the_index_per_label_after_label_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    label_changes(&db.session());
    db.close().unwrap();
    assert_after_label_changes(&path);
    assert_after_label_changes_again(&path);
}

/// After a second close, the record was written again by a checkpoint of the reopened database.
fn assert_after_label_changes_again(path: &Path) {
    let db = GrafeoDB::open(path).unwrap();
    let session = db.session();
    assert!(reads_index(&session, ":L"));
    assert!(!reads_index(&session, ":Other"));
    assert_found(&db, "L", 9, &["Vincent"], None);
    drop(session);
    db.close().unwrap();
}

const CHILD_PATH: &str = "GRAFEO_PER_LABEL_INDEX_PATH";

#[test]
fn label_changes_child() {
    let Some(path) = std::env::var_os(CHILD_PATH) else {
        return;
    };
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    label_changes(&db.session());
    std::process::exit(0);
}

/// Only the log has it: the record and the label changes replay.
#[test]
fn a_crash_replays_the_index_and_the_label_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crashed.grafeo");
    run_child(grafeo_common::child_test!("label_changes_child").env(CHILD_PATH, &path));
    assert_after_label_changes(&path);
}

/// A drop of the index in the log: it stays gone after a replay, and a checkpoint.
#[test]
fn a_dropped_index_stays_dropped_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dropped.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    label_changes(&session);
    run_all(&session, &["DROP INDEX by_k"]);
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(!reads_index(&session, ":L"));
    assert!(!db.has_property_index("k"));
    assert_found(&db, "L", 1, &["Alix"], None);
    drop(session);
    db.close().unwrap();
}

/// A copy of the database or a graph, a snapshot and a compaction keep the index per label.
#[test]
fn copies_keep_the_index_per_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    label_changes(&session);

    let imported = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();
    assert!(reads_index(&imported.session(), ":L"), "a snapshot");
    assert!(!reads_index(&imported.session(), ":Other"), "a snapshot");
    assert_found(&imported, "L", 1, &["Alix"], None);
    assert_found(&imported, "L", 2, &[], None);

    let mut compacted = GrafeoDB::new_in_memory();
    label_changes(&compacted.session());
    compacted.compact().unwrap();
    let after = compacted.session();
    assert!(reads_index(&after, ":L"), "a compaction");
    assert!(!reads_index(&after, ":Other"), "a compaction");
    drop(after);
    assert_found(&compacted, "L", 1, &["Alix"], None);
    assert_found(&compacted, "L", 2, &[], None);
}

/// A rule's write check, through a label added: the node added to the label is held to the rule by
/// the index, and an undo of it leaves the value free.
#[test]
fn a_rule_sees_a_label_added_through_its_index() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(
        &session,
        &[
            "CREATE CONSTRAINT c1 FOR (n:L) ON (n.k) UNIQUE",
            "INSERT (:L {name: 'Alix', k: 1})",
            "INSERT (:Other {name: 'Gus', k: 1})",
            "INSERT (:Other {name: 'Mia', k: 2})",
        ],
    );
    assert!(
        session.execute("MATCH (n {name: 'Gus'}) SET n:L").is_err(),
        "a label added onto a taken value"
    );
    assert_found(&db, "L", 1, &["Alix"], None);
    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n {name: 'Mia'}) SET n:L"]);
    session.rollback().unwrap();
    run_all(&session, &["INSERT (:L {name: 'Jules', k: 2})"]);
    assert!(session.execute("MATCH (n {name: 'Mia'}) SET n:L").is_err());
    assert_found(&db, "L", 2, &["Jules"], None);
    run_all(&session, &["MATCH (n {name: 'Alix'}) REMOVE n:L"]);
    run_all(&session, &["MATCH (n {name: 'Gus'}) SET n:L"]);
    assert_found(&db, "L", 1, &["Gus"], None);
}

// ---------------------------------------------------------------------------
// One transaction, several index statements; a view that disagrees with the newest labels
// ---------------------------------------------------------------------------

/// A transaction whose view still has the label writes a value while another's removal of the label
/// stands: its own lookup finds what it wrote, as a plan with no index does. If the write conflicts
/// with the removal there is nothing to find.
#[test]
fn a_write_under_a_view_that_still_sees_a_removed_label_finds_itself() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1, spare: 0})"],
    );
    let mut open = db.session();
    open.begin_transaction().unwrap();
    assert_eq!(
        sorted(&open, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
        names(&["Alix"])
    );
    run_all(&session, &["MATCH (n:Other) REMOVE n:L"]);
    if open.execute("MATCH (n {name: 'Alix'}) SET n.k = 5").is_ok() {
        let usual = sorted(&open, "MATCH (n:L) WHERE n.k = 5 RETURN n.name");
        let keyed = sorted(
            &open,
            "UNWIND [5] AS x MATCH (n:L) WHERE n.k = x RETURN n.name",
        );
        open.set_planner_rules(PlannerRules::plain());
        let plain = sorted(&open, "MATCH (n:L) WHERE n.k = 5 RETURN n.name");
        assert_eq!(usual, plain, "a constant");
        assert_eq!(keyed, plain, "a key from the rows");
        assert_eq!(plain, names(&["Alix"]));
    }
    let _ = open.rollback();
    assert_found(&db, "L", 1, &[], None);
}

/// A drop and a create on one label and key in one transaction: the new one stands, and the rows of
/// a commit are all in it; a rollback leaves the old one as it stood.
#[test]
fn a_drop_and_a_create_of_one_index_in_a_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swap.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let mut session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L {name: 'Alix', k: 1})"]);
    let listed = |session: &Session| sorted(session, "SHOW INDEXES");
    let before = listed(&session);

    session.begin_transaction().unwrap();
    run_all(
        &session,
        &["DROP INDEX by_k", "CREATE INDEX swapped FOR (n:L) ON (n.k)"],
    );
    session.rollback().unwrap();
    assert_eq!(listed(&session), before);
    assert!(reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Alix"], None);

    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "DROP INDEX by_k",
            "INSERT (:L {name: 'Gus', k: 1})",
            "CREATE INDEX swapped FOR (n:L) ON (n.k)",
            "INSERT (:L {name: 'Mia', k: 1})",
        ],
    );
    session.commit().unwrap();
    let after = listed(&session);
    assert_eq!(after.len(), 1, "{after:?}");
    assert!(after[0].contains("swapped"), "{after:?}");
    assert!(reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Alix", "Gus", "Mia"], None);
    drop(session);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(listed(&session), after);
    assert!(reads_index(&session, ":L"));
    assert_found(&db, "L", 1, &["Alix", "Gus", "Mia"], None);
    drop(session);
    db.close().unwrap();
}

/// A create on a label and key that has an index already: nothing is built, in one transaction or
/// two, and an `IF NOT EXISTS` of the same name changes nothing.
#[test]
fn a_second_create_on_one_label_and_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L {name: 'Alix', k: 1})"]);
    let bytes = index_bytes(&db);
    run_all(
        &session,
        &["CREATE INDEX IF NOT EXISTS by_k FOR (n:L) ON (n.k)"],
    );
    assert_eq!(sorted(&session, "SHOW INDEXES").len(), 1);
    assert_eq!(index_bytes(&db), bytes, "no second structure");
    assert!(
        session
            .execute("CREATE INDEX by_k FOR (n:M) ON (n.k)")
            .is_err(),
        "a name stands for one definition"
    );
    assert_found(&db, "L", 1, &["Alix"], None);
}

/// A name over the API's index: it serves every label, and the name is what lists it.
#[test]
fn a_name_over_a_key_wide_index_builds_nothing_and_keeps_it_key_wide() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:L {name: 'Alix', k: 1}), (:Other {name: 'Gus', k: 1})"],
    );
    let bytes = index_bytes(&db);
    run_all(&session, &[INDEX]);
    assert_eq!(index_bytes(&db), bytes, "nothing built");
    let listed = sorted(&session, "SHOW INDEXES");
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert!(listed[0].contains("by_k"), "{listed:?}");
    for pattern in [":L", ":Other", ""] {
        assert!(reads_index(&session, pattern), "{pattern}");
    }
    assert_found(&db, "L", 1, &["Alix"], None);
    assert_found(&db, "Other", 1, &["Gus"], None);
}

/// A's own lookup of a value it wrote, with B's removal of the label already applied, committed or
/// still pending. A's view has the label, so the row is A's to find, whatever the newest labels say.
/// Where the two writes conflict there is nothing to look up, and the test says which.
#[test]
fn a_value_written_under_a_label_another_removed_is_found_by_its_writer() {
    for b_commits in [true, false] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
        let (mut a, mut b) = (db.session(), db.session());
        a.begin_transaction().unwrap();
        assert_eq!(
            sorted(&a, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
            names(&["Alix"]),
            "A's view has the label"
        );
        b.begin_transaction().unwrap();
        run_all(&b, &["MATCH (n:Other) REMOVE n:L"]);
        if b_commits {
            b.commit().unwrap();
        }
        match a.execute("MATCH (n {name: 'Alix'}) SET n.k = 2") {
            Ok(_) => {
                for query in lookups("L", 2) {
                    assert_eq!(
                        sorted(&a, &query),
                        names(&["Alix"]),
                        "B committed: {b_commits}: {query}"
                    );
                }
                let _ = a.rollback();
            }
            Err(error) => {
                assert!(
                    error.to_string().contains("GRAFEO-T001"),
                    "only a conflict may stop the write: {error}"
                );
                let _ = a.rollback();
            }
        }
        if !b_commits {
            b.commit().unwrap();
        }
        assert_found(&db, "L", 1, &[], None);
        assert_found(&db, "Other", 1, &["Alix"], None);
    }
}

/// A reader between A's commit of a value and B's commit of a label's removal would see the committed
/// label and the new value, and find the row. Statements cannot reach it today: B's pending removal
/// and A's write of the node conflict, so A does not commit. The first branch is a guard, should that
/// conflict ever go; the second is what happens now.
#[test]
fn a_reader_between_a_value_and_a_removal_finds_the_row() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[INDEX, "INSERT (:L:Other {name: 'Alix', k: 1})"]);
    let (mut a, mut b) = (db.session(), db.session());
    a.begin_transaction().unwrap();
    b.begin_transaction().unwrap();
    assert_eq!(
        sorted(&a, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
        names(&["Alix"])
    );
    run_all(&b, &["MATCH (n:Other) REMOVE n:L"]);
    let written = a.execute("MATCH (n {name: 'Alix'}) SET n.k = 2");
    if written.is_ok() && a.commit().is_ok() {
        assert_found(&db, "L", 2, &["Alix"], None);
        assert_found(&db, "L", 1, &[], None);
        if b.commit().is_ok() {
            assert_found(&db, "L", 2, &[], None);
            assert_found(&db, "Other", 2, &["Alix"], None);
        } else {
            assert_found(&db, "L", 2, &["Alix"], None);
        }
    } else {
        let _ = a.rollback();
        b.commit().unwrap();
        assert_found(&db, "L", 1, &[], None);
        assert_found(&db, "Other", 1, &["Alix"], None);
    }
}

/// With an index on both labels, the plan reads one of them; which one follows the label counts, so
/// only that an index is read is pinned here.
#[test]
fn a_lookup_with_an_index_on_both_labels_reads_one() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            INDEX,
            "CREATE INDEX by_k_m FOR (n:M) ON (n.k)",
            "INSERT (:L:M {name: 'Alix', k: 1}), (:L {name: 'Gus', k: 1}), (:M {name: 'Mia', k: 1})",
        ],
    );
    assert!(reads_index(&session, ":L:M"));
    assert!(reads_index(&session, ":M:L"));
    for query in [
        "MATCH (n:L:M) WHERE n.k = 1 RETURN n.name",
        "UNWIND [1] AS x MATCH (n:M:L) WHERE n.k = x RETURN n.name",
    ] {
        assert_eq!(sorted(&session, query), names(&["Alix"]), "{query}");
    }
}

// ---------------------------------------------------------------------------
// A union of labels
// ---------------------------------------------------------------------------

const BY_FILE: &str = "CREATE INDEX by_file FOR (n:File) ON (n.k)";
const BY_OTHER: &str = "CREATE INDEX by_other FOR (n:Other) ON (n.k)";
const BY_THIRD: &str = "CREATE INDEX by_third FOR (n:Third) ON (n.k)";

/// `File`, `Other`, a node of both, `Third`, and one of no listed label, all with `k` 1.
fn labelled(session: &Session) {
    run_all(
        session,
        &[
            "INSERT (:File {name: 'Alix', k: 1})",
            "INSERT (:Other {name: 'Gus', k: 1})",
            "INSERT (:File:Other {name: 'Mia', k: 1})",
            "INSERT (:Third {name: 'Jules', k: 1})",
            "INSERT (:Fourth {name: 'Hans', k: 1})",
            "INSERT (:File {name: 'Vincent', k: 2})",
        ],
    );
}

/// Each way of writing the union, as a filter and with a key from the rows.
fn union_of(labels: &str, value: i64) -> [String; 3] {
    [
        format!("MATCH (n) WHERE ({labels}) AND n.k = {value} RETURN n.name"),
        format!("UNWIND [{value}] AS x MATCH (n) WHERE ({labels}) AND n.k = x RETURN n.name"),
        format!("MATCH (n) WHERE n.k = {value} AND ({labels}) RETURN n.name"),
    ]
}

/// The rows of each form with the usual plan and with one that reads no index. Sorted but not
/// deduplicated: a node of both labels appears once.
fn assert_union_rows(
    db: &GrafeoDB,
    labels: &str,
    value: i64,
    wanted: &[&str],
    at: Option<EpochId>,
) {
    let usual = db.session();
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    if let Some(epoch) = at {
        usual.set_viewing_epoch(epoch);
        scan.set_viewing_epoch(epoch);
    }
    let wanted = names(wanted);
    for query in union_of(labels, value) {
        assert_eq!(sorted(&scan, &query), wanted, "no index: {query}");
        assert_eq!(sorted(&usual, &query), wanted, "usual: {query}");
    }
}

/// Whether the plan of the form with a key from the rows reads an index on `k`.
fn union_reads_index(session: &Session, labels: &str) -> bool {
    plan_of(
        session,
        &format!("UNWIND [1] AS x MATCH (n) WHERE ({labels}) AND n.k = x RETURN n.name"),
    )
    .contains("[index: k]")
}

/// With every listed label indexed the plan reads the indexes; a node of both labels appears once.
#[test]
fn a_union_of_indexed_labels_reads_them_and_gives_each_node_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    let labels = "n:File OR n:Other";
    assert!(union_reads_index(&session, labels));
    assert_union_rows(&db, labels, 1, &["Alix", "Gus", "Mia"], None);
    assert_union_rows(&db, labels, 2, &["Vincent"], None);
    assert_union_rows(&db, labels, 3, &[], None);
    assert_eq!(
        sorted(
            &session,
            "MATCH (n) WHERE (n:File OR n:Other) AND n.k = 1 RETURN count(n) AS c"
        ),
        vec![format!("{:?}", vec![Value::Int64(3)])],
        "a count sees the node of both labels once"
    );
}

/// Three listed labels, written in any order; and a label written twice.
#[test]
fn a_union_of_more_labels_reads_them_in_any_order() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER, BY_THIRD]);
    labelled(&session);
    let every = &["Alix", "Gus", "Mia", "Jules"];
    for labels in [
        "n:File OR n:Other OR n:Third",
        "n:Third OR n:File OR n:Other",
        "(n:File OR n:Other) OR n:Third",
        "n:File OR (n:Other OR n:Third)",
    ] {
        assert!(union_reads_index(&session, labels), "{labels}");
        assert_union_rows(&db, labels, 1, every, None);
    }
    assert_union_rows(&db, "n:File OR n:File", 1, &["Alix", "Mia"], None);
}

/// One listed label with no index: not the union, the same rows by a key-wide index or the scan.
#[test]
fn a_union_with_a_label_not_indexed_reads_no_union() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE]);
    labelled(&session);
    let labels = "n:File OR n:Other";
    // The rows first: an index read for a label with no index loses that label's nodes.
    assert_union_rows(&db, labels, 1, &["Alix", "Gus", "Mia"], None);
    assert!(
        !union_reads_index(&session, labels),
        "a scan: Other has none"
    );

    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let session = db.session();
    run_all(&session, &[BY_FILE]);
    labelled(&session);
    assert!(union_reads_index(&session, labels), "the key-wide one");
    assert_union_rows(&db, labels, 1, &["Alix", "Gus", "Mia"], None);
}

/// What only looks like a union of labels: every result need not have a listed label.
#[test]
fn a_condition_that_is_no_union_of_labels_scans_with_the_same_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    for (condition, wanted) in [
        ("n:File OR n.name = 'Hans'", vec!["Alix", "Hans", "Mia"]),
        ("NOT n:File", vec!["Gus", "Hans", "Jules"]),
        ("n:File AND NOT n:Other", vec!["Alix"]),
        ("NOT (n:File OR n:Other)", vec!["Hans", "Jules"]),
    ] {
        assert_union_rows(&db, condition, 1, &wanted, None);
    }
    // `OR` of whole conditions, the key under one side only: `k` does not hold of every result.
    let query = "MATCH (n) WHERE n:Third OR (n:File AND n.k = 2) RETURN n.name";
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    let rows = names(&["Jules", "Vincent"]);
    assert_eq!(sorted(&session, query), rows);
    assert_eq!(sorted(&scan, query), rows);
}

/// A label that changes, at each view, under the union: added, removed, undone.
#[test]
fn a_union_follows_labels_that_change() {
    let db = with_history();
    let mut session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    let labels = "n:File OR n:Other";
    let before = db.current_epoch();
    run_all(
        &session,
        &[
            "MATCH (n {name: 'Jules'}) SET n:Other",
            "MATCH (n {name: 'Mia'}) REMOVE n:File",
            "MATCH (n {name: 'Alix'}) REMOVE n:File",
        ],
    );
    assert_union_rows(&db, labels, 1, &["Gus", "Jules", "Mia"], None);
    assert_union_rows(&db, labels, 1, &["Alix", "Gus", "Mia"], Some(before));

    session.begin_transaction().unwrap();
    run_all(&session, &["MATCH (n {name: 'Hans'}) SET n:File:Other"]);
    for query in union_of(labels, 1) {
        assert_eq!(
            sorted(&session, &query),
            names(&["Gus", "Hans", "Jules", "Mia"]),
            "the transaction sees its own, once: {query}"
        );
    }
    session.rollback().unwrap();
    assert_union_rows(&db, labels, 1, &["Gus", "Jules", "Mia"], None);
}

/// A far end with a union of labels, and a union on a pattern's own label expression.
#[test]
fn a_union_at_a_far_end_and_in_a_label_expression_gives_the_same_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    run_all(
        &session,
        &[
            "INSERT (:Start {name: 'Butch'})",
            "MATCH (s:Start), (n) WHERE n.k = 1 AND n <> s INSERT (s)-[:R]->(n)",
        ],
    );
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    let rows = names(&["Alix", "Gus", "Mia"]);
    for query in [
        "MATCH (:Start)-[:R]->(n) WHERE (n:File OR n:Other) AND n.k = 1 RETURN n.name",
        "MATCH (n:File|Other) WHERE n.k = 1 RETURN n.name",
        "UNWIND [1] AS x MATCH (n:File|Other) WHERE n.k = x RETURN n.name",
    ] {
        assert_eq!(sorted(&session, query), rows, "{query}");
        assert_eq!(sorted(&scan, query), rows, "no index: {query}");
    }
}

/// Written as a label expression in the pattern, with both labels indexed, the union is read too:
/// a constant key, a key from the rows and a parameter, with a node of both labels once.
#[test]
fn a_label_expression_reads_the_union_of_its_indexes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    let plan = plan_of(
        &session,
        "UNWIND [1] AS x MATCH (n:File|Other) WHERE n.k = x RETURN n.name",
    );
    assert!(plan.contains("[index: k]"), "{plan}");
    let plan = plan_of(&session, "MATCH (n:File|Third) WHERE n.k = 1 RETURN n.name");
    assert!(!plan.contains("[index: k]"), "Third has none: {plan}");
    let rows = names(&["Alix", "Gus", "Mia"]);
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    for query in [
        "MATCH (n:File|Other {k: 1}) RETURN n.name",
        "MATCH (n:File|Other) WHERE n.k = 1 RETURN n.name",
        "MATCH (:Fourth) MATCH (n:File|Other) WHERE n.k = 1 RETURN n.name",
    ] {
        assert_eq!(sorted(&session, query), rows, "{query}");
        assert_eq!(sorted(&scan, query), rows, "no index: {query}");
    }
    let params = || std::collections::HashMap::from([("v".to_string(), Value::Int64(1))]);
    let keyed = "MATCH (n:File|Other) WHERE n.k = $v RETURN n.name";
    let found = |session: &Session| {
        let result = session.execute_with_params(keyed, params()).unwrap();
        let mut found: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
        found.sort();
        found
    };
    assert_eq!(found(&session), rows, "a parameter");
    assert_eq!(found(&scan), rows, "a parameter, no index");
}

/// `IN` takes each value through each label's index, a node once per value it matches.
#[test]
fn a_union_with_in_gives_each_node_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    let rows = names(&["Alix", "Gus", "Mia", "Vincent"]);
    for query in [
        "MATCH (n:File|Other) WHERE n.k IN [1, 2] RETURN n.name",
        "MATCH (n) WHERE (n:File OR n:Other) AND n.k IN [1, 2] RETURN n.name",
        "MATCH (n:File|Other) WHERE n.k IN [1, 1, 2] RETURN n.name",
        "UNWIND [[1, 2]] AS xs MATCH (n:File|Other) WHERE n.k IN xs RETURN n.name",
    ] {
        assert_eq!(sorted(&session, query), rows, "{query}");
        assert_eq!(sorted(&scan, query), rows, "no index: {query}");
    }
}

/// A hop's candidates count across the union: a value on many nodes of both labels gives the rows
/// of a plan with no index, each node once, past any cap on how many a lookup takes.
#[test]
fn a_far_end_union_past_a_cap_counts_across_its_labels() {
    const EACH: i64 = 700;
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            BY_FILE,
            BY_OTHER,
            "INSERT (:Start {name: 'Butch'})",
            &format!(
                "UNWIND range(1, {EACH}) AS i INSERT (:File {{name: 'f' + toString(i), k: 1}})"
            ),
            &format!(
                "UNWIND range(1, {EACH}) AS i INSERT (:Other {{name: 'o' + toString(i), k: 1}})"
            ),
            &format!(
                "UNWIND range(1, {EACH}) AS i INSERT (:File:Other {{name: 'b' + toString(i), k: 1}})"
            ),
            "INSERT (:Fourth {name: 'Hans', k: 1})",
            "MATCH (s:Start), (n) WHERE n.k = 1 INSERT (s)-[:R]->(n)",
        ],
    );
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    let total = 3 * EACH;
    for query in [
        "MATCH (:Start)-[:R]->(n:File|Other) WHERE n.k = 1 RETURN count(n) AS c",
        "MATCH (:Start)-[:R]->(n) WHERE (n:File OR n:Other) AND n.k = 1 RETURN count(n) AS c",
        "UNWIND [1] AS x MATCH (:Start)-[:R]->(n:File|Other) WHERE n.k = x RETURN count(n) AS c",
        "MATCH (n:File|Other {k: 1}) RETURN count(n) AS c",
        "UNWIND [1] AS x MATCH (n:File|Other) WHERE n.k = x RETURN count(n) AS c",
    ] {
        let want = vec![format!("{:?}", vec![Value::Int64(total)])];
        assert_eq!(sorted(&session, query), want, "{query}");
        assert_eq!(sorted(&scan, query), want, "no index: {query}");
    }
    // Rows too, as a count could be answered without reading them.
    let distinct = "MATCH (:Start)-[:R]->(n:File|Other) WHERE n.k = 1 RETURN n.name";
    let found = sorted(&session, distinct);
    let mut unique = found.clone();
    unique.dedup();
    assert_eq!(found.len(), usize::try_from(total).unwrap(), "each node");
    assert_eq!(unique.len(), found.len(), "once");
}

/// A key with no node under one label, none under either, and a label no node has: the plan's
/// answer while planning a constant is the same as at run time.
#[test]
fn a_union_of_a_label_with_no_nodes_finds_the_others() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            BY_FILE,
            "CREATE INDEX by_nobody FOR (n:Nobody) ON (n.k)",
            "INSERT (:File {name: 'Alix', k: 1})",
        ],
    );
    assert_union_rows(&db, "n:File OR n:Nobody", 1, &["Alix"], None);
    assert_union_rows(&db, "n:Nobody OR n:File", 1, &["Alix"], None);
    assert_union_rows(&db, "n:Nobody", 1, &[], None);
    assert_union_rows(&db, "n:File OR n:Nobody", 9, &[], None);
    run_all(&session, &["INSERT (:Nobody {name: 'Gus', k: 1})"]);
    assert_union_rows(&db, "n:File OR n:Nobody", 1, &["Alix", "Gus"], None);
}

/// Per-label for one, key-wide for all: the key-wide index answers, with no node twice.
#[test]
fn a_union_with_a_key_wide_index_beside_a_per_label_one_gives_each_node_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &[BY_FILE, BY_OTHER]);
    labelled(&session);
    db.create_property_index("k").unwrap();
    assert!(union_reads_index(&session, "n:File OR n:Other"));
    assert!(union_reads_index(&session, "n:File OR n:Third"));
    assert_union_rows(&db, "n:File OR n:Other", 1, &["Alix", "Gus", "Mia"], None);
    assert_union_rows(&db, "n:File OR n:Third", 1, &["Alix", "Jules", "Mia"], None);
}

/// A rule's index counts as any other: a union over two rules reads both, across a reopen.
#[test]
fn a_union_over_rule_indexes_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("union.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE CONSTRAINT ca FOR (n:File) ON (n.k) UNIQUE",
            "CREATE CONSTRAINT cb FOR (n:Other) ON (n.k) UNIQUE",
            "INSERT (:File {name: 'Alix', k: 1}), (:Other {name: 'Gus', k: 1})",
        ],
    );
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(union_reads_index(&session, "n:File OR n:Other"));
    assert_union_rows(&db, "n:File OR n:Other", 1, &["Alix", "Gus"], None);
    drop(session);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The EXPLAIN note: an index on the key exists, and this lookup cannot read it
// ---------------------------------------------------------------------------

/// The database of the note's tests: `k` on three labels and `a` on one, one index on each key.
fn noted(statements: &[&str]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:File {name: 'Alix', k: 1, a: 1})",
            "INSERT (:Other {name: 'Gus', k: 1, a: 1})",
            "INSERT (:Third {name: 'Mia', k: 1, a: 1})",
        ],
    );
    run_all(&session, statements);
    db
}

const NOTE_ON_FILE: &str = "[no index: k is indexed on :File only]";

/// The note's text is the stable output, so it is pinned whole, with the usual hint in front.
#[test]
fn the_note_names_the_label_an_index_is_on() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    let plan = plan_of(&session, "MATCH (n:Other) WHERE n.k = 1 RETURN n.name");
    assert!(plan.contains(NOTE_ON_FILE), "{plan}");
    assert!(!plan.contains("[index: k]"), "{plan}");
    // With no label, the index on `File` is read, and the rest compared: no note.
    let plan = plan_of(&session, "MATCH (n) WHERE n.k = 1 RETURN n.name");
    assert!(plan.contains("[indexes and remainder: k]"), "{plan}");
    assert!(!plan.contains("[no index"), "{plan}");
}

/// Pinned by a literal or a key from the rows, by `=` or `IN`.
#[test]
fn the_note_shows_for_each_way_of_pinning_the_node() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    for query in [
        "MATCH (n:Other) WHERE n.k = 1 RETURN n.name",
        "MATCH (n:Other {k: 1}) RETURN n.name",
        "MATCH (n:Other) WHERE 1 = n.k RETURN n.name",
        "MATCH (n:Other) WHERE n.k IN [1, 2] RETURN n.name",
        "UNWIND [1] AS x MATCH (n:Other) WHERE n.k = x RETURN n.name",
        "UNWIND [[1, 2]] AS xs MATCH (n:Other) WHERE n.k IN xs RETURN n.name",
        "UNWIND [1] AS x MATCH (n) WHERE n.k = x RETURN n.name",
    ] {
        let plan = plan_of(&session, query);
        assert!(plan.contains(NOTE_ON_FILE), "{query}\n{plan}");
    }
}

/// Several labels, sorted, whatever order they were indexed in.
#[test]
fn the_note_lists_several_labels_sorted() {
    for statements in [
        [
            "CREATE INDEX by_other FOR (n:Other) ON (n.k)",
            "CREATE INDEX by_file FOR (n:File) ON (n.k)",
        ],
        [
            "CREATE INDEX by_file FOR (n:File) ON (n.k)",
            "CREATE INDEX by_other FOR (n:Other) ON (n.k)",
        ],
    ] {
        let db = noted(&statements);
        let plan = plan_of(&db.session(), "MATCH (n:Third) WHERE n.k = 1 RETURN n.name");
        assert!(
            plan.contains("[no index: k is indexed on :File, :Other only]"),
            "{plan}"
        );
    }
}

/// No note where an index serves, where there is none on the key, and for a seek by id.
#[test]
fn the_note_stays_away_when_an_index_is_read_or_there_is_none() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    for query in [
        "UNWIND [1] AS x MATCH (n:File) WHERE n.k = x RETURN n.name",
        "UNWIND [1] AS x MATCH (n:File:Other) WHERE n.k = x RETURN n.name",
        // No index on the key at all.
        "MATCH (n:Other) WHERE n.a = 1 RETURN n.name",
        // Not pinned: a range, an inequality, the key against another property.
        "MATCH (n:Other) WHERE n.k > 0 RETURN n.name",
        "MATCH (n:Other) WHERE n.k <> 2 RETURN n.name",
        "MATCH (n:Other) WHERE n.k = n.a RETURN n.name",
        // A seek by id: the note is about a lookup that has to scan.
        "UNWIND [0] AS x MATCH (n:Other) WHERE id(n) = x AND n.k = 1 RETURN n.name",
    ] {
        let plan = plan_of(&session, query);
        assert!(!plan.contains("[no index"), "{query}\n{plan}");
    }
    // The API's index covers every label, so nothing is out of reach.
    let db = noted(&[]);
    db.create_property_index("k").unwrap();
    let plan = plan_of(&db.session(), "MATCH (n:Other) WHERE n.k = 1 RETURN n.name");
    assert!(!plan.contains("[no index"), "{plan}");
}

/// A union: read when each listed label has an index, and the note when one does not.
#[test]
fn the_note_on_a_union_follows_whether_it_can_be_read() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    // A key from the rows reads no index, as the note says.
    let query = "UNWIND [1] AS x MATCH (n:File|Other) WHERE n.k = x RETURN n.name";
    let plan = plan_of(&session, query);
    assert!(plan.contains(NOTE_ON_FILE), "{query}\n{plan}");
    // A constant key reads the index it has and compares the rest.
    let query = "MATCH (n) WHERE (n:File OR n:Other) AND n.k = 1 RETURN n.name";
    let plan = plan_of(&session, query);
    assert!(
        plan.contains("[indexes and remainder: k]"),
        "{query}\n{plan}"
    );
    assert!(!plan.contains("[no index"), "{query}\n{plan}");
    run_all(&session, &["CREATE INDEX by_other FOR (n:Other) ON (n.k)"]);
    let plan = plan_of(
        &session,
        "UNWIND [1] AS x MATCH (n:File|Other) WHERE n.k = x RETURN n.name",
    );
    assert!(!plan.contains("[no index"), "{plan}");
    assert!(plan.contains("[index: k]"), "{plan}");
}

/// With several such keys, the first conjunct's.
#[test]
fn the_note_names_the_first_conjuncts_key() {
    let db = noted(&[
        "CREATE INDEX by_file FOR (n:File) ON (n.k)",
        "CREATE INDEX by_file_a FOR (n:File) ON (n.a)",
    ]);
    let session = db.session();
    let plan = plan_of(
        &session,
        "MATCH (n:Other) WHERE n.k = 1 AND n.a = 1 RETURN n.name",
    );
    assert!(plan.contains(NOTE_ON_FILE), "{plan}");
    let plan = plan_of(
        &session,
        "MATCH (n:Other) WHERE n.a = 1 AND n.k = 1 RETURN n.name",
    );
    assert!(
        plan.contains("[no index: a is indexed on :File only]"),
        "{plan}"
    );
    assert_eq!(plan.matches("[no index").count(), 1, "{plan}");
}

/// A drop takes the note with it, and an index on the label replaces it by the read.
#[test]
fn the_note_follows_the_indexes_there_are() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    let query = "UNWIND [1] AS x MATCH (n:Other) WHERE n.k = x RETURN n.name";
    assert!(plan_of(&session, query).contains(NOTE_ON_FILE));
    run_all(&session, &["CREATE INDEX by_other FOR (n:Other) ON (n.k)"]);
    let plan = plan_of(&session, query);
    assert!(
        plan.contains("[index: k]") && !plan.contains("[no index"),
        "{plan}"
    );
    run_all(&session, &["DROP INDEX by_other", "DROP INDEX by_file"]);
    let plan = plan_of(&session, query);
    assert!(
        !plan.contains("[no index") && !plan.contains("[index: k]"),
        "{plan}"
    );
}

/// It is only a note: the rows of a query with it are those of a plan with no index, and running
/// the query does not put it anywhere but in the plan's text.
#[test]
fn the_note_changes_no_rows() {
    let db = noted(&["CREATE INDEX by_file FOR (n:File) ON (n.k)"]);
    let session = db.session();
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    for (query, rows) in [
        ("MATCH (n:Other) WHERE n.k = 1 RETURN n.name", vec!["Gus"]),
        (
            "MATCH (n) WHERE n.k = 1 RETURN n.name",
            vec!["Alix", "Gus", "Mia"],
        ),
    ] {
        assert_eq!(sorted(&session, query), names(&rows), "{query}");
        assert_eq!(sorted(&scan, query), names(&rows), "no index: {query}");
    }
}

/// A's `SET` on a node whose labels B changed and committed after A's snapshot: the write is taken,
/// and the commit fails with a write conflict, whichever way the labels moved. B's stands whole, and
/// the index and a plan with no index give the same rows, also after a collection.
#[test]
fn a_value_written_after_a_committed_label_change_conflicts_at_its_commit() {
    for (b_statement, labels, name) in [
        (
            "MATCH (n:Other) REMOVE n:L",
            vec!["Other"],
            "B removed the label",
        ),
        (
            "MATCH (n:Other) SET n:M",
            vec!["L", "M", "Other"],
            "B added another label",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(
            &session,
            &[
                INDEX,
                "CREATE INDEX by_k_m FOR (n:M) ON (n.k)",
                "INSERT (:L:Other {name: 'Alix', k: 1})",
            ],
        );
        let mut a = db.session();
        a.begin_transaction().unwrap();
        assert_eq!(
            sorted(&a, "MATCH (n:L) WHERE n.k = 1 RETURN n.name"),
            names(&["Alix"])
        );
        run_all(&session, &[b_statement]);
        a.execute("MATCH (n {name: 'Alix'}) SET n.k = 2")
            .unwrap_or_else(|error| panic!("{name}: the write is taken: {error}"));
        let error = a.commit().expect_err(name);
        assert!(error.to_string().contains("GRAFEO-T001"), "{name}: {error}");
        let wanted = vec![format!(
            "{:?}",
            vec![
                Value::List(
                    labels
                        .iter()
                        .map(|label| Value::from(*label))
                        .collect::<Vec<_>>()
                        .into()
                ),
                Value::Int64(1),
            ]
        )];
        assert_eq!(
            sorted(&session, "MATCH (n {name: 'Alix'}) RETURN labels(n), n.k"),
            wanted,
            "{name}: B's change, and A's value gone"
        );
        let mut scan = db.session();
        scan.set_planner_rules(PlannerRules::plain());
        for round in ["after the conflict", "after a collection"] {
            for label in ["L", "M", "Other"] {
                for value in [1, 2] {
                    for query in lookups(label, value) {
                        assert_eq!(
                            sorted(&session, &query),
                            sorted(&scan, &query),
                            "{name}, {round}: {query}"
                        );
                    }
                }
            }
            db.gc();
        }
    }
}

// ---------------------------------------------------------------------------
// Labels checked in a filter above the key's filter
// ---------------------------------------------------------------------------

/// The plan of a query in a language, whichever parser wrote the labels' filter.
fn plan_in(session: &Session, language: &str, query: &str) -> String {
    let text = format!("EXPLAIN {query}");
    let result = match language {
        "cypher" => session.execute_cypher(&text),
        _ => session.execute(&text),
    };
    result.unwrap().rows()[0][0].as_str().unwrap().to_string()
}

/// How many rows a query reads, as a plain index read and a scan of the label differ by far.
fn scanned(session: &Session, language: &str, query: &str) -> usize {
    let result = match language {
        "cypher" => session.execute_cypher(query),
        _ => session.execute(query),
    };
    usize::try_from(result.unwrap().rows_scanned().unwrap()).unwrap()
}

/// With each listed label indexed, every way of writing the union is read through them, also where
/// the labels' filter sits above the key's: no note says otherwise, and few rows are read.
#[test]
fn the_note_stays_away_when_the_labels_are_checked_above_the_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "UNWIND range(1, 3000) AS i INSERT (:File {name: 'f', k: i})",
            "UNWIND range(1, 3000) AS i INSERT (:Other {name: 'o', k: i})",
            BY_FILE,
            BY_OTHER,
        ],
    );
    for (language, query) in [
        ("gql", "MATCH (n:File|Other {k: 7}) RETURN n.k"),
        ("gql", "MATCH (n:File|Other) WHERE n.k = 7 RETURN n.k"),
        ("gql", "MATCH (n {k: 7}) WHERE n:File OR n:Other RETURN n.k"),
        (
            "cypher",
            "MATCH (n {k: 7}) WHERE n:File OR n:Other RETURN n.k",
        ),
        (
            "gql",
            "MATCH (n) WHERE (n:File OR n:Other) AND n.k = 7 RETURN n.k",
        ),
        (
            "cypher",
            "MATCH (n) WHERE (n:File OR n:Other) AND n.k = 7 RETURN n.k",
        ),
    ] {
        let plan = plan_in(&session, language, query);
        assert!(!plan.contains("[no index"), "{language}: {query}\n{plan}");
        assert!(
            scanned(&session, language, query) < 100,
            "{language}: {query} reads through the indexes\n{plan}"
        );
    }
}

/// The same for one label, where the label's filter sits above the key's.
#[test]
fn the_note_stays_away_when_one_label_is_checked_above_the_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "UNWIND range(1, 3000) AS i INSERT (:File {name: 'f', k: i})",
            BY_FILE,
        ],
    );
    for (language, query) in [
        ("gql", "MATCH (n {k: 7}) WHERE n:File RETURN n.k"),
        ("cypher", "MATCH (n {k: 7}) WHERE n:File RETURN n.k"),
        ("cypher", "MATCH (n:File {k: 7}) RETURN n.k"),
        ("gql", "MATCH (n:File {k: 7}) RETURN n.k"),
    ] {
        let plan = plan_in(&session, language, query);
        assert!(!plan.contains("[no index"), "{language}: {query}\n{plan}");
    }
}

/// The note's other half: when no index is read for the key it shows, in either filter order.
#[test]
fn the_note_shows_in_either_filter_order_when_no_index_is_read() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "UNWIND range(1, 3000) AS i INSERT (:File {name: 'f', k: i})",
            "UNWIND range(1, 3000) AS i INSERT (:Other {name: 'o', k: i})",
            BY_FILE,
        ],
    );
    for (language, query) in [
        ("gql", "MATCH (n:Other {k: 7}) RETURN n.k"),
        ("cypher", "MATCH (n:Other {k: 7}) RETURN n.k"),
        ("gql", "MATCH (n {k: 7}) WHERE n:Other RETURN n.k"),
        ("cypher", "MATCH (n {k: 7}) WHERE n:Other RETURN n.k"),
        ("gql", "MATCH (n) WHERE n.k = 7 AND n:Other RETURN n.k"),
        ("cypher", "MATCH (n) WHERE n.k = 7 AND n:Other RETURN n.k"),
    ] {
        let plan = plan_in(&session, language, query);
        assert!(
            plan.contains(NOTE_ON_FILE),
            "{language}: {query}
{plan}"
        );
        assert!(
            scanned(&session, language, query) >= 3000,
            "{language}: {query} scans, as the note says
{plan}"
        );
    }
}

/// A far end with its label and a key, written in GQL or Cypher, goes to the key through the index
/// on that label: the two languages put the label's filter on either side of the key's. A hub's
/// edges are not walked, and the rows are those of a plan with no index.
#[test]
fn a_hop_to_a_key_reads_the_far_labels_index_in_each_language() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "UNWIND range(0, 2999) AS i INSERT (:Far {k: i})",
            "UNWIND range(0, 2999) AS i INSERT (:Other {k: i + 3000})",
            "INSERT (:Start {s: 1})",
            "MATCH (a:Start), (m) WHERE m.k >= 0 INSERT (a)-[:R]->(m)",
            "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
        ],
    );
    let mut scan = db.session();
    scan.set_planner_rules(PlannerRules::plain());
    for language in ["gql", "cypher"] {
        for query in [
            "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = 7 RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.s IN [1, 2] RETURN a.s, m.k",
        ] {
            let plan = plan_in(&session, language, query);
            assert!(
                plan.contains("[expand to key: k]"),
                "{language}: {query}\n{plan}"
            );
            assert!(
                scanned(&session, language, query) < 100,
                "{language}: {query} walks the hub\n{plan}"
            );
            let run = |session: &Session| {
                let result = match language {
                    "cypher" => session.execute_cypher(query),
                    _ => session.execute(query),
                }
                .unwrap();
                result.row_count()
            };
            assert_eq!(run(&session), 1, "{language}: {query}");
            assert_eq!(run(&scan), 1, "no index, {language}: {query}");
        }
    }
}

/// The note says a label has no index. With the label's index there, a plan that goes to the key
/// another way has no note: a hash join on the rows, a key from an earlier row, an empty list, and
/// the rules off. The rows are those of the plain plan.
#[test]
fn the_note_stays_away_from_plans_that_go_to_the_key_another_way() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:File {k: 1}), (:File {k: 2})", BY_FILE],
    );
    let hash_join = "UNWIND [{id: 1}, {id: 2}] AS r MATCH (n:File) WHERE n.k = r.id RETURN n.k";
    let plan = plan_in(&session, "gql", hash_join);
    assert!(plan.contains("[hash join"), "{plan}");
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    for (query, rows) in [
        (hash_join, vec!["[Int64(1)]", "[Int64(2)]"]),
        (
            "UNWIND [1] AS x MATCH (n:File) WHERE n.k = x RETURN n.k",
            vec!["[Int64(1)]"],
        ),
        ("MATCH (n:File) WHERE n.k IN [] RETURN n.k", vec![]),
        (
            "MATCH (n:File) WHERE n.k = 1 RETURN n.k",
            vec!["[Int64(1)]"],
        ),
    ] {
        for (rules, reader) in [("all", &session), ("plain", &plain)] {
            let plan = plan_in(reader, "gql", query);
            assert!(
                !plan.contains("[no index"),
                "rules {rules}: {query}\n{plan}"
            );
            assert_eq!(sorted(reader, query), rows, "rules {rules}: {query}");
        }
    }
    for rule in PlannerRule::ALL {
        let mut off = db.session();
        off.set_planner_rules(PlannerRules::all().with(rule, false));
        for query in [hash_join, "MATCH (n:File) WHERE n.k = 1 RETURN n.k"] {
            let plan = plan_in(&off, "gql", query);
            assert!(!plan.contains("[no index"), "{rule:?} off: {query}\n{plan}");
        }
    }
}

/// A key from an earlier row of a statement that writes: the plan has no note, without running.
#[test]
fn the_note_stays_away_from_a_key_after_a_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:File {k: 1}), (:File {k: 2})", BY_FILE],
    );
    let query = "INSERT (:A {v: 1}) WITH 1 AS x MATCH (n:File) WHERE n.k = x RETURN n.k";
    let plan = plan_in(&session, "gql", query);
    assert!(!plan.contains("[no index"), "{plan}");
    assert_eq!(sorted(&session, query), vec!["[Int64(1)]".to_string()]);
}
