//! The planner's node counts (`GraphStore::label_node_estimate` and
//! `node_estimate`, 0.6.1): the row-group store reads them from its counts
//! without building the statistics, and they must be what the statistics
//! say and what a committed reader finds:
//!
//! - after creates, deletes (of nodes with one label and with two), label
//!   adds and removes, and a label whose last node goes;
//! - 0 for a label no node has, never had, or only an edge type has;
//! - an open transaction moves nothing until it commits, and a rollback
//!   nothing at all;
//! - a store that only overrides `statistics()` (a projection, and the null
//!   store) answers through the default methods;
//! - a two-label scan starts from the label with fewer nodes now, after
//!   the counts flip.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test label_node_estimates
//! ```

#![cfg(feature = "gql")]

use grafeo_core::graph::{GraphStore, NullGraphStore};
use grafeo_engine::{Config, GrafeoDB, ProjectionSpec};

/// Every label the tests use, an unknown one, a lowercase spelling and an
/// edge type's name.
const LABELS: [&str; 6] = ["Person", "Admin", "City", "Unknown", "person", "LIVES_IN"];

/// Alix, Gus and Vincent are Persons, Mia a Person and Admin, Paris and
/// Prague Cities; Alix and Mia LIVES_IN Paris.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (:Person {name: 'Gus'}), \
         (:Person {name: 'Vincent'}), (mia:Person:Admin {name: 'Mia'}), \
         (paris:City {name: 'Paris'}), (:City {name: 'Prague'}), \
         (alix)-[:LIVES_IN]->(paris), (mia)-[:LIVES_IN]->(paris)",
    )
    .unwrap();
    db
}

fn found(db: &GrafeoDB, pattern: &str) -> u64 {
    let rows = db
        .execute(&format!("MATCH {pattern} RETURN n"))
        .unwrap_or_else(|error| panic!("{pattern}: {error}"))
        .rows()
        .len();
    u64::try_from(rows).unwrap()
}

/// The estimates of `store`, checked against its statistics: per label of
/// [`LABELS`], then the total.
fn estimates(store: &dyn GraphStore, context: &str) -> (Vec<u64>, u64) {
    let statistics = store.statistics();
    let per_label = LABELS
        .iter()
        .map(|label| {
            let estimate = store.label_node_estimate(label);
            let listed = statistics.get_label(label).map_or(0, |l| l.node_count);
            assert_eq!(
                estimate, listed,
                "{context}: {label} against the statistics"
            );
            estimate
        })
        .collect();
    let total = store.node_estimate();
    assert_eq!(
        total, statistics.total_nodes,
        "{context}: the total against the statistics"
    );
    (per_label, total)
}

/// The row-group store's estimates, which must also be what a committed
/// reader finds by scanning. Returns `[Person, Admin, City]` and the total.
fn counts(db: &GrafeoDB, context: &str) -> ([u64; 3], u64) {
    let store = db.store().unwrap();
    let (per_label, total) = estimates(&*store, context);
    for (label, estimate) in LABELS.iter().zip(&per_label) {
        assert_eq!(
            *estimate,
            found(db, &format!("(n:{label})")),
            "{context}: {label} against a scan"
        );
    }
    assert_eq!(
        total,
        found(db, "(n)"),
        "{context}: the total against a scan"
    );
    ([per_label[0], per_label[1], per_label[2]], total)
}

#[test]
fn an_empty_store_counts_nothing() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(counts(&db, "empty"), ([0, 0, 0], 0));
}

#[test]
fn the_estimates_follow_creates_deletes_and_label_changes() {
    let db = database();
    assert_eq!(counts(&db, "built"), ([4, 1, 2], 6));

    db.execute("MATCH (n {name: 'Gus'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(counts(&db, "Gus deleted"), ([3, 1, 2], 5));

    db.execute("MATCH (n {name: 'Vincent'}) SET n:Admin")
        .unwrap();
    assert_eq!(counts(&db, "Vincent an Admin"), ([3, 2, 2], 5));

    db.execute("MATCH (n {name: 'Mia'}) REMOVE n:Admin")
        .unwrap();
    db.execute("MATCH (n {name: 'Vincent'}) REMOVE n:Admin")
        .unwrap();
    assert_eq!(counts(&db, "no Admin left"), ([3, 0, 2], 5));

    db.execute("MATCH (n {name: 'Mia'}) SET n:Admin").unwrap();
    assert_eq!(counts(&db, "Mia an Admin again"), ([3, 1, 2], 5));

    // A node with both labels: each count moves.
    db.execute("MATCH (n {name: 'Mia'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(counts(&db, "Mia deleted"), ([2, 0, 2], 4));

    db.execute("INSERT (:City {name: 'Berlin'}), (:Person {name: 'Jules'})")
        .unwrap();
    assert_eq!(counts(&db, "Berlin and Jules created"), ([3, 0, 3], 6));
}

/// Inside a transaction its writes are not in the committed counts; the
/// rollback leaves them as they were, the commit moves them.
#[test]
fn a_transaction_moves_the_counts_only_when_it_commits() {
    let db = database();
    let writes = [
        "INSERT (:Person {name: 'Jules'}), (:Person:Admin {name: 'Butch'})",
        "MATCH (n {name: 'Paris'}) DETACH DELETE n",
        "MATCH (n {name: 'Alix'}) SET n:Admin",
    ];

    let mut session = db.session();
    session.begin_transaction().unwrap();
    for write in writes {
        session.execute(write).unwrap();
    }
    assert_eq!(
        session
            .execute("MATCH (n:Admin) RETURN n")
            .unwrap()
            .rows()
            .len(),
        3,
        "the transaction sees its own Admins"
    );
    assert_eq!(counts(&db, "open"), ([4, 1, 2], 6));
    session.rollback().unwrap();
    assert_eq!(counts(&db, "rolled back"), ([4, 1, 2], 6));

    session.begin_transaction().unwrap();
    for write in writes {
        session.execute(write).unwrap();
    }
    session.commit().unwrap();
    assert_eq!(counts(&db, "committed"), ([6, 3, 1], 7));
}

/// A projection overrides only `statistics()`: its estimates are those, a
/// label outside its spec counts the projected nodes with it, and both
/// agree with a query on the projection.
#[test]
fn a_projection_answers_through_the_default_methods() {
    let db = database();
    db.execute("MATCH (n {name: 'Vincent'}) SET n:Admin")
        .unwrap();
    db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
        .unwrap();
    let snapshot = db.projection("people").unwrap();
    let (per_label, total) = estimates(snapshot.store(), "projection");
    assert_eq!(&per_label[..3], [4, 2, 0], "Person, Admin, City");
    assert_eq!(total, 4);

    let view = GrafeoDB::with_read_store(snapshot.shared(), Config::in_memory()).unwrap();
    for (label, estimate) in LABELS.iter().zip(&per_label) {
        assert_eq!(
            *estimate,
            found(&view, &format!("(n:{label})")),
            "{label} against a scan of the projection"
        );
    }
    assert_eq!(total, found(&view, "(n)"));
}

#[test]
fn the_null_store_counts_nothing() {
    let store = NullGraphStore;
    assert_eq!(estimates(&store, "null store"), (vec![0; LABELS.len()], 0));
}

/// The scan label of `MATCH (n:Person:Admin)` in the plan.
fn scanned(db: &GrafeoDB) -> String {
    let plan = db
        .execute("EXPLAIN MATCH (n:Person:Admin) RETURN n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect::<Vec<_>>()
        .join("\n");
    ["Person", "Admin"]
        .into_iter()
        .find(|label| plan.contains(&format!("Scan (n:{label})")))
        .unwrap_or_else(|| panic!("no Person or Admin scan in {plan}"))
        .to_string()
}

/// A two-label match scans the label with fewer nodes, as the counts are
/// now: Admin while few nodes are Admins, Person once Admins outnumber
/// Persons. Both plans return the same nodes.
#[test]
fn a_two_label_scan_follows_the_counts() {
    let db = database();
    assert_eq!(scanned(&db), "Admin", "1 Admin against 4 Persons");
    db.execute("UNWIND range(1, 10) AS i INSERT (:Admin {name: 'admin ' + toString(i)})")
        .unwrap();
    assert_eq!(scanned(&db), "Person", "11 Admins against 4 Persons");
    assert_eq!(found(&db, "(n:Person:Admin)"), 1);
    db.execute("MATCH (n:Admin) WHERE NOT n:Person DETACH DELETE n")
        .unwrap();
    assert_eq!(scanned(&db), "Admin", "the extra Admins deleted");
    assert_eq!(found(&db, "(n:Person:Admin)"), 1);
}
