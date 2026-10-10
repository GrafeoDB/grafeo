//! Creating an edge while a transaction deletes one of its endpoints.
//!
//! An edge needs both its endpoints. A transaction that creates one claims
//! them: a transaction that deletes one of them at the same time conflicts
//! with it, and the later of the two fails (first writer wins, as for two
//! writes of one node); a delete committed after the creating transaction
//! began makes that transaction's commit fail, and the other way around. An
//! edge to a node the transaction itself deleted is refused. So no committed
//! edge ever ends at a deleted node, in a plain store or a compacted one.
//! Claims conflict with deletes only: two transactions that create edges to
//! one node, or one that creates an edge to a node while another sets its
//! properties, both commit. An edge the schema refuses claims nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test edge_endpoint_conflicts
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Error;
use grafeo_engine::{GrafeoDB, Session};

/// Alix, Gus and Vincent, without edges.
const PEOPLE: &str = "INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'}), \
                      (:Person {name: 'Vincent'})";

/// Alix knows Gus.
const ALIX_KNOWS_GUS: &str =
    "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(g)";

/// Vincent knows Gus.
const VINCENT_KNOWS_GUS: &str =
    "MATCH (v:Person {name: 'Vincent'}), (g:Person {name: 'Gus'}) INSERT (v)-[:KNOWS]->(g)";

/// Gus goes, with his edges.
const DELETE_GUS: &str = "MATCH (g:Person {name: 'Gus'}) DETACH DELETE g";

/// The databases each test runs on: a plain one, and one compacted after
/// its people were written.
fn databases() -> Vec<(&'static str, GrafeoDB)> {
    let plain = GrafeoDB::new_in_memory();
    plain.execute(PEOPLE).unwrap();
    let mut compacted = GrafeoDB::new_in_memory();
    compacted.execute(PEOPLE).unwrap();
    compacted.compact().unwrap();
    vec![("plain", plain), ("compacted", compacted)]
}

/// What the tests compare: the people, who knows whom, and the number of
/// nodes and edges the store reports (a dangling edge is not in the
/// matches, but the edge count counts it).
fn state(db: &GrafeoDB) -> (Vec<String>, Vec<(String, String)>, usize, usize) {
    let text = |value: &Value| match value {
        Value::String(text) => text.to_string(),
        other => panic!("not a name: {other:?}"),
    };
    let people = db
        .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| text(&row[0]))
        .collect();
    let knows = db
        .execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b \
             ORDER BY a, b",
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| (text(&row[0]), text(&row[1])))
        .collect();
    (people, knows, db.node_count(), db.edge_count())
}

/// The state once Gus is gone: no edge is left, none points at him.
fn without_gus() -> (Vec<String>, Vec<(String, String)>, usize, usize) {
    (
        vec!["Alix".to_string(), "Vincent".to_string()],
        Vec::new(),
        2,
        0,
    )
}

/// The state with Gus and Alix's edge to him.
fn alix_knows_gus() -> (Vec<String>, Vec<(String, String)>, usize, usize) {
    (
        vec!["Alix".to_string(), "Gus".to_string(), "Vincent".to_string()],
        vec![("Alix".to_string(), "Gus".to_string())],
        3,
        1,
    )
}

/// Whether `error` is a write conflict.
fn is_conflict(error: &Error) -> bool {
    error.to_string().to_lowercase().contains("conflict")
}

/// Whether `session` sees Gus.
fn sees_gus(session: &Session) -> bool {
    session
        .execute("MATCH (g:Person {name: 'Gus'}) RETURN g")
        .unwrap()
        .row_count()
        == 1
}

/// The ids of Alix and Gus.
fn alix_and_gus(db: &GrafeoDB) -> (NodeId, NodeId) {
    let ids = db
        .execute("MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) RETURN id(a), id(g)")
        .unwrap();
    let id = |column: usize| match ids.rows()[0][column] {
        Value::Int64(id) => NodeId::new(u64::try_from(id).unwrap()),
        ref other => panic!("not an id: {other:?}"),
    };
    (id(0), id(1))
}

/// A transaction deletes Gus; another then creates an edge to him. When the
/// linking transaction still sees Gus (the delete is not committed), the
/// edge is a conflict; the delete commits, and no edge points at Gus.
///
/// A plain store hides a pending delete from other transactions today (its
/// delete is stamped with the deleting transaction's start epoch, a known
/// bug of its own), so there the linker finds no Gus and creates nothing; a
/// compacted base keeps its deletes pending until the commit.
#[test]
fn an_edge_to_a_node_another_transaction_deletes_is_a_conflict() {
    for (kind, db) in databases() {
        let mut deleter = db.session();
        deleter.begin_transaction().unwrap();
        deleter.execute(DELETE_GUS).unwrap();

        let mut linker = db.session();
        linker.begin_transaction().unwrap();
        let saw_gus = sees_gus(&linker);
        let created = linker.execute(ALIX_KNOWS_GUS).map(|_| ());
        deleter.commit().unwrap();
        // A failed statement leaves the transaction open, with nothing in it.
        let committed = linker.commit();
        let outcome = format!(
            "{kind}: the link {created:?}, its commit {committed:?}, which left {:?}",
            state(&db)
        );
        if saw_gus {
            assert!(
                created.as_ref().is_err_and(is_conflict),
                "the edge to the node being deleted is a conflict; {outcome}"
            );
        }
        assert_eq!(state(&db), without_gus(), "{outcome}");
    }
}

/// The other order: a transaction creates an edge to Gus, then another
/// deletes him. The delete is the conflict, and the edge commits.
#[test]
fn deleting_a_node_another_transaction_links_to_is_a_conflict() {
    for (kind, db) in databases() {
        let mut linker = db.session();
        linker.begin_transaction().unwrap();
        linker.execute(ALIX_KNOWS_GUS).unwrap();

        let mut deleter = db.session();
        deleter.begin_transaction().unwrap();
        let deleted = deleter.execute(DELETE_GUS).map(|_| ());
        deleter.rollback().unwrap();
        linker.commit().unwrap();
        assert!(
            deleted.as_ref().is_err_and(is_conflict),
            "{kind}: deleting the node an open transaction links to is a conflict, \
             got {deleted:?}"
        );
        assert_eq!(
            state(&db),
            alix_knows_gus(),
            "{kind}: Gus stays, with the edge"
        );
    }
}

/// A transaction that began before the delete of Gus committed and still
/// sees him can create an edge to him; its commit then fails, as a commit
/// does after a conflicting write committed after it began.
#[test]
fn an_edge_to_a_node_deleted_after_the_transaction_began_fails_its_commit() {
    for (kind, db) in databases() {
        let mut linker = db.session();
        linker.begin_transaction().unwrap();
        db.execute(DELETE_GUS).unwrap();

        let saw_gus = sees_gus(&linker);
        let created = linker.execute(ALIX_KNOWS_GUS).map(|_| ());
        let committed = linker.commit();
        let outcome = format!(
            "{kind}: the link {created:?}, its commit {committed:?}, which left {:?}",
            state(&db)
        );
        if saw_gus {
            assert!(
                created.as_ref().is_err_and(is_conflict)
                    || committed.as_ref().is_err_and(is_conflict),
                "an edge to a node deleted after the transaction began is a conflict; {outcome}"
            );
        }
        assert_eq!(state(&db), without_gus(), "{outcome}");
    }
}

/// The other order: an edge to Gus committed after a transaction began makes
/// that transaction's delete of Gus fail.
#[test]
fn deleting_a_node_linked_to_after_the_transaction_began_fails_its_commit() {
    for (kind, db) in databases() {
        let mut deleter = db.session();
        deleter.begin_transaction().unwrap();
        db.execute(ALIX_KNOWS_GUS).unwrap();

        let deleted = deleter.execute(DELETE_GUS).and_then(|_| deleter.commit());
        assert!(
            deleted.as_ref().is_err_and(is_conflict),
            "{kind}: the delete of a node linked to meanwhile fails, got {deleted:?}"
        );
        assert_eq!(
            state(&db),
            alix_knows_gus(),
            "{kind}: Gus stays, with the edge"
        );
    }
}

/// An edge to a node the transaction itself deleted is refused, in the
/// statement that deletes it (which then changes nothing) and in a later
/// direct call: the node is gone for the transaction.
#[test]
fn an_edge_to_a_node_the_transaction_deleted_is_refused() {
    for (kind, db) in databases() {
        let (alix, gus) = alix_and_gus(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let in_one_statement = session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
                 DETACH DELETE g INSERT (a)-[:KNOWS]->(g)",
            )
            .map(|_| ());
        assert!(
            sees_gus(&session),
            "{kind}: the failed statement deletes nothing, got {in_one_statement:?}"
        );
        session.execute(DELETE_GUS).unwrap();
        let later = session.create_edge(alix, gus, "KNOWS");
        session.commit().unwrap();
        assert!(
            in_one_statement.is_err() && later.is_err(),
            "{kind}: an edge to the deleted node is refused, got {in_one_statement:?} in the \
             deleting statement and {later:?} later, which left {:?}",
            state(&db)
        );
        assert_eq!(state(&db), without_gus(), "{kind}");
    }
}

/// An edge the schema refuses claims nothing: while the transaction that
/// tried to create it is still open, another one deletes the endpoint
/// without a conflict.
#[test]
fn a_refused_edge_leaves_its_endpoints_unclaimed() {
    for (kind, db) in databases() {
        db.execute("CREATE EDGE TYPE KNOWS (since INT64)").unwrap();
        let mut linker = db.session();
        linker.begin_transaction().unwrap();
        let refused = linker
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS {since: 'yesterday'}]->(g)",
            )
            .map(|_| ());
        assert!(
            refused.as_ref().is_err_and(|error| !is_conflict(error)),
            "{kind}: the schema refuses the edge, got {refused:?}"
        );

        let mut deleter = db.session();
        deleter.begin_transaction().unwrap();
        let deleted = deleter.execute(DELETE_GUS).map(|_| ());
        assert!(
            deleted.is_ok(),
            "{kind}: no edge claims Gus, got {deleted:?}"
        );
        deleter.commit().unwrap();
        linker.commit().unwrap();
        assert_eq!(state(&db), without_gus(), "{kind}");
    }
}

/// A direct call outside any transaction cannot create an edge to a node
/// whose delete is committed.
#[test]
fn a_direct_edge_to_a_deleted_node_is_refused() {
    for (kind, db) in databases() {
        let (alix, gus) = alix_and_gus(&db);
        db.execute(DELETE_GUS).unwrap();
        let created = db.create_edge(alix, gus, "KNOWS");
        assert!(
            created.is_err(),
            "{kind}: refused, got {created:?}, which left {:?}",
            state(&db)
        );
        assert_eq!(state(&db), without_gus(), "{kind}");
    }
}

/// Two transactions that create edges to the same node both commit: an
/// edge's claim on its endpoints conflicts only with a delete.
#[test]
fn two_transactions_link_to_one_node_without_a_conflict() {
    for (kind, db) in databases() {
        let mut alix = db.session();
        let mut vincent = db.session();
        alix.begin_transaction().unwrap();
        vincent.begin_transaction().unwrap();
        alix.execute(ALIX_KNOWS_GUS).unwrap();
        vincent.execute(VINCENT_KNOWS_GUS).unwrap();
        alix.commit().unwrap();
        vincent.commit().unwrap();

        assert_eq!(
            state(&db),
            (
                vec!["Alix".to_string(), "Gus".to_string(), "Vincent".to_string()],
                vec![
                    ("Alix".to_string(), "Gus".to_string()),
                    ("Vincent".to_string(), "Gus".to_string())
                ],
                3,
                2
            ),
            "{kind}: both edges"
        );
    }
}

/// A transaction that sets properties of Gus and one that creates an edge
/// to him both commit.
#[test]
fn a_property_write_and_an_edge_to_the_node_do_not_conflict() {
    for (kind, db) in databases() {
        let mut writer = db.session();
        let mut linker = db.session();
        writer.begin_transaction().unwrap();
        linker.begin_transaction().unwrap();
        writer
            .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 19")
            .unwrap();
        linker.execute(ALIX_KNOWS_GUS).unwrap();
        writer
            .execute("MATCH (g:Person {name: 'Gus'}) SET g.city = 'Paris'")
            .unwrap();
        linker.commit().unwrap();
        writer.commit().unwrap();

        let gus = db
            .execute("MATCH (a:Person)-[:KNOWS]->(g:Person) RETURN g.age, g.city")
            .unwrap();
        assert_eq!(
            gus.rows(),
            [vec![Value::Int64(19), Value::from("Paris")]],
            "{kind}: the edge and both values"
        );
    }
}
