//! Deletes after `compact()` follow the transaction.
//!
//! Deleting a node or edge written before `compact()` hides it from the
//! deleting transaction only until the commit: a rollback (or a savepoint
//! rollback) brings it back, other sessions keep seeing it, and a checkpoint
//! writes only committed deletes. A node changed and then deleted stays
//! deleted, and a node created after `compact()` can be deleted after a
//! reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_transactions
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

/// The names of the people, sorted.
const PEOPLE: &str = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";

/// Who knows whom, since when, sorted.
const KNOWS: &str = "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
                     RETURN a.name AS a, b.name AS b, k.since AS since ORDER BY a";

/// The names of the cities.
const CITIES: &str = "MATCH (c:City) RETURN c.name AS name";

/// Gus's age, if Gus is there.
const GUS_AGE: &str = "MATCH (g:Person {name: 'Gus'}) RETURN g.age AS age";

/// A database or a session that runs read queries.
trait Reads {
    /// The rows `query` returns.
    fn rows(&self, query: &str) -> Vec<Vec<Value>>;

    /// The people, who knows whom, and Gus's age: what the tests compare.
    fn state(&self) -> (Vec<Vec<Value>>, Vec<Vec<Value>>, Vec<Vec<Value>>) {
        (self.rows(PEOPLE), self.rows(KNOWS), self.rows(GUS_AGE))
    }
}

impl Reads for GrafeoDB {
    fn rows(&self, query: &str) -> Vec<Vec<Value>> {
        self.execute(query).unwrap().rows().to_vec()
    }
}

impl Reads for Session {
    fn rows(&self, query: &str) -> Vec<Vec<Value>> {
        self.execute(query).unwrap().rows().to_vec()
    }
}

/// One row per name.
fn names(names: &[&str]) -> Vec<Vec<Value>> {
    names.iter().map(|name| vec![Value::from(*name)]).collect()
}

/// Alix knows Gus (since 1988), who knows Mia (since 2019).
const INSERT_PEOPLE: &str = "INSERT (:Person {name: 'Alix', age: 33})\
                             -[:KNOWS {since: 1988}]->(:Person {name: 'Gus', age: 19})\
                             -[:KNOWS {since: 2019}]->(:Person {name: 'Mia', age: 88})";

/// The state of the database [`INSERT_PEOPLE`] wrote.
fn everyone() -> (Vec<Vec<Value>>, Vec<Vec<Value>>, Vec<Vec<Value>>) {
    (
        names(&["Alix", "Gus", "Mia"]),
        vec![
            vec![Value::from("Alix"), Value::from("Gus"), Value::Int64(1988)],
            vec![Value::from("Gus"), Value::from("Mia"), Value::Int64(2019)],
        ],
        vec![vec![Value::Int64(19)]],
    )
}

/// The state once Gus and his edges are deleted.
fn without_gus() -> (Vec<Vec<Value>>, Vec<Vec<Value>>, Vec<Vec<Value>>) {
    (names(&["Alix", "Mia"]), Vec::new(), Vec::new())
}

/// An in-memory database holding [`INSERT_PEOPLE`], compacted: all of it is
/// in the base.
fn compacted_people() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(INSERT_PEOPLE).unwrap();
    db.compact().unwrap();
    db
}

/// A transaction deletes Gus from the base: until it ends, only it misses him
/// and his edges; other sessions and transactions, older or newer, still see
/// them, and the rollback brings them back for the deleting session too.
#[test]
#[ignore = "#412: the plain store shows an uncommitted delete to other sessions"]
fn a_rolled_back_base_delete_restores_the_node_and_its_edges() {
    let db = compacted_people();
    assert_eq!(db.state(), everyone());
    let mut older = db.session();
    older.begin_transaction().unwrap();

    let mut deleter = db.session();
    deleter.begin_transaction().unwrap();
    deleter
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    assert_eq!(deleter.state(), without_gus(), "the deleting transaction");
    assert_eq!(db.state(), everyone(), "a session outside the transaction");
    assert_eq!(older.state(), everyone(), "a transaction begun before");
    let mut newer = db.session();
    newer.begin_transaction().unwrap();
    assert_eq!(newer.state(), everyone(), "a transaction begun after");
    newer.rollback().unwrap();

    deleter.rollback().unwrap();
    assert_eq!(
        deleter.state(),
        everyone(),
        "the session after its rollback"
    );
    assert_eq!(db.state(), everyone(), "after the rollback");
    older.rollback().unwrap();
}

/// A committed base delete is seen by every transaction begun after it.
#[test]
#[ignore = "#412: the plain store shows an uncommitted delete to other sessions"]
fn a_committed_base_delete_is_seen_after_the_commit() {
    let db = compacted_people();
    let mut deleter = db.session();
    deleter.begin_transaction().unwrap();
    deleter
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    assert_eq!(db.state(), everyone(), "before the commit");
    deleter.commit().unwrap();

    assert_eq!(db.state(), without_gus());
    let mut newer = db.session();
    newer.begin_transaction().unwrap();
    assert_eq!(newer.state(), without_gus());
    newer.rollback().unwrap();
}

/// A savepoint rollback brings back what was deleted after the savepoint and
/// keeps what was deleted before it, also once committed.
#[test]
#[ignore = "#412: the plain store shows an uncommitted delete to other sessions"]
fn a_savepoint_rollback_restores_only_the_later_base_deletes() {
    let db = compacted_people();
    let mut deleter = db.session();
    deleter.begin_transaction().unwrap();
    deleter
        .execute("MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() DELETE k")
        .unwrap();
    deleter.savepoint("before_mia").unwrap();
    deleter
        .execute("MATCH (m:Person {name: 'Mia'}) DETACH DELETE m")
        .unwrap();
    assert_eq!(deleter.rows(PEOPLE), names(&["Alix", "Gus"]));
    deleter.rollback_to_savepoint("before_mia").unwrap();

    let gus_knows_mia = vec![vec![
        Value::from("Gus"),
        Value::from("Mia"),
        Value::Int64(2019),
    ]];
    assert_eq!(deleter.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
    assert_eq!(
        deleter.rows(KNOWS),
        gus_knows_mia,
        "Alix's edge stays deleted"
    );
    assert_eq!(db.state(), everyone(), "nothing is committed yet");
    deleter.commit().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
    assert_eq!(db.rows(KNOWS), gus_knows_mia);
}

/// Guarantee 10: a transaction begun before another session's commit still
/// sees the base nodes and edges that later writes copied into the overlay,
/// with their labels and values: a copy is there at every epoch, as the base
/// entity it copies is.
#[test]
fn a_transaction_begun_before_a_commit_sees_base_entities_copied_after_it() {
    let db = compacted_people();
    let mut older = db.session();
    older.begin_transaction().unwrap();
    assert_eq!(older.state(), everyone(), "before the commit");

    // A commit after the older transaction began moves the epoch on; then
    // writes copy Gus (a new edge from him) and Alix's KNOWS edge (a value
    // the older transaction does not read) into the overlay.
    db.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    db.execute(
        "MATCH (g:Person {name: 'Gus'}), (c:City {name: 'Berlin'}) \
         INSERT (g)-[:LIVES_IN]->(c)",
    )
    .unwrap();
    db.execute("MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() SET k.note = 'Paris'")
        .unwrap();

    assert_eq!(
        older.state(),
        everyone(),
        "the older transaction sees the copied Gus, his age and both KNOWS edges"
    );
    assert_eq!(
        older.rows("MATCH (g:Person {name: 'Gus'}) RETURN labels(g) AS labels"),
        vec![vec![Value::List(vec![Value::from("Person")].into())]],
        "the copied Gus keeps his label for the older transaction"
    );
    assert_eq!(
        older.rows(CITIES),
        Vec::<Vec<Value>>::new(),
        "the commit after the older transaction began stays unseen"
    );
    older.rollback().unwrap();

    assert_eq!(db.state(), everyone(), "a new reader");
    assert_eq!(
        db.rows("MATCH (:Person {name: 'Gus'})-[:LIVES_IN]->(c:City) RETURN c.name AS name"),
        names(&["Berlin"]),
        "a new reader sees the edge the copy got"
    );
}

/// A delete without `DETACH` refuses a node with edges, but not the edges the
/// transaction deleted itself, which the base keeps until the commit.
#[test]
fn a_delete_without_detach_counts_only_the_edges_the_transaction_kept() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() DELETE k")
        .unwrap();
    let refused = session.execute("MATCH (g:Person {name: 'Gus'}) DELETE g");
    assert!(
        refused.is_err(),
        "Gus still knows Mia, so he cannot be deleted without DETACH"
    );
    session.rollback().unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Gus'})-[k:KNOWS]-() DELETE k")
        .unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) DELETE g")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(db.state(), without_gus());
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod file {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use grafeo_common::testing::child_process;

    use super::*;

    /// A database at `path` holding [`INSERT_PEOPLE`], compacted and closed:
    /// the file holds the compacted base.
    fn compacted_file(path: &Path) {
        let mut db = GrafeoDB::open(path).unwrap();
        db.execute(INSERT_PEOPLE).unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    /// Gus gets a new age (which copies him into the overlay), then a
    /// committed `DETACH DELETE`: after a close and a reopen he stays deleted,
    /// although the overlay no longer holds the copy (N1).
    #[test]
    fn a_base_node_updated_then_deleted_stays_deleted_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        {
            let db = GrafeoDB::open(&path).unwrap();
            db.execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 3")
                .unwrap();
            db.execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
                .unwrap();
            assert_eq!(db.state(), without_gus());
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.state(), without_gus(), "after the reopen");
        db.close().unwrap();
    }

    /// Nodes created after `compact()` are deleted after a reopen, with or
    /// without their edges, and stay deleted (N6).
    #[test]
    fn a_node_created_after_compact_can_be_deleted_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        {
            let db = GrafeoDB::open(&path).unwrap();
            db.execute(
                "INSERT (:Person {name: 'Vincent'})-[:KNOWS {since: 3}]->\
                 (:Person {name: 'Jules'})",
            )
            .unwrap();
            db.execute("INSERT (:City {name: 'Prague'})").unwrap();
            db.close().unwrap();
        }
        {
            let db = GrafeoDB::open(&path).unwrap();
            db.execute("MATCH (v:Person {name: 'Vincent'}) DETACH DELETE v")
                .unwrap();
            db.execute("MATCH (c:City {name: 'Prague'}) DELETE c")
                .unwrap();
            assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Jules", "Mia"]));
            assert_eq!(db.rows(CITIES), Vec::<Vec<Value>>::new());
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Jules", "Mia"]));
        assert_eq!(db.rows(CITIES), Vec::<Vec<Value>>::new());
        assert_eq!(db.rows(KNOWS), everyone().1, "Vincent's edge went with him");
        db.close().unwrap();
    }

    /// The database path a child process works on.
    const PATH_VAR: &str = "GRAFEO_COMPACT_TRANSACTIONS_PATH";
    /// Which child: "open", "updated_open" or "committed".
    const CHILD_VAR: &str = "GRAFEO_COMPACT_TRANSACTIONS_CHILD";
    /// Exit code of a child that reached its end.
    const EXITED: i32 = 19;

    /// The WAL next to the database file.
    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Runs the child `which` on the database at `path`; it exits without
    /// `close()`, like a crash, so the parent reopens the file and the WAL
    /// the child left.
    fn crash_child(which: &str, path: &Path) {
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "file::compact_transactions_child", "--nocapture"])
                .env(CHILD_VAR, which)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert_eq!(
            output.status.code(),
            Some(EXITED),
            "the child {which} exited early:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            sidecar_wal(path).exists(),
            "the child left its WAL, as a crash does"
        );
    }

    /// Child-process entry for [`crash_child`]; a no-op when run directly.
    #[test]
    fn compact_transactions_child() {
        let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        if which == "updated_open" {
            // A committed new age, which copies Gus into the overlay.
            db.execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 3")
                .unwrap();
        }
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
            .unwrap();
        match which.as_str() {
            // The checkpoint runs while the delete is open, and the process
            // ends with the transaction still open.
            "open" | "updated_open" => {
                db.wal_checkpoint().unwrap();
            }
            // The delete commits, then a checkpoint writes it.
            "committed" => {
                session.commit().unwrap();
                db.wal_checkpoint().unwrap();
            }
            other => panic!("unknown child {other}"),
        }
        std::process::exit(EXITED);
    }

    /// A checkpoint while a transaction deletes Gus from the base, then a
    /// crash: the file keeps Gus and his edges, with their properties.
    #[test]
    fn a_checkpoint_during_an_open_base_delete_keeps_the_node() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        crash_child("open", &path);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.state(), everyone());
        db.close().unwrap();
    }

    /// Guarantee 9, second half: a committed new age copies Gus into the
    /// overlay, then a checkpoint while a transaction deletes him, then a
    /// crash: the file keeps the copy with the committed age (and his name),
    /// and his edges.
    #[test]
    fn a_checkpoint_during_an_open_delete_of_an_updated_base_node_keeps_the_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        crash_child("updated_open", &path);
        let db = GrafeoDB::open(&path).unwrap();
        let (people, knows, _) = everyone();
        assert_eq!(
            db.state(),
            (people, knows, vec![vec![Value::Int64(3)]]),
            "Gus with the committed age, and both edges"
        );
        db.close().unwrap();
    }

    /// A committed base delete, a checkpoint, then a crash: Gus stays deleted.
    #[test]
    fn a_checkpoint_after_a_committed_base_delete_keeps_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        crash_child("committed", &path);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.state(), without_gus());
        db.close().unwrap();
    }
}
