//! The history retention limit's count at an open, exactly (H4, H6): the bytes each time-index
//! entry holds after a reopen, or a crash and its replay, are those the same writes give in one
//! run, for every form the history takes: a replaced value, a label set, a deleted node with its
//! edges, a delete then a create, a row created and deleted in one transaction (which replaced
//! nothing), and a named graph's writes.

use std::collections::HashMap;
use std::path::Path;

use grafeo_common::types::Value;

use crate::config::{Config, HistoryRetention};
use crate::database::GrafeoDB;

/// A limit far above what the writes replace: the count runs, nothing is dropped.
const LIMIT: u64 = 1 << 30;

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_history_retention(HistoryRetention::Forever)
            .with_history_retention_limit(LIMIT),
    )
    .unwrap()
}

/// A value of `bytes` bytes, distinct per `tag`.
fn big(tag: &str, bytes: usize) -> Value {
    Value::from(format!("{tag}{}", "V".repeat(bytes)).as_str())
}

fn run(db: &GrafeoDB, query: &str, params: &[(&str, Value)]) {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect();
    db.execute_with_params(query, params)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
}

/// The writes before the reopen (or the checkpoint before the crash): values, label sets.
fn first_writes(db: &GrafeoDB) {
    run(
        db,
        "INSERT (:Person {name: 'Alix', bio: $a}), \
         (:Person {name: 'Gus', bio: $g})-[:KNOWS {note: $n}]->(:Person {name: 'Mia', bio: $m})",
        &[
            ("a", big("alix", 4096)),
            ("g", big("gus", 4096)),
            ("n", big("note", 2048)),
            ("m", big("mia", 4096)),
        ],
    );
    run(
        db,
        "MATCH (p:Person {name: 'Alix'}) SET p.bio = $bio",
        &[("bio", big("alix2", 8192))],
    );
    run(db, "MATCH (p:Person {name: 'Alix'}) SET p:Engineer", &[]);
    run(db, "MATCH (p:Person {name: 'Alix'}) REMOVE p:Engineer", &[]);
}

/// The writes after it: a row created and deleted in one transaction, a deleted node with its
/// edge, a delete then a create, a label set, and a named graph's writes.
fn second_writes(db: &GrafeoDB) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_with_params(
            "INSERT (:Temp {bio: $bio})",
            HashMap::from([("bio".to_string(), big("temp", 8192))]),
        )
        .unwrap();
    session.execute("MATCH (t:Temp) DELETE t").unwrap();
    session.commit().unwrap();
    drop(session);
    run(db, "MATCH (p:Person {name: 'Gus'}) DETACH DELETE p", &[]);
    run(db, "MATCH (p:Person {name: 'Mia'}) DELETE p", &[]);
    run(
        db,
        "INSERT (:Person {name: 'Vincent', bio: $bio})",
        &[("bio", big("vincent", 4096))],
    );
    run(db, "MATCH (p:Person {name: 'Alix'}) SET p:Manager", &[]);
    run(db, "CREATE GRAPH g", &[]);
    let session = db.session();
    session.use_graph("g");
    for (round, bytes) in [(0, 2048), (1, 4096), (2, 1024)] {
        session
            .execute_with_params(
                if round == 0 {
                    "INSERT (:City {name: 'Amsterdam', note: $note})"
                } else {
                    "MATCH (c:City {name: 'Amsterdam'}) SET c.note = $note"
                },
                HashMap::from([("note".to_string(), big(&format!("city{round}"), bytes))]),
            )
            .unwrap();
    }
    session.execute("MATCH (c:City) DELETE c").unwrap();
}

/// The time index's totals: each entry's epoch and the bytes replaced up to it.
fn totals(db: &GrafeoDB) -> Vec<(u64, u64)> {
    db.session()
        .inner()
        .unwrap()
        .transaction_manager()
        .history_totals()
}

/// The writes in one run of a database, and its totals.
fn reference() -> Vec<(u64, u64)> {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("one.grafeo"));
    first_writes(&db);
    second_writes(&db);
    let totals = totals(&db);
    db.close().unwrap();
    totals
}

#[test]
fn a_reopen_counts_every_form_of_the_history_as_one_run() {
    let expected = reference();
    assert!(
        expected.windows(2).any(|pair| pair[1].1 > pair[0].1),
        "the writes replace something: {expected:?}"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reopened.grafeo");
    let db = open(&path);
    first_writes(&db);
    db.close().unwrap();
    let db = open(&path);
    second_writes(&db);
    assert_eq!(totals(&db), expected);
    db.close().unwrap();
}

/// Every write, a close and a reopen: the whole history comes from the file's chunks.
#[test]
fn a_reopen_after_every_write_counts_as_one_run() {
    let expected = reference();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("loaded.grafeo");
    let db = open(&path);
    first_writes(&db);
    second_writes(&db);
    db.close().unwrap();
    let db = open(&path);
    assert_eq!(totals(&db), expected);
    db.close().unwrap();
}

/// A crash after the second writes, which the log alone holds: replay rebuilds them, and the
/// open counts them with the loaded history, a row created and deleted in one transaction as
/// nothing.
#[cfg(all(feature = "wal", feature = "testing-crash-injection"))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;

    use super::{first_writes, open, reference, second_writes, totals};

    const PATH_VAR: &str = "GRAFEO_HISTORY_LIMIT_FORMS_CRASH_PATH";

    /// Child-process entry: the first writes, a checkpoint, the second writes, then an exit
    /// without `close()`. A no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = open(&PathBuf::from(path));
        first_writes(&db);
        db.wal_checkpoint().unwrap();
        second_writes(&db);
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    #[test]
    fn a_replay_counts_every_form_of_the_history_as_one_run() {
        let expected = reference();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crashed.grafeo");
        let output =
            child_process::output(grafeo_common::child_test!("crash_child").env(PATH_VAR, &path))
                .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let db = open(&path);
        assert_eq!(totals(&db), expected);
        db.close().unwrap();
    }
}
