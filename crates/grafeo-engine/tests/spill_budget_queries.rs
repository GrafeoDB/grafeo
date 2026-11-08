//! The queries that spill against the budget (#751, with the operators' memory grants): a sort
//! past a 1 MiB query limit spills, so queries spilling at once share the one cap, the room bound
//! holds on a spill volume of its own, and an encrypted database refuses instead of spilling.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main spill_budget_queries::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "spill"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

/// A sort that materializes all its rows: no `LIMIT`, which a top-k rewrite would keep to its
/// k rows; the test takes the first five itself.
const SORT: &str = "UNWIND range(1, 200000) AS i RETURN i ORDER BY i DESC";

fn sort_rows(db: &GrafeoDB) -> Result<Vec<Vec<Value>>, Error> {
    db.execute(SORT).map(|result| result.rows()[..5].to_vec())
}

/// The sort's answer, from a database that never spills.
fn want() -> Vec<Vec<Value>> {
    sort_rows(&GrafeoDB::new_in_memory()).unwrap()
}

/// `result` is the S001 refusal naming `bound`.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>, bound: &str) {
    let error = result.expect_err("refused");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(text.contains(bound), "names the {bound}: {text}");
}

/// The bytes under `<path>.spill`.
fn spill_bytes(path: &Path) -> u64 {
    fn under(dir: &Path) -> u64 {
        std::fs::read_dir(dir).map_or(0, |entries| {
            entries
                .map(|entry| {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        under(&path)
                    } else {
                        std::fs::metadata(&path).unwrap().len()
                    }
                })
                .sum()
        })
    }
    under(&spill_root(path))
}

fn spill_root(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.spill", path.display()))
}

/// Queries that spill at once share the one cap: the spilled bytes never pass it, each query
/// answers or is refused naming the spill limit, and the account is back at 0 afterwards.
#[test]
fn concurrent_spilling_queries_share_the_cap() {
    let want = want();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ghent.grafeo");
    let cap = 4 * MIB as u64;
    let db = Arc::new(
        GrafeoDB::with_config(
            Config::persistent(&path)
                .with_query_memory_limit(MIB)
                .with_spill_limit(cap),
        )
        .unwrap(),
    );
    let done = Arc::new(AtomicBool::new(false));
    let highest = Arc::new(AtomicU64::new(0));
    let sampler = {
        let (db, done, highest) = (Arc::clone(&db), Arc::clone(&done), Arc::clone(&highest));
        std::thread::spawn(move || {
            while !done.load(Ordering::SeqCst) {
                let spilled = db.memory_usage().spill.total_bytes;
                highest.fetch_max(spilled, Ordering::SeqCst);
                std::thread::yield_now();
            }
        })
    };
    let queries: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || sort_rows(&db))
        })
        .collect();
    for query in queries {
        match query.join().unwrap() {
            Ok(rows) => assert_eq!(rows, want),
            Err(error) => assert_refused(Err::<(), _>(error), "spill limit"),
        }
    }
    done.store(true, Ordering::SeqCst);
    sampler.join().unwrap();
    assert!(
        highest.load(Ordering::SeqCst) <= cap,
        "{} bytes spilled at once under a cap of {cap}",
        highest.load(Ordering::SeqCst)
    );
    assert_eq!(db.memory_usage().spill.total_bytes, 0);
    assert_eq!(spill_bytes(&path), 0, "no spill file is left");
}

/// The room bound holds on a spill volume of its own: a spill path whose volume has too little
/// free space is refused naming the room; with room the sort answers.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn the_room_bound_holds_on_another_spill_volume() {
    use grafeo_storage::file::space::with_free_space_under;

    let want = want();
    let dir = tempfile::tempdir().unwrap();
    let spill = tempfile::tempdir().unwrap();
    let path = dir.path().join("bruges.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_spill_path(spill.path())
            .with_query_memory_limit(MIB)
            .with_unlimited_spill(),
    )
    .unwrap();
    db.execute("UNWIND range(1, 2000) AS i INSERT (:Person {i: i})")
        .unwrap();
    assert_refused(
        with_free_space_under(spill.path(), Some(4_321), || sort_rows(&db)),
        "room",
    );
    assert_eq!(
        with_free_space_under(spill.path(), Some(1 << 40), || sort_rows(&db)).unwrap(),
        want
    );
    db.close().unwrap();
}

/// An encrypted database spills nothing: a sort past the query limit is refused naming that
/// limit, and no spill file is written.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_database_spills_nothing() {
    use grafeo_common::encryption::KeyChain;
    use grafeo_engine::config::EncryptionConfig;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mechelen.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_encryption(EncryptionConfig::new(Arc::new(KeyChain::new([19; 32]))))
            .with_query_memory_limit(MIB)
            .with_unlimited_spill(),
    )
    .unwrap();
    assert_refused(sort_rows(&db), "query memory limit");
    assert_eq!(spill_bytes(&path), 0, "nothing was spilled");
    assert_eq!(db.memory_usage().spill.total_bytes, 0);
    db.close().unwrap();
}
