//! The per-query memory limit (#751): it caps a statement's operator memory, defaults to the
//! global limit and never exceeds it; at the cap an operator spills first and fails only when it
//! cannot. Each bound has a pair: spill impossible (`with_spill_limit(0)`) gives the error naming
//! that bound; spill possible gives the in-memory answer; a failed spill leaves no spill file; a
//! higher limit lets the same statement succeed. Tests match on the bound's name.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main memory_query_limit::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

/// `result` is the S001 refusal naming `bound` and `setting`.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>, bound: &str, setting: &str) {
    let error = result.expect_err("refused at the limit");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(text.contains(bound), "names the {bound}: {text}");
    if !setting.is_empty() {
        assert!(text.contains(setting), "names {setting}: {text}");
    }
    assert!(
        text.chars().any(|c| c.is_ascii_digit()),
        "carries the numbers: {text}"
    );
}

fn count(db: &GrafeoDB, label: &str) -> usize {
    db.execute(&format!("MATCH (n:{label}) RETURN n"))
        .unwrap()
        .rows()
        .len()
}

/// A sort that materializes all its rows: no `LIMIT`, which a top-k rewrite would keep to its
/// k rows; the test takes the first five itself.
const SORT: &str = "UNWIND range(1, 200000) AS i RETURN i ORDER BY i DESC";

fn sort_rows(db: &GrafeoDB) -> Result<Vec<Vec<grafeo_common::types::Value>>, Error> {
    db.execute(SORT).map(|result| result.rows()[..5].to_vec())
}

fn with(config: Config) -> GrafeoDB {
    GrafeoDB::with_config(config).unwrap()
}

/// The report's limit is the budget the database runs under, and the per-query limit defaults
/// to it. The default budget itself is pinned in `default_memory_budget.rs`.
#[test]
fn the_query_limit_defaults_to_the_global_one() {
    let db = GrafeoDB::new_in_memory();
    let usage = db.memory_usage();
    assert_eq!(usage.limit_bytes, db.buffer_manager().budget());
    assert_eq!(usage.query_limit_bytes, usage.limit_bytes);
}

#[test]
fn the_query_limit_never_exceeds_the_global_one() {
    let db = with(
        Config::in_memory()
            .with_memory_limit(32 * MIB)
            .with_query_memory_limit(256 * MIB),
    );
    assert_eq!(db.memory_usage().query_limit_bytes, 32 * MIB);
    let db = with(
        Config::in_memory()
            .with_memory_limit(256 * MIB)
            .with_query_memory_limit(8 * MIB),
    );
    assert_eq!(db.memory_usage().query_limit_bytes, 8 * MIB);
}

/// The query memory limit's pair: a sort past it spills and gives the in-memory answer; with no
/// spill it fails naming the query limit; a higher query limit lets it through.
#[test]
fn a_sort_past_the_query_limit_spills_or_names_the_limit() {
    let want = sort_rows(&GrafeoDB::new_in_memory()).unwrap();
    let spill = tempfile::tempdir().unwrap();
    let spilling = with(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_path(spill.path()),
    );
    assert_eq!(
        sort_rows(&spilling).unwrap(),
        want,
        "spilled, the same rows"
    );

    let no_spill = with(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_limit(0),
    );
    assert_refused(
        sort_rows(&no_spill),
        "query memory limit",
        "with_query_memory_limit",
    );

    let higher = with(
        Config::in_memory()
            .with_query_memory_limit(512 * MIB)
            .with_spill_limit(0),
    );
    assert_eq!(
        sort_rows(&higher).unwrap(),
        want,
        "a higher limit, in memory"
    );
}

/// An in-memory database without a spill path has nowhere to spill: a sort past the query
/// limit is refused naming that limit.
#[test]
fn an_in_memory_database_without_a_spill_path_refuses_at_the_query_limit() {
    let db = with(Config::in_memory().with_query_memory_limit(MIB));
    assert_refused(
        sort_rows(&db),
        "query memory limit",
        "with_query_memory_limit",
    );
}

/// A writing statement that fails at the query limit changes nothing.
#[test]
fn a_write_refused_at_the_query_limit_changes_nothing() {
    let db = with(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_limit(0),
    );
    assert_refused(
        db.execute("UNWIND range(1, 200000) AS i WITH i ORDER BY i DESC INSERT (:S {i: i})"),
        "query memory limit",
        "with_query_memory_limit",
    );
    assert_eq!(count(&db, "S"), 0);
}

#[test]
fn two_concurrent_queries_each_keep_their_own_limit() {
    let want = sort_rows(&GrafeoDB::new_in_memory()).unwrap();
    let spill = tempfile::tempdir().unwrap();
    let db = std::sync::Arc::new(with(
        Config::in_memory()
            .with_query_memory_limit(2 * MIB)
            .with_spill_path(spill.path()),
    ));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let db = std::sync::Arc::clone(&db);
            std::thread::spawn(move || sort_rows(&db).unwrap())
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), want);
    }
}

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
    under(&PathBuf::from(format!("{}.spill", path.display())))
}

/// The spill limit's pair: a spill past a small cap fails naming the spill limit and leaves no
/// spill file; a zero cap the same; unlimited spill gives the in-memory answer; reopened with a
/// higher cap the same statement succeeds.
#[test]
fn a_spill_past_the_spill_limit_names_it_and_leaves_no_file() {
    let want = sort_rows(&GrafeoDB::new_in_memory()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    for cap in [64 * 1024_u64, 0] {
        let db = with(
            Config::persistent(&path)
                .with_query_memory_limit(MIB)
                .with_spill_limit(cap),
        );
        assert_refused(sort_rows(&db), "spill limit", "with_spill_limit");
        assert_eq!(spill_bytes(&path), 0, "cap {cap}: no spill file is left");
        assert_eq!(db.memory_usage().spill.total_bytes, 0);
        assert_eq!(db.memory_usage().spill.limit_bytes, Some(cap));
        db.close().unwrap();
    }
    let db = with(
        Config::persistent(&path)
            .with_query_memory_limit(MIB)
            .with_unlimited_spill(),
    );
    assert_eq!(sort_rows(&db).unwrap(), want);
    assert_eq!(db.memory_usage().spill.limit_bytes, None);
    db.close().unwrap();
    let db = with(
        Config::persistent(&path)
            .with_query_memory_limit(MIB)
            .with_spill_limit(1 << 30),
    );
    assert_eq!(
        sort_rows(&db).unwrap(),
        want,
        "a higher cap lets it through"
    );
    db.close().unwrap();
}

/// The checkpoint room's pair: a spill that would leave too little room for the database's own
/// writes fails naming the room; with room it succeeds.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_spill_into_the_checkpoint_room_names_the_room() {
    use grafeo_storage::file::space::with_free_space;
    let want = sort_rows(&GrafeoDB::new_in_memory()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    let db = with(
        Config::persistent(&path)
            .with_query_memory_limit(MIB)
            .with_unlimited_spill(),
    );
    db.execute("UNWIND range(1, 2000) AS i INSERT (:Person {i: i})")
        .unwrap();
    assert_refused(with_free_space(Some(4_321), || sort_rows(&db)), "room", "");
    assert_eq!(
        with_free_space(Some(1 << 40), || sort_rows(&db)).unwrap(),
        want
    );
    db.close().unwrap();
}
