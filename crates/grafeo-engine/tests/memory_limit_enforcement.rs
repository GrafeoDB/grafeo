//! `Config::with_memory_limit` is a hard limit on the memory a database
//! tracks (#398).
//!
//! The contract: when a write or a query would take the database past its
//! limit, it first spills what it can (operators, and for a database file
//! its vectors and cold columns); when that is not enough, the statement
//! fails with a memory-limit error that names the bytes used and the limit,
//! and changes nothing. Today only the spillable memory is bounded (operator
//! spill, the vector cache, and each path search, which gets a quarter of
//! the limit); store writes are not, so the tests of the store's bound are
//! ignored until #398 is fixed.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test memory_limit_enforcement -- --include-ignored
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const LIMIT: usize = 32 * 1024 * 1024;

/// The bytes of one embedding's values (384 `f32`).
const EMBEDDING_BYTES: usize = 384 * 4;

/// A 384-dimension embedding, as the #398 reporter wrote them.
fn embedding(seed: usize) -> Value {
    #[expect(
        clippy::cast_precision_loss,
        reason = "small seeds; the values only need to differ"
    )]
    let base = seed as f32;
    Value::Vector((0..384).map(|i| base + i as f32 / 384.0).collect())
}

/// Writes embedding nodes until a write fails, at most `cap` of them: the
/// number written and the error, if one came.
fn write_until_refused(db: &GrafeoDB, cap: usize) -> (usize, Option<String>) {
    for written in 0..cap {
        let result = db
            .create_node(&["Blob"])
            .and_then(|id| db.set_node_property(id, "embedding", embedding(written)));
        if let Err(error) = result {
            return (written, Some(error.to_string()));
        }
    }
    (cap, None)
}

/// The limit is reported as configured, and each path search gets a
/// quarter of it: a search a 4 MiB limit refuses (its budget is 1 MiB)
/// runs under the default budget.
#[test]
fn memory_limit_is_reported_and_bounds_path_searches() {
    let complete = |db: &GrafeoDB| {
        db.execute("UNWIND range(0, 11) AS i INSERT (:K {i: i})")
            .unwrap();
        db.execute("MATCH (a:K), (b:K) WHERE a.i <> b.i INSERT (a)-[:E]->(b)")
            .unwrap();
    };
    let query = "MATCH (a:K {i: 0})-[:E*1..5]->(b) RETURN count(b)";

    let limit = 4 * 1024 * 1024;
    let limited = GrafeoDB::with_config(Config::in_memory().with_memory_limit(limit)).unwrap();
    assert_eq!(limited.memory_limit(), Some(limit));
    complete(&limited);
    let error = limited
        .execute(query)
        .expect_err("the search needs more than 1 MiB")
        .to_string();
    assert!(
        error.contains("more than the 1 MiB of memory a path search may use"),
        "{error}"
    );

    let unlimited = GrafeoDB::new_in_memory();
    assert_eq!(unlimited.memory_limit(), None);
    complete(&unlimited);
    // Every walk of 1 to 5 edges from one node of the complete graph on 12.
    assert_eq!(
        unlimited.execute(query).unwrap().rows(),
        [[Value::Int64(11 + 121 + 1331 + 14_641 + 161_051)]]
    );
}

/// An in-memory database has nothing to spill its store to: writes past its
/// limit fail with the memory-limit error, which names the limit, and the
/// failed write changes nothing. Every earlier write is there, the tracked
/// memory stays within the limit, and the database still answers reads.
#[test]
#[ignore = "#398: store writes are not bounded by the memory limit yet"]
fn writes_past_the_memory_limit_fail_or_spill() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(LIMIT)).unwrap();
    // Three times what fits: about 1.5 KiB per embedding.
    let (written, error) = write_until_refused(&db, 3 * LIMIT / 1536);
    let error = error.expect("a write past the limit fails");
    assert!(
        error.to_lowercase().contains("memory limit")
            && (error.contains("32 MiB") || error.contains(&LIMIT.to_string())),
        "the error names the limit: {error}"
    );
    assert!(written > 0, "writes within the limit succeed");
    let count = i64::try_from(written).unwrap();
    assert_eq!(
        db.execute("MATCH (b:Blob) WHERE b.embedding IS NOT NULL RETURN count(b)")
            .unwrap()
            .rows(),
        [[Value::Int64(count)]],
        "every write before the refused one, and nothing of it"
    );
    let held = db.memory_usage().total_bytes;
    assert!(
        held <= LIMIT && held >= written * EMBEDDING_BYTES,
        "the tracked memory {held} counts the {written} embeddings and stays within the limit {LIMIT}"
    );
}

/// The bytes of every file under `dir`, 0 when it does not exist.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn bytes_under(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .map(|entry| {
            let path = entry.unwrap().path();
            if path.is_dir() {
                bytes_under(&path)
            } else {
                usize::try_from(std::fs::metadata(&path).unwrap().len()).unwrap()
            }
        })
        .sum()
}

/// A database file spills its vectors at its limit instead of failing:
/// three times what fits in memory is written without an error, every
/// embedding reads back, the tracked memory stays within the limit, and
/// what it does not hold is in the spill directory beside the file.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
#[ignore = "#398: store writes are not bounded by the memory limit yet"]
fn a_persistent_database_spills_vectors_at_its_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(LIMIT)).unwrap();
    let cap = 3 * LIMIT / 1536;
    let (written, error) = write_until_refused(&db, cap);
    assert_eq!(
        (written, error),
        (cap, None),
        "a database file spills, it does not fail"
    );
    let held = db.memory_usage().total_bytes;
    let mut spill = path.as_os_str().to_owned();
    spill.push(".spill");
    let spilled = bytes_under(std::path::Path::new(&spill));
    assert!(
        held <= LIMIT && held + spilled >= cap * EMBEDDING_BYTES,
        "{held} bytes held (limit {LIMIT}) and {spilled} spilled hold the {cap} embeddings"
    );
    for seed in [0, cap / 2, cap - 1] {
        let rows = db
            .execute_with_params(
                "MATCH (b:Blob) WHERE b.embedding = $e RETURN count(b)",
                std::iter::once(("e".to_string(), embedding(seed))).collect(),
            )
            .unwrap();
        assert_eq!(rows.rows(), [[Value::Int64(1)]], "embedding {seed}");
    }
    db.close().unwrap();
}

/// `memory_usage()` counts the values a database holds: a limit on tracked
/// memory bounds nothing if 15 MB of embeddings are tracked as a few.
#[test]
fn memory_usage_counts_vector_values() {
    let db = GrafeoDB::new_in_memory();
    let before = db.memory_usage().total_bytes;
    let (written, error) = write_until_refused(&db, 10_000);
    assert_eq!((written, error), (10_000, None));
    let held = db.memory_usage().total_bytes;
    assert!(
        held - before >= written * EMBEDDING_BYTES,
        "{written} embeddings of {EMBEDDING_BYTES} bytes are tracked as {} bytes",
        held - before
    );
}
