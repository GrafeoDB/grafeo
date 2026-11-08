//! `Config::with_memory_limit` is a hard limit on the memory a database
//! tracks (#398).
//!
//! The contract: when a write or a query would take the database past its
//! limit, it first spills what it can (operators, and for a database file
//! its indexed vectors); when that is not enough, the statement fails with a
//! memory-limit error that names the bytes used and the limit, and changes
//! nothing. Vectors without an index and cold columns do not spill yet
//! (#713): their test is a strict expected failure until they do.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main memory_limit_enforcement::
//! ```

#![cfg(feature = "gql")]

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
    // The limit binds the granted bytes (the caches outside it), with one grant step for the
    // forced overshoot of one op; the report counts the embeddings.
    let usage = db.memory_usage();
    let granted = usage.buffer_manager.allocated_bytes;
    let step = db.buffer_manager().grant_step();
    assert!(
        granted <= LIMIT + step,
        "{granted} bytes granted, past the limit {LIMIT} and one step of {step}"
    );
    assert!(
        usage.total_bytes >= written * EMBEDDING_BYTES,
        "the tracked memory {} counts the {written} embeddings",
        usage.total_bytes
    );
}

/// A strict expected failure: `body` must panic (the behavior `spec` names is not there yet,
/// `issue` tracks it); if it passes, the test fails so the mark is removed.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn xfail(spec: &str, issue: &str, body: impl FnOnce()) {
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err();
    assert!(
        failed,
        "XPASS: {spec} ({issue}) holds now; remove the xfail"
    );
}

/// The rows of a row group.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
const ROW_GROUP: usize = 65_536;

/// The dimensions of the spill rows' vectors: 128 bytes each, so one row group's embedding
/// column is 8 MiB and four groups pass the 32 MiB limit.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
const DIMENSIONS: usize = 32;

/// A vector of [`DIMENSIONS`] that differs per seed.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn small_vector(seed: usize) -> Value {
    #[expect(
        clippy::cast_precision_loss,
        reason = "small seeds; values only differ"
    )]
    let base = seed as f32;
    Value::Vector((0..DIMENSIONS).map(|i| base + i as f32 / 64.0).collect())
}

/// Writes `:Blob` nodes `from..to` with an embedding each; the first id written. Panics on a
/// refused write, naming the row.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn write_blobs(db: &GrafeoDB, from: usize, to: usize) -> grafeo_common::types::NodeId {
    use grafeo_common::types::PropertyKey;

    let mut first = None;
    for seed in from..to {
        let id = db
            .create_node_with_props(
                &["Blob"],
                [(PropertyKey::new("embedding"), small_vector(seed))],
            )
            .unwrap_or_else(|error| panic!("row {seed} of {to}: {error}"));
        first.get_or_insert(id);
    }
    first.expect("rows were written")
}

/// A database file opened with `config`, with the vector index when `index`.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn blobs_file(config: Config, index: bool) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    #[cfg(feature = "vector-index")]
    if index {
        db.create_vector_index(
            "Blob",
            "embedding",
            Some(DIMENSIONS),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    }
    db
}

/// The limit at which `rows` blobs (with `index`, a vector index first) load only by spilling
/// the older row groups' columns: measured on a database without a limit holding the same rows,
/// what stays granted once every column is spilled (the store and the vector index, which a
/// spill never frees and which grows with every row), plus 1.1 times one row group's column,
/// plus four grant steps. The newest group (at most one group's column) and the whole index fit
/// under it; two groups' columns do not.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn pressure_limit(dir: &std::path::Path, index: bool, rows: usize) -> usize {
    let path = dir.join("measured.grafeo");
    let db = blobs_file(Config::persistent(&path), index);
    write_blobs(&db, 0, rows);
    let everything = db.memory_usage().buffer_manager.allocated_bytes;
    db.buffer_manager()
        .spill_consumer_by_name("section:VectorStore");
    let without_columns = db.memory_usage().buffer_manager.allocated_bytes;
    let step = db.buffer_manager().grant_step();
    db.close().unwrap();
    let column = (everything - without_columns) * ROW_GROUP / rows;
    without_columns + column * 11 / 10 + 4 * step
}

/// A database file at its limit holding four row groups and a bit of `:Blob` nodes with an
/// embedding each (with `index`, a vector index on them first), at the [`pressure_limit`] of
/// those rows: the newest group never spills under pressure, so the writes into groups 1 to 4
/// complete because the full groups' columns spill. Returns the database, its limit, the first
/// node's id and the number of rows; panics on a refused write.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
fn spilled_across_row_groups(
    dir: &std::path::Path,
    index: bool,
) -> (GrafeoDB, usize, grafeo_common::types::NodeId, usize) {
    let rows = 4 * ROW_GROUP + 1_000;
    let limit = pressure_limit(dir, index, rows);
    let path = dir.join("vectors.grafeo");
    let db = blobs_file(Config::persistent(&path).with_memory_limit(limit), index);
    let first = write_blobs(&db, 0, rows);
    (db, limit, first, rows)
}

/// A database file spills its indexed vectors at its limit instead of failing: rows over
/// several row groups load under the limit, the spill happened, and every sampled embedding
/// reads back.
#[cfg(all(feature = "grafeo-file", feature = "wal", feature = "vector-index"))]
#[test]
fn a_persistent_database_spills_vectors_at_its_limit() {
    let dir = tempfile::tempdir().unwrap();
    let (db, limit, _, rows) = spilled_across_row_groups(dir.path(), true);
    let usage = db.memory_usage();
    assert!(
        usage.spill.vector_column_bytes > 0,
        "earlier groups' embedding columns spilled: {:?}",
        usage.spill
    );
    let step = db.buffer_manager().grant_step();
    assert!(
        usage.buffer_manager.allocated_bytes <= limit + step,
        "{} bytes granted under the limit {limit} and one step of {step}",
        usage.buffer_manager.allocated_bytes
    );
    for seed in [0, ROW_GROUP + 1, rows / 2, rows - 1] {
        let found = db
            .execute_with_params(
                "MATCH (b:Blob) WHERE b.embedding = $e RETURN count(b)",
                std::iter::once(("e".to_string(), small_vector(seed))).collect(),
            )
            .unwrap();
        assert_eq!(found.rows(), [[Value::Int64(1)]], "embedding {seed}");
    }
    db.close().unwrap();
}

/// Vectors without an index are a cold column the limit cannot spill yet: the write past the
/// limit fails instead.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_persistent_database_spills_unindexed_vectors_at_its_limit() {
    xfail(
        "Config::with_memory_limit: a database file spills its vectors and cold columns",
        "#713",
        || {
            let dir = tempfile::tempdir().unwrap();
            let (db, _, _, _) = spilled_across_row_groups(dir.path(), false);
            assert!(db.memory_usage().spill.vector_column_bytes > 0);
            db.close().unwrap();
        },
    );
}

/// A write into a row group whose embedding column just spilled, at the limit, succeeds: the
/// stale value's tail write needs no room for the spilled column (held as a #713 strict
/// xfail until patch 8's write path made it hold; the xfail came off).
#[cfg(all(feature = "grafeo-file", feature = "wal", feature = "vector-index"))]
#[test]
fn a_write_into_a_spilled_row_group_at_the_limit_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _, first, _) = spilled_across_row_groups(dir.path(), true);
    db.set_node_property(first, "embedding", small_vector(1_000_000))
        .unwrap();
    db.close().unwrap();
}

/// Vectors that all sit in the newest row group (fewer than 65,536 rows) have nothing to spill:
/// the consumer keeps the group writes still go to. A write past the limit is then refused with
/// the memory limit's error, the store as before and nothing spilled. The limit is what 20,000
/// rows with the index are granted, measured without a limit, so it binds within the group.
#[cfg(all(feature = "grafeo-file", feature = "wal", feature = "vector-index"))]
#[test]
fn vectors_only_in_the_newest_row_group_are_refused_not_spilled() {
    use grafeo_common::types::PropertyKey;

    let dir = tempfile::tempdir().unwrap();
    let measure = dir.path().join("twenty_thousand.grafeo");
    let limit = {
        let db = blobs_file(Config::persistent(&measure), true);
        write_blobs(&db, 0, 20_000);
        let granted = db.memory_usage().buffer_manager.allocated_bytes;
        db.close().unwrap();
        granted
    };
    let path = dir.path().join("append_only.grafeo");
    let db = blobs_file(Config::persistent(&path).with_memory_limit(limit), true);
    let mut written = 0;
    let mut refusal = None;
    while written < 60_000 {
        match db.create_node_with_props(
            &["Blob"],
            [(PropertyKey::new("embedding"), small_vector(written))],
        ) {
            Ok(_) => written += 1,
            Err(error) => {
                refusal = Some(error);
                break;
            }
        }
    }
    let error = refusal.expect("the limit binds within the newest row group");
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
    assert!(
        error.to_string().contains("memory limit"),
        "names the memory limit: {error}"
    );
    let count = i64::try_from(written).unwrap();
    assert_eq!(
        db.execute("MATCH (b:Blob) RETURN count(b)").unwrap().rows(),
        [[Value::Int64(count)]],
        "the refused write left nothing"
    );
    assert_eq!(
        db.memory_usage().spill.vector_column_bytes,
        0,
        "nothing of the newest group spilled"
    );
    db.close().unwrap();
}

/// Under pressure the newest row group keeps its column: with one full group and part of a
/// second, a load at a limit that forces eviction spills group 0's embedding column and not
/// group 1's: one spilled column, the newest group being group 1, and spilled bytes of one full
/// group (as an explicit spill of a database holding exactly one group measures them), not
/// that plus the second group's.
#[cfg(all(feature = "grafeo-file", feature = "wal", feature = "vector-index"))]
#[test]
#[allow(
    deprecated,
    reason = "which row group spilled is read from the raw store, which no session shows"
)]
fn pressure_spills_an_older_row_group_and_keeps_the_newest() {
    let dir = tempfile::tempdir().unwrap();
    // One full group: what an explicit spill of it writes.
    let one_group_column = {
        let path = dir.path().join("one_group.grafeo");
        let db = blobs_file(Config::persistent(&path), true);
        write_blobs(&db, 0, ROW_GROUP);
        db.buffer_manager()
            .spill_consumer_by_name("section:VectorStore");
        let column = db.memory_usage().spill.vector_column_bytes;
        db.close().unwrap();
        column
    };
    assert!(one_group_column > 0, "an explicit spill writes the group");

    let path = dir.path().join("two_groups.grafeo");
    let db = blobs_file(
        Config::persistent(&path).with_memory_limit(pressure_limit(
            dir.path(),
            true,
            ROW_GROUP + 60_000,
        )),
        true,
    );
    write_blobs(&db, 0, ROW_GROUP + 60_000);
    // Spill files are per group and column, and the rows hold one embedding column: exactly
    // group 0's column is spilled, and group 1 is the newest.
    let store = db.store().unwrap();
    assert_eq!(store.append_node_group(), Some(1), "group 1 is the newest");
    assert_eq!(
        store.memory().spilled_columns,
        1,
        "group 0's column spilled, the newest group's did not"
    );
    let spilled = db.memory_usage().spill.vector_column_bytes;
    assert!(
        spilled * 4 >= one_group_column * 3 && spilled * 4 <= one_group_column * 5,
        "group 0 spilled and group 1 stayed: {spilled} bytes spilled, one group is {one_group_column}"
    );
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
