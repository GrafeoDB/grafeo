//! The spill budget (#751): one cap for everything an open spills, a default from the spill
//! volume's free space, crashed queries' leftovers removed before the cap is measured, and the
//! report naming who spilled. The cap's own refusal (`with_spill_limit`, `with_unlimited_spill`, a
//! zero cap) and the room on the database's volume are in `memory_enforcement`; the queries that
//! spill against the cap are in `spill_budget_queries`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main spill_budget::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;
const GIB: u64 = 1 << 30;

fn spill_root(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.spill", path.display()))
}

/// The default cap is half the spill volume's free space at the open; when the volume cannot
/// be read, the memory budget.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn the_default_cap_is_half_the_free_space_at_open() {
    use grafeo_storage::file::space::with_free_space_under;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("antwerp.grafeo");
    GrafeoDB::open(&path).unwrap().close().unwrap();

    let db = with_free_space_under(dir.path(), Some(10 * GIB), || {
        GrafeoDB::open(&path).unwrap()
    });
    assert_eq!(db.memory_usage().spill.limit_bytes, Some(5 * GIB));
    db.close().unwrap();
    drop(db);

    let db = with_free_space_under(dir.path(), None, || {
        GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(64 * MIB)).unwrap()
    });
    assert_eq!(
        db.memory_usage().spill.limit_bytes,
        Some(64 * MIB as u64),
        "the memory budget, when the free space cannot be read"
    );
    db.close().unwrap();
}

/// A crashed query's spill directory is removed when the database opens, before the cap is
/// measured: a 1 GiB leftover neither stays on disk nor shrinks the default cap.
#[test]
fn a_crashed_query_leftover_is_removed_before_the_cap_is_measured() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("leuven.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let cap = db.memory_usage().spill.limit_bytes.expect("a default cap");
    db.close().unwrap();
    drop(db);

    let leftover = spill_root(&path).join("query_7");
    std::fs::create_dir_all(&leftover).unwrap();
    let run = std::fs::File::create(leftover.join("run_0.spill")).unwrap();
    run.set_len(GIB).unwrap();
    drop(run);

    let db = GrafeoDB::open(&path).unwrap();
    assert!(!leftover.exists(), "the leftover is removed at the open");
    let after = db.memory_usage().spill.limit_bytes.expect("a default cap");
    // Half the leftover would be 512 MiB off the cap; other writers on the volume move it less.
    assert!(
        after + 256 * MIB as u64 >= cap,
        "the cap was measured after the removal: {cap} before, {after} with the leftover"
    );
    db.close().unwrap();
}

/// The report's spill part names who spilled: a spilled vector column shows as
/// `vector_column_bytes` and in `total_bytes`, the queries' parts are 0 once they ended, and
/// the parts add up to the total; an unlimited cap reports no limit.
#[cfg(all(feature = "vector-index", feature = "mmap"))]
#[test]
fn the_report_names_who_spilled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("namur.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_unlimited_spill()).unwrap();
    for i in 0..200_u32 {
        let id = db.create_node(&["Item"]).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "small numbers")]
        let embedding: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32 / 1_600.0).collect();
        db.set_node_property(id, "embedding", Value::Vector(embedding.into()))
            .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(8), None, None, None, None)
        .unwrap();
    db.buffer_manager()
        .spill_consumer_by_name("section:VectorStore");
    let spill = db.memory_usage().spill;
    assert!(
        spill.vector_column_bytes > 0,
        "the column spilled: {spill:?}"
    );
    assert_eq!(spill.sort_bytes, 0);
    assert_eq!(spill.aggregate_bytes, 0);
    assert_eq!(
        spill.total_bytes,
        spill.sort_bytes + spill.aggregate_bytes + spill.vector_column_bytes
    );
    assert_eq!(spill.limit_bytes, None, "unlimited");
    db.close().unwrap();
}
