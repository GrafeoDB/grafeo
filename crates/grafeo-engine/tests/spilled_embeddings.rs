//! Spilled embeddings stay part of the database (#594).
//!
//! Spilling a vector property column moves its values into a cache file that
//! the column reads through: the database file, its copies and every reader
//! still see them, and the cache goes with the database. Before, a spill moved
//! the values into `<file>.spill` alone: copies (`save`, `to_memory`, backups,
//! snapshots) and the file on its own lost them, `RETURN n.embedding` read
//! NULL while spilled, a crash after a reload lost them for good, and a reload
//! on a read-only open deleted the spill file.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test spilled_embeddings
//! cargo test -p grafeo-engine --features full,encryption --test spilled_embeddings
//! ```
//!
//! (`--all-features` turns on `temporal`, which has no vector spill.)

#![cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    feature = "gql",
    feature = "wal",
    not(feature = "temporal"),
    not(miri)
))]

use std::path::{Path, PathBuf};

use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::testing::child_process;
use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, GrafeoDB};

/// `config` with the vector section forced to disk.
fn force_disk(config: Config) -> Config {
    config.with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        },
    )
}

fn embedding_of(i: u8) -> Vec<f32> {
    vec![1.0, f32::from(i) / 10.0, 0.0, 0.0]
}

/// Three indexed `:Item` nodes with embeddings close to the x axis.
fn indexed_items(db: &GrafeoDB) -> Vec<NodeId> {
    let ids = (0..3u8)
        .map(|i| {
            db.create_node_with_props(
                &["Item"],
                [("embedding", Value::Vector(embedding_of(i).into()))],
            )
            .unwrap()
        })
        .collect();
    db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
        .unwrap();
    ids
}

/// The embeddings of the `:Item` nodes as GQL returns them, in id order
/// (`None` for a node without one).
fn embeddings(db: &GrafeoDB) -> Vec<Option<Vec<f32>>> {
    db.execute("MATCH (n:Item) RETURN n.embedding AS e ORDER BY id(n)")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Vector(v) => Some(v.to_vec()),
            _ => None,
        })
        .collect()
}

/// What the three items hold before anything changes.
fn all_three() -> Vec<Option<Vec<f32>>> {
    (0..3).map(|i| Some(embedding_of(i))).collect()
}

/// The names in `dir`, sorted.
fn names_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

fn spill_dir(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.spill", path.display()))
}

fn cache_dir(path: &Path) -> PathBuf {
    spill_dir(path).join("cache")
}

/// A file database at `path` with three indexed embeddings, spilled.
fn spilled(path: &Path) -> (GrafeoDB, Vec<NodeId>) {
    let db = GrafeoDB::with_config(force_disk(Config::persistent(path))).unwrap();
    let ids = indexed_items(&db);
    assert!(db.buffer_manager().spill_all() > 0, "nothing was spilled");
    (db, ids)
}

// ── Readers and copies see spilled values ──────────────────────────

#[test]
fn return_embedding_reads_spilled_values() {
    let dir = tempfile::tempdir().unwrap();
    let (db, ids) = spilled(&dir.path().join("amsterdam.grafeo"));
    assert_eq!(embeddings(&db), all_three());
    let node = db.get_node(ids[1]).unwrap();
    assert_eq!(
        node.get_property("embedding"),
        Some(&Value::Vector(embedding_of(1).into()))
    );
}

#[test]
fn save_keeps_spilled_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = spilled(&dir.path().join("berlin.grafeo"));
    let copy = dir.path().join("berlin-copy.grafeo");
    db.save(&copy).unwrap();
    db.close().unwrap();
    assert_eq!(embeddings(&GrafeoDB::open(&copy).unwrap()), all_three());
}

#[test]
fn to_memory_keeps_spilled_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = spilled(&dir.path().join("paris.grafeo"));
    assert_eq!(embeddings(&db.to_memory().unwrap()), all_three());
}

#[test]
fn a_backup_keeps_spilled_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = spilled(&dir.path().join("prague.grafeo"));
    let backups = dir.path().join("backups");
    let segment = db.backup_full(&backups).unwrap();
    db.close().unwrap();
    let restored = dir.path().join("prague-restored.grafeo");
    GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &restored).unwrap();
    assert_eq!(embeddings(&GrafeoDB::open(&restored).unwrap()), all_three());
}

#[test]
fn a_snapshot_keeps_spilled_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _) = spilled(&dir.path().join("barcelona.grafeo"));
    let bytes = db.export_snapshot().unwrap();
    assert_eq!(
        embeddings(&GrafeoDB::import_snapshot(&bytes).unwrap()),
        all_three()
    );
}

/// Closed while spilled: the database file alone, copied elsewhere, holds
/// every embedding.
#[test]
fn the_file_alone_holds_spilled_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let (db, _) = spilled(&path);
    db.close().unwrap();
    drop(db);
    let elsewhere = tempfile::tempdir().unwrap();
    let copy = elsewhere.path().join("amsterdam.grafeo");
    std::fs::copy(&path, &copy).unwrap();
    assert_eq!(embeddings(&GrafeoDB::open(&copy).unwrap()), all_three());
}

/// A spill takes the embeddings off the heap (the purpose of spilling), and
/// a reload puts them back.
#[test]
fn a_spill_takes_the_embeddings_off_the_heap() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(force_disk(Config::persistent(
        dir.path().join("prague.grafeo"),
    )))
    .unwrap();
    for i in 0..500u16 {
        let embedding: Vec<f32> = (0..16u16).map(|j| f32::from(i * 16 + j)).collect();
        db.create_node_with_props(&["Item"], [("embedding", Value::Vector(embedding.into()))])
            .unwrap();
    }
    db.create_vector_index("Item", "embedding", Some(16), None, None, None, None)
        .unwrap();
    let heap = || db.memory_usage().store.node_properties_bytes;
    let before = heap();

    assert!(db.buffer_manager().spill_all() > 0);
    let spilled = heap();
    assert!(
        spilled * 10 < before,
        "spilled: {spilled} bytes, before: {before}"
    );
    assert!(db.reload_eligible(1.0) > 0);
    assert!(heap() >= before / 2, "reloaded: {} bytes", heap());
}

// ── Changes while spilled ──────────────────────────────────────────

/// A node deleted and an embedding removed while spilled stay gone after
/// the reload and after a reopen (a reload brought them back).
#[test]
fn removals_while_spilled_stay_removed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let (db, ids) = spilled(&path);
    assert!(db.delete_node(ids[0]).unwrap());
    assert!(db.remove_node_property(ids[1], "embedding").unwrap());
    assert_eq!(embeddings(&db), vec![None, Some(embedding_of(2))]);

    assert!(db.reload_eligible(1.0) > 0);
    assert_eq!(embeddings(&db), vec![None, Some(embedding_of(2))]);
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), vec![None, Some(embedding_of(2))]);
    assert!(db.get_node(ids[0]).is_none());
}

/// Search on a spilled index answers as it did in memory, and reads an
/// embedding written while spilled.
#[test]
fn search_while_spilled_answers_as_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(force_disk(Config::persistent(
        dir.path().join("paris.grafeo"),
    )))
    .unwrap();
    let ids = indexed_items(&db);
    let query = [1.0, 0.2, 0.0, 0.0];
    let in_memory = db
        .vector_search("Item", "embedding", &query, 3, None, None)
        .unwrap();
    assert!(db.buffer_manager().spill_all() > 0);
    let spilled = db
        .vector_search("Item", "embedding", &query, 3, None, None)
        .unwrap();
    assert_eq!(spilled, in_memory);
    assert_eq!(spilled[0].0, ids[2]);

    db.set_node_property(
        ids[0],
        "embedding",
        Value::Vector(vec![1.0, 0.2, 0.0, 0.0].into()),
    )
    .unwrap();
    let nearest = db
        .vector_search("Item", "embedding", &query, 1, None, None)
        .unwrap();
    assert!(
        nearest[0].1 < 1e-6,
        "the embedding written while spilled is read: {nearest:?}"
    );
}

// ── Crashes ────────────────────────────────────────────────────────

const CHILD_PATH: &str = "GRAFEO_SPILLED_EMBEDDINGS_PATH";
const CHILD_CASE: &str = "GRAFEO_SPILLED_EMBEDDINGS_CASE";

/// Runs `case` of [`crash_child`] on `path` in a child process, which exits
/// without closing the database.
fn crash_in_child(path: &Path, case: &str) {
    let output = child_process::output(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(CHILD_PATH, path)
            .env(CHILD_CASE, case),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "the child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child-process entry for [`crash_in_child`]; a no-op when run directly.
#[test]
fn crash_child() {
    let (Some(path), Ok(case)) = (std::env::var_os(CHILD_PATH), std::env::var(CHILD_CASE)) else {
        return;
    };
    let (db, _) = spilled(Path::new(&path));
    if case == "after_reload" {
        db.wal_checkpoint().unwrap();
        assert!(db.reload_eligible(1.0) > 0);
    }
    // A crash: no close, no checkpoint, no destructors.
    std::process::exit(0);
}

/// Spilled, checkpointed, reloaded, crashed: every embedding is there
/// (they were in the spill file only, which the reload had deleted).
#[test]
fn a_crash_after_a_reload_keeps_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    crash_in_child(&path, "after_reload");
    assert_eq!(embeddings(&GrafeoDB::open(&path).unwrap()), all_three());
}

/// A crash while spilled leaves the cache behind; the next read-write open
/// has every embedding and removes the stale cache.
#[test]
fn a_crash_while_spilled_keeps_embeddings_and_the_next_open_removes_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    crash_in_child(&path, "while_spilled");
    assert_eq!(names_in(&cache_dir(&path)).len(), 1, "the crash left it");
    // A 0.5.x spill file beside the cache is not the cache's to remove (one
    // of an index this database does not have, which the fold leaves alone).
    let old = spill_dir(&path).join("vectors_Doc%3Aembedding.bin");
    std::fs::write(&old, b"0.5.x").unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert!(!cache_dir(&path).exists(), "the stale cache was removed");
    assert_eq!(std::fs::read(&old).unwrap(), b"0.5.x", "the old file stays");
    assert_eq!(embeddings(&db), all_three());
}

// ── Where the cache lives ──────────────────────────────────────────

/// The cache goes with the database: a closed database still reads its
/// spilled values, and dropping it removes the cache and `<file>.spill`.
#[test]
fn the_cache_goes_with_the_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let (db, _) = spilled(&path);
    assert_eq!(names_in(&cache_dir(&path)).len(), 1, "the cache file");
    db.close().unwrap();
    assert_eq!(embeddings(&db), all_three(), "reads after close");
    drop(db);
    assert!(!spill_dir(&path).exists(), "{:?}", names_in(dir.path()));
}

/// After `close()` a handle still reads (and reloads) its spilled
/// embeddings but spills no more: the next read-write open owns
/// `<file>.spill/cache/`, and dropping the closed handle leaves that open's
/// cache file alone.
#[test]
fn a_closed_handle_spills_no_more_and_leaves_the_next_open_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let (first, _) = spilled(&path);
    let first_files = names_in(&cache_dir(&path));
    assert_eq!(first_files.len(), 1);
    first.close().unwrap();

    let second = GrafeoDB::with_config(force_disk(Config::persistent(&path))).unwrap();
    let second_files: Vec<String> = names_in(&cache_dir(&path))
        .into_iter()
        .filter(|name| !first_files.contains(name))
        .collect();
    assert_eq!(second_files.len(), 1, "the second open spilled at open");

    assert_eq!(embeddings(&first), all_three(), "the closed handle reads");
    first.reload_eligible(1.0);
    assert_eq!(
        first.buffer_manager().spill_all(),
        0,
        "no spill after close"
    );
    assert_eq!(embeddings(&first), all_three());
    drop(first);
    let left = names_in(&cache_dir(&path));
    assert!(
        second_files.iter().all(|name| left.contains(name)),
        "dropping the closed handle took the second open's file: {left:?}"
    );
    assert_eq!(embeddings(&second), all_three());
}

/// Dropping the vector index of a spilled column reloads the column, which
/// nothing would reload otherwise; its cache file goes.
#[test]
fn dropping_the_index_of_a_spilled_column_reloads_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let (db, _) = spilled(&path);
    assert_eq!(names_in(&cache_dir(&path)).len(), 1);
    assert!(db.drop_vector_index("Item", "embedding").unwrap());
    assert_eq!(names_in(&cache_dir(&path)), Vec::<String>::new());
    assert_eq!(embeddings(&db), all_three());
}

/// A reload deletes its cache file, and a second spill writes a new one.
#[test]
fn a_reload_deletes_its_cache_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let (db, _) = spilled(&path);
    assert_eq!(names_in(&cache_dir(&path)).len(), 1, "the cache file");
    assert!(db.reload_eligible(1.0) > 0);
    assert_eq!(names_in(&cache_dir(&path)), Vec::<String>::new());
    assert!(db.buffer_manager().spill_all() > 0);
    assert_eq!(names_in(&cache_dir(&path)).len(), 1);
    assert_eq!(embeddings(&db), all_three());
}

/// A read-only open spills to the system temp directory and writes nothing
/// beside the database; its reload deletes nothing there either.
#[test]
fn a_read_only_open_spills_to_the_temp_directory() {
    let dir = tempfile::tempdir().unwrap();
    let name = format!("ro-{}.grafeo", std::process::id());
    let path = dir.path().join(&name);
    {
        let db = GrafeoDB::open(&path).unwrap();
        indexed_items(&db);
        db.close().unwrap();
    }
    let beside = names_in(dir.path());

    let db = GrafeoDB::with_config(force_disk(Config::read_only(&path))).unwrap();
    let prefix = format!("grafeo-{name}-{}-", std::process::id());
    let temp_dirs = || -> Vec<String> {
        names_in(&std::env::temp_dir())
            .into_iter()
            .filter(|n| n.starts_with(&prefix))
            .collect()
    };
    // `ForceDisk` spilled the embeddings at open; this spills nothing more.
    db.buffer_manager().spill_all();
    let temp = temp_dirs();
    assert_eq!(temp.len(), 1, "the read-only cache");
    assert_eq!(
        names_in(&std::env::temp_dir().join(&temp[0])).len(),
        1,
        "the cache file"
    );
    assert_eq!(embeddings(&db), all_three());
    assert!(db.reload_eligible(1.0) > 0);
    assert_eq!(embeddings(&db), all_three());
    assert_eq!(names_in(dir.path()), beside, "nothing beside the database");
    drop(db);
    assert_eq!(
        temp_dirs(),
        Vec::<String>::new(),
        "removed with the database"
    );
    assert_eq!(names_in(dir.path()), beside);
}

/// Databases that share an explicit spill path each spill into their own
/// directory, and dropping one removes only its own.
#[test]
fn databases_sharing_a_spill_path_keep_their_own_cache() {
    let dir = tempfile::tempdir().unwrap();
    let shared = dir.path().join("shared-spill");
    let open = |config: Config| {
        let db = GrafeoDB::with_config(force_disk(config.with_spill_path(&shared))).unwrap();
        indexed_items(&db);
        assert!(db.buffer_manager().spill_all() > 0);
        db
    };
    let file_db = open(Config::persistent(dir.path().join("paris.grafeo")));
    let memory_db = open(Config::in_memory());
    let pid = std::process::id();
    let caches = || names_in(&shared);
    let both = caches();
    assert_eq!(both.len(), 2, "{both:?}");
    assert!(
        both.iter()
            .any(|n| n.starts_with(&format!("grafeo-paris.grafeo-{pid}-"))),
        "{both:?}"
    );
    assert!(
        both.iter()
            .any(|n| n.starts_with(&format!("grafeo-memory-{pid}-"))),
        "{both:?}"
    );

    drop(file_db);
    let left = caches();
    assert_eq!(left.len(), 1, "{left:?}");
    assert!(left[0].starts_with("grafeo-memory-"), "{left:?}");
    assert_eq!(embeddings(&memory_db), all_three());
    drop(memory_db);
    assert_eq!(caches(), Vec::<String>::new());
    assert!(
        !dir.path().join("paris.grafeo.spill").exists(),
        "nothing beside the file"
    );
}

/// An encrypted database never spills: a spill file would hold its
/// embeddings in plaintext.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypted_database_never_spills() {
    use grafeo_common::encryption::KeyChain;
    use grafeo_engine::config::EncryptionConfig;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encrypted.grafeo");
    // (`TierOverride::ForceDisk` with encryption is refused by the config.)
    let config = Config::persistent(&path)
        .with_encryption(EncryptionConfig::new(Arc::new(KeyChain::new([7; 32]))));
    let db = GrafeoDB::with_config(config).unwrap();
    indexed_items(&db);
    assert_eq!(db.buffer_manager().spill_all(), 0);
    assert_eq!(embeddings(&db), all_three());
    assert!(!spill_dir(&path).exists());
    db.close().unwrap();
    drop(db);
    assert_eq!(names_in(dir.path()), vec!["encrypted.grafeo".to_string()]);
}
