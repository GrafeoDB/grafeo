//! Embeddings an older build spilled come back into the database (#594).
//!
//! Before 0.6 a vector spill moved the embeddings of a column into
//! `<file>.spill/vectors_<label>%3A<property>.bin` and out of the database. An
//! open folds them back into their columns: a read-write open makes them
//! durable and deletes the old files, a read-only open keeps them in memory
//! and changes nothing. The fixtures (`fixtures/closed-while-spilled/`, see its
//! README) were closed while spilled by 0.5.44 and by a 0.6 development build.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test legacy_spill
//! cargo test -p grafeo-engine --features full,testing-crash-injection --test legacy_spill
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "vector-index",
    not(miri)
))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const SPILL_FILE: &str = "vectors_Item%3Aembedding.bin";

/// A copy of the fixture written by `version`, and the path of its database.
fn fixture(version: &str) -> (tempfile::TempDir, PathBuf) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/closed-while-spilled")
        .join(version);
    let dir = tempfile::tempdir().unwrap();
    copy_tree(&source, dir.path());
    let path = dir.path().join("spilled.grafeo");
    (dir, path)
}

fn copy_tree(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Every file under `dir` with its bytes.
fn files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap().flatten() {
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                let relative = entry.path().strip_prefix(dir).unwrap().to_path_buf();
                found.insert(relative, std::fs::read(entry.path()).unwrap());
            }
        }
    }
    found
}

/// The embedding of each `:Item`, by name (`None` for an item without one).
fn embeddings(db: &GrafeoDB) -> Vec<(String, Option<Vec<f32>>)> {
    db.execute("MATCH (n:Item) RETURN n.name AS name, n.embedding AS e ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            let name = match &row[0] {
                Value::String(name) => name.to_string(),
                other => panic!("name: {other:?}"),
            };
            let embedding = match &row[1] {
                Value::Vector(vector) => Some(vector.to_vec()),
                _ => None,
            };
            (name, embedding)
        })
        .collect()
}

/// What every fixture holds once its spill file is folded back in: Alix's
/// embedding from the spill file, Gus's newer one from the database file,
/// Jules's back (a removal while spilled before 0.6 is not recorded, as its
/// reload brought it back too), and no Vincent (deleted while spilled).
fn folded() -> Vec<(String, Option<Vec<f32>>)> {
    vec![
        ("Alix".to_string(), Some(vec![3.0, 19.0, 88.0])),
        ("Gus".to_string(), Some(vec![1988.0, 3.0, 19.0])),
        ("Jules".to_string(), Some(vec![3.19, 19.88, 88.3])),
    ]
}

fn spill_dir(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.spill", path.display()))
}

/// The nearest `:Item` to Alix's embedding, by name.
fn nearest_to_alix(db: &GrafeoDB) -> String {
    let hits = db
        .vector_search("Item", "embedding", &[3.0, 19.0, 88.0], 1, None, None)
        .unwrap();
    let node = db.get_node(hits[0].0).unwrap();
    match node.get_property("name") {
        Some(Value::String(name)) => name.to_string(),
        other => panic!("name: {other:?}"),
    }
}

/// A read-write open folds the spill file back in, makes it durable and
/// deletes the old files: the database file alone then holds everything.
#[test]
fn a_read_write_open_folds_the_spilled_embeddings_back_in() {
    let (dir, path) = fixture("0.6.0-dev");
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), folded());
    assert_eq!(nearest_to_alix(&db), "Alix");
    assert!(!spill_dir(&path).exists(), "the old files are gone");
    db.close().unwrap();
    drop(db);

    let elsewhere = tempfile::tempdir().unwrap();
    let copy = elsewhere.path().join("spilled.grafeo");
    std::fs::copy(&path, &copy).unwrap();
    assert_eq!(embeddings(&GrafeoDB::open(&copy).unwrap()), folded());
    drop(dir);
}

/// A read-only open folds the spill file in memory and changes nothing on
/// disk, for a 0.6 file and for a 0.5.x one.
#[test]
fn a_read_only_open_folds_in_memory_and_changes_nothing() {
    for version in ["0.6.0-dev", "0.5.44"] {
        let (dir, path) = fixture(version);
        let before = files(dir.path());
        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(embeddings(&db), folded(), "{version}");
        assert_eq!(nearest_to_alix(&db), "Alix", "{version}");
        db.reload_eligible(1.0);
        drop(db);
        assert_eq!(
            files(dir.path()),
            before,
            "{version}: the read-only open wrote"
        );
    }
}

/// A 0.5.x database closed while spilled migrates with its embeddings: the
/// read-only read of the old database folds them in before the 0.6 image is
/// written. Its spill directory is kept with the old file, as
/// `<p>.pre-0.6.spill`, so the way back to 0.5.x keeps them too, and a
/// read-only open of the kept copy folds them in memory.
#[test]
fn a_0_5_database_closed_while_spilled_migrates_with_its_embeddings() {
    let (dir, path) = fixture("0.5.44");
    let spill_files = files(&spill_dir(&path));
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), folded());
    db.close().unwrap();
    drop(db);
    assert!(!spill_dir(&path).exists(), "the spill directory moved");
    assert_eq!(
        files(&dir.path().join("spilled.grafeo.pre-0.6.spill")),
        spill_files,
        "kept byte for byte"
    );
    assert_eq!(embeddings(&GrafeoDB::open(&path).unwrap()), folded());

    let kept = GrafeoDB::open_read_only(dir.path().join("spilled.grafeo.pre-0.6")).unwrap();
    assert_eq!(embeddings(&kept), folded(), "the kept copy");
}

/// A 0.5.x database configured with a spill path migrates with every
/// embedding, and the way back keeps them: the 0.5.x file in the spill path
/// stays (the migrating open filled nothing from it, and never deletes there),
/// and the kept copy read with that path has them all.
#[test]
fn a_0_5_database_with_a_configured_spill_path_keeps_the_way_back() {
    let (dir, path) = fixture("0.5.44");
    let shared = dir.path().join("shared");
    std::fs::create_dir(&shared).unwrap();
    std::fs::rename(spill_dir(&path).join(SPILL_FILE), shared.join(SPILL_FILE)).unwrap();
    let before = files(&shared);

    let db = GrafeoDB::with_config(Config::persistent(&path).with_spill_path(&shared)).unwrap();
    assert_eq!(embeddings(&db), folded());
    db.close().unwrap();
    drop(db);
    assert_eq!(files(&shared), before, "the 0.5.x file stays");
    assert_eq!(
        embeddings(&GrafeoDB::open(&path).unwrap()),
        folded(),
        "the 0.6 file holds them"
    );
    let kept = GrafeoDB::with_config(
        Config::read_only(dir.path().join("spilled.grafeo.pre-0.6")).with_spill_path(&shared),
    )
    .unwrap();
    assert_eq!(
        embeddings(&kept),
        folded(),
        "the kept copy with its spill path"
    );
}

/// A migration into an encrypted file keeps the spilled embeddings: the new
/// file holds them (read back with the key), and the 0.5.x spill directory is
/// kept, byte for byte, as `<p>.pre-0.6.spill`.
#[cfg(feature = "encryption")]
#[test]
fn an_encrypting_migration_keeps_the_spilled_embeddings() {
    use std::sync::Arc;

    use grafeo_common::encryption::KeyChain;
    use grafeo_engine::config::EncryptionConfig;

    let (dir, path) = fixture("0.5.44");
    let spill_files = files(&spill_dir(&path));
    let chain = Arc::new(KeyChain::new([19; 32]));
    let keyed = || {
        let mut config = Config::persistent(&path);
        config.encryption = Some(EncryptionConfig {
            key_chain: Arc::clone(&chain),
        });
        config
    };

    let db = GrafeoDB::with_config(keyed()).unwrap();
    assert_eq!(embeddings(&db), folded());
    db.close().unwrap();
    drop(db);
    assert_eq!(
        files(&dir.path().join("spilled.grafeo.pre-0.6.spill")),
        spill_files,
        "kept byte for byte"
    );
    assert!(GrafeoDB::open(&path).is_err(), "the new file needs the key");
    assert_eq!(
        embeddings(&GrafeoDB::with_config(keyed()).unwrap()),
        folded()
    );
}

/// Old files in a configured spill path, which other databases may share,
/// are read by the migration of a 0.5.x database, only for its own vector
/// indexes, and nothing there is deleted: not the read file (another database
/// may still need it, and the kept copy does), not a file of an index this
/// database does not have, not the cache directories of other opens. The
/// embeddings are in the new database file.
#[test]
fn old_files_in_a_spill_path_are_folded_only_for_this_databases_indexes() {
    let (dir, path) = fixture("0.5.44");
    let shared = dir.path().join("shared");
    let cache = shared.join("grafeo-berlin.grafeo-19-0000000000001988");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::rename(spill_dir(&path).join(SPILL_FILE), shared.join(SPILL_FILE)).unwrap();
    let foreign = shared.join("vectors_Doc%3Aembedding.bin");
    std::fs::copy(shared.join(SPILL_FILE), &foreign).unwrap();
    std::fs::copy(shared.join(SPILL_FILE), cache.join(SPILL_FILE)).unwrap();
    let before = files(&shared);

    let db = GrafeoDB::with_config(Config::persistent(&path).with_spill_path(&shared)).unwrap();
    assert_eq!(embeddings(&db), folded());
    assert_eq!(files(&shared), before, "nothing in the shared path changes");
    db.close().unwrap();
    drop(db);
    assert!(shared.exists(), "the configured spill path stays");
    assert_eq!(embeddings(&GrafeoDB::open(&path).unwrap()), folded());
}

/// What the 0.6.0-dev fixture holds without its spill file: Gus's newer
/// embedding, which the database file holds, and nothing for Alix and Jules.
fn unfolded() -> Vec<(String, Option<Vec<f32>>)> {
    vec![
        ("Alix".to_string(), None),
        ("Gus".to_string(), Some(vec![1988.0, 3.0, 19.0])),
        ("Jules".to_string(), None),
    ]
}

/// A 0.6 database never reads old files in a configured spill path: they are
/// read only with a 0.5.x database. So one written by a 0.6 development build
/// that spilled there does not get those embeddings back (a documented
/// limit), and the file stays as it is.
#[test]
fn a_0_6_database_does_not_read_old_files_in_a_configured_spill_path() {
    let (dir, path) = fixture("0.6.0-dev");
    let configured = dir.path().join("configured");
    std::fs::create_dir(&configured).unwrap();
    std::fs::rename(
        spill_dir(&path).join(SPILL_FILE),
        configured.join(SPILL_FILE),
    )
    .unwrap();
    let before = files(&configured);

    let db = GrafeoDB::with_config(Config::persistent(&path).with_spill_path(&configured)).unwrap();
    assert_eq!(embeddings(&db), unfolded());
    drop(db);
    let db = GrafeoDB::with_config(Config::read_only(&path).with_spill_path(&configured)).unwrap();
    assert_eq!(embeddings(&db), unfolded(), "read-only");
    drop(db);
    assert_eq!(files(&configured), before, "the file stays");
}

/// After the migration of a 0.5.x database with a configured spill path, an
/// embedding removed in 0.6 stays removed, open after open, while the old file
/// that held it is still in the spill path (for the kept copy).
#[test]
fn an_embedding_removed_after_the_migration_stays_removed() {
    let (dir, path) = fixture("0.5.44");
    let shared = dir.path().join("shared");
    std::fs::create_dir(&shared).unwrap();
    std::fs::rename(spill_dir(&path).join(SPILL_FILE), shared.join(SPILL_FILE)).unwrap();
    let before = files(&shared);
    let open =
        || GrafeoDB::with_config(Config::persistent(&path).with_spill_path(&shared)).unwrap();

    let db = open();
    assert_eq!(embeddings(&db), folded());
    db.execute("MATCH (n:Item {name: 'Alix'}) REMOVE n.embedding")
        .unwrap();
    db.close().unwrap();
    drop(db);
    let mut without_alix = folded();
    without_alix[0].1 = None;
    for reopen in ["first", "second"] {
        let db = open();
        assert_eq!(embeddings(&db), without_alix, "{reopen} reopen");
        db.close().unwrap();
        drop(db);
    }
    assert_eq!(files(&shared), before, "the old file is still there");
}

/// An open of a database without an index for a spill file leaves the file
/// alone, also in `<file>.spill`.
#[test]
fn a_spill_file_without_an_index_here_is_left_alone() {
    let (dir, path) = fixture("0.6.0-dev");
    let foreign = spill_dir(&path).join("vectors_Doc%3Aembedding.bin");
    std::fs::copy(spill_dir(&path).join(SPILL_FILE), &foreign).unwrap();
    let foreign_bytes = std::fs::read(&foreign).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), folded());
    assert!(!spill_dir(&path).join(SPILL_FILE).exists());
    assert_eq!(std::fs::read(&foreign).unwrap(), foreign_bytes);
    drop(db);
    drop(dir);
}

/// The directory of `<file>.spill` that old files a read-write open could
/// not fold in completely move to.
fn kept_dir(path: &Path) -> PathBuf {
    spill_dir(path).join("kept")
}

/// An old spill file that does not read as one (cut short, another magic, no
/// header, a count no file can hold) is refused for good: its embeddings stay
/// out, a read-only open leaves it where it is, and a read-write open moves
/// it to `<file>.spill/kept/`, byte for byte, so no later open reads it again.
#[test]
fn an_old_spill_file_that_is_not_one_moves_to_kept() {
    let (_source_dir, source) = fixture("0.6.0-dev");
    let original = std::fs::read(spill_dir(&source).join(SPILL_FILE)).unwrap();
    let mut other_magic = original.clone();
    other_magic[..8].copy_from_slice(b"GRAFVEC0");
    let mut huge_count = original[..64].to_vec();
    huge_count[16..24].copy_from_slice(&(u64::MAX / 16).to_le_bytes());
    for (damage, bytes) in [
        ("cut short", original[..original.len() - 1].to_vec()),
        ("another magic", other_magic),
        ("no header", b"GRAFVEC1".to_vec()),
        ("a count no file can hold", huge_count),
    ] {
        let (_dir, path) = fixture("0.6.0-dev");
        let old = spill_dir(&path).join(SPILL_FILE);
        std::fs::write(&old, &bytes).unwrap();

        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(embeddings(&db), unfolded(), "{damage}: read-only");
        drop(db);
        assert_eq!(
            std::fs::read(&old).unwrap(),
            bytes,
            "{damage}: a read-only open moves nothing"
        );

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(embeddings(&db), unfolded(), "{damage}");
        assert!(!old.exists(), "{damage}: the file left the top level");
        assert_eq!(
            std::fs::read(kept_dir(&path).join(SPILL_FILE)).unwrap(),
            bytes,
            "{damage}: kept byte for byte"
        );
    }
}

/// A file refused for good (here vectors of 2 dimensions against an index of
/// 3) costs one rebuild and one checkpoint: the read-write open moves it to
/// `kept/` once both are done (an injected checkpoint failure ends the first
/// open and moves nothing), and the next open finds no old file, so it
/// neither rebuilds the index nor writes a checkpoint (an injected failure
/// would end it, and it passes no crash point, the rebuild's included).
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_refused_file_moves_to_kept_and_the_next_open_neither_rebuilds_nor_checkpoints() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at, with_failure_at};

    let (_dir, path) = fixture("0.6.0-dev");
    let old = spill_dir(&path).join(SPILL_FILE);
    write_old(&old, 2, &[(0, vec![3.0, 19.0])]);
    let bytes = std::fs::read(&old).unwrap();
    assert!(
        with_failure_at(1, || GrafeoDB::open(&path)).is_err(),
        "the first open writes a checkpoint"
    );
    assert_eq!(
        std::fs::read(&old).unwrap(),
        bytes,
        "a failed checkpoint moves nothing"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert!(!old.exists(), "the file left the top level");
    assert_eq!(
        std::fs::read(kept_dir(&path).join(SPILL_FILE)).unwrap(),
        bytes,
        "kept byte for byte"
    );
    db.close().unwrap();
    drop(db);

    let reopened = with_failure_at(1, || GrafeoDB::open(&path));
    assert!(
        reopened.is_ok(),
        "the next open writes a checkpoint: {:?}",
        reopened.err()
    );
    drop(reopened);
    let CrashResult::Completed(reopened) = with_crash_at(1, || GrafeoDB::open(&path)) else {
        panic!("the next open passed a crash point: a rebuild or a checkpoint");
    };
    let db = reopened.unwrap();
    assert_eq!(
        search_names(&db, &[1988.0, 3.0, 19.0], 3),
        vec!["Gus"],
        "the index rebuilt once"
    );
    assert_eq!(
        std::fs::read(kept_dir(&path).join(SPILL_FILE)).unwrap(),
        bytes,
        "nothing reads kept/"
    );
}

/// A kept file is never replaced: a name taken in `kept/` gets the first free
/// numeric suffix.
#[test]
fn a_kept_file_never_replaces_another() {
    let (_dir, path) = fixture("0.6.0-dev");
    let old = spill_dir(&path).join(SPILL_FILE);
    write_old(&old, 2, &[(0, vec![3.0, 19.0])]);
    let bytes = std::fs::read(&old).unwrap();
    let kept = kept_dir(&path);
    std::fs::create_dir(&kept).unwrap();
    std::fs::write(kept.join(SPILL_FILE), b"Vincent").unwrap();
    std::fs::write(kept.join(format!("{SPILL_FILE}.1")), b"Mia").unwrap();

    drop(GrafeoDB::open(&path).unwrap());
    assert!(!old.exists(), "the file left the top level");
    assert_eq!(
        files(&kept),
        BTreeMap::from([
            (PathBuf::from(SPILL_FILE), b"Vincent".to_vec()),
            (PathBuf::from(format!("{SPILL_FILE}.1")), b"Mia".to_vec()),
            (PathBuf::from(format!("{SPILL_FILE}.2")), bytes),
        ])
    );
}

/// Folding in is a load step, like WAL replay: no change event and no new
/// epoch. An open of the same database without its old spill file has the
/// same epoch and the same history for every item.
#[cfg(feature = "cdc")]
#[test]
fn folding_in_adds_no_change_event_and_no_epoch() {
    let open = |with_spill_file: bool| {
        let (dir, path) = fixture("0.6.0-dev");
        if !with_spill_file {
            std::fs::remove_dir_all(spill_dir(&path)).unwrap();
        }
        let db = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
        (dir, db)
    };
    let (_plain_dir, plain) = open(false);
    let (_folded_dir, folding) = open(true);
    assert_eq!(embeddings(&folding), folded());
    assert_ne!(embeddings(&plain), folded(), "the plain open folds nothing");

    assert_eq!(folding.current_epoch(), plain.current_epoch());
    for name in ["Alix", "Gus", "Jules"] {
        let ids = folding.find_nodes_by_property("name", &Value::from(name));
        assert_eq!(ids.len(), 1, "{name}");
        assert_eq!(
            format!("{:?}", folding.history(ids[0]).unwrap()),
            format!("{:?}", plain.history(ids[0]).unwrap()),
            "the history of {name}"
        );
    }
}

// ── Crashes ────────────────────────────────────────────────────────

/// The database a crash child opens.
#[cfg(feature = "testing-crash-injection")]
const PATH_VAR: &str = "GRAFEO_LEGACY_SPILL_PATH";
/// The crash point a crash child crashes at.
#[cfg(feature = "testing-crash-injection")]
const POINT_VAR: &str = "GRAFEO_LEGACY_SPILL_POINT";
/// Exit code of a child that crashed.
#[cfg(feature = "testing-crash-injection")]
const CRASHED: i32 = 3;
/// Exit code of a child whose open completed.
#[cfg(feature = "testing-crash-injection")]
const OPENED: i32 = 4;

/// Opens `path` read-write in a child process that crashes at the
/// `point`-th crash point. Returns the crash point's name, or `None` when the
/// open completed.
#[cfg(feature = "testing-crash-injection")]
fn open_in_child(point: u64, path: &Path) -> Option<String> {
    let output = grafeo_common::testing::child_process::output(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(POINT_VAR, point.to_string())
            .env(PATH_VAR, path),
    )
    .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    match output.status.code() {
        Some(OPENED) => None,
        Some(CRASHED) => {
            let (_, name) = stderr
                .lines()
                .find_map(|line| line.split_once("crash injection at: "))
                .unwrap_or_else(|| panic!("the child names no crash point:\n{stderr}"));
            Some(name.trim().to_string())
        }
        other => panic!("point {point}: the child exited with {other:?}:\n{stderr}"),
    }
}

/// Child-process entry for [`open_in_child`]; a no-op when run directly.
/// The crash exits from the panic hook, so the open database is never
/// dropped (a drop would close it: a clean close, not a crash).
#[cfg(feature = "testing-crash-injection")]
#[test]
fn crash_child() {
    let (Ok(point), Some(path)) = (std::env::var(POINT_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    std::panic::set_hook(Box::new(|info| {
        eprintln!("{info}");
        std::process::exit(CRASHED);
    }));
    grafeo_common::testing::crash::enable_crash_at(point.parse().unwrap());
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    grafeo_common::testing::crash::disable_crash();
    std::mem::forget(db);
    std::process::exit(OPENED);
}

/// A crash anywhere in a read-write open that folds a spill file in loses
/// nothing: the old file is still there after every crash (it is deleted only
/// once the database file holds its values), and the next open has every
/// embedding and deletes it. When the open completes and the process exits
/// without closing the database, the file is gone and the database file holds
/// every embedding: only the open's own checkpoint can have put them there.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_while_folding_in_loses_nothing() {
    let mut points = Vec::new();
    for point in 1.. {
        let (_dir, path) = fixture("0.6.0-dev");
        let crashed = open_in_child(point, &path);
        let Some(name) = crashed else {
            assert!(
                !spill_dir(&path).join(SPILL_FILE).exists(),
                "the completed open deleted the old file"
            );
            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(embeddings(&db), folded(), "after an open without a close");
            break;
        };
        assert!(
            spill_dir(&path).join(SPILL_FILE).exists(),
            "after a crash at {name} the old file is gone"
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(embeddings(&db), folded(), "after a crash at {name}");
        assert!(!spill_dir(&path).exists(), "after a crash at {name}");
        db.close().unwrap();
        points.push(name);
        assert!(point < 500, "the open never completed: {points:?}");
    }
    eprintln!("crash points swept: {points:?}");
    for expected in [
        "legacy_spill:rebuild",
        "legacy_spill:after_fill",
        "legacy_spill:before_delete",
    ] {
        assert!(
            points.iter().any(|name| name == expected),
            "the sweep missed {expected}: {points:?}"
        );
    }
}

/// A failed checkpoint ends the open and keeps the old file, byte for byte,
/// as the database file does not hold its values; the next open folds it and
/// deletes it.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_failed_checkpoint_keeps_the_old_file() {
    let (_dir, path) = fixture("0.6.0-dev");
    let old = spill_dir(&path).join(SPILL_FILE);
    let bytes = std::fs::read(&old).unwrap();
    let opened = grafeo_common::testing::crash::with_failure_at(1, || GrafeoDB::open(&path));
    assert!(opened.is_err(), "the failed checkpoint ends the open");
    drop(opened);
    assert_eq!(std::fs::read(&old).unwrap(), bytes, "the old file stays");
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), folded());
    assert!(!old.exists());
}

/// Writes an old spill file of `dimensions` with `records`, as the old builds
/// did.
fn write_old(path: &Path, dimensions: usize, records: &[(u64, Vec<f32>)]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GRAFVEC1");
    bytes.extend_from_slice(&(dimensions as u64).to_le_bytes());
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    bytes.resize(64, 0);
    for (id, vector) in records {
        bytes.extend_from_slice(&id.to_le_bytes());
        for value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// A spill file whose vectors have another number of dimensions than the
/// index (2 and 0 against 3; a foreign or damaged file) is not read, by a
/// read-only or a read-write open of either fixture: the embeddings it held
/// stay out (Gus keeps the one the database file holds), the file stays, byte
/// for byte (in `kept/` after a read-write open of a 0.6 file, with the kept
/// copy after a migration), and vector search works.
#[test]
fn a_spill_file_of_another_dimension_is_not_read() {
    let expected = vec![
        ("Alix".to_string(), None),
        ("Gus".to_string(), Some(vec![1988.0, 3.0, 19.0])),
        ("Jules".to_string(), None),
    ];
    for version in ["0.6.0-dev", "0.5.44"] {
        for dimensions in [2, 0] {
            let context = format!("{version}, {dimensions} dimensions");
            let vector: Vec<f32> = [3.0, 19.0].into_iter().take(dimensions).collect();
            let records: Vec<(u64, Vec<f32>)> = (0..4).map(|id| (id, vector.clone())).collect();

            let (_dir, path) = fixture(version);
            let old = spill_dir(&path).join(SPILL_FILE);
            write_old(&old, dimensions, &records);
            let bytes = std::fs::read(&old).unwrap();
            let db = GrafeoDB::open_read_only(&path).unwrap();
            assert_eq!(embeddings(&db), expected, "{context}: read-only");
            assert!(
                db.vector_search("Item", "embedding", &[3.0, 19.0, 88.0], 3, None, None)
                    .is_ok(),
                "{context}: read-only search"
            );
            drop(db);

            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(embeddings(&db), expected, "{context}");
            assert!(
                db.vector_search("Item", "embedding", &[3.0, 19.0, 88.0], 3, None, None)
                    .is_ok(),
                "{context}: search"
            );
            db.close().unwrap();
            drop(db);
            // A 0.5.x database migrated: its spill directory is kept with it.
            let kept = if version == "0.5.44" {
                PathBuf::from(format!("{}.pre-0.6.spill", path.display())).join(SPILL_FILE)
            } else {
                kept_dir(&path).join(SPILL_FILE)
            };
            assert_eq!(
                std::fs::read(&kept).unwrap(),
                bytes,
                "{context}: the file stays"
            );
            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(embeddings(&db), expected, "{context}: reopened");
        }
    }
}

/// The embeddings of the `:Person` nodes, by name.
fn person_embeddings(db: &GrafeoDB) -> Vec<(String, Option<Vec<f32>>)> {
    db.execute("MATCH (p:Person) RETURN p.name AS name, p.embedding AS e ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            let name = match &row[0] {
                Value::String(name) => name.to_string(),
                other => panic!("name: {other:?}"),
            };
            let embedding = match &row[1] {
                Value::Vector(vector) => Some(vector.to_vec()),
                _ => None,
            };
            (name, embedding)
        })
        .collect()
}

/// After the migration of a 0.5.x database with a configured spill path, a
/// removed embedding stays removed, and new nodes get no old embedding, open
/// after open, while the old file holding the ids of deleted nodes
/// (Vincent's, deleted in 0.5.x, and Jules's, deleted in 0.6) is still in the
/// path. Since 0.6 a reopen keeps each graph's next id, so the new nodes take
/// ids above every id the old file holds instead of those of deleted nodes.
#[test]
fn reused_ids_get_no_old_embedding_after_a_migration() {
    let (dir, path) = fixture("0.5.44");
    let shared = dir.path().join("shared");
    std::fs::create_dir(&shared).unwrap();
    std::fs::rename(spill_dir(&path).join(SPILL_FILE), shared.join(SPILL_FILE)).unwrap();
    let before = files(&shared);
    let open =
        || GrafeoDB::with_config(Config::persistent(&path).with_spill_path(&shared)).unwrap();

    let db = open();
    assert_eq!(embeddings(&db), folded());
    db.execute("MATCH (n:Item {name: 'Jules'}) DETACH DELETE n")
        .unwrap();
    db.execute("MATCH (n:Item {name: 'Alix'}) REMOVE n.embedding")
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = open();
    db.execute("INSERT (:Person {name: 'Mia'}), (:Person {name: 'Butch'})")
        .unwrap();
    let ids: Vec<Value> = db
        .execute("MATCH (p:Person) RETURN id(p) AS id ORDER BY id")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    assert_eq!(
        ids,
        vec![Value::Int64(4), Value::Int64(5)],
        "the new nodes take fresh ids, not those of deleted nodes the old file holds"
    );
    db.close().unwrap();
    drop(db);

    for reopen in ["first", "second"] {
        let db = open();
        assert_eq!(
            embeddings(&db),
            vec![
                ("Alix".to_string(), None),
                ("Gus".to_string(), Some(vec![1988.0, 3.0, 19.0])),
            ],
            "{reopen} reopen"
        );
        assert_eq!(
            person_embeddings(&db),
            vec![("Butch".to_string(), None), ("Mia".to_string(), None)],
            "{reopen} reopen"
        );
        db.close().unwrap();
        drop(db);
    }
    assert_eq!(files(&shared), before, "the old file is still there");
}

/// The fold fills only nodes of the index's label: a `:Person` node whose id
/// an old file in `<file>.spill` holds gets nothing, the `:Item` beside it
/// gets its embedding.
#[test]
fn the_fold_fills_only_nodes_of_the_index_label() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let item = db
        .create_node_with_props(&["Item"], [("name", Value::from("Alix"))])
        .unwrap();
    let person = db
        .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
        .unwrap();
    db.create_vector_index("Item", "embedding", Some(3), None, None, None, None)
        .unwrap();
    db.close().unwrap();
    drop(db);
    std::fs::create_dir(spill_dir(&path)).unwrap();
    write_old(
        &spill_dir(&path).join(SPILL_FILE),
        3,
        &[
            (item.as_u64(), vec![3.0, 19.0, 88.0]),
            (person.as_u64(), vec![19.0, 88.0, 3.0]),
        ],
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        embeddings(&db),
        vec![("Alix".to_string(), Some(vec![3.0, 19.0, 88.0]))]
    );
    assert_eq!(person_embeddings(&db), vec![("Mia".to_string(), None)]);
}

/// The `:Item` names a vector search returns, nearest first.
fn search_names(db: &GrafeoDB, query: &[f32], k: usize) -> Vec<String> {
    db.vector_search("Item", "embedding", query, k, None, None)
        .unwrap()
        .into_iter()
        .map(
            |(id, _)| match db.get_node(id).unwrap().get_property("name") {
                Some(Value::String(name)) => name.to_string(),
                other => panic!("name: {other:?}"),
            },
        )
        .collect()
}

/// The vector index of a database closed while spilled is rebuilt from the
/// stored values: its saved topology was kept while the values could not be
/// read, so it missed Gus's embedding set then. A search for Gus's own vector
/// finds him first, read-only, read-write and after a reopen, for both
/// fixtures; with a file of another dimension (nothing folded), a search
/// returns only Gus, never Alix or Jules, who have no embedding.
#[test]
fn search_finds_the_embeddings_the_fold_brings_back() {
    let gus = [1988.0, 3.0, 19.0];
    for version in ["0.6.0-dev", "0.5.44"] {
        let (_dir, path) = fixture(version);
        let db = GrafeoDB::open_read_only(&path).unwrap();
        let hits = search_names(&db, &gus, 3);
        assert_eq!(
            hits.first().map(String::as_str),
            Some("Gus"),
            "{version}: read-only {hits:?}"
        );
        assert_eq!(hits.len(), 3, "{version}: read-only");
        drop(db);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(search_names(&db, &gus, 3)[0], "Gus", "{version}");
        db.close().unwrap();
        drop(db);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(search_names(&db, &gus, 3)[0], "Gus", "{version}: reopened");

        let (_dir, path) = fixture(version);
        let records: Vec<(u64, Vec<f32>)> = (0..4).map(|id| (id, vec![3.0, 19.0])).collect();
        write_old(&spill_dir(&path).join(SPILL_FILE), 2, &records);
        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(
            search_names(&db, &gus, 3),
            vec!["Gus"],
            "{version}: nothing folded, read-only"
        );
        drop(db);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            search_names(&db, &gus, 3),
            vec!["Gus"],
            "{version}: nothing folded"
        );
        db.close().unwrap();
        drop(db);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            search_names(&db, &gus, 3),
            vec!["Gus"],
            "{version}: nothing folded, reopened"
        );
    }
}

/// A node that lost the index's label while spilled has its embedding only in
/// the old file (0.5.x kept a property when a label went, and its reload
/// brought the value back). The fold fills only nodes of the label, so Butch,
/// a `:Person` now, gets nothing; the read-write open then moves the file to
/// `<file>.spill/kept/`, byte for byte, instead of deleting it, so his
/// embedding is not lost, and no later open reads it.
#[test]
fn the_embedding_of_a_node_that_lost_the_label_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.create_node_with_props(
        &["Item"],
        [
            ("name", Value::from("Alix")),
            ("embedding", Value::Vector(vec![3.0, 19.0, 88.0].into())),
        ],
    )
    .unwrap();
    let butch = db
        .create_node_with_props(&["Person"], [("name", Value::from("Butch"))])
        .unwrap();
    db.create_vector_index("Item", "embedding", Some(3), None, None, None, None)
        .unwrap();
    db.close().unwrap();
    drop(db);
    std::fs::create_dir(spill_dir(&path)).unwrap();
    let old = spill_dir(&path).join(SPILL_FILE);
    write_old(&old, 3, &[(butch.as_u64(), vec![19.0, 88.0, 3.0])]);
    let bytes = std::fs::read(&old).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(person_embeddings(&db), vec![("Butch".to_string(), None)]);
    assert!(!old.exists(), "the file left the top level");
    assert_eq!(
        std::fs::read(kept_dir(&path).join(SPILL_FILE)).unwrap(),
        bytes,
        "Butch's embedding is kept, byte for byte"
    );
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        person_embeddings(&db),
        vec![("Butch".to_string(), None)],
        "nothing reads kept/"
    );
    assert_eq!(
        files(&kept_dir(&path)),
        BTreeMap::from([(PathBuf::from(SPILL_FILE), bytes)])
    );
}

/// A fold that fills nothing (the file holds only Gus's old embedding, and
/// the database file holds his newer one) still rebuilds the index, and the
/// rebuilt index is durable before the file goes: after an open that exits
/// without a close, the file is gone and a search for Gus's vector finds him.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_rebuild_without_a_fill_is_durable() {
    let (_dir, path) = fixture("0.6.0-dev");
    write_old(
        &spill_dir(&path).join(SPILL_FILE),
        3,
        &[(1, vec![19.0, 88.0, 3.0])],
    );
    assert_eq!(open_in_child(u64::MAX, &path), None, "the open completes");
    assert!(
        !spill_dir(&path).join(SPILL_FILE).exists(),
        "the old file was deleted"
    );
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(embeddings(&db), unfolded(), "nothing was filled");
    assert_eq!(search_names(&db, &[1988.0, 3.0, 19.0], 3), vec!["Gus"]);
}

/// A 0.5.x database that reloaded its spilled embeddings before it closed has
/// no old file, and the index it saved still misses Gus's embedding, set
/// while spilled. Every vector index of a 0.5.x database is rebuilt from the
/// stored values while it is read in place: a read-only open (in memory), the
/// migration (so the 0.6 file holds the rebuilt index, and its next open
/// rebuilds nothing) and a read-only open of the kept copy. Alix and Jules
/// had their embeddings only in the removed file: a search finds only Gus.
#[test]
fn a_0_5_database_without_its_spill_file_gets_its_vector_index_rebuilt() {
    let (dir, path) = fixture("0.5.44");
    std::fs::remove_dir_all(spill_dir(&path)).unwrap();
    let gus = [1988.0, 3.0, 19.0];

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(search_names(&db, &gus, 3), vec!["Gus"], "read-only");
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(search_names(&db, &gus, 3), vec!["Gus"], "migrated");
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        search_names(&db, &gus, 3),
        vec!["Gus"],
        "the 0.6 file holds the rebuilt index"
    );
    drop(db);
    let kept = GrafeoDB::open_read_only(dir.path().join("spilled.grafeo.pre-0.6")).unwrap();
    assert_eq!(search_names(&kept, &gus, 3), vec!["Gus"], "the kept copy");
}
