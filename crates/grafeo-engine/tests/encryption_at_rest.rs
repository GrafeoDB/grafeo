//! Encryption at rest, end to end.
//!
//! A database opened with `Config::encryption` writes its `.grafeo` file and
//! its sidecar WAL encrypted, with keys the key chain derives for that
//! database (`"grafeo-container"` and `"grafeo-wal"`, each with the database
//! id), and opens only with that key chain. These tests also run in CI with
//! the minimal feature set:
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test encryption_at_rest
//! cargo test -p grafeo-engine --no-default-features \
//!     --features lpg,gql,wal,grafeo-file,encryption --test encryption_at_rest
//! ```

// Miri cannot interpret the AES-NI intrinsics aes-gcm uses.
#![cfg(all(
    feature = "encryption",
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    not(miri)
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::encryption::KeyChain;
use grafeo_common::testing::child_process;
use grafeo_common::types::{EpochId, Value};
use grafeo_engine::config::EncryptionConfig;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::v3::header::FileHeaderV3;
use grafeo_storage::wal::WalRecovery;

/// A property value whose bytes must never be found in an encrypted file.
const MARKER: &str = "Amsterdam-3-19-88-marker";

/// A key chain whose master key is 32 bytes of `seed`.
fn key_chain(seed: u8) -> Arc<KeyChain> {
    Arc::new(KeyChain::new([seed; 32]))
}

fn with_key(mut config: Config, chain: &Arc<KeyChain>) -> Config {
    config.encryption = Some(EncryptionConfig {
        key_chain: Arc::clone(chain),
    });
    config
}

/// A read-write configuration of the database at `path` with `chain`.
fn encrypted(path: &Path, chain: &Arc<KeyChain>) -> Config {
    with_key(Config::persistent(path), chain)
}

/// A read-only configuration of the database at `path` with `chain`.
fn encrypted_read_only(path: &Path, chain: &Arc<KeyChain>) -> Config {
    with_key(Config::read_only(path), chain)
}

fn open_encrypted(path: &Path, chain: &Arc<KeyChain>) -> GrafeoDB {
    GrafeoDB::with_config(encrypted(path, chain))
        .unwrap_or_else(|error| panic!("{} opens with its key: {error}", path.display()))
}

/// The error of an open that must fail.
fn error_of(result: grafeo_common::utils::error::Result<GrafeoDB>) -> String {
    match result {
        Ok(_) => panic!("the open succeeded, expected an error"),
        Err(error) => error.to_string(),
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn sidecar_wal(path: &Path) -> PathBuf {
    with_suffix(path, ".wal")
}

/// The file header of the `.grafeo` file at `path`, read without a key.
fn file_header(path: &Path) -> FileHeaderV3 {
    let bytes = std::fs::read(path).unwrap();
    FileHeaderV3::decode(&bytes[..4096]).unwrap()
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Every file under `dir`, recursively.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// The names of the people in the default graph, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    rows(db, "MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .into_iter()
        .map(|row| row[0].clone())
        .collect()
}

fn names(people: &[&str]) -> Vec<Value> {
    people.iter().map(|name| Value::from(*name)).collect()
}

fn copy(from: &Path, to: &Path) {
    if from.is_dir() {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            copy(&entry.path(), &to.join(entry.file_name()));
        }
    } else {
        std::fs::copy(from, to).unwrap();
    }
}

/// Writes an encrypted database at `path` holding Alix and closes it.
fn write_alix(path: &Path, chain: &Arc<KeyChain>) {
    let db = open_encrypted(path, chain);
    db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        .unwrap();
    db.close().unwrap();
}

#[test]
fn an_encrypted_database_keeps_its_data_across_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let chain = key_chain(3);

    {
        let db = open_encrypted(&path, &chain);
        let alix = db
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::from("Alix")),
                    ("city", Value::from("Amsterdam")),
                ],
            )
            .unwrap();
        let gus = db
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::from("Gus")),
                    ("city", Value::from("Berlin")),
                ],
            )
            .unwrap();
        db.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::from(2019_i64))])
            .unwrap();
        db.create_property_index("city").unwrap();
        db.execute("CREATE GRAPH berlin").unwrap();
        let in_berlin = db.session();
        in_berlin.use_graph("berlin");
        in_berlin
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.close().unwrap();
    }

    assert!(
        file_header(&path).encrypted,
        "the file header of a database created with a key sets the encrypted flag"
    );
    let db = open_encrypted(&path, &chain);
    assert_eq!(people(&db), names(&["Alix", "Gus"]));
    assert_eq!(
        rows(
            &db,
            "MATCH (a)-[r:KNOWS]->(b) RETURN a.name, r.since, b.name, a.city, b.city"
        ),
        vec![vec![
            Value::from("Alix"),
            Value::from(2019_i64),
            Value::from("Gus"),
            Value::from("Amsterdam"),
            Value::from("Berlin"),
        ]]
    );
    assert!(
        db.has_property_index("city"),
        "the property index survives the reopen"
    );
    assert_eq!(
        db.find_nodes_by_property("city", &Value::from("Berlin"))
            .len(),
        1,
        "the index finds Gus"
    );
    assert_eq!(db.list_graphs(), vec!["berlin".to_string()]);
    let in_berlin = db.graph("berlin").unwrap();
    assert_eq!(
        in_berlin
            .execute("MATCH (n) RETURN n.name")
            .unwrap()
            .rows()
            .to_vec(),
        vec![vec![Value::from("Vincent")]]
    );
    db.close().unwrap();
}

// --- A writer that exits without close() --------------------------------------

const CHILD_PATH_VAR: &str = "GRAFEO_ENCRYPTION_AT_REST_PATH";
/// The seed of the child's key chain, or `none` for an unencrypted database.
const CHILD_KEY_VAR: &str = "GRAFEO_ENCRYPTION_AT_REST_KEY";

/// In a child process: opens the database at `path` (encrypted with
/// `key_chain(seed)` when `key` is given), writes Mia with the marker,
/// checkpoints (so the file holds her), writes Jules with the marker (so only
/// the sidecar WAL holds him) and exits without `close()`.
fn write_in_child_and_exit(path: &Path, key: Option<u8>) {
    let output = child_process::output(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "write_child", "--nocapture"])
            .env(CHILD_PATH_VAR, path)
            .env(
                CHILD_KEY_VAR,
                key.map_or_else(|| "none".to_string(), |seed| seed.to_string()),
            ),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "the child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child-process entry for [`write_in_child_and_exit`]; a no-op when run
/// directly.
#[test]
fn write_child() {
    let (Some(path), Ok(key)) = (
        std::env::var_os(CHILD_PATH_VAR),
        std::env::var(CHILD_KEY_VAR),
    ) else {
        return;
    };
    let path = PathBuf::from(path);
    let config = match key.as_str() {
        "none" => Config::persistent(&path),
        seed => encrypted(&path, &key_chain(seed.parse().unwrap())),
    };
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(&format!(
        "INSERT (:Person {{name: 'Mia', note: '{MARKER}'}})"
    ))
    .unwrap();
    db.wal_checkpoint().unwrap();
    db.execute(&format!(
        "INSERT (:Person {{name: 'Jules', note: '{MARKER}'}})"
    ))
    .unwrap();
    db.wal().unwrap().sync().unwrap();
    // Exit without running destructors: nothing is closed or checkpointed.
    std::process::exit(0);
}

#[test]
fn an_encrypted_wal_replays_with_the_key_after_an_exit_without_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    write_in_child_and_exit(&path, Some(19));

    let wal_files = files_under(&sidecar_wal(&path));
    assert!(
        wal_files
            .iter()
            .any(|file| std::fs::metadata(file).unwrap().len() > 0),
        "the child left its last write in the sidecar WAL: {wal_files:?}"
    );

    // The WAL key is derived per database: "grafeo-wal" with the id.
    let database_id = file_header(&path).database_id;
    let chain = key_chain(19);
    let mut recovery = WalRecovery::new(sidecar_wal(&path));
    recovery.set_encryptor(chain.encryptor_for("grafeo-wal", &database_id.to_le_bytes()));
    assert!(
        !recovery.recover().unwrap().is_empty(),
        "the WAL decrypts with the key derived from \"grafeo-wal\" and the database id"
    );

    let db = open_encrypted(&path, &chain);
    assert_eq!(
        people(&db),
        names(&["Jules", "Mia"]),
        "Mia comes from the file, Jules from the encrypted WAL"
    );
    db.close().unwrap();
    let db = open_encrypted(&path, &chain);
    assert_eq!(people(&db), names(&["Jules", "Mia"]));
}

/// Every file under `dir` with its bytes, sorted by path.
fn contents_under(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut contents: Vec<(PathBuf, Vec<u8>)> = files_under(dir)
        .into_iter()
        .map(|file| {
            let bytes = std::fs::read(&file).unwrap();
            (file, bytes)
        })
        .collect();
    contents.sort();
    contents
}

/// `Config::wal_enabled` decides only whether new commits are logged: a
/// read-write open with it off still replays the sidecar WAL a writer left
/// without `close()` (decrypted with the database's WAL key when it is
/// encrypted), and its `close()` writes those commits to the file before it
/// removes the WAL, so every later open finds them, with or without a WAL.
#[test]
fn an_open_with_the_wal_off_replays_the_sidecar_wal_and_keeps_its_commits() {
    for key in [None, Some(19)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("barcelona.grafeo");
        write_in_child_and_exit(&path, key);
        assert!(
            !files_under(&sidecar_wal(&path)).is_empty(),
            "key {key:?}: the child left Jules in the sidecar WAL"
        );
        let config = |wal_enabled: bool| {
            let mut config = match key {
                None => Config::persistent(&path),
                Some(seed) => encrypted(&path, &key_chain(seed)),
            };
            config.wal_enabled = wal_enabled;
            config
        };

        // What an open with the WAL off sees, whether its close() leaves the
        // sidecar WAL, and what a reopen (also with the WAL off) finds.
        let db = GrafeoDB::with_config(config(false))
            .unwrap_or_else(|error| panic!("key {key:?}: the open fails: {error}"));
        let seen = people(&db);
        db.close().unwrap();
        drop(db);
        let wal_left = sidecar_wal(&path).exists();
        let db = GrafeoDB::with_config(config(false)).unwrap();
        let reopened = people(&db);
        db.close().unwrap();
        drop(db);
        assert_eq!(
            (seen, wal_left, reopened),
            (names(&["Jules", "Mia"]), false, names(&["Jules", "Mia"])),
            "key {key:?}: the open with the WAL off replays Jules, and close() writes him to \
             the file before it removes the WAL"
        );

        for wal_enabled in [false, true] {
            let db = GrafeoDB::with_config(config(wal_enabled)).unwrap();
            assert_eq!(
                people(&db),
                names(&["Jules", "Mia"]),
                "key {key:?}: a reopen (WAL {wal_enabled}) finds the replayed commit in the file"
            );
            db.close().unwrap();
        }
    }
}

/// A read-only open of a 0.6 file that a writer left without `close()` (a
/// crash) replays its sidecar WAL into memory, decrypted with the database's
/// WAL key when it is encrypted: it sees every commit. It writes nothing, so
/// every file is byte for byte as the writer left it.
#[test]
fn a_read_only_open_replays_the_sidecar_wal_and_writes_nothing() {
    for key in [None, Some(3)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        write_in_child_and_exit(&path, key);
        assert!(
            !files_under(&sidecar_wal(&path)).is_empty(),
            "key {key:?}: the child left Jules in the sidecar WAL"
        );
        let before = contents_under(dir.path());

        let config = match key {
            None => Config::read_only(&path),
            Some(seed) => encrypted_read_only(&path, &key_chain(seed)),
        };
        let db = GrafeoDB::with_config(config)
            .unwrap_or_else(|error| panic!("key {key:?}: the read-only open fails: {error}"));
        assert_eq!(
            people(&db),
            names(&["Jules", "Mia"]),
            "key {key:?}: Mia comes from the file, Jules from the sidecar WAL"
        );
        db.close().unwrap();
        drop(db);

        let mut after = contents_under(dir.path());
        // A read-only open may create an empty spill directory, nothing else.
        after.retain(|(file, _)| !file.to_string_lossy().contains(".spill"));
        assert!(
            after == before,
            "key {key:?}: the read-only open changed files: {:?}",
            after.iter().map(|(file, _)| file).collect::<Vec<_>>()
        );
    }
}

/// The files under `dir` that hold the marker.
fn files_with_marker(dir: &Path) -> Vec<PathBuf> {
    files_under(dir)
        .into_iter()
        .filter(|file| contains(&std::fs::read(file).unwrap(), MARKER.as_bytes()))
        .collect()
}

#[test]
fn no_file_of_an_encrypted_database_holds_a_written_value_in_plaintext() {
    let dir = tempfile::tempdir().unwrap();

    // Without a key the file and the WAL hold the marker, so the checks below
    // mean something.
    let plain_dir = dir.path().join("plain");
    let plain = plain_dir.join("plain.grafeo");
    write_in_child_and_exit(&plain, None);
    let found = files_with_marker(&plain_dir);
    assert!(
        found.contains(&plain),
        "an unencrypted database's file holds the marker: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|file| file.starts_with(sidecar_wal(&plain))),
        "an unencrypted database's sidecar WAL holds the marker: {found:?}"
    );

    // Every file next to the encrypted database: its file, its sidecar WAL,
    // and anything else an open, a checkpoint or a spill could leave.
    let encrypted_dir = dir.path().join("encrypted");
    let path = encrypted_dir.join("encrypted.grafeo");
    write_in_child_and_exit(&path, Some(88));
    assert!(
        !files_under(&sidecar_wal(&path)).is_empty(),
        "the child left a sidecar WAL"
    );
    assert_eq!(
        files_with_marker(&encrypted_dir),
        Vec::<PathBuf>::new(),
        "no file next to the encrypted database holds the marker"
    );

    let db = open_encrypted(&path, &key_chain(88));
    assert_eq!(
        rows(&db, "MATCH (p:Person) RETURN p.note AS note ORDER BY note"),
        vec![vec![Value::from(MARKER)], vec![Value::from(MARKER)]],
        "the key reads both values back"
    );
    db.close().unwrap();
    assert_eq!(
        files_with_marker(&encrypted_dir),
        Vec::<PathBuf>::new(),
        "nor after a reopen and a clean close"
    );
}

// --- Opens that must fail ------------------------------------------------------

#[test]
fn an_encrypted_database_does_not_open_without_its_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    write_alix(&path, &key_chain(3));
    let before = std::fs::read(&path).unwrap();

    let error = error_of(GrafeoDB::open(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    let error = error_of(GrafeoDB::open_read_only(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused open changes nothing"
    );
    assert!(
        !sidecar_wal(&path).exists(),
        "a refused open creates no sidecar WAL"
    );
}

#[test]
fn an_encrypted_database_does_not_open_with_another_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    write_alix(&path, &key_chain(3));
    let before = std::fs::read(&path).unwrap();

    let other = key_chain(19);
    let error = error_of(GrafeoDB::with_config(encrypted(&path, &other)));
    assert!(error.contains("wrong key"), "{error}");
    let error = error_of(GrafeoDB::with_config(encrypted_read_only(&path, &other)));
    assert!(error.contains("wrong key"), "{error}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "an open with another key changes nothing"
    );
    assert!(!sidecar_wal(&path).exists());

    let db = open_encrypted(&path, &key_chain(3));
    assert_eq!(people(&db), names(&["Alix"]));
}

/// A writer that exited without `close()` left its last changes in the
/// sidecar WAL: an open with another key, or without one, is refused before
/// the WAL is read or written, and the right key still recovers them.
#[test]
fn another_key_is_refused_before_the_wal_of_a_crashed_writer_is_touched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crashed.grafeo");
    write_in_child_and_exit(&path, Some(88));
    let wal_contents = || {
        let mut found: Vec<(PathBuf, Vec<u8>)> = files_under(&sidecar_wal(&path))
            .into_iter()
            .map(|file| {
                let bytes = std::fs::read(&file).unwrap();
                (file, bytes)
            })
            .collect();
        found.sort();
        found
    };
    let file_before = std::fs::read(&path).unwrap();
    let wal_before = wal_contents();
    assert!(!wal_before.is_empty(), "the child left a sidecar WAL");

    let error = error_of(GrafeoDB::with_config(encrypted(&path, &key_chain(19))));
    assert!(error.contains("wrong key"), "{error}");
    let error = error_of(GrafeoDB::open(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        file_before,
        "the file is unchanged"
    );
    assert_eq!(wal_contents(), wal_before, "the sidecar WAL is unchanged");

    let db = open_encrypted(&path, &key_chain(88));
    assert_eq!(people(&db), names(&["Jules", "Mia"]));
}

/// A new encrypted database is never created next to a sidecar WAL that holds
/// files: its new database id gives another WAL key, so the records would fail
/// to decrypt, count as a torn tail, and be deleted at the next checkpoint.
/// The open fails, names the WAL, and changes nothing.
#[test]
fn a_new_encrypted_database_is_never_created_next_to_a_sidecar_wal_with_files() {
    let dir = tempfile::tempdir().unwrap();
    let other = dir.path().join("other.grafeo");
    write_in_child_and_exit(&other, Some(88));
    let path = dir.path().join("new.grafeo");
    copy(&sidecar_wal(&other), &sidecar_wal(&path));
    let wal_bytes = |path: &Path| {
        let mut files: Vec<(PathBuf, Vec<u8>)> = files_under(&sidecar_wal(path))
            .into_iter()
            .map(|file| {
                let bytes = std::fs::read(&file).unwrap();
                (file, bytes)
            })
            .collect();
        files.sort();
        files
    };
    let leftover = wal_bytes(&path);
    assert!(!leftover.is_empty(), "the copied WAL holds files");

    let error = error_of(GrafeoDB::with_config(encrypted(&path, &key_chain(88))));
    assert!(
        error.contains(&sidecar_wal(&path).display().to_string()),
        "the error names the sidecar WAL: {error}"
    );
    assert!(!path.exists(), "no database file is created");
    assert_eq!(wal_bytes(&path), leftover, "the WAL is left as it was");
}

#[test]
fn a_key_for_an_unencrypted_database_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let chain = key_chain(3);

    let error = error_of(GrafeoDB::with_config(encrypted(&path, &chain)));
    assert!(error.contains("the database is not encrypted"), "{error}");
    let error = error_of(GrafeoDB::with_config(encrypted_read_only(&path, &chain)));
    assert!(error.contains("the database is not encrypted"), "{error}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused open changes nothing"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), names(&["Gus"]));
}

#[test]
fn an_encrypted_database_opens_read_only_with_its_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let chain = key_chain(3);
    write_alix(&path, &chain);
    let before = std::fs::read(&path).unwrap();

    let db = GrafeoDB::with_config(encrypted_read_only(&path, &chain)).unwrap();
    assert_eq!(people(&db), names(&["Alix"]));
    assert!(
        db.execute("INSERT (:Person {name: 'Mia'})").is_err(),
        "a read-only database refuses writes"
    );
    db.close().unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a read-only open writes nothing"
    );
}

#[test]
fn an_in_memory_database_with_a_key_is_a_configuration_error() {
    let config = with_key(Config::in_memory(), &key_chain(3));
    let invalid = config
        .validate()
        .expect_err("encryption without a persistent path is invalid");
    assert!(invalid.to_string().contains("persistent"), "{invalid}");

    let error = error_of(GrafeoDB::with_config(config));
    assert!(error.contains("persistent"), "{error}");
}

#[test]
fn an_encrypted_database_has_no_spill_path() {
    let dir = tempfile::tempdir().unwrap();
    let plain = GrafeoDB::open(dir.path().join("plain.grafeo")).unwrap();
    assert!(
        plain.buffer_manager().config().spill_path.is_some(),
        "an unencrypted persistent database spills next to its file"
    );
    plain.close().unwrap();

    let path = dir.path().join("encrypted.grafeo");
    let db = open_encrypted(&path, &key_chain(3));
    assert_eq!(
        db.buffer_manager().config().spill_path,
        None,
        "spill files would hold the data of an encrypted database in plaintext"
    );
    db.close().unwrap();

    let spill = dir.path().join("spill");
    let config = encrypted(&path, &key_chain(3)).with_spill_path(&spill);
    let invalid = config
        .validate()
        .expect_err("a spill path with encryption is invalid");
    assert!(invalid.to_string().contains("spill"), "{invalid}");
    let error = error_of(GrafeoDB::with_config(config));
    assert!(error.contains("spill"), "{error}");
    assert!(
        !spill.exists(),
        "nothing is written to a refused spill path"
    );
}

/// Under memory pressure an unencrypted database moves its compacted base to
/// a spill file next to it; an encrypted one keeps it in memory and writes
/// nothing next to its file but the file and its sidecar WAL.
#[cfg(all(feature = "compact-store", feature = "spill", feature = "mmap"))]
#[test]
fn an_encrypted_database_spills_nothing_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let spilled = |root: &Path, key: Option<u8>| {
        let path = root.join("people.grafeo");
        let config = match key {
            Some(seed) => encrypted(&path, &key_chain(seed)),
            None => Config::persistent(&path),
        };
        let mut db = GrafeoDB::with_config(config).unwrap();
        for name in ["Alix", "Gus", "Vincent", "Mia", "Jules"] {
            db.execute(&format!(
                "INSERT (:Person {{name: '{name}', note: '{MARKER}'}})"
            ))
            .unwrap();
        }
        db.compact().unwrap();
        db.buffer_manager().spill_all();
        let found: Vec<PathBuf> = files_under(root)
            .into_iter()
            .filter(|file| *file != path && !file.starts_with(sidecar_wal(&path)))
            .collect();
        db.close().unwrap();
        found
    };

    let plain_dir = dir.path().join("plain");
    let found = spilled(&plain_dir, None);
    assert!(
        !found.is_empty(),
        "the unencrypted database wrote spill files next to its file"
    );

    let encrypted_dir = dir.path().join("encrypted");
    assert_eq!(
        spilled(&encrypted_dir, Some(3)),
        Vec::<PathBuf>::new(),
        "the encrypted database wrote nothing but its file and its sidecar WAL"
    );
    assert_eq!(files_with_marker(&encrypted_dir), Vec::<PathBuf>::new());
}

/// A new path without the `.grafeo` extension (a WAL directory before 0.6) is
/// a single file: with a key it is created encrypted, holds no written value
/// in plaintext, and opens only with its key.
#[test]
fn a_key_on_a_new_path_without_the_extension_creates_an_encrypted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin");
    let chain = key_chain(3);

    let db = open_encrypted(&path, &chain);
    db.execute(&format!(
        "INSERT (:Person {{name: 'Alix', note: '{MARKER}'}})"
    ))
    .unwrap();
    db.close().unwrap();
    drop(db);

    assert!(path.is_file(), "the database is a single file");
    assert!(file_header(&path).encrypted, "the file is encrypted");
    assert_eq!(
        files_with_marker(dir.path()),
        Vec::<PathBuf>::new(),
        "no file holds the written value in plaintext"
    );
    let error = error_of(GrafeoDB::open(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    let db = open_encrypted(&path, &chain);
    assert_eq!(people(&db), names(&["Alix"]));
}

// --- Keys are per database -----------------------------------------------------

#[test]
fn two_databases_with_one_key_chain_have_different_keys() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.grafeo");
    let second = dir.path().join("second.grafeo");
    let chain = key_chain(3);
    write_alix(&first, &chain);
    write_alix(&second, &chain);

    let first_id = file_header(&first).database_id;
    let second_id = file_header(&second).database_id;
    assert_ne!(first_id, second_id, "every database has its own id");

    let container_key = |id: u128| chain.encryptor_for("grafeo-container", &id.to_le_bytes());
    GrafeoFileManager::open_read_only(&first, Some(container_key(first_id)))
        .expect("the key derived from \"grafeo-container\" and its id reads the first file")
        .close()
        .unwrap();
    match GrafeoFileManager::open_read_only(&first, Some(container_key(second_id))) {
        Ok(_) => panic!("the second database's key decrypted the first database's directory"),
        Err(error) => assert!(error.to_string().contains("wrong key"), "{error}"),
    }
}

// --- Copies --------------------------------------------------------------------

#[test]
fn saving_an_encrypted_database_writes_an_encrypted_file_with_its_own_id() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let copy_path = dir.path().join("copy.grafeo");
    let chain = key_chain(3);

    let db = open_encrypted(&path, &chain);
    db.execute(&format!(
        "INSERT (:Person {{name: 'Alix', note: '{MARKER}'}})"
    ))
    .unwrap();
    db.save(&copy_path).unwrap();
    assert!(
        contains(&db.export_snapshot().unwrap(), MARKER.as_bytes()),
        "export_snapshot returns plaintext, as its documentation says"
    );
    db.close().unwrap();

    let header = file_header(&copy_path);
    assert!(header.encrypted, "the saved copy is encrypted");
    assert_ne!(
        header.database_id,
        file_header(&path).database_id,
        "the copy is a database of its own, with its own id and keys"
    );
    assert!(!contains(
        &std::fs::read(&copy_path).unwrap(),
        MARKER.as_bytes()
    ));
    let error = error_of(GrafeoDB::open(&copy_path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    let copied = open_encrypted(&copy_path, &chain);
    assert_eq!(people(&copied), names(&["Alix"]));
}

/// `save` to a path without the `.grafeo` extension (a WAL directory before
/// 0.6) writes a single file, encrypted as `save` to a `.grafeo` path is.
#[test]
fn saving_an_encrypted_database_to_a_path_without_the_extension_writes_an_encrypted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let target = dir.path().join("copy");
    let chain = key_chain(3);
    let db = open_encrypted(&path, &chain);
    db.execute(&format!(
        "INSERT (:Person {{name: 'Alix', note: '{MARKER}'}})"
    ))
    .unwrap();

    db.save(&target).unwrap();
    db.close().unwrap();
    assert!(target.is_file(), "the copy is a single file");
    let header = file_header(&target);
    assert!(header.encrypted, "the copy is encrypted");
    assert_ne!(
        header.database_id,
        file_header(&path).database_id,
        "the copy is a database of its own"
    );
    assert!(
        !contains(&std::fs::read(&target).unwrap(), MARKER.as_bytes()),
        "the copy holds no written value in plaintext"
    );
    let copied = open_encrypted(&target, &chain);
    assert_eq!(people(&copied), names(&["Alix"]));
}

/// `to_memory` copies the data, not the key: an in-memory database cannot be
/// encrypted, so a copy saved from it is plaintext. Save from the encrypted
/// database instead.
#[test]
fn an_in_memory_copy_of_an_encrypted_database_saves_unencrypted() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_encrypted(&dir.path().join("amsterdam.grafeo"), &key_chain(3));
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let copy = db.to_memory().unwrap();
    let saved = dir.path().join("from_memory.grafeo");
    copy.save(&saved).unwrap();
    assert!(
        !file_header(&saved).encrypted,
        "the in-memory copy has no key, as its documentation says"
    );
}

// --- Backups -------------------------------------------------------------------

/// The backups of an encrypted database: a full backup holding Alix, then an
/// incremental segment with Jules (whose note is the marker) and Mia.
struct Backups {
    dir: PathBuf,
    /// The epoch of the full backup.
    full_epoch: EpochId,
    /// The epoch right after Jules was written.
    after_jules: EpochId,
    /// The last epoch of the incremental segment.
    last_epoch: EpochId,
}

fn encrypted_backups(root: &Path, chain: &Arc<KeyChain>) -> Backups {
    let dir = root.join("backups");
    let db = open_encrypted(&root.join("source.grafeo"), chain);
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let full = db.backup_full(&dir).unwrap();
    db.execute(&format!(
        "INSERT (:Person {{name: 'Jules', note: '{MARKER}'}})"
    ))
    .unwrap();
    let after_jules = db.current_epoch();
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    let last = db.backup_incremental(&dir).unwrap();
    db.close().unwrap();
    assert!(
        files_with_marker(&dir).is_empty(),
        "the backups of an encrypted database do not hold the marker"
    );
    Backups {
        dir,
        full_epoch: full.end_epoch,
        after_jules,
        last_epoch: last.end_epoch,
    }
}

/// Without the key the incremental segments of an encrypted backup cannot be
/// replayed: the restore fails, naming the keyed restore, and writes nothing.
/// A restore that needs only the full backup copies the encrypted file.
#[test]
fn the_keyless_restore_refuses_the_segments_of_an_encrypted_backup() {
    let dir = tempfile::tempdir().unwrap();
    let chain = key_chain(3);
    let backups = encrypted_backups(dir.path(), &chain);
    let restore_dir = dir.path().join("restored");
    std::fs::create_dir(&restore_dir).unwrap();
    let output = restore_dir.join("restored.grafeo");

    let error = GrafeoDB::restore_to_epoch(&backups.dir, backups.last_epoch, &output)
        .expect_err("the segments need the key")
        .to_string();
    assert!(error.contains("restore_to_epoch_with"), "{error}");
    assert_eq!(
        files_under(&restore_dir),
        Vec::<PathBuf>::new(),
        "a refused restore writes nothing"
    );

    GrafeoDB::restore_to_epoch(&backups.dir, backups.full_epoch, &output).unwrap();
    assert!(file_header(&output).encrypted);
    let db = open_encrypted(&output, &chain);
    assert_eq!(people(&db), names(&["Alix"]));
}

fn encryption(chain: &Arc<KeyChain>) -> EncryptionConfig {
    EncryptionConfig {
        key_chain: Arc::clone(chain),
    }
}

/// The keyed restore replays the encrypted segments, and what it writes (the
/// restored file and its sidecar WAL) is encrypted with the backup's keys.
#[test]
fn the_keyed_restore_replays_the_segments_of_an_encrypted_backup() {
    let dir = tempfile::tempdir().unwrap();
    let chain = key_chain(3);
    let backups = encrypted_backups(dir.path(), &chain);
    let restore_dir = dir.path().join("restored");
    std::fs::create_dir(&restore_dir).unwrap();
    let output = restore_dir.join("restored.grafeo");

    GrafeoDB::restore_to_epoch_with(
        &backups.dir,
        backups.last_epoch,
        &output,
        &encryption(&chain),
    )
    .unwrap();
    assert!(file_header(&output).encrypted);
    assert!(
        !files_under(&sidecar_wal(&output)).is_empty(),
        "the replayed records wait in the restored sidecar WAL"
    );
    assert_eq!(
        files_with_marker(&restore_dir),
        Vec::<PathBuf>::new(),
        "nothing the restore wrote holds the marker"
    );
    let error = error_of(GrafeoDB::open(&output));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );

    let db = open_encrypted(&output, &chain);
    assert_eq!(people(&db), names(&["Alix", "Jules", "Mia"]));
    db.close().unwrap();
    let db = open_encrypted(&output, &chain);
    assert_eq!(people(&db), names(&["Alix", "Jules", "Mia"]));
}

#[test]
fn the_keyed_restore_stops_at_an_epoch_inside_the_segments() {
    let dir = tempfile::tempdir().unwrap();
    let chain = key_chain(3);
    let backups = encrypted_backups(dir.path(), &chain);
    let output = dir.path().join("restored.grafeo");

    GrafeoDB::restore_to_epoch_with(
        &backups.dir,
        backups.after_jules,
        &output,
        &encryption(&chain),
    )
    .unwrap();
    let db = open_encrypted(&output, &chain);
    assert_eq!(
        people(&db),
        names(&["Alix", "Jules"]),
        "Mia was written after the target epoch"
    );
}

/// The keyed restore checks the key against the full backup before it writes
/// anything, and refuses a key for a backup that is not encrypted.
#[test]
fn the_keyed_restore_refuses_another_key_and_an_unencrypted_backup() {
    let dir = tempfile::tempdir().unwrap();
    let backups = encrypted_backups(dir.path(), &key_chain(3));
    let restore_dir = dir.path().join("restored");
    std::fs::create_dir(&restore_dir).unwrap();
    let output = restore_dir.join("restored.grafeo");

    let error = GrafeoDB::restore_to_epoch_with(
        &backups.dir,
        backups.last_epoch,
        &output,
        &encryption(&key_chain(19)),
    )
    .expect_err("another key")
    .to_string();
    assert!(error.contains("wrong key"), "{error}");
    assert_eq!(files_under(&restore_dir), Vec::<PathBuf>::new());

    let plain_dir = dir.path().join("plain");
    let plain_backups = plain_dir.join("backups");
    {
        let db = GrafeoDB::open(plain_dir.join("source.grafeo")).unwrap();
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        let full = db.backup_full(&plain_backups).unwrap();
        db.close().unwrap();
        let error = GrafeoDB::restore_to_epoch_with(
            &plain_backups,
            full.end_epoch,
            &output,
            &encryption(&key_chain(3)),
        )
        .expect_err("a key for an unencrypted backup")
        .to_string();
        assert!(error.contains("not encrypted"), "{error}");
    }
    assert_eq!(files_under(&restore_dir), Vec::<PathBuf>::new());
}

// --- Databases written by 0.5.x ------------------------------------------------
//
// The released fixtures hold RDF triples (the file and the WAL directory) and
// vector and text indexes (the file), which a build without `triple-store`,
// `vector-index` and `text-index` refuses (see `rdf_file_without_triple_store`
// and `search_indexes_without_their_features`): their migrations run in
// builds with these features.

/// The 0.5.44 database whose second session is only in its sidecar WAL.
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/unflushed.grafeo")
}

/// Copies the fixture file to `to` and its sidecar WAL to `<to>.wal`.
fn copy_fixture(to: &Path) {
    copy(&fixture(), to);
    copy(&sidecar_wal(&fixture()), &sidecar_wal(to));
}

/// The people and the sorted named graphs of `db`, as queries see them.
#[cfg(feature = "triple-store")]
fn summary(db: &GrafeoDB) -> (Vec<Vec<Value>>, Vec<String>) {
    let mut graphs = db.list_graphs();
    graphs.sort();
    (
        rows(
            db,
            "MATCH (p:Person) RETURN p.name AS name, p.email, p.age ORDER BY name",
        ),
        graphs,
    )
}

#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
#[test]
fn a_05x_database_opened_with_a_key_is_migrated_into_an_encrypted_file() {
    let dir = tempfile::tempdir().unwrap();
    let expected = {
        let reference = dir.path().join("reference.grafeo");
        copy_fixture(&reference);
        let db = GrafeoDB::open_read_only(&reference).unwrap();
        let found = summary(&db);
        db.close().unwrap();
        assert!(found.0.len() > 1, "the fixture holds people: {found:?}");
        found
    };
    // The migrated database gets a directory of its own, so every file in it
    // is the migration's.
    let migrated_dir = dir.path().join("migrated");
    std::fs::create_dir(&migrated_dir).unwrap();
    let path = migrated_dir.join("db.grafeo");
    copy_fixture(&path);
    let chain = key_chain(3);

    let db = open_encrypted(&path, &chain);
    assert_eq!(summary(&db), expected, "the migration keeps the data");
    db.close().unwrap();

    assert!(
        file_header(&path).encrypted,
        "the migrated file is encrypted"
    );
    let kept = with_suffix(&path, ".pre-0.6");
    assert_eq!(
        std::fs::read(&kept).unwrap(),
        std::fs::read(fixture()).unwrap(),
        "the kept 0.5.x copy is the fixture, byte for byte (it stays unencrypted)"
    );
    for file in files_under(&sidecar_wal(&fixture())) {
        let relative = file.strip_prefix(sidecar_wal(&fixture())).unwrap();
        assert_eq!(
            std::fs::read(sidecar_wal(&kept).join(relative)).unwrap(),
            std::fs::read(&file).unwrap(),
            "the kept sidecar WAL file {} is unchanged",
            relative.display()
        );
    }

    // The fixture's values are in the kept 0.5.x files (`<path>.pre-0.6`,
    // `<path>.pre-0.6.wal/`), and in no other file next to the migrated
    // database: not in the migrated file, its sidecar WAL, or anything else
    // the migration and the opens left.
    let (kept_files, other_files): (Vec<PathBuf>, Vec<PathBuf>) = files_under(&migrated_dir)
        .into_iter()
        .partition(|file| file.to_string_lossy().starts_with(&*kept.to_string_lossy()));
    assert!(
        other_files.contains(&path),
        "the scan covers the migrated file: {other_files:?}"
    );
    let kept_bytes: Vec<u8> = kept_files
        .iter()
        .flat_map(|file| std::fs::read(file).unwrap())
        .collect();
    for value in ["alix@example.org", "gus@example.org", "mia@example.org"] {
        assert!(
            contains(&kept_bytes, value.as_bytes()),
            "the kept 0.5.x files hold {value} in plaintext"
        );
        for file in &other_files {
            assert!(
                !contains(&std::fs::read(file).unwrap(), value.as_bytes()),
                "{} holds {value} in plaintext",
                file.display()
            );
        }
    }

    let error = error_of(GrafeoDB::open(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    let db = open_encrypted(&path, &chain);
    assert_eq!(summary(&db), expected);
}

#[test]
fn a_05x_database_opened_read_only_with_a_key_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let error = error_of(GrafeoDB::with_config(encrypted_read_only(
        &path,
        &key_chain(3),
    )));
    assert!(error.contains("the database is not encrypted"), "{error}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(fixture()).unwrap(),
        "the 0.5.x file is unchanged"
    );
    assert!(
        !with_suffix(&path, ".pre-0.6").exists(),
        "a read-only open never migrates"
    );
}

/// The 0.5.44 WAL-directory database: `wal/` and the `LOCK` file 0.5.44 creates.
fn directory_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/directory")
}

/// A read-write open of a 0.5.x WAL directory with a key migrates it into an
/// encrypted single file at the same path. The directory is kept as
/// `<path>.pre-0.6/` byte for byte (it is the user's 0.5.x data, never
/// encrypted), and no other file next to the database holds its values in
/// plaintext.
#[cfg(feature = "triple-store")]
#[test]
fn a_05x_wal_directory_opened_with_a_key_is_migrated_into_an_encrypted_file() {
    let dir = tempfile::tempdir().unwrap();
    let expected = {
        let reference = dir.path().join("reference");
        copy(&directory_fixture(), &reference);
        let db = GrafeoDB::open_read_only(&reference).unwrap();
        let found = summary(&db);
        db.close().unwrap();
        assert!(found.0.len() > 1, "the fixture holds people: {found:?}");
        found
    };
    // The migrated database gets a directory of its own, so every file in it
    // is the migration's.
    let migrated_dir = dir.path().join("migrated");
    std::fs::create_dir(&migrated_dir).unwrap();
    let path = migrated_dir.join("db");
    copy(&directory_fixture(), &path);
    let chain = key_chain(3);

    let db = open_encrypted(&path, &chain);
    assert_eq!(summary(&db), expected, "the migration keeps the data");
    db.close().unwrap();
    drop(db);

    assert!(path.is_file(), "the migrated database is a single file");
    assert!(
        file_header(&path).encrypted,
        "the migrated file is encrypted"
    );
    let kept = with_suffix(&path, ".pre-0.6");
    let relative = |root: &Path| {
        let mut found: Vec<(PathBuf, Vec<u8>)> = files_under(root)
            .into_iter()
            .map(|file| {
                let bytes = std::fs::read(&file).unwrap();
                (file.strip_prefix(root).unwrap().to_path_buf(), bytes)
            })
            .collect();
        found.sort();
        found
    };
    assert!(
        relative(&kept) == relative(&directory_fixture()),
        "the kept 0.5.x directory is the fixture, byte for byte (it stays unencrypted)"
    );

    let (kept_files, other_files): (Vec<PathBuf>, Vec<PathBuf>) = files_under(&migrated_dir)
        .into_iter()
        .partition(|file| file.starts_with(&kept));
    assert!(
        other_files.contains(&path),
        "the scan covers the migrated file: {other_files:?}"
    );
    let kept_bytes: Vec<u8> = kept_files
        .iter()
        .flat_map(|file| std::fs::read(file).unwrap())
        .collect();
    for value in ["alix@example.org", "gus@example.org", "mia@example.org"] {
        assert!(
            contains(&kept_bytes, value.as_bytes()),
            "the kept 0.5.x directory holds {value} in plaintext"
        );
        for file in &other_files {
            assert!(
                !contains(&std::fs::read(file).unwrap(), value.as_bytes()),
                "{} holds {value} in plaintext",
                file.display()
            );
        }
    }

    let error = error_of(GrafeoDB::open(&path));
    assert!(
        error.contains("the database is encrypted and needs its key"),
        "{error}"
    );
    let db = open_encrypted(&path, &chain);
    assert_eq!(summary(&db), expected);
}

/// A read-only open of a 0.5.x WAL directory with a key fails as for a 0.5.x
/// file: the directory is not encrypted, and only a read-write open migrates
/// it into an encrypted file. Nothing changes.
#[test]
fn a_05x_wal_directory_opened_read_only_with_a_key_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    copy(&directory_fixture(), &path);

    let error = error_of(GrafeoDB::with_config(encrypted_read_only(
        &path,
        &key_chain(3),
    )));
    assert!(error.contains("the database is not encrypted"), "{error}");
    assert!(path.join("wal").is_dir(), "the directory is unchanged");
    assert!(
        !with_suffix(&path, ".pre-0.6").exists(),
        "a read-only open never migrates"
    );
}
