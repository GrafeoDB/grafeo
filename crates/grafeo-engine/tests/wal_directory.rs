//! WAL directories: 0.5.x databases that 0.6 loads and migrates, and never
//! creates.
//!
//! Before 0.6, a new database at a path without the `.grafeo` extension was a
//! directory holding its WAL (`<path>/wal/`). 0.6 creates a single file at
//! every new path, whatever its extension (its WAL is the sidecar
//! `<path>.wal/` while it is open), and refuses the deprecated
//! `StorageFormat::WalDirectory` for a new path. An existing 0.5.x directory
//! is loaded by replaying every file of its WAL: a read-write open migrates it
//! to a single file at the same path, a read-only open reads it in place.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full,temporal --test wal_directory
//! ```
//!
//! The named-graph epoch test needs `temporal` (CI runs every feature).

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{NodeId, TransactionId, Value};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::detect::{OnDisk, detect};
use grafeo_storage::wal::{WalConfig, WalManager, WalRecord};

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn names(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// A new path is a single file whatever its extension, with its WAL in the
/// sidecar `<path>.wal/` while it is open, and the data written through
/// queries with the default configuration is there after a reopen (#252, the
/// server's usage). Before 0.6 these paths became WAL directories.
#[test]
fn a_new_path_without_the_grafeo_extension_is_a_single_file() {
    for name in ["db", "db.db"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);

        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
            .unwrap();
        assert!(path.is_file(), "{name}: the database is a file");
        assert!(
            with_suffix(&path, ".wal").is_dir(),
            "{name}: its WAL is the sidecar <path>.wal/ while it is open"
        );
        db.close().unwrap();
        drop(db);

        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::Current,
            "{name}: the file is in the 0.6 format"
        );
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(
            names(&db),
            [Value::from("Alix"), Value::from("Gus")],
            "{name}: the data after a reopen"
        );
        db.close().unwrap();
    }
}

/// The deprecated `StorageFormat::WalDirectory` creates nothing: on a new
/// path the open fails, naming the path and the format to use instead, and
/// leaves no file or directory behind, not even the missing parent
/// directories of the path.
#[test]
#[allow(
    deprecated,
    reason = "pins what the deprecated `StorageFormat::WalDirectory` still does until 0.7.0"
)]
fn the_wal_directory_format_no_longer_creates_a_database() {
    use grafeo_engine::config::StorageFormat;

    for name in ["db", "db.grafeo", "amsterdam/berlin/db"] {
        let dir = tempfile::tempdir().unwrap();
        // Joined part by part, so the path uses the platform's separator.
        let path = name
            .split('/')
            .fold(dir.path().to_path_buf(), |path, part| path.join(part));
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let error = match GrafeoDB::with_config(config) {
            Ok(_) => panic!("{name}: a WAL directory was created"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("no longer created")
                && error.contains(&path.display().to_string())
                && error.contains("StorageFormat::Auto"),
            "{name}: the error says WAL directories are no longer created, names the path and \
             the format to use: {error}"
        );
        let left: Vec<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert!(left.is_empty(), "{name}: the refused open left {left:?}");
    }
}

/// Writes a database in the 0.5.x WAL-directory layout at `path`: `<path>/wal/`
/// holding 19 committed people (each with its `idx`) across several WAL files,
/// as a 0.5.x process left it.
fn write_rotated_directory(path: &Path) {
    let wal_dir = path.join("wal");
    // Tiny log files, so the commits span several rotated files.
    let wal = WalManager::with_config(
        &wal_dir,
        WalConfig {
            max_log_size: 100,
            ..WalConfig::default()
        },
    )
    .unwrap();
    for idx in 0..19u64 {
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(idx),
            labels: vec!["Person".to_string()],
        })
        .unwrap();
        wal.log(&WalRecord::SetNodeProperty {
            id: NodeId::new(idx),
            key: "idx".to_string(),
            value: Value::Int64(i64::try_from(idx).unwrap()),
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(idx + 1),
        })
        .unwrap();
    }
    wal.sync().unwrap();
    drop(wal);
    let logs = std::fs::read_dir(&wal_dir)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|ext| ext == "log")
        })
        .count();
    assert!(logs > 2, "the WAL rotated: {logs} log files");
}

/// The count and the sum of `idx` of the people in `db`.
fn people_and_sum(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN count(p), sum(p.idx)")
        .unwrap()
        .rows()[0]
        .clone()
}

/// A 0.5.x directory whose WAL rotated is loaded from every one of its files,
/// not only the active one: by a read-only open, by `open_in_memory`, and by
/// the read-write open that migrates it, after which the migrated file holds
/// it all.
#[test]
fn a_0_5_directory_whose_wal_rotated_loads_every_file() {
    let expected = vec![Value::Int64(19), Value::Int64((0..19).sum())];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotated");
    write_rotated_directory(&path);

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(people_and_sum(&db), expected, "read-only");
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::open_in_memory(&path).unwrap();
    assert_eq!(people_and_sum(&db), expected, "in memory");
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people_and_sum(&db), expected, "migrated");
    db.close().unwrap();
    drop(db);
    assert!(path.is_file(), "the migrated database is a single file");
    assert!(
        with_suffix(&path, ".pre-0.6").join("wal").is_dir(),
        "the 0.5.x directory is kept as <path>.pre-0.6/"
    );
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people_and_sum(&db), expected, "the migrated file reopened");
    db.close().unwrap();
}

/// Writes a database in the 0.5.43 WAL-directory layout at `path`, as 0.5.43
/// logged it: one commit in the default graph (Alix), then four commits in the
/// named graph `trips` without a switch back to the default graph (0.5.43
/// never logged one). In `trips`, node 1 is named Paris, later Amsterdam, and
/// node 3 (Berlin) comes last.
#[cfg(feature = "temporal")]
fn write_0_5_43_named_graph_directory(path: &Path) {
    let wal = WalManager::open(path.join("wal")).unwrap();
    let mut transaction = 0;
    let mut log = |records: Vec<WalRecord>| {
        transaction += 1;
        for record in records {
            wal.log(&record).unwrap();
        }
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(transaction),
        })
        .unwrap();
    };
    let city = |id: u64, name: &str| {
        vec![
            WalRecord::CreateNode {
                id: NodeId::new(id),
                labels: vec!["City".to_string()],
            },
            WalRecord::SetNodeProperty {
                id: NodeId::new(id),
                key: "name".to_string(),
                value: Value::from(name),
            },
        ]
    };
    log(vec![
        WalRecord::CreateNode {
            id: NodeId::new(0),
            labels: vec!["Person".to_string()],
        },
        WalRecord::SetNodeProperty {
            id: NodeId::new(0),
            key: "name".to_string(),
            value: Value::from("Alix"),
        },
    ]);
    let mut paris = vec![
        WalRecord::CreateNamedGraph {
            name: "trips".to_string(),
        },
        WalRecord::SwitchGraph {
            name: Some("trips".to_string()),
        },
    ];
    paris.extend(city(1, "Paris"));
    log(paris);
    log(city(2, "Prague"));
    log(vec![WalRecord::SetNodeProperty {
        id: NodeId::new(1),
        key: "name".to_string(),
        value: Value::from("Amsterdam"),
    }]);
    log(city(3, "Berlin"));
    wal.sync().unwrap();
}

/// What an open of the named-graph directory shows: the database's epoch,
/// the cities in `trips`, and the name history of node 1 there.
#[cfg(feature = "temporal")]
fn named_graph_view(
    db: &GrafeoDB,
) -> (
    grafeo_common::types::EpochId,
    Vec<Value>,
    Vec<(grafeo_common::types::EpochId, Value)>,
) {
    let cities = db
        .graph("trips")
        .unwrap()
        .execute("MATCH (c:City) RETURN c.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    db.set_current_graph(Some("trips")).unwrap();
    let history = db.get_node_property_history(NodeId::new(1), "name");
    db.set_current_graph(None).unwrap();
    (db.current_epoch(), cities, history)
}

/// A 0.5.43 directory whose last commits all went to a named graph is read at
/// one epoch, the highest any graph reached during the replay: every open (read
/// only, `open_in_memory`, the migration, a reopen of the migrated file) shows
/// the latest values with their whole history, at the same epoch, and a new
/// commit continues above it. Replay advances the epoch of the graph a commit
/// lands in, and 0.5.43 never switched back to the default graph, so a read at
/// the default graph's epoch would show a stale name and miss Berlin.
#[cfg(feature = "temporal")]
#[test]
fn a_0_5_43_directory_is_read_at_the_epoch_its_named_graphs_reached() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trips");
    write_0_5_43_named_graph_directory(&path);

    let read_only = {
        let db = GrafeoDB::open_read_only(&path).unwrap();
        let view = named_graph_view(&db);
        db.close().unwrap();
        view
    };
    let (epoch, cities, history) = &read_only;
    assert_eq!(
        cities,
        &[
            Value::from("Amsterdam"),
            Value::from("Berlin"),
            Value::from("Prague")
        ],
        "read-only: every city with its latest name"
    );
    let names: Vec<&Value> = history.iter().map(|(_, value)| value).collect();
    assert_eq!(
        names,
        [&Value::from("Paris"), &Value::from("Amsterdam")],
        "read-only: the whole name history of node 1"
    );
    assert!(
        history.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && history.iter().all(|(at, _)| at <= epoch),
        "read-only: the history is in epoch order, up to the database's epoch {epoch:?}: \
         {history:?}"
    );

    let in_memory = named_graph_view(&GrafeoDB::open_in_memory(&path).unwrap());
    assert_eq!(
        in_memory, read_only,
        "open_in_memory: the same epoch and data"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(named_graph_view(&db), read_only, "migrated: the same");
    db.close().unwrap();
    drop(db);
    assert!(path.is_file(), "the directory was migrated");

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(named_graph_view(&db), read_only, "reopened: the same");
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert!(
        db.current_epoch() > *epoch,
        "a new commit continues above epoch {epoch:?}, at {:?}",
        db.current_epoch()
    );
    db.close().unwrap();
}

/// The disk usage of a 0.5.x directory read in place is the size of the files
/// in it, counted recursively (its WAL files and the `LOCK` 0.5.44 left).
#[test]
fn a_0_5_directory_read_in_place_reports_its_size() {
    fn size_under(dir: &Path) -> u64 {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    size_under(&path)
                } else {
                    std::fs::metadata(&path).unwrap().len()
                }
            })
            .sum()
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

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/directory"),
        &path,
    );
    let expected = size_under(&path);
    assert!(expected > 0, "the fixture's WAL holds records");

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(
        db.detailed_stats()
            .disk_bytes
            .map(|bytes| u64::try_from(bytes).unwrap()),
        Some(expected),
        "the disk usage is the size of the directory"
    );
    db.close().unwrap();
}

/// Regression test for GrafeoDB/grafeo#221 (WAL deadlock on the second batch),
/// on a database at a path without an extension, which is now a single file.
///
/// Direct CRUD calls on a persistent database with WAL would deadlock on
/// the second batch of writes because `sync_all()` was called while holding
/// the `active_log` mutex, and Batch mode triggers sync when >100ms have
/// elapsed since the last sync (i.e., the first write of the second batch).
#[test]
fn second_batch_crud_does_not_deadlock() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("deadlock_test");

    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("open");

    // First batch: create nodes with properties
    for i in 0..10 {
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "name", Value::from(format!("Node{i}")))
            .unwrap();
        db.set_node_property(id, "index", Value::Int64(i)).unwrap();
    }

    // Sleep long enough to trigger Batch mode sync threshold (default 100ms)
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Second batch: this would deadlock before the fix because the first
    // write_frame triggers sync_all() while holding active_log.
    for i in 10..20 {
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "name", Value::from(format!("Node{i}")))
            .unwrap();
        db.set_node_property(id, "index", Value::Int64(i)).unwrap();
    }

    assert_eq!(db.node_count(), 20);
    db.close().expect("close");

    // Verify data survives reopen
    let db = GrafeoDB::with_config(Config::persistent(&path)).expect("reopen");
    assert_eq!(db.node_count(), 20, "all nodes should survive reopen");
    db.close().expect("close");
}
