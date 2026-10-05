//! Integration tests for the single-file `.grafeo` database format.

#![cfg(feature = "grafeo-file")]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

#[cfg(all(feature = "wal", feature = "lpg"))]
#[path = "common/unlistable.rs"]
mod unlistable;

/// Helper: extract string values from column 0 of query result rows.
fn extract_strings(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

/// Helper: compute sidecar WAL path for a .grafeo file.
fn sidecar_wal_path(db_path: &std::path::Path) -> std::path::PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

// =========================================================================
// Basic create, open, and reopen
// =========================================================================

#[test]
fn create_new_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);

    // File should exist
    assert!(path.exists());
    assert!(path.is_file());

    db.close().unwrap();
}

#[test]
fn insert_close_reopen_persists_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("persist.grafeo");

    // Create and populate
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);
        db.close().unwrap();
    }

    // Sidecar WAL should be gone after close
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should be removed after close"
    );

    // Reopen and verify
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);

        // Verify data is queryable
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Gus"]);
        db.close().unwrap();
    }
}

#[test]
fn save_as_grafeo_file_from_in_memory() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("exported.grafeo");

    // Create in-memory DB and populate
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    assert_eq!(db.node_count(), 2);

    // Save as .grafeo file
    db.save(&path).unwrap();

    // Open the file and verify
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(extract_strings(result.rows()), vec!["Amsterdam", "Berlin"]);
    db2.close().unwrap();
}

#[test]
fn wal_checkpoint_writes_to_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    // Checkpoint should write snapshot to file
    db.wal_checkpoint().unwrap();

    // Verify the file manager has a non-empty header
    let fm = db.file_manager().expect("should have file manager");
    let header = fm.active_header();
    // v2 format uses section directory (snapshot_length = 0),
    // v1 used snapshot_length > 0. Either is valid.
    assert!(header.iteration > 0, "checkpoint should have been written");
    assert_eq!(header.node_count, 1);
    assert_eq!(header.edge_count, 0);

    db.close().unwrap();
}

#[test]
fn multiple_checkpoints_alternate_headers() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("multi.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    db.wal_checkpoint().unwrap();

    let fm = db.file_manager().unwrap();
    assert_eq!(fm.active_header().iteration, 1);

    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(fm.active_header().iteration, 2);
    assert_eq!(fm.active_header().node_count, 2);

    db.close().unwrap();

    // Reopen and verify both nodes are there
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 2);
    db2.close().unwrap();
}

/// The extension decides nothing: a new path without `.grafeo` is a single
/// file too (it was a WAL directory before 0.6), and an existing file opens as
/// what it is.
#[test]
fn auto_detect_uses_a_single_file_for_a_path_without_the_extension() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("test_legacy");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert!(
        db.file_manager().is_some(),
        "a path without the extension uses the single-file format"
    );

    let session = db.session();
    session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    db.close().unwrap();
    drop(db);

    assert!(path.is_file(), "the database is a file, not a directory");
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.node_count(), 1, "the reopened file holds Butch");
    db.close().unwrap();
}

#[test]
fn info_reports_persistence() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("info.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let info = db.info();
    assert!(info.is_persistent);
    assert!(info.path.is_some());
    db.close().unwrap();
}

// =========================================================================
// Checkpoint merging: data inserted between checkpoints is preserved
// =========================================================================

#[test]
fn checkpoint_merges_incremental_writes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("merge.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    // Batch 1: insert 3 nodes, checkpoint
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 3);

    // Batch 2: insert 2 more nodes, modify one, checkpoint again
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    db.wal_checkpoint().unwrap();
    assert_eq!(db.file_manager().unwrap().active_header().node_count, 5);

    // Batch 3: delete a node, add an edge, checkpoint
    session
        .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Jules'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    db.wal_checkpoint().unwrap();

    let header = db.file_manager().unwrap().active_header();
    assert_eq!(header.node_count, 4);
    assert_eq!(header.edge_count, 1);
    assert_eq!(header.iteration, 3); // 3 checkpoints = iteration 3

    db.close().unwrap();

    // Reopen and verify final state
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 4);
    assert_eq!(db2.edge_count(), 1);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Alix", "Jules", "Mia", "Vincent"]
    );

    // Verify the property survived
    let result = session2
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(31));

    db2.close().unwrap();
}

#[test]
fn writes_after_checkpoint_survive_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("post_checkpoint.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    // Write MORE data after checkpoint, then close (without explicit checkpoint)
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session.execute("INSERT (:City {name: 'Prague'})").unwrap();
    db.close().unwrap(); // close() does its own checkpoint

    // Reopen: all 3 cities should be present
    let db2 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db2.node_count(), 3);

    let session2 = db2.session();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Amsterdam", "Berlin", "Prague"]
    );
    db2.close().unwrap();
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn empty_database_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.grafeo");

    // Create, close immediately, reopen
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.close().unwrap();
    }
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
        db.close().unwrap();
    }
}

#[test]
fn multiple_reopen_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("cycles.grafeo");

    // Cycle 1: create with data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 2: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 1);
        db.session()
            .execute("INSERT (:Person {name: 'Gus'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: add more data
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 2);
        db.session()
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 4: verify all data present
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(
            extract_strings(result.rows()),
            vec!["Alix", "Gus", "Vincent"]
        );
        db.close().unwrap();
    }
}

#[test]
fn large_property_values_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("large_props.grafeo");

    let big_string = "x".repeat(100_000);

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute_with_params(
                "INSERT (:Doc {content: $text})",
                [("text".to_string(), Value::String(big_string.clone().into()))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session.execute("MATCH (d:Doc) RETURN d.content").unwrap();
        match &result.rows()[0][0] {
            Value::String(s) => assert_eq!(s.len(), 100_000),
            other => panic!("expected String, got {other:?}"),
        }
        db.close().unwrap();
    }
}

#[test]
fn diverse_property_types_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "INSERT (:Thing { \
                    str_val: 'hello', \
                    int_val: 42, \
                    float_val: 3.14, \
                    bool_val: true, \
                    list_val: [1, 2, 3] \
                 })",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (t:Thing) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.list_val",
            )
            .unwrap();
        let row = &result.rows()[0];
        assert_eq!(row[0], Value::String("hello".into()));
        assert_eq!(row[1], Value::Int64(42));
        assert!(matches!(row[2], Value::Float64(_)));
        assert_eq!(row[3], Value::Bool(true));
        assert!(matches!(row[4], Value::List(_)));
        db.close().unwrap();
    }
}

#[test]
fn open_nonexistent_creates_new() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("new.grafeo");

    assert!(!path.exists());
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert!(path.exists());
    assert_eq!(db.node_count(), 0);
    db.close().unwrap();
}

#[test]
fn file_grows_and_shrinks_with_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("size.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    let initial_size = fm.file_size().unwrap();

    // Add substantial data
    let session = db.session();
    for i in 0..100 {
        session
            .execute(&format!(
                "INSERT (:Node {{idx: {i}, data: '{}'}})",
                "x".repeat(1000)
            ))
            .unwrap();
    }
    db.wal_checkpoint().unwrap();
    let large_size = fm.file_size().unwrap();
    assert!(large_size > initial_size, "file should grow with data");

    // Delete most data. A checkpoint writes its image next to the active
    // one, which it must not overwrite: the first after the delete goes into
    // the free pages around the larger image, the second reuses the larger
    // image's pages and cuts the file after its own.
    session
        .execute("MATCH (n:Node) WHERE n.idx > 5 DELETE n")
        .unwrap();
    db.wal_checkpoint().unwrap();
    db.wal_checkpoint().unwrap();
    let small_size = fm.file_size().unwrap();
    assert!(
        small_size < large_size,
        "file should shrink after deleting data: {small_size} >= {large_size}"
    );

    db.close().unwrap();
}

#[test]
fn sidecar_wal_exists_during_operation() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_lifecycle.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Sidecar WAL should exist while DB is open
    let wal = sidecar_wal_path(&path);
    assert!(wal.exists(), "sidecar WAL should exist during operation");
    assert!(wal.is_dir(), "sidecar WAL should be a directory");

    db.close().unwrap();

    // After close, sidecar should be cleaned up
    assert!(!wal.exists(), "sidecar WAL should be removed after close");
}

#[test]
fn checkpoint_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("idempotent.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Multiple checkpoints without intervening writes should be safe
    db.wal_checkpoint().unwrap();
    let iter1 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter2 = db.file_manager().unwrap().active_header().iteration;

    db.wal_checkpoint().unwrap();
    let iter3 = db.file_manager().unwrap().active_header().iteration;

    // Each checkpoint increments the iteration counter
    assert_eq!(iter2, iter1 + 1);
    assert_eq!(iter3, iter2 + 1);

    // Data is still consistent
    assert_eq!(db.node_count(), 1);

    db.close().unwrap();
}

#[test]
fn concurrent_sessions_before_close() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sessions.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

        // Multiple sessions writing data
        let s1 = db.session();
        let s2 = db.session();

        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s2.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s1.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(db.node_count(), 3);
        db.close().unwrap();
    }
}

#[test]
fn named_graphs_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_graphs.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("USE GRAPH DEFAULT").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        // Default graph should have Gus
        let session = db.session();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Gus"]);

        // Social graph should have Alix
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
fn edges_with_properties_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_props.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS {since: 2020, strength: 0.95}]->(b)",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        let result = session
            .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(2020));
        assert!(matches!(result.rows()[0][1], Value::Float64(_)));
        db.close().unwrap();
    }
}

// =========================================================================
// Corruption and validation
// =========================================================================

#[test]
fn corrupt_snapshot_detected_on_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("corrupt.grafeo");

    // Write valid data, and note where the active image's directory is
    let directory_offset = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        db.file_manager().unwrap().active_header().root.offset
    };

    // Corrupt the directory block the active database header points at.
    // Its bytes are integrity-checked at open and the corruption must be
    // surfaced rather than masked.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(directory_offset)).unwrap();
        file.write_all(b"CORRUPTED DATA HERE!!!").unwrap();
    }

    let result = GrafeoDB::with_config(Config::persistent(&path));
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    let signals_corruption = err_msg.contains("checksum")
        || err_msg.contains("section directory")
        || err_msg.contains("failed to parse");
    assert!(
        signals_corruption,
        "expected a corruption-related error, got: {err_msg}"
    );
}

#[test]
fn validate_reports_clean_state() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("valid.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    let validation = db.validate();
    assert!(validation.errors.is_empty(), "should have no errors");
    db.close().unwrap();
}

// =========================================================================
// WAL status and detailed stats
// =========================================================================

/// The total size of the files under `dir`, recursively (0 when it is missing).
#[cfg(all(feature = "wal", feature = "lpg"))]
fn size_under(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
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

/// A sidecar WAL that exists but cannot be listed fails the open, naming it,
/// instead of opening the file without the commits only the WAL holds. Listed
/// again, the same open replays them.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn a_sidecar_wal_that_cannot_be_listed_fails_the_open() {
    use grafeo_common::types::{NodeId, TransactionId};
    use grafeo_storage::wal::{WalManager, WalRecord};

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("berlin.grafeo");
    GrafeoDB::with_config(Config::persistent(&path))
        .unwrap()
        .close()
        .unwrap();
    let sidecar = sidecar_wal_path(&path);
    {
        // A commit that only the sidecar WAL holds, as a process that exited
        // without `close()` leaves it.
        let wal = WalManager::open(&sidecar).unwrap();
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(3),
            labels: vec!["Person".to_string()],
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(19),
        })
        .unwrap();
        wal.sync().unwrap();
    }
    let file = std::fs::read(&path).unwrap();

    let opened = {
        let Some(_denied) = unlistable::Unlistable::deny(&sidecar) else {
            return;
        };
        GrafeoDB::with_config(Config::persistent(&path)).map(|db| db.node_count())
    };
    let error = match opened {
        Ok(nodes) => panic!("the open replayed a sidecar WAL it cannot list ({nodes} nodes)"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains(&sidecar.display().to_string()),
        "the error names the sidecar WAL: {error}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        file,
        "the refused open changed nothing"
    );

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        db.node_count(),
        1,
        "listed again, the WAL replays its commit"
    );
    db.close().unwrap();
}

/// The WAL of a database file is its sidecar `<path>.wal/`, and its disk usage
/// is the file plus that sidecar WAL.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn wal_status_and_disk_usage_count_the_file_and_its_sidecar_wal() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("amsterdam.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        .unwrap();
    assert_eq!(
        db.wal_status().path,
        Some(sidecar_wal_path(&path)),
        "the WAL is the sidecar of the database file"
    );
    let disk = db
        .detailed_stats()
        .disk_bytes
        .expect("a persistent database reports its disk usage");
    let file = std::fs::metadata(&path).unwrap().len();
    let wal = size_under(&sidecar_wal_path(&path));
    assert!(wal > 0, "the commit is in the sidecar WAL");
    assert_eq!(
        u64::try_from(disk).unwrap(),
        file + wal,
        "the disk usage is the file ({file} bytes) plus the sidecar WAL ({wal} bytes)"
    );
    db.close().unwrap();
}

#[test]
fn wal_status_reflects_single_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_status.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let status = db.wal_status();
    assert!(status.enabled);

    db.close().unwrap();
}

#[test]
fn detailed_stats_with_grafeo_file() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("stats.grafeo");

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();

    for i in 0..10 {
        session
            .execute(&format!("INSERT (:Node {{idx: {i}}})"))
            .unwrap();
    }

    let stats = db.detailed_stats();
    assert_eq!(stats.node_count, 10);
    assert_eq!(stats.edge_count, 0);

    db.close().unwrap();
}

// =========================================================================
// File locking
// =========================================================================

#[test]
fn second_open_of_same_file_is_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("locked.grafeo");

    let db1 = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db1.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Second open should fail because the file is locked
    let result = GrafeoDB::open(&path);
    assert!(result.is_err(), "second open should fail due to file lock");

    db1.close().unwrap();

    // After close, open should succeed
    let db2 = GrafeoDB::open(&path).unwrap();
    assert_eq!(db2.node_count(), 1);
    db2.close().unwrap();
}

#[test]
fn lock_released_on_drop() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_lock.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session().execute("INSERT (:X {v: 1})").unwrap();
        // Drop without explicit close: lock should still be released
    }

    // Should be able to open after drop
    let db2 = GrafeoDB::open(&path).unwrap();
    // Data may or may not persist (no explicit close/checkpoint), but open should succeed
    db2.close().unwrap();
}

// =========================================================================
// Schema (DDL) persistence
// =========================================================================

#[test]
fn node_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema.grafeo");

    // Create DB and define node types
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
            .unwrap();
        session
            .execute("CREATE NODE TYPE Company (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        db.close().unwrap();
    }

    // Reopen and verify types survived
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify data
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);

        // Verify node type definitions survived via SHOW NODE TYPES
        let result = session.execute("SHOW NODE TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"Person".to_string()),
            "Person type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"Company".to_string()),
            "Company type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn edge_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("edge_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE WORKS_AT (role STRING)")
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW EDGE TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"KNOWS".to_string()),
            "KNOWS type missing: {type_names:?}"
        );
        assert!(
            type_names.contains(&"WORKS_AT".to_string()),
            "WORKS_AT type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn graph_type_definitions_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("graph_types.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        session
            .execute("CREATE EDGE TYPE KNOWS (since INT64)")
            .unwrap();
        session
            .execute(
                "CREATE GRAPH TYPE SocialGraph (\
                 NODE TYPE Person (name STRING),\
                 EDGE TYPE KNOWS (since INT64)\
                 )",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW GRAPH TYPES").unwrap();
        let type_names = extract_strings(result.rows());
        assert!(
            type_names.contains(&"SocialGraph".to_string()),
            "SocialGraph type missing: {type_names:?}"
        );

        db.close().unwrap();
    }
}

#[test]
fn schema_survives_export_import_roundtrip() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    // Export and import
    let snapshot = db.export_snapshot().unwrap();
    let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session2 = db2.session();

    // Verify data
    let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(extract_strings(result.rows()), vec!["Alix"]);

    // Verify schema
    let result = session2.execute("SHOW NODE TYPES").unwrap();
    let type_names = extract_strings(result.rows());
    assert!(
        type_names.contains(&"Person".to_string()),
        "Person type missing after import: {type_names:?}"
    );
}

#[test]
#[cfg(feature = "algos")]
fn stored_procedures_persist() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("procedures.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute(
                "CREATE PROCEDURE get_people() RETURNS (name STRING) \
                 AS { MATCH (p:Person) RETURN p.name AS name }",
            )
            .unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        // Insert data so the procedure has something to return
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        // CALL the procedure to verify it survived the roundtrip
        let result = session.execute("CALL get_people()").unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

#[test]
fn schema_with_data_across_multiple_cycles() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("schema_cycles.grafeo");

    // Cycle 1: Create type + insert
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    // Cycle 2: Add another type + more data
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Verify first type survived
        let result = session.execute("SHOW NODE TYPES").unwrap();
        assert!(extract_strings(result.rows()).contains(&"Person".to_string()));

        session
            .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
            .unwrap();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        db.close().unwrap();
    }

    // Cycle 3: Verify everything
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("SHOW NODE TYPES").unwrap();
        let types = extract_strings(result.rows());
        assert!(types.contains(&"Person".to_string()), "Person missing");
        assert!(types.contains(&"City".to_string()), "City missing");

        assert_eq!(db.node_count(), 2);
        db.close().unwrap();
    }
}

// =========================================================================
// WAL-disabled single-file mode (issue #185)
// =========================================================================

/// Verifies the bug fixed in #185: file manager was previously gated behind
/// `wal_enabled`, so opening a single file with WAL disabled produced no output.
/// With the fix, checkpoint-on-close persists the snapshot correctly.
#[test]
fn wal_disabled_single_file_persists_on_close() {
    use grafeo_engine::config::StorageFormat;

    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("no_wal.grafeo");

    {
        let config = Config {
            wal_enabled: false,
            ..Config::persistent(&path).with_storage_format(StorageFormat::Auto)
        };
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        assert_eq!(db.node_count(), 2);

        // No sidecar WAL should exist: WAL is disabled
        assert!(
            !sidecar_wal_path(&path).exists(),
            "no sidecar WAL should be created when wal_enabled: false"
        );

        db.close().unwrap();
    }

    // Sidecar WAL should not exist (was never created)
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should not exist after close with wal_enabled: false"
    );

    // File should exist and contain the checkpointed data
    assert!(path.exists() && path.is_file());

    {
        let config = Config {
            wal_enabled: false,
            ..Config::persistent(&path).with_storage_format(StorageFormat::Auto)
        };
        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(
            db.node_count(),
            2,
            "data must survive close-reopen with WAL disabled"
        );

        let session = db.session();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);
        db.close().unwrap();
    }
}

// =========================================================================
// Drop: implicit close persists data and cleans up sidecar WAL
// =========================================================================

/// Dropping a GrafeoDB without explicitly calling close() still persists all
/// data and cleans up the sidecar WAL, because GrafeoDB::Drop calls close().
/// This verifies that implicit close (via drop) is as safe as explicit close.
#[test]
fn drop_persists_data_and_removes_sidecar_wal() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("drop_implicit_close.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        // Drop triggers close(), which checkpoints and removes sidecar WAL
        drop(db);
    }

    // Sidecar WAL must be cleaned up (drop triggers close())
    assert!(
        !sidecar_wal_path(&path).exists(),
        "sidecar WAL should be removed after implicit close via drop"
    );

    // File must contain the checkpointed data
    assert!(path.exists() && path.is_file());

    // Reopen: both nodes must be present
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(
            db.node_count(),
            2,
            "both nodes must survive implicit close via drop"
        );
        let result = db
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Jules", "Vincent"]);
        db.close().unwrap();
    }
}

// =========================================================================
// Concurrent read-only access
// =========================================================================

/// Two open_read_only handles on the same file can coexist (shared locks).
#[test]
fn two_concurrent_read_only_opens() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("shared_ro.grafeo");

    // Writer: create, populate, close (releases exclusive lock)
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        db.close().unwrap();
    }

    // Both read-only handles must open and read successfully
    let ro1 = GrafeoDB::open_read_only(&path).unwrap();
    let ro2 = GrafeoDB::open_read_only(&path).unwrap();

    assert_eq!(ro1.node_count(), 1);
    assert_eq!(ro2.node_count(), 1);

    let r1 = ro1
        .session()
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap();
    let r2 = ro2
        .session()
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap();
    assert_eq!(extract_strings(r1.rows()), vec!["Amsterdam"]);
    assert_eq!(extract_strings(r2.rows()), vec!["Amsterdam"]);

    ro1.close().unwrap();
    ro2.close().unwrap();
}

/// open_read_only must fail while a writer holds the exclusive lock.
#[test]
fn read_only_blocked_while_writer_holds_lock() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("writer_lock.grafeo");

    // Create the file first (writer creates it)
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Mia'})")
            .unwrap();
        db.close().unwrap();
    }

    // Writer holds exclusive lock
    let writer = GrafeoDB::with_config(Config::persistent(&path)).unwrap();

    // Read-only open must fail (cannot acquire shared lock while exclusive held)
    let result = GrafeoDB::open_read_only(&path);
    assert!(
        result.is_err(),
        "open_read_only must fail while writer holds exclusive lock"
    );

    writer.close().unwrap();

    // After writer releases, read-only must succeed
    let ro = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(ro.node_count(), 1);
    ro.close().unwrap();
}

// =========================================================================
// WAL and checkpoint interaction
// =========================================================================

/// Data written to the WAL after the last checkpoint is recovered on reopen.
/// This covers the case where the process exits between a checkpoint and
/// the subsequent close (e.g. a crash or ungraceful shutdown).
#[test]
fn wal_data_after_checkpoint_survives_drop() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_after_checkpoint.grafeo");

    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();

        // First batch: checkpointed to snapshot
        session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
        db.wal_checkpoint().unwrap();

        let fm = db.file_manager().unwrap();
        assert_eq!(
            fm.active_header().node_count,
            1,
            "checkpoint must capture first node"
        );

        // Second batch: in WAL only, not yet in snapshot
        session
            .execute("INSERT (:Person {name: 'Shosanna'})")
            .unwrap();

        // Drop without close: second node is only in sidecar WAL
        drop(db);
    }

    // Reopen: both nodes must be present (first from snapshot, second from WAL)
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 2);
        let result = db
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Butch", "Shosanna"]);
        db.close().unwrap();
    }
}

// ── Read-only checkpoint guard tests ─────────────────────────────

/// Regression: wal_checkpoint() on a read-only database should be a no-op,
/// not an error. Read-only databases have no WAL and the on-disk file is
/// already a valid snapshot.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn wal_checkpoint_on_read_only_is_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ro_checkpoint.grafeo");

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&path).unwrap();
    // Should succeed (no-op), not fail with "read-only mode" error.
    db.wal_checkpoint()
        .expect("wal_checkpoint on read-only should be a no-op");
    assert_eq!(db.node_count(), 1);
}

/// Regression: save() from a read-only database should succeed.
/// save() creates a NEW file at the target path, reading from self.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn save_from_read_only_database() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("ro_save_src.grafeo");
    let dest_path = dir.path().join("ro_save_dest.grafeo");

    {
        let db = GrafeoDB::open(&src_path).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&src_path).unwrap();
    db.save(&dest_path)
        .expect("save from read-only should succeed");

    let restored = GrafeoDB::open(&dest_path).unwrap();
    assert_eq!(restored.node_count(), 2);
    restored.close().unwrap();
}

/// Every file and directory under `root` (spill directories left out), with
/// the bytes of each file, sorted.
#[cfg(all(feature = "wal", feature = "lpg"))]
fn contents_under(root: &std::path::Path) -> Vec<(std::path::PathBuf, Option<Vec<u8>>)> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.to_string_lossy().contains(".spill") {
                continue;
            }
            if path.is_dir() {
                found.push((path.clone(), None));
                pending.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                found.push((path, Some(bytes)));
            }
        }
    }
    found.sort();
    found
}

/// `save()` never writes into or over something that exists: a database
/// file (with or without an extension), an empty directory, or a 0.5.x WAL
/// directory (where 0.5.x saved into the database already there). Each fails
/// with an error naming the path and leaves it as it was.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn save_fails_at_an_existing_path_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let source = GrafeoDB::new_in_memory();
    source.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let file = dir.path().join("amsterdam.grafeo");
    let other = GrafeoDB::new_in_memory();
    other.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    other.save(&file).unwrap();
    let without_extension = dir.path().join("berlin");
    other.save(&without_extension).unwrap();
    let empty = dir.path().join("paris");
    std::fs::create_dir(&empty).unwrap();
    let wal_directory = dir.path().join("prague");
    std::fs::create_dir_all(wal_directory.join("wal")).unwrap();
    std::fs::write(wal_directory.join("wal").join("wal_00000000.log"), b"Mia").unwrap();
    let before = contents_under(dir.path());

    for target in [&file, &without_extension, &empty, &wal_directory] {
        let error = source
            .save(target)
            .expect_err("save() refuses an existing path");
        assert!(
            error.to_string().contains(&target.display().to_string()),
            "{}: the error names the path: {error}",
            target.display()
        );
    }
    assert!(
        contents_under(dir.path()) == before,
        "the refused saves changed nothing"
    );
    let gus = GrafeoDB::open_read_only(&file).unwrap();
    assert_eq!(gus.node_count(), 1, "the existing database is as it was");
    gus.close().unwrap();
}

/// `save("copy/")` names the file `copy`, as an open of `copy/` does: the
/// path is normalized, and the side files go next to it.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn save_to_a_path_with_a_trailing_separator_writes_the_file_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let source = GrafeoDB::new_in_memory();
    source.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let mut with_separator = dir.path().join("copy").into_os_string();
    with_separator.push(std::path::MAIN_SEPARATOR_STR);
    source
        .save(&with_separator)
        .unwrap_or_else(|error| panic!("save() to {with_separator:?}: {error}"));

    let copy = dir.path().join("copy");
    assert!(copy.is_file(), "the copy is the file {}", copy.display());
    let db = GrafeoDB::open(&with_separator).unwrap();
    assert_eq!(db.node_count(), 1, "the copy opens by the same spelling");
    db.close().unwrap();
}

// =========================================================================
// Layered overlay deletion durability (#323 follow-up)
// =========================================================================

/// Regression test: when a database has been compacted (so deletes go
/// through the LayeredStore's `deleted_from_base_*` sets rather than
/// directly modifying an LpgStore), those deletions must survive a
/// close/reopen cycle. Before the OverlayDeletions section landed, the
/// deletion sets were in-memory only and previously-deleted base nodes
/// silently reappeared on reopen.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn deleted_base_nodes_stay_deleted_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("layered_delete_persist.grafeo");

    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        // Drop the session so its lock is released before compact() takes
        // the write lock on the store.
        drop(session);

        // Compact: pushes the three nodes into the columnar base and
        // installs a LayeredStore. Subsequent deletes go through
        // `LayeredStore::delete_node` and write to
        // `deleted_from_base_nodes`, which is exactly the path we need
        // to durably persist.
        db.compact().expect("compact should succeed");

        let session = db.session();
        session
            .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
            .unwrap();
        drop(session);

        // Verify the delete is visible in the running database.
        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Vincent"]);
        drop(session);

        db.close().unwrap();
    }

    // Reopen and confirm Gus did not reappear.
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(
        extract_strings(result.rows()),
        vec!["Alix", "Vincent"],
        "previously-deleted base node must stay deleted across reopen"
    );
    drop(session);
    db.close().unwrap();
}

/// Companion test for edge deletion through the LayeredStore.
#[cfg(all(feature = "compact-store", feature = "lpg"))]
#[test]
fn deleted_base_edges_stay_deleted_across_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("layered_edge_delete_persist.grafeo");

    {
        let mut db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        drop(session);

        db.compact().expect("compact should succeed");

        let session = db.session();
        session
            .execute("MATCH (:Person)-[r:KNOWS]->(:Person) DELETE r")
            .unwrap();
        let result = session
            .execute("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(0));
        drop(session);

        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    let session = db.session();
    let result = session
        .execute("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(0),
        "previously-deleted base edge must stay deleted across reopen"
    );
    drop(session);
    db.close().unwrap();
}

// =========================================================================
// A sidecar WAL left next to a missing database file
// =========================================================================

/// Copies the files of the directory `from` into a new directory `to`.
#[cfg(feature = "wal")]
fn copy_directory(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// The name and bytes of every file in the directory `dir`, sorted.
#[cfg(feature = "wal")]
fn directory_files(dir: &std::path::Path) -> Vec<(std::ffi::OsString, Vec<u8>)> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect();
    files.sort();
    files
}

/// A new database is never created next to a sidecar WAL that holds files:
/// they belong to another database (one that was at this path, or a WAL
/// moved here), and the new one would replay its records. The open fails,
/// names the WAL, and changes nothing; an empty sidecar directory is no
/// obstacle.
#[cfg(feature = "wal")]
#[test]
fn a_new_database_is_never_created_next_to_a_sidecar_wal_with_files() {
    let dir = tempfile::TempDir::new().unwrap();
    let other = dir.path().join("other.grafeo");
    let path = dir.path().join("new.grafeo");
    let wal = sidecar_wal_path(&path);
    {
        let db = GrafeoDB::with_config(Config::persistent(&other)).unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        db.wal().unwrap().sync().unwrap();
        copy_directory(&sidecar_wal_path(&other), &wal);
        db.close().unwrap();
    }
    let leftover = directory_files(&wal);
    assert!(!leftover.is_empty(), "the copied WAL holds files");

    let error = match GrafeoDB::with_config(Config::persistent(&path)) {
        Ok(db) => panic!(
            "a database was created next to the leftover WAL, holding {} nodes",
            db.node_count()
        ),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains(&wal.display().to_string()),
        "the error names the sidecar WAL: {error}"
    );
    assert!(!path.exists(), "no database file is created");
    assert_eq!(directory_files(&wal), leftover, "the WAL is left as it was");

    // Without files, the sidecar directory is no obstacle.
    std::fs::remove_dir_all(&wal).unwrap();
    std::fs::create_dir(&wal).unwrap();
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.node_count(), 0, "a new, empty database");
    db.close().unwrap();
}

/// `open_in_memory` reads a 0.6 file as a read-only open does: no exclusive
/// lock (it works while a reader holds the file), no checkpoint and no WAL
/// removal when the copy closes. The files stay byte for byte as they were,
/// also after writes to the in-memory copy and its `close()`.
#[cfg(all(feature = "wal", feature = "lpg"))]
#[test]
fn open_in_memory_changes_nothing_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }
    let before = contents_under(dir.path());

    let reader = GrafeoDB::open_read_only(&path).unwrap();
    let copy = GrafeoDB::open_in_memory(&path)
        .unwrap_or_else(|error| panic!("open_in_memory while a reader holds the file: {error}"));
    copy.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert_eq!(copy.node_count(), 2, "the copy takes writes");
    copy.close().unwrap();
    drop(copy);
    reader.close().unwrap();
    drop(reader);

    assert!(
        contents_under(dir.path()) == before,
        "open_in_memory, writes to the copy and its close() leave every file as it was"
    );
}
