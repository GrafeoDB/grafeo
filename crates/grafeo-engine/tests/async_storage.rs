//! Integration tests for the `async-storage` feature.
//!
//! These tests verify that async WAL checkpoint and snapshot operations
//! work correctly end-to-end through the GrafeoDB API.

#![cfg(feature = "async-storage")]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Helper: extract string values from single-column result rows.
fn extract_strings(rows: &[Vec<Value>]) -> Vec<String> {
    let mut values: Vec<String> = rows
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    values.sort();
    values
}

#[tokio::test]
async fn test_async_checkpoint_persists_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("checkpoint.grafeo");

    // Create, populate, and checkpoint
    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS {since: 2020}]->(b)",
            )
            .unwrap();
        db.async_wal_checkpoint().await.unwrap();
        db.close().unwrap();
    }

    // Reopen and verify
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);

        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Gus"]);

        let result = session
            .execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(2020));
        db.close().unwrap();
    }
}

#[tokio::test]
async fn test_async_checkpoint_multiple_sessions() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("multi_session.grafeo");

    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());

        // Session 1: insert Alix
        let s1 = db.session();
        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Session 2: insert Gus
        let s2 = db.session();
        s2.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Session 3: insert Vincent
        let s3 = db.session();
        s3.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        db.async_wal_checkpoint().await.unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
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

#[tokio::test]
async fn test_async_snapshot_then_continue_writing() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("snapshot_continue.grafeo");

    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let session = db.session();

        // Insert pre-snapshot data
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.async_wal_checkpoint().await.unwrap();

        // Insert post-snapshot data
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 2);

        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Gus"]);
        db.close().unwrap();
    }
}

#[tokio::test]
async fn test_async_checkpoint_empty_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("empty.grafeo");

    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    // Checkpoint an empty database should succeed
    db.async_wal_checkpoint().await.unwrap();
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
    db.close().unwrap();
}

#[tokio::test]
async fn test_async_checkpoint_after_deletes() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("deletes.grafeo");

    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Delete Gus
        session
            .execute("MATCH (p:Person {name: 'Gus'}) DELETE p")
            .unwrap();

        db.async_wal_checkpoint().await.unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 2);

        let session = db.session();
        let result = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix", "Vincent"]);
        db.close().unwrap();
    }
}

#[tokio::test]
async fn test_async_checkpoint_with_properties_and_labels() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("props_labels.grafeo");

    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let session = db.session();

        // Node with multiple labels and various property types
        session
            .execute("INSERT (:Person:Employee {name: 'Alix', age: 30, score: 9.5, active: true})")
            .unwrap();

        // Node with list property
        session
            .execute("INSERT (:Person {name: 'Gus', tags: ['engineer', 'lead']})")
            .unwrap();

        // Node with map property
        session
            .execute("INSERT (:Person {name: 'Vincent', metadata: {role: 'admin', level: 5}})")
            .unwrap();

        db.async_wal_checkpoint().await.unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.node_count(), 3);

        let session = db.session();

        // Verify Alix's properties
        let result = session
            .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age, p.score, p.active")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(30));
        assert_eq!(result.rows()[0][1], Value::Float64(9.5));
        assert_eq!(result.rows()[0][2], Value::Bool(true));

        // Verify Alix has both labels
        let result = session
            .execute("MATCH (p:Employee {name: 'Alix'}) RETURN p.name")
            .unwrap();
        assert_eq!(result.rows().len(), 1);

        // Verify Gus's list property
        let result = session
            .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.tags")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        match &result.rows()[0][0] {
            Value::List(list) => {
                assert_eq!(list.len(), 2);
            }
            other => panic!("Expected List, got {other:?}"),
        }

        // Verify Vincent's map property
        let result = session
            .execute("MATCH (p:Person {name: 'Vincent'}) RETURN p.metadata")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        match &result.rows()[0][0] {
            Value::Map(map) => {
                assert!(map.len() >= 2);
            }
            other => panic!("Expected Map, got {other:?}"),
        }

        db.close().unwrap();
    }
}

#[tokio::test]
async fn test_async_checkpoint_named_graphs() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("named_graphs.grafeo");

    {
        let db = Arc::new(GrafeoDB::open(&path).unwrap());
        let session = db.session();

        // Create named graphs and insert data
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("USE GRAPH DEFAULT").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        db.async_wal_checkpoint().await.unwrap();
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        // Default graph should have Gus
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Gus"]);

        // Social graph should have Alix
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        assert_eq!(extract_strings(result.rows()), vec!["Alix"]);

        db.close().unwrap();
    }
}

#[tokio::test]
async fn test_async_operations_do_not_block_each_other() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("non_blocking.grafeo");

    let db = Arc::new(GrafeoDB::open(&path).unwrap());

    // The sleep should complete without being blocked by the checkpoint
    let (checkpoint_result, ()) = tokio::join!(
        db.async_wal_checkpoint(),
        tokio::time::sleep(std::time::Duration::from_millis(1)),
    );

    checkpoint_result.unwrap();
    db.close().unwrap();
}

#[tokio::test]
async fn test_async_checkpoint_idempotent() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("idempotent.grafeo");

    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Call async_wal_checkpoint twice in a row: both should succeed
    db.async_wal_checkpoint().await.unwrap();
    db.async_wal_checkpoint().await.unwrap();

    assert_eq!(db.node_count(), 1);
    db.close().unwrap();
}
