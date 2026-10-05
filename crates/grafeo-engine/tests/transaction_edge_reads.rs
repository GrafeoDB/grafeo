//! A transaction reads the edges it created, with their type, on every kind
//! of store: in memory, with CDC, in a `.grafeo` file, in a single file
//! without that extension and after `compact()`. Persistent databases used
//! to read the type of a new edge at the committed state, so `-[:KNOWS]->`
//! missed it and `type(r)` was null until the commit.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test transaction_edge_reads
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The rows of `query` run in `session`, as text, sorted.
fn rows(session: &grafeo_engine::Session, query: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = session
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

/// Inserts an edge in a transaction and reads it back by its type, before
/// and after the commit.
fn reads_its_new_edge(db: &GrafeoDB, store: &str) {
    db.execute("INSERT (:P {name: 'a'}), (:P {name: 'b'})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:P {name: 'a'}), (b:P {name: 'b'}) INSERT (a)-[:KNOWS]->(b)")
        .unwrap();
    let checks = [
        ("MATCH (a:P)-[:KNOWS]->(b) RETURN b.name", vec![vec!["b"]]),
        ("MATCH (b:P)<-[:KNOWS]-(a) RETURN a.name", vec![vec!["a"]]),
        ("MATCH ()-[r]->() RETURN type(r)", vec![vec!["KNOWS"]]),
    ];
    for (query, expected) in &checks {
        assert_eq!(
            rows(&session, query),
            *expected,
            "{store}, in the transaction: {query}"
        );
    }
    session.commit().unwrap();
    for (query, expected) in &checks {
        assert_eq!(
            rows(&db.session(), query),
            *expected,
            "{store}, committed: {query}"
        );
    }
}

#[test]
fn in_memory() {
    reads_its_new_edge(&GrafeoDB::new_in_memory(), "in memory");
}

#[cfg(feature = "cdc")]
#[test]
fn in_memory_with_cdc() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    reads_its_new_edge(&db, "in memory with CDC");
}

#[cfg(feature = "grafeo-file")]
#[test]
fn in_a_grafeo_file() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("edges.grafeo")).unwrap();
    reads_its_new_edge(&db, "a .grafeo file");
    db.close().unwrap();
}

#[cfg(all(feature = "grafeo-file", feature = "cdc"))]
#[test]
fn in_a_grafeo_file_with_cdc() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(Config::persistent(dir.path().join("edges.grafeo")).with_cdc())
        .unwrap();
    reads_its_new_edge(&db, "a .grafeo file with CDC");
    db.close().unwrap();
}

#[cfg(feature = "grafeo-file")]
#[test]
fn in_a_file_without_the_grafeo_extension() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        Config::persistent(dir.path().join("edges"))
            .with_storage_format(grafeo_engine::config::StorageFormat::SingleFile),
    )
    .unwrap();
    reads_its_new_edge(&db, "a file without the .grafeo extension");
    db.close().unwrap();
}

#[cfg(feature = "compact-store")]
#[test]
fn after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Q {name: 'old'})").unwrap();
    db.compact().unwrap();
    reads_its_new_edge(&db, "after compact()");
}
