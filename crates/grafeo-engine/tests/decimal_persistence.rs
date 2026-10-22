//! Exact decimals survive every way a database is saved and read back
//! (#610): a checkpoint and a reopen, a read-only open, a crash with WAL
//! replay, and a snapshot export and import. Each decimal keeps its value and
//! its scale (`1.50` stays `1.50`), a column of one scale and one of several,
//! and filters and sums over the reopened values are exact.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test decimal_persistence
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use std::path::Path;

use grafeo_engine::{Config, GrafeoDB};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// The rows of `query`, each value as `Debug` prints it, so a decimal's
/// scale shows.
fn printed(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect()
}

/// Two hundred items with a price of scale 2 (a column of one scale) and a
/// weight of several scales.
fn write_items(db: &GrafeoDB) {
    for i in 0..200 {
        let weight = match i % 3 {
            0 => "1.5",
            1 => "1.50",
            _ => "0.125",
        };
        db.execute(&format!(
            "INSERT (:Item {{n: {i}, price: {i}.{:02}, weight: {weight}}})",
            i % 100
        ))
        .unwrap();
    }
}

fn assert_items(db: &GrafeoDB) {
    assert_eq!(
        printed(
            db,
            "MATCH (i:Item) WHERE i.n IN [0, 1, 2, 199] RETURN i.n, i.price, i.weight ORDER BY i.n"
        ),
        [
            "[Int64(0), Decimal(0.00), Decimal(1.5)]",
            "[Int64(1), Decimal(1.01), Decimal(1.50)]",
            "[Int64(2), Decimal(2.02), Decimal(0.125)]",
            "[Int64(199), Decimal(199.99), Decimal(1.50)]",
        ]
    );
    // Exact over the reopened values: the sum of n + (n mod 100) / 100.
    assert_eq!(
        printed(db, "MATCH (i:Item) RETURN sum(i.price)"),
        ["[Decimal(19999.00)]"]
    );
    assert_eq!(
        printed(db, "MATCH (i:Item) WHERE i.price > 197.50 RETURN count(*)"),
        ["[Int64(3)]"],
        "197.97, 198.98 and 199.99"
    );
    assert_eq!(
        printed(db, "MATCH (i:Item) WHERE i.weight = 1.5 RETURN count(*)"),
        ["[Int64(134)]"]
    );
}

#[test]
fn decimals_survive_a_checkpoint_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("items.grafeo");
    let db = open(&path);
    write_items(&db);
    db.close().unwrap();
    drop(db);

    let db = open(&path);
    assert_items(&db);
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    assert_items(&db);
}

#[test]
fn decimals_survive_a_crash_and_a_wal_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "decimals_survive_a_crash_and_a_wal_replay",
        open,
        write_items,
    );
    assert_items(&db);
}

#[test]
fn decimals_survive_a_snapshot() {
    let db = GrafeoDB::new_in_memory();
    write_items(&db);
    let snapshot = db.export_snapshot().unwrap();
    let imported = GrafeoDB::import_snapshot(&snapshot).unwrap();
    assert_items(&imported);
}
