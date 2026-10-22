//! `memory_usage()` counts what property values hold on the heap.
//!
//! A property column held its values' slots in the count, but not what a
//! vector, a string, a list or a map holds besides: 96 MB of 384-dimension
//! embeddings were reported as about 4.6 MB. Each test here writes one kind
//! of value to 2,000 nodes and checks that the node property bytes grow by
//! at least what the values hold and at most that plus two slots per row
//! and 64 KiB, in memory and in a reopened file, and that the parts of the
//! breakdown add up to its totals.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test memory_usage_values
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{Decimal, PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::memory_usage::MemoryUsage;

const ROWS: usize = 2_000;

/// The value of `kind` for row `row`: different on each row, so nothing is
/// shared or compressed away.
fn value(kind: &str, row: usize) -> Value {
    #[expect(clippy::cast_precision_loss, reason = "small numbers")]
    let base = row as f32;
    match kind {
        "vector" => Value::Vector((0..384).map(|i| base + i as f32).collect()),
        "string" => Value::from(format!("{row:04}-{}", "Amsterdam canals ".repeat(60))),
        "bytes" => Value::Bytes(
            (0..1_000)
                .map(|i| u8::try_from((row + i) % 251).unwrap())
                .collect(),
        ),
        "list" => Value::List(
            (0..100)
                .map(|i| Value::Int64(i64::try_from(row * 100 + i).unwrap()))
                .collect(),
        ),
        "map" => Value::Map(Arc::new(
            (0..20)
                .map(|i| {
                    (
                        PropertyKey::new(format!("key{i}")),
                        Value::from(format!("{row}-Berlin-{i:02}")),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        )),
        "decimal" => Value::Decimal(Decimal::parse(&format!("{row}.25")).unwrap()),
        other => panic!("unknown kind {other}"),
    }
}

const KINDS: [&str; 6] = ["vector", "string", "bytes", "list", "map", "decimal"];

/// Writes `ROWS` nodes with the property `p` of `kind`; the bytes the
/// values hold on the heap.
fn write(db: &GrafeoDB, kind: &str) -> usize {
    let mut payload = 0;
    for row in 0..ROWS {
        let value = value(kind, row);
        payload += value.estimated_size_bytes();
        db.create_node_with_props(&["K"], [("p", value)]).unwrap();
    }
    payload
}

/// The node property bytes grew from `before` by what the values hold, within
/// the tolerance of this file.
fn assert_counted(usage: &MemoryUsage, before: usize, payload: usize, what: &str) {
    let grown = usage.store.node_properties_bytes - before;
    let slack = 2 * ROWS * size_of::<Value>() + 64 * 1024;
    assert!(
        grown >= payload && grown <= payload + slack,
        "{what}: the property bytes grew by {grown}, for {payload} bytes of values (slack {slack})"
    );
}

/// The parts of the breakdown add up to its totals.
fn assert_sums(usage: &MemoryUsage, what: &str) {
    let store = &usage.store;
    assert_eq!(
        store.total_bytes,
        store.nodes_bytes
            + store.edges_bytes
            + store.node_properties_bytes
            + store.edge_properties_bytes,
        "{what}: the store total"
    );
    assert_eq!(
        usage.total_bytes,
        usage.store.total_bytes
            + usage.indexes.total_bytes
            + usage.mvcc.total_bytes
            + usage.caches.total_bytes
            + usage.string_pool.total_bytes
            + usage.buffer_manager.allocated_bytes
            + usage.rdf.total_bytes
            + usage.cdc.total_bytes,
        "{what}: the total"
    );
}

#[test]
fn memory_usage_counts_every_value_kind_in_memory() {
    for kind in KINDS {
        let db = GrafeoDB::new_in_memory();
        let before = db.memory_usage().store.node_properties_bytes;
        let payload = write(&db, kind);
        let usage = db.memory_usage();
        assert_counted(&usage, before, payload, kind);
        assert_sums(&usage, kind);
    }
}

/// After a reopen the values are read back from the file: once every value
/// was read, they are counted as in memory.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn memory_usage_counts_every_value_kind_after_a_reopen() {
    for kind in KINDS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let payload = write(&db, kind);
        db.close().unwrap();
        drop(db);

        let db = GrafeoDB::open(&path).unwrap();
        let read = db.execute("MATCH (n:K) RETURN n.p").unwrap();
        assert_eq!(read.rows().len(), ROWS, "{kind}");
        let usage = db.memory_usage();
        assert_counted(&usage, 0, payload, &format!("{kind} after a reopen"));
        assert_sums(&usage, kind);
        db.close().unwrap();
    }
}
