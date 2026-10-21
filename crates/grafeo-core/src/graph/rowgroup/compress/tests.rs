//! Compression on request: memory, reads at every view, chains, thaws.

use grafeo_common::change::DataOp;
use grafeo_common::types::{NodeId, PropertyKey, Value};

use crate::graph::apply::{ChangeTarget, Writer};
use crate::graph::conformance::{Reader, Tx, committed, image, labels, next_epoch, properties};
use crate::graph::rowgroup::RowGroupStore;

/// Writes `op` at once at the next epoch.
fn immediately(store: &RowGroupStore, op: &DataOp) {
    store
        .apply(
            op,
            Writer::Immediate {
                epoch: next_epoch(store),
                before_images: false,
            },
        )
        .unwrap();
}

/// A store of 5,000 people with an age each, written at once and
/// collected.
fn ages() -> RowGroupStore {
    let store = RowGroupStore::new();
    for id in store.reserve_node_ids(5_000).unwrap() {
        immediately(
            &store,
            &DataOp::CreateNode {
                id: NodeId::new(id),
                labels: labels(&["Person"]),
                properties: properties(&[("age", Value::Int64(i64::try_from(id % 90).unwrap()))]),
            },
        );
    }
    store.collect(store.current_epoch());
    store
}

/// The age a reader sees of node `id`.
fn age_of(seen: &crate::graph::conformance::Image, id: u64) -> Option<Value> {
    seen.nodes[&id]
        .1
        .iter()
        .find(|(key, _)| key == "age")
        .map(|(_, value)| value.clone())
}

/// Compression makes the hot columns cold: the column memory goes, the
/// cold chunks hold less, and every reader reads what it read before.
#[test]
fn compression_frees_the_hot_columns_and_reads_stay_the_same() {
    let store = ages();
    let before = store.memory();
    let seen = committed(&store);
    let report = store.compress().unwrap();
    assert_eq!((report.columns, report.rows), (1, 5_000), "{report:?}");
    let after = store.memory();
    assert!(
        after.node_columns_bytes * 10 < before.node_columns_bytes,
        "the decoded values are gone: {before:?} then {after:?}"
    );
    assert!(after.node_cold_bytes > 0, "{after:?}");
    assert!(
        after.total_bytes() < before.total_bytes(),
        "{} bytes before, {} after",
        before.total_bytes(),
        after.total_bytes()
    );
    assert!(report.freed_bytes > 0, "{report:?}");
    assert_eq!(committed(&store), seen);
    assert_eq!(
        store.compress().unwrap().columns,
        0,
        "a cold column stays as it is"
    );
}

/// A column whose update chains a reader at an earlier epoch still needs
/// stays hot, and that reader keeps reading what it read; once the chains
/// are collected the column compresses. A write thaws a compressed column,
/// and the next compression takes it again.
#[test]
fn chains_keep_a_column_hot_until_they_are_collected() {
    let store = ages();
    let before = store.current_epoch();
    let seen_before = image(&store, Reader::At(before));
    let mut tx = Tx::begin(&store, 40);
    tx.set_node(NodeId::new(7), "age", Value::Int64(41))
        .unwrap();
    tx.commit().unwrap();
    let report = store.compress().unwrap();
    assert_eq!(
        (report.columns, report.kept_with_chains),
        (0, 1),
        "{report:?}"
    );
    assert_eq!(image(&store, Reader::At(before)), seen_before);
    store.collect(store.current_epoch());
    assert_eq!(store.compress().unwrap().columns, 1);
    assert_eq!(age_of(&committed(&store), 7), Some(Value::Int64(41)));

    immediately(
        &store,
        &DataOp::SetNodeProperty {
            id: NodeId::new(8),
            key: PropertyKey::new("age"),
            value: Value::Int64(19),
        },
    );
    assert!(store.memory().node_columns_bytes > 0, "the write thawed it");
    store.collect(store.current_epoch());
    assert_eq!(store.compress().unwrap().columns, 1);
    let seen = committed(&store);
    assert_eq!(age_of(&seen, 8), Some(Value::Int64(19)));
    assert_eq!(age_of(&seen, 7), Some(Value::Int64(41)));
}

/// Strings have no point view: their column stays hot.
#[test]
fn a_string_column_stays_hot() {
    let store = RowGroupStore::new();
    let id = store.reserve_node_ids(1).unwrap().start;
    immediately(
        &store,
        &DataOp::CreateNode {
            id: NodeId::new(id),
            labels: labels(&["City"]),
            properties: properties(&[("name", Value::from("Prague"))]),
        },
    );
    store.collect(store.current_epoch());
    let report = store.compress().unwrap();
    assert_eq!(
        (report.columns, report.kept_unsupported),
        (0, 1),
        "{report:?}"
    );
}
