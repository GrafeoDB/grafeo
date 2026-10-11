//! What the store costs on the write and read paths, per store, through the
//! interfaces the engine uses: `ChangeTarget::apply` (and `stamp`) for
//! writes, the read traits for reads.
//!
//! The direct-call question (an immediate call costs about 470 ns, a private
//! transaction about 1,070 ns, of which the store's `apply` about 210 ns)
//! is answered here per store: `LpgStore` now, the row-group store from
//! workstream H1a on, with the same cases. Each write case starts from the
//! same graph: 10,000 people with a name and an age, each knowing two others.
//!
//! ```bash
//! cargo bench -p grafeo-core --bench store_apply
//! ```
//!
//! Uses `codspeed-criterion-compat` in place of `criterion`, as the other
//! benches do.
#![allow(
    missing_docs,
    reason = "criterion_group! from codspeed-criterion-compat generates functions without doc comments"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::change::{ChangeSet, DataModel, DataOp, GraphRef};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_core::graph::apply::{Applied, ChangeTarget, Writer};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphStoreMut};

/// The people in the graph each case starts from.
const PEOPLE: u64 = 10_000;

/// A store the cases run on.
trait Store: GraphStoreMut + ChangeTarget {}

impl<T: GraphStoreMut + ChangeTarget> Store for T {}

/// The epoch after the store's current one.
fn next_epoch(store: &dyn Store) -> EpochId {
    EpochId::new(store.current_epoch().as_u64() + 1)
}

/// Applies `op` as an immediate write at the next epoch.
fn immediate(store: &dyn Store, op: &DataOp, before_images: bool) -> Applied {
    let writer = Writer::Immediate {
        epoch: next_epoch(store),
        before_images,
    };
    store.apply(op, writer).expect("the write applies")
}

/// 10,000 people, each knowing the next two, replayed at the first epoch;
/// returns the ids of the people.
fn people(store: &dyn Store) -> Vec<NodeId> {
    let writer = Writer::Replay {
        epoch: next_epoch(store),
    };
    let first = store.reserve_node_ids(PEOPLE).expect("node ids").start;
    let ids: Vec<NodeId> = (first..first + PEOPLE).map(NodeId::new).collect();
    for (row, id) in ids.iter().enumerate() {
        let op = DataOp::CreateNode {
            id: *id,
            labels: [ArcStr::from("Person")].into_iter().collect(),
            properties: vec![
                (PropertyKey::new("name"), Value::from(format!("Alix {row}"))),
                (PropertyKey::new("age"), Value::Int64(19)),
            ],
        };
        store.apply(&op, writer).expect("a person");
    }
    let edges = store.reserve_edge_ids(2 * PEOPLE).expect("edge ids").start;
    for (row, id) in ids.iter().enumerate() {
        for hop in 1..=2_usize {
            let op = DataOp::CreateEdge {
                id: EdgeId::new(edges + 2 * row as u64 + hop as u64 - 1),
                src: *id,
                dst: ids[(row + hop) % ids.len()],
                edge_type: ArcStr::from("KNOWS"),
                properties: Vec::new(),
            };
            store.apply(&op, writer).expect("an edge");
        }
    }
    ids
}

/// The write and read cases on stores from `make`, named `store`.
fn cases<S: Store + 'static>(c: &mut Criterion, store_name: &str, make: fn() -> S) {
    let mut group = c.benchmark_group(format!("store_apply/{store_name}"));

    for before_images in [false, true] {
        let images = if before_images { "images" } else { "no_images" };

        let store = make();
        people(&store);
        group.bench_function(format!("immediate_create_node/{images}"), |b| {
            b.iter(|| {
                let id = NodeId::new(store.reserve_node_ids(1).expect("an id").start);
                let op = DataOp::CreateNode {
                    id,
                    labels: [ArcStr::from("Person")].into_iter().collect(),
                    properties: vec![
                        (PropertyKey::new("name"), Value::from("Gus")),
                        (PropertyKey::new("age"), Value::Int64(3)),
                    ],
                };
                black_box(immediate(&store, &op, before_images))
            });
        });

        let store = make();
        let ids = people(&store);
        let mut row = 0_usize;
        group.bench_function(format!("immediate_set_node_property/{images}"), |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                let op = DataOp::SetNodeProperty {
                    id: ids[row],
                    key: PropertyKey::new("age"),
                    value: Value::Int64(i64::try_from(row).unwrap_or(0)),
                };
                black_box(immediate(&store, &op, before_images))
            });
        });

        let store = make();
        let ids = people(&store);
        let mut row = 0_usize;
        group.bench_function(format!("immediate_create_edge/{images}"), |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                let id = EdgeId::new(store.reserve_edge_ids(1).expect("an id").start);
                let op = DataOp::CreateEdge {
                    id,
                    src: ids[row],
                    dst: ids[(row * 7 + 3) % ids.len()],
                    edge_type: ArcStr::from("KNOWS"),
                    properties: Vec::new(),
                };
                black_box(immediate(&store, &op, before_images))
            });
        });

        let store = make();
        let ids = people(&store);
        let mut row = 0_usize;
        group.bench_function(format!("immediate_label_add_and_remove/{images}"), |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                let label = ArcStr::from("Traveller");
                let add = DataOp::AddNodeLabel {
                    id: ids[row],
                    label: label.clone(),
                };
                let remove = DataOp::RemoveNodeLabel {
                    id: ids[row],
                    label,
                };
                black_box(immediate(&store, &add, before_images));
                black_box(immediate(&store, &remove, before_images))
            });
        });
    }

    // A transaction of one write: apply as the transaction, record the
    // entry, stamp it at the next epoch (the store's part of a private
    // transaction; the transaction manager's part is not in here).
    let store = make();
    let ids = people(&store);
    let mut row = 0_usize;
    let mut transaction = 1_000_u64;
    group.bench_function("transaction_set_node_property_and_stamp", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            transaction += 1;
            let id = TransactionId::new(transaction);
            let writer = Writer::Transaction {
                id,
                snapshot: store.current_epoch(),
            };
            let op = DataOp::SetNodeProperty {
                id: ids[row],
                key: PropertyKey::new("age"),
                value: Value::Int64(i64::try_from(row).unwrap_or(0)),
            };
            let mut set = ChangeSet::new();
            let slot = set
                .slot(GraphRef {
                    model: DataModel::Lpg,
                    key: None,
                })
                .expect("a slot");
            if let Applied::Changed { before, version } =
                store.apply(&op, writer).expect("the write applies")
            {
                set.push(slot, op, before, version).expect("the entry");
            }
            store
                .stamp(id, &mut set.entries().iter(), next_epoch(&store))
                .expect("the stamp");
        });
    });

    // The change set's part of the case above, without the store: a new set,
    // its slot and one entry. Subtract it to get the store's part.
    let op = DataOp::SetNodeProperty {
        id: NodeId::new(1),
        key: PropertyKey::new("age"),
        value: Value::Int64(3),
    };
    group.bench_function("change_set_record_one", |b| {
        b.iter(|| {
            let mut set = ChangeSet::new();
            let slot = set
                .slot(GraphRef {
                    model: DataModel::Lpg,
                    key: None,
                })
                .expect("a slot");
            set.push(
                slot,
                op.clone(),
                grafeo_common::change::Before::Value(Some(Value::Int64(19))),
                grafeo_common::change::PendingVersion::Created,
            )
            .expect("the entry");
            black_box(set)
        });
    });

    // Reads of the committed graph.
    let store = make();
    let ids = people(&store);
    let age = PropertyKey::new("age");
    let mut row = 0_usize;
    group.bench_function("read_node_property", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.get_node_property(ids[row], &age))
        });
    });
    group.bench_function("read_node_at_snapshot", |b| {
        let epoch = store.current_epoch();
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.get_node_versioned(ids[row], epoch, TransactionId::new(7)))
        });
    });
    group.bench_function("expand_outgoing", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.edges_from(ids[row], Direction::Outgoing))
        });
    });
    group.bench_function("label_scan_10k", |b| {
        b.iter(|| black_box(store.nodes_by_label("Person").len()));
    });
    group.finish();
}

fn lpg_store() -> LpgStore {
    LpgStore::new().expect("a new store")
}

fn bench_lpg_store(c: &mut Criterion) {
    cases(c, "lpg", lpg_store);
}

fn bench_rowgroup_store(c: &mut Criterion) {
    cases(c, "rowgroup", RowGroupStore::new);
}

criterion_group!(benches, bench_lpg_store, bench_rowgroup_store);
criterion_main!(benches);
