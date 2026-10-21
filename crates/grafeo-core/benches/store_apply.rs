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

use bytes::Bytes;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use grafeo_common::change::{ChangeSet, DataModel, DataOp, GraphRef};
use grafeo_common::storage::{ChunkCaps, ChunkMeta, SectionSink, SectionSource};
use grafeo_common::types::{
    ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value,
};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::apply::{Applied, ChangeTarget, Writer};
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::rowgroup::section::{
    GraphToWrite, LPG_SECTION_VERSION, read_lpg_section, write_lpg_section,
};
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut};

/// The people in the graph each case starts from.
const PEOPLE: u64 = 10_000;

/// How often the write cases with an older reader collect: the reader holds
/// the horizon back between collects, as one that runs for a while does.
/// The other write cases ask for a collect after each write or commit, as
/// the engine does on the row-group store (#412 S6, decision B), which runs
/// it once a bounded number of writes came since the last one.
const COLLECT_EVERY: u32 = 1_024;

/// A store the cases run on.
trait Store: GraphStoreMut + ChangeTarget {}

impl<T: GraphStoreMut + ChangeTarget> Store for T {}

/// A store with a property index.
trait Indexed: Store {
    fn create_property_index(&self, property: &str);
}

impl Indexed for LpgStore {
    fn create_property_index(&self, property: &str) {
        LpgStore::create_property_index(self, property);
    }
}

impl Indexed for RowGroupStore {
    fn create_property_index(&self, property: &str) {
        RowGroupStore::create_property_index(self, property);
    }
}

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

/// Calls `collect` every [`COLLECT_EVERY`] calls, counted in `calls`.
fn collect_now_and_then<S>(store: &S, collect: fn(&S), calls: &mut u32) {
    *calls += 1;
    if *calls == COLLECT_EVERY {
        *calls = 0;
        collect(store);
    }
}

/// The write and read cases on stores from `make`, named `store`.
/// `collect` drops what no reader needs any more (the row-group store's
/// update chains and version stamps, in the vectors it lists, when a
/// collect is due): the cases without an older reader call it after each
/// write or commit, as the engine does, the ones with an older reader now
/// and then.
fn cases<S: Indexed + 'static>(
    c: &mut Criterion,
    store_name: &str,
    make: fn() -> S,
    collect: fn(&S),
) {
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
        let mut calls = 0;
        group.bench_function(format!("immediate_set_node_property/{images}"), |b| {
            b.iter(|| {
                collect_now_and_then(&store, collect, &mut calls);
                row = (row + 1) % ids.len();
                let op = DataOp::SetNodeProperty {
                    id: ids[row],
                    key: PropertyKey::new("age"),
                    value: Value::Int64(i64::try_from(row).unwrap_or(0)),
                };
                black_box(immediate(&store, &op, before_images))
            });
        });

        // The same writes while no reader reads before their epoch: each
        // keeps what it replaces, and the collect after it drops that once a
        // collect is due.
        let store = make();
        let ids = people(&store);
        let mut row = 0_usize;
        group.bench_function(
            format!("immediate_set_node_property_no_older_reader/{images}"),
            |b| {
                b.iter(|| {
                    row = (row + 1) % ids.len();
                    let op = DataOp::SetNodeProperty {
                        id: ids[row],
                        key: PropertyKey::new("age"),
                        value: Value::Int64(i64::try_from(row).unwrap_or(0)),
                    };
                    let applied = immediate(&store, &op, before_images);
                    collect(&store);
                    black_box(applied)
                });
            },
        );

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
        let mut calls = 0;
        group.bench_function(format!("immediate_label_add_and_remove/{images}"), |b| {
            b.iter(|| {
                collect_now_and_then(&store, collect, &mut calls);
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
    // entry, stamp it at the next epoch, collect (the store's part of a
    // private transaction; the transaction manager's part is not in here).
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
            collect(&store);
        });
    });

    // The same transaction on a store with an index on the key it writes:
    // the index keeps the old value for older readers until the collect.
    let store = make();
    store.create_property_index("age");
    let ids = people(&store);
    let mut row = 0_usize;
    group.bench_function("transaction_set_indexed_property_and_stamp", |b| {
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
            collect(&store);
        });
    });

    // Lookups through an index on `name` (one node per name): without a
    // view, at a transaction's view, and maybe-equal at the view.
    let store = make();
    store.create_property_index("name");
    let ids = people(&store);
    let names: Vec<Value> = (0..ids.len())
        .map(|row| Value::from(format!("Alix {row}")))
        .collect();
    let view = ReadView::transaction(TransactionId::new(7), store.current_epoch());
    let mut row = 0_usize;
    group.bench_function("index_lookup", |b| {
        b.iter(|| {
            row = (row + 1) % names.len();
            black_box(store.find_nodes_by_property("name", &names[row]))
        });
    });
    group.bench_function("index_lookup_at_view", |b| {
        b.iter(|| {
            row = (row + 1) % names.len();
            black_box(store.find_nodes_by_property_at("name", &names[row], view))
        });
    });
    group.bench_function("index_maybe_equal_at_view", |b| {
        b.iter(|| {
            row = (row + 1) % names.len();
            black_box(store.find_nodes_maybe_equal_at("name", &names[row], view))
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

    // Reads at a transaction's view, and the reads of now with the
    // visibility checks the engine made before the reads at a view.
    let view = ReadView::transaction(TransactionId::new(7), store.current_epoch());
    group.bench_function("read_node_property_at_view", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.node_property_at(ids[row], "age", view))
        });
    });
    group.bench_function("expand_outgoing_at_view", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.edges_from_at(ids[row], Direction::Outgoing, view))
        });
    });
    group.bench_function("expand_outgoing_then_filter", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            let mut edges = store.edges_from(ids[row], Direction::Outgoing);
            edges.retain(|&(_, edge)| {
                store.is_edge_visible_versioned(edge, view.epoch, TransactionId::new(7))
            });
            black_box(edges)
        });
    });
    group.bench_function("label_scan_10k_at_view", |b| {
        b.iter(|| black_box(store.nodes_by_label_at("Person", view).len()));
    });
    group.bench_function("label_scan_10k_then_filter", |b| {
        b.iter(|| {
            let ids = store.nodes_by_label("Person");
            black_box(
                store
                    .filter_visible_node_ids_versioned(&ids, view.epoch, TransactionId::new(7))
                    .len(),
            )
        });
    });

    // Reads by a committed reader while an open transaction has written
    // every person's age and the label `Traveller`: where a store keeps
    // the old values for older readers, each read walks what it kept.
    let store = make();
    let ids = people(&store);
    let committed = ReadView::committed(store.current_epoch());
    let writer = Writer::Transaction {
        id: TransactionId::new(7),
        snapshot: store.current_epoch(),
    };
    for (row, id) in ids.iter().enumerate() {
        for op in [
            DataOp::SetNodeProperty {
                id: *id,
                key: PropertyKey::new("age"),
                value: Value::Int64(i64::try_from(row).unwrap_or(0)),
            },
            DataOp::AddNodeLabel {
                id: *id,
                label: ArcStr::from("Traveller"),
            },
        ] {
            store.apply(&op, writer).expect("the open write");
        }
    }
    group.bench_function("read_node_property_at_view_with_open_writes", |b| {
        b.iter(|| {
            row = (row + 1) % ids.len();
            black_box(store.node_property_at(ids[row], "age", committed))
        });
    });
    group.bench_function("label_scan_10k_at_view_with_open_writes", |b| {
        b.iter(|| black_box(store.nodes_by_label_at("Person", committed).len()));
    });
    group.finish();
}

fn lpg_store() -> LpgStore {
    LpgStore::new().expect("a new store")
}

fn bench_lpg_store(c: &mut Criterion) {
    // `LpgStore` keeps no old values to collect.
    cases(c, "lpg", lpg_store, |_| {});
}

fn bench_rowgroup_store(c: &mut Criterion) {
    cases(c, "rowgroup", RowGroupStore::new, |store| {
        store.collect_after_write(store.current_epoch());
    });
}

/// The row-group store with and without backward edges (10b), paired: an
/// edge create (one adjacency entry fewer without them), an outgoing expand
/// (alike), an incoming expand (a scan of the edge table's targets without
/// them) and the in-degree, on the graph of [`people`].
fn bench_rowgroup_backward_edges(c: &mut Criterion) {
    for (name, backward_edges) in [("kept", true), ("not_kept", false)] {
        let mut group = c.benchmark_group(format!("store_apply/rowgroup_backward_edges_{name}"));
        let store = RowGroupStore::with_backward_edges(backward_edges);
        let ids = people(&store);
        let mut row = 0_usize;
        group.bench_function("immediate_create_edge", |b| {
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
                black_box(immediate(&store, &op, false))
            });
        });

        let store = RowGroupStore::with_backward_edges(backward_edges);
        let ids = people(&store);
        store.collect(store.current_epoch());
        group.bench_function("expand_outgoing", |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                black_box(store.edges_from(ids[row], Direction::Outgoing))
            });
        });
        group.bench_function("expand_incoming", |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                black_box(store.edges_from(ids[row], Direction::Incoming))
            });
        });
        group.bench_function("in_degree", |b| {
            b.iter(|| {
                row = (row + 1) % ids.len();
                black_box(store.in_degree(ids[row]))
            });
        });
        group.finish();
    }
}

/// The same encoded section supplies every cold-read iteration.
#[derive(Default)]
struct MemorySection {
    metas: Vec<ChunkMeta>,
    bytes: Vec<Bytes>,
}

impl SectionSink for MemorySection {
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
        self.metas.push(meta);
        self.bytes.push(Bytes::copy_from_slice(bytes));
        Ok(())
    }
}

impl SectionSource for MemorySection {
    fn chunks(&self) -> &[ChunkMeta] {
        &self.metas
    }

    fn fetch(&self, index: usize) -> Result<Bytes> {
        Ok(self.bytes[index].clone())
    }

    fn stored_length(&self, index: usize) -> Result<u64> {
        Ok(u64::try_from(self.bytes[index].len()).expect("a chunk length fits u64"))
    }

    fn section_version(&self) -> u8 {
        LPG_SECTION_VERSION
    }
}

fn cold_property_section() -> (MemorySection, NodeId) {
    const NODES: u64 = 65_536;
    let store = RowGroupStore::new();
    let first = store.reserve_node_ids(NODES).expect("node ids").start;
    let writer = Writer::Replay {
        epoch: EpochId::new(1),
    };
    for row in 0..NODES {
        let op = DataOp::CreateNode {
            id: NodeId::new(first + row),
            labels: Default::default(),
            properties: vec![(
                PropertyKey::new("ordinal"),
                Value::Int64(i64::try_from(row).expect("the row fits i64")),
            )],
        };
        store.apply(&op, writer).expect("a node");
    }

    let mut section = MemorySection::default();
    write_lpg_section(
        &[GraphToWrite {
            id: 0,
            name: "",
            store: &store,
        }],
        1,
        store.current_epoch(),
        ChunkCaps::DEFAULT,
        &mut section,
    )
    .expect("write the property section");
    (section, NodeId::new(first + NODES - 1))
}

fn load_property_section(section: &MemorySection) -> RowGroupStore {
    let (next_graph, mut graphs) = read_lpg_section(section).expect("read the property section");
    assert_eq!((next_graph, graphs.len()), (1, 1));
    let store = graphs.remove(0).store;
    assert_eq!(store.cold_columns(), 1, "the property starts encoded");
    store
}

fn bench_rowgroup_cold_reads(c: &mut Criterion) {
    let (section, last) = cold_property_section();
    let key = PropertyKey::new("ordinal");
    let expected = Some(Value::Int64(65_535));
    let mut group = c.benchmark_group("store_apply/rowgroup_cold");

    group.bench_function("first_point_read_65536", |b| {
        b.iter_batched(
            || load_property_section(&section),
            |store| {
                assert_eq!(
                    black_box(store.get_node_property(black_box(last), black_box(&key))),
                    expected,
                );
                // Returning the store keeps its destruction outside timing.
                store
            },
            BatchSize::PerIteration,
        );
    });

    let hot = load_property_section(&section);
    // A same-value write thaws the column on both baseline and candidate.
    hot.set_node_property(last, "ordinal", Value::Int64(65_535));
    assert_eq!(hot.cold_columns(), 0, "the control uses a hot column");
    group.bench_function("hot_point_read_65536", |b| {
        b.iter(|| {
            assert_eq!(
                black_box(hot.get_node_property(black_box(last), black_box(&key))),
                expected,
            );
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lpg_store,
    bench_rowgroup_store,
    bench_rowgroup_backward_edges,
    bench_rowgroup_cold_reads
);
criterion_main!(benches);
