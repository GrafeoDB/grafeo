//! Release benchmark: row-oriented vs storage-level transactional batch writes.
//!
//! Compares the existing per-row `Session::create_node_with_props` /
//! `create_edge_with_props` path against the new transaction-aware storage-level
//! batch APIs (`create_nodes_with_props_transactional` /
//! `create_edges_with_props_transactional`) for the same workload, inside one
//! explicit transaction with a single commit — the shape the incremental
//! WritePlan lane uses (plan §5 performance gate).
//!
//! Run with the shared NVMe target + sccache (release):
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/cargo-targets/jfrie-grafeo-session-batch/target \
//! SCCACHE_DIR=/data/sccache TMPDIR=/data/tmp RUSTC_WRAPPER=sccache \
//!   cargo bench -p grafeo-engine --bench transactional_batch_bench
//! ```
// reason: criterion_group! expansion does not carry doc comments.
#![allow(missing_docs)]
// reason: benchmark loop counters are bounded by NODES/EDGES, fit i64.
#![allow(clippy::cast_possible_wrap)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use grafeo_common::types::{PropertyKey, TransactionId, Value};
use grafeo_core::graph::lpg::{BatchEdgeCreate, BatchNodeCreate, LpgStore};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::{TransactionalEdgeCreate, TransactionalNodeCreate};

/// Nodes per benchmark iteration. Sized to expose lock-hoisting wins without
/// making a single sample slow.
const NODES: usize = 5_000;
/// Edges created (chained), per iteration.
const EDGES: usize = 5_000;

fn build_node_specs(n: usize) -> Vec<TransactionalNodeCreate> {
    (0..n)
        .map(|i| {
            TransactionalNodeCreate::new(["Person"])
                .with_property("id", Value::from(i as i64))
                .with_property("name", Value::from(format!("User{i}")))
                .with_property("age", Value::from(20 + (i % 50) as i64))
        })
        .collect()
}

/// Row-oriented baseline: one public row call per node/edge, single commit.
fn row_oriented(nodes: usize, edges: usize) {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().expect("begin");

    let mut ids = Vec::with_capacity(nodes);
    for i in 0..nodes {
        let id = session
            .create_node_with_props(
                &["Person"],
                [
                    ("id", Value::from(i as i64)),
                    ("name", Value::from(format!("User{i}"))),
                    ("age", Value::from(20 + (i % 50) as i64)),
                ],
            )
            .expect("create node");
        ids.push(id);
    }
    for i in 0..edges {
        session
            .create_edge_with_props(
                ids[i],
                ids[(i + 1) % nodes],
                "KNOWS",
                [("w", Value::from(1i64))],
            )
            .expect("create edge");
    }

    session.commit().expect("commit");
}

/// Storage-level batch: one batch call per entity kind, single commit.
fn storage_batched(specs: &[TransactionalNodeCreate], edges: usize) {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().expect("begin");

    let ids = session
        .create_nodes_with_props_transactional(specs)
        .expect("batch nodes");
    let nodes = ids.len();
    let edge_specs: Vec<TransactionalEdgeCreate> = (0..edges)
        .map(|i| {
            TransactionalEdgeCreate::new(ids[i], ids[(i + 1) % nodes], "KNOWS")
                .with_property("w", Value::from(1i64))
        })
        .collect();
    session
        .create_edges_with_props_transactional(&edge_specs)
        .expect("batch edges");

    session.commit().expect("commit");
}

fn bench_row_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("transactional_batch_write");
    group.throughput(Throughput::Elements((NODES + EDGES) as u64));
    group.sample_size(20);

    let specs = build_node_specs(NODES);

    group.bench_function("row_oriented_nodes_edges", |b| {
        b.iter(|| row_oriented(std::hint::black_box(NODES), std::hint::black_box(EDGES)));
    });

    group.bench_function("storage_batched_nodes_edges", |b| {
        b.iter(|| storage_batched(std::hint::black_box(&specs), std::hint::black_box(EDGES)));
    });

    group.finish();
}

/// Storage-only comparison: isolates the lock-hoisting win of the batch
/// primitive from session/WAL overhead, which is identical in both session
/// paths and dominates at small scale.
fn bench_storage_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("transactional_batch_storage_only");
    group.throughput(Throughput::Elements((NODES + EDGES) as u64));
    group.sample_size(20);

    let tx = TransactionId::new(7);

    // Pre-build batch inputs once (not measured).
    let node_specs: Vec<BatchNodeCreate<'_>> = (0..NODES)
        .map(|i| BatchNodeCreate {
            labels: &["Person"],
            properties: vec![
                (PropertyKey::new("id"), Value::from(i as i64)),
                (PropertyKey::new("name"), Value::from(format!("User{i}"))),
                (PropertyKey::new("age"), Value::from(20 + (i % 50) as i64)),
            ],
        })
        .collect();

    group.bench_function("row_oriented", |b| {
        b.iter(|| {
            let store = LpgStore::new().unwrap();
            let epoch = store.new_epoch();
            let mut ids = Vec::with_capacity(NODES);
            for i in 0..NODES {
                let id = store.create_node_with_props_versioned(
                    &["Person"],
                    [
                        ("id", Value::from(i as i64)),
                        ("name", Value::from(format!("User{i}"))),
                        ("age", Value::from(20 + (i % 50) as i64)),
                    ],
                    epoch,
                    tx,
                );
                ids.push(id);
            }
            for i in 0..EDGES {
                store.create_edge_versioned(ids[i], ids[(i + 1) % NODES], "KNOWS", epoch, tx);
            }
            std::hint::black_box(ids);
        });
    });

    group.bench_function("storage_batched", |b| {
        b.iter(|| {
            let store = LpgStore::new().unwrap();
            let epoch = store.new_epoch();
            let ids =
                store.create_nodes_batch_versioned(std::hint::black_box(&node_specs), epoch, tx);
            let edge_specs: Vec<BatchEdgeCreate<'_>> = (0..EDGES)
                .map(|i| BatchEdgeCreate {
                    source: ids[i],
                    target: ids[(i + 1) % NODES],
                    edge_type: "KNOWS",
                    properties: vec![],
                })
                .collect();
            let eids = store.create_edges_batch_versioned(&edge_specs, epoch, tx);
            std::hint::black_box(eids);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_row_vs_batch, bench_storage_only);
criterion_main!(benches);
