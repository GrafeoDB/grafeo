//! Benchmarks for index structures.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use graphos_common::types::NodeId;
use graphos_core::index::adjacency::ChunkedAdjacencyList;
use graphos_core::index::hash::HashIndex;

fn bench_adjacency_insert(c: &mut Criterion) {
    c.bench_function("adjacency_insert_1000", |b| {
        b.iter(|| {
            let mut adj = ChunkedAdjacencyList::new();
            for i in 0..1000u64 {
                adj.add_edge(NodeId(i % 100), NodeId(i), i as u32);
            }
            black_box(adj)
        });
    });
}

fn bench_adjacency_lookup(c: &mut Criterion) {
    let mut adj = ChunkedAdjacencyList::new();
    for i in 0..10000u64 {
        adj.add_edge(NodeId(i % 100), NodeId(i), i as u32);
    }

    c.bench_function("adjacency_lookup", |b| {
        b.iter(|| {
            for i in 0..100u64 {
                black_box(adj.get_neighbors(NodeId(i)));
            }
        });
    });
}

fn bench_hash_index_insert(c: &mut Criterion) {
    c.bench_function("hash_index_insert_1000", |b| {
        b.iter(|| {
            let mut index: HashIndex<u64, NodeId> = HashIndex::new();
            for i in 0..1000u64 {
                index.insert(i, NodeId(i));
            }
            black_box(index)
        });
    });
}

fn bench_hash_index_lookup(c: &mut Criterion) {
    let mut index: HashIndex<u64, NodeId> = HashIndex::new();
    for i in 0..10000u64 {
        index.insert(i, NodeId(i));
    }

    c.bench_function("hash_index_lookup", |b| {
        b.iter(|| {
            for i in 0..1000u64 {
                black_box(index.get(&i));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_adjacency_insert,
    bench_adjacency_lookup,
    bench_hash_index_insert,
    bench_hash_index_lookup,
);

criterion_main!(benches);
