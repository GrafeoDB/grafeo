//! Benchmarks for index structures.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_core::index::adjacency::ChunkedAdjacency;
use grafeo_core::index::hash::HashIndex;
use grafeo_core::index::vector::{
    DistanceMetric, HnswConfig, HnswIndex, ProductQuantizer, ScalarQuantizer,
    brute_force_knn, compute_distance,
};

fn bench_adjacency_insert(c: &mut Criterion) {
    c.bench_function("adjacency_insert_1000", |b| {
        b.iter(|| {
            let adj = ChunkedAdjacency::new();
            for i in 0..1000u64 {
                adj.add_edge(NodeId(i % 100), NodeId(i), EdgeId(i));
            }
            black_box(adj)
        });
    });
}

fn bench_adjacency_lookup(c: &mut Criterion) {
    let adj = ChunkedAdjacency::new();
    for i in 0..10000u64 {
        adj.add_edge(NodeId(i % 100), NodeId(i), EdgeId(i));
    }

    c.bench_function("adjacency_lookup", |b| {
        b.iter(|| {
            for i in 0..100u64 {
                black_box(adj.neighbors(NodeId(i)));
            }
        });
    });
}

fn bench_hash_index_insert(c: &mut Criterion) {
    c.bench_function("hash_index_insert_1000", |b| {
        b.iter(|| {
            let index: HashIndex<u64, NodeId> = HashIndex::new();
            for i in 0..1000u64 {
                index.insert(i, NodeId(i));
            }
            black_box(index)
        });
    });
}

fn bench_hash_index_lookup(c: &mut Criterion) {
    let index: HashIndex<u64, NodeId> = HashIndex::new();
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

// ============================================================================
// Vector Index Benchmarks
// ============================================================================

fn generate_random_vectors(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    // Simple deterministic pseudo-random generator
    let mut state = seed;
    (0..count)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect()
}

fn bench_distance_computation(c: &mut Criterion) {
    let dims = 384; // Common embedding size
    let vectors = generate_random_vectors(2, dims, 42);
    let v1 = &vectors[0];
    let v2 = &vectors[1];

    let mut group = c.benchmark_group("distance");

    group.bench_function("cosine_384d", |b| {
        b.iter(|| black_box(compute_distance(v1, v2, DistanceMetric::Cosine)));
    });

    group.bench_function("euclidean_384d", |b| {
        b.iter(|| black_box(compute_distance(v1, v2, DistanceMetric::Euclidean)));
    });

    group.bench_function("dot_product_384d", |b| {
        b.iter(|| black_box(compute_distance(v1, v2, DistanceMetric::DotProduct)));
    });

    group.finish();
}

fn bench_brute_force_knn(c: &mut Criterion) {
    let dims = 128;

    let mut group = c.benchmark_group("brute_force_knn");

    for &count in &[100, 1000, 10000] {
        let vectors = generate_random_vectors(count, dims, 42);
        let query = generate_random_vectors(1, dims, 123)[0].clone();

        let indexed: Vec<(NodeId, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64), v.as_slice()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("k10", count),
            &indexed,
            |b, indexed| {
                b.iter(|| {
                    black_box(brute_force_knn(
                        indexed.iter().map(|(id, v)| (*id, *v)),
                        &query,
                        10,
                        DistanceMetric::Cosine,
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_insert(c: &mut Criterion) {
    let dims = 128;

    let mut group = c.benchmark_group("hnsw_insert");

    for &count in &[100, 500, 1000] {
        let vectors = generate_random_vectors(count, dims, 42);

        group.bench_with_input(BenchmarkId::new("vectors", count), &vectors, |b, vectors| {
            b.iter(|| {
                let config = HnswConfig::new(dims, DistanceMetric::Cosine);
                let index = HnswIndex::with_seed(config, 12345);
                for (i, vec) in vectors.iter().enumerate() {
                    index.insert(NodeId::new(i as u64), vec);
                }
                black_box(index)
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let dims = 128;
    let count = 5000;

    let vectors = generate_random_vectors(count, dims, 42);
    let config = HnswConfig::new(dims, DistanceMetric::Cosine);
    let index = HnswIndex::with_seed(config, 12345);

    for (i, vec) in vectors.iter().enumerate() {
        index.insert(NodeId::new(i as u64), vec);
    }

    let query = generate_random_vectors(1, dims, 999)[0].clone();

    let mut group = c.benchmark_group("hnsw_search");

    for &k in &[1, 10, 50] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| black_box(index.search(&query, k)));
        });
    }

    group.finish();
}

fn bench_scalar_quantization(c: &mut Criterion) {
    let dims = 384;
    let count = 1000;

    let vectors = generate_random_vectors(count, dims, 42);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let quantizer = ScalarQuantizer::train(&refs);

    let query = &vectors[0];

    let mut group = c.benchmark_group("scalar_quantization");

    group.bench_function("train_1000x384", |b| {
        b.iter(|| black_box(ScalarQuantizer::train(&refs)));
    });

    group.bench_function("quantize_384d", |b| {
        b.iter(|| black_box(quantizer.quantize(query)));
    });

    let quantized = quantizer.quantize(query);
    let quantized2 = quantizer.quantize(&vectors[1]);

    group.bench_function("distance_u8_384d", |b| {
        b.iter(|| black_box(quantizer.distance_u8(&quantized, &quantized2)));
    });

    group.finish();
}

fn bench_product_quantization(c: &mut Criterion) {
    let dims = 128; // Must be divisible by num_subvectors
    let count = 500;
    let num_subvectors = 8;

    let vectors = generate_random_vectors(count, dims, 42);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let quantizer = ProductQuantizer::train(&refs, num_subvectors, 256, 5);

    let query = &vectors[0];

    let mut group = c.benchmark_group("product_quantization");

    group.bench_function("train_500x128_pq8", |b| {
        b.iter(|| black_box(ProductQuantizer::train(&refs, num_subvectors, 256, 5)));
    });

    group.bench_function("quantize_128d_pq8", |b| {
        b.iter(|| black_box(quantizer.quantize(query)));
    });

    let codes = quantizer.quantize(query);

    group.bench_function("asymmetric_distance_128d_pq8", |b| {
        b.iter(|| black_box(quantizer.asymmetric_distance(query, &codes)));
    });

    group.bench_function("build_distance_table_128d_pq8", |b| {
        b.iter(|| black_box(quantizer.build_distance_table(query)));
    });

    let table = quantizer.build_distance_table(query);

    group.bench_function("distance_with_table_pq8", |b| {
        b.iter(|| black_box(quantizer.distance_with_table(&table, &codes)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_adjacency_insert,
    bench_adjacency_lookup,
    bench_hash_index_insert,
    bench_hash_index_lookup,
    bench_distance_computation,
    bench_brute_force_knn,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_scalar_quantization,
    bench_product_quantization,
);

criterion_main!(benches);
