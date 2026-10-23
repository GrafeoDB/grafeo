//! Targeted regression benchmarks for documented performance vectors.
//!
//! These benchmarks cover the three regression categories identified in
//! performance-degradation.md (50-120% multi-hop regression between v0.5.6
//! and v0.5.21):
//!
//! 1. Multi-hop traversal: vtable dispatch + MVCC compounding at depth
//! 2. Repeated execution: parser overhead compounding across many queries
//! 3. Edge type filtering: MVCC edge_type_versioned lookup overhead
//!
//! A case that times a path another case already times is left out (one-hop
//! and two-hop patterns: query_bench; the same paths at larger sizes or
//! counts), which keeps CodSpeed's run short.
//!
//! Run with: cargo bench -p grafeo-engine --bench regression_bench
//! All features: cargo bench --all-features --bench regression_bench
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_engine::GrafeoDB;

// ============================================================================
// Setup helpers
// ============================================================================

/// Sets up a social graph with Person nodes and KNOWS edges.
fn setup_social_graph(node_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..node_count {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}}})",
            i,
            i,
            20 + (i % 50)
        );
        session.execute(&query).unwrap();
    }

    let edge_count = node_count * edge_multiplier;
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 7 + 13) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) CREATE (a)-[:KNOWS]->(b)",
                src, dst
            );
            let _ = session.execute(&query);
        }
    }

    db
}

/// Sets up a graph with multiple edge types for filtering benchmarks.
/// Creates KNOWS, FOLLOWS, and LIKES edges between Person nodes.
fn setup_multi_type_graph(node_count: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for i in 0..node_count {
        let query = format!("INSERT (:Person {{id: {}, name: 'User{}'}})", i, i);
        session.execute(&query).unwrap();
    }

    let types = ["KNOWS", "FOLLOWS", "LIKES"];
    let edges_per_type = node_count * 3;
    for (type_idx, edge_type) in types.iter().enumerate() {
        for i in 0..edges_per_type {
            let src = i % node_count;
            let dst = (i * (type_idx + 3) + 7) % node_count;
            if src != dst {
                let query = format!(
                    "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) CREATE (a)-[:{}]->(b)",
                    src, dst, edge_type
                );
                let _ = session.execute(&query);
            }
        }
    }

    db
}

// ============================================================================
// Multi-hop traversal benchmarks
// ============================================================================

fn bench_3hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_3hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d) \
                     RETURN DISTINCT d.id LIMIT 5000",
                )
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// Repeated execution benchmarks
// ============================================================================

fn bench_repeat_unique_100(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_unique_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                let query = format!("MATCH (n:Person {{id: {}}}) RETURN n.name", i);
                let result = session.execute(&query).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_repeat_cached_500(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_cached_500", |b| {
        b.iter(|| {
            for _ in 0..500 {
                let result = session
                    .execute("MATCH (n:Person {id: 42}) RETURN n.name")
                    .unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Edge type filtering benchmarks
// ============================================================================

fn bench_edge_filter_single(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_edge_filter_single", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_edge_filter_follows(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    // Second edge type filter: tests MVCC edge_type_versioned with a
    // different type in the same graph (compare against single/KNOWS).
    group.bench_function("regression_edge_filter_follows", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:FOLLOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_edge_filter_any(c: &mut Criterion) {
    let db = setup_multi_type_graph(2_000);
    let session = db.session();

    let mut group = c.benchmark_group("edge_filter");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_edge_filter_any", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// Groups and main
// ============================================================================

criterion_group!(multihop_benches, bench_3hop_1k);

criterion_group!(
    repeated_benches,
    bench_repeat_unique_100,
    bench_repeat_cached_500,
);

criterion_group!(
    edge_filter_benches,
    bench_edge_filter_single,
    bench_edge_filter_follows,
    bench_edge_filter_any,
);

criterion_main!(multihop_benches, repeated_benches, edge_filter_benches);
