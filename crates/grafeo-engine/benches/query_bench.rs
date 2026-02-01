//! End-to-end query benchmarks for the Grafeo engine.
//!
//! Measures full query execution time from GQL string → result,
//! covering parsing, planning, optimization, and execution.
//!
//! Run with: cargo bench -p grafeo-engine

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_engine::GrafeoDB;

/// Sets up a small social graph for benchmarking.
/// Returns the database instance ready for queries.
fn setup_social_graph(node_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create Person nodes with properties
    for i in 0..node_count {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}}})",
            i,
            i,
            20 + (i % 50)
        );
        session.execute(&query).unwrap();
    }

    // Create KNOWS edges using CREATE (not INSERT - INSERT doesn't work after MATCH in GQL)
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

fn bench_node_lookup(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_node_lookup_by_property", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (n:Person {id: 42}) RETURN n.name")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_pattern_match(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_1hop_pattern", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.name")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_two_hop_pattern(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_2hop_pattern", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN DISTINCT c.id",
                )
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_aggregation_count(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_count_all", |b| {
        b.iter(|| {
            let result = session.execute("MATCH (n:Person) RETURN COUNT(n)").unwrap();
            black_box(result)
        });
    });
}

fn bench_filter_range(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_filter_range", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (n:Person) WHERE n.age > 50 RETURN n.id")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_insert_single_node(c: &mut Criterion) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let mut counter = 0u64;
    c.bench_function("query_insert_single_node", |b| {
        b.iter(|| {
            let query = format!("INSERT (:Bench {{id: {}}})", counter);
            counter += 1;
            let result = session.execute(&query).unwrap();
            black_box(result)
        });
    });
}

criterion_group!(
    benches,
    bench_node_lookup,
    bench_pattern_match,
    bench_two_hop_pattern,
    bench_aggregation_count,
    bench_filter_range,
    bench_insert_single_node,
);

criterion_main!(benches);
