//! The graph algorithms and the vector search, through `CALL`, on graphs large enough that the
//! algorithm's own loops are the cost: the line by which a change to those loops (such as the
//! query deadline's tick, 0.6.2) is checked against the 2 to 3 percent the project allows.
//!
//! Run with: cargo bench -p grafeo-engine --all-features --bench algorithms_bench
#![allow(
    missing_docs,
    reason = "criterion_group! from codspeed-criterion-compat generates wrapper functions without doc comments"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A seeded graph of `nodes` nodes and `nodes * degree` edges, each with a weight `w` from 1 to
/// 9, on a ring (so every node is reached) plus seeded chords.
fn graph(nodes: usize, degree: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..nodes)
        .map(|_| db.create_node(&["V"]).expect("a node"))
        .collect();
    let mut state: u64 = 0x5EED_0A16_0B1E;
    let mut next = |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        usize::try_from(state % bound as u64).expect("below the bound")
    };
    for (index, &from) in ids.iter().enumerate() {
        for hop in 0..degree {
            let to = if hop == 0 {
                ids[(index + 1) % nodes]
            } else {
                ids[next(nodes)]
            };
            let edge = db.create_edge(from, to, "E").expect("an edge");
            let weight = i64::try_from(1 + next(9)).expect("a small weight");
            db.set_edge_property(edge, "w", Value::Int64(weight))
                .expect("a weight");
        }
    }
    db
}

/// `count` documents with a seeded vector of `dimensions` values, indexed.
fn documents(count: usize, dimensions: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut state: u64 = 0xD0C5_5EED_0001;
    let mut unit = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(
            clippy::cast_precision_loss,
            reason = "the top 24 bits of the state are below 2^24, exact in an f32"
        )]
        let value = (state >> 40) as f32 / 16_777_216.0;
        value
    };
    for _ in 0..count {
        let vector: Vec<f32> = (0..dimensions).map(|_| unit()).collect();
        let node = db.create_node(&["Doc"]).expect("a document");
        db.set_node_property(node, "emb", Value::Vector(vector.into()))
            .expect("a vector");
    }
    db.create_vector_index(
        "Doc",
        "emb",
        Some(dimensions),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .expect("the index");
    db
}

fn run(db: &GrafeoDB, query: &str) {
    let result = db.execute(query).expect("the call");
    black_box(result.rows().len());
}

fn bench_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let large = graph(20_000, 8);
    let source = large
        .execute("MATCH (n:V) RETURN id(n) AS id ORDER BY id LIMIT 1")
        .expect("a source")
        .rows()[0][0]
        .clone();
    let dijkstra = format!(
        "CALL grafeo.dijkstra({{source: {}, weight: 'w'}}) YIELD node_id",
        source.as_int64().expect("an id")
    );
    for (name, query) in [
        (
            "pagerank_20k",
            "CALL grafeo.pagerank({max_iterations: 20, tolerance: 0.0}) YIELD node_id",
        ),
        ("dijkstra_20k", dijkstra.as_str()),
        ("louvain_20k", "CALL grafeo.louvain() YIELD node_id"),
        (
            "clustering_coefficient_20k",
            "CALL grafeo.clustering_coefficient() YIELD node_id",
        ),
    ] {
        group.bench_function(name, |b| b.iter(|| run(&large, query)));
    }

    let small = graph(400, 6);
    group.bench_function("floyd_warshall_400", |b| {
        b.iter(|| run(&small, "CALL grafeo.floyd_warshall({weight: 'w'})"));
    });
    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.sample_size(30);
    let db = documents(20_000, 32);
    let query: Vec<String> = (0..32)
        .map(|i| format!("{}", f64::from(i) / 32.0))
        .collect();
    let call = format!(
        "CALL grafeo.search.vector('Doc', 'emb', [{}], 10) YIELD node_id",
        query.join(", ")
    );
    group.bench_function("search_vector_20k_k10", |b| b.iter(|| run(&db, &call)));
    group.finish();
}

criterion_group!(benches, bench_algorithms, bench_vector_search);
criterion_main!(benches);
