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
//! Run with: cargo bench -p grafeo-engine --bench regression_bench
//! All features: cargo bench --all-features --bench regression_bench
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Duration;

#[cfg(codspeed)]
use criterion::criterion_main;
use criterion::{Criterion, criterion_group};

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

fn bench_1hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_1hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_2hop_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_2hop_1k", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) \
                     RETURN DISTINCT c.id",
                )
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

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

fn bench_1hop_5k(c: &mut Criterion) {
    let db = setup_social_graph(5_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_1hop_5k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
                .unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_fan_out_5k(c: &mut Criterion) {
    let db = setup_social_graph(5_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("multihop");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_fan_out_5k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b) RETURN COUNT(b)")
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

fn bench_repeat_unique_500(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    let mut group = c.benchmark_group("repeated");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("regression_repeat_unique_500", |b| {
        b.iter(|| {
            for i in 0..500 {
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

criterion_group!(
    multihop_benches,
    bench_1hop_1k,
    bench_2hop_1k,
    bench_3hop_1k,
    bench_1hop_5k,
    bench_fan_out_5k,
);

criterion_group!(
    repeated_benches,
    bench_repeat_unique_100,
    bench_repeat_unique_500,
    bench_repeat_cached_500,
);

criterion_group!(
    edge_filter_benches,
    bench_edge_filter_single,
    bench_edge_filter_follows,
    bench_edge_filter_any,
);

/// Native grouped callers that borrow their results.
fn bench_native_aggregate_resource(c: &mut Criterion) {
    bench_aggregate_resource(c, false);
}

/// The same callers transferring rows through the existing public API.
fn bench_native_owned_aggregate_resource(c: &mut Criterion) {
    bench_aggregate_resource(c, true);
}

fn bench_aggregate_resource(c: &mut Criterion, owned: bool) {
    #[cfg(all(feature = "lpg", feature = "gql", feature = "spill"))]
    {
        use criterion::BatchSize;
        use grafeo_common::memory::buffer::MemoryRegion;
        use grafeo_common::types::Value;
        use grafeo_common::utils::error::Error;
        use grafeo_engine::Config;

        const GROUPS: usize = 1024;
        const BUDGET: usize = 32 << 20;
        const QUERY: &str = "MATCH (n:AggregateCost) RETURN n.bucket AS bucket, sum(n.value) AS total, count(*) AS count";
        let validate = |result: &grafeo_engine::database::QueryResult| {
            assert_eq!(result.row_count(), GROUPS);
            let mut seen = [false; GROUPS];
            for row in result.rows() {
                let [
                    Value::Int64(bucket),
                    Value::Int64(total),
                    Value::Int64(count),
                ] = row.as_slice()
                else {
                    panic!("unexpected aggregate row: {row:?}");
                };
                let bucket = usize::try_from(*bucket).unwrap();
                assert!(bucket < GROUPS && !seen[bucket]);
                seen[bucket] = true;
                assert_eq!((*total, *count), (10, 4));
            }
            assert!(seen.into_iter().all(|value| value));
        };
        let consume = |result: grafeo_engine::database::QueryResult| {
            if owned {
                let rows = result.into_rows();
                black_box(&rows);
                drop(rows);
            } else {
                black_box(&result);
                drop(result);
            }
        };
        let mut group = c.benchmark_group(if owned {
            "native_owned_aggregate_resource"
        } else {
            "native_aggregate_resource"
        });
        group.sample_size(50);
        group.measurement_time(Duration::from_secs(10));
        group.warm_up_time(Duration::from_secs(3));
        for (name, configured, pressure, cold) in [
            ("groups_1024_rows_4096_ordinary_cold", false, false, true),
            ("groups_1024_rows_4096_cached", false, false, false),
            (
                "groups_1024_rows_4096_configured_resident",
                true,
                false,
                false,
            ),
            (
                "groups_1024_rows_4096_pressure_90_percent",
                true,
                true,
                false,
            ),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let spill_path = directory.path().join("spill");
            let mut config = Config::in_memory()
                .with_memory_limit(BUDGET)
                .without_query_timeout();
            if configured {
                config = config.with_spill_path(&spill_path);
            }
            let database = GrafeoDB::with_config(config).unwrap();
            for value in 1..=4_i64 {
                for bucket in 0..GROUPS {
                    database
                        .create_node_with_props(
                            &["AggregateCost"],
                            [
                                ("bucket", Value::Int64(i64::try_from(bucket).unwrap())),
                                ("value", Value::Int64(value)),
                            ],
                        )
                        .expect("insert grouped benchmark fixture");
                }
            }
            let manager = database.buffer_manager();
            let baseline = manager.allocated();
            let held = pressure.then(|| {
                manager
                    .try_allocate(
                        (BUDGET * 9 / 10).checked_sub(baseline).unwrap(),
                        MemoryRegion::ExecutionBuffers,
                    )
                    .unwrap()
            });
            let held_total = manager.allocated();
            if pressure {
                assert_eq!(held_total, BUDGET * 9 / 10);
            }
            let session = database.session();
            if pressure {
                // Spill directories are lazy. This error proves that this exact
                // grouped workload reaches a filesystem spill, without a watcher.
                std::fs::write(&spill_path, b"blocked spill directory").unwrap();
                let error = session.execute(QUERY).unwrap_err();
                assert!(matches!(error, Error::Io(_)), "{error}");
                assert_eq!(manager.allocated(), held_total);
                std::fs::remove_file(&spill_path).unwrap();
            }
            database.query_cache().clear();
            let before = database.query_cache().stats();
            let result = session.execute(QUERY).unwrap();
            validate(&result);
            drop(result);
            assert_eq!(manager.allocated(), held_total);
            let primed = database.query_cache().stats();
            assert!(primed.optimized_misses > before.optimized_misses);
            let result = session.execute(QUERY).unwrap();
            validate(&result);
            drop(result);
            let warmed = database.query_cache().stats();
            assert!(warmed.optimized_hits > primed.optimized_hits);
            assert_eq!(manager.allocated(), held_total);
            if configured && !pressure {
                assert!(!spill_path.exists(), "resident control must not spill");
            }

            group.bench_function(name, |b| {
                if cold {
                    // PerIteration prevents Criterion from batching cache clears
                    // ahead of several queries and accidentally timing warm calls.
                    b.iter_batched(
                        || database.query_cache().clear(),
                        |()| {
                            let result = session.execute(black_box(QUERY)).unwrap();
                            consume(result);
                        },
                        BatchSize::PerIteration,
                    );
                } else {
                    b.iter(|| {
                        let result = session.execute(black_box(QUERY)).unwrap();
                        consume(result);
                    });
                }
            });
            assert_eq!(manager.allocated(), held_total);
            if spill_path.exists() {
                assert_eq!(std::fs::read_dir(&spill_path).unwrap().count(), 0);
            }
            drop(held);
            assert_eq!(manager.allocated(), baseline);
        }
        group.finish();
    }
    #[cfg(not(all(feature = "lpg", feature = "gql", feature = "spill")))]
    let _ = (c, owned);
}

criterion_group!(
    native_aggregate_benches,
    bench_native_aggregate_resource,
    bench_native_owned_aggregate_resource,
);

#[cfg(codspeed)]
criterion_main!(
    multihop_benches,
    repeated_benches,
    edge_filter_benches,
    native_aggregate_benches
);

#[cfg(not(codspeed))]
fn main() {
    // The cost protocol supplies its positional filter first. Inspect only
    // that argument so a baseline named after this group is not a filter.
    if std::env::args().nth(1).is_some_and(|filter| {
        filter.starts_with("native_aggregate_resource")
            || filter.starts_with("native_owned_aggregate_resource")
    }) {
        native_aggregate_benches();
    } else {
        multihop_benches();
        repeated_benches();
        edge_filter_benches();
        native_aggregate_benches();
    }
    // Match the pinned Criterion main macro's final configuration/summary.
    Criterion::default().configure_from_args().final_summary();
}
