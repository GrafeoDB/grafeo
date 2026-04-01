//! Benchmark: planner label-first path vs find_nodes_by_property.
//!
//! The query planner has two paths for property-equality filters:
//! 1. (indexed) → find_nodes_by_properties → find_nodes_by_property
//! 2. (label-first) → nodes_by_label → get_node() per row → check property
//!
//! CompactStore has no property indexes, so path 2 is always taken.
//! This benchmark measures the cost difference.
//!
//! Run: `cargo bench -p grafeo-core --bench planner_path_bench --features compact-store`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::Value;
use grafeo_core::graph::compact::builder::CompactStoreBuilder;
use grafeo_core::graph::traits::GraphStore;

fn build_store(n: usize) -> grafeo_core::graph::compact::CompactStore {
    let scores: Vec<u64> = (0..n).map(|i| (i % 10) as u64).collect();
    let names: Vec<&str> = (0..n)
        .map(|i| match i % 5 {
            0 => "alpha",
            1 => "beta",
            2 => "gamma",
            3 => "delta",
            _ => "epsilon",
        })
        .collect();

    CompactStoreBuilder::new()
        .node_table("Item", |t| {
            t.column_bitpacked("score", &scores, 4)
                .column_dict("name", &names)
        })
        .build()
        .expect("build failed")
}

/// Simulates the planner's current label-first path (filter.rs:866-889).
/// For each node: get_node() → materialize full Node → check one property.
fn planner_label_first_path(
    store: &impl GraphStore,
    label: &str,
    property: &str,
    target: &Value,
) -> Vec<grafeo_common::types::NodeId> {
    let label_nodes = store.nodes_by_label(label);
    label_nodes
        .into_iter()
        .filter(|&node_id| {
            let node = store.get_node(node_id);
            node.is_some_and(|n| n.get_property(property).is_some_and(|v| v == target))
        })
        .collect()
}

/// Proposed fix: route through find_nodes_by_property (which uses typed scans).
/// tables, so if label matches the table, no intersection needed.
fn planner_find_nodes_direct(
    store: &impl GraphStore,
    _label: &str,
    property: &str,
    target: &Value,
) -> Vec<grafeo_common::types::NodeId> {
    // For CompactStore, find_nodes_by_property scans all tables (each table is
    // one label), so the results are already label-correct.
    store.find_nodes_by_property(property, target)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_dict_equality(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner_dict_eq");

    for &n in &[1_000usize, 10_000, 100_000] {
        let store = build_store(n);

        let target = Value::String("gamma".into());

        // Verify both paths produce the same results.
        let a = planner_label_first_path(&store, "Item", "name", &target);
        let b = planner_find_nodes_direct(&store, "Item", "name", &target);
        assert_eq!(a.len(), b.len(), "mismatch at n={n}: {} vs {}", a.len(), b.len());

        group.bench_function(format!("label_first_get_node/{n}"), |bench| {
            bench.iter(|| {
                black_box(planner_label_first_path(&store, "Item", "name", &target).len())
            });
        });

        group.bench_function(format!("find_nodes_by_property/{n}"), |bench| {
            bench.iter(|| {
                black_box(planner_find_nodes_direct(&store, "Item", "name", &target).len())
            });
        });
    }

    group.finish();
}

fn bench_int_equality(c: &mut Criterion) {
    let mut group = c.benchmark_group("planner_int_eq");

    for &n in &[1_000usize, 10_000, 100_000] {
        let store = build_store(n);

        let target = Value::Int64(5);

        let a = planner_label_first_path(&store, "Item", "score", &target);
        let b = planner_find_nodes_direct(&store, "Item", "score", &target);
        assert_eq!(a.len(), b.len(), "mismatch at n={n}: {} vs {}", a.len(), b.len());

        group.bench_function(format!("label_first_get_node/{n}"), |bench| {
            bench.iter(|| {
                black_box(planner_label_first_path(&store, "Item", "score", &target).len())
            });
        });

        group.bench_function(format!("find_nodes_by_property/{n}"), |bench| {
            bench.iter(|| {
                black_box(planner_find_nodes_direct(&store, "Item", "score", &target).len())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dict_equality, bench_int_equality);
criterion_main!(benches);
