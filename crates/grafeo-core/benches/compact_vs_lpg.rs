//! Head-to-head benchmark: CompactStore vs LpgStore through the GraphStore trait.
//!
//! Both stores are loaded with identical data, then queried through the same
//! trait interface. This measures the real end-to-end performance difference,
//! not raw primitives.
//!
//! Run: cargo bench -p grafeo-core --bench compact_vs_lpg --features "compact-store,vector-index"

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grafeo_common::types::{PropertyKey, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::compact::CompactStoreBuilder;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::traits::GraphStore;
use std::hint::black_box;

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Build both stores with identical data at the given scale.
fn build_stores(
    num_items: usize,
    num_activities: usize,
    num_accounts: usize,
) -> (LpgStore, grafeo_core::graph::compact::CompactStore) {
    // --- CompactStore ---
    let scores: Vec<u64> = (0..num_items).map(|i| (i % 10) as u64).collect();
    let item_names: Vec<&str> = (0..num_items)
        .map(|i| match i % 5 {
            0 => "alpha",
            1 => "beta",
            2 => "gamma",
            3 => "delta",
            _ => "epsilon",
        })
        .collect();

    let ratings: Vec<u64> = (0..num_activities).map(|i| (i % 5 + 1) as u64).collect();
    let activity_item_ids: Vec<u64> = (0..num_activities)
        .map(|i| (i % num_items) as u64)
        .collect();
    let activity_account_ids: Vec<u64> = (0..num_activities)
        .map(|i| (i % num_accounts) as u64)
        .collect();

    let account_names: Vec<&str> = (0..num_accounts)
        .map(|i| match i % 4 {
            0 => "user_a",
            1 => "user_b",
            2 => "user_c",
            _ => "user_d",
        })
        .collect();

    let activity_on_edges: Vec<(u32, u32)> = (0..num_activities)
        .map(|i| (i as u32, (i % num_items) as u32))
        .collect();

    let performed_by_edges: Vec<(u32, u32)> = (0..num_activities)
        .map(|i| (i as u32, (i % num_accounts) as u32))
        .collect();

    let compact = CompactStoreBuilder::new()
        .node_table("Item", |t| {
            t.column_bitpacked("score", &scores, 4)
                .column_dict("name", &item_names)
        })
        .node_table("Activity", |t| {
            t.column_bitpacked("rating", &ratings, 4)
                .column_bitpacked("item_id", &activity_item_ids, 20)
                .column_bitpacked("account_id", &activity_account_ids, 20)
        })
        .node_table("Account", |t| t.column_dict("name", &account_names))
        .rel_table("ACTIVITY_ON", "Activity", "Item", |r| {
            r.edges(activity_on_edges.clone()).backward(true)
        })
        .rel_table("PERFORMED_BY", "Activity", "Account", |r| {
            r.edges(performed_by_edges.clone()).backward(true)
        })
        .build()
        .expect("CompactStore build failed");

    // --- LpgStore with same data ---
    let lpg = LpgStore::new().expect("LpgStore creation failed");

    let mut item_ids = Vec::with_capacity(num_items);
    for i in 0..num_items {
        let id = lpg.create_node(&["Item"]);
        lpg.set_node_property(id, "score", Value::Int64((i % 10) as i64));
        lpg.set_node_property(
            id,
            "name",
            Value::String(arcstr::ArcStr::from(item_names[i])),
        );
        item_ids.push(id);
    }

    let mut activity_ids = Vec::with_capacity(num_activities);
    for i in 0..num_activities {
        let id = lpg.create_node(&["Activity"]);
        lpg.set_node_property(id, "rating", Value::Int64((i % 5 + 1) as i64));
        lpg.set_node_property(id, "item_id", Value::Int64((i % num_items) as i64));
        lpg.set_node_property(id, "account_id", Value::Int64((i % num_accounts) as i64));
        activity_ids.push(id);
    }

    let mut account_ids = Vec::with_capacity(num_accounts);
    for i in 0..num_accounts {
        let id = lpg.create_node(&["Account"]);
        lpg.set_node_property(
            id,
            "name",
            Value::String(arcstr::ArcStr::from(account_names[i])),
        );
        account_ids.push(id);
    }

    for i in 0..num_activities {
        lpg.create_edge(activity_ids[i], item_ids[i % num_items], "ACTIVITY_ON");
        lpg.create_edge(
            activity_ids[i],
            account_ids[i % num_accounts],
            "PERFORMED_BY",
        );
    }

    (lpg, compact)
}

// ---------------------------------------------------------------------------
// Benchmark: nodes_by_label scan
// ---------------------------------------------------------------------------

fn bench_nodes_by_label(c: &mut Criterion) {
    let mut group = c.benchmark_group("nodes_by_label");

    for &(ni, na, nac, label) in &[
        (100, 1_000, 50, "100i_1Ka_50ac"),
        (1_000, 10_000, 500, "1Ki_10Ka_500ac"),
        (10_000, 100_000, 5_000, "10Ki_100Ka_5Kac"),
    ] {
        let (lpg, compact) = build_stores(ni, na, nac);

        group.bench_with_input(BenchmarkId::new("lpg", label), &(), |b, ()| {
            b.iter(|| black_box(lpg.nodes_by_label("Activity").len()));
        });

        group.bench_with_input(BenchmarkId::new("compact", label), &(), |b, ()| {
            b.iter(|| black_box(compact.nodes_by_label("Activity").len()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: get_node_property (random access)
// ---------------------------------------------------------------------------

fn bench_get_node_property(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_node_property");

    for &(ni, na, nac, label) in &[
        (1_000, 10_000, 500, "1Ki_10Ka"),
        (10_000, 100_000, 5_000, "10Ki_100Ka"),
    ] {
        let (lpg, compact) = build_stores(ni, na, nac);

        let lpg_ids = lpg.nodes_by_label("Activity");
        let compact_ids = compact.nodes_by_label("Activity");

        let mut state = 12345u64;
        let lookups: Vec<usize> = (0..10_000)
            .map(|_| (lcg(&mut state) as usize) % na)
            .collect();

        let rating_key = PropertyKey::from("rating");

        group.bench_with_input(BenchmarkId::new("lpg", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut sum = 0i64;
                for &i in idxs {
                    if let Some(Value::Int64(v)) = lpg.get_node_property(lpg_ids[i], &rating_key) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("compact", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut sum = 0i64;
                for &i in idxs {
                    if let Some(Value::Int64(v)) =
                        compact.get_node_property(compact_ids[i], &rating_key)
                    {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: edges_from (outgoing traversal)
// ---------------------------------------------------------------------------

fn bench_edges_from_outgoing(c: &mut Criterion) {
    let mut group = c.benchmark_group("edges_from_outgoing");

    for &(ni, na, nac, label) in &[
        (1_000, 10_000, 500, "1Ki_10Ka"),
        (10_000, 100_000, 5_000, "10Ki_100Ka"),
    ] {
        let (lpg, compact) = build_stores(ni, na, nac);

        let lpg_ids = lpg.nodes_by_label("Activity");
        let compact_ids = compact.nodes_by_label("Activity");

        let mut state = 22222u64;
        let lookups: Vec<usize> = (0..10_000)
            .map(|_| (lcg(&mut state) as usize) % na)
            .collect();

        group.bench_with_input(BenchmarkId::new("lpg", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut total = 0usize;
                for &i in idxs {
                    total += lpg.edges_from(lpg_ids[i], Direction::Outgoing).count();
                }
                black_box(total)
            });
        });

        group.bench_with_input(BenchmarkId::new("compact", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut total = 0usize;
                for &i in idxs {
                    total += compact
                        .edges_from(compact_ids[i], Direction::Outgoing)
                        .len();
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: edges_from (incoming traversal)
// ---------------------------------------------------------------------------

fn bench_edges_from_incoming(c: &mut Criterion) {
    let mut group = c.benchmark_group("edges_from_incoming");

    for &(ni, na, nac, label) in &[
        (1_000, 10_000, 500, "1Ki_10Ka"),
        (10_000, 100_000, 5_000, "10Ki_100Ka"),
    ] {
        let (lpg, compact) = build_stores(ni, na, nac);

        let lpg_item_ids = lpg.nodes_by_label("Item");
        let compact_item_ids = compact.nodes_by_label("Item");

        let mut state = 33333u64;
        let lookups: Vec<usize> = (0..10_000)
            .map(|_| (lcg(&mut state) as usize) % ni)
            .collect();

        group.bench_with_input(BenchmarkId::new("lpg", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut total = 0usize;
                for &i in idxs {
                    total += lpg.edges_from(lpg_item_ids[i], Direction::Incoming).count();
                }
                black_box(total)
            });
        });

        group.bench_with_input(BenchmarkId::new("compact", label), &lookups, |b, idxs| {
            b.iter(|| {
                let mut total = 0usize;
                for &i in idxs {
                    total += compact
                        .edges_from(compact_item_ids[i], Direction::Incoming)
                        .len();
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: get_node (full entity materialization)
// ---------------------------------------------------------------------------

fn bench_get_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_node");

    let (lpg, compact) = build_stores(10_000, 100_000, 5_000);

    let lpg_ids = lpg.nodes_by_label("Activity");
    let compact_ids = compact.nodes_by_label("Activity");

    let mut state = 44444u64;
    let lookups: Vec<usize> = (0..10_000)
        .map(|_| (lcg(&mut state) as usize) % 100_000)
        .collect();

    group.bench_with_input(
        BenchmarkId::new("lpg", "100K_activities"),
        &lookups,
        |b, idxs| {
            b.iter(|| {
                let mut count = 0usize;
                for &i in idxs {
                    if lpg.get_node(lpg_ids[i]).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("compact", "100K_activities"),
        &lookups,
        |b, idxs| {
            b.iter(|| {
                let mut count = 0usize;
                for &i in idxs {
                    if compact.get_node(compact_ids[i]).is_some() {
                        count += 1;
                    }
                }
                black_box(count)
            });
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: memory comparison
// ---------------------------------------------------------------------------

fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    for &(ni, na, nac, label) in &[
        (1_000, 10_000, 500, "1Ki_10Ka_500ac"),
        (10_000, 100_000, 5_000, "10Ki_100Ka_5Kac"),
    ] {
        let (lpg, compact) = build_stores(ni, na, nac);

        let (store_mem, index_mem, mvcc_mem, _) = lpg.memory_breakdown();
        let lpg_bytes = store_mem.total_bytes + index_mem.total_bytes + mvcc_mem.total_bytes;
        let compact_bytes = compact.memory_bytes();
        let total_nodes = ni + na + nac;
        let total_edges = na * 2;

        eprintln!("\n=== {label} ({total_nodes} nodes, {total_edges} edges) ===");
        eprintln!(
            "LpgStore:     {lpg_bytes:>12} bytes ({:.1} MB, {:.0} bytes/node)",
            lpg_bytes as f64 / 1_048_576.0,
            lpg_bytes as f64 / total_nodes as f64
        );
        eprintln!("  store:      {:>12} bytes", store_mem.total_bytes);
        eprintln!("  indexes:    {:>12} bytes", index_mem.total_bytes);
        eprintln!("  mvcc:       {:>12} bytes", mvcc_mem.total_bytes);
        eprintln!(
            "CompactStore: {compact_bytes:>12} bytes ({:.1} MB, {:.0} bytes/node)",
            compact_bytes as f64 / 1_048_576.0,
            compact_bytes as f64 / total_nodes as f64
        );

        // Dummy benches to trigger the eprintln output
        group.bench_function(format!("lpg_memory/{label}"), |b| {
            b.iter(|| black_box(lpg_bytes));
        });
        group.bench_function(format!("compact_memory/{label}"), |b| {
            b.iter(|| black_box(compact_bytes));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: statistics / cardinality estimation
// ---------------------------------------------------------------------------

fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    let (lpg, compact) = build_stores(10_000, 100_000, 5_000);

    group.bench_function("lpg_estimate_cardinality", |b| {
        b.iter(|| black_box(lpg.estimate_label_cardinality("Activity")));
    });

    group.bench_function("compact_estimate_cardinality", |b| {
        b.iter(|| black_box(compact.estimate_label_cardinality("Activity")));
    });

    group.bench_function("lpg_statistics", |b| {
        b.iter(|| black_box(lpg.statistics()));
    });

    group.bench_function("compact_statistics", |b| {
        b.iter(|| black_box(compact.statistics()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_nodes_by_label,
    bench_get_node_property,
    bench_edges_from_outgoing,
    bench_edges_from_incoming,
    bench_get_node,
    bench_memory,
    bench_statistics,
);
criterion_main!(benches);
