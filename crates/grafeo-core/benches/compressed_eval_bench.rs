//! Benchmark: property scan performance in CompactStore.
//!
//! Measures the cost of `find_nodes_by_property` and `find_nodes_in_range`
//! using the current decode path vs typed codec-native evaluation.
//!
//! Run: `cargo bench -p grafeo-core --bench compressed_eval_bench --features compact-store`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::Value;
use grafeo_core::graph::compact::CompactStore;
use grafeo_core::graph::compact::builder::CompactStoreBuilder;
use grafeo_core::graph::compact::column::ColumnCodec;
use grafeo_core::graph::traits::GraphStore;
use grafeo_core::storage::bitpack::BitPackedInts;
use grafeo_core::storage::dictionary::DictionaryBuilder;

/// Build a test graph at the given scale.
/// - `n` Item nodes with "score" (4-bit) and "name" (dict, 5 values)
/// - `n * 5` Activity nodes with "rating" (4-bit)
fn build_store(n: usize) -> CompactStore {
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
    let activity_count = n * 5;
    let ratings: Vec<u64> = (0..activity_count).map(|i| ((i % 5) + 1) as u64).collect();

    CompactStoreBuilder::new()
        .node_table("Item", |t| {
            t.column_bitpacked("score", &scores, 4)
                .column_dict("name", &names)
        })
        .node_table("Activity", |t| t.column_bitpacked("rating", &ratings, 4))
        .build()
        .expect("Failed to build test store")
}

// ---------------------------------------------------------------------------
// Benchmark: find_nodes_by_property (equality) on dictionary column
// ---------------------------------------------------------------------------

fn bench_find_by_dict_property(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_nodes_by_property_dict");

    for &n in &[1_000usize, 10_000, 100_000] {
        let store = build_store(n);

        // find_nodes_by_property uses ColumnCodec::find_eq internally.
        group.bench_function(format!("name_eq_gamma/{n}"), |b| {
            b.iter(|| {
                let results = store.find_nodes_by_property("name", &Value::String("gamma".into()));
                black_box(results.len())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: find_nodes_by_property (equality) on bitpacked column
// ---------------------------------------------------------------------------

fn bench_find_by_int_property(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_nodes_by_property_int");

    for &n in &[1_000usize, 10_000, 100_000] {
        let store = build_store(n);

        group.bench_function(format!("score_eq_5/{n}"), |b| {
            b.iter(|| {
                let results = store.find_nodes_by_property("score", &Value::Int64(5));
                black_box(results.len())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: find_nodes_in_range on bitpacked column
// ---------------------------------------------------------------------------

fn bench_find_in_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_nodes_in_range");

    for &n in &[1_000usize, 10_000, 100_000] {
        let store = build_store(n);

        // score > 7 (20% selectivity)
        group.bench_function(format!("score_gt_7/{n}"), |b| {
            b.iter(|| {
                let results =
                    store.find_nodes_in_range("score", Some(&Value::Int64(7)), None, false, false);
                black_box(results.len())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: raw codec-level comparison (isolate the decode cost)
//
// Compares three paths at 100K rows:
//   1. ColumnCodec::get() → Value → compare (current path for other consumers)
//   2. Typed accessor (bp.get() / dict.get()) → native type → compare
//   3. Bulk find_eq / find_in_range (new methods)
// ---------------------------------------------------------------------------

fn bench_codec_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_paths");

    let n = 100_000usize;

    // --- BitPacked column ---
    let values: Vec<u64> = (0..n).map(|i| (i % 10) as u64).collect();
    let bp = BitPackedInts::pack_with_bits(&values, 4);
    let col = ColumnCodec::BitPacked(bp.clone());

    group.bench_function("bitpacked/via_value/100K", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for i in 0..n {
                if let Some(Value::Int64(v)) = col.get(i) {
                    if v == 5 {
                        count += 1;
                    }
                }
            }
            black_box(count)
        });
    });

    group.bench_function("bitpacked/find_eq/100K", |b| {
        b.iter(|| black_box(col.find_eq(&Value::Int64(5)).len()))
    });

    // --- Dictionary column ---
    let labels = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let string_values: Vec<&str> = (0..n).map(|i| labels[i % 5]).collect();
    let mut builder = DictionaryBuilder::new();
    for &s in &string_values {
        builder.add(s);
    }
    let dict = builder.build();
    let dict_col = ColumnCodec::Dict(dict);

    group.bench_function("dict/via_value/100K", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for i in 0..n {
                if let Some(Value::String(ref s)) = dict_col.get(i) {
                    if s.as_str() == "gamma" {
                        count += 1;
                    }
                }
            }
            black_box(count)
        });
    });

    group.bench_function("dict/find_eq/100K", |b| {
        let target = Value::String("gamma".into());
        b.iter(|| black_box(dict_col.find_eq(&target).len()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_find_by_dict_property,
    bench_find_by_int_property,
    bench_find_in_range,
    bench_codec_paths,
);
criterion_main!(benches);
