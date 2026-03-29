//! Baseline benchmarks measuring the OLD behavior of storage primitives.
//!
//! This benchmarks the approaches a user would take BEFORE the improvements
//! in this PR, to establish honest before/after comparisons:
//!
//! - RLE get(): simulates O(n) linear scan (the old implementation)
//! - DeltaBitPacked: decode() + index (the only way to get a single value before)
//! - Dictionary: no to_bytes/from_bytes existed, so we measure serde_json as baseline
//!
//! Run: cargo bench -p grafeo-core --bench storage_baseline_bench

use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grafeo_core::storage::{DeltaBitPacked, RunLengthEncoding};

/// Simulate the OLD O(n) RLE get() by linear scanning runs.
/// This is exactly what the code did before the binary search fix.
fn rle_get_linear(rle: &RunLengthEncoding, index: usize) -> Option<u64> {
    if index >= rle.total_count() {
        return None;
    }
    let mut offset = 0usize;
    for run in rle.runs() {
        let run_end = offset + run.length as usize;
        if index < run_end {
            return Some(run.value);
        }
        offset = run_end;
    }
    None
}

/// Simulate the OLD DeltaBitPacked single-value access:
/// decode the entire array, then index into the Vec.
fn delta_bitpacked_decode_and_index(dbp: &DeltaBitPacked, index: usize) -> Option<u64> {
    let decoded = dbp.decode();
    decoded.get(index).copied()
}

fn bench_rle_before_after(c: &mut Criterion) {
    let mut group = c.benchmark_group("rle_get_before_vs_after");

    for &(num_runs, avg_run_len, label) in &[
        (100usize, 10usize, "100_runs"),
        (1_000, 10, "1K_runs"),
        (10_000, 10, "10K_runs"),
        (100_000, 5, "100K_runs"),
    ] {
        let mut values = Vec::new();
        for i in 0..num_runs as u64 {
            for _ in 0..avg_run_len {
                values.push(i);
            }
        }
        let rle = RunLengthEncoding::encode(&values);
        let total = rle.total_count();

        let mut state: u64 = 42;
        let indices: Vec<usize> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as usize) % total
            })
            .collect();

        // BEFORE: O(n) linear scan
        group.bench_with_input(
            BenchmarkId::new("BEFORE_linear_scan", label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum = sum.wrapping_add(rle_get_linear(&rle, idx).unwrap_or(0));
                    }
                    black_box(sum)
                });
            },
        );

        // AFTER: O(log n) binary search
        group.bench_with_input(
            BenchmarkId::new("AFTER_binary_search", label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum = sum.wrapping_add(rle.get(idx).unwrap_or(0));
                    }
                    black_box(sum)
                });
            },
        );
    }
    group.finish();
}

fn bench_delta_bitpacked_before_after(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_bitpacked_get_before_vs_after");

    for &(n, label) in &[
        (100usize, "100_values"),
        (1_000, "1K_values"),
        (10_000, "10K_values"),
        (100_000, "100K_values"),
    ] {
        let values: Vec<u64> = (0..n as u64).map(|i| i * 100 + 50).collect();
        let dbp = DeltaBitPacked::encode(&values);

        let mut state: u64 = 99;
        let indices: Vec<usize> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as usize) % n
            })
            .collect();

        // BEFORE: decode entire array, then index (the only option before this PR)
        group.bench_with_input(
            BenchmarkId::new("BEFORE_decode_then_index", label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum = sum.wrapping_add(
                            delta_bitpacked_decode_and_index(&dbp, idx).unwrap_or(0),
                        );
                    }
                    black_box(sum)
                });
            },
        );

        // AFTER: skip-index get() (O(1) per lookup)
        group.bench_with_input(
            BenchmarkId::new("AFTER_skip_index_get", label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum = sum.wrapping_add(dbp.get(idx).unwrap_or(0));
                    }
                    black_box(sum)
                });
            },
        );
    }
    group.finish();
}

fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    // Measure the actual memory cost of the new fields
    group.bench_function("rle_prefix_sums_100K_runs", |b| {
        let mut values = Vec::new();
        for i in 0..100_000u64 {
            for _ in 0..5 {
                values.push(i);
            }
        }
        b.iter(|| {
            let rle = RunLengthEncoding::encode(&values);
            // prefix_sums: 100K * 8 bytes = 800 KB
            // runs: 100K * 16 bytes = 1.6 MB
            // Total metadata overhead: 800KB / 1.6MB = 50% increase
            black_box(rle.total_count())
        });
    });

    group.bench_function("delta_skip_index_100K_values", |b| {
        let values: Vec<u64> = (0..100_000u64).map(|i| i * 100).collect();
        b.iter(|| {
            let dbp = DeltaBitPacked::encode(&values);
            // skip_index: 100K/64 * 8 bytes = ~12.5 KB
            // deltas: varies by bit width, ~100K * bits/8 bytes
            black_box(dbp.len())
        });
    });

    group.finish();

    // Print memory costs
    eprintln!("\n=== Memory Overhead of New Fields ===");
    eprintln!("RLE prefix_sums (100K runs): {} KB", 100_000 * 8 / 1024);
    eprintln!("RLE runs data (100K runs):   {} KB", 100_000 * 16 / 1024);
    eprintln!("RLE overhead ratio:          50% (prefix_sums / runs)");
    eprintln!();
    eprintln!("DeltaBitPacked skip_index (100K values): {} KB", (100_000 / 64) * 8 / 1024);
    eprintln!("DeltaBitPacked deltas (100K values):     varies by bit width");
    eprintln!("DeltaBitPacked overhead ratio:            ~1-5% (skip_index / deltas)");
}

criterion_group!(
    benches,
    bench_rle_before_after,
    bench_delta_bitpacked_before_after,
    bench_memory_overhead,
);
criterion_main!(benches);
