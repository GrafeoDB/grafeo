//! Benchmarks for storage primitive improvements.
//!
//! Validates that DictionaryEncoding serialization, RunLengthEncoding binary
//! search get(), and DeltaBitPacked random access do not regress performance
//! on existing operations while improving the new ones.
//!
//! Run: cargo bench -p grafeo-core --bench storage_primitives_bench

use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grafeo_core::storage::{
    BitPackedInts, DeltaBitPacked, DictionaryEncoding, RunLengthEncoding,
    DictionaryBuilder,
};

// ---------------------------------------------------------------------------
// DictionaryEncoding: serialization round-trip + lookup (no regression)
// ---------------------------------------------------------------------------

fn bench_dictionary(c: &mut Criterion) {
    let mut group = c.benchmark_group("dictionary");

    for &(n, unique, label) in &[
        (1_000usize, 10usize, "1K_vals_10_unique"),
        (100_000, 100, "100K_vals_100_unique"),
        (1_000_000, 1000, "1M_vals_1K_unique"),
    ] {
        // Build dictionary
        let strings: Vec<String> = (0..unique).map(|i| format!("string_{i:05}")).collect();
        let mut builder = DictionaryBuilder::new();
        for i in 0..n {
            builder.add(&strings[i % unique]);
        }
        let dict = builder.build();

        // Benchmark: lookup (existing operation — must not regress)
        group.bench_with_input(
            BenchmarkId::new("lookup_10K", label),
            &dict,
            |b, dict| {
                b.iter(|| {
                    let mut count = 0usize;
                    let step = (dict.len() / 10_000).max(1);
                    for i in (0..dict.len()).step_by(step).take(10_000) {
                        if dict.get(i).is_some() {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );

        // Benchmark: to_bytes (NEW)
        group.bench_with_input(
            BenchmarkId::new("to_bytes", label),
            &dict,
            |b, dict| {
                b.iter(|| {
                    black_box(dict.to_bytes())
                });
            },
        );

        // Benchmark: from_bytes (NEW)
        let bytes = dict.to_bytes();
        group.bench_with_input(
            BenchmarkId::new("from_bytes", label),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    black_box(DictionaryEncoding::from_bytes(bytes).unwrap())
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RunLengthEncoding: get() random access (was O(n), now O(log n))
// ---------------------------------------------------------------------------

fn bench_rle_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("rle_get");

    for &(num_runs, avg_run_len, label) in &[
        (100usize, 10usize, "100_runs_len10"),
        (1_000, 10, "1K_runs_len10"),
        (10_000, 10, "10K_runs_len10"),
        (100_000, 5, "100K_runs_len5"),
    ] {
        // Build RLE with known structure
        let mut values = Vec::new();
        for i in 0..num_runs as u64 {
            for _ in 0..avg_run_len {
                values.push(i);
            }
        }
        let rle = RunLengthEncoding::encode(&values);
        let total = rle.total_count();

        // Generate 10K random lookup indices
        let mut state: u64 = 42;
        let indices: Vec<usize> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as usize) % total
            })
            .collect();

        // Benchmark: random access get() (was O(n), now O(log n))
        group.bench_with_input(
            BenchmarkId::new("random_access_10K", label),
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

        // Benchmark: sequential decode (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("decode", label),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(rle.decode())
                });
            },
        );

        // Benchmark: iterator (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("iter_sum", label),
            &(),
            |b, _| {
                b.iter(|| {
                    let sum: u64 = rle.iter().sum();
                    black_box(sum)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// DeltaBitPacked: get() random access (NEW) vs decode (existing)
// ---------------------------------------------------------------------------

fn bench_delta_bitpacked(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_bitpacked");

    for &(n, label) in &[
        (100usize, "100_values"),
        (1_000, "1K_values"),
        (10_000, "10K_values"),
        (100_000, "100K_values"),
    ] {
        let values: Vec<u64> = (0..n as u64).map(|i| i * 100 + 50).collect();
        let dbp = DeltaBitPacked::encode(&values);

        // Generate 10K random lookup indices
        let mut state: u64 = 99;
        let indices: Vec<usize> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as usize) % n
            })
            .collect();

        // Benchmark: random access get() (NEW — using skip index)
        group.bench_with_input(
            BenchmarkId::new("get_random_10K", label),
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

        // Benchmark: full decode (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("decode", label),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(dbp.decode())
                });
            },
        );

        // Benchmark: serialization round-trip (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("serialize_roundtrip", label),
            &(),
            |b, _| {
                b.iter(|| {
                    let bytes = dbp.to_bytes();
                    black_box(DeltaBitPacked::from_bytes(&bytes).unwrap())
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// BitPackedInts: verify no regression on existing get() and unpack()
// ---------------------------------------------------------------------------

fn bench_bitpacked_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitpacked_baseline");

    for &(n, bits, label) in &[
        (10_000usize, 4u8, "10K_4bit"),
        (100_000, 21, "100K_21bit"),
        (1_000_000, 4, "1M_4bit"),
    ] {
        let max_val = 1u64 << bits;
        let mut state: u64 = 12345;
        let values: Vec<u64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state % max_val
            })
            .collect();
        let bp = BitPackedInts::pack_with_bits(&values, bits);

        let mut state2: u64 = 67890;
        let indices: Vec<usize> = (0..10_000)
            .map(|_| {
                state2 = state2.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state2 as usize) % n
            })
            .collect();

        // Benchmark: random access get() (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("get_random_10K", label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum = sum.wrapping_add(bp.get(idx).unwrap_or(0));
                    }
                    black_box(sum)
                });
            },
        );

        // Benchmark: full unpack (existing — must not regress)
        group.bench_with_input(
            BenchmarkId::new("unpack", label),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(bp.unpack())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dictionary,
    bench_rle_get,
    bench_delta_bitpacked,
    bench_bitpacked_baseline,
);
criterion_main!(benches);
