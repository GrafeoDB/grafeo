//! Benchmarks for TopKOperator vs LimitOperator(SortOperator(...)).
//!
//! Confirms the spec claim that TopK is strictly faster than the unfused
//! composition at every N tested. The asserted O(k) memory bound lives in
//! the unit test (`top_k_skips_materialization_for_losers`) — `#[cfg(test)]`-
//! gated counters aren't visible from criterion bench files (the dep crate
//! is built without `cfg(test)`).
//!
//! Uses `codspeed-criterion-compat` so the same file runs under
//! plain `cargo bench` and under `cargo codspeed`.
#![allow(missing_docs)]
#![allow(clippy::cast_possible_truncation)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::LogicalType;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::chunk::DataChunkBuilder;
use grafeo_core::execution::operators::{
    LimitOperator, Operator, OperatorResult, SortKey, SortOperator, TopKOperator,
};

const K: usize = 50;

struct VecSource {
    chunks: Vec<DataChunk>,
    pos: usize,
}

impl VecSource {
    fn new(values: &[i64], chunk_size: usize) -> Self {
        let mut chunks = Vec::new();
        for window in values.chunks(chunk_size) {
            let mut b = DataChunkBuilder::new(&[LogicalType::Int64]);
            for &v in window {
                b.column_mut(0).unwrap().push_int64(v);
                b.advance_row();
            }
            chunks.push(b.finish());
        }
        Self { chunks, pos: 0 }
    }
}

impl Operator for VecSource {
    fn next(&mut self) -> OperatorResult {
        if self.pos < self.chunks.len() {
            let c = std::mem::replace(&mut self.chunks[self.pos], DataChunk::empty());
            self.pos += 1;
            Ok(Some(c))
        } else {
            Ok(None)
        }
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn name(&self) -> &'static str {
        "VecSource"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

fn pseudo_random_values(n: usize) -> Vec<i64> {
    (0..n).map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 1_000_000_007) as i64).collect()
}

fn drain<O: Operator + ?Sized>(op: &mut O) -> usize {
    let mut count = 0;
    while let Some(c) = op.next().unwrap() {
        count += c.row_count();
    }
    count
}

fn bench_top_k_vs_sort_limit_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_vs_sort_limit_time");
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let values = pseudo_random_values(n);

        group.bench_function(format!("top_k/N={n}"), |b| {
            b.iter(|| {
                let source = Box::new(VecSource::new(&values, 2048));
                let mut op = TopKOperator::new(
                    source,
                    vec![SortKey::descending(0)],
                    K,
                    vec![LogicalType::Int64],
                );
                black_box(drain(&mut op))
            });
        });

        group.bench_function(format!("sort_limit/N={n}"), |b| {
            b.iter(|| {
                let source = Box::new(VecSource::new(&values, 2048));
                let sort = Box::new(SortOperator::new(
                    source,
                    vec![SortKey::descending(0)],
                    vec![LogicalType::Int64],
                ));
                let mut limit = LimitOperator::new(sort, K, vec![LogicalType::Int64]);
                black_box(drain(&mut limit))
            });
        });
    }
    group.finish();
}

fn bench_top_k_drain(c: &mut Criterion) {
    // Times TopK build-then-drain end-to-end so CodSpeed can detect
    // regressions in the build/eviction loop. The asserted O(k) memory
    // bound lives in Task 4's unit test.
    let mut group = c.benchmark_group("top_k_drain");
    group.sample_size(10);
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let values = pseudo_random_values(n);
        group.bench_function(format!("N={n}"), |b| {
            b.iter(|| {
                let source = Box::new(VecSource::new(&values, 2048));
                let mut op = TopKOperator::new(
                    source,
                    vec![SortKey::descending(0)],
                    K,
                    vec![LogicalType::Int64],
                );
                black_box(drain(&mut op))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_top_k_vs_sort_limit_time, bench_top_k_drain);
criterion_main!(benches);
