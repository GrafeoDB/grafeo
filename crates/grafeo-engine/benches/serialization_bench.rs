//! Serialization benchmarks for snapshot export/import and Value encoding.
//!
//! Covers the bincode hot paths used by persistence, WAL, and spill-to-disk.
//!
//! Run with: cargo bench -p grafeo-engine --bench serialization_bench
// Bench values are small known constants
#![allow(clippy::cast_possible_wrap)]
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Build a small graph (~50 nodes, ~100 edges) representative of typical workloads.
fn build_bench_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for i in 0..50u64 {
        let n = db.create_node(&["Person"]).unwrap();
        db.set_node_property(n, "name", Value::String(format!("User{i}").into()))
            .unwrap();
        db.set_node_property(n, "age", Value::Int64(20 + (i % 50) as i64))
            .unwrap();
        db.set_node_property(
            n,
            "bio",
            Value::String("A short biography for benchmarking serialization throughput.".into()),
        )
        .unwrap();
    }
    for i in 0..100u64 {
        let src = grafeo_common::types::NodeId::new(i % 50);
        let dst = grafeo_common::types::NodeId::new((i * 7 + 13) % 50);
        let e = db.create_edge(src, dst, "KNOWS").unwrap();
        db.set_edge_property(e, "weight", Value::Float64(i as f64 * 0.1))
            .unwrap();
    }
    db
}

// ---------------------------------------------------------------------------
// Snapshot export / import
// ---------------------------------------------------------------------------

fn bench_snapshot_export(c: &mut Criterion) {
    let db = build_bench_db();
    c.bench_function("snapshot_export_50n_100e", |b| {
        b.iter(|| black_box(db.export_snapshot().unwrap()));
    });
}

fn bench_snapshot_import(c: &mut Criterion) {
    let db = build_bench_db();
    let bytes = db.export_snapshot().unwrap();
    c.bench_function("snapshot_import_50n_100e", |b| {
        b.iter(|| black_box(GrafeoDB::import_snapshot(&bytes).unwrap()));
    });
}

fn bench_snapshot_roundtrip(c: &mut Criterion) {
    let db = build_bench_db();
    c.bench_function("snapshot_roundtrip_50n_100e", |b| {
        b.iter(|| {
            let bytes = db.export_snapshot().unwrap();
            black_box(GrafeoDB::import_snapshot(&bytes).unwrap());
        });
    });
}

// ---------------------------------------------------------------------------
// Value encoding / decoding (bincode hot path)
// ---------------------------------------------------------------------------

fn bench_value_encode(c: &mut Criterion) {
    let values: Vec<Value> = vec![
        Value::Int64(42),
        Value::Float64(9.81),
        Value::String("hello world".into()),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into()),
    ];

    c.bench_function("value_encode_mixed_6", |b| {
        b.iter(|| {
            for v in &values {
                black_box(bincode::serde::encode_to_vec(v, bincode::config::standard()).unwrap());
            }
        });
    });
}

fn bench_value_decode(c: &mut Criterion) {
    let values: Vec<Value> = vec![
        Value::Int64(42),
        Value::Float64(9.81),
        Value::String("hello world".into()),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into()),
    ];

    let encoded: Vec<Vec<u8>> = values
        .iter()
        .map(|v| bincode::serde::encode_to_vec(v, bincode::config::standard()).unwrap())
        .collect();

    c.bench_function("value_decode_mixed_6", |b| {
        b.iter(|| {
            for bytes in &encoded {
                let (v, _): (Value, _) =
                    bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
                black_box(v);
            }
        });
    });
}

criterion_group!(
    serialization,
    bench_snapshot_export,
    bench_snapshot_import,
    bench_snapshot_roundtrip,
    bench_value_encode,
    bench_value_decode,
);
criterion_main!(serialization);
