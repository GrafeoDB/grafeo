//! Memory usage benchmarks for regression detection.
//!
//! Each benchmark creates a workload, measures creation time via Criterion,
//! then records the absolute memory footprint to a JSON snapshot file at
//! `target/criterion/memory_snapshot.json`. The CI comparison script reads
// Bench indices are small known values
#![allow(clippy::cast_possible_wrap)]
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]
//! that file and checks against absolute bounds in `bench-thresholds.toml`.
//!
//! Run with: cargo bench --all-features --bench memory_bench

use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Snapshot recording
// ============================================================================

/// Collects memory snapshots from benchmarks and writes them to JSON on drop.
struct SnapshotRecorder {
    entries: Vec<(String, usize)>,
}

impl SnapshotRecorder {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn record(&mut self, name: &str, bytes: usize) {
        self.entries.push((name.to_string(), bytes));
    }

    fn write_json(&self) {
        let dir = PathBuf::from("target/criterion");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("memory_snapshot.json");
        let mut file = std::fs::File::create(&path).expect("failed to create memory_snapshot.json");
        write!(file, "{{").unwrap();
        for (i, (name, bytes)) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(file, ",").unwrap();
            }
            write!(file, "\n  \"{name}\": {bytes}").unwrap();
        }
        writeln!(file, "\n}}").unwrap();
        eprintln!("Memory snapshot written to {}", path.display());
    }
}

static RECORDER: Mutex<Option<SnapshotRecorder>> = Mutex::new(None);

fn recorder_record(name: &str, bytes: usize) {
    let mut guard = RECORDER.lock().unwrap();
    let recorder = guard.get_or_insert_with(SnapshotRecorder::new);
    recorder.record(name, bytes);
}

fn recorder_flush() {
    let guard = RECORDER.lock().unwrap();
    if let Some(ref recorder) = *guard {
        recorder.write_json();
    }
}

// ============================================================================
// Setup helpers
// ============================================================================

/// Creates a social graph using the direct CRUD API for speed.
///
/// Using the query API (MATCH + CREATE) for edges is O(n) per edge due to
/// property-based node lookup, making large graphs prohibitively slow.
/// The CRUD API bypasses parsing and planning entirely.
fn setup_social_graph(node_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // Create nodes via CRUD API, collecting their IDs
    let mut node_ids = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "id", Value::Int64(i as i64))
            .unwrap();
        db.set_node_property(id, "name", Value::String(format!("User{i}").into()))
            .unwrap();
        db.set_node_property(id, "age", Value::Int64((20 + (i % 50)) as i64))
            .unwrap();
        node_ids.push(id);
    }

    // Create edges via CRUD API using collected IDs (O(1) per edge)
    let edge_count = node_count * edge_multiplier;
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 7 + 13) % node_count;
        if src != dst {
            db.create_edge(node_ids[src], node_ids[dst], "KNOWS")
                .unwrap();
        }
    }

    db
}

// ============================================================================
// Memory benchmarks
// ============================================================================

fn bench_memory_empty(c: &mut Criterion) {
    c.bench_function("memory_empty_db", |b| {
        b.iter(|| {
            let db = GrafeoDB::new_in_memory();
            black_box(db.memory_usage().total_bytes)
        });
    });

    let db = GrafeoDB::new_in_memory();
    recorder_record("memory_empty_db", db.memory_usage().total_bytes);
}

fn bench_memory_1k(c: &mut Criterion) {
    c.bench_function("memory_1k_nodes_5k_edges", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let db = setup_social_graph(1_000, 5);
                black_box(db.memory_usage().total_bytes);
            }
            start.elapsed()
        });
    });

    let db = setup_social_graph(1_000, 5);
    recorder_record("memory_1k_nodes_5k_edges", db.memory_usage().total_bytes);
}

fn bench_memory_10k(c: &mut Criterion) {
    c.bench_function("memory_10k_nodes_50k_edges", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let db = setup_social_graph(10_000, 5);
                black_box(db.memory_usage().total_bytes);
            }
            start.elapsed()
        });
    });

    let db = setup_social_graph(10_000, 5);
    recorder_record("memory_10k_nodes_50k_edges", db.memory_usage().total_bytes);
}

fn bench_memory_after_queries(c: &mut Criterion) {
    c.bench_function("memory_after_100_queries", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let db = setup_social_graph(1_000, 5);
                let session = db.session();
                for i in 0..100 {
                    let query = format!("MATCH (n:Person {{id: {}}}) RETURN n.name", i);
                    let _ = session.execute(&query);
                }
                black_box(db.memory_usage().total_bytes);
            }
            start.elapsed()
        });
    });

    let db = setup_social_graph(1_000, 5);
    let session = db.session();
    for i in 0..100 {
        let query = format!("MATCH (n:Person {{id: {}}}) RETURN n.name", i);
        let _ = session.execute(&query);
    }
    recorder_record("memory_after_100_queries", db.memory_usage().total_bytes);
}

#[cfg(feature = "vector-index")]
fn bench_memory_vector_index(c: &mut Criterion) {
    c.bench_function("memory_vector_index_1k", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let db = GrafeoDB::new_in_memory();
                let session = db.session();

                // Create nodes with 128-dim vector embeddings
                for i in 0..1_000 {
                    let dims: Vec<String> = (0..128)
                        .map(|d| format!("{:.4}", (i * 128 + d) as f64 / 128_000.0))
                        .collect();
                    let vec_str = dims.join(", ");
                    let query = format!("INSERT (:Item {{id: {}, embedding: [{vec_str}]}})", i);
                    session.execute(&query).unwrap();
                }
                db.create_vector_index(
                    "Item",
                    "embedding",
                    Some(128),
                    Some("cosine"),
                    None,
                    None,
                    None,
                )
                .unwrap();
                black_box(db.memory_usage().total_bytes);
            }
            start.elapsed()
        });
    });

    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..1_000 {
        let dims: Vec<String> = (0..128)
            .map(|d| format!("{:.4}", (i * 128 + d) as f64 / 128_000.0))
            .collect();
        let vec_str = dims.join(", ");
        let query = format!("INSERT (:Item {{id: {}, embedding: [{vec_str}]}})", i);
        session.execute(&query).unwrap();
    }
    db.create_vector_index(
        "Item",
        "embedding",
        Some(128),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();
    recorder_record("memory_vector_index_1k", db.memory_usage().total_bytes);
}

// ============================================================================
// Flush and main
// ============================================================================

fn flush_snapshots(c: &mut Criterion) {
    // Dummy benchmark that just flushes the recorder.
    // Criterion requires at least one bench_function call per group function.
    c.bench_function("_memory_flush", |b| {
        b.iter(|| black_box(0));
    });
    recorder_flush();
}

// Memory benchmarks create entire databases per iteration, so we use a
// small sample size and short measurement time to keep CI under 5 minutes.
#[cfg(feature = "vector-index")]
criterion_group! {
    name = memory_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets =
        bench_memory_empty,
        bench_memory_1k,
        bench_memory_10k,
        bench_memory_after_queries,
        bench_memory_vector_index,
        flush_snapshots,
}

#[cfg(not(feature = "vector-index"))]
criterion_group! {
    name = memory_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(1));
    targets =
        bench_memory_empty,
        bench_memory_1k,
        bench_memory_10k,
        bench_memory_after_queries,
        flush_snapshots,
}

criterion_main!(memory_benches);
