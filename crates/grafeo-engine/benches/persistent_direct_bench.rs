//! Persistent direct-write costs for LPG commit staging.
//!
//! Each sample times 64 creations or property updates with Sync or NoSync WAL.
//! Database open, WAL initialization, close and drop are outside measurement.
//!
//! Run: cargo bench -p grafeo-engine --bench persistent_direct_bench
#![allow(
    missing_docs,
    reason = "criterion_group! generates undocumented wrapper functions"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_engine::GrafeoDB;

#[cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]
fn bench_persistent_direct_writes(c: &mut Criterion) {
    use criterion::{BatchSize, Throughput};
    use grafeo_common::types::Value;
    use grafeo_engine::Config;
    use grafeo_engine::config::{DurabilityMode, StorageFormat};

    const WRITES: u64 = 64;
    let mut group = c.benchmark_group("persistent_direct");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(WRITES));

    for (name, durability) in [
        ("nosync", DurabilityMode::NoSync),
        ("sync", DurabilityMode::Sync),
    ] {
        let open = || {
            let dir = tempfile::tempdir().unwrap();
            let db = GrafeoDB::with_config(
                Config::persistent(dir.path().join("db.grafeo"))
                    .with_storage_format(StorageFormat::Auto)
                    .with_wal_durability(durability),
            )
            .unwrap();
            // Open the WAL and initialize the label outside the timed writes.
            let node = db.create_node(&["DirectProbe"]).unwrap();
            (db, dir, node)
        };

        group.bench_function(format!("create_64_{name}"), |b| {
            b.iter_batched(
                open,
                |(db, dir, _)| {
                    for _ in 0..WRITES {
                        black_box(db.create_node(&["DirectProbe"]).unwrap());
                    }
                    // Return the database so close/checkpoint/drop stay outside
                    // the timer. Tuple order closes it before deleting the path.
                    (db, dir)
                },
                BatchSize::PerIteration,
            );
        });
        group.bench_function(format!("set_property_64_{name}"), |b| {
            b.iter_batched(
                open,
                |(db, dir, node)| {
                    for revision in 0..WRITES {
                        db.set_node_property(
                            node,
                            "revision",
                            Value::Int64(i64::try_from(black_box(revision)).unwrap()),
                        )
                        .unwrap();
                    }
                    (db, dir)
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(persistent_direct_benches, bench_persistent_direct_writes);
criterion_main!(persistent_direct_benches);
