//! The memory of grouped queries (#674): a grouping that spills and reloads
//! asks the buffer manager about its consumers a bounded number of times,
//! whatever its input, and a grouping refused at the query memory limit is
//! refused again from its cached plan, each time holding nothing after.
//!
//! The scenarios come from #649. The core's grouping cases (a deadline after a
//! spill, a cut spill file, empty SPARQL sums and averages through a spill)
//! are in `grafeo-core`'s `push/aggregate` tests.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main grouped_query_memory::
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, StorageTier};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

/// A consumer that holds nothing and counts how often the buffer manager asks
/// what it holds.
struct ScanCounter(AtomicUsize);

impl MemoryConsumer for ScanCounter {
    fn name(&self) -> &str {
        "grouped-query-scan-counter"
    }

    fn memory_usage(&self) -> usize {
        self.0.fetch_add(1, Ordering::Relaxed);
        0
    }

    fn eviction_priority(&self) -> u8 {
        0
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::ExecutionBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    fn current_tier(&self) -> StorageTier {
        StorageTier::Uninitialized
    }
}

/// A grouping of 1,024 groups under pressure spills (a spill path that is a
/// file fails it) and reloads; how often the buffer manager reads its
/// consumers does not grow with the input, from 4 rows a group to 16. The
/// answer is whole, and the grouping holds nothing and leaves no file after.
#[cfg(feature = "spill")]
#[test]
fn a_spilling_grouping_reads_the_consumers_a_bounded_number_of_times() {
    const GROUPS: usize = 1024;
    const BUDGET: usize = 32 * MIB;
    let mut scans = Vec::new();
    for rows_per_group in [4_i64, 16] {
        let root = tempfile::tempdir().unwrap();
        let spill_path = root.path().join("spill");
        let db = GrafeoDB::with_config(
            Config::in_memory()
                .with_memory_limit(BUDGET)
                .with_spill_path(&spill_path)
                .without_query_timeout(),
        )
        .unwrap();
        let query = format!(
            "UNWIND range(0, {}) AS i RETURN i % {GROUPS} AS bucket, sum(i) AS total, count(*) AS c",
            rows_per_group * i64::try_from(GROUPS).unwrap() - 1
        );
        let manager = db.buffer_manager();
        let baseline = manager.allocated();
        let held = manager
            .try_allocate(BUDGET * 9 / 10 - baseline, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let held_total = manager.allocated();
        let session = db.session();

        // A spill path that is a file: the grouping fails as it spills.
        std::fs::write(&spill_path, b"not a directory").unwrap();
        assert!(
            session.execute(&query).is_err(),
            "the grouping spills under pressure"
        );
        assert_eq!(manager.allocated(), held_total);
        std::fs::remove_file(&spill_path).unwrap();

        let counter = Arc::new(ScanCounter(AtomicUsize::new(0)));
        manager.register_consumer(Arc::clone(&counter) as Arc<dyn MemoryConsumer>);
        let result = session.execute(&query).unwrap();
        scans.push(counter.0.load(Ordering::Relaxed));

        assert_eq!(result.rows().len(), GROUPS);
        let mut seen = vec![false; GROUPS];
        let groups = i64::try_from(GROUPS).unwrap();
        for row in result.rows() {
            let [
                Value::Int64(bucket),
                Value::Int64(total),
                Value::Int64(count),
            ] = row.as_slice()
            else {
                panic!("a grouped row: {row:?}");
            };
            let index = usize::try_from(*bucket).unwrap();
            assert!(!seen[index], "bucket {bucket} twice");
            seen[index] = true;
            // The bucket's rows are bucket + k * GROUPS for k below rows_per_group.
            let expected =
                rows_per_group * bucket + groups * rows_per_group * (rows_per_group - 1) / 2;
            assert_eq!(
                (*total, *count),
                (expected, rows_per_group),
                "bucket {bucket}"
            );
        }
        assert!(seen.into_iter().all(|present| present));
        drop(result);
        assert_eq!(
            manager.allocated(),
            held_total,
            "the grouping holds nothing after"
        );
        assert!(
            std::fs::read_dir(&spill_path).map_or(true, |mut entries| entries.next().is_none()),
            "and leaves no spill file"
        );
        drop(held);
        assert_eq!(manager.allocated(), baseline);
    }
    assert_eq!(
        scans[0], scans[1],
        "consumer reads for 4 and 16 rows a group: {scans:?}"
    );
}

/// `result` is the S001 refusal naming the query memory limit.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, grafeo_common::utils::error::Error>) {
    let error = result.expect_err("refused at the query memory limit");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(
        text.contains("query memory limit"),
        "names the bound: {text}"
    );
}

/// A grouping refused at the query memory limit is refused the same way from
/// its cached plan, each time giving back all it held; the session answers
/// the next statement, and the same grouping under a higher limit answers.
#[test]
fn a_refused_grouping_is_refused_again_from_its_cached_plan_and_holds_nothing() {
    let query = "UNWIND range(1, 200000) AS i RETURN i % 100000 AS k, count(*) AS n";
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_limit(0),
    )
    .unwrap();
    let execution = |db: &GrafeoDB| db.memory_usage().buffer_manager.execution_buffers_bytes;
    let before = execution(&db);
    let session = db.session();

    // A cached plan's hit: the session's own entry or the shared cache's.
    let hits = |db: &GrafeoDB| {
        let stats = db.query_cache().stats();
        stats.optimized_hits + stats.session_hits
    };
    assert_refused(session.execute(query));
    assert_eq!(
        execution(&db),
        before,
        "a cold plan's refusal holds nothing"
    );
    let cached = hits(&db);
    assert_refused(session.execute(query));
    assert!(
        hits(&db) > cached,
        "the second run takes the cached plan: {:?}",
        db.query_cache().stats()
    );
    assert_eq!(
        execution(&db),
        before,
        "a cached plan's refusal holds nothing"
    );

    let rows = session
        .execute("UNWIND range(1, 1000) AS i RETURN i % 10 AS k, count(*) AS n")
        .unwrap();
    assert_eq!(rows.rows().len(), 10);
    assert_eq!(execution(&db), before);

    let higher = GrafeoDB::with_config(
        Config::in_memory()
            .with_query_memory_limit(512 * MIB)
            .with_spill_limit(0),
    )
    .unwrap();
    assert_eq!(higher.execute(query).unwrap().rows().len(), 100_000);
}
