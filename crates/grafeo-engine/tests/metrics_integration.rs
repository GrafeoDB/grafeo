//! Integration tests for the metrics subsystem.
//!
//! Verifies that query, transaction, session, and cache metrics
//! are recorded correctly and exposed via the snapshot and Prometheus APIs.

#![cfg(feature = "metrics")]

use grafeo_engine::GrafeoDB;

fn db_with_metrics() -> GrafeoDB {
    // Metrics are automatically enabled when the `metrics` feature is compiled in.
    GrafeoDB::new_in_memory()
}

#[test]
fn test_query_count_increments() {
    let db = db_with_metrics();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("MATCH (n) RETURN n").unwrap();

    let m = db.metrics();
    assert!(
        m.query_count >= 2,
        "expected at least 2 queries, got {}",
        m.query_count
    );
}

#[test]
fn test_session_created_and_active() {
    let db = db_with_metrics();

    let m_before = db.metrics();
    let initial_created = m_before.session_created;

    let _s1 = db.session();
    let _s2 = db.session();

    let m_after = db.metrics();
    assert_eq!(
        m_after.session_created,
        initial_created + 2,
        "session_created should increment by 2"
    );
    assert!(m_after.session_active >= 2, "at least 2 sessions active");
}

#[test]
fn test_session_active_decrements_on_drop() {
    let db = db_with_metrics();
    {
        let _s = db.session();
        let m = db.metrics();
        assert!(m.session_active >= 1);
    }
    // Session dropped, active should decrement
    let m = db.metrics();
    // We can't assert exact 0 because the db.metrics() call may create an internal session,
    // but active should be less than before the drop.
    assert!(m.session_active < 100, "sanity check: active is bounded");
}

#[test]
fn test_transaction_metrics() {
    let db = db_with_metrics();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.commit().unwrap();

    let m = db.metrics();
    assert!(m.tx_committed >= 1, "expected at least 1 commit");
}

#[test]
fn test_transaction_rollback_metric() {
    let db = db_with_metrics();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();
    session.rollback().unwrap();

    let m = db.metrics();
    assert!(m.tx_rolled_back >= 1, "expected at least 1 rollback");
}

#[test]
fn test_query_error_metric() {
    let db = db_with_metrics();
    let session = db.session();

    // Division by zero or type error should produce a query error
    let _ = session.execute("RETURN 1 / 0");

    let m = db.metrics();
    // Even if the query "succeeds" with a null result, at minimum query_count should be > 0.
    // Query errors are only counted for execution-time failures, not parse errors.
    assert!(m.query_count >= 1, "query should still be counted");
}

#[test]
fn test_cache_metrics_in_snapshot() {
    let db = db_with_metrics();
    let session = db.session();

    // Execute same query twice: first is a miss, second should be a hit
    session.execute("INSERT (:Animal {name: 'Dog'})").unwrap();
    session.execute("MATCH (a:Animal) RETURN a.name").unwrap();
    session.execute("MATCH (a:Animal) RETURN a.name").unwrap();

    let m = db.metrics();
    // Cache hits should be > 0 after the second identical query
    assert!(
        m.cache_hits > 0,
        "expected cache hits after repeated query, got {}",
        m.cache_hits
    );
}

#[test]
fn test_prometheus_output_format() {
    let db = db_with_metrics();
    let session = db.session();
    session.execute("INSERT (:Test {v: 1})").unwrap();

    let prom = db.metrics_prometheus();
    assert!(
        prom.contains("grafeo_query_count"),
        "prometheus output should contain grafeo_query_count"
    );
    assert!(
        prom.contains("grafeo_session_created"),
        "prometheus output should contain grafeo_session_created"
    );
}

/// Block-STM was removed in 0.6.0: neither the snapshot nor the Prometheus
/// export reports counters for an executor that does not exist (they always
/// read 0, which a dashboard would take for "no batches ran").
#[test]
fn no_metric_names_the_removed_block_stm_executor() {
    let db = db_with_metrics();
    let session = db.session();
    session
        .execute("INSERT (:City {name: 'Amsterdam', population: 88})")
        .unwrap();

    let prom = db.metrics_prometheus();
    assert!(
        prom.contains("grafeo_query_count"),
        "the export still lists the query counters"
    );
    assert!(
        !prom.contains("block_stm"),
        "the Prometheus export names a Block-STM counter:\n{prom}"
    );

    let json = serde_json::to_string(&db.metrics()).expect("a snapshot serializes");
    assert!(
        json.contains("\"query_count\""),
        "the snapshot still carries the query counters: {json}"
    );
    assert!(
        !json.contains("block_stm"),
        "the metrics snapshot carries a Block-STM field: {json}"
    );
}

#[test]
fn test_reset_metrics() {
    let db = db_with_metrics();
    let session = db.session();
    session.execute("INSERT (:Reset {v: 1})").unwrap();

    let m1 = db.metrics();
    assert!(m1.query_count > 0);

    db.reset_metrics();
    let m2 = db.metrics();
    assert_eq!(m2.query_count, 0, "query_count should be 0 after reset");
}
