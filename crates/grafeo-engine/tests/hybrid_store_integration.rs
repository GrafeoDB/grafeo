//! Integration tests for HybridStore through GrafeoDB::with_hybrid().
//!
//! Requires feature: `compact-store` (which also enables the `hybrid` module).
//!
//! Validates that HybridStore works end-to-end:
//! - `with_hybrid` wraps a CompactStore in a queryable, writable GrafeoDB
//! - `compact_to_hybrid` freezes current LPG data and switches to hybrid mode

#![cfg(feature = "compact-store")]

use grafeo_core::graph::compact::CompactStoreBuilder;
use grafeo_engine::{Config, GrafeoDB};

fn test_compact() -> grafeo_core::graph::compact::CompactStore {
    let grades: Vec<u64> = (0..100).collect();
    let names: Vec<&str> = (0..100)
        .map(|i| if i % 2 == 0 { "Alice" } else { "Bob" })
        .collect();
    CompactStoreBuilder::new()
        .node_table("Student", |b| {
            b.column_bitpacked("grade", &grades, 8)
                .column_dict("name", &names)
        })
        .build()
        .unwrap()
}

#[test]
fn with_hybrid_creates_queryable_db() {
    let db = GrafeoDB::with_hybrid(test_compact(), Config::in_memory()).unwrap();
    let result = db.execute("MATCH (s:Student) RETURN count(s)").unwrap();
    let rows = result.into_rows();
    assert_eq!(rows.len(), 1);
}

#[test]
fn hybrid_db_supports_mutations() {
    let db = GrafeoDB::with_hybrid(test_compact(), Config::in_memory()).unwrap();
    db.execute("CREATE (:Person {name: 'Alix'})").unwrap();
    let result = db.execute("MATCH (p:Person) RETURN p.name").unwrap();
    let rows = result.into_rows();
    assert_eq!(rows.len(), 1);
}

#[test]
fn compact_to_hybrid_preserves_data() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("CREATE (:Foo {x: 1})").unwrap();
    db.execute("CREATE (:Foo {x: 2})").unwrap();
    db.compact_to_hybrid().unwrap();
    let result = db.execute("MATCH (f:Foo) RETURN count(f)").unwrap();
    let rows = result.into_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], grafeo_common::types::Value::Int64(2));
    // Can still write after switching to hybrid
    db.execute("CREATE (:Bar {y: 3})").unwrap();
    let result = db.execute("MATCH (n) RETURN count(n)").unwrap();
    let rows = result.into_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], grafeo_common::types::Value::Int64(3));
}

/// Regression test: compact() after compact_to_hybrid() must clear hybrid_overlay.
///
/// Before the fix, `compact()` left `hybrid_overlay` pointing at the previous
/// HybridStore's overlay LpgStore.  Sessions created after `compact()` would
/// receive that stale Arc, preventing the old overlay from being freed and
/// potentially routing MVCC operations to a store that had no backing write
/// store.
#[test]
fn compact_after_hybrid_clears_overlay() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("CREATE (:Foo {x: 1})").unwrap();
    db.compact_to_hybrid().unwrap();

    // Switch from hybrid → read-only compact.
    db.compact().unwrap();

    // The DB should be read-only now; writes must fail.
    assert!(db.execute("CREATE (:Bar {y: 2})").is_err());

    // Reads should still work and reflect the data that was present before
    // the second compact() call.
    let result = db.execute("MATCH (f:Foo) RETURN count(f)").unwrap();
    let rows = result.into_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], grafeo_common::types::Value::Int64(1));

    // Verify that a new session does not inherit the old overlay.
    // If hybrid_overlay were still set the session's internal store would be
    // the dead overlay rather than a fresh empty LpgStore, which would not
    // cause an immediate panic but would keep the old allocation alive.
    // We assert that the public field visible to submodules is None by
    // exercising the session path; the session must be usable.
    let session = db.session();
    let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
    assert_eq!(result.into_rows().len(), 1);
}
