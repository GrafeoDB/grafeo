//! Golden fixture tests for binary format stability.
//!
//! These tests deserialize a **committed** binary fixture (`snapshot_v4.bin`)
//! and verify the current code can still read it correctly. If a code change
//! silently alters the binary layout (bincode bump, serde derive change, enum
//! variant reorder), these tests fail immediately.
//!
//! Three layers of protection:
//! 1. **Backward-read**: load committed bytes, assert correct data
//! 2. **Round-trip**: import golden -> export -> re-import, verify no data loss
//! 3. **Byte-equality**: export from identical graph, assert exact byte match
//!
//! ## When these tests fail
//!
//! - **Accidental breakage** (no `SNAPSHOT_VERSION` bump): fix the regression.
//! - **Intentional format change** (version bumped):
//!   1. Archive the old fixture: `cp fixtures/snapshot_v4.bin fixtures/archive/`
//!   2. Regenerate:
//!      ```
//!      cargo test --all-features -p grafeo-engine --test golden_format -- regenerate_snapshot_fixture --ignored
//!      ```
//!   3. Rename the new fixture to match the new version (e.g. `snapshot_v5.bin`)
//!   4. Update `EXPECTED_SNAPSHOT_VERSION` and `golden_bytes()` below
//!   5. Commit both the archived and new fixture files

mod generate_fixture;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Must match `SNAPSHOT_VERSION` in `persistence.rs`.
const EXPECTED_SNAPSHOT_VERSION: u8 = 4;

/// Load the committed golden fixture bytes.
fn golden_bytes() -> &'static [u8] {
    include_bytes!("fixtures/snapshot_v4.bin")
}

// ---------------------------------------------------------------------------
// Backward-read tests: can today's code deserialize yesterday's bytes?
// ---------------------------------------------------------------------------

#[test]
fn golden_snapshot_version_byte() {
    let bytes = golden_bytes();
    assert_eq!(
        bytes[0], EXPECTED_SNAPSHOT_VERSION,
        "fixture version byte ({}) does not match EXPECTED_SNAPSHOT_VERSION ({}), regenerate the fixture",
        bytes[0], EXPECTED_SNAPSHOT_VERSION,
    );
}

#[test]
fn golden_import_succeeds() {
    let bytes = golden_bytes();
    if let Err(e) = GrafeoDB::import_snapshot(bytes) {
        panic!("failed to import golden fixture: {e}");
    }
}

#[test]
fn golden_node_count() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    assert_eq!(db.node_count(), 3);
}

#[test]
fn golden_edge_count() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    assert_eq!(db.edge_count(), 2);
}

#[test]
fn golden_node_labels() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let session = db.session();

    let persons = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(persons.rows().len(), 2);
    assert_eq!(persons.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(persons.rows()[1][0], Value::String("Gus".into()));

    let companies = session.execute("MATCH (c:Company) RETURN c.name").unwrap();
    assert_eq!(companies.rows().len(), 1);
    assert_eq!(companies.rows()[0][0], Value::String("Acme Corp".into()));
}

#[test]
fn golden_node_properties() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(30));

    let result = session
        .execute("MATCH (p:Person) WHERE p.name = 'Gus' RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(25));
}

#[test]
fn golden_multi_labels() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let session = db.session();

    let employees = session.execute("MATCH (e:Employee) RETURN e.name").unwrap();
    assert_eq!(employees.rows().len(), 1);
    assert_eq!(employees.rows()[0][0], Value::String("Gus".into()));
}

#[test]
fn golden_edge_types_and_properties() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let session = db.session();

    let knows = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
        .unwrap();
    assert_eq!(knows.rows().len(), 1);
    assert_eq!(knows.rows()[0][0], Value::Int64(2020));

    let works = session
        .execute("MATCH ()-[e:WORKS_AT]->() RETURN e.role")
        .unwrap();
    assert_eq!(works.rows().len(), 1);
    assert_eq!(works.rows()[0][0], Value::String("Engineer".into()));
}

#[test]
fn golden_edge_connectivity() {
    let db = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let session = db.session();

    // Alix -[:KNOWS]-> Gus
    let result = session
        .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));

    // Gus -[:WORKS_AT]-> Acme Corp
    let result = session
        .execute("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][1], Value::String("Acme Corp".into()));
}

// ---------------------------------------------------------------------------
// Round-trip through golden fixture: import -> re-export -> re-import
// ---------------------------------------------------------------------------

#[test]
fn golden_round_trip_preserves_data() {
    // Import the golden fixture, re-export, re-import, verify data survives.
    // This catches format drift that is not backward-compatible (e.g. a field
    // type change that serializes but produces different logical data).
    let db1 = GrafeoDB::import_snapshot(golden_bytes()).unwrap();
    let re_exported = db1.export_snapshot().unwrap();
    let db2 = GrafeoDB::import_snapshot(&re_exported).unwrap();

    assert_eq!(db2.node_count(), 3);
    assert_eq!(db2.edge_count(), 2);

    let session = db2.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(30));

    let result = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(2020));

    let result = session
        .execute("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][1], Value::String("Acme Corp".into()));
}

// ---------------------------------------------------------------------------
// Byte-equality stability: deterministic export enables exact comparison
// ---------------------------------------------------------------------------

#[test]
fn golden_byte_equality() {
    // Re-create the same graph and check the export is byte-for-byte identical.
    // Export ordering is deterministic (nodes/edges sorted by ID, properties by
    // key, labels alphabetically), so any difference means the format drifted.
    let db = generate_fixture::build_fixture_db();
    let fresh_bytes = db.export_snapshot().unwrap();
    let golden = golden_bytes();

    assert_eq!(
        fresh_bytes.len(),
        golden.len(),
        "snapshot byte length changed: golden={}, fresh={}, format may have drifted",
        golden.len(),
        fresh_bytes.len(),
    );

    assert_eq!(
        fresh_bytes.as_slice(),
        golden,
        "snapshot bytes differ despite same graph data, export may not be deterministic",
    );
}
