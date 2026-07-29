//! One-shot helper to regenerate the golden snapshot fixture.
//!
//! Run with: cargo test --all-features -p grafeo-engine --test golden_format -- regenerate_snapshot_fixture --ignored
//! Then commit the updated fixture file. This test is ignored by default.
//!
//! **When to regenerate:** only after an intentional snapshot format change
//! (i.e. bumping `SNAPSHOT_VERSION`). If the golden test in `golden_format.rs`
//! fails without a version bump, fix the regression instead of regenerating.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Builds the canonical fixture graph used by both the generator and the
/// golden format tests. Keep this in sync with the assertions in
/// `golden_format.rs`.
pub fn build_fixture_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();

    // 3 nodes with different labels and properties
    let alix = db.create_node(&["Person"]).unwrap();
    db.set_node_property(alix, "name", Value::String("Alix".into()))
        .unwrap();
    db.set_node_property(alix, "age", Value::Int64(30)).unwrap();

    let gus = db.create_node(&["Person", "Employee"]).unwrap();
    db.set_node_property(gus, "name", Value::String("Gus".into()))
        .unwrap();
    db.set_node_property(gus, "age", Value::Int64(25)).unwrap();

    let acme = db.create_node(&["Company"]).unwrap();
    db.set_node_property(acme, "name", Value::String("Acme Corp".into()))
        .unwrap();

    // 2 edges with properties
    let knows = db.create_edge(alix, gus, "KNOWS");
    db.set_edge_property(knows, "since", Value::Int64(2020));

    let works = db.create_edge(gus, acme, "WORKS_AT");
    db.set_edge_property(works, "role", Value::String("Engineer".into()));

    db
}

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_snapshot_fixture() {
    let db = build_fixture_db();
    let bytes = db.export_snapshot().unwrap();

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v4.bin"
    );
    std::fs::write(path, &bytes).unwrap();
    println!("Wrote {} bytes to {path}", bytes.len());
    println!("First byte (snapshot version): {}", bytes[0]);
}
