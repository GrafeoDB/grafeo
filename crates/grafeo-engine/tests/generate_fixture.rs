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
    // Written straight to the store, all at epoch 0, so the export matches the
    // checked-in bytes: the database's write API commits each call and moves
    // the epoch, which the exported property versions record.
    let store = db.store();

    // 3 nodes with different labels and properties
    let alix = store.create_node(&["Person"]);
    store.set_node_property(alix, "name", Value::String("Alix".into()));
    store.set_node_property(alix, "age", Value::Int64(30));

    let gus = store.create_node(&["Person", "Employee"]);
    store.set_node_property(gus, "name", Value::String("Gus".into()));
    store.set_node_property(gus, "age", Value::Int64(25));

    let acme = store.create_node(&["Company"]);
    store.set_node_property(acme, "name", Value::String("Acme Corp".into()));

    // 2 edges with properties
    let knows = store.create_edge(alix, gus, "KNOWS");
    store.set_edge_property(knows, "since", Value::Int64(2020));

    let works = store.create_edge(gus, acme, "WORKS_AT");
    store.set_edge_property(works, "role", Value::String("Engineer".into()));

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
