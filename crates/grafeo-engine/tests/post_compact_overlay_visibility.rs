//! Regression tests for writes performed after `GrafeoDB::compact()`
//! remaining visible to subsequent GQL `MATCH` queries.
//!
//! Pre-fix behaviour: `LayeredStore::is_node_visible_at_epoch` (and its
//! edge / versioned / epoch siblings) fell through to the base store's
//! visibility check for any id not in `dirty_node_ids`. Overlay-only
//! nodes (post-`compact()` writes) are not "dirty" in that sense —
//! `dirty_node_ids` only tracks overlay modifications of base nodes —
//! so the base was asked, didn't know the id, and returned false.
//! Result: post-compact writes silently vanished from reads.
//!
//! Tracked upstream as GrafeoDB/grafeo#302.
//!
//! ```bash
//! cargo test -p grafeo-engine --features "compact-store lpg gql" \
//!     --test post_compact_overlay_visibility
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn int_scalar(db: &GrafeoDB, q: &str) -> i64 {
    let s = db.session();
    match &s.execute(q).unwrap().rows()[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64 for `{q}`, got {other:?}"),
    }
}

fn row_count(db: &GrafeoDB, q: &str) -> usize {
    let s = db.session();
    s.execute(q).unwrap().rows().len()
}

#[test]
fn node_created_after_compact_is_visible() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_node(&["X"]).unwrap();
    db.compact().expect("compact");
    db.create_node(&["Y"]).unwrap();

    assert_eq!(int_scalar(&db, "MATCH (n) RETURN count(n)"), 2);
    assert_eq!(row_count(&db, "MATCH (n:Y) RETURN n"), 1);
    assert_eq!(row_count(&db, "MATCH (n:X) RETURN n"), 1);
}

#[test]
fn multiple_post_compact_nodes_visible() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_node(&["Base"]).unwrap();
    db.compact().expect("compact");
    for _ in 0..5 {
        db.create_node(&["Overlay"]).unwrap();
    }

    assert_eq!(int_scalar(&db, "MATCH (n) RETURN count(n)"), 6);
    assert_eq!(int_scalar(&db, "MATCH (n:Overlay) RETURN count(n)"), 5);
    assert_eq!(int_scalar(&db, "MATCH (n:Base) RETURN count(n)"), 1);
}

#[test]
fn post_compact_edge_visible() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]).unwrap();
    let b = db.create_node(&["B"]).unwrap();
    db.create_edge(a, b, "PRE");
    db.compact().expect("compact");

    // New nodes + edge entirely in the overlay.
    let c = db.create_node(&["A"]).unwrap();
    let d = db.create_node(&["B"]).unwrap();
    db.create_edge(c, d, "POST");

    assert_eq!(int_scalar(&db, "MATCH ()-[r]->() RETURN count(r)"), 2);
    assert_eq!(int_scalar(&db, "MATCH ()-[r:POST]->() RETURN count(r)"), 1);
    assert_eq!(int_scalar(&db, "MATCH ()-[r:PRE]->() RETURN count(r)"), 1);
}

#[test]
fn post_compact_edge_between_base_and_overlay_nodes() {
    let mut db = GrafeoDB::new_in_memory();
    let base_a = db.create_node(&["A"]).unwrap();
    db.compact().expect("compact");
    let overlay_b = db.create_node(&["B"]).unwrap();
    db.create_edge(base_a, overlay_b, "CROSS");

    // The edge connects a base node to an overlay node — visibility has
    // to work on both sides.
    assert_eq!(int_scalar(&db, "MATCH ()-[r]->() RETURN count(r)"), 1);
    assert_eq!(
        row_count(&db, "MATCH (a:A)-[r:CROSS]->(b:B) RETURN a, r, b"),
        1
    );
}

#[test]
fn post_compact_node_property_survives_reread() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_node(&["X"]).unwrap();
    db.compact().expect("compact");
    let n = db.create_node(&["Y"]).unwrap();
    db.set_node_property(n, "label", Value::String("hello".into()))
        .unwrap();

    let s = db.session();
    let r = s.execute("MATCH (n:Y) RETURN n.label").unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(
        r.rows()[0][0],
        Value::String("hello".into()),
        "overlay-only node property should be readable"
    );
}
