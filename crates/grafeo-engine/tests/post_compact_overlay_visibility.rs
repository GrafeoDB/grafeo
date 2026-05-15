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
    db.create_node(&["X"]);
    db.compact().expect("compact");
    db.create_node(&["Y"]);

    assert_eq!(int_scalar(&db, "MATCH (n) RETURN count(n)"), 2);
    assert_eq!(row_count(&db, "MATCH (n:Y) RETURN n"), 1);
    assert_eq!(row_count(&db, "MATCH (n:X) RETURN n"), 1);
}

#[test]
fn multiple_post_compact_nodes_visible() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_node(&["Base"]);
    db.compact().expect("compact");
    for _ in 0..5 {
        db.create_node(&["Overlay"]);
    }

    assert_eq!(int_scalar(&db, "MATCH (n) RETURN count(n)"), 6);
    assert_eq!(int_scalar(&db, "MATCH (n:Overlay) RETURN count(n)"), 5);
    assert_eq!(int_scalar(&db, "MATCH (n:Base) RETURN count(n)"), 1);
}

#[test]
fn post_compact_edge_visible() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.create_edge(a, b, "PRE");
    db.compact().expect("compact");

    // New nodes + edge entirely in the overlay.
    let c = db.create_node(&["A"]);
    let d = db.create_node(&["B"]);
    db.create_edge(c, d, "POST");

    assert_eq!(int_scalar(&db, "MATCH ()-[r]->() RETURN count(r)"), 2);
    assert_eq!(int_scalar(&db, "MATCH ()-[r:POST]->() RETURN count(r)"), 1);
    assert_eq!(int_scalar(&db, "MATCH ()-[r:PRE]->() RETURN count(r)"), 1);
}

#[test]
fn post_compact_edge_between_base_and_overlay_nodes() {
    let mut db = GrafeoDB::new_in_memory();
    let base_a = db.create_node(&["A"]);
    db.compact().expect("compact");
    let overlay_b = db.create_node(&["B"]);
    db.create_edge(base_a, overlay_b, "CROSS");

    // The edge connects a base node to an overlay node — visibility has
    // to work on both sides.
    assert_eq!(int_scalar(&db, "MATCH ()-[r]->() RETURN count(r)"), 1);
    assert_eq!(
        row_count(&db, "MATCH (a:A)-[r:CROSS]->(b:B) RETURN a, r, b"),
        1
    );
}

/// Regression: a property-anchored edge that lives entirely in the
/// snapshot tier must remain reachable after an unrelated overlay write
/// promotes one of its endpoints. The reproduction is the GQL pattern
/// `MATCH (a {id: $x})-[:T]->(b {id: $y})` — the planner may walk from
/// either anchor, so both promotion directions need to work.
///
/// Before the fix, `LayeredStore::edges_from` and `neighbors` skipped the
/// base layer whenever the source node was marked dirty. `ensure_in_overlay`
/// only copies labels and properties into the overlay, never adjacency, so
/// a dirty endpoint silently lost all of its pre-compact edges.
#[test]
fn post_compact_property_anchored_edge_survives_unrelated_overlay_write() {
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.set_node_property(a, "id", Value::Int64(1));
    db.set_node_property(b, "id", Value::Int64(2));
    db.create_edge(a, b, "T");
    db.compact().expect("compact");

    // Both endpoints are now entirely in the snapshot tier and the
    // BUSINESS_MEMBER-shaped pattern resolves cleanly.
    assert_eq!(
        row_count(
            &db,
            "MATCH (a:A {id: 1})-[:T]->(b:B {id: 2}) RETURN true"
        ),
        1,
        "double-anchor edge query should match before any overlay write"
    );

    // Unrelated overlay write that promotes `a` (it becomes the endpoint of
    // a brand-new overlay edge). The original snapshot-tier `T` edge between
    // `a` and `b` is not touched in any way.
    let c = db.create_node(&["C"]);
    db.set_node_property(c, "id", Value::Int64(99));
    db.create_edge(a, c, "UNRELATED");

    assert_eq!(
        row_count(
            &db,
            "MATCH (a:A {id: 1})-[:T]->(b:B {id: 2}) RETURN true"
        ),
        1,
        "snapshot-tier T edge must still match after an unrelated overlay write \
         promotes its source endpoint"
    );

    // And the same when the destination endpoint is the one that gets
    // promoted — the planner may anchor on either side.
    let mut db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["A"]);
    let b = db.create_node(&["B"]);
    db.set_node_property(a, "id", Value::Int64(1));
    db.set_node_property(b, "id", Value::Int64(2));
    db.create_edge(a, b, "T");
    db.compact().expect("compact");

    let c = db.create_node(&["C"]);
    db.set_node_property(c, "id", Value::Int64(99));
    db.create_edge(c, b, "UNRELATED"); // promotes b

    assert_eq!(
        row_count(
            &db,
            "MATCH (a:A {id: 1})-[:T]->(b:B {id: 2}) RETURN true"
        ),
        1,
        "snapshot-tier T edge must still match after an unrelated overlay write \
         promotes its destination endpoint"
    );
}

#[test]
fn post_compact_node_property_survives_reread() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_node(&["X"]);
    db.compact().expect("compact");
    let n = db.create_node(&["Y"]);
    db.set_node_property(n, "label", Value::String("hello".into()));

    let s = db.session();
    let r = s.execute("MATCH (n:Y) RETURN n.label").unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(
        r.rows()[0][0],
        Value::String("hello".into()),
        "overlay-only node property should be readable"
    );
}
