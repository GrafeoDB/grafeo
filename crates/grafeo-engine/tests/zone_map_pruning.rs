//! Zone-map pruning: skip a filter's input when statistics prove nothing can
//! match, but only when the statistics describe what the variable is bound to.
//!
//! Node statistics used to be applied to edge and map variables that shared a
//! property key with nodes, silently returning no rows. The spec cases in
//! `tests/spec/lpg/cypher/regression.gtest` cover the wrong-result side; these
//! tests pin that pruning still happens where it is valid.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test zone_map_pruning
//! ```

#![allow(missing_docs)]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute_cypher("CREATE (:N {id: 'a', w: 1})-[:T {id: 'e1', w: 50}]->(:N {id: 'b', w: 2})")
        .unwrap();
    db
}

fn profile(db: &GrafeoDB, query: &str) -> String {
    let result = db
        .session()
        .execute_cypher(&format!("PROFILE {query}"))
        .unwrap();
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("expected a profile string, got {other:?}"),
    }
}

/// Whether a filter was replaced by an empty result: the pruned filter shows as
/// `Empty (<predicate>)`; `Empty ()` is the single-row source of UNWIND.
fn pruned(plan: &str) -> bool {
    plan.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("Empty (") && !line.starts_with("Empty ()")
    })
}

#[test]
fn node_statistics_still_prune_node_filters() {
    let db = db();
    let plan = profile(&db, "MATCH (n:N) WHERE n.w = 1000 RETURN n.id");
    assert!(plan.contains("Empty (n.w Eq 1000)"), "{plan}");
}

#[test]
fn edge_statistics_prune_edge_filters() {
    let db = db();
    let plan = profile(&db, "MATCH ()-[r]->() WHERE r.w = 1000 RETURN r.id");
    assert!(plan.contains("Empty (r.w Eq 1000)"), "{plan}");

    // A value inside the edge range (and outside the node range) is not pruned.
    let plan = profile(&db, "MATCH ()-[r]->() WHERE r.w = 50 RETURN r.id");
    assert!(!pruned(&plan), "{plan}");
}

#[test]
fn map_and_projected_values_are_never_pruned() {
    let db = db();
    for query in [
        "UNWIND [{w: 1000}] AS u WITH u WHERE u.w = 1000 RETURN u.w",
        "MATCH (n:N) WITH n.w AS w WHERE w = 1000 RETURN w",
        "MATCH (n:N {id: 'a'}) SET n.w = 1000 WITH n WHERE n.w = 1000 RETURN n.id",
    ] {
        let plan = profile(&db, query);
        assert!(!pruned(&plan), "{query}\n{plan}");
    }
}

#[test]
fn values_written_earlier_in_the_query_are_not_pruned() {
    // `b` is a plain node scan, but the zone map was built before the SET, so
    // it does not know about w = 999 yet.
    let db = db();
    let rows = db
        .session()
        .execute_cypher(
            "MATCH (a:N {id: 'a'}) SET a.w = 999 WITH a MATCH (b:N) WHERE b.w = 999 RETURN b.id",
        )
        .unwrap();
    assert_eq!(rows.rows(), [[Value::from("a")]]);
}
