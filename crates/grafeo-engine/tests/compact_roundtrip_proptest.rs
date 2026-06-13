//! Property-based round-trip tests for [`GrafeoDB::compact`].
//!
//! Generates arbitrary small LPG graphs, applies them to two fresh in-memory
//! databases, compacts one, then asserts that a battery of GQL queries returns
//! equivalent cardinalities and row shapes on both. Catches regressions in the
//! live → [`LayeredStore`] (CompactStore base + LpgStore overlay) conversion
//! path introduced in 0.5.31–0.5.32.
//!
//! Content-equality (`sum(n.num)` etc.) and post-`compact()` write visibility
//! are exercised by the sibling fix PRs for GrafeoDB/grafeo#301 and #302,
//! which ship deterministic tests alongside the fix.
//!
//! ```bash
//! cargo test -p grafeo-engine --features "compact-store lpg gql" \
//!     --test compact_roundtrip_proptest
//!
//! # bump coverage locally:
//! PROPTEST_CASES=1024 cargo test -p grafeo-engine ...
//! ```
//!
//! [`LayeredStore`]: grafeo_core::graph::compact::layered::LayeredStore

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;
use proptest::prelude::*;

// ── Input space ──────────────────────────────────────────────────

/// A minimal serialisable description of a graph, used to build two identical
/// databases deterministically from one proptest-generated seed.
#[derive(Debug, Clone)]
struct GraphSpec {
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
}

#[derive(Debug, Clone)]
struct NodeSpec {
    label: &'static str,
    num: Option<i64>,
    name: Option<String>,
}

#[derive(Debug, Clone)]
struct EdgeSpec {
    src: usize, // index into `nodes`
    dst: usize,
    kind: &'static str,
}

const LABELS: &[&str] = &["A", "B", "C"];
const EDGE_KINDS: &[&str] = &["R1", "R2"];

fn label_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just(LABELS[0]), Just(LABELS[1]), Just(LABELS[2])]
}

fn edge_kind_strategy() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just(EDGE_KINDS[0]), Just(EDGE_KINDS[1])]
}

fn node_spec_strategy() -> impl Strategy<Value = NodeSpec> {
    // Bound `num` so sum() across ~20 nodes can't overflow i64.
    (
        label_strategy(),
        proptest::option::of(-1_000_000_000i64..1_000_000_000i64),
        proptest::option::of("[a-z]{1,8}"),
    )
        .prop_map(|(label, num, name)| NodeSpec { label, num, name })
}

fn graph_spec_strategy() -> impl Strategy<Value = GraphSpec> {
    prop::collection::vec(node_spec_strategy(), 1..=20).prop_flat_map(|nodes| {
        let n = nodes.len();
        let edges = prop::collection::vec(
            (0..n, 0..n, edge_kind_strategy()).prop_map(|(src, dst, kind)| EdgeSpec {
                src,
                dst,
                kind,
            }),
            0..=30,
        );
        (Just(nodes), edges).prop_map(|(nodes, edges)| GraphSpec { nodes, edges })
    })
}

// ── Application ──────────────────────────────────────────────────

fn apply_spec(spec: &GraphSpec, db: &GrafeoDB) {
    let mut ids: Vec<NodeId> = Vec::with_capacity(spec.nodes.len());
    for n in &spec.nodes {
        let id = db.create_node(&[n.label]).unwrap();
        if let Some(num) = n.num {
            db.set_node_property(id, "num", Value::Int64(num)).unwrap();
        }
        if let Some(s) = &n.name {
            db.set_node_property(id, "name", Value::String(s.clone().into()))
                .unwrap();
        }
        ids.push(id);
    }
    for e in &spec.edges {
        let _ = db.create_edge(ids[e.src], ids[e.dst], e.kind);
    }
}

// ── Query helpers ────────────────────────────────────────────────

fn scalar(db: &GrafeoDB, query: &str) -> Value {
    let session = db.session();
    let result = session
        .execute(query)
        .unwrap_or_else(|e| panic!("query failed: {query} — {e:?}"));
    let rows = result.rows();
    assert_eq!(
        rows.len(),
        1,
        "expected 1 row for `{query}`, got {}",
        rows.len()
    );
    rows[0][0].clone()
}

fn row_count(db: &GrafeoDB, query: &str) -> usize {
    let session = db.session();
    let result = session
        .execute(query)
        .unwrap_or_else(|e| panic!("query failed: {query} — {e:?}"));
    result.rows().len()
}

// ── Oracle ───────────────────────────────────────────────────────

fn assert_scalar_equivalent(a: &GrafeoDB, b: &GrafeoDB, query: &str) {
    let va = scalar(a, query);
    let vb = scalar(b, query);
    assert_eq!(
        va, vb,
        "divergence on scalar query `{query}`: live={va:?}, compacted={vb:?}"
    );
}

fn assert_rows_equivalent(a: &GrafeoDB, b: &GrafeoDB, query: &str) {
    let ra = row_count(a, query);
    let rb = row_count(b, query);
    assert_eq!(
        ra, rb,
        "divergence on row-count query `{query}`: live={ra}, compacted={rb}"
    );
}

fn assert_equivalent(live: &GrafeoDB, compacted: &GrafeoDB) {
    // ── Cardinality ──────────────────────────────────────────────
    assert_scalar_equivalent(live, compacted, "MATCH (n) RETURN count(n)");
    assert_scalar_equivalent(live, compacted, "MATCH ()-[r]->() RETURN count(r)");

    for label in LABELS {
        let q = format!("MATCH (n:{label}) RETURN count(n)");
        assert_scalar_equivalent(live, compacted, &q);
    }
    for kind in EDGE_KINDS {
        let q = format!("MATCH ()-[r:{kind}]->() RETURN count(r)");
        assert_scalar_equivalent(live, compacted, &q);
    }
    for src_label in LABELS {
        for kind in EDGE_KINDS {
            for dst_label in LABELS {
                let q =
                    format!("MATCH (a:{src_label})-[r:{kind}]->(b:{dst_label}) RETURN count(r)");
                assert_scalar_equivalent(live, compacted, &q);
            }
        }
    }

    // ── Row shape ────────────────────────────────────────────────
    assert_rows_equivalent(live, compacted, "MATCH (n) RETURN n");
    assert_rows_equivalent(live, compacted, "MATCH (a)-[r]->(b) RETURN a, r, b");
}

// ── Properties ───────────────────────────────────────────────────

proptest! {
    // 128 cases balances coverage against CI wall-clock. Bump locally via
    // `PROPTEST_CASES=1024 cargo test ...` when investigating a flake.
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Cardinality and row-shape queries agree between a freshly built live
    /// database and the same spec compacted to `LayeredStore`.
    #[test]
    fn compact_preserves_cardinality_and_shape(spec in graph_spec_strategy()) {
        let live = GrafeoDB::new_in_memory();
        apply_spec(&spec, &live);

        let mut compacted = GrafeoDB::new_in_memory();
        apply_spec(&spec, &compacted);
        compacted.compact().expect("compact");

        assert_equivalent(&live, &compacted);
    }
}

// ── Fixed regression seeds ───────────────────────────────────────
//
// Concrete shapes worth asserting explicitly so regressions surface even
// when proptest shrinking is disabled.

#[test]
fn empty_graph_compact_is_noop() {
    let live = GrafeoDB::new_in_memory();
    let mut compacted = GrafeoDB::new_in_memory();
    compacted.compact().expect("compact empty");
    assert_equivalent(&live, &compacted);
}

#[test]
fn single_node_no_properties() {
    let live = GrafeoDB::new_in_memory();
    live.create_node(&["A"]);

    let mut compacted = GrafeoDB::new_in_memory();
    compacted.create_node(&["A"]);
    compacted.compact().expect("compact");

    assert_equivalent(&live, &compacted);
}

#[test]
fn self_loop_survives_compact() {
    let live = GrafeoDB::new_in_memory();
    let n = live.create_node(&["A"]);
    live.create_edge(n, n, "R1");

    let mut compacted = GrafeoDB::new_in_memory();
    let m = compacted.create_node(&["A"]);
    compacted.create_edge(m, m, "R1");
    compacted.compact().expect("compact");

    assert_equivalent(&live, &compacted);
}
