//! Branch coverage for `query/planner/lpg/filter.rs`.
//!
//! The primary `filter_pushdown.rs` suite exercises the common pushdown shapes.
//! These tests target the rarer branches of filter planning: deeply nested
//! EXISTS extraction, chained NOT EXISTS semi-joins, COUNT subquery comparison
//! in more shapes (flipped operator, AND-combined, with remaining predicate),
//! label-first scan with Int/Float coerced equality, zone map OR short circuit,
//! and every BETWEEN ordering variant.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test filter_planner_branches
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Social graph used throughout: 4 Person nodes with outgoing KNOWS / WORKS_AT
/// edges and a handful of Company nodes.
fn social_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();
    let jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("Berlin".into())),
            ],
        )
        .unwrap();
    let mia = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Mia".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Paris".into())),
            ],
        )
        .unwrap();
    let beatrix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Beatrix".into())),
                ("age", Value::Int64(40)),
                ("city", Value::String("Prague".into())),
            ],
        )
        .unwrap();

    let acme = session
        .create_node_with_props(&["Company"], [("name", Value::String("Acme".into()))])
        .unwrap();
    let globex = session
        .create_node_with_props(&["Company"], [("name", Value::String("Globex".into()))])
        .unwrap();

    // KNOWS graph: Vincent -> Jules, Vincent -> Mia, Jules -> Mia, Beatrix -> Vincent
    session.create_edge(vincent, jules, "KNOWS").unwrap();
    session.create_edge(vincent, mia, "KNOWS").unwrap();
    session.create_edge(jules, mia, "KNOWS").unwrap();
    session.create_edge(beatrix, vincent, "KNOWS").unwrap();

    // WORKS_AT: Vincent + Jules at Acme, Mia at Globex. Beatrix unemployed.
    session.create_edge(vincent, acme, "WORKS_AT").unwrap();
    session.create_edge(jules, acme, "WORKS_AT").unwrap();
    session.create_edge(mia, globex, "WORKS_AT").unwrap();

    db
}

fn sorted_names(result: &QueryResult, col: usize) -> Vec<String> {
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .map(|r| match &r[col] {
            Value::String(s) => s.to_string(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// Equality on a label scan without an index, Int/Float cross-type
// ---------------------------------------------------------------------------

#[test]
fn label_first_equality_coerces_int_property_to_float_literal() {
    // `age` is Int64 in the store; the literal is Float64. The filter over
    // the label scan (no index) decides membership with `=` and must accept
    // the cross-type match.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.age = 30.0 RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

#[test]
fn label_first_equality_coerces_float_literal_on_left() {
    // Literal on the left: the same `=` decides.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE 25.0 = n.age RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
}

// ---------------------------------------------------------------------------
// Zone map OR short-circuit (check_zone_map_for_predicate OR branch)
// ---------------------------------------------------------------------------

#[test]
fn zone_map_or_branch_evaluates_without_crashing() {
    // Two literal comparisons OR'd together. The zone map walks both sides
    // and combines them via the OR branch. This is a smoke test: the query
    // must return the union regardless of zone-map decisions.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) WHERE n.city = 'Prague' OR n.city = 'Paris' \
             RETURN n.name",
        )
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Beatrix", "Mia"]);
}

// ---------------------------------------------------------------------------
// Property index path with remaining predicate (try_plan_filter_with_property_index)
// ---------------------------------------------------------------------------

#[test]
fn indexed_equality_plus_range_uses_remaining_predicate_path() {
    let db = social_graph();
    db.create_property_index("city").unwrap();
    let session = db.session();

    // city = 'Amsterdam' should be pushed down through the index; the age
    // range stays as a remaining predicate wrapped around the NodeListOperator.
    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'Amsterdam' AND n.age > 25 RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

// ---------------------------------------------------------------------------
// COUNT subquery: AND-combined, flipped operator (literal on left), and
// remaining predicate handling in plan_count_as_apply.
// ---------------------------------------------------------------------------

#[test]
fn count_subquery_combined_with_and_extracts_count_branch() {
    let db = social_graph();
    let session = db.session();

    // COUNT { ... } > 0 extracts the COUNT branch, leaving n.age > 30 as
    // the remaining predicate inside plan_count_as_apply.
    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE COUNT { MATCH (n)-[:KNOWS]->() } > 0 \
               AND n.age > 30 \
             RETURN n.name",
        )
        .unwrap();
    // Mia has no outgoing KNOWS; Beatrix (40, KNOWS Vincent) passes both.
    assert_eq!(sorted_names(&result, 0), vec!["Beatrix"]);
}

#[test]
fn count_subquery_with_and_on_left_branch_also_extracted() {
    // Same as above but the COUNT is on the left side of the AND; the
    // extractor must walk both sides of the AND.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE n.age > 30 \
               AND COUNT { MATCH (n)-[:KNOWS]->() } > 0 \
             RETURN n.name",
        )
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Beatrix"]);
}

#[test]
fn count_subquery_with_flipped_operator_literal_on_left() {
    // `0 < COUNT { ... }` flips Lt -> Gt inside extract_count_from_binary.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE 0 < COUNT { MATCH (n)-[:KNOWS]->() } RETURN n.name")
        .unwrap();
    // Everyone except Mia has an outgoing KNOWS edge.
    assert_eq!(
        sorted_names(&result, 0),
        vec!["Beatrix", "Jules", "Vincent"]
    );
}

#[test]
fn count_subquery_with_flipped_ge_operator() {
    // `1 <= COUNT { ... }` flips Le -> Ge.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE 1 <= COUNT { MATCH (n)-[:KNOWS]->() } \
             RETURN n.name",
        )
        .unwrap();
    assert_eq!(
        sorted_names(&result, 0),
        vec!["Beatrix", "Jules", "Vincent"]
    );
}

#[test]
fn count_subquery_equality_zero_keeps_left_joined_nulls() {
    // COUNT { ... } = 0 must keep Mia (no outgoing KNOWS). This verifies
    // the Left join + COUNT-of-right-column path that yields zero for
    // unmatched outer rows.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE COUNT { MATCH (n)-[:KNOWS]->() } = 0 \
             RETURN n.name",
        )
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Mia"]);
}

// ---------------------------------------------------------------------------
// Multiple chained NOT EXISTS (plan_exists_as_semi_join_with_input recursion,
// including its remaining-predicate branch).
// ---------------------------------------------------------------------------

#[cfg(feature = "cypher")]
#[test]
fn two_not_exists_in_and_chain_as_anti_joins() {
    // Two NOT EXISTS patterns plus a property predicate. extract_complex_exists
    // peels one off, plan_exists_as_semi_join builds the first anti-join,
    // then the recursive handler sees another complex NOT EXISTS in the
    // remaining predicate and chains a second anti-join, finally applying
    // the scalar predicate as a FilterOperator.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (n:Person) \
             WHERE NOT EXISTS { MATCH (n)-[:KNOWS]->(m)-[:WORKS_AT]->(c) } \
               AND NOT EXISTS { MATCH (n)-[:WORKS_AT]->(c) WHERE c.name = 'Globex' } \
               AND n.age >= 25 \
             RETURN n.name",
        )
        .unwrap();
    // Vincent KNOWS Jules -> Acme (fails 1st clause).
    // Jules KNOWS Mia -> Globex (fails 1st clause).
    // Mia has no outgoing KNOWS (passes 1st) and WORKS_AT Globex (fails 2nd).
    // Beatrix KNOWS Vincent -> Acme (fails 1st clause).
    // => no matches.
    assert!(
        result.rows().is_empty(),
        "expected no rows, got {:?}",
        result.rows()
    );
}

#[cfg(feature = "cypher")]
#[test]
fn not_exists_and_exists_chained_with_scalar_tail() {
    // One NOT EXISTS + one EXISTS with a tail scalar predicate. Exercises
    // the semi-join + anti-join chain plus the FilterOperator tail in
    // plan_exists_as_semi_join_with_input.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (n:Person) \
             WHERE EXISTS { MATCH (n)-[:KNOWS]->(m)-[:WORKS_AT]->(c) } \
               AND NOT EXISTS { MATCH (n)-[:WORKS_AT]->(c) WHERE c.name = 'Globex' } \
               AND n.age >= 25 \
             RETURN n.name",
        )
        .unwrap();
    // Vincent: KNOWS Jules->Acme (EXISTS ok); WORKS_AT Acme (NOT EXISTS ok); 30>=25.
    // Jules: KNOWS Mia->Globex (EXISTS ok); WORKS_AT Acme (NOT EXISTS ok); 25>=25.
    // Beatrix: KNOWS Vincent->Acme (EXISTS ok); no WORKS_AT (NOT EXISTS ok); 40>=25.
    // Mia: no outgoing KNOWS (EXISTS fails).
    assert_eq!(
        sorted_names(&result, 0),
        vec!["Beatrix", "Jules", "Vincent"]
    );
}

// ---------------------------------------------------------------------------
// extract_complex_exists deeper recursion: EXISTS buried beneath an AND
// sibling that is itself a compound scalar expression.
// ---------------------------------------------------------------------------

#[cfg(feature = "cypher")]
#[test]
fn complex_exists_with_compound_scalar_sibling() {
    // `(scalar1 AND scalar2) AND EXISTS { multihop }` -- the Cypher translator
    // builds a left-leaning tree. extract_complex_exists must descend through
    // the non-EXISTS sibling to find the EXISTS and reconstruct the remaining
    // predicate as `scalar1 AND scalar2`.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (n:Person) \
             WHERE n.age >= 25 AND n.age <= 40 \
               AND EXISTS { MATCH (n)-[:KNOWS]->(m)-[:WORKS_AT]->(c) } \
             RETURN n.name",
        )
        .unwrap();
    // Matches: Vincent, Jules, Beatrix.
    assert_eq!(
        sorted_names(&result, 0),
        vec!["Beatrix", "Jules", "Vincent"]
    );
}

// ---------------------------------------------------------------------------
// BETWEEN: every ordering variant of extract_between_predicate. The existing
// pushdown tests cover (Ge, Le), (Gt, Lt), (Ge, Lt). Here we fill in the rest.
// ---------------------------------------------------------------------------

#[test]
fn between_gt_le_variant() {
    let db = social_graph();
    let session = db.session();

    // n.age > 25 AND n.age <= 35 => Vincent(30), Mia(35)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age > 25 AND n.age <= 35 RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Mia", "Vincent"]);
}

#[test]
fn between_reversed_le_ge_variant() {
    let db = social_graph();
    let session = db.session();

    // Reversed: max first -- n.age <= 35 AND n.age >= 25 => Jules, Vincent, Mia
    let result = session
        .execute("MATCH (n:Person) WHERE n.age <= 35 AND n.age >= 25 RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Jules", "Mia", "Vincent"]);
}

#[test]
fn between_reversed_lt_ge_variant() {
    let db = social_graph();
    let session = db.session();

    // n.age < 40 AND n.age >= 30 => Vincent(30), Mia(35)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age < 40 AND n.age >= 30 RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Mia", "Vincent"]);
}

#[test]
fn between_reversed_le_gt_variant() {
    let db = social_graph();
    let session = db.session();

    // n.age <= 35 AND n.age > 25 => Vincent(30), Mia(35)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age <= 35 AND n.age > 25 RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Mia", "Vincent"]);
}

#[test]
fn between_reversed_lt_gt_variant() {
    let db = social_graph();
    let session = db.session();

    // n.age < 40 AND n.age > 25 => Vincent(30), Mia(35)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age < 40 AND n.age > 25 RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Mia", "Vincent"]);
}

#[test]
fn between_mismatched_variable_falls_through() {
    // When both sides of the AND compare range-style but on different
    // variables, extract_between_predicate must return None and the filter
    // falls through to the generic plan. The result must still be correct.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (a:Person), (b:Person) \
             WHERE a.age > 30 AND b.age < 30 \
             RETURN a.name, b.name \
             ORDER BY a.name, b.name",
        )
        .unwrap();
    // a in {Mia(35), Beatrix(40)}, b = Jules(25) only.
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn range_predicate_only_lt_literal_on_left() {
    // Reversed Lt -> Gt via extract_range_predicate.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE 30 < n.age RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Beatrix", "Mia"]);
}

#[test]
fn range_predicate_only_gt_literal_on_left() {
    // 40 > n.age means n.age < 40.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE 40 > n.age RETURN n.name")
        .unwrap();
    assert_eq!(sorted_names(&result, 0), vec!["Jules", "Mia", "Vincent"]);
}

// ---------------------------------------------------------------------------
// Equality with NULL literal must NOT be pushed down as an index lookup
// (extract_property_equality rejects NULL under three-valued logic).
// ---------------------------------------------------------------------------

#[test]
fn equality_with_null_literal_is_not_pushed_down() {
    let db = social_graph();
    db.create_property_index("city").unwrap();
    let session = db.session();

    // `n.city = NULL` is always UNKNOWN, so no rows. The important thing is
    // that the planner does not treat NULL as an indexable equality (that
    // path would produce surprising results).
    let result = session
        .execute("MATCH (n:Person) WHERE n.city = NULL RETURN n.name")
        .unwrap();
    assert!(result.rows().is_empty(), "{:?}", result.rows());
}

// ---------------------------------------------------------------------------
// Compound equalities over a label scan
// ---------------------------------------------------------------------------

#[test]
fn compound_equality_all_pushed_down_no_remaining_predicate() {
    // Two equality conditions on the target variable.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) WHERE n.city = 'Amsterdam' AND n.name = 'Vincent' RETURN n.age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(30));
}

#[test]
fn compound_equality_with_unindexed_third_property() {
    // n.city = 'Amsterdam' AND n.name = 'Vincent' AND n.age > 25: two
    // equalities and a range in one AND.
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE n.city = 'Amsterdam' AND n.name = 'Vincent' AND n.age > 25 \
             RETURN n.name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}
