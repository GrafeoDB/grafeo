//! openCypher matches relationships isomorphically: a MATCH clause binds one
//! relationship at most once, across all its patterns and the relationships
//! of its variable-length patterns (openCypher 9, "Uniqueness"; TCK Match3
//! [15] and [16], Match4 [7]). Grafeo's Cypher used to match them
//! homomorphically, as GQL does by default, so one relationship could serve
//! two hops. A Cypher MATCH now behaves as GQL `MATCH DIFFERENT EDGES`, and
//! an unbounded variable-length pattern ends on its own, past the hundred
//! hops a walk stops at.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test relationship_uniqueness
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus, Vincent and Mia, and Amsterdam. Relationships with `w` 1 to 6:
/// KNOWS Alix->Gus (1), Gus->Vincent (2), Vincent->Alix (3), a second KNOWS
/// Alix->Gus (4), LIVES_IN Alix->Amsterdam (5) and a LIKES loop on Mia (6).
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (vincent:Person {name: 'Vincent'}), (mia:Person {name: 'Mia'}), \
         (ams:City {name: 'Amsterdam'}), \
         (alix)-[:KNOWS {w: 1}]->(gus), (gus)-[:KNOWS {w: 2}]->(vincent), \
         (vincent)-[:KNOWS {w: 3}]->(alix), (alix)-[:KNOWS {w: 4}]->(gus), \
         (alix)-[:LIVES_IN {w: 5}]->(ams), (mia)-[:LIKES {w: 6}]->(mia)",
    )
    .unwrap();
    db
}

/// A chain of 105 `Step` nodes (`i` 0 to 104) linked by NEXT.
fn chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("UNWIND range(0, 104) AS i CREATE (:Step {i: i})")
        .unwrap();
    db.execute_cypher("MATCH (a:Step), (b:Step) WHERE b.i = a.i + 1 CREATE (a)-[:NEXT]->(b)")
        .unwrap();
    db
}

/// The rows of `result` as text, sorted.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = result
        .unwrap()
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
        other => panic!("unexpected value {other:?}"),
    }
}

/// `rows` for literal text cells.
fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

fn cypher(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    rows(db.execute_cypher(query))
}

// ---------------------------------------------------------------------------
// Fixed-length patterns
// ---------------------------------------------------------------------------

/// The only relationship that ends where it starts is Mia's loop, so a path
/// that takes the same relationship twice has no match in Cypher. GQL lets
/// edges repeat by default: there it is the loop, twice.
#[test]
fn a_relationship_variable_used_twice_in_a_path_matches_nothing() {
    let db = graph();
    let query = "MATCH (a)-[r]->(b)-[r]->(c) RETURN a.name, b.name, c.name";
    assert_eq!(cypher(&db, query), expected(&[]));
    assert_eq!(rows(db.execute(query)), expected(&[&["Mia", "Mia", "Mia"]]));
}

/// Two KNOWS relationships lead into Gus, one into Vincent and one into
/// Alix: only Gus's two can be paired, in both orders.
#[test]
fn two_relationship_patterns_never_bind_one_relationship() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a)-[r1:KNOWS]->(b)<-[r2:KNOWS]-(c) RETURN a.name, c.name, r1.w, r2.w"
        ),
        expected(&[&["Alix", "Alix", "1", "4"], &["Alix", "Alix", "4", "1"]])
    );
}

/// Anonymous relationships bind relationships too.
#[test]
fn anonymous_relationships_are_unique_as_well() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a)-[:KNOWS]->(b)<-[:KNOWS]-(c) RETURN a.name, b.name, c.name"
        ),
        expected(&[&["Alix", "Gus", "Alix"], &["Alix", "Gus", "Alix"]])
    );
}

/// The patterns of one MATCH share the rule, wherever the commas put them.
#[test]
fn comma_separated_patterns_of_one_match_share_the_rule() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a)-[r1:KNOWS]->(b), (c)-[r2:KNOWS]->(b) RETURN a.name, c.name, r1.w, r2.w"
        ),
        expected(&[&["Alix", "Alix", "1", "4"], &["Alix", "Alix", "4", "1"]])
    );
}

/// An undirected two-hop pattern never goes back over the relationship it
/// came on: 14 of the 22 walks (each middle node with KNOWS degree d gives
/// d * (d - 1): Alix 6, Gus 6, Vincent 2), the count of GQL's DIFFERENT
/// EDGES.
#[test]
fn an_undirected_two_hop_pattern_never_comes_back_over_its_relationship() {
    let db = graph();
    let pattern = "(a)-[e1:KNOWS]-(b)-[e2:KNOWS]-(c) RETURN count(*)";
    assert_eq!(
        cypher(&db, &format!("MATCH {pattern}")),
        expected(&[&["14"]])
    );
    assert_eq!(
        rows(db.execute(&format!("MATCH DIFFERENT EDGES {pattern}"))),
        expected(&[&["14"]])
    );
}

/// The rule holds per MATCH clause: a later MATCH may bind a relationship an
/// earlier one bound (each of the four KNOWS relationships with itself, and
/// Gus's two in both orders).
#[test]
fn separate_match_clauses_may_bind_one_relationship() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a)-[r1:KNOWS]->(b) MATCH (c)-[r2:KNOWS]->(b) RETURN count(*)"
        ),
        expected(&[&["6"]])
    );
}

/// A relationship bound by an earlier clause is a relationship of the later
/// MATCH that names it: the other pattern of that MATCH binds another one.
#[test]
fn a_bound_relationship_is_unique_in_the_match_that_names_it() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH ()-[r:KNOWS {w: 1}]->() MATCH (a)-[r]->(b)<-[s:KNOWS]-(c) RETURN s.w"
        ),
        expected(&[&["4"]])
    );
}

// ---------------------------------------------------------------------------
// Variable-length patterns
// ---------------------------------------------------------------------------

/// From Gus the KNOWS trails are Gus->Vincent (2), ->Alix (2, 3), and back
/// to Gus over either of Alix's two relationships (2, 3, 1 and 2, 3, 4):
/// from there the only way on is relationship 2 again. Walks would go round
/// the triangle (12 rows up to 6 hops).
#[test]
fn a_variable_length_pattern_repeats_no_relationship() {
    let db = graph();
    let want = expected(&[&["Vincent"], &["Alix"], &["Gus"], &["Gus"]]);
    assert_eq!(
        cypher(
            &db,
            "MATCH (g:Person {name: 'Gus'})-[:KNOWS*1..6]->(c) RETURN c.name"
        ),
        want
    );
    assert_eq!(
        cypher(
            &db,
            "MATCH (g:Person {name: 'Gus'})-[:KNOWS*]->(c) RETURN c.name"
        ),
        want,
        "an unbounded pattern ends with the trails"
    );
    assert_eq!(
        cypher(
            &db,
            "MATCH p = (g:Person {name: 'Gus'})-[:KNOWS*]->(c) RETURN length(p)"
        ),
        expected(&[&["1"], &["2"], &["3"], &["3"]])
    );
}

/// A variable-length pattern takes no relationship another pattern of the
/// MATCH binds: after Alix's first hop to Gus, the way back to Gus takes
/// Alix's other relationship.
#[test]
fn a_variable_length_pattern_takes_no_relationship_of_another_pattern() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b)-[:KNOWS*1..3]->(c) RETURN r.w, c.name"
        ),
        expected(&[
            &["1", "Vincent"],
            &["1", "Alix"],
            &["1", "Gus"],
            &["4", "Vincent"],
            &["4", "Alix"],
            &["4", "Gus"],
        ])
    );
}

/// openCypher TCK Match4 [7]: a bound relationship between two
/// variable-length patterns of at most one hop, on a chain of three
/// relationships, each matched in both directions.
#[test]
fn tck_match4_7_a_bound_relationship_between_variable_length_patterns() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (n0:Node), (n1:Node), (n2:Node), (n3:Node), \
         (n0)-[:EDGE]->(n1), (n1)-[:EDGE]->(n2), (n2)-[:EDGE]->(n3)",
    )
    .unwrap();
    assert_eq!(
        cypher(
            &db,
            "MATCH ()-[r:EDGE]-() MATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m) RETURN count(p) AS c"
        ),
        expected(&[&["32"]])
    );
}

/// An unbounded pattern is no longer cut at a hundred hops: the walk cap
/// (`min + 100`) is not needed for paths that repeat no relationship.
#[test]
fn an_unbounded_pattern_reaches_past_a_hundred_hops() {
    let db = chain();
    assert_eq!(
        cypher(
            &db,
            "MATCH (a:Step {i: 0})-[:NEXT*]->(b:Step {i: 104}) RETURN count(*)"
        ),
        expected(&[&["1"]])
    );
    assert_eq!(
        cypher(
            &db,
            "MATCH (a:Step {i: 0}), (b:Step {i: 104}) RETURN EXISTS { MATCH (a)-[:NEXT*]->(b) }"
        ),
        expected(&[&["true"]]),
        "the EXISTS form agrees"
    );
    assert_eq!(
        rows(
            db.execute(
                "MATCH TRAIL (a:Step {i: 0})-[:NEXT]->{1,}(b:Step {i: 104}) RETURN count(*)"
            )
        ),
        expected(&[&["1"]]),
        "a GQL TRAIL ends on its own too"
    );
}

// ---------------------------------------------------------------------------
// Other places a pattern is matched
// ---------------------------------------------------------------------------

/// Mia's only relationship is her loop, so no two-hop pattern starts at her:
/// OPTIONAL MATCH gives nulls.
#[test]
fn optional_match_keeps_the_rule() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (m:Person {name: 'Mia'}) OPTIONAL MATCH (m)-[r1]->(x)-[r2]->(y) \
             RETURN m.name, y.name"
        ),
        expected(&[&["Mia", "null"]])
    );
}

/// The pattern of an EXISTS subquery, a pattern predicate and a pattern
/// comprehension binds each relationship once as well.
#[test]
fn subquery_patterns_keep_the_rule() {
    let db = graph();
    assert_eq!(
        cypher(
            &db,
            "MATCH (m:Person) WHERE EXISTS { MATCH (m)-[r1]->(x)-[r2]->(m) } RETURN m.name"
        ),
        expected(&[])
    );
    assert_eq!(
        cypher(&db, "MATCH (m:Person) WHERE (m)-->()-->(m) RETURN m.name"),
        expected(&[])
    );
    assert_eq!(
        cypher(
            &db,
            "MATCH (m:Person {name: 'Mia'}) RETURN [(m)-[r1]->(x)-[r2]->(y) | y.name] AS names"
        ),
        expected(&[&["[]"]])
    );
}
