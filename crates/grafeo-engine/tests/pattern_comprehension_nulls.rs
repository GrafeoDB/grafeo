//! A pattern comprehension has one item per match, a null projection among
//! them: `[(n)-[:T]->(b) | b.name]` is `[null]` for one match whose `b` has no
//! name (openCypher TCK 2024.3 Pattern2 [4] [5]). The comprehension collects
//! its projection, and `collect` skips nulls, so such a match was dropped.
//! `collect` itself still skips them.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main pattern_comprehension_nulls::
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// `k` 1 to 3: 1 points at 2, which has a name, 2 at 3, which has none; 4
/// points at 5 (no name), 6 (`'x'`) and 7 (no name), its edges named alike.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (a {k: 1}), (b {k: 2, name: 'val'}), (c {k: 3}), \
         (a)-[:T {name: 'val'}]->(b), (b)-[:T]->(c)",
    )
    .unwrap();
    db.execute_cypher(
        "CREATE (d {k: 4}), (d)-[:U {i: 1}]->({k: 5}), (d)-[:U {i: 2, name: 'x'}]->({k: 6, name: 'x'}), \
         (d)-[:U {i: 3}]->({k: 7})",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute_cypher(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn list(items: &[Value]) -> Value {
    Value::List(items.to_vec().into())
}

/// A match whose projection is null gives a null item, for a new node and a
/// new relationship variable (TCK Pattern2 [4] [5]).
#[test]
fn a_null_projection_is_an_item() {
    let db = database();
    let val = Value::from("val");
    for query in [
        "MATCH (n) WHERE n.k <= 3 RETURN n.k AS k, [(n)-[:T]->(b) | b.name] AS list ORDER BY k",
        "MATCH (n) WHERE n.k <= 3 RETURN n.k AS k, [(n)-[r:T]->() | r.name] AS list ORDER BY k",
        // The same through a WITH.
        "MATCH (n) WHERE n.k <= 3 WITH n.k AS k, [(n)-[:T]->(b) | b.name] AS list \
         RETURN k, list ORDER BY k",
    ] {
        assert_eq!(
            rows(&db, query),
            vec![
                vec![Value::Int64(1), list(std::slice::from_ref(&val))],
                vec![Value::Int64(2), list(&[Value::Null])],
                vec![Value::Int64(3), list(&[])],
            ],
            "{query}"
        );
    }
}

/// Nulls and values mixed keep the order of the matches: the names are at
/// the places of their matches' `i`, read by the same comprehension.
#[test]
fn nulls_keep_the_order_of_the_matches() {
    let db = database();
    let found = rows(
        &db,
        "MATCH (d {k: 4}) RETURN [(d)-[r:U]->() | r.i] AS i, [(d)-[r:U]->() | r.name] AS names",
    );
    let (Value::List(order), Value::List(names)) = (&found[0][0], &found[0][1]) else {
        panic!("{found:?}");
    };
    let expected: Vec<Value> = order
        .iter()
        .map(|i| match i {
            Value::Int64(2) => Value::from("x"),
            _ => Value::Null,
        })
        .collect();
    assert_eq!(names.as_ref(), expected.as_slice(), "{found:?}");
    assert_eq!(names.len(), 3, "{found:?}");
}

/// In a WHERE (the comprehension lowered to a subquery), a null item counts:
/// `2`'s one match has no name, and its list has size 1.
#[test]
fn a_null_item_counts_in_a_where() {
    let db = database();
    assert_eq!(
        rows(
            &db,
            "MATCH (n) WHERE n.k <= 3 AND size([(n)-[:T]->(b) | b.name]) = 1 \
             RETURN n.k AS k ORDER BY k"
        ),
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]]
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (d {k: 4}) WITH d WHERE size([(d)-[:U]->(m) | m.name]) = 3 RETURN d.k AS k"
        ),
        vec![vec![Value::Int64(4)]]
    );
}

/// `collect` still skips nulls.
#[test]
fn collect_still_skips_nulls() {
    let db = database();
    assert_eq!(
        rows(
            &db,
            "MATCH (d {k: 4})-[:U]->(m) RETURN collect(m.name) AS names"
        ),
        vec![vec![list(&[Value::from("x")])]]
    );
}
