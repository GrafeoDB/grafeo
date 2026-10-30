//! A write sees the writes of the rows before it in the same statement,
//! while what feeds it runs in full chunks (the #677 chunk fix): the input
//! of the lowest write is read whole, then handed to the write one row at a
//! time.
//!
//! - a chain whose second write reads the first write's property of an
//!   earlier row (`SET p.b = p.a, p.a = x`: b ends as 1);
//! - a SET that accumulates over the rows (`coalesce(p.n, 0) + x`), on one
//!   node and on several, and across more rows than a chunk holds;
//! - label changes between property writes;
//! - RETURN after a write keeps the input's row order;
//! - CREATE per row, and a DELETE after a SET.
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test writes_row_at_a_time
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn one(db: &GrafeoDB, language: &str, query: &str) -> Value {
    rows(db, language, query)[0][0].clone()
}

/// One P node with `a: 0`.
fn single() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {id: 1, a: 0})").unwrap();
    db
}

/// Row 1 sets b to a (0) and a to 1; row 2 sets b to a, which row 1 wrote
/// (1), and a to 2.
#[test]
fn the_second_write_reads_the_first_writes_value_of_an_earlier_row() {
    for language in LANGUAGES {
        let db = single();
        rows(
            &db,
            language,
            "MATCH (p:P) UNWIND [1, 2] AS x SET p.b = p.a, p.a = x",
        );
        assert_eq!(
            rows(&db, language, "MATCH (p:P) RETURN p.a, p.b"),
            [[Value::Int64(2), Value::Int64(1)]],
            "{language}"
        );
    }
}

#[test]
fn a_set_accumulates_over_the_rows() {
    for language in LANGUAGES {
        let db = single();
        rows(
            &db,
            language,
            "MATCH (p:P) UNWIND [10, 20, 30] AS x SET p.n = coalesce(p.n, 0) + x",
        );
        assert_eq!(
            one(&db, language, "MATCH (p:P) RETURN p.n"),
            Value::Int64(60),
            "{language}"
        );
    }
}

/// Three nodes, each meeting the rows 1 to 4: each sums to 10.
#[test]
fn a_set_accumulates_on_several_nodes() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        db.execute("UNWIND range(1, 3) AS i INSERT (:P {id: i})")
            .unwrap();
        rows(
            &db,
            language,
            "MATCH (p:P) UNWIND range(1, 4) AS x SET p.n = coalesce(p.n, 0) + x",
        );
        assert_eq!(
            rows(&db, language, "MATCH (p:P) RETURN p.n ORDER BY p.id"),
            [[Value::Int64(10)], [Value::Int64(10)], [Value::Int64(10)]],
            "{language}"
        );
    }
}

/// More rows than one chunk: the sum of 1 to 5000, and a count of the rows.
#[test]
fn across_more_rows_than_a_chunk_holds() {
    for language in LANGUAGES {
        let db = single();
        rows(
            &db,
            language,
            "MATCH (p:P) UNWIND range(1, 5000) AS x SET p.n = coalesce(p.n, 0) + x, \
             p.c = coalesce(p.c, 0) + 1",
        );
        assert_eq!(
            rows(&db, language, "MATCH (p:P) RETURN p.n, p.c"),
            [[Value::Int64(12_502_500), Value::Int64(5000)]],
            "{language}"
        );
    }
}

/// Label changes between property writes see the rows before them.
#[test]
fn label_changes_between_property_writes() {
    for language in LANGUAGES {
        let db = single();
        rows(
            &db,
            language,
            "MATCH (p:P) UNWIND [1, 2, 3] AS x SET p:Seen REMOVE p:Seen \
             SET p.c = coalesce(p.c, 0) + 1",
        );
        assert_eq!(
            rows(&db, language, "MATCH (p:P) RETURN p.c, p:Seen"),
            [[Value::Int64(3), Value::Bool(false)]],
            "{language}"
        );
    }
}

/// RETURN after a write keeps the rows in the input's order, and the last
/// row's value is what stays.
#[test]
fn return_after_a_write_keeps_the_input_order() {
    for language in LANGUAGES {
        let db = single();
        let returned = rows(
            &db,
            language,
            "UNWIND [3, 1, 2, 5, 4] AS x MATCH (p:P {id: 1}) SET p.last = x RETURN x",
        );
        assert_eq!(
            returned,
            [3, 1, 2, 5, 4].map(|x| vec![Value::Int64(x)]),
            "{language}"
        );
        assert_eq!(
            one(&db, language, "MATCH (p:P) RETURN p.last"),
            Value::Int64(4),
            "{language}"
        );
    }
}

/// CREATE per row, then a SET over the rows on what was created.
#[test]
fn create_per_row_then_set_on_it() {
    for (language, create) in [("gql", "INSERT"), ("cypher", "CREATE")] {
        let db = GrafeoDB::new_in_memory();
        rows(
            &db,
            language,
            &format!("UNWIND range(1, 3000) AS x {create} (q:Q {{x: x}}) SET q.y = x * 2"),
        );
        assert_eq!(
            rows(&db, language, "MATCH (q:Q) RETURN count(q), sum(q.y)"),
            [[Value::Int64(3000), Value::Int64(9_003_000)]],
            "{language}"
        );
    }
}

/// A SET then a DETACH DELETE of the same nodes: the nodes are gone.
#[test]
fn a_delete_after_a_set() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        db.execute("UNWIND range(1, 10) AS i INSERT (:P {id: i})")
            .unwrap();
        rows(
            &db,
            language,
            "MATCH (p:P) WHERE p.id <= 5 SET p.gone = true DETACH DELETE p",
        );
        assert_eq!(
            one(&db, language, "MATCH (p:P) RETURN count(p)"),
            Value::Int64(5),
            "{language}"
        );
    }
}
