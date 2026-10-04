//! Without `ORDER BY` the row order is unspecified. The `shuffle_unordered`
//! test option makes that visible: every result without `ORDER BY` comes back
//! in random order, and ordered results stay as they are.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test row_order
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// A database with the option on and the values 0 to 49 on `:A` nodes.
fn shuffled_database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_shuffle_unordered(true)).unwrap();
    db.execute("UNWIND range(0, 49) AS v INSERT (:A {v: v})")
        .unwrap();
    db
}

/// The different orders of the first column over five runs.
fn orders(run: impl Fn() -> Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    let mut orders: Vec<Vec<Value>> = Vec::new();
    for _ in 0..5 {
        let order: Vec<Value> = run().into_iter().map(|row| row[0].clone()).collect();
        if !orders.contains(&order) {
            orders.push(order);
        }
    }
    orders
}

fn gql(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query).unwrap().rows().to_vec()
}

fn ints(values: impl IntoIterator<Item = i64>) -> Vec<Value> {
    values.into_iter().map(Value::Int64).collect()
}

#[test]
fn results_without_order_by_are_shuffled() {
    let db = shuffled_database();
    let unordered = orders(|| gql(&db, "MATCH (n:A) RETURN n.v"));
    assert!(unordered.len() > 1, "five runs gave one order");
    for order in &unordered {
        let mut sorted = order.clone();
        sorted.sort_by_key(|value| value.as_int64());
        assert_eq!(sorted, ints(0..50), "every row once");
    }
    // Grouped results have no order either.
    let groups = orders(|| gql(&db, "MATCH (n:A) RETURN n.v % 10 AS g, count(n)"));
    assert!(groups.len() > 1);
}

/// A stream is shuffled one chunk at a time, so it keeps its bounded memory:
/// over several chunks, each chunk holds the same rows in every run (the rows
/// the scan put there), in an order that changes, and every row comes back.
/// A shuffle of the whole result would move rows between chunks.
#[test]
fn streamed_results_are_shuffled_per_chunk() {
    let db = GrafeoDB::with_config(Config::in_memory().with_shuffle_unordered(true)).unwrap();
    db.execute("UNWIND range(0, 4999) AS v INSERT (:A {v: v})")
        .unwrap();
    let chunks = || {
        let mut stream = db.execute_streaming("MATCH (n:A) RETURN n.v").unwrap();
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next_chunk().unwrap() {
            let column = chunk.column(0).unwrap();
            let values: Vec<i64> = chunk
                .selected_indices()
                .map(|row| column.get_value(row).and_then(|v| v.as_int64()).unwrap())
                .collect();
            chunks.push(values);
        }
        chunks
    };
    let sorted = |chunks: &[Vec<i64>]| -> Vec<Vec<i64>> {
        chunks
            .iter()
            .map(|chunk| {
                let mut chunk = chunk.clone();
                chunk.sort_unstable();
                chunk
            })
            .collect()
    };

    let first = chunks();
    assert!(first.len() > 1, "{} chunks", first.len());
    let mut all: Vec<i64> = first.iter().flatten().copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..5000).collect::<Vec<_>>(), "every row once");

    let runs: Vec<Vec<Vec<i64>>> = (0..4).map(|_| chunks()).collect();
    for run in &runs {
        assert_eq!(sorted(run), sorted(&first), "each chunk keeps its rows");
    }
    assert!(
        runs.iter().any(|run| *run != first),
        "five runs gave one order"
    );
}

#[test]
fn ordered_results_keep_their_order() {
    let db = shuffled_database();
    for (query, expected) in [
        ("MATCH (n:A) RETURN n.v ORDER BY n.v", ints(0..50)),
        (
            "MATCH (n:A) RETURN n.v ORDER BY n.v DESC LIMIT 5",
            ints([49, 48, 47, 46, 45]),
        ),
        ("MATCH (n:A) RETURN n.v ORDER BY n.v SKIP 45", ints(45..50)),
        (
            "MATCH (n:A) RETURN DISTINCT n.v % 5 AS r ORDER BY r",
            ints(0..5),
        ),
    ] {
        assert_eq!(orders(|| gql(&db, query)), [expected], "{query}");
    }
}

#[test]
fn the_option_is_off_by_default() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 49) AS v INSERT (:A {v: v})")
        .unwrap();
    assert_eq!(orders(|| gql(&db, "MATCH (n:A) RETURN n.v")).len(), 1);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_results_are_shuffled_too() {
    let db = shuffled_database();
    let session = db.session();
    let cypher = |query: &str| session.execute_cypher(query).unwrap().rows().to_vec();
    assert!(orders(|| cypher("MATCH (n:A) RETURN n.v")).len() > 1);
    assert_eq!(
        orders(|| cypher("MATCH (n:A) RETURN n.v ORDER BY n.v")),
        [ints(0..50)]
    );
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_results_are_shuffled_too() {
    let db = GrafeoDB::with_config(Config::in_memory().with_shuffle_unordered(true)).unwrap();
    for i in 0..30 {
        db.execute_sparql(&format!(
            "INSERT DATA {{ <http://ex/s{i}> <http://ex/v> \"{i:02}\" }}"
        ))
        .unwrap();
    }
    let sparql = |query: &str| db.execute_sparql(query).unwrap().rows().to_vec();
    assert!(orders(|| sparql("SELECT ?v WHERE { ?s <http://ex/v> ?v }")).len() > 1);
    assert_eq!(
        orders(|| sparql("SELECT ?v WHERE { ?s <http://ex/v> ?v } ORDER BY ?v")).len(),
        1
    );
}
