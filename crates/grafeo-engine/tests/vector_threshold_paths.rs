//! A similarity or distance filter near or at an exact threshold gives the
//! same nodes whichever way it runs: through the vector index (`VectorScan`,
//! the filter on a label scan of an indexed property), per row (the same
//! filter after an input row, which the index cannot serve), and without an
//! index. The thresholds are the per-row values of the nodes themselves,
//! one ulp to either side, and offsets inside the scan's own margin, so each
//! filter sits on or right next to a node's value. The index computed the
//! value another way than the per-row function, so a node at its own
//! threshold was in one answer and not the other.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test vector_threshold_paths
//! ```

#![cfg(all(feature = "gql", feature = "vector-index"))]

use std::collections::BTreeSet;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const QUERY_VECTOR: &str = "[0.3, -0.7, 0.2]";
const NODES: usize = 200;
/// The nodes whose values serve as thresholds (every fifth one).
const STEP: usize = 5;

/// The 3-dimensional vectors of the nodes, three decimals each, from a fixed
/// linear congruential sequence.
fn vectors() -> Vec<[f64; 3]> {
    let mut state: u64 = 0x5EED_1946;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // A value in [-1, 1] with three decimals.
        let thousandths = i64::try_from((state >> 33) % 2001).unwrap() - 1000;
        f64::from(i32::try_from(thousandths).unwrap()) / 1000.0
    };
    (0..NODES).map(|_| [next(), next(), next()]).collect()
}

/// A database holding the nodes, with a vector index of `metric` on `emb`
/// when one is given.
fn database(metric: Option<&str>) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if let Some(metric) = metric {
        db.execute(&format!(
            "CREATE VECTOR INDEX d_emb ON :D(emb) DIMENSION 3 METRIC '{metric}'"
        ))
        .unwrap();
    }
    for (i, v) in vectors().iter().enumerate() {
        db.execute(&format!(
            "INSERT (:D {{i: {i}, emb: vector([{:?}, {:?}, {:?}])}})",
            v[0], v[1], v[2]
        ))
        .unwrap();
    }
    db
}

/// The `i` of each row of `query`, in order.
fn ids(db: &GrafeoDB, query: &str) -> Vec<i64> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(i) => i,
            ref other => panic!("{query}: i is {other:?}"),
        })
        .collect()
}

/// Each node's value of `function` against the query vector, per row.
fn values(db: &GrafeoDB, function: &str) -> Vec<f64> {
    let rows = db
        .execute(&format!(
            "MATCH (d:D) RETURN d.i AS i, {function}(d.emb, {QUERY_VECTOR}) AS v ORDER BY i"
        ))
        .unwrap();
    rows.rows()
        .iter()
        .map(|row| match row[1] {
            Value::Float64(v) => v,
            ref other => panic!("{function}: {other:?}"),
        })
        .collect()
}

/// The thresholds around `value`: itself, one ulp below and above, and
/// offsets inside the scan's margin of 1e-4.
fn around(value: f64) -> Vec<f64> {
    let mut thresholds = vec![value, value.next_down(), value.next_up()];
    for offset in [1e-6, 5e-5] {
        thresholds.push(value + offset);
        thresholds.push(value - offset);
    }
    thresholds
}

/// For every fifth node, the filter `function(d.emb, q) op t` for each
/// threshold `t` around the node's own value, through the index (with
/// `metric`), per row and without an index: the cases whose answers differ.
/// `tail` follows the filter (RETURN and LIMIT).
fn disagreements(function: &str, metric: &str, ops: &[&str], tail: &str) -> Vec<String> {
    let indexed = database(Some(metric));
    let plain = database(None);
    let node_values = values(&plain, function);
    let mut wrong = Vec::new();
    for op in ops {
        for (node, value) in node_values.iter().enumerate().step_by(STEP) {
            for threshold in around(*value) {
                // Scientific notation: a float literal in GQL, exact on the way back.
                let filter = format!("{function}(d.emb, {QUERY_VECTOR}) {op} {threshold:e}");
                let scan = ids(&indexed, &format!("MATCH (d:D) WHERE {filter} {tail}"));
                let per_row = ids(
                    &indexed,
                    &format!("UNWIND [1] AS u MATCH (d:D) WHERE {filter} {tail}"),
                );
                let unindexed = ids(&plain, &format!("MATCH (d:D) WHERE {filter} {tail}"));
                if scan != per_row || scan != unindexed {
                    let set = |v: &[i64]| v.iter().copied().collect::<BTreeSet<_>>();
                    let only = |a: &[i64], b: &[i64]| -> Vec<i64> {
                        set(a).difference(&set(b)).copied().collect()
                    };
                    wrong.push(format!(
                        "{op} {threshold:e} (node {node}, value {value:e}): index only {:?}, \
                         per row only {:?}, without the index {}",
                        only(&scan, &per_row),
                        only(&per_row, &scan),
                        if unindexed == per_row {
                            "as per row"
                        } else {
                            "differs from per row"
                        }
                    ));
                }
            }
        }
    }
    wrong
}

fn assert_agree(wrong: &[String]) {
    assert!(
        wrong.is_empty(),
        "{} filters differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The index path is the vector scan for each metric, so the comparison is
/// between two ways of computing, not one way twice.
#[test]
fn the_threshold_filter_runs_through_the_vector_index() {
    for (metric, filter) in [
        (
            "cosine",
            "cosine_similarity(d.emb, [0.3, -0.7, 0.2]) >= 0.5e0",
        ),
        (
            "euclidean",
            "euclidean_distance(d.emb, [0.3, -0.7, 0.2]) <= 0.5e0",
        ),
        (
            "manhattan",
            "manhattan_distance(d.emb, [0.3, -0.7, 0.2]) <= 0.5e0",
        ),
    ] {
        let db = database(Some(metric));
        let profile = db
            .execute(&format!(
                "PROFILE MATCH (d:D) WHERE {filter} RETURN d.i AS i"
            ))
            .unwrap();
        let text = format!("{:?}", profile.rows());
        assert!(text.contains("VectorScan"), "{metric}: {text}");
        let per_row = db
            .execute(&format!(
                "PROFILE UNWIND [1] AS u MATCH (d:D) WHERE {filter} RETURN d.i AS i"
            ))
            .unwrap();
        let text = format!("{:?}", per_row.rows());
        assert!(!text.contains("VectorScan"), "{metric}: {text}");
    }
}

#[test]
fn a_cosine_threshold_gives_the_same_nodes_on_every_path() {
    assert_agree(&disagreements(
        "cosine_similarity",
        "cosine",
        &[">=", ">", "<=", "<"],
        "RETURN d.i AS i ORDER BY i",
    ));
}

#[test]
fn a_euclidean_threshold_gives_the_same_nodes_on_every_path() {
    assert_agree(&disagreements(
        "euclidean_distance",
        "euclidean",
        &["<=", "<", ">=", ">"],
        "RETURN d.i AS i ORDER BY i",
    ));
}

#[test]
fn a_manhattan_threshold_gives_the_same_nodes_on_every_path() {
    assert_agree(&disagreements(
        "manhattan_distance",
        "manhattan",
        &["<=", "<", ">=", ">"],
        "RETURN d.i AS i ORDER BY i",
    ));
}

/// A LIMIT after the threshold filter keeps the first nodes of the same
/// answer, in id order and in similarity order.
#[test]
fn a_threshold_with_a_limit_gives_the_same_first_nodes_on_every_path() {
    let mut wrong = disagreements(
        "cosine_similarity",
        "cosine",
        &[">=", ">"],
        "RETURN d.i AS i ORDER BY i LIMIT 7",
    );
    wrong.extend(disagreements(
        "cosine_similarity",
        "cosine",
        &[">="],
        &format!(
            "RETURN d.i AS i ORDER BY cosine_similarity(d.emb, {QUERY_VECTOR}) DESC, i LIMIT 5"
        ),
    ));
    wrong.extend(disagreements(
        "euclidean_distance",
        "euclidean",
        &["<="],
        &format!("RETURN d.i AS i ORDER BY euclidean_distance(d.emb, {QUERY_VECTOR}), i LIMIT 5"),
    ));
    assert_agree(&wrong);
}

/// The `(i, value)` rows of `query`, by `i`.
fn scores(db: &GrafeoDB, query: &str) -> Vec<(i64, f64)> {
    let mut rows: Vec<(i64, f64)> = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(i), Value::Float64(v)) => (*i, *v),
            other => panic!("{query}: {other:?}"),
        })
        .collect();
    rows.sort_by_key(|(i, _)| *i);
    rows
}

/// `RETURN function(...)` gives the function's value through the index, as
/// per row and without one: after a threshold filter and after a top-k plan.
/// Through a cosine index it gave the index's distance, `1 - similarity`.
#[test]
fn the_returned_score_is_the_functions_value_on_every_path() {
    let mut failures = Vec::new();
    for (function, metric, op, threshold, order) in [
        ("cosine_similarity", "cosine", ">=", "-0.2e0", "DESC"),
        ("euclidean_distance", "euclidean", "<=", "1.2e0", "ASC"),
        ("manhattan_distance", "manhattan", "<=", "1.6e0", "ASC"),
        ("dot_product", "dot_product", ">=", "-0.2e0", "DESC"),
    ] {
        let indexed = database(Some(metric));
        let plain = database(None);
        let call = format!("{function}(d.emb, {QUERY_VECTOR})");
        for (shape, tail) in [
            (
                "a threshold filter",
                format!("WHERE {call} {op} {threshold} RETURN d.i AS i, {call} AS s"),
            ),
            (
                "a threshold and a top-k",
                format!(
                    "WHERE {call} {op} {threshold} RETURN d.i AS i, {call} AS s \
                     ORDER BY {call} {order}, i LIMIT 9"
                ),
            ),
            (
                "a top-k",
                format!("RETURN d.i AS i, {call} AS s ORDER BY {call} {order}, i LIMIT 9"),
            ),
        ] {
            let scan = scores(&indexed, &format!("MATCH (d:D) {tail}"));
            let per_row = scores(&indexed, &format!("UNWIND [1] AS u MATCH (d:D) {tail}"));
            let unindexed = scores(&plain, &format!("MATCH (d:D) {tail}"));
            if scan.is_empty() {
                failures.push(format!("{function}, {shape}: no rows"));
            }
            if scan != per_row || scan != unindexed {
                failures.push(format!(
                    "{function}, {shape}: through the index {:?}, per row {:?}",
                    &scan[..scan.len().min(4)],
                    &per_row[..per_row.len().min(4)]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The values themselves, on two vectors: cosine similarity 0.6 and 1 to
/// [1, 0], never the distances 0.4 and 0, through a cosine index.
#[test]
fn a_cosine_index_returns_the_similarity_not_the_distance() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE VECTOR INDEX e ON :V(emb) DIMENSION 2 METRIC 'cosine'")
        .unwrap();
    db.execute("INSERT (:V {i: 1, emb: vector([0.6, 0.8])}), (:V {i: 2, emb: vector([1.0, 0.0])})")
        .unwrap();
    for query in [
        "MATCH (d:V) WHERE cosine_similarity(d.emb, [1.0, 0.0]) >= 0.5e0 \
         RETURN d.i AS i, cosine_similarity(d.emb, [1.0, 0.0]) AS s",
        "MATCH (d:V) WHERE cosine_similarity(d.emb, [1.0, 0.0]) >= 0.5e0 \
         RETURN d.i AS i, cosine_similarity(d.emb, [1.0, 0.0]) AS s \
         ORDER BY cosine_similarity(d.emb, [1.0, 0.0]) DESC LIMIT 2",
    ] {
        let rows = scores(&db, query);
        assert_eq!(rows.len(), 2, "{query}");
        assert!((rows[0].1 - 0.6).abs() < 1e-6, "{query}: {rows:?}");
        assert!((rows[1].1 - 1.0).abs() < 1e-6, "{query}: {rows:?}");
    }
}

/// `text_score` keeps reusing the text scan's column: its returned value
/// after a text threshold and a top-k is the per-row value.
#[cfg(feature = "text-index")]
#[test]
fn a_text_score_after_a_text_scan_is_the_per_row_value() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Doc {i: 1, body: 'canals and bridges of Amsterdam'}), \
         (:Doc {i: 2, body: 'canals canals canals'}), \
         (:Doc {i: 3, body: 'bridges of Prague'})",
    )
    .unwrap();
    db.execute("CREATE INDEX doc_body FOR (d:Doc) ON (d.body) USING TEXT")
        .unwrap();
    let tail = "WHERE text_score(d.body, 'canals') > 0.0e0 \
                RETURN d.i AS i, text_score(d.body, 'canals') AS s \
                ORDER BY text_score(d.body, 'canals') DESC, i LIMIT 2";
    let scan = scores(&db, &format!("MATCH (d:Doc) {tail}"));
    let per_row = scores(&db, &format!("UNWIND [1] AS u MATCH (d:Doc) {tail}"));
    assert_eq!(scan.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(scan, per_row);
    assert!(scan.iter().all(|(_, s)| *s > 0.0), "{scan:?}");
}
