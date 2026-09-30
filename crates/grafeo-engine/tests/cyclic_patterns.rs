//! Cyclic patterns match exactly the rows their acyclic form with an
//! equality filter matches: a triangle written as comma-separated patterns,
//! and a path that returns to an earlier variable.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test cyclic_patterns
//! ```

use grafeo_engine::GrafeoDB;

/// Four `:P` nodes `n: 1..4` with `:K` edges 1->2, 2->3, 3->1, 3->4, 4->2 and
/// 2->1: two directed triangles (1-2-3 and 2-3-4) and one 2-cycle (1-2).
fn cycles() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {n: 1}), (:P {n: 2}), (:P {n: 3}), (:P {n: 4})")
        .unwrap();
    for (a, b) in [(1, 2), (2, 3), (3, 1), (3, 4), (4, 2), (2, 1)] {
        db.execute(&format!(
            "MATCH (a:P {{n: {a}}}), (b:P {{n: {b}}}) INSERT (a)-[:K]->(b)"
        ))
        .unwrap();
    }
    db
}

/// The rows of `query`, each value an integer, sorted.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Vec<i64>> {
    let mut rows: Vec<Vec<i64>> = result
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| {
                    value
                        .as_int64()
                        .unwrap_or_else(|| panic!("expected an integer, got {value:?}"))
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

/// Every rotation of both directed triangles, as `[a, b, c]`.
fn triangles() -> Vec<Vec<i64>> {
    vec![
        vec![1, 2, 3],
        vec![2, 3, 1],
        vec![2, 3, 4],
        vec![3, 1, 2],
        vec![3, 4, 2],
        vec![4, 2, 3],
    ]
}

/// Three comma-separated patterns that close a triangle form a cyclic join;
/// it must find each triangle row once and keep the `WHERE`.
#[test]
fn a_triangle_of_comma_patterns_matches_each_triangle_once() {
    let db = cycles();
    let triangle = "MATCH (a:P)-[:K]->(b), (b)-[:K]->(c), (c)-[:K]->(a)";

    assert_eq!(
        rows(db.execute(&format!("{triangle} RETURN a.n, b.n, c.n"))),
        triangles()
    );
    assert_eq!(
        rows(db.execute(&format!("{triangle} WHERE b.n = 3 RETURN a.n, b.n, c.n"))),
        vec![vec![2, 3, 1], vec![2, 3, 4]]
    );
    #[cfg(feature = "cypher")]
    assert_eq!(
        rows(db.execute_cypher(&format!("{triangle} RETURN a.n, b.n, c.n"))),
        triangles()
    );
}
