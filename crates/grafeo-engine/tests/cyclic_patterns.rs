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

/// A path that returns to an earlier variable closes the cycle: it matches
/// the rows of the same path ending in a fresh variable filtered to equal it.
#[test]
fn a_path_back_to_an_earlier_variable_closes_the_cycle() {
    let db = cycles();

    // The 2-cycle 1 <-> 2, from both ends.
    assert_eq!(
        rows(db.execute("MATCH (a)-[:K]->(b)-[:K]->(a) RETURN a.n, b.n")),
        vec![vec![1, 2], vec![2, 1]]
    );
    assert_eq!(
        rows(db.execute("MATCH (a)-[:K]->(b)-[:K]->(c)-[:K]->(a) RETURN a.n, b.n, c.n")),
        triangles()
    );
    // The same cycle closed by a second MATCH.
    assert_eq!(
        rows(db.execute("MATCH (a)-[:K]->(b) MATCH (b)-[:K]->(a) RETURN a.n, b.n")),
        vec![vec![1, 2], vec![2, 1]]
    );
    // Each form agrees with the fresh variable and an equality filter.
    assert_eq!(
        rows(
            db.execute("MATCH (a)-[:K]->(b)-[:K]->(c)-[:K]->(d) WHERE d = a RETURN a.n, b.n, c.n")
        ),
        triangles()
    );
}

/// A variable-length hop back to an earlier variable closes the cycle too:
/// the same rows as ending in a fresh variable filtered to equal it.
#[test]
fn a_variable_length_path_back_to_an_earlier_variable_closes_the_cycle() {
    let db = cycles();
    let closed = rows(db.execute("MATCH (a)-[:K]->(b)-[:K]->{1,2}(a) RETURN a.n, b.n"));
    // One hop back over the 2-cycle, two hops back around each triangle.
    assert_eq!(
        closed,
        vec![
            vec![1, 2],
            vec![1, 2],
            vec![2, 1],
            vec![2, 3],
            vec![2, 3],
            vec![3, 1],
            vec![3, 4],
            vec![4, 2],
        ]
    );
    assert_eq!(
        closed,
        rows(db.execute("MATCH (a)-[:K]->(b)-[:K]->{1,2}(d) WHERE d = a RETURN a.n, b.n"))
    );
}

#[cfg(feature = "cypher")]
#[test]
fn a_cypher_path_back_to_an_earlier_variable_closes_the_cycle() {
    let db = cycles();
    assert_eq!(
        rows(db.execute_cypher("MATCH (a)-[:K]->(b)-[:K]->(a) RETURN a.n, b.n")),
        vec![vec![1, 2], vec![2, 1]]
    );
    assert_eq!(
        rows(db.execute_cypher("MATCH (a)-[:K]->(b)-[:K]->(c)-[:K]->(a) RETURN a.n, b.n, c.n")),
        triangles()
    );
}

#[cfg(feature = "sql-pgq")]
#[test]
fn a_sql_pgq_path_back_to_an_earlier_variable_closes_the_cycle() {
    let db = cycles();
    assert_eq!(
        rows(db.session().execute_sql(
            "SELECT an, bn FROM GRAPH_TABLE (MATCH (a)-[:K]->(b)-[:K]->(a) COLUMNS (a.n AS an, b.n AS bn))"
        )),
        vec![vec![1, 2], vec![2, 1]]
    );
}

/// After `WITH b` the earlier `a` is out of scope: the second `MATCH` binds a
/// new `a`, so every edge out of `b` counts, cycle or not.
#[cfg(feature = "cypher")]
#[test]
fn a_variable_out_of_scope_is_bound_again() {
    let db = cycles();
    let query = "MATCH (a)-[:K]->(b) WITH b MATCH (b)-[:K]->(a) RETURN b.n, a.n";
    assert_eq!(rows(db.execute(query)), rows(db.execute_cypher(query)));
    assert_eq!(
        rows(db.execute(query)),
        vec![
            vec![1, 2],
            vec![1, 2],
            vec![2, 1],
            vec![2, 1],
            vec![2, 3],
            vec![2, 3],
            vec![3, 1],
            vec![3, 4],
            vec![4, 2],
        ]
    );
}

/// A user variable spelled like the name the rewrite gives the closing node
/// still closes its cycle.
#[test]
fn a_cycle_on_a_variable_named_like_an_internal_one_closes() {
    let db = cycles();
    assert_eq!(
        rows(db.execute(
            "MATCH (_cycle_end_0)-[:K]->(b)-[:K]->(_cycle_end_0) RETURN _cycle_end_0.n, b.n"
        )),
        vec![vec![1, 2], vec![2, 1]]
    );
}

/// The closing node's name is not one that a later clause binds: that clause
/// scans its own nodes, so each 2-cycle row pairs with all four nodes.
#[test]
fn a_later_variable_named_like_an_internal_one_binds_on_its_own() {
    let db = cycles();
    assert_eq!(
        rows(db.execute("MATCH (a)-[:K]->(b)-[:K]->(a) MATCH (_cycle_end_0:P) RETURN count(*)")),
        vec![vec![8]]
    );
}

/// An anonymous node is a node of its own, also next to a user variable
/// spelled like the names anonymous nodes get: every edge out of `_v0` counts.
#[cfg(feature = "sql-pgq")]
#[test]
fn a_sql_pgq_anonymous_node_is_not_a_user_variable() {
    let db = cycles();
    assert_eq!(
        rows(
            db.session().execute_sql(
                "SELECT n FROM GRAPH_TABLE (MATCH (_v0)-[:K]->() COLUMNS (_v0.n AS n))"
            )
        ),
        vec![vec![1], vec![2], vec![2], vec![3], vec![3], vec![4]]
    );
}
