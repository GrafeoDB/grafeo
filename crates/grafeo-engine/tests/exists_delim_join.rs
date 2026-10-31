//! A top-level EXISTS runs as a delim join (DuckDB's dependent join): the
//! subquery runs once over the distinct values of the variables it shares
//! with the outer rows, and the result joins back to every outer row. The
//! rows are those of evaluating the EXISTS per row: several EXISTS sharing
//! `n`, an EXISTS sharing `n` and `m`, NOT EXISTS, outer rows that repeat a
//! value (each kept once, as itself), an empty outer. A subquery that would
//! fail for a node no outer row holds does not run for it. PROFILE shows the
//! subquery's scan and expand, with the row counts of one run over the
//! distinct values, not one run per outer row. In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test exists_delim_join
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Six Methods (`i` 1001 to 1006; 1001 to 1004 active with `zero: 1`, 1005
/// and 1006 inactive with `zero: 0`), each with an `R` edge to a Target
/// (`i` 1 to 6) and three CALLS edges to Calls (`k` 1 to 3). USES: 1001 and
/// 1002 use each other, 1002 uses 1003, 1003 uses 1004, 1004 uses itself,
/// 1005 uses 1001. And 200 plain Graph nodes (`i` 1 to 200, `zero: 0`).
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 200) AS i INSERT (:Graph {i: i, zero: 0})")
        .unwrap();
    db.execute(concat!(
        "UNWIND range(1, 6) AS k ",
        "INSERT (:Graph:Method {i: 1000 + k, active: k <= 4, zero: CASE WHEN k <= 4 THEN 1 ELSE 0 END})",
        "-[:R]->(:Target {i: k})"
    ))
    .unwrap();
    db.execute("MATCH (n:Method) UNWIND range(1, 3) AS k INSERT (n)-[:CALLS]->(:Call {k: k})")
        .unwrap();
    for (from, to) in [
        (1001, 1002),
        (1002, 1001),
        (1002, 1003),
        (1003, 1004),
        (1004, 1004),
        (1005, 1001),
    ] {
        db.execute(&format!(
            "MATCH (a:Method {{i: {from}}}), (b:Method {{i: {to}}}) INSERT (a)-[:USES]->(b)"
        ))
        .unwrap();
    }
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn ints(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

/// Several EXISTS sharing `n`, with a plain conjunct.
#[test]
fn several_exists_sharing_a_variable() {
    let db = database();
    let query = concat!(
        "MATCH (n:Method) WHERE n.active ",
        "AND EXISTS { MATCH (n)-[:R]->(t) WHERE t.i > 1 } ",
        "AND EXISTS { MATCH (n)-[:USES]->() } ",
        "RETURN n.i AS i ORDER BY i"
    );
    for language in LANGUAGES {
        assert_eq!(
            rows(&db, language, query),
            ints(&[1002, 1003, 1004]),
            "{language}"
        );
    }
}

/// An EXISTS sharing two variables: the pairs whose USES goes both ways.
#[test]
fn an_exists_sharing_two_variables() {
    let db = database();
    let query = concat!(
        "MATCH (n:Method)-[:USES]->(m:Method) WHERE EXISTS { MATCH (m)-[:USES]->(n) } ",
        "RETURN n.i AS a, m.i AS b ORDER BY a, b"
    );
    let pair = |a: i64, b: i64| vec![Value::Int64(a), Value::Int64(b)];
    for language in LANGUAGES {
        assert_eq!(
            rows(&db, language, query),
            [pair(1001, 1002), pair(1002, 1001), pair(1004, 1004)],
            "{language}"
        );
    }
}

/// NOT EXISTS with a WHERE: the Methods that use no Method above 1003.
#[test]
fn not_exists() {
    let db = database();
    let query = concat!(
        "MATCH (n:Method) WHERE NOT EXISTS { MATCH (n)-[:USES]->(m) WHERE m.i > 1003 } ",
        "RETURN n.i AS i ORDER BY i"
    );
    for language in LANGUAGES {
        assert_eq!(
            rows(&db, language, query),
            ints(&[1001, 1002, 1005, 1006]),
            "{language}"
        );
    }
}

/// Outer rows that repeat `n` (one per CALLS edge) each stay once: the
/// join gives each outer row back, not once per match of the subquery and
/// not once per distinct value.
#[test]
fn outer_rows_with_repeated_values_stay_once() {
    let db = database();
    for language in LANGUAGES {
        let query = concat!(
            "MATCH (n:Method)-[:CALLS]->(c) WHERE EXISTS { MATCH (n)-[:USES]->() } ",
            "RETURN n.i AS i, c.k AS k ORDER BY i, k"
        );
        let found = rows(&db, language, query);
        let expected: Vec<Vec<Value>> = (1001..=1005)
            .flat_map(|i| (1..=3).map(move |k| vec![Value::Int64(i), Value::Int64(k)]))
            .collect();
        assert_eq!(found, expected, "{language}");
        // 1001 has two USES matches (1001 is used by 1002 and 1005, uses
        // 1002); a semi join keeps its rows once all the same.
        let query = concat!(
            "MATCH (n:Method)-[:CALLS]->(c) WHERE EXISTS { MATCH (n)<-[:USES]-() } ",
            "RETURN count(*) AS rows"
        );
        assert_eq!(rows(&db, language, query), ints(&[12]), "{language}");
    }
}

/// An empty outer: no rows, and no error from the subquery.
#[test]
fn an_empty_outer() {
    let db = database();
    for language in LANGUAGES {
        for query in [
            "MATCH (n:Missing) WHERE EXISTS { MATCH (n)-[:USES]->() } RETURN count(n) AS c",
            concat!(
                "MATCH (n:Method) WHERE n.i > 5000 ",
                "AND EXISTS { MATCH (m:Graph) WHERE 10 / m.zero > n.i } RETURN count(n) AS c"
            ),
        ] {
            assert_eq!(
                rows(&db, language, query),
                ints(&[0]),
                "{language} `{query}`"
            );
        }
    }
}

/// The subquery divides by `n.zero`, 0 on the inactive Methods only: it
/// runs for the active outer rows, never for a node no outer row holds.
#[test]
fn the_subquery_never_runs_for_a_node_no_outer_row_holds() {
    let db = database();
    for language in LANGUAGES {
        for query in [
            concat!(
                "MATCH (n:Method) WHERE n.active ",
                "AND EXISTS { MATCH (n)-[:R]->(t) WHERE 10 / n.zero > 0 } RETURN count(n) AS c"
            ),
            concat!(
                "MATCH (n:Method) WHERE n.active ",
                "AND EXISTS { MATCH (t:Target) WHERE t.i = n.i - 1000 AND 10 / n.zero > 0 } ",
                "RETURN count(n) AS c"
            ),
        ] {
            assert_eq!(
                rows(&db, language, query),
                ints(&[4]),
                "{language} `{query}`"
            );
        }
    }
}

/// The row counts PROFILE gives the lines that contain `needle`.
fn profiled_rows(text: &str, needle: &str) -> Vec<u64> {
    text.lines()
        .filter(|line| line.contains(needle))
        .filter_map(|line| {
            let rest = line.split("rows=").nth(1)?;
            rest.split_whitespace().next()?.parse().ok()
        })
        .collect()
}

fn profile(db: &GrafeoDB, language: &str, query: &str) -> String {
    let result = rows(db, language, &format!("PROFILE {query}"));
    match &result[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language} PROFILE `{query}`: {other:?}"),
    }
}

/// With every Method repeated three times in the outer rows, the subquery's
/// expand of `R` runs once per distinct Method: 6 rows, not 18.
#[test]
fn profile_shows_one_run_over_the_distinct_values() {
    let db = database();
    let query = concat!(
        "MATCH (n:Method)-[:CALLS]->(c) WHERE EXISTS { MATCH (n)-[:R]->(t) WHERE t.i > 2 } ",
        "RETURN n.i AS i"
    );
    for language in LANGUAGES {
        let text = profile(&db, language, query);
        assert_eq!(
            profiled_rows(&text, "[:R]"),
            [6],
            "{language}: the subquery's expand, one run per distinct n: {text}"
        );
        assert!(
            text.lines().next().unwrap_or_default().contains("rows=12"),
            "{language}: {text}"
        );
    }
}

/// A subquery that shares `n` only through its WHERE scans its label once
/// for all outer rows (206 Graph nodes), not once per outer row (18 times).
#[test]
fn profile_shows_one_scan_for_a_subquery_correlated_in_its_where() {
    let db = database();
    let query = concat!(
        "MATCH (n:Method)-[:CALLS]->(c) WHERE EXISTS { MATCH (m:Graph) WHERE m.i = n.i - 999 } ",
        "RETURN n.i AS i"
    );
    for language in LANGUAGES {
        assert_eq!(
            rows(&db, language, &format!("{query} ORDER BY i")).len(),
            18,
            "{language}"
        );
        let text = profile(&db, language, query);
        let scanned: u64 = profiled_rows(&text, "(m:Graph)").iter().sum();
        assert!(
            (1..=206).contains(&scanned),
            "{language}: the subquery's scan of Graph, {scanned} rows: {text}"
        );
    }
}
