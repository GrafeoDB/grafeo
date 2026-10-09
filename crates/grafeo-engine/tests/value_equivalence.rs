//! Grouping keys and DISTINCT use equivalence; min and max use the order of
//! ORDER BY; UNWIND keeps its list variable.
//!
//! Grouping and DISTINCT take two values for one when they are the same value
//! (openCypher "equivalence"; ISO/IEC 39075:2024 "not distinct"): numbers by
//! their value, so 3 and 3.0 are one and -0.0 is 0.0, NaN is NaN, and lists,
//! maps, paths and vectors item by item. A grouping key keeps its value: the
//! first of its group, in its own type. min and max order their values as
//! ORDER BY does (openCypher orderability), whatever their types.
//!
//! The spec tests (`tests/spec/rosetta/grouping_and_distinct_equivalence.gtest`,
//! `min_max_over_mixed_values.gtest`, `unwind_keeps_its_list.gtest`,
//! `two_argument_aggregates.gtest`) cover the values as text; these check the
//! value types they cannot see, a database with a spill path, SPARQL's MIN
//! and MAX, and SQL/PGQ's two-argument aggregates.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --all-features --test value_equivalence
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.session()
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
        .rows()
        .to_vec()
}

/// The languages that run the GQL and Cypher forms of a query.
fn languages() -> Vec<&'static str> {
    if cfg!(feature = "cypher") {
        vec!["gql", "cypher"]
    } else {
        vec!["gql"]
    }
}

/// The rows sorted by their text, so they compare whatever the group order.
fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

// ============================================================================
// Grouping keys keep their value and their type
// ============================================================================

/// A float key comes back as the float, at the end of the query (an
/// aggregate the query ends with) and before another clause; it came back as
/// the integer of its bits, 1.5 as 4609434218613702656.
#[test]
fn a_float_grouping_key_stays_a_float() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        for query in [
            "UNWIND [1.5, 1.5, 2.5] AS x RETURN x, count(*) AS c",
            "UNWIND [1.5, 1.5, 2.5] AS x WITH x, count(*) AS c RETURN x, c",
            "UNWIND [1.5, 1.5, 2.5] AS x WITH x, count(*) AS c RETURN x, c ORDER BY c",
        ] {
            assert_eq!(
                sorted(rows(&db, language, query)),
                [
                    vec![Value::Float64(1.5), Value::Int64(2)],
                    vec![Value::Float64(2.5), Value::Int64(1)],
                ],
                "{language}: {query}"
            );
        }
    }
}

/// 3 and 3.0 are one group, and the group keeps the first of them, in its
/// type: an integer when 3 came first, a float when 3.0 did.
#[test]
fn an_integer_and_a_float_of_one_value_are_one_group_that_keeps_the_first() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        for (list, first) in [
            ("[3, 3.0]", Value::Int64(3)),
            ("[3.0, 3]", Value::Float64(3.0)),
        ] {
            for query in [
                format!("UNWIND {list} AS x RETURN x, count(*) AS c"),
                format!("UNWIND {list} AS x WITH x, count(*) AS c RETURN x, c"),
            ] {
                assert_eq!(
                    rows(&db, language, &query),
                    [vec![first.clone(), Value::Int64(2)]],
                    "{language}: {query}"
                );
            }
        }
    }
}

/// With a spill path a grouped aggregate files its groups in partitions that
/// can spill to disk, under the representatives of their keys: the same
/// values still meet in one group there, which keeps the first of them.
#[cfg(feature = "spill")]
#[test]
fn equal_grouping_keys_are_one_group_with_a_spill_path_too() {
    let spill = tempfile::tempdir().unwrap();
    let db =
        GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_spill_path(spill.path()))
            .unwrap();
    for language in languages() {
        let query =
            "UNWIND [3, 3.0, -0.0, 0.0, 19, 3] AS x RETURN x, count(*) AS c ORDER BY c DESC, x";
        let result = rows(&db, language, query);
        assert_eq!(
            result,
            [
                vec![Value::Int64(3), Value::Int64(3)],
                vec![Value::Float64(-0.0), Value::Int64(2)],
                vec![Value::Int64(19), Value::Int64(1)],
            ],
            "{language}: {query}"
        );
        assert!(
            matches!(result[1][0], Value::Float64(zero) if zero.is_sign_negative()),
            "the group of -0.0 and 0.0 keeps -0.0, its first value: {result:?}"
        );
    }
}

/// An integer above 2^53 and the float nearest to it are different numbers,
/// so they are two groups and two distinct values: the integer must not be
/// rounded to a float to compare them.
#[test]
fn a_large_integer_and_the_float_beside_it_stay_apart() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        let query = "UNWIND [9007199254740993, 9007199254740992.0] AS x \
                     RETURN count(DISTINCT x) AS d";
        assert_eq!(
            rows(&db, language, query),
            [vec![Value::Int64(2)]],
            "{language}: {query}"
        );
        let query = "UNWIND [9007199254740993, 9007199254740992.0] AS x \
                     WITH x, count(*) AS c RETURN c";
        assert_eq!(
            rows(&db, language, query),
            [vec![Value::Int64(1)], vec![Value::Int64(1)]],
            "{language}: {query}"
        );
    }
}

/// A vector key stays a vector and groups by all its items; it became the
/// text `Vector([1; 2 dims])`, one group for every vector with the same first
/// item and length.
#[test]
fn a_vector_grouping_key_stays_a_vector() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        let query = "UNWIND [vector([1.0, 2.0]), vector([1.0, 3.0]), vector([1.0, 2.0])] AS x \
                     WITH x, count(*) AS c RETURN x, c ORDER BY c DESC";
        assert_eq!(
            rows(&db, language, query),
            [
                vec![Value::Vector(vec![1.0, 2.0].into()), Value::Int64(2)],
                vec![Value::Vector(vec![1.0, 3.0].into()), Value::Int64(1)],
            ],
            "{language}: {query}"
        );
    }
}

/// Lists and maps holding numbers of one value are one group, which keeps
/// the first of them with its own items.
#[test]
fn lists_and_maps_of_equal_numbers_are_one_group() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        let query = "UNWIND [[1.5, 3], [1.5, 3.0]] AS x WITH x, count(*) AS c RETURN x, c";
        assert_eq!(
            rows(&db, language, query),
            [vec![
                Value::List(vec![Value::Float64(1.5), Value::Int64(3)].into()),
                Value::Int64(2)
            ]],
            "{language}: {query}"
        );
        let query = "UNWIND [{k: 3.0}, {k: 3}] AS x WITH x, count(*) AS c RETURN x.k AS k, c";
        assert_eq!(
            rows(&db, language, query),
            [vec![Value::Float64(3.0), Value::Int64(2)]],
            "{language}: {query}"
        );
    }
}

// ============================================================================
// DISTINCT over paths
// ============================================================================

/// Alix knows Gus, Vincent and Jules, who each know Mia: three paths of
/// length 2 from Alix to Mia.
fn three_paths() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})-[:KNOWS]->(m:Person {name: 'Mia'}), \
             (a)-[:KNOWS]->(:Person {name: 'Vincent'})-[:KNOWS]->(m), \
             (a)-[:KNOWS]->(:Person {name: 'Jules'})-[:KNOWS]->(m)",
        )
        .unwrap();
    db
}

/// DISTINCT tells paths apart by their nodes and edges; it took every path
/// of one length for one, so count(DISTINCT p) over the three paths gave 1.
#[test]
fn distinct_tells_paths_of_one_length_apart() {
    let db = three_paths();
    let pattern =
        "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(:Person {name: 'Mia'})";
    for language in languages() {
        for (rest, expected) in [
            ("RETURN count(DISTINCT p) AS d", 3),
            ("RETURN size(collect(DISTINCT p)) AS d", 3),
            ("WITH DISTINCT p RETURN count(*) AS d", 3),
            ("WITH p, count(*) AS c RETURN count(*) AS d", 3),
        ] {
            let query = format!("{pattern} {rest}");
            assert_eq!(
                rows(&db, language, &query),
                [vec![Value::Int64(expected)]],
                "{language}: {query}"
            );
        }
        let query = format!("{pattern} RETURN DISTINCT p");
        let paths = rows(&db, language, &query);
        assert_eq!(paths.len(), 3, "{language}: {query}: {paths:?}");
        assert!(
            paths
                .iter()
                .all(|row| matches!(row[..], [Value::Path { .. }])),
            "{language}: {query}: {paths:?}"
        );
    }
}

// ============================================================================
// min and max use the order of ORDER BY
// ============================================================================

/// min and max of values of different types do not depend on the input
/// order: a string comes before numbers, so min is the string and max the
/// largest number.
#[test]
fn min_and_max_over_mixed_types_do_not_depend_on_the_input_order() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        for list in ["[3, 'a', 1]", "['a', 3, 1]", "[1, 3, 'a']"] {
            for query in [
                format!("UNWIND {list} AS x RETURN min(x) AS lo, max(x) AS hi"),
                format!("UNWIND {list} AS x WITH min(x) AS lo, max(x) AS hi RETURN lo, hi"),
            ] {
                assert_eq!(
                    rows(&db, language, &query),
                    [vec![Value::String("a".into()), Value::Int64(3)]],
                    "{language}: {query}"
                );
            }
        }
    }
}

/// NaN is the largest number, so it is max; min skips nothing but nulls.
#[test]
fn max_of_numbers_with_nan_is_nan() {
    let db = GrafeoDB::new_in_memory();
    for language in languages() {
        let query = "UNWIND [3, 0.0 / 0.0, 19, null] AS x RETURN min(x) AS lo, max(x) AS hi";
        let result = rows(&db, language, query);
        assert_eq!(result[0][0], Value::Int64(3), "{language}: {query}");
        assert!(
            matches!(result[0][1], Value::Float64(hi) if hi.is_nan()),
            "{language}: {query}: {result:?}"
        );
    }
}

/// SPARQL keeps comparing literals that read as numbers by their numbers:
/// its rows hold RDF literals as text, and `9` is less than `10`.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_min_and_max_compare_numeric_literals_as_numbers() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/a> <http://ex.org/v> 10 . \
         <http://ex.org/b> <http://ex.org/v> 9 . <http://ex.org/c> <http://ex.org/v> 88 . }",
    )
    .unwrap();
    let result = db
        .execute_sparql(
            "SELECT (MIN(?v) AS ?lo) (MAX(?v) AS ?hi) WHERE { ?s <http://ex.org/v> ?v }",
        )
        .unwrap();
    let values: Vec<String> = result.rows()[0].iter().map(ToString::to_string).collect();
    assert_eq!(values, ["\"9\"", "\"88\""], "{:?}", result.rows());
}

// ============================================================================
// UNWIND keeps its list variable
// ============================================================================

/// The variable that holds the list keeps its value after UNWIND (GQL FOR);
/// it read null.
#[test]
fn unwind_keeps_the_variable_that_holds_its_list() {
    let db = GrafeoDB::new_in_memory();
    let list = Value::List(vec![Value::Int64(3), Value::Int64(19)].into());
    let mut queries = vec![
        ("gql", "LET l = [3, 19] FOR x IN l RETURN l, x"),
        ("gql", "LET l = [3, 19] UNWIND l AS x RETURN l, x"),
    ];
    if cfg!(feature = "cypher") {
        queries.push(("cypher", "WITH [3, 19] AS l UNWIND l AS x RETURN l, x"));
    }
    for (language, query) in queries {
        assert_eq!(
            rows(&db, language, query),
            [
                vec![list.clone(), Value::Int64(3)],
                vec![list.clone(), Value::Int64(19)],
            ],
            "{language}: {query}"
        );
    }
}

// ============================================================================
// SQL/PGQ aggregates that take two arguments
// ============================================================================

/// COVAR_SAMP, REGR_COUNT and PERCENTILE_DISC read their second argument;
/// SQL/PGQ dropped it, so they gave null, 0 and the median.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_two_argument_aggregates_read_both_arguments() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:P {x: 3, y: 9}), (:P {x: 19, y: 41}), (:P {x: 88, y: 179})")
        .unwrap();
    let result = rows(
        &db,
        "sql",
        "SELECT COVAR_SAMP(y, x) AS cs, REGR_COUNT(y, x) AS n, PERCENTILE_DISC(x, 1.0) AS hi \
         FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.x AS x, p.y AS y))",
    );
    let Value::Float64(covariance) = result[0][0] else {
        panic!("COVAR_SAMP: {result:?}");
    };
    assert!(
        (covariance - 12_242.0 / 3.0).abs() < 1e-9,
        "COVAR_SAMP: {result:?}"
    );
    assert_eq!(result[0][1], Value::Int64(3), "REGR_COUNT: {result:?}");
    assert_eq!(
        result[0][2],
        Value::Float64(88.0),
        "PERCENTILE_DISC: {result:?}"
    );
}
