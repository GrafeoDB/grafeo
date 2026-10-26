//! Seam tests for aggregate semantics and expression edge cases
//! (ISO/IEC 39075 Section 20.9, 21).
//!
//! Tests COUNT(*) vs COUNT(expr) semantics with NULLs, aggregates on
//! empty sets, NULLIF/COALESCE edge cases, and CASE expressions.
//!
//! ```bash
//! cargo test -p grafeo-engine --test seam_aggregates_expressions
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ============================================================================
// 1. COUNT(*) vs COUNT(expr) semantics
// ============================================================================

mod count_semantics {
    use super::*;

    #[test]
    fn count_star_counts_all_rows() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(3));
    }

    #[test]
    fn count_star_on_empty_returns_zero() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Should return one row");
        assert_eq!(result.rows()[0][0], Value::Int64(0));
    }

    #[test]
    fn count_expr_skips_nulls() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap(); // no age
        session
            .execute("INSERT (:Person {name: 'Vincent', age: 35})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(n.age) AS cnt")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(2),
            "COUNT(expr) should skip NULLs"
        );
    }

    #[test]
    fn count_star_vs_count_expr_difference() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap(); // no age

        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(*) AS total, COUNT(n.age) AS with_age")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(2),
            "COUNT(*) counts all rows"
        );
        assert_eq!(
            result.rows()[0][1],
            Value::Int64(1),
            "COUNT(n.age) skips NULL age"
        );
    }

    #[test]
    fn count_distinct() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', city: 'Amsterdam'})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent', city: 'Berlin'})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(DISTINCT n.city) AS cities")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(2));
    }
}

// ============================================================================
// 2. Aggregate NULL handling
// ============================================================================

mod aggregate_nulls {
    use super::*;

    #[test]
    fn sum_skips_nulls() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Data {val: 10})").unwrap();
        session.execute("INSERT (:Data {})").unwrap(); // no val
        session.execute("INSERT (:Data {val: 20})").unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN SUM(n.val) AS s")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(30),
            "SUM should skip NULLs"
        );
    }

    #[test]
    fn avg_skips_nulls() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Data {val: 10})").unwrap();
        session.execute("INSERT (:Data {})").unwrap();
        session.execute("INSERT (:Data {val: 20})").unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN AVG(n.val) AS a")
            .unwrap();
        // AVG(10, 20) = 15.0 (NULL is excluded from count)
        assert_eq!(result.rows()[0][0], Value::Float64(15.0));
    }

    #[test]
    fn min_skips_nulls() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Data {})").unwrap();
        session.execute("INSERT (:Data {val: 30})").unwrap();
        session.execute("INSERT (:Data {val: 10})").unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN MIN(n.val) AS m")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(10));
    }

    #[test]
    fn max_skips_nulls() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Data {})").unwrap();
        session.execute("INSERT (:Data {val: 10})").unwrap();
        session.execute("INSERT (:Data {val: 30})").unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN MAX(n.val) AS m")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(30));
    }

    #[test]
    fn collect_includes_nulls() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Data {val: 10})").unwrap();
        session.execute("INSERT (:Data {})").unwrap();
        session.execute("INSERT (:Data {val: 20})").unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN COLLECT(n.val) AS c")
            .unwrap();
        match &result.rows()[0][0] {
            Value::List(list) => {
                // COLLECT may or may not include NULLs depending on implementation.
                // The key invariant: it should be a list.
                assert!(
                    list.len() >= 2,
                    "COLLECT should have at least the non-null values"
                );
            }
            other => panic!("COLLECT should return a List, got: {other:?}"),
        }
    }
}

// ============================================================================
// 3. NULLIF and COALESCE edge cases
// ============================================================================

mod nullif_coalesce {
    use super::*;

    #[test]
    fn nullif_equal_returns_null() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN NULLIF(5, 5) AS r").unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "NULLIF(x, x) should return NULL"
        );
    }

    #[test]
    fn nullif_unequal_returns_first() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN NULLIF(5, 3) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(5));
    }

    #[test]
    fn coalesce_first_non_null() {
        let db = db();
        let session = db.session();

        let result = session
            .execute("RETURN COALESCE(NULL, NULL, 3) AS r")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(3));
    }

    #[test]
    fn coalesce_all_null() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN COALESCE(NULL, NULL) AS r").unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "COALESCE of all NULLs should be NULL"
        );
    }

    #[test]
    fn coalesce_first_value_wins() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN COALESCE(1, 2, 3) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(1));
    }
}

// ============================================================================
// 4. CASE expression edge cases
// ============================================================================

mod case_expressions {
    use super::*;

    #[test]
    fn simple_case_match() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let result = session
            .execute(
                "MATCH (n:Person) RETURN CASE n.age WHEN 30 THEN 'young' WHEN 50 THEN 'senior' ELSE 'unknown' END AS category",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("young".into()));
    }

    #[test]
    fn searched_case() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let result = session
            .execute(
                "MATCH (n:Person) RETURN CASE WHEN n.age < 20 THEN 'teen' WHEN n.age < 40 THEN 'adult' ELSE 'senior' END AS category",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("adult".into()));
    }

    #[test]
    fn case_else_default() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session
            .execute(
                "MATCH (n:Person) RETURN CASE WHEN n.age > 50 THEN 'old' ELSE 'no age' END AS r",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("no age".into()));
    }

    #[test]
    fn case_no_else_returns_null() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN CASE WHEN n.age > 50 THEN 'old' END AS r")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "CASE with no ELSE and no match should return NULL"
        );
    }
}

// ============================================================================
// 5. Type coercion and casting edge cases
// ============================================================================

mod type_coercion {
    use super::*;

    #[test]
    fn cast_int_to_string() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST(42 AS STRING) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("42".into()));
    }

    #[test]
    fn cast_string_to_int() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST('42' AS INT64) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(42));
    }

    #[test]
    fn cast_float_to_int_truncates() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST(3.9 AS INT64) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(3));
    }

    #[test]
    fn cast_int_to_float() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST(42 AS FLOAT64) AS r").unwrap();
        assert_eq!(result.rows()[0][0], Value::Float64(42.0));
    }

    #[test]
    fn cast_bool_to_string() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST(true AS STRING) AS r").unwrap();
        match &result.rows()[0][0] {
            Value::String(s) => assert!(
                s == "true" || s == "TRUE",
                "CAST(true AS STRING) should produce 'true', got '{s}'"
            ),
            other => panic!("Expected String, got: {other:?}"),
        }
    }

    #[test]
    fn cast_null_stays_null() {
        let db = db();
        let session = db.session();

        let result = session.execute("RETURN CAST(NULL AS INT64) AS r").unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "CAST(NULL) should remain NULL"
        );
    }
}

// Aggregates inside a composite projection must be evaluated over the input,
// before the surrounding expression runs (#539).
#[cfg(all(feature = "gql", feature = "cypher"))]
mod aggregate_discovery_539 {
    use super::*;
    use std::collections::BTreeMap;

    fn list(values: Vec<Value>) -> Value {
        Value::List(values.into())
    }

    fn map(fields: Vec<(&str, Value)>) -> Value {
        Value::Map(
            fields
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect::<BTreeMap<_, _>>()
                .into(),
        )
    }

    fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
        db.execute_language(query, language, None)
            .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
            .rows()
            .to_vec()
    }

    fn projection(language: &str, expression: &str, expected: Value) {
        let db = db();
        let query = format!("UNWIND [3, 1, 2] AS n WITH n ORDER BY n RETURN {expression} AS value");
        assert_eq!(
            rows(&db, language, &query),
            vec![vec![expected]],
            "{language} `{query}`"
        );
    }

    macro_rules! language_cases {
        ($module:ident, $language:literal) => {
            mod $module {
                use super::*;

                #[test]
                fn aggregate_in_index_base() {
                    projection($language, "collect(n)[0]", Value::Int64(1));
                }

                #[test]
                fn aggregate_in_index_expression() {
                    projection($language, "['zero', 'one', 'two'][count(n) - 1]", Value::from("two"));
                }

                #[test]
                fn aggregate_in_slice_base() {
                    projection($language, "collect(n)[1..]", list(vec![Value::Int64(2), Value::Int64(3)]));
                }

                #[test]
                fn aggregate_in_slice_bounds() {
                    projection($language, "[10, 20, 30][count(n) - 2..count(n)]", list(vec![Value::Int64(20), Value::Int64(30)]));
                }

                #[test]
                fn aggregate_in_map_literal() {
                    projection($language, "{names: collect(n)}", map(vec![("names", list(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]))]));
                }

                #[test]
                fn property_access_on_collected_maps() {
                    let db = db();
                    let query = "UNWIND [{route: 'A1'}, {route: 'B2'}] AS n WITH n ORDER BY n.route RETURN head(collect(n)).route AS route";
                    assert_eq!(rows(&db, $language, query), vec![vec![Value::from("A1")]]);
                }

                #[test]
                fn aggregates_in_a_list_literal() {
                    projection($language, "[collect(n)[0], count(n)]", list(vec![Value::Int64(1), Value::Int64(3)]));
                }

                #[test]
                fn multiple_aggregates_in_nested_wrappers() {
                    projection($language, "{first: collect(n)[0], last: collect(n)[size(collect(n)) - 1]}", map(vec![("first", Value::Int64(1)), ("last", Value::Int64(3))]));
                }

                #[test]
                fn wrapped_aggregate_with_a_grouping_key() {
                    let db = db();
                    let query = "UNWIND [{city: 'A', name: 'Alix'}, {city: 'A', name: 'Gus'}, {city: 'B', name: 'Mia'}] AS n WITH n ORDER BY n.city, n.name RETURN n.city AS city, collect(n.name)[0] AS first ORDER BY city";
                    assert_eq!(rows(&db, $language, query), vec![vec![Value::from("A"), Value::from("Alix")], vec![Value::from("B"), Value::from("Mia")]]);
                }

                #[test]
                fn wrapped_aggregate_on_empty_input() {
                    let db = db();
                    let query = "UNWIND [] AS n RETURN collect(n)[0] AS first, collect(n)[..] AS all_values";
                    assert_eq!(rows(&db, $language, query), vec![vec![Value::Null, list(vec![])]]);
                }

                #[test]
                fn aggregate_in_a_comprehension_source() {
                    projection($language, "[x IN collect(n) | x + 1]", list(vec![Value::Int64(2), Value::Int64(3), Value::Int64(4)]));
                }

                #[test]
                fn aggregate_in_a_list_predicate_source() {
                    projection($language, "all(x IN collect(n) WHERE x > 0)", Value::Bool(true));
                }

                #[test]
                fn aggregate_in_a_reduce_source() {
                    projection($language, "reduce(total = 0, x IN collect(n) | total + x)", Value::Int64(6));
                }

                #[test]
                fn outer_aggregates_in_locally_bound_bodies() {
                    projection($language, "[x IN [10] | sum(n) + x]", list(vec![Value::Int64(16)]));
                    projection($language, "all(x IN [1, 2] WHERE x < sum(n))", Value::Bool(true));
                    projection($language, "reduce(total = 0, x IN [1, 2] | total + x + sum(n))", Value::Int64(15));
                }

                #[test]
                fn an_aggregate_cannot_capture_a_shadowing_local() {
                    use grafeo_common::utils::error::{Error, QueryErrorKind};
                    let db = db();
                    for (expression, name) in [
                        ("[x IN [1, 2] | sum(x)]", "x"),
                        ("all(x IN [1, 2] WHERE sum(x) > 0)", "x"),
                        ("reduce(total = 0, x IN [1, 2] | total + sum(x))", "x"),
                        ("reduce(total = 0, x IN [1, 2] | sum(total) + x)", "total"),
                    ] {
                        let query = format!("UNWIND [3, 1, 2] AS x RETURN {expression} AS value");
                        match db.execute_language(&query, $language, None) {
                            Err(Error::Query(error)) => {
                                assert_eq!(error.kind, QueryErrorKind::Semantic, "{query}");
                                assert!(error.message.contains(&format!("locally bound variable '{name}'")), "{query}: {error}");
                            }
                            other => panic!("{} `{query}`: expected a binding error, got {other:?}", $language),
                        }
                    }
                }
            }
        };
    }

    language_cases!(gql, "gql");
    language_cases!(cypher, "cypher");

    #[test]
    fn gql_aggregate_in_a_let_binding() {
        projection(
            "gql",
            "LET names = collect(n) IN names[0] END",
            Value::Int64(1),
        );
    }

    #[test]
    fn cypher_aggregate_in_a_map_projection() {
        let db = db();
        db.execute("INSERT (:P {name: 'Alix'}), (:P {name: 'Gus'})")
            .unwrap();
        let query = "MATCH (n:P) WITH n, n { .name, names: collect(n.name) } AS projected RETURN projected.name AS name, projected.names AS names ORDER BY name";
        assert_eq!(
            rows(&db, "cypher", query),
            vec![
                vec![Value::from("Alix"), list(vec![Value::from("Alix")])],
                vec![Value::from("Gus"), list(vec![Value::from("Gus")])]
            ]
        );
    }

    #[test]
    fn gql_let_aliases_keep_their_enclosing_binding() {
        projection("gql", "LET x = n IN sum(x) END", Value::Int64(6));
        projection(
            "gql",
            "LET x = n IN [n IN [10] | sum(x) + n] END",
            list(vec![Value::Int64(16)]),
        );
    }

    #[test]
    fn gql_a_let_alias_does_not_hide_a_local_aggregate_argument() {
        use grafeo_common::utils::error::{Error, QueryErrorKind};
        let db = db();
        let query = "UNWIND [3, 1, 2] AS x RETURN [x IN [1, 2] | LET y = x IN sum(y) END] AS value";
        match db.execute_language(query, "gql", None) {
            Err(Error::Query(error)) => {
                assert_eq!(error.kind, QueryErrorKind::Semantic);
                assert!(
                    error.message.contains("locally bound variable 'x'"),
                    "{error}"
                );
            }
            other => panic!("expected a local binding error, got {other:?}"),
        }
    }
}

#[cfg(all(feature = "gql", feature = "cypher"))]
mod aggregate_discovery_539_correlated_scope {
    use super::*;

    fn check(language: &str, setup: &str) {
        let db = db();
        db.execute_language(setup, language, None).unwrap();
        for (query, expected) in [
            (
                "MATCH (n:P) RETURN n, size([p IN [n] WHERE EXISTS { MATCH (p)-[:R]->() } | p]) AS score",
                vec![0, 1],
            ),
            (
                "MATCH (n:P) RETURN n, count(*) + size([p IN [n] WHERE EXISTS { MATCH (p)-[:R]->() } | p]) AS score",
                vec![1, 2],
            ),
        ] {
            let result = db
                .execute_language(query, language, None)
                .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
            assert_eq!(result.rows().len(), 2, "{language} `{query}`");
            let mut scores: Vec<i64> = result
                .rows()
                .iter()
                .map(|row| {
                    assert_eq!(row.len(), 2, "{language} `{query}`");
                    match &row[1] {
                        Value::Int64(score) => *score,
                        other => {
                            panic!("{language} `{query}`: expected integer score, got {other:?}")
                        }
                    }
                })
                .collect();
            scores.sort_unstable();
            assert_eq!(scores, expected, "{language} `{query}`");
        }
    }

    #[test]
    fn gql_wrapped_aggregate_keeps_a_correlated_comprehension_binding() {
        check("gql", "INSERT (a:P)-[:R]->(:Q), (:P)");
    }

    #[test]
    fn cypher_wrapped_aggregate_keeps_a_correlated_comprehension_binding() {
        check("cypher", "CREATE (a:P)-[:R]->(:Q), (:P)");
    }
}

#[cfg(feature = "gql")]
#[test]
fn horizontal_sum_stays_per_path_inside_a_map() {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::GrafeoDB;
    use std::collections::BTreeMap;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P)-[:R {w: 2}]->(:P), (:P)-[:R {w: 3}]->(:P)")
        .unwrap();

    let bare = db
        .execute("MATCH (a:P)-[e:R]->{1,1}(b:P) RETURN sum(e.w) AS total ORDER BY total")
        .unwrap();
    assert_eq!(bare.columns, ["total"]);
    assert_eq!(bare.rows(), &[vec![Value::Int64(2)], vec![Value::Int64(3)]]);

    let wrapped = db
        .execute("MATCH (a:P)-[e:R]->{1,1}(b:P) RETURN {total: sum(e.w)} AS result")
        .unwrap();
    assert_eq!(wrapped.columns, ["result"]);
    let mut rows = wrapped.rows().to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    let expected: Vec<Vec<Value>> = [2, 3]
        .into_iter()
        .map(|total| {
            vec![Value::Map(
                BTreeMap::from([(PropertyKey::new("total"), Value::Int64(total))]).into(),
            )]
        })
        .collect();
    assert_eq!(rows, expected);
}

#[cfg(feature = "gql")]
#[test]
fn horizontal_sum_stays_per_path_through_nested_wrappers() {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P)-[:R {w: 2}]->(:P), (:P)-[:R {w: 3}]->(:P)")
        .unwrap();
    for expression in [
        "{total: sum(e.w)}.total",
        "[sum(e.w)][0]",
        "[sum(e.w)][0..1][0]",
        "LET total = sum(e.w) IN total END",
        "LET path = e IN sum(path.w) END",
    ] {
        let query =
            format!("MATCH (a:P)-[e:R]->{{1,1}}(b:P) RETURN {expression} AS total ORDER BY total");
        let result = db.execute(&query).unwrap();
        assert_eq!(result.columns, ["total"], "{query}");
        assert_eq!(
            result.rows(),
            &[vec![Value::Int64(2)], vec![Value::Int64(3)]],
            "{query}"
        );
    }
}

#[cfg(feature = "gql")]
#[test]
fn horizontal_wrapper_traversal_keeps_a_shadowing_local() {
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P)-[:R {w: 2}]->(:P)").unwrap();
    let error = db
        .execute("MATCH (a:P)-[e:R]->{1,1}(b:P) RETURN [e IN [{w: 7}] | sum(e.w)] AS values")
        .unwrap_err();
    assert!(
        error.to_string().contains("locally bound variable 'e'"),
        "{error}"
    );
}

// Append to seam_aggregates_expressions.rs; existing db()/Value imports remain.
#[cfg(feature = "gql")]
#[test]
fn gql_let_capture_keeps_a_correlated_shadowing_local() {
    let db = db();
    db.execute("INSERT (a:P)-[:R]->(:Q), (:P)").unwrap();
    for (query, expected) in [
        (
            "MATCH (n:P), (other:P) WHERE n <> other RETURN n, other, size([p IN [other] WHERE EXISTS { MATCH (p)-[:R]->() } | p]) AS score ORDER BY id(n)",
            vec![0, 1],
        ),
        (
            "MATCH (n:P), (other:P) WHERE n <> other RETURN n, other, LET source = n IN count(*) + size([n IN [other] WHERE EXISTS { MATCH (n)-[:R]->() } | source]) END AS score ORDER BY id(n)",
            vec![1, 2],
        ),
    ] {
        let result = db
            .execute(query)
            .unwrap_or_else(|error| panic!("`{query}`: {error}"));
        assert_eq!(result.rows().len(), 2, "{query}");
        let scores: Vec<i64> = result
            .rows()
            .iter()
            .map(|row| {
                assert_eq!(row.len(), 3, "{query}");
                match &row[2] {
                    Value::Int64(score) => *score,
                    other => panic!("`{query}`: expected integer score, got {other:?}"),
                }
            })
            .collect();
        // Keep query order: sorting these scores would hide swapped bindings.
        assert_eq!(scores, expected, "{query}");
    }
}

#[cfg(all(feature = "gql", feature = "cypher"))]
mod aggregate_discovery_539_local_subquery_guard {
    use super::*;
    use grafeo_common::utils::error::{Error, QueryErrorKind};

    fn check(language: &str, setup: &str) {
        let db = db();
        db.execute_language(setup, language, None).unwrap();
        let query = "MATCH (n:P) RETURN n, [p IN [n] | sum(CASE WHEN EXISTS { MATCH (p)-[:R]->() } THEN 1 ELSE 0 END)] AS values";
        match db.execute_language(query, language, None) {
            Err(Error::Query(error)) => {
                assert_eq!(
                    error.kind,
                    QueryErrorKind::Unsupported,
                    "{language} `{query}`: {error}"
                );
                assert!(
                    error
                        .message
                        .contains("cannot be lifted across a local binding"),
                    "{language} `{query}`: {error}"
                );
            }
            other => panic!(
                "{language} `{query}`: expected an unsupported local/subquery aggregate error, got {other:?}"
            ),
        }
    }

    #[test]
    fn gql_aggregate_cannot_capture_a_local_through_a_correlated_subquery() {
        check("gql", "INSERT (a:P)-[:R]->(:Q), (:P)");
    }

    #[test]
    fn cypher_aggregate_cannot_capture_a_local_through_a_correlated_subquery() {
        check("cypher", "CREATE (a:P)-[:R]->(:Q), (:P)");
    }
}

// Append to seam_aggregates_expressions.rs; existing db()/Value imports remain.
#[cfg(feature = "gql")]
#[test]
fn gql_aggregate_let_liveness_and_simultaneous_bindings() {
    let db = db();
    for (expression, expected) in [
        ("LET x = n IN LET y = x IN sum(y) END END", 6),
        ("LET x = n, y = n + 1 IN sum(x) + sum(y) END", 15),
        // Both values read incoming n; the scalar n is then bound to 10.
        ("LET n = 10, x = n IN sum(x) + n END", 16),
    ] {
        let query = format!("UNWIND [1, 2, 3] AS n RETURN {expression} AS value");
        let result = db
            .execute(&query)
            .unwrap_or_else(|error| panic!("`{query}`: {error}"));
        assert_eq!(result.rows(), &[vec![Value::Int64(expected)]], "{query}");
    }
}

#[cfg(all(feature = "gql", feature = "cypher"))]
mod aggregate_discovery_539_subquery_arguments {
    use super::*;
    use grafeo_common::utils::error::{Error, QueryErrorKind};

    fn check(language: &str, setup: &str) {
        let db = db();
        db.execute_language(setup, language, None).unwrap();
        for (query, expected) in [
            (
                "MATCH (n:P) RETURN n, sum(CASE WHEN EXISTS { MATCH (n)-[:R]->() } THEN 1 ELSE 0 END) AS score ORDER BY id(n)",
                vec![1, 0],
            ),
            (
                "MATCH (n:P) RETURN sum(size([p IN [n] WHERE EXISTS { MATCH (p)-[:R]->() } | p])) AS score",
                vec![1],
            ),
            (
                "UNWIND [1, 2, 3] AS n RETURN sum(reduce(total = 0, x IN [n] | total + x)) AS score",
                vec![6],
            ),
        ] {
            let result = db
                .execute_language(query, language, None)
                .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
            let scores: Vec<_> = result
                .rows()
                .iter()
                .map(|row| row.last().unwrap().clone())
                .collect();
            let expected: Vec<_> = expected.into_iter().map(Value::Int64).collect();
            assert_eq!(scores, expected, "{language} `{query}`");
        }
        let query = "MATCH (n:P) RETURN [p IN collect(n) | sum(CASE WHEN EXISTS { MATCH (x:Q) } THEN 1 ELSE 0 END)] AS values";
        match db.execute_language(query, language, None) {
            Err(Error::Query(error)) => {
                assert_eq!(
                    error.kind,
                    QueryErrorKind::Unsupported,
                    "{language}: {error}"
                );
                assert!(
                    error
                        .message
                        .contains("cannot be lifted across a local binding"),
                    "{language}: {error}"
                );
            }
            other => panic!("expected the explicit local/subquery capability boundary: {other:?}"),
        }
    }

    #[test]
    fn gql_keeps_supported_subquery_aggregate_arguments() {
        check("gql", "INSERT (a:P)-[:R]->(:Q), (:P)");
    }

    #[test]
    fn cypher_keeps_supported_subquery_aggregate_arguments() {
        check("cypher", "CREATE (a:P)-[:R]->(:Q), (:P)");
    }
}

#[cfg(feature = "gql")]
mod gql_shared_let_scope {
    use super::*;
    use grafeo_common::types::PropertyKey;
    use std::collections::BTreeMap;

    fn rows(query: &str) -> Vec<Vec<Value>> {
        db().execute_language(query, "gql", None)
            .unwrap_or_else(|error| panic!("`{query}`: {error}"))
            .rows()
            .to_vec()
    }

    fn ints(values: &[i64]) -> Value {
        Value::List(
            values
                .iter()
                .copied()
                .map(Value::Int64)
                .collect::<Vec<_>>()
                .into(),
        )
    }

    #[test]
    fn ordinary_let_substitutes_map_index_and_slice() {
        let query = "RETURN LET x = [1, 2, 3], i = 1 IN {picked: x[i], part: x[0..2], length: size(x)} END AS value";
        let expected = Value::Map(
            BTreeMap::from([
                (PropertyKey::new("picked"), Value::Int64(2)),
                (PropertyKey::new("part"), ints(&[1, 2])),
                (PropertyKey::new("length"), Value::Int64(3)),
            ])
            .into(),
        );
        assert_eq!(rows(query), vec![vec![expected]]);
    }

    #[test]
    fn ordinary_let_substitutes_an_indexed_list_per_input() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET x = n IN [x][0] END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), Value::Int64(1)],
                vec![Value::Int64(2), Value::Int64(2)],
                vec![Value::Int64(3), Value::Int64(3)],
            ]
        );
    }

    #[test]
    fn ordinary_simultaneous_let_values_read_the_enclosing_scope() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET n = 10, x = n IN x + n END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), Value::Int64(11)],
                vec![Value::Int64(2), Value::Int64(12)],
                vec![Value::Int64(3), Value::Int64(13)],
            ]
        );
    }

    #[test]
    fn ordinary_nested_let_preserves_shadowed_binding_values() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET x = n IN LET x = x + 10, y = x IN [x, y] END END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), ints(&[11, 1])],
                vec![Value::Int64(2), ints(&[12, 2])],
                vec![Value::Int64(3), ints(&[13, 3])],
            ]
        );
    }

    #[test]
    fn ordinary_let_alias_is_not_captured_by_a_comprehension() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET x = n IN [n IN [10] | x + n] END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), ints(&[11])],
                vec![Value::Int64(2), ints(&[12])],
                vec![Value::Int64(3), ints(&[13])],
            ]
        );
    }

    #[test]
    fn ordinary_let_alias_is_not_captured_by_a_predicate() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET x = n IN any(n IN [10] WHERE n > x) END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), Value::Bool(true)],
                vec![Value::Int64(2), Value::Bool(true)],
                vec![Value::Int64(3), Value::Bool(true)],
            ]
        );
    }

    #[test]
    fn ordinary_let_alias_is_not_captured_by_a_reduce_accumulator() {
        let query = "UNWIND [3, 1, 2] AS n RETURN n AS input, LET x = n IN reduce(n = 0, p IN [10] | n + p + x) END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), Value::Int64(11)],
                vec![Value::Int64(2), Value::Int64(12)],
                vec![Value::Int64(3), Value::Int64(13)],
            ]
        );
    }

    #[test]
    fn unused_let_input_does_not_become_a_grouping_key() {
        let query = "UNWIND [1, 2, 3] AS n RETURN LET unused = n IN sum(n) END AS value";
        assert_eq!(rows(query), vec![vec![Value::Int64(6)]]);
    }

    #[test]
    fn order_by_wrapped_alias_uses_its_projected_value() {
        let query = "UNWIND [1, 3, 2] AS n RETURN -n AS n ORDER BY [n][0]";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(-3)],
                vec![Value::Int64(-2)],
                vec![Value::Int64(-1)],
            ]
        );
    }

    #[test]
    fn let_binding_evaluates_a_volatile_value_once() {
        for (query, count) in [
            (
                "UNWIND [1, 2, 3] AS n RETURN LET x = rand() IN x = x END AS same",
                3,
            ),
            (
                "UNWIND [1, 2, 3] AS n RETURN LET x = rand() IN count(*) > 0 AND x = x END AS same",
                1,
            ),
            (
                "UNWIND [1, 2, 3] AS n RETURN LET x = abs(rand()) IN x = x END AS same",
                3,
            ),
            (
                "UNWIND [1, 2, 3] AS n RETURN LET x = abs(rand()) IN count(*) > 0 AND x = x END AS same",
                1,
            ),
        ] {
            // rand() returns a finite value: reusing one binding compares
            // that same value with itself, for every input and aggregate row.
            assert_eq!(rows(query), vec![vec![Value::Bool(true)]; count], "{query}");
        }
    }
    #[test]
    fn order_by_keeps_input_columns_with_unrelated_outputs() {
        for query in [
            "UNWIND [2, 1] AS n RETURN n + 10 AS value, rand() AS unused ORDER BY n NEXT RETURN value",
            "UNWIND [2, 1] AS n RETURN n + 10 AS value ORDER BY 0 * rand(), n NEXT RETURN value",
            "UNWIND [2, 1] AS n RETURN n + 10 AS value, 0 * rand() AS r ORDER BY r, n NEXT RETURN value",
        ] {
            assert_eq!(
                rows(query),
                vec![vec![Value::Int64(11)], vec![Value::Int64(12)]],
                "{query}"
            );
        }
    }
    #[test]
    fn order_by_keeps_input_columns_beside_an_opaque_key() {
        let db = db();
        db.execute_language("INSERT (:P {v: 2}), (:P {v: 1})", "gql", None)
            .unwrap();
        for query in [
            "MATCH (n:P) RETURN n.v + 10 AS value, EXISTS { MATCH (n)-[:R]->() } AS unused ORDER BY n.v NEXT RETURN value",
            "MATCH (n:P) RETURN n.v + 10 AS value ORDER BY EXISTS { MATCH (n)-[:R]->() }, n.v NEXT RETURN value",
        ] {
            let result = db
                .execute_language(query, "gql", None)
                .unwrap_or_else(|error| panic!("`{query}`: {error}"));
            assert_eq!(
                result.rows(),
                &[vec![Value::Int64(11)], vec![Value::Int64(12)]],
                "{query}"
            );
        }
    }

    #[test]
    fn lexical_let_grouping_keeps_the_same_key_across_translation() {
        let query = "UNWIND [1, 1, 2] AS n RETURN LET x = toInteger(0 * rand()), y = n IN x + y END AS k, count(*) AS c GROUP BY k ORDER BY k";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Int64(2), Value::Int64(1)]
            ]
        );
    }
    #[test]
    fn nested_lexical_lets_keep_enclosing_values_and_reserved_names() {
        let query = "UNWIND [3, 1, 2] AS _let_values_0 RETURN _let_values_0 AS input, LET x = toInteger(0 * rand()), y = _let_values_0 IN LET x = toInteger(0 * rand()), z = x + y IN [x, z, y] END END AS value ORDER BY input";
        assert_eq!(
            rows(query),
            vec![
                vec![Value::Int64(1), ints(&[0, 1, 1])],
                vec![Value::Int64(2), ints(&[0, 2, 2])],
                vec![Value::Int64(3), ints(&[0, 3, 3])],
            ]
        );
    }
}
