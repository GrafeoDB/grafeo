//! Aggregation semantics: DISTINCT in every aggregate, aggregates in a GQL
//! HAVING, GROUP BY without an aggregate, and grouping keys that are lists.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --all-features --test aggregate_semantics
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

/// The single value of a query that returns one row of one column.
fn value(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let rows = rows(db, language, query);
    assert_eq!(rows.len(), 1, "{language}: {query}: {rows:?}");
    assert_eq!(rows[0].len(), 1, "{language}: {query}: {rows:?}");
    rows[0][0].clone()
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

fn list(values: &[Value]) -> Value {
    Value::List(values.to_vec().into())
}

/// One `:N` node per value of `xs`, each with `y = 2 * x + 3` and `g = 'g'`.
fn numbers(xs: &[i64]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for x in xs {
        session
            .execute(&format!(
                "INSERT (:N {{g: 'g', x: {x}, y: {}}})",
                2 * x + 3 + x % 19
            ))
            .unwrap();
    }
    db
}

// ============================================================================
// DISTINCT in the statistical aggregates
// ============================================================================
//
// A set quantifier DISTINCT removes the duplicate values before the
// aggregate is computed (ISO/IEC 39075:2024 20.9 <aggregate function>, ISO/IEC
// 9075-2 <set function specification>), for every aggregate that takes it:
// `agg(DISTINCT x)` over values with copies is `agg(x)` over each value once.

/// The aggregates per language, as `(language, with DISTINCT, without)`.
fn statistical_aggregates() -> Vec<(&'static str, &'static str, &'static str)> {
    let mut aggregates = vec![
        ("gql", "stddev_samp(DISTINCT n.x)", "stddev_samp(n.x)"),
        ("gql", "stddev_pop(DISTINCT n.x)", "stddev_pop(n.x)"),
        ("gql", "var_samp(DISTINCT n.x)", "var_samp(n.x)"),
        ("gql", "var_pop(DISTINCT n.x)", "var_pop(n.x)"),
        (
            "gql",
            "percentile_disc(DISTINCT n.x, 0.5)",
            "percentile_disc(n.x, 0.5)",
        ),
        (
            "gql",
            "percentile_cont(DISTINCT n.x, 0.5)",
            "percentile_cont(n.x, 0.5)",
        ),
    ];
    if cfg!(feature = "cypher") {
        aggregates.extend([
            ("cypher", "stDev(DISTINCT n.x)", "stDev(n.x)"),
            ("cypher", "stDevP(DISTINCT n.x)", "stDevP(n.x)"),
            ("cypher", "variance(DISTINCT n.x)", "variance(n.x)"),
            (
                "cypher",
                "percentileDisc(DISTINCT n.x, 0.5)",
                "percentileDisc(n.x, 0.5)",
            ),
            (
                "cypher",
                "percentileCont(DISTINCT n.x, 0.5)",
                "percentileCont(n.x, 0.5)",
            ),
        ]);
    }
    aggregates
}

#[test]
fn statistical_aggregates_drop_duplicate_values_with_distinct() {
    let with_copies = numbers(&[3, 3, 3, 19, 88]);
    let once = numbers(&[3, 19, 88]);
    for (language, distinct, plain) in statistical_aggregates() {
        for (shape, query) in [
            ("ungrouped", "MATCH (n:N) RETURN {} AS v"),
            ("grouped", "MATCH (n:N) WITH n.g AS g, {} AS v RETURN v"),
        ] {
            let expected = value(&once, language, &query.replace("{}", plain));
            let copies_counted = value(&with_copies, language, &query.replace("{}", plain));
            assert_ne!(
                copies_counted, expected,
                "{language} {shape}: the copies must change {plain}, or the test proves nothing"
            );
            assert_eq!(
                value(&with_copies, language, &query.replace("{}", distinct)),
                expected,
                "{language} {shape}: {distinct} over 3, 3, 3, 19, 88 is {plain} over 3, 19, 88"
            );
        }
    }
}

#[test]
fn binary_set_functions_drop_duplicate_pairs_with_distinct() {
    let with_copies = numbers(&[3, 3, 3, 19, 88]);
    let once = numbers(&[3, 19, 88]);
    for function in [
        "covar_samp",
        "covar_pop",
        "corr",
        "regr_slope",
        "regr_count",
        "regr_sxx",
        "regr_avgx",
    ] {
        let plain = format!("MATCH (n:N) RETURN {function}(n.y, n.x) AS v");
        let distinct = format!("MATCH (n:N) RETURN {function}(DISTINCT n.y, n.x) AS v");
        let expected = value(&once, "gql", &plain);
        assert_ne!(
            value(&with_copies, "gql", &plain),
            expected,
            "{function}: the copies must change the result, or the test proves nothing"
        );
        assert_eq!(
            value(&with_copies, "gql", &distinct),
            expected,
            "{function}(DISTINCT ...) counts each (y, x) pair once"
        );
    }
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_statistical_aggregates_drop_duplicate_values_with_distinct() {
    let with_copies = numbers(&[3, 3, 3, 19, 88]);
    let once = numbers(&[3, 19, 88]);
    for function in ["STDDEV_SAMP", "STDDEV_POP", "VAR_SAMP", "VAR_POP"] {
        let query = |argument: &str| {
            format!(
                "SELECT {function}({argument}) AS v \
                 FROM GRAPH_TABLE (MATCH (n:N) COLUMNS (n.x AS x))"
            )
        };
        let expected = value(&once, "sql", &query("x"));
        assert_ne!(
            value(&with_copies, "sql", &query("x")),
            expected,
            "{function}"
        );
        assert_eq!(
            value(&with_copies, "sql", &query("DISTINCT x")),
            expected,
            "{function}(DISTINCT x)"
        );
    }
}

// ============================================================================
// GQL HAVING with an aggregate, and GROUP BY without one
// ============================================================================
//
// A HAVING condition is evaluated per group, and an aggregate in it is
// computed over the rows of that group, whether or not the RETURN list
// computes it too (ISO/IEC 9075-2 <having clause>; Grafeo's GQL HAVING follows
// SQL). GROUP BY makes one row per group (ISO/IEC 39075:2024 <group by
// clause>), also when the RETURN list has no aggregate.

/// Alix knows Gus, Vincent and Mia; Jules knows Gus and Mia; Vincent knows Mia.
fn acquaintances() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (alix:Person {name: 'Alix', age: 19}), (gus:Person {name: 'Gus', age: 3}), \
                    (vincent:Person {name: 'Vincent', age: 88}), (mia:Person {name: 'Mia', age: 19}), \
                    (jules:Person {name: 'Jules', age: 3}), \
                    (alix)-[:KNOWS]->(gus), (alix)-[:KNOWS]->(vincent), (alix)-[:KNOWS]->(mia), \
                    (jules)-[:KNOWS]->(gus), (jules)-[:KNOWS]->(mia), (vincent)-[:KNOWS]->(mia)",
        )
        .unwrap();
    db
}

fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

#[test]
fn having_computes_an_aggregate_that_the_return_list_also_computes() {
    let db = acquaintances();
    let result = db
        .session()
        .execute(
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k, count(*) AS c \
             GROUP BY m.name HAVING count(*) > 1",
        )
        .unwrap();
    assert_eq!(result.columns, ["k", "c"], "HAVING adds no column");
    assert_eq!(
        sorted(result.rows().to_vec()),
        [
            vec![text("Gus"), Value::Int64(2)],
            vec![text("Mia"), Value::Int64(3)],
        ]
    );
}

#[test]
fn having_computes_an_aggregate_that_the_return_list_does_not() {
    let db = acquaintances();
    let result = db
        .session()
        .execute(
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k, collect(n.name) AS knowers \
             GROUP BY m.name HAVING count(*) > 2",
        )
        .unwrap();
    assert_eq!(
        result.columns,
        ["k", "knowers"],
        "the aggregate HAVING reads stays out of the result"
    );
    assert_eq!(result.rows().len(), 1, "{:?}", result.rows());
    assert_eq!(result.rows()[0][0], text("Mia"));
    let Value::List(knowers) = &result.rows()[0][1] else {
        panic!("collect gives a list: {:?}", result.rows());
    };
    let mut knowers: Vec<String> = knowers.iter().map(|v| format!("{v:?}")).collect();
    knowers.sort();
    assert_eq!(
        knowers.len(),
        3,
        "Alix, Jules and Vincent know Mia: {knowers:?}"
    );
}

#[test]
fn having_reads_aggregates_inside_an_expression() {
    let db = acquaintances();
    // Gus is known by Alix (19) and Jules (3): average 11; Mia by Alix, Jules
    // and Vincent: (19 + 3 + 88) / 3; Vincent by Alix only.
    assert_eq!(
        sorted(rows(
            &db,
            "gql",
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k \
             GROUP BY m.name HAVING sum(n.age) / count(*) > 10 AND max(n.age) < 88"
        )),
        [vec![text("Gus")], vec![text("Vincent")]]
    );
}

#[test]
fn group_by_without_an_aggregate_returns_one_row_per_group() {
    let db = acquaintances();
    assert_eq!(
        sorted(rows(
            &db,
            "gql",
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN n.name AS k GROUP BY n.name"
        )),
        [
            vec![text("Alix")],
            vec![text("Jules")],
            vec![text("Vincent")]
        ]
    );
    assert_eq!(
        sorted(rows(
            &db,
            "gql",
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN n.name AS k GROUP BY n.name HAVING count(*) > 1"
        )),
        [vec![text("Alix")], vec![text("Jules")]],
        "with a HAVING aggregate too"
    );
}

#[test]
fn having_reads_grouping_keys_and_return_aliases() {
    let db = acquaintances();
    let gus_and_vincent = [
        vec![text("Gus"), Value::Int64(2)],
        vec![text("Vincent"), Value::Int64(1)],
    ];
    for (query, expected) in [
        (
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k, count(*) AS c \
             GROUP BY m.name HAVING m.name <> 'Mia'",
            gus_and_vincent.to_vec(),
        ),
        (
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k, count(*) AS c \
             GROUP BY m.name HAVING k <> 'Mia'",
            gus_and_vincent.to_vec(),
        ),
        (
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k, count(*) + 1 AS c \
             GROUP BY m.name HAVING c > 2",
            vec![
                vec![text("Gus"), Value::Int64(3)],
                vec![text("Mia"), Value::Int64(4)],
            ],
        ),
        (
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k GROUP BY m.name HAVING k <> 'Mia'",
            vec![vec![text("Gus")], vec![text("Vincent")]],
        ),
        // Without GROUP BY the items that are no aggregate are the keys
        (
            "MATCH (n:Person)-[:KNOWS]->(m) RETURN m.name AS k HAVING count(*) > 1",
            vec![vec![text("Gus")], vec![text("Mia")]],
        ),
    ] {
        assert_eq!(sorted(rows(&db, "gql", query)), expected, "{query}");
    }
}

// ============================================================================
// A list as a grouping key next to an aggregate
// ============================================================================
//
// A grouping key keeps its value, a list too (ISO/IEC 39075:2024 <group by
// clause>; openCypher WITH grouping keys): whatever the aggregate beside it.

/// Person 5 studies at UvA and works at Siemens.
fn student() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (p:Person {id: 5})-[:STUDY_AT]->(:Uni {name: 'UvA'}), \
                    (p)-[:WORK_AT]->(:Company {name: 'Siemens'})",
        )
        .unwrap();
    db
}

fn grouping_languages() -> Vec<&'static str> {
    if cfg!(feature = "cypher") {
        vec!["gql", "cypher"]
    } else {
        vec!["gql"]
    }
}

#[test]
fn a_list_from_collect_stays_a_list_as_a_grouping_key() {
    let db = student();
    let unis = list(&[text("UvA")]);
    for language in grouping_languages() {
        for aggregate in [
            "collect(1)",
            "sum(1)",
            "avg(1)",
            "min(1)",
            "max(1)",
            "count(1)",
        ] {
            let query = format!(
                "MATCH (f:Person)-[:STUDY_AT]->(u) WITH f, collect(u.name) AS unis \
                 WITH f, {aggregate} AS a, unis RETURN unis"
            );
            assert_eq!(
                value(&db, language, &query),
                unis,
                "{language}: the key after {aggregate}"
            );
            let query = format!(
                "MATCH (f:Person)-[:STUDY_AT]->(u) WITH f, collect(u.name) AS unis \
                 WITH unis, {aggregate} AS a RETURN unis"
            );
            assert_eq!(
                value(&db, language, &query),
                unis,
                "{language}: the only key, before {aggregate}"
            );
        }
    }
}

#[test]
fn a_list_literal_stays_a_list_as_a_grouping_key() {
    let db = student();
    let expected = list(&[Value::Int64(5), Value::Int64(1)]);
    for language in grouping_languages() {
        for aggregate in ["collect(1)", "sum(1)", "count(1)"] {
            let query = format!(
                "MATCH (f:Person) WITH f, [f.id, 1] AS l WITH f, {aggregate} AS a, l RETURN l"
            );
            assert_eq!(
                value(&db, language, &query),
                expected,
                "{language}: after {aggregate}"
            );
        }
    }
}

#[test]
fn a_list_grouping_key_and_its_collected_values_both_survive() {
    let db = student();
    for language in grouping_languages() {
        let result = rows(
            &db,
            language,
            "MATCH (f:Person)-[:STUDY_AT]->(u) WITH f, collect(u.name) AS unis \
             WITH f, collect(1) AS ones, unis RETURN unis, ones",
        );
        assert_eq!(
            result,
            [vec![list(&[text("UvA")]), list(&[Value::Int64(1)])]],
            "{language}"
        );
    }
}
