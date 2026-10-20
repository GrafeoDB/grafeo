//! Aggregate calls as the languages define them: an aggregate takes its own
//! number of arguments (`count(a, b)` counted `a`, `sum()` counted the rows),
//! a COUNT subquery in an aggregate's argument runs per row (it was an
//! internal error), GQL takes HAVING right after GROUP BY (before ORDER BY,
//! as SQL orders them; it was a syntax error), and an SQL/PGQ SELECT item
//! reads only grouping keys (one that read another column silently became a
//! grouping key and split the groups).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test aggregate_calls
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "cypher",
    feature = "sql-pgq"
))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_engine::GrafeoDB;

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

/// Asserts a semantic error whose message holds `names`.
fn refused(db: &GrafeoDB, language: &str, query: &str, names: &str) {
    match db.execute_language(query, language, None) {
        Err(Error::Query(error)) => {
            assert_eq!(error.kind, QueryErrorKind::Semantic, "{language} `{query}`");
            assert!(
                error.message.contains(names),
                "{language} `{query}`: {}",
                error.message
            );
        }
        other => panic!("{language} `{query}`: expected a semantic error, got {other:?}"),
    }
}

#[test]
fn an_aggregate_takes_its_number_of_arguments() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {v: 3}), (:P {v: 19})").unwrap();
    for language in ["gql", "cypher"] {
        refused(
            &db,
            language,
            "MATCH (p:P) RETURN count(p.v, p.v) AS c",
            "Function 'count' takes 0 or 1 arguments, got 2",
        );
        refused(
            &db,
            language,
            "MATCH (p:P) RETURN sum() AS s",
            "Function 'sum' takes 1 argument, got 0",
        );
        refused(
            &db,
            language,
            "MATCH (p:P) RETURN collect(p.v, p.v) AS l",
            "Function 'collect' takes 1 argument, got 2",
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:P) RETURN count(*) AS a, count(p.v) AS b, sum(p.v) AS s"
            ),
            [[Value::Int64(2), Value::Int64(2), Value::Int64(22)]],
            "{language}"
        );
    }
    refused(
        &db,
        "sql",
        "SELECT SUM() AS s FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.v AS v))",
        "Function 'SUM' takes 1 argument, got 0",
    );
}

/// A COUNT subquery in an aggregate's argument counts per row: rows 1 and 2
/// look for k 300 and 299, of which 300 exists.
#[test]
fn a_count_subquery_inside_an_aggregate_runs_per_row() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:T {k: 300})").unwrap();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [1, 2] AS i RETURN sum(COUNT { MATCH (t:T) WHERE t.k = 301 - i }) AS s"
            ),
            [[Value::Int64(1)]],
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [1, 2, 3] AS i \
                 RETURN COUNT { MATCH (t:T) WHERE t.k = 301 - i } AS found, count(*) AS c \
                 ORDER BY found"
            ),
            [
                [Value::Int64(0), Value::Int64(2)],
                [Value::Int64(1), Value::Int64(1)]
            ],
            "{language}: a COUNT subquery as a grouping key"
        );
    }
}

/// GQL takes HAVING right after GROUP BY, before ORDER BY, and after the
/// RETURN clause as before; not both.
#[test]
fn gql_having_after_group_by() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:P {city: 'Paris'}), (:P {city: 'Paris'}), (:P {city: 'Prague'}), \
         (:P {city: 'Berlin'}), (:P {city: 'Berlin'})",
    )
    .unwrap();
    let paris_berlin = [[Value::from("Berlin")], [Value::from("Paris")]];
    assert_eq!(
        rows(
            &db,
            "gql",
            "MATCH (p:P) RETURN p.city AS city GROUP BY city HAVING count(*) > 1 ORDER BY city"
        ),
        paris_berlin
    );
    assert_eq!(
        rows(
            &db,
            "gql",
            "MATCH (p:P) RETURN p.city AS city GROUP BY city ORDER BY city HAVING count(*) > 1"
        ),
        paris_berlin
    );
    let error = db
        .execute(
            "MATCH (p:P) RETURN p.city AS city GROUP BY city HAVING count(*) > 1 \
             ORDER BY city HAVING count(*) > 0",
        )
        .expect_err("two HAVING");
    assert!(error.to_string().contains("one HAVING"), "{error}");
}

/// Without GROUP BY the grouping keys are the items without an aggregate, and
/// an item with one reads only them, aggregates and constants: one that
/// reads another variable is refused (openCypher's
/// AmbiguousAggregationExpression; it returned null), and a part that is a
/// key reads the key (it was null too).
#[test]
fn an_implicitly_grouped_item_reads_grouping_keys() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {g: 'f', b: 2}), (:P {g: 'f', b: 5}), (:P {g: 'm', b: 7})")
        .unwrap();
    for language in ["gql", "cypher"] {
        refused(
            &db,
            language,
            "MATCH (p:P) RETURN p.g AS g, count(*) + p.b AS n",
            "RETURN item n reads p beside an aggregate, but p is not a grouping key",
        );
        refused(
            &db,
            language,
            "MATCH (p:P) RETURN count(*) + p.b AS n",
            "(it has none)",
        );
        refused(
            &db,
            language,
            "MATCH (p:P) WITH p.g AS g, sum(p.b) * p.b AS n RETURN g, n",
            "WITH item n reads p",
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:P) RETURN p.b AS b, count(*) + p.b AS n ORDER BY b"
            ),
            [
                [Value::Int64(2), Value::Int64(3)],
                [Value::Int64(5), Value::Int64(6)],
                [Value::Int64(7), Value::Int64(8)]
            ],
            "{language}: p.b is a key"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:P) WITH p, count(*) * 10 + p.b AS n RETURN n ORDER BY n"
            ),
            [[Value::Int64(12)], [Value::Int64(15)], [Value::Int64(17)]],
            "{language}: a property of the key p"
        );
    }
    // An import of a CALL body has one value per call.
    for (language, query) in [
        (
            "gql",
            "UNWIND [100, 200] AS t CALL (t) { MATCH (p:P) RETURN count(*) + t AS n } \
             RETURN n ORDER BY n",
        ),
        (
            "cypher",
            "UNWIND [100, 200] AS t CALL (t) { MATCH (p:P) RETURN count(*) + t AS n } \
             RETURN n ORDER BY n",
        ),
    ] {
        assert_eq!(
            rows(&db, language, query),
            [[Value::Int64(103)], [Value::Int64(203)]],
            "{language}"
        );
    }
    assert_eq!(
        rows(
            &db,
            "gql",
            "UNWIND [100] AS t CALL (t) { MATCH (p:P) RETURN p.g AS g, count(*) + t AS n GROUP BY g } \
             RETURN g, n ORDER BY g"
        ),
        [
            [Value::from("f"), Value::Int64(102)],
            [Value::from("m"), Value::Int64(101)]
        ],
        "with GROUP BY too"
    );
}

/// An SQL/PGQ SELECT item reads grouping keys only: a key, an expression of
/// keys or a constant.
#[test]
fn an_sql_pgq_select_item_reads_grouping_keys() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:P {city: 'Paris', name: 'Alix'}), (:P {city: 'Paris', name: 'Gus'}), \
         (:P {city: 'Prague', name: 'Mia'})",
    )
    .unwrap();
    refused(
        &db,
        "sql",
        "SELECT city, name, COUNT(*) AS c \
         FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.city AS city, p.name AS name)) GROUP BY city",
        "name",
    );
    let mut grouped = rows(
        &db,
        "sql",
        "SELECT city, UPPER(city) AS loud, 1 AS one, COUNT(*) AS c \
         FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.city AS city, p.name AS name)) GROUP BY city",
    );
    grouped.sort_by_key(|row| row[0].to_string());
    // An item over a computed key reads the key.
    let mut lengths = rows(
        &db,
        "sql",
        "SELECT CHAR_LENGTH(name) + 1 AS l, COUNT(*) AS c \
         FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.city AS city, p.name AS name)) \
         GROUP BY CHAR_LENGTH(name)",
    );
    lengths.sort_by_key(|row| row[0].to_string());
    assert_eq!(
        lengths,
        [
            [Value::Int64(4), Value::Int64(2)],
            [Value::Int64(5), Value::Int64(1)]
        ]
    );
    assert_eq!(
        grouped,
        [
            [
                Value::from("Paris"),
                Value::from("PARIS"),
                Value::Int64(1),
                Value::Int64(2)
            ],
            [
                Value::from("Prague"),
                Value::from("PRAGUE"),
                Value::Int64(1),
                Value::Int64(1)
            ]
        ]
    );
}
