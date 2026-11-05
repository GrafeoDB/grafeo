//! SQL/PGQ gaps (0.6.1): ORDER BY may name an alias of the outer SELECT (an
//! output column, a computed one, a path function), as standard SQL allows
//! and as an alias from COLUMNS already could; and a parameter may be named
//! after a reserved word (`$from`, `$select`, `$order`, ...), as GQL and
//! Cypher allow, also next to the same keyword in the query.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_outer_aliases_and_keyword_parameters::
//! ```

#![cfg(all(feature = "gql", feature = "sql-pgq"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix (30) knows Gus (25), Gus knows Vincent (40); Mia (35) knows no one.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Person {name: 'Alix', age: 30})-[:KNOWS]->(g:Person {name: 'Gus', age: 25})\
         -[:KNOWS]->(:Person {name: 'Vincent', age: 40}), (:Person {name: 'Mia', age: 35})",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str, params: Option<HashMap<String, Value>>) -> Vec<Vec<Value>> {
    db.execute_language(query, "sql-pgq", params)
        .unwrap_or_else(|error| panic!("`{query}`: {error}"))
        .rows()
        .to_vec()
}

fn ints(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

#[test]
fn order_by_an_alias_of_the_outer_select() {
    let db = database();
    assert_eq!(
        rows(
            &db,
            "SELECT g.age AS years FROM GRAPH_TABLE (MATCH (a:Person) COLUMNS (a.age AS age)) AS g \
             ORDER BY years",
            None
        ),
        ints(&[25, 30, 35, 40])
    );
    assert_eq!(
        rows(
            &db,
            "SELECT age AS years FROM GRAPH_TABLE (MATCH (a:Person) COLUMNS (a.age AS age)) \
             ORDER BY years DESC",
            None
        ),
        ints(&[40, 35, 30, 25])
    );
}

#[test]
fn order_by_a_computed_alias_of_the_outer_select() {
    let db = database();
    assert_eq!(
        rows(
            &db,
            "SELECT g.age + 1 AS older FROM GRAPH_TABLE (MATCH (a:Person) COLUMNS (a.age AS age)) AS g \
             ORDER BY older DESC",
            None
        ),
        ints(&[41, 36, 31, 26])
    );
    // The alias and the column it hides: ORDER BY the alias, not the column.
    assert_eq!(
        rows(
            &db,
            "SELECT 100 - age AS age FROM GRAPH_TABLE (MATCH (a:Person) COLUMNS (a.age AS age)) \
             ORDER BY age",
            None
        ),
        ints(&[60, 65, 70, 75])
    );
}

#[test]
fn order_by_an_alias_of_a_path_function() {
    let db = database();
    // Paths of one and two KNOWS from Alix: two nodes, then three.
    assert_eq!(
        rows(
            &db,
            "SELECT CARDINALITY(NODES(p)) AS nodes FROM GRAPH_TABLE \
             (MATCH p = (a:Person WHERE a.name = 'Alix')-[:KNOWS]->{1,2}(b:Person) COLUMNS (p AS p)) \
             ORDER BY nodes DESC",
            None
        ),
        ints(&[3, 2])
    );
}

/// The reserved words of the SQL/PGQ lexer that a parameter name might be.
const KEYWORDS: [&str; 15] = [
    "from", "select", "where", "match", "order", "limit", "as", "group", "by", "and", "or", "not",
    "null", "true", "is",
];

#[test]
fn a_parameter_named_after_a_reserved_word_binds() {
    let db = database();
    let want = rows(
        &db,
        "SELECT n FROM GRAPH_TABLE (MATCH (a:Person) WHERE a.age >= $to COLUMNS (a.name AS n)) \
         ORDER BY n",
        Some(HashMap::from([("to".to_string(), Value::Int64(30))])),
    );
    assert_eq!(want.len(), 3, "Alix, Mia and Vincent");
    for keyword in KEYWORDS {
        for name in [keyword.to_string(), keyword.to_uppercase()] {
            let query = format!(
                "SELECT n FROM GRAPH_TABLE (MATCH (a:Person) WHERE a.age >= ${name} \
                 COLUMNS (a.name AS n)) ORDER BY n"
            );
            let params = HashMap::from([(name.clone(), Value::Int64(30))]);
            assert_eq!(rows(&db, &query, Some(params)), want, "${name}");
        }
    }
}

#[test]
fn a_keyword_parameter_next_to_the_same_keyword() {
    let db = database();
    // $from inside FROM's graph table, $limit in LIMIT, $order beside ORDER BY.
    let params = HashMap::from([
        ("from".to_string(), Value::Int64(26)),
        ("limit".to_string(), Value::Int64(2)),
        ("order".to_string(), Value::Int64(1)),
    ]);
    assert_eq!(
        rows(
            &db,
            "SELECT n, age + $order AS next FROM GRAPH_TABLE (MATCH (a:Person) WHERE a.age >= $from \
             COLUMNS (a.name AS n, a.age AS age)) ORDER BY n LIMIT $limit",
            Some(params)
        ),
        vec![
            vec![Value::from("Alix"), Value::Int64(31)],
            vec![Value::from("Mia"), Value::Int64(36)],
        ]
    );
}

#[test]
fn gql_and_cypher_already_take_a_keyword_parameter() {
    // Control: the other languages.
    let db = database();
    let params = || Some(HashMap::from([("from".to_string(), Value::Int64(36))]));
    for (language, query) in [
        ("gql", "MATCH (a:Person) WHERE a.age >= $from RETURN a.name"),
        (
            "cypher",
            "MATCH (a:Person) WHERE a.age >= $from RETURN a.name",
        ),
    ] {
        let got = db
            .execute_language(query, language, params())
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(got, vec![vec![Value::from("Vincent")]], "{language}");
    }
}

/// LIMIT and OFFSET take a parameter: a whole non-negative number limits and
/// skips; a negative one, a fraction or a text fails with the same GQLSTATUS
/// and code as GQL's LIMIT and SKIP give for it.
#[test]
fn limit_and_offset_parameters_behave_as_in_gql() {
    use grafeo_common::utils::gqlstatus::GqlStatus;

    let db = database();
    let sql = |clause: &str, value: Value| {
        db.execute_language(
            &format!(
                "SELECT n FROM GRAPH_TABLE (MATCH (a:Person) COLUMNS (a.name AS n)) ORDER BY n {clause}"
            ),
            "sql-pgq",
            Some(HashMap::from([("v".to_string(), value)])),
        )
    };
    let gql = |clause: &str, value: Value| {
        db.execute_language(
            &format!("MATCH (a:Person) RETURN a.name AS n ORDER BY n {clause}"),
            "gql",
            Some(HashMap::from([("v".to_string(), value)])),
        )
    };
    let names =
        |result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>| {
            result
                .unwrap()
                .rows()
                .iter()
                .map(|row| row[0].clone())
                .collect::<Vec<_>>()
        };
    assert_eq!(
        names(sql("LIMIT $v", Value::Int64(2))),
        [Value::from("Alix"), Value::from("Gus")]
    );
    assert_eq!(
        names(sql("OFFSET $v", Value::Int64(2))),
        [Value::from("Mia"), Value::from("Vincent")]
    );
    assert_eq!(
        names(sql("LIMIT $v OFFSET 1", Value::Int64(0))),
        Vec::<Value>::new()
    );
    for (sql_clause, gql_clause) in [("LIMIT $v", "LIMIT $v"), ("OFFSET $v", "SKIP $v")] {
        for value in [Value::Int64(-1), Value::Float64(1.5), Value::from("two")] {
            let from_sql = sql(sql_clause, value.clone())
                .expect_err(&format!("SQL/PGQ {sql_clause} {value:?}"));
            let from_gql =
                gql(gql_clause, value.clone()).expect_err(&format!("GQL {gql_clause} {value:?}"));
            assert_eq!(
                GqlStatus::from(&from_sql),
                GqlStatus::from(&from_gql),
                "{sql_clause} {value:?}: SQL/PGQ `{from_sql}` against GQL `{from_gql}`"
            );
            assert_eq!(
                from_sql.error_code(),
                from_gql.error_code(),
                "{sql_clause} {value:?}"
            );
        }
    }
}
