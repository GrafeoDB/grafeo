//! Procedure arguments: a parameter or another constant expression in a
//! `CALL` argument runs exactly like the same literal, in GQL and Cypher, and
//! an argument the procedure cannot use (one that reads a row, or a value of
//! the wrong type) is an error, never the parameter's default.

#![cfg(feature = "algos")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Alix -> Gus -> Vincent -> Mia, and Alix -> Jules.
fn chain_and_branch() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})\
         -[:KNOWS]->(:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Mia'}), \
         (a)-[:KNOWS]->(:Person {name: 'Jules'})",
    )
    .unwrap();
    db
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// The node id of the person called `name`.
fn id_of(db: &GrafeoDB, name: &str) -> i64 {
    let result = db
        .execute_with_params(
            "MATCH (p:Person {name: $name}) RETURN id(p)",
            params(&[("name", Value::from(name))]),
        )
        .unwrap();
    match &result.rows()[0][0] {
        Value::Int64(id) => *id,
        other => panic!("id() returned {other:?}"),
    }
}

/// Every row of `result` as written, for an exact comparison.
fn rows(result: &QueryResult) -> Vec<Vec<Value>> {
    result.rows().to_vec()
}

/// The name of each node a BFS or DFS reached, with its value in `column`.
fn by_name(db: &GrafeoDB, result: &QueryResult, column: usize) -> Vec<(String, i64)> {
    let names: HashMap<i64, String> = db
        .execute("MATCH (p:Person) RETURN id(p), p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(id), Value::String(name)) => (*id, name.to_string()),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    let mut out: Vec<(String, i64)> = result
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[column]) {
            (Value::Int64(id), Value::Int64(value)) => (names[id].clone(), *value),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort();
    out
}

const PAGERANK_GQL: &str =
    "CALL grafeo.pagerank({}) YIELD node_id, score RETURN node_id, score ORDER BY node_id";

fn pagerank(db: &GrafeoDB, arguments: &str, values: &[(&str, Value)]) -> Vec<Vec<Value>> {
    let query = PAGERANK_GQL.replace("{}", arguments);
    rows(&db.execute_with_params(&query, params(values)).unwrap())
}

// ---------------------------------------------------------------------------
// Parameters run like literals
// ---------------------------------------------------------------------------

#[test]
fn positional_parameters_run_like_the_same_literals() {
    let db = chain_and_branch();
    let literal = pagerank(&db, "0.5, 1, 0.0001", &[]);
    let parameters = pagerank(
        &db,
        "$d, $m, $t",
        &[
            ("d", Value::Float64(0.5)),
            ("m", Value::Int64(1)),
            ("t", Value::Float64(0.0001)),
        ],
    );
    assert_eq!(
        parameters, literal,
        "$d, $m, $t must run like 0.5, 1, 0.0001"
    );
    assert_ne!(
        literal,
        pagerank(&db, "", &[]),
        "the literal arguments must change the scores, or the test proves nothing"
    );
}

#[test]
fn parameters_in_a_map_argument_run_like_the_same_literals() {
    let db = chain_and_branch();
    let literal = pagerank(&db, "{damping: 0.5, max_iterations: 1}", &[]);
    let parameters = pagerank(
        &db,
        "{damping: $d, max_iterations: $m}",
        &[("d", Value::Float64(0.5)), ("m", Value::Int64(1))],
    );
    assert_eq!(parameters, literal);
    assert_ne!(literal, pagerank(&db, "", &[]));
}

#[test]
fn a_map_parameter_names_the_arguments() {
    let db = chain_and_branch();
    let literal = pagerank(&db, "{damping: 0.5, max_iterations: 1}", &[]);
    let config = Value::Map(std::sync::Arc::new(
        [
            ("damping".into(), Value::Float64(0.5)),
            ("max_iterations".into(), Value::Int64(1)),
        ]
        .into_iter()
        .collect(),
    ));
    assert_eq!(pagerank(&db, "$config", &[("config", config)]), literal);
}

#[test]
fn a_required_argument_from_a_parameter_runs() {
    let db = chain_and_branch();
    let alix = id_of(&db, "Alix");
    let result = db
        .execute_with_params(
            "CALL grafeo.bfs($s) YIELD node_id, depth RETURN node_id, depth",
            params(&[("s", Value::Int64(alix))]),
        )
        .unwrap();
    assert_eq!(
        by_name(&db, &result, 1),
        vec![
            ("Alix".to_string(), 0),
            ("Gus".to_string(), 1),
            ("Jules".to_string(), 1),
            ("Mia".to_string(), 3),
            ("Vincent".to_string(), 2),
        ]
    );
}

#[test]
fn a_missing_parameter_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute("CALL grafeo.pagerank($d) YIELD score RETURN score")
        .expect_err("an argument without its parameter must fail, not run with the default");
    assert!(
        error.to_string().contains("Missing parameter: $d"),
        "unexpected error: {error}"
    );
}

#[test]
fn constant_expressions_run_like_their_values() {
    let db = chain_and_branch();
    let literal = pagerank(&db, "0.5, 1, 0.0001", &[]);
    assert_eq!(pagerank(&db, "0.25 + 0.25, 3 - 2, 0.0001", &[]), literal);
    assert_eq!(
        pagerank(
            &db,
            "$d / 2, $m, 0.0001",
            &[("d", Value::Float64(1.0)), ("m", Value::Int64(1))]
        ),
        literal
    );
    // The variable of a list comprehension is its own, not the row's.
    assert_eq!(
        pagerank(&db, "[x IN [1, 2] | x * 0.25][1], 1, 0.0001", &[]),
        literal
    );
}

#[test]
fn an_argument_that_reads_the_graph_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute("CALL grafeo.pagerank({damping: COUNT { MATCH (n) } / 10.0}) YIELD score")
        .expect_err("a subquery is not a constant");
    let message = error.to_string();
    assert!(
        message.contains("Argument 'damping' of grafeo.pagerank")
            && message.contains("must be a constant"),
        "unexpected error: {message}"
    );
}

#[test]
fn an_argument_that_cannot_be_evaluated_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute("CALL grafeo.pagerank(1 / 0) YIELD score")
        .expect_err("1 / 0 has no value to run with");
    assert!(
        error
            .to_string()
            .contains("Argument 'damping' of grafeo.pagerank cannot be evaluated"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_null_argument_keeps_the_default() {
    let db = chain_and_branch();
    assert_eq!(
        pagerank(&db, "$d, 3", &[("d", Value::Null)]),
        pagerank(&db, "0.85, 3", &[])
    );
}

#[test]
fn an_integer_runs_where_a_float_is_expected() {
    let db = chain_and_branch();
    // A damping of 1 (an integer) is the damping 1.0, not the default 0.85.
    assert_eq!(pagerank(&db, "1, 3", &[]), pagerank(&db, "1.0, 3", &[]));
    assert_eq!(
        pagerank(&db, "$d, 3", &[("d", Value::Int64(1))]),
        pagerank(&db, "1.0, 3", &[])
    );
    assert_ne!(pagerank(&db, "1.0, 3", &[]), pagerank(&db, "0.85, 3", &[]));
}

#[test]
fn an_argument_of_the_wrong_type_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute_with_params(
            "CALL grafeo.pagerank($d) YIELD score RETURN score",
            params(&[("d", Value::from("high"))]),
        )
        .expect_err("a string damping must fail, not run with the default damping");
    let message = error.to_string();
    assert!(
        message.contains("damping") && message.contains("grafeo.pagerank"),
        "the error must name the argument and the procedure: {message}"
    );
}

#[test]
fn a_parameter_used_in_a_stored_procedure_call_runs() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE add_city(name STRING) RETURNS (n INTEGER) AS { \
         INSERT (c:City {name: $name}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute_with_params(
        "CALL add_city($city) YIELD n RETURN n",
        params(&[("city", Value::from("Prague"))]),
    )
    .unwrap();
    db.execute("CALL add_city('Pra' || 'gue') YIELD n RETURN n")
        .unwrap();
    let cities = db
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(
        cities,
        vec![vec![Value::from("Prague")], vec![Value::from("Prague")]]
    );
}

// ---------------------------------------------------------------------------
// Cypher
// ---------------------------------------------------------------------------

#[cfg(feature = "cypher")]
#[test]
fn cypher_parameters_run_like_the_same_literals() {
    let db = chain_and_branch();
    let query =
        "CALL grafeo.pagerank({}) YIELD node_id, score RETURN node_id, score ORDER BY node_id";
    let literal = rows(
        &db.execute_cypher(&query.replace("{}", "0.5, 1, 0.0001"))
            .unwrap(),
    );
    let parameters = rows(
        &db.execute_cypher_with_params(
            &query.replace("{}", "$d, $m, $t"),
            params(&[
                ("d", Value::Float64(0.5)),
                ("m", Value::Int64(1)),
                ("t", Value::Float64(0.0001)),
            ]),
        )
        .unwrap(),
    );
    assert_eq!(parameters, literal);

    let alix = id_of(&db, "Alix");
    let bfs = db
        .execute_cypher_with_params(
            "CALL grafeo.bfs($s) YIELD node_id, depth RETURN node_id, depth",
            params(&[("s", Value::Int64(alix))]),
        )
        .unwrap();
    assert_eq!(bfs.row_count(), 5);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_argument_that_reads_a_row_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute_cypher(
            "MATCH (p:Person {name: 'Alix'}) WITH id(p) AS s \
             CALL grafeo.bfs(s) YIELD node_id RETURN node_id",
        )
        .expect_err("an argument read from a row must fail, not run without it");
    assert!(
        error.to_string().contains("must be a constant"),
        "unexpected error: {error}"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_missing_parameter_is_an_error() {
    let db = chain_and_branch();
    let error = db
        .execute_cypher("CALL grafeo.bfs($s) YIELD node_id RETURN node_id")
        .expect_err("a missing parameter must fail");
    assert!(
        error.to_string().contains("Missing parameter: $s"),
        "unexpected error: {error}"
    );
}

// ---------------------------------------------------------------------------
// Vector search arguments
// ---------------------------------------------------------------------------

#[cfg(all(feature = "vector-index", feature = "lpg"))]
mod vector {
    use super::*;

    fn docs() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        for (title, embedding) in [
            ("Amsterdam", [1.0_f32, 0.0, 0.0]),
            ("Berlin", [0.9, 0.1, 0.0]),
            ("Paris", [0.0, 1.0, 0.0]),
        ] {
            let node = db.create_node(&["Doc"]).unwrap();
            db.set_node_property(node, "title", Value::from(title))
                .unwrap();
            db.set_node_property(node, "emb", Value::Vector(embedding.to_vec().into()))
                .unwrap();
        }
        db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .unwrap();
        db
    }

    const SEARCH: &str = "CALL grafeo.search.vector('Doc', 'emb', {}, 2) \
                          YIELD node_id, distance RETURN node_id, distance";

    #[test]
    fn a_query_vector_from_a_parameter_runs_like_the_literal() {
        let db = docs();
        let literal = rows(
            &db.execute(&SEARCH.replace("{}", "[1.0, 0.0, 0.0]"))
                .unwrap(),
        );
        assert_eq!(literal.len(), 2);
        let as_list = Value::List(
            vec![
                Value::Float64(1.0),
                Value::Float64(0.0),
                Value::Float64(0.0),
            ]
            .into(),
        );
        assert_eq!(
            rows(
                &db.execute_with_params(&SEARCH.replace("{}", "$q"), params(&[("q", as_list)]))
                    .unwrap()
            ),
            literal
        );
        let as_vector = Value::Vector(vec![1.0_f32, 0.0, 0.0].into());
        assert_eq!(
            rows(
                &db.execute_with_params(&SEARCH.replace("{}", "$q"), params(&[("q", as_vector)]))
                    .unwrap()
            ),
            literal
        );
    }

    #[test]
    fn a_list_argument_keeps_its_computed_elements() {
        let db = docs();
        let literal = rows(
            &db.execute(&SEARCH.replace("{}", "[1.0, 0.0, 0.0]"))
                .unwrap(),
        );
        assert_eq!(
            rows(
                &db.execute(&SEARCH.replace("{}", "[1.0, 0.0 + 0.0, 0.0]"))
                    .unwrap()
            ),
            literal,
            "0.0 + 0.0 is an element of the query vector, not left out"
        );
    }

    #[test]
    fn a_query_vector_of_another_dimension_is_an_error() {
        let db = docs();
        for procedure in ["grafeo.search.vector", "grafeo.search.mmr"] {
            let error = db
                .execute_with_params(
                    &format!("CALL {procedure}('Doc', 'emb', $q, 2) YIELD node_id RETURN node_id"),
                    params(&[(
                        "q",
                        Value::List(vec![Value::Float64(1.0), Value::Float64(0.0)].into()),
                    )]),
                )
                .expect_err("a 2-dimensional query on a 3-dimensional index must fail, not panic");
            assert!(
                error.to_string().contains("dimension"),
                "{procedure}: unexpected error: {error}"
            );
        }
    }
}
