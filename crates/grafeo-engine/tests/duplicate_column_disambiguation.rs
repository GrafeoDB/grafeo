//! Duplicate result column names (#371).
//!
//! Bindings read rows by column name, so two columns with the same name lose
//! data silently. A `QueryResult` with a repeated column name is rejected with
//! an error (eager and streaming paths), and unaliased expressions are named
//! after their source text so distinct expressions (`id(s)`, `id(t)`,
//! `count(a)`, `count(b)`, two different `CASE`s) get distinct names without
//! aliases.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Two `Person` nodes (ids 0 and 1) joined by a single `KNOWS` edge (id 0):
/// `(s {name:"s", age:30}) -[r:KNOWS]-> (t {name:"t", age:25})`.
fn one_edge() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let s = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("s".into())),
                ("age", Value::Int64(30)),
            ],
        )
        .unwrap();
    let t = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("t".into())),
                ("age", Value::Int64(25)),
            ],
        )
        .unwrap();
    session.create_edge(s, t, "KNOWS");
    db
}

fn assert_duplicate_column_error(
    result: grafeo_common::utils::error::Result<impl std::fmt::Debug>,
) {
    match result {
        Ok(r) => panic!("expected a duplicate-column error, got {r:?}"),
        Err(e) => assert!(
            e.to_string().contains("duplicate column name"),
            "error should name the duplicate column, got: {e}"
        ),
    }
}

// Unaliased id() calls get distinct, correctly valued columns.
#[test]
fn unaliased_id_calls_get_distinct_names() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (s)-[r]->(t) RETURN id(s), id(t), id(r)")
        .unwrap();
    assert_eq!(r.columns, vec!["id(s)", "id(t)", "id(r)"]);
    assert_eq!(r.row_count(), 1);
    assert_eq!(
        r.rows()[0],
        vec![Value::Int64(0), Value::Int64(1), Value::Int64(0)]
    );
}

// A repeated property column (a.name, a.name) is rejected.
#[test]
fn duplicate_property_column_is_rejected() {
    let db = one_edge();
    assert_duplicate_column_error(
        db.session()
            .execute("MATCH (a:Person) RETURN a.name, a.name"),
    );
}

// A repeated alias (AS x, AS x) is rejected.
#[test]
fn duplicate_alias_is_rejected() {
    let db = one_edge();
    assert_duplicate_column_error(
        db.session()
            .execute("MATCH (s)-[r]->(t) RETURN id(s) AS x, id(t) AS x"),
    );
}

// Control: distinct aliases are accepted.
#[test]
fn distinct_aliases_are_accepted() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (s)-[r]->(t) RETURN id(s) AS sid, id(t) AS tid")
        .unwrap();
    assert_eq!(r.columns, vec!["sid", "tid"]);
    assert_eq!(r.rows()[0], vec![Value::Int64(0), Value::Int64(1)]);
}

// Arithmetic and unary expressions are named as written, literals included.
#[test]
fn arithmetic_expressions_are_named_as_written() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (a:Person {name: 's'}) RETURN a.age + 1, -a.age, (a.age + 1) * 2")
        .unwrap();
    assert_eq!(r.columns, vec!["a.age + 1", "-a.age", "(a.age + 1) * 2"]);
    assert_eq!(
        r.rows()[0],
        vec![Value::Int64(31), Value::Int64(-30), Value::Int64(62)]
    );
}

// Literals are named as they are written, not by their internal type: `1` and
// `1.0` stay distinct, strings are single-quoted with quotes escaped.
#[test]
fn literals_are_named_as_written() {
    let db = one_edge();
    let r = db
        .session()
        .execute(r#"RETURN 1, 1.0, 'x', "it's", true, NULL, [1, 2]"#)
        .unwrap();
    assert_eq!(
        r.columns,
        vec!["1", "1.0", "'x'", r"'it\'s'", "true", "NULL", "[1, 2]"]
    );
    assert_eq!(r.rows()[0][0], Value::Int64(1));
    assert_eq!(r.rows()[0][1], Value::Float64(1.0));
    assert_eq!(r.rows()[0][3], Value::String("it's".into()));
}

// Two different CASE expressions used to both be named `case` and collide.
#[test]
fn different_case_expressions_get_distinct_names() {
    let db = one_edge();
    let r = db
        .session()
        .execute(
            "MATCH (a:Person {name: 's'}) \
             RETURN CASE WHEN a.age > 26 THEN 1 ELSE 0 END, \
                    CASE WHEN a.age > 40 THEN 1 ELSE 0 END, \
                    CASE a.name WHEN 's' THEN 'yes' END",
        )
        .unwrap();
    assert_eq!(
        r.columns,
        vec![
            "CASE WHEN a.age > 26 THEN 1 ELSE 0 END",
            "CASE WHEN a.age > 40 THEN 1 ELSE 0 END",
            "CASE a.name WHEN 's' THEN 'yes' END",
        ]
    );
    assert_eq!(
        r.rows()[0],
        vec![
            Value::Int64(1),
            Value::Int64(0),
            Value::String("yes".into())
        ]
    );
}

// Unaliased aggregates render their arguments.
#[test]
fn unaliased_aggregates_get_distinct_names() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN count(a), count(b)")
        .unwrap();
    assert_eq!(r.columns, vec!["count(a)", "count(b)"]);
    assert_eq!(r.rows()[0], vec![Value::Int64(1), Value::Int64(1)]);

    let r = db
        .session()
        .execute("MATCH (a:Person) RETURN count(*), count(DISTINCT a)")
        .unwrap();
    assert_eq!(r.columns, vec!["count(*)", "count(DISTINCT a)"]);
}

// The percentile and the separator are part of an aggregate's name: two
// percentiles of one column are different results.
#[test]
fn aggregate_parameters_are_part_of_the_name() {
    let db = one_edge();
    let r = db
        .session()
        .execute(
            "MATCH (a:Person) \
             RETURN percentile_cont(a.age, 0.5), percentile_cont(a.age, 0.9)",
        )
        .unwrap();
    assert_eq!(
        r.columns,
        vec!["percentile_cont(a.age, 0.5)", "percentile_cont(a.age, 0.9)"]
    );
    assert_eq!(
        r.rows()[0],
        vec![Value::Float64(27.5), Value::Float64(29.5)]
    );

    let r = db
        .session()
        .execute("MATCH (a:Person) RETURN group_concat(a.name, ';'), group_concat(a.name, '|')")
        .unwrap();
    assert_eq!(
        r.columns,
        vec!["group_concat(a.name, ';')", "group_concat(a.name, '|')"]
    );
}

// Cypher expressions that used to share a generic name (`n{...}`, `reduce`,
// `list_comprehension`) are rendered in full.
#[cfg(feature = "cypher")]
#[test]
fn cypher_projections_and_comprehensions_get_distinct_names() {
    let db = one_edge();
    let session = db.session();

    let r = session
        .execute_cypher("MATCH (a:Person {name: 's'}) RETURN a{.name}, a{.age, next: a.age + 1}")
        .unwrap();
    assert_eq!(r.columns, vec!["a{.name}", "a{.age, next: a.age + 1}"]);

    let r = session
        .execute_cypher(
            "RETURN reduce(acc = 0, x IN [1, 2] | acc + x), reduce(acc = 0, x IN [3, 4] | acc + x)",
        )
        .unwrap();
    assert_eq!(
        r.columns,
        vec![
            "reduce(acc = 0, x IN [1, 2] | acc + x)",
            "reduce(acc = 0, x IN [3, 4] | acc + x)",
        ]
    );
    assert_eq!(r.rows()[0], vec![Value::Int64(3), Value::Int64(7)]);

    let r = session
        .execute_cypher(
            "RETURN [x IN [1, 2] | x * 2], [x IN [1, 2] WHERE x > 1], \
                    all(x IN [1, 2] WHERE x > 0), any(x IN [1, 2] WHERE x > 1)",
        )
        .unwrap();
    assert_eq!(
        r.columns,
        vec![
            "[x IN [1, 2] | x * 2]",
            "[x IN [1, 2] WHERE x > 1]",
            "all(x IN [1, 2] WHERE x > 0)",
            "any(x IN [1, 2] WHERE x > 1)",
        ]
    );
}

// SPARQL: `SELECT ?s ?s` is rejected; a repeated `AS ?x` alias fails at parse time.
#[cfg(feature = "sparql")]
fn rdf_db_one_triple() -> GrafeoDB {
    use grafeo_engine::config::{Config, GraphModel};
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <urn:s> <urn:p> <urn:o> }")
        .unwrap();
    db
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_duplicate_variable_is_rejected() {
    let db = rdf_db_one_triple();
    assert_duplicate_column_error(
        db.session()
            .execute_sparql("SELECT ?s ?s WHERE { ?s ?p ?o }"),
    );
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_duplicate_alias_is_rejected_at_parse() {
    let db = rdf_db_one_triple();
    match db
        .session()
        .execute_sparql("SELECT (?s AS ?x) (?o AS ?x) WHERE { ?s ?p ?o }")
    {
        Ok(r) => panic!("expected a parse error, got columns {:?}", r.columns),
        Err(e) => assert!(
            e.to_string().contains("duplicate projection variable '?x'"),
            "parse error should name the repeated alias, got: {e}"
        ),
    }
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_distinct_variables_are_accepted() {
    let db = rdf_db_one_triple();
    let r = db
        .session()
        .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .unwrap();
    assert_eq!(r.columns, vec!["s"]);
    assert_eq!(r.row_count(), 1);
}

// Streaming: duplicate column names are rejected when the stream opens.
#[test]
fn streaming_rejects_duplicate_columns_at_open() {
    let db = one_edge();
    assert_duplicate_column_error(db.execute_streaming("MATCH (a:Person) RETURN a.name, a.name"));

    let stream = db
        .execute_streaming("MATCH (s)-[r]->(t) RETURN id(s), id(t), id(r)")
        .unwrap();
    assert_eq!(stream.columns(), ["id(s)", "id(t)", "id(r)"]);
}
