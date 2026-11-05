//! SPARQL `EXISTS` that reads the row, and `EXISTS` at each position an expression holds.
//!
//! SPARQL 1.1 section 18.6 substitutes the row's values into the pattern, so `EXISTS` answers per row
//! by them, also where the sub-query has no variable in common with the row. A variable the row
//! leaves unbound stays unbound. A sub-query that reads the row other than in its top filter is
//! refused for now, with an error and never a wrong answer. `correlated_exists.gtest` runs the same
//! table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_correlated_exists:: -- --ignored
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const DATA: &str = "INSERT DATA { <urn:a> <urn:p> 1 . <urn:a> <urn:q> 2 . <urn:a> <urn:q> 3 . <urn:a> <urn:r> 0 . <urn:b> <urn:p> 3 . <urn:b> <urn:q> 2 . <urn:b> <urn:t> 0 . <urn:e> <urn:q> 9 . GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:q> 2 } GRAPH <urn:g2> { <urn:d> <urn:p> 5 . <urn:d> <urn:q> 2 . <urn:e> <urn:p> 6 } }";

/// Name, query, rows, each its cells joined.
const READS_THE_ROW: &[(&str, &str, &[&str])] = &[
    (
        "shared_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
        &["urn:a"],
    ),
    (
        "shared_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
        &["urn:b"],
    ),
    (
        "no_shared_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x FILTER(?x > ?o) }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "no_shared_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { ?t <urn:q> ?x FILTER(?x > ?o) }) }",
        &[],
    ),
    (
        "no_shared_exists_none_above",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x FILTER(?x > ?o && ?x < 9) }) }",
        &["urn:a"],
    ),
    (
        "no_shared_not_exists_none_above",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { ?t <urn:q> ?x FILTER(?x > ?o && ?x < 9) }) }",
        &["urn:b"],
    ),
    (
        "conjunct_and_a_local",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o && ?x < 3) }) }",
        &["urn:a"],
    ),
    (
        "conjunct_and_a_local_finds_none",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o && ?x < 2) }) }",
        &[],
    ),
    (
        "conjunct_and_a_local_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { ?s <urn:q> ?x FILTER(?x > ?o && ?x < 2) }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "two_filters",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) FILTER(?x = 3) }) }",
        &["urn:a"],
    ),
    (
        "two_row_conjuncts",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) FILTER(?x <= ?o + 1) }) }",
        &["urn:a"],
    ),
    (
        "an_or_is_one_conjunct",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o || ?x = 2) }) }",
        &["urn:a", "urn:b"],
    ),
    (
        "the_row_on_both_sides",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?o + ?x = 5) }) }",
        &["urn:b"],
    ),
    (
        "the_row_alone",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?o = 3) }) }",
        &["urn:b"],
    ),
    (
        "unbound_in_the_row_is_false",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:r> ?w } FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?w) }) }",
        &["urn:a"],
    ),
    (
        "unbound_in_the_row_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:r> ?w } FILTER(NOT EXISTS { ?s <urn:q> ?x FILTER(?x > ?w) }) }",
        &["urn:b"],
    ),
    (
        "unbound_in_the_other_row",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:t> ?w } FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?w) }) }",
        &["urn:b"],
    ),
    (
        "unbound_read_by_bound",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:r> ?w } FILTER(EXISTS { ?s <urn:q> ?x FILTER(BOUND(?w)) }) }",
        &["urn:a"],
    ),
    (
        "unbound_read_by_coalesce",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:r> ?w } FILTER(EXISTS { ?s <urn:q> ?x FILTER(COALESCE(?w, 5) > 1) }) }",
        &["urn:b"],
    ),
    (
        "two_found_one_row",
        "SELECT ?s ?o WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
        &["urn:a|1"],
    ),
    (
        "count_of_rows",
        "SELECT (COUNT(*) AS ?c) WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x FILTER(?x > ?o) }) }",
        &["2"],
    ),
    (
        "no_row_variable_read",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > 2) }) }",
        &["urn:a"],
    ),
    (
        "in_a_bind",
        "SELECT ?s ?e WHERE { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) } AS ?e) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "in_a_projection",
        "SELECT ?s (EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) } AS ?e) WHERE { ?s <urn:p> ?o }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "in_an_if",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(IF(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }, ?o = 1, ?o = 9)) }",
        &["urn:a"],
    ),
    (
        "under_an_and",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(?o > 0 && EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
        &["urn:a"],
    ),
    (
        "inside_each_graph",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) } }",
        &["urn:g1|urn:c"],
    ),
    (
        "inside_each_graph_not_exists",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(NOT EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) } }",
        &["urn:g2|urn:d", "urn:g2|urn:e"],
    ),
];

const ORDERED: &[(&str, &str, &[&str])] = &[(
    "in_order_by",
    "SELECT ?s WHERE { ?s <urn:p> ?o } ORDER BY (EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) })",
    &["urn:b", "urn:a"],
)];

/// Name, statement, a read and its rows.
const UPDATES: &[(&str, &str, &str, &[&str])] = &[(
    "update_where",
    "INSERT { ?s <urn:w> 1 } WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
    "SELECT ?s WHERE { ?s <urn:w> 1 }",
    &["urn:a"],
)];

const REFUSED: &[(&str, &str)] = &[
    (
        "a_filter_in_a_nested_group",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x { FILTER(?x > ?o) } }) }",
    ),
    (
        "a_filter_in_a_nested_group_beside_a_triple",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x { ?s <urn:q> ?y FILTER(?y > ?o) } }) }",
    ),
    (
        "a_filter_in_optional",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x OPTIONAL { ?s <urn:r> ?y FILTER(?y > ?o) } }) }",
    ),
    (
        "a_filter_in_union",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { ?s <urn:q> ?x FILTER(?x > ?o) } UNION { ?s <urn:r> ?x } }) }",
    ),
    (
        "a_filter_in_minus",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x MINUS { ?s <urn:r> ?y FILTER(?y > ?o) } }) }",
    ),
    (
        "a_filter_in_graph",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { GRAPH ?h { ?s <urn:q> ?x FILTER(?x > ?o) } }) }",
    ),
    (
        "a_bind",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) }) }",
    ),
    (
        "a_nested_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(EXISTS { ?s <urn:r> ?y FILTER(?y > ?o) }) }) }",
    ),
    (
        "a_nested_not_exists",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(NOT EXISTS { ?s <urn:r> ?y FILTER(?y > ?o) }) }) }",
    ),
    (
        "a_sub_select",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s <urn:q> ?x FILTER(?x > ?o) } } }) }",
    ),
    (
        "a_sub_select_having",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s <urn:q> ?x } GROUP BY ?s HAVING(COUNT(?x) > ?o) } }) }",
    ),
    (
        "may_leave_unbound_and_the_row_binds",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x OPTIONAL { ?t <urn:r> ?o } }) }",
    ),
    (
        "may_leave_unbound_read_by_a_filter",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x OPTIONAL { ?t <urn:r> ?o } FILTER(?o = 1) }) }",
    ),
    (
        "in_a_bind",
        "SELECT ?s ?e WHERE { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) } AS ?e) }",
    ),
    (
        "in_a_projection",
        "SELECT ?s (EXISTS { ?s <urn:q> ?x BIND(?x + ?o AS ?y) } AS ?e) WHERE { ?s <urn:p> ?o }",
    ),
];

/// Name, query, the columns, and rows: a mark for `EXISTS` must not show in the columns.
const POSITIONS: &[(&str, &str, &[&str], &[&str])] = &[
    (
        "select_star_of_a_filter",
        "SELECT * WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:r> ?y }) }",
        &["s", "o"],
        &["urn:a|1"],
    ),
    (
        "select_star_of_a_bind",
        "SELECT * WHERE { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:r> ?y } AS ?e) }",
        &["s", "o", "e"],
        &["urn:a|1|true", "urn:b|3|false"],
    ),
    (
        "select_star_of_a_sub_select",
        "SELECT * WHERE { { SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:r> ?y }) } } }",
        &["s"],
        &["urn:a"],
    ),
    (
        "select_star_of_a_correlated",
        "SELECT * WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x FILTER(?x > ?o) }) }",
        &["s", "o"],
        &["urn:a|1"],
    ),
    (
        "having",
        "SELECT ?s (COUNT(?v) AS ?c) WHERE { ?s ?p ?v } GROUP BY ?s HAVING(EXISTS { ?s <urn:r> ?x })",
        &["s", "c"],
        &["urn:a|4"],
    ),
    (
        "having_not_exists",
        "SELECT ?s (COUNT(?v) AS ?c) WHERE { ?s ?p ?v } GROUP BY ?s HAVING(NOT EXISTS { ?s <urn:r> ?x })",
        &["s", "c"],
        &["urn:b|3", "urn:e|1"],
    ),
    (
        "an_aggregates_argument",
        "SELECT (SUM(IF(EXISTS { ?s <urn:r> ?x }, 1, 0)) AS ?c) WHERE { ?s <urn:p> ?o }",
        &["c"],
        &["1"],
    ),
    (
        "an_aggregates_argument_per_group",
        "SELECT ?o (SUM(IF(EXISTS { ?s <urn:r> ?x }, 1, 0)) AS ?c) WHERE { ?s <urn:p> ?o } GROUP BY ?o",
        &["o", "c"],
        &["1|1", "3|0"],
    ),
    (
        "inside_graph_a_filter",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:q> ?x }) } }",
        &["g", "s"],
        &["urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    (
        "inside_graph_a_bind",
        "SELECT ?g ?s ?e WHERE { GRAPH ?g { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:q> ?x } AS ?e) } }",
        &["g", "s", "e"],
        &[
            "urn:g1|urn:c|true",
            "urn:g2|urn:d|true",
            "urn:g2|urn:e|false",
        ],
    ),
    (
        "distinct_of_a_projection",
        "SELECT DISTINCT (EXISTS { ?s <urn:r> ?x } AS ?e) WHERE { ?s <urn:p> ?o }",
        &["e"],
        &["true", "false"],
    ),
];

fn data() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute_sparql(DATA).unwrap();
    db
}

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        // A term by its plain value: an IRI without its brackets, a literal without its quotes and type.
        Value::RdfTerm(term) => {
            let text = term.to_ntriples();
            match text.strip_prefix('<') {
                Some(iri) => iri.trim_end_matches('>').to_string(),
                None => text.split('"').nth(1).unwrap_or("").to_string(),
            }
        }
        other => format!("{other:?}"),
    }
}

fn rows_in_order(db: &GrafeoDB, query: &str) -> Result<Vec<String>, String> {
    let result = db
        .session()
        .execute_sparql(query)
        .map_err(|e| e.to_string())?;
    Ok(result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect())
}

fn rows_of(db: &GrafeoDB, query: &str) -> Result<Vec<String>, String> {
    let mut rows = rows_in_order(db, query)?;
    rows.sort();
    Ok(rows)
}

fn sorted(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(ToString::to_string).collect();
    rows.sort();
    rows
}

#[test]
#[ignore = "waits for #740: EXISTS reads the row by the Architect's ruling"]
fn exists_answers_per_row_by_the_rows_values() {
    let db = data();
    let mut wrong = Vec::new();
    for (name, query, wanted) in READS_THE_ROW {
        match rows_of(&db, query) {
            Ok(rows) if rows == sorted(wanted) => {}
            other => wrong.push(format!("{name}: {query}: {other:?}, wanted {wanted:?}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
#[ignore = "waits for #740: EXISTS reads the row by the Architect's ruling"]
fn exists_that_reads_the_row_orders_rows() {
    let db = data();
    for (name, query, wanted) in ORDERED {
        let rows = rows_in_order(&db, query).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(rows, *wanted, "{name}: {query}");
    }
}

#[test]
#[ignore = "waits for #740: EXISTS reads the row by the Architect's ruling"]
fn an_update_reads_the_row_in_its_exists() {
    for (name, statement, read, wanted) in UPDATES {
        let db = data();
        rows_of(&db, statement).unwrap_or_else(|error| panic!("{name}: {statement}: {error}"));
        assert_eq!(rows_of(&db, read).unwrap(), sorted(wanted), "{name}");
    }
}

/// An error, naming `EXISTS`, and not rows of any kind: on 0.6.1 these answered wrongly.
#[test]
#[ignore = "waits for #740: EXISTS reads the row by the Architect's ruling"]
fn exists_that_reads_the_row_elsewhere_is_refused() {
    let db = data();
    let mut wrong = Vec::new();
    for (name, query) in REFUSED {
        match rows_of(&db, query) {
            Err(error) if error.contains("EXISTS") => {}
            other => wrong.push(format!("{name}: {query}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Each position holds an `EXISTS`, and what it adds to a row stays out of the columns.
#[test]
#[ignore = "waits for #740: EXISTS outside a FILTER"]
fn exists_holds_at_each_position() {
    let db = data();
    let mut wrong = Vec::new();
    for (name, query, columns, wanted) in POSITIONS {
        let result = match db.session().execute_sparql(query) {
            Ok(result) => result,
            Err(error) => {
                wrong.push(format!("{name}: {query}: {error}"));
                continue;
            }
        };
        if result.columns != *columns {
            wrong.push(format!(
                "{name}: columns {:?}, wanted {columns:?}",
                result.columns
            ));
        }
        match rows_of(&db, query) {
            Ok(rows) if rows == sorted(wanted) => {}
            other => wrong.push(format!("{name}: {query}: {other:?}, wanted {wanted:?}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
