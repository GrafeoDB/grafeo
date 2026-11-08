//! SPARQL `EXISTS` that reads the row, and `EXISTS` at each position an expression holds.
//!
//! SPARQL 1.1 section 18.6 substitutes the row's values into the pattern, so `EXISTS` answers per row
//! by them, also where the sub-query has no variable in common with the row. A variable the row
//! leaves unbound stays unbound. `correlated_exists.gtest` runs the same table on every binding, and
//! `sparql_exists_per_row.rs` holds the forms that read the row anywhere in the pattern.
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
fn exists_that_reads_the_row_orders_rows() {
    let db = data();
    for (name, query, wanted) in ORDERED {
        let rows = rows_in_order(&db, query).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(rows, *wanted, "{name}: {query}");
    }
}

#[test]
fn an_update_reads_the_row_in_its_exists() {
    for (name, statement, read, wanted) in UPDATES {
        let db = data();
        rows_of(&db, statement).unwrap_or_else(|error| panic!("{name}: {statement}: {error}"));
        assert_eq!(rows_of(&db, read).unwrap(), sorted(wanted), "{name}");
    }
}

/// Each position holds an `EXISTS`, and what it adds to a row stays out of the columns.
#[test]
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

/// Name, data, query, rows.
const WITH_THEIR_DATA: &[(&str, &str, &str, &[&str])] = &[
    (
        "a_filter_in_graph",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:a> <urn:q> 2 . <urn:b> <urn:p> 3 . <urn:b> <urn:q> 2 . <urn:c> <urn:p> 1 . <urn:d> <urn:p> 3 . GRAPH <urn:g1> { <urn:c> <urn:q> 2 . <urn:d> <urn:q> 2 } }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { GRAPH ?h { ?s <urn:q> ?x FILTER(?x > ?o) } }) }",
        &["urn:c"],
    ),
    (
        "a_filter_in_graph_not",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:a> <urn:q> 2 . <urn:b> <urn:p> 3 . <urn:b> <urn:q> 2 . <urn:c> <urn:p> 1 . <urn:d> <urn:p> 3 . GRAPH <urn:g1> { <urn:c> <urn:q> 2 . <urn:d> <urn:q> 2 } }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { GRAPH ?h { ?s <urn:q> ?x FILTER(?x > ?o) } }) }",
        &["urn:a", "urn:b", "urn:d"],
    ),
    (
        "a_filter_in_a_named_graph",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:a> <urn:q> 2 . <urn:b> <urn:p> 3 . <urn:b> <urn:q> 2 . <urn:c> <urn:p> 1 . <urn:d> <urn:p> 3 . GRAPH <urn:g1> { <urn:c> <urn:q> 2 . <urn:d> <urn:q> 2 } }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { GRAPH <urn:g1> { ?s <urn:q> ?x FILTER(?x > ?o) } }) }",
        &["urn:c"],
    ),
    (
        "sub_select_lists_the_graph_over_an_optional",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g ?s WHERE { GRAPH ?g { { SELECT ?g ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:link> ?g } } } } }",
        &["urn:g1|urn:c", "urn:g1|urn:y", "urn:g2|urn:d"],
    ),
    (
        "sub_select_lists_the_graph_it_never_binds_is_filled",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g ?s WHERE { GRAPH ?g { { SELECT ?g ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:none> ?g } } } } }",
        &[
            "urn:g1|urn:c",
            "urn:g1|urn:x",
            "urn:g1|urn:y",
            "urn:g2|urn:d",
        ],
    ),
    (
        "sub_select_lists_another_value_of_the_graph_is_dropped",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g ?s WHERE { GRAPH ?g { { SELECT ?g ?s WHERE { ?s <urn:away> ?g } } } }",
        &[],
    ),
    (
        "sub_select_lists_the_graph_as_its_only_value",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g WHERE { GRAPH ?g { { SELECT ?g WHERE { ?s <urn:link> ?g } } } }",
        &["urn:g1"],
    ),
    (
        "row_key_unbound_is_free",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:z> ?k } FILTER(EXISTS { ?k <urn:q> ?x FILTER(?x > 2) }) }",
        &["urn:b"],
    ),
    (
        "row_key_unbound_is_free_not",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:z> ?k } FILTER(NOT EXISTS { ?k <urn:q> ?x FILTER(?x > 2) }) }",
        &["urn:a"],
    ),
    (
        "row_key_unbound_with_a_filter_on_the_row",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:z> ?k } FILTER(EXISTS { ?k <urn:q> ?x FILTER(?x > ?o && ?x < 5) }) }",
        &["urn:a"],
    ),
    (
        "row_key_unbound_with_a_filter_on_the_row_not",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:z> ?k } FILTER(NOT EXISTS { ?k <urn:q> ?x FILTER(?x > ?o && ?x < 5) }) }",
        &["urn:b"],
    ),
    (
        "a_filter_reading_a_row_value_left_unbound_is_false",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:w> ?k } FILTER(EXISTS { ?t <urn:q> ?x FILTER(?x = ?k) }) }",
        &["urn:a"],
    ),
    (
        "a_filter_reading_a_row_value_left_unbound_is_false_not",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o OPTIONAL { ?s <urn:w> ?k } FILTER(NOT EXISTS { ?t <urn:q> ?x FILTER(?x = ?k) }) }",
        &["urn:b"],
    ),
];

/// Name, data, query, rows of the statement it profiles.
const PROFILED: &[(&str, &str, &str, usize)] = &[
    (
        "graph_variable",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o } }",
        4,
    ),
    (
        "graph_variable_empty_group",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g WHERE { GRAPH ?g { } }",
        2,
    ),
    (
        "graph_of_a_sub_select",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?g ?c WHERE { GRAPH ?g { { SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } } } }",
        2,
    ),
    (
        "graph_not_in_the_dataset",
        "INSERT DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 1 . <urn:c> <urn:link> <urn:g1> . <urn:x> <urn:p> 2 . <urn:x> <urn:link> <urn:other> . <urn:y> <urn:p> 3 } GRAPH <urn:g2> { <urn:d> <urn:p> 4 . <urn:d> <urn:away> <urn:g1> } }",
        "SELECT ?x WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        0,
    ),
    (
        "marked_exists",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:w> ?y }) }",
        1,
    ),
    (
        "marked_not_exists",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(NOT EXISTS { ?s <urn:w> ?y }) }",
        1,
    ),
    (
        "marked_reading_the_row",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(EXISTS { ?t <urn:q> ?x FILTER(?x > ?o) }) }",
        2,
    ),
    (
        "marked_in_a_bind",
        "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:p> 3 . <urn:a> <urn:z> <urn:b> . <urn:a> <urn:w> 2 . <urn:b> <urn:q> 2 . <urn:e> <urn:q> 9 }",
        "SELECT ?s ?e WHERE { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:w> ?y } AS ?e) }",
        2,
    ),
];

/// A sub-SELECT listing the graph's variable joins with the graph, and a row key the row leaves
/// unbound is free in the sub-query.
#[test]
fn a_listed_graph_and_an_unbound_row_key_answer_by_the_join() {
    let mut wrong = Vec::new();
    for (name, data, query, want) in WITH_THEIR_DATA {
        let db = GrafeoDB::new_in_memory();
        db.session().execute_sparql(data).unwrap();
        let found = rows_of(&db, query);
        if found.as_ref() != Ok(&sorted(want)) {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `PROFILE` runs what the plain statement runs: it shows each operator, and its rows are the
/// statement's, over `GRAPH ?g` and over a test of `EXISTS`, which the plan marks.
#[test]
fn profile_reads_a_graph_variable_and_a_marked_query() {
    for (name, data, query, rows) in PROFILED {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute_sparql(data).unwrap();
        let plain = session.execute_sparql(query).unwrap();
        assert_eq!(plain.row_count(), *rows, "{name}");
        let profiled = session.execute_sparql(&format!("PROFILE {query}"));
        let profiled = profiled.unwrap_or_else(|error| panic!("{name}: {error}"));
        let text = format!("{:?}", profiled.rows());
        assert!(text.contains("rows"), "{name}: {text}");
        let explained = session.execute_sparql(&format!("EXPLAIN {query}"));
        explained.unwrap_or_else(|error| panic!("{name}: {error}"));
        // Profiling runs the statement once and leaves the next run as it was.
        assert_eq!(
            session.execute_sparql(query).unwrap().rows(),
            plain.rows(),
            "{name}"
        );
    }
}
