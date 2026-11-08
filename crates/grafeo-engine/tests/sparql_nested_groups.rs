//! A group is its own scope, whatever it holds (SPARQL 1.1 18.2.2, 18.3.2.7): a `FILTER` alone in a nested group
//! does not read the outer group's variables, a `BIND` alone there starts from no variables, and an `OPTIONAL` or
//! `MINUS` alone there has the empty solution on its left, then joins. A group of one triple block, `UNION`, `GRAPH`,
//! sub-SELECT or `VALUES` answers as that element does. `nested_groups.gtest` runs the same table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_nested_groups::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

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

fn rows_in_order(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let result = session
        .execute_sparql(query)
        .map_err(|error| error.to_string())?;
    Ok(result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect())
}

fn rows_of(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let mut rows = rows_in_order(session, query)?;
    rows.sort();
    Ok(rows)
}

fn holding(data: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute_sparql(data).unwrap();
    db
}

/// Every case runs, so one failure names all that fail with it.
fn each(data: &str, cases: &[(&str, &str, &[&str])]) {
    let db = holding(data);
    let session = db.session();
    let mut wrong = Vec::new();
    for (name, query, want) in cases {
        let mut want: Vec<String> = want.iter().map(ToString::to_string).collect();
        want.sort();
        let found = rows_of(&session, query);
        if found.as_ref() != Ok(&want) {
            wrong.push(format!(
                "{name}: {query}\n  found {found:?}\n  want {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        cases.len(),
        wrong.join("\n")
    );
}

const DATA: &str = "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :a :q 2 . :b :q 2 . GRAPH :g { :a :t :k } }";

const CASES: &[(&str, &str, &[&str])] = &[
    (
        "filter_reads_the_outer_variable",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x { FILTER(?x > 0) } }",
        &[],
    ),
    (
        "filter_bound_of_the_outer_variable",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { FILTER(BOUND(?s)) } }",
        &[],
    ),
    (
        "filter_not_bound_of_the_outer_variable",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { FILTER(!BOUND(?s)) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "filter_with_no_variable_true",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { FILTER(1 > 0) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "filter_with_no_variable_false",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { FILTER(1 > 2) } }",
        &[],
    ),
    (
        "filter_in_two_groups",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x { { FILTER(?x > 0) } } }",
        &[],
    ),
    (
        "filter_group_first",
        "PREFIX : <urn:> SELECT ?s WHERE { { FILTER(?s = :a) } ?s :t :k }",
        &[],
    ),
    (
        "filter_group_between",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { FILTER(?s = :a) } ?s :q ?x }",
        &[],
    ),
    (
        "filter_beside_a_triple_is_right",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x { ?s :q ?y FILTER(?x > 0) } }",
        &[],
    ),
    (
        "filter_of_its_own_triple",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x { ?s :q ?y FILTER(?y > 0) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "filter_at_the_same_level_reads_it",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x FILTER(?x > 0) }",
        &["urn:a", "urn:b"],
    ),
    (
        "bind_reads_the_outer_variable",
        "PREFIX : <urn:> SELECT ?s ?y WHERE { ?s :t :k { BIND(?s AS ?y) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "bind_of_a_literal",
        "PREFIX : <urn:> SELECT ?s ?y WHERE { ?s :t :k { BIND(1 AS ?y) } }",
        &["urn:a|1", "urn:b|1"],
    ),
    (
        "bind_of_an_outer_variable_name",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { BIND(:a AS ?s) } }",
        &["urn:a"],
    ),
    (
        "optional_joins_as_a_group",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { OPTIONAL { ?s :p ?o } } }",
        &["urn:a"],
    ),
    (
        "optional_with_no_match_keeps_all",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { OPTIONAL { ?s :none ?o } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "optional_with_a_filter",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { OPTIONAL { ?s :p ?o FILTER(?o = :c) } } }",
        &["urn:a"],
    ),
    (
        "optional_at_the_same_level_keeps_all",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k OPTIONAL { ?s :p ?o } }",
        &["urn:a", "urn:b"],
    ),
    (
        "minus_takes_nothing_from_a_group_of_none",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { MINUS { ?s :p ?o } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "minus_at_the_same_level_takes",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k MINUS { ?s :p ?o } }",
        &["urn:b"],
    ),
    (
        "optional_of_a_filter_alone_keeps_the_left",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :q ?x OPTIONAL { FILTER(?x > 5) } }",
        &["urn:a", "urn:b"],
    ),
    (
        "optional_of_a_group_of_a_filter",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k OPTIONAL { { FILTER(?s = :a) } } }",
        &["urn:a", "urn:b"],
    ),
    (
        "union_branch_of_a_filter",
        "PREFIX : <urn:> SELECT ?s WHERE { { { FILTER(1 > 2) } } UNION { ?s :p ?o } }",
        &["urn:a"],
    ),
    (
        "filter_group_in_a_graph",
        "PREFIX : <urn:> SELECT ?s WHERE { GRAPH :g { ?s :t :k { FILTER(BOUND(?s)) } } }",
        &[],
    ),
    (
        "a_triple_alone",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { ?s :p ?o } }",
        &["urn:a"],
    ),
    (
        "a_union_alone",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { { ?s :p ?o } UNION { ?s :none ?o } } }",
        &["urn:a"],
    ),
    (
        "a_graph_alone",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { GRAPH :g { ?s :t :k } } }",
        &["urn:a"],
    ),
    (
        "a_sub_select_alone",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { SELECT ?s WHERE { ?s :p ?o } } }",
        &["urn:a"],
    ),
    (
        "values_alone",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { VALUES ?s { :a } } }",
        &["urn:a"],
    ),
    (
        "an_empty_group",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k { } }",
        &["urn:a", "urn:b"],
    ),
];
/// Name, update, the query that shows it, rows.
const UPDATES: &[(&str, &str, &str, &[&str])] = &[
    (
        "update_filter_group",
        "PREFIX : <urn:> INSERT { :r :saw ?s } WHERE { ?s :q ?x { FILTER(?x > 0) } }",
        "PREFIX : <urn:> SELECT ?s WHERE { :r :saw ?s }",
        &[],
    ),
    (
        "update_optional_group",
        "PREFIX : <urn:> INSERT { :r :saw ?s } WHERE { ?s :t :k { OPTIONAL { ?s :p ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { :r :saw ?s }",
        &["urn:a"],
    ),
    (
        "update_minus_group",
        "PREFIX : <urn:> DELETE { ?s :t :k } WHERE { ?s :t :k { MINUS { ?s :p ?o } } }",
        "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k }",
        &[],
    ),
];

#[test]
fn a_nested_group_is_its_own_scope() {
    each(DATA, CASES);
}

#[test]
fn an_update_reads_a_nested_group_as_a_query_does() {
    for (name, update, query, want) in UPDATES {
        let db = holding(DATA);
        let session = db.session();
        session
            .execute_sparql(update)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let want: Vec<String> = want.iter().map(ToString::to_string).collect();
        assert_eq!(rows_of(&session, query).unwrap(), want, "{name}");
    }
}
