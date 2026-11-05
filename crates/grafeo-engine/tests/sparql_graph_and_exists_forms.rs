//! SPARQL graph patterns and `EXISTS` as an expression (#737 to #740).
//!
//! Grammar rule 54 allows one `.` after a pattern that is not a triple. A graph the dataset does not
//! name is empty whatever its pattern reads (18.6), so an update over it writes nothing. A sub-SELECT
//! inside `GRAPH ?g` runs once per graph, with `?g` bound. `EXISTS` is an expression (17.4.1.4), so
//! it holds in a `BIND`, a projection, a function and `ORDER BY`. `graph_and_exists_forms.gtest` runs
//! the same table on every binding.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_graph_and_exists_forms::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const DATA: &str = "INSERT DATA { <urn:a> <urn:p> 1 . <urn:b> <urn:q> 2 . GRAPH <urn:g1> { <urn:c> <urn:p> 4 } GRAPH <urn:g2> { <urn:d> <urn:p> 5 . <urn:e> <urn:p> 6 } }";

/// Each form, with `@` for the `.`.
const DOTS: &[(&str, &str)] = &[
    (
        "after_filter",
        "SELECT ?s WHERE { ?s <urn:p> ?o . FILTER(?o = 1) @ }",
    ),
    ("after_a_nested_group", "SELECT * WHERE { { ?s ?q ?y } @ }"),
    (
        "in_optional",
        "SELECT * WHERE { ?a ?b ?c OPTIONAL { { ?s ?q ?y } @ } }",
    ),
    (
        "in_minus",
        "SELECT * WHERE { ?a ?b ?c MINUS { { ?a ?b ?c } @ } }",
    ),
    (
        "in_graph_variable",
        "SELECT * WHERE { GRAPH ?g { { ?s ?q ?y } @ } }",
    ),
    (
        "filter_in_graph",
        "SELECT ?o WHERE { GRAPH <urn:g1> { <urn:c> <urn:p> ?o . FILTER(?o = 4) @ } }",
    ),
    (
        "after_optional",
        "SELECT * WHERE { ?s ?p ?o OPTIONAL { ?s ?q ?y } @ }",
    ),
    (
        "after_minus",
        "SELECT * WHERE { ?s ?p ?o MINUS { ?s <urn:q> ?y } @ }",
    ),
    ("after_graph", "SELECT * WHERE { GRAPH ?g { ?s ?p ?o } @ }"),
    ("after_bind", "SELECT * WHERE { ?s ?p ?o BIND(1 AS ?x) @ }"),
    (
        "after_values",
        "SELECT * WHERE { VALUES ?s { <urn:a> } @ ?s ?p ?o }",
    ),
    (
        "after_union",
        "SELECT * WHERE { { ?s <urn:p> ?o } UNION { ?s <urn:q> ?o } @ }",
    ),
    (
        "then_more_triples",
        "SELECT ?s ?y WHERE { ?s <urn:p> ?o FILTER(?o = 1) @ ?s <urn:p> ?y }",
    ),
];

/// What the grammar does not allow.
const REFUSED: &[&str] = &[
    "SELECT * WHERE { . ?s ?p ?o }",
    "SELECT * WHERE { ?s ?p ?o . . }",
    "SELECT * WHERE { ?s ?p ?o FILTER(?o = 1) . . }",
    "SELECT * WHERE { { ?s ?p ?o } . . }",
    "SELECT * WHERE { . }",
];

/// A query and its rows, each its cells joined.
const CASES: &[(&str, &str, &[&str])] = &[
    (
        "no_graph_bind",
        "SELECT ?x WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        &[],
    ),
    (
        "no_graph_values",
        "SELECT ?x WHERE { GRAPH <urn:nograph> { VALUES ?x { 1 } } }",
        &[],
    ),
    (
        "no_graph_empty_group",
        "SELECT * WHERE { GRAPH <urn:nograph> { } }",
        &[],
    ),
    (
        "no_graph_nested_bind",
        "SELECT ?x WHERE { GRAPH <urn:nograph> { { BIND(1 AS ?x) } } }",
        &[],
    ),
    (
        "no_graph_filter_only",
        "SELECT * WHERE { GRAPH <urn:nograph> { FILTER(true) } }",
        &[],
    ),
    (
        "no_graph_under_optional",
        "SELECT ?s ?x WHERE { ?s <urn:p> ?o OPTIONAL { GRAPH <urn:nograph> { BIND(1 AS ?x) } } }",
        &["urn:a|null"],
    ),
    (
        "a_named_graph_bind",
        "SELECT ?x WHERE { GRAPH <urn:g1> { BIND(1 AS ?x) } }",
        &["1"],
    ),
    (
        "a_named_graph_empty_group",
        "SELECT ?g WHERE { GRAPH ?g { } }",
        &["urn:g1", "urn:g2"],
    ),
    (
        "each_graph_bind",
        "SELECT ?g ?x WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &["urn:g1|1", "urn:g2|1"],
    ),
    (
        "graph_visit_default_graph_is_not_one",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:q> ?o } }",
        &[],
    ),
    (
        "graph_visit_named_ones_only",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s ?p ?o } }",
        &["urn:g1|urn:c", "urn:g2|urn:d", "urn:g2|urn:e"],
    ),
    (
        "graph_visit_from_named",
        "SELECT ?g ?s FROM NAMED <urn:g1> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &["urn:g1|urn:c"],
    ),
    (
        "graph_visit_from_named_bind",
        "SELECT ?g ?x FROM NAMED <urn:g1> WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &["urn:g1|1"],
    ),
    (
        "graph_visit_from_named_empty_group",
        "SELECT ?g FROM NAMED <urn:g2> WHERE { GRAPH ?g { } }",
        &["urn:g2"],
    ),
    (
        "graph_visit_from_named_of_no_graph",
        "SELECT ?g FROM NAMED <urn:nograph> WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &["urn:nograph"],
    ),
    (
        "graph_visit_from_named_a_constant_it_lists",
        "SELECT ?x FROM NAMED <urn:nograph> WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        &["1"],
    ),
    (
        "graph_visit_from_named_a_constant_it_does_not_list",
        "SELECT ?x FROM NAMED <urn:g1> WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        &[],
    ),
    (
        "graph_visit_from_named_of_no_graph_a_triple",
        "SELECT ?g ?s FROM NAMED <urn:nograph> WHERE { GRAPH ?g { ?s ?p ?o } }",
        &[],
    ),
    (
        "sub_select_count_of_a_listed_graph_with_no_triples",
        "SELECT ?g ?c FROM NAMED <urn:nograph> WHERE { GRAPH ?g { { SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } } } }",
        &["urn:nograph|0"],
    ),
    (
        "graph_visit_from_named_a_constant_outside_it",
        "SELECT ?x FROM NAMED <urn:g1> WHERE { GRAPH <urn:g2> { BIND(1 AS ?x) } }",
        &[],
    ),
    (
        "sub_select_from_named",
        "SELECT ?g ?s FROM NAMED <urn:g2> WHERE { GRAPH ?g { { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &["urn:g2|urn:d", "urn:g2|urn:e"],
    ),
    (
        "sub_select_binds_the_graph",
        "SELECT ?g ?s WHERE { GRAPH ?g { { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &["urn:g1|urn:c", "urn:g2|urn:d", "urn:g2|urn:e"],
    ),
    (
        "sub_select_aggregate_per_graph",
        "SELECT ?g ?c WHERE { GRAPH ?g { { SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } } } }",
        &["urn:g1|1", "urn:g2|2"],
    ),
    (
        "sub_select_limit_per_graph",
        "SELECT ?g ?s WHERE { GRAPH ?g { { SELECT ?s WHERE { ?s <urn:p> ?o } ORDER BY ?s LIMIT 1 } } }",
        &["urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    (
        "sub_select_in_a_named_graph",
        "SELECT ?s WHERE { GRAPH <urn:g1> { { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &["urn:c"],
    ),
    (
        "sub_select_in_no_graph",
        "SELECT ?s WHERE { GRAPH <urn:nograph> { { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &[],
    ),
    (
        "sub_select_with_the_graph_fixed",
        "SELECT ?g ?s WHERE { VALUES ?g { <urn:g1> } GRAPH ?g { { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &["urn:g1|urn:c"],
    ),
    (
        "sub_select_beside_a_triple",
        "SELECT ?g ?s ?v WHERE { GRAPH ?g { ?s <urn:p> ?v { SELECT ?s WHERE { ?s <urn:p> ?o } } } }",
        &["urn:g1|urn:c|4", "urn:g2|urn:d|5", "urn:g2|urn:e|6"],
    ),
    (
        "bind_exists_true",
        "SELECT ?w WHERE { BIND(EXISTS { <urn:a> <urn:p> ?o } AS ?w) }",
        &["true"],
    ),
    (
        "bind_exists_false",
        "SELECT ?w WHERE { BIND(EXISTS { <urn:a> <urn:q> ?o } AS ?w) }",
        &["false"],
    ),
    (
        "bind_not_exists",
        "SELECT ?w WHERE { BIND(NOT EXISTS { <urn:a> <urn:q> ?o } AS ?w) }",
        &["true"],
    ),
    (
        "bind_exists_per_row",
        "SELECT ?s ?w WHERE { ?s ?p ?v BIND(EXISTS { ?s <urn:p> ?o } AS ?w) }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "project_exists",
        "SELECT (EXISTS { <urn:a> <urn:p> ?o } AS ?w) WHERE { }",
        &["true"],
    ),
    (
        "project_exists_per_row",
        "SELECT ?s (EXISTS { ?s <urn:p> ?o } AS ?w) WHERE { ?s ?p ?v }",
        &["urn:a|true", "urn:b|false"],
    ),
    (
        "project_not_exists",
        "SELECT ?s (NOT EXISTS { ?s <urn:p> ?o } AS ?w) WHERE { ?s ?p ?v }",
        &["urn:a|false", "urn:b|true"],
    ),
    (
        "exists_inside_a_function",
        "SELECT ?s (IF(EXISTS { ?s <urn:p> ?o }, 1, 0) AS ?w) WHERE { ?s ?p ?v }",
        &["urn:a|1", "urn:b|0"],
    ),
    (
        "bound_exists_in_a_filter",
        "SELECT ?s WHERE { ?s ?p ?v BIND(EXISTS { ?s <urn:p> ?o } AS ?w) FILTER(?w) }",
        &["urn:a"],
    ),
    (
        "exists_in_a_filter_guard",
        "SELECT ?s WHERE { ?s ?p ?v FILTER(EXISTS { ?s <urn:p> ?o }) }",
        &["urn:a"],
    ),
    (
        "exists_inside_each_graph_bind",
        "SELECT ?g ?s ?w WHERE { GRAPH ?g { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:p> 4 } AS ?w) } }",
        &[
            "urn:g1|urn:c|true",
            "urn:g2|urn:d|false",
            "urn:g2|urn:e|false",
        ],
    ),
    (
        "exists_inside_each_graph_filter",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:p> 4 }) } }",
        &["urn:g1|urn:c"],
    ),
    (
        "exists_inside_a_graph",
        "SELECT ?g ?w WHERE { GRAPH ?g { <urn:c> ?p ?v BIND(EXISTS { <urn:c> <urn:p> ?o } AS ?w) } }",
        &["urn:g1|true"],
    ),
];

/// An update, a read and its rows.
const UPDATES: &[(&str, &str, &str, &[&str])] = &[
    (
        "a_listed_graph_keeps_its_row_after_its_last_triple_goes",
        "DELETE DATA { GRAPH <urn:g1> { <urn:c> <urn:p> 4 } }",
        "SELECT ?g FROM NAMED <urn:g1> WHERE { GRAPH ?g { BIND(1 AS ?x) } }",
        &["urn:g1"],
    ),
    (
        "update_with_no_graph_writes_nothing",
        "INSERT { <urn:t> <urn:p> 9 } WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        "SELECT ?v WHERE { <urn:t> <urn:p> ?v }",
        &[],
    ),
    (
        "delete_with_no_graph_removes_nothing",
        "DELETE { <urn:a> <urn:p> 1 } WHERE { GRAPH <urn:nograph> { BIND(1 AS ?x) } }",
        "SELECT ?v WHERE { <urn:a> <urn:p> ?v }",
        &["1"],
    ),
    (
        "update_binds_exists",
        "INSERT { <urn:t> <urn:p> ?w } WHERE { BIND(EXISTS { <urn:a> <urn:p> ?o } AS ?w) }",
        "SELECT ?v WHERE { <urn:t> <urn:p> ?v }",
        &["true"],
    ),
    (
        "update_sub_select_binds_the_graph",
        "INSERT { <urn:t> <urn:from> ?g } WHERE { GRAPH ?g { { SELECT ?s WHERE { ?s <urn:p> 4 } } } }",
        "SELECT ?v WHERE { <urn:t> <urn:from> ?v }",
        &["urn:g1"],
    ),
    (
        "update_filters_on_exists_inside_a_graph",
        "INSERT { <urn:t> <urn:from> ?g } WHERE { GRAPH ?g { ?s <urn:p> ?o FILTER(EXISTS { ?s <urn:p> 4 }) } }",
        "SELECT ?v WHERE { <urn:t> <urn:from> ?v }",
        &["urn:g1"],
    ),
    (
        "update_binds_exists_inside_a_graph",
        "INSERT { ?s <urn:w> ?w } WHERE { GRAPH ?g { ?s <urn:p> ?o BIND(EXISTS { ?s <urn:p> 4 } AS ?w) } }",
        "SELECT ?s ?v WHERE { ?s <urn:w> ?v }",
        &["urn:c|true", "urn:d|false", "urn:e|false"],
    ),
    (
        "update_where_with_a_dot",
        "DELETE { ?s <urn:p> ?o } WHERE { ?s <urn:p> ?o . FILTER(?o = 1) . }",
        "SELECT ?s WHERE { ?s <urn:p> ?o }",
        &[],
    ),
    (
        "an_empty_named_graph_is_a_graph",
        "CREATE GRAPH <urn:empty>",
        "SELECT ?x WHERE { GRAPH <urn:empty> { BIND(1 AS ?x) } }",
        &["1"],
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

fn rows_of(db: &GrafeoDB, query: &str) -> Result<Vec<String>, String> {
    let result = db
        .session()
        .execute_sparql(query)
        .map_err(|e| e.to_string())?;
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    Ok(rows)
}

fn sorted(rows: &[&str]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(ToString::to_string).collect();
    rows.sort();
    rows
}

/// The dot changes nothing: each form answers as it does without it.
#[test]
fn a_dot_after_a_pattern_changes_nothing() {
    let db = data();
    for (name, form) in DOTS {
        let without = rows_of(&db, &form.replace('@', ""))
            .unwrap_or_else(|error| panic!("{name}, no dot: {error}"));
        let with = rows_of(&db, &form.replace('@', "."))
            .unwrap_or_else(|error| panic!("{name}, with its dot: {error}"));
        assert_eq!(with, without, "{name}");
    }
}

/// `SERVICE` is not run, but its `.` parses: the error is that, not a syntax error.
#[test]
fn a_dot_after_service_is_not_a_syntax_error() {
    let db = data();
    let error = rows_of(&db, "SELECT * WHERE { SERVICE <urn:e> { ?s ?p ?o } . }").unwrap_err();
    assert!(error.contains("SERVICE"), "{error}");
}

/// A `.` the grammar does not allow is refused, by the parser.
#[test]
fn a_dot_the_grammar_does_not_allow_is_refused() {
    let db = data();
    for form in REFUSED {
        let error = rows_of(&db, form).expect_err(form);
        assert!(error.contains("unexpected"), "{form}: {error}");
    }
}

fn each_case(prefix: &str) {
    let db = data();
    for (name, query, wanted) in CASES {
        if !name.starts_with(prefix) {
            continue;
        }
        let rows = rows_of(&db, query).unwrap_or_else(|error| panic!("{name}: {query}: {error}"));
        assert_eq!(rows, sorted(wanted), "{name}: {query}");
    }
}

#[test]
fn a_graph_no_dataset_names_is_empty() {
    each_case("no_graph");
}

#[test]
fn a_named_graph_keeps_its_answers() {
    each_case("a_named_graph");
    each_case("each_graph");
}

/// `GRAPH ?g` visits the named graphs, never the default one, and `FROM NAMED` narrows them.
#[test]
fn graph_visits_the_named_graphs_of_the_dataset() {
    each_case("graph_visit");
}

#[test]
fn a_sub_select_inside_graph_runs_per_graph() {
    each_case("sub_select");
}

#[test]
fn exists_in_a_bind() {
    each_case("bind_");
    each_case("bound_");
}

/// #740's error reached a projection too, on 0.6.1.
#[test]
fn exists_in_a_projection() {
    each_case("project_");
}

#[test]
fn exists_inside_a_function_and_a_graph() {
    each_case("exists_inside");
}

/// `EXISTS` in a `FILTER` read as it did before #740.
#[test]
fn exists_in_a_filter_still_reads() {
    each_case("exists_in_a_filter");
}

#[test]
fn exists_in_order_by() {
    let db = data();
    let result = db
        .session()
        .execute_sparql("SELECT ?s WHERE { ?s ?p ?v } ORDER BY (EXISTS { ?s <urn:p> ?o })")
        .unwrap();
    let rows: Vec<String> = result.rows().iter().map(|row| cell(&row[0])).collect();
    assert_eq!(rows, vec!["urn:b", "urn:a"]);
}

fn each_update(chosen: impl Fn(&str) -> bool) {
    for (name, statement, read, wanted) in UPDATES {
        if !chosen(name) {
            continue;
        }
        let db = data();
        rows_of(&db, statement).unwrap_or_else(|error| panic!("{name}: {statement}: {error}"));
        assert_eq!(
            rows_of(&db, read).unwrap_or_else(|error| panic!("{name}: {read}: {error}")),
            sorted(wanted),
            "{name}: {statement}"
        );
    }
}

const UPDATES_OF_739: &[&str] = &["update_sub_select_binds_the_graph"];
const UPDATES_OF_740: &[&str] = &[
    "update_binds_exists",
    "update_filters_on_exists_inside_a_graph",
    "update_binds_exists_inside_a_graph",
];

/// An update's `WHERE` reads as a query's.
#[test]
fn an_update_reads_its_where_as_a_query_does() {
    each_update(|name| !UPDATES_OF_739.contains(&name) && !UPDATES_OF_740.contains(&name));
}

#[test]
fn an_update_reads_a_sub_select_inside_graph() {
    each_update(|name| UPDATES_OF_739.contains(&name));
}

#[test]
fn an_update_reads_exists_in_its_where() {
    each_update(|name| UPDATES_OF_740.contains(&name));
}

/// A reader that began before a graph was made reads the dataset it began with: what `GRAPH` finds
/// and which graphs `GRAPH ?g` visits are those of its view.
#[test]
#[ignore = "waits for #718: the graph list is read at the reader's view"]
fn a_graph_made_after_a_readers_snapshot_is_not_in_its_dataset() {
    let db = data();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let names = "SELECT ?g WHERE { GRAPH ?g { } }";
    let bind = "SELECT ?x WHERE { GRAPH <urn:new> { BIND(1 AS ?x) } }";
    let run = |session: &grafeo_engine::session::Session, query: &str| -> Vec<String> {
        let mut rows: Vec<String> = session
            .execute_sparql(query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| cell(&row[0]))
            .collect();
        rows.sort();
        rows
    };
    assert_eq!(run(&reader, bind), Vec::<String>::new(), "before");
    db.session()
        .execute_sparql("CREATE GRAPH <urn:new>")
        .unwrap();
    assert_eq!(
        run(&reader, bind),
        Vec::<String>::new(),
        "the snapshot, after"
    );
    assert_eq!(
        run(&reader, names),
        vec!["urn:g1", "urn:g2"],
        "the snapshot's graphs"
    );
    reader.rollback().unwrap();
    assert_eq!(run(&reader, bind), vec!["1"], "a new reader");
    assert_eq!(run(&reader, names), vec!["urn:g1", "urn:g2", "urn:new"]);
}
