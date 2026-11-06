//! Each route of `EXISTS` is an optimization of the per-row definition, so on generated data every
//! form answers as it does forced per row, and each form takes the route its design names. Also the
//! forms the routes are most likely to get wrong: a test inside `GRAPH ?g` whose sub-SELECT reads
//! another graph's triple, and `IF(EXISTS ...)` in an `OPTIONAL`'s `FILTER`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_exists_routes::
//! ```

use std::fmt::Write;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
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

fn rows_of(session: &Session, query: &str) -> Result<Vec<String>, String> {
    let result = session
        .execute_sparql(query)
        .map_err(|error| error.to_string())?;
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    Ok(rows)
}

fn per_row(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::ExistsRunOnce, false));
    session
}

// ---------------------------------------------------------------------------
// The route each form takes
// ---------------------------------------------------------------------------

const DATA: &str = "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :d :q :e . GRAPH <urn:g> { :c :r :e } }";

/// How `PROFILE` shows a route.
const SEMI_OR_ANTI: &str = "HashJoin (";
const ROW_CHECK: &str = "[ExistsRowCheck]";
const MARK: &str = "[ExistsMark]";
const PER_ROW: &str = "[ExistsPerRow]";

/// The pattern, and the route the design names for it in a filter of its own.
const ROUTES: &[(&str, &str)] = &[
    ("?s :p ?o", SEMI_OR_ANTI),
    ("FILTER(?s = :a)", ROW_CHECK),
    (":d :q ?o FILTER(?s = :a)", ROW_CHECK),
    (":d :q ?o { FILTER(?s = :a) }", PER_ROW),
    (":d :q ?o OPTIONAL { ?s :p ?y }", PER_ROW),
    (":d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) }", PER_ROW),
    ("?s :t :k MINUS { ?s :t :k }", PER_ROW),
    (":d :q ?o MINUS { :d :q :e }", PER_ROW),
    (":d :q ?o MINUS { :nobody :q ?z }", PER_ROW),
    ("BIND(:a AS ?s)", SEMI_OR_ANTI),
    ("BIND(?s AS ?y) FILTER(?y = :a)", PER_ROW),
    ("SELECT ?s { ?s :p ?o }", SEMI_OR_ANTI),
    ("SELECT ?o { ?s :p ?o }", PER_ROW),
    ("SELECT ?o { ?x :p ?o FILTER(?x = ?s) }", PER_ROW),
    ("SELECT ?s { ?s :t :k } LIMIT 1", PER_ROW),
    ("SELECT (COUNT(*) AS ?n) { ?s :p ?o }", PER_ROW),
    (
        "SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a)",
        PER_ROW,
    ),
    ("SELECT (?s AS ?y) { :d :q ?o }", PER_ROW),
    ("{ ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) }", PER_ROW),
    ("GRAPH <urn:g> { ?s :r* ?y }", SEMI_OR_ANTI),
    (":d :q ?o FILTER EXISTS { ?s :p ?y }", PER_ROW),
    (":d :q ?o FILTER(BOUND(?s))", ROW_CHECK),
];

fn profile_of(session: &Session, query: &str) -> String {
    let result = session
        .execute_sparql(&format!("PROFILE {query}"))
        .unwrap_or_else(|error| panic!("PROFILE {query}: {error}"));
    result
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

const ROUTE_NAMES: [&str; 4] = [SEMI_OR_ANTI, ROW_CHECK, MARK, PER_ROW];

/// Each form takes its route, with `NOT EXISTS` too, and the rule forces every one per row.
#[test]
fn each_form_takes_the_route_it_is_owed() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    let forced = per_row(&db);
    let mut wrong = Vec::new();
    for (pattern, route) in ROUTES {
        for not in ["", "NOT "] {
            let query = format!(
                "PREFIX : <urn:> SELECT ?s WHERE {{ ?s :t :k FILTER {not}EXISTS {{ {pattern} }} }}"
            );
            let profile = profile_of(&session, &query);
            if !profile.contains(route) {
                let shown: Vec<_> = ROUTE_NAMES
                    .iter()
                    .filter(|name| profile.contains(**name))
                    .collect();
                wrong.push(format!("{query}\n  wanted {route}, found {shown:?}"));
            }
            assert_eq!(
                rows_of(&session, &query),
                rows_of(&forced, &query),
                "{query}"
            );
            let profile = profile_of(&forced, &query);
            if !profile.contains(PER_ROW) {
                wrong.push(format!("{query}\n  forced, and not per row\n{profile}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A test that is not the whole predicate, or whose variable the row may leave unbound, is a mark.
#[test]
fn a_test_that_is_not_the_whole_predicate_is_a_mark() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(DATA).unwrap();
    let forced = per_row(&db);
    for form in [
        "SELECT ?s ?u WHERE { ?s :t :k OPTIONAL { ?s :w ?u } FILTER EXISTS { :a :p ?u } }",
        "SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { ?s :p ?o } AS ?e) }",
        "SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { ?s :p ?o } || ?s = :b) }",
    ] {
        let query = format!("PREFIX : <urn:> {form}");
        let profile = profile_of(&session, &query);
        assert!(profile.contains(MARK), "{query}\n{profile}");
        assert_eq!(
            rows_of(&session, &query),
            rows_of(&forced, &query),
            "{query}"
        );
    }
}

// ---------------------------------------------------------------------------
// The oracle: every route answers as per row, on generated data
// ---------------------------------------------------------------------------

/// A small generator, so a failing seed is the same on every machine.
struct Numbers(u64);

impl Numbers {
    fn below(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(limit).unwrap()).unwrap()
    }
}

/// Triples over few terms, so joins meet and graphs share them: terms, literals and a self-loop.
fn generated(seed: u64) -> String {
    let mut numbers = Numbers(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let subjects = [":a", ":b", ":c", ":d", ":k", ":e", "_:n"];
    let predicates = [":t", ":p", ":q", ":r", ":w"];
    let objects = [":a", ":b", ":c", ":d", ":e", ":k", "1", "2", "\"Alix\""];
    let triple = |numbers: &mut Numbers| {
        format!(
            "{} {} {} . ",
            subjects[numbers.below(subjects.len())],
            predicates[numbers.below(predicates.len())],
            objects[numbers.below(objects.len())]
        )
    };
    let mut data = String::from("PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . ");
    for _ in 0..8 + numbers.below(12) {
        data.push_str(&triple(&mut numbers));
    }
    for graph in ["<urn:g>", "<urn:h>"] {
        write!(data, "GRAPH {graph} {{ :a :t :k . ").unwrap();
        for _ in 0..numbers.below(8) {
            data.push_str(&triple(&mut numbers));
        }
        data.push_str("} ");
    }
    data.push('}');
    data
}

/// The patterns of the routes above and more, reading what a row may leave unbound.
const PATTERNS: &[&str] = &[
    "?s :p ?o",
    "?s :p ?u",
    "?u :p ?o",
    "?s :p :c",
    ":d :q ?o",
    "FILTER(?s = :a)",
    "FILTER(BOUND(?u))",
    "FILTER(!BOUND(?u))",
    ":d :q ?o FILTER(?s = :a)",
    ":d :q ?o FILTER(?u = 1)",
    ":d :q ?o { FILTER(?s = :a) }",
    ":d :q ?o OPTIONAL { ?s :p ?y }",
    ":d :q ?o OPTIONAL { ?u :p ?y }",
    ":d :q ?o OPTIONAL { :d :q ?y FILTER(?s = :a) }",
    "?s :t :k OPTIONAL { ?s :w ?y } FILTER(!BOUND(?y))",
    "?s :t :k MINUS { ?s :t :k }",
    "?s :t :k MINUS { ?s :p ?y }",
    ":d :q ?o MINUS { :d :q :e }",
    ":d :q ?o MINUS { :nobody :q ?z }",
    "BIND(:a AS ?s)",
    "BIND(:a AS ?u)",
    "BIND(?s AS ?y) FILTER(?y = :a)",
    "SELECT ?s { ?s :p ?o }",
    "SELECT ?o { ?s :p ?o }",
    "SELECT ?u { ?s :p ?u }",
    "SELECT ?o { ?x :p ?o FILTER(?x = ?s) }",
    "SELECT ?s { ?s :t :k } LIMIT 1",
    "SELECT ?s { ?s :t :k } ORDER BY ?s LIMIT 1 OFFSET 1",
    "SELECT DISTINCT ?s { ?s ?p ?o }",
    "SELECT (COUNT(*) AS ?n) { ?s :p ?o }",
    "SELECT ?s (COUNT(*) AS ?n) { ?s :p ?o } GROUP BY ?s",
    "SELECT ?k { :d :q ?k } GROUP BY ?k HAVING(?s = :a)",
    "SELECT (?s AS ?y) { :d :q ?o }",
    "SELECT (:a AS ?s) { :d :q ?o }",
    "{ ?s :p ?o } UNION { :d :q ?z FILTER(?s = :b) }",
    "{ ?s :p ?o } UNION { ?s :q ?o }",
    "GRAPH <urn:g> { ?s :p* ?y }",
    "GRAPH <urn:g> { ?s :p ?y }",
    "GRAPH ?h { ?s :p ?y }",
    "GRAPH <urn:nograph> { ?s :p ?y }",
    "?s :p? ?y",
    "?y :p+ ?s",
    "?s :p/:q ?y",
    ":d :q ?o FILTER EXISTS { ?s :p ?y }",
    ":d :q ?o FILTER NOT EXISTS { ?s :p ?y }",
    ":d :q ?o FILTER(BOUND(?s))",
    "?s :p ?o FILTER(?o = 1 || ?o = :c)",
    "VALUES ?s { :a :c }",
    "VALUES ?u { 1 UNDEF }",
    "?s :p ?o . ?o :q ?y",
    "{ }",
];

/// Where a test may stand: `@` is its pattern. The row binds `?s`, may leave `?u` unbound and, in a
/// graph, binds `?g`.
const POSITIONS: &[&str] = &[
    "SELECT ?s WHERE { ?s :t :k FILTER EXISTS { @ } }",
    "SELECT ?s WHERE { ?s :t :k FILTER NOT EXISTS { @ } }",
    "SELECT ?s ?u WHERE { ?s :t :k OPTIONAL { ?s :w ?u } FILTER EXISTS { @ } }",
    "SELECT ?s ?u WHERE { ?s :t :k OPTIONAL { ?s :w ?u } FILTER NOT EXISTS { @ } }",
    "SELECT ?s WHERE { ?s :t :k FILTER(EXISTS { @ } || ?s = :b) }",
    "SELECT ?s WHERE { ?s :t :k FILTER(!EXISTS { @ } && ?s = :a) }",
    "SELECT ?s ?e WHERE { ?s :t :k BIND(EXISTS { @ } AS ?e) }",
    "SELECT ?s (EXISTS { @ } AS ?e) WHERE { ?s :t :k }",
    "SELECT ?s ?e WHERE { ?s :t :k BIND(IF(EXISTS { @ }, 1, 2) AS ?e) }",
    "SELECT ?s ?v WHERE { ?s :t :k OPTIONAL { ?s :p ?v FILTER EXISTS { @ } } }",
    "SELECT ?s ?v WHERE { ?s :t :k OPTIONAL { ?s :p ?v FILTER NOT EXISTS { @ } } }",
    "SELECT ?s ?v WHERE { ?s :t :k OPTIONAL { ?s :p ?v FILTER(IF(EXISTS { @ }, true, false)) } }",
    "SELECT ?g ?s WHERE { GRAPH ?g { ?s :t :k FILTER EXISTS { @ } } }",
    "SELECT ?g ?s WHERE { GRAPH ?g { ?s :t :k FILTER NOT EXISTS { @ } } }",
    "SELECT ?s WHERE { GRAPH <urn:g> { ?s :t :k FILTER EXISTS { @ } } }",
    "SELECT ?s WHERE { { SELECT ?s WHERE { ?s :t :k FILTER EXISTS { @ } } } }",
    "SELECT ?s WHERE { ?s :t :k FILTER EXISTS { ?s :t :k FILTER EXISTS { @ } } }",
    "SELECT ?s (COUNT(*) AS ?n) WHERE { ?s :t :k } GROUP BY ?s HAVING(EXISTS { @ })",
    "SELECT ?s WHERE { ?s :t :k } ORDER BY DESC(EXISTS { @ }) ?s",
];

/// Every position of every pattern on each data set, the routes against per row. A form may fail, so
/// long as it fails both ways: the point is the agreement.
#[test]
fn every_route_answers_as_per_row_on_generated_data() {
    let mut wrong = Vec::new();
    let mut cases = 0;
    let mut answered = 0;
    for seed in 1..=8 {
        let data = generated(seed);
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute_sparql(&data).unwrap();
        let forced = per_row(&db);
        for pattern in PATTERNS {
            for position in POSITIONS {
                let query = format!("PREFIX : <urn:> {}", position.replace('@', pattern));
                let found = rows_of(&session, &query);
                let wanted = rows_of(&forced, &query);
                cases += 1;
                answered += usize::from(wanted.is_ok());
                if found != wanted {
                    wrong.push(format!(
                        "seed {seed}: {query}\n  routes  {found:?}\n  per row {wanted:?}"
                    ));
                }
            }
        }
    }
    let shown: Vec<&str> = wrong.iter().take(40).map(String::as_str).collect();
    assert!(
        wrong.is_empty(),
        "{} of {cases} differ, the first of them:\n{}",
        wrong.len(),
        shown.join("\n")
    );
    assert!(
        answered * 10 >= cases * 9,
        "{answered} of {cases} answered: most forms must run for this to say anything"
    );
}

// ---------------------------------------------------------------------------
// A test inside a graph, and a test in an OPTIONAL's FILTER
// ---------------------------------------------------------------------------

/// `b` holds `:q` in its own graph, `a` only in the other, and `c` only in the default graph.
const GRAPH_DATA: &str = "PREFIX : <urn:> INSERT DATA { :c :q 9 . GRAPH <urn:g1> { :a :p 1 . :b :p 2 . :b :q 3 . :c :p 4 } GRAPH <urn:g2> { :a :q 5 . :d :p 6 } }";

/// Name, query, rows. A test inside `GRAPH` reads the graph at hand (SPARQL 1.1 18.6), through a
/// sub-SELECT too: `a`'s `:q` in the other graph does not count. A `GRAPH ?g` inside the test is
/// another matter: the filter runs before the outer `GRAPH ?g` binds ?g, so the inner one reaches
/// every named graph, and `a`'s `:q` in g2 counts (`sparql_graph_variable_scope.rs`).
const IN_A_GRAPH: &[(&str, &str, &[&str])] = &[
    (
        "exists",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { ?s :q ?x }) } }",
        &["urn:g1|urn:b"],
    ),
    (
        "exists_a_sub_select",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s :q ?x } } }) } }",
        &["urn:g1|urn:b"],
    ),
    (
        "not_exists",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(NOT EXISTS { ?s :q ?x }) } }",
        &["urn:g1|urn:a", "urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    (
        "not_exists_a_sub_select",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(NOT EXISTS { { SELECT ?s WHERE { ?s :q ?x } } }) } }",
        &["urn:g1|urn:a", "urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    (
        "a_sub_select_that_projects_another",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { { SELECT ?x WHERE { ?s :q ?x } } }) } }",
        &["urn:g1|urn:b"],
    ),
    (
        "a_sub_select_with_a_count",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { { SELECT (COUNT(*) AS ?n) WHERE { ?s :q ?x } } FILTER(?n > 0) }) } }",
        &["urn:g1|urn:b"],
    ),
    (
        "bound_in_a_bind",
        "SELECT ?g ?s ?e WHERE { GRAPH ?g { ?s :p ?o BIND(EXISTS { { SELECT ?s WHERE { ?s :q ?x } } } AS ?e) } }",
        &[
            "urn:g1|urn:a|false",
            "urn:g1|urn:b|true",
            "urn:g1|urn:c|false",
            "urn:g2|urn:d|false",
        ],
    ),
    (
        "in_a_named_graph",
        "SELECT ?s WHERE { GRAPH <urn:g1> { ?s :p ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s :q ?x } } }) } }",
        &["urn:b"],
    ),
    (
        "in_the_default_graph_it_reads_the_default_graph",
        "SELECT ?s WHERE { ?s :q ?o FILTER(EXISTS { { SELECT ?s WHERE { ?s :q ?x } } }) }",
        &["urn:c"],
    ),
    (
        "a_graph_inside_the_test_reads_that_graph",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { GRAPH <urn:g2> { ?s :q ?x } }) } }",
        &["urn:g1|urn:a"],
    ),
    (
        "a_graph_variable_inside_the_test_is_free",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { GRAPH ?g { ?s :q ?x } }) } }",
        &["urn:g1|urn:a", "urn:g1|urn:b"],
    ),
    (
        "if_in_a_graph",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(IF(EXISTS { { SELECT ?s WHERE { ?s :q ?x } } }, true, false)) } }",
        &["urn:g1|urn:b"],
    ),
];

#[test]
fn a_test_inside_a_graph_reads_that_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(GRAPH_DATA).unwrap();
    let forced = per_row(&db);
    let mut wrong = Vec::new();
    for (name, query, want) in IN_A_GRAPH {
        let query = format!("PREFIX : <urn:> {query}");
        for (route, reader) in [("routes", &session), ("per row", &forced)] {
            let found = rows_of(reader, &query);
            if found
                .as_ref()
                .ok()
                .map(Vec::as_slice)
                .is_none_or(|rows| rows.iter().map(String::as_str).ne(want.iter().copied()))
            {
                wrong.push(format!(
                    "{name}, {route}: {query}\n  found {found:?}\n  want {want:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

const OPTIONAL_DATA: &str = "PREFIX : <urn:> INSERT DATA { :a :t :k . :b :t :k . :a :p :c . :a :p :d . :b :p :c . :c :r 1 . :d :m :a }";

/// Name, query, rows. The test sees the pair of the left and right rows, per pair, and a pair it
/// rejects leaves the left row with the right side unbound.
const IN_AN_OPTIONAL: &[(&str, &str, &[&str])] = &[
    (
        "if_reads_the_right",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { ?o :r ?z }, true, false)) } }",
        &["urn:a|urn:c", "urn:b|urn:c"],
    ),
    (
        "if_keeps_the_left_when_it_rejects_all",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { ?o :nothing ?z }, true, false)) } }",
        &["urn:a|null", "urn:b|null"],
    ),
    (
        "if_the_other_way_round",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { ?o :r ?z }, false, true)) } }",
        &["urn:a|urn:d", "urn:b|null"],
    ),
    (
        "if_reads_both_sides",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { ?o :m ?s }, true, false)) } }",
        &["urn:a|urn:d", "urn:b|null"],
    ),
    (
        "if_reads_the_left_alone",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { :a :p ?o FILTER(IF(EXISTS { ?s :p :d }, true, false)) } }",
        &["urn:a|urn:c", "urn:a|urn:d", "urn:b|null"],
    ),
    (
        "if_with_no_shared_variable",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { :c :r 1 }, true, false)) } }",
        &["urn:a|urn:c", "urn:a|urn:d", "urn:b|urn:c"],
    ),
    (
        "if_a_sub_select",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { { SELECT ?o WHERE { ?o :r ?z } } }, true, false)) } }",
        &["urn:a|urn:c", "urn:b|urn:c"],
    ),
    (
        "if_a_body_a_run_without_the_row_cannot_answer",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { SELECT ?z { ?x :r ?z FILTER(?x = ?o) } }, true, false)) } }",
        &["urn:a|urn:c", "urn:b|urn:c"],
    ),
    (
        "if_in_an_and",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(?s = :a && IF(EXISTS { ?o :r ?z }, true, false)) } }",
        &["urn:a|urn:c", "urn:b|null"],
    ),
    (
        "not_of_if",
        "SELECT ?s ?o WHERE { ?s :t :k OPTIONAL { ?s :p ?o FILTER(!IF(EXISTS { ?o :r ?z }, true, false)) } }",
        &["urn:a|urn:d", "urn:b|null"],
    ),
    (
        "if_inside_a_graph",
        "SELECT ?s ?o WHERE { GRAPH <urn:o> { ?s :t :k OPTIONAL { ?s :p ?o FILTER(IF(EXISTS { ?o :r ?z }, true, false)) } } }",
        &["urn:a|urn:c"],
    ),
];

#[test]
fn if_exists_in_an_optional_reads_each_pair() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute_sparql(OPTIONAL_DATA).unwrap();
    session
        .execute_sparql("PREFIX : <urn:> INSERT DATA { GRAPH <urn:o> { :a :t :k . :a :p :c . :a :p :d . :c :r 1 } }")
        .unwrap();
    let forced = per_row(&db);
    let mut wrong = Vec::new();
    for (name, query, want) in IN_AN_OPTIONAL {
        let query = format!("PREFIX : <urn:> {query}");
        for (route, reader) in [("routes", &session), ("per row", &forced)] {
            let found = rows_of(reader, &query);
            if found
                .as_ref()
                .ok()
                .map(Vec::as_slice)
                .is_none_or(|rows| rows.iter().map(String::as_str).ne(want.iter().copied()))
            {
                wrong.push(format!(
                    "{name}, {route}: {query}\n  found {found:?}\n  want {want:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
