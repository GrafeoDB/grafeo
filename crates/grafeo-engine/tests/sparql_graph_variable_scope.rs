//! `GRAPH ?g { P }` evaluates `P` in each named graph, then joins `?g` (SPARQL 1.1 18.6, and the 1.2
//! draft's "Evaluation of Graph"): inside `P`, `?g` is free. An `EXISTS`, a `NOT EXISTS`, an `OPTIONAL`'s
//! right side and a `MINUS` in `P` see it unbound, so a nested `GRAPH ?g` there ranges over every named
//! graph; a nested `GRAPH ?g` on a join or union path of `P` binds it and the join keeps the rows of the
//! graph at hand. A `FILTER` on `?g` inside `P` sees it unbound, and a triple of `P` that names `?g` binds
//! it before the join. `FROM NAMED` limits the graphs both levels range over. rdflib gives the same
//! answers (the oracle) but for the `MINUS` pair inside `EXISTS`, where it follows SPARQL 1.1
//! and grafeo the 1.2 draft by decision. Also #728's two SPARQL threads: a graph a transaction empties is
//! listed alike before and after its commit, and a sub-SELECT alone in a group takes no `.` after it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_graph_variable_scope::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const DATA: &str = "PREFIX : <urn:> INSERT DATA { :c :q 9 . \
     GRAPH <urn:g1> { :a :p 1 . :b :p 2 . :b :q 3 . :c :p 4 . :x :in <urn:g1> } \
     GRAPH <urn:g2> { :a :q 5 . :d :p 6 . :y :in <urn:g1> . :z :in <urn:g2> } }";

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Null => "null".to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
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

fn rows_of(session: &Session, query: &str) -> Vec<String> {
    let result = session
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    rows
}

fn with_graphs() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute_sparql(DATA).unwrap();
    db
}

/// The query of each form and its rows, sorted, as `g|s` or `g|s|x`.
const FORMS: &[(&str, &str, &[&str])] = &[
    // a: `:a :q 5` is in g2, which the free ?g of the EXISTS reaches.
    (
        "exists",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { GRAPH ?g { ?s :q ?x } }) } }",
        &["urn:g1|urn:a", "urn:g1|urn:b"],
    ),
    (
        "not_exists",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(NOT EXISTS { GRAPH ?g { ?s :q ?x } }) } }",
        &["urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    // a's optional match binds ?g to g2, which the join with g1 then drops: a has no row at all.
    (
        "optional",
        "SELECT ?g ?s ?x WHERE { GRAPH ?g { ?s :p ?o OPTIONAL { GRAPH ?g { ?s :q ?x } } } }",
        &["urn:g1|urn:b|3", "urn:g1|urn:c|null", "urn:g2|urn:d|null"],
    ),
    (
        "minus",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o MINUS { GRAPH ?g { ?s :q ?x } } } }",
        &["urn:g1|urn:c", "urn:g2|urn:d"],
    ),
    // On a join path the nested GRAPH ?g keeps to the graph at hand: as before.
    (
        "join",
        "SELECT ?g ?s ?x WHERE { GRAPH ?g { ?s :p ?o . GRAPH ?g { ?s :q ?x } } }",
        &["urn:g1|urn:b|3"],
    ),
    (
        "union",
        "SELECT ?g ?s WHERE { GRAPH ?g { { ?s :p 1 } UNION { GRAPH ?g { ?s :q ?x } } } }",
        &["urn:g1|urn:a", "urn:g1|urn:b", "urn:g2|urn:a"],
    ),
    // ?g is unbound in the FILTER: the comparison is an error, so no row passes.
    (
        "filter_on_g",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(?g = <urn:g1>) } }",
        &[],
    ),
    // The triple binds ?g; only a row whose object is its own graph joins with it.
    (
        "g_as_a_term",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g } }",
        &["urn:g1|urn:x", "urn:g2|urn:z"],
    ),
    // ?g is unbound inside the group: BOUND is false there.
    (
        "bound_g",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(BOUND(?g)) } }",
        &[],
    ),
    (
        "not_bound_g",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :p ?o FILTER(!BOUND(?g)) } }",
        &[
            "urn:g1|urn:a",
            "urn:g1|urn:b",
            "urn:g1|urn:c",
            "urn:g2|urn:d",
        ],
    ),
    // A MINUS that shares no variable removes nothing, a group that names ?g as a term too.
    (
        "minus_shares_nothing",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g MINUS { ?x :q ?y } } }",
        &["urn:g1|urn:x", "urn:g2|urn:z"],
    ),
    (
        "sub_select_in_a_term_pattern",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g { SELECT ?s WHERE { ?s :in ?h } } } }",
        &["urn:g1|urn:x", "urn:g2|urn:z"],
    ),
    // `:y :in <urn:g1>` is in g2: as ?h it is g2, as the triple's object g1, so it joins nothing.
    (
        "nested_graph_h",
        "SELECT ?g ?h ?s WHERE { GRAPH ?g { ?s :in ?g . GRAPH ?h { ?s :in ?h } } }",
        &["urn:g1|urn:g1|urn:x", "urn:g2|urn:g2|urn:z"],
    ),
    (
        "from_named_term_pattern",
        "SELECT ?g ?s FROM NAMED <urn:g2> WHERE { GRAPH ?g { ?s :in ?g } }",
        &["urn:g2|urn:z"],
    ),
    (
        "from_named",
        "SELECT ?g ?s FROM NAMED <urn:g1> WHERE { GRAPH ?g { ?s :p ?o FILTER(EXISTS { GRAPH ?g { ?s :q ?x } }) } }",
        &["urn:g1|urn:b"],
    ),
    // A MINUS inside an EXISTS whose right side names no variable of the row. By the SPARQL 1.2
    // draft's EXISTS (17.4.1.4, chosen by the user on 2026-11-05) every group of the test starts from
    // the row, the MINUS's right side too, so it shares the row's ?s and removes it: EXISTS keeps no
    // row, NOT EXISTS both. The graph's hidden column plays no part (`sparql_exists_per_row::case_8`
    // without a graph). rdflib, a SPARQL 1.1 oracle, gives the reverse (18.5: a disjoint domain keeps
    // the row); the two differ here by decision.
    (
        "exists_minus_shares_nothing",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g FILTER(EXISTS { MINUS { ?x :q ?y } }) } }",
        &[],
    ),
    (
        "not_exists_minus_shares_nothing",
        "SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g FILTER(NOT EXISTS { MINUS { ?x :q ?y } }) } }",
        &["urn:g1|urn:x", "urn:g2|urn:z"],
    ),
];

/// Where `?g` is a term of the group's one pattern, the scan itself enforces it (0.6.2): one scan
/// with `?g` in the object position and no hidden graph column. Where a side may leave the column unbound (an `OPTIONAL` group), the hidden graph's
/// check stays above the join and EXPLAIN names it under its hidden name: pinned, so a rename
/// shows here.
#[test]
fn explain_shows_the_hidden_graph_only_where_its_check_stays() {
    let db = with_graphs();
    let session = db.session();
    let plan = |query: &str| {
        let result = session
            .execute_sparql(&format!("EXPLAIN PREFIX : <urn:> {query}"))
            .unwrap();
        format!("{:?}", result.rows())
    };
    let object = plan("SELECT ?g ?s WHERE { GRAPH ?g { ?s :in ?g } }");
    assert!(!object.contains("graph of GRAPH ?g"), "{object}");
    assert!(object.contains("TripleScan"), "{object}");
    let optional =
        plan("SELECT ?g ?s ?x WHERE { GRAPH ?g { ?s :p ?o OPTIONAL { GRAPH ?g { ?s :q ?x } } } }");
    assert!(optional.contains("graph of GRAPH ?g at 0"), "{optional}");
}

#[test]
fn graph_binds_its_variable_after_the_pattern() {
    let db = with_graphs();
    let session = db.session();
    let mut wrong = Vec::new();
    for (name, query, want) in FORMS {
        let found = rows_of(&session, &format!("PREFIX : <urn:> {query}"));
        if found != *want {
            wrong.push(format!("{name}: {found:?}, want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `SELECT *` over a group that names its graph's variable as a term: the result has the variables
/// the query names and no other column (the graph's own column is hidden, EG11 option A).
#[test]
fn select_star_shows_no_hidden_column() {
    let db = with_graphs();
    let session = db.session();
    let result = session
        .execute_sparql("PREFIX : <urn:> SELECT * WHERE { GRAPH ?g { ?s :in ?g } }")
        .unwrap();
    let columns: std::collections::BTreeSet<&str> =
        result.columns.iter().map(String::as_str).collect();
    assert_eq!(columns, std::collections::BTreeSet::from(["g", "s"]));
    assert_eq!(result.rows().len(), 2);
}

/// #728 (`rdf/named_graphs.rs:47`, b): a graph a transaction gave its only triple and then took it from
/// again is listed alike before the commit and after it (ruled: listed).
#[test]
fn a_graph_a_transaction_empties_answers_alike_before_and_after_the_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("INSERT DATA { GRAPH <urn:g> { <urn:s> <urn:p> 1 } }")
        .unwrap();
    session
        .execute_sparql("DELETE DATA { GRAPH <urn:g> { <urn:s> <urn:p> 1 } }")
        .unwrap();
    let bind = "SELECT ?x WHERE { GRAPH <urn:g> { BIND(1 AS ?x) } }";
    let names = "SELECT ?g WHERE { GRAPH ?g { } }";
    let before = (rows_of(&session, bind), rows_of(&session, names));
    session.commit().unwrap();
    let after = (rows_of(&session, bind), rows_of(&session, names));
    assert_eq!(before, after, "the same answers across the commit");
    assert_eq!(after.0, ["1"], "the graph is listed");
    assert_eq!(after.1, ["urn:g"]);
    let fresh = db.session();
    assert_eq!(
        (rows_of(&fresh, bind), rows_of(&fresh, names)),
        after,
        "a later session"
    );
}

/// #728 (`sparql/parser.rs:798`): a group that is a sub-SELECT holds only it (SPARQL 1.1 grammar,
/// `GroupGraphPattern ::= '{' ( SubSelect | GroupGraphPatternSub ) '}'`), so a `.` after it is an
/// error; in a group of its own inside the group it may take one.
#[test]
fn a_sub_select_alone_in_a_group_takes_no_dot() {
    let db = with_graphs();
    let session = db.session();
    let alone =
        "PREFIX : <urn:> SELECT ?s WHERE { SELECT ?s WHERE { GRAPH <urn:g1> { ?s :p ?o } } . }";
    assert!(session.execute_sparql(alone).is_err(), "{alone}");
    for query in [
        "PREFIX : <urn:> SELECT ?s WHERE { { SELECT ?s WHERE { GRAPH <urn:g1> { ?s :p ?o } } } . }",
        "PREFIX : <urn:> SELECT ?s WHERE { SELECT ?s WHERE { GRAPH <urn:g1> { ?s :p ?o } } }",
    ] {
        assert_eq!(
            rows_of(&session, query),
            ["urn:a", "urn:b", "urn:c"],
            "{query}"
        );
    }
}
