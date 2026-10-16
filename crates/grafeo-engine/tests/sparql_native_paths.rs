//! Caller-level acceptance for unbounded SPARQL property paths.
//!
//! Exact row multisets catch a depth cap, missing zero-hop terms, duplicate
//! closure pairs and lost route multiplicity in an ordinary fixed sequence.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;

use grafeo_common::types::Value;
use grafeo_common::utils::error::ErrorCode;
use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::database::QueryResult;
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

// The shared path fixture's EDGES table, represented as RDF triples. The
// duplicate OTHER edge deliberately becomes one RDF triple, unlike LPG edges.
const EDGES: &[(&str, &str, &str)] = &[
    ("s", "a", "REL"),
    ("s", "b", "REL"),
    ("a", "d", "REL"),
    ("b", "d", "REL"),
    ("d", "e", "REL"),
    ("e", "f", "REL"),
    ("f", "e", "REL"),
    ("b", "g", "REL"),
    ("g", "g", "REL"),
    ("a", "h", "REL"),
    ("h", "d", "REL"),
    ("s", "x", "OTHER"),
    ("s", "x", "OTHER"),
    ("u", "v", "REL"),
];

// Hand-checked positive-length REL reachability: 7 + 4 + 4 + 2 + 2 + 2 +
// 1 + 3 + 1 = 26 pairs. This table does not run a traversal as its oracle.
const REL_REACHABLE: &[(&str, &[&str])] = &[
    ("s", &["a", "b", "d", "e", "f", "g", "h"]),
    ("a", &["d", "e", "f", "h"]),
    ("b", &["d", "e", "f", "g"]),
    ("d", &["e", "f"]),
    ("e", &["e", "f"]),
    ("f", &["e", "f"]),
    ("g", &["g"]),
    ("h", &["d", "e", "f"]),
    ("u", &["v"]),
];

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
}

fn iri(local: &str) -> String {
    format!("http://example.org/{local}")
}

fn insert_data(db: &GrafeoDB, triples: &str) {
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {triples} }}"))
        .expect("insert RDF path fixture");
}

/// A directed 51-edge chain, using one predicate or alternating predicates.
fn chain_db(predicates: &[&str]) -> GrafeoDB {
    let db = rdf_db();
    let mut triples = String::new();
    for index in 0..51 {
        writeln!(
            triples,
            "e:n{index} e:{} e:n{} .",
            predicates[index % predicates.len()],
            index + 1
        )
        .unwrap();
    }
    insert_data(&db, &triples);
    db
}

fn adversarial_db() -> GrafeoDB {
    let db = rdf_db();
    let mut triples = String::new();
    for (from, to, predicate) in EDGES {
        writeln!(triples, "e:{from} e:{predicate} e:{to} .").unwrap();
    }
    insert_data(&db, &triples);
    db
}

fn endpoints(locals: &[&str]) -> Vec<Vec<String>> {
    locals.iter().map(|local| vec![iri(local)]).collect()
}

fn chain_endpoints(indices: std::ops::RangeInclusive<usize>) -> Vec<Vec<String>> {
    indices
        .map(|index| vec![iri(&format!("n{index}"))])
        .collect()
}

fn assert_rows(db: &GrafeoDB, query: &str, count: usize, expected: Vec<Vec<String>>) {
    let result = db
        .execute_sparql(&format!("{PREFIX} {query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_result_rows(&result, query, count, expected);
}

fn assert_session_rows(session: &Session, query: &str, count: usize, expected: Vec<Vec<String>>) {
    let result = session
        .execute_sparql(&format!("{PREFIX} {query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_result_rows(&result, query, count, expected);
}

fn assert_result_rows(
    result: &QueryResult,
    query: &str,
    count: usize,
    mut expected: Vec<Vec<String>>,
) {
    assert_eq!(expected.len(), count, "independent expected row count");
    assert_eq!(result.row_count(), count, "{query}");
    let mut actual: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(value) => value.to_string(),
                    other => panic!("{query}: expected a string RDF value, got {other:?}"),
                })
                .collect()
        })
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected, "exact row multiset for {query}");
}

#[test]
fn chain_plus_has_no_depth_cap() {
    let db = chain_db(&["p"]);
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n0 e:p+ ?o }",
        51,
        chain_endpoints(1..=51),
    );
}

#[test]
fn chain_star_has_no_depth_cap() {
    let db = chain_db(&["p"]);
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n0 e:p* ?o }",
        52,
        chain_endpoints(0..=51),
    );
}

#[test]
fn reverse_bound_chain_plus_has_no_depth_cap() {
    let db = chain_db(&["p"]);
    assert_rows(
        &db,
        "SELECT ?s WHERE { ?s e:p+ e:n51 }",
        51,
        chain_endpoints(0..=50),
    );
}

#[test]
fn reverse_bound_chain_star_has_no_depth_cap() {
    let db = chain_db(&["p"]);
    assert_rows(
        &db,
        "SELECT ?s WHERE { ?s e:p* e:n51 }",
        52,
        chain_endpoints(0..=51),
    );
}

#[test]
fn free_chain_plus_has_exact_pairs() {
    let db = chain_db(&["p"]);
    // A directed chain reaches exactly the pairs with start < end:
    // 52 * 51 / 2 = 1326, including n0 -> n51.
    let expected = (0..=51)
        .flat_map(|start| {
            (start + 1..=51)
                .map(move |end| vec![iri(&format!("n{start}")), iri(&format!("n{end}"))])
        })
        .collect();
    assert_rows(&db, "SELECT ?s ?o WHERE { ?s e:p+ ?o }", 1326, expected);
}

#[test]
fn free_chain_star_has_exact_pairs() {
    let db = chain_db(&["p"]);
    // Zero-hop adds each of the 52 chain terms exactly once: 1326 + 52.
    let expected = (0..=51)
        .flat_map(|start| {
            (start..=51).map(move |end| vec![iri(&format!("n{start}")), iri(&format!("n{end}"))])
        })
        .collect();
    assert_rows(&db, "SELECT ?s ?o WHERE { ?s e:p* ?o }", 1378, expected);
}

#[test]
fn sequence_star_has_no_depth_cap() {
    let db = rdf_db();
    let mut triples = String::new();
    for index in 0..51 {
        let start = 2 * index;
        writeln!(triples, "e:n{start} e:p e:n{} .", start + 1).unwrap();
        writeln!(triples, "e:n{} e:q e:n{} .", start + 1, start + 2).unwrap();
    }
    insert_data(&db, &triples);
    let expected = (0..=51)
        .map(|index| vec![iri(&format!("n{}", 2 * index))])
        .collect();
    assert_rows(&db, "SELECT ?o WHERE { e:n0 (e:p/e:q)* ?o }", 52, expected);
}

#[test]
fn alternative_paths_have_no_depth_cap() {
    let db = chain_db(&["p", "q"]);
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n0 (e:p|e:q)+ ?o }",
        51,
        chain_endpoints(1..=51),
    );
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n0 (e:p|e:q)* ?o }",
        52,
        chain_endpoints(0..=51),
    );
}

#[test]
fn inverse_paths_have_no_depth_cap() {
    let db = chain_db(&["p"]);
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n51 (^e:p)+ ?o }",
        51,
        chain_endpoints(0..=50),
    );
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:n51 (^e:p)* ?o }",
        52,
        chain_endpoints(0..=51),
    );
}

#[test]
fn adversarial_paths_return_exact_endpoints() {
    let db = adversarial_db();
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:REL+ ?o }",
        7,
        endpoints(&["a", "b", "d", "e", "f", "g", "h"]),
    );
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:REL* ?o }",
        8,
        endpoints(&["s", "a", "b", "d", "e", "f", "g", "h"]),
    );
}

#[test]
fn adversarial_paths_return_exact_free_pairs() {
    let db = adversarial_db();
    let positive: Vec<Vec<String>> = REL_REACHABLE
        .iter()
        .flat_map(|(from, tos)| tos.iter().map(move |to| vec![iri(from), iri(to)]))
        .collect();
    assert_rows(
        &db,
        "SELECT ?s ?o WHERE { ?s e:REL+ ?o }",
        26,
        positive.clone(),
    );
    // e, f and g already reach themselves by a positive-length cycle.
    // x occurs only in OTHER triples, but its zero-length REL path still exists.
    let mut reflexive = positive;
    reflexive.extend(
        ["s", "a", "b", "d", "h", "u", "v", "x"]
            .into_iter()
            .map(|node| vec![iri(node), iri(node)]),
    );
    assert_rows(&db, "SELECT ?s ?o WHERE { ?s e:REL* ?o }", 34, reflexive);
}

#[test]
fn repeated_endpoint_variable_keeps_only_reflexive_pairs() {
    let db = adversarial_db();
    assert_rows(
        &db,
        "SELECT ?x WHERE { ?x e:REL+ ?x }",
        3,
        endpoints(&["e", "f", "g"]),
    );
    assert_rows(
        &db,
        "SELECT ?x WHERE { ?x e:REL* ?x }",
        11,
        endpoints(&["a", "b", "d", "e", "f", "g", "h", "s", "u", "v", "x"]),
    );
}

#[test]
fn fixed_sequence_preserves_route_multiplicity() {
    let db = adversarial_db();
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:REL/e:REL ?o }",
        4,
        endpoints(&["d", "d", "g", "h"]),
    );
}

#[test]
fn duplicate_rdf_triples_do_not_multiply_path_results() {
    let db = adversarial_db();
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:OTHER ?o }",
        1,
        endpoints(&["x"]),
    );
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:OTHER+ ?o }",
        1,
        endpoints(&["x"]),
    );
}

fn pairs(locals: &[(&str, &str)]) -> Vec<Vec<String>> {
    locals.iter().map(|(a, b)| vec![iri(a), iri(b)]).collect()
}

fn dataset_db() -> GrafeoDB {
    let db = rdf_db();
    insert_data(
        &db,
        "e:s e:p e:default .
         GRAPH e:g1 { e:s e:p e:a . e:a e:p e:b . }
         GRAPH e:g2 { e:s e:p e:a . e:b e:p e:c . }
         GRAPH e:g3 { e:s e:p e:excluded . }",
    );
    db
}

#[test]
fn paths_respect_default_graph_named_graphs_and_dataset_union() {
    let db = dataset_db();
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:s e:p+ ?o }",
        1,
        endpoints(&["default"]),
    );
    assert_rows(
        &db,
        "SELECT ?o WHERE { GRAPH e:g1 { e:s e:p+ ?o } }",
        2,
        endpoints(&["a", "b"]),
    );
    // The b -> c step is in another graph; the duplicate s -> a is one triple
    // in the merged default graph, and listing g1 twice adds no multiplicity.
    assert_rows(
        &db,
        "SELECT ?o FROM e:g1 FROM e:g2 FROM e:g1 WHERE { e:s e:p+ ?o }",
        3,
        endpoints(&["a", "b", "c"]),
    );
    assert_rows(
        &db,
        "SELECT ?g ?o WHERE { GRAPH ?g { e:s e:p+ ?o } }",
        4,
        pairs(&[("g1", "a"), ("g1", "b"), ("g2", "a"), ("g3", "excluded")]),
    );
    assert_rows(
        &db,
        "SELECT ?g ?o FROM NAMED e:g1 FROM NAMED e:g2 WHERE { GRAPH ?g { e:s e:p+ ?o } }",
        3,
        pairs(&[("g1", "a"), ("g1", "b"), ("g2", "a")]),
    );
    assert_rows(
        &db,
        "SELECT ?o FROM NAMED e:g1 WHERE { GRAPH e:g3 { e:s e:p+ ?o } }",
        0,
        vec![],
    );
    assert_rows(
        &db,
        "SELECT ?o FROM NAMED e:g1 WHERE { e:s e:p+ ?o }",
        0,
        vec![],
    );
}

#[test]
fn repeated_graph_and_endpoint_variable_is_one_exact_binding() {
    let db = rdf_db();
    insert_data(
        &db,
        "GRAPH e:g1 { e:g1 e:p e:a . e:a e:p e:b . e:unrelated e:p e:a . }
         GRAPH e:g2 { e:g2 e:p e:a . e:a e:p e:c . }",
    );
    assert_rows(
        &db,
        "SELECT ?g ?o WHERE { GRAPH ?g { ?g e:p+ ?o } }",
        4,
        pairs(&[("g1", "a"), ("g1", "b"), ("g2", "a"), ("g2", "c")]),
    );
    // The object shares the graph variable too, rather than creating another
    // output column or accepting an endpoint from the wrong graph.
    insert_data(
        &db,
        "GRAPH e:g1 { e:s e:p e:g1 . } GRAPH e:g2 { e:s e:p e:g2 . }",
    );
    assert_rows(
        &db,
        "SELECT ?g WHERE { GRAPH ?g { e:s e:p+ ?g } }",
        2,
        endpoints(&["g1", "g2"]),
    );
}

#[test]
fn paths_apply_pending_operations_in_order_only_in_the_writing_session() {
    let db = rdf_db();
    insert_data(
        &db,
        "e:s e:p e:a . e:a e:p e:b . GRAPH e:g { e:s e:p e:a . e:a e:p e:b . }",
    );
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    for update in [
        "DELETE DATA { e:a e:p e:b . e:s e:p e:a . GRAPH e:g { e:a e:p e:b . } }",
        "INSERT DATA { e:a e:p e:b . e:s e:p e:a . e:a e:p e:c . e:a e:p e:d .
                       GRAPH e:g { e:a e:p e:c . } }",
        "DELETE DATA { e:a e:p e:c . }",
    ] {
        writer
            .execute_sparql(&format!("{PREFIX} {update}"))
            .unwrap();
    }
    let default = "SELECT ?o WHERE { e:s e:p+ ?o }";
    let named = "SELECT ?o WHERE { GRAPH e:g { e:s e:p+ ?o } }";
    assert_session_rows(&writer, default, 3, endpoints(&["a", "b", "d"]));
    assert_session_rows(&writer, named, 2, endpoints(&["a", "c"]));
    assert_session_rows(&reader, default, 2, endpoints(&["a", "b"]));
    assert_session_rows(&reader, named, 2, endpoints(&["a", "b"]));
    writer.rollback().unwrap();
    assert_session_rows(&writer, default, 2, endpoints(&["a", "b"]));
    assert_session_rows(&writer, named, 2, endpoints(&["a", "b"]));
}

fn insert_terms(db: &GrafeoDB, terms: &[(Term, Term)]) {
    for (subject, object) in terms {
        assert!(db.rdf_store().insert(Triple::new(
            subject.clone(),
            Term::iri(iri("p")),
            object.clone(),
        )));
    }
}

#[test]
fn paths_keep_iri_literal_and_blank_literal_spellings_distinct() {
    let db = rdf_db();
    insert_terms(
        &db,
        &[
            (Term::iri(iri("root")), Term::iri(iri("value"))),
            (Term::iri(iri("root")), Term::literal(iri("value"))),
            (Term::iri(iri("root")), Term::blank("same")),
            (Term::iri(iri("root")), Term::literal("_:same")),
            (Term::iri(iri("value")), Term::iri(iri("iri_end"))),
            (Term::blank("same"), Term::iri(iri("blank_end"))),
            (
                Term::iri(iri("literal_parent")),
                Term::literal(iri("value")),
            ),
        ],
    );
    // Distinct RDF terms can have the same public lexical value. Keep both
    // rows rather than treating those strings as the traversal's identity.
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:root e:p+ ?o }",
        6,
        vec![
            vec![iri("value")],
            vec![iri("value")],
            vec!["_:same".into()],
            vec!["_:same".into()],
            vec![iri("iri_end")],
            vec![iri("blank_end")],
        ],
    );
    assert_rows(
        &db,
        "SELECT ?s WHERE { ?s e:p+ e:value }",
        1,
        endpoints(&["root"]),
    );
    assert_rows(
        &db,
        "SELECT ?s WHERE { ?s e:p+ \"http://example.org/value\" }",
        2,
        endpoints(&["root", "literal_parent"]),
    );
}

#[test]
fn paths_keep_datatypes_before_lexical_projection() {
    let db = rdf_db();
    // Public store insertion retains arbitrary datatype IRIs. The existing
    // SPARQL literal translator does not retain all custom datatype syntax.
    insert_terms(
        &db,
        &[
            (
                Term::iri(iri("root")),
                Term::typed_literal("same", iri("type1")),
            ),
            (
                Term::iri(iri("root")),
                Term::typed_literal("same", iri("type2")),
            ),
        ],
    );
    let same = || vec![vec!["same".into()]];
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:root e:p+ ?o }",
        2,
        vec![vec!["same".into()], vec!["same".into()]],
    );
    for datatype in ["type1", "type2"] {
        assert_rows(
            &db,
            &format!("SELECT ?o WHERE {{ e:root e:p+ ?o FILTER(DATATYPE(?o) = e:{datatype}) }}"),
            1,
            same(),
        );
    }
}

#[test]
fn paths_keep_languages_before_lexical_projection() {
    let db = rdf_db();
    insert_data(&db, "e:root e:p \"same\"@en . e:root e:p \"same\"@fr .");
    assert_rows(
        &db,
        "SELECT ?o WHERE { e:root e:p+ ?o }",
        2,
        vec![vec!["same".into()], vec!["same".into()]],
    );
    for language in ["en", "fr"] {
        assert_rows(
            &db,
            &format!("SELECT ?o WHERE {{ e:root e:p+ ?o FILTER(LANG(?o) = \"{language}\") }}"),
            1,
            vec![vec!["same".into()]],
        );
    }
}

#[test]
fn optional_path_join_keeps_unmatched_input() {
    let db = rdf_db();
    insert_data(
        &db,
        "e:s e:seed e:yes . e:u e:seed e:yes . e:s e:p e:a . e:a e:p e:b .",
    );
    let query = "SELECT ?s ?o WHERE { ?s e:seed e:yes OPTIONAL { ?s e:p+ ?o } }";
    let result = db.execute_sparql(&format!("{PREFIX} {query}")).unwrap();
    assert_eq!(result.row_count(), 3);
    let mut actual: Vec<Vec<Option<String>>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(value) => Some(value.to_string()),
                    Value::Null => None,
                    other => panic!("{query}: unexpected value {other:?}"),
                })
                .collect()
        })
        .collect();
    let mut expected = vec![
        vec![Some(iri("s")), Some(iri("a"))],
        vec![Some(iri("s")), Some(iri("b"))],
        vec![Some(iri("u")), None],
    ];
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}

#[test]
fn explain_accepts_native_path_without_returning_query_rows() {
    let db = chain_db(&["p"]);
    let result = db
        .execute_sparql(&format!(
            "EXPLAIN {PREFIX} SELECT ?o WHERE {{ e:n0 e:p+ ?o }}"
        ))
        .unwrap();
    assert_eq!(result.columns, ["plan"]);
    assert_eq!(result.row_count(), 1);
    assert!(matches!(&result.rows()[0][0], Value::String(plan) if !plan.is_empty()));
}

#[cfg(feature = "lpg")]
#[test]
fn processor_profile_reports_native_path_cardinality() {
    use grafeo_engine::query::{QueryLanguage, QueryProcessor};
    use std::sync::Arc;

    let db = chain_db(&["p"]);
    // Session::execute_sparql does not dispatch PROFILE. This is the existing
    // public RDF processor route that plans and executes profiling wrappers.
    let processor = QueryProcessor::with_rdf(db.store(), Arc::clone(db.rdf_store()));
    let result = processor
        .process(
            &format!("EXPLAIN ANALYZE {PREFIX} SELECT ?o WHERE {{ e:n0 e:p+ ?o }}"),
            QueryLanguage::Sparql,
            None,
        )
        .unwrap();
    assert_eq!(result.columns, ["profile"]);
    assert_eq!(result.row_count(), 1);
    let Value::String(profile) = &result.rows()[0][0] else {
        panic!("expected profile text");
    };
    let path_lines: Vec<_> = profile
        .lines()
        .filter(|line| line.trim_start().starts_with("RdfPropertyPath "))
        .collect();
    assert_eq!(path_lines.len(), 1, "{profile}");
    assert_eq!(
        path_lines[0]
            .split("rows=")
            .nth(1)
            .and_then(|tail| tail.split_whitespace().next()),
        Some("51"),
        "the native path operator's own output cardinality: {profile}",
    );
}

#[test]
fn unsupported_inner_repetition_returns_semantic_error() {
    let db = rdf_db();
    insert_data(&db, "e:s e:p e:a .");
    for path in ["(e:p+)*", "(!e:p)+"] {
        let error = db
            .execute_sparql(&format!("{PREFIX} SELECT ?o WHERE {{ e:s {path} ?o }}"))
            .expect_err("unsupported path must not return a partial query result");
        assert_eq!(
            error.error_code(),
            ErrorCode::QuerySemantic,
            "{path}: {error}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn expired_deadline_returns_timeout_instead_of_partial_path_rows() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_query_timeout(std::time::Duration::ZERO),
    )
    .unwrap();
    insert_terms(&db, &[(Term::iri(iri("s")), Term::iri(iri("a")))]);
    let error = db
        .execute_sparql(&format!("{PREFIX} SELECT ?o WHERE {{ e:s e:p+ ?o }}"))
        .expect_err("expired deadline must not return a partial query result");
    assert_eq!(error.error_code(), ErrorCode::QueryTimeout, "{error}");
}

#[test]
fn exhausted_memory_returns_error_instead_of_partial_path_rows() {
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_graph_model(GraphModel::Rdf)
            .with_memory_limit(1),
    )
    .unwrap();
    insert_terms(&db, &[(Term::iri(iri("s")), Term::iri(iri("a")))]);
    let error = db
        .execute_sparql(&format!("{PREFIX} SELECT ?o WHERE {{ e:s e:p+ ?o }}"))
        .expect_err("exhausted memory must not return a partial query result");
    assert_eq!(error.error_code(), ErrorCode::QueryExecution, "{error}");
}

#[test]
fn pending_only_named_graph_paths_follow_transaction_lifecycle() {
    let db = rdf_db();
    let mut writer = db.session();
    let reader = db.session();
    let assert_visible = |session: &Session, nodes: &[&str]| {
        for query in [
            "SELECT ?o WHERE { GRAPH e:g { e:s e:p+ ?o } }",
            "SELECT ?o FROM e:g WHERE { e:s e:p+ ?o }",
        ] {
            assert_session_rows(session, query, nodes.len(), endpoints(nodes));
        }
        for query in [
            "SELECT ?g ?o WHERE { GRAPH ?g { e:s e:p+ ?o } }",
            "SELECT ?g ?o FROM NAMED e:g WHERE { GRAPH ?g { e:s e:p+ ?o } }",
        ] {
            let expected = nodes.iter().map(|node| vec![iri("g"), iri(node)]).collect();
            assert_session_rows(session, query, nodes.len(), expected);
        }
    };
    let insert = format!("{PREFIX} INSERT DATA {{ GRAPH e:g {{ e:s e:p e:a . e:a e:p e:b . }} }}");

    writer.begin_transaction().unwrap();
    writer.execute_sparql(&insert).unwrap();
    assert_visible(&writer, &["a", "b"]);
    assert_visible(&reader, &[]);

    writer
        .execute_sparql(&format!(
            "{PREFIX} DELETE DATA {{ GRAPH e:g {{ e:a e:p e:b . }} }}"
        ))
        .unwrap();
    assert_visible(&writer, &["a"]);
    assert_visible(&reader, &[]);

    writer
        .execute_sparql(&format!(
            "{PREFIX} INSERT DATA {{ GRAPH e:g {{ e:a e:p e:b . }} }}"
        ))
        .unwrap();
    assert_visible(&writer, &["a", "b"]);
    assert_visible(&reader, &[]);

    writer.rollback().unwrap();
    assert_visible(&writer, &[]);
    assert_visible(&reader, &[]);

    let mut committer = db.session();
    committer.begin_transaction().unwrap();
    committer.execute_sparql(&insert).unwrap();
    assert_visible(&committer, &["a", "b"]);
    assert_visible(&reader, &[]);
    committer.commit().unwrap();
    assert_visible(&committer, &["a", "b"]);
    assert_visible(&writer, &["a", "b"]);
    assert_visible(&reader, &["a", "b"]);
}

#[test]
fn pending_empty_named_graph_keeps_bound_star_reflexivity_after_commit() {
    let assert_named_graph = |session: &Session, graph: &str, present: bool| {
        let count = usize::from(present);
        let expected_endpoints = if present { endpoints(&["s"]) } else { vec![] };
        assert_session_rows(
            session,
            &format!("SELECT ?o WHERE {{ GRAPH e:{graph} {{ e:s e:p* ?o }} }}"),
            count,
            expected_endpoints,
        );
        for query in [
            "SELECT ?g ?o WHERE { GRAPH ?g { e:s e:p* ?o } }".to_owned(),
            format!("SELECT ?g ?o FROM NAMED e:{graph} WHERE {{ GRAPH ?g {{ e:s e:p* ?o }} }}"),
        ] {
            let expected = if present {
                pairs(&[(graph, "s")])
            } else {
                vec![]
            };
            assert_session_rows(session, &query, count, expected);
        }
    };

    let db = rdf_db();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    for update in [
        "INSERT DATA { GRAPH e:g { e:s e:p e:a . } }",
        "DELETE DATA { GRAPH e:g { e:s e:p e:a . } }",
    ] {
        writer
            .execute_sparql(&format!("{PREFIX} {update}"))
            .unwrap();
    }
    // Deleting the last triple empties the graph created by INSERT; it does
    // not erase its existence, which bound-star GRAPH queries can observe.
    assert_named_graph(&writer, "g", true);
    assert_named_graph(&reader, "g", false);
    let from_graph = "SELECT ?o FROM e:g WHERE { e:s e:p* ?o }";
    assert_session_rows(&writer, from_graph, 1, endpoints(&["s"]));
    writer.commit().unwrap();
    for session in [&writer, &reader] {
        assert_named_graph(session, "g", true);
        assert_session_rows(session, from_graph, 1, endpoints(&["s"]));
    }

    let absent_db = rdf_db();
    let mut deleter = absent_db.session();
    let observer = absent_db.session();
    deleter.begin_transaction().unwrap();
    deleter
        .execute_sparql(&format!(
            "{PREFIX} DELETE DATA {{ GRAPH e:h {{ e:s e:p e:a . }} }}"
        ))
        .unwrap();
    assert_named_graph(&deleter, "h", false);
    assert_named_graph(&observer, "h", false);
    deleter.commit().unwrap();
    assert_named_graph(&deleter, "h", false);
    assert_named_graph(&observer, "h", false);
    // FROM's empty active default graph can itself yield bound reflexivity;
    // named GRAPH queries above distinguish absence from an existing graph.

    let assert_extra_graph = |session: &Session, graph: &str, present: bool| {
        let count = usize::from(present);
        let expected = if present { endpoints(&["s"]) } else { vec![] };
        assert_session_rows(
            session,
            &format!("SELECT ?o WHERE {{ GRAPH e:{graph} {{ e:s e:p* ?o }} }}"),
            count,
            expected,
        );
        let expected_graphs = if present {
            pairs(&[("g", "s"), (graph, "s")])
        } else {
            pairs(&[("g", "s")])
        };
        assert_session_rows(
            session,
            "SELECT ?g ?o WHERE { GRAPH ?g { e:s e:p* ?o } }",
            1 + count,
            expected_graphs,
        );
        let expected_named = if present {
            pairs(&[(graph, "s")])
        } else {
            vec![]
        };
        assert_session_rows(
            session,
            &format!("SELECT ?g ?o FROM NAMED e:{graph} WHERE {{ GRAPH ?g {{ e:s e:p* ?o }} }}"),
            count,
            expected_named,
        );
    };
    let insert_then_delete = |session: &Session, graph: &str| {
        for operation in ["INSERT", "DELETE"] {
            session
                .execute_sparql(&format!(
                    "{PREFIX} {operation} DATA {{ GRAPH e:{graph} {{ e:s e:p e:a . }} }}"
                ))
                .unwrap();
        }
    };

    writer.begin_transaction().unwrap();
    insert_then_delete(&writer, "u");
    assert_extra_graph(&writer, "u", true);
    assert_extra_graph(&reader, "u", false);
    writer.rollback().unwrap();
    for session in [&writer, &reader] {
        assert_extra_graph(session, "u", false);
    }

    writer.begin_transaction().unwrap();
    writer.savepoint("sp").unwrap();
    insert_then_delete(&writer, "v");
    assert_extra_graph(&writer, "v", true);
    assert_extra_graph(&reader, "v", false);
    writer.rollback_to_savepoint("sp").unwrap();
    for session in [&writer, &reader] {
        assert_extra_graph(session, "v", false);
    }
    writer.commit().unwrap();
    for session in [&writer, &reader] {
        assert_extra_graph(session, "v", false);
    }
}
