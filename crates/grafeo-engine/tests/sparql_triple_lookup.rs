//! A SPARQL triple pattern whose subject, predicate or object the rows before
//! it always bind looks its triples up per row through the store's indexes
//! (0.6.1, LDBC IS1 on RDF); past a limit that grows with the pattern's
//! estimate (at most 1,024 rows) it joins as a hash join does. Its rows must be
//! those of the plain join, which the same query
//! with the pattern in a sub-select gives (a sub-select is not looked up);
//! each case compares the two, and the main ones check their rows:
//!
//! - the bound subject, object or predicate, and two bound positions;
//! - VALUES of 1 row (looked up) and of 1,024, 1,025 and 5,000 rows (past
//!   the limit of a pattern of 2,000 triples), with repeated rows;
//! - a term a BIND computes (an IRI and a literal), a term no triple has,
//!   a term of another datatype or language;
//! - two triple patterns (the lookup side chosen by the estimates);
//! - an OPTIONAL left side that holds the free variable (the plain join);
//! - a constant GRAPH, FROM over two graphs that share a triple (its
//!   duplicates as before), GRAPH ?g (the plain join);
//! - a transaction's pending inserts and deletes, in both modes;
//! - PROFILE: the lookup operator and its rows;
//! - a UNION of patterns (an alternative path `?p (e:knows|^e:knows) ?f`,
//!   or written out), looked up in each branch: the rows of the two
//!   single-direction queries together, duplicates kept, a variable of one
//!   branch unbound in the other's rows, three branches, the UNION before
//!   the pattern that binds; past the limit; an OPTIONAL of one; a blank
//!   node as the looked-up term; and the UNION shapes left to the plain join
//!   (a branch without the bound variable, with a FILTER or an OPTIONAL, a
//!   repeated variable, GRAPH ?g).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_triple_lookup::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]
#![allow(
    deprecated,
    reason = "the test reaches the row-group store itself, which only the deprecated GrafeoDB::store() gives, until 0.7.0 removes it"
)]

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

/// People p0 to p1999: `e:name "n{i}"`, `e:age {i % 50}`, `e:knows p{i+1}`
/// (p1999 none); p1 also has `e:name "Alix"@nl`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut data = String::new();
    for i in 0..2_000 {
        write!(data, "e:p{i} e:name \"n{i}\" ; e:age {} ", i % 50).unwrap();
        if i < 1_999 {
            write!(data, "; e:knows e:p{} ", i + 1).unwrap();
        }
        data.push_str(". ");
    }
    data.push_str("e:p1 e:name \"Alix\"@nl . ");
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {data} }}"))
        .unwrap();
    db
}

/// A term's text: an IRI's local name, a literal's lexical form (with its
/// language), `-` unbound.
fn text(value: &Value) -> String {
    use grafeo_common::types::rdf::Term;
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://example.org/")
                .unwrap_or(iri.as_str())
                .to_string(),
            Term::Literal(literal) => match literal.language() {
                Some(language) => format!("{}@{language}", literal.value()),
                None => literal.value().to_string(),
            },
            other => format!("{other:?}"),
        },
        Value::Null => "-".to_string(),
        other => format!("{other:?}"),
    }
}

fn sorted(result: &QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect::<Vec<_>>().join(" "))
        .collect();
    rows.sort();
    rows
}

type Execute<'a> = &'a dyn Fn(&str) -> Result<QueryResult>;

fn run(execute: Execute<'_>, query: &str) -> std::result::Result<Vec<String>, String> {
    execute(&format!("{PREFIX} {query}"))
        .map(|result| sorted(&result))
        .map_err(|error| error.to_string())
}

/// `SELECT {projection} {dataset} WHERE { {left} {pattern} }` against the
/// same with the pattern in a sub-select over its own variables; they must
/// agree. Returns the rows.
fn agree_with(
    execute: Execute<'_>,
    projection: &str,
    dataset: &str,
    left: &str,
    pattern: &str,
    pattern_variables: &str,
) -> Vec<String> {
    let looked_up = format!("SELECT {projection} {dataset} WHERE {{ {left} {pattern} }}");
    let joined = format!(
        "SELECT {projection} {dataset} WHERE {{ {left} {{ SELECT {pattern_variables} WHERE {{ {pattern} }} }} }}"
    );
    let got = run(execute, &looked_up);
    assert_eq!(got, run(execute, &joined), "`{looked_up}` against the join");
    got.unwrap_or_else(|error| panic!("`{looked_up}`: {error}"))
}

fn agree(
    db: &GrafeoDB,
    projection: &str,
    left: &str,
    pattern: &str,
    variables: &str,
) -> Vec<String> {
    agree_with(
        &|q| db.execute_sparql(q),
        projection,
        "",
        left,
        pattern,
        variables,
    )
}

/// The profile text of `query` on the committed store.
fn profile(db: &GrafeoDB, query: &str) -> String {
    let processor = QueryProcessor::with_rdf(db.store().unwrap(), Arc::clone(db.rdf_store()));
    let result = processor
        .process(
            &format!("EXPLAIN ANALYZE {PREFIX} {query}"),
            QueryLanguage::Sparql,
            None,
        )
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{other:?}"),
    }
}

/// `VALUES ?p { ... }` of `count` rows: p0, p1, ... wrapping at 2,000, so
/// past 2,000 rows repeat.
fn values_of_people(count: usize) -> String {
    let mut values = String::from("VALUES ?p { ");
    for i in 0..count {
        write!(values, "e:p{} ", i % 2_000).unwrap();
    }
    values.push('}');
    values
}

#[test]
fn each_bound_position() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?p { e:p1 e:p7 }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p1 Alix@nl", "p1 n1", "p7 n7"],
        "the subject"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?n { \"n5\" \"n9\" }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p5 n5", "p9 n9"],
        "the object"
    );
    assert_eq!(
        agree(
            &db,
            "?r ?o",
            "VALUES ?r { e:name e:age }",
            "e:p3 ?r ?o .",
            "?r ?o"
        ),
        ["age 3", "name n3"],
        "the predicate"
    );
    let all_ages = agree(
        &db,
        "?s ?o",
        "VALUES ?r { e:age }",
        "?s ?r ?o .",
        "?s ?r ?o",
    );
    assert_eq!(all_ages.len(), 2_000, "the predicate alone");
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES (?p ?n) { (e:p1 \"n1\") (e:p1 \"n2\") (e:p2 \"n2\") }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p1 n1", "p2 n2"],
        "subject and object"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?q",
            "VALUES (?p ?r) { (e:p4 e:knows) (e:p1999 e:knows) }",
            "?p ?r ?q .",
            "?p ?r ?q"
        ),
        ["p4 p5"],
        "subject and predicate"
    );
}

/// 1 row looks up; 1,024, 1,025 and 5,000 rows are past the limit of a
/// pattern of 2,000 triples (and 1,025 past the most any pattern looks up).
/// Each row joins: repeated rows give their rows again.
#[test]
fn values_on_both_sides_of_the_switch() {
    let db = database();
    for count in [1, 1_024, 1_025, 5_000] {
        let rows = agree(
            &db,
            "?p ?a",
            &values_of_people(count),
            "?p e:age ?a .",
            "?p ?a",
        );
        assert_eq!(rows.len(), count, "{count} rows, one age each");
        let names = agree(
            &db,
            "?p ?n",
            &values_of_people(count),
            "?p e:name ?n .",
            "?p ?n",
        );
        let with_alix = (0..count).filter(|i| i % 2_000 == 1).count();
        assert_eq!(names.len(), count + with_alix, "{count}: p1 has two names");
    }
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?p { e:p7 e:p7 e:p7 }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p7 n7", "p7 n7", "p7 n7"]
    );
}

/// A term a BIND computes, a term no triple has, and literals that differ
/// from a stored one only by language or datatype.
#[test]
fn computed_unknown_and_near_terms() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "BIND(IRI(CONCAT(STR(e:), \"p\", \"7\")) AS ?p)",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p7 n7"],
        "a computed IRI"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "BIND(CONCAT(\"n\", \"9\") AS ?n)",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p9 n9"],
        "a computed literal"
    );
    for left in [
        "VALUES ?p { e:nobody }",
        "VALUES ?n { \"zzz\" }",
        "VALUES ?n { \"n5\"@en }",
        "VALUES ?n { \"Alix\" }",
    ] {
        assert_eq!(
            agree(&db, "?p ?n", left, "?p e:name ?n .", "?p ?n"),
            Vec::<String>::new(),
            "{left}"
        );
    }
    for left in [
        "VALUES ?a { \"7\" }",
        "VALUES ?a { 7.0 }",
        "VALUES ?a { \"07\"^^<http://www.w3.org/2001/XMLSchema#integer> }",
    ] {
        agree(&db, "?p ?a", left, "?p e:age ?a .", "?p ?a");
    }
}

/// Two triple patterns: one is looked up with the rows of the other, both
/// orders written.
#[test]
fn two_triple_patterns() {
    let db = database();
    let execute = |q: &str| db.execute_sparql(q);
    for (query, every_pair) in [
        (
            "SELECT ?p ?n WHERE { ?p e:age 7 . ?p e:name ?n . }",
            "SELECT ?p ?n WHERE { ?p e:age 7 . { SELECT ?p ?n WHERE { ?p e:name ?n } } }",
        ),
        (
            "SELECT ?p ?n WHERE { ?p e:name ?n . ?p e:age 7 . }",
            "SELECT ?p ?n WHERE { { SELECT ?p ?n WHERE { ?p e:name ?n } } ?p e:age 7 . }",
        ),
        (
            "SELECT ?p ?q WHERE { ?p e:name \"n5\" . ?p e:knows ?q . ?q e:name ?m . }",
            "SELECT ?p ?q WHERE { ?p e:name \"n5\" . \
             { SELECT ?p ?q ?m WHERE { ?p e:knows ?q . ?q e:name ?m } } }",
        ),
    ] {
        assert_eq!(run(&execute, query), run(&execute, every_pair), "{query}");
    }
    assert_eq!(
        run(
            &execute,
            "SELECT ?p ?n WHERE { ?p e:age 7 . ?p e:name ?n . }"
        )
        .unwrap()
        .len(),
        40,
        "ages 7 of 2,000 people"
    );
}

/// An OPTIONAL left side holds ?n without always binding it: the plain join
/// for `?p e:name ?n` (the OPTIONAL itself looks its pattern up from ?p).
#[test]
fn an_optional_left_side_holding_the_free_variable() {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ e:p7 e:nick \"n7\" . e:p57 e:nick \"x\" . }}"
    ))
    .unwrap();
    // Nicks of the people of other ages, so the OPTIONAL is the larger side
    // (a side of two triples is a hash join's build side instead).
    let others = (0..2_000)
        .filter(|i| i % 50 != 7)
        .fold(String::new(), |mut out, i| {
            use std::fmt::Write;
            write!(out, "e:p{i} e:nick \"m{i}\" . ").unwrap();
            out
        });
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {others} }}"))
        .unwrap();
    let rows = agree(
        &db,
        "?p ?n",
        "?p e:age 7 . OPTIONAL { ?p e:nick ?n }",
        "?p e:name ?n .",
        "?p ?n",
    );
    assert_eq!(rows.len(), 39, "p57's nick is no name of it, p7's is");
    assert!(rows.contains(&"p7 n7".to_string()));
    let text = profile(
        &db,
        "SELECT ?p ?n WHERE { ?p e:age 7 . OPTIONAL { ?p e:nick ?n } ?p e:name ?n . }",
    );
    assert_eq!(
        text.matches("RdfTripleLookup").count(),
        1,
        "only the OPTIONAL looks up: {text}"
    );
}

/// A constant GRAPH; FROM over g1 and g2, which both hold `e:q1 e:name "q1"`
/// (FROM returns that triple twice today: the lookup must not change it);
/// GRAPH ?g (the plain join).
#[test]
fn graphs() {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ \
         GRAPH e:g1 {{ e:q1 e:name \"q1\" . e:q2 e:name \"q2\" . }} \
         GRAPH e:g2 {{ e:q1 e:name \"q1\" . e:q3 e:name \"q3\" . }} }}"
    ))
    .unwrap();
    let execute = |q: &str| db.execute_sparql(q);
    let left = "VALUES ?p { e:q1 e:q2 e:q3 e:p1 }";
    let in_g1 = format!("SELECT ?p ?n WHERE {{ {left} GRAPH e:g1 {{ ?p e:name ?n }} }}");
    let in_g1_joined = format!(
        "SELECT ?p ?n WHERE {{ {left} GRAPH e:g1 {{ {{ SELECT ?p ?n WHERE {{ ?p e:name ?n }} }} }} }}"
    );
    assert_eq!(run(&execute, &in_g1), run(&execute, &in_g1_joined));
    assert_eq!(
        run(&execute, &in_g1).unwrap(),
        ["q1 q1", "q2 q2"],
        "a constant GRAPH"
    );
    let from = agree_with(
        &execute,
        "?p ?n",
        "FROM e:g1 FROM e:g2",
        left,
        "?p e:name ?n .",
        "?p ?n",
    );
    let from_scan = run(
        &execute,
        "SELECT ?p ?n FROM e:g1 FROM e:g2 WHERE { ?p e:name ?n . FILTER(?p IN (e:q1, e:q2, e:q3, e:p1)) }",
    )
    .unwrap();
    assert_eq!(from, from_scan, "FROM as a scan gives it");
    let graph_free = "SELECT ?g ?n WHERE { VALUES ?p { e:q1 } GRAPH ?g { ?p e:name ?n } }";
    let graph_free_pairs = "SELECT ?g ?n WHERE { VALUES ?p { e:q1 } \
                            { SELECT ?g ?p ?n WHERE { GRAPH ?g { ?p e:name ?n } } } }";
    assert_eq!(run(&execute, graph_free), run(&execute, graph_free_pairs));
    assert_eq!(run(&execute, graph_free).unwrap(), ["g1 q1", "g2 q1"]);
}

/// A transaction's pending insert and delete, with 10 rows (looked up) and
/// 1,025 (joined as a hash join does): the writer sees them, a reader does
/// not, the rollback drops them.
#[test]
fn a_transactions_pending_changes() {
    let db = database();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!("{PREFIX} INSERT DATA {{ e:p2 e:name \"Gus\" . }}"))
        .unwrap();
    writer
        .execute_sparql(&format!("{PREFIX} DELETE DATA {{ e:p3 e:name \"n3\" . }}"))
        .unwrap();
    let names = |session: &grafeo_engine::session::Session, count: usize| {
        let execute = |q: &str| session.execute_sparql(q);
        let rows = agree_with(
            &execute,
            "?p ?n",
            "",
            &values_of_people(count),
            "?p e:name ?n .",
            "?p ?n",
        );
        (
            rows.contains(&"p2 Gus".to_string()),
            rows.contains(&"p3 n3".to_string()),
        )
    };
    for count in [10, 1_025] {
        assert_eq!(names(&writer, count), (true, false), "writer, {count} rows");
        assert_eq!(names(&reader, count), (false, true), "reader, {count} rows");
    }
    writer.rollback().unwrap();
    for count in [10, 1_025] {
        assert_eq!(
            names(&writer, count),
            (false, true),
            "after the rollback, {count} rows"
        );
    }
}

/// PROFILE shows the lookup with its rows; the sub-select form does not.
#[test]
fn profile_shows_the_lookup() {
    let db = database();
    let text = profile(
        &db,
        "SELECT ?p ?n WHERE { VALUES ?p { e:p1 e:p7 e:p8 } ?p e:name ?n . }",
    );
    let lookups: Vec<&str> = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("RdfTripleLookup"))
        .collect();
    assert_eq!(lookups.len(), 1, "{text}");
    assert!(lookups[0].contains("rows=4"), "p1 has two names: {text}");
    let joined = profile(
        &db,
        "SELECT ?p ?n WHERE { VALUES ?p { e:p1 e:p7 e:p8 } { SELECT ?p ?n WHERE { ?p e:name ?n } } }",
    );
    assert!(!joined.contains("RdfTripleLookup"), "{joined}");
}

/// The rows of `forward` and of `inverse`, together and sorted: what their
/// UNION gives (a bag: a row both give is there twice).
fn both(db: &GrafeoDB, forward: &str, inverse: &str) -> Vec<String> {
    let execute = |q: &str| db.execute_sparql(q);
    let mut rows = run(&execute, forward).unwrap();
    rows.extend(run(&execute, inverse).unwrap());
    rows.sort();
    rows
}

/// A join into a UNION of triple patterns that each hold a variable the
/// other side binds (an alternative path `?p (e:knows|^e:knows) ?f` is one):
/// each row looks its triples up in every branch. Its rows are those of the
/// plain join and of the two single-direction queries together.
#[test]
fn a_union_of_patterns_looks_up_each_branch() {
    let db = database();
    let left = "VALUES ?p { e:p0 e:p5 e:p1999 e:nobody }";
    let expected = both(
        &db,
        &format!("SELECT ?p ?f WHERE {{ {left} ?p e:knows ?f }}"),
        &format!("SELECT ?p ?f WHERE {{ {left} ?f e:knows ?p }}"),
    );
    assert_eq!(
        expected,
        ["p0 p1", "p1999 p1998", "p5 p4", "p5 p6"],
        "p0 knows only p1, p1999 is only known"
    );
    for pattern in [
        "?p (e:knows|^e:knows) ?f .",
        "{ ?p e:knows ?f } UNION { ?f e:knows ?p }",
        "{ ?f e:knows ?p } UNION { ?p e:knows ?f }",
    ] {
        assert_eq!(
            agree(&db, "?p ?f", left, pattern, "?p ?f"),
            expected,
            "{pattern}"
        );
    }
    // The UNION before the pattern that binds ?p.
    let execute = |q: &str| db.execute_sparql(q);
    let union_first =
        "SELECT ?p ?f WHERE { { ?p e:knows ?f } UNION { ?f e:knows ?p } ?p e:age 7 . }";
    let union_first_joined = "SELECT ?p ?f WHERE { \
         { SELECT ?p ?f WHERE { { ?p e:knows ?f } UNION { ?f e:knows ?p } } } ?p e:age 7 . }";
    let rows = run(&execute, union_first).unwrap();
    assert_eq!(rows, run(&execute, union_first_joined).unwrap());
    assert_eq!(rows.len(), 80, "40 people of age 7, two friends each");
}

/// Bag semantics: a branch twice gives its rows twice, a triple both branches
/// match (a self loop) twice; a variable of one branch only is unbound in the
/// other's rows; three branches.
#[test]
fn a_union_lookup_keeps_duplicates_and_unbound_variables() {
    let db = database();
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ e:p5 e:knows e:p5 . }}"))
        .unwrap();
    let left = "VALUES ?p { e:p5 e:p7 }";
    assert_eq!(
        agree(
            &db,
            "?p ?f",
            left,
            "{ ?p e:knows ?f } UNION { ?p e:knows ?f }",
            "?p ?f"
        ),
        ["p5 p5", "p5 p5", "p5 p6", "p5 p6", "p7 p8", "p7 p8"],
        "the same branch twice"
    );
    assert_eq!(
        agree(&db, "?p ?f", left, "?p (e:knows|^e:knows) ?f .", "?p ?f"),
        ["p5 p4", "p5 p5", "p5 p5", "p5 p6", "p7 p6", "p7 p8"],
        "the self loop matches both directions"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?f ?n",
            left,
            "{ ?p e:knows ?f } UNION { ?p e:name ?n }",
            "?p ?f ?n"
        ),
        ["p5 - n5", "p5 p5 -", "p5 p6 -", "p7 - n7", "p7 p8 -"],
        "each branch leaves the other's variable unbound"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?o",
            "VALUES ?p { e:p3 }",
            "?p (e:knows|^e:knows|e:age) ?o .",
            "?p ?o"
        ),
        ["p3 3", "p3 p2", "p3 p4"],
        "three branches"
    );
}

/// UNION shapes the lookup leaves to the plain join, which must still agree:
/// a branch without the bound variable, a branch with a FILTER or an
/// OPTIONAL, a repeated variable, a GRAPH ?g branch; and rows past the limit.
#[test]
fn union_shapes_the_lookup_leaves_alone() {
    let db = database();
    let left = "VALUES ?p { e:p5 e:p7 }";
    for (pattern, variables) in [
        ("{ ?p e:knows ?f } UNION { ?f e:age 49 }", "?p ?f"),
        (
            "{ ?p e:knows ?f OPTIONAL { ?f e:nick ?n } } UNION { ?f e:knows ?p }",
            "?p ?f ?n",
        ),
        ("{ ?p e:knows ?p } UNION { ?f e:knows ?p }", "?p ?f"),
        (
            "{ GRAPH ?g { ?p e:knows ?f } } UNION { ?f e:knows ?p }",
            "?p ?f ?g",
        ),
    ] {
        agree(&db, variables, left, pattern, variables);
    }
    assert_eq!(
        agree(
            &db,
            "?p ?f",
            left,
            "{ ?p e:knows ?f FILTER(?f != e:p6) } UNION { ?f e:knows ?p }",
            "?p ?f"
        ),
        ["p5 p4", "p7 p6", "p7 p8"],
        "the FILTER holds in its branch only"
    );
    for count in [1_024, 1_025, 5_000] {
        let rows = agree(
            &db,
            "?p ?f",
            &values_of_people(count),
            "?p (e:knows|^e:knows) ?f .",
            "?p ?f",
        );
        // p0 and p1999 have one friend each, the others two.
        let ends = (0..count)
            .filter(|i| matches!(i % 2_000, 0 | 1_999))
            .count();
        assert_eq!(rows.len(), 2 * count - ends, "{count} rows");
    }
}

/// An OPTIONAL of a UNION of patterns looks each branch up and keeps a row
/// none of them matches; a blank node as a looked-up term.
#[test]
fn an_optional_union_and_blank_nodes() {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ _:b e:knows e:p5 . _:b e:knows e:p9 . }}"
    ))
    .unwrap();
    let execute = |q: &str| db.execute_sparql(q);
    let optional = "SELECT ?p ?f WHERE { VALUES ?p { e:p0 e:nobody } \
                    OPTIONAL { ?p (e:knows|^e:knows) ?f } }";
    let optional_joined = "SELECT ?p ?f WHERE { VALUES ?p { e:p0 e:nobody } \
                           OPTIONAL { { SELECT ?p ?f WHERE { ?p (e:knows|^e:knows) ?f } } } }";
    assert_eq!(run(&execute, optional), run(&execute, optional_joined));
    assert_eq!(run(&execute, optional).unwrap(), ["nobody -", "p0 p1"]);
    // ?b is the blank node the rows bind: its friends are looked up from it.
    let through_blank = "SELECT ?g WHERE { VALUES ?p { e:p5 } ?b e:knows ?p . \
                         FILTER(isBlank(?b)) ?b (e:knows|^e:knows) ?g . }";
    let through_blank_joined = "SELECT ?g WHERE { VALUES ?p { e:p5 } ?b e:knows ?p . \
                                FILTER(isBlank(?b)) \
                                { SELECT ?b ?g WHERE { ?b (e:knows|^e:knows) ?g } } }";
    assert_eq!(
        run(&execute, through_blank),
        run(&execute, through_blank_joined)
    );
    assert_eq!(run(&execute, through_blank).unwrap(), ["p5", "p9"]);
}

/// PROFILE shows one lookup for a join into a UNION, with its rows and the
/// UNION's branches in its line, and no operator with more rows (a UNION
/// that read every `e:knows` triple); an OPTIONAL of one the same. A pattern
/// that binds ?p (one triple) is not looked up per UNION row instead.
#[test]
fn profile_shows_the_union_lookup() {
    let db = database();
    for (query, rows) in [
        (
            "SELECT ?p ?f WHERE { VALUES ?p { e:p5 e:p7 } ?p (e:knows|^e:knows) ?f . }",
            4,
        ),
        (
            "SELECT ?p ?f WHERE { VALUES ?p { e:p5 e:p7 } { ?p e:knows ?f } UNION { ?f e:knows ?p } }",
            4,
        ),
        (
            "SELECT ?p ?f WHERE { ?p (e:knows|^e:knows) ?f . VALUES ?p { e:p5 e:p7 } }",
            4,
        ),
        (
            "SELECT ?p ?f WHERE { VALUES ?p { e:p5 e:p7 } OPTIONAL { ?p (e:knows|^e:knows) ?f } }",
            4,
        ),
        (
            "SELECT ?p ?f WHERE { ?p e:name \"n5\" . ?p (e:knows|^e:knows) ?f . }",
            2,
        ),
        (
            "SELECT ?p ?f WHERE { ?p (e:knows|^e:knows) ?f . ?p e:name \"n5\" . }",
            2,
        ),
    ] {
        let text = profile(&db, query);
        let lookups: Vec<&str> = text
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with("RdfTripleLookup"))
            .collect();
        assert_eq!(lookups.len(), 1, "{query}: {text}");
        assert!(
            lookups[0].contains("RdfUnion 2 branches"),
            "{query}: {text}"
        );
        assert!(
            lookups[0].contains(&format!("rows={rows}")),
            "{query}: {text}"
        );
        // No operator reads the 1,999 `e:knows` triples.
        let most = text
            .split("rows=")
            .skip(1)
            .filter_map(|rest| rest.split_whitespace().next()?.parse::<usize>().ok())
            .max();
        assert_eq!(most, Some(rows), "{query}: {text}");
    }
}
