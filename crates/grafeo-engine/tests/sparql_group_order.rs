//! The order of a SPARQL group's patterns (0.6.1): each chain of inner joins
//! is rebuilt to start at its most selective input and to join next the
//! cheapest one connected to the variables bound so far. An inner join
//! commutes, so the rows must not depend on the written order: each case runs
//! the group in several written orders and compares the rows, and checks them
//! against rows worked out by hand:
//!
//! - a star on one subject, a chain, a cross product of two disconnected
//!   parts, VALUES written first, in the middle and last;
//! - an OPTIONAL whose maybe-unbound variable a later pattern shares (the
//!   join stays compatible), MINUS, BIND between patterns (a boundary),
//!   FILTER inside the group, GRAPH ?g, a property path in the chain and a
//!   sub-select;
//! - PROFILE: a VALUES block or a pattern with a constant written last is
//!   joined first.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_group_order
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

/// People p0 to p49: `e:name "n{i}"`, `e:age {20 + i % 10}`, living in one of
/// five cities (`e:livesIn`, each city with `e:cityName`), `e:knows` the next
/// one (p49 nobody). p1 has `e:nick "Gus"`; owners o1 and o2 own the nicks
/// "Gus" and "Zed" (`e:owns`). p1 to p4 are in graph g1 with `e:member`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let cities = ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"];
    let mut data = String::new();
    for (k, city) in cities.iter().enumerate() {
        write!(data, "e:c{k} e:cityName \"{city}\" . ").unwrap();
    }
    for i in 0..50 {
        write!(
            data,
            "e:p{i} a e:Person ; e:name \"n{i}\" ; e:age {} ; e:livesIn e:c{} . ",
            20 + i % 10,
            i % 5
        )
        .unwrap();
        if i < 49 {
            write!(data, "e:p{i} e:knows e:p{} . ", i + 1).unwrap();
        }
    }
    data.push_str("e:p1 e:nick \"Gus\" . e:o1 e:owns \"Gus\" . e:o2 e:owns \"Zed\" . ");
    data.push_str("GRAPH e:g1 { e:p1 e:member e:g1 . e:p2 e:member e:g1 . e:p3 e:member e:g1 . e:p4 e:member e:g1 . } ");
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {data} }}"))
        .unwrap();
    db
}

/// A term's text: an IRI's local name, a literal's lexical form, `-` unbound.
fn text(value: &Value) -> String {
    use grafeo_common::types::rdf::Term;
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://example.org/")
                .unwrap_or(iri.as_str())
                .to_string(),
            Term::Literal(literal) => literal.value().to_string(),
            other => format!("{other:?}"),
        },
        Value::Null => "-".to_string(),
        other => format!("{other:?}"),
    }
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<String> {
    let result = db
        .execute_sparql(&format!("{PREFIX} {query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect::<Vec<_>>().join(" "))
        .collect();
    rows.sort();
    rows
}

/// `SELECT {projection} WHERE { ... }` with the group's `parts` in each of
/// `orders` (indexes into `parts`); every order must give the same rows,
/// which are returned.
fn same_in_every_order(
    db: &GrafeoDB,
    projection: &str,
    parts: &[&str],
    orders: &[&[usize]],
) -> Vec<String> {
    let mut first: Option<(String, Vec<String>)> = None;
    for order in orders {
        let group: Vec<&str> = order.iter().map(|&i| parts[i]).collect();
        let query = format!("SELECT {projection} WHERE {{ {} }}", group.join(" "));
        let got = rows(db, &query);
        match &first {
            None => first = Some((query, got)),
            Some((written, want)) => assert_eq!(&got, want, "`{query}` against `{written}`"),
        }
    }
    first.expect("an order").1
}

/// Every order of `n` parts, for small `n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..=rest.len() {
            let mut order = rest.clone();
            order.insert(at, n - 1);
            out.push(order);
        }
    }
    out
}

fn every_order(db: &GrafeoDB, projection: &str, parts: &[&str]) -> Vec<String> {
    let orders = permutations(parts.len());
    let orders: Vec<&[usize]> = orders.iter().map(Vec::as_slice).collect();
    same_in_every_order(db, projection, parts, &orders)
}

#[test]
fn a_star_on_one_subject() {
    let db = database();
    let got = every_order(
        &db,
        "?p ?n ?a ?city",
        &[
            "VALUES ?p { e:p3 e:p17 }",
            "?p a e:Person .",
            "?p e:name ?n .",
            "?p e:age ?a .",
            "?p e:livesIn ?c . ?c e:cityName ?city .",
        ],
    );
    assert_eq!(got, ["p17 n17 27 Paris", "p3 n3 23 Prague"]);
}

#[test]
fn a_chain_from_a_constant() {
    let db = database();
    let got = every_order(
        &db,
        "?b ?c ?n",
        &[
            "e:p10 e:knows ?b .",
            "?b e:knows ?c .",
            "?c e:name ?n .",
            "?c e:livesIn e:c2 .",
        ],
    );
    assert_eq!(got, ["p11 p12 n12"]);
}

/// Two parts that share no variable: every pair of their rows.
#[test]
fn a_cross_product() {
    let db = database();
    let got = every_order(
        &db,
        "?p ?c",
        &[
            "?p e:name \"n7\" .",
            "?c e:cityName ?name .",
            "?p e:age 27 .",
        ],
    );
    assert_eq!(got, ["p7 c0", "p7 c1", "p7 c2", "p7 c3", "p7 c4"]);
}

#[test]
fn values_first_in_the_middle_and_last() {
    let db = database();
    let got = every_order(
        &db,
        "?p ?n",
        &[
            "?p e:name ?n .",
            "VALUES ?p { e:p1 e:p2 e:p2 e:nobody }",
            "?p e:age ?a .",
        ],
    );
    assert_eq!(got, ["p1 n1", "p2 n2", "p2 n2"], "each VALUES row joins");
}

/// After an OPTIONAL, ?k may be unbound: p1 has the nick Gus, p2 none. A
/// later pattern shares ?k, so p1 joins only the owner of Gus and p2 every
/// owner (an unbound variable is compatible with any value).
#[test]
fn an_optional_variable_a_later_pattern_shares() {
    let db = database();
    let query = "SELECT ?p ?x WHERE { VALUES ?p { e:p1 e:p2 } ?p e:name ?n . \
                 OPTIONAL { ?p e:nick ?k } ?x e:owns ?k . }";
    assert_eq!(rows(&db, query), ["p1 o1", "p2 o1", "p2 o2"]);
    let later_first = "SELECT ?p ?x WHERE { ?p e:name ?n . VALUES ?p { e:p1 e:p2 } \
                       OPTIONAL { ?p e:nick ?k } ?x e:owns ?k . }";
    assert_eq!(rows(&db, later_first), ["p1 o1", "p2 o1", "p2 o2"]);
}

/// FILTER anywhere in the group, MINUS last: a MINUS written first would
/// take nothing from the empty group before it (SPARQL 1.1 sec 18.6), so it
/// stays after the patterns it removes from.
#[test]
fn minus_and_filter_inside_the_group() {
    let db = database();
    let parts = [
        "?p e:livesIn e:c0 .",
        "?p e:age ?a .",
        "FILTER(?a >= 25)",
        "MINUS { ?p e:name \"n5\" }",
    ];
    let orders: Vec<Vec<usize>> = permutations(3)
        .into_iter()
        .map(|mut order| {
            order.push(3);
            order
        })
        .collect();
    let orders: Vec<&[usize]> = orders.iter().map(Vec::as_slice).collect();
    let got = same_in_every_order(&db, "?p", &parts, &orders);
    assert_eq!(got, ["p15", "p25", "p35", "p45"], "ages 25 in c0, but p5");
}

/// BIND ends the patterns before it: the ones after join with its rows.
#[test]
fn bind_between_patterns() {
    let db = database();
    let query = "SELECT ?p ?q WHERE { VALUES ?p { e:p3 } ?p e:age ?a . \
                 BIND(?a + 1 AS ?b) ?q e:age ?b . ?q e:livesIn e:c4 . }";
    assert_eq!(
        rows(&db, query),
        ["p3 p14", "p3 p24", "p3 p34", "p3 p4", "p3 p44"]
    );
    let written_other_way = "SELECT ?p ?q WHERE { ?p e:age ?a . VALUES ?p { e:p3 } \
                             BIND(?a + 1 AS ?b) ?q e:livesIn e:c4 . ?q e:age ?b . }";
    assert_eq!(rows(&db, written_other_way), rows(&db, query));
}

#[test]
fn a_graph_variable_group() {
    let db = database();
    let got = every_order(
        &db,
        "?g ?p ?a",
        &[
            "GRAPH ?g { ?p e:member ?g }",
            "?p e:age ?a .",
            "?p e:livesIn e:c3 .",
        ],
    );
    assert_eq!(got, ["g1 p3 23"]);
}

#[test]
fn a_property_path_in_the_chain() {
    let db = database();
    let got = every_order(
        &db,
        "?q ?n",
        &[
            "?p e:knows+ ?q .",
            "?q e:name ?n .",
            "VALUES ?p { e:p45 }",
            "?q e:age ?a .",
        ],
    );
    assert_eq!(got, ["p46 n46", "p47 n47", "p48 n48", "p49 n49"]);
}

#[test]
fn a_sub_select_in_the_chain() {
    let db = database();
    let got = every_order(
        &db,
        "?p ?n",
        &[
            "{ SELECT ?p WHERE { ?p e:age 21 } }",
            "?p e:name ?n .",
            "?p e:livesIn e:c1 .",
        ],
    );
    assert_eq!(got, ["p1 n1", "p11 n11", "p21 n21", "p31 n31", "p41 n41"]);
}

/// The profile text of `query` (the physical plan with each operator's
/// rows), on the committed store.
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

/// The operator lines of a plan, deepest last: the last is the first input
/// of the left-deep chain.
fn operator_lines(plan: &str) -> Vec<&str> {
    plan.lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty() && !line.starts_with("Total") && !line.starts_with("Planning")
        })
        .collect()
}

/// A VALUES block written last is the chain's first input (`RdfBind` over
/// `SingleRow` at the bottom), and of three patterns on ?p the one with a
/// constant object, written last, is read first: the deepest scan reads one
/// row, not the fifty Persons.
#[test]
fn the_selective_input_is_read_first() {
    let db = database();
    let text = profile(
        &db,
        "SELECT ?n WHERE { ?p a e:Person . ?p e:name ?n . VALUES ?p { e:p3 } }",
    );
    let lines = operator_lines(&text);
    assert!(
        lines.len() >= 2 && lines[lines.len() - 2].starts_with("RdfBind"),
        "VALUES first: {text}"
    );
    let text = profile(
        &db,
        "SELECT ?n WHERE { ?p a e:Person . ?p e:name ?n . ?p e:name \"n3\" . }",
    );
    let deepest = *operator_lines(&text).last().unwrap();
    assert!(
        deepest.starts_with("RdfTripleScan") && deepest.contains("rows=1 "),
        "the constant pattern first: {text}"
    );
}
