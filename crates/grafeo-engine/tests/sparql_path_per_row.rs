//! A SPARQL property path whose subject or object the rows before it always
//! bind runs as one search per row from that end (0.6.1, the IC13 shape
//! `?a :id 1 . ?b :id 2 . ?a (:KNOWS|^:KNOWS)+ ?b`). Its rows must be those
//! of the path over every pair joined with the rows, which the same query
//! with the path in a sub-select gives (a sub-select is not searched per
//! row); each case compares the two, and the main ones check their rows:
//!
//! - `+`, `*`, `?`, sequences, alternatives and inverses, the bound end as
//!   subject, as object and both;
//! - the end bound by a triple, by inline VALUES (repeated and unknown
//!   terms) and by a BIND (a constant and a computed IRI);
//! - `?x p+ ?x`; zero-length `*` with both ends bound and equal, and from a
//!   term no triple has (a constant start is itself, a bound one no row);
//! - an OPTIONAL left side that holds the free end, a GRAPH ?g that the
//!   left side binds (both searched over every pair);
//! - FROM over two graphs, FROM NAMED and GRAPH ?g, a transaction's pending
//!   inserts and deletes; the FROM merge counting a triple in both graphs
//!   once (ignored: it is returned twice, 0.6.0 too), a sub-select inside
//!   GRAPH ?g (ignored: it reads every graph, ?g unbound, 0.6.0 too);
//! - `?` with a constant object (ignored: other subjects and duplicates,
//!   0.6.0 too);
//! - PROFILE: the per-row operator and its rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_path_per_row
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

/// REL edges with branches, a two-cycle (e, f), a self-loop (g) and a
/// separate pair (u, v); one OTHER edge.
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
    ("u", "v", "REL"),
];

const NODES: &[&str] = &["s", "a", "b", "d", "e", "f", "g", "h", "x", "u", "v"];

/// The edges in the default graph, and each node's `e:id` (its name).
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut triples = String::new();
    for (from, to, predicate) in EDGES {
        writeln!(triples, "e:{from} e:{predicate} e:{to} .").unwrap();
    }
    for node in NODES {
        writeln!(triples, "e:{node} e:id \"{node}\" .").unwrap();
    }
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {triples} }}"))
        .unwrap();
    db
}

/// A term's text: an IRI's local name after the example base, a literal's
/// lexical form, `-` for an unbound value.
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
        Value::Bool(b) => b.to_string(),
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

/// The rows of `query`, sorted, or the error's text.
fn run(
    execute: &dyn Fn(&str) -> grafeo_common::utils::error::Result<QueryResult>,
    query: &str,
) -> Result<Vec<String>, String> {
    execute(&format!("{PREFIX} {query}"))
        .map(|result| sorted(&result))
        .map_err(|error| error.to_string())
}

/// `SELECT {projection} WHERE { {left} ?a {path} ?b }` against the same
/// with the path in a sub-select; they must agree. Returns the rows.
fn agree_with(
    execute: &dyn Fn(&str) -> grafeo_common::utils::error::Result<QueryResult>,
    projection: &str,
    left: &str,
    path: &str,
) -> Vec<String> {
    let searched = format!("SELECT {projection} WHERE {{ {left} ?a {path} ?b }}");
    let every_pair = format!(
        "SELECT {projection} WHERE {{ {left} {{ SELECT ?a ?b WHERE {{ ?a {path} ?b }} }} }}"
    );
    let got = run(execute, &searched);
    assert_eq!(
        got,
        run(execute, &every_pair),
        "`{searched}` against every pair"
    );
    got.unwrap_or_else(|error| panic!("`{searched}`: {error}"))
}

fn agree(db: &GrafeoDB, left: &str, path: &str) -> Vec<String> {
    agree_with(&|q| db.execute_sparql(q), "?a ?b", left, path)
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

const PATHS: &[&str] = &[
    "e:REL+",
    "e:REL*",
    "e:REL?",
    "e:REL/e:REL",
    "(e:REL|e:OTHER)+",
    "^e:REL+",
    "(e:REL|^e:REL)+",
    "e:REL/^e:REL",
    "(e:REL/e:REL)+",
];

#[test]
fn the_subject_bound_by_a_triple() {
    let db = database();
    for path in PATHS {
        agree(&db, "?a e:id \"s\" .", path);
        agree(&db, "?a e:id ?n .", path);
    }
    assert_eq!(
        agree(&db, "?a e:id \"b\" .", "e:REL+"),
        ["b d", "b e", "b f", "b g"]
    );
}

#[test]
fn the_object_bound_by_a_triple() {
    let db = database();
    for path in PATHS {
        agree(&db, "?b e:id \"d\" .", path);
        agree(&db, "?b e:id ?n .", path);
    }
    assert_eq!(
        agree(&db, "?b e:id \"d\" .", "e:REL+"),
        ["a d", "b d", "h d", "s d"]
    );
}

#[test]
fn both_ends_bound() {
    let db = database();
    for path in PATHS {
        agree(&db, "?a e:id \"s\" . ?b e:id \"f\" .", path);
        agree(&db, "?a e:id ?m . ?b e:id ?n .", path);
    }
    assert_eq!(
        agree(&db, "?a e:id \"s\" . ?b e:id \"f\" .", "e:REL+"),
        ["s f"]
    );
    assert_eq!(
        agree(&db, "?a e:id \"u\" . ?b e:id \"s\" .", "e:REL+"),
        Vec::<String>::new()
    );
    let ask = |query: &str| run(&|q| db.execute_sparql(q), query).unwrap();
    assert_eq!(
        ask("ASK { ?a e:id \"s\" . ?b e:id \"f\" . ?a (e:REL|^e:REL)+ ?b }"),
        ["true"]
    );
    assert_eq!(
        ask("ASK { ?a e:id \"s\" . ?b e:id \"v\" . ?a (e:REL|^e:REL)+ ?b }"),
        ["false"]
    );
}

/// VALUES binds the start, a term twice (two rows each) and one no triple
/// has; BIND binds it as a constant and as a computed IRI.
#[test]
fn ends_bound_by_values_and_bind() {
    let db = database();
    for path in PATHS {
        agree(&db, "VALUES ?a { e:s e:b e:s e:nowhere }", path);
        agree(&db, "VALUES ?b { e:d e:e e:d }", path);
        agree(
            &db,
            "VALUES (?a ?b) { (e:s e:f) (e:u e:s) (e:g e:g) (e:s e:s) }",
            path,
        );
        agree(&db, "BIND(e:b AS ?a)", path);
        agree(&db, "BIND(IRI(CONCAT(STR(e:), \"b\")) AS ?a)", path);
    }
    let rows = agree(&db, "VALUES ?a { e:b e:b }", "e:REL+");
    assert_eq!(rows.len(), 8, "each VALUES row searched: {rows:?}");
    assert_eq!(
        agree(&db, "BIND(IRI(CONCAT(STR(e:), \"b\")) AS ?a)", "e:REL+"),
        ["b d", "b e", "b f", "b g"],
        "a computed start IRI"
    );
}

/// `?x p+ ?x`: the nodes on a cycle.
#[test]
fn the_same_variable_at_both_ends() {
    let db = database();
    let execute = |q: &str| db.execute_sparql(q);
    for path in ["e:REL+", "(e:REL|^e:REL)+", "e:REL*"] {
        let searched = run(
            &execute,
            &format!("SELECT ?x WHERE {{ ?x e:id ?n . ?x {path} ?x }}"),
        );
        let every_pair = run(
            &execute,
            &format!("SELECT ?x WHERE {{ ?x e:id ?n . {{ SELECT ?x WHERE {{ ?x {path} ?x }} }} }}"),
        );
        assert_eq!(searched, every_pair, "{path}");
    }
    assert_eq!(
        run(&execute, "SELECT ?x WHERE { ?x e:id ?n . ?x e:REL+ ?x }").unwrap(),
        ["e", "f", "g"]
    );
}

/// `*` with both ends bound to one term: one row, also for a node with no
/// edge of the path (x has only OTHER).
#[test]
fn zero_length_with_both_ends_equal() {
    let db = database();
    for node in ["s", "u", "x"] {
        assert_eq!(
            agree(
                &db,
                &format!("?a e:id \"{node}\" . ?b e:id \"{node}\" ."),
                "e:REL*"
            ),
            [format!("{node} {node}")],
            "{node}"
        );
    }
    let ask = run(
        &|q| db.execute_sparql(q),
        "ASK { ?a e:id \"x\" . ?b e:id \"x\" . ?a e:REL* ?b }",
    )
    .unwrap();
    assert_eq!(ask, ["true"]);
}

/// An OPTIONAL left side holds ?b without always binding it: the pairs are
/// joined as they are without a search per row.
#[test]
fn an_optional_left_side_holding_the_free_end() {
    let db = database();
    for path in ["e:REL+", "e:REL*", "(e:REL|^e:REL)+"] {
        agree(&db, "?a e:id ?n . OPTIONAL { ?a e:OTHER ?b }", path);
        agree(&db, "?a e:id \"s\" . OPTIONAL { ?a e:REL ?b }", path);
    }
    assert_eq!(
        agree(&db, "?a e:id \"s\" . OPTIONAL { ?a e:OTHER ?b }", "e:REL*"),
        Vec::<String>::new(),
        "s reaches x only over OTHER"
    );
}

/// Named graphs: g1 and g2 both connect s to c. FROM over both, FROM NAMED
/// with GRAPH ?g, a free GRAPH ?g and one the left side binds; each against
/// every pair.
#[test]
fn named_graphs_and_from() {
    let db = named_graphs();
    let execute = |q: &str| db.execute_sparql(q);
    for (projection, left, path, dataset) in [
        (
            "?a ?b",
            "?a e:id \"s\" . ?b e:id \"c\" .",
            "e:p+",
            "FROM e:g1 FROM e:g2",
        ),
        ("?a ?b", "?a e:id \"s\" .", "e:p+", "FROM e:g1 FROM e:g2"),
        ("?a ?b", "?b e:id \"c\" .", "e:p*", "FROM e:g1"),
    ] {
        let searched = format!("SELECT {projection} {dataset} WHERE {{ {left} ?a {path} ?b }}");
        let every_pair = format!(
            "SELECT {projection} {dataset} WHERE {{ {left} {{ SELECT ?a ?b WHERE {{ ?a {path} ?b }} }} }}"
        );
        assert_eq!(
            run(&execute, &searched),
            run(&execute, &every_pair),
            "{searched}"
        );
    }
    let named = "SELECT ?g ?a ?b FROM NAMED e:g1 FROM NAMED e:g2 WHERE { GRAPH ?g { ?a e:id \"s\" . ?b e:id \"c\" . ?a e:p+ ?b } }";
    assert_eq!(
        run(&execute, named).unwrap(),
        ["g1 s c", "g2 s c"],
        "once per named graph"
    );
    let graph_free = "SELECT ?g ?b WHERE { ?a e:id \"s\" . GRAPH ?g { ?a e:p+ ?b } }";
    let graph_free_pairs = "SELECT ?g ?b WHERE { ?a e:id \"s\" . { SELECT ?g ?a ?b WHERE { GRAPH ?g { ?a e:p+ ?b } } } }";
    assert_eq!(run(&execute, graph_free), run(&execute, graph_free_pairs));
    assert_eq!(
        run(&execute, graph_free).unwrap(),
        ["g1 a", "g1 c", "g2 b", "g2 c"]
    );
    let graph_bound =
        "SELECT ?g ?b WHERE { VALUES ?g { e:g2 } ?a e:id \"s\" . GRAPH ?g { ?a e:p+ ?b } }";
    let graph_bound_pairs = "SELECT ?g ?b WHERE { VALUES ?g { e:g2 } ?a e:id \"s\" . { SELECT ?g ?a ?b WHERE { GRAPH ?g { ?a e:p+ ?b } } } }";
    assert_eq!(run(&execute, graph_bound), run(&execute, graph_bound_pairs));
    assert_eq!(run(&execute, graph_bound).unwrap(), ["g2 b", "g2 c"]);
}

/// The base graph plus g1 (s, a, c) and g2 (s, b, c), each with the ids of
/// s and c.
fn named_graphs() -> GrafeoDB {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ GRAPH e:g1 {{ e:s e:p e:a . e:a e:p e:c . e:s e:id \"s\" . e:c e:id \"c\" . }} GRAPH e:g2 {{ e:s e:p e:b . e:b e:p e:c . e:s e:id \"s\" . e:c e:id \"c\" . }} }}"
    ))
    .unwrap();
    db
}

/// FROM over two graphs is their merge, a set of triples: `e:s e:id "s"`,
/// in both, is one triple, so the pair (s, c) is one solution.
#[test]
#[ignore = "FROM over two graphs returns a triple that is in both twice (0.6.0 too)"]
fn from_two_graphs_counts_a_triple_in_both_once() {
    let db = named_graphs();
    let execute = |q: &str| db.execute_sparql(q);
    assert_eq!(
        run(
            &execute,
            "SELECT ?a FROM e:g1 FROM e:g2 WHERE { ?a e:id \"s\" }"
        )
        .unwrap(),
        ["s"]
    );
    assert_eq!(
        run(
            &execute,
            "SELECT ?a ?b FROM e:g1 FROM e:g2 WHERE { ?a e:id \"s\" . ?b e:id \"c\" . ?a e:p+ ?b }"
        )
        .unwrap(),
        ["s c"]
    );
}

/// `?` (ZeroOrOnePath, SPARQL 1.1 sec 18.5): the zero-length pairs of every
/// term of the graph, and a constant object reached in one hop or itself.
#[test]
#[ignore = "p? with a constant object returns other subjects and duplicates, and its zero-length pairs cover only the predicate's terms (0.6.0 too)"]
fn zero_or_one_with_a_constant_object() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ e:s e:REL e:a . e:a e:REL e:b . e:x e:OTHER e:y . }}"
    ))
    .unwrap();
    let execute = |q: &str| db.execute_sparql(q);
    assert_eq!(
        run(&execute, "SELECT ?a WHERE { ?a e:REL? e:b }").unwrap(),
        ["a", "b"]
    );
    assert_eq!(
        run(&execute, "SELECT ?a WHERE { ?a e:REL? e:nowhere }").unwrap(),
        ["nowhere"]
    );
    assert_eq!(
        run(&execute, "SELECT ?a ?b WHERE { ?a e:REL? ?b }").unwrap(),
        ["a a", "a b", "b b", "s a", "s s", "x x", "y y"]
    );
}

/// `GRAPH ?g { { SELECT ... } }`: the sub-select runs in each named graph
/// and binds ?g, as the same pattern without the sub-select does.
#[test]
#[ignore = "a sub-select inside GRAPH ?g reads every graph and leaves ?g unbound (0.6.0 too)"]
fn a_subselect_inside_a_graph_variable() {
    let db = named_graphs();
    let execute = |q: &str| db.execute_sparql(q);
    let direct = run(&execute, "SELECT ?g ?a ?b WHERE { GRAPH ?g { ?a e:p ?b } }");
    assert_eq!(
        run(
            &execute,
            "SELECT ?g ?a ?b WHERE { GRAPH ?g { { SELECT ?a ?b WHERE { ?a e:p ?b } } } }"
        ),
        direct
    );
    assert_eq!(direct.unwrap(), ["g1 a c", "g1 s a", "g2 b c", "g2 s b"]);
}

/// A zero-length path from a term no triple has: a constant start is the
/// path's own term (one solution, itself), while a start the rows bind
/// (VALUES) joins the path's solutions over the graph's terms, which do not
/// hold it: no solution (SPARQL 1.1 sec 18.6, ZeroLengthPath). A constant
/// end with the other end bound by the rows gives the constant's own
/// solution (sec 18.5), in either direction.
#[test]
fn zero_length_from_a_term_outside_the_graph() {
    let db = database();
    let execute = |q: &str| db.execute_sparql(q);
    for path in ["e:REL*", "e:REL?"] {
        assert_eq!(
            run(
                &execute,
                &format!("SELECT ?b WHERE {{ e:nowhere {path} ?b }}")
            )
            .unwrap(),
            ["nowhere"],
            "{path}: a constant start"
        );
        assert_eq!(
            agree(&db, "VALUES ?a { e:nowhere }", path),
            Vec::<String>::new(),
            "{path}: a start the rows bind"
        );
    }
    // A constant end with the other end bound by the rows: the constant's
    // own zero-length solution joins the row (`?` with a constant object is
    // wrong before this change, see `zero_or_one_with_a_constant_object`).
    for path in ["e:REL*"] {
        for (query, every_pair) in [
            (
                format!("SELECT ?b WHERE {{ VALUES ?b {{ e:nowhere }} e:nowhere {path} ?b }}"),
                format!(
                    "SELECT ?b WHERE {{ VALUES ?b {{ e:nowhere }} {{ SELECT ?b WHERE {{ e:nowhere {path} ?b }} }} }}"
                ),
            ),
            (
                format!("SELECT ?a WHERE {{ VALUES ?a {{ e:nowhere }} ?a {path} e:nowhere }}"),
                format!(
                    "SELECT ?a WHERE {{ VALUES ?a {{ e:nowhere }} {{ SELECT ?a WHERE {{ ?a {path} e:nowhere }} }} }}"
                ),
            ),
        ] {
            assert_eq!(run(&execute, &query), run(&execute, &every_pair), "{query}");
            assert_eq!(run(&execute, &query).unwrap(), ["nowhere"], "{query}");
        }
    }
}

/// A transaction's pending inserts and deletes: the writer's searches see
/// them, a reader's do not, and the rollback drops them.
#[test]
fn a_transactions_pending_changes() {
    let db = database();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!(
            "{PREFIX} DELETE DATA {{ e:a e:REL e:d . }} ; \
             INSERT DATA {{ e:u e:REL e:s . }}"
        ))
        .or_else(|_| {
            writer
                .execute_sparql(&format!("{PREFIX} DELETE DATA {{ e:a e:REL e:d . }}"))
                .and_then(|_| {
                    writer.execute_sparql(&format!("{PREFIX} INSERT DATA {{ e:u e:REL e:s . }}"))
                })
        })
        .unwrap();
    let in_writer = |session: &Session| {
        let execute = |q: &str| session.execute_sparql(q);
        for path in ["e:REL+", "(e:REL|^e:REL)+", "e:REL*"] {
            agree_with(&execute, "?a ?b", "?a e:id \"u\" .", path);
            agree_with(&execute, "?a ?b", "?b e:id \"d\" .", path);
        }
        agree_with(&execute, "?a ?b", "?a e:id \"u\" .", "e:REL+")
    };
    assert_eq!(
        in_writer(&writer),
        [
            "u a", "u b", "u d", "u e", "u f", "u g", "u h", "u s", "u v"
        ]
    );
    assert_eq!(in_writer(&reader), ["u v"]);
    writer.rollback().unwrap();
    assert_eq!(in_writer(&writer), ["u v"]);
}

/// PROFILE shows the per-row operator with its rows, and no closure over
/// every pair; the sub-select form shows the path on its own.
#[test]
fn profile_shows_the_search_per_row() {
    let db = database();
    let text = profile(
        &db,
        "SELECT ?a ?b WHERE { VALUES ?a { e:s e:b } ?a e:REL+ ?b }",
    );
    let per_row: Vec<&str> = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("RdfPropertyPathPerRow"))
        .collect();
    assert_eq!(per_row.len(), 1, "{text}");
    assert!(per_row[0].contains("rows=11"), "7 from s, 4 from b: {text}");
    let pairs = profile(
        &db,
        "SELECT ?a ?b WHERE { VALUES ?a { e:s e:b } { SELECT ?a ?b WHERE { ?a e:REL+ ?b } } }",
    );
    assert!(!pairs.contains("RdfPropertyPathPerRow"), "{pairs}");
}
