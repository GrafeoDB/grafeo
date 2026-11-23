//! A SPARQL group inside `GRAPH <iri> { }`, or after `FROM <iri>`, is estimated from the
//! statistics of the graphs it reads, so it is planned as the same group on the default graph:
//! it looks the other patterns up from the subjects it matches, and the rows it scans follow the
//! matching subjects, not the size of the graph. `GRAPH ?g` is ordered by the sum of the graphs,
//! a graph the dataset does not list or the store does not hold is estimated empty, and a write
//! to a named graph refreshes its estimates.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// The named graph the data goes into.
const GRAPH: &str = "urn:g:data";

/// Subjects added per `INSERT DATA`.
const CHUNK: usize = 250;

/// The issue's four shapes, each a group of patterns on `?s`, with the triple patterns each
/// holds.
const SHAPES: [(&str, &str, usize); 4] = [
    (
        "two patterns",
        "?s <urn:p:owner> <urn:e:alix> . ?s <urn:p:status> ?a .",
        2,
    ),
    (
        "four patterns",
        "?s <urn:p:owner> <urn:e:alix> . ?s <urn:p:status> ?a . ?s <urn:p:until> ?b . ?s <urn:p:by> ?c .",
        4,
    ),
    (
        "two OPTIONAL groups",
        "?s <urn:p:owner> <urn:e:alix> . OPTIONAL { ?s <urn:p:status> ?a } OPTIONAL { ?s <urn:p:until> ?b }",
        3,
    ),
    (
        "the type first",
        "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t ; <urn:p:owner> <urn:e:alix> ; <urn:p:status> ?a .",
        3,
    ),
];

/// Groups with a property path or a sub-select beside the patterns: each costs as much as on
/// the default graph, not as a path or an input over an empty graph.
const MIXED_SHAPES: [(&str, &str); 3] = [
    (
        "a path beside two patterns",
        "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:c:Thing> . ?s <urn:p:status> ?a . ?s <urn:p:knows>+ ?x .",
    ),
    (
        "a path from Alix's subjects",
        "?s <urn:p:owner> <urn:e:alix> . ?s <urn:p:knows>+ ?x .",
    ),
    (
        "a sub-select beside two patterns",
        "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:c:Thing> . ?s <urn:p:status> ?a . { SELECT ?s ?b WHERE { ?s <urn:p:by> ?b } }",
    ),
];

/// The triples of subject `i`: a type, a status, an until, a by, one it knows and an owner,
/// Alix's when `owned`, else one of its own.
fn subject(i: usize, owned: bool) -> String {
    let owner = if owned {
        "<urn:e:alix>".to_string()
    } else {
        format!("<urn:e:o{i}>")
    };
    format!(
        "<urn:e:s{i}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:c:Thing> ; \
         <urn:p:status> \"open\" ; <urn:p:until> {i} ; <urn:p:by> <urn:e:b{i}> ; \
         <urn:p:knows> <urn:e:b{i}> ; <urn:p:owner> {owner} ."
    )
}

/// Inserts subjects `from..to` into `graph` (the default graph when `None`), Alix's when
/// `owned`.
fn insert(db: &GrafeoDB, graph: Option<&str>, from: usize, to: usize, owned: bool) {
    for start in (from..to).step_by(CHUNK) {
        let triples: Vec<String> = (start..to.min(start + CHUNK))
            .map(|i| subject(i, owned))
            .collect();
        let body = triples.join("\n");
        let statement = match graph {
            Some(graph) => format!("INSERT DATA {{ GRAPH <{graph}> {{ {body} }} }}"),
            None => format!("INSERT DATA {{ {body} }}"),
        };
        db.execute_sparql(&statement).unwrap();
    }
}

/// `n` unrelated subjects and three of Alix's, in `graph` (the default graph when `None`).
fn store(graph: Option<&str>, n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    insert(&db, graph, 0, n, false);
    insert(&db, graph, n, n + 3, true);
    db
}

/// `body` as a SELECT, inside `GRAPH <graph> { }` when one is given.
fn select(graph: Option<&str>, body: &str) -> String {
    match graph {
        Some(graph) => format!("SELECT * WHERE {{ GRAPH <{graph}> {{ {body} }} }}"),
        None => format!("SELECT * WHERE {{ {body} }}"),
    }
}

/// The rows of `result`, sorted, and the rows it scanned.
fn answer(result: &QueryResult) -> (Vec<String>, u64) {
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    let scanned = result
        .rows_scanned
        .expect("every query reports rows_scanned");
    (rows, scanned)
}

/// The rows of `query` on `db`, sorted, and the rows it scanned.
fn run(db: &GrafeoDB, query: &str) -> (Vec<String>, u64) {
    answer(
        &db.execute_sparql(query)
            .unwrap_or_else(|error| panic!("{query}: {error}")),
    )
}

/// The profile of `query` on `db`: each operator with the rows it gave, its times left out.
fn profile(db: &GrafeoDB, query: &str) -> Vec<String> {
    let result = db
        .execute_sparql(&format!("EXPLAIN ANALYZE {query}"))
        .unwrap_or_else(|error| panic!("EXPLAIN ANALYZE {query}: {error}"));
    let text = match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{other:?}"),
    };
    text.lines()
        .filter(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with("Total") && !line.starts_with("Planning")
        })
        .map(|line| {
            line.split_whitespace()
                .filter(|word| !word.starts_with("time="))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Each shape inside `GRAPH <iri>` gives the rows the same shape gives on the default graph and
/// scans as many rows, at two sizes: the rows scanned follow Alix's three subjects, not the
/// size of the graph.
#[test]
fn a_join_in_a_named_graph_reads_the_matching_subjects() {
    let mut scanned_at = Vec::new();
    for n in [200, 2_000] {
        let default = store(None, n);
        let named = store(Some(GRAPH), n);
        let mut scanned = Vec::new();
        for (shape, body, _) in SHAPES {
            let in_default = run(&default, &select(None, body));
            let in_named = run(&named, &select(Some(GRAPH), body));
            assert_eq!(in_named.0, in_default.0, "{shape} at {n}: rows");
            assert_eq!(
                in_named.1, in_default.1,
                "{shape} at {n}: rows scanned in the named graph and in the default graph"
            );
            scanned.push(in_named.1);
        }
        scanned_at.push(scanned);
    }
    assert_eq!(
        scanned_at[0], scanned_at[1],
        "rows scanned at 200 and at 2,000 subjects"
    );
}

/// A group inside `GRAPH <iri>`, or after `FROM <iri>`, is planned as the same group on the
/// default graph: the same operators, giving the same rows each, with a property path or a
/// sub-select beside the patterns too.
#[test]
fn a_group_in_a_named_graph_is_planned_as_on_the_default_graph() {
    let n = 2_000;
    let default = store(None, n);
    let named = store(Some(GRAPH), n);
    let bodies = SHAPES
        .iter()
        .map(|(shape, body, _)| (*shape, *body))
        .chain(MIXED_SHAPES);
    for (shape, body) in bodies {
        let on_default = profile(&default, &select(None, body));
        assert_eq!(
            profile(&named, &select(Some(GRAPH), body)),
            on_default,
            "{shape} inside GRAPH <iri>"
        );
        assert_eq!(
            profile(
                &named,
                &format!("SELECT * FROM <{GRAPH}> WHERE {{ {body} }}")
            ),
            on_default,
            "{shape} after FROM <iri>"
        );
    }
}

/// `GRAPH ?g` over a large graph of unrelated subjects and a small one with Alix's: the rows
/// are those of a store holding the small graph alone, and the order starts at Alix's subjects,
/// estimated over the sum of the graphs. A pattern with a graph variable is not looked up from
/// the rows before it, so the rows scanned grow with the large graph; the first scan of the
/// plan does not.
#[test]
fn a_graph_variable_is_ordered_by_the_sum_of_the_graphs() {
    let small = |db: &GrafeoDB, n: usize| {
        insert(db, Some("urn:g:small"), n, n + 3, true);
        insert(db, Some("urn:g:small"), n + 3, n + 13, false);
    };
    for n in [200, 2_000] {
        let db = GrafeoDB::new_in_memory();
        insert(&db, Some("urn:g:large"), 0, n, false);
        small(&db, n);
        let alone = GrafeoDB::new_in_memory();
        small(&alone, n);
        for (shape, body, patterns) in SHAPES {
            let query = format!("SELECT * WHERE {{ GRAPH ?g {{ {body} }} }}");
            assert_eq!(
                run(&db, &query).0,
                run(&alone, &query).0,
                "{shape} at {n}: Alix's rows"
            );
            let scans = scan_rows(&profile(&db, &query));
            assert_eq!(
                scans.len(),
                patterns,
                "{shape} at {n}: one scan per pattern, each one that ran: {scans:?}"
            );
            assert_eq!(
                scans[0], 3,
                "{shape} at {n}: the first scan reads Alix's subjects"
            );
        }
    }
}

/// The rows each triple scan of a profile gave, in its order: the first is the first input of
/// the deepest join.
fn scan_rows(profile: &[String]) -> Vec<u64> {
    profile
        .iter()
        .filter(|line| line.starts_with("RdfTripleScan"))
        .map(|line| {
            line.split_whitespace()
                .find_map(|word| word.strip_prefix("rows="))
                .unwrap_or_else(|| panic!("rows in {line}"))
                .parse()
                .unwrap()
        })
        .collect()
}

/// A `FROM <iri>` makes its graph the default graph: each shape then scans as many rows as on
/// the store's default graph, at both sizes.
#[test]
fn a_from_clause_reads_by_the_graphs_it_merges() {
    let mut scanned_at = Vec::new();
    for n in [200, 2_000] {
        let default = store(None, n);
        let named = store(Some(GRAPH), n);
        let mut scanned = Vec::new();
        for (shape, body, _) in SHAPES {
            let in_default = run(&default, &select(None, body));
            let in_from = run(
                &named,
                &format!("SELECT * FROM <{GRAPH}> WHERE {{ {body} }}"),
            );
            assert_eq!(in_from, in_default, "{shape} at {n}: rows and rows scanned");
            scanned.push(in_from.1);
        }
        scanned_at.push(scanned);
    }
    assert_eq!(
        scanned_at[0], scanned_at[1],
        "rows scanned at 200 and at 2,000 subjects"
    );
}

/// Two `FROM` clauses merge their graphs: with Alix's subjects in one and the others in the
/// other, each shape gives the default graph's rows, and the rows it scans do not grow with
/// the other graph.
#[test]
fn two_from_clauses_read_by_the_sum_of_their_graphs() {
    let mut scanned_at = Vec::new();
    for n in [200, 2_000] {
        let default = store(None, n);
        let split = GrafeoDB::new_in_memory();
        insert(&split, Some("urn:g:rest"), 0, n, false);
        insert(&split, Some("urn:g:alix"), n, n + 3, true);
        let mut scanned = Vec::new();
        for (shape, body, _) in SHAPES {
            let query = format!("SELECT * FROM <urn:g:alix> FROM <urn:g:rest> WHERE {{ {body} }}");
            let (rows, rows_scanned) = run(&split, &query);
            assert_eq!(
                rows,
                run(&default, &select(None, body)).0,
                "{shape} at {n}: rows"
            );
            scanned.push(rows_scanned);
        }
        scanned_at.push(scanned);
    }
    assert_eq!(
        scanned_at[0], scanned_at[1],
        "rows scanned at 200 and at 2,000 subjects of the other graph"
    );
}

/// A `GRAPH <iri>` the dataset's `FROM NAMED` does not list is planned as a graph the store
/// does not hold: the rows and the rows scanned are those of a store without it, however large
/// it is.
#[test]
fn a_graph_the_dataset_does_not_list_is_planned_as_absent() {
    let with_it = store(Some(GRAPH), 2_000);
    insert(&with_it, Some("urn:g:small"), 0, 3, true);
    let without_it = GrafeoDB::new_in_memory();
    insert(&without_it, Some("urn:g:small"), 0, 3, true);
    let query = format!(
        "SELECT * FROM NAMED <urn:g:small> WHERE {{ GRAPH <urn:g:small> {{ ?s <urn:p:owner> <urn:e:alix> }} GRAPH <{GRAPH}> {{ ?s <urn:p:status> ?a }} }}"
    );
    assert_eq!(run(&with_it, &query), run(&without_it, &query));
    for (shape, body, _) in SHAPES {
        let query =
            format!("SELECT * FROM NAMED <urn:g:small> WHERE {{ GRAPH <{GRAPH}> {{ {body} }} }}");
        assert_eq!(
            run(&with_it, &query),
            (Vec::new(), 0),
            "{shape}: rows and rows scanned"
        );
    }
}

/// A `GRAPH <iri>` the store does not hold is estimated empty, not as the default graph: written
/// last in a chain of three with two patterns on the default graph, it is read first, so the
/// rows scanned do not grow with the default graph.
#[test]
fn a_graph_the_store_does_not_hold_is_estimated_empty() {
    let mut scanned_at = Vec::new();
    for n in [200, 2_000] {
        let db = store(None, n);
        let query = "SELECT * WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t . ?s <urn:p:until> ?b . GRAPH <urn:g:none> { ?s <urn:p:status> ?a } }";
        let (rows, scanned) = run(&db, query);
        assert_eq!(rows, Vec::<String>::new(), "at {n}: rows");
        scanned_at.push(scanned);
    }
    assert_eq!(
        scanned_at[0], scanned_at[1],
        "rows scanned at 200 and at 2,000 subjects of the default graph"
    );
}

/// A path that alone reads its named graph is costed from that graph's statistics: beside two
/// patterns on the default graph, the group is planned as the same group with the path on the
/// default graph, not with the path first as a path over a graph without statistics.
#[test]
fn a_path_alone_in_its_graph_is_costed_from_that_graph() {
    let n = 2_000;
    let db = store(None, n);
    let knows: Vec<String> = (0..n + 3)
        .map(|i| format!("<urn:e:s{i}> <urn:p:knows> <urn:e:b{i}> ."))
        .collect();
    for chunk in knows.chunks(CHUNK) {
        db.execute_sparql(&format!(
            "INSERT DATA {{ GRAPH <{GRAPH}> {{ {} }} }}",
            chunk.join(" ")
        ))
        .unwrap();
    }
    let patterns = "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:c:Thing> . ?s <urn:p:status> ?a .";
    let on_default = profile(
        &db,
        &format!("SELECT * WHERE {{ {patterns} ?s <urn:p:knows>+ ?x }}"),
    );
    assert_eq!(
        profile(
            &db,
            &format!("SELECT * WHERE {{ {patterns} GRAPH <{GRAPH}> {{ ?s <urn:p:knows>+ ?x }} }}")
        ),
        on_default
    );
}

/// With `FROM NAMED` alone the default graph is empty: a basic pattern gives no rows and scans
/// none, however large the store's default graph is.
#[test]
fn a_basic_pattern_under_from_named_alone_reads_nothing() {
    let db = store(None, 2_000);
    insert(&db, Some(GRAPH), 0, 3, true);
    for (shape, body, _) in SHAPES {
        let query = format!("SELECT * FROM NAMED <{GRAPH}> WHERE {{ {body} }}");
        assert_eq!(
            run(&db, &query),
            (Vec::new(), 0),
            "{shape}: rows and rows scanned"
        );
    }
}

/// An `INSERT DATA` into the named graph refreshes its estimates: a join first run while the
/// graph holds Alix's subjects alone scans, once the graph has grown, as many rows as on a
/// graph that was loaded whole.
#[test]
fn an_insert_into_the_named_graph_refreshes_its_estimates() {
    let n = 2_000;
    let grown = GrafeoDB::new_in_memory();
    insert(&grown, Some(GRAPH), n, n + 3, true);
    for (_, body, _) in SHAPES {
        grown.execute_sparql(&select(Some(GRAPH), body)).unwrap();
    }
    insert(&grown, Some(GRAPH), 0, n, false);
    let whole = store(Some(GRAPH), n);
    for (shape, body, _) in SHAPES {
        let query = select(Some(GRAPH), body);
        assert_eq!(
            run(&grown, &query),
            run(&whole, &query),
            "{shape}: rows and rows scanned after the graph grew"
        );
    }
}
