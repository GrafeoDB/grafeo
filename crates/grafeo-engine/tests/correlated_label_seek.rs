//! A `MATCH` after another one without a shared variable runs once per row of
//! the one before it (#455). The conjuncts of its `WHERE` that read only the
//! earlier variables filter those rows before the later scan, and a key from
//! the row finds the later nodes through a property index, also when the
//! pattern has more labels (`(t:Graph:TypeDefinition {filePath: f.path})`,
//! whose other labels are checks on the scanned node). The rows are those of
//! the plan as written: in GQL and Cypher, with and without an index, inside
//! and outside a transaction, and in subqueries.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test correlated_label_seek
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// A pattern with both labels, in either order.
const PATTERNS: [&str; 2] = ["t:Graph:TypeDefinition", "t:TypeDefinition:Graph"];

/// Which label gets 19 more nodes: none (four nodes each), `Graph` or
/// `TypeDefinition`, so that each of them is the one scanned.
const SIZES: [Option<&str>; 3] = [None, Some("Graph"), Some("TypeDefinition")];

fn run(
    session: &Session,
    language: Language,
    query: &str,
    params: &[(&str, Value)],
) -> Vec<Vec<Value>> {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The rows of `query` in a stable order (the query does not order them).
fn sorted(
    session: &Session,
    language: Language,
    query: &str,
    params: &[(&str, Value)],
) -> Vec<Vec<Value>> {
    let mut rows = run(session, language, query, params);
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

/// The text of the plan that `EXPLAIN` or `PROFILE` returns for `query`.
fn plan_text(
    session: &Session,
    language: Language,
    query: &str,
    params: &[(&str, Value)],
) -> String {
    run(session, language, query, params)
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A plan line without its indentation and its pushdown hint (`[...]`).
fn operator(line: &str) -> &str {
    let line = line.trim();
    match line.rfind(" [") {
        Some(hint) if line.ends_with(']') => &line[..hint],
        _ => line,
    }
}

/// Whether the plan has `upper` with `lower` as its input, hints aside.
fn sits_on(plan: &str, upper: &str, lower: &str) -> bool {
    let lines: Vec<&str> = plan.lines().collect();
    let indent = |line: &str| line.len() - line.trim_start().len();
    lines.windows(2).any(|pair| {
        operator(pair[0]) == upper
            && operator(pair[1]) == lower
            && indent(pair[1]) == indent(pair[0]) + 2
    })
}

/// The lines of a plan, trimmed.
fn lines(plan: &str) -> Vec<&str> {
    plan.lines().map(str::trim).collect()
}

fn strings(row: &[&str]) -> Vec<Value> {
    row.iter().map(|item| Value::from(*item)).collect()
}

fn table(rows: &[&[&str]]) -> Vec<Vec<Value>> {
    rows.iter().map(|row| strings(row)).collect()
}

fn list(items: &[&str]) -> Value {
    Value::List(strings(items).into())
}

// ---------------------------------------------------------------------------
// Conjuncts on the earlier variables filter the earlier rows
// ---------------------------------------------------------------------------

/// Functions Alix (`f1`), Gus (`f2`), Vincent (`f3`, not active), Jules (`f4`)
/// and Mia (`f5`) with `CALLS` edges, and `Model` nodes that point at them by
/// `source_identifier`: one for `f1`, `f3` and `f4`, two for `f2`, none for
/// `f5`. With `indexed`, a property index on `source_identifier`.
fn calls(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("source_identifier").unwrap();
    }
    db.execute(
        "INSERT (:Function {id: 'f1', name: 'Alix', active: true}), \
                (:Function {id: 'f2', name: 'Gus', active: true}), \
                (:Function {id: 'f3', name: 'Vincent', active: false}), \
                (:Function {id: 'f4', name: 'Jules', active: true}), \
                (:Function {id: 'f5', name: 'Mia', active: true})",
    )
    .unwrap();
    db.execute(
        "UNWIND [['f1', 'f2'], ['f1', 'f3'], ['f2', 'f4'], ['f4', 'f1'], ['f5', 'f2'], ['f2', 'f5']] AS pair \
         MATCH (a:Function {id: pair[0]}), (b:Function {id: pair[1]}) \
         INSERT (a)-[:CALLS]->(b)",
    )
    .unwrap();
    db.execute(
        "INSERT (:Model {source_identifier: 'f1', city: 'Amsterdam'}), \
                (:Model {source_identifier: 'f2', city: 'Berlin'}), \
                (:Model {source_identifier: 'f2', city: 'Paris'}), \
                (:Model {source_identifier: 'f3', city: 'Prague'}), \
                (:Model {source_identifier: 'f4', city: 'Barcelona'})",
    )
    .unwrap();
    db
}

/// The query of #455 as written there, with a `WHERE` after each `MATCH`
/// (Cypher only).
const CYPHER_CALLS: &str = "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
     WHERE graph_src.active = true AND graph_tgt.active = true \
     MATCH (model_src:Model), (model_tgt:Model) \
     WHERE model_src.source_identifier = graph_src.id \
       AND model_tgt.source_identifier = graph_tgt.id \
     RETURN graph_src.id, graph_tgt.id, model_src.city, model_tgt.city";

/// The same query with one `WHERE` after both `MATCH` clauses.
const ONE_WHERE_CALLS: &str = "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
     MATCH (model_src:Model), (model_tgt:Model) \
     WHERE graph_src.active = true AND graph_tgt.active = true \
       AND model_src.source_identifier = graph_src.id \
       AND model_tgt.source_identifier = graph_tgt.id \
     RETURN graph_src.id, graph_tgt.id, model_src.city, model_tgt.city";

/// `ONE_WHERE_CALLS` with a `WHERE` that cannot be split into conjuncts (the
/// added disjunct is never true): it stays above the last scan, as every
/// filter of a later `MATCH` did before.
const UNSPLIT_CALLS: &str = "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
     MATCH (model_src:Model), (model_tgt:Model) \
     WHERE (graph_src.active = true AND graph_tgt.active = true \
       AND model_src.source_identifier = graph_src.id \
       AND model_tgt.source_identifier = graph_tgt.id) OR model_tgt IS NULL \
     RETURN graph_src.id, graph_tgt.id, model_src.city, model_tgt.city";

/// The rows of the queries on `calls`, sorted.
fn called_models() -> Vec<Vec<Value>> {
    table(&[
        &["f1", "f2", "Amsterdam", "Berlin"],
        &["f1", "f2", "Amsterdam", "Paris"],
        &["f2", "f4", "Berlin", "Barcelona"],
        &["f2", "f4", "Paris", "Barcelona"],
        &["f4", "f1", "Barcelona", "Amsterdam"],
    ])
}

/// Each conjunct of the second `WHERE` filters the rows right above the scan
/// of the node it reads, so the rows of the first `MATCH` and of
/// `model_src` are cut down before `model_tgt` is scanned for each of them.
#[test]
fn each_conjunct_filters_right_above_the_scan_of_its_node() {
    let db = calls(false);
    let session = db.session();
    for (language, query) in [
        (Language::Cypher, CYPHER_CALLS),
        (Language::Cypher, ONE_WHERE_CALLS),
        (Language::Gql, ONE_WHERE_CALLS),
    ] {
        let plan = plan_text(&session, language, &format!("EXPLAIN {query}"), &[]);
        let context = format!("{language:?} `{query}`:\n{plan}");
        assert!(
            sits_on(
                &plan,
                "Filter (model_src.source_identifier Eq graph_src.id)",
                "NodeScan (model_src:Model)"
            ),
            "{context}"
        );
        assert!(
            sits_on(
                &plan,
                "Filter (model_tgt.source_identifier Eq graph_tgt.id)",
                "NodeScan (model_tgt:Model)"
            ),
            "{context}"
        );
    }
    // The conjuncts on the first `MATCH` go below both scans, each to the node
    // it reads, also when they are the first `MATCH`'s own `WHERE`.
    for (language, query) in [
        (Language::Cypher, CYPHER_CALLS),
        (Language::Cypher, ONE_WHERE_CALLS),
        (Language::Gql, ONE_WHERE_CALLS),
    ] {
        let plan = plan_text(&session, language, &format!("EXPLAIN {query}"), &[]);
        assert!(
            sits_on(
                &plan,
                "Filter (graph_tgt.active Eq true)",
                "Expand (graph_src)->[:CALLS]->(graph_tgt)"
            ) && sits_on(
                &plan,
                "Filter (graph_src.active Eq true)",
                "NodeScan (graph_src:*)"
            ),
            "{language:?}:\n{plan}"
        );
    }
}

/// The rows are those the filter above the last scan returned, in the same
/// order: the plan that cannot split its `WHERE` is the plan from before.
#[test]
fn the_rows_are_those_of_the_filter_above_the_last_scan() {
    for indexed in [false, true] {
        let db = calls(indexed);
        let session = db.session();
        let before = run(&session, Language::Gql, UNSPLIT_CALLS, &[]);
        for (language, query) in [
            (Language::Cypher, CYPHER_CALLS),
            (Language::Cypher, ONE_WHERE_CALLS),
            (Language::Gql, ONE_WHERE_CALLS),
        ] {
            let context = format!("{language:?} `{query}` (index: {indexed})");
            if !indexed {
                assert_eq!(run(&session, language, query, &[]), before, "{context}");
            }
            assert_eq!(
                sorted(&session, language, query, &[]),
                called_models(),
                "{context}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// A key from the row seeks through the label checks
// ---------------------------------------------------------------------------

/// Files `amsterdam.rs`, `berlin.rs` and `paris.rs`, and type definitions with
/// a `filePath`: Alix and Mia (`Graph` and `TypeDefinition`) and Vincent
/// (`TypeDefinition` only) in `amsterdam.rs`, Gus (both labels) in
/// `berlin.rs`, Jules (`Graph` only) in `paris.rs`. Each label has four nodes;
/// `more` gives one of them 19 more, in `prague.rs`, which no file has. With
/// `indexed`, a property index on `filePath`.
fn files(indexed: bool, more: Option<&str>) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("filePath").unwrap();
    }
    db.execute("INSERT (:File {path: 'amsterdam.rs'}), (:File {path: 'berlin.rs'}), (:File {path: 'paris.rs'})")
        .unwrap();
    db.execute(
        "INSERT (:Graph:TypeDefinition {name: 'Alix', filePath: 'amsterdam.rs'}), \
                (:TypeDefinition {name: 'Vincent', filePath: 'amsterdam.rs'}), \
                (:Graph:TypeDefinition {name: 'Gus', filePath: 'berlin.rs'}), \
                (:Graph {name: 'Jules', filePath: 'paris.rs'}), \
                (:Graph:TypeDefinition {name: 'Mia', filePath: 'amsterdam.rs'})",
    )
    .unwrap();
    if let Some(label) = more {
        db.execute(&format!(
            "UNWIND range(1, 19) AS i INSERT (:{label} {{name: 'Marcellus', filePath: 'prague.rs'}})"
        ))
        .unwrap();
    }
    db
}

/// The type definitions with both labels per file, sorted.
fn definitions() -> Vec<Vec<Value>> {
    table(&[
        &["amsterdam.rs", "Alix"],
        &["amsterdam.rs", "Mia"],
        &["berlin.rs", "Gus"],
    ])
}

/// The paths the `UNWIND` queries look up: every file's, and one only nodes
/// with a single label have.
fn paths() -> Vec<(&'static str, Value)> {
    vec![(
        "paths",
        list(&["amsterdam.rs", "berlin.rs", "paris.rs", "prague.rs"]),
    )]
}

/// The queries whose second pattern, `labels`, takes its key from the row.
fn keyed(labels: &str) -> Vec<(Language, String)> {
    vec![
        (
            Language::Gql,
            format!("MATCH (f:File) MATCH ({labels} {{filePath: f.path}}) RETURN f.path, t.name"),
        ),
        (
            Language::Gql,
            format!(
                "MATCH (f:File) MATCH ({labels}) WHERE t.filePath = f.path RETURN f.path, t.name"
            ),
        ),
        (
            Language::Gql,
            format!(
                "UNWIND $paths AS path MATCH ({labels} {{filePath: path}}) RETURN path, t.name"
            ),
        ),
        (
            Language::Cypher,
            format!(
                "MATCH (f:File) WITH f LIMIT 200 MATCH ({labels} {{filePath: f.path}}) RETURN f.path, t.name"
            ),
        ),
        (
            Language::Cypher,
            format!(
                "MATCH (f:File) MATCH ({labels}) WHERE t.filePath = f.path RETURN f.path, t.name"
            ),
        ),
        (
            Language::Cypher,
            format!(
                "UNWIND $paths AS path MATCH ({labels} {{filePath: path}}) RETURN path, t.name"
            ),
        ),
    ]
}

/// The label the planner scans: the one with fewer nodes, or the first one
/// written when both have as many.
fn scanned(labels: &str, more: Option<&str>) -> String {
    let written = labels.trim_start_matches("t:").split(':').next().unwrap();
    let label = match more {
        None => written,
        Some("Graph") => "TypeDefinition",
        Some(_) => "Graph",
    };
    format!("NodeScan (t:{label})")
}

/// With an index on `filePath`, `EXPLAIN` shows a lookup for the key from the
/// row, after the label with the fewest nodes was chosen for the scan, in
/// either label order; the rows are those without the index: only nodes with
/// both labels. (In the `MATCH` forms the key filter moves below the label
/// check; `the_seek_takes_over_the_label_checks` has the seek through it.)
#[test]
fn a_key_from_the_row_finds_the_nodes_with_both_labels() {
    for more in SIZES {
        let (sought, scanned_db) = (files(true, more), files(false, more));
        let (with_index, without_index) = (sought.session(), scanned_db.session());
        for labels in PATTERNS {
            for (language, query) in keyed(labels) {
                let context = format!("{language:?} `{query}` (more {more:?})");
                let plan = plan_text(&with_index, language, &format!("EXPLAIN {query}"), &paths());
                assert!(plan.contains("[index: filePath]"), "{context}:\n{plan}");
                assert!(
                    lines(&plan).contains(&scanned(labels, more).as_str()),
                    "{context}:\n{plan}"
                );
                assert_eq!(
                    sorted(&with_index, language, &query, &paths()),
                    definitions(),
                    "{context}, with the index"
                );
                assert_eq!(
                    sorted(&without_index, language, &query, &paths()),
                    definitions(),
                    "{context}, without the index"
                );
            }
        }
    }
}

/// The queries whose key filter stays above the label check, because the
/// optimizer does not see the variable of the key bound below it: an `UNWIND`
/// variable, or the row a `CALL` subquery imports. Each with the key as
/// `EXPLAIN` shows it.
fn keyed_over_checks(labels: &str) -> Vec<(Language, String, &'static str)> {
    vec![
        (
            Language::Gql,
            format!(
                "UNWIND $paths AS path MATCH ({labels} {{filePath: path}}) RETURN path, t.name"
            ),
            "path",
        ),
        (
            Language::Cypher,
            format!(
                "UNWIND $paths AS path MATCH ({labels} {{filePath: path}}) RETURN path, t.name"
            ),
            "path",
        ),
        (
            Language::Gql,
            format!(
                "MATCH (f:File) CALL (f) {{ MATCH ({labels} {{filePath: f.path}}) \
                 RETURN t.name AS name }} RETURN f.path, name"
            ),
            "f.path",
        ),
        (
            Language::Cypher,
            format!(
                "MATCH (f:File) CALL {{ WITH f MATCH ({labels} {{filePath: f.path}}) \
                 RETURN t.name AS name }} RETURN f.path, name"
            ),
            "f.path",
        ),
    ]
}

/// The seek takes over the check of the other label: `EXPLAIN` shows the
/// lookup on the key filter right above that check, which sits on the scan of
/// the label with the fewest nodes. The rows are those without the index.
#[test]
fn the_seek_takes_over_the_label_checks() {
    for more in SIZES {
        let (sought, scanned_db) = (files(true, more), files(false, more));
        let (with_index, without_index) = (sought.session(), scanned_db.session());
        for labels in PATTERNS {
            let scan = scanned(labels, more);
            let other = if scan.ends_with("(t:Graph)") {
                "TypeDefinition"
            } else {
                "Graph"
            };
            let check = format!("Filter (hasLabel(t, \"{other}\"))");
            for (language, query, key) in keyed_over_checks(labels) {
                let context = format!("{language:?} `{query}` (more {more:?})");
                let plan = plan_text(&with_index, language, &format!("EXPLAIN {query}"), &paths());
                let lookup = format!("Filter (t.filePath Eq {key}) [index: filePath]");
                assert!(
                    lines(&plan).contains(&lookup.as_str())
                        && lines(&plan).contains(&check.as_str())
                        && sits_on(&plan, &format!("Filter (t.filePath Eq {key})"), &check)
                        && sits_on(&plan, &check, &scan),
                    "{context}:\n{plan}"
                );
                assert_eq!(
                    sorted(&with_index, language, &query, &paths()),
                    definitions(),
                    "{context}, with the index"
                );
                assert_eq!(
                    sorted(&without_index, language, &query, &paths()),
                    definitions(),
                    "{context}, without the index"
                );
            }
        }
    }
}

/// A check that is not a label, the `WHERE` of an element pattern, is taken
/// over the same way.
#[test]
fn the_seek_takes_over_an_element_where() {
    let (sought, scanned_db) = (files(true, None), files(false, None));
    let query = "UNWIND $paths AS path \
                 MATCH (t:Graph:TypeDefinition WHERE t.name <> 'Alix') WHERE t.filePath = path \
                 RETURN path, t.name";
    let plan = plan_text(
        &sought.session(),
        Language::Gql,
        &format!("EXPLAIN {query}"),
        &paths(),
    );
    assert!(
        lines(&plan).contains(&"Filter (t.filePath Eq path) [index: filePath]")
            && lines(&plan).contains(&"Filter (t.name Ne \"Alix\")")
            && sits_on(
                &plan,
                "Filter (t.filePath Eq path)",
                "Filter (hasLabel(t, \"TypeDefinition\"))"
            )
            && sits_on(
                &plan,
                "Filter (hasLabel(t, \"TypeDefinition\"))",
                "Filter (t.name Ne \"Alix\")"
            )
            && sits_on(&plan, "Filter (t.name Ne \"Alix\")", "NodeScan (t:Graph)"),
        "{plan}"
    );
    let expected = table(&[&["amsterdam.rs", "Mia"], &["berlin.rs", "Gus"]]);
    for db in [&sought, &scanned_db] {
        assert_eq!(
            sorted(&db.session(), Language::Gql, query, &paths()),
            expected
        );
    }
}

/// `PROFILE` shows the seek that runs, and the checks it took over.
#[test]
fn profile_shows_the_seek_and_its_checks() {
    let db = files(true, None);
    let session = db.session();
    for labels in PATTERNS {
        for (language, query, _) in keyed_over_checks(labels) {
            let profile = plan_text(&session, language, &format!("PROFILE {query}"), &paths());
            let operators = lines(&profile);
            assert!(
                operators
                    .iter()
                    .any(|line| line.starts_with("NodeSeek (t:")),
                "{language:?} `{query}`:\n{profile}"
            );
            assert!(
                !operators
                    .iter()
                    .any(|line| ["Scan (t:", "NodeScan (t:", "NestedLoopJoin (t:"]
                        .iter()
                        .any(|scan| line.starts_with(scan))),
                "{language:?} `{query}`:\n{profile}"
            );
            assert!(
                operators
                    .iter()
                    .any(|line| line.starts_with("Filter (hasLabel(t, ")),
                "{language:?} `{query}`:\n{profile}"
            );
        }
    }
}

/// A seek looks a key up when its input row arrives, so after a write in its
/// input it would miss what the input writes for later rows: here the nodes
/// for the first rows' keys are written by rows of the input's second chunk
/// (a chunk holds 2048 rows). Such a scan reads its whole input first
/// instead, with or without more labels, and `EXPLAIN` shows no lookup.
#[test]
fn no_seek_after_a_write_in_the_input() {
    for (language, write) in [
        (
            Language::Cypher,
            "UNWIND range(1, 2100) AS i CREATE (n:N:M {k: i}) WITH i WHERE i <= 3",
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 2100) INSERT (n:N:M {k: i}) WITH i WHERE i <= 3",
        ),
    ] {
        for pattern in ["t:N", "t:N:M", "t:M:N"] {
            let db = GrafeoDB::new_in_memory();
            db.create_property_index("k").unwrap();
            let session = db.session();
            let query =
                format!("{write} MATCH ({pattern} {{k: 2101 - i}}) RETURN i, t.k ORDER BY i");
            let plan = plan_text(&session, language, &format!("EXPLAIN {query}"), &[]);
            assert!(!plan.contains("[index"), "{language:?} `{query}`:\n{plan}");
            assert_eq!(
                run(&session, language, &query, &[]),
                [
                    [Value::Int64(1), Value::Int64(2100)],
                    [Value::Int64(2), Value::Int64(2099)],
                    [Value::Int64(3), Value::Int64(2098)],
                ],
                "{language:?} `{query}`"
            );
        }
    }
}

/// In an open transaction, a node created with both labels is found through
/// the index, and nodes that lost one of the labels in it are not; one that
/// got the missing label is. After a rollback the rows are those from before.
#[test]
fn a_transaction_sees_the_labels_it_changed() {
    let changed = table(&[
        &["amsterdam.rs", "Alix"],
        &["amsterdam.rs", "Vincent"],
        &["paris.rs", "Butch"],
    ]);
    for indexed in [true, false] {
        let db = files(indexed, None);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Graph:TypeDefinition {name: 'Butch', filePath: 'paris.rs'})")
            .unwrap();
        session
            .execute_cypher("MATCH (t {name: 'Mia'}) REMOVE t:Graph")
            .unwrap();
        session
            .execute_cypher("MATCH (t {name: 'Gus'}) REMOVE t:TypeDefinition")
            .unwrap();
        session
            .execute_cypher("MATCH (t {name: 'Vincent'}) SET t:Graph")
            .unwrap();
        for labels in PATTERNS {
            for (language, query) in keyed(labels) {
                assert_eq!(
                    sorted(&session, language, &query, &paths()),
                    changed,
                    "{language:?} `{query}` in the transaction (index: {indexed})"
                );
            }
        }
        session.rollback().unwrap();
        for labels in PATTERNS {
            for (language, query) in keyed(labels) {
                assert_eq!(
                    sorted(&session, language, &query, &paths()),
                    definitions(),
                    "{language:?} `{query}` after the rollback (index: {indexed})"
                );
            }
        }
    }
}

/// Subqueries that match the pattern per outer row keep their rows: one that
/// finds new nodes for the key from the row, and one whose outer row binds
/// the node already, which the seek checks instead of looking it up (its
/// labels included).
#[test]
fn subqueries_keep_their_rows() {
    for more in SIZES {
        for indexed in [false, true] {
            let db = files(indexed, more);
            let session = db.session();
            for labels in PATTERNS {
                let cases: Vec<(String, Vec<Vec<Value>>)> = vec![
                    (
                        format!(
                            "MATCH (f:File), (t) WHERE EXISTS {{ MATCH ({labels} {{filePath: f.path}}) }} \
                             RETURN f.path, t.name"
                        ),
                        definitions(),
                    ),
                    (
                        format!(
                            "MATCH (f:File), (t) CALL {{ WITH f, t MATCH ({labels} {{filePath: f.path}}) \
                             RETURN t.name AS name }} RETURN f.path, name"
                        ),
                        definitions(),
                    ),
                    (
                        format!(
                            "MATCH (f:File) CALL {{ WITH f MATCH ({labels} {{filePath: f.path}}) \
                             RETURN t.name AS name }} RETURN f.path, name"
                        ),
                        definitions(),
                    ),
                    (
                        format!(
                            "MATCH (f:File) WHERE EXISTS {{ MATCH ({labels} {{filePath: f.path}}) }} \
                             RETURN f.path"
                        ),
                        table(&[&["amsterdam.rs"], &["berlin.rs"]]),
                    ),
                    (
                        format!(
                            "MATCH (f:File) RETURN f.path, \
                             COUNT {{ MATCH ({labels} {{filePath: f.path}}) }} AS found"
                        ),
                        vec![
                            vec![Value::from("amsterdam.rs"), Value::Int64(2)],
                            vec![Value::from("berlin.rs"), Value::Int64(1)],
                            vec![Value::from("paris.rs"), Value::Int64(0)],
                        ],
                    ),
                ];
                for language in LANGUAGES {
                    for (query, expected) in &cases {
                        assert_eq!(
                            &sorted(&session, language, query, &[]),
                            expected,
                            "{language:?} `{query}` (index: {indexed}, more {more:?})"
                        );
                    }
                }
            }
        }
    }
}

/// A conjunct on the first `MATCH` alone filters its rows before the second
/// scan, in GQL and in Cypher; the rows and their order stay those of the
/// filter above the second scan.
#[test]
fn a_conjunct_on_the_first_match_filters_it_before_the_second() {
    let expected = table(&[
        &["amsterdam.rs", "Alix"],
        &["amsterdam.rs", "Mia"],
        &["amsterdam.rs", "Vincent"],
        &["berlin.rs", "Gus"],
    ]);
    let split = "MATCH (f:File) MATCH (t:TypeDefinition) \
                 WHERE f.path <> 'paris.rs' AND t.filePath = f.path RETURN f.path, t.name";
    let unsplit = "MATCH (f:File) MATCH (t:TypeDefinition) \
                   WHERE (f.path <> 'paris.rs' AND t.filePath = f.path) OR t IS NULL \
                   RETURN f.path, t.name";
    let cypher_only = "MATCH (f:File) WHERE f.path <> 'paris.rs' \
                       MATCH (t:TypeDefinition) WHERE t.filePath = f.path RETURN f.path, t.name";
    for indexed in [false, true] {
        let db = files(indexed, None);
        let session = db.session();
        let before = run(&session, Language::Gql, unsplit, &[]);
        for (language, query) in [
            (Language::Gql, split),
            (Language::Cypher, split),
            (Language::Cypher, cypher_only),
        ] {
            let context = format!("{language:?} `{query}` (index: {indexed})");
            let plan = plan_text(&session, language, &format!("EXPLAIN {query}"), &[]);
            assert!(
                sits_on(
                    &plan,
                    "Filter (f.path Ne \"paris.rs\")",
                    "NodeScan (f:File)"
                ) && sits_on(
                    &plan,
                    "Filter (t.filePath Eq f.path)",
                    "NodeScan (t:TypeDefinition)"
                ),
                "{context}:\n{plan}"
            );
            if indexed {
                assert!(
                    lines(&plan).contains(&"Filter (t.filePath Eq f.path) [index: filePath]"),
                    "{context}:\n{plan}"
                );
            } else {
                assert_eq!(run(&session, language, query, &[]), before, "{context}");
            }
            assert_eq!(
                sorted(&session, language, query, &[]),
                expected,
                "{context}"
            );
        }
    }
}
