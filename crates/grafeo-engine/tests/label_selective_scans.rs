//! A node pattern with several labels scans the label with the fewest nodes
//! and checks the others per row, and a lookup through a property index checks
//! the pattern's labels on each node the index finds (#457). The rows are those
//! of the pattern as written: in GQL and Cypher, with and without an index,
//! inside and outside a transaction.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test label_selective_scans
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, ProjectionSpec, Session};

/// Nodes with `Graph` and not `Repository`.
const GRAPH_ONLY: usize = 1988;
/// Nodes with `Repository` and not `Graph`.
const REPOSITORY_ONLY: usize = 19;
/// Nodes with both labels: `r0`, `r1` and `r2`.
const BOTH: usize = 3;

/// A skewed graph: 1,988 `Graph` nodes (`g0`..`g1987`), three nodes with
/// `Graph` and `Repository` (`r0`..`r2`) and 19 `Repository` nodes (`p0`..`p18`),
/// each with an `id` and a number `n` (its index); every `r<i>` has a `HAS`
/// edge to `g<i>`. `Tag` and `Topic` have three nodes each, one of them shared
/// (`t0`). With `indexed`, a property index on `id`.
fn skewed(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("id").unwrap();
    }
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Graph {{id: 'g' + toString(i), n: i}})",
        GRAPH_ONLY - 1
    ))
    .unwrap();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Graph:Repository {{id: 'r' + toString(i), n: i}})",
        BOTH - 1
    ))
    .unwrap();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Repository {{id: 'p' + toString(i), n: i}})",
        REPOSITORY_ONLY - 1
    ))
    .unwrap();
    db.execute(
        "MATCH (r:Repository), (g:Graph) WHERE r.id STARTS WITH 'r' AND g.id = 'g' + toString(r.n) \
         INSERT (r)-[:HAS]->(g)",
    )
    .unwrap();
    db.execute("INSERT (:Tag:Topic {id: 't0'}), (:Tag {id: 't1'}), (:Tag {id: 't2'})")
        .unwrap();
    db.execute("INSERT (:Topic {id: 'u1'}), (:Topic {id: 'u2'})")
        .unwrap();
    db
}

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

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

/// One-column rows of strings, in the order given.
fn ids(items: &[&str]) -> Vec<Vec<Value>> {
    items.iter().map(|id| vec![Value::from(*id)]).collect()
}

fn list(items: &[&str]) -> Value {
    Value::List(
        items
            .iter()
            .map(|item| Value::from(*item))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// The text of a plan that `EXPLAIN` or `PROFILE` returns.
fn plan_text(session: &Session, language: Language, query: &str) -> String {
    run(session, language, query, &[])
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The lines of a plan that scan nodes.
fn scans(plan: &str) -> Vec<&str> {
    plan.lines()
        .map(str::trim)
        .filter(|line| line.starts_with("NodeScan") || line.starts_with("Scan"))
        .collect()
}

#[test]
fn the_label_with_the_fewest_nodes_is_scanned() {
    let db = skewed(false);
    let session = db.session();
    for language in LANGUAGES {
        for (pattern, scanned) in [
            ("(n:Graph:Repository)", "NodeScan (n:Repository)"),
            ("(n:Repository:Graph)", "NodeScan (n:Repository)"),
            ("(n:Graph:Repository {n: 1})", "NodeScan (n:Repository)"),
            // No node has `Missing`: it is the smallest, and the scan finds nothing.
            ("(n:Graph:Missing:Repository)", "NodeScan (n:Missing)"),
        ] {
            let plan = plan_text(
                &session,
                language,
                &format!("EXPLAIN MATCH {pattern} RETURN n.id"),
            );
            assert_eq!(scans(&plan), [scanned], "{language:?} {pattern}:\n{plan}");
        }
    }
}

/// `EXPLAIN` shows the label the planner scans wherever the pattern is: the
/// start of a path, an `OPTIONAL MATCH`, below a write.
#[test]
fn explain_shows_the_scanned_label_below_any_operator() {
    let db = skewed(false);
    let session = db.session();
    for language in LANGUAGES {
        for query in [
            "MATCH (n:Graph:Repository)-[:HAS]->(m) RETURN m.id",
            "MATCH (a:Graph {id: 'g88'}) OPTIONAL MATCH (n:Graph:Repository) RETURN a.id, n.id",
            "MATCH (n:Graph:Repository) SET n.touched = true",
            "MATCH (n:Graph:Repository) DETACH DELETE n",
        ] {
            let plan = plan_text(&session, language, &format!("EXPLAIN {query}"));
            let scanned = scans(&plan);
            assert!(
                scanned.contains(&"NodeScan (n:Repository)")
                    && !scanned.contains(&"NodeScan (n:Graph)"),
                "{language:?} `{query}`:\n{plan}"
            );
        }
        let profile = plan_text(
            &session,
            language,
            "PROFILE MATCH (a:Graph {id: 'g88'}) OPTIONAL MATCH (n:Graph:Repository) RETURN a.id, n.id",
        );
        assert!(
            scans(&profile)
                .iter()
                .any(|line| line.starts_with("Scan (n:Repository)")),
            "{language:?} the OPTIONAL MATCH runs as EXPLAIN shows it:\n{profile}"
        );
    }
}

#[test]
fn labels_with_as_many_nodes_keep_the_order_they_are_written_in() {
    let db = skewed(false);
    let session = db.session();
    for language in LANGUAGES {
        let tag_first = plan_text(
            &session,
            language,
            "EXPLAIN MATCH (n:Tag:Topic) RETURN n.id",
        );
        assert_eq!(
            scans(&tag_first),
            ["NodeScan (n:Tag)"],
            "{language:?}:\n{tag_first}"
        );
        let topic_first = plan_text(
            &session,
            language,
            "EXPLAIN MATCH (n:Topic:Tag) RETURN n.id",
        );
        assert_eq!(
            scans(&topic_first),
            ["NodeScan (n:Topic)"],
            "{language:?}:\n{topic_first}"
        );
    }
}

/// The scan reads the nodes of the small label only: `PROFILE` counts the rows
/// each operator returns, and the scan returns one row per node it reads.
#[test]
fn the_scan_reads_only_the_nodes_of_the_smallest_label() {
    let db = skewed(false);
    let session = db.session();
    for language in LANGUAGES {
        for pattern in ["(n:Graph:Repository)", "(n:Repository:Graph)"] {
            let profile = plan_text(
                &session,
                language,
                &format!("PROFILE MATCH {pattern} RETURN n.id"),
            );
            let scan = scans(&profile);
            assert_eq!(scan.len(), 1, "{language:?} {pattern}:\n{profile}");
            assert!(
                scan[0].starts_with("Scan (n:Repository)")
                    && scan[0].contains(&format!("rows={} ", REPOSITORY_ONLY + BOTH)),
                "{language:?} {pattern} reads the {} Repository nodes, not the {} Graph nodes:\n{profile}",
                REPOSITORY_ONLY + BOTH,
                GRAPH_ONLY + BOTH,
            );
        }
    }
}

/// The shapes of a node pattern with labels, each with the rows it must
/// return (sorted, unless the query orders them).
fn shapes() -> Vec<(&'static str, Vec<(&'static str, Value)>, Vec<Vec<Value>>)> {
    let some_ids = list(&["g19", "r0", "p3", "missing", "g19", "r2"]);
    let mut repository: Vec<String> = (0..BOTH).map(|i| format!("r{i}")).collect();
    repository.extend((0..REPOSITORY_ONLY).map(|i| format!("p{i}")));
    repository.sort();
    let repository: Vec<&str> = repository.iter().map(String::as_str).collect();
    vec![
        ("MATCH (n:Repository) RETURN n.id", vec![], ids(&repository)),
        (
            "MATCH (n:Graph:Repository) RETURN n.id",
            vec![],
            ids(&["r0", "r1", "r2"]),
        ),
        (
            "MATCH (n:Repository:Graph) RETURN n.id",
            vec![],
            ids(&["r0", "r1", "r2"]),
        ),
        (
            "MATCH (n:Repository:Graph) RETURN n.id ORDER BY n.id DESC",
            vec![],
            ids(&["r2", "r1", "r0"]),
        ),
        (
            "MATCH (n:Graph:Repository) RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(3)]],
        ),
        (
            "MATCH (n:Graph:Repository {id: 'r1'}) RETURN n.id",
            vec![],
            ids(&["r1"]),
        ),
        (
            "MATCH (n:Repository:Graph {id: 'r1'}) RETURN n.id",
            vec![],
            ids(&["r1"]),
        ),
        (
            "MATCH (n:Graph:Repository {id: 'g3'}) RETURN n.id",
            vec![],
            ids(&[]),
        ),
        (
            "MATCH (n:Repository:Graph {id: 'p3'}) RETURN n.id",
            vec![],
            ids(&[]),
        ),
        (
            "MATCH (n:Graph:Repository {n: 2}) RETURN n.id",
            vec![],
            ids(&["r2"]),
        ),
        (
            "MATCH (n:Graph) WHERE n:Repository RETURN n.id",
            vec![],
            ids(&["r0", "r1", "r2"]),
        ),
        (
            "MATCH (n:Graph {id: $x}) RETURN n.id",
            vec![("x", Value::from("r2"))],
            ids(&["r2"]),
        ),
        (
            "MATCH (n:Graph {id: $x}) RETURN n.id",
            vec![("x", Value::from("p3"))],
            ids(&[]),
        ),
        (
            "MATCH (n:Graph) WHERE n.id IN $ids RETURN n.id",
            vec![("ids", some_ids.clone())],
            ids(&["g19", "r0", "r2"]),
        ),
        (
            "MATCH (n:Graph:Repository) WHERE n.id IN $ids RETURN n.id",
            vec![("ids", some_ids.clone())],
            ids(&["r0", "r2"]),
        ),
        (
            "MATCH (n:Repository:Graph) WHERE n.id IN $ids RETURN n.id",
            vec![("ids", some_ids)],
            ids(&["r0", "r2"]),
        ),
        (
            "MATCH (n:Graph:Repository) WHERE n.n > 0 RETURN n.id",
            vec![],
            ids(&["r1", "r2"]),
        ),
        (
            "MATCH (n:Repository:Graph)-[:HAS]->(m) RETURN n.id, m.id",
            vec![],
            vec![
                vec![Value::from("r0"), Value::from("g0")],
                vec![Value::from("r1"), Value::from("g1")],
                vec![Value::from("r2"), Value::from("g2")],
            ],
        ),
        (
            "MATCH (a:Graph {id: 'g88'}) MATCH (n:Graph:Repository) RETURN a.id, n.id",
            vec![],
            vec![
                vec![Value::from("g88"), Value::from("r0")],
                vec![Value::from("g88"), Value::from("r1")],
                vec![Value::from("g88"), Value::from("r2")],
            ],
        ),
        (
            "MATCH (a:Graph {id: 'g88'}) OPTIONAL MATCH (n:Graph:Repository {id: 'p1'}) RETURN a.id, n.id",
            vec![],
            vec![vec![Value::from("g88"), Value::Null]],
        ),
        ("MATCH (n:Tag:Topic) RETURN n.id", vec![], ids(&["t0"])),
        ("MATCH (n:Topic:Tag) RETURN n.id", vec![], ids(&["t0"])),
        (
            "MATCH (n:Graph:Missing:Repository) RETURN n.id",
            vec![],
            ids(&[]),
        ),
        // A label under NOT, OR, XOR, CASE or `= true` is no requirement of the
        // node: the scan keeps the pattern's label.
        (
            "MATCH (n:Graph) WHERE NOT n:Repository RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(1988)]],
        ),
        (
            "MATCH (n:Graph) WHERE n:Repository OR n.n = 5 RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(4)]],
        ),
        (
            "MATCH (n:Graph) WHERE NOT (n:Repository AND n.n = 1) RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(1990)]],
        ),
        (
            "MATCH (n:Graph:Repository) WHERE NOT n:Tag RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(3)]],
        ),
        (
            "MATCH (n:Graph) WHERE n:Repository XOR n.n = 1 RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(3)]],
        ),
        (
            "MATCH (n:Graph) WHERE CASE WHEN n:Repository THEN true ELSE false END RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(3)]],
        ),
        (
            "MATCH (n:Graph) WHERE (n:Repository) = true RETURN count(n)",
            vec![],
            vec![vec![Value::Int64(3)]],
        ),
    ]
}

#[test]
fn every_shape_returns_the_rows_of_the_pattern() {
    for indexed in [false, true] {
        let db = skewed(indexed);
        let mut session = db.session();
        for in_transaction in [false, true] {
            if in_transaction {
                session.begin_transaction().unwrap();
            }
            for language in LANGUAGES {
                for (query, params, expected) in shapes() {
                    let actual = if query.contains("ORDER BY") {
                        run(&session, language, query, &params)
                    } else {
                        sorted(&session, language, query, &params)
                    };
                    assert_eq!(
                        actual, expected,
                        "{language:?} `{query}` {params:?} (index: {indexed}, transaction: {in_transaction})"
                    );
                }
            }
            if in_transaction {
                session.rollback().unwrap();
            }
        }
    }
}

/// In an open transaction, a node created with both labels is found, and a
/// node that lost one of them in it is not, whichever label is scanned; after
/// a rollback the rows are those from before.
#[test]
fn a_transaction_sees_the_labels_it_changed() {
    for indexed in [false, true] {
        let db = skewed(indexed);
        for language in LANGUAGES {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute("INSERT (:Graph:Repository {id: 'r3', n: 3})")
                .unwrap();
            // `r1` loses the label that is scanned, `r2` the one checked per row.
            session
                .execute_cypher("MATCH (n {id: 'r1'}) REMOVE n:Repository")
                .unwrap();
            session
                .execute_cypher("MATCH (n {id: 'r2'}) REMOVE n:Graph")
                .unwrap();
            let in_transaction: Vec<(&str, Vec<(&str, Value)>, Vec<Vec<Value>>)> = vec![
                (
                    "MATCH (n:Graph:Repository) RETURN n.id",
                    vec![],
                    ids(&["r0", "r3"]),
                ),
                (
                    "MATCH (n:Repository:Graph) RETURN n.id",
                    vec![],
                    ids(&["r0", "r3"]),
                ),
                (
                    "MATCH (n:Graph:Repository {id: 'r3'}) RETURN n.id",
                    vec![],
                    ids(&["r3"]),
                ),
                (
                    "MATCH (n:Repository:Graph {id: 'r1'}) RETURN n.id",
                    vec![],
                    ids(&[]),
                ),
                (
                    "MATCH (n:Graph:Repository {id: 'r2'}) RETURN n.id",
                    vec![],
                    ids(&[]),
                ),
                (
                    "MATCH (n:Graph {id: $x}) RETURN n.id",
                    vec![("x", Value::from("r1"))],
                    ids(&["r1"]),
                ),
                (
                    "MATCH (n:Graph {id: $x}) RETURN n.id",
                    vec![("x", Value::from("r2"))],
                    ids(&[]),
                ),
                (
                    "MATCH (n:Repository:Graph) WHERE n.id IN $ids RETURN n.id",
                    vec![("ids", list(&["r0", "r1", "r2", "r3"]))],
                    ids(&["r0", "r3"]),
                ),
                (
                    "MATCH (n:Graph) WHERE n.id IN $ids RETURN n.id",
                    vec![("ids", list(&["r0", "r1", "r2", "r3"]))],
                    ids(&["r0", "r1", "r3"]),
                ),
            ];
            for (query, params, expected) in in_transaction {
                assert_eq!(
                    sorted(&session, language, query, &params),
                    expected,
                    "{language:?} `{query}` {params:?} in the transaction (index: {indexed})"
                );
            }
            session.rollback().unwrap();
            for (query, params, expected) in shapes() {
                if query.contains("ORDER BY") {
                    continue;
                }
                assert_eq!(
                    sorted(&session, language, query, &params),
                    expected,
                    "{language:?} `{query}` {params:?} after the rollback (index: {indexed})"
                );
            }
        }
    }
}

/// A text index answers a filter for the label it was made for, so the scan
/// keeps the label as written when either label has the index: scanning the
/// other could turn a per-row check into an index search, or the reverse.
#[cfg(feature = "text-index")]
#[test]
fn a_label_with_a_text_index_for_the_filter_stays_scanned() {
    for indexed_label in ["Doc", "Draft"] {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "UNWIND range(0, 18) AS i INSERT (:Doc {id: 'd' + toString(i), body: 'graph notes ' + toString(i)})",
        )
        .unwrap();
        db.execute(
            "UNWIND range(0, 2) AS i INSERT (:Doc:Draft {id: 'x' + toString(i), body: 'draft graph'})",
        )
        .unwrap();
        db.execute("INSERT (:Doc:Draft {id: 'x3', body: 'nothing here'})")
            .unwrap();
        db.create_text_index(indexed_label, "body").unwrap();
        let session = db.session();
        for language in LANGUAGES {
            let context = format!("{language:?}, index on {indexed_label}");
            let query = "MATCH (n:Doc:Draft) WHERE text_match(n.body, 'draft') RETURN n.id";
            let plan = plan_text(&session, language, &format!("EXPLAIN {query}"));
            assert_eq!(scans(&plan), ["NodeScan (n:Doc)"], "{context}:\n{plan}");
            let expected = ids(&["x0", "x1", "x2"]);
            assert_eq!(
                sorted(&session, language, query, &[]),
                expected,
                "{context}"
            );
            assert_eq!(
                sorted(
                    &session,
                    language,
                    "MATCH (n:Draft:Doc) WHERE text_match(n.body, 'draft') RETURN n.id",
                    &[]
                ),
                expected,
                "{context}, with the labels the other way around"
            );
        }
    }
}

/// A vector index answers a filter for the label it was made for, so the scan
/// keeps the label as written when either label has the index: an index
/// search of the other label is approximate where the per-row check is exact.
#[cfg(feature = "vector-index")]
#[test]
fn a_label_with_a_vector_index_for_the_filter_stays_scanned() {
    for indexed_label in ["Item", "Pick"] {
        let db = GrafeoDB::new_in_memory();
        for i in 0..19_u8 {
            let labels: &[&str] = if i < 3 { &["Item", "Pick"] } else { &["Item"] };
            let node = db.create_node(labels).unwrap();
            db.set_node_property(node, "id", Value::from(format!("i{i}").as_str()))
                .unwrap();
            db.set_node_property(
                node,
                "vec",
                Value::Vector(vec![f32::from(i) / 19.0, 0.0, 0.0].into()),
            )
            .unwrap();
        }
        db.create_vector_index(
            indexed_label,
            "vec",
            Some(3),
            Some("euclidean"),
            None,
            None,
            None,
        )
        .unwrap();
        let session = db.session();
        for language in LANGUAGES {
            let context = format!("{language:?}, index on {indexed_label}");
            let query = "MATCH (n:Item:Pick) WHERE euclidean_distance(n.vec, [0.0, 0.0, 0.0]) < 0.1 RETURN n.id";
            let plan = plan_text(&session, language, &format!("EXPLAIN {query}"));
            assert_eq!(scans(&plan), ["NodeScan (n:Item)"], "{context}:\n{plan}");
            let expected = ids(&["i0", "i1"]);
            assert_eq!(
                sorted(&session, language, query, &[]),
                expected,
                "{context}"
            );
            assert_eq!(
                sorted(
                    &session,
                    language,
                    "MATCH (n:Pick:Item) WHERE euclidean_distance(n.vec, [0.0, 0.0, 0.0]) < 0.1 RETURN n.id",
                    &[]
                ),
                expected,
                "{context}, with the labels the other way around"
            );
        }
    }
}

/// A read of a past epoch finds the nodes that had the labels at the epoch,
/// whichever label has the fewest nodes now: `r1` lost `Repository` after it.
/// It scans the label as written, and `EXPLAIN` shows that.
#[cfg(feature = "temporal")]
#[test]
fn a_read_of_a_past_epoch_sees_the_labels_of_the_epoch() {
    let db = skewed(false);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.commit().unwrap();
    let before = db.current_epoch();
    session.begin_transaction().unwrap();
    session
        .execute_cypher("MATCH (n {id: 'r1'}) REMOVE n:Repository")
        .unwrap();
    session.commit().unwrap();
    for language in LANGUAGES {
        let query = "MATCH (n:Graph:Repository) RETURN n.id";
        assert_eq!(
            sorted(&session, language, query, &[]),
            ids(&["r0", "r2"]),
            "{language:?} now"
        );
        let now = plan_text(&session, language, &format!("EXPLAIN {query}"));
        assert_eq!(
            scans(&now),
            ["NodeScan (n:Repository)"],
            "{language:?} now:\n{now}"
        );
        session.set_viewing_epoch(before);
        let then = sorted(&session, language, query, &[]);
        let then_plan = plan_text(&session, language, &format!("EXPLAIN {query}"));
        session.clear_viewing_epoch();
        assert_eq!(
            then,
            ids(&["r0", "r1", "r2"]),
            "{language:?} at the epoch before"
        );
        assert_eq!(
            scans(&then_plan),
            ["NodeScan (n:Graph)"],
            "{language:?} at the epoch before:\n{then_plan}"
        );
    }
}

/// MERGE with several labels finds the node that has them all, whichever
/// order they are written in and whether or not an index finds it, and
/// creates one only when no node has them all. Without an index it reads the
/// label with the fewest nodes (see the `merge` operator's own tests).
#[test]
fn merge_with_several_labels_finds_the_node_that_has_them_all() {
    // With the five Tag and Topic nodes.
    let total = GRAPH_ONLY + BOTH + REPOSITORY_ONLY + 5;
    let count = |session: &Session, language: Language, query: &str| -> Value {
        run(session, language, query, &[])[0][0].clone()
    };
    for indexed in [false, true] {
        for language in LANGUAGES {
            for labels in ["Graph:Repository", "Repository:Graph"] {
                let db = skewed(indexed);
                let session = db.session();
                let context = format!("{language:?} {labels} (index: {indexed})");
                assert_eq!(
                    run(
                        &session,
                        language,
                        &format!("MERGE (n:{labels} {{id: 'r1'}}) RETURN n.id, n.n"),
                        &[]
                    ),
                    [vec![Value::from("r1"), Value::Int64(1)]],
                    "{context}: the existing node"
                );
                assert_eq!(
                    count(&session, language, "MATCH (n) RETURN count(n)"),
                    Value::Int64(i64::try_from(total).unwrap()),
                    "{context}: nothing created"
                );
                // `g3` has one of the labels only: MERGE creates a node with both.
                assert_eq!(
                    run(
                        &session,
                        language,
                        &format!("MERGE (n:{labels} {{id: 'g3'}}) RETURN n.id, n.n"),
                        &[]
                    ),
                    [vec![Value::from("g3"), Value::Null]],
                    "{context}: a new node"
                );
                assert_eq!(
                    sorted(
                        &session,
                        language,
                        "MATCH (n:Graph:Repository) RETURN n.id",
                        &[]
                    ),
                    ids(&["g3", "r0", "r1", "r2"]),
                    "{context}: the new node has both labels"
                );
                assert_eq!(
                    count(&session, language, "MATCH (n {id: 'g3'}) RETURN count(n)"),
                    Value::Int64(2),
                    "{context}: the node with one label stays"
                );
            }
        }
    }
}

/// On a compacted database the label with the fewest nodes is scanned too,
/// counted over the compacted base and the changes made since. The nodes with
/// both labels are made after `compact()` here; `compact_labels.rs` checks
/// the label choice over compacted nodes with several labels.
#[test]
fn a_compacted_database_scans_the_label_with_the_fewest_nodes() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Graph {{id: 'g' + toString(i), n: i}})",
        GRAPH_ONLY - 1
    ))
    .unwrap();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Repository {{id: 'p' + toString(i), n: i}})",
        REPOSITORY_ONLY - 1
    ))
    .unwrap();
    db.compact().unwrap();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Graph:Repository {{id: 'r' + toString(i), n: i}})",
        BOTH - 1
    ))
    .unwrap();
    db.execute("MATCH (n:Graph {id: 'g88'}) SET n:Repository")
        .unwrap();
    db.execute("MATCH (n:Repository {id: 'p0'}) DELETE n")
        .unwrap();
    db.execute("MATCH (n:Graph {id: 'g19'}) DELETE n").unwrap();
    let session = db.session();
    // Repository: 19 compacted, less p0, plus r0..r2 and g88.
    let repository = REPOSITORY_ONLY - 1 + BOTH + 1;
    for language in LANGUAGES {
        for pattern in ["(n:Graph:Repository)", "(n:Repository:Graph)"] {
            let plan = plan_text(
                &session,
                language,
                &format!("EXPLAIN MATCH {pattern} RETURN n.id"),
            );
            assert_eq!(
                scans(&plan),
                ["NodeScan (n:Repository)"],
                "{language:?} {pattern}:\n{plan}"
            );
            let profile = plan_text(
                &session,
                language,
                &format!("PROFILE MATCH {pattern} RETURN n.id"),
            );
            assert!(
                scans(&profile)[0].contains(&format!("rows={repository} ")),
                "{language:?} {pattern} reads the {repository} Repository nodes:\n{profile}"
            );
            assert_eq!(
                sorted(
                    &session,
                    language,
                    &format!("MATCH {pattern} RETURN n.id"),
                    &[]
                ),
                ids(&["g88", "r0", "r1", "r2"]),
                "{language:?} {pattern}"
            );
        }
    }
}

/// A projection finds the nodes of a label outside its spec that are in it.
/// The projection of `Person` holds Alix (`Person` and `Admin`), Gus and Mia;
/// Vincent, Jules and Butch are `Admin` only, outside it. In the projection
/// `Admin` has one node and `Person` three, so a pattern with both scans
/// `Admin`: the count of a label outside the spec is the exact number of its
/// nodes in the projection (in the graph `Admin` has four, more than `Person`).
#[test]
fn a_projection_finds_the_nodes_of_a_label_outside_its_spec() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person:Admin {name: 'Alix'}), (:Person {name: 'Gus'}), (:Person {name: 'Mia'})",
    )
    .unwrap();
    db.execute(
        "INSERT (:Admin {name: 'Vincent'}), (:Admin {name: 'Jules'}), (:Admin {name: 'Butch'})",
    )
    .unwrap();
    assert!(
        db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
            .unwrap()
    );
    let view =
        GrafeoDB::with_read_store(db.projection("people").unwrap(), Config::in_memory()).unwrap();
    let session = view.session();
    let alix = ids(&["Alix"]);
    let one = vec![vec![Value::Int64(1)]];
    for language in LANGUAGES {
        let where_label = sorted(
            &session,
            language,
            "MATCH (n:Person) WHERE n:Admin RETURN n.name",
            &[],
        );
        assert_eq!(where_label, alix, "{language:?} the label checked per row");
        for (query, expected) in [
            ("MATCH (n:Person:Admin) RETURN n.name", &where_label),
            ("MATCH (n:Admin:Person) RETURN n.name", &where_label),
            ("MATCH (n:Admin) RETURN n.name", &alix),
            ("MATCH (n:Person:Admin) RETURN count(n)", &one),
            ("MATCH (n:Admin) RETURN count(n)", &one),
            ("MATCH (n) WHERE 'Admin' IN labels(n) RETURN n.name", &alix),
            (
                "MATCH (n:Person) RETURN n.name",
                &ids(&["Alix", "Gus", "Mia"]),
            ),
        ] {
            assert_eq!(
                &sorted(&session, language, query, &[]),
                expected,
                "{language:?} `{query}` on the projection"
            );
        }
        for pattern in ["(n:Person:Admin)", "(n:Admin:Person)"] {
            let plan = plan_text(
                &session,
                language,
                &format!("EXPLAIN MATCH {pattern} RETURN n.name"),
            );
            assert_eq!(
                scans(&plan),
                ["NodeScan (n:Admin)"],
                "{language:?} {pattern} scans the label with the fewest nodes in the projection:\n{plan}"
            );
        }
    }
    assert_eq!(
        sorted(
            &db.session(),
            Language::Gql,
            "MATCH (n:Admin) RETURN count(n)",
            &[]
        ),
        vec![vec![Value::Int64(4)]],
        "the graph itself has four Admin nodes"
    );
}
