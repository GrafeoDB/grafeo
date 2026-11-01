//! OPTIONAL MATCH controls on the shared adversarial topology (#678, item 8).

#![cfg(feature = "gql")]

#[path = "support/adversarial_paths.rs"]
mod adversarial_paths;

use std::collections::{BTreeMap, BTreeSet};

use adversarial_paths::{EDGES, add_parallel_rel, adversarial_path_graph, node_ids};
use grafeo_common::types::Value;
use grafeo_engine::transaction::IsolationLevel;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    #[cfg(feature = "cypher")]
    Cypher,
}

const LANGUAGES: &[Language] = &[
    Language::Gql,
    #[cfg(feature = "cypher")]
    Language::Cypher,
];

fn rows(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        #[cfg(feature = "cypher")]
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?}: {query}: {error}"))
        .rows()
        .to_vec()
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value.to_string()),
        other => panic!("expected text or null, got {other:?}"),
    }
}

fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::Null => None,
        Value::Int64(value) => Some(*value),
        other => panic!("expected integer or null, got {other:?}"),
    }
}

fn edge_ids(value: &Value) -> Vec<i64> {
    let Value::List(values) = value else {
        panic!("expected edge-id list, got {value:?}");
    };
    values
        .iter()
        .map(|value| integer(value).expect("an edge id"))
        .collect()
}

/// Resolve edge ids independently of the shortest-path result, using the fixture table.
fn edge_catalog(db: &GrafeoDB) -> BTreeMap<i64, (String, String)> {
    rows(
        &db.session(),
        Language::Gql,
        "MATCH (a:Node)-[e]->(b:Node) RETURN id(e), a.id, b.id, e.slot",
    )
    .into_iter()
    .map(|row| {
        let slot = usize::try_from(integer(&row[3]).unwrap()).unwrap();
        let (from, to, _) = if slot == EDGES.len() {
            ("s", "a", "REL")
        } else {
            EDGES[slot]
        };
        assert_eq!(text(&row[1]).as_deref(), Some(from));
        assert_eq!(text(&row[2]).as_deref(), Some(to));
        (
            integer(&row[0]).unwrap(),
            (from.to_string(), to.to_string()),
        )
    })
    .collect()
}

#[test]
fn optional_shortest_preserves_free_targets_and_all_parallel_paths() {
    for indexed in [false, true] {
        for &language in LANGUAGES {
            let db = adversarial_path_graph(indexed);
            add_parallel_rel(&db);
            let catalog = edge_catalog(&db);
            assert_eq!(catalog.len(), EDGES.len() + 1);
            for all in [false, true] {
                let query = match language {
                    Language::Gql => format!(
                        "UNWIND [1, 0] AS budget OPTIONAL MATCH p = {} \
                         (s:Node {{id: 's'}})-[r:REL WHERE r.cost <= budget]->{{1,2}}(b:Node) \
                         RETURN budget, b.id, length(p), [e IN edges(p) | id(e)]",
                        if all { "ALL SHORTEST" } else { "ANY SHORTEST" },
                    ),
                    #[cfg(feature = "cypher")]
                    Language::Cypher => format!(
                        "UNWIND [1, 0] AS budget OPTIONAL MATCH p = {}(\
                         (s:Node {{id: 's'}})-[:REL*1..2 {{cost: budget}}]->(b:Node)) \
                         RETURN budget, b.id, length(p), [e IN edges(p) | id(e)]",
                        if all {
                            "allShortestPaths"
                        } else {
                            "shortestPath"
                        },
                    ),
                };
                let actual = rows(&db.session(), language, &query);
                let mut counts = BTreeMap::<String, usize>::new();
                let mut distinct_paths = BTreeSet::new();
                let mut fallbacks = 0;
                for row in actual {
                    if integer(&row[0]) == Some(0) {
                        assert!(row[1..].iter().all(|value| matches!(value, Value::Null)));
                        fallbacks += 1;
                        continue;
                    }
                    assert_eq!(integer(&row[0]), Some(1));
                    let target = text(&row[1]).unwrap();
                    let hops = usize::try_from(integer(&row[2]).unwrap()).unwrap();
                    let ids = edge_ids(&row[3]);
                    assert_eq!(ids.len(), hops);
                    assert_eq!(
                        hops,
                        if matches!(target.as_str(), "a" | "b") {
                            1
                        } else {
                            2
                        }
                    );
                    let mut current = "s";
                    for id in &ids {
                        let (from, to) = catalog.get(id).expect("edge id belongs to fixture");
                        assert_eq!(from, current);
                        current = to;
                    }
                    assert_eq!(current, target);
                    assert!(distinct_paths.insert((target.clone(), ids)));
                    *counts.entry(target).or_default() += 1;
                }
                assert_eq!(fallbacks, 1, "{language:?}, indexed={indexed}, all={all}");
                let expected = if all {
                    BTreeMap::from([("a", 2), ("b", 1), ("d", 3), ("g", 1), ("h", 2)])
                } else {
                    BTreeMap::from([("a", 1), ("b", 1), ("d", 1), ("g", 1), ("h", 1)])
                };
                assert_eq!(
                    counts,
                    expected
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v))
                        .collect()
                );
            }
        }
    }
}

#[test]
fn optional_exists_keeps_every_left_row_and_parallel_edge() {
    for indexed in [false, true] {
        for &language in LANGUAGES {
            let db = adversarial_path_graph(indexed);
            let query = "MATCH (a:Node) OPTIONAL MATCH (a)-[r:OTHER]->(b) \
                         WHERE EXISTS { MATCH ()-[:OTHER]->(b) } RETURN a.id, b.id, id(r)";
            let actual = rows(&db.session(), language, query);
            assert_eq!(actual.len(), 12);
            let mut unmatched = BTreeSet::new();
            let mut matched_ids = BTreeSet::new();
            for row in actual {
                let source = text(&row[0]).unwrap();
                if source == "s" {
                    assert_eq!(text(&row[1]).as_deref(), Some("x"));
                    assert!(matched_ids.insert(integer(&row[2]).unwrap()));
                } else {
                    assert!(matches!(row[1], Value::Null));
                    assert!(matches!(row[2], Value::Null));
                    assert!(unmatched.insert(source));
                }
            }
            assert_eq!(matched_ids.len(), 2);
            assert_eq!(
                unmatched,
                node_ids()
                    .into_iter()
                    .filter(|id| *id != "s")
                    .map(str::to_string)
                    .collect()
            );
            let false_query = query.replace("MATCH ()-[:OTHER]->(b)", "MATCH (b)-[:MISSING]->()");
            let actual = rows(&db.session(), language, &false_query);
            assert_eq!(actual.len(), 11);
            let mut sources = BTreeSet::new();
            for row in actual {
                assert!(sources.insert(text(&row[0]).unwrap()));
                assert!(row[1..].iter().all(|value| matches!(value, Value::Null)));
            }
            assert_eq!(
                sources,
                node_ids().into_iter().map(str::to_string).collect()
            );
        }
    }
}

#[test]
fn a_pinned_parallel_edge_retains_its_shortest_path_identity() {
    for indexed in [false, true] {
        let db = adversarial_path_graph(indexed);
        let actual = rows(
            &db.session(),
            Language::Gql,
            "MATCH (:Node {id: 's'})-[e:OTHER]->(:Node {id: 'x'}) \
             MATCH p = ANY SHORTEST (a)-[e]->(b) \
             RETURN id(e), a.id, b.id, [r IN edges(p) | id(r)]",
        );
        assert_eq!(actual.len(), 2);
        let mut ids = BTreeSet::new();
        for row in actual {
            let id = integer(&row[0]).unwrap();
            assert!(ids.insert(id));
            assert_eq!(text(&row[1]).as_deref(), Some("s"));
            assert_eq!(text(&row[2]).as_deref(), Some("x"));
            assert_eq!(edge_ids(&row[3]), [id]);
        }
    }
}

fn optional_edges(session: &Session, language: Language) -> Vec<(String, i64)> {
    let mut found: Vec<_> = rows(
        session,
        language,
        "MATCH (s:Node {id: 's'}) OPTIONAL MATCH (s)-[r:OTHER]->(b) \
         WHERE EXISTS { MATCH ()-[:OTHER]->(b) } RETURN b.id, id(r)",
    )
    .into_iter()
    .map(|row| {
        (
            text(&row[0]).expect("a matching target"),
            integer(&row[1]).unwrap(),
        )
    })
    .collect();
    found.sort();
    found
}

fn free_targets(session: &Session, language: Language) -> Vec<String> {
    let mut found: Vec<_> = rows(
        session,
        language,
        "MATCH (:Node {id: 's'})-[r]->(b) RETURN b.id",
    )
    .iter()
    .map(|row| text(&row[0]).unwrap())
    .collect();
    found.sort();
    found
}

#[test]
fn optional_match_keeps_explicit_snapshot_and_own_edge_versions() {
    for indexed in [false, true] {
        for &language in LANGUAGES {
            let db = adversarial_path_graph(indexed);
            let before = optional_edges(&db.session(), language);
            assert_eq!(
                before
                    .iter()
                    .map(|(target, _)| target.as_str())
                    .collect::<Vec<_>>(),
                ["x", "x"]
            );
            assert_ne!(before[0].1, before[1].1);
            let mut reader = db.session();
            reader
                .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
                .unwrap();
            assert_eq!(optional_edges(&reader, language), before);
            let mut writer = db.session();
            writer
                .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
                .unwrap();
            writer
                .execute("MATCH (:Node {id: 's'})-[r:OTHER]->() DELETE r")
                .unwrap();
            let inserted = writer
                .execute(
                    "MATCH (s:Node {id: 's'}), (a:Node {id: 'a'}) \
                 INSERT (s)-[r:OTHER {cost: 1}]->(a) RETURN id(r)",
                )
                .unwrap();
            let id = integer(&inserted.rows()[0][0]).unwrap();
            assert!(before.iter().all(|(_, old)| *old != id));
            let own = vec![("a".to_string(), id)];
            assert_eq!(optional_edges(&writer, language), own);
            assert_eq!(free_targets(&writer, language), ["a", "a", "b"]);
            assert_eq!(optional_edges(&reader, language), before);
            assert_eq!(free_targets(&reader, language), ["a", "b", "x", "x"]);
            assert_eq!(optional_edges(&db.session(), language), before);
            writer.commit().unwrap();
            assert_eq!(optional_edges(&reader, language), before);
            assert_eq!(free_targets(&reader, language), ["a", "b", "x", "x"]);
            assert_eq!(optional_edges(&db.session(), language), own);
            reader.rollback().unwrap();
        }
    }
}

#[test]
fn optional_exists_correlates_each_candidate_before_null_extension() {
    for indexed in [false, true] {
        for &language in LANGUAGES {
            let db = adversarial_path_graph(indexed);
            add_parallel_rel(&db);
            let catalog = edge_catalog(&db);
            assert_eq!(catalog.len(), EDGES.len() + 1);
            let actual = rows(
                &db.session(),
                language,
                "MATCH (a:Node) OPTIONAL MATCH (a)-[r:REL]->(b) \
                 WHERE EXISTS { MATCH (b)-[:REL]->(:Node {id: 'd'}) } \
                 RETURN a.id, b.id, id(r)",
            );
            assert_eq!(actual.len(), 13, "{language:?}, indexed={indexed}");
            let mut counts = BTreeMap::new();
            let mut matched_ids = BTreeSet::new();
            let mut unmatched = BTreeSet::new();
            for row in actual {
                let source = text(&row[0]).unwrap();
                if matches!(row[1], Value::Null) {
                    assert!(matches!(row[2], Value::Null));
                    assert!(unmatched.insert(source));
                } else {
                    let pair = (source, text(&row[1]).unwrap());
                    let id = integer(&row[2]).unwrap();
                    assert!(matched_ids.insert(id));
                    assert_eq!(catalog.get(&id), Some(&pair));
                    *counts.entry(pair).or_insert(0) += 1;
                }
            }
            assert_eq!(
                counts,
                BTreeMap::from([
                    (("s".to_string(), "a".to_string()), 2),
                    (("s".to_string(), "b".to_string()), 1),
                    (("a".to_string(), "h".to_string()), 1),
                ])
            );
            assert_eq!(matched_ids.len(), 4);
            assert_eq!(
                unmatched,
                node_ids()
                    .into_iter()
                    .filter(|id| !matches!(*id, "s" | "a"))
                    .map(str::to_string)
                    .collect()
            );
        }
    }
}
