//! An OPTIONAL shortest path between two nodes the rows already bind (the
//! LDBC Interactive query 13 shape): one search per row, from its start to
//! its end, never a join of every start with every end. The length is that
//! of a breadth-first search over the same edges; no path gives null (IC13
//! returns -1 for it). A path from a node to itself does not exist
//! (`shortestPath` with the same start and end gives no path, user decision
//! 2026-10-30).
//!
//! - six people, every pair, in GQL (`ANY SHORTEST`) and Cypher
//!   (`shortestPath`), bound by a MATCH and by parameters;
//! - 1000 people and 18000 KNOWS edges, 40 pairs, against the search;
//! - the same start and end: `shortestPath` gives null under OPTIONAL and no
//!   row under MATCH, `*0..` the path of length 0; GQL `ANY SHORTEST` keeps
//!   its WALK rule (out and back with `+`, length 0 with `*`);
//! - the LDBC short query 7 shape (an OPTIONAL edge between two bound nodes
//!   through the message's author);
//! - an OPTIONAL part with a node of its own keeps the join (no per-row
//!   search), and a search after a write in the same statement sees it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test optional_shortest_path_both_ends
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::{BTreeSet, HashMap, VecDeque};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const GQL: &str = "MATCH (a:Person {id: $from}), (b:Person {id: $to}) \
                   OPTIONAL MATCH p = ANY SHORTEST (a)-[:KNOWS]-+(b) \
                   RETURN CASE WHEN p IS NULL THEN -1 ELSE length(p) END AS hops";
const CYPHER: &str = "MATCH (a:Person {id: $from}), (b:Person {id: $to}) \
                      OPTIONAL MATCH p = shortestPath((a)-[:KNOWS*]-(b)) \
                      RETURN CASE WHEN p IS NULL THEN -1 ELSE length(p) END AS hops";

/// People `0..people` and the KNOWS edges between them, with an id index,
/// as LDBC loads them.
fn graph(people: usize, edges: &[(usize, usize)]) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id").unwrap();
    let ids: Vec<_> = (0..people)
        .map(|i| {
            let id = db.create_node(&["Person"]).unwrap();
            db.set_node_property(id, "id", Value::Int64(i64::try_from(i).unwrap()))
                .unwrap();
            id
        })
        .collect();
    for &(from, to) in edges {
        db.create_edge(ids[from], ids[to], "KNOWS").unwrap();
    }
    db
}

/// The hops of a breadth-first search from `from` to `to`, edges either way;
/// -1 when there is no path or the two are the same node.
fn search(people: usize, edges: &[(usize, usize)], from: usize, to: usize) -> i64 {
    if from == to {
        return -1;
    }
    let mut next = vec![Vec::new(); people];
    for &(a, b) in edges {
        next[a].push(b);
        next[b].push(a);
    }
    let mut hops = vec![-1_i64; people];
    hops[from] = 0;
    let mut queue = VecDeque::from([from]);
    while let Some(node) = queue.pop_front() {
        for &other in &next[node] {
            if hops[other] < 0 {
                hops[other] = hops[node] + 1;
                if other == to {
                    return hops[other];
                }
                queue.push_back(other);
            }
        }
    }
    -1
}

fn hops(db: &GrafeoDB, language: &str, from: usize, to: usize) -> i64 {
    let params = HashMap::from([
        (
            "from".to_string(),
            Value::Int64(i64::try_from(from).unwrap()),
        ),
        ("to".to_string(), Value::Int64(i64::try_from(to).unwrap())),
    ]);
    let query = if language == "gql" { GQL } else { CYPHER };
    let result = db
        .execute_language(query, language, Some(params))
        .unwrap_or_else(|error| panic!("{language} {from} to {to}: {error}"));
    assert_eq!(result.rows().len(), 1, "{language} {from} to {to}");
    match &result.rows()[0][0] {
        Value::Int64(hops) => *hops,
        other => panic!("{language} {from} to {to}: {other:?}"),
    }
}

/// Alix 0, Gus 1, Vincent 2, Jules 3, Mia 4, and Butch 5 who knows nobody.
/// Alix-Gus, Gus-Vincent, Vincent-Jules, Alix-Mia, Mia-Jules (some stored
/// the other way round, so the search reads both directions).
const SMALL: [(usize, usize); 5] = [(0, 1), (2, 1), (2, 3), (0, 4), (3, 4)];

#[test]
fn every_pair_of_six_people() {
    let db = graph(6, &SMALL);
    for language in ["gql", "cypher"] {
        for from in 0..6 {
            for to in 0..6 {
                if from == to {
                    continue;
                }
                assert_eq!(
                    hops(&db, language, from, to),
                    search(6, &SMALL, from, to),
                    "{language} {from} to {to}"
                );
            }
        }
    }
}

/// The pairs come from rows (UNWIND), not parameters: each row searches its
/// own pair.
#[test]
fn pairs_from_rows() {
    let db = graph(6, &SMALL);
    // In the order of `ORDER BY a, b`.
    let pairs = [(0_usize, 3_usize), (0, 5), (1, 3), (4, 2), (5, 1)];
    let list = pairs
        .iter()
        .map(|(a, b)| format!("[{a}, {b}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let expected: Vec<Vec<Value>> = pairs
        .iter()
        .map(|&(a, b)| {
            vec![
                Value::Int64(i64::try_from(a).unwrap()),
                Value::Int64(i64::try_from(b).unwrap()),
                Value::Int64(search(6, &SMALL, a, b)),
            ]
        })
        .collect();
    for (language, path) in [
        ("gql", "ANY SHORTEST (a)-[:KNOWS]-+(b)"),
        ("cypher", "shortestPath((a)-[:KNOWS*]-(b))"),
    ] {
        let query = format!(
            "UNWIND [{list}] AS pair MATCH (a:Person {{id: pair[0]}}), (b:Person {{id: pair[1]}}) \
             OPTIONAL MATCH p = {path} \
             RETURN a.id AS a, b.id AS b, CASE WHEN p IS NULL THEN -1 ELSE length(p) END AS hops \
             ORDER BY a, b"
        );
        let got = db
            .execute_language(&query, language, None)
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(got.rows(), expected, "{language}");
    }
}

/// `shortestPath` from a node to itself gives no path: -1, not the walk out
/// and back over one edge (length 2).
#[test]
fn the_same_start_and_end_have_no_shortest_path() {
    let db = graph(6, &SMALL);
    for from in 0..6 {
        assert_eq!(hops(&db, "cypher", from, from), -1, "{from}");
    }
}

/// 1000 people, 18000 KNOWS edges drawn by a fixed generator (no self
/// loops), 40 pairs: each length is the search's.
#[test]
fn a_thousand_people() {
    let people = 1000;
    let mut state: u64 = 0x5eed_1c13;
    let mut draw = |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(state >> 33).unwrap() % bound
    };
    let mut edges = Vec::with_capacity(18_000);
    while edges.len() < 18_000 {
        let (a, b) = (draw(people), draw(people));
        if a != b {
            edges.push((a, b));
        }
    }
    // Half the people know only within the first 600: some pairs between
    // the two groups stay unreachable.
    let edges: Vec<_> = edges
        .into_iter()
        .map(|(a, b)| {
            if a < 600 || b < 600 {
                (a % 600, b % 600)
            } else {
                (a, b)
            }
        })
        .filter(|(a, b)| a != b)
        .collect();
    let db = graph(people, &edges);
    let mut unreachable = 0;
    for _ in 0..40 {
        let (from, to) = (draw(people), draw(people));
        if from == to {
            continue;
        }
        let want = search(people, &edges, from, to);
        unreachable += usize::from(want < 0);
        for language in ["gql", "cypher"] {
            assert_eq!(
                hops(&db, language, from, to),
                want,
                "{language} {from} to {to}"
            );
        }
    }
    assert!(unreachable > 0, "the pairs include unreachable ones");
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn int_rows(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

/// `shortestPath` with the same start and end: no row under MATCH; `*0..`
/// gives the path of length 0.
#[test]
fn the_same_start_and_end_under_match() {
    let db = graph(6, &SMALL);
    let shortest = "MATCH (a:Person {id: 0}), (b:Person {id: 0}) \
                    MATCH p = shortestPath((a)-[:KNOWS*]-(b)) RETURN length(p) AS hops";
    assert_eq!(rows(&db, "cypher", shortest), int_rows(&[]));
    let from_zero = shortest.replace("[:KNOWS*]", "[:KNOWS*0..]");
    assert_eq!(rows(&db, "cypher", &from_zero), int_rows(&[0]));
    let optional = from_zero.replacen(" MATCH p", " OPTIONAL MATCH p", 1);
    assert_eq!(rows(&db, "cypher", &optional), int_rows(&[0]));
}

/// GQL `ANY SHORTEST` is a WALK: from Alix to herself, `-+` goes out and back
/// over one edge (length 2) and `-*` is the empty path (length 0); Butch, who
/// knows nobody, has no `-+` path to himself.
#[test]
fn gql_any_shortest_keeps_its_walk_rule_for_the_same_ends() {
    let db = graph(6, &SMALL);
    let query = |person: i64, quantifier: &str| {
        format!(
            "MATCH (a:Person {{id: {person}}}), (b:Person {{id: {person}}}) \
             OPTIONAL MATCH p = ANY SHORTEST (a)-[:KNOWS]-{quantifier}(b) \
             RETURN CASE WHEN p IS NULL THEN -1 ELSE length(p) END AS hops"
        )
    };
    assert_eq!(rows(&db, "gql", &query(0, "+")), int_rows(&[2]));
    assert_eq!(rows(&db, "gql", &query(0, "*")), int_rows(&[0]));
    assert_eq!(rows(&db, "gql", &query(5, "+")), int_rows(&[-1]));
    assert_eq!(rows(&db, "gql", &query(5, "*")), int_rows(&[0]));
}

/// The LDBC short query 7 shape: the replies of a message, each with whether
/// its author knows the message's author (an OPTIONAL edge between two
/// nodes the row binds, through the author). Checked against the KNOWS
/// edges, both directions.
#[test]
fn the_is7_shape() {
    // Message 0 by Alix (person 0); replies 1 to 5 by persons 1 to 5.
    let db = graph(6, &SMALL);
    db.execute("INSERT (:Message {id: 100})").unwrap();
    db.execute("MATCH (m:Message {id: 100}), (a:Person {id: 0}) INSERT (m)-[:HAS_CREATOR]->(a)")
        .unwrap();
    for author in 1..6 {
        db.execute(&format!(
            "MATCH (m:Message {{id: 100}}), (p:Person {{id: {author}}}) \
             INSERT (m)<-[:REPLY_OF]-(:Comment {{id: {}}})-[:HAS_CREATOR]->(p)",
            300 + author
        ))
        .unwrap();
    }
    let knows_alix: BTreeSet<i64> = SMALL
        .iter()
        .filter_map(|&(a, b)| match (a, b) {
            (0, other) | (other, 0) => Some(i64::try_from(other).unwrap()),
            _ => None,
        })
        .collect();
    let expected: Vec<Vec<Value>> = (1..6)
        .map(|author| {
            vec![
                Value::Int64(300 + author),
                Value::Int64(author),
                Value::Bool(knows_alix.contains(&author)),
            ]
        })
        .collect();
    let query = "MATCH (m:Message {id: 100})<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) \
                 OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) \
                 RETURN c.id AS comment, p.id AS author, r IS NOT NULL AS knows ORDER BY comment";
    for language in ["gql", "cypher"] {
        assert_eq!(rows(&db, language, query), expected, "{language}");
    }
}

/// An OPTIONAL part with a node of its own (`m`) keeps the join: no per-row
/// search in the plan, and the counts of every Person's KNOWS edges.
#[test]
fn an_optional_part_with_a_node_of_its_own_keeps_the_join() {
    let db = graph(6, &SMALL);
    let query = "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m) \
                 RETURN n.id AS id, count(m) AS c ORDER BY id";
    let expected: Vec<Vec<Value>> = (0..6)
        .map(|person| {
            let out = SMALL.iter().filter(|&&(a, _)| a == person).count();
            vec![
                Value::Int64(i64::try_from(person).unwrap()),
                Value::Int64(i64::try_from(out).unwrap()),
            ]
        })
        .collect();
    for language in ["gql", "cypher"] {
        assert_eq!(rows(&db, language, query), expected, "{language}");
        let profile = rows(&db, language, &format!("PROFILE {query}"));
        let Value::String(text) = &profile[0][0] else {
            panic!("{language}: {profile:?}");
        };
        assert!(
            !text.contains("CorrelatedSubplan"),
            "{language}: no per-row search: {text}"
        );
    }
}

/// A search after a write in the same statement sees the new edge: Butch
/// (5) knows nobody until the statement adds Butch to Mia (4).
#[test]
fn a_search_after_a_write_sees_it() {
    for (language, write) in [("gql", "INSERT"), ("cypher", "CREATE")] {
        let db = graph(6, &SMALL);
        let path = if language == "gql" {
            "ANY SHORTEST (a)-[:KNOWS]-+(b)"
        } else {
            "shortestPath((a)-[:KNOWS*]-(b))"
        };
        let query = format!(
            "MATCH (a:Person {{id: 5}}), (b:Person {{id: 1}}), (mia:Person {{id: 4}}) \
             {write} (a)-[:KNOWS]->(mia) WITH a, b \
             OPTIONAL MATCH p = {path} \
             RETURN CASE WHEN p IS NULL THEN -1 ELSE length(p) END AS hops"
        );
        // Butch-Mia-Alix-Gus: 3 hops.
        assert_eq!(rows(&db, language, &query), int_rows(&[3]), "{language}");
    }
}
