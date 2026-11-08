//! A `CALL { }` body combined by a set operation passes its columns on to every reader after
//! the CALL, for every operator alike (ISO/IEC 39075 14.2 SR 10, 15.2 SR 10 and 11): a property
//! read, a MATCH from the returned node, a nested CALL (scoped and implicit), `id()`, and a
//! MERGE on it, which must bind the node itself and create nothing new. The operators come from
//! an exhaustive match, so a new one does not compile until it is listed. The oracle is
//! computed: each branch run on its own, and the multiset operation on the names (each person's
//! name is unique).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main call_body_set_operations::
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeMap;

use grafeo_adapters::query::gql::CompositeOp;
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The keyword of each set operation; NEXT is not one.
fn keyword(op: CompositeOp) -> Option<&'static str> {
    match op {
        CompositeOp::Union => Some("UNION"),
        CompositeOp::UnionAll => Some("UNION ALL"),
        CompositeOp::Except => Some("EXCEPT"),
        CompositeOp::ExceptAll => Some("EXCEPT ALL"),
        CompositeOp::Intersect => Some("INTERSECT"),
        CompositeOp::IntersectAll => Some("INTERSECT ALL"),
        CompositeOp::Otherwise => Some("OTHERWISE"),
        CompositeOp::Next => None,
    }
}

const OPS: [CompositeOp; 8] = [
    CompositeOp::Union,
    CompositeOp::UnionAll,
    CompositeOp::Except,
    CompositeOp::ExceptAll,
    CompositeOp::Intersect,
    CompositeOp::IntersectAll,
    CompositeOp::Otherwise,
    CompositeOp::Next,
];

/// Alix knows Gus, Gus knows Vincent, Alix works at the company.
fn social() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'}), (:Person {name: 'Vincent'}), \
         (:Company {name: 'Grafeo OSS'})",
        "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(g)",
        "MATCH (g:Person {name: 'Gus'}), (v:Person {name: 'Vincent'}) INSERT (g)-[:KNOWS]->(v)",
        "MATCH (a:Person {name: 'Alix'}), (c:Company) INSERT (a)-[:WORKS_AT]->(c)",
    ] {
        db.execute(statement).unwrap();
    }
    db
}

/// Three pairs of branches: every person against those who know someone; a person per
/// outgoing edge (Alix twice) against the same, so the ALL forms see duplicates; and an empty
/// left (no person works at a person) against those who know someone, so OTHERWISE runs its
/// right side.
const BRANCHES: [(&str, &str); 3] = [
    (
        "MATCH (n:Person) RETURN n AS v",
        "MATCH (n:Person)-[:KNOWS]->() RETURN n AS v",
    ),
    (
        "MATCH (n:Person)-[]->() RETURN n AS v",
        "MATCH (n:Person)-[:KNOWS]->() RETURN n AS v",
    ),
    (
        "MATCH (n:Person)-[:WORKS_AT]->(:Person) RETURN n AS v",
        "MATCH (n:Person)-[:KNOWS]->() RETURN n AS v",
    ),
];

fn strings(db: &GrafeoDB, query: &str) -> Vec<String> {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut out: Vec<String> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(s) => s.to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    out.sort();
    out
}

fn counts(names: &[String]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for name in names {
        *map.entry(name.clone()).or_insert(0) += 1;
    }
    map
}

/// The names the combined body yields, from the branches' names.
fn oracle(op: CompositeOp, left: &[String], right: &[String]) -> Vec<String> {
    let (l, r) = (counts(left), counts(right));
    let mut out = Vec::new();
    match op {
        CompositeOp::Union => out.extend(l.keys().chain(r.keys()).cloned()),
        CompositeOp::UnionAll => out.extend(left.iter().chain(right).cloned()),
        CompositeOp::Except => out.extend(l.keys().filter(|k| !r.contains_key(*k)).cloned()),
        CompositeOp::ExceptAll => {
            for (name, n) in &l {
                let m = r.get(name).copied().unwrap_or(0);
                out.extend(std::iter::repeat_n(name.clone(), n.saturating_sub(m)));
            }
        }
        CompositeOp::Intersect => out.extend(l.keys().filter(|k| r.contains_key(*k)).cloned()),
        CompositeOp::IntersectAll => {
            for (name, n) in &l {
                let m = r.get(name).copied().unwrap_or(0);
                out.extend(std::iter::repeat_n(name.clone(), (*n).min(m)));
            }
        }
        CompositeOp::Otherwise => {
            out.extend(if left.is_empty() { right } else { left }.iter().cloned());
        }
        CompositeOp::Next => unreachable!("not a set operation"),
    }
    out.sort();
    out.dedup_by(|a, b| a == b && matches!(op, CompositeOp::Union));
    out
}

#[test]
fn every_set_operation_passes_its_columns_to_every_reader() {
    let db = social();
    let knows: BTreeMap<String, Vec<String>> = {
        let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for pair in strings(&db, "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name") {
            let (a, b) = pair.split_once('|').unwrap();
            map.entry(a.to_string()).or_default().push(b.to_string());
        }
        map
    };
    let ids: BTreeMap<String, String> = strings(&db, "MATCH (n:Person) RETURN n.name, id(n)")
        .into_iter()
        .map(|pair| {
            let (name, id) = pair.split_once('|').unwrap();
            (name.to_string(), id.to_string())
        })
        .collect();
    let mut wrong = Vec::new();
    for op in OPS {
        let Some(keyword) = keyword(op) else {
            continue;
        };
        for (left, right) in BRANCHES {
            let names =
                |branch: &str| strings(&db, &branch.replace("RETURN n AS v", "RETURN n.name AS v"));
            let want = oracle(op, &names(left), &names(right));
            let body = format!("CALL {{ {left} {keyword} {right} }}");

            let mut check = |reader: &str, expected: Vec<String>| {
                let query = format!("{body} {reader}");
                match db.execute(&query) {
                    Ok(_) => {
                        let got = strings(&db, &query);
                        let mut expected = expected;
                        expected.sort();
                        if got != expected {
                            wrong.push(format!("{query}: {got:?}, want {expected:?}"));
                        }
                    }
                    Err(error) => wrong.push(format!("{query}: {error}")),
                }
            };
            check("RETURN v.name AS a", want.clone());
            check(
                "MATCH (v)-[:KNOWS]->(b) RETURN v.name AS a, b.name AS b",
                want.iter()
                    .flat_map(|name| {
                        knows
                            .get(name)
                            .into_iter()
                            .flatten()
                            .map(move |b| format!("{name}|{b}"))
                    })
                    .collect(),
            );
            check("CALL (v) { RETURN v.name AS w } RETURN w", want.clone());
            check(
                "CALL { WITH 1 AS k RETURN v.name AS w } RETURN w",
                want.clone(),
            );
            check(
                "RETURN id(v) AS i",
                want.iter().map(|name| ids[name].clone()).collect(),
            );

            // A MERGE on the returned node binds it: one SEEN edge and one :Seen node per distinct
            // node, no node made for v; the rollback leaves nothing.
            let mut session = db.session();
            session.begin_transaction().unwrap();
            let merge = format!("{body} MERGE (v)-[:SEEN]->(:Seen)");
            if let Err(error) = session.execute(&merge) {
                wrong.push(format!("{merge}: {error}"));
            } else {
                let seen = session
                    .execute("MATCH (p:Person)-[:SEEN]->() RETURN p.name AS n")
                    .unwrap()
                    .rows()
                    .len();
                let distinct = counts(&want).len();
                if seen != distinct {
                    wrong.push(format!("{merge}: {seen} SEEN edges, want {distinct}"));
                }
                let nodes = session.execute("MATCH (n) RETURN n").unwrap().rows().len();
                if nodes != 4 + distinct {
                    wrong.push(format!("{merge}: {nodes} nodes, want {}", 4 + distinct));
                }
            }
            session.rollback().unwrap();
        }
    }
    assert_eq!(
        strings(&db, "MATCH ()-[r:SEEN]->() RETURN r"),
        Vec::<String>::new(),
        "a rollback left SEEN edges"
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Duplicates are decided by identity: two nodes with equal properties are two rows.
#[test]
fn two_equal_looking_nodes_are_two_rows() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:T {k: 1}), (:T {k: 1})").unwrap();
    let rows = db
        .execute("CALL { MATCH (n:T) RETURN n AS v INTERSECT MATCH (n:T) RETURN n AS v } RETURN count(*) AS c")
        .unwrap();
    assert_eq!(rows.rows()[0][0], Value::Int64(2));
}

/// A body ending in OTHERWISE binds the returned node for a MATCH and a MERGE after it.
#[test]
fn an_otherwise_body_binds_the_node_for_match_and_merge() {
    let db = social();
    let body = "CALL { MATCH (n:Person {name: 'Alix'}) RETURN n AS v OTHERWISE MATCH (n:Company) RETURN n AS v }";
    assert_eq!(
        strings(
            &db,
            &format!("{body} MATCH (v)-[:KNOWS]->(b) RETURN b.name")
        ),
        ["Gus"]
    );
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(&format!("{body} MERGE (v)-[:TAGGED]->(:Tag {{name: 't'}})"))
        .unwrap();
    let tagged = session
        .execute("MATCH (a:Person)-[:TAGGED]->(:Tag) RETURN a.name")
        .unwrap()
        .rows()
        .len();
    assert_eq!(tagged, 1, "one TAGGED edge, from Alix");
    let nodes = session.execute("MATCH (n) RETURN n").unwrap().rows().len();
    assert_eq!(nodes, 5, "the four nodes and one :Tag, no node made for v");
    session.rollback().unwrap();
    assert_eq!(
        strings(&db, "MATCH ()-[r:TAGGED]->() RETURN r"),
        Vec::<String>::new(),
        "the rollback left a TAGGED edge"
    );
}
