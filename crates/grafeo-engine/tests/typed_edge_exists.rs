//! An EXISTS of one edge (the typed edge check, 0.6.1) stops at the first
//! visible edge of the type; its answer is that of listing the edges, which
//! each case checks against `COUNT {...} > 0` (COUNT keeps listing):
//!
//! - a node with thousands of edges of other types and one of the type, and
//!   one without it;
//! - Both with a self-loop, and an end the row binds (`(n)-[:T]-(n)`);
//! - an end label checked after the edge (`(n)-[:T]->(:Target)`);
//! - a variable-length pattern's first edge;
//! - an incoming check on a store without incoming lists;
//! - an edge created or deleted in the transaction, and a past epoch.
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test typed_edge_exists
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Hub (k 1): 3000 OTHER edges to Leaf nodes and one WANTED edge to a Target.
/// Busy (k 2): 3000 OTHER edges, no WANTED. Plain (k 3): one WANTED edge to
/// a Leaf (not a Target). Loop (k 4): one LOOP self-loop. Lonely (k 5): no
/// edges.
fn database(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (:N {k: 1}), (:N {k: 2}), (:N {k: 3}), (:N {k: 4}), (:N {k: 5}), (:Target {k: 10})",
    )
    .unwrap();
    for k in [1, 2] {
        db.execute(&format!(
            "MATCH (n:N {{k: {k}}}) UNWIND range(1, 3000) AS i INSERT (n)-[:OTHER]->(:Leaf {{i: i}})"
        ))
        .unwrap();
    }
    db.execute("MATCH (n:N {k: 1}), (t:Target) INSERT (n)-[:WANTED]->(t)")
        .unwrap();
    db.execute("MATCH (n:N {k: 3}) INSERT (n)-[:WANTED]->(:Leaf {i: 0})")
        .unwrap();
    db.execute("MATCH (n:N {k: 4}) INSERT (n)-[:LOOP]->(n)")
        .unwrap();
    db
}

fn database_default() -> GrafeoDB {
    database(Config::in_memory().with_history_retention(HistoryRetention::Forever))
}

fn ks(db: &GrafeoDB, language: &str, query: &str) -> Vec<i64> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(k) => k,
            ref other => panic!("{other:?}"),
        })
        .collect()
}

/// The N nodes `pattern` (an EXISTS body over `n`) holds for, and the same
/// with `COUNT {...} > 0`, which must agree.
fn holds(db: &GrafeoDB, language: &str, pattern: &str) -> Vec<i64> {
    let exists = ks(
        db,
        language,
        &format!("MATCH (n:N) WHERE EXISTS {{ {pattern} }} RETURN n.k ORDER BY n.k"),
    );
    let counted = ks(
        db,
        language,
        &format!("MATCH (n:N) WHERE COUNT {{ {pattern} }} > 0 RETURN n.k ORDER BY n.k"),
    );
    assert_eq!(
        exists, counted,
        "{language} `{pattern}`: EXISTS against COUNT"
    );
    exists
}

#[test]
fn one_edge_of_the_type_among_thousands() {
    let db = database_default();
    for language in LANGUAGES {
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]->()"),
            [1, 3],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:OTHER]->()"),
            [1, 2],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:MISSING]->()"),
            Vec::<i64>::new(),
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)<-[:WANTED]-()"),
            Vec::<i64>::new(),
            "{language}: no N has an incoming WANTED"
        );
        assert_eq!(
            ks(
                &db,
                language,
                "MATCH (n:N {k: 1}) RETURN COUNT { MATCH (n)-[:OTHER]->() } AS c"
            ),
            [3000],
            "{language}: COUNT lists every edge"
        );
    }
}

#[test]
fn both_directions_and_a_self_loop() {
    let db = database_default();
    for language in LANGUAGES {
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:LOOP]-()"),
            [4],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:LOOP]-(n)"),
            [4],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]-()"),
            [1, 3],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]-(n)"),
            Vec::<i64>::new(),
            "{language}"
        );
    }
}

/// The end label is checked after the edge: k 3's WANTED goes to a Leaf.
#[test]
fn an_end_label_after_the_edge() {
    let db = database_default();
    for language in LANGUAGES {
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]->(:Target)"),
            [1],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]->(:Leaf)"),
            [3],
            "{language}"
        );
    }
}

/// A variable-length pattern checks its first edge the same way.
#[test]
fn a_variable_length_first_edge() {
    let db = database_default();
    for (language, one_or_two, loop_path) in [
        ("gql", "-[:WANTED]->{1,2}", "-[:LOOP]->{1,3}"),
        ("cypher", "-[:WANTED*1..2]->", "-[:LOOP*1..3]->"),
    ] {
        assert_eq!(
            holds(&db, language, &format!("MATCH (n){one_or_two}()")),
            [1, 3],
            "{language}"
        );
        assert_eq!(
            holds(&db, language, &format!("MATCH (n){loop_path}()")),
            [4],
            "{language}"
        );
    }
}

/// Without incoming lists the incoming check still finds the edges.
#[test]
fn an_incoming_check_without_incoming_lists() {
    let db = database(Config::in_memory().without_backward_edges());
    for language in LANGUAGES {
        assert_eq!(
            ks(
                &db,
                language,
                "MATCH (t:Target) WHERE EXISTS { MATCH (t)<-[:WANTED]-() } RETURN t.k"
            ),
            [10],
            "{language}"
        );
        assert_eq!(
            ks(
                &db,
                language,
                "MATCH (t:Target) WHERE EXISTS { MATCH (t)<-[:OTHER]-() } RETURN t.k"
            ),
            Vec::<i64>::new(),
            "{language}"
        );
        assert_eq!(
            holds(&db, language, "MATCH (n)<-[:LOOP]-()"),
            [4],
            "{language}"
        );
    }
}

/// In a transaction: k 1's WANTED edge deleted and one created for k 2; the
/// check sees both changes inside and none after the rollback.
#[test]
fn edges_created_and_deleted_in_the_transaction() {
    let db = database_default();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:N {k: 1})-[r:WANTED]->() DELETE r")
        .unwrap();
    session
        .execute("MATCH (n:N {k: 2}), (t:Target) INSERT (n)-[:WANTED]->(t)")
        .unwrap();
    let inside: Vec<i64> = session
        .execute("MATCH (n:N) WHERE EXISTS { MATCH (n)-[:WANTED]->() } RETURN n.k ORDER BY n.k")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(k) => k,
            ref other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(inside, [2, 3]);
    session.rollback().unwrap();
    for language in LANGUAGES {
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]->()"),
            [1, 3],
            "{language}"
        );
    }
}

/// At a past epoch: before k 5 got a WANTED edge and k 3 lost its own.
#[test]
fn at_a_past_epoch() {
    let db = database_default();
    let before = db.current_epoch();
    db.execute("MATCH (n:N {k: 5}), (t:Target) INSERT (n)-[:WANTED]->(t)")
        .unwrap();
    db.execute("MATCH (:N {k: 3})-[r:WANTED]->() DELETE r")
        .unwrap();
    for language in LANGUAGES {
        assert_eq!(
            holds(&db, language, "MATCH (n)-[:WANTED]->()"),
            [1, 5],
            "{language} now"
        );
    }
    let then: Vec<i64> = db
        .execute_at_epoch(
            "MATCH (n:N) WHERE EXISTS { MATCH (n)-[:WANTED]->() } RETURN n.k ORDER BY n.k",
            before,
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(k) => k,
            ref other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(then, [1, 3]);
}
