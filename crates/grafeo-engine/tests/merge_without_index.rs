//! MERGE over many rows without a property index looks each row's values up
//! among the pattern's nodes, kept current from the transaction's writes
//! (0.6.1). Its matches must be those of a scan, which a property index
//! leaves in place: each case runs the same statements on a store without
//! an index and on one with an index on the match properties, compares the
//! statement's rows and the nodes after it, and checks the expected ones:
//!
//! - duplicate keys among the rows merge into one node, ON CREATE and ON
//!   MATCH included;
//! - 1, '1', true and a float stay apart (a float row reads the store);
//!   -0.0 and 0.0 through an index (ignored: MERGE's index lookup tells
//!   them apart);
//! - ON CREATE SET of the key, which later rows see; SET, SET += and REMOVE
//!   of the key and label changes after the MERGE, which run after its
//!   every row (clause by clause, as with an index);
//! - two match properties, one missing on some nodes;
//! - a node deleted later in the statement;
//! - nodes an earlier statement of the transaction created in bulk, an
//!   explicit transaction and its rollback;
//! - a two-label pattern, and every match bound (two nodes, one key).
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test merge_without_index
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// A database without an index and one with an index on each of `keys`,
/// each after `setup` (GQL).
fn databases(setup: &[&str], keys: &[&str]) -> [(&'static str, GrafeoDB); 2] {
    let build = |index: bool| {
        let db = GrafeoDB::new_in_memory();
        if index {
            for key in keys {
                db.create_property_index(key).unwrap();
            }
        }
        for statement in setup {
            db.execute(statement)
                .unwrap_or_else(|error| panic!("`{statement}`: {error}"));
        }
        db
    };
    [("no index", build(false)), ("index", build(true))]
}

/// Every node as `labels (sorted) properties`, sorted.
fn nodes(db: &GrafeoDB) -> Vec<String> {
    let mut nodes: Vec<String> = db
        .execute_cypher("MATCH (n) RETURN labels(n) AS l, properties(n) AS p")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            let Value::List(labels) = &row[0] else {
                panic!("{:?}", row[0]);
            };
            let mut labels: Vec<String> = labels.iter().map(|l| format!("{l:?}")).collect();
            labels.sort();
            format!("{} {:?}", labels.join(":"), row[1])
        })
        .collect();
    nodes.sort();
    nodes
}

fn sorted_rows(db: &GrafeoDB, language: &str, statement: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_language(statement, language, None)
        .unwrap_or_else(|error| panic!("{language} `{statement}`: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// Runs `statement` after `setup` without and with the index, in each
/// language; the rows and the nodes after it must agree. Calls `check` with
/// the database (no index) after the statement and the language.
fn agree(setup: &[&str], keys: &[&str], statement: &str, check: impl Fn(&GrafeoDB, &str)) {
    for language in LANGUAGES {
        let [(_, plain), (_, indexed)] = databases(setup, keys);
        let rows = sorted_rows(&plain, language, statement);
        assert_eq!(
            rows,
            sorted_rows(&indexed, language, statement),
            "{language} `{statement}`: the rows without and with the index"
        );
        assert_eq!(
            nodes(&plain),
            nodes(&indexed),
            "{language} `{statement}`: the nodes without and with the index"
        );
        check(&plain, language);
        check(&indexed, language);
    }
}

fn ints(db: &GrafeoDB, query: &str) -> Vec<i64> {
    db.execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(v) => v,
            ref other => panic!("`{query}`: {other:?}"),
        })
        .collect()
}

#[test]
fn duplicate_keys_merge_into_one_node() {
    agree(
        &[],
        &["id"],
        "UNWIND [1, 2, 1, 3, 2, 1] AS k MERGE (m:M {id: k}) \
         ON CREATE SET m.c = 1 ON MATCH SET m.c = m.c + 1",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.c ORDER BY m.id"),
                [3, 2, 1],
                "{language}"
            );
        },
    );
    // More rows than a chunk holds, each key three times.
    agree(
        &[],
        &["id"],
        "UNWIND range(1, 3000) AS i WITH i % 1000 AS k MERGE (m:M {id: k}) \
         ON CREATE SET m.c = 1 ON MATCH SET m.c = m.c + 1",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN count(m)"),
                [1000],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN sum(m.c)"),
                [3000],
                "{language}"
            );
        },
    );
}

/// 1, '1' and true are three keys, and a float a fourth (a float row reads
/// the store).
#[test]
fn values_of_different_kinds_stay_apart() {
    agree(
        &[],
        &["id"],
        "UNWIND [1, '1', true, 1, '1', true, 1.5, 1.5, 1] AS k MERGE (m:M {id: k}) \
         ON MATCH SET m.hits = coalesce(m.hits, 0) + 1",
        |db, language| {
            assert_eq!(ints(db, "MATCH (m:M) RETURN count(m)"), [4], "{language}");
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.hits ORDER BY m.hits"),
                [1, 1, 1, 2],
                "{language}: true, '1', 1.5 once more, 1 twice"
            );
        },
    );
}

/// `-0.0 = 0.0`: without an index MERGE matches the node, through a
/// property index it creates a second one (MATCH finds it either way).
#[test]
#[ignore = "MERGE through a property index tells 0.0 and -0.0 apart where = joins them (0.6.0 too)"]
fn negative_zero_matches_zero_with_an_index() {
    for (name, db) in databases(&[], &["id"]) {
        db.execute_cypher("CREATE (:M {id: -0.0})").unwrap();
        db.execute_cypher("UNWIND [0.0, 1.5, 0.0] AS k MERGE (m:M {id: k})")
            .unwrap();
        assert_eq!(
            ints(&db, "MATCH (m:M) RETURN count(m)"),
            [2],
            "{name}: -0.0 and 1.5"
        );
        assert_eq!(
            ints(&db, "MATCH (m:M {id: 0.0}) RETURN count(m)"),
            [1],
            "{name}: MATCH by 0.0 finds -0.0"
        );
    }
}

/// ON CREATE SET moves a created node's key: a later row with the new key
/// matches it.
#[test]
fn a_key_set_on_create_counts_for_later_rows() {
    agree(
        &[],
        &["id"],
        "UNWIND [1, 11, 2, 12, 3, 13] AS k MERGE (m:M {id: k}) ON CREATE SET m.id = k + 10",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.id ORDER BY m.id"),
                [11, 12, 13],
                "{language}"
            );
        },
    );
}

/// A SET after the MERGE runs after the MERGE's every row (clause by
/// clause): the MERGE creates 1, 5, 2, 7 and 3, then the SET moves each.
#[test]
fn a_later_set_of_the_key_runs_after_every_merge_row() {
    agree(
        &[],
        &["id"],
        "UNWIND [[1, 5], [5, 6], [2, 7], [7, 8], [3, 9]] AS p MERGE (m:M {id: p[0]}) \
         SET m.id = p[1]",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.id ORDER BY m.id"),
                [5, 6, 7, 8, 9],
                "{language}"
            );
        },
    );
    // The same through SET += (Cypher's map update).
    let statement = "UNWIND [[1, 5], [5, 6], [2, 7], [7, 8], [3, 9]] AS p \
                     MERGE (m:M {id: p[0]}) SET m += {id: p[1], seen: true}";
    let [(_, plain), (_, indexed)] = databases(&[], &["id"]);
    for db in [&plain, &indexed] {
        db.execute_cypher(statement).unwrap();
        assert_eq!(
            ints(db, "MATCH (m:M) RETURN m.id ORDER BY m.id"),
            [5, 6, 7, 8, 9]
        );
    }
    assert_eq!(nodes(&plain), nodes(&indexed));
}

/// A REMOVE of the key after the MERGE: the k 1 rows all bind the one node
/// the first created, whose key goes after.
#[test]
fn a_later_remove_of_the_key() {
    agree(
        &[],
        &["id"],
        "UNWIND [1, 2, 1, 3, 1, 2] AS k MERGE (m:M {id: k}) \
         WITH m, k WHERE k = 1 REMOVE m.id",
        |db, language| {
            assert_eq!(ints(db, "MATCH (m:M) RETURN count(m)"), [3], "{language}");
            assert_eq!(
                ints(db, "MATCH (m:M) WHERE m.id IS NULL RETURN count(m)"),
                [1],
                "{language}"
            );
        },
    );
}

/// A label removed after the MERGE, and a label a later SET adds to
/// another node with the key: the next statement's MERGE binds both.
#[test]
fn labels_removed_or_added_by_a_later_write() {
    agree(
        &[],
        &["id"],
        "UNWIND [1, 2, 1, 3, 1, 2] AS k MERGE (m:M {id: k}) \
         WITH m, k WHERE k = 1 REMOVE m:M SET m:Gone",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:Gone) RETURN count(m)"),
                [1],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.id ORDER BY m.id"),
                [2, 3],
                "{language}"
            );
        },
    );
    agree(
        &["INSERT (:Other {id: 2}), (:Other {id: 4})"],
        &["id"],
        "UNWIND [1, 2, 3, 2, 4, 4] AS k MERGE (m:M {id: k}) WITH m, k \
         MATCH (o:Other {id: k}) SET o:M RETURN k, m.id",
        |db, language| {
            assert_eq!(
                ints(
                    db,
                    "MATCH (m:M) WHERE NOT m:Other RETURN m.id ORDER BY m.id"
                ),
                [1, 2, 3, 4],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH (m:M:Other) RETURN m.id ORDER BY m.id"),
                [2, 4],
                "{language}"
            );
            assert_eq!(
                ints(
                    db,
                    "UNWIND [2, 4, 2, 3] AS k MERGE (m:M {id: k}) RETURN count(*)"
                ),
                [7],
                "{language}: 2 and 4 bind two nodes each"
            );
        },
    );
}

/// `{a, b}`: a node without b matches no row; equal pairs merge.
#[test]
fn two_match_properties_one_missing_on_some_nodes() {
    agree(
        &["INSERT (:M {a: 1, b: 1, was: 'there'}), (:M {a: 1})"],
        &["a", "b"],
        "UNWIND [[1, 1], [1, 2], [1, 1], [1, 2], [2, 1], [1, 2]] AS p \
         MERGE (m:M {a: p[0], b: p[1]}) ON MATCH SET m.hits = coalesce(m.hits, 0) + 1",
        |db, language| {
            assert_eq!(ints(db, "MATCH (m:M) RETURN count(m)"), [4], "{language}");
            assert_eq!(
                ints(db, "MATCH (m:M {a: 1, b: 1}) RETURN m.hits"),
                [2],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH (m:M {a: 1, b: 2}) RETURN m.hits"),
                [2],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH (m:M) WHERE m.b IS NULL RETURN count(m)"),
                [1],
                "{language}: the node without b untouched"
            );
        },
    );
}

#[test]
fn a_node_deleted_later_in_the_statement() {
    agree(
        &["INSERT (:M {id: 1})-[:R]->(:N)"],
        &["id"],
        "UNWIND [1, 2, 1, 3, 1] AS k MERGE (m:M {id: k}) \
         WITH m, k WHERE k = 1 DETACH DELETE m",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN m.id ORDER BY m.id"),
                [2, 3],
                "{language}"
            );
            assert_eq!(
                ints(db, "MATCH ()-[r:R]->() RETURN count(r)"),
                [0],
                "{language}"
            );
        },
    );
}

/// Nodes an earlier statement of the transaction created in bulk are
/// matched; the commit keeps them, a rollback leaves nothing.
#[test]
fn bulk_created_nodes_earlier_in_the_transaction() {
    for (language, create) in [("gql", "INSERT"), ("cypher", "CREATE")] {
        for commit in [true, false] {
            for (name, db) in databases(&["INSERT (:M {id: -1})"], &["id"]) {
                let mut session = db.session();
                session.begin_transaction().unwrap();
                session
                    .execute_language(
                        &format!("UNWIND range(1, 2000) AS i {create} (:M {{id: i}})"),
                        language,
                        None,
                    )
                    .unwrap();
                session
                    .execute_language(
                        "UNWIND [5, 2001, 5, -1, 2001, 7, 2001] AS k MERGE (m:M {id: k}) \
                         ON MATCH SET m.hits = coalesce(m.hits, 0) + 1",
                        language,
                        None,
                    )
                    .unwrap();
                let inside = |query: &str| -> Vec<Value> {
                    session
                        .execute_language(query, "cypher", None)
                        .unwrap()
                        .rows()
                        .iter()
                        .map(|row| row[0].clone())
                        .collect()
                };
                assert_eq!(
                    inside("MATCH (m:M) RETURN count(m)"),
                    [Value::Int64(2002)],
                    "{language} {name}"
                );
                assert_eq!(
                    inside(
                        "MATCH (m:M) WHERE m.id IN [-1, 5, 7, 2001] \
                         RETURN m.hits ORDER BY m.id"
                    ),
                    [1, 2, 1, 2].map(Value::Int64),
                    "{language} {name}"
                );
                if commit {
                    session.commit().unwrap();
                    assert_eq!(
                        ints(&db, "MATCH (m:M) RETURN count(m)"),
                        [2002],
                        "{language} {name}"
                    );
                } else {
                    session.rollback().unwrap();
                    assert_eq!(
                        ints(&db, "MATCH (m:M) RETURN m.id"),
                        [-1],
                        "{language} {name}"
                    );
                }
            }
        }
    }
}

/// An explicit transaction: a MERGE matches what an earlier statement
/// created and what its own earlier rows created.
#[test]
fn an_explicit_transaction_sees_its_earlier_statements() {
    for language in LANGUAGES {
        for (name, db) in databases(&[], &["id"]) {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_language("INSERT (:M {id: 1, from: 'insert'})", "gql", None)
                .unwrap();
            session
                .execute_language(
                    "UNWIND [2, 1, 2, 3, 1, 3] AS k MERGE (m:M {id: k}) \
                     ON CREATE SET m.from = 'merge' ON MATCH SET m.n = coalesce(m.n, 0) + 1",
                    language,
                    None,
                )
                .unwrap();
            session.commit().unwrap();
            assert_eq!(
                ints(&db, "MATCH (m:M) RETURN m.n ORDER BY m.id"),
                [2, 1, 1],
                "{language} {name}"
            );
        }
    }
}

#[test]
fn a_two_label_pattern() {
    agree(
        &["INSERT (:A:B {id: 1}), (:A {id: 2}), (:B {id: 2})"],
        &["id"],
        "UNWIND [1, 2, 1, 2, 3, 3] AS k MERGE (m:A:B {id: k})",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:A:B) RETURN m.id ORDER BY m.id"),
                [1, 2, 3],
                "{language}"
            );
            assert_eq!(ints(db, "MATCH (m) RETURN count(m)"), [5], "{language}");
        },
    );
}

/// Two nodes with one key: each row binds both.
#[test]
fn every_match_is_bound() {
    agree(
        &["INSERT (:M {id: 1, n: 'a'}), (:M {id: 1, n: 'b'}), (:M {id: 2, n: 'c'})"],
        &["id"],
        "UNWIND [1, 2, 1, 1] AS k MERGE (m:M {id: k}) RETURN k, m.n",
        |db, language| {
            assert_eq!(
                ints(db, "MATCH (m:M) RETURN count(m)"),
                [3],
                "{language}: nothing created"
            );
        },
    );
    for language in LANGUAGES {
        let [(_, db), _] = databases(
            &["INSERT (:M {id: 1, n: 'a'}), (:M {id: 1, n: 'b'}), (:M {id: 2, n: 'c'})"],
            &["id"],
        );
        let rows = sorted_rows(
            &db,
            language,
            "UNWIND [1, 2, 1, 1] AS k MERGE (m:M {id: k}) RETURN k, m.n",
        );
        assert_eq!(rows.len(), 7, "{language}: 2 + 1 + 2 + 2 rows");
    }
}
