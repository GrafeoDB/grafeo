//! A statement that writes reads the input of every join below the write in
//! full first (#677, part 1): each row sees the graph as it was when the
//! statement started, as ISO GQL and openCypher define a statement. Before,
//! rows the join produced after the first write already saw it, so a SET
//! that moved a joined key matched the moved nodes again, a DELETE hid the
//! nodes later rows should have matched, and a CREATE fed the join new rows.
//!
//! Every input here is long (3000 rows) so that the join produces its rows
//! in more than one chunk, after the first writes.

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

fn run(db: &GrafeoDB, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => db.execute(query),
        Language::Cypher => db.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn count(db: &GrafeoDB, language: Language, query: &str) -> i64 {
    match run(db, language, query).as_slice() {
        [row] => row[0].as_int64().unwrap(),
        rows => panic!("{language:?} `{query}`: {rows:?}"),
    }
}

/// The sorted `k` of every `:T` node.
fn keys(db: &GrafeoDB) -> Vec<i64> {
    let mut keys: Vec<i64> = run(db, Language::Gql, "MATCH (t:T) RETURN t.k")
        .iter()
        .map(|row| row[0].as_int64().unwrap())
        .collect();
    keys.sort_unstable();
    keys
}

/// Ten `:T` nodes, `k` 0 to 9, with a property index on `k` when `indexed`.
fn ten(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 9) AS k INSERT (:T {k: k})")
        .unwrap();
    if indexed {
        db.create_property_index("k").unwrap();
    }
    db
}

/// A SET above a join that moves the joined key: each of the ten nodes
/// matches once, for the row with its original key, and moves once. The
/// rows from 2048 on must not meet the moved nodes again.
#[test]
fn a_set_above_a_join_changes_each_row_once() {
    for indexed in [false, true] {
        for language in LANGUAGES {
            let db = ten(indexed);
            let query = "UNWIND range(0, 2999) AS i MATCH (t:T) WHERE t.k = i \
                         SET t.k = t.k + 2048 RETURN count(*) AS n";
            assert_eq!(
                count(&db, language, query),
                10,
                "{language:?} indexed={indexed}"
            );
            assert_eq!(
                keys(&db),
                (2048..2058).collect::<Vec<_>>(),
                "{language:?} indexed={indexed}: each node moved once"
            );
        }
    }
}

/// The same with the key in a pattern property map, which an index on `k`
/// answers with a seek per row.
#[test]
fn a_set_above_an_index_seek_changes_each_row_once() {
    for language in LANGUAGES {
        let db = ten(true);
        let query = "UNWIND range(0, 2999) AS i MATCH (t:T {k: i}) \
                     SET t.k = t.k + 2048 RETURN count(*) AS n";
        assert_eq!(count(&db, language, query), 10, "{language:?}");
        assert_eq!(keys(&db), (2048..2058).collect::<Vec<_>>(), "{language:?}");
    }
}

/// A join of two patterns over the same nodes, with the SET on one side:
/// every pair that matched at the start is changed, and only those.
#[test]
fn a_set_above_a_join_of_two_patterns_sees_the_start() {
    for indexed in [false, true] {
        for language in LANGUAGES {
            let db = ten(indexed);
            // Pairs (a, b) with b.k = a.k + 1: nine at the start. Moving b by
            // 100 must not make new pairs or lose the pairs already found.
            let query = "MATCH (a:T) MATCH (b:T) WHERE b.k = a.k + 1 \
                         SET b.k = b.k + 100 RETURN count(*) AS n";
            assert_eq!(
                count(&db, language, query),
                9,
                "{language:?} indexed={indexed}"
            );
            let mut expected: Vec<i64> = std::iter::once(0).chain(101..110).collect();
            expected.sort_unstable();
            assert_eq!(keys(&db), expected, "{language:?} indexed={indexed}");
        }
    }
}

/// A DELETE whose join reads the nodes it deletes: every row that matched at
/// the start is a row of the result, though its node was deleted by an
/// earlier row (deleting a deleted node does nothing).
#[test]
fn a_delete_above_a_join_sees_the_start() {
    for indexed in [false, true] {
        for language in LANGUAGES {
            let db = ten(indexed);
            let query = "UNWIND range(0, 2999) AS i MATCH (t:T) WHERE t.k = i % 10 \
                         DETACH DELETE t RETURN count(*) AS n";
            assert_eq!(
                count(&db, language, query),
                3000,
                "{language:?} indexed={indexed}: every row matched at the start"
            );
            assert_eq!(
                keys(&db),
                Vec::<i64>::new(),
                "{language:?} indexed={indexed}"
            );
        }
    }
}

/// A CREATE whose join reads the label it creates: the rows are those of
/// the start, so the new nodes feed no further rows.
#[test]
fn a_create_above_a_join_sees_the_start() {
    for indexed in [false, true] {
        for language in LANGUAGES {
            let db = ten(indexed);
            let query = match language {
                Language::Gql => {
                    "UNWIND range(0, 2999) AS i MATCH (t:T) WHERE t.k = i \
                     INSERT (:T {k: i + 10}) RETURN count(*) AS n"
                }
                Language::Cypher => {
                    "UNWIND range(0, 2999) AS i MATCH (t:T) WHERE t.k = i \
                     CREATE (:T {k: i + 10}) RETURN count(*) AS n"
                }
            };
            assert_eq!(
                count(&db, language, query),
                10,
                "{language:?} indexed={indexed}"
            );
            assert_eq!(
                keys(&db),
                (0..20).collect::<Vec<_>>(),
                "{language:?} indexed={indexed}: ten created, no more"
            );
        }
    }
}

/// The control: a write with no join below it reads its rows as before, and
/// its plan gets no step that reads them first.
#[test]
fn a_write_without_a_join_keeps_its_plan() {
    for language in LANGUAGES {
        let db = ten(false);
        let query = "MATCH (t:T) SET t.k = t.k + 1 RETURN count(*) AS n";
        let plan = run(&db, language, &format!("EXPLAIN {query}"))
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !plan.contains("Eager"),
            "{language:?}: no eager step without a join:\n{plan}"
        );
        assert_eq!(count(&db, language, query), 10, "{language:?}");
        assert_eq!(keys(&db), (1..11).collect::<Vec<_>>(), "{language:?}");
    }
}
