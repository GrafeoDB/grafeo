//! A `MERGE` after a write in the same query sees everything the query wrote
//! before it, as clause-at-a-time semantics ask (ISO GQL, openCypher): it
//! reads its whole writing input first, as `MATCH` and `OPTIONAL MATCH` do,
//! so a row finds what a later row's earlier clause wrote instead of creating
//! it again. Here the row `i` writes the node `k: i`, and its `MERGE` looks
//! for `k: n + 1 - i`, which a later row writes for the first half of the
//! rows. A `MERGE` still sees what it created for the rows before: two rows
//! with one key create one node. For node and relationship patterns, with
//! `ON MATCH SET` and `ON CREATE SET`, with and without a property index, in
//! GQL and Cypher. These queries write, so they are no cases of the
//! differential test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test merge_after_write
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// The rows the input writes, one node `(:P {k: i})` each.
const ROWS: i64 = 300;

/// More rows than a chunk holds (2048).
const MANY_ROWS: i64 = 2100;

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// The writing input of `rows` rows, `(:P {k: i})` each, followed by `rest`,
/// in which `LAST` stands for `rows + 1`.
fn after_node_writes(language: Language, rows: i64, rest: &str) -> String {
    let rest = rest.replace("LAST", &(rows + 1).to_string());
    match language {
        Language::Gql => format!("FOR i IN range(1, {rows}) INSERT (:P {{k: i}}) WITH i {rest}"),
        Language::Cypher => {
            format!("UNWIND range(1, {rows}) AS i CREATE (:P {{k: i}}) WITH i {rest}")
        }
    }
}

/// The number of `P` nodes, and of those with a `matched` and a `created`
/// property.
const COUNT_P: &str = "MATCH (p:P) RETURN count(p), count(p.matched), count(p.created)";

/// Runs the writing input of `rows` rows followed by `rest` on a new
/// database, with and without a property index on `k`, in both languages,
/// and checks [`COUNT_P`] after it.
fn assert_merges(rows: i64, rest: &str, counts: [i64; 3]) {
    for language in LANGUAGES {
        for indexed in [false, true] {
            if rows > ROWS && !indexed {
                // A scan per row of so many rows would make the test slow.
                continue;
            }
            let db = GrafeoDB::new_in_memory();
            if indexed {
                db.create_property_index("k").unwrap();
            }
            let session = db.session();
            let query = after_node_writes(language, rows, rest);
            run(&session, language, &query);
            assert_eq!(
                run(&session, language, COUNT_P),
                [counts.map(int)],
                "{language:?} `{query}` (index: {indexed})"
            );
        }
    }
}

#[test]
fn a_merge_after_a_write_finds_the_later_rows_nodes() {
    assert_merges(ROWS, "MERGE (t:P {k: LAST - i})", [ROWS, 0, 0]);
    assert_merges(MANY_ROWS, "MERGE (t:P {k: LAST - i})", [MANY_ROWS, 0, 0]);
}

/// Every row matches, so `ON MATCH SET` runs for every node and `ON CREATE
/// SET` for none.
#[test]
fn a_merge_after_a_write_matches_with_on_match_set() {
    assert_merges(
        ROWS,
        "MERGE (t:P {k: LAST - i}) ON MATCH SET t.matched = i ON CREATE SET t.created = i",
        [ROWS, ROWS, 0],
    );
}

/// The row 1 looks for `k: LAST`, which no row writes: that one node is
/// created, the others are matched.
#[test]
fn a_merge_after_a_write_creates_only_what_no_row_wrote() {
    assert_merges(
        ROWS,
        "MERGE (t:P {k: LAST + 1 - i}) ON MATCH SET t.matched = i ON CREATE SET t.created = i",
        [ROWS + 1, ROWS - 1, 1],
    );
}

/// GQL takes a `MERGE` right after an `INSERT`.
#[test]
fn a_merge_right_after_an_insert_finds_the_later_rows_nodes() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        if indexed {
            db.create_property_index("k").unwrap();
        }
        let session = db.session();
        let query = format!(
            "FOR i IN range(1, {ROWS}) INSERT (:P {{k: i}}) MERGE (t:P {{k: {LAST} - i}}) \
             ON MATCH SET t.matched = i ON CREATE SET t.created = i",
            LAST = ROWS + 1
        );
        run(&session, Language::Gql, &query);
        assert_eq!(
            run(&session, Language::Gql, COUNT_P),
            [[int(ROWS), int(ROWS), int(0)]],
            "`{query}` (index: {indexed})"
        );
    }
}

/// A `MERGE` after a write still sees what it created for the rows before
/// its own: the two rows of key 3 create one node.
#[test]
fn a_merge_after_a_write_sees_its_own_earlier_rows() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = match language {
            Language::Gql => {
                "FOR i IN [1, 2] INSERT (:N {k: i}) WITH i \
                              FOR k IN [3, 3, 19] MERGE (:P {k: k})"
            }
            Language::Cypher => {
                "UNWIND [1, 2] AS i CREATE (:N {k: i}) WITH i \
                                 UNWIND [3, 3, 19] AS k MERGE (:P {k: k})"
            }
        };
        run(&session, language, query);
        assert_eq!(
            run(&session, language, COUNT_P),
            [[int(2), int(0), int(0)]],
            "{language:?} `{query}`"
        );
    }
}

/// The row `i` merges the hubs `k: i` and `k: i + 1` and links them, then
/// merges the link from `k: 5 - i` to `k: 6 - i`, which the row `5 - i`
/// writes: for the first rows a later one. Every link is matched.
#[test]
fn a_relationship_merge_after_a_write_finds_the_later_rows_edges() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = match language {
            Language::Gql => {
                "FOR i IN range(1, 4) MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i + 1}) \
                 INSERT (a)-[:R]->(b) WITH i MERGE (c:Hub {k: 5 - i}) MERGE (d:Hub {k: 6 - i}) \
                 MERGE (c)-[r:R]->(d) ON MATCH SET r.matched = i ON CREATE SET r.created = i"
            }
            Language::Cypher => {
                "UNWIND range(1, 4) AS i MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i + 1}) \
                 CREATE (a)-[:R]->(b) WITH i MERGE (c:Hub {k: 5 - i}) MERGE (d:Hub {k: 6 - i}) \
                 MERGE (c)-[r:R]->(d) ON MATCH SET r.matched = i ON CREATE SET r.created = i"
            }
        };
        run(&session, language, query);
        assert_eq!(
            run(
                &session,
                language,
                "MATCH ()-[r:R]->() RETURN count(r), count(r.matched), count(r.created)"
            ),
            [[int(4), int(4), int(0)]],
            "{language:?} `{query}`"
        );
        assert_eq!(
            run(&session, language, "MATCH (h:Hub) RETURN count(h)"),
            [[int(5)]],
            "{language:?} `{query}`"
        );
    }
}

/// The same links between the hubs `k: 1` to `k: 5` that are there before
/// the query, with the relationship `MERGE` right after the write of the
/// links (no `MERGE` of nodes between them).
#[test]
fn a_relationship_merge_right_after_a_create_finds_the_later_rows_edges() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run(
            &session,
            Language::Cypher,
            "UNWIND range(1, 5) AS k CREATE (:Hub {k: k})",
        );
        let (rows, insert) = match language {
            Language::Gql => ("FOR i IN range(1, 4)", "INSERT"),
            Language::Cypher => ("UNWIND range(1, 4) AS i", "CREATE"),
        };
        let query = format!(
            "{rows} MATCH (a:Hub {{k: i}}), (b:Hub {{k: i + 1}}), (c:Hub {{k: 5 - i}}), \
             (d:Hub {{k: 6 - i}}) {insert} (a)-[:R]->(b) \
             MERGE (c)-[r:R]->(d) ON MATCH SET r.matched = i ON CREATE SET r.created = i"
        );
        run(&session, language, &query);
        assert_eq!(
            run(
                &session,
                language,
                "MATCH ()-[r:R]->() RETURN count(r), count(r.matched), count(r.created)"
            ),
            [[int(4), int(4), int(0)]],
            "{language:?} `{query}`"
        );
    }
}

/// `PROFILE` runs the plan too, and writes what the query writes.
#[test]
fn a_merge_after_a_write_profiles() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = after_node_writes(language, 3, "MERGE (t:P {k: LAST - i}) RETURN count(*)");
        let rows = run(&session, language, &format!("PROFILE {query}"));
        let Value::String(text) = &rows[0][0] else {
            panic!("{language:?} `PROFILE {query}` returned {rows:?}");
        };
        assert!(
            text.lines().next().unwrap_or_default().contains("rows=1"),
            "{language:?} the top line of\n{text}"
        );
        assert_eq!(
            run(&session, language, COUNT_P),
            [[int(3), int(0), int(0)]],
            "{language:?} `PROFILE {query}`"
        );
    }
}
