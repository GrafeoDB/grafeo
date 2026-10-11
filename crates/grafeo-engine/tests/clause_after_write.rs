//! A clause after a write in the same statement reads the whole clause before
//! it, as ISO GQL and openCypher define a statement: a `RETURN`, `WITH`,
//! `WHERE`, `ORDER BY`, an aggregate or `LIMIT` after a `SET`, `REMOVE`,
//! `DELETE`, `CREATE` or `MERGE` sees what every row of the write did, not
//! only what the rows up to its own did. An `UNWIND` passes its rows on one at
//! a time, so without that the row `i` of `UNWIND range(1, 4) AS i MERGE
//! (h:Hub) SET h.c = i RETURN h.c` read `i` where every row reads 4. EXPLAIN
//! marks each clause the planner reads after a write (one per write), and
//! none in a statement that only reads. In GQL and Cypher. These queries
//! write, so they are no cases of the differential test corpus (which only
//! reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test clause_after_write
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// What EXPLAIN adds to a clause the planner reads after a write.
const MARKER: &str = "[after the write]";

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

/// The lines of `EXPLAIN query`.
fn explain(session: &Session, language: Language, query: &str) -> Vec<String> {
    let rows = run(session, language, &format!("EXPLAIN {query}"));
    let Value::String(text) = &rows[0][0] else {
        panic!("{language:?} `EXPLAIN {query}` returned {rows:?}");
    };
    text.lines().map(str::to_string).collect()
}

/// The lines of `EXPLAIN query` that carry the marker, each as the name of
/// its operator.
fn marked(session: &Session, language: Language, query: &str) -> Vec<String> {
    explain(session, language, query)
        .iter()
        .filter(|line| line.contains(MARKER))
        .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
        .collect()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// The rows `(i, found)` for `i` from 1 to 4.
fn each_row(found: Value) -> Vec<Vec<Value>> {
    (1..=4).map(|i| vec![int(i), found.clone()]).collect()
}

/// The write the tests read after: the row `i` merges the one hub and sets
/// `h.c = i`, so after it `h.c` is 4.
fn hub_write(language: Language) -> &'static str {
    match language {
        Language::Gql => "FOR i IN range(1, 4) MERGE (h:Hub) SET h.c = i",
        Language::Cypher => "UNWIND range(1, 4) AS i MERGE (h:Hub) SET h.c = i",
    }
}

/// Runs the hub write followed by each of `rests` in `languages`, each on a
/// new database, and checks the rows.
fn assert_after_hub_write(languages: &[Language], rests: &[&str], expected: &[Vec<Value>]) {
    for &language in languages {
        for rest in rests {
            let session = GrafeoDB::new_in_memory().session();
            let query = format!("{} {rest}", hub_write(language));
            assert_eq!(
                run(&session, language, &query),
                expected,
                "{language:?} `{query}`"
            );
        }
    }
}

#[test]
fn a_return_after_a_set_reads_the_whole_write() {
    assert_after_hub_write(
        &LANGUAGES,
        &["RETURN i, h.c", "WITH h, i RETURN i, h.c"],
        &each_row(int(4)),
    );
}

/// A node a `RETURN` returns is read when the row is returned: its map has
/// what the whole write left.
#[test]
fn a_returned_node_after_a_set_has_the_whole_write() {
    for language in LANGUAGES {
        let session = GrafeoDB::new_in_memory().session();
        let query = format!("{} RETURN i, h", hub_write(language));
        let counters: Vec<Option<Value>> = run(&session, language, &query)
            .iter()
            .map(|row| match &row[1] {
                Value::Map(map) => map.get(&PropertyKey::new("c")).cloned(),
                other => panic!("{language:?} `{query}` returned {other:?} for h"),
            })
            .collect();
        assert_eq!(
            counters,
            vec![Some(int(4)); 4],
            "{language:?} `{query}`: h.c of each row"
        );
    }
}

/// Every row passes `WHERE h.c = 4`, not only the last.
#[test]
fn a_where_after_a_set_reads_the_whole_write() {
    assert_after_hub_write(
        &LANGUAGES,
        &["WITH h, i WHERE h.c = 4 RETURN i, h.c"],
        &each_row(int(4)),
    );
}

#[test]
fn an_aggregate_after_a_set_reads_the_whole_write() {
    assert_after_hub_write(
        &LANGUAGES,
        &[
            "RETURN sum(h.c) AS total",
            "WITH h, i RETURN sum(h.c) AS total",
        ],
        &[vec![int(16)]],
    );
}

/// The sort reads 4 for every row, so the rows keep the order of `i`, the
/// second key.
#[test]
fn an_order_by_after_a_set_reads_the_whole_write() {
    assert_after_hub_write(
        &LANGUAGES,
        &["RETURN i, h.c ORDER BY h.c DESC, i"],
        &each_row(int(4)),
    );
    assert_after_hub_write(
        &[Language::Cypher],
        &["WITH h, i ORDER BY h.c DESC, i RETURN i, h.c"],
        &each_row(int(4)),
    );
}

/// The row `i` merges the hubs `k: i` and `k: i - 1` and removes the label of
/// the second, so the hub of each row but the last loses it to the row after
/// its own.
#[test]
fn labels_after_a_remove_read_the_whole_write() {
    let seen =
        |labels: &[&str]| Value::from(labels.iter().map(|l| Value::from(*l)).collect::<Vec<_>>());
    let expected: Vec<Vec<Value>> = (1..=4)
        .map(|i| {
            let labels = if i < 4 {
                seen(&["Hub"])
            } else {
                seen(&["Hub", "Seen"])
            };
            vec![int(i), labels]
        })
        .collect();
    for (language, rows) in [
        (Language::Gql, "FOR i IN range(1, 4)"),
        (Language::Cypher, "UNWIND range(1, 4) AS i"),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        run(
            &session,
            Language::Cypher,
            "UNWIND range(0, 4) AS k CREATE (:Hub:Seen {k: k})",
        );
        let query = format!(
            "{rows} MERGE (a:Hub {{k: i}}) MERGE (b:Hub {{k: i - 1}}) REMOVE b:Seen \
             WITH a, i RETURN i, labels(a) AS l"
        );
        assert_eq!(
            run(&session, language, &query),
            expected,
            "{language:?} `{query}`"
        );
    }
}

/// Each of the two rows deletes its own `a` and reads the name of `b`, the
/// other row's `a`: after the whole delete both are gone. The `UNWIND` passes
/// the two rows of the `MATCH` on one at a time. Cypher only: GQL takes no
/// `DELETE` after a `FOR` or a `WITH` yet; a `DELETE` right after its `MATCH`
/// reads the rows the `MATCH` passes on all at once.
#[test]
fn a_return_after_a_delete_reads_the_whole_delete() {
    for (language, query, expected) in [
        (
            Language::Cypher,
            "MATCH (a:P), (b:P) WHERE a.k <> b.k UNWIND [1] AS x DELETE a RETURN b.name AS name",
            vec![vec![Value::Null], vec![Value::Null]],
        ),
        (
            Language::Cypher,
            "MATCH (a:P), (b:P) WHERE a.k <> b.k UNWIND [1] AS x DELETE a \
             RETURN count(b.name) AS named",
            vec![vec![int(0)]],
        ),
        (
            Language::Cypher,
            "MATCH (p:P) WITH p ORDER BY p.k WITH collect(p) AS ps UNWIND [0, 1] AS i \
             WITH i, ps[i] AS a, ps[1 - i] AS b DELETE a RETURN i, b.name AS name",
            vec![vec![int(0), Value::Null], vec![int(1), Value::Null]],
        ),
        (
            Language::Gql,
            "MATCH (a:P), (b:P) WHERE a.k <> b.k DELETE a RETURN b.name AS name",
            vec![vec![Value::Null], vec![Value::Null]],
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        run(
            &session,
            Language::Cypher,
            "CREATE (:P {k: 1, name: 'Alix'}), (:P {k: 2, name: 'Gus'})",
        );
        assert_eq!(
            run(&session, language, query),
            expected,
            "{language:?} `{query}`"
        );
        assert_eq!(
            run(&session, language, "MATCH (p:P) RETURN count(p)"),
            [[int(0)]],
            "{language:?} `{query}` deletes both"
        );
    }
}

/// A later `WITH` counts every node the write created, through a `MATCH` or
/// a `COUNT` subquery (both read their whole writing input first already).
#[test]
fn a_count_of_created_nodes_in_a_later_with_sees_them_all() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i CREATE (:W) WITH i MATCH (w:W) \
             WITH i, count(w) AS c RETURN i, c ORDER BY i",
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i CREATE (:W) WITH i, COUNT { (w:W) } AS c RETURN i, c",
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 4) INSERT (:W) WITH i MATCH (w:W) RETURN i, count(w) AS c ORDER BY i",
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 4) INSERT (:W) WITH i, COUNT { MATCH (w:W) } AS c RETURN i, c",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query),
            each_row(int(4)),
            "{language:?} `{query}`"
        );
    }
}

/// A `LIMIT` (and GQL's `FINISH`, a `LIMIT 0`) after a write cuts the rows
/// after the write, not the write: all ten nodes are there.
#[test]
fn a_limit_after_a_write_keeps_every_write() {
    for (language, query, rows) in [
        (
            Language::Cypher,
            "UNWIND range(1, 10) AS i CREATE (n:L {i: i}) RETURN n.i AS i LIMIT 1",
            1,
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 10) AS i CREATE (:L {i: i}) RETURN i LIMIT 1",
            1,
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 10) AS i CREATE (:L {i: i}) WITH i LIMIT 3 RETURN i",
            3,
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 10) INSERT (:L {i: i}) RETURN i LIMIT 1",
            1,
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 10) INSERT (:L {i: i}) FINISH",
            0,
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query).len(),
            rows,
            "{language:?} `{query}`"
        );
        assert_eq!(
            run(&session, language, "MATCH (n:L) RETURN count(n)"),
            [[int(10)]],
            "{language:?} `{query}` writes every row"
        );
    }
}

/// EXPLAIN marks the clause that reads after the write: one per write, the
/// first one that reads the graph. A `WITH` of variables only passes the
/// rows on, and a clause after a marked one reads rows that are complete.
#[test]
fn explain_marks_the_clause_read_after_a_write() {
    for (language, rest, expected) in [
        (Language::Cypher, "WITH h, i RETURN i, h.c", vec!["Return"]),
        (Language::Gql, "WITH h, i RETURN i, h.c", vec!["Return"]),
        (
            Language::Cypher,
            "RETURN sum(h.c) AS total",
            vec!["Aggregate"],
        ),
        (Language::Gql, "RETURN sum(h.c) AS total", vec!["Aggregate"]),
        // The WHERE runs before the WITH's projection.
        (
            Language::Cypher,
            "WITH h, i WHERE h.c = 4 RETURN i, h.c",
            vec!["Filter"],
        ),
        (
            Language::Cypher,
            "WITH h, i ORDER BY h.c DESC, i RETURN i, h.c",
            vec!["Sort"],
        ),
        (
            Language::Cypher,
            "WITH h, i LIMIT 1 RETURN i",
            vec!["Limit"],
        ),
        // Nothing the rows read changes: the count needs no complete rows.
        (Language::Cypher, "RETURN count(*) AS rows", vec![]),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let query = format!("{} {rest}", hub_write(language));
        assert_eq!(
            marked(&session, language, &query),
            expected,
            "{language:?} `EXPLAIN {query}`:\n{}",
            explain(&session, language, &query).join("\n")
        );
    }
}

/// A join reads the whole write before its first row (an `OPTIONAL MATCH`
/// reads its writing input first and its own pattern whole), so the clause
/// after it is not read again: EXPLAIN marks nothing, and every row finds
/// the hub as the whole write left it.
#[test]
fn a_clause_after_a_join_after_a_write_is_not_read_again() {
    for language in LANGUAGES {
        let query = format!(
            "{} WITH h, i OPTIONAL MATCH (t:Hub) RETURN i, t.c AS found",
            hub_write(language)
        );
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            marked(&session, language, &query),
            Vec::<String>::new(),
            "{language:?} `EXPLAIN {query}`:\n{}",
            explain(&session, language, &query).join("\n")
        );
        assert_eq!(
            run(&session, language, &query),
            each_row(int(4)),
            "{language:?} `{query}`"
        );
    }
}

/// A `MATCH` after a write reads its whole input first already, so a `MERGE`
/// after it reads complete rows and is not marked (the `RETURN` after the
/// `MERGE` is, for the `MERGE` writes). A `MERGE` right after the write is,
/// and so is a second `MERGE` right after it (the first one writes). Every
/// `MERGE` finds the node the write created for its key.
#[test]
fn a_merge_after_a_match_after_a_write_reads_the_rows_once() {
    for (language, query, expected) in [
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i CREATE (:N {k: i}) WITH i MATCH (a:N {k: 5 - i}) \
             MERGE (b:N {k: i}) RETURN i, a.k AS ak, b.k AS bk",
            vec!["Return"],
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 4) INSERT (:N {k: i}) WITH i MATCH (a:N {k: 5 - i}) \
             MERGE (b:N {k: i}) RETURN i, a.k AS ak, b.k AS bk",
            vec!["Return"],
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i CREATE (:N {k: i}) WITH i MERGE (a:N {k: 5 - i}) \
             MERGE (b:N {k: i}) RETURN i, a.k AS ak, b.k AS bk",
            vec!["Return", "Merge", "Merge"],
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            marked(&session, language, query),
            expected,
            "{language:?} `EXPLAIN {query}`:\n{}",
            explain(&session, language, query).join("\n")
        );
        assert_eq!(
            run(&session, language, query),
            (1..=4)
                .map(|i| vec![int(i), int(5 - i), int(i)])
                .collect::<Vec<_>>(),
            "{language:?} `{query}`"
        );
        assert_eq!(
            run(&session, language, "MATCH (n:N) RETURN count(n)"),
            [[int(4)]],
            "{language:?} `{query}` merges what the write created"
        );
    }
}

/// A statement that only reads has no clause read after a write: its plan is
/// the one it had before.
#[test]
fn a_statement_that_only_reads_marks_no_clause() {
    let session = GrafeoDB::new_in_memory().session();
    run(
        &session,
        Language::Cypher,
        "UNWIND range(1, 4) AS i CREATE (:Hub {c: i})-[:R]->(:Q {i: i})",
    );
    for (language, query) in [
        (
            Language::Cypher,
            "MATCH (h:Hub) WITH h WHERE h.c > 1 RETURN h.c ORDER BY h.c",
        ),
        (
            Language::Gql,
            "MATCH (h:Hub) WITH h WHERE h.c > 1 RETURN h.c ORDER BY h.c",
        ),
        (
            Language::Cypher,
            "MATCH (h:Hub)-[:R]->(q) RETURN sum(q.i) AS total",
        ),
        (
            Language::Gql,
            "MATCH (h:Hub)-[:R]->(q) RETURN sum(q.i) AS total",
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i MATCH (h:Hub {c: i}) RETURN h LIMIT 2",
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 4) MATCH (h:Hub {c: i}) RETURN h LIMIT 2",
        ),
        (
            Language::Cypher,
            "MATCH (h:Hub) OPTIONAL MATCH (h)-[:R]->(q) RETURN h.c, q.i ORDER BY h.c",
        ),
        (
            Language::Cypher,
            "MATCH (h:Hub) CALL { WITH h MATCH (h)-[:R]->(q) RETURN q } RETURN q.i",
        ),
    ] {
        assert_eq!(
            marked(&session, language, query),
            Vec::<String>::new(),
            "{language:?} `EXPLAIN {query}`"
        );
    }
}

/// `PROFILE` runs these plans too (each on a new database, as the query
/// writes): the top line counts the rows the query returns.
#[test]
fn clauses_after_a_write_profile() {
    for language in LANGUAGES {
        for rest in [
            "RETURN i, h.c",
            "WITH h, i WHERE h.c = 4 RETURN i, h.c",
            "RETURN i, h.c ORDER BY h.c DESC, i",
        ] {
            let query = format!("{} {rest}", hub_write(language));
            let rows = run(
                &GrafeoDB::new_in_memory().session(),
                language,
                &format!("PROFILE {query}"),
            );
            let Value::String(text) = &rows[0][0] else {
                panic!("{language:?} `PROFILE {query}` returned {rows:?}");
            };
            assert!(
                text.lines().next().unwrap_or_default().contains("rows=4"),
                "{language:?} the top line of\n{text}"
            );
        }
    }
}
