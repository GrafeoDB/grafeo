//! A pattern after a write in the same query sees everything the query wrote
//! before it, as clause-at-a-time semantics ask (ISO GQL, openCypher): an
//! `OPTIONAL MATCH`, a `MATCH` that goes on from a node the writing input
//! binds (an expand, a variable-length expand, a chain of expands, a shortest
//! path, the labels and properties it checks), and a part of a `MATCH` joined
//! to another on a shared variable read the store only after the whole
//! writing input is read. An `UNWIND` passes its rows on one at a time, so
//! without that a row sees only what the rows before it wrote, or nothing.
//! In GQL and Cypher, with and without a property index. These queries
//! write, so they are no cases of the differential test corpus (which only
//! reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test pattern_after_write
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// The rows the writing input of an `OPTIONAL MATCH` writes and returns, one
/// node `(:N {k: i})` each.
const ROWS: i64 = 300;

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

/// The operator lines of `PROFILE query`, each as its operator name and
/// `rows=` count, from the top down.
fn profile(session: &Session, language: Language, query: &str) -> Vec<(String, u64)> {
    let rows = run(session, language, &format!("PROFILE {query}"));
    let Value::String(text) = &rows[0][0] else {
        panic!("{language:?} `PROFILE {query}` returned {rows:?}");
    };
    text.lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with("Total time:"))
        .map(|line| {
            let name = line.split_whitespace().next().unwrap_or("").to_string();
            let count = line
                .split("rows=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|count| count.parse().ok())
                .unwrap_or_else(|| panic!("no row count in profile line `{line}`"));
            (name, count)
        })
        .collect()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// Rows of integers (`None` for null).
fn table(rows: &[&[Option<i64>]]) -> Vec<Vec<Value>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|cell| cell.map_or(Value::Null, int))
                .collect()
        })
        .collect()
}

/// The writing input of `ROWS` rows, `(:N {k: i})` each, followed by `rest`,
/// in which `LAST` stands for `ROWS + 1`.
fn after_node_writes(language: Language, rest: &str) -> String {
    let rest = rest.replace("LAST", &(ROWS + 1).to_string());
    match language {
        Language::Gql => format!("FOR i IN range(1, {ROWS}) INSERT (:N {{k: i}}) WITH i {rest}"),
        Language::Cypher => {
            format!("UNWIND range(1, {ROWS}) AS i CREATE (:N {{k: i}}) WITH i {rest}")
        }
    }
}

/// Runs [`after_node_writes`] with `rest` on a new database, with and without
/// a property index on `k`, in both languages, and checks the rows.
fn assert_after_node_writes(rest: &str, expected: &[Vec<Value>]) {
    assert_after_node_writes_in(&LANGUAGES, rest, expected);
}

/// [`assert_after_node_writes`] in `languages` (GQL takes no `WHERE` on a
/// `MATCH` after a `WITH`, only one in the element pattern, which Cypher
/// does not take).
fn assert_after_node_writes_in(languages: &[Language], rest: &str, expected: &[Vec<Value>]) {
    for &language in languages {
        for indexed in [false, true] {
            let db = GrafeoDB::new_in_memory();
            if indexed {
                db.create_property_index("k").unwrap();
            }
            let session = db.session();
            let query = after_node_writes(language, rest);
            assert_eq!(
                run(&session, language, &query),
                expected,
                "{language:?} `{query}` (index: {indexed})"
            );
        }
    }
}

/// The row `i` looks for the node `k: 301 - i`, which a later row writes for
/// the first half of the rows.
#[test]
fn an_optional_match_sees_the_whole_write() {
    let all = table(&[&[Some(ROWS)]]);
    let first_three = table(&[
        &[Some(1), Some(ROWS)],
        &[Some(2), Some(ROWS - 1)],
        &[Some(3), Some(ROWS - 2)],
    ]);
    assert_after_node_writes(
        "OPTIONAL MATCH (t:N {k: LAST - i}) RETURN count(t) AS found",
        &all,
    );
    assert_after_node_writes_in(
        &[Language::Cypher],
        "OPTIONAL MATCH (t:N) WHERE t.k = LAST - i RETURN count(t) AS found",
        &all,
    );
    assert_after_node_writes_in(
        &[Language::Gql],
        "OPTIONAL MATCH (t:N WHERE t.k = LAST - i) RETURN count(t) AS found",
        &all,
    );
    assert_after_node_writes(
        "OPTIONAL MATCH (t:N {k: LAST - i}) RETURN i, t.k AS k ORDER BY i LIMIT 3",
        &first_three,
    );
}

/// A row whose key has no node keeps nulls, so the condition that reads the
/// row decides the matches and filters no row: the row `i` looks for `k: 302
/// - i`, which no row writes for `i = 1`.
#[test]
fn an_optional_match_after_a_write_keeps_the_rows_without_a_match() {
    let expected = table(&[
        &[Some(1), None],
        &[Some(2), Some(ROWS)],
        &[Some(3), Some(ROWS - 1)],
    ]);
    assert_after_node_writes(
        "OPTIONAL MATCH (t:N {k: LAST + 1 - i}) RETURN i, t.k AS k ORDER BY i LIMIT 3",
        &expected,
    );
    assert_after_node_writes_in(
        &[Language::Cypher],
        "OPTIONAL MATCH (t:N) WHERE t.k = LAST + 1 - i RETURN i, t.k AS k ORDER BY i LIMIT 3",
        &expected,
    );
    assert_after_node_writes_in(
        &[Language::Gql],
        "OPTIONAL MATCH (t:N WHERE t.k = LAST + 1 - i) RETURN i, t.k AS k ORDER BY i LIMIT 3",
        &expected,
    );
}

/// GQL takes an `OPTIONAL MATCH` right after an `INSERT`: the condition that
/// reads the row (a variable of the `FOR`, or a property of the inserted
/// node) decides the matches there too.
#[test]
fn an_optional_match_right_after_an_insert_sees_the_whole_write() {
    let first_three = table(&[
        &[Some(1), Some(ROWS)],
        &[Some(2), Some(ROWS - 1)],
        &[Some(3), Some(ROWS - 2)],
    ]);
    for key in ["LAST - i", "LAST - n.k"] {
        let key = key.replace("LAST", &(ROWS + 1).to_string());
        for indexed in [false, true] {
            let db = GrafeoDB::new_in_memory();
            if indexed {
                db.create_property_index("k").unwrap();
            }
            let session = db.session();
            let query = format!(
                "FOR i IN range(1, {ROWS}) INSERT (n:N {{k: i}})                  OPTIONAL MATCH (t:N {{k: {key}}}) RETURN i, t.k AS k ORDER BY i LIMIT 3"
            );
            assert_eq!(
                run(&session, Language::Gql, &query),
                first_three,
                "`{query}` (index: {indexed})"
            );
        }
    }
}

/// A `CALL` subquery that writes passes the rows of a `WITH` on to the
/// `OPTIONAL MATCH` after it, which sees all of what the subquery wrote for
/// every row (Cypher: GQL takes no `WITH` after a `CALL`).
#[test]
fn an_optional_match_after_a_writing_call_sees_the_whole_write() {
    let first_three = table(&[
        &[Some(1), Some(ROWS)],
        &[Some(2), Some(ROWS - 1)],
        &[Some(3), Some(ROWS - 2)],
    ]);
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        if indexed {
            db.create_property_index("k").unwrap();
        }
        let session = db.session();
        let query = format!(
            "UNWIND range(1, {ROWS}) AS i CALL {{ WITH i CREATE (:N {{k: i}}) }} WITH i              OPTIONAL MATCH (t:N {{k: {LAST} - i}}) RETURN i, t.k AS k ORDER BY i LIMIT 3",
            LAST = ROWS + 1
        );
        assert_eq!(
            run(&session, Language::Cypher, &query),
            first_three,
            "`{query}` (index: {indexed})"
        );
    }
}

/// An `OPTIONAL MATCH` after a write that an aggregate reads whole still
/// sees the write: its own pattern is read before its input otherwise.
#[test]
fn an_optional_match_after_an_aggregated_write_sees_it() {
    for (language, write) in [
        (Language::Gql, "FOR k IN range(1, 3) INSERT (:N {k: k})"),
        (
            Language::Cypher,
            "UNWIND range(1, 3) AS k CREATE (:N {k: k})",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "{write} WITH count(*) AS c UNWIND range(1, 3) AS i \
             OPTIONAL MATCH (t:N {{k: 4 - i}}) RETURN i, t.k AS k ORDER BY i"
        );
        assert_eq!(
            run(&session, language, &query),
            table(&[
                &[Some(1), Some(3)],
                &[Some(2), Some(2)],
                &[Some(3), Some(1)]
            ]),
            "{language:?} `{query}`"
        );
    }
}

/// The hub write: the row `i` merges the one hub and links a new `(:Q {i:
/// i})` with a `(:T)` behind it to the hub, so every row after the `WITH`
/// sees four of each.
fn hub_write(language: Language) -> &'static str {
    match language {
        Language::Gql => {
            "FOR i IN range(1, 4) MERGE (h:Hub) INSERT (h)-[:R]->(:Q {i: i})-[:S]->(:T) WITH h, i"
        }
        Language::Cypher => {
            "UNWIND range(1, 4) AS i MERGE (h:Hub) CREATE (h)-[:R]->(:Q {i: i})-[:S]->(:T) \
             WITH h, i"
        }
    }
}

/// Runs the hub write followed by each `rest` in `languages` on a new
/// database (with or without factorized execution), and checks that every
/// row `i` from 1 to 4 has `found` in its second column.
fn assert_after_hub_writes(languages: &[Language], rests: &[&str], found: i64, factorized: bool) {
    let expected: Vec<Vec<Value>> = (1..=4).map(|i| vec![int(i), int(found)]).collect();
    for &language in languages {
        for rest in rests {
            let config = if factorized {
                Config::in_memory()
            } else {
                Config::in_memory().without_factorized_execution()
            };
            let db = GrafeoDB::with_config(config).unwrap();
            let session = db.session();
            let query = format!("{} {rest}", hub_write(language));
            assert_eq!(
                run(&session, language, &query),
                expected,
                "{language:?} `{query}` (factorized: {factorized})"
            );
        }
    }
}

#[test]
fn an_expand_from_a_written_node_sees_the_whole_write() {
    assert_after_hub_writes(
        &LANGUAGES,
        &[
            "MATCH (h)-[:R]->(q) RETURN i, count(q) AS found ORDER BY i",
            "MATCH (h)-[:R]->(q:Q) RETURN i, count(q) AS found ORDER BY i",
            "MATCH (h)-[e:R]->(q) RETURN i, count(e) AS found ORDER BY i",
            "MATCH (h)-[:R]->(q) RETURN i, sum(q.i) - 6 AS found ORDER BY i",
        ],
        4,
        true,
    );
    // GQL takes no `WHERE` on a `MATCH` after a `WITH` (only one in the
    // element pattern, which Cypher does not take), and no `MATCH` after a
    // `WITH` that does not go on from it.
    assert_after_hub_writes(
        &[Language::Gql],
        &["MATCH (h)-[:R]->(q WHERE q.i > 0) RETURN i, count(q) AS found ORDER BY i"],
        4,
        true,
    );
    assert_after_hub_writes(
        &[Language::Cypher],
        &[
            "MATCH (h)-[:R]->(q) WHERE q.i > 0 RETURN i, count(q) AS found ORDER BY i",
            "MATCH (h)-[:R]->(q) WITH i, count(q) AS found RETURN i, found ORDER BY i",
        ],
        4,
        true,
    );
}

/// A variable-length expand and a chain of expands read their whole input
/// first on their own (with factorized execution off, the chain is two
/// expands): four `Q` at one hop and four `T` at two.
#[test]
fn a_longer_pattern_from_a_written_node_sees_the_whole_write() {
    for factorized in [true, false] {
        assert_after_hub_writes(
            &LANGUAGES,
            &["MATCH (h)-[*1..2]->(q) RETURN i, count(q) AS found ORDER BY i"],
            8,
            factorized,
        );
        assert_after_hub_writes(
            &LANGUAGES,
            &["MATCH (h)-[]->(q)-[]->(t) RETURN i, count(t) AS found ORDER BY i"],
            4,
            factorized,
        );
    }
}

/// An aggregate with no group key straight over a chain of expands from a
/// written node (which the planner may run factorized, without the rows of
/// the chain) counts what the whole write left: each of the four rows
/// reaches four `T`, so the count is sixteen, with factorized execution on
/// and off.
#[test]
fn an_aggregate_over_a_chain_from_a_written_node_counts_the_whole_write() {
    for factorized in [true, false] {
        for language in LANGUAGES {
            for rest in [
                "MATCH (h)-[:R]->(q)-[:S]->(t) RETURN count(*) AS found",
                "MATCH (h)-[:R]->(q)-[:S]->(t) RETURN count(t) AS found",
                "MATCH (h)-[]->(q)-[]->(t) RETURN count(*) AS found",
                "MATCH (h)-[:R]->(q)-[:S]->(t) WITH count(*) AS found RETURN found",
                "MATCH (h)-[:R]->(q) RETURN count(*) AS found",
            ] {
                let config = if factorized {
                    Config::in_memory()
                } else {
                    Config::in_memory().without_factorized_execution()
                };
                let db = GrafeoDB::with_config(config).unwrap();
                let session = db.session();
                let query = format!("{} {rest}", hub_write(language));
                assert_eq!(
                    run(&session, language, &query),
                    vec![vec![int(16)]],
                    "{language:?} `{query}` (factorized: {factorized})"
                );
            }
        }
    }
}

/// An `OPTIONAL MATCH` from a written node, and a part of a `MATCH` joined
/// to another on a shared variable, see the whole write too.
#[test]
fn a_joined_pattern_from_a_written_node_sees_the_whole_write() {
    assert_after_hub_writes(
        &LANGUAGES,
        &[
            "OPTIONAL MATCH (h)-[:R]->(q:Q {i: 4}) RETURN i, q.i AS found ORDER BY i",
            "MATCH (h)-[:R]->(q), (q)<-[:R]-(g) RETURN i, count(*) AS found ORDER BY i",
        ],
        4,
        true,
    );
}

/// The row `i` merges the hubs `k: i - 1` and `k: i` and links them, so the
/// path from the hub `k: i - 1` of the row to the last hub `k: 4` has `5 - i`
/// edges, all but one written by later rows.
#[test]
fn a_shortest_path_from_a_written_node_sees_the_whole_write() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i MERGE (a:Hub {k: i - 1}) MERGE (b:Hub {k: i}) \
             MERGE (z:Hub {k: 4}) CREATE (a)-[:R]->(b) WITH a, z, i \
             MATCH p = shortestPath((a)-[*]->(z)) RETURN i, length(p) AS hops ORDER BY i",
        ),
        (
            Language::Gql,
            "FOR i IN range(1, 4) MERGE (a:Hub {k: i - 1}) MERGE (b:Hub {k: i}) \
             MERGE (z:Hub {k: 4}) INSERT (a)-[:R]->(b) WITH a, z, i \
             MATCH p = ANY SHORTEST (a)-[*]->(z) RETURN i, length(p) AS hops ORDER BY i",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        assert_eq!(
            run(&session, language, query),
            (1..=4)
                .map(|i| vec![int(i), int(5 - i)])
                .collect::<Vec<_>>(),
            "{language:?} `{query}`"
        );
    }
}

/// The row `i` merges the hubs `k: i` and `k: i - 1` and marks the second,
/// so the hub of each row but the last is marked by the row after it: the
/// labels and properties a `MATCH` checks on a written node are those of the
/// whole write.
#[test]
fn the_checks_on_a_written_node_see_the_whole_write() {
    let marked = table(&[&[Some(1)], &[Some(2)], &[Some(3)]]);
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN range(1, 4) MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i - 1}) \
             SET b:Seen, b.seen = true WITH a, i",
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i - 1}) \
             SET b:Seen, b.seen = true WITH a, i",
        ),
    ] {
        for rest in [
            "MATCH (a:Seen) RETURN i ORDER BY i",
            "MATCH (a {seen: true}) RETURN i ORDER BY i",
            "MATCH (a:Hub:Seen) RETURN i ORDER BY i",
        ] {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            let query = format!("{write} {rest}");
            assert_eq!(
                run(&session, language, &query),
                marked,
                "{language:?} `{query}`"
            );
        }
    }
}

/// More rows than a chunk holds (2048) come from a scan in chunks, each
/// written whole before it is passed on: the `MATCH` after the write sees
/// the edges of every chunk, not only those of the chunks before.
#[test]
fn a_match_after_a_write_of_more_rows_than_a_chunk_sees_them_all() {
    const SOURCES: i64 = 2100;
    for (language, query) in [
        (
            Language::Gql,
            "MATCH (s:S) MERGE (h:Hub) INSERT (h)-[:R]->(:Q {i: s.i}) WITH h, s \
             WHERE s.i <= 3 MATCH (h)-[:R]->(q) RETURN s.i AS i, count(q) AS found ORDER BY i",
        ),
        (
            Language::Cypher,
            "MATCH (s:S) MERGE (h:Hub) CREATE (h)-[:R]->(:Q {i: s.i}) WITH h, s \
             WHERE s.i <= 3 MATCH (h)-[:R]->(q) RETURN s.i AS i, count(q) AS found ORDER BY i",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute_cypher(&format!(
                "UNWIND range(1, {SOURCES}) AS i CREATE (:S {{i: i}})"
            ))
            .unwrap();
        assert_eq!(
            run(&session, language, query),
            (1..=3)
                .map(|i| vec![int(i), int(SOURCES)])
                .collect::<Vec<_>>(),
            "{language:?} `{query}`"
        );
    }
}

/// The write in which the row `i` sets `c: i` on the one hub, so that after
/// it the hub has `c: 4`.
fn hub_key_write(language: Language) -> &'static str {
    match language {
        Language::Gql => "FOR i IN range(1, 4) MATCH (h:Hub) SET h.c = i WITH h, i",
        Language::Cypher => "UNWIND range(1, 4) AS i MATCH (h:Hub) SET h.c = i WITH h, i",
    }
}

/// A pattern with a filter on constants after a write finds what the whole
/// write left, as one that reads the row does: every row `i` finds the hub
/// with `c: 4`, which the last row writes. An `OPTIONAL MATCH` (with a
/// property map, a `WHERE` of an equality, an `IN` list or a range, with and
/// without a label), the part of a `MATCH` joined to another on a shared
/// variable, a `CALL` subquery and a `COUNT` or `EXISTS` subquery used to
/// find it in none of the rows: the planner looked the hub up while
/// planning, before the write (by its label, an index on `c`, or the zone
/// map of `c`, which knew only `c: 3`). With and without an index on `c`,
/// and with the hub without `c` or with `c: 3` before the query.
#[test]
fn a_constant_filter_after_a_write_sees_the_whole_write() {
    let mut queries = Vec::new();
    for language in LANGUAGES {
        for rest in [
            "OPTIONAL MATCH (t:Hub {c: 4}) RETURN i, t.c AS found ORDER BY i",
            "OPTIONAL MATCH (t {c: 4}) RETURN i, t.c AS found ORDER BY i",
            "MATCH (t:Hub {c: 4}) RETURN i, t.c AS found ORDER BY i",
            "MATCH (h)-[:R]->(q), (t:Hub {c: 4})-[:R]->(q) RETURN i, t.c AS found ORDER BY i",
            "RETURN i, COUNT { MATCH (t:Hub {c: 4}) } + 3 AS found ORDER BY i",
            "WITH i WHERE EXISTS { MATCH (t:Hub {c: 4}) } RETURN i, 4 AS found ORDER BY i",
            // A subquery tied to the row by `h`, whose part joined on `q`
            // has the filter on constants.
            "WITH h, i WHERE EXISTS { MATCH (h)-[:R]->(q), (t:Hub {c: 4})-[:R]->(q) } \
             RETURN i, 4 AS found ORDER BY i",
        ] {
            queries.push((language, format!("{} {rest}", hub_key_write(language))));
        }
    }
    for rest in [
        "OPTIONAL MATCH (t:Hub) WHERE t.c = 4 RETURN i, t.c AS found ORDER BY i",
        "OPTIONAL MATCH (t:Hub) WHERE t.c IN [4, 88] RETURN i, t.c AS found ORDER BY i",
        "OPTIONAL MATCH (t:Hub) WHERE t.c > 3 RETURN i, t.c AS found ORDER BY i",
        "CALL { MATCH (t:Hub {c: 4}) RETURN t.c AS found } RETURN i, found ORDER BY i",
        "CALL { WITH h MATCH (h)-[:R]->(q), (t:Hub {c: 4})-[:R]->(q) RETURN t.c AS found } \
         RETURN i, found ORDER BY i",
    ] {
        let write = hub_key_write(Language::Cypher);
        queries.push((Language::Cypher, format!("{write} {rest}")));
    }
    for rest in [
        "OPTIONAL MATCH (t:Hub WHERE t.c = 4) RETURN i, t.c AS found ORDER BY i",
        "OPTIONAL MATCH (t:Hub WHERE t.c IN [4, 88]) RETURN i, t.c AS found ORDER BY i",
        "OPTIONAL MATCH (t:Hub WHERE t.c > 3) RETURN i, t.c AS found ORDER BY i",
    ] {
        let write = hub_key_write(Language::Gql);
        queries.push((Language::Gql, format!("{write} {rest}")));
    }
    // GQL takes a `CALL` after a `MERGE`, not after a `SET` or `WITH`.
    queries.push((
        Language::Gql,
        concat!(
            "FOR i IN range(1, 4) MERGE (h:Hub) ON MATCH SET h.c = i ",
            "CALL () { MATCH (t:Hub {c: 4}) RETURN t.c AS found } RETURN i, found ORDER BY i"
        )
        .to_string(),
    ));
    let expected: Vec<Vec<Value>> = (1..=4).map(|i| vec![int(i), int(4)]).collect();
    let mut wrong = Vec::new();
    for (language, query) in &queries {
        for setup in [
            "CREATE (:Hub)-[:R]->(:Q)",
            "CREATE (:Hub {c: 3})-[:R]->(:Q)",
        ] {
            for indexed in [false, true] {
                let db = GrafeoDB::new_in_memory();
                if indexed {
                    db.create_property_index("c").unwrap();
                }
                let session = db.session();
                session.execute_cypher(setup).unwrap();
                let rows = run(&session, *language, query);
                if rows != expected {
                    wrong.push(format!(
                        "{language:?} `{query}` after `{setup}` (index: {indexed}): {rows:?}"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "expected {expected:?} from:
{}",
        wrong.join(
            "
"
        )
    );
}

/// With an index on the key, a filter on a constant key after a write still
/// looks the key up in the index, when it runs instead of while planning:
/// `PROFILE` shows a seek, which finds the hub for each of the four rows, and
/// `EXPLAIN` the index on the filter, in an `OPTIONAL MATCH` and a `CALL`
/// subquery. A key the planner does not look up while planning, a call of a
/// function, takes the seek too.
#[test]
fn a_constant_key_after_a_write_is_looked_up_when_it_runs() {
    for (language, rest) in [
        (
            Language::Gql,
            "OPTIONAL MATCH (t:Hub {c: 4}) RETURN i, t.c AS found ORDER BY i",
        ),
        (
            Language::Cypher,
            "OPTIONAL MATCH (t:Hub {c: 4}) RETURN i, t.c AS found ORDER BY i",
        ),
        (
            Language::Gql,
            "OPTIONAL MATCH (t:Hub WHERE t.c = toInteger('4')) RETURN i, t.c AS found ORDER BY i",
        ),
        (
            Language::Cypher,
            "OPTIONAL MATCH (t:Hub) WHERE t.c = toInteger('4') RETURN i, t.c AS found ORDER BY i",
        ),
        (
            Language::Cypher,
            "CALL { MATCH (t:Hub) WHERE t.c = toInteger('4') RETURN t.c AS found } \
             RETURN i, found ORDER BY i",
        ),
    ] {
        let query = format!("{} {rest}", hub_key_write(language));
        let session = |db: &GrafeoDB| {
            db.create_property_index("c").unwrap();
            let session = db.session();
            session.execute_cypher("CREATE (:Hub)-[:R]->(:Q)").unwrap();
            session
        };
        let db = GrafeoDB::new_in_memory();
        let plan = run(&session(&db), language, &format!("EXPLAIN {query}"));
        let Value::String(plan) = &plan[0][0] else {
            panic!("{language:?} `EXPLAIN {query}` returned {plan:?}");
        };
        assert!(
            plan.contains("[index: c]"),
            "{language:?} `EXPLAIN {query}`:\n{plan}"
        );
        let db = GrafeoDB::new_in_memory();
        let lines = profile(&session(&db), language, &query);
        assert!(
            lines.iter().any(|(name, _)| name == "NodeSeek"),
            "{language:?} `PROFILE {query}`: {lines:?}"
        );
        let db = GrafeoDB::new_in_memory();
        assert_eq!(
            run(&session(&db), language, &query),
            (1..=4).map(|i| vec![int(i), int(4)]).collect::<Vec<_>>(),
            "{language:?} `{query}`"
        );
    }
}

/// `PROFILE` runs these plans too (each on a new database, as the query
/// writes): every operator has its line, and the top line counts the rows the
/// query returns, which are those it returns without `PROFILE`.
#[test]
fn patterns_after_a_write_profile() {
    for language in LANGUAGES {
        for rest in [
            "MATCH (h)-[:R]->(q) RETURN i, count(q) AS found ORDER BY i",
            "OPTIONAL MATCH (h)-[:R]->(q:Q {i: 4}) RETURN i, q.i AS found ORDER BY i",
            "MATCH (h)-[:R]->(q), (q)<-[:R]-(g) RETURN i, count(*) AS found ORDER BY i",
        ] {
            let query = format!("{} {rest}", hub_write(language));
            let lines = profile(&GrafeoDB::new_in_memory().session(), language, &query);
            assert_eq!(
                lines.first().map(|(_, rows)| *rows),
                Some(4),
                "{language:?} `PROFILE {query}`: {lines:?}"
            );
            assert_eq!(
                run(&GrafeoDB::new_in_memory().session(), language, &query),
                (1..=4).map(|i| vec![int(i), int(4)]).collect::<Vec<_>>(),
                "{language:?} `{query}` after its profile"
            );
        }
    }
}
