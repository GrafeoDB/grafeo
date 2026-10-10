//! `PROFILE` of a query that starts without `MATCH`: `RETURN 1`, a first
//! `WITH`, `OPTIONAL MATCH` or `CALL { ... }`, an aggregate (`RETURN
//! count(*)`). Such a query starts from the one empty row of the logical
//! plan's `Empty`, which the planner replaces by a single row; `PROFILE` walks
//! the logical plan and needs a profile entry for that `Empty` too (it
//! panicked when the planner recorded none, and an aggregate failed with
//! "Empty plan" even without `PROFILE`). In GQL and Cypher, the profile has
//! the single row as its leaf, and its top line counts the rows the query
//! returns.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test profile_without_match
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::panic::{AssertUnwindSafe, catch_unwind};

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// Alix (`A`), who knows Gus (`B`).
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:A {name: 'Alix'})-[:KNOWS]->(:B {name: 'Gus'})")
        .unwrap();
    db
}

fn execute(
    session: &Session,
    language: Language,
    query: &str,
) -> grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult> {
    match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    }
}

/// The text `PROFILE query` returns, after checking it is one `profile` row.
fn profile(session: &Session, language: Language, query: &str) -> String {
    let result = execute(session, language, &format!("PROFILE {query}"))
        .unwrap_or_else(|error| panic!("{language:?} `PROFILE {query}` failed: {error}"));
    assert_eq!(
        result.columns,
        ["profile"],
        "{language:?} `PROFILE {query}`"
    );
    assert_eq!(result.rows().len(), 1, "{language:?} `PROFILE {query}`");
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language:?} `PROFILE {query}` returned {other:?}"),
    }
}

/// The operator lines of a profile (without the total time), each as its
/// indentation depth, operator name and `rows=` count.
fn operators(profile: &str) -> Vec<(usize, String, u64)> {
    profile
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with("Total time:"))
        .map(|line| {
            let depth = (line.len() - line.trim_start().len()) / 2;
            let name = line
                .trim_start()
                .split(' ')
                .next()
                .unwrap_or("")
                .to_string();
            let rows = line
                .split("rows=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|count| count.parse().ok())
                .unwrap_or_else(|| panic!("no row count in profile line `{line}`"));
            (depth, name, rows)
        })
        .collect()
}

/// `PROFILE query` in `languages`: the operator names from the top down, with
/// the depth and the row count of each line.
fn assert_profile(languages: &[Language], query: &str, expected: &[(usize, &str, u64)]) {
    for &language in languages {
        let db = people();
        let session = db.session();
        let text = profile(&session, language, query);
        let lines = operators(&text);
        let expected: Vec<(usize, String, u64)> = expected
            .iter()
            .map(|(depth, name, rows)| (*depth, (*name).to_string(), *rows))
            .collect();
        assert_eq!(lines, expected, "{language:?} `PROFILE {query}`:\n{text}");
    }
}

#[test]
fn profile_of_a_standalone_return() {
    assert_profile(
        &LANGUAGES,
        "RETURN 1 AS x",
        &[(0, "Project", 1), (1, "SingleRow", 1)],
    );
}

#[test]
fn profile_of_a_first_optional_match() {
    assert_profile(
        &LANGUAGES,
        "OPTIONAL MATCH (a:A) RETURN a.name",
        &[
            (0, "Project", 1),
            (1, "HashJoin", 1),
            (2, "SingleRow", 1),
            (2, "Scan", 1),
        ],
    );
}

/// The scan after the `WITH` joins each of its rows (the `NestedLoopJoin` is
/// the scan's line).
#[test]
fn profile_of_a_first_with_before_a_match() {
    // GQL starts a query with FOR, not WITH (see the UNWIND forms below).
    assert_profile(
        &[Language::Cypher],
        "WITH 1 AS x MATCH (b:B) RETURN x, b.name",
        &[
            (0, "Project", 1),
            (1, "NestedLoopJoin", 1),
            (2, "Project", 1),
            (3, "SingleRow", 1),
        ],
    );
}

/// The outer `RETURN x` passes the `Apply`'s column on, so its line names the
/// `Apply` it reads.
#[test]
fn profile_of_a_first_call_subquery() {
    assert_profile(
        &LANGUAGES,
        "CALL { RETURN 1 AS x } RETURN x",
        &[
            (0, "Apply", 1),
            (1, "Apply", 1),
            (2, "SingleRow", 1),
            (2, "Project", 1),
            (3, "SingleRow", 1),
        ],
    );
}

#[test]
fn profile_of_a_first_unwind() {
    for (language, query) in [
        (Language::Cypher, "UNWIND [3, 19, 88] AS x RETURN x"),
        (Language::Gql, "FOR x IN [3, 19, 88] RETURN x"),
    ] {
        assert_profile(
            &[language],
            query,
            &[(0, "Project", 3), (1, "Unwind", 3), (2, "SingleRow", 1)],
        );
    }
}

/// A `MERGE` that comes first runs once without an input row, unless it
/// computes a property, which it does on the one empty row.
#[test]
fn profile_of_a_first_merge() {
    assert_profile(
        &LANGUAGES,
        "MERGE (m:A {name: 'Alix'}) RETURN m.name",
        &[(0, "Project", 1), (1, "Merge", 1), (2, "Empty", 0)],
    );
    assert_profile(
        &[Language::Cypher],
        "MERGE (m:A {name: toString('Alix')}) RETURN m.name",
        &[(0, "Project", 1), (1, "Merge", 1), (2, "SingleRow", 1)],
    );
}

/// An aggregate that comes first aggregates the one empty row. The `RETURN`
/// passes the aggregate's columns on, so its line names the aggregate it
/// reads, as after a `MATCH`.
#[test]
fn profile_of_a_standalone_aggregate() {
    for query in ["RETURN count(*) AS c", "RETURN 3 AS x, count(*) AS c"] {
        let aggregate = if query.contains("x,") {
            "HashAggregate"
        } else {
            "SimpleAggregate"
        };
        assert_profile(
            &LANGUAGES,
            query,
            &[(0, aggregate, 1), (1, aggregate, 1), (2, "SingleRow", 1)],
        );
    }
}

/// Queries that start without `MATCH`, for both languages, or for one when
/// the other has no such clause.
fn queries_without_match() -> Vec<(Language, &'static str)> {
    let both = [
        "RETURN 1 AS x",
        "RETURN 3 AS x, 'Alix' AS name",
        "RETURN DISTINCT 3 AS x",
        "RETURN 3 AS x ORDER BY x",
        "RETURN 3 AS x ORDER BY x + 1",
        "RETURN 3 AS x SKIP 0 LIMIT 1",
        "RETURN [x IN [3, 19] | x + 1] AS xs",
        "RETURN 1 AS x UNION RETURN 2 AS x",
        "RETURN 1 AS x UNION ALL RETURN 1 AS x",
        "UNWIND [3, 19, 88] AS x RETURN x",
        "UNWIND [3, 19, 88] AS x WITH x WHERE x > 3 RETURN x",
        "UNWIND [3, 19] AS x RETURN sum(x) AS total",
        "UNWIND [3, 19] AS x MATCH (a:A) RETURN x, a.name",
        "OPTIONAL MATCH (a:A) RETURN a.name",
        "OPTIONAL MATCH (z:Missing) RETURN z",
        "OPTIONAL MATCH (a:A)-[:KNOWS]->(b) RETURN a.name, b.name",
        "OPTIONAL MATCH (a:A) WITH a MATCH (b:B) RETURN a.name, b.name",
        "OPTIONAL MATCH (a:A) OPTIONAL MATCH (b:B) RETURN a.name, b.name",
        "OPTIONAL MATCH (a:A) WHERE a.name = 'Gus' RETURN a",
        "CALL { RETURN 1 AS x } RETURN x",
        "CALL { MATCH (a:A) RETURN a.name AS name } RETURN name",
        "CALL { OPTIONAL MATCH (a:A) RETURN a.name AS name } RETURN name",
        "CALL { RETURN 1 AS x UNION RETURN 2 AS x } RETURN x",
        "CALL { RETURN 1 AS x } CALL { RETURN 2 AS y } RETURN x, y",
        "RETURN EXISTS { MATCH (a:A) } AS found",
        "RETURN COUNT { MATCH (a:A) } AS found",
        "RETURN EXISTS { MATCH (a:A)-[:KNOWS]->(b:B) } AS found",
        "MERGE (m:A {name: 'Alix'}) RETURN m.name",
        "MERGE (m:C {name: 'Mia'}) RETURN m.name",
        "RETURN count(*)",
        "RETURN count(*) AS c",
        "RETURN sum(3) AS s, avg(19) AS a, min(88) AS m",
        "RETURN collect(3) AS l",
        "RETURN size(collect(3)) AS s",
        "RETURN count(3) + 1 AS c",
        "RETURN count(DISTINCT 3) AS c",
        "RETURN 3 AS x, count(*) AS c",
        "RETURN count(*) AS c ORDER BY c",
        "RETURN DISTINCT count(*) AS c",
        "RETURN count(*) AS c SKIP 0 LIMIT 1",
        "RETURN max(3) AS m UNION RETURN 19 AS m",
        "CALL { RETURN count(*) AS c } RETURN c",
    ];
    let mut queries: Vec<(Language, &'static str)> = both
        .iter()
        .flat_map(|query| LANGUAGES.map(|language| (language, *query)))
        .collect();
    // GQL starts a query with FOR, not WITH, LET or FILTER.
    for query in [
        "INSERT (c:C {k: 3}) RETURN c.k",
        "FOR x IN [3, 19] RETURN x",
        "FOR x IN [3, 19] LET y = x + 1 RETURN y",
        "FOR x IN [3, 19] FILTER x > 3 RETURN x",
        "RETURN VALUE { MATCH (a:A) RETURN a.name } AS name",
        "CALL { RETURN 1 AS x } FILTER x > 0 RETURN x",
        "CALL { RETURN 1 AS x } LET y = x + 1 RETURN y",
        "RETURN count(*) AS c NEXT RETURN c",
    ] {
        queries.push((Language::Gql, query));
    }
    for query in [
        "CREATE (c:C {k: 3}) RETURN c.k",
        "WITH 1 AS x RETURN x",
        "WITH 1 AS x WHERE x > 0 RETURN x",
        "WITH 3 AS x, 19 AS y RETURN x + y AS z",
        "WITH 3 AS x RETURN count(x) AS c",
        "WITH 1 AS x RETURN x ORDER BY x LIMIT 1",
        "WITH [3, 19, 88] AS xs UNWIND xs AS x RETURN x",
        "WITH 1 AS x MATCH (b:B) RETURN x, b.name",
        "WITH 1 AS x OPTIONAL MATCH (b:B) RETURN x, b.name",
        "WITH 1 AS x OPTIONAL MATCH (z:Missing) RETURN x, z",
        "WITH 1 AS x CALL { WITH x RETURN x + 1 AS y } RETURN y",
        "WITH 1 AS x CALL { RETURN 2 AS y } RETURN x, y",
        "WITH 1 AS x WHERE EXISTS { MATCH (a:A) } RETURN x",
        "WITH 1 AS x RETURN x, COUNT { MATCH (a:A) WHERE a.name = 'Alix' } AS found",
        "WITH 'Alix' AS name MERGE (m:A {name: name}) RETURN m.name",
        "MERGE (m:A {name: toString('Alix')}) RETURN m.name",
        "WITH count(*) AS c RETURN c",
        "WITH count(*) AS c WHERE c > 0 RETURN c",
        "WITH collect(3) AS l UNWIND l AS x RETURN x",
        "RETURN percentileDisc(3, 0.5) AS p, stDev(3) AS d",
    ] {
        queries.push((Language::Cypher, query));
    }
    queries
}

/// Every query in the list above profiles without an error or a panic, and
/// the top line of its profile counts the rows the query returns: an entry
/// missing for an operator of the logical plan panicked, and one too many
/// would put another operator's counts on the top line.
#[test]
fn every_query_without_match_profiles() {
    let mut failures = Vec::new();
    for (language, query) in queries_without_match() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let returned = {
                let db = people();
                let session = db.session();
                execute(&session, language, query)
                    .unwrap_or_else(|error| panic!("the query failed: {error}"))
                    .rows()
                    .len()
            };
            let db = people();
            let session = db.session();
            let text = profile(&session, language, query);
            assert!(text.contains("Total time:"), "{text}");
            // The operators of a subquery in an expression (EXISTS, COUNT,
            // VALUE) record profile entries the logical tree has no place
            // for, which shifts the lines of such a query: a separate bug,
            // so its top line is not checked here.
            if ["EXISTS {", "COUNT {", "VALUE {"]
                .iter()
                .any(|subquery| query.contains(subquery))
            {
                return;
            }
            let lines = operators(&text);
            let top = lines.first().map(|(_, _, rows)| *rows);
            assert_eq!(
                top,
                Some(u64::try_from(returned).unwrap()),
                "the top line of the profile counts the returned rows:\n{text}"
            );
        }));
        if let Err(panic) = outcome {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(ToString::to_string))
                .unwrap_or_default();
            failures.push(format!("{language:?} `PROFILE {query}`: {message}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
