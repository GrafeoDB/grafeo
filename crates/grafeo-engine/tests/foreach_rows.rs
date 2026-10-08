//! `FOREACH` runs its updates once per item of its list for each row, and
//! passes each row on once, as it came in (openCypher): a list of two items
//! writes twice and keeps the one row, an empty list or null writes nothing
//! and keeps every row, and the `FOREACH` variable and what its updates bind
//! are not variables after it. Its updates see all of what the clauses before
//! it wrote. Cypher only: GQL has no `FOREACH` (its `FOR` is an `UNWIND`).
//! These queries write, so they are no cases of the differential test corpus
//! (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test foreach_rows
//! ```

#![cfg(all(feature = "lpg", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

fn run(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// The number of nodes with the label `Z`.
fn written(session: &Session) -> Value {
    run(session, "MATCH (z:Z) RETURN count(z)")[0][0].clone()
}

/// Each query runs on a new database: the rows it returns and the `Z` nodes
/// it writes, whatever the list holds.
#[test]
fn foreach_keeps_one_row_per_input_row() {
    for (query, rows, nodes) in [
        (
            "UNWIND [1] AS i FOREACH (x IN [1, 2] | CREATE (:Z)) RETURN count(*) AS rows",
            1,
            2,
        ),
        (
            "UNWIND [1, 2, 3] AS i FOREACH (x IN [] | CREATE (:Z)) RETURN count(*) AS rows",
            3,
            0,
        ),
        (
            "UNWIND [1, 2, 3] AS i FOREACH (x IN null | CREATE (:Z)) RETURN count(*) AS rows",
            3,
            0,
        ),
        (
            "UNWIND [3, 19] AS i FOREACH (x IN range(1, i) | CREATE (:Z {i: i, x: x})) \
             RETURN count(*) AS rows",
            2,
            22,
        ),
        // Nested: two items, each with three.
        (
            "UNWIND [1] AS i FOREACH (x IN [1, 2] | FOREACH (y IN [1, 2, 3] | \
             CREATE (:Z {x: x, y: y}))) RETURN count(*) AS rows",
            1,
            6,
        ),
        // Two updates per item.
        (
            "UNWIND [1, 2] AS i FOREACH (x IN [3, 19] | CREATE (:Z {x: x}) CREATE (:Z {x: -x})) \
             RETURN count(*) AS rows",
            2,
            8,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        assert_eq!(run(&session, query), [[int(rows)]], "`{query}`");
        assert_eq!(written(&session), int(nodes), "the nodes `{query}` writes");
    }
}

/// The rows after `FOREACH` are those before it, values and order included,
/// and only their variables are there: `RETURN *` returns `i` alone.
#[test]
fn foreach_passes_each_row_on_unchanged() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        run(
            &session,
            "UNWIND [19, 3, 88] AS i FOREACH (x IN range(1, 2) | CREATE (:Z {i: i})) RETURN i"
        ),
        [[int(19)], [int(3)], [int(88)]]
    );
    let result = session
        .execute_cypher("UNWIND [3] AS i FOREACH (x IN [1, 2] | CREATE (z:Z)) RETURN *")
        .unwrap();
    assert_eq!(result.columns, ["i"], "the columns of `RETURN *`");
    assert_eq!(result.rows(), [[int(3)]]);
}

/// The `FOREACH` variable and the variables its updates bind are not
/// variables after it.
#[test]
fn foreach_variables_stay_inside_it() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (query, variable) in [
        (
            "UNWIND [3] AS i FOREACH (x IN [1, 2] | CREATE (:Z)) RETURN x",
            "x",
        ),
        (
            "UNWIND [3] AS i FOREACH (x IN [1, 2] | CREATE (z:Z)) RETURN z",
            "z",
        ),
    ] {
        let error = session
            .execute_cypher(query)
            .expect_err(&format!("`{query}` reads a variable of the FOREACH"));
        assert!(
            error
                .to_string()
                .contains(&format!("Undefined variable '{variable}'")),
            "`{query}`: {error}"
        );
    }
    assert_eq!(
        written(&session),
        int(0),
        "a query that fails writes nothing"
    );
}

/// The updates of a `FOREACH` after a `MATCH` act on the matched nodes, and
/// each item's update sees the ones before it.
#[test]
fn foreach_updates_the_matched_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run(
        &session,
        "CREATE (:Person {name: 'Alix'}), (:Person {name: 'Gus'})",
    );
    assert_eq!(
        run(
            &session,
            "MATCH (p:Person) FOREACH (x IN [3, 19, 88] | SET p.n = coalesce(p.n, 0) + x) \
             RETURN p.name AS name, p.n AS n ORDER BY name"
        ),
        [
            [Value::from("Alix"), int(110)],
            [Value::from("Gus"), int(110)]
        ]
    );
    assert_eq!(
        run(
            &session,
            "MATCH (p:Person) FOREACH (city IN ['Amsterdam', 'Berlin'] | \
             CREATE (p)-[:VISITED]->(:City {name: city})) RETURN p.name AS name ORDER BY name"
        ),
        [[Value::from("Alix")], [Value::from("Gus")]]
    );
    assert_eq!(
        run(
            &session,
            "MATCH (p:Person)-[:VISITED]->(c:City) RETURN p.name AS name, count(c) AS cities \
             ORDER BY name"
        ),
        [[Value::from("Alix"), int(2)], [Value::from("Gus"), int(2)]]
    );
    // As the last clause, without a RETURN.
    run(
        &session,
        "MATCH (p:Person) FOREACH (x IN [1, 2] | CREATE (:Z {name: p.name}))",
    );
    assert_eq!(written(&session), int(4));
}

/// The row `i` writes `(:N {k: i})`, and its `FOREACH` merges `k: 3 - i`,
/// which the other row writes: the `FOREACH` runs after the whole write, so
/// both merges find their node.
#[test]
fn foreach_after_a_write_sees_the_whole_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        run(
            &session,
            "UNWIND [1, 2] AS i CREATE (:N {k: i}) WITH i \
             FOREACH (x IN [1] | MERGE (:N {k: 3 - i})) RETURN count(*) AS rows"
        ),
        [[int(2)]]
    );
    assert_eq!(run(&session, "MATCH (n:N) RETURN count(n)"), [[int(2)]]);
}

/// `PROFILE` runs the plan too: its top line counts the one row.
#[test]
fn foreach_profiles() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let rows = run(
        &session,
        "PROFILE UNWIND [1] AS i FOREACH (x IN [1, 2] | CREATE (:Z)) RETURN count(*)",
    );
    let Value::String(text) = &rows[0][0] else {
        panic!("PROFILE returned {rows:?}");
    };
    let top = text.lines().next().unwrap_or_default();
    assert!(top.contains("rows=1"), "the top line of\n{text}");
    assert_eq!(
        written(&session),
        int(2),
        "PROFILE writes as the query does"
    );
}

/// `FOREACH` can be the first clause, as the first example of the Cypher
/// mutations guide shows: it runs its updates once for the one row a query
/// starts from.
#[test]
fn foreach_can_start_a_query() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run(
        &session,
        "FOREACH (name IN ['Alix', 'Gus', 'Vincent'] | CREATE (:Person {name: name}))",
    );
    let mut names: Vec<Value> = run(&session, "MATCH (p:Person) RETURN p.name")
        .into_iter()
        .map(|row| row[0].clone())
        .collect();
    names.sort_by_key(ToString::to_string);
    assert_eq!(
        names,
        [
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Vincent")
        ]
    );
    run(&session, "FOREACH (x IN [3, 19] | CREATE (:Z {x: x}))");
    assert_eq!(written(&session), int(2));
}
