//! A pattern comprehension runs as an item of `RETURN` or `WITH`, alone or
//! inside an expression, as an aggregate's argument, and in a `WHERE` (of
//! `MATCH`, `OPTIONAL MATCH`, `WITH` or another pattern comprehension). In the
//! body of a list comprehension, a list predicate or a `reduce` (which reads
//! its own iteration variable), in `ORDER BY`, `SET` and `UNWIND` it is not
//! supported yet: the statement fails with GRAFEO-Q004 (unsupported, GQLSTATUS
//! 42000) naming the form. Both used to fail with the internal error
//! GRAFEO-X001.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main pattern_comprehension_placement::
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_common::utils::error::ErrorCode;
use grafeo_engine::GrafeoDB;

/// `(:X {n: 1})-[:R]->(:Y)` and `(:X {n: 2})-[:R]->(:Z)`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:X {n: 1})-[:R]->(:Y), (:X {n: 2})-[:R]->(:Z)")
        .unwrap();
    db
}

fn int_rows(values: &[&[i64]]) -> Vec<Vec<Value>> {
    values
        .iter()
        .map(|row| row.iter().map(|&value| Value::Int64(value)).collect())
        .collect()
}

fn check(db: &GrafeoDB, query: &str, expected: &[Vec<Value>]) {
    let result = db
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_eq!(result.rows(), expected, "{query}");
}

/// The positions the rewrite does not reach yet: the bodies that read their
/// own iteration variable, `ORDER BY`, `SET` and `UNWIND`. Each fails as
/// unsupported, with a message that names the pattern comprehension, never
/// as an internal error.
#[test]
fn an_unreached_position_is_unsupported_not_internal() {
    let db = database();
    for query in [
        // openCypher TCK, Pattern2 [7]
        "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
        "MATCH (n:X) RETURN [x IN [n] WHERE size([(x)-->(:Y) | 1]) > 0 | x.n] AS list",
        "MATCH (n:X) RETURN all(x IN [n] WHERE size([(x)-->() | 1]) > 0) AS every",
        "MATCH (n:X) RETURN reduce(s = 0, x IN [n] | s + size([(x)-->() | 1])) AS total",
        "MATCH (n:X) WHERE any(x IN [n] WHERE size([(x)-->(:Y) | 1]) > 0) RETURN n.n AS n",
        "MATCH (n:X) RETURN n.n AS n ORDER BY size([(n)-->(:Y) | 1]) DESC",
        "MATCH (n:X) WITH n ORDER BY size([(n)-->(:Y) | 1]) RETURN n.n AS n",
        "MATCH (n:X) SET n.c = size([(n)-->() | 1]) RETURN n.c AS c",
        "MATCH (n:X) UNWIND [(n)-->(m) | m.n] AS v RETURN count(v) AS c",
    ] {
        let error = db
            .execute_cypher(query)
            .expect_err(&format!("{query} should be refused"));
        assert_eq!(
            error.error_code(),
            ErrorCode::QueryUnsupported,
            "{query}: {error}"
        );
        let message = error.to_string();
        assert!(
            message.contains("pattern comprehension"),
            "{query}: {message}"
        );
        assert!(!message.contains("bug"), "{query}: {message}");
    }
}

/// In a `WHERE` the comprehension runs once per row, over that row's
/// variables.
#[test]
fn in_a_match_where() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) WHERE size([(n)-->(:Y) | 1]) > 0 RETURN n.n AS n",
        &int_rows(&[&[1]]),
    );
    check(
        &db,
        "MATCH (n:X) WHERE [(n)-->(m) | labels(m)[0]] = ['Z'] RETURN n.n AS n",
        &int_rows(&[&[2]]),
    );
}

/// A comprehension that matches nothing gives an empty list, not null.
#[test]
fn an_empty_comprehension_in_a_where_is_an_empty_list() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) WHERE [(n)-->(:W) | 1] = [] RETURN n.n AS n ORDER BY n",
        &int_rows(&[&[1], &[2]]),
    );
}

/// A comprehension that names no variable of the row gives the same list for
/// every row.
#[test]
fn an_uncorrelated_comprehension_in_a_where() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) WHERE size([(y:Y)<--() | 1]) = 1 RETURN n.n AS n ORDER BY n",
        &int_rows(&[&[1], &[2]]),
    );
}

/// In the `WHERE` of an `OPTIONAL MATCH` the comprehension decides whether
/// the optional part matches; a row it rejects stays, with a null `m`.
#[test]
fn in_an_optional_match_where() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) OPTIONAL MATCH (n)-->(m) WHERE size([(m)<--(:X {n: 1}) | 1]) > 0 \
         RETURN n.n AS n, m IS NULL AS missing ORDER BY n",
        &[
            vec![Value::Int64(1), Value::Bool(false)],
            vec![Value::Int64(2), Value::Bool(true)],
        ],
    );
}

#[test]
fn in_a_with_where() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) WITH n WHERE size([(n)-->(:Z) | 1]) = 1 RETURN n.n AS n",
        &int_rows(&[&[2]]),
    );
    check(
        &db,
        "MATCH (n:X) WITH n, n.n AS k WHERE size([(n)-->(:Y) | k]) = 1 RETURN k",
        &int_rows(&[&[1]]),
    );
}

/// The `WHERE` of a pattern comprehension may hold another one.
#[test]
fn in_the_where_of_a_pattern_comprehension() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) RETURN n.n AS n, \
         [(n)-->(m) WHERE size([(m)<--(:X {n: 2}) | 1]) = 1 | labels(m)[0]] AS ls ORDER BY n",
        &[
            vec![Value::Int64(1), Value::List(Vec::new().into())],
            vec![Value::Int64(2), Value::List(vec![Value::from("Z")].into())],
        ],
    );
}

/// The positions rewritten before this change still run.
#[test]
fn in_return_with_and_aggregates() {
    let db = database();
    check(
        &db,
        "MATCH (n:X) RETURN n.n AS n, size([(n)-->(:Y) | 1]) AS c ORDER BY n",
        &int_rows(&[&[1, 1], &[2, 0]]),
    );
    check(
        &db,
        "MATCH (n:X) WITH n, [(n)-->(m) | labels(m)[0]] AS ls RETURN n.n AS n, ls ORDER BY n",
        &[
            vec![Value::Int64(1), Value::List(vec![Value::from("Y")].into())],
            vec![Value::Int64(2), Value::List(vec![Value::from("Z")].into())],
        ],
    );
    check(
        &db,
        "MATCH (n:X) RETURN sum(size([(n)-->(:Y) | 1])) AS total",
        &int_rows(&[&[1]]),
    );
}
