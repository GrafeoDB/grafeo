//! Regular expressions and LIKE patterns compile once per query, not once per
//! row (#458), and `=~` matches the whole string (openCypher).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test regex_patterns
//! ```

#![cfg(any(feature = "regex", feature = "regex-lite"))]

use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// `n` `:File` nodes, every third one named like a route or an api.
fn files(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for i in 0..n {
        let name = match i % 3 {
            0 => format!("src/routes/r{i}.py"),
            1 => format!("src/API/a{i}.py"),
            _ => format!("src/util/u{i}.py"),
        };
        db.create_node_with_props(&["File"], [("fileName", Value::from(name))])
            .unwrap();
    }
    db
}

fn count(result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>) -> i64 {
    result.unwrap().rows()[0][0].as_int64().unwrap()
}

/// Runs `query` and returns its count and how long it took.
fn timed(
    run: impl FnOnce() -> grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> (i64, Duration) {
    let started = Instant::now();
    let rows = count(run());
    (rows, started.elapsed())
}

/// Whether `pattern` took at most a small multiple of `plain`, the same
/// filter written without a pattern. Compiling the pattern per row made it
/// about 100 times slower (#458: 235 ms against 1.9 ms over 1,500 rows).
fn close_to(pattern: Duration, plain: Duration) -> bool {
    pattern <= plain * 10 + Duration::from_millis(50)
}

#[cfg(feature = "cypher")]
#[test]
fn a_regex_filter_compiles_its_pattern_once() {
    let db = files(6_000);
    let (plain_rows, plain) = timed(|| {
        db.execute_cypher(
            "MATCH (n:File) WHERE toLower(n.fileName) CONTAINS 'route'              OR toLower(n.fileName) CONTAINS 'api' RETURN count(n)",
        )
    });
    let (rows, pattern) = timed(|| {
        db.execute_cypher(
            "MATCH (n:File) WHERE n.fileName =~ '(?i).*(route|api).*' RETURN count(n)",
        )
    });
    assert_eq!((rows, plain_rows), (4_000, 4_000));
    assert!(
        close_to(pattern, plain),
        "=~ took {pattern:?}, CONTAINS {plain:?}"
    );
}

#[test]
fn a_like_filter_compiles_its_pattern_once() {
    let db = files(6_000);
    let (plain_rows, plain) = timed(|| {
        db.execute("MATCH (n:File) WHERE n.fileName STARTS WITH 'src/routes/' RETURN count(n)")
    });
    let (rows, pattern) =
        timed(|| db.execute("MATCH (n:File) WHERE n.fileName LIKE 'src/routes/%' RETURN count(n)"));
    assert_eq!((rows, plain_rows), (2_000, 2_000));
    assert!(
        close_to(pattern, plain),
        "LIKE took {pattern:?}, STARTS WITH {plain:?}"
    );
}
