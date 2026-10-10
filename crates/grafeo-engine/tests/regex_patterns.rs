//! Regular expressions and LIKE patterns compile once per query, not once per
//! row (#458), and `=~` matches the whole string (openCypher). GQL has `=~`
//! too, a Grafeo extension with Cypher's meaning, in every place an
//! expression goes. A pattern that is not a regular expression is an error
//! that names it, before any row is read.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test regex_patterns
//! ```

#![cfg(any(feature = "regex", feature = "regex-lite"))]

use std::collections::HashMap;
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

// ============================================================================
// `=~` in GQL, and patterns that are not regular expressions
// ============================================================================

/// Directories with paths a project tree has: tests, specs, caches, a `.git`
/// directory, and `src/agit`, which has `git` but no `.git`.
fn directories() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Directory {path: 'src/tests'}), (:Directory {path: 'src/app'}), \
         (:Directory {path: 'lib/spec/models'}), (:Directory {path: 'src/__pycache__'}), \
         (:Directory {path: 'web/node_modules/left-pad'}), (:Directory {path: 'src/.git/hooks'}), \
         (:Directory {path: 'src/agit'}), (:Directory {name: 'unnamed'})",
    )
    .unwrap();
    db
}

/// The single column of `result`, as strings (a null as `null`).
fn texts(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<String> {
    result
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            Value::Null => "null".to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// The paths Deriva's derivation queries leave out (tests, specs, caches and
/// version control), with the pattern they use.
const EXCLUDED: &str = r"'.*(test|spec|__pycache__|node_modules|\.git).*'";

#[test]
fn gql_regex_match_keeps_the_paths_of_an_alternation() {
    let db = directories();
    let matched = texts(db.execute(&format!(
        "MATCH (n:Directory) WHERE n.path =~ {EXCLUDED} RETURN n.path ORDER BY n.path"
    )));
    assert_eq!(
        matched,
        [
            "lib/spec/models",
            "src/.git/hooks",
            "src/__pycache__",
            "src/tests",
            "web/node_modules/left-pad"
        ]
    );
    // `\.git` is a dot and `git`: `src/agit` stays out. A directory without
    // a path is unknown, so NOT keeps it out as well.
    let kept = texts(db.execute(&format!(
        "MATCH (n:Directory) WHERE NOT n.path =~ {EXCLUDED} RETURN n.path ORDER BY n.path"
    )));
    assert_eq!(kept, ["src/agit", "src/app"]);
}

#[cfg(feature = "cypher")]
#[test]
fn gql_and_cypher_regex_matches_keep_the_same_paths() {
    let db = directories();
    for negation in ["", "NOT "] {
        let query = format!(
            "MATCH (n:Directory) WHERE {negation}n.path =~ {EXCLUDED} RETURN n.path ORDER BY n.path"
        );
        assert_eq!(
            texts(db.execute(&query)),
            texts(db.execute_cypher(&query)),
            "{query}"
        );
    }
}

#[test]
fn gql_regex_match_matches_the_whole_string() {
    let db = directories();
    assert!(
        texts(db.execute("MATCH (n:Directory) WHERE n.path =~ 'src' RETURN n.path")).is_empty(),
        "a pattern that matches a prefix only must not match"
    );
    assert_eq!(
        texts(db.execute("MATCH (n:Directory) WHERE n.path =~ '(?i)SRC/APP' RETURN n.path")),
        ["src/app"]
    );
}

#[test]
fn gql_regex_match_is_a_value_in_return_set_and_inline_where() {
    let db = directories();
    let flags = texts(db.execute(
        "MATCH (n:Directory) RETURN n.path =~ 'src/.*' AS in_src ORDER BY n.path NULLS LAST",
    ));
    assert_eq!(
        flags,
        [
            "false", "true", "true", "true", "true", "true", "false", "null"
        ],
        "one flag per directory, null without a path"
    );

    let inline =
        texts(db.execute(
            "MATCH (n:Directory WHERE n.path =~ 'src/[a-z]+') RETURN n.path ORDER BY n.path",
        ));
    assert_eq!(inline, ["src/agit", "src/app", "src/tests"]);

    db.execute(r"MATCH (n:Directory) SET n.hidden = n.path =~ '.*/\..*'")
        .unwrap();
    let hidden = texts(db.execute("MATCH (n:Directory) WHERE n.hidden = true RETURN n.path"));
    assert_eq!(hidden, ["src/.git/hooks"]);

    let map = db
        .execute("MATCH (n:Directory {path: 'src/app'}) RETURN {in_src: n.path =~ 'src/.*'} AS m")
        .unwrap();
    assert_eq!(
        map.rows()[0][0].to_string(),
        "{in_src: true}",
        "a map value holds the match"
    );
}

#[test]
fn gql_regex_match_takes_its_pattern_from_a_parameter() {
    let db = directories();
    let params = HashMap::from([("pattern".to_string(), Value::from(".*_.*"))]);
    let paths = texts(db.execute_with_params(
        "MATCH (n:Directory) WHERE n.path =~ $pattern RETURN n.path ORDER BY n.path",
        params,
    ));
    assert_eq!(paths, ["src/__pycache__", "web/node_modules/left-pad"]);
}

/// The message of `result`'s error.
fn error_of(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> String {
    match result {
        Ok(rows) => panic!("expected an error, got {:?}", rows.rows()),
        Err(error) => error.to_string(),
    }
}

#[test]
fn an_invalid_pattern_is_an_error_that_names_it() {
    let db = directories();
    let error = error_of(db.execute("MATCH (n:Directory) WHERE n.path =~ '(src' RETURN n.path"));
    assert!(
        error.contains("Invalid regular expression '(src'"),
        "{error}"
    );
    let error = error_of(db.execute("MATCH (n:Directory) RETURN n.path =~ 'src)|(lib' AS m"));
    assert!(
        error.contains("Invalid regular expression 'src)|(lib'"),
        "a pattern is checked alone, not inside the group that makes it match the whole string: {error}"
    );
    let params = HashMap::from([("pattern".to_string(), Value::from("[a-"))]);
    let error = error_of(db.execute_with_params(
        "MATCH (n:Directory) WHERE n.path =~ $pattern RETURN n.path",
        params,
    ));
    assert!(
        error.contains("Invalid regular expression '[a-'"),
        "{error}"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn an_invalid_cypher_pattern_is_an_error_that_names_it() {
    let db = directories();
    let error =
        error_of(db.execute_cypher("MATCH (n:Directory) WHERE n.path =~ '(src' RETURN n.path"));
    assert!(
        error.contains("Invalid regular expression '(src'"),
        "{error}"
    );
}
