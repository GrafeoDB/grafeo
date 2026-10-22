//! Functions and operators that need a feature this build may not have: a
//! regular expression match (`=~`) and LIKE need `regex` or `regex-lite`,
//! `text_score` and `text_match` need `text-index`. Without the feature a
//! query that uses one is an error that says what the build lacks, before
//! any row is read; it used to be null for every row, so a WHERE kept none.
//! With the feature, the text functions also need a text index on the
//! property (#308).
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test functions_without_their_features
//! cargo test -p grafeo-engine --all-features --test functions_without_their_features
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Two directories.
fn directories() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Directory {path: 'src/tests'}), (:Directory {path: 'src/app'})")
        .unwrap();
    db
}

/// The number of rows of `result`, which must not be an error.
#[cfg(any(feature = "regex", feature = "regex-lite", feature = "text-index"))]
fn row_count(query: &str, result: Result<QueryResult>) -> usize {
    result
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .len()
}

/// The message of `result`'s error.
fn error_of(query: &str, result: Result<QueryResult>) -> String {
    match result {
        Ok(rows) => panic!("{query}: expected an error, got {:?}", rows.rows()),
        Err(error) => error.to_string(),
    }
}

/// Queries with a regular expression match or a LIKE pattern.
const PATTERN_QUERIES: &[&str] = &[
    "MATCH (d:Directory) WHERE d.path =~ '.*test.*' RETURN d.path",
    "MATCH (d:Directory) RETURN d.path =~ 'src/.*' AS in_src",
    "MATCH (d:Directory) WHERE d.path LIKE '%test%' RETURN d.path",
];

#[cfg(not(any(feature = "regex", feature = "regex-lite")))]
#[test]
fn a_pattern_match_without_regex_says_the_build_has_none() {
    let db = directories();
    for query in PATTERN_QUERIES {
        let error = error_of(query, db.execute(query));
        assert!(
            error.contains("this build has no regular expressions")
                && error.contains("'regex'")
                && error.contains("'regex-lite'"),
            "{query}: {error}"
        );
    }
}

#[cfg(any(feature = "regex", feature = "regex-lite"))]
#[test]
fn a_pattern_match_with_regex_runs() {
    let db = directories();
    let counts: Vec<usize> = PATTERN_QUERIES
        .iter()
        .map(|query| row_count(query, db.execute(query)))
        .collect();
    assert_eq!(counts, [1, 2, 1]);
}

/// Queries that call a text search function.
const TEXT_QUERIES: &[&str] = &[
    "MATCH (d:Directory) RETURN text_score(d.path, 'tests') AS score",
    "MATCH (d:Directory) WHERE text_match(d.path, 'tests') RETURN d.path",
];

#[cfg(not(feature = "text-index"))]
#[test]
fn a_text_function_without_text_index_says_the_build_lacks_it() {
    let db = directories();
    for (query, function) in TEXT_QUERIES.iter().zip(["text_score", "text_match"]) {
        let error = error_of(query, db.execute(query));
        assert!(
            error.contains(&format!(
                "Function '{function}' is not available in this build"
            )) && error.contains("'text-index'"),
            "{query}: {error}"
        );
    }
}

#[cfg(feature = "text-index")]
#[test]
fn a_text_function_with_text_index_runs() {
    let db = directories();
    db.create_text_index("Directory", "path").unwrap();
    let counts: Vec<usize> = TEXT_QUERIES
        .iter()
        .map(|query| row_count(query, db.execute(query)))
        .collect();
    assert_eq!(counts, [2, 1], "a score per directory, one match");
}

#[cfg(feature = "text-index")]
#[test]
fn a_text_function_without_a_text_index_says_to_create_one() {
    let db = directories();
    for (query, function) in TEXT_QUERIES.iter().zip(["text_score", "text_match"]) {
        let error = error_of(query, db.execute(query));
        assert!(
            error.contains(&format!(
                "{function}(d.path, ...) needs a text index on 'path'"
            )) && error.contains("USING TEXT"),
            "{query}: {error}"
        );
    }
}
