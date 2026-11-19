//! Regular expressions and LIKE patterns compile once per query, not once per
//! row (#458), and `=~` matches the whole string (openCypher). GQL has `=~`
//! too, a Grafeo extension with Cypher's meaning, in every place an
//! expression goes. A pattern that is not a regular expression is an error
//! that names it, before any row is read.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main regex_patterns::
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

// ============================================================================
// SPARQL `REGEX` and `REPLACE` (#675)
// ============================================================================

/// The integer a SPARQL `COUNT` answers, an `xsd:integer` literal.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_count(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> i64 {
    match &result.unwrap().rows()[0][0] {
        Value::RdfTerm(term) => term
            .as_literal()
            .expect("a literal")
            .value()
            .parse()
            .unwrap(),
        other => panic!("expected an xsd:integer, got {other:?}"),
    }
}

/// [`timed`] for a SPARQL `COUNT`.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn timed_sparql(
    run: impl FnOnce() -> grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> (i64, Duration) {
    let started = Instant::now();
    let rows = sparql_count(run());
    (rows, started.elapsed())
}

/// `n` resources, each with a file name like `files` gives the nodes.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn rdf_files(n: usize) -> GrafeoDB {
    use std::fmt::Write;

    let db = GrafeoDB::new_in_memory();
    let mut data = String::from("INSERT DATA {\n");
    for i in 0..n {
        let name = match i % 3 {
            0 => format!("src/routes/r{i}.py"),
            1 => format!("src/API/a{i}.py"),
            _ => format!("src/util/u{i}.py"),
        };
        writeln!(
            data,
            "  <http://ex.org/f{i}> <http://ex.org/name> \"{name}\" ."
        )
        .unwrap();
    }
    data.push('}');
    db.execute_sparql(&data).unwrap();
    db
}

/// A SPARQL `REGEX` with a constant pattern and flags compiles it once, not once per row: as
/// close to the same filter written with `CONTAINS` as `=~` is in Cypher.
/// Timed against `regex`'s compile cost; a debug build with only `regex-lite` sits too close to
/// the bound to say anything.
#[cfg(all(feature = "sparql", feature = "triple-store", feature = "regex"))]
#[test]
fn a_sparql_regex_compiles_its_pattern_once() {
    let db = rdf_files(6_000);
    let (plain_rows, plain) = timed_sparql(|| {
        db.execute_sparql(
            "SELECT (COUNT(*) AS ?n) WHERE { ?f <http://ex.org/name> ?name \
             FILTER(CONTAINS(LCASE(?name), 'route') || CONTAINS(LCASE(?name), 'api')) }",
        )
    });
    let (rows, pattern) = timed_sparql(|| {
        db.execute_sparql(
            "SELECT (COUNT(*) AS ?n) WHERE { ?f <http://ex.org/name> ?name \
             FILTER(REGEX(?name, '(route|api)', 'i')) }",
        )
    });
    assert_eq!((rows, plain_rows), (4_000, 4_000));
    assert!(
        close_to(pattern, plain),
        "REGEX took {pattern:?}, CONTAINS {plain:?}"
    );
    let (replaced, _) = timed_sparql(|| {
        db.execute_sparql(
            "SELECT (COUNT(*) AS ?n) WHERE { ?f <http://ex.org/name> ?name \
             FILTER(REPLACE(?name, 'API', 'api', 'q') != ?name) }",
        )
    });
    assert_eq!(
        replaced, 2_000,
        "REPLACE with the q flag rewrites the API names"
    );
}

/// Patterns taken from the rows: each row's pattern and flags give its answer, more distinct
/// patterns than the cache holds give the same answers, and a pattern that does not compile is
/// an error of its row alone (the FILTER drops that row, 17.2.2).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_regex_patterns_taken_from_rows() {
    use std::fmt::Write;

    let db = GrafeoDB::new_in_memory();
    let mut data = String::from("INSERT DATA {\n");
    // 300 rows, each matching its own pattern: more than the 256 patterns a cache keeps.
    for i in 0..300 {
        writeln!(
            data,
            "  <http://ex.org/r{i}> <http://ex.org/text> \"Item-{i}\" ; <http://ex.org/pattern> \"^item-{i}$\" ."
        )
        .unwrap();
    }
    data.push_str("  <http://ex.org/bad> <http://ex.org/text> \"Item-x\" ; <http://ex.org/pattern> \"(unclosed\" .\n");
    data.push('}');
    db.execute_sparql(&data).unwrap();
    let count = |query: &str| -> i64 { sparql_count(db.execute_sparql(query)) };
    let each =
        "SELECT (COUNT(*) AS ?n) WHERE { ?r <http://ex.org/text> ?t ; <http://ex.org/pattern> ?p ";
    assert_eq!(count(&format!("{each} FILTER(REGEX(?t, ?p, 'i')) }}")), 300);
    assert_eq!(
        count(&format!("{each} FILTER(REGEX(?t, ?p)) }}")),
        0,
        "case matters without i"
    );
    assert_eq!(
        count(&format!("{each} FILTER(!REGEX(?t, ?p, 'i')) }}")),
        0,
        "the bad pattern's row is an error under NOT too, so dropped"
    );
    assert_eq!(
        count("SELECT (COUNT(*) AS ?n) WHERE { ?r <http://ex.org/text> ?t }"),
        301
    );
}

/// Two flag sets of one pattern in one query compile apart: the case-blind
/// `REGEX` keeps every row and the case-sensitive one none, so a pattern
/// cached under the other's flags would change the count. `q` quotes the
/// pattern, with `i` too.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_regex_flag_sets_in_one_query_compile_apart() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/a> <http://ex.org/text> \"A.C\" . \
         <http://ex.org/b> <http://ex.org/text> \"AXC\" . \
         <http://ex.org/c> <http://ex.org/text> \"abc\" }",
    )
    .unwrap();
    let count = |filter: &str| -> i64 {
        sparql_count(db.execute_sparql(&format!(
            "SELECT (COUNT(*) AS ?n) WHERE {{ ?r <http://ex.org/text> ?t FILTER({filter}) }}"
        )))
    };
    // Case-blind "a.c" matches all three; case-sensitive matches only "abc".
    assert_eq!(count("REGEX(?t, 'a.c', 'i') && !REGEX(?t, 'a.c')"), 2);
    assert_eq!(count("!REGEX(?t, 'a.c', 'i') && REGEX(?t, 'a.c')"), 0);
    // Quoted: "a.c" is the literal text, so only "A.C" with i, and nothing without.
    assert_eq!(count("REGEX(?t, 'a.c', 'qi')"), 1);
    assert_eq!(count("REGEX(?t, 'a.c', 'qi') && !REGEX(?t, 'a.c', 'q')"), 1);
}

/// `REPLACE` with a pattern that matches the empty string is an error
/// (SPARQL 1.1, 17.4.3.15): the value is unbound, and a filter on it drops
/// the row; a pattern that does not match empty replaces as before.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_replace_with_a_pattern_matching_empty_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql("INSERT DATA { <http://ex.org/a> <http://ex.org/text> \"abc\" }")
        .unwrap();
    let count = |filter: &str| -> i64 {
        sparql_count(db.execute_sparql(&format!(
            "SELECT (COUNT(*) AS ?n) WHERE {{ ?r <http://ex.org/text> ?t FILTER({filter}) }}"
        )))
    };
    assert_eq!(
        count("BOUND(REPLACE(?t, 'x*', '-'))"),
        0,
        "x* matches empty"
    );
    assert_eq!(
        count("REPLACE(?t, 'x*', '-') = 'abc'"),
        0,
        "an error, not the text"
    );
    assert_eq!(
        count("REPLACE(?t, 'b', '-') = 'a-c'"),
        1,
        "a non-empty match replaces"
    );
}
