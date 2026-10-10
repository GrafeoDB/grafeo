//! A vector or text search in a later MATCH keeps the rows of the clauses
//! before it: `MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(...) >
//! 0.5` returns every file with every similar document. The index search
//! replaces only a scan without input; a scan that runs for each input row
//! checks the condition per row. `PROFILE` shows which: a `VectorScan` or
//! `TextScan` for the search, and `EXPLAIN` marks a label-first scan
//! `[label-first]`, which the search replaces.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test search_after_a_match
//! ```

#![cfg(all(feature = "text-index", feature = "vector-index", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Files `a.rs` and `b.rs`, and documents with a 2-dimensional `emb` and a
/// `body`: Amsterdam ([1, 0], 'graph database'), Berlin ([0, 1], 'rust
/// compiler'), Paris ([0.8, 0.6], 'graph theory') and Prague ([0.3, 0.95],
/// 'query planner'). Against [1, 0] the cosine similarity is 1 for
/// Amsterdam, 0.8 for Paris, about 0.3 for Prague and 0 for Berlin. A cosine
/// vector index on `emb` and a text index on `body` of `Doc`.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:File {name: 'a.rs'}), (:File {name: 'b.rs'})")
        .unwrap();
    for (name, emb, body) in [
        ("Amsterdam", "[1.0, 0.0]", "graph database"),
        ("Berlin", "[0.0, 1.0]", "rust compiler"),
        ("Paris", "[0.8, 0.6]", "graph theory"),
        ("Prague", "[0.3, 0.95]", "query planner"),
    ] {
        db.execute(&format!(
            "INSERT (:Doc {{name: '{name}', emb: vector({emb}), body: '{body}'}})"
        ))
        .unwrap();
    }
    db.create_vector_index("Doc", "emb", Some(2), Some("cosine"), None, None, None)
        .expect("create vector index");
    db.create_text_index("Doc", "body")
        .expect("create text index");
    db
}

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    #[cfg(feature = "cypher")]
    Cypher,
}

/// The languages the build has: GQL and, with the `cypher` feature, Cypher.
fn languages() -> Vec<Language> {
    vec![
        Language::Gql,
        #[cfg(feature = "cypher")]
        Language::Cypher,
    ]
}

fn run(db: &GrafeoDB, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => db.execute(query),
        #[cfg(feature = "cypher")]
        Language::Cypher => db.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The rows of `query` as text, sorted.
fn rows(db: &GrafeoDB, language: Language, query: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = run(db, language, query)
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    Value::Int64(number) => number.to_string(),
                    other => panic!("unexpected value {other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

/// `rows` for literal text cells.
fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

/// The text of the plan that `EXPLAIN` or `PROFILE` returns for `query`.
fn plan(db: &GrafeoDB, language: Language, query: &str) -> String {
    run(db, language, query)
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Checks the rows of `query` in every language against `want`, and that
/// its physical plan does not search an index: the scan runs for each
/// input row.
fn assert_rows_per_row(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    for language in languages() {
        assert_eq!(
            rows(db, language, query),
            expected(want),
            "{language:?}: {query}"
        );
        let profile = plan(db, language, &format!("PROFILE {query}"));
        assert!(
            !profile.contains("VectorScan") && !profile.contains("TextScan"),
            "{language:?} `{query}` searched an index:\n{profile}"
        );
    }
}

/// Checks the rows of `query` in every language against `want`, and that
/// its physical plan searches with `operator` (`VectorScan` or `TextScan`)
/// while `EXPLAIN` shows no label-first scan for its filter.
fn assert_rows_searched(db: &GrafeoDB, query: &str, operator: &str, want: &[&[&str]]) {
    for language in languages() {
        assert_eq!(
            rows(db, language, query),
            expected(want),
            "{language:?}: {query}"
        );
        let profile = plan(db, language, &format!("PROFILE {query}"));
        assert!(
            profile.contains(operator),
            "{language:?} `{query}` did not search with {operator}:\n{profile}"
        );
        let explain = plan(db, language, &format!("EXPLAIN {query}"));
        assert!(
            !explain.contains("[label-first]"),
            "{language:?} `{query}` scans the label:\n{explain}"
        );
    }
}

/// Every file with each of the documents.
fn with_files(documents: &[&str]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for file in ["a.rs", "b.rs"] {
        for document in documents {
            rows.push(vec![file.to_string(), (*document).to_string()]);
        }
    }
    rows.sort();
    rows
}

/// `assert_rows_per_row` for the rows of `with_files(documents)`.
fn assert_files_with(db: &GrafeoDB, query: &str, documents: &[&str]) {
    let want = with_files(documents);
    let want: Vec<Vec<&str>> = want
        .iter()
        .map(|row| row.iter().map(String::as_str).collect())
        .collect();
    let want: Vec<&[&str]> = want.iter().map(Vec::as_slice).collect();
    assert_rows_per_row(db, query, &want);
}

#[test]
fn a_vector_search_after_a_match_keeps_its_rows() {
    let db = graph();
    for condition in [
        "cosine_similarity(d.emb, [1.0, 0.0]) > 0.5",
        "cosine_similarity(d.emb, [1.0, 0.0]) >= 0.75",
        "euclidean_distance(d.emb, [1.0, 0.0]) < 0.7",
    ] {
        assert_files_with(
            &db,
            &format!("MATCH (f:File) MATCH (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            &["Amsterdam", "Paris"],
        );
        assert_files_with(
            &db,
            &format!("MATCH (f:File), (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            &["Amsterdam", "Paris"],
        );
    }
}

/// Returning only the document: one row per file and document, not one per
/// document.
#[test]
fn a_vector_search_after_a_match_returns_a_row_per_input_row() {
    let db = graph();
    assert_rows_per_row(
        &db,
        "MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 \
         RETURN d.name",
        &[&["Amsterdam"], &["Amsterdam"], &["Paris"], &["Paris"]],
    );
    assert_rows_per_row(
        &db,
        "MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 \
         RETURN count(*)",
        &[&["4"]],
    );
}

#[test]
fn a_text_search_after_a_match_keeps_its_rows() {
    let db = graph();
    for condition in [
        "text_match(d.body, 'graph')",
        "text_score(d.body, 'graph') > 0.0",
    ] {
        assert_files_with(
            &db,
            &format!("MATCH (f:File) MATCH (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            &["Amsterdam", "Paris"],
        );
        assert_files_with(
            &db,
            &format!("MATCH (f:File), (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            &["Amsterdam", "Paris"],
        );
    }
}

/// A vector and a text condition together, under AND and under OR.
#[test]
fn a_vector_and_text_search_after_a_match_keeps_its_rows() {
    let db = graph();
    for (condition, documents) in [
        (
            "cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 AND text_match(d.body, 'theory')",
            &["Paris"][..],
        ),
        (
            "cosine_similarity(d.emb, [1.0, 0.0]) > 0.9 OR text_match(d.body, 'rust')",
            &["Amsterdam", "Berlin"][..],
        ),
    ] {
        assert_files_with(
            &db,
            &format!("MATCH (f:File) MATCH (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            documents,
        );
        assert_files_with(
            &db,
            &format!("MATCH (f:File), (d:Doc) WHERE {condition} RETURN f.name, d.name"),
            documents,
        );
    }
}

/// The rows of an UNWIND before the search, and a condition on the input
/// row beside the search.
#[test]
fn a_search_keeps_unwound_rows_and_reads_them() {
    let db = graph();
    assert_rows_per_row(
        &db,
        "UNWIND [3, 19] AS n MATCH (d:Doc) WHERE text_match(d.body, 'graph') \
         RETURN n, d.name",
        &[
            &["19", "Amsterdam"],
            &["19", "Paris"],
            &["3", "Amsterdam"],
            &["3", "Paris"],
        ],
    );
    assert_rows_per_row(
        &db,
        "MATCH (f:File) MATCH (d:Doc) \
         WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 AND f.name = 'b.rs' \
         RETURN f.name, d.name",
        &[&["b.rs", "Amsterdam"], &["b.rs", "Paris"]],
    );
}

/// Ordered by a score and cut: the top rows over every file and document,
/// not over the documents alone.
#[test]
fn a_top_k_by_score_after_a_match_keeps_its_rows() {
    let db = graph();
    for order in [
        "cosine_similarity(d.emb, [1.0, 0.0]) DESC",
        "euclidean_distance(d.emb, [1.0, 0.0])",
        "text_score(d.body, 'graph') DESC",
    ] {
        assert_files_with(
            &db,
            &format!("MATCH (f:File) MATCH (d:Doc) RETURN f.name, d.name ORDER BY {order} LIMIT 4"),
            &["Amsterdam", "Paris"],
        );
    }
}

/// A scan without input still searches the index.
#[test]
fn a_search_without_input_uses_the_index() {
    let db = graph();
    assert_rows_searched(
        &db,
        "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 RETURN d.name",
        "VectorScan",
        &[&["Amsterdam"], &["Paris"]],
    );
    assert_rows_searched(
        &db,
        "MATCH (d:Doc) WHERE text_match(d.body, 'graph') RETURN d.name",
        "TextScan",
        &[&["Amsterdam"], &["Paris"]],
    );
    for operator in ["VectorScan", "TextScan"] {
        assert_rows_searched(
            &db,
            "MATCH (d:Doc) \
             WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 AND text_match(d.body, 'theory') \
             RETURN d.name",
            operator,
            &[&["Paris"]],
        );
    }
}
