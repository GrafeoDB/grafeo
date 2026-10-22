//! Runtime support for .gtest spec tests.
//!
//! Provides helper functions called by the build.rs-generated test code:
//! dataset loading, query execution, and result comparison.

use std::fs;
use std::path::PathBuf;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Root directory of the grafeo repository.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Load a dataset .setup file into the database.
///
/// # Panics
///
/// Panics if the file at `relative_path` cannot be read or if any query in the file fails to execute.
pub fn load_dataset(db: &GrafeoDB, relative_path: &str) {
    let path = repo_root().join(relative_path);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read dataset {}: {e}", path.display()));

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        db.execute(trimmed)
            .unwrap_or_else(|e| panic!("Dataset query failed: {trimmed}\nError: {e}"));
    }
}

/// Execute a query in the specified language, panicking on failure.
pub fn execute_query(db: &GrafeoDB, language: &str, query: &str) {
    execute_query_with_model(db, language, "", query);
}

/// Execute a query with model-aware dispatch, panicking on failure.
///
/// # Panics
///
/// Panics if `execute_query_result` returns an error for the given query and language.
pub fn execute_query_with_model(db: &GrafeoDB, language: &str, model: &str, query: &str) {
    execute_query_result(db, language, model, query).unwrap_or_else(|e| {
        panic!("Query failed: {query}\nLanguage: {language}\nModel: {model}\nError: {e}")
    });
}

/// Execute a query using the (language, model) dispatch key, returning the Result.
///
/// # Errors
///
/// Returns `Err` if the underlying database executor fails (parse error, execution error, etc.).
///
/// # Panics
///
/// Panics if `language` is not a supported language identifier.
pub fn execute_query_result(
    db: &GrafeoDB,
    language: &str,
    model: &str,
    query: &str,
) -> Result<QueryResult> {
    // The (language, model) pair determines dispatch. For most languages,
    // model is informational. For GraphQL, model=rdf routes to the RDF
    // triple store executor via execute_language("graphql-rdf", ...).
    match (language, model) {
        ("gql" | "", _) => db.execute(query),
        #[cfg(feature = "cypher")]
        ("cypher", _) => db.execute_cypher(query),
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        ("sparql", _) => db.execute_sparql(query),
        #[cfg(feature = "gremlin")]
        ("gremlin", _) => db.execute_gremlin(query),
        #[cfg(feature = "graphql")]
        ("graphql", _) => db.execute_graphql(query),
        #[cfg(all(feature = "graphql", feature = "triple-store"))]
        ("graphql-rdf", _) => db.execute_language(query, "graphql-rdf", None),
        #[cfg(feature = "sql-pgq")]
        ("sql-pgq" | "sql_pgq", _) => db.execute_sql(query),
        (other, _) => panic!("Unsupported language: {other}"),
    }
}

/// Execute a query with parameters, returning the Result.
///
/// Delegates to `GrafeoDB::execute_language` which routes the query to the
/// correct language executor and substitutes parameters before planning.
///
/// # Errors
///
/// Returns `Err` if the query fails to parse, plan, or execute.
pub fn execute_query_result_with_params(
    db: &GrafeoDB,
    language: &str,
    _model: &str,
    query: &str,
    params: std::collections::HashMap<String, Value>,
) -> Result<QueryResult> {
    // Normalize the language key so execute_language can dispatch correctly.
    let lang_key = match language {
        "sql-pgq" | "sql_pgq" => "sql-pgq",
        other => other,
    };
    db.execute_language(query, lang_key, Some(params))
}

/// Convert a QueryResult into rows of string values for comparison.
pub fn result_to_strings(result: &QueryResult) -> Vec<Vec<String>> {
    result
        .rows()
        .iter()
        .map(|row| row.iter().map(value_to_string).collect())
        .collect()
}

/// Convert a Value to its canonical string representation for comparison.
pub fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => {
            if f.is_infinite() {
                if f.is_sign_positive() {
                    "Infinity".to_string()
                } else {
                    "-Infinity".to_string()
                }
            } else if f.is_nan() {
                "NaN".to_string()
            } else {
                format!("{f}")
            }
        }
        Value::String(s) => s.to_string(),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Map(map) => {
            let mut entries: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{k}: {}", value_to_string(v)))
                .collect();
            entries.sort();
            format!("{{{}}}", entries.join(", "))
        }
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Timestamp(ts) => ts.to_string(),
        Value::ZonedDatetime(zdt) => zdt.to_string(),
        Value::Duration(d) => d.to_string(),
        Value::Vector(v) => format!("{v:?}"),
        Value::Bytes(b) => format!("bytes[{}]", b.len()),
        Value::Path { nodes, edges } => {
            format!("path[{} nodes, {} edges]", nodes.len(), edges.len())
        }
        Value::GCounter(counts) => {
            let total: u64 = counts.values().sum();
            format!("{total}")
        }
        Value::OnCounter { pos, neg } => {
            // reason: individual CRDT counter values are small, sum fits i64
            #[allow(clippy::cast_possible_wrap)]
            let net = pos.values().sum::<u64>() as i64 - neg.values().sum::<u64>() as i64;
            format!("{net}")
        }
        _ => value.to_string(),
    }
}

/// Assert rows match after sorting both sides.
///
/// # Panics
///
/// Panics if row counts differ, column counts differ within any row, or any cell value does not match.
pub fn assert_rows_sorted(result: &QueryResult, expected: &[Vec<String>]) {
    let mut actual = result_to_strings(result);
    let mut expected = expected.to_vec();

    // Sort both
    actual.sort();
    expected.sort();

    assert_eq!(
        actual.len(),
        expected.len(),
        "Row count mismatch: got {} rows, expected {}\nActual: {actual:?}\nExpected: {expected:?}",
        actual.len(),
        expected.len()
    );

    for (i, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            act.len(),
            exp.len(),
            "Column count mismatch at row {i}: got {} cols, expected {}\nActual row: {act:?}\nExpected row: {exp:?}",
            act.len(),
            exp.len()
        );
        for (j, (a, e)) in act.iter().zip(exp.iter()).enumerate() {
            assert_eq!(
                a, e,
                "Mismatch at sorted row {i}, col {j}: got '{a}', expected '{e}'\nFull actual row: {act:?}\nFull expected row: {exp:?}"
            );
        }
    }
}

/// Assert rows match in exact order.
///
/// # Panics
///
/// Panics if row counts differ, column counts differ within any row, or any cell value does not match.
pub fn assert_rows_ordered(result: &QueryResult, expected: &[Vec<String>]) {
    let actual = result_to_strings(result);

    assert_eq!(
        actual.len(),
        expected.len(),
        "Row count mismatch: got {} rows, expected {}\nActual: {actual:?}\nExpected: {expected:?}",
        actual.len(),
        expected.len()
    );

    for (i, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            act.len(),
            exp.len(),
            "Column count mismatch at row {i}: got {} cols, expected {}\nActual row: {act:?}\nExpected row: {exp:?}",
            act.len(),
            exp.len()
        );
        for (j, (a, e)) in act.iter().zip(exp.iter()).enumerate() {
            assert_eq!(
                a, e,
                "Mismatch at row {i}, col {j}: got '{a}', expected '{e}'\nFull actual row: {act:?}\nFull expected row: {exp:?}"
            );
        }
    }
}

/// Assert that result column names match expected names exactly.
///
/// # Panics
///
/// Panics if the result column names do not match the `expected` slice.
pub fn assert_columns(result: &QueryResult, expected: &[&str]) {
    let actual: Vec<&str> = result.columns.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        actual, expected,
        "Column mismatch: got {actual:?}, expected {expected:?}"
    );
}

/// Assert rows match with floating-point tolerance.
///
/// Cells that parse as `f64` on both sides are compared within `10^(-precision)`.
/// All other cells use exact string comparison.
///
/// # Panics
///
/// Panics if row counts differ, column counts differ, or any cell value falls outside the tolerance.
pub fn assert_rows_with_precision(result: &QueryResult, expected: &[Vec<String>], precision: u32) {
    let actual = result_to_strings(result);
    // reason: precision is typically 0..10, fits i32 negated
    #[allow(clippy::cast_possible_wrap)]
    let tolerance = 10f64.powi(-(precision as i32));

    assert_eq!(
        actual.len(),
        expected.len(),
        "Row count mismatch: got {} rows, expected {}\nActual: {actual:?}\nExpected: {expected:?}",
        actual.len(),
        expected.len()
    );

    for (i, (act_row, exp_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            act_row.len(),
            exp_row.len(),
            "Column count mismatch at row {i}: got {} cols, expected {}",
            act_row.len(),
            exp_row.len()
        );
        for (j, (a, e)) in act_row.iter().zip(exp_row.iter()).enumerate() {
            if let (Ok(af), Ok(ef)) = (a.parse::<f64>(), e.parse::<f64>()) {
                assert!(
                    (af - ef).abs() < tolerance,
                    "Float mismatch at row {i}, col {j}: got {af}, expected {ef} (tolerance {tolerance})"
                );
            } else {
                assert_eq!(
                    a, e,
                    "Mismatch at row {i}, col {j}: got '{a}', expected '{e}'"
                );
            }
        }
    }
}

/// Assert result hash matches (MD5 of sorted, pipe-delimited rows).
///
/// # Panics
///
/// Panics if the computed MD5 hash of the sorted, pipe-delimited rows does not match `expected_hash`.
pub fn assert_hash(result: &QueryResult, expected_hash: &str) {
    let mut rows = result_to_strings(result);
    rows.sort();

    let mut ctx = md5::Context::new();
    for row in &rows {
        ctx.consume(row.join("|").as_bytes());
        ctx.consume(b"\n");
    }
    let hash = format!("{:x}", ctx.finalize());

    assert_eq!(
        hash, expected_hash,
        "Hash mismatch: got '{hash}', expected '{expected_hash}'\nRows: {rows:?}"
    );
}

// Include the generated test code
#[cfg(test)]
mod generated {
    use super::*;
    use grafeo_engine::GrafeoDB;

    include!(concat!(env!("OUT_DIR"), "/spec_tests.rs"));
}
