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

/// A parameter written as an integer past `i64`, typed as a binding types one
/// (see "Parameter values" in the spec README): an exact decimal up to 38
/// digits, and past that the error a binding gives, naming the parameter.
///
/// # Errors
///
/// An integer with more than 38 digits.
pub fn wide_integer_param(name: &str, text: &str) -> Result<Value> {
    Value::from_number_text(text).map_err(|error| {
        grafeo_common::utils::error::Error::InvalidValue(format!("parameter '{name}': {error}"))
    })
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
///
/// # Panics
///
/// On an RDF term of a kind the runner does not know, so a new kind fails
/// the spec run instead of comparing as some other text.
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
        // An RDF term as its lexical form: an IRI bare, a blank node as
        // `_:` and its label, a literal as its text (a `terms:`
        // expectation checks kind, language tag and datatype).
        Value::RdfTerm(term) => match term.as_ref() {
            grafeo_common::types::rdf::Term::Iri(iri) => iri.as_str().to_string(),
            grafeo_common::types::rdf::Term::BlankNode(node) => format!("_:{}", node.id()),
            grafeo_common::types::rdf::Term::Literal(literal) => literal.value().to_string(),
            other => panic!(
                "{}",
                grafeo_common::types::rdf::UnsupportedTermKind::of(other)
            ),
        },
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

/// Assert that the last statement failed as an error case expects: its
/// message contains `message` and its GQLSTATUS is `gql_status`, for each
/// that is given.
///
/// # Panics
///
/// Panics if the statement succeeded, if the message lacks `message`, or if
/// the error's GQLSTATUS is not `gql_status`.
pub fn assert_error(result: Result<QueryResult>, message: Option<&str>, gql_status: Option<&str>) {
    let expected = match (message, gql_status) {
        (Some(message), Some(status)) => {
            format!("error containing '{message}' with GQLSTATUS {status}")
        }
        (Some(message), None) => format!("error containing '{message}'"),
        (None, Some(status)) => format!("error with GQLSTATUS {status}"),
        (None, None) => "an error".to_string(),
    };
    let Err(error) = result else {
        panic!("Expected {expected} but query succeeded");
    };
    let text = error.to_string();
    if let Some(message) = message {
        assert!(
            text.contains(message),
            "Error '{text}' does not contain '{message}'"
        );
    }
    if let Some(status) = gql_status {
        let actual = grafeo_common::utils::GqlStatus::from(&error).to_string();
        assert_eq!(
            actual, status,
            "Error '{text}' has GQLSTATUS {actual}, expected {status}"
        );
    }
}

/// The XSD namespace an expected `xsd:` datatype stands for.
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
/// The RDF namespace an expected `rdf:` datatype stands for.
const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// A literal's text as N-Triples writes it, between the quotes.
fn escape_lexical(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

/// A result cell as the N-Triples form of the RDF term it holds: `<iri>`,
/// `_:label`, `"text"` (an `xsd:string`), `"text"@tag` or `"text"^^<datatype>`;
/// `null` when unbound.
///
/// # Panics
///
/// On a value that is not an RDF term, and on a term of a kind the runner
/// does not know: a `terms:` expectation compares terms only.
#[must_use]
pub fn term_to_string(value: &Value) -> String {
    use grafeo_common::types::rdf::{Literal, Term};
    let term = match value {
        Value::Null => return "null".to_string(),
        Value::RdfTerm(term) => term,
        other => panic!("a `terms:` expectation got a value that is not an RDF term: {other:?}"),
    };
    match term.as_ref() {
        Term::Iri(iri) => format!("<{}>", iri.as_str()),
        Term::BlankNode(node) => format!("_:{}", node.id()),
        Term::Literal(literal) => {
            let text = escape_lexical(literal.value());
            match literal.language() {
                Some(tag) => format!("\"{text}\"@{}", tag.to_lowercase()),
                None if literal.datatype() == Literal::XSD_STRING => format!("\"{text}\""),
                None => format!("\"{text}\"^^<{}>", literal.datatype()),
            }
        }
        other => panic!(
            "{}",
            grafeo_common::types::rdf::UnsupportedTermKind::of(other)
        ),
    }
}

/// An expected term written for a `terms:` expectation, in the form
/// [`term_to_string`] gives: an `xsd:` or `rdf:` datatype expanded, an
/// explicit `xsd:string` dropped (it is the simple literal) and a language
/// tag lowercased.
#[must_use]
pub fn normalize_expected_term(expected: &str) -> String {
    let Some(rest) = expected.strip_prefix('"') else {
        return expected.to_string();
    };
    // The closing quote is the last one: a datatype or tag has none.
    let Some(end) = rest.rfind('"') else {
        return expected.to_string();
    };
    let (text, suffix) = (&rest[..end], &rest[end + 1..]);
    if let Some(tag) = suffix.strip_prefix('@') {
        return format!("\"{text}\"@{}", tag.to_lowercase());
    }
    if let Some(datatype) = suffix.strip_prefix("^^") {
        let iri = if let Some(local) = datatype.strip_prefix("xsd:") {
            format!("{XSD}{local}")
        } else if let Some(local) = datatype.strip_prefix("rdf:") {
            format!("{RDF}{local}")
        } else {
            datatype
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        };
        if iri == format!("{XSD}string") {
            return format!("\"{text}\"");
        }
        return format!("\"{text}\"^^<{iri}>");
    }
    expected.to_string()
}

/// Assert that each cell holds the exact RDF term `expected` names (see
/// [`term_to_string`] and [`normalize_expected_term`]), in order when
/// `ordered`, as a multiset otherwise. A blank node matches a blank node; in
/// order, the labels must also correspond one to one, as `_:a` and `_:b` in
/// the expectation name two different nodes.
///
/// # Panics
///
/// When a row or a cell differs, or a cell holds no RDF term.
pub fn assert_terms(result: &QueryResult, expected: &[Vec<String>], ordered: bool) {
    let actual: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(term_to_string).collect())
        .collect();
    let expected: Vec<Vec<String>> = expected
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| normalize_expected_term(cell))
                .collect()
        })
        .collect();
    let blank = |cell: &String| {
        if cell.starts_with("_:") {
            "_:".to_string()
        } else {
            cell.clone()
        }
    };
    let without_labels = |rows: &[Vec<String>]| -> Vec<Vec<String>> {
        rows.iter()
            .map(|row| row.iter().map(blank).collect())
            .collect()
    };
    let (mut got, mut want) = (without_labels(&actual), without_labels(&expected));
    if !ordered {
        got.sort();
        want.sort();
    }
    assert_eq!(
        got, want,
        "terms differ\nactual:   {actual:?}\nexpected: {expected:?}"
    );
    if ordered {
        let mut pairs: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        let mut back: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for (row_a, row_e) in actual.iter().zip(&expected) {
            for (a, e) in row_a.iter().zip(row_e) {
                if a.starts_with("_:") {
                    assert_eq!(
                        *pairs.entry(e.as_str()).or_insert(a.as_str()),
                        a.as_str(),
                        "{e} names two blank nodes\nactual: {actual:?}"
                    );
                    assert_eq!(
                        *back.entry(a.as_str()).or_insert(e.as_str()),
                        e.as_str(),
                        "blank node {a} is named both {e} and another label\nactual: {actual:?}"
                    );
                }
            }
        }
    }
}

// Include the generated test code
#[cfg(test)]
mod generated {
    use super::*;
    use grafeo_engine::GrafeoDB;

    include!(concat!(env!("OUT_DIR"), "/spec_tests.rs"));
}
