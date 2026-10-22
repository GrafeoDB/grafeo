//! Single-shot query execution command.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use grafeo_common::types::Value;

use crate::output::{self, Format, format_duration_ms};
use crate::{OutputFormat, QueryLanguage};

/// Options for the single-shot query command.
pub struct QueryRunOptions<'a> {
    pub path: &'a Path,
    pub query: Option<String>,
    pub file: Option<PathBuf>,
    pub stdin: bool,
    pub params: &'a [String],
    pub lang: QueryLanguage,
    pub format: OutputFormat,
    pub quiet: bool,
    pub timing: bool,
    pub max_width: Option<usize>,
}

/// Run the query command.
pub fn run(opts: QueryRunOptions<'_>) -> Result<()> {
    let QueryRunOptions {
        path,
        query,
        file,
        stdin,
        params,
        lang,
        format,
        quiet,
        timing,
        max_width,
    } = opts;
    // Resolve the query string from one of three sources
    let query_str = if let Some(q) = query {
        q
    } else if let Some(f) = file {
        std::fs::read_to_string(&f)
            .with_context(|| format!("Failed to read query file: {}", f.display()))?
    } else if stdin {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read query from stdin")?;
        buf
    } else {
        anyhow::bail!("Provide a query string, --file, or --stdin");
    };

    // Parse parameters
    let param_map = parse_params(params)?;

    let db = super::open_existing(path)?;
    let session = db.session();

    let result = if param_map.is_empty() {
        execute_query(&session, &query_str, lang)?
    } else {
        execute_query_with_params(&session, &query_str, param_map, lang)?
    };

    let fmt: Format = format.into();

    // Format rows as strings, applying max_width truncation
    let headers = result.columns.clone();
    let rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| truncate_value(format_value(v), max_width))
                .collect()
        })
        .collect();

    output::print_result_table(&headers, &rows, fmt, quiet);

    // Print row count + optional timing footer
    if !quiet && fmt == Format::Table {
        let timing_str = if timing {
            result
                .execution_time_ms
                .map_or_else(String::new, |ms| format!(" ({})", format_duration_ms(ms)))
        } else {
            String::new()
        };
        println!(
            "{} row{}{}",
            rows.len(),
            if rows.len() == 1 { "" } else { "s" },
            timing_str
        );
    }

    Ok(())
}

/// Execute a query without parameters.
fn execute_query(
    session: &grafeo_engine::Session,
    query: &str,
    lang: QueryLanguage,
) -> Result<grafeo_engine::database::QueryResult> {
    let result = match lang {
        QueryLanguage::Gql => session.execute(query),
        #[cfg(feature = "cypher")]
        QueryLanguage::Cypher => session.execute_cypher(query),
        #[cfg(not(feature = "cypher"))]
        QueryLanguage::Cypher => {
            anyhow::bail!("Cypher support not enabled (compile with --features cypher)")
        }
        #[cfg(feature = "sparql")]
        QueryLanguage::Sparql => session.execute_sparql(query),
        #[cfg(not(feature = "sparql"))]
        QueryLanguage::Sparql => {
            anyhow::bail!("SPARQL support not enabled (compile with --features sparql)")
        }
        #[cfg(feature = "sql-pgq")]
        QueryLanguage::Sql => session.execute_sql(query),
        #[cfg(not(feature = "sql-pgq"))]
        QueryLanguage::Sql => {
            anyhow::bail!("SQL/PGQ support not enabled (compile with --features sql-pgq)")
        }
    };
    result.context("Query execution failed")
}

/// Execute a query with parameters.
fn execute_query_with_params(
    session: &grafeo_engine::Session,
    query: &str,
    params: std::collections::HashMap<String, Value>,
    lang: QueryLanguage,
) -> Result<grafeo_engine::database::QueryResult> {
    let result = match lang {
        QueryLanguage::Gql => session.execute_with_params(query, params),
        // Cypher with params falls back to GQL execute_with_params
        #[cfg(feature = "cypher")]
        QueryLanguage::Cypher => session.execute_with_params(query, params),
        #[cfg(not(feature = "cypher"))]
        QueryLanguage::Cypher => {
            anyhow::bail!("Cypher support not enabled (compile with --features cypher)")
        }
        #[cfg(feature = "sparql")]
        QueryLanguage::Sparql => session.execute_sparql_with_params(query, params),
        #[cfg(not(feature = "sparql"))]
        QueryLanguage::Sparql => {
            anyhow::bail!("SPARQL support not enabled (compile with --features sparql)")
        }
        #[cfg(feature = "sql-pgq")]
        QueryLanguage::Sql => session.execute_sql_with_params(query, params),
        #[cfg(not(feature = "sql-pgq"))]
        QueryLanguage::Sql => {
            anyhow::bail!("SQL/PGQ support not enabled (compile with --features sql-pgq)")
        }
    };
    result.context("Query execution failed")
}

/// Parse key=value parameter strings into a HashMap.
fn parse_params(params: &[String]) -> Result<std::collections::HashMap<String, Value>> {
    let mut map = std::collections::HashMap::new();
    for param in params {
        let (key, value) = param.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("Invalid parameter format: '{}' (expected key=value)", param)
        })?;

        let value = parse_value(value);
        map.insert(key.to_string(), value);
    }
    Ok(map)
}

/// Parse a string value into a typed Value.
fn parse_value(s: &str) -> Value {
    // Try integer
    if let Ok(i) = s.parse::<i64>() {
        return Value::Int64(i);
    }
    // Try float
    if let Ok(f) = s.parse::<f64>() {
        return Value::Float64(f);
    }
    // Try boolean
    match s.to_lowercase().as_str() {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }
    // Default to string
    Value::from(s)
}

/// Truncate a string to `max_width` characters, appending `...` if truncated.
fn truncate_value(s: String, max_width: Option<usize>) -> String {
    match max_width {
        Some(w) if s.len() > w && w > 3 => {
            let mut truncated = s;
            truncated.truncate(w - 3);
            truncated.push_str("...");
            truncated
        }
        _ => s,
    }
}

/// Format a Value for display.
pub fn format_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => format!("{f}"),
        Value::String(s) => format!("\"{s}\""),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(format_value).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Map(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Bytes(b) => format!("<{} bytes>", b.len()),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- truncate_value ---

    #[test]
    fn test_truncate_none_returns_original() {
        assert_eq!(truncate_value("hello world".into(), None), "hello world");
    }

    #[test]
    fn test_truncate_short_string_unchanged() {
        assert_eq!(truncate_value("hi".into(), Some(10)), "hi");
    }

    #[test]
    fn test_truncate_exact_width_unchanged() {
        assert_eq!(truncate_value("12345".into(), Some(5)), "12345");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate_value("hello world".into(), Some(8)), "hello...");
    }

    #[test]
    fn test_truncate_width_too_small_returns_original() {
        // When max_width <= 3, can't fit "...", so return original
        assert_eq!(truncate_value("hello".into(), Some(3)), "hello");
        assert_eq!(truncate_value("hello".into(), Some(2)), "hello");
    }

    #[test]
    fn test_truncate_width_4() {
        assert_eq!(truncate_value("hello".into(), Some(4)), "h...");
    }

    // --- parse_value ---

    #[test]
    fn test_parse_value_integer() {
        assert!(matches!(parse_value("42"), Value::Int64(42)));
        assert!(matches!(parse_value("-7"), Value::Int64(-7)));
        assert!(matches!(parse_value("0"), Value::Int64(0)));
    }

    #[test]
    fn test_parse_value_float() {
        assert!(matches!(parse_value("1.5"), Value::Float64(f) if (f - 1.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_parse_value_bool() {
        assert!(matches!(parse_value("true"), Value::Bool(true)));
        assert!(matches!(parse_value("True"), Value::Bool(true)));
        assert!(matches!(parse_value("FALSE"), Value::Bool(false)));
    }

    #[test]
    fn test_parse_value_null() {
        assert!(matches!(parse_value("null"), Value::Null));
        assert!(matches!(parse_value("NULL"), Value::Null));
    }

    #[test]
    fn test_parse_value_string() {
        match parse_value("hello") {
            Value::String(s) => assert_eq!(s.as_str(), "hello"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    // --- parse_params ---

    #[test]
    fn test_parse_params_valid() {
        let params = vec!["name=Alix".to_string(), "age=30".to_string()];
        let map = parse_params(&params).unwrap();
        assert_eq!(map.len(), 2);
        assert!(matches!(map.get("age"), Some(Value::Int64(30))));
    }

    #[test]
    fn test_parse_params_empty() {
        let map = parse_params(&[]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_params_invalid_format() {
        let params = vec!["no_equals_sign".to_string()];
        assert!(parse_params(&params).is_err());
    }

    #[test]
    fn test_parse_params_value_with_equals() {
        // "key=val=ue" should split at first '='
        let params = vec!["expr=a=b".to_string()];
        let map = parse_params(&params).unwrap();
        match map.get("expr") {
            Some(Value::String(s)) => assert_eq!(s.as_str(), "a=b"),
            other => panic!("expected String(\"a=b\"), got {other:?}"),
        }
    }

    // --- format_value ---

    #[test]
    fn test_format_value_null() {
        assert_eq!(format_value(&Value::Null), "null");
    }

    #[test]
    fn test_format_value_bool() {
        assert_eq!(format_value(&Value::Bool(true)), "true");
        assert_eq!(format_value(&Value::Bool(false)), "false");
    }

    #[test]
    fn test_format_value_int() {
        assert_eq!(format_value(&Value::Int64(42)), "42");
        assert_eq!(format_value(&Value::Int64(-1)), "-1");
    }

    #[test]
    fn test_format_value_float() {
        assert_eq!(format_value(&Value::Float64(1.5)), "1.5");
    }

    #[test]
    fn test_format_value_string() {
        assert_eq!(format_value(&Value::from("hello")), "\"hello\"");
    }

    #[test]
    fn test_format_value_list() {
        let list = Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into());
        assert_eq!(format_value(&list), "[1, 2, 3]");
    }

    #[test]
    fn test_format_value_bytes() {
        assert_eq!(
            format_value(&Value::Bytes(vec![0, 1, 2, 3, 4].into())),
            "<5 bytes>"
        );
    }
}
