//! CSV and JSON Lines import commands.

use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};

use crate::output;
use crate::{ImportCommands, OutputFormat};

/// Run an import subcommand.
pub fn run(cmd: ImportCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        ImportCommands::Csv {
            file,
            path,
            headers,
            separator,
            label,
        } => import_csv(&file, &path, headers, separator.as_deref(), &label, quiet),
        ImportCommands::Jsonl { file, path, label } => import_jsonl(&file, &path, &label, quiet),
    }
}

/// Build a `LOAD DATA` query for CSV import and execute it via the engine.
fn import_csv(
    file: &Path,
    db_path: &Path,
    headers: bool,
    separator: Option<&str>,
    label: &str,
    quiet: bool,
) -> Result<()> {
    // Canonicalize file path so the engine can find it
    let abs_path = file
        .canonicalize()
        .with_context(|| format!("File not found: {}", file.display()))?;
    let path_str = escape_single_quotes(&abs_path.to_string_lossy().replace('\\', "/"));

    // Read column headers from the first line so we can build the property map
    let columns = read_csv_columns(&abs_path, headers, separator)?;

    // Normalize separator alias (e.g. "TAB" -> '\t') before embedding in the query
    let normalized_sep = separator.map(parse_separator).transpose()?;

    // Build the LOAD DATA query (label is sanitized inside build_csv_query)
    let safe_label = sanitize_identifier(label);
    let query = build_csv_query(&path_str, headers, normalized_sep, &safe_label, &columns)?;

    // Open or create the database and execute
    let db = open_or_create(db_path)?;
    let session = db.session();

    // Count existing nodes before import to compute the delta
    let before_count = count_nodes_with_label(&session, &safe_label);

    session
        .execute(&query)
        .with_context(|| "Import query failed")?;

    let count = count_nodes_with_label(&session, &safe_label) - before_count;

    output::status(
        &format!(
            "Imported {count} nodes with label '{label}' from {}",
            file.display()
        ),
        quiet,
    );
    Ok(())
}

/// Read column names from the first line of a CSV file.
fn read_csv_columns(
    abs_path: &Path,
    headers: bool,
    separator: Option<&str>,
) -> Result<Vec<String>> {
    if headers {
        let f = std::fs::File::open(abs_path)
            .with_context(|| format!("Failed to open file: {}", abs_path.display()))?;
        let mut reader = BufReader::new(f);
        let mut header_line = String::new();
        reader
            .read_line(&mut header_line)
            .context("Failed to read header line")?;
        let sep = separator.unwrap_or(",");
        let sep_char = parse_separator(sep)?;
        Ok(header_line
            .trim()
            .split(sep_char)
            .map(|h| h.trim().trim_matches('"').to_string())
            .filter(|h| !h.is_empty())
            .collect::<Vec<_>>())
    } else {
        Ok(Vec::new())
    }
}

/// Build a LOAD DATA query string for CSV import.
///
/// This is a pure function that constructs the GQL query from its components,
/// making it easy to test independently of file I/O.
fn build_csv_query(
    path_str: &str,
    headers: bool,
    separator: Option<char>,
    label: &str,
    columns: &[String],
) -> Result<String> {
    let header_clause = if headers { " WITH HEADERS" } else { "" };
    let separator_clause = match separator {
        Some(ch) => format!(
            " FIELDTERMINATOR '{}'",
            escape_single_quotes(&ch.to_string())
        ),
        None => String::new(),
    };

    let insert_clause = if headers && !columns.is_empty() {
        let props = columns
            .iter()
            .map(|col| {
                let safe_col = sanitize_identifier(col);
                format!("{safe_col}: row.{safe_col}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("INSERT (:{label} {{{props}}})")
    } else {
        // Without headers, we cannot map columns by name.
        // Insert nodes with no properties (positional access is limited).
        format!("INSERT (:{label} {{}})")
    };

    Ok(format!(
        "LOAD DATA FROM '{path_str}' FORMAT CSV{header_clause} AS row{separator_clause} {insert_clause}"
    ))
}

/// Build a `LOAD DATA` query for JSONL import and execute it via the engine.
fn import_jsonl(file: &Path, db_path: &Path, label: &str, quiet: bool) -> Result<()> {
    let abs_path = file
        .canonicalize()
        .with_context(|| format!("File not found: {}", file.display()))?;
    let path_str = escape_single_quotes(&abs_path.to_string_lossy().replace('\\', "/"));

    // Read the first line to discover JSON keys for property mapping
    let keys = read_jsonl_keys(&abs_path)?;

    let safe_label = sanitize_identifier(label);
    let query = build_jsonl_query(&path_str, &safe_label, &keys);

    let db = open_or_create(db_path)?;
    let session = db.session();

    let before_count = count_nodes_with_label(&session, &safe_label);

    session
        .execute(&query)
        .with_context(|| "Import query failed")?;

    let count = count_nodes_with_label(&session, &safe_label) - before_count;

    output::status(
        &format!(
            "Imported {count} nodes with label '{label}' from {}",
            file.display()
        ),
        quiet,
    );
    Ok(())
}

/// Read JSON keys from the first non-empty line of a JSONL file.
fn read_jsonl_keys(abs_path: &Path) -> Result<Vec<String>> {
    let f = std::fs::File::open(abs_path)
        .with_context(|| format!("Failed to open file: {}", abs_path.display()))?;
    let reader = BufReader::new(f);
    let mut keys = Vec::new();
    for line in reader.lines() {
        let line = line.context("Failed to read JSONL file")?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
        {
            keys = obj.keys().cloned().collect();
        }
        break;
    }
    Ok(keys)
}

/// Build a LOAD DATA query string for JSONL import.
///
/// This is a pure function that constructs the GQL query from its components.
fn build_jsonl_query(path_str: &str, label: &str, keys: &[String]) -> String {
    let insert_clause = if keys.is_empty() {
        format!("INSERT (:{label} {{}})")
    } else {
        let props = keys
            .iter()
            .map(|key| {
                let safe_key = sanitize_identifier(key);
                format!("{safe_key}: row.{safe_key}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("INSERT (:{label} {{{props}}})")
    };

    format!("LOAD DATA FROM '{path_str}' FORMAT JSONL AS row {insert_clause}")
}

/// Open an existing database or create a new one at the given path.
fn open_or_create(path: &Path) -> Result<grafeo_engine::GrafeoDB> {
    if path.exists() {
        super::open_existing(path)
    } else {
        grafeo_engine::GrafeoDB::open(path)
            .with_context(|| format!("Failed to create database at {}", path.display()))
    }
}

/// Parse a separator string into a single character.
fn parse_separator(s: &str) -> Result<char> {
    match s {
        "\\t" | "TAB" | "tab" => Ok('\t'),
        _ if s.len() == 1 => Ok(s.chars().next().unwrap()),
        _ => anyhow::bail!("Separator must be a single character, got: '{s}'"),
    }
}

/// Escape single quotes in a string for safe embedding in a GQL string literal.
fn escape_single_quotes(s: &str) -> String {
    s.replace('\'', "\\'")
}

/// Count nodes with a given label.
fn count_nodes_with_label(session: &grafeo_engine::Session, label: &str) -> i64 {
    session
        .execute(&format!("MATCH (n:{label}) RETURN count(n) AS c"))
        .ok()
        .and_then(|r| r.rows().first().cloned())
        .and_then(|row| row.first().cloned())
        .and_then(|v| match v {
            grafeo_common::types::Value::Int64(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0)
}

/// Sanitize a column/key name for use as a GQL identifier.
///
/// Strips characters that are not alphanumeric or underscores, and ensures
/// the identifier does not start with a digit.
fn sanitize_identifier(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "_col".to_string()
    } else if sanitized.starts_with(|c: char| c.is_ascii_digit()) {
        format!("_{sanitized}")
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── sanitize_identifier ──────────────────────────────────────────

    #[test]
    fn sanitize_identifier_alphanumeric_unchanged() {
        assert_eq!(sanitize_identifier("name"), "name");
        assert_eq!(sanitize_identifier("age_years"), "age_years");
        assert_eq!(sanitize_identifier("col123"), "col123");
    }

    #[test]
    fn sanitize_identifier_replaces_special_chars() {
        assert_eq!(sanitize_identifier("first-name"), "first_name");
        assert_eq!(sanitize_identifier("col.value"), "col_value");
        assert_eq!(sanitize_identifier("a b c"), "a_b_c");
    }

    #[test]
    fn sanitize_identifier_empty_returns_placeholder() {
        assert_eq!(sanitize_identifier(""), "_col");
    }

    #[test]
    fn sanitize_identifier_leading_digit_gets_prefix() {
        assert_eq!(sanitize_identifier("1col"), "_1col");
        assert_eq!(sanitize_identifier("42"), "_42");
    }

    // ── parse_separator ──────────────────────────────────────────────

    #[test]
    fn parse_separator_single_char() {
        assert_eq!(parse_separator(",").unwrap(), ',');
        assert_eq!(parse_separator(";").unwrap(), ';');
        assert_eq!(parse_separator("|").unwrap(), '|');
    }

    #[test]
    fn parse_separator_tab_aliases() {
        assert_eq!(parse_separator("\\t").unwrap(), '\t');
        assert_eq!(parse_separator("TAB").unwrap(), '\t');
        assert_eq!(parse_separator("tab").unwrap(), '\t');
    }

    #[test]
    fn parse_separator_multi_char_error() {
        let result = parse_separator(",,");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("single character"), "unexpected error: {msg}");
    }

    // ── escape_single_quotes ─────────────────────────────────────────

    #[test]
    fn escape_single_quotes_no_quotes() {
        assert_eq!(escape_single_quotes("hello"), "hello");
    }

    #[test]
    fn escape_single_quotes_with_quotes() {
        assert_eq!(escape_single_quotes("it's"), "it\\'s");
    }

    // ── build_csv_query ──────────────────────────────────────────────

    #[test]
    fn build_csv_query_with_headers() {
        let columns = vec!["name".to_string(), "age".to_string()];
        let query = build_csv_query("/data/file.csv", true, None, "Person", &columns).unwrap();

        assert!(query.starts_with("LOAD DATA FROM '/data/file.csv' FORMAT CSV"));
        assert!(query.contains("WITH HEADERS"));
        assert!(query.contains("name: row.name"));
        assert!(query.contains("age: row.age"));
        assert!(query.contains(":Person"));
    }

    #[test]
    fn build_csv_query_without_headers() {
        let query = build_csv_query("/data/file.csv", false, None, "Item", &[]).unwrap();

        assert!(query.contains("FORMAT CSV AS row"));
        assert!(!query.contains("WITH HEADERS"));
        assert!(query.contains("INSERT (:Item {})"));
    }

    #[test]
    fn build_csv_query_with_separator() {
        let columns = vec!["x".to_string()];
        let query = build_csv_query("/data/f.csv", true, Some(';'), "Node", &columns).unwrap();

        assert!(query.contains("FIELDTERMINATOR ';'"));
    }

    #[test]
    fn build_csv_query_with_tab_separator() {
        let columns = vec!["col1".to_string()];
        let query = build_csv_query("/data/f.tsv", true, Some('\t'), "Row", &columns).unwrap();

        assert!(query.contains("FIELDTERMINATOR '"));
    }

    #[test]
    fn build_csv_query_sanitizes_column_names() {
        let columns = vec!["first-name".to_string(), "e.mail".to_string()];
        let query = build_csv_query("/f.csv", true, None, "User", &columns).unwrap();

        assert!(query.contains("first_name: row.first_name"));
        assert!(query.contains("e_mail: row.e_mail"));
    }

    #[test]
    fn build_csv_query_headers_true_but_empty_columns() {
        let query = build_csv_query("/f.csv", true, None, "Empty", &[]).unwrap();
        assert!(query.contains("INSERT (:Empty {})"));
    }

    // ── build_jsonl_query ────────────────────────────────────────────

    #[test]
    fn build_jsonl_query_with_keys() {
        let keys = vec!["name".to_string(), "city".to_string()];
        let query = build_jsonl_query("/data/file.jsonl", "Person", &keys);

        assert!(query.starts_with("LOAD DATA FROM '/data/file.jsonl' FORMAT JSONL"));
        assert!(query.contains("name: row.name"));
        assert!(query.contains("city: row.city"));
        assert!(query.contains(":Person"));
    }

    #[test]
    fn build_jsonl_query_no_keys() {
        let query = build_jsonl_query("/data/empty.jsonl", "Thing", &[]);
        assert!(query.contains("INSERT (:Thing {})"));
    }

    #[test]
    fn build_jsonl_query_sanitizes_keys() {
        let keys = vec!["user-id".to_string(), "e.mail".to_string()];
        let query = build_jsonl_query("/f.jsonl", "Account", &keys);

        assert!(query.contains("user_id: row.user_id"));
        assert!(query.contains("e_mail: row.e_mail"));
    }

    // ── read_csv_columns ─────────────────────────────────────────────

    #[test]
    fn read_csv_columns_with_headers() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        std::fs::write(&file_path, "name,age,city\nAlix,30,Amsterdam\n").unwrap();

        let columns = read_csv_columns(&file_path, true, None).unwrap();
        assert_eq!(columns, vec!["name", "age", "city"]);
    }

    #[test]
    fn read_csv_columns_with_custom_separator() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.tsv");
        std::fs::write(&file_path, "name\tage\nAlix\t30\n").unwrap();

        let columns = read_csv_columns(&file_path, true, Some("\\t")).unwrap();
        assert_eq!(columns, vec!["name", "age"]);
    }

    #[test]
    fn read_csv_columns_no_headers() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        std::fs::write(&file_path, "Alix,30\n").unwrap();

        let columns = read_csv_columns(&file_path, false, None).unwrap();
        assert!(columns.is_empty(), "{columns:?}");
    }

    #[test]
    fn read_csv_columns_quoted_headers() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        std::fs::write(&file_path, "\"name\",\"age\"\nAlix,30\n").unwrap();

        let columns = read_csv_columns(&file_path, true, None).unwrap();
        assert_eq!(columns, vec!["name", "age"]);
    }

    #[test]
    fn read_csv_columns_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.csv");
        std::fs::write(&file_path, "").unwrap();

        let columns = read_csv_columns(&file_path, true, None).unwrap();
        assert!(columns.is_empty(), "{columns:?}");
    }

    // ── read_jsonl_keys ──────────────────────────────────────────────

    #[test]
    fn read_jsonl_keys_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.jsonl");
        let mut f = std::fs::File::create(&file_path).unwrap();
        writeln!(f, r#"{{"name": "Alix", "age": 30}}"#).unwrap();
        writeln!(f, r#"{{"name": "Gus", "age": 25}}"#).unwrap();

        let keys = read_jsonl_keys(&file_path).unwrap();
        assert!(keys.contains(&"name".to_string()));
        assert!(keys.contains(&"age".to_string()));
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn read_jsonl_keys_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty.jsonl");
        std::fs::write(&file_path, "").unwrap();

        let keys = read_jsonl_keys(&file_path).unwrap();
        assert!(keys.is_empty(), "{keys:?}");
    }

    #[test]
    fn read_jsonl_keys_skips_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.jsonl");
        let mut f = std::fs::File::create(&file_path).unwrap();
        writeln!(f).unwrap();
        writeln!(f, r#"{{"city": "Berlin"}}"#).unwrap();

        let keys = read_jsonl_keys(&file_path).unwrap();
        assert_eq!(keys, vec!["city"]);
    }

    #[test]
    fn read_jsonl_keys_invalid_json_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("bad.jsonl");
        std::fs::write(&file_path, "not valid json\n").unwrap();

        let keys = read_jsonl_keys(&file_path).unwrap();
        assert!(keys.is_empty(), "{keys:?}");
    }

    // ── import_csv error cases ───────────────────────────────────────

    #[test]
    fn import_csv_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.csv");
        let db_path = dir.path().join("test.db");

        let result = import_csv(&missing, &db_path, true, None, "Node", true);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not found") || msg.contains("nonexistent"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn import_jsonl_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nonexistent.jsonl");
        let db_path = dir.path().join("test.db");

        let result = import_jsonl(&missing, &db_path, "Node", true);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not found") || msg.contains("nonexistent"),
            "unexpected: {msg}"
        );
    }

    // ── end-to-end import tests ──────────────────────────────────────

    #[test]
    fn import_csv_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("people.csv");
        std::fs::write(&csv_path, "name,age\nAlix,30\nGus,25\n").unwrap();
        let db_path = dir.path().join("test.db");

        import_csv(&csv_path, &db_path, true, None, "Person", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "Person");
        assert_eq!(count, 2, "expected 2 Person nodes");
    }

    #[test]
    fn import_csv_with_separator_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("data.tsv");
        std::fs::write(&csv_path, "name\tcity\nAlix\tAmsterdam\nGus\tBerlin\n").unwrap();
        let db_path = dir.path().join("test.db");

        import_csv(&csv_path, &db_path, true, Some("\\t"), "Citizen", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "Citizen");
        assert_eq!(count, 2, "expected 2 Citizen nodes");
    }

    #[test]
    fn import_csv_no_headers_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("raw.csv");
        std::fs::write(&csv_path, "Alix,30\nGus,25\nVincent,35\n").unwrap();
        let db_path = dir.path().join("test.db");

        import_csv(&csv_path, &db_path, false, None, "RawRow", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "RawRow");
        assert_eq!(count, 3, "expected 3 RawRow nodes");
    }

    #[test]
    fn import_csv_empty_file_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("empty.csv");
        std::fs::write(&csv_path, "").unwrap();
        let db_path = dir.path().join("test.db");

        import_csv(&csv_path, &db_path, true, None, "Empty", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "Empty");
        assert_eq!(count, 0, "expected 0 nodes from empty CSV");
    }

    #[test]
    fn import_jsonl_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl_path = dir.path().join("people.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"name": "Alix", "city": "Amsterdam"}}"#).unwrap();
            writeln!(f, r#"{{"name": "Gus", "city": "Berlin"}}"#).unwrap();
        }
        let db_path = dir.path().join("test.db");

        import_jsonl(&jsonl_path, &db_path, "Resident", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "Resident");
        assert_eq!(count, 2, "expected 2 Resident nodes");
    }

    #[test]
    fn import_jsonl_empty_file_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl_path = dir.path().join("empty.jsonl");
        std::fs::write(&jsonl_path, "").unwrap();
        let db_path = dir.path().join("test.db");

        import_jsonl(&jsonl_path, &db_path, "Ghost", true).unwrap();

        let db = open_or_create(&db_path).unwrap();
        let session = db.session();
        let count = count_nodes_with_label(&session, "Ghost");
        assert_eq!(count, 0, "expected 0 nodes from empty JSONL");
    }
}
