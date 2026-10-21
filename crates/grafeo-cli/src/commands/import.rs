//! CSV and JSON Lines import commands.
//!
//! Both run `grafeo_engine::import::import_nodes`, the import the Python and
//! Node.js bindings use too.

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

/// Import each row of a CSV file as a node.
fn import_csv(
    file: &Path,
    db_path: &Path,
    headers: bool,
    separator: Option<&str>,
    label: &str,
    quiet: bool,
) -> Result<()> {
    let delimiter = separator.map(parse_separator).transpose()?;
    import(
        file,
        db_path,
        label,
        quiet,
        Format::Csv { headers, delimiter },
    )
}

/// Import each line of a JSON Lines file as a node.
fn import_jsonl(file: &Path, db_path: &Path, label: &str, quiet: bool) -> Result<()> {
    import(file, db_path, label, quiet, Format::Jsonl)
}

/// The file format, as the command line gives it.
enum Format {
    Csv {
        headers: bool,
        delimiter: Option<char>,
    },
    Jsonl,
}

#[cfg(feature = "gql")]
fn import(file: &Path, db_path: &Path, label: &str, quiet: bool, format: Format) -> Result<()> {
    use grafeo_engine::import::{ImportFormat, import_nodes};

    let format = match format {
        Format::Csv { headers, delimiter } => ImportFormat::Csv { headers, delimiter },
        Format::Jsonl => ImportFormat::Jsonl,
    };
    std::fs::metadata(file).with_context(|| format!("File not found: {}", file.display()))?;
    let db = open_or_create(db_path)?;
    let count = import_nodes(&db.session(), file, label, format)
        .with_context(|| format!("Import of {} failed", file.display()))?;
    output::status(
        &format!(
            "Imported {count} nodes with label '{label}' from {}",
            file.display()
        ),
        quiet,
    );
    Ok(())
}

#[cfg(not(feature = "gql"))]
fn import(
    _file: &Path,
    _db_path: &Path,
    _label: &str,
    _quiet: bool,
    _format: Format,
) -> Result<()> {
    anyhow::bail!("import needs a build with the 'gql' feature")
}

/// Open an existing database or create a new one at the given path.
#[cfg(feature = "gql")]
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
        _ if s.chars().count() == 1 => Ok(s.chars().next().unwrap_or(',')),
        _ => anyhow::bail!("Separator must be a single character, got: '{s}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_separator ─────────────────────────────────────────────

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
        let result = parse_separator("ab");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("single character"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_separator_one_non_ascii_char() {
        assert_eq!(parse_separator("§").unwrap(), '§');
    }

    // ── import ───────────────────────────────────────────────────────

    #[cfg(feature = "gql")]
    mod import {
        use super::super::*;

        /// `RETURN` of `query` on the database at `db_path`, each row as text.
        fn rows(db_path: &Path, query: &str) -> Vec<String> {
            let db = open_or_create(db_path).unwrap();
            let result = db.session().execute(query).unwrap();
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        }

        fn names(db_path: &Path, query: &str) -> Vec<String> {
            let db = open_or_create(db_path).unwrap();
            let result = db.session().execute(query).unwrap();
            let mut names: Vec<String> = result
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    grafeo_common::types::Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            names.sort();
            names
        }

        #[test]
        fn a_missing_file_is_an_error() {
            let dir = tempfile::tempdir().unwrap();
            let missing = dir.path().join("nonexistent.csv");
            let db_path = dir.path().join("test.grafeo");
            let msg = import_csv(&missing, &db_path, true, None, "Node", true)
                .unwrap_err()
                .to_string();
            assert!(msg.contains("File not found"), "unexpected: {msg}");
            let msg = import_jsonl(&missing, &db_path, "Node", true)
                .unwrap_err()
                .to_string();
            assert!(msg.contains("File not found"), "unexpected: {msg}");
            assert!(!db_path.exists(), "a failed import created the database");
        }

        #[test]
        fn csv_columns_keep_their_names() {
            let dir = tempfile::tempdir().unwrap();
            let csv_path = dir.path().join("people.csv");
            std::fs::write(
                &csv_path,
                "\u{feff}name,home city\nAlix,Amsterdam\nGus,Berlin\n",
            )
            .unwrap();
            let db_path = dir.path().join("test.grafeo");
            import_csv(&csv_path, &db_path, true, None, "Person", true).unwrap();
            assert_eq!(
                names(
                    &db_path,
                    "MATCH (p:Person) RETURN p.name + ' ' + p.`home city` AS v"
                ),
                ["Alix Amsterdam", "Gus Berlin"]
            );
        }

        #[test]
        fn a_separator_and_a_label_with_a_space() {
            let dir = tempfile::tempdir().unwrap();
            let csv_path = dir.path().join("data.tsv");
            std::fs::write(&csv_path, "name\tcity\nAlix\tAmsterdam\nGus\tBerlin\n").unwrap();
            let db_path = dir.path().join("test.grafeo");
            import_csv(&csv_path, &db_path, true, Some("\\t"), "City Dweller", true).unwrap();
            assert_eq!(
                names(&db_path, "MATCH (c:`City Dweller`) RETURN c.city AS v"),
                ["Amsterdam", "Berlin"]
            );
        }

        #[test]
        fn csv_columns_without_headers_are_numbered() {
            let dir = tempfile::tempdir().unwrap();
            let csv_path = dir.path().join("raw.csv");
            std::fs::write(&csv_path, "Alix,30\nGus,25\nVincent\n").unwrap();
            let db_path = dir.path().join("test.grafeo");
            import_csv(&csv_path, &db_path, false, None, "RawRow", true).unwrap();
            assert_eq!(
                rows(
                    &db_path,
                    "MATCH (r:RawRow) RETURN r.column0 AS a, r.column1 AS b ORDER BY a"
                ),
                [
                    "[String(\"Alix\"), String(\"30\")]",
                    "[String(\"Gus\"), String(\"25\")]",
                    "[String(\"Vincent\"), Null]",
                ]
            );
        }

        #[test]
        fn an_empty_file_imports_nothing() {
            let dir = tempfile::tempdir().unwrap();
            let csv_path = dir.path().join("empty.csv");
            std::fs::write(&csv_path, "").unwrap();
            let jsonl_path = dir.path().join("empty.jsonl");
            std::fs::write(&jsonl_path, "").unwrap();
            let db_path = dir.path().join("test.grafeo");
            import_csv(&csv_path, &db_path, true, None, "Empty", true).unwrap();
            import_jsonl(&jsonl_path, &db_path, "Ghost", true).unwrap();
            assert_eq!(
                rows(&db_path, "MATCH (n) RETURN count(n) AS c"),
                ["[Int64(0)]"]
            );
        }

        #[cfg(feature = "jsonl-import")]
        #[test]
        fn jsonl_keys_of_later_lines_are_kept() {
            let dir = tempfile::tempdir().unwrap();
            let jsonl_path = dir.path().join("people.jsonl");
            std::fs::write(
                &jsonl_path,
                "{\"name\": \"Alix\"}\n\n{\"name\": \"Gus\", \"home city\": \"Berlin\"}\n",
            )
            .unwrap();
            let db_path = dir.path().join("test.grafeo");
            import_jsonl(&jsonl_path, &db_path, "Resident", true).unwrap();
            assert_eq!(
                names(&db_path, "MATCH (r:Resident) RETURN r.`home city` AS v"),
                ["Berlin", "Null"]
            );
        }
    }
}
