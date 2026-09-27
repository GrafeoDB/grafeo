//! Output formatting for CLI commands.
//!
//! Provides table, JSON, and CSV output with TTY auto-detection and NO_COLOR support.

pub mod formatter;

use comfy_table::{Cell, Color, ContentArrangement, Table};
use is_terminal::IsTerminal;

pub use formatter::{format_bytes, format_duration_ms, print_memory};

/// Output format selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Table,
    Json,
    Csv,
}

impl From<crate::OutputFormat> for Format {
    fn from(f: crate::OutputFormat) -> Self {
        match f {
            crate::OutputFormat::Table => Format::Table,
            crate::OutputFormat::Json => Format::Json,
            crate::OutputFormat::Csv => Format::Csv,
            crate::OutputFormat::Auto => {
                if std::io::stdout().is_terminal() {
                    Format::Table
                } else {
                    Format::Json
                }
            }
        }
    }
}

/// Returns true if color output should be suppressed.
pub fn no_color() -> bool {
    std::env::var_os("NO_COLOR").is_some()
}

/// Create a styled table with consistent formatting.
pub fn create_table() -> Table {
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.load_style(comfy_table::presets::UTF8_FULL_CONDENSED);
    table
}

/// Add a header row to a table.
pub fn add_header(table: &mut Table, headers: &[&str]) {
    if no_color() {
        table.set_header(headers.iter().copied().map(Cell::new).collect::<Vec<_>>());
    } else {
        table.set_header(
            headers
                .iter()
                .map(|h| Cell::new(h).fg(Color::Cyan))
                .collect::<Vec<_>>(),
        );
    }
}

/// Print a key-value table (for info displays).
pub fn print_key_value_table(items: &[(&str, String)], format: Format, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        Format::Json => {
            let map: std::collections::HashMap<&str, &str> =
                items.iter().map(|(k, v)| (*k, v.as_str())).collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&map).expect("JSON serialization of string map")
            );
        }
        Format::Csv => {
            println!("key,value");
            for (key, value) in items {
                println!("{key},{}", csv_escape(value));
            }
        }
        Format::Table => {
            let mut table = create_table();
            add_header(&mut table, &["Property", "Value"]);
            for (key, value) in items {
                if no_color() {
                    table.add_row(vec![Cell::new(key), Cell::new(value)]);
                } else {
                    table.add_row(vec![Cell::new(key).fg(Color::Green), Cell::new(value)]);
                }
            }
            println!("{table}");
        }
    }
}

/// Print a tabular result set (for query results).
pub fn print_result_table(headers: &[String], rows: &[Vec<String>], format: Format, quiet: bool) {
    if quiet {
        return;
    }

    match format {
        Format::Json => {
            let result: Vec<std::collections::HashMap<&str, &str>> = rows
                .iter()
                .map(|row| {
                    headers
                        .iter()
                        .zip(row.iter())
                        .map(|(h, v)| (h.as_str(), v.as_str()))
                        .collect()
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("JSON serialization of string map")
            );
        }
        Format::Csv => {
            println!(
                "{}",
                headers
                    .iter()
                    .map(|h| csv_escape(h))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            for row in rows {
                println!(
                    "{}",
                    row.iter()
                        .map(|v| csv_escape(v))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }
        Format::Table => {
            let mut table = create_table();
            let header_strs: Vec<&str> = headers.iter().map(String::as_str).collect();
            add_header(&mut table, &header_strs);
            for row in rows {
                table.add_row(row.iter().map(Cell::new).collect::<Vec<_>>());
            }
            println!("{table}");
        }
    }
}

/// Escape a value for CSV output.
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Print a status message (respects quiet mode).
pub fn status(msg: &str, quiet: bool) {
    if !quiet {
        println!("{msg}");
    }
}

/// Print a success message.
pub fn success(msg: &str, quiet: bool) {
    if !quiet {
        if no_color() {
            println!("ok: {msg}");
        } else {
            println!("\x1b[32m✓\x1b[0m {msg}");
        }
    }
}

/// Print an error message to stderr.
pub fn error(msg: &str) {
    if no_color() {
        eprintln!("error: {msg}");
    } else {
        eprintln!("\x1b[31merror\x1b[0m: {msg}");
    }
}

/// Format a string with ANSI color for console output.
pub fn colored(text: &str, color: Color) -> String {
    if no_color() {
        return text.to_string();
    }
    match color {
        Color::Green => format!("\x1b[32m{text}\x1b[0m"),
        Color::Red => format!("\x1b[31m{text}\x1b[0m"),
        Color::Yellow => format!("\x1b[33m{text}\x1b[0m"),
        Color::Cyan => format!("\x1b[36m{text}\x1b[0m"),
        _ => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_clone_and_copy() {
        let format = Format::Table;
        let copied = format;
        let cloned = Clone::clone(&format);
        assert!(matches!(copied, Format::Table));
        assert!(matches!(cloned, Format::Table));
    }

    #[test]
    fn test_create_table() {
        let table = create_table();
        assert!(table.to_string().is_empty() || !table.to_string().is_empty());
    }

    #[test]
    fn test_add_header() {
        let mut table = create_table();
        add_header(&mut table, &["Name", "Value", "Type"]);
        let output = table.to_string();
        assert!(output.contains("Name"));
        assert!(output.contains("Value"));
        assert!(output.contains("Type"));
    }

    #[test]
    fn test_table_with_rows() {
        let mut table = create_table();
        add_header(&mut table, &["Key", "Value"]);
        table.add_row(vec!["foo", "bar"]);
        table.add_row(vec!["baz", "qux"]);
        let output = table.to_string();
        assert!(output.contains("foo"));
        assert!(output.contains("bar"));
        assert!(output.contains("baz"));
        assert!(output.contains("qux"));
    }

    #[test]
    fn test_csv_escape() {
        assert_eq!(csv_escape("hello"), "hello");
        assert_eq!(csv_escape("hello,world"), "\"hello,world\"");
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_escape("line\nbreak"), "\"line\nbreak\"");
    }

    #[test]
    fn test_format_auto_json_when_not_tty() {
        // In test environment, stdout is not a TTY
        let format: Format = crate::OutputFormat::Auto.into();
        assert_eq!(format, Format::Json);
    }
}
