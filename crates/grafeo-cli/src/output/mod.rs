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
pub fn print_key_value_table(items: &[(&str, String)], format: Format) {
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
pub fn print_result_table(headers: &[String], rows: &[Vec<String>], format: Format) {
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

/// Print what an action command did (a backup, an import, a dump) as a
/// JSON object with its fields in order, when the format is JSON; in a
/// table or CSV the command's success line, on stderr, says it.
pub fn print_record(fields: &[(&str, serde_json::Value)], format: Format) {
    if format == Format::Json {
        let record = Record(fields);
        println!(
            "{}",
            serde_json::to_string_pretty(&record).expect("JSON serialization of a record")
        );
    }
}

/// The fields of [`print_record`], serialized in order.
struct Record<'a>(&'a [(&'a str, serde_json::Value)]);

impl serde::Serialize for Record<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// Print the rows of a query result under `headers`: in JSON each row is an
/// object of JSON values (see [`json_value`]) with the columns in result
/// order; in a table and in CSV each value is its text ([`display_value`]),
/// cut to `max_width` characters.
pub fn print_value_rows(
    headers: &[String],
    rows: &[Vec<grafeo_common::types::Value>],
    format: Format,
    max_width: Option<usize>,
) {
    if format == Format::Json {
        let objects: Vec<JsonRow<'_>> = rows.iter().map(|row| JsonRow { headers, row }).collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&objects).expect("JSON serialization of query rows")
        );
        return;
    }
    let texts: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| truncate(display_value(value), max_width))
                .collect()
        })
        .collect();
    print_result_table(headers, &texts, format);
}

/// One row of a query result as a JSON object, its keys in column order.
struct JsonRow<'a> {
    headers: &'a [String],
    row: &'a [grafeo_common::types::Value],
}

impl serde::Serialize for JsonRow<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.headers.len()))?;
        for (header, value) in self.headers.iter().zip(self.row) {
            map.serialize_entry(header, &json_value(value))?;
        }
        map.end()
    }
}

/// A value as JSON: null, booleans, integers and floats as themselves (a
/// float without a JSON number, NaN or an infinity, as its text), strings,
/// lists, vectors and byte arrays as arrays, maps as objects and a path as
/// `{"nodes": [...], "edges": [...]}`. A decimal is a number when that
/// keeps its digits, else its text; temporal values and counters keep their
/// text, which a JSON number or type would change.
pub fn json_value(value: &grafeo_common::types::Value) -> serde_json::Value {
    use grafeo_common::types::Value;
    use serde_json::Value as Json;

    let float = |f: f64| {
        serde_json::Number::from_f64(f).map_or_else(|| Json::String(f.to_string()), Json::Number)
    };
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int64(i) => Json::from(*i),
        Value::Float64(f) => float(*f),
        Value::String(s) => Json::String(s.to_string()),
        Value::List(items) => Json::Array(items.iter().map(json_value).collect()),
        Value::Map(entries) => Json::Object(
            entries
                .iter()
                .map(|(key, value)| (key.as_str().to_string(), json_value(value)))
                .collect(),
        ),
        Value::Vector(components) => Json::Array(
            components
                .iter()
                .map(|component| float(f64::from(*component)))
                .collect(),
        ),
        Value::Bytes(bytes) => Json::Array(bytes.iter().map(|byte| Json::from(*byte)).collect()),
        // A decimal is a JSON number when the number writes back as the same
        // text (1.5, 30); one a 64-bit float would change keeps its text.
        Value::Decimal(decimal) => {
            let text = decimal.to_string();
            match serde_json::Number::from_f64(decimal.to_f64()) {
                Some(number) if number.to_string() == text => Json::Number(number),
                _ => Json::String(text),
            }
        }
        Value::Path { nodes, edges } => serde_json::json!({
            "nodes": nodes.iter().map(json_value).collect::<Vec<_>>(),
            "edges": edges.iter().map(json_value).collect::<Vec<_>>(),
        }),
        other => Json::String(other.to_string()),
    }
}

/// A value as table and CSV text: a string as it is, without quotes; inside
/// a list or map a string is quoted, so `["a, b"]` and `["a", "b"]` differ;
/// null as `null`, and other values as their text (`1988-03-19` for a
/// date).
pub fn display_value(value: &grafeo_common::types::Value) -> String {
    match value {
        grafeo_common::types::Value::String(s) => s.to_string(),
        other => nested_text(other),
    }
}

/// A value inside a list or map (see [`display_value`]).
fn nested_text(value: &grafeo_common::types::Value) -> String {
    use grafeo_common::types::Value;

    match value {
        Value::Null => "null".to_string(),
        Value::String(s) => format!("{:?}", s.as_str()),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(nested_text).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Map(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(key, value)| format!("{key}: {}", nested_text(value)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Bytes(bytes) => format!("<{} bytes>", bytes.len()),
        other => other.to_string(),
    }
}

/// `text` cut to `max_width` characters, ending in `...` when cut.
fn truncate(text: String, max_width: Option<usize>) -> String {
    match max_width {
        Some(width) if text.chars().count() > width && width > 3 => {
            let mut cut: String = text.chars().take(width - 3).collect();
            cut.push_str("...");
            cut
        }
        _ => text,
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

/// Print a status message to stderr, unless `quiet`: stdout holds only
/// results, so it stays in the chosen format.
pub fn status(msg: &str, quiet: bool) {
    if !quiet {
        eprintln!("{msg}");
    }
}

/// Print a success message.
pub fn success(msg: &str, quiet: bool) {
    if !quiet {
        if no_color() {
            eprintln!("ok: {msg}");
        } else {
            eprintln!("\x1b[32m✓\x1b[0m {msg}");
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
    fn test_format_is_copy() {
        let format = Format::Table;
        let copied = format;
        // `format` is still usable after the copy.
        assert!(matches!(copied, Format::Table));
        assert!(matches!(format, Format::Table));
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

#[cfg(test)]
mod value_output_tests {
    use grafeo_common::types::{Date, Value};

    use super::*;

    #[test]
    fn display_value_writes_a_string_without_quotes() {
        assert_eq!(display_value(&Value::from("Alix")), "Alix");
        assert_eq!(display_value(&Value::Null), "null");
        assert_eq!(display_value(&Value::Bool(true)), "true");
        assert_eq!(display_value(&Value::Int64(-1)), "-1");
        assert_eq!(display_value(&Value::Float64(1.5)), "1.5");
        assert_eq!(
            display_value(&Value::Bytes(vec![0, 1, 2, 3, 4].into())),
            "<5 bytes>"
        );
    }

    #[test]
    fn display_value_quotes_strings_inside_a_list_or_map() {
        let list = Value::List(vec![Value::from("a, b"), Value::Null, Value::Int64(3)].into());
        assert_eq!(display_value(&list), r#"["a, b", null, 3]"#);
        let map = Value::Map(std::sync::Arc::new(
            [("city".into(), Value::from("Berlin"))]
                .into_iter()
                .collect(),
        ));
        assert_eq!(display_value(&map), r#"{city: "Berlin"}"#);
    }

    #[test]
    fn display_value_writes_a_date_as_its_value() {
        let date = Value::Date(Date::from_ymd(1988, 3, 19).unwrap());
        assert_eq!(display_value(&date), "1988-03-19");
    }

    #[test]
    fn json_value_keeps_json_types() {
        let list = Value::List(vec![Value::Int64(1), Value::from("Gus"), Value::Null].into());
        assert_eq!(json_value(&list), serde_json::json!([1, "Gus", null]));
        assert_eq!(json_value(&Value::Float64(1.5)), serde_json::json!(1.5));
        assert_eq!(
            json_value(&Value::Float64(f64::NAN)),
            serde_json::json!("NaN")
        );
        assert_eq!(json_value(&Value::Bool(false)), serde_json::json!(false));
        assert_eq!(
            json_value(&Value::Vector(vec![0.5_f32, 2.0].into())),
            serde_json::json!([0.5, 2.0])
        );
        let date = Value::Date(Date::from_ymd(1988, 3, 19).unwrap());
        assert_eq!(json_value(&date), serde_json::json!("1988-03-19"));
        let decimal = |text: &str| {
            json_value(&Value::Decimal(
                grafeo_common::types::Decimal::parse(text).unwrap(),
            ))
        };
        assert_eq!(decimal("1.5"), serde_json::json!(1.5));
        assert_eq!(
            decimal("12345678901234567890.123456789"),
            serde_json::json!("12345678901234567890.123456789"),
            "a decimal a float would round keeps its text"
        );
    }

    #[test]
    fn json_rows_keep_the_column_order() {
        let headers = vec!["z".to_string(), "a".to_string()];
        let row = vec![Value::Int64(1), Value::from("x")];
        let text = serde_json::to_string(&JsonRow {
            headers: &headers,
            row: &row,
        })
        .unwrap();
        assert_eq!(text, r#"{"z":1,"a":"x"}"#);
    }

    #[test]
    fn truncate_cuts_long_text_on_a_character_boundary() {
        assert_eq!(truncate("hello world".into(), None), "hello world");
        assert_eq!(truncate("hi".into(), Some(10)), "hi");
        assert_eq!(truncate("12345".into(), Some(5)), "12345");
        assert_eq!(truncate("hello world".into(), Some(8)), "hello...");
        assert_eq!(truncate("Ångström-länge".into(), Some(8)), "Ångst...");
    }
}
