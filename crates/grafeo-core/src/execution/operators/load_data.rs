//! LOAD DATA operator for reading CSV, JSONL, and Parquet files.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use grafeo_common::types::{ArcStr, LogicalType, PropertyKey, Value};

/// File format for the load data operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoadDataFormat {
    /// CSV (comma-separated values).
    Csv,
    /// JSON Lines (one JSON object per line).
    Jsonl,
    /// Apache Parquet columnar format.
    Parquet,
}

/// Operator that reads a data file and produces one row per record.
///
/// CSV with headers: each row is a `Value::Map` with column names as keys.
/// CSV without headers: each row is a `Value::List` of string values.
/// JSONL: each row is a `Value::Map` from JSON object fields.
/// Parquet: each row is a `Value::Map` from column names.
pub struct LoadDataOperator {
    /// File format.
    format: LoadDataFormat,
    /// Buffered reader for CSV/JSONL files.
    reader: Option<BufReader<File>>,
    /// Column headers (if CSV WITH HEADERS).
    headers: Option<Vec<String>>,
    /// Whether the CSV has headers.
    with_headers: bool,
    /// File path (for reset).
    path: String,
    /// Field separator (CSV only).
    delimiter: u8,
    /// Whether the file has been opened.
    opened: bool,
    /// Buffered Parquet rows (read all upfront, then iterate).
    #[cfg(feature = "parquet-import")]
    parquet_rows: Option<std::vec::IntoIter<Value>>,
}

impl LoadDataOperator {
    /// Creates a new LOAD DATA operator.
    pub fn new(
        path: String,
        format: LoadDataFormat,
        with_headers: bool,
        field_terminator: Option<char>,
        _variable: String,
    ) -> Self {
        let delimiter = field_terminator.map_or(b',', |c| {
            let mut buf = [0u8; 4];
            c.encode_utf8(&mut buf);
            buf[0]
        });

        Self {
            format,
            reader: None,
            headers: None,
            with_headers,
            path,
            delimiter,
            opened: false,
            #[cfg(feature = "parquet-import")]
            parquet_rows: None,
        }
    }

    /// Opens the file and reads headers if needed (CSV/JSONL).
    fn open_text(&mut self) -> Result<(), OperatorError> {
        let file_path = strip_file_prefix(&self.path);

        let file = File::open(file_path).map_err(|e| {
            OperatorError::Execution(format!(
                "Failed to open {} file '{}': {}",
                format_name(self.format),
                self.path,
                e
            ))
        })?;
        let mut reader = BufReader::new(file);

        if self.format == LoadDataFormat::Csv && self.with_headers {
            let mut header_line = String::new();
            reader.read_line(&mut header_line).map_err(|e| {
                OperatorError::Execution(format!("Failed to read CSV headers: {e}"))
            })?;
            // Strip BOM if present
            let header_line = header_line.strip_prefix('\u{feff}').unwrap_or(&header_line);
            let header_line = header_line.trim_end_matches(['\r', '\n']);
            self.headers = Some(parse_csv_row(header_line, self.delimiter));
        }

        self.reader = Some(reader);
        self.opened = true;
        Ok(())
    }

    /// Reads the next CSV record.
    fn next_csv(&mut self) -> OperatorResult {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| OperatorError::Internal("CSV reader not initialized".to_string()))?;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .map_err(|e| OperatorError::Execution(format!("Failed to read CSV line: {e}")))?;

            if bytes_read == 0 {
                return Ok(None); // EOF
            }

            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                continue; // skip blank lines
            }

            let fields = parse_csv_row(trimmed, self.delimiter);

            let row_value = if let Some(headers) = &self.headers {
                // WITH HEADERS: produce a Map
                let mut map = BTreeMap::new();
                for (i, header) in headers.iter().enumerate() {
                    let value = fields.get(i).map_or(Value::Null, |s| {
                        if s.is_empty() {
                            Value::Null
                        } else {
                            Value::String(ArcStr::from(s.as_str()))
                        }
                    });
                    map.insert(PropertyKey::from(header.as_str()), value);
                }
                Value::Map(Arc::new(map))
            } else {
                // Without headers: produce a List
                let values: Vec<Value> = fields
                    .into_iter()
                    .map(|s| {
                        if s.is_empty() {
                            Value::Null
                        } else {
                            Value::String(ArcStr::from(s.as_str()))
                        }
                    })
                    .collect();
                Value::List(Arc::from(values))
            };

            return Ok(Some(build_single_row_chunk(row_value)));
        }
    }

    /// Reads the next JSONL record.
    #[cfg(feature = "jsonl-import")]
    fn next_jsonl(&mut self) -> OperatorResult {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| OperatorError::Internal("JSONL reader not initialized".to_string()))?;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader
                .read_line(&mut line)
                .map_err(|e| OperatorError::Execution(format!("Failed to read JSONL line: {e}")))?;

            if bytes_read == 0 {
                return Ok(None); // EOF
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // skip blank lines
            }

            let json_value: serde_json::Value = serde_json::from_str(trimmed)
                .map_err(|e| OperatorError::Execution(format!("Failed to parse JSON line: {e}")))?;

            let row_value = json_to_value(&json_value);
            return Ok(Some(build_single_row_chunk(row_value)));
        }
    }

    /// Reads the next JSONL record (stub when feature disabled).
    #[cfg(not(feature = "jsonl-import"))]
    fn next_jsonl(&mut self) -> OperatorResult {
        Err(OperatorError::from(
            grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Unsupported,
                    "this build cannot load JSONL files (the `jsonl-import` feature)",
                ),
            ),
        ))
    }

    /// Opens and reads all rows from a Parquet file into a buffer.
    #[cfg(feature = "parquet-import")]
    fn open_parquet(&mut self) -> Result<(), OperatorError> {
        use parquet::file::reader::FileReader;

        let file_path = strip_file_prefix(&self.path);
        let file = File::open(file_path).map_err(|e| {
            OperatorError::Execution(format!(
                "Failed to open Parquet file '{}': {}",
                self.path, e
            ))
        })?;

        let reader = parquet::file::reader::SerializedFileReader::new(file).map_err(|e| {
            OperatorError::Execution(format!(
                "Failed to read Parquet file '{}': {}",
                self.path, e
            ))
        })?;

        let row_iter = reader.get_row_iter(None).map_err(|e| {
            OperatorError::Execution(format!("Failed to create Parquet row iterator: {e}"))
        })?;

        let mut rows = Vec::new();
        for row_result in row_iter {
            let row = row_result.map_err(|e| {
                OperatorError::Execution(format!("Failed to read Parquet row: {e}"))
            })?;
            rows.push(parquet_row_to_value(&row));
        }

        self.parquet_rows = Some(rows.into_iter());
        self.opened = true;
        Ok(())
    }

    /// Reads the next buffered Parquet record.
    #[cfg(feature = "parquet-import")]
    fn next_parquet(&mut self) -> OperatorResult {
        let rows = self
            .parquet_rows
            .as_mut()
            .ok_or_else(|| OperatorError::Internal("Parquet reader not initialized".to_string()))?;

        match rows.next() {
            Some(row_value) => Ok(Some(build_single_row_chunk(row_value))),
            None => Ok(None), // EOF
        }
    }
}

impl Operator for LoadDataOperator {
    fn next(&mut self) -> OperatorResult {
        match self.format {
            LoadDataFormat::Csv => {
                if !self.opened {
                    self.open_text()?;
                }
                self.next_csv()
            }
            LoadDataFormat::Jsonl => {
                if !self.opened {
                    self.open_text()?;
                }
                self.next_jsonl()
            }
            LoadDataFormat::Parquet => {
                #[cfg(feature = "parquet-import")]
                {
                    if !self.opened {
                        self.open_parquet()?;
                    }
                    self.next_parquet()
                }
                #[cfg(not(feature = "parquet-import"))]
                Err(OperatorError::from(
                    grafeo_common::utils::error::Error::Query(
                        grafeo_common::utils::error::QueryError::new(
                            grafeo_common::utils::error::QueryErrorKind::Unsupported,
                            "this build cannot load Parquet files (the `parquet-import` feature)",
                        ),
                    ),
                ))
            }
        }
    }

    fn reset(&mut self) {
        self.reader = None;
        self.headers = None;
        self.opened = false;
        #[cfg(feature = "parquet-import")]
        {
            self.parquet_rows = None;
        }
    }

    fn name(&self) -> &'static str {
        match self.format {
            LoadDataFormat::Csv => "LoadCsv",
            LoadDataFormat::Jsonl => "LoadJsonl",
            LoadDataFormat::Parquet => "LoadParquet",
        }
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Strips `file:///` or `file://` prefix from a path (Neo4j convention).
fn strip_file_prefix(path: &str) -> &str {
    path.strip_prefix("file:///")
        .or_else(|| path.strip_prefix("file://"))
        .unwrap_or(path)
}

/// Returns a human-readable format name.
fn format_name(format: LoadDataFormat) -> &'static str {
    match format {
        LoadDataFormat::Csv => "CSV",
        LoadDataFormat::Jsonl => "JSONL",
        LoadDataFormat::Parquet => "Parquet",
    }
}

/// Builds a single-row `DataChunk` with one column containing the given value.
fn build_single_row_chunk(value: Value) -> crate::execution::DataChunk {
    let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
    if let Some(col) = builder.column_mut(0) {
        col.push_value(value);
    }
    builder.advance_row();
    builder.finish()
}

/// Parses a single CSV row into fields, respecting quoted fields.
///
/// Handles:
/// - Unquoted fields separated by the delimiter
/// - Double-quoted fields (can contain delimiters, newlines, and escaped quotes)
/// - Escaped quotes within quoted fields (`""` becomes `"`)
fn parse_csv_row(line: &str, delimiter: u8) -> Vec<String> {
    let delim = delimiter as char;
    let mut fields = Vec::new();
    let mut chars = line.chars().peekable();
    let mut field = String::new();

    loop {
        if chars.peek() == Some(&'"') {
            // Quoted field
            chars.next(); // consume opening quote
            loop {
                match chars.next() {
                    Some('"') => {
                        if chars.peek() == Some(&'"') {
                            // Escaped quote
                            chars.next();
                            field.push('"');
                        } else {
                            // End of quoted field
                            break;
                        }
                    }
                    Some(c) => field.push(c),
                    None => break, // Unterminated quote, take what we have
                }
            }
            // Skip to delimiter or end
            match chars.peek() {
                Some(c) if *c == delim => {
                    chars.next();
                }
                _ => {}
            }
            fields.push(std::mem::take(&mut field));
        } else {
            // Unquoted field
            loop {
                match chars.peek() {
                    Some(c) if *c == delim => {
                        chars.next();
                        break;
                    }
                    Some(_) => {
                        field.push(chars.next().unwrap());
                    }
                    None => break,
                }
            }
            fields.push(std::mem::take(&mut field));
        }

        if chars.peek().is_none() {
            break;
        }
    }

    fields
}

// ============================================================================
// JSONL helpers
// ============================================================================

/// Converts a `serde_json::Value` to a `grafeo_common::types::Value`.
#[cfg(feature = "jsonl-import")]
fn json_to_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::String(ArcStr::from(n.to_string().as_str()))
            }
        }
        serde_json::Value::String(s) => Value::String(ArcStr::from(s.as_str())),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(Arc::from(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map = BTreeMap::new();
            for (key, val) in obj {
                map.insert(PropertyKey::from(key.as_str()), json_to_value(val));
            }
            Value::Map(Arc::new(map))
        }
    }
}

// ============================================================================
// Parquet helpers
// ============================================================================

/// Converts a Parquet `Row` to a `Value::Map`.
#[cfg(feature = "parquet-import")]
fn parquet_row_to_value(row: &parquet::record::Row) -> Value {
    use parquet::record::Field;

    let mut map = BTreeMap::new();
    for (name, field) in row.get_column_iter() {
        let value = match field {
            Field::Null => Value::Null,
            Field::Bool(b) => Value::Bool(*b),
            Field::Byte(b) => Value::Int64(i64::from(*b)),
            Field::Short(s) => Value::Int64(i64::from(*s)),
            Field::Int(i) => Value::Int64(i64::from(*i)),
            Field::Long(l) => Value::Int64(*l),
            Field::UByte(b) => Value::Int64(i64::from(*b)),
            Field::UShort(s) => Value::Int64(i64::from(*s)),
            Field::UInt(i) => Value::Int64(i64::from(*i)),
            Field::ULong(l) => {
                // u64 may overflow i64, store as string if too large
                if let Ok(i) = i64::try_from(*l) {
                    Value::Int64(i)
                } else {
                    Value::String(ArcStr::from(l.to_string().as_str()))
                }
            }
            Field::Float(f) => Value::Float64(f64::from(*f)),
            Field::Double(d) => Value::Float64(*d),
            Field::Str(s) => Value::String(ArcStr::from(s.as_str())),
            Field::Bytes(b) => Value::Bytes(Arc::from(b.data().to_vec())),
            Field::Decimal(d) => {
                // Convert decimal to f64 for simplicity
                Value::Float64(decimal_to_f64(d))
            }
            Field::Float16(f) => Value::Float64(f64::from(*f)),
            Field::Group(row) => parquet_row_to_value(row),
            Field::ListInternal(list) => {
                let items: Vec<Value> =
                    list.elements().iter().map(parquet_field_to_value).collect();
                Value::List(Arc::from(items))
            }
            Field::MapInternal(map_internal) => {
                let mut inner_map = BTreeMap::new();
                for (key_field, val_field) in map_internal.entries() {
                    let key_str = match key_field {
                        Field::Str(s) => s.clone(),
                        other => format!("{other}"),
                    };
                    inner_map.insert(
                        PropertyKey::from(key_str.as_str()),
                        parquet_field_to_value(val_field),
                    );
                }
                Value::Map(Arc::new(inner_map))
            }
            Field::TimestampMillis(ms) => Value::Int64(*ms),
            Field::TimestampMicros(us) => Value::Int64(*us),
            Field::TimeMillis(ms) => Value::Int64(i64::from(*ms)),
            Field::TimeMicros(us) => Value::Int64(*us),
            Field::Date(days) => Value::Int64(i64::from(*days)),
        };
        map.insert(PropertyKey::from(name.as_str()), value);
    }
    Value::Map(Arc::new(map))
}

/// Converts a single Parquet field to a Value.
#[cfg(feature = "parquet-import")]
fn parquet_field_to_value(field: &parquet::record::Field) -> Value {
    use parquet::record::Field;

    match field {
        Field::Null => Value::Null,
        Field::Bool(b) => Value::Bool(*b),
        Field::Byte(b) => Value::Int64(i64::from(*b)),
        Field::Short(s) => Value::Int64(i64::from(*s)),
        Field::Int(i) => Value::Int64(i64::from(*i)),
        Field::Long(l) => Value::Int64(*l),
        Field::UByte(b) => Value::Int64(i64::from(*b)),
        Field::UShort(s) => Value::Int64(i64::from(*s)),
        Field::UInt(i) => Value::Int64(i64::from(*i)),
        Field::ULong(l) => {
            if let Ok(i) = i64::try_from(*l) {
                Value::Int64(i)
            } else {
                Value::String(ArcStr::from(l.to_string().as_str()))
            }
        }
        Field::Float(f) => Value::Float64(f64::from(*f)),
        Field::Double(d) => Value::Float64(*d),
        Field::Str(s) => Value::String(ArcStr::from(s.as_str())),
        Field::Bytes(b) => Value::Bytes(Arc::from(b.data().to_vec())),
        Field::Decimal(d) => Value::Float64(decimal_to_f64(d)),
        Field::Float16(f) => Value::Float64(f64::from(*f)),
        Field::Group(row) => parquet_row_to_value(row),
        Field::ListInternal(list) => {
            let items: Vec<Value> = list.elements().iter().map(parquet_field_to_value).collect();
            Value::List(Arc::from(items))
        }
        Field::MapInternal(map_internal) => {
            let mut inner_map = BTreeMap::new();
            for (key_field, val_field) in map_internal.entries() {
                let key_str = match key_field {
                    Field::Str(s) => s.clone(),
                    other => format!("{other}"),
                };
                inner_map.insert(
                    PropertyKey::from(key_str.as_str()),
                    parquet_field_to_value(val_field),
                );
            }
            Value::Map(Arc::new(inner_map))
        }
        Field::TimestampMillis(ms) => Value::Int64(*ms),
        Field::TimestampMicros(us) => Value::Int64(*us),
        Field::TimeMillis(ms) => Value::Int64(i64::from(*ms)),
        Field::TimeMicros(us) => Value::Int64(*us),
        Field::Date(days) => Value::Int64(i64::from(*days)),
    }
}

/// Converts a Parquet Decimal to f64.
#[cfg(feature = "parquet-import")]
fn decimal_to_f64(d: &parquet::data_type::Decimal) -> f64 {
    let bytes = d.data();
    let scale = d.scale();
    // Interpret bytes as big-endian signed integer
    let mut value: i128 = if !bytes.is_empty() && bytes[0] & 0x80 != 0 {
        -1 // sign-extend for negative
    } else {
        0
    };
    for &b in bytes {
        value = (value << 8) | i128::from(b);
    }
    value as f64 / 10f64.powi(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_simple() {
        let fields = parse_csv_row("a,b,c", b',');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_csv_quoted() {
        let fields = parse_csv_row(r#""hello","world""#, b',');
        assert_eq!(fields, vec!["hello", "world"]);
    }

    #[test]
    fn test_parse_csv_escaped_quotes() {
        let fields = parse_csv_row(r#""say ""hi""","ok""#, b',');
        assert_eq!(fields, vec![r#"say "hi""#, "ok"]);
    }

    #[test]
    fn test_parse_csv_delimiter_in_quoted() {
        let fields = parse_csv_row(r#""a,b",c"#, b',');
        assert_eq!(fields, vec!["a,b", "c"]);
    }

    #[test]
    fn test_parse_csv_empty_fields() {
        let fields = parse_csv_row("a,,c", b',');
        assert_eq!(fields, vec!["a", "", "c"]);
    }

    #[test]
    fn test_parse_csv_tab_delimiter() {
        let fields = parse_csv_row("a\tb\tc", b'\t');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_csv_single_field() {
        let fields = parse_csv_row("hello", b',');
        assert_eq!(fields, vec!["hello"]);
    }

    #[test]
    fn test_strip_file_prefix() {
        assert_eq!(strip_file_prefix("file:///data.csv"), "data.csv");
        assert_eq!(strip_file_prefix("file://data.csv"), "data.csv");
        assert_eq!(strip_file_prefix("data.csv"), "data.csv");
        assert_eq!(strip_file_prefix("/tmp/data.csv"), "/tmp/data.csv");
    }

    #[test]
    fn test_format_name() {
        assert_eq!(format_name(LoadDataFormat::Csv), "CSV");
        assert_eq!(format_name(LoadDataFormat::Jsonl), "JSONL");
        assert_eq!(format_name(LoadDataFormat::Parquet), "Parquet");
    }

    #[test]
    fn test_load_data_into_any() {
        let op = LoadDataOperator::new(
            "test.csv".to_string(),
            LoadDataFormat::Csv,
            true,
            None,
            "row".to_string(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<LoadDataOperator>().is_ok());
    }

    #[cfg(feature = "jsonl-import")]
    mod jsonl_tests {
        use super::*;

        #[test]
        fn test_json_to_value_null() {
            assert!(matches!(
                json_to_value(&serde_json::Value::Null),
                Value::Null
            ));
        }

        #[test]
        fn test_json_to_value_bool() {
            assert!(matches!(
                json_to_value(&serde_json::Value::Bool(true)),
                Value::Bool(true)
            ));
        }

        #[test]
        fn test_json_to_value_integer() {
            let json: serde_json::Value = serde_json::from_str("42").unwrap();
            assert!(matches!(json_to_value(&json), Value::Int64(42)));
        }

        #[test]
        fn test_json_to_value_float() {
            let json: serde_json::Value = serde_json::from_str("1.5").unwrap();
            match json_to_value(&json) {
                Value::Float64(f) => assert!((f - 1.5_f64).abs() < f64::EPSILON),
                other => panic!("expected Float64, got {other:?}"),
            }
        }

        #[test]
        fn test_json_to_value_string() {
            let json: serde_json::Value = serde_json::from_str(r#""hello""#).unwrap();
            match json_to_value(&json) {
                Value::String(s) => assert_eq!(s.as_str(), "hello"),
                other => panic!("expected String, got {other:?}"),
            }
        }

        #[test]
        fn test_json_to_value_array() {
            let json: serde_json::Value = serde_json::from_str("[1, 2, 3]").unwrap();
            match json_to_value(&json) {
                Value::List(items) => {
                    assert_eq!(items.len(), 3);
                    assert!(matches!(items[0], Value::Int64(1)));
                }
                other => panic!("expected List, got {other:?}"),
            }
        }

        #[test]
        fn test_json_to_value_object() {
            let json: serde_json::Value =
                serde_json::from_str(r#"{"name": "Alix", "age": 30}"#).unwrap();
            match json_to_value(&json) {
                Value::Map(map) => {
                    assert_eq!(map.len(), 2);
                    assert!(matches!(
                        map.get(&PropertyKey::from("age")),
                        Some(Value::Int64(30))
                    ));
                }
                other => panic!("expected Map, got {other:?}"),
            }
        }
    }
}
