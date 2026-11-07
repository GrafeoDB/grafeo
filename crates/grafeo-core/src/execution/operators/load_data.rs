//! LOAD DATA operator for reading CSV, JSONL, and Parquet files.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
#[cfg(feature = "jsonl-import")]
use grafeo_common::types::NumberTextError;
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
    /// The number of the last physical line read (CSV/JSONL), for errors.
    line: u64,
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
            line: 0,
            delimiter,
            opened: false,
            #[cfg(feature = "parquet-import")]
            parquet_rows: None,
        }
    }

    /// Opens the file and reads headers if needed (CSV/JSONL).
    fn open_text(&mut self) -> Result<(), OperatorError> {
        let file_path = local_file(&self.path)?;

        let file = File::open(&file_path).map_err(|e| {
            OperatorError::Execution(format!(
                "Failed to open {} file '{}': {}",
                format_name(self.format),
                self.path,
                e
            ))
        })?;
        let mut reader = BufReader::new(file);

        if self.format == LoadDataFormat::Csv && self.with_headers {
            let record = read_csv_record(&mut reader, self.delimiter, true, &mut self.line)
                .map_err(|e| OperatorError::Execution(format!("Failed to read CSV headers: {e}")))?
                .unwrap_or_default();
            self.headers = Some(parse_csv_row(&record, self.delimiter));
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

        loop {
            let Some(record) = read_csv_record(reader, self.delimiter, false, &mut self.line)
                .map_err(|e| OperatorError::Execution(format!("Failed to read CSV line: {e}")))?
            else {
                return Ok(None); // EOF
            };

            if record.is_empty() {
                continue; // skip blank lines
            }

            let fields = parse_csv_row(&record, self.delimiter);

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
            self.line += 1;
            let number = self.line;

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // skip blank lines
            }

            let json_value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
                OperatorError::Execution(format!("Failed to parse JSON line {number}: {e}"))
            })?;

            let row_value = json_to_value(&json_value).map_err(|error| match error {
                NumberTextError::Invalid(_) => {
                    OperatorError::Execution(format!("line {number}: {error}"))
                }
                _ => OperatorError::numeric_out_of_range(format!("line {number}: {error}")),
            })?;
            return Ok(Some(build_single_row_chunk(row_value)));
        }
    }

    /// Reads the next JSONL record (stub when feature disabled).
    #[cfg(not(feature = "jsonl-import"))]
    fn next_jsonl(&mut self) -> OperatorResult {
        Err(OperatorError::from(
            grafeo_common::utils::error::Error::from(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                "this build cannot load JSONL files (the `jsonl-import` feature)",
            )),
        ))
    }

    /// Opens and reads all rows from a Parquet file into a buffer.
    #[cfg(feature = "parquet-import")]
    fn open_parquet(&mut self) -> Result<(), OperatorError> {
        use parquet::basic::CompressionCodec;
        use parquet::file::reader::FileReader;

        let file_path = local_file(&self.path)?;
        let file = File::open(&file_path).map_err(|e| {
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

        // The parquet crate is built with Snappy only (the default of pyarrow,
        // pandas and Spark). Its own error for another codec names a cargo
        // feature, so name the file's codec before a page is read.
        for row_group in reader.metadata().row_groups() {
            for column in row_group.columns() {
                let codec = column.compression_codec();
                if !matches!(
                    codec,
                    CompressionCodec::UNCOMPRESSED | CompressionCodec::SNAPPY
                ) {
                    return Err(OperatorError::from(
                        grafeo_common::utils::error::Error::from(
                            grafeo_common::utils::error::QueryError::new(
                                grafeo_common::utils::error::QueryErrorKind::Unsupported,
                                format!(
                                    "the Parquet file '{}' is compressed with {codec}; this \
                                     build reads uncompressed and Snappy Parquet files",
                                    self.path
                                ),
                            ),
                        ),
                    ));
                }
            }
        }

        let schema = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .root_schema_ptr();
        let row_iter = reader.get_row_iter(None).map_err(|e| {
            OperatorError::Execution(format!("Failed to create Parquet row iterator: {e}"))
        })?;

        let mut rows = Vec::new();
        for row_result in row_iter {
            let row = row_result.map_err(|e| {
                OperatorError::Execution(format!("Failed to read Parquet row: {e}"))
            })?;
            rows.push(parquet_values::parquet_row_to_value(&row, Some(&schema)));
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
                    grafeo_common::utils::error::Error::from(
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
        self.line = 0;
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

/// The local file `path` names: a path as given, or a `file:` URL's (RFC 8089, see
/// [`grafeo_common::utils::file_url`]).
fn local_file(path: &str) -> Result<std::path::PathBuf, OperatorError> {
    use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

    // An invalid parameter value (22023) naming the path, as the load procedures report it.
    grafeo_common::utils::file_url::local_path(path).map_err(|error| {
        OperatorError::Wrapped(Box::new(Error::from(QueryError::new(
            QueryErrorKind::InvalidParameterValue,
            format!("the path names no local file: {error}"),
        ))))
    })
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

/// Reads a complete CSV record, including line breaks inside quoted fields.
/// The BOM is removed only when reading headers, as before.
fn read_csv_record(
    reader: &mut impl BufRead,
    delimiter: u8,
    strip_bom: bool,
    line: &mut u64,
) -> std::io::Result<Option<String>> {
    let mut record = String::new();
    let mut quoted = false;
    let mut quoted_start = 0;
    let mut field_start = true;
    loop {
        let start = record.len();
        if reader.read_line(&mut record)? == 0 {
            if quoted {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unterminated quoted field starting at line {quoted_start}"),
                ));
            }
            return Ok((!record.is_empty()).then_some(record));
        }
        *line += 1;
        if strip_bom && start == 0 && record.starts_with('\u{feff}') {
            record.replace_range(..'\u{feff}'.len_utf8(), "");
        }
        let mut chars = record[start..].chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '"' if quoted => {
                    if chars.peek() == Some(&'"') {
                        chars.next(); // Escaped quote, not the end of the field.
                    } else {
                        quoted = false;
                    }
                }
                '"' if field_start => {
                    quoted = true;
                    quoted_start = *line;
                    field_start = false;
                }
                c if !quoted && c == char::from(delimiter) => field_start = true,
                _ => field_start = false,
            }
        }
        if !quoted {
            let end = record.trim_end_matches(['\r', '\n']).len();
            record.truncate(end);
            return Ok(Some(record));
        }
    }
}

/// Parses a complete CSV record into fields, respecting quoted fields.
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

/// Converts a `serde_json::Value` to a `grafeo_common::types::Value`. A
/// number follows [`Value::from_number_text`] on its source text (the
/// workspace reads JSON with `arbitrary_precision`): an integer is an INT64
/// when it fits 64 bits, else an exact DECIMAL up to 38 digits; any other
/// number is a FLOAT64.
#[cfg(feature = "jsonl-import")]
fn json_to_value(json: &serde_json::Value) -> Result<Value, NumberTextError> {
    Ok(match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => Value::from_number_text(n.as_str())?,
        serde_json::Value::String(s) => Value::String(ArcStr::from(s.as_str())),
        serde_json::Value::Array(arr) => {
            let items = arr
                .iter()
                .map(json_to_value)
                .collect::<Result<Vec<_>, _>>()?;
            Value::List(Arc::from(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map = BTreeMap::new();
            for (key, val) in obj {
                map.insert(PropertyKey::from(key.as_str()), json_to_value(val)?);
            }
            Value::Map(Arc::new(map))
        }
    })
}

// ============================================================================
// Parquet helpers
// ============================================================================

/// Parquet rows as Grafeo values.
#[cfg(feature = "parquet-import")]
mod parquet_values {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::{
        ArcStr, Date, Decimal, MAX_DECIMAL_DIGITS, PropertyKey, Time, Timestamp, Value,
    };
    use parquet::schema::types::Type as ParquetType;

    /// Converts a Parquet `Row` to a `Value::Map`, each column read by its type
    /// in `schema` (the file's message type, or the group the row is).
    pub(super) fn parquet_row_to_value(
        row: &parquet::record::Row,
        schema: Option<&ParquetType>,
    ) -> Value {
        // The reader yields a group's columns in the order of its fields.
        let fields = schema.filter(|ty| ty.is_group()).map(|ty| ty.get_fields());
        let mut map = BTreeMap::new();
        for (index, (name, field)) in row.get_column_iter().enumerate() {
            let ty = fields
                .and_then(|fields| fields.get(index))
                .map(AsRef::as_ref);
            map.insert(
                PropertyKey::from(name.as_str()),
                parquet_field_to_value(field, ty),
            );
        }
        Value::Map(Arc::new(map))
    }

    /// Converts a single Parquet field to a Value. `ty` is the field's type in
    /// the file's schema: the record reader marks dates, times, timestamps and
    /// decimals by their converted type, which a nanosecond time or timestamp
    /// does not have, so those come as plain integers and only the logical type
    /// in the schema tells them apart.
    fn parquet_field_to_value(field: &parquet::record::Field, ty: Option<&ParquetType>) -> Value {
        use parquet::basic::{LogicalType as ParquetLogicalType, TimeUnit};
        use parquet::record::Field;

        match field {
            Field::Null => Value::Null,
            Field::Bool(b) => Value::Bool(*b),
            Field::Byte(b) => Value::Int64(i64::from(*b)),
            Field::Short(s) => Value::Int64(i64::from(*s)),
            Field::Int(i) => Value::Int64(i64::from(*i)),
            Field::Long(l) => match ty.and_then(|ty| ty.get_basic_info().logical_type_ref()) {
                Some(ParquetLogicalType::Timestamp(t)) if t.unit == TimeUnit::NANOS => {
                    // Grafeo keeps microseconds; the rest is dropped toward the
                    // past, so an instant before the epoch stays before it.
                    Value::Timestamp(Timestamp::from_micros(l.div_euclid(1_000)))
                }
                Some(ParquetLogicalType::Time(t)) if t.unit == TimeUnit::NANOS => {
                    time_of_day(*l, 1)
                }
                _ => Value::Int64(*l),
            },
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
            Field::Decimal(d) => parquet_decimal(field, d),
            Field::Float16(f) => Value::Float64(f64::from(*f)),
            Field::Group(row) => parquet_row_to_value(row, ty),
            Field::ListInternal(list) => {
                let element = ty.and_then(list_element_type);
                let items: Vec<Value> = list
                    .elements()
                    .iter()
                    .map(|item| parquet_field_to_value(item, element))
                    .collect();
                Value::List(Arc::from(items))
            }
            Field::MapInternal(map_internal) => {
                // A map is a group of one repeated `key_value` group: key, value.
                let value_type = ty
                    .filter(|ty| ty.is_group())
                    .and_then(|ty| ty.get_fields().first())
                    .filter(|key_value| key_value.is_group())
                    .and_then(|key_value| key_value.get_fields().get(1))
                    .map(AsRef::as_ref);
                let mut inner_map = BTreeMap::new();
                for (key_field, val_field) in map_internal.entries() {
                    let key_str = match key_field {
                        Field::Str(s) => s.clone(),
                        other => format!("{other}"),
                    };
                    inner_map.insert(
                        PropertyKey::from(key_str.as_str()),
                        parquet_field_to_value(val_field, value_type),
                    );
                }
                Value::Map(Arc::new(inner_map))
            }
            Field::TimestampMillis(ms) => match ms.checked_mul(1_000) {
                Some(micros) => Value::Timestamp(Timestamp::from_micros(micros)),
                // Past the range Grafeo's microseconds hold: kept as written.
                None => Value::Int64(*ms),
            },
            Field::TimestampMicros(us) => Value::Timestamp(Timestamp::from_micros(*us)),
            Field::TimeMillis(ms) => time_of_day(i64::from(*ms), 1_000_000),
            Field::TimeMicros(us) => time_of_day(*us, 1_000),
            Field::Date(days) => Value::Date(Date::from_days(*days)),
        }
    }

    /// The type of a list's elements, by the rules the record reader follows:
    /// a `LIST` group holds one repeated field, which is the element itself in
    /// the legacy two-level form and wraps it in the three-level one. A repeated
    /// field outside a `LIST` group reads as a list of itself.
    fn list_element_type(ty: &ParquetType) -> Option<&ParquetType> {
        use parquet::basic::{ConvertedType, LogicalType as ParquetLogicalType, Repetition};

        let is_repeated = |ty: &ParquetType| {
            let info = ty.get_basic_info();
            info.has_repetition() && info.repetition() == Repetition::REPEATED
        };
        let is_list = |ty: &ParquetType| {
            ty.is_group()
                && match ty.get_basic_info().logical_type_ref() {
                    Some(logical) => logical == &ParquetLogicalType::List,
                    None => ty.get_basic_info().converted_type() == ConvertedType::LIST,
                }
        };
        if !is_list(ty) {
            return is_repeated(ty).then_some(ty);
        }
        let repeated = ty.get_fields().first()?.as_ref();
        let single_repeated_child = repeated.is_group()
            && repeated.get_fields().len() == 1
            && is_repeated(&repeated.get_fields()[0]);
        let is_element = !(is_list(repeated) || single_repeated_child)
            && (repeated.is_primitive()
                || repeated.get_fields().len() > 1
                || repeated.name() == "array"
                || repeated.name().ends_with("_tuple"));
        if is_element {
            Some(repeated)
        } else {
            repeated.get_fields().first().map(AsRef::as_ref)
        }
    }

    /// A time of day of `count` units of `nanos_per_unit` nanoseconds since
    /// midnight, or the count as written when it is not within a day.
    fn time_of_day(count: i64, nanos_per_unit: u64) -> Value {
        u64::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(nanos_per_unit))
            .and_then(Time::from_nanos)
            .map_or(Value::Int64(count), Value::Time)
    }

    /// A Parquet decimal as an exact DECIMAL at the column's scale. A column
    /// declared wider than Grafeo's 38 digits is its exact text, every value of
    /// it, as the Arrow export writes such a column.
    fn parquet_decimal(field: &parquet::record::Field, d: &parquet::data_type::Decimal) -> Value {
        let exact = u8::try_from(d.scale())
            .ok()
            .filter(|_| d.precision() <= i32::from(MAX_DECIMAL_DIGITS))
            .zip(decimal_mantissa(d.data()))
            .and_then(|(scale, mantissa)| Decimal::new(mantissa, scale).ok());
        match exact {
            Some(decimal) => Value::Decimal(decimal),
            // The record API writes a decimal's digits exactly.
            None => Value::String(ArcStr::from(field.to_string().as_str())),
        }
    }

    /// The big-endian two's complement integer of `bytes`, when it fits 128
    /// bits.
    fn decimal_mantissa(bytes: &[u8]) -> Option<i128> {
        let negative = bytes.first().is_some_and(|first| first & 0x80 != 0);
        bytes
            .iter()
            .try_fold(if negative { -1 } else { 0 }, |value: i128, &byte| {
                value.checked_mul(256)?.checked_add(i128::from(byte))
            })
    }
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
    fn a_file_url_names_its_absolute_path_and_a_path_is_kept() {
        let path = |name: &str| local_file(name).unwrap().to_string_lossy().into_owned();
        assert_eq!(path("file:///tmp/data.csv"), "/tmp/data.csv");
        assert_eq!(path("data.csv"), "data.csv");
        assert_eq!(path("/tmp/data.csv"), "/tmp/data.csv");
        assert!(local_file("file://server/data.csv").is_err());
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

    mod csv_records {
        use super::*;

        fn operator(
            content: &str,
            headers: bool,
            delimiter: Option<char>,
        ) -> (tempfile::NamedTempFile, LoadDataOperator) {
            let file = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(file.path(), content).unwrap();
            let op = LoadDataOperator::new(
                file.path().to_str().unwrap().to_string(),
                LoadDataFormat::Csv,
                headers,
                delimiter,
                "row".to_string(),
            );
            (file, op)
        }

        fn records(op: &mut LoadDataOperator) -> Vec<Value> {
            let mut rows = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                assert_eq!(chunk.row_count(), 1);
                rows.push(chunk.column(0).unwrap().get_value(0).unwrap());
            }
            rows
        }

        fn list(values: &[&str]) -> Value {
            Value::List(Arc::from(
                values
                    .iter()
                    .map(|value| Value::String((*value).into()))
                    .collect::<Vec<_>>(),
            ))
        }

        #[test]
        fn quoted_line_breaks_and_escaped_quotes_form_one_record() {
            for ending in ["\n", "\r\n"] {
                let note = format!("first{ending}{ending}said \"hi\"");
                let content = format!(
                    "{ending}Alix;\"first{ending}{ending}said \"\"hi\"\"\"{ending}Gus;quiet"
                );
                let (_file, mut op) = operator(&content, false, Some(';'));
                assert_eq!(
                    records(&mut op),
                    [list(&["Alix", &note]), list(&["Gus", "quiet"])],
                    "line ending {ending:?}"
                );
            }
        }

        #[test]
        fn multiline_headers_keep_bom_handling_and_column_names() {
            let (_file, mut op) = operator(
                "\u{feff}\"home\ncity\",\"note\r\ntext\"\r\nAmsterdam,\"first\nlast\"",
                true,
                None,
            );
            assert_eq!(
                records(&mut op),
                [Value::Map(Arc::new(BTreeMap::from([
                    (
                        PropertyKey::from("home\ncity"),
                        Value::String("Amsterdam".into())
                    ),
                    (
                        PropertyKey::from("note\r\ntext"),
                        Value::String("first\nlast".into())
                    ),
                ])))]
            );
        }

        #[test]
        fn reset_replays_records_larger_than_the_reader_buffer() {
            let text = format!("{}\n{}", "x".repeat(9_000), "y".repeat(9_000));
            let content = format!("\"{text}\",last\nnext,row\n");
            let (_file, mut op) = operator(&content, false, None);
            let expected = [list(&[&text, "last"]), list(&["next", "row"])];
            assert_eq!(records(&mut op), expected);
            assert!(op.next().unwrap().is_none());
            op.reset();
            assert_eq!(records(&mut op), expected);
        }

        #[test]
        fn unquoted_quotes_do_not_consume_the_next_record() {
            let (_file, mut op) = operator("a\"b,c\nnext,row", false, None);
            assert_eq!(
                records(&mut op),
                [list(&["a\"b", "c"]), list(&["next", "row"])]
            );
        }

        #[test]
        fn an_unclosed_quote_reports_its_opening_line() {
            for ending in ["\n", "\r\n"] {
                for (content, headers, delimiter, line) in [
                    ("\"unfinished", false, None, 1),
                    ("\"unfinished\nmore\n", false, None, 1),
                    ("first,row\n\n\"unfinished", false, None, 3),
                    ("name,note\nAlix,\"unfinished\nmore", true, None, 2),
                    ("\u{feff}\"unfinished", true, None, 1),
                    (
                        "\"multi\nline\",note\nAlix,ok\n\nGus,\"unfinished\nlast",
                        true,
                        None,
                        5,
                    ),
                    ("a;\"ok\nstill\";\"unfinished\nend", false, Some(';'), 2),
                ] {
                    let content = content.replace('\n', ending);
                    let (_file, mut op) = operator(&content, headers, delimiter);
                    for _ in 0..2 {
                        let error = loop {
                            match op.next() {
                                Ok(Some(_)) => {}
                                Ok(None) => panic!("unterminated field accepted: {content:?}"),
                                Err(error) => break error,
                            }
                        };
                        assert!(
                            error.to_string().contains(&format!(
                                "unterminated quoted field starting at line {line}"
                            )),
                            "{content:?}: {error}"
                        );
                        op.reset();
                    }
                }
            }
        }
    }

    #[cfg(feature = "jsonl-import")]
    mod jsonl_tests {
        use super::*;

        /// [`super::json_to_value`] for a value that has one.
        fn json_to_value(json: &serde_json::Value) -> Value {
            super::super::json_to_value(json).unwrap()
        }

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

    /// Parquet files, uncompressed and Snappy-compressed (the default of
    /// pyarrow, pandas and Spark), load the same rows.
    #[cfg(feature = "parquet-import")]
    mod parquet_records {
        use super::*;
        use grafeo_common::types::{Date, Time, Timestamp};
        use parquet::basic::Compression;
        use parquet::data_type::{
            ByteArray, ByteArrayType, DataType, FixedLenByteArray, FixedLenByteArrayType,
            Int32Type, Int64Type,
        };
        use parquet::file::metadata::ParquetMetaDataWriter;
        use parquet::file::properties::WriterProperties;
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::file::writer::{SerializedColumnWriter, SerializedFileWriter};
        use parquet::schema::parser::parse_message_type;

        /// Alix (31) and Gus (no age), in one row group, with `compression`.
        fn write_people(compression: Compression) -> tempfile::NamedTempFile {
            let schema = parse_message_type(
                "message people { REQUIRED BYTE_ARRAY name (UTF8); OPTIONAL INT64 age; }",
            )
            .unwrap();
            let properties = WriterProperties::builder()
                .set_compression(compression)
                .build();
            let file = tempfile::NamedTempFile::new().unwrap();
            let mut writer = SerializedFileWriter::new(
                file.reopen().unwrap(),
                Arc::new(schema),
                Arc::new(properties),
            )
            .unwrap();
            let mut row_group = writer.next_row_group().unwrap();
            let mut names = row_group.next_column().unwrap().unwrap();
            names
                .typed::<ByteArrayType>()
                .write_batch(
                    &[ByteArray::from("Alix"), ByteArray::from("Gus")],
                    None,
                    None,
                )
                .unwrap();
            names.close().unwrap();
            let mut ages = row_group.next_column().unwrap().unwrap();
            ages.typed::<Int64Type>()
                .write_batch(&[31], Some(&[1, 0]), None)
                .unwrap();
            ages.close().unwrap();
            row_group.close().unwrap();
            writer.close().unwrap();
            file
        }

        fn load(file: &tempfile::NamedTempFile) -> Vec<(Option<Value>, Option<Value>)> {
            let mut op = LoadDataOperator::new(
                file.path().to_str().unwrap().to_string(),
                LoadDataFormat::Parquet,
                true,
                None,
                "row".to_string(),
            );
            let mut rows = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                match chunk.column(0).unwrap().get_value(0).unwrap() {
                    Value::Map(map) => rows.push((
                        map.get(&PropertyKey::from("name")).cloned(),
                        map.get(&PropertyKey::from("age")).cloned(),
                    )),
                    other => panic!("expected Map, got {other:?}"),
                }
            }
            rows
        }

        #[test]
        fn snappy_and_uncompressed_files_load_the_same_rows() {
            let want = vec![
                (Some(Value::from("Alix")), Some(Value::Int64(31))),
                (Some(Value::from("Gus")), Some(Value::Null)),
            ];
            for compression in [Compression::UNCOMPRESSED, Compression::SNAPPY] {
                let file = write_people(compression);
                // The writer compressed what it was asked to.
                let reader = SerializedFileReader::new(file.reopen().unwrap()).unwrap();
                let row_group = reader.metadata().row_group(0);
                for column in row_group.columns() {
                    assert_eq!(column.compression(), compression);
                }
                assert_eq!(load(&file), want, "{compression:?}");
            }
        }

        /// A copy of `file` whose column chunks say they are compressed with
        /// `codec`, the pages left as they are. This build cannot write those
        /// codecs, and the loader refuses the file before it reads a page.
        fn marked_compressed_with(
            file: &tempfile::NamedTempFile,
            codec: Compression,
        ) -> tempfile::NamedTempFile {
            let bytes = std::fs::read(file.path()).unwrap();
            // The file ends with its metadata, the metadata's length (four
            // bytes, little-endian) and `PAR1`.
            let length_at = bytes.len() - 8;
            let length = u32::from_le_bytes(bytes[length_at..length_at + 4].try_into().unwrap());
            let metadata_at = length_at - usize::try_from(length).unwrap();
            let reader = SerializedFileReader::new(file.reopen().unwrap()).unwrap();
            let mut builder = reader.metadata().clone().into_builder();
            let row_groups = builder
                .take_row_groups()
                .into_iter()
                .map(|row_group| {
                    let columns = row_group
                        .columns()
                        .iter()
                        .map(|column| {
                            column
                                .clone()
                                .into_builder()
                                .set_compression(codec)
                                .build()
                                .unwrap()
                        })
                        .collect();
                    row_group
                        .into_builder()
                        .set_column_metadata(columns)
                        .build()
                        .unwrap()
                })
                .collect();
            let metadata = builder.set_row_groups(row_groups).build();
            let mut marked = bytes[..metadata_at].to_vec();
            ParquetMetaDataWriter::new(&mut marked, &metadata)
                .finish()
                .unwrap();
            let copy = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(copy.path(), marked).unwrap();
            copy
        }

        /// A file in a codec this build lacks fails with the codec's name
        /// and the codecs it reads, not the parquet crate's compile-time
        /// feature (`zstd`, `flate2` for gzip).
        #[test]
        fn another_codec_is_refused_naming_it() {
            let snappy = write_people(Compression::SNAPPY);
            for (codec, name) in [
                (Compression::ZSTD(Default::default()), "ZSTD"),
                (Compression::GZIP(Default::default()), "GZIP"),
                (Compression::LZ4_RAW, "LZ4_RAW"),
                (Compression::BROTLI(Default::default()), "BROTLI"),
            ] {
                let file = marked_compressed_with(&snappy, codec);
                let mut op = LoadDataOperator::new(
                    file.path().to_str().unwrap().to_string(),
                    LoadDataFormat::Parquet,
                    true,
                    None,
                    "row".to_string(),
                );
                let message = op.next().unwrap_err().to_string();
                let path = file.path().display();
                assert!(
                    message.ends_with(&format!(
                        "the Parquet file '{path}' is compressed with {name}; this build \
                         reads uncompressed and Snappy Parquet files"
                    )),
                    "{message}"
                );
            }
        }

        /// Writes one row group of `schema`, uncompressed, each column filled
        /// by the next writer in `columns`.
        fn write_columns(
            schema: &str,
            columns: Vec<Box<dyn FnOnce(&mut SerializedColumnWriter<'_>)>>,
        ) -> tempfile::NamedTempFile {
            let schema = parse_message_type(schema).unwrap();
            let file = tempfile::NamedTempFile::new().unwrap();
            let mut writer = SerializedFileWriter::new(
                file.reopen().unwrap(),
                Arc::new(schema),
                Arc::new(WriterProperties::builder().build()),
            )
            .unwrap();
            let mut row_group = writer.next_row_group().unwrap();
            for write in columns {
                let mut column = row_group.next_column().unwrap().unwrap();
                write(&mut column);
                column.close().unwrap();
            }
            row_group.close().unwrap();
            writer.close().unwrap();
            file
        }

        /// A column writer that writes `values` of type `T` with the given
        /// definition and repetition levels.
        fn column<T: DataType>(
            values: Vec<T::T>,
            definitions: Option<Vec<i16>>,
            repetitions: Option<Vec<i16>>,
        ) -> Box<dyn FnOnce(&mut SerializedColumnWriter<'_>)>
        where
            T::T: 'static,
        {
            Box::new(move |writer| {
                writer
                    .typed::<T>()
                    .write_batch(&values, definitions.as_deref(), repetitions.as_deref())
                    .unwrap();
            })
        }

        /// The rows of a Parquet file, each a map from column to value.
        fn load_rows(file: &tempfile::NamedTempFile) -> Vec<BTreeMap<PropertyKey, Value>> {
            let mut op = LoadDataOperator::new(
                file.path().to_str().unwrap().to_string(),
                LoadDataFormat::Parquet,
                true,
                None,
                "row".to_string(),
            );
            let mut rows = Vec::new();
            while let Some(chunk) = op.next().unwrap() {
                match chunk.column(0).unwrap().get_value(0).unwrap() {
                    Value::Map(map) => rows.push((*map).clone()),
                    other => panic!("expected Map, got {other:?}"),
                }
            }
            rows
        }

        fn timestamp(micros: i64) -> Value {
            Value::Timestamp(Timestamp::from_micros(micros))
        }

        fn time(nanos: u64) -> Value {
            Value::Time(Time::from_nanos(nanos).unwrap())
        }

        /// The decimal `mantissa / 10^scale`, checked to keep its scale.
        fn decimal(value: &Value, mantissa: i128, scale: u8) {
            match value {
                Value::Decimal(d) => {
                    assert_eq!((d.mantissa(), d.scale()), (mantissa, scale), "{d}");
                }
                other => panic!("expected the decimal {mantissa}e-{scale}, got {other:?}"),
            }
        }

        /// 2026-10-31T10:00:00Z in milliseconds since the epoch.
        const HALLOWEEN_MILLIS: i64 = 1_793_440_800_000;

        /// Each temporal and decimal type pyarrow, pandas, polars and Spark
        /// write loads as the Grafeo value of that type: DATE as a date,
        /// TIMESTAMP in every unit, with or without UTC adjustment, as a
        /// datetime (as a Python datetime and the Arrow export's
        /// `Timestamp(us, UTC)` are), TIME as a time of day and DECIMAL as an
        /// exact decimal at the column's scale.
        #[test]
        fn temporal_and_decimal_columns_load_as_their_types() {
            let file = write_columns(
                "message typed {
                    REQUIRED INT32 day (DATE);
                    REQUIRED INT64 local_millis (TIMESTAMP(MILLIS,false));
                    REQUIRED INT64 utc_micros (TIMESTAMP(MICROS,true));
                    REQUIRED INT64 utc_nanos (TIMESTAMP(NANOS,true));
                    REQUIRED INT64 before_epoch_nanos (TIMESTAMP(NANOS,false));
                    REQUIRED INT32 time_millis (TIME(MILLIS,false));
                    REQUIRED INT64 time_micros (TIME(MICROS,true));
                    REQUIRED INT64 time_nanos (TIME(NANOS,false));
                    REQUIRED INT32 price (DECIMAL(9,2));
                    REQUIRED INT64 balance (DECIMAL(18,4));
                    REQUIRED FIXED_LEN_BYTE_ARRAY (16) widest (DECIMAL(38,0));
                    REQUIRED BYTE_ARRAY cents (DECIMAL(20,2));
                }",
                vec![
                    column::<Int32Type>(vec![20_757], None, None),
                    column::<Int64Type>(vec![HALLOWEEN_MILLIS], None, None),
                    column::<Int64Type>(vec![HALLOWEEN_MILLIS * 1_000 + 250_000], None, None),
                    column::<Int64Type>(
                        vec![HALLOWEEN_MILLIS * 1_000_000 + 123_456_789],
                        None,
                        None,
                    ),
                    column::<Int64Type>(vec![-1], None, None),
                    column::<Int32Type>(vec![37_800_500], None, None),
                    column::<Int64Type>(vec![37_800_000_250], None, None),
                    column::<Int64Type>(vec![37_800_000_000_001], None, None),
                    column::<Int32Type>(vec![12_345], None, None),
                    column::<Int64Type>(vec![-98_765_432_100], None, None),
                    column::<FixedLenByteArrayType>(
                        vec![FixedLenByteArray::from(
                            (10_i128.pow(38) - 1).to_be_bytes().to_vec(),
                        )],
                        None,
                        None,
                    ),
                    column::<ByteArrayType>(vec![ByteArray::from(vec![0xFF_u8])], None, None),
                ],
            );
            let rows = load_rows(&file);
            assert_eq!(rows.len(), 1);
            let row = &rows[0];
            let get = |name: &str| row.get(&PropertyKey::from(name)).unwrap();

            assert_eq!(
                get("day"),
                &Value::Date(Date::from_ymd(2026, 10, 31).unwrap())
            );
            assert_eq!(get("local_millis"), &timestamp(HALLOWEEN_MILLIS * 1_000));
            assert_eq!(
                get("utc_micros"),
                &timestamp(HALLOWEEN_MILLIS * 1_000 + 250_000)
            );
            // Grafeo keeps microseconds: the nanoseconds below one are dropped,
            // toward the past, so one before the epoch stays before it.
            assert_eq!(
                get("utc_nanos"),
                &timestamp(HALLOWEEN_MILLIS * 1_000 + 123_456)
            );
            assert_eq!(get("before_epoch_nanos"), &timestamp(-1));
            // 10:30:00.5, 10:30:00.00025 and 10:30:00.000000001.
            assert_eq!(get("time_millis"), &time(37_800_500_000_000));
            assert_eq!(get("time_micros"), &time(37_800_000_250_000));
            assert_eq!(get("time_nanos"), &time(37_800_000_000_001));
            decimal(get("price"), 12_345, 2);
            decimal(get("balance"), -98_765_432_100, 4);
            decimal(get("widest"), 10_i128.pow(38) - 1, 0);
            decimal(get("cents"), -1, 2);
        }

        /// A decimal column wider than Grafeo's 38 digits loads as its exact
        /// text, as the Arrow export writes such a column.
        #[test]
        fn a_decimal_column_past_38_digits_loads_as_its_text() {
            let mut widest = vec![0_u8];
            widest.extend_from_slice(&u128::MAX.to_be_bytes());
            let file = write_columns(
                "message wide {
                    REQUIRED FIXED_LEN_BYTE_ARRAY (17) huge (DECIMAL(40,2));
                    REQUIRED FIXED_LEN_BYTE_ARRAY (17) small (DECIMAL(40,2));
                }",
                vec![
                    column::<FixedLenByteArrayType>(
                        vec![FixedLenByteArray::from(widest)],
                        None,
                        None,
                    ),
                    column::<FixedLenByteArrayType>(
                        vec![FixedLenByteArray::from({
                            let mut small = vec![0xFF_u8; 17];
                            small[16] = 0xFE;
                            small
                        })],
                        None,
                        None,
                    ),
                ],
            );
            let rows = load_rows(&file);
            assert_eq!(
                rows[0].get(&PropertyKey::from("huge")),
                Some(&Value::from("3402823669209384634633746074317682114.55"))
            );
            // Every value of the column is text, also one that would fit.
            assert_eq!(
                rows[0].get(&PropertyKey::from("small")),
                Some(&Value::from("-0.02"))
            );
        }

        /// Nested columns read by their own types too: a timestamp in a list,
        /// in a legacy two-level list, in a struct and as a map's value.
        #[test]
        fn nested_temporal_values_load_as_their_types() {
            let file = write_columns(
                "message nested {
                    OPTIONAL group stamps (LIST) {
                        REPEATED group list {
                            OPTIONAL INT64 element (TIMESTAMP(NANOS,true));
                        }
                    }
                    OPTIONAL group legacy (LIST) {
                        REPEATED INT64 at (TIMESTAMP(NANOS,false));
                    }
                    OPTIONAL group visit {
                        REQUIRED BYTE_ARRAY city (UTF8);
                        REQUIRED INT64 at (TIME(NANOS,false));
                    }
                    OPTIONAL group arrivals (MAP) {
                        REPEATED group key_value {
                            REQUIRED BYTE_ARRAY key (UTF8);
                            OPTIONAL INT64 value (TIMESTAMP(NANOS,true));
                        }
                    }
                }",
                vec![
                    column::<Int64Type>(vec![1_000, 2_000], Some(vec![3, 3]), Some(vec![0, 1])),
                    column::<Int64Type>(vec![3_000, 4_000], Some(vec![2, 2]), Some(vec![0, 1])),
                    column::<ByteArrayType>(
                        vec![ByteArray::from("Amsterdam")],
                        Some(vec![1]),
                        None,
                    ),
                    column::<Int64Type>(vec![37_800_000_000_001], Some(vec![1]), None),
                    column::<ByteArrayType>(
                        vec![ByteArray::from("Berlin")],
                        Some(vec![2]),
                        Some(vec![0]),
                    ),
                    column::<Int64Type>(vec![5_000], Some(vec![3]), Some(vec![0])),
                ],
            );
            let rows = load_rows(&file);
            let get = |name: &str| rows[0].get(&PropertyKey::from(name)).unwrap().clone();
            assert_eq!(
                get("stamps"),
                Value::List(Arc::from(vec![timestamp(1), timestamp(2)]))
            );
            // The record reader nests a legacy list of a repeated primitive
            // one level deeper; its elements still read by their type.
            assert_eq!(
                get("legacy"),
                Value::List(Arc::from(vec![Value::List(Arc::from(vec![
                    timestamp(3),
                    timestamp(4)
                ]))]))
            );
            let mut visit = BTreeMap::new();
            visit.insert(PropertyKey::from("city"), Value::from("Amsterdam"));
            visit.insert(PropertyKey::from("at"), time(37_800_000_000_001));
            assert_eq!(get("visit"), Value::Map(Arc::new(visit)));
            let mut arrivals = BTreeMap::new();
            arrivals.insert(PropertyKey::from("Berlin"), timestamp(5));
            assert_eq!(get("arrivals"), Value::Map(Arc::new(arrivals)));
        }
    }
}
