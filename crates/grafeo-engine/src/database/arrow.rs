//! Arrow IPC export for query results.
//!
//! Converts [`QueryResult`](super::QueryResult) to Arrow [`RecordBatch`] and serializes to Arrow IPC format.
//! Feature-gated behind `arrow-export`.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int64Builder, StringBuilder,
};
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};

use grafeo_common::{LogicalType, Value};

/// Errors from Arrow export operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArrowExportError {
    /// Error from the Arrow library.
    #[error("Arrow error: {0}")]
    Arrow(#[from] ArrowError),
}

/// Maps a grafeo [`LogicalType`] to an Arrow [`DataType`].
///
/// Falls back to `Utf8` for types that have no direct Arrow equivalent.
fn logical_type_to_arrow(logical_type: &LogicalType) -> DataType {
    match logical_type {
        LogicalType::Null => DataType::Null,
        LogicalType::Bool => DataType::Boolean,
        LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64 => {
            DataType::Int64
        }
        LogicalType::Float32 | LogicalType::Float64 => DataType::Float64,
        LogicalType::String => DataType::Utf8,
        LogicalType::Bytes => DataType::Binary,
        LogicalType::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        LogicalType::Date => DataType::Date32,
        LogicalType::Time => DataType::Time64(TimeUnit::Nanosecond),
        LogicalType::Duration => DataType::Utf8, // ISO 8601 string (Arrow Duration lacks months)
        LogicalType::ZonedDatetime | LogicalType::ZonedTime => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        LogicalType::Vector(dim) => DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            i32::try_from(*dim).unwrap_or(0),
        ),
        LogicalType::List(_)
        | LogicalType::Map { .. }
        | LogicalType::Struct(_)
        | LogicalType::Node
        | LogicalType::Edge
        | LogicalType::Path
        | LogicalType::Any => DataType::Utf8,
        _ => DataType::Utf8,
    }
}

/// Infers the Arrow [`DataType`] for a column from its [`LogicalType`] hint and actual values.
///
/// If the logical type is `Any` (unknown), scans values to find the dominant type.
/// Falls back to `Utf8` for heterogeneous columns.
fn infer_column_type(logical_type: &LogicalType, column: &[&Value]) -> DataType {
    if *logical_type != LogicalType::Any {
        return logical_type_to_arrow(logical_type);
    }

    // Scan values to find the dominant non-null type
    let mut seen_type: Option<DataType> = None;
    for value in column {
        let dt = match value {
            Value::Null => continue,
            Value::Bool(_) => DataType::Boolean,
            Value::Int64(_) => DataType::Int64,
            Value::Float64(_) => DataType::Float64,
            Value::String(_) => DataType::Utf8,
            Value::Bytes(_) => DataType::Binary,
            Value::Timestamp(_) => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            Value::Date(_) => DataType::Date32,
            Value::Time(_) => DataType::Time64(TimeUnit::Nanosecond),
            Value::Duration(_) => DataType::Utf8,
            Value::ZonedDatetime(_) => {
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
            }
            Value::Vector(v) => DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                i32::try_from(v.len()).unwrap_or(0),
            ),
            Value::List(_)
            | Value::Map(_)
            | Value::Path { .. }
            | Value::GCounter(_)
            | Value::OnCounter { .. } => DataType::Utf8,
            _ => DataType::Utf8,
        };

        match &seen_type {
            None => seen_type = Some(dt),
            Some(existing) if *existing == dt => {}
            Some(_) => return DataType::Utf8, // Mixed types: fall back to string
        }
    }

    seen_type.unwrap_or(DataType::Null)
}

/// Builds an Arrow [`ArrayRef`] from a column of [`Value`] references.
fn build_array(column: &[&Value], target_type: &DataType) -> Result<ArrayRef, ArrowExportError> {
    let len = column.len();

    match target_type {
        DataType::Null => Ok(Arc::new(arrow_array::NullArray::new(len)) as ArrayRef),
        DataType::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(len);
            for value in column {
                match value {
                    Value::Bool(b) => builder.append_value(*b),
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(len);
            for value in column {
                match value {
                    Value::Int64(i) => builder.append_value(*i),
                    // reason: intentional lossy f64-to-i64 coercion for Arrow column
                    #[allow(clippy::cast_possible_truncation)]
                    Value::Float64(f) => builder.append_value(*f as i64),
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(len);
            for value in column {
                match value {
                    Value::Float64(f) => builder.append_value(*f),
                    Value::Int64(i) => builder.append_value(*i as f64),
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(len, len * 32);
            for value in column {
                match value {
                    Value::Null => builder.append_null(),
                    Value::String(s) => builder.append_value(s.as_str()),
                    other => builder.append_value(other.to_string()),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Binary => {
            let mut builder = BinaryBuilder::with_capacity(len, len * 64);
            for value in column {
                match value {
                    Value::Bytes(b) => builder.append_value(b.as_ref()),
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let mut builder = Int64Builder::with_capacity(len);
            for value in column {
                match value {
                    Value::Timestamp(ts) => builder.append_value(ts.as_micros()),
                    Value::ZonedDatetime(zdt) => {
                        builder.append_value(zdt.as_timestamp().as_micros());
                    }
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            let int_array = builder.finish();
            // Reinterpret as TimestampMicrosecondArray
            let data = int_array.into_data();
            let ts_data = data
                .into_builder()
                .data_type(DataType::Timestamp(
                    TimeUnit::Microsecond,
                    Some("UTC".into()),
                ))
                .build()?;
            Ok(Arc::new(arrow_array::TimestampMicrosecondArray::from(ts_data)) as ArrayRef)
        }
        DataType::Date32 => {
            let values: Vec<Option<i32>> = column
                .iter()
                .map(|v| match v {
                    Value::Date(d) => Some(d.as_days()),
                    _ => None,
                })
                .collect();
            Ok(Arc::new(arrow_array::Date32Array::from(values)) as ArrayRef)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let mut builder = Int64Builder::with_capacity(len);
            for value in column {
                match value {
                    // reason: time-of-day nanos < 86_400e9, well within i64 range
                    #[allow(clippy::cast_possible_wrap)]
                    Value::Time(t) => builder.append_value(t.as_nanos() as i64),
                    Value::Null => builder.append_null(),
                    _ => builder.append_null(),
                }
            }
            let int_array = builder.finish();
            let data = int_array
                .into_data()
                .into_builder()
                .data_type(DataType::Time64(TimeUnit::Nanosecond))
                .build()?;
            Ok(Arc::new(arrow_array::Time64NanosecondArray::from(data)) as ArrayRef)
        }
        DataType::FixedSizeList(_, dim) => {
            // reason: Arrow FixedSizeList dimension is always non-negative
            #[allow(clippy::cast_sign_loss)]
            let dim_usize = *dim as usize;
            let mut float_builder = Float32Builder::with_capacity(len * dim_usize);
            let mut null_mask = Vec::with_capacity(len);
            for value in column {
                match value {
                    Value::Vector(v) if v.len() == dim_usize => {
                        for f in v.iter() {
                            float_builder.append_value(*f);
                        }
                        null_mask.push(true);
                    }
                    Value::Null => {
                        for _ in 0..dim_usize {
                            float_builder.append_value(0.0);
                        }
                        null_mask.push(false);
                    }
                    _ => {
                        for _ in 0..dim_usize {
                            float_builder.append_value(0.0);
                        }
                        null_mask.push(false);
                    }
                }
            }
            let values_array = float_builder.finish();
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            let list_array = FixedSizeListArray::try_new(
                field,
                *dim,
                Arc::new(values_array),
                Some(null_mask.into()),
            )?;
            Ok(Arc::new(list_array) as ArrayRef)
        }
        // Fallback: serialize as string
        _ => {
            let mut builder = StringBuilder::with_capacity(len, len * 32);
            for value in column {
                match value {
                    Value::Null => builder.append_null(),
                    other => builder.append_value(other.to_string()),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
    }
}

/// Converts a [`QueryResult`](super::QueryResult) to an Arrow [`RecordBatch`].
///
/// # Errors
///
/// Returns [`ArrowExportError`] if column type inference fails or Arrow
/// array construction encounters incompatible data.
pub fn query_result_to_record_batch(
    columns: &[String],
    column_types: &[LogicalType],
    rows: &[Vec<Value>],
) -> Result<RecordBatch, ArrowExportError> {
    if columns.is_empty() {
        let schema = Arc::new(Schema::empty());
        return Ok(RecordBatch::new_empty(schema));
    }

    let num_cols = columns.len();
    let num_rows = rows.len();

    // Extract column-oriented data
    let mut col_values: Vec<Vec<&Value>> = vec![Vec::with_capacity(num_rows); num_cols];
    for row in rows {
        for (col_idx, value) in row.iter().enumerate() {
            if col_idx < num_cols {
                col_values[col_idx].push(value);
            }
        }
    }

    // Infer types and build arrays
    let mut fields = Vec::with_capacity(num_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(num_cols);

    for (col_idx, col_name) in columns.iter().enumerate() {
        let logical_type = column_types.get(col_idx).unwrap_or(&LogicalType::Any);
        let values = &col_values[col_idx];
        let arrow_type = infer_column_type(logical_type, values);

        fields.push(Field::new(col_name.as_str(), arrow_type.clone(), true));
        arrays.push(build_array(values, &arrow_type)?);
    }

    let schema = Arc::new(Schema::new(fields));
    Ok(RecordBatch::try_new(schema, arrays)?)
}

/// Serializes a [`RecordBatch`] to Arrow IPC stream format bytes.
///
/// # Errors
///
/// Returns [`ArrowExportError`] if IPC stream encoding fails.
pub fn record_batch_to_ipc_stream(batch: &RecordBatch) -> Result<Vec<u8>, ArrowExportError> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

// =========================================================================
// Bulk export: nodes and edges to Arrow RecordBatch
// =========================================================================

#[cfg(feature = "lpg")]
mod bulk_export {
    use std::collections::HashSet;
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder, StringBuilder as LB, UInt64Builder};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use grafeo_common::LogicalType;
    use grafeo_common::types::Value;
    use grafeo_core::graph::lpg::{Edge, Node};

    use super::{ArrowExportError, build_array, infer_column_type, record_batch_to_ipc_stream};

    /// Structural column names for nodes (property keys matching these are skipped).
    const RESERVED_NODE_COLS: &[&str] = &["_id", "_labels"];

    /// Structural column names for edges (property keys matching these are skipped).
    const RESERVED_EDGE_COLS: &[&str] = &["_id", "_source", "_target", "_type"];

    /// Discovers property keys in first-seen order, skipping reserved column names.
    fn discover_property_keys<'a>(
        properties_iter: impl Iterator<Item = impl Iterator<Item = &'a str>>,
        reserved: &[&str],
    ) -> Vec<String> {
        let mut keys = Vec::new();
        let mut seen = HashSet::new();
        for prop_keys in properties_iter {
            for key in prop_keys {
                if seen.insert(key.to_owned()) && !reserved.contains(&key) {
                    keys.push(key.to_owned());
                }
            }
        }
        keys
    }

    /// Converts a slice of [`Node`]s to an Arrow [`RecordBatch`].
    ///
    /// Schema: `_id` (UInt64), `_labels` (List\<Utf8\>), plus one nullable column per
    /// unique property key. Structural columns are underscore-prefixed to avoid
    /// collision with user property names. Property types are inferred from
    /// values; mixed-type columns fall back to Utf8.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`] on Arrow construction failure.
    pub fn nodes_to_record_batch(nodes: &[Node]) -> Result<RecordBatch, ArrowExportError> {
        let num_rows = nodes.len();

        // Discover property keys in first-seen order
        let prop_keys = discover_property_keys(
            nodes
                .iter()
                .map(|n| n.properties.iter().map(|(k, _)| k.as_str())),
            RESERVED_NODE_COLS,
        );

        // Build structural columns
        let mut id_builder = UInt64Builder::with_capacity(num_rows);
        let mut labels_builder = ListBuilder::new(LB::new());

        for node in nodes {
            id_builder.append_value(node.id.0);
            for label in &node.labels {
                labels_builder.values().append_value(&**label);
            }
            labels_builder.append(true);
        }

        let mut fields: Vec<Field> = vec![
            Field::new("_id", DataType::UInt64, false),
            Field::new(
                "_labels",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                false,
            ),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(labels_builder.finish()),
        ];

        // Build property columns
        for key in &prop_keys {
            let prop_key = grafeo_common::types::PropertyKey::new(key.clone());
            let values: Vec<&Value> = nodes
                .iter()
                .map(|n| n.properties.get(&prop_key).unwrap_or(&Value::Null))
                .collect();
            let arrow_type = infer_column_type(&LogicalType::Any, &values);
            fields.push(Field::new(key.as_str(), arrow_type.clone(), true));
            arrays.push(build_array(&values, &arrow_type)?);
        }

        let schema = Arc::new(Schema::new(fields));
        Ok(RecordBatch::try_new(schema, arrays)?)
    }

    /// Converts a slice of [`Edge`]s to an Arrow [`RecordBatch`].
    ///
    /// Schema: `_id` (UInt64), `_type` (Utf8), `_source` (UInt64), `_target` (UInt64),
    /// plus one nullable column per unique property key. Structural columns are
    /// underscore-prefixed to avoid collision with user property names.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`] on Arrow construction failure.
    pub fn edges_to_record_batch(edges: &[Edge]) -> Result<RecordBatch, ArrowExportError> {
        let num_rows = edges.len();

        // Discover property keys in first-seen order
        let prop_keys = discover_property_keys(
            edges
                .iter()
                .map(|e| e.properties.iter().map(|(k, _)| k.as_str())),
            RESERVED_EDGE_COLS,
        );

        // Build structural columns
        let mut id_builder = UInt64Builder::with_capacity(num_rows);
        let mut type_builder = StringBuilder::with_capacity(num_rows, num_rows * 16);
        let mut source_builder = UInt64Builder::with_capacity(num_rows);
        let mut target_builder = UInt64Builder::with_capacity(num_rows);

        for edge in edges {
            id_builder.append_value(edge.id.0);
            type_builder.append_value(&*edge.edge_type);
            source_builder.append_value(edge.src.0);
            target_builder.append_value(edge.dst.0);
        }

        let mut fields: Vec<Field> = vec![
            Field::new("_id", DataType::UInt64, false),
            Field::new("_type", DataType::Utf8, false),
            Field::new("_source", DataType::UInt64, false),
            Field::new("_target", DataType::UInt64, false),
        ];
        let mut arrays: Vec<ArrayRef> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(type_builder.finish()),
            Arc::new(source_builder.finish()),
            Arc::new(target_builder.finish()),
        ];

        // Build property columns
        for key in &prop_keys {
            let prop_key = grafeo_common::types::PropertyKey::new(key.clone());
            let values: Vec<&Value> = edges
                .iter()
                .map(|e| e.properties.get(&prop_key).unwrap_or(&Value::Null))
                .collect();
            let arrow_type = infer_column_type(&LogicalType::Any, &values);
            fields.push(Field::new(key.as_str(), arrow_type.clone(), true));
            arrays.push(build_array(&values, &arrow_type)?);
        }

        let schema = Arc::new(Schema::new(fields));
        Ok(RecordBatch::try_new(schema, arrays)?)
    }

    /// Serializes nodes to Arrow IPC stream format bytes.
    ///
    /// Convenience wrapper: `nodes_to_record_batch` + `record_batch_to_ipc_stream`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`] on Arrow construction or IPC encoding failure.
    pub fn nodes_to_ipc_stream(nodes: &[Node]) -> Result<Vec<u8>, ArrowExportError> {
        let batch = nodes_to_record_batch(nodes)?;
        record_batch_to_ipc_stream(&batch)
    }

    /// Serializes edges to Arrow IPC stream format bytes.
    ///
    /// Convenience wrapper: `edges_to_record_batch` + `record_batch_to_ipc_stream`.
    ///
    /// # Errors
    ///
    /// Returns [`ArrowExportError`] on Arrow construction or IPC encoding failure.
    pub fn edges_to_ipc_stream(edges: &[Edge]) -> Result<Vec<u8>, ArrowExportError> {
        let batch = edges_to_record_batch(edges)?;
        record_batch_to_ipc_stream(&batch)
    }
}

#[cfg(feature = "lpg")]
pub use bulk_export::{
    edges_to_ipc_stream, edges_to_record_batch, nodes_to_ipc_stream, nodes_to_record_batch,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc as StdArc;

    use arrow_array::Array;
    use arrow_schema::DataType;
    use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};
    use grafeo_common::{LogicalType, PropertyKey, Value};

    use super::{query_result_to_record_batch, record_batch_to_ipc_stream};

    fn make_result(
        columns: Vec<&str>,
        types: Vec<LogicalType>,
        rows: Vec<Vec<Value>>,
    ) -> (Vec<String>, Vec<LogicalType>, Vec<Vec<Value>>) {
        (columns.into_iter().map(String::from).collect(), types, rows)
    }

    #[test]
    fn test_empty_result() {
        let (cols, types, rows) = make_result(vec![], vec![], vec![]);
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_null_column() {
        let (cols, types, rows) = make_result(
            vec!["x"],
            vec![LogicalType::Null],
            vec![vec![Value::Null], vec![Value::Null]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Null);
    }

    #[test]
    fn test_bool_column() {
        let (cols, types, rows) = make_result(
            vec!["flag"],
            vec![LogicalType::Bool],
            vec![vec![Value::Bool(true)], vec![Value::Bool(false)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .unwrap();
        assert!(arr.value(0));
        assert!(!arr.value(1));
    }

    #[test]
    fn test_int64_column() {
        let (cols, types, rows) = make_result(
            vec!["age"],
            vec![LogicalType::Int64],
            vec![
                vec![Value::Int64(30)],
                vec![Value::Null],
                vec![Value::Int64(-5)],
            ],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 30);
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), -5);
    }

    #[test]
    fn test_float64_column() {
        let (cols, types, rows) = make_result(
            vec!["score"],
            vec![LogicalType::Float64],
            vec![vec![Value::Float64(3.125)], vec![Value::Float64(-0.5)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .unwrap();
        assert!((arr.value(0) - 3.125).abs() < f64::EPSILON);
    }

    #[test]
    fn test_string_column() {
        let (cols, types, rows) = make_result(
            vec!["name"],
            vec![LogicalType::String],
            vec![
                vec![Value::String("Alix".into())],
                vec![Value::Null],
                vec![Value::String("Gus".into())],
            ],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        assert_eq!(arr.value(0), "Alix");
        assert!(arr.is_null(1));
        assert_eq!(arr.value(2), "Gus");
    }

    #[test]
    fn test_bytes_column() {
        let (cols, types, rows) = make_result(
            vec!["data"],
            vec![LogicalType::Bytes],
            vec![vec![Value::Bytes(StdArc::from(vec![1u8, 2, 3].as_slice()))]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap();
        assert_eq!(arr.value(0), &[1, 2, 3]);
    }

    #[test]
    fn test_timestamp_column() {
        let ts = Timestamp::from_micros(1_700_000_000_000_000);
        let (cols, types, rows) = make_result(
            vec!["created"],
            vec![LogicalType::Timestamp],
            vec![vec![Value::Timestamp(ts)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(arr.value(0), 1_700_000_000_000_000);
    }

    #[test]
    fn test_date_column() {
        let date = Date::from_ymd(2025, 6, 15).unwrap();
        let (cols, types, rows) = make_result(
            vec!["birthday"],
            vec![LogicalType::Date],
            vec![vec![Value::Date(date)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_time_column() {
        let time = Time::from_hms(14, 30, 0).unwrap();
        let (cols, types, rows) = make_result(
            vec!["alarm"],
            vec![LogicalType::Time],
            vec![vec![Value::Time(time)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_duration_as_string() {
        let dur = Duration::new(2, 5, 1_000_000_000);
        let (cols, types, rows) = make_result(
            vec!["interval"],
            vec![LogicalType::Duration],
            vec![vec![Value::Duration(dur)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        // Duration maps to Utf8
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_zoned_datetime_column() {
        let zdt = ZonedDatetime::from_timestamp_offset(
            Timestamp::from_micros(1_700_000_000_000_000),
            3600,
        );
        let (cols, types, rows) = make_result(
            vec!["event_at"],
            vec![LogicalType::ZonedDatetime],
            vec![vec![Value::ZonedDatetime(zdt)]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(arr.value(0), 1_700_000_000_000_000);
    }

    #[test]
    fn test_vector_column() {
        let vec3 = Value::Vector(StdArc::from(vec![1.0f32, 2.0, 3.0].as_slice()));
        let (cols, types, rows) = make_result(
            vec!["embedding"],
            vec![LogicalType::Vector(3)],
            vec![vec![vec3]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_rows(), 1);
        match batch.schema().field(0).data_type() {
            DataType::FixedSizeList(_, 3) => {}
            other => panic!("Expected FixedSizeList(_, 3), got {other:?}"),
        }
    }

    #[test]
    fn test_list_as_string() {
        let list = Value::List(StdArc::from(vec![Value::Int64(1), Value::Int64(2)]));
        let (cols, types, rows) = make_result(
            vec!["items"],
            vec![LogicalType::List(Box::new(LogicalType::Int64))],
            vec![vec![list]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_map_as_string() {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("key"), Value::String("val".into()));
        let map_val = Value::Map(StdArc::from(map));
        let (cols, types, rows) = make_result(
            vec!["props"],
            vec![LogicalType::Map {
                key: Box::new(LogicalType::String),
                value: Box::new(LogicalType::String),
            }],
            vec![vec![map_val]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_heterogeneous_column_falls_back_to_string() {
        let (cols, types, rows) = make_result(
            vec!["mixed"],
            vec![LogicalType::Any],
            vec![vec![Value::Int64(42)], vec![Value::String("hello".into())]],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(*batch.schema().field(0).data_type(), DataType::Utf8);
    }

    #[test]
    fn test_multi_column() {
        let (cols, types, rows) = make_result(
            vec!["name", "age", "active"],
            vec![LogicalType::String, LogicalType::Int64, LogicalType::Bool],
            vec![
                vec![
                    Value::String("Alix".into()),
                    Value::Int64(30),
                    Value::Bool(true),
                ],
                vec![
                    Value::String("Gus".into()),
                    Value::Int64(25),
                    Value::Bool(false),
                ],
            ],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn test_ipc_roundtrip() {
        let (cols, types, rows) = make_result(
            vec!["id", "name"],
            vec![LogicalType::Int64, LogicalType::String],
            vec![
                vec![Value::Int64(1), Value::String("Alix".into())],
                vec![Value::Int64(2), Value::String("Gus".into())],
            ],
        );
        let batch = query_result_to_record_batch(&cols, &types, &rows).unwrap();
        let ipc_bytes = record_batch_to_ipc_stream(&batch).unwrap();
        assert!(!ipc_bytes.is_empty(), "ipc_bytes is empty");

        // Read back
        let cursor = std::io::Cursor::new(ipc_bytes);
        let reader = arrow_ipc::reader::StreamReader::try_new(cursor, None).unwrap();
        let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
        assert_eq!(batches[0].num_columns(), 2);
    }

    // =====================================================================
    // Bulk export: nodes and edges
    // =====================================================================

    #[cfg(feature = "lpg")]
    mod bulk_export_tests {
        use super::*;
        use grafeo_common::types::{EdgeId, NodeId};
        use grafeo_core::graph::lpg::{Edge, Node};

        fn make_node(id: u64, labels: &[&str]) -> Node {
            let mut node = Node::new(NodeId(id));
            for label in labels {
                node.labels.push((*label).into());
            }
            node
        }

        fn make_edge(id: u64, src: u64, dst: u64, edge_type: &str) -> Edge {
            Edge::new(EdgeId(id), NodeId(src), NodeId(dst), edge_type)
        }

        #[test]
        fn test_nodes_empty() {
            let batch = crate::database::arrow::nodes_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 2); // id, labels
        }

        #[test]
        fn test_nodes_basic() {
            let mut alix = make_node(1, &["Person"]);
            alix.properties
                .insert(PropertyKey::new("name"), Value::String("Alix".into()));
            alix.properties
                .insert(PropertyKey::new("age"), Value::Int64(30));

            let mut gus = make_node(2, &["Person", "Developer"]);
            gus.properties
                .insert(PropertyKey::new("name"), Value::String("Gus".into()));

            let batch = crate::database::arrow::nodes_to_record_batch(&[alix, gus]).unwrap();
            assert_eq!(batch.num_rows(), 2);
            // id, labels, name, age
            assert_eq!(batch.num_columns(), 4);
            assert_eq!(batch.schema().field(0).name(), "_id");
            assert_eq!(batch.schema().field(1).name(), "_labels");
            assert_eq!(batch.schema().field(2).name(), "name");
            assert_eq!(batch.schema().field(3).name(), "age");
        }

        #[test]
        fn test_nodes_reserved_column_skipped() {
            let mut node = make_node(1, &["Test"]);
            node.properties
                .insert(PropertyKey::new("_id"), Value::Int64(999)); // should be skipped
            node.properties
                .insert(PropertyKey::new("score"), Value::Float64(0.95));

            let batch = crate::database::arrow::nodes_to_record_batch(&[node]).unwrap();
            // _id (structural), _labels (structural), score (property)
            assert_eq!(batch.num_columns(), 3);
            assert_eq!(batch.schema().field(2).name(), "score");
        }

        /// Regression: properties named "id" or "labels" must NOT be dropped.
        /// Old code reserved these bare names, causing silent data loss.
        #[test]
        fn test_nodes_property_named_id_preserved() {
            let mut node = make_node(1, &["Method"]);
            node.properties
                .insert(PropertyKey::new("id"), Value::String("custom-uuid".into()));
            node.properties
                .insert(PropertyKey::new("labels"), Value::String("meta".into()));

            let batch = crate::database::arrow::nodes_to_record_batch(&[node]).unwrap();
            // _id, _labels, id (property), labels (property)
            assert_eq!(batch.num_columns(), 4);
            let names: Vec<_> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert!(
                names.contains(&"id".to_string()),
                "property 'id' must be preserved"
            );
            assert!(
                names.contains(&"labels".to_string()),
                "property 'labels' must be preserved"
            );
        }

        #[test]
        fn test_nodes_ipc_roundtrip() {
            let mut node = make_node(1, &["Person"]);
            node.properties
                .insert(PropertyKey::new("name"), Value::String("Alix".into()));

            let ipc_bytes = crate::database::arrow::nodes_to_ipc_stream(&[node]).unwrap();
            assert!(!ipc_bytes.is_empty(), "ipc_bytes is empty");

            let cursor = std::io::Cursor::new(ipc_bytes);
            let reader = arrow_ipc::reader::StreamReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].num_rows(), 1);
        }

        #[test]
        fn test_edges_empty() {
            let batch = crate::database::arrow::edges_to_record_batch(&[]).unwrap();
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 4); // _id, _type, _source, _target
        }

        #[test]
        fn test_edges_basic() {
            let mut edge = make_edge(1, 10, 20, "KNOWS");
            edge.properties
                .insert(PropertyKey::new("since"), Value::Int64(2020));

            let batch = crate::database::arrow::edges_to_record_batch(&[edge]).unwrap();
            assert_eq!(batch.num_rows(), 1);
            // _id, _type, _source, _target, since
            assert_eq!(batch.num_columns(), 5);
            assert_eq!(batch.schema().field(0).name(), "_id");
            assert_eq!(batch.schema().field(1).name(), "_type");
            assert_eq!(batch.schema().field(2).name(), "_source");
            assert_eq!(batch.schema().field(3).name(), "_target");
            assert_eq!(batch.schema().field(4).name(), "since");
        }

        /// Regression: properties named "source" or "target" must NOT be dropped.
        /// Old code reserved these bare names, causing silent data loss.
        #[test]
        fn test_edges_property_named_source_preserved() {
            let mut edge = make_edge(1, 10, 20, "CALLS");
            edge.properties
                .insert(PropertyKey::new("source"), Value::String("jdt".into()));
            edge.properties
                .insert(PropertyKey::new("confidence"), Value::Float64(0.9));

            let batch = crate::database::arrow::edges_to_record_batch(&[edge]).unwrap();
            // _id, _type, _source, _target, source (property), confidence
            assert_eq!(batch.num_columns(), 6);
            let names: Vec<_> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert!(
                names.contains(&"source".to_string()),
                "property 'source' must be preserved"
            );
            assert!(names.contains(&"confidence".to_string()));
        }

        /// Regression: boolean properties must appear in Arrow export.
        #[test]
        fn test_nodes_boolean_properties_preserved() {
            let mut node = make_node(1, &["Method"]);
            node.properties
                .insert(PropertyKey::new("name"), Value::String("foo".into()));
            node.properties
                .insert(PropertyKey::new("is_exported"), Value::Bool(true));
            node.properties
                .insert(PropertyKey::new("is_test"), Value::Bool(false));

            let batch = crate::database::arrow::nodes_to_record_batch(&[node]).unwrap();
            let names: Vec<_> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert!(
                names.contains(&"is_exported".to_string()),
                "bool property 'is_exported' must be present"
            );
            assert!(
                names.contains(&"is_test".to_string()),
                "bool property 'is_test' must be present"
            );
        }

        #[test]
        fn test_edges_ipc_roundtrip() {
            let edge = make_edge(1, 10, 20, "KNOWS");
            let ipc_bytes = crate::database::arrow::edges_to_ipc_stream(&[edge]).unwrap();
            assert!(!ipc_bytes.is_empty(), "ipc_bytes is empty");

            let cursor = std::io::Cursor::new(ipc_bytes);
            let reader = arrow_ipc::reader::StreamReader::try_new(cursor, None).unwrap();
            let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].num_rows(), 1);
            assert_eq!(batches[0].num_columns(), 4);
        }
    }
}
