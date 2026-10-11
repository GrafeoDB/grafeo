//! Arrow IPC export for query results.
//!
//! Converts [`QueryResult`](super::QueryResult) to Arrow [`RecordBatch`] and serializes to Arrow IPC format.
//! Feature-gated behind `arrow-export`.
//!
//! # Column types
//!
//! A column's Arrow type comes from its values (or from the result's column
//! type where that names a scalar type):
//!
//! - a list is an Arrow `List` of its elements' type, also nested;
//! - a map is an Arrow `Struct` with one field per key that any map of the
//!   column has, each field typed by its values: a key a map lacks is null
//!   in that row (a `Struct` keeps each key's own type, where an Arrow `Map`
//!   needs one type for all values, and polars reads it natively);
//! - a duration is a `Struct` of `months`, `days` and `nanos` (`Int64`): an
//!   Arrow `Duration` cannot hold months, and polars cannot read an Arrow
//!   `Interval`;
//! - a vector is a `FixedSizeList` of `Float32`.
//!
//! Nulls are null at every level. Values of different types in one column
//! (or one list) are written as their GQL text in a `Utf8` column, except
//! integers and floats together, which are `Float64`; vectors of different
//! lengths are text too.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int64Builder, NullBufferBuilder,
    OffsetBufferBuilder, StringBuilder,
};
use arrow_array::{ArrayRef, FixedSizeListArray, ListArray, RecordBatch, StructArray};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, Field, Fields, Schema, TimeUnit};

use grafeo_common::{LogicalType, PropertyKey, Value};

/// Errors from Arrow export operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArrowExportError {
    /// Error from the Arrow library.
    #[error("Arrow error: {0}")]
    Arrow(#[from] ArrowError),
}

/// The Arrow type of a list whose elements have type `element`.
fn list_of(element: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", element, true)))
}

/// The Arrow type of a map with these keys and value types, in this order.
fn struct_of(fields: impl IntoIterator<Item = (String, DataType)>) -> DataType {
    DataType::Struct(
        fields
            .into_iter()
            .map(|(name, data_type)| Field::new(name, data_type, true))
            .collect::<Fields>(),
    )
}

/// The Arrow type of a duration: its months, days and nanoseconds.
fn duration_type() -> DataType {
    struct_of(
        ["months", "days", "nanos"]
            .into_iter()
            .map(|name| (name.to_string(), DataType::Int64)),
    )
}

/// The Arrow type of `value`, or `None` for null (see the module docs).
fn value_type(value: &Value) -> Option<DataType> {
    Some(match value {
        Value::Null => return None,
        Value::Bool(_) => DataType::Boolean,
        Value::Int64(_) => DataType::Int64,
        Value::Float64(_) => DataType::Float64,
        Value::String(_) => DataType::Utf8,
        Value::Bytes(_) => DataType::Binary,
        Value::Timestamp(_) | Value::ZonedDatetime(_) => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        Value::Date(_) => DataType::Date32,
        Value::Time(_) => DataType::Time64(TimeUnit::Nanosecond),
        Value::Duration(_) => duration_type(),
        Value::Vector(v) => DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            i32::try_from(v.len()).unwrap_or(0),
        ),
        Value::List(items) => list_of(common_type(items.iter())),
        Value::Map(map) => struct_of(map.iter().map(|(key, value)| {
            (
                key.as_str().to_string(),
                value_type(value).unwrap_or(DataType::Null),
            )
        })),
        // Paths and counters have no Arrow form: their text
        _ => DataType::Utf8,
    })
}

/// The one Arrow type that holds values of types `left` and `right` (see
/// the module docs): `Utf8` text when there is none.
fn unify(left: DataType, right: DataType) -> DataType {
    match (left, right) {
        (left, right) if left == right => left,
        (DataType::Null, other) | (other, DataType::Null) => other,
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => {
            DataType::Float64
        }
        (DataType::List(left), DataType::List(right)) => {
            list_of(unify(left.data_type().clone(), right.data_type().clone()))
        }
        (DataType::Struct(left), DataType::Struct(right)) => {
            let mut fields: Vec<(String, DataType)> = left
                .iter()
                .map(|field| (field.name().clone(), field.data_type().clone()))
                .collect();
            for field in &right {
                match fields.iter_mut().find(|(name, _)| name == field.name()) {
                    Some((_, data_type)) => {
                        let known = std::mem::replace(data_type, DataType::Null);
                        *data_type = unify(known, field.data_type().clone());
                    }
                    None => fields.push((field.name().clone(), field.data_type().clone())),
                }
            }
            struct_of(fields)
        }
        _ => DataType::Utf8,
    }
}

/// The Arrow type that holds every one of `values` (`Null` when all are).
fn common_type<'a>(values: impl Iterator<Item = &'a Value>) -> DataType {
    values.filter_map(value_type).fold(DataType::Null, unify)
}

/// The value of the struct field `key` for `value`, a map or a duration:
/// null for a key the map lacks and for a null row.
fn struct_field(value: &Value, key: &PropertyKey) -> Value {
    match value {
        Value::Map(map) => map.get(key).cloned().unwrap_or(Value::Null),
        Value::Duration(duration) => match key.as_str() {
            "months" => Value::Int64(duration.months()),
            "days" => Value::Int64(duration.days()),
            "nanos" => Value::Int64(duration.nanos()),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
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
        LogicalType::Duration => duration_type(),
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
/// A hint that names a scalar type gives the column's type; for any other
/// (`Any`, a list, a map, a duration) the values do (see the module docs).
fn infer_column_type(logical_type: &LogicalType, column: &[&Value]) -> DataType {
    match logical_type {
        LogicalType::Any
        | LogicalType::List(_)
        | LogicalType::Map { .. }
        | LogicalType::Struct(_)
        | LogicalType::Duration => common_type(column.iter().copied()),
        other => logical_type_to_arrow(other),
    }
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
        DataType::List(field) => {
            let mut offsets = OffsetBufferBuilder::<i32>::new(len);
            let mut nulls = NullBufferBuilder::new(len);
            let mut elements: Vec<&Value> = Vec::new();
            for value in column {
                let length = match value {
                    Value::List(items) => {
                        elements.extend(items.iter());
                        nulls.append_non_null();
                        items.len()
                    }
                    _ => {
                        nulls.append_null();
                        0
                    }
                };
                offsets.try_push_length(length).map_err(|error| {
                    ArrowError::InvalidArgumentError(format!(
                        "the lists of one column hold too many values: {error}"
                    ))
                })?;
            }
            let offsets = offsets.try_finish().map_err(|error| {
                ArrowError::InvalidArgumentError(format!(
                    "the lists of one column hold more than {} values: {error}",
                    i32::MAX
                ))
            })?;
            let values = build_array(&elements, field.data_type())?;
            Ok(Arc::new(ListArray::try_new(
                Arc::clone(field),
                offsets,
                values,
                nulls.finish(),
            )?) as ArrayRef)
        }
        DataType::Struct(fields) => {
            let mut nulls = NullBufferBuilder::new(len);
            for value in column {
                nulls.append(matches!(value, Value::Map(_) | Value::Duration(_)));
            }
            if fields.is_empty() {
                return Ok(Arc::new(StructArray::new_empty_fields(len, nulls.finish())) as ArrayRef);
            }
            let mut children = Vec::with_capacity(fields.len());
            for field in fields {
                let key = PropertyKey::new(field.name().as_str());
                let values: Vec<Value> = column
                    .iter()
                    .map(|value| struct_field(value, &key))
                    .collect();
                let values: Vec<&Value> = values.iter().collect();
                children.push(build_array(&values, field.data_type())?);
            }
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                nulls.finish(),
            )?) as ArrayRef)
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

    /// Row `row` of `array` as text: `null`, a number, a string, `[...]` for
    /// a list and `{key: value, ...}` for a struct.
    fn render(array: &dyn Array, row: usize) -> String {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Float32Type, Float64Type, Int64Type};
        if array.data_type() == &DataType::Null || array.is_null(row) {
            return "null".to_string();
        }
        match array.data_type() {
            DataType::Boolean => array.as_boolean().value(row).to_string(),
            DataType::Int64 => array.as_primitive::<Int64Type>().value(row).to_string(),
            DataType::Float64 => format!("{:?}", array.as_primitive::<Float64Type>().value(row)),
            DataType::Float32 => format!("{:?}", array.as_primitive::<Float32Type>().value(row)),
            DataType::Utf8 => format!("'{}'", array.as_string::<i32>().value(row)),
            DataType::List(_) => {
                let items = array.as_list::<i32>().value(row);
                let items: Vec<String> = (0..items.len()).map(|i| render(&items, i)).collect();
                format!("[{}]", items.join(", "))
            }
            DataType::FixedSizeList(..) => {
                let items = array.as_fixed_size_list().value(row);
                let items: Vec<String> = (0..items.len()).map(|i| render(&items, i)).collect();
                format!("[{}]", items.join(", "))
            }
            DataType::Struct(fields) => {
                let record = array.as_struct();
                let fields: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, field)| {
                        format!("{}: {}", field.name(), render(record.column(i), row))
                    })
                    .collect();
                format!("{{{}}}", fields.join(", "))
            }
            other => format!("<{other:?}>"),
        }
    }

    /// The one column of a result whose rows are `values`, with `hint` as
    /// its column type: its Arrow type and each row as text (see `render`).
    fn column(hint: LogicalType, values: Vec<Value>) -> (DataType, Vec<String>) {
        let rows: Vec<Vec<Value>> = values.into_iter().map(|value| vec![value]).collect();
        let batch = query_result_to_record_batch(&["c".to_string()], &[hint], &rows).unwrap();
        let array = batch.column(0);
        let rendered = (0..array.len()).map(|row| render(array, row)).collect();
        (array.data_type().clone(), rendered)
    }

    fn list(values: Vec<Value>) -> Value {
        Value::List(StdArc::from(values))
    }

    fn map(entries: &[(&str, Value)]) -> Value {
        Value::Map(StdArc::new(
            entries
                .iter()
                .map(|(key, value)| (PropertyKey::from(*key), value.clone()))
                .collect(),
        ))
    }

    fn text(value: &str) -> Value {
        Value::String(value.into())
    }

    #[test]
    fn a_duration_is_a_struct_of_months_days_and_nanos() {
        let (data_type, rows) = column(
            LogicalType::Duration,
            vec![
                Value::Duration(Duration::new(3, 19, 88)),
                Value::Null,
                Value::Duration(Duration::new(0, 0, 0)),
            ],
        );
        assert_eq!(data_type, super::duration_type());
        assert_eq!(
            rows,
            [
                "{months: 3, days: 19, nanos: 88}",
                "null",
                "{months: 0, days: 0, nanos: 0}"
            ]
        );
    }

    #[test]
    fn a_list_is_an_arrow_list_of_its_elements() {
        for hint in [
            LogicalType::List(Box::new(LogicalType::Int64)),
            LogicalType::Any,
        ] {
            let (data_type, rows) = column(
                hint,
                vec![
                    list(vec![Value::Int64(3), Value::Int64(19)]),
                    Value::Null,
                    list(vec![]),
                    list(vec![Value::Int64(88), Value::Null]),
                ],
            );
            assert_eq!(data_type, super::list_of(DataType::Int64));
            assert_eq!(rows, ["[3, 19]", "null", "[]", "[88, null]"]);
        }
    }

    #[test]
    fn nested_lists_and_lists_of_maps_keep_their_shape() {
        let (data_type, rows) = column(
            LogicalType::Any,
            vec![list(vec![
                list(vec![text("Alix")]),
                list(vec![text("Gus"), text("Mia")]),
            ])],
        );
        assert_eq!(data_type, super::list_of(super::list_of(DataType::Utf8)));
        assert_eq!(rows, ["[['Alix'], ['Gus', 'Mia']]"]);

        let (_, rows) = column(
            LogicalType::Any,
            vec![list(vec![
                map(&[("city", text("Paris"))]),
                map(&[("city", text("Prague")), ("years", Value::Int64(3))]),
            ])],
        );
        assert_eq!(
            rows,
            ["[{city: 'Paris', years: null}, {city: 'Prague', years: 3}]"]
        );
    }

    #[test]
    fn a_map_is_a_struct_of_every_key_of_the_column() {
        // A key a map lacks is null in that row; a null row is null.
        let (data_type, rows) = column(
            LogicalType::Any,
            vec![
                map(&[("age", Value::Int64(19)), ("name", text("Alix"))]),
                map(&[("city", text("Paris")), ("name", text("Gus"))]),
                Value::Null,
                map(&[("tags", list(vec![text("a")])), ("name", Value::Null)]),
            ],
        );
        let DataType::Struct(fields) = &data_type else {
            panic!("expected a struct, got {data_type:?}");
        };
        let fields: Vec<(String, DataType)> = fields
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect();
        assert_eq!(
            fields,
            [
                ("age".to_string(), DataType::Int64),
                ("name".to_string(), DataType::Utf8),
                ("city".to_string(), DataType::Utf8),
                ("tags".to_string(), super::list_of(DataType::Utf8)),
            ]
        );
        assert_eq!(
            rows,
            [
                "{age: 19, name: 'Alix', city: null, tags: null}",
                "{age: null, name: 'Gus', city: 'Paris', tags: null}",
                "null",
                "{age: null, name: null, city: null, tags: ['a']}",
            ]
        );
    }

    #[test]
    fn a_map_hint_reads_the_values() {
        let (data_type, rows) = column(
            LogicalType::Map {
                key: Box::new(LogicalType::String),
                value: Box::new(LogicalType::Any),
            },
            vec![map(&[("k", Value::Float64(0.5))])],
        );
        assert!(matches!(data_type, DataType::Struct(_)), "{data_type:?}");
        assert_eq!(rows, ["{k: 0.5}"]);
    }

    #[test]
    fn values_of_different_types_in_one_column() {
        // Integers and floats are floats.
        let (data_type, rows) = column(
            LogicalType::Any,
            vec![Value::Int64(3), Value::Float64(2.5), Value::Null],
        );
        assert_eq!(data_type, DataType::Float64);
        assert_eq!(rows, ["3.0", "2.5", "null"]);
        // Any other mix is the values' text, at the level where the types
        // differ: the column, a list's elements or a map's key.
        let (data_type, rows) = column(
            LogicalType::Any,
            vec![list(vec![Value::Int64(3)]), Value::Int64(19), Value::Null],
        );
        assert_eq!(data_type, DataType::Utf8);
        assert_eq!(rows, ["'[3]'", "'19'", "null"]);
        let (data_type, rows) = column(
            LogicalType::Any,
            vec![list(vec![Value::Int64(3), text("x"), Value::Null])],
        );
        assert_eq!(data_type, super::list_of(DataType::Utf8));
        assert_eq!(rows, ["['3', 'x', null]"]);
        let (_, rows) = column(
            LogicalType::Any,
            vec![
                map(&[("k", Value::Int64(88))]),
                map(&[("k", text("Berlin"))]),
            ],
        );
        assert_eq!(rows, ["{k: '88'}", "{k: 'Berlin'}"]);
        // Vectors of different lengths are text.
        let (data_type, _) = column(
            LogicalType::Any,
            vec![
                Value::Vector(StdArc::from(vec![0.5f32].as_slice())),
                Value::Vector(StdArc::from(vec![0.5f32, 3.0].as_slice())),
            ],
        );
        assert_eq!(data_type, DataType::Utf8);
    }

    #[test]
    fn nested_types_survive_an_ipc_roundtrip() {
        let rows = vec![vec![
            list(vec![Value::Int64(3)]),
            map(&[("name", text("Vincent"))]),
            Value::Duration(Duration::new(0, 3, 19)),
        ]];
        let columns: Vec<String> = ["l", "m", "d"].map(String::from).to_vec();
        let batch =
            query_result_to_record_batch(&columns, &vec![LogicalType::Any; 3], &rows).unwrap();
        let ipc_bytes = record_batch_to_ipc_stream(&batch).unwrap();
        let reader =
            arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(ipc_bytes), None)
                .unwrap();
        let read: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
        assert_eq!(read[0].schema(), batch.schema());
        let rendered: Vec<String> = (0..3).map(|i| render(read[0].column(i), 0)).collect();
        assert_eq!(
            rendered,
            [
                "[3]",
                "{name: 'Vincent'}",
                "{months: 0, days: 3, nanos: 19}"
            ]
        );
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
        fn properties_keep_their_types_in_node_and_edge_batches() {
            let mut alix = make_node(1, &["Person"]);
            alix.properties
                .insert(PropertyKey::new("tags"), list(vec![text("a"), text("b")]));
            alix.properties.insert(
                PropertyKey::new("address"),
                map(&[("city", text("Amsterdam")), ("number", Value::Int64(3))]),
            );
            alix.properties.insert(
                PropertyKey::new("wait"),
                Value::Duration(Duration::new(0, 19, 0)),
            );
            let gus = make_node(2, &["Person"]);
            let batch = crate::database::arrow::nodes_to_record_batch(&[alix, gus]).unwrap();
            let rendered = |name: &str| -> Vec<String> {
                let array = batch.column_by_name(name).unwrap();
                (0..array.len()).map(|row| render(array, row)).collect()
            };
            assert_eq!(rendered("_labels"), ["['Person']", "['Person']"]);
            assert_eq!(rendered("tags"), ["['a', 'b']", "null"]);
            assert_eq!(
                rendered("address"),
                ["{city: 'Amsterdam', number: 3}", "null"]
            );
            assert_eq!(
                rendered("wait"),
                ["{months: 0, days: 19, nanos: 0}", "null"]
            );

            let mut edge = make_edge(1, 10, 20, "KNOWS");
            edge.properties.insert(
                PropertyKey::new("weights"),
                list(vec![Value::Float64(0.5), Value::Int64(3)]),
            );
            let batch = crate::database::arrow::edges_to_record_batch(&[edge]).unwrap();
            let weights = batch.column_by_name("weights").unwrap();
            assert_eq!(
                weights.data_type(),
                &super::super::list_of(DataType::Float64)
            );
            assert_eq!(render(weights, 0), "[0.5, 3.0]");
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
