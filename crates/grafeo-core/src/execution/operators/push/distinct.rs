//! Push-based distinct operator.

use crate::execution::chunk::{ColumnTypes, DataChunk};
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::vector::ValueVector;
use grafeo_common::types::{HashableValue, Value};
use std::collections::HashSet;

/// Row key with cached per-value hashes and complete typed equality.
#[derive(Debug, Clone)]
struct RowKey(Vec<(u64, HashableValue)>);

impl PartialEq for RowKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|((_, left), (_, right))| left == right)
    }
}

impl Eq for RowKey {}

impl std::hash::Hash for RowKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for (hash, _) in &self.0 {
            hash.hash(state);
        }
    }
}

impl RowKey {
    fn from_row(chunk: &DataChunk, row: usize, columns: &[usize]) -> Self {
        let parts = columns
            .iter()
            .map(|&column| Self::key_part(chunk, row, column))
            .collect();
        Self(parts)
    }

    fn from_all_columns(chunk: &DataChunk, row: usize) -> Self {
        let parts = (0..chunk.column_count())
            .map(|column| Self::key_part(chunk, row, column))
            .collect();
        Self(parts)
    }

    fn key_part(chunk: &DataChunk, row: usize, column: usize) -> (u64, HashableValue) {
        let value = HashableValue::from(
            chunk
                .column(column)
                .and_then(|column| column.get_value(row))
                .unwrap_or(Value::Null),
        );
        (hash_value(&value), value)
    }
}

fn hash_value(value: &HashableValue) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    hash_value_into(value, &mut hasher);
    hasher.finish()
}

/// Hashes the complete value using the same semantics as row-key equality.
fn hash_value_into(value: &HashableValue, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;

    value.hash(hasher);
}

/// Push-based distinct operator.
///
/// Filters out duplicate rows based on all columns or specified columns.
/// This operator maintains state (seen values) but can produce output
/// incrementally as new unique rows arrive.
pub struct DistinctPushOperator {
    /// Columns to check for distinctness (None = all columns).
    columns: Option<Vec<usize>>,
    /// Set of seen row keys.
    seen: HashSet<RowKey>,
}

impl DistinctPushOperator {
    /// Create a distinct operator on all columns.
    pub fn new() -> Self {
        Self {
            columns: None,
            seen: HashSet::new(),
        }
    }

    /// Create a distinct operator on specific columns.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self {
            columns: Some(columns),
            seen: HashSet::new(),
        }
    }

    /// Get the number of unique rows seen.
    pub fn unique_count(&self) -> usize {
        self.seen.len()
    }
}

impl Default for DistinctPushOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Find rows that are new (not seen before)
        let mut new_indices = Vec::new();

        for row in chunk.selected_indices() {
            let key = match &self.columns {
                Some(cols) => RowKey::from_row(&chunk, row, cols),
                None => RowKey::from_all_columns(&chunk, row),
            };

            if self.seen.insert(key) {
                new_indices.push(row);
            }
        }

        if new_indices.is_empty() {
            return Ok(true);
        }

        // Copy retained rows using their input column types, including entity IDs.
        let mut columns: Vec<ValueVector> = chunk
            .columns()
            .iter()
            .map(|column| ValueVector::with_capacity(column.data_type().clone(), new_indices.len()))
            .collect();
        for row in new_indices {
            for (column_index, column) in columns.iter_mut().enumerate() {
                let value = chunk
                    .column(column_index)
                    .and_then(|column| column.get_value(row))
                    .unwrap_or(Value::Null);
                column.push_value(value);
            }
        }

        sink.consume(DataChunk::new(columns))
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Nothing to finalize - all output was produced incrementally
        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "DistinctPush"
    }
}

/// Push-based distinct operator that materializes all input first.
///
/// This is a true pipeline breaker that buffers all rows and produces
/// distinct output in the finalize phase. Use this when you need
/// deterministic ordering of output.
pub struct DistinctMaterializingOperator {
    /// Columns to check for distinctness.
    columns: Option<Vec<usize>>,
    /// Buffered unique rows.
    rows: Vec<Vec<Value>>,
    /// Set of seen row keys.
    seen: HashSet<RowKey>,
    /// Types of the buffered input columns.
    column_types: ColumnTypes,
}

impl DistinctMaterializingOperator {
    /// Create a distinct operator on all columns.
    pub fn new() -> Self {
        Self {
            columns: None,
            rows: Vec::new(),
            seen: HashSet::new(),
            column_types: ColumnTypes::default(),
        }
    }

    /// Create a distinct operator on specific columns.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self {
            columns: Some(columns),
            rows: Vec::new(),
            seen: HashSet::new(),
            column_types: ColumnTypes::default(),
        }
    }
}

impl Default for DistinctMaterializingOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctMaterializingOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        self.column_types.add(&chunk);

        let num_cols = chunk.column_count();

        for row in chunk.selected_indices() {
            let key = match &self.columns {
                Some(cols) => RowKey::from_row(&chunk, row, cols),
                None => RowKey::from_all_columns(&chunk, row),
            };

            if self.seen.insert(key) {
                // Store the full row
                let row_values: Vec<Value> = (0..num_cols)
                    .map(|col| {
                        chunk
                            .column(col)
                            .and_then(|c| c.get_value(row))
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                self.rows.push(row_values);
            }
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.rows.is_empty() {
            return Ok(());
        }

        let mut columns: Vec<ValueVector> = self
            .column_types
            .types()
            .iter()
            .map(|data_type| ValueVector::with_capacity(data_type.clone(), self.rows.len()))
            .collect();

        for row in &self.rows {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                let val = row.get(col_idx).cloned().unwrap_or(Value::Null);
                col.push(val);
            }
        }

        let chunk = DataChunk::new(columns);
        sink.consume(chunk)?;

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "DistinctMaterializing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::sink::CollectorSink;
    use grafeo_common::types::{EdgeId, LogicalType, NodeId, Time, ZonedDatetime};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_distinct_all_unique() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5);
        assert_eq!(distinct.unique_count(), 5);
    }

    #[test]
    fn test_distinct_with_duplicates() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 1, 3, 2, 1, 4]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 4); // 1, 2, 3, 4
        assert_eq!(distinct.unique_count(), 4);
    }

    #[test]
    fn test_distinct_all_same() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[5, 5, 5, 5, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(distinct.unique_count(), 1);
    }

    #[test]
    fn test_distinct_multiple_chunks() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 3]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[2, 3, 4]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[3, 4, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5); // 1, 2, 3, 4, 5
    }

    #[test]
    fn test_distinct_materializing() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        // All output comes in finalize
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 7); // 7 unique values
    }

    fn create_mixed_chunk(values: &[Value]) -> DataChunk {
        let vector = ValueVector::from_values(values);
        DataChunk::new(vec![vector])
    }

    fn value_rows_chunk(rows: &[Vec<Value>]) -> DataChunk {
        let columns = (0..rows[0].len())
            .map(|column| {
                ValueVector::from_values(
                    &rows
                        .iter()
                        .map(|row| row[column].clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        DataChunk::new(columns)
    }

    fn collected_value_rows(sink: &CollectorSink) -> Vec<Vec<Value>> {
        let mut rows = Vec::new();
        for chunk in sink.chunks() {
            for row in chunk.selected_indices() {
                rows.push(
                    chunk
                        .columns()
                        .iter()
                        .map(|column| column.get_value(row).unwrap())
                        .collect(),
                );
            }
        }
        rows
    }

    fn same_total_counters() -> [Value; 2] {
        [
            Value::GCounter(Arc::new(HashMap::from([
                ("left".into(), 4),
                ("right".into(), 6),
            ]))),
            Value::GCounter(Arc::new(HashMap::from([
                ("left".into(), 5),
                ("right".into(), 5),
            ]))),
        ]
    }

    #[test]
    fn test_distinct_preserves_crdt_replica_identity() {
        let [first, second] = same_total_counters();
        assert_ne!(first, second);
        assert_eq!(first.to_string(), second.to_string());
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(
                create_mixed_chunk(&[first.clone(), second.clone(), first.clone()]),
                &mut sink,
            )
            .unwrap();
        distinct
            .push(
                create_mixed_chunk(&[second.clone(), first.clone()]),
                &mut sink,
            )
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(collected_value_rows(&sink), vec![vec![first], vec![second]]);
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_materializing_preserves_crdt_replica_identity() {
        let [first, second] = same_total_counters();
        assert_ne!(first, second);
        assert_eq!(first.to_string(), second.to_string());
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(
                create_mixed_chunk(&[first.clone(), second.clone(), first.clone()]),
                &mut sink,
            )
            .unwrap();
        distinct
            .push(
                create_mixed_chunk(&[second.clone(), first.clone()]),
                &mut sink,
            )
            .unwrap();
        assert!(sink.is_empty());
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(collected_value_rows(&sink), vec![vec![first], vec![second]]);
    }

    #[test]
    fn test_distinct_preserves_typed_nested_rows_across_chunks() {
        let nested = |value| Value::List(vec![Value::List(vec![value].into())].into());
        let expected = vec![
            vec![Value::Int64(0)],
            vec![Value::Float64(0.0)],
            vec![nested(Value::Int64(0))],
            vec![nested(Value::Float64(0.0))],
            vec![nested(Value::Bytes(vec![7, 1, 2].into()))],
            vec![nested(Value::Bytes(vec![7, 3, 4].into()))],
            vec![Value::Null],
        ];
        for mut distinct in [
            Box::new(DistinctPushOperator::new()) as Box<dyn PushOperator>,
            Box::new(DistinctMaterializingOperator::new()),
        ] {
            let mut sink = CollectorSink::new();
            distinct
                .push(value_rows_chunk(&expected), &mut sink)
                .unwrap();
            distinct
                .push(value_rows_chunk(&expected), &mut sink)
                .unwrap();
            distinct.finalize(&mut sink).unwrap();
            assert_eq!(collected_value_rows(&sink), expected, "{}", distinct.name());
        }
    }

    #[test]
    fn test_distinct_selected_columns_preserve_full_rows() {
        let [first, second] = same_total_counters();
        let expected = vec![
            vec![first.clone(), Value::from("first")],
            vec![second.clone(), Value::from("second")],
            vec![Value::Null, Value::from("null")],
        ];
        let duplicates = vec![
            vec![second, Value::from("ignored second payload")],
            vec![first, Value::from("ignored first payload")],
            vec![Value::Null, Value::from("ignored null payload")],
        ];
        for mut distinct in [
            Box::new(DistinctPushOperator::on_columns(vec![0])) as Box<dyn PushOperator>,
            Box::new(DistinctMaterializingOperator::on_columns(vec![0])),
        ] {
            let mut sink = CollectorSink::new();
            distinct
                .push(value_rows_chunk(&expected), &mut sink)
                .unwrap();
            distinct
                .push(value_rows_chunk(&duplicates), &mut sink)
                .unwrap();
            distinct.finalize(&mut sink).unwrap();
            assert_eq!(collected_value_rows(&sink), expected, "{}", distinct.name());
        }
    }

    #[test]
    fn test_distinct_preserves_raw_float_bits() {
        let expected_bits = [
            0,
            (-0.0_f64).to_bits(),
            0x7ff8_0000_0000_0001,
            0x7ff8_0000_0000_0002,
        ];
        let values: Vec<_> = expected_bits
            .iter()
            .map(|&bits| Value::Float64(f64::from_bits(bits)))
            .collect();
        for mut distinct in [
            Box::new(DistinctPushOperator::new()) as Box<dyn PushOperator>,
            Box::new(DistinctMaterializingOperator::new()),
        ] {
            let mut sink = CollectorSink::new();
            distinct
                .push(create_mixed_chunk(&values), &mut sink)
                .unwrap();
            distinct
                .push(create_mixed_chunk(&values), &mut sink)
                .unwrap();
            distinct.finalize(&mut sink).unwrap();
            let actual: Vec<_> = collected_value_rows(&sink)
                .iter()
                .map(|row| match row.as_slice() {
                    [Value::Float64(value)] => value.to_bits(),
                    other => panic!("expected one unchanged float, got {other:?}"),
                })
                .collect();
            assert_eq!(actual, expected_bits, "{}", distinct.name());
        }
    }

    #[test]
    fn test_distinct_temporal_equality_preserves_first_values() {
        let first_datetime =
            Value::ZonedDatetime(ZonedDatetime::parse("2024-06-15T10:30:00+05:30").unwrap());
        let same_datetime =
            Value::ZonedDatetime(ZonedDatetime::parse("2024-06-15T05:00:00Z").unwrap());
        let first_time = Value::Time(Time::parse("14:00:00+01:00").unwrap());
        let same_time = Value::Time(Time::parse("13:00:00Z").unwrap());
        let local_time = Value::Time(Time::parse("13:00:00").unwrap());
        assert_eq!(first_datetime, same_datetime);
        assert_eq!(first_time, same_time);
        assert_ne!(first_time, local_time);

        let expected = vec![
            vec![first_datetime.clone()],
            vec![first_time.clone()],
            vec![local_time.clone()],
            vec![Value::List(vec![first_datetime, first_time].into())],
        ];
        let duplicates = vec![
            vec![same_datetime.clone()],
            vec![same_time.clone()],
            vec![local_time],
            vec![Value::List(vec![same_datetime, same_time].into())],
        ];
        for mut distinct in [
            Box::new(DistinctPushOperator::new()) as Box<dyn PushOperator>,
            Box::new(DistinctMaterializingOperator::new()),
        ] {
            let mut sink = CollectorSink::new();
            distinct
                .push(value_rows_chunk(&expected), &mut sink)
                .unwrap();
            distinct
                .push(value_rows_chunk(&duplicates), &mut sink)
                .unwrap();
            distinct.finalize(&mut sink).unwrap();
            let actual = collected_value_rows(&sink);
            assert_eq!(actual, expected, "{}", distinct.name());
            assert_eq!(actual[0][0].to_string(), "2024-06-15T10:30:00+05:30");
            assert_eq!(actual[1][0].to_string(), "14:00:00+01:00");
        }
    }

    #[test]
    fn test_distinct_preserves_node_edge_columns() {
        for mut distinct in [
            Box::new(DistinctPushOperator::new()) as Box<dyn PushOperator>,
            Box::new(DistinctMaterializingOperator::new()),
        ] {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Edge]);
            for id in [1, 1, 2] {
                builder.column_mut(0).unwrap().push_node_id(NodeId::new(id));
                builder.column_mut(1).unwrap().push_edge_id(EdgeId::new(id));
                builder.advance_row();
            }
            let mut sink = CollectorSink::new();
            distinct.push(builder.finish(), &mut sink).unwrap();
            distinct.finalize(&mut sink).unwrap();

            assert_eq!(sink.row_count(), 2, "{}", distinct.name());
            assert_eq!(sink.chunks().len(), 1);
            let chunk = &sink.chunks()[0];
            assert_eq!(
                chunk.column_types(),
                [LogicalType::Node, LogicalType::Edge],
                "{}",
                distinct.name()
            );
            for (row, id) in [1, 2].into_iter().enumerate() {
                assert_eq!(
                    chunk.column(0).unwrap().get_node_id(row),
                    Some(NodeId::new(id))
                );
                assert_eq!(
                    chunk.column(1).unwrap().get_edge_id(row),
                    Some(EdgeId::new(id))
                );
            }
        }
    }

    #[test]
    fn test_distinct_null_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Null, Value::Null, Value::Int64(1)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2); // Null + 1
    }

    #[test]
    fn test_distinct_bool_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Bool(true), Value::Bool(false), Value::Bool(true)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_float_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Float64(1.0),
            Value::Float64(2.0),
            Value::Float64(1.0),
            Value::Float64(f64::NAN),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 3); // 1.0, 2.0, NaN
    }

    #[test]
    fn test_distinct_string_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk =
            create_mixed_chunk(&[Value::from("Alix"), Value::from("Gus"), Value::from("Alix")]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_bytes_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Bytes(vec![1u8, 2, 3].into()),
            Value::Bytes(vec![4u8, 5, 6].into()),
            Value::Bytes(vec![1u8, 2, 3].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_list_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
            Value::List(vec![Value::Int64(3), Value::Int64(4)].into()),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_map_values() {
        use std::collections::BTreeMap;

        let mut map1 = BTreeMap::new();
        map1.insert("a".into(), Value::Int64(1));
        let mut map2 = BTreeMap::new();
        map2.insert("b".into(), Value::Int64(2));

        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Map(map1.clone().into()),
            Value::Map(map2.into()),
            Value::Map(map1.into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_vector_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::Vector(vec![3.0_f32, 4.0].into()),
            Value::Vector(vec![1.0_f32, 2.0].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_path_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let path1 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::Int64(10)].into(),
        };
        let path2 = Value::Path {
            nodes: vec![Value::Int64(3), Value::Int64(4)].into(),
            edges: vec![Value::Int64(20)].into(),
        };

        let chunk = create_mixed_chunk(&[path1.clone(), path2, path1]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_mixed_types_are_distinct() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        // Different types with "similar" content should be distinct
        let chunk = create_mixed_chunk(&[
            Value::Int64(1),
            Value::Float64(1.0),
            Value::from("1"),
            Value::Bool(true),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 4);
    }

    #[test]
    fn test_hash_value_deterministic() {
        // Same value should always produce the same hash
        let v1 = HashableValue::from(Value::from("test"));
        let v2 = HashableValue::from(Value::from("test"));
        assert_eq!(hash_value(&v1), hash_value(&v2));

        // Different values should (almost certainly) produce different hashes
        let v3 = HashableValue::from(Value::from("other"));
        assert_ne!(hash_value(&v1), hash_value(&v3));
    }
}
