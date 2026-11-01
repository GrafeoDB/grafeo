//! Map collect operator.
//!
//! Reads all rows from a child operator and collects key-value pairs
//! into a single `Value::Map`. Used for Gremlin `groupCount()` semantics.

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use grafeo_common::types::{LogicalType, PropertyKey, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Collects key-value pairs from child rows into a single Map value.
pub struct MapCollectOperator {
    child: Box<dyn Operator>,
    key_col: usize,
    value_col: usize,
    done: bool,
}

impl MapCollectOperator {
    /// Creates a new map collect operator.
    #[must_use]
    pub fn new(child: Box<dyn Operator>, key_col: usize, value_col: usize) -> Self {
        Self {
            child,
            key_col,
            value_col,
            done: false,
        }
    }
}

impl Operator for MapCollectOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        let mut map = BTreeMap::new();
        while let Some(chunk) = self.child.next()? {
            crate::execution::deadline::check()?;
            for row in chunk.selected_indices() {
                let key = chunk.column(self.key_col).and_then(|c| c.get_value(row));
                let value = chunk.column(self.value_col).and_then(|c| c.get_value(row));
                if let (Some(k), Some(v)) = (key, value) {
                    let key_str: PropertyKey = match &k {
                        Value::String(s) => PropertyKey::from(s.as_str()),
                        other => PropertyKey::from(format!("{other}").as_str()),
                    };
                    map.insert(key_str, v);
                }
            }
        }

        let mut output = DataChunk::with_capacity(&[LogicalType::Any], 1);
        output
            .column_mut(0)
            .expect("column 0 exists: single-column schema")
            .push_value(Value::Map(Arc::new(map)));
        output.set_count(1);

        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.done = false;
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "MapCollect"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;

    /// A simple mock operator that yields pre-built chunks one at a time.
    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position >= self.chunks.len() {
                return Ok(None);
            }
            let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
            self.position += 1;
            Ok(Some(chunk))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Helper: builds a two-column chunk (String key, Int64 value) from slices.
    fn build_chunk(keys: &[&str], values: &[i64]) -> DataChunk {
        assert_eq!(keys.len(), values.len());
        let mut chunk =
            DataChunk::with_capacity(&[LogicalType::String, LogicalType::Int64], keys.len());
        for key in keys {
            chunk
                .column_mut(0)
                .unwrap()
                .push_value(Value::String((*key).into()));
        }
        for val in values {
            chunk.column_mut(1).unwrap().push_value(Value::Int64(*val));
        }
        chunk.set_count(keys.len());
        chunk
    }

    #[test]
    fn test_basic_map_collection() {
        let chunk = build_chunk(&["NYC", "LA"], &[2, 1]);
        let mock = MockOperator::new(vec![chunk]);

        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        // First call: single row with a Map value
        let result = op.next().unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        match value {
            Value::Map(map) => {
                assert_eq!(map.len(), 2);
                assert_eq!(map.get(&PropertyKey::new("NYC")), Some(&Value::Int64(2)));
                assert_eq!(map.get(&PropertyKey::new("LA")), Some(&Value::Int64(1)));
            }
            other => panic!("Expected Value::Map, got {:?}", other),
        }

        // Second call: exhausted
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_empty_input_produces_empty_map() {
        let mock = MockOperator::new(vec![]);

        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        let result = op.next().unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.row_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        match value {
            Value::Map(map) => {
                assert!(map.is_empty(), "Expected empty map, got {map:?}");
            }
            other => panic!("Expected Value::Map, got {:?}", other),
        }

        // Second call: exhausted
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_reset_allows_reprocessing() {
        let chunk = build_chunk(&["a"], &[10]);
        let mock = MockOperator::new(vec![chunk]);

        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        // Consume the first result
        let result = op.next().unwrap();
        assert!(result.is_some());
        assert!(op.next().unwrap().is_none());

        // After reset, the child mock is also reset but its chunks were
        // consumed via mem::replace, so we get an empty map (the mock's
        // chunks are replaced with DataChunk::empty()). The important
        // thing is that reset() clears the `done` flag and produces a
        // new result instead of returning None.
        op.reset();
        let result = op.next().unwrap();
        assert!(
            result.is_some(),
            "After reset, next() should produce a result"
        );

        // The result is a single-row chunk with a Map
        let result = result.unwrap();
        assert_eq!(result.row_count(), 1);
        let value = result.column(0).unwrap().get_value(0).unwrap();
        assert!(
            matches!(value, Value::Map(_)),
            "Expected Value::Map after reset"
        );
    }

    #[test]
    fn test_name_returns_map_collect() {
        let mock = MockOperator::new(vec![]);
        let op = MapCollectOperator::new(Box::new(mock), 0, 1);
        assert_eq!(op.name(), "MapCollect");
    }

    #[test]
    fn test_multiple_chunks_merged_into_single_map() {
        let chunk1 = build_chunk(&["x", "y"], &[1, 2]);
        let chunk2 = build_chunk(&["z"], &[3]);
        let mock = MockOperator::new(vec![chunk1, chunk2]);

        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        match value {
            Value::Map(map) => {
                assert_eq!(map.len(), 3);
                assert_eq!(map.get(&PropertyKey::new("x")), Some(&Value::Int64(1)));
                assert_eq!(map.get(&PropertyKey::new("y")), Some(&Value::Int64(2)));
                assert_eq!(map.get(&PropertyKey::new("z")), Some(&Value::Int64(3)));
            }
            other => panic!("Expected Value::Map, got {:?}", other),
        }
    }

    #[test]
    fn test_duplicate_keys_last_value_wins() {
        // When the same key appears multiple times, the last value should win
        // because BTreeMap::insert overwrites.
        let chunk = build_chunk(&["k", "k"], &[1, 2]);
        let mock = MockOperator::new(vec![chunk]);

        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        let result = op.next().unwrap().unwrap();
        let value = result.column(0).unwrap().get_value(0).unwrap();
        match value {
            Value::Map(map) => {
                assert_eq!(map.len(), 1);
                assert_eq!(
                    map.get(&PropertyKey::new("k")),
                    Some(&Value::Int64(2)),
                    "Last value should win for duplicate keys"
                );
            }
            other => panic!("Expected Value::Map, got {:?}", other),
        }
    }

    #[test]
    fn test_non_string_keys_converted_via_display() {
        // When the key column contains non-string values (e.g. Int64),
        // they should be converted to strings via Display formatting.
        let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64, LogicalType::String], 2);
        chunk.column_mut(0).unwrap().push_value(Value::Int64(42));
        chunk.column_mut(0).unwrap().push_value(Value::Int64(99));
        chunk
            .column_mut(1)
            .unwrap()
            .push_value(Value::String("val_a".into()));
        chunk
            .column_mut(1)
            .unwrap()
            .push_value(Value::String("val_b".into()));
        chunk.set_count(2);

        let mock = MockOperator::new(vec![chunk]);
        let mut op = MapCollectOperator::new(Box::new(mock), 0, 1);

        let result = op.next().unwrap().unwrap();
        let value = result.column(0).unwrap().get_value(0).unwrap();
        match value {
            Value::Map(map) => {
                assert_eq!(map.len(), 2);
                assert_eq!(
                    map.get(&PropertyKey::new("42")),
                    Some(&Value::String("val_a".into()))
                );
                assert_eq!(
                    map.get(&PropertyKey::new("99")),
                    Some(&Value::String("val_b".into()))
                );
            }
            other => panic!("Expected Value::Map, got {:?}", other),
        }
    }

    #[test]
    fn test_map_collect_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = MapCollectOperator::new(Box::new(mock), 0, 1);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<MapCollectOperator>().is_ok());
    }
}
