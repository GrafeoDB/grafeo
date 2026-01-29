//! ValueVector for columnar data storage.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};

/// Default vector capacity (tuples per vector).
pub const DEFAULT_VECTOR_CAPACITY: usize = 2048;

/// A columnar vector of values.
///
/// ValueVector stores data in columnar format for efficient SIMD processing
/// and cache utilization during query execution.
#[derive(Debug, Clone)]
pub struct ValueVector {
    /// The logical type of values in this vector.
    data_type: LogicalType,
    /// The actual data storage.
    data: VectorData,
    /// Number of valid entries.
    len: usize,
    /// Validity bitmap (true = valid, false = null).
    validity: Option<Vec<bool>>,
}

/// Internal storage for vector data.
#[derive(Debug, Clone)]
enum VectorData {
    /// Boolean values.
    Bool(Vec<bool>),
    /// 64-bit integers.
    Int64(Vec<i64>),
    /// 64-bit floats.
    Float64(Vec<f64>),
    /// Strings (stored as Arc for cheap cloning).
    String(Vec<Arc<str>>),
    /// Node IDs.
    NodeId(Vec<NodeId>),
    /// Edge IDs.
    EdgeId(Vec<EdgeId>),
    /// Generic values (fallback for complex types).
    Generic(Vec<Value>),
}

impl ValueVector {
    /// Creates a new empty generic vector.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(LogicalType::Any, DEFAULT_VECTOR_CAPACITY)
    }

    /// Creates a new empty vector with the given type.
    #[must_use]
    pub fn with_type(data_type: LogicalType) -> Self {
        Self::with_capacity(data_type, DEFAULT_VECTOR_CAPACITY)
    }

    /// Creates a vector from a slice of values.
    pub fn from_values(values: &[Value]) -> Self {
        let mut vec = Self::new();
        for value in values {
            vec.push_value(value.clone());
        }
        vec
    }

    /// Creates a new vector with the given capacity.
    #[must_use]
    pub fn with_capacity(data_type: LogicalType, capacity: usize) -> Self {
        let data = match &data_type {
            LogicalType::Bool => VectorData::Bool(Vec::with_capacity(capacity)),
            LogicalType::Int8 | LogicalType::Int16 | LogicalType::Int32 | LogicalType::Int64 => {
                VectorData::Int64(Vec::with_capacity(capacity))
            }
            LogicalType::Float32 | LogicalType::Float64 => {
                VectorData::Float64(Vec::with_capacity(capacity))
            }
            LogicalType::String => VectorData::String(Vec::with_capacity(capacity)),
            LogicalType::Node => VectorData::NodeId(Vec::with_capacity(capacity)),
            LogicalType::Edge => VectorData::EdgeId(Vec::with_capacity(capacity)),
            _ => VectorData::Generic(Vec::with_capacity(capacity)),
        };

        Self {
            data_type,
            data,
            len: 0,
            validity: None,
        }
    }

    /// Returns the data type of this vector.
    #[must_use]
    pub fn data_type(&self) -> &LogicalType {
        &self.data_type
    }

    /// Returns the number of entries in this vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the value at index is null.
    #[must_use]
    pub fn is_null(&self, index: usize) -> bool {
        self.validity
            .as_ref()
            .map_or(false, |v| !v.get(index).copied().unwrap_or(true))
    }

    /// Sets the value at index to null.
    pub fn set_null(&mut self, index: usize) {
        if self.validity.is_none() {
            self.validity = Some(vec![true; self.len]);
        }
        if let Some(validity) = &mut self.validity {
            if index < validity.len() {
                validity[index] = false;
            }
        }
    }

    /// Pushes a boolean value.
    pub fn push_bool(&mut self, value: bool) {
        if let VectorData::Bool(vec) = &mut self.data {
            vec.push(value);
            self.len += 1;
        }
    }

    /// Pushes an integer value.
    pub fn push_int64(&mut self, value: i64) {
        if let VectorData::Int64(vec) = &mut self.data {
            vec.push(value);
            self.len += 1;
        }
    }

    /// Pushes a float value.
    pub fn push_float64(&mut self, value: f64) {
        if let VectorData::Float64(vec) = &mut self.data {
            vec.push(value);
            self.len += 1;
        }
    }

    /// Pushes a string value.
    pub fn push_string(&mut self, value: impl Into<Arc<str>>) {
        if let VectorData::String(vec) = &mut self.data {
            vec.push(value.into());
            self.len += 1;
        }
    }

    /// Pushes a node ID.
    pub fn push_node_id(&mut self, value: NodeId) {
        if let VectorData::NodeId(vec) = &mut self.data {
            vec.push(value);
            self.len += 1;
        }
    }

    /// Pushes an edge ID.
    pub fn push_edge_id(&mut self, value: EdgeId) {
        if let VectorData::EdgeId(vec) = &mut self.data {
            vec.push(value);
            self.len += 1;
        }
    }

    /// Pushes a generic value.
    pub fn push_value(&mut self, value: Value) {
        // Handle null values specially - push a default and mark as null
        if matches!(value, Value::Null) {
            match &mut self.data {
                VectorData::Bool(vec) => vec.push(false),
                VectorData::Int64(vec) => vec.push(0),
                VectorData::Float64(vec) => vec.push(0.0),
                VectorData::String(vec) => vec.push("".into()),
                VectorData::NodeId(vec) => vec.push(NodeId::new(0)),
                VectorData::EdgeId(vec) => vec.push(EdgeId::new(0)),
                VectorData::Generic(vec) => vec.push(Value::Null),
            }
            self.len += 1;
            self.set_null(self.len - 1);
            return;
        }

        match (&mut self.data, &value) {
            (VectorData::Bool(vec), Value::Bool(b)) => vec.push(*b),
            (VectorData::Int64(vec), Value::Int64(i)) => vec.push(*i),
            (VectorData::Float64(vec), Value::Float64(f)) => vec.push(*f),
            (VectorData::String(vec), Value::String(s)) => vec.push(s.clone()),
            // Handle Int64 -> NodeId conversion (from get_value roundtrip)
            (VectorData::NodeId(vec), Value::Int64(i)) => vec.push(NodeId::new(*i as u64)),
            // Handle Int64 -> EdgeId conversion (from get_value roundtrip)
            (VectorData::EdgeId(vec), Value::Int64(i)) => vec.push(EdgeId::new(*i as u64)),
            (VectorData::Generic(vec), _) => vec.push(value),
            _ => {
                // Type mismatch - push a default value to maintain vector alignment
                match &mut self.data {
                    VectorData::Bool(vec) => vec.push(false),
                    VectorData::Int64(vec) => vec.push(0),
                    VectorData::Float64(vec) => vec.push(0.0),
                    VectorData::String(vec) => vec.push("".into()),
                    VectorData::NodeId(vec) => vec.push(NodeId::new(0)),
                    VectorData::EdgeId(vec) => vec.push(EdgeId::new(0)),
                    VectorData::Generic(vec) => vec.push(value),
                }
            }
        }
        self.len += 1;
    }

    /// Gets a boolean value at index.
    #[must_use]
    pub fn get_bool(&self, index: usize) -> Option<bool> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Bool(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets an integer value at index.
    #[must_use]
    pub fn get_int64(&self, index: usize) -> Option<i64> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Int64(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets a float value at index.
    #[must_use]
    pub fn get_float64(&self, index: usize) -> Option<f64> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::Float64(vec) = &self.data {
            vec.get(index).copied()
        } else {
            None
        }
    }

    /// Gets a string value at index.
    #[must_use]
    pub fn get_string(&self, index: usize) -> Option<&str> {
        if self.is_null(index) {
            return None;
        }
        if let VectorData::String(vec) = &self.data {
            vec.get(index).map(|s| s.as_ref())
        } else {
            None
        }
    }

    /// Gets a node ID at index.
    #[must_use]
    pub fn get_node_id(&self, index: usize) -> Option<NodeId> {
        if self.is_null(index) {
            return None;
        }
        match &self.data {
            VectorData::NodeId(vec) => vec.get(index).copied(),
            // Handle Generic vectors that contain node IDs stored as Int64
            VectorData::Generic(vec) => match vec.get(index) {
                Some(Value::Int64(i)) => Some(NodeId::new(*i as u64)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Gets an edge ID at index.
    #[must_use]
    pub fn get_edge_id(&self, index: usize) -> Option<EdgeId> {
        if self.is_null(index) {
            return None;
        }
        match &self.data {
            VectorData::EdgeId(vec) => vec.get(index).copied(),
            // Handle Generic vectors that contain edge IDs stored as Int64
            VectorData::Generic(vec) => match vec.get(index) {
                Some(Value::Int64(i)) => Some(EdgeId::new(*i as u64)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Gets a value at index as a generic Value.
    #[must_use]
    pub fn get_value(&self, index: usize) -> Option<Value> {
        if self.is_null(index) {
            return Some(Value::Null);
        }

        match &self.data {
            VectorData::Bool(vec) => vec.get(index).map(|&v| Value::Bool(v)),
            VectorData::Int64(vec) => vec.get(index).map(|&v| Value::Int64(v)),
            VectorData::Float64(vec) => vec.get(index).map(|&v| Value::Float64(v)),
            VectorData::String(vec) => vec.get(index).map(|v| Value::String(v.clone())),
            VectorData::NodeId(vec) => vec.get(index).map(|&v| Value::Int64(v.as_u64() as i64)),
            VectorData::EdgeId(vec) => vec.get(index).map(|&v| Value::Int64(v.as_u64() as i64)),
            VectorData::Generic(vec) => vec.get(index).cloned(),
        }
    }

    /// Alias for get_value.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        self.get_value(index)
    }

    /// Alias for push_value.
    pub fn push(&mut self, value: Value) {
        self.push_value(value);
    }

    /// Returns a slice of the underlying boolean data.
    #[must_use]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if let VectorData::Bool(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying integer data.
    #[must_use]
    pub fn as_int64_slice(&self) -> Option<&[i64]> {
        if let VectorData::Int64(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying float data.
    #[must_use]
    pub fn as_float64_slice(&self) -> Option<&[f64]> {
        if let VectorData::Float64(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying node ID data.
    #[must_use]
    pub fn as_node_id_slice(&self) -> Option<&[NodeId]> {
        if let VectorData::NodeId(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns a slice of the underlying edge ID data.
    #[must_use]
    pub fn as_edge_id_slice(&self) -> Option<&[EdgeId]> {
        if let VectorData::EdgeId(vec) = &self.data {
            Some(vec)
        } else {
            None
        }
    }

    /// Returns the logical type of this vector.
    #[must_use]
    pub fn logical_type(&self) -> LogicalType {
        self.data_type.clone()
    }

    /// Copies a row from this vector to the destination vector.
    ///
    /// The destination vector should have a compatible type. The value at `row`
    /// is read from this vector and pushed to the destination vector.
    pub fn copy_row_to(&self, row: usize, dest: &mut ValueVector) {
        if self.is_null(row) {
            dest.push_value(Value::Null);
            return;
        }

        match &self.data {
            VectorData::Bool(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_bool(v);
                }
            }
            VectorData::Int64(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_int64(v);
                }
            }
            VectorData::Float64(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_float64(v);
                }
            }
            VectorData::String(vec) => {
                if let Some(v) = vec.get(row) {
                    dest.push_string(v.clone());
                }
            }
            VectorData::NodeId(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_node_id(v);
                }
            }
            VectorData::EdgeId(vec) => {
                if let Some(&v) = vec.get(row) {
                    dest.push_edge_id(v);
                }
            }
            VectorData::Generic(vec) => {
                if let Some(v) = vec.get(row) {
                    dest.push_value(v.clone());
                }
            }
        }
    }

    /// Clears all data from this vector.
    pub fn clear(&mut self) {
        match &mut self.data {
            VectorData::Bool(vec) => vec.clear(),
            VectorData::Int64(vec) => vec.clear(),
            VectorData::Float64(vec) => vec.clear(),
            VectorData::String(vec) => vec.clear(),
            VectorData::NodeId(vec) => vec.clear(),
            VectorData::EdgeId(vec) => vec.clear(),
            VectorData::Generic(vec) => vec.clear(),
        }
        self.len = 0;
        self.validity = None;
    }
}

impl Default for ValueVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int64_vector() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);

        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get_int64(0), Some(1));
        assert_eq!(vec.get_int64(1), Some(2));
        assert_eq!(vec.get_int64(2), Some(3));
    }

    #[test]
    fn test_string_vector() {
        let mut vec = ValueVector::with_type(LogicalType::String);

        vec.push_string("hello");
        vec.push_string("world");

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_string(0), Some("hello"));
        assert_eq!(vec.get_string(1), Some("world"));
    }

    #[test]
    fn test_null_values() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);

        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        assert!(!vec.is_null(1));
        vec.set_null(1);
        assert!(vec.is_null(1));

        assert_eq!(vec.get_int64(0), Some(1));
        assert_eq!(vec.get_int64(1), None); // Null
        assert_eq!(vec.get_int64(2), Some(3));
    }

    #[test]
    fn test_get_value() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(42);

        let value = vec.get_value(0);
        assert_eq!(value, Some(Value::Int64(42)));
    }

    #[test]
    fn test_slice_access() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(1);
        vec.push_int64(2);
        vec.push_int64(3);

        let slice = vec.as_int64_slice().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
    }

    #[test]
    fn test_clear() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(1);
        vec.push_int64(2);

        vec.clear();

        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
    }
}
