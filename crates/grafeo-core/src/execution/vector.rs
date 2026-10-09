//! ValueVector for columnar data storage.

use arcstr::ArcStr;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};

/// Default vector capacity (tuples per vector).
pub const DEFAULT_VECTOR_CAPACITY: usize = 2048;

/// A columnar vector of values.
///
/// ValueVector stores data in columnar format for efficient SIMD processing
/// and cache utilization during query execution.
///
/// A vector of a typed column (bool, integer, float, string, node or edge)
/// keeps its values in typed storage. A value that does not fit that type
/// means a planner or an operator declared the wrong column type: debug
/// builds panic so tests find it, and release builds turn the vector generic
/// ([`LogicalType::Any`]), so every value is kept. A vector never writes a
/// default value (`0`, `''`, node 0) in place of one it cannot hold.
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
    /// Strings (stored as ArcStr for cheap cloning).
    String(Vec<ArcStr>),
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
            self.validity = Some(vec![true; index + 1]);
        }
        if let Some(validity) = &mut self.validity {
            if validity.len() <= index {
                validity.resize(index + 1, true);
            }
            validity[index] = false;
        }
    }

    /// Pushes a boolean value (see the type for a value that does not fit).
    pub fn push_bool(&mut self, value: bool) {
        match &mut self.data {
            VectorData::Bool(vec) => vec.push(value),
            VectorData::Generic(vec) => vec.push(Value::Bool(value)),
            _ => self.push_unfit(Value::Bool(value)),
        }
        self.len += 1;
    }

    /// Pushes an integer value (see the type for a value that does not fit).
    pub fn push_int64(&mut self, value: i64) {
        match &mut self.data {
            VectorData::Int64(vec) => vec.push(value),
            VectorData::Generic(vec) => vec.push(Value::Int64(value)),
            _ => self.push_unfit(Value::Int64(value)),
        }
        self.len += 1;
    }

    /// Pushes a float value (see the type for a value that does not fit).
    pub fn push_float64(&mut self, value: f64) {
        match &mut self.data {
            VectorData::Float64(vec) => vec.push(value),
            VectorData::Generic(vec) => vec.push(Value::Float64(value)),
            _ => self.push_unfit(Value::Float64(value)),
        }
        self.len += 1;
    }

    /// Pushes a string value (see the type for a value that does not fit).
    pub fn push_string(&mut self, value: impl Into<ArcStr>) {
        match &mut self.data {
            VectorData::String(vec) => vec.push(value.into()),
            VectorData::Generic(vec) => vec.push(Value::String(value.into())),
            _ => self.push_unfit(Value::String(value.into())),
        }
        self.len += 1;
    }

    /// Pushes a node ID (see the type for a value that does not fit).
    pub fn push_node_id(&mut self, value: NodeId) {
        // A generic vector keeps an ID as its bits in an i64, which
        // `get_node_id` casts back to the same u64.
        let id = Value::Int64(value.as_u64().cast_signed());
        match &mut self.data {
            VectorData::NodeId(vec) => vec.push(value),
            VectorData::Generic(vec) => vec.push(id),
            _ => self.push_unfit(id),
        }
        self.len += 1;
    }

    /// Pushes an edge ID (see the type for a value that does not fit).
    pub fn push_edge_id(&mut self, value: EdgeId) {
        // As in `push_node_id`: the ID's bits, which `get_edge_id` reads back.
        let id = Value::Int64(value.as_u64().cast_signed());
        match &mut self.data {
            VectorData::EdgeId(vec) => vec.push(value),
            VectorData::Generic(vec) => vec.push(id),
            _ => self.push_unfit(id),
        }
        self.len += 1;
    }

    /// Stores `value`, which does not fit this vector's typed storage, by
    /// turning the vector generic: every value so far is kept as a [`Value`]
    /// (nulls stay null) and the type becomes [`LogicalType::Any`]. The
    /// caller counts the row.
    ///
    /// # Panics
    ///
    /// In debug builds, always: a value that does not fit means a planner or
    /// an operator declared the wrong column type, and tests should find it.
    fn push_unfit(&mut self, value: Value) {
        debug_assert!(
            false,
            "a {} value does not fit a {:?} column: a planner or an operator declared the wrong column type",
            value.type_name(),
            self.data_type
        );
        let mut values: Vec<Value> = (0..self.len)
            .map(|row| self.get_value(row).unwrap_or(Value::Null))
            .collect();
        values.push(value);
        self.data = VectorData::Generic(values);
        self.data_type = LogicalType::Any;
    }

    /// Pushes a generic value (see the type for a value that does not fit).
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

        match (&mut self.data, value) {
            (VectorData::Bool(vec), Value::Bool(b)) => vec.push(b),
            (VectorData::Int64(vec), Value::Int64(i)) => vec.push(i),
            (VectorData::Float64(vec), Value::Float64(f)) => vec.push(f),
            (VectorData::String(vec), Value::String(s)) => vec.push(s),
            // Handle Int64 -> NodeId conversion (from get_value roundtrip)
            // reason: ID encoding: i64 <-> u64 round-trip
            #[allow(clippy::cast_sign_loss)]
            (VectorData::NodeId(vec), Value::Int64(i)) => vec.push(NodeId::new(i as u64)),
            // Handle Int64 -> EdgeId conversion (from get_value roundtrip)
            // reason: ID encoding: i64 <-> u64 round-trip
            #[allow(clippy::cast_sign_loss)]
            (VectorData::EdgeId(vec), Value::Int64(i)) => vec.push(EdgeId::new(i as u64)),
            (VectorData::Generic(vec), value) => vec.push(value),
            (_, value) => self.push_unfit(value),
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
                // reason: ID encoding: i64 <-> u64 round-trip
                #[allow(clippy::cast_sign_loss)]
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
                // reason: ID encoding: i64 <-> u64 round-trip
                #[allow(clippy::cast_sign_loss)]
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
            // reason: entity IDs stored as i64, standard encoding
            VectorData::NodeId(vec) => vec.get(index).map(|&v| {
                // reason: entity IDs are sequential counters, well within i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v.as_u64() as i64);
                val
            }),
            // reason: entity IDs stored as i64, standard encoding
            // reason: entity IDs are sequential counters, well within i64::MAX
            VectorData::EdgeId(vec) => vec.get(index).map(|&v| {
                // reason: entity IDs are sequential counters, well within i64::MAX
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v.as_u64() as i64);
                val
            }),
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
    /// is read from this vector and pushed to the destination vector (see the
    /// type for a value that does not fit the destination).
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

    /// Typed push methods fall back to VectorData::Generic when the vector
    /// was created with LogicalType::Any. This exercises the safety-net arms
    /// added to prevent silent data loss on type mismatch.
    #[test]
    fn test_generic_fallback_push_int64() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_int64(42);
        vec.push_int64(-7);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Int64(42)));
        assert_eq!(vec.get_value(1), Some(Value::Int64(-7)));
    }

    #[test]
    fn test_generic_fallback_push_bool() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_bool(true);
        vec.push_bool(false);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Bool(true)));
        assert_eq!(vec.get_value(1), Some(Value::Bool(false)));
    }

    #[test]
    fn test_generic_fallback_push_float64() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_float64(1.23);
        vec.push_float64(-0.5);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::Float64(1.23)));
        assert_eq!(vec.get_value(1), Some(Value::Float64(-0.5)));
    }

    #[test]
    fn test_generic_fallback_push_string() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_string("hello");
        vec.push_string("world");
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_value(0), Some(Value::String("hello".into())));
        assert_eq!(vec.get_value(1), Some(Value::String("world".into())));
    }

    /// Mixed typed pushes into a Generic vector preserve each value's type.
    #[test]
    fn test_generic_fallback_mixed_types() {
        let mut vec = ValueVector::with_type(LogicalType::Any);
        vec.push_int64(1);
        vec.push_string("two");
        vec.push_bool(true);
        vec.push_float64(99.5);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get_value(0), Some(Value::Int64(1)));
        assert_eq!(vec.get_value(1), Some(Value::String("two".into())));
        assert_eq!(vec.get_value(2), Some(Value::Bool(true)));
        assert_eq!(vec.get_value(3), Some(Value::Float64(99.5)));
    }

    // A value that does not fit the vector's type means a planner or an
    // operator declared the wrong column type. Debug builds (and so every
    // test run without --release) panic to find it; release builds keep the
    // value in a generic vector. It used to write `''`, `0`, `0.0`, `false`
    // or node or edge 0 in its place (a typed push wrote nothing, so the
    // column fell out of step with the others).

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit"))]
    fn a_list_in_a_string_column_is_kept_not_written_as_an_empty_string() {
        let mut vec = ValueVector::with_type(LogicalType::String);
        vec.push_value(Value::String("Alix".into()));
        vec.push_value(Value::Null);
        let list = Value::List(vec![Value::Int64(3), Value::Int64(19)].into());
        vec.push_value(list.clone());
        vec.push_value(Value::String("Gus".into()));
        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get_value(0), Some(Value::String("Alix".into())));
        assert_eq!(vec.get_value(1), Some(Value::Null), "a null stays null");
        assert_eq!(vec.get_value(2), Some(list));
        assert_eq!(vec.get_value(3), Some(Value::String("Gus".into())));
        assert_eq!(
            vec.data_type(),
            &LogicalType::Any,
            "the vector says it holds values of any type"
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit"))]
    fn a_string_in_a_node_column_is_kept_not_read_as_node_zero() {
        let mut vec = ValueVector::with_type(LogicalType::Node);
        vec.push_node_id(NodeId::new(3));
        vec.push_value(Value::String("Amsterdam".into()));
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get_node_id(0), Some(NodeId::new(3)));
        assert_eq!(vec.get_value(1), Some(Value::String("Amsterdam".into())));
        assert_eq!(vec.get_node_id(1), None, "a string is no node");
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit"))]
    fn a_float_in_an_integer_column_is_kept_not_written_as_zero() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(19);
        vec.push_value(Value::Float64(88.5));
        assert_eq!(vec.get_value(0), Some(Value::Int64(19)));
        assert_eq!(vec.get_value(1), Some(Value::Float64(88.5)));
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit"))]
    fn a_typed_push_that_does_not_fit_keeps_the_column_in_step() {
        let mut vec = ValueVector::with_type(LogicalType::Int64);
        vec.push_int64(3);
        vec.push_string("Mia");
        vec.push_bool(true);
        assert_eq!(vec.len(), 3, "every push adds a row");
        assert_eq!(vec.get_value(0), Some(Value::Int64(3)));
        assert_eq!(vec.get_value(1), Some(Value::String("Mia".into())));
        assert_eq!(vec.get_value(2), Some(Value::Bool(true)));
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit"))]
    fn a_row_copied_into_a_column_of_another_type_is_kept() {
        let mut source = ValueVector::with_type(LogicalType::String);
        source.push_string("Vincent");
        let mut destination = ValueVector::with_type(LogicalType::Float64);
        destination.push_float64(3.5);
        source.copy_row_to(0, &mut destination);
        assert_eq!(destination.len(), 2);
        assert_eq!(destination.get_value(0), Some(Value::Float64(3.5)));
        assert_eq!(
            destination.get_value(1),
            Some(Value::String("Vincent".into()))
        );
    }

    #[test]
    fn values_that_fit_keep_the_typed_storage() {
        let mut vec = ValueVector::with_type(LogicalType::Node);
        vec.push_value(Value::Int64(19));
        vec.push_value(Value::Null);
        vec.push_node_id(NodeId::new(88));
        assert_eq!(vec.data_type(), &LogicalType::Node);
        assert_eq!(vec.as_node_id_slice().map(<[NodeId]>::len), Some(3));
        assert_eq!(vec.get_node_id(0), Some(NodeId::new(19)));
        assert!(vec.is_null(1));
        assert_eq!(vec.get_node_id(2), Some(NodeId::new(88)));
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
