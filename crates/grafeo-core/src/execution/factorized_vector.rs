//! Factorized vector for avoiding Cartesian product materialization.
//!
//! A `FactorizedVector` can represent data in two states:
//! - **Flat**: One value per logical row (same as `ValueVector`)
//! - **Unflat**: Multiple values per parent row, with offset arrays tracking boundaries
//!
//! This enables multi-hop graph traversals without duplicating source columns.
//!
//! # Example
//!
//! For a 2-hop traversal where node A has neighbors `[B1, B2]` and node A' has neighbor `[B3]`:
//!
//! ```text
//! Flat representation (current):
//!   Row 0: (A, B1)
//!   Row 1: (A, B2)
//!   Row 2: (A', B3)
//!   -> 3 rows, A duplicated twice
//!
//! Factorized representation:
//!   Level 0 (flat): [A, A']           (2 values)
//!   Level 1 (unflat): [B1, B2, B3]    (3 values)
//!   Offsets: [0, 2, 3]                (A's neighbors at 0..2, A's at 2..3)
//!   -> 5 values total, no duplication
//! ```

use grafeo_common::types::{LogicalType, Value};

use super::vector::ValueVector;

/// A vector that can represent nested/repeated values without duplication.
///
/// In flat mode, this behaves like a regular `ValueVector` - one value per logical row.
/// In unflat mode, values are grouped by parent, with offset arrays tracking boundaries.
#[derive(Debug, Clone)]
pub struct FactorizedVector {
    /// The underlying data storage.
    data: ValueVector,
    /// The factorization state.
    state: FactorizedState,
}

/// The factorization state of a vector.
#[derive(Debug, Clone)]
pub enum FactorizedState {
    /// Flat: one value per logical row. `data.len() == logical_row_count`.
    Flat,
    /// Unflat: multiple values per logical row, grouped by parent.
    Unflat(UnflatMetadata),
}

/// Metadata for unflat vectors.
///
/// Tracks how values are grouped by their parent rows using an offset array.
#[derive(Debug, Clone)]
pub struct UnflatMetadata {
    /// Offset array: `offsets[i]` is the start index in data for parent `i`.
    /// Length is `parent_count + 1`, where the last element equals `data.len()`.
    offsets: Vec<u32>,
    /// Number of parent rows this vector is grouped by.
    parent_count: usize,
}

impl FactorizedVector {
    /// Creates a new flat factorized vector from a `ValueVector`.
    ///
    /// In flat mode, there's one value per logical row.
    #[must_use]
    pub fn flat(data: ValueVector) -> Self {
        Self {
            data,
            state: FactorizedState::Flat,
        }
    }

    /// Creates a new unflat factorized vector.
    ///
    /// # Arguments
    ///
    /// * `data` - The underlying values
    /// * `offsets` - Offset array where `offsets[i]` is the start index for parent `i`.
    ///   Must have length `parent_count + 1`.
    /// * `parent_count` - Number of parent rows
    ///
    /// # Panics
    ///
    /// Panics if `offsets.len() != parent_count + 1` or if offsets are invalid.
    #[must_use]
    pub fn unflat(data: ValueVector, offsets: Vec<u32>, parent_count: usize) -> Self {
        debug_assert_eq!(
            offsets.len(),
            parent_count + 1,
            "offsets must have length parent_count + 1"
        );
        debug_assert!(
            offsets.last().copied() == Some(data.len() as u32),
            "last offset must equal data length"
        );

        Self {
            data,
            state: FactorizedState::Unflat(UnflatMetadata {
                offsets,
                parent_count,
            }),
        }
    }

    /// Creates an empty flat factorized vector with the given type.
    #[must_use]
    pub fn empty(data_type: LogicalType) -> Self {
        Self::flat(ValueVector::with_type(data_type))
    }

    /// Returns true if this vector is in flat state.
    #[must_use]
    pub fn is_flat(&self) -> bool {
        matches!(self.state, FactorizedState::Flat)
    }

    /// Returns true if this vector is in unflat state.
    #[must_use]
    pub fn is_unflat(&self) -> bool {
        matches!(self.state, FactorizedState::Unflat(_))
    }

    /// Returns the number of physical values stored.
    ///
    /// This is the actual storage size, not the logical row count.
    #[must_use]
    pub fn physical_len(&self) -> usize {
        self.data.len()
    }

    /// Returns the logical type of the underlying data.
    #[must_use]
    pub fn data_type(&self) -> LogicalType {
        self.data.logical_type()
    }

    /// Returns a reference to the underlying `ValueVector`.
    #[must_use]
    pub fn data(&self) -> &ValueVector {
        &self.data
    }

    /// Returns a mutable reference to the underlying `ValueVector`.
    ///
    /// Use with caution - modifying the data directly may invalidate offsets.
    pub fn data_mut(&mut self) -> &mut ValueVector {
        &mut self.data
    }

    /// Returns the offset array for unflat vectors, or `None` for flat vectors.
    #[must_use]
    pub fn offsets(&self) -> Option<&[u32]> {
        match &self.state {
            FactorizedState::Flat => None,
            FactorizedState::Unflat(meta) => Some(&meta.offsets),
        }
    }

    /// Returns the parent count for unflat vectors.
    ///
    /// For flat vectors, returns the data length (each value is its own parent).
    #[must_use]
    pub fn parent_count(&self) -> usize {
        match &self.state {
            FactorizedState::Flat => self.data.len(),
            FactorizedState::Unflat(meta) => meta.parent_count,
        }
    }

    /// Gets the count of values for a specific parent index.
    ///
    /// For flat vectors, always returns 1.
    #[must_use]
    pub fn count_for_parent(&self, parent_idx: usize) -> usize {
        match &self.state {
            FactorizedState::Flat => 1,
            FactorizedState::Unflat(meta) => {
                if parent_idx >= meta.parent_count {
                    return 0;
                }
                let start = meta.offsets[parent_idx] as usize;
                let end = meta.offsets[parent_idx + 1] as usize;
                end - start
            }
        }
    }

    /// Gets the start and end indices for a specific parent.
    ///
    /// Returns `(start, end)` where values for this parent are at indices `start..end`.
    #[must_use]
    pub fn range_for_parent(&self, parent_idx: usize) -> (usize, usize) {
        match &self.state {
            FactorizedState::Flat => (parent_idx, parent_idx + 1),
            FactorizedState::Unflat(meta) => {
                if parent_idx >= meta.parent_count {
                    return (0, 0);
                }
                let start = meta.offsets[parent_idx] as usize;
                let end = meta.offsets[parent_idx + 1] as usize;
                (start, end)
            }
        }
    }

    /// Gets a value at a physical index.
    #[must_use]
    pub fn get_physical(&self, physical_idx: usize) -> Option<Value> {
        self.data.get_value(physical_idx)
    }

    /// Gets a NodeId at a physical index.
    #[must_use]
    pub fn get_node_id_physical(
        &self,
        physical_idx: usize,
    ) -> Option<grafeo_common::types::NodeId> {
        self.data.get_node_id(physical_idx)
    }

    /// Gets an EdgeId at a physical index.
    #[must_use]
    pub fn get_edge_id_physical(
        &self,
        physical_idx: usize,
    ) -> Option<grafeo_common::types::EdgeId> {
        self.data.get_edge_id(physical_idx)
    }

    /// Gets a value for a parent at a relative offset within that parent's values.
    ///
    /// For flat vectors, `relative_idx` must be 0.
    #[must_use]
    pub fn get_for_parent(&self, parent_idx: usize, relative_idx: usize) -> Option<Value> {
        let (start, end) = self.range_for_parent(parent_idx);
        let physical_idx = start + relative_idx;
        if physical_idx >= end {
            return None;
        }
        self.data.get_value(physical_idx)
    }

    /// Computes the total logical row count given parent multiplicities.
    ///
    /// For a flat vector at level 0, this is just `data.len()`.
    /// For an unflat vector, this is the sum of values for each parent,
    /// multiplied by that parent's multiplicity.
    ///
    /// # Arguments
    ///
    /// * `parent_multiplicities` - For each parent index, how many logical rows it represents.
    ///   If `None`, each parent represents 1 logical row.
    #[must_use]
    pub fn logical_row_count(&self, parent_multiplicities: Option<&[usize]>) -> usize {
        match &self.state {
            FactorizedState::Flat => self.data.len(),
            FactorizedState::Unflat(meta) => {
                let mut total = 0;
                for i in 0..meta.parent_count {
                    let count = self.count_for_parent(i);
                    let mult = parent_multiplicities.map_or(1, |m| m.get(i).copied().unwrap_or(1));
                    total += count * mult;
                }
                total
            }
        }
    }

    /// Flattens this vector to a regular `ValueVector`.
    ///
    /// For flat vectors, returns a clone of the underlying data.
    /// For unflat vectors, duplicates values according to parent multiplicities.
    ///
    /// # Arguments
    ///
    /// * `parent_multiplicities` - For each parent, how many times to repeat its values.
    ///   If `None`, each parent's values appear once.
    #[must_use]
    pub fn flatten(&self, parent_multiplicities: Option<&[usize]>) -> ValueVector {
        match &self.state {
            FactorizedState::Flat => {
                // If multiplicities are provided, we need to duplicate values
                if let Some(mults) = parent_multiplicities {
                    let capacity = mults.iter().sum();
                    let mut result = ValueVector::with_capacity(self.data.logical_type(), capacity);
                    for (i, &mult) in mults.iter().enumerate() {
                        if let Some(value) = self.data.get_value(i) {
                            for _ in 0..mult {
                                result.push_value(value.clone());
                            }
                        }
                    }
                    result
                } else {
                    self.data.clone()
                }
            }
            FactorizedState::Unflat(meta) => {
                let capacity = self.logical_row_count(parent_multiplicities);
                let mut result =
                    ValueVector::with_capacity(self.data.logical_type(), capacity.max(1));

                for parent_idx in 0..meta.parent_count {
                    let mult = parent_multiplicities
                        .map_or(1, |m| m.get(parent_idx).copied().unwrap_or(1));
                    let (start, end) = self.range_for_parent(parent_idx);

                    for _ in 0..mult {
                        for phys_idx in start..end {
                            if let Some(value) = self.data.get_value(phys_idx) {
                                result.push_value(value);
                            }
                        }
                    }
                }
                result
            }
        }
    }

    /// Creates an iterator over (parent_idx, physical_idx, value) tuples.
    pub fn iter_with_parent(&self) -> impl Iterator<Item = (usize, usize, Value)> + '_ {
        FactorizedVectorIter {
            vector: self,
            parent_idx: 0,
            physical_idx: 0,
        }
    }
}

/// Iterator over factorized vector values with parent indices.
struct FactorizedVectorIter<'a> {
    vector: &'a FactorizedVector,
    parent_idx: usize,
    physical_idx: usize,
}

impl Iterator for FactorizedVectorIter<'_> {
    type Item = (usize, usize, Value);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.vector.state {
            FactorizedState::Flat => {
                if self.physical_idx >= self.vector.data.len() {
                    return None;
                }
                let value = self.vector.data.get_value(self.physical_idx)?;
                let result = (self.physical_idx, self.physical_idx, value);
                self.physical_idx += 1;
                self.parent_idx += 1;
                Some(result)
            }
            FactorizedState::Unflat(meta) => {
                // Skip empty parents
                while self.parent_idx < meta.parent_count {
                    let (_start, end) = self.vector.range_for_parent(self.parent_idx);
                    if self.physical_idx < end {
                        let value = self.vector.data.get_value(self.physical_idx)?;
                        let result = (self.parent_idx, self.physical_idx, value);
                        self.physical_idx += 1;
                        return Some(result);
                    }
                    self.parent_idx += 1;
                    if self.parent_idx < meta.parent_count {
                        let (new_start, _) = self.vector.range_for_parent(self.parent_idx);
                        self.physical_idx = new_start;
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId};

    use super::*;

    #[test]
    fn test_flat_vector() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);
        data.push_int64(3);

        let vec = FactorizedVector::flat(data);

        assert!(vec.is_flat());
        assert!(!vec.is_unflat());
        assert_eq!(vec.physical_len(), 3);
        assert_eq!(vec.parent_count(), 3);
        assert_eq!(vec.count_for_parent(0), 1);
        assert_eq!(vec.count_for_parent(1), 1);
        assert_eq!(vec.count_for_parent(2), 1);
    }

    #[test]
    fn test_unflat_vector() {
        // Create neighbors: parent 0 has [10, 20], parent 1 has [30]
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3]; // parent 0: 0..2, parent 1: 2..3
        let vec = FactorizedVector::unflat(data, offsets, 2);

        assert!(!vec.is_flat());
        assert!(vec.is_unflat());
        assert_eq!(vec.physical_len(), 3);
        assert_eq!(vec.parent_count(), 2);
        assert_eq!(vec.count_for_parent(0), 2);
        assert_eq!(vec.count_for_parent(1), 1);
        assert_eq!(vec.range_for_parent(0), (0, 2));
        assert_eq!(vec.range_for_parent(1), (2, 3));
    }

    #[test]
    fn test_empty_vector() {
        let vec = FactorizedVector::empty(LogicalType::Int64);

        assert!(vec.is_flat());
        assert_eq!(vec.physical_len(), 0);
        assert_eq!(vec.data_type(), LogicalType::Int64);
    }

    #[test]
    fn test_data_type() {
        let mut data = ValueVector::with_type(LogicalType::String);
        data.push_string("hello");

        let vec = FactorizedVector::flat(data);
        assert_eq!(vec.data_type(), LogicalType::String);
    }

    #[test]
    fn test_data_access() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(42);

        let vec = FactorizedVector::flat(data);

        // Test immutable access
        assert_eq!(vec.data().len(), 1);
        assert_eq!(vec.data().get_int64(0), Some(42));
    }

    #[test]
    fn test_data_mut() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(42);

        let mut vec = FactorizedVector::flat(data);

        // Modify via mutable access
        vec.data_mut().push_int64(100);
        assert_eq!(vec.physical_len(), 2);
    }

    #[test]
    fn test_offsets_flat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);

        let vec = FactorizedVector::flat(data);
        assert!(vec.offsets().is_none());
    }

    #[test]
    fn test_offsets_unflat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets.clone(), 2);

        assert_eq!(vec.offsets(), Some(offsets.as_slice()));
    }

    #[test]
    fn test_get_physical() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.get_physical(0), Some(Value::Int64(10)));
        assert_eq!(vec.get_physical(1), Some(Value::Int64(20)));
        assert_eq!(vec.get_physical(2), None);
    }

    #[test]
    fn test_get_node_id_physical() {
        let mut data = ValueVector::with_type(LogicalType::Node);
        data.push_node_id(NodeId::new(100));
        data.push_node_id(NodeId::new(200));

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.get_node_id_physical(0), Some(NodeId::new(100)));
        assert_eq!(vec.get_node_id_physical(1), Some(NodeId::new(200)));
        assert_eq!(vec.get_node_id_physical(2), None);
    }

    #[test]
    fn test_get_edge_id_physical() {
        let mut data = ValueVector::with_type(LogicalType::Edge);
        data.push_edge_id(EdgeId::new(100));
        data.push_edge_id(EdgeId::new(200));

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.get_edge_id_physical(0), Some(EdgeId::new(100)));
        assert_eq!(vec.get_edge_id_physical(1), Some(EdgeId::new(200)));
        assert_eq!(vec.get_edge_id_physical(2), None);
    }

    #[test]
    fn test_range_for_parent_flat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);
        data.push_int64(3);

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.range_for_parent(0), (0, 1));
        assert_eq!(vec.range_for_parent(1), (1, 2));
        assert_eq!(vec.range_for_parent(2), (2, 3));
    }

    #[test]
    fn test_range_for_parent_out_of_bounds() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let offsets = vec![0, 2];
        let vec = FactorizedVector::unflat(data, offsets, 1);

        // Out of bounds should return (0, 0)
        assert_eq!(vec.range_for_parent(10), (0, 0));
    }

    #[test]
    fn test_count_for_parent_out_of_bounds() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let offsets = vec![0, 2];
        let vec = FactorizedVector::unflat(data, offsets, 1);

        // Out of bounds should return 0
        assert_eq!(vec.count_for_parent(10), 0);
    }

    #[test]
    fn test_get_for_parent() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        // Parent 0's values
        assert_eq!(vec.get_for_parent(0, 0), Some(Value::Int64(10)));
        assert_eq!(vec.get_for_parent(0, 1), Some(Value::Int64(20)));
        assert_eq!(vec.get_for_parent(0, 2), None); // Out of range

        // Parent 1's values
        assert_eq!(vec.get_for_parent(1, 0), Some(Value::Int64(30)));
        assert_eq!(vec.get_for_parent(1, 1), None); // Out of range
    }

    #[test]
    fn test_get_for_parent_flat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.get_for_parent(0, 0), Some(Value::Int64(10)));
        assert_eq!(vec.get_for_parent(1, 0), Some(Value::Int64(20)));
        assert_eq!(vec.get_for_parent(0, 1), None); // Only one value per parent in flat mode
    }

    #[test]
    fn test_flatten_flat_no_mults() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);
        data.push_int64(3);

        let vec = FactorizedVector::flat(data);
        let flat = vec.flatten(None);

        assert_eq!(flat.len(), 3);
        assert_eq!(flat.get_int64(0), Some(1));
        assert_eq!(flat.get_int64(2), Some(3));
    }

    #[test]
    fn test_flatten_flat_with_mults() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let vec = FactorizedVector::flat(data);

        // Duplicate first value 3 times, second value 2 times
        let mults = [3, 2];
        let flat = vec.flatten(Some(&mults));

        assert_eq!(flat.len(), 5);
        assert_eq!(flat.get_int64(0), Some(10));
        assert_eq!(flat.get_int64(1), Some(10));
        assert_eq!(flat.get_int64(2), Some(10));
        assert_eq!(flat.get_int64(3), Some(20));
        assert_eq!(flat.get_int64(4), Some(20));
    }

    #[test]
    fn test_flatten_unflat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        // Flatten without multiplicities
        let flat = vec.flatten(None);
        assert_eq!(flat.len(), 3);
        assert_eq!(flat.get_int64(0), Some(10));
        assert_eq!(flat.get_int64(1), Some(20));
        assert_eq!(flat.get_int64(2), Some(30));
    }

    #[test]
    fn test_flatten_with_multiplicities() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        // Parent 0 has mult 2, parent 1 has mult 1
        // So: [10,20], [10,20], [30]
        let mults = [2, 1];
        let flat = vec.flatten(Some(&mults));

        assert_eq!(flat.len(), 5);
        assert_eq!(flat.get_int64(0), Some(10));
        assert_eq!(flat.get_int64(1), Some(20));
        assert_eq!(flat.get_int64(2), Some(10));
        assert_eq!(flat.get_int64(3), Some(20));
        assert_eq!(flat.get_int64(4), Some(30));
    }

    #[test]
    fn test_logical_row_count_flat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);
        data.push_int64(3);

        let vec = FactorizedVector::flat(data);

        assert_eq!(vec.logical_row_count(None), 3);
    }

    #[test]
    fn test_logical_row_count() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        // Without multiplicities: 2 + 1 = 3
        assert_eq!(vec.logical_row_count(None), 3);

        // With multiplicities [2, 3]: 2*2 + 1*3 = 7
        let mults = [2, 3];
        assert_eq!(vec.logical_row_count(Some(&mults)), 7);
    }

    #[test]
    fn test_logical_row_count_missing_mult() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        // Only provide one multiplicity - second should default to 1
        let mults = [2];
        assert_eq!(vec.logical_row_count(Some(&mults)), 5); // 2*2 + 1*1 = 5
    }

    #[test]
    fn test_iter_with_parent_flat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);

        let vec = FactorizedVector::flat(data);
        let items: Vec<_> = vec.iter_with_parent().collect();

        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (0, 0, Value::Int64(1)));
        assert_eq!(items[1], (1, 1, Value::Int64(2)));
    }

    #[test]
    fn test_iter_with_parent_unflat() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);
        data.push_int64(30);

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);
        let items: Vec<_> = vec.iter_with_parent().collect();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (0, 0, Value::Int64(10)));
        assert_eq!(items[1], (0, 1, Value::Int64(20)));
        assert_eq!(items[2], (1, 2, Value::Int64(30)));
    }

    #[test]
    fn test_iter_with_parent_empty_parents() {
        // Parent 0 has [], parent 1 has [10, 20], parent 2 has []
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let offsets = vec![0, 0, 2, 2];
        let vec = FactorizedVector::unflat(data, offsets, 3);
        let items: Vec<_> = vec.iter_with_parent().collect();

        // Should skip empty parents
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (1, 0, Value::Int64(10)));
        assert_eq!(items[1], (1, 1, Value::Int64(20)));
    }

    #[test]
    fn test_iter_with_parent_empty() {
        let data = ValueVector::with_type(LogicalType::Int64);
        let vec = FactorizedVector::flat(data);
        let items: Vec<_> = vec.iter_with_parent().collect();

        assert!(items.is_empty());
    }

    #[test]
    fn test_empty_parents() {
        // Parent 0 has [], parent 1 has [10, 20], parent 2 has []
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(10);
        data.push_int64(20);

        let offsets = vec![0, 0, 2, 2]; // parent 0: empty, parent 1: 0..2, parent 2: empty
        let vec = FactorizedVector::unflat(data, offsets, 3);

        assert_eq!(vec.count_for_parent(0), 0);
        assert_eq!(vec.count_for_parent(1), 2);
        assert_eq!(vec.count_for_parent(2), 0);
        assert_eq!(vec.logical_row_count(None), 2);
    }

    #[test]
    fn test_node_id_vector() {
        let mut data = ValueVector::with_type(LogicalType::Node);
        data.push_node_id(NodeId::new(100));
        data.push_node_id(NodeId::new(200));
        data.push_node_id(NodeId::new(300));

        let offsets = vec![0, 2, 3];
        let vec = FactorizedVector::unflat(data, offsets, 2);

        assert_eq!(vec.count_for_parent(0), 2);
        assert_eq!(vec.count_for_parent(1), 1);

        // Check physical access
        let flat = vec.flatten(None);
        assert_eq!(flat.get_node_id(0), Some(NodeId::new(100)));
        assert_eq!(flat.get_node_id(1), Some(NodeId::new(200)));
        assert_eq!(flat.get_node_id(2), Some(NodeId::new(300)));
    }

    #[test]
    fn test_factorized_state_debug() {
        let state_flat = FactorizedState::Flat;
        assert!(format!("{state_flat:?}").contains("Flat"));

        let state_unflat = FactorizedState::Unflat(UnflatMetadata {
            offsets: vec![0, 1],
            parent_count: 1,
        });
        assert!(format!("{state_unflat:?}").contains("Unflat"));
    }

    #[test]
    fn test_unflat_metadata_debug() {
        let meta = UnflatMetadata {
            offsets: vec![0, 2, 5],
            parent_count: 2,
        };
        let debug = format!("{meta:?}");
        assert!(debug.contains("offsets"));
        assert!(debug.contains("parent_count"));
    }

    #[test]
    fn test_factorized_vector_clone() {
        let mut data = ValueVector::with_type(LogicalType::Int64);
        data.push_int64(1);
        data.push_int64(2);

        let vec = FactorizedVector::flat(data);
        let cloned = vec.clone();

        assert_eq!(vec.physical_len(), cloned.physical_len());
        assert_eq!(vec.is_flat(), cloned.is_flat());
    }
}
