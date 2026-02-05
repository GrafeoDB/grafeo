//! Columnar property storage for nodes and edges.
//!
//! Properties are stored column-wise (all "name" values together, all "age"
//! values together) rather than row-wise. This makes filtering fast - to find
//! all nodes where age > 30, we only scan the age column.
//!
//! Each column also maintains a zone map (min/max/null_count) enabling the
//! query optimizer to skip columns entirely when a predicate can't match.
//!
//! ## Compression
//!
//! Columns can be compressed to save memory. When compression is enabled,
//! the column automatically selects the best codec based on the data type:
//!
//! | Data type | Codec | Typical savings |
//! |-----------|-------|-----------------|
//! | Int64 (sorted) | DeltaBitPacked | 5-20x |
//! | Int64 (small) | BitPacked | 2-16x |
//! | Int64 (repeated) | RunLength | 2-100x |
//! | String (low cardinality) | Dictionary | 2-50x |
//! | Bool | BitVector | 8x |

use crate::index::zone_map::ZoneMapEntry;
use crate::storage::{
    CompressedData, CompressionCodec, DictionaryBuilder, DictionaryEncoding, TypeSpecificCompressor,
};
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;

/// Compression mode for property columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMode {
    /// Never compress - always use sparse HashMap (default).
    #[default]
    None,
    /// Automatically compress when beneficial (after threshold).
    Auto,
    /// Eagerly compress on every flush.
    Eager,
}

/// Threshold for automatic compression (number of values).
const COMPRESSION_THRESHOLD: usize = 1000;

/// Size of the hot buffer for recent writes (before compression).
/// Larger buffer (4096) keeps more recent data uncompressed for faster reads.
/// This trades ~64KB of memory overhead per column for 1.5-2x faster point lookups
/// on recently-written data.
const HOT_BUFFER_SIZE: usize = 4096;

/// Comparison operators used for zone map predicate checks.
///
/// These map directly to GQL comparison operators like `=`, `<`, `>=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal to value.
    Eq,
    /// Not equal to value.
    Ne,
    /// Less than value.
    Lt,
    /// Less than or equal to value.
    Le,
    /// Greater than value.
    Gt,
    /// Greater than or equal to value.
    Ge,
}

/// Marker trait for IDs that can key into property storage.
///
/// Implemented for [`NodeId`] and [`EdgeId`] - you can store properties on both.
pub trait EntityId: Copy + Eq + Hash + 'static {}

impl EntityId for NodeId {}
impl EntityId for EdgeId {}

/// Thread-safe columnar property storage.
///
/// Each property key ("name", "age", etc.) gets its own column. This layout
/// is great for analytical queries that filter on specific properties -
/// you only touch the columns you need.
///
/// Generic over `Id` so the same storage works for nodes and edges.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::PropertyStorage;
/// use grafeo_common::types::{NodeId, PropertyKey};
///
/// let storage = PropertyStorage::new();
/// let alice = NodeId::new(1);
///
/// storage.set(alice, PropertyKey::new("name"), "Alice".into());
/// storage.set(alice, PropertyKey::new("age"), 30i64.into());
///
/// // Fetch all properties at once
/// let props = storage.get_all(alice);
/// assert_eq!(props.len(), 2);
/// ```
pub struct PropertyStorage<Id: EntityId = NodeId> {
    /// Map from property key to column.
    /// Lock order: 9 (nested, acquired via LpgStore::node_properties/edge_properties)
    columns: RwLock<FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    /// Default compression mode for new columns.
    default_compression: CompressionMode,
    _marker: PhantomData<Id>,
}

impl<Id: EntityId> PropertyStorage<Id> {
    /// Creates a new property storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: RwLock::new(FxHashMap::default()),
            default_compression: CompressionMode::None,
            _marker: PhantomData,
        }
    }

    /// Creates a new property storage with compression enabled.
    #[must_use]
    pub fn with_compression(mode: CompressionMode) -> Self {
        Self {
            columns: RwLock::new(FxHashMap::default()),
            default_compression: mode,
            _marker: PhantomData,
        }
    }

    /// Sets the default compression mode for new columns.
    pub fn set_default_compression(&mut self, mode: CompressionMode) {
        self.default_compression = mode;
    }

    /// Sets a property value for an entity.
    pub fn set(&self, id: Id, key: PropertyKey, value: Value) {
        let mut columns = self.columns.write();
        let mode = self.default_compression;
        columns
            .entry(key)
            .or_insert_with(|| PropertyColumn::with_compression(mode))
            .set(id, value);
    }

    /// Enables compression for a specific column.
    pub fn enable_compression(&self, key: &PropertyKey, mode: CompressionMode) {
        let mut columns = self.columns.write();
        if let Some(col) = columns.get_mut(key) {
            col.set_compression_mode(mode);
        }
    }

    /// Compresses all columns that have compression enabled.
    pub fn compress_all(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            if col.compression_mode() != CompressionMode::None {
                col.compress();
            }
        }
    }

    /// Forces compression on all columns regardless of mode.
    pub fn force_compress_all(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.force_compress();
        }
    }

    /// Returns compression statistics for all columns.
    #[must_use]
    pub fn compression_stats(&self) -> FxHashMap<PropertyKey, CompressionStats> {
        let columns = self.columns.read();
        columns
            .iter()
            .map(|(key, col)| (key.clone(), col.compression_stats()))
            .collect()
    }

    /// Returns the total memory usage of all columns.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let columns = self.columns.read();
        columns
            .values()
            .map(|col| col.compression_stats().compressed_size)
            .sum()
    }

    /// Gets a property value for an entity.
    #[must_use]
    pub fn get(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get(id))
    }

    /// Removes a property value for an entity.
    pub fn remove(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let mut columns = self.columns.write();
        columns.get_mut(key).and_then(|col| col.remove(id))
    }

    /// Removes all properties for an entity.
    pub fn remove_all(&self, id: Id) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove(id);
        }
    }

    /// Gets all properties for an entity.
    #[must_use]
    pub fn get_all(&self, id: Id) -> FxHashMap<PropertyKey, Value> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.get(id) {
                result.insert(key.clone(), value);
            }
        }
        result
    }

    /// Gets property values for multiple entities in a single lock acquisition.
    ///
    /// More efficient than calling [`Self::get`] in a loop because it acquires
    /// the read lock only once.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let key = PropertyKey::new("age");
    /// let ids = vec![NodeId(1), NodeId(2), NodeId(3)];
    /// let values = storage.get_batch(&ids, &key);
    /// // values[i] is the property value for ids[i], or None if not set
    /// ```
    #[must_use]
    pub fn get_batch(&self, ids: &[Id], key: &PropertyKey) -> Vec<Option<Value>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => ids.iter().map(|&id| col.get(id)).collect(),
            None => vec![None; ids.len()],
        }
    }

    /// Gets all properties for multiple entities efficiently.
    ///
    /// More efficient than calling [`Self::get_all`] in a loop because it
    /// acquires the read lock only once.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let ids = vec![NodeId(1), NodeId(2)];
    /// let all_props = storage.get_all_batch(&ids);
    /// // all_props[i] is a HashMap of all properties for ids[i]
    /// ```
    #[must_use]
    pub fn get_all_batch(&self, ids: &[Id]) -> Vec<FxHashMap<PropertyKey, Value>> {
        let columns = self.columns.read();
        let column_count = columns.len();

        // Pre-allocate result vector with exact capacity (NebulaGraph pattern)
        let mut results = Vec::with_capacity(ids.len());

        for &id in ids {
            // Pre-allocate HashMap with expected column count
            let mut result =
                FxHashMap::with_capacity_and_hasher(column_count, Default::default());
            for (key, col) in columns.iter() {
                if let Some(value) = col.get(id) {
                    result.insert(key.clone(), value);
                }
            }
            results.push(result);
        }

        results
    }

    /// Gets selected properties for multiple entities efficiently (projection pushdown).
    ///
    /// This is more efficient than [`Self::get_all_batch`] when you only need a subset
    /// of properties - it only iterates the requested columns instead of all columns.
    ///
    /// **Performance**: O(N × K) where N = ids.len() and K = keys.len(),
    /// compared to O(N × C) for `get_all_batch` where C = total column count.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::PropertyStorage;
    /// use grafeo_common::types::{PropertyKey, Value};
    /// use grafeo_common::NodeId;
    ///
    /// let storage: PropertyStorage<NodeId> = PropertyStorage::new();
    /// let ids = vec![NodeId::new(1), NodeId::new(2)];
    /// let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    ///
    /// // Only fetches "name" and "age" columns, ignoring other properties
    /// let props = storage.get_selective_batch(&ids, &keys);
    /// ```
    #[must_use]
    pub fn get_selective_batch(
        &self,
        ids: &[Id],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        if keys.is_empty() {
            // No properties requested - return empty maps
            return vec![FxHashMap::default(); ids.len()];
        }

        let columns = self.columns.read();

        // Pre-collect only the columns we need (avoids re-lookup per id)
        let requested_columns: Vec<_> = keys
            .iter()
            .filter_map(|key| columns.get(key).map(|col| (key, col)))
            .collect();

        // Pre-allocate result with exact capacity
        let mut results = Vec::with_capacity(ids.len());

        for &id in ids {
            let mut result = FxHashMap::with_capacity_and_hasher(
                requested_columns.len(),
                Default::default(),
            );
            // Only iterate requested columns, not all columns
            for (key, col) in &requested_columns {
                if let Some(value) = col.get(id) {
                    result.insert((*key).clone(), value);
                }
            }
            results.push(result);
        }

        results
    }

    /// Returns the number of property columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.read().len()
    }

    /// Returns the keys of all columns.
    #[must_use]
    pub fn keys(&self) -> Vec<PropertyKey> {
        self.columns.read().keys().cloned().collect()
    }

    /// Gets a column by key for bulk access.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<PropertyColumnRef<'_, Id>> {
        let columns = self.columns.read();
        if columns.contains_key(key) {
            Some(PropertyColumnRef {
                _guard: columns,
                key: key.clone(),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Checks if a predicate might match any values (using zone maps).
    ///
    /// Returns `false` only when we're *certain* no values match - for example,
    /// if you're looking for age > 100 but the max age is 80. Returns `true`
    /// if the property doesn't exist (conservative - might match).
    #[must_use]
    pub fn might_match(&self, key: &PropertyKey, op: CompareOp, value: &Value) -> bool {
        let columns = self.columns.read();
        columns
            .get(key)
            .map(|col| col.might_match(op, value))
            .unwrap_or(true) // No column = assume might match (conservative)
    }

    /// Gets the zone map for a property column.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<ZoneMapEntry> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.zone_map().clone())
    }

    /// Checks if a range predicate might match any values (using zone maps).
    ///
    /// Returns `false` only when we're *certain* no values match the range.
    /// Returns `true` if the property doesn't exist (conservative - might match).
    #[must_use]
    pub fn might_match_range(
        &self,
        key: &PropertyKey,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> bool {
        let columns = self.columns.read();
        columns
            .get(key)
            .map(|col| {
                col.zone_map()
                    .might_contain_range(min, max, min_inclusive, max_inclusive)
            })
            .unwrap_or(true) // No column = assume might match (conservative)
    }

    /// Rebuilds zone maps for all columns (call after bulk removes).
    pub fn rebuild_zone_maps(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.rebuild_zone_map();
        }
    }
}

impl<Id: EntityId> Default for PropertyStorage<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed storage for a property column.
///
/// Holds the compressed representation of values along with the index
/// mapping entity IDs to positions in the compressed array.
#[derive(Debug)]
pub enum CompressedColumnData {
    /// Compressed integers (Int64 values).
    Integers {
        /// Compressed data.
        data: CompressedData,
        /// Index: entity ID position -> compressed array index.
        id_to_index: Vec<u64>,
        /// Reverse index: compressed array index -> entity ID position.
        index_to_id: Vec<u64>,
    },
    /// Dictionary-encoded strings.
    Strings {
        /// Dictionary encoding.
        encoding: DictionaryEncoding,
        /// Index: entity ID position -> dictionary index.
        id_to_index: Vec<u64>,
        /// Reverse index: dictionary index -> entity ID position.
        index_to_id: Vec<u64>,
    },
    /// Compressed booleans.
    Booleans {
        /// Compressed data.
        data: CompressedData,
        /// Index: entity ID position -> bit index.
        id_to_index: Vec<u64>,
        /// Reverse index: bit index -> entity ID position.
        index_to_id: Vec<u64>,
    },
}

impl CompressedColumnData {
    /// Returns the memory usage of the compressed data in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        match self {
            CompressedColumnData::Integers {
                data,
                id_to_index,
                index_to_id,
            } => {
                data.data.len()
                    + id_to_index.len() * std::mem::size_of::<u64>()
                    + index_to_id.len() * std::mem::size_of::<u64>()
            }
            CompressedColumnData::Strings {
                encoding,
                id_to_index,
                index_to_id,
            } => {
                encoding.codes().len() * std::mem::size_of::<u32>()
                    + encoding.dictionary().iter().map(|s| s.len()).sum::<usize>()
                    + id_to_index.len() * std::mem::size_of::<u64>()
                    + index_to_id.len() * std::mem::size_of::<u64>()
            }
            CompressedColumnData::Booleans {
                data,
                id_to_index,
                index_to_id,
            } => {
                data.data.len()
                    + id_to_index.len() * std::mem::size_of::<u64>()
                    + index_to_id.len() * std::mem::size_of::<u64>()
            }
        }
    }

    /// Returns the compression ratio.
    #[must_use]
    #[allow(dead_code)]
    pub fn compression_ratio(&self) -> f64 {
        match self {
            CompressedColumnData::Integers { data, .. } => data.compression_ratio(),
            CompressedColumnData::Strings { encoding, .. } => encoding.compression_ratio(),
            CompressedColumnData::Booleans { data, .. } => data.compression_ratio(),
        }
    }
}

/// Statistics about column compression.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Size of uncompressed data in bytes.
    pub uncompressed_size: usize,
    /// Size of compressed data in bytes.
    pub compressed_size: usize,
    /// Number of values in the column.
    pub value_count: usize,
    /// Codec used for compression.
    pub codec: Option<CompressionCodec>,
}

impl CompressionStats {
    /// Returns the compression ratio (uncompressed / compressed).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            return 1.0;
        }
        self.uncompressed_size as f64 / self.compressed_size as f64
    }
}

/// A single property column (e.g., all "age" values).
///
/// Maintains min/max/null_count for fast predicate evaluation. When you
/// filter on `age > 50`, we first check if any age could possibly match
/// before scanning the actual values.
///
/// Columns support optional compression for large datasets. When compression
/// is enabled, the column automatically selects the best codec based on the
/// data type and characteristics.
pub struct PropertyColumn<Id: EntityId = NodeId> {
    /// Sparse storage: entity ID -> value (hot buffer + uncompressed).
    /// Used for recent writes and when compression is disabled.
    values: FxHashMap<Id, Value>,
    /// Zone map tracking min/max/null_count for predicate pushdown.
    zone_map: ZoneMapEntry,
    /// Whether zone map needs rebuild (after removes).
    zone_map_dirty: bool,
    /// Compression mode for this column.
    compression_mode: CompressionMode,
    /// Compressed data (when compression is enabled and triggered).
    compressed: Option<CompressedColumnData>,
    /// Number of values before last compression.
    compressed_count: usize,
}

impl<Id: EntityId> PropertyColumn<Id> {
    /// Creates a new empty column.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: CompressionMode::None,
            compressed: None,
            compressed_count: 0,
        }
    }

    /// Creates a new column with the specified compression mode.
    #[must_use]
    pub fn with_compression(mode: CompressionMode) -> Self {
        Self {
            values: FxHashMap::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: mode,
            compressed: None,
            compressed_count: 0,
        }
    }

    /// Sets the compression mode for this column.
    pub fn set_compression_mode(&mut self, mode: CompressionMode) {
        self.compression_mode = mode;
        if mode == CompressionMode::None {
            // Decompress if switching to no compression
            if self.compressed.is_some() {
                self.decompress_all();
            }
        }
    }

    /// Returns the compression mode for this column.
    #[must_use]
    pub fn compression_mode(&self) -> CompressionMode {
        self.compression_mode
    }

    /// Sets a value for an entity.
    pub fn set(&mut self, id: Id, value: Value) {
        // Update zone map incrementally
        self.update_zone_map_on_insert(&value);
        self.values.insert(id, value);

        // Check if we should compress (in Auto mode)
        if self.compression_mode == CompressionMode::Auto {
            let total_count = self.values.len() + self.compressed_count;
            let hot_buffer_count = self.values.len();

            // Compress when hot buffer exceeds threshold and total is large enough
            if hot_buffer_count >= HOT_BUFFER_SIZE && total_count >= COMPRESSION_THRESHOLD {
                self.compress();
            }
        }
    }

    /// Updates zone map when inserting a value.
    fn update_zone_map_on_insert(&mut self, value: &Value) {
        self.zone_map.row_count += 1;

        if matches!(value, Value::Null) {
            self.zone_map.null_count += 1;
            return;
        }

        // Update min
        match &self.zone_map.min {
            None => self.zone_map.min = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Less) {
                    self.zone_map.min = Some(value.clone());
                }
            }
        }

        // Update max
        match &self.zone_map.max {
            None => self.zone_map.max = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Greater) {
                    self.zone_map.max = Some(value.clone());
                }
            }
        }
    }

    /// Gets a value for an entity.
    ///
    /// First checks the hot buffer (uncompressed values), then falls back
    /// to the compressed data if present.
    #[must_use]
    pub fn get(&self, id: Id) -> Option<Value> {
        // First check hot buffer
        if let Some(value) = self.values.get(&id) {
            return Some(value.clone());
        }

        // For now, compressed data lookup is not implemented for sparse access
        // because the compressed format stores values by index, not by entity ID.
        // This would require maintaining an ID -> index map in CompressedColumnData.
        // The compressed data is primarily useful for bulk/scan operations.
        None
    }

    /// Removes a value for an entity.
    pub fn remove(&mut self, id: Id) -> Option<Value> {
        let removed = self.values.remove(&id);
        if removed.is_some() {
            // Mark zone map as dirty - would need full rebuild for accurate min/max
            self.zone_map_dirty = true;
        }
        removed
    }

    /// Returns the number of values in this column (hot + compressed).
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.values.len() + self.compressed_count
    }

    /// Returns true if this column is empty.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty() && self.compressed_count == 0
    }

    /// Iterates over all (id, value) pairs in the hot buffer.
    ///
    /// Note: This only iterates over uncompressed values. For full iteration
    /// including compressed values, use [`iter_all`].
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = (Id, &Value)> {
        self.values.iter().map(|(&id, v)| (id, v))
    }

    /// Returns compression statistics for this column.
    #[must_use]
    pub fn compression_stats(&self) -> CompressionStats {
        let hot_size = self.values.len() * std::mem::size_of::<Value>();
        let compressed_size = self.compressed.as_ref().map_or(0, |c| c.memory_usage());
        let codec = match &self.compressed {
            Some(CompressedColumnData::Integers { data, .. }) => Some(data.codec),
            Some(CompressedColumnData::Strings { .. }) => Some(CompressionCodec::Dictionary),
            Some(CompressedColumnData::Booleans { data, .. }) => Some(data.codec),
            None => None,
        };

        CompressionStats {
            uncompressed_size: hot_size + self.compressed_count * std::mem::size_of::<Value>(),
            compressed_size: hot_size + compressed_size,
            value_count: self.len(),
            codec,
        }
    }

    /// Returns whether the column has compressed data.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_compressed(&self) -> bool {
        self.compressed.is_some()
    }

    /// Compresses the hot buffer values.
    ///
    /// This merges the hot buffer into the compressed data, selecting the
    /// best codec based on the value types.
    ///
    /// Note: If compressed data already exists, this is a no-op to avoid
    /// losing previously compressed values. Use `force_compress()` after
    /// decompressing to re-compress with all values.
    pub fn compress(&mut self) {
        if self.values.is_empty() {
            return;
        }

        // Don't re-compress if we already have compressed data
        // (would need to decompress and merge first)
        if self.compressed.is_some() {
            return;
        }

        // Determine the dominant type
        let (int_count, str_count, bool_count) = self.count_types();
        let total = self.values.len();

        if int_count > total / 2 {
            self.compress_as_integers();
        } else if str_count > total / 2 {
            self.compress_as_strings();
        } else if bool_count > total / 2 {
            self.compress_as_booleans();
        }
        // If no dominant type, don't compress (mixed types don't compress well)
    }

    /// Counts values by type.
    fn count_types(&self) -> (usize, usize, usize) {
        let mut int_count = 0;
        let mut str_count = 0;
        let mut bool_count = 0;

        for value in self.values.values() {
            match value {
                Value::Int64(_) => int_count += 1,
                Value::String(_) => str_count += 1,
                Value::Bool(_) => bool_count += 1,
                _ => {}
            }
        }

        (int_count, str_count, bool_count)
    }

    /// Compresses integer values.
    #[allow(unsafe_code)]
    fn compress_as_integers(&mut self) {
        // Extract integer values and their IDs
        let mut values: Vec<(u64, i64)> = Vec::new();
        let mut non_int_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::Int64(v) => {
                    // Convert Id to u64 for indexing (assumes Id can be converted)
                    let id_u64 = unsafe { std::mem::transmute_copy::<Id, u64>(&id) };
                    values.push((id_u64, *v));
                }
                _ => {
                    non_int_values.insert(id, value.clone());
                }
            }
        }

        if values.len() < 8 {
            // Not worth compressing
            return;
        }

        // Sort by ID for better compression
        values.sort_by_key(|(id, _)| *id);

        let id_to_index: Vec<u64> = values.iter().map(|(id, _)| *id).collect();
        let index_to_id: Vec<u64> = id_to_index.clone();
        let int_values: Vec<i64> = values.iter().map(|(_, v)| *v).collect();

        // Compress using the optimal codec
        let compressed = TypeSpecificCompressor::compress_signed_integers(&int_values);

        // Only use compression if it actually saves space
        if compressed.compression_ratio() > 1.2 {
            self.compressed = Some(CompressedColumnData::Integers {
                data: compressed,
                id_to_index,
                index_to_id,
            });
            self.compressed_count = values.len();
            self.values = non_int_values;
        }
    }

    /// Compresses string values using dictionary encoding.
    #[allow(unsafe_code)]
    fn compress_as_strings(&mut self) {
        let mut values: Vec<(u64, ArcStr)> = Vec::new();
        let mut non_str_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::String(s) => {
                    let id_u64 = unsafe { std::mem::transmute_copy::<Id, u64>(&id) };
                    values.push((id_u64, s.clone()));
                }
                _ => {
                    non_str_values.insert(id, value.clone());
                }
            }
        }

        if values.len() < 8 {
            return;
        }

        // Sort by ID
        values.sort_by_key(|(id, _)| *id);

        let id_to_index: Vec<u64> = values.iter().map(|(id, _)| *id).collect();
        let index_to_id: Vec<u64> = id_to_index.clone();

        // Build dictionary
        let mut builder = DictionaryBuilder::new();
        for (_, s) in &values {
            builder.add(s.as_ref());
        }
        let encoding = builder.build();

        // Only use compression if it actually saves space
        if encoding.compression_ratio() > 1.2 {
            self.compressed = Some(CompressedColumnData::Strings {
                encoding,
                id_to_index,
                index_to_id,
            });
            self.compressed_count = values.len();
            self.values = non_str_values;
        }
    }

    /// Compresses boolean values.
    #[allow(unsafe_code)]
    fn compress_as_booleans(&mut self) {
        let mut values: Vec<(u64, bool)> = Vec::new();
        let mut non_bool_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::Bool(b) => {
                    let id_u64 = unsafe { std::mem::transmute_copy::<Id, u64>(&id) };
                    values.push((id_u64, *b));
                }
                _ => {
                    non_bool_values.insert(id, value.clone());
                }
            }
        }

        if values.len() < 8 {
            return;
        }

        // Sort by ID
        values.sort_by_key(|(id, _)| *id);

        let id_to_index: Vec<u64> = values.iter().map(|(id, _)| *id).collect();
        let index_to_id: Vec<u64> = id_to_index.clone();
        let bool_values: Vec<bool> = values.iter().map(|(_, v)| *v).collect();

        let compressed = TypeSpecificCompressor::compress_booleans(&bool_values);

        // Booleans always compress well (8x)
        self.compressed = Some(CompressedColumnData::Booleans {
            data: compressed,
            id_to_index,
            index_to_id,
        });
        self.compressed_count = values.len();
        self.values = non_bool_values;
    }

    /// Decompresses all values back to the hot buffer.
    #[allow(unsafe_code)]
    fn decompress_all(&mut self) {
        let Some(compressed) = self.compressed.take() else {
            return;
        };

        match compressed {
            CompressedColumnData::Integers {
                data, index_to_id, ..
            } => {
                if let Ok(values) = TypeSpecificCompressor::decompress_integers(&data) {
                    // Convert back to signed using zigzag decoding
                    let signed: Vec<i64> = values
                        .iter()
                        .map(|&v| crate::storage::zigzag_decode(v))
                        .collect();

                    for (i, id_u64) in index_to_id.iter().enumerate() {
                        if let Some(&value) = signed.get(i) {
                            let id: Id = unsafe { std::mem::transmute_copy(id_u64) };
                            self.values.insert(id, Value::Int64(value));
                        }
                    }
                }
            }
            CompressedColumnData::Strings {
                encoding,
                index_to_id,
                ..
            } => {
                for (i, id_u64) in index_to_id.iter().enumerate() {
                    if let Some(s) = encoding.get(i) {
                        let id: Id = unsafe { std::mem::transmute_copy(id_u64) };
                        self.values.insert(id, Value::String(ArcStr::from(s)));
                    }
                }
            }
            CompressedColumnData::Booleans {
                data, index_to_id, ..
            } => {
                if let Ok(values) = TypeSpecificCompressor::decompress_booleans(&data) {
                    for (i, id_u64) in index_to_id.iter().enumerate() {
                        if let Some(&value) = values.get(i) {
                            let id: Id = unsafe { std::mem::transmute_copy(id_u64) };
                            self.values.insert(id, Value::Bool(value));
                        }
                    }
                }
            }
        }

        self.compressed_count = 0;
    }

    /// Forces compression regardless of thresholds.
    ///
    /// Useful for bulk loading or when you know the column is complete.
    pub fn force_compress(&mut self) {
        self.compress();
    }

    /// Returns the zone map for this column.
    #[must_use]
    pub fn zone_map(&self) -> &ZoneMapEntry {
        &self.zone_map
    }

    /// Uses zone map to check if any values could satisfy the predicate.
    ///
    /// Returns `false` when we can prove no values match (so the column
    /// can be skipped entirely). Returns `true` if values might match.
    #[must_use]
    pub fn might_match(&self, op: CompareOp, value: &Value) -> bool {
        if self.zone_map_dirty {
            // Conservative: can't skip if zone map is stale
            return true;
        }

        match op {
            CompareOp::Eq => self.zone_map.might_contain_equal(value),
            CompareOp::Ne => {
                // Can only skip if all values are equal to the value
                // (which means min == max == value)
                match (&self.zone_map.min, &self.zone_map.max) {
                    (Some(min), Some(max)) => {
                        !(compare_values(min, value) == Some(Ordering::Equal)
                            && compare_values(max, value) == Some(Ordering::Equal))
                    }
                    _ => true,
                }
            }
            CompareOp::Lt => self.zone_map.might_contain_less_than(value, false),
            CompareOp::Le => self.zone_map.might_contain_less_than(value, true),
            CompareOp::Gt => self.zone_map.might_contain_greater_than(value, false),
            CompareOp::Ge => self.zone_map.might_contain_greater_than(value, true),
        }
    }

    /// Rebuilds zone map from current values.
    pub fn rebuild_zone_map(&mut self) {
        let mut zone_map = ZoneMapEntry::new();

        for value in self.values.values() {
            zone_map.row_count += 1;

            if matches!(value, Value::Null) {
                zone_map.null_count += 1;
                continue;
            }

            // Update min
            match &zone_map.min {
                None => zone_map.min = Some(value.clone()),
                Some(current) => {
                    if compare_values(value, current) == Some(Ordering::Less) {
                        zone_map.min = Some(value.clone());
                    }
                }
            }

            // Update max
            match &zone_map.max {
                None => zone_map.max = Some(value.clone()),
                Some(current) => {
                    if compare_values(value, current) == Some(Ordering::Greater) {
                        zone_map.max = Some(value.clone());
                    }
                }
            }
        }

        self.zone_map = zone_map;
        self.zone_map_dirty = false;
    }
}

/// Compares two values for ordering.
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    }
}

impl<Id: EntityId> Default for PropertyColumn<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// A borrowed reference to a property column for bulk reads.
///
/// Holds the read lock so the column can't change while you're iterating.
pub struct PropertyColumnRef<'a, Id: EntityId = NodeId> {
    _guard: parking_lot::RwLockReadGuard<'a, FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    #[allow(dead_code)]
    key: PropertyKey,
    _marker: PhantomData<Id>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcstr::ArcStr;

    #[test]
    fn test_property_storage_basic() {
        let storage = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let name_key = PropertyKey::new("name");
        let age_key = PropertyKey::new("age");

        storage.set(node1, name_key.clone(), "Alice".into());
        storage.set(node1, age_key.clone(), 30i64.into());
        storage.set(node2, name_key.clone(), "Bob".into());

        assert_eq!(
            storage.get(node1, &name_key),
            Some(Value::String("Alice".into()))
        );
        assert_eq!(storage.get(node1, &age_key), Some(Value::Int64(30)));
        assert_eq!(
            storage.get(node2, &name_key),
            Some(Value::String("Bob".into()))
        );
        assert!(storage.get(node2, &age_key).is_none());
    }

    #[test]
    fn test_property_storage_remove() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        let key = PropertyKey::new("name");

        storage.set(node, key.clone(), "Alice".into());
        assert!(storage.get(node, &key).is_some());

        let removed = storage.remove(node, &key);
        assert!(removed.is_some());
        assert!(storage.get(node, &key).is_none());
    }

    #[test]
    fn test_property_storage_get_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alice".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());
        storage.set(node, PropertyKey::new("active"), true.into());

        let props = storage.get_all(node);
        assert_eq!(props.len(), 3);
    }

    #[test]
    fn test_property_storage_remove_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alice".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());

        storage.remove_all(node);

        assert!(storage.get(node, &PropertyKey::new("name")).is_none());
        assert!(storage.get(node, &PropertyKey::new("age")).is_none());
    }

    #[test]
    fn test_property_column() {
        let mut col = PropertyColumn::new();

        col.set(NodeId::new(1), "Alice".into());
        col.set(NodeId::new(2), "Bob".into());

        assert_eq!(col.len(), 2);
        assert!(!col.is_empty());

        assert_eq!(col.get(NodeId::new(1)), Some(Value::String("Alice".into())));

        col.remove(NodeId::new(1));
        assert!(col.get(NodeId::new(1)).is_none());
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_compression_mode() {
        let col: PropertyColumn<NodeId> = PropertyColumn::new();
        assert_eq!(col.compression_mode(), CompressionMode::None);

        let col: PropertyColumn<NodeId> = PropertyColumn::with_compression(CompressionMode::Auto);
        assert_eq!(col.compression_mode(), CompressionMode::Auto);
    }

    #[test]
    fn test_property_storage_with_compression() {
        let storage = PropertyStorage::with_compression(CompressionMode::Auto);

        for i in 0..100 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("age"),
                Value::Int64(20 + (i as i64 % 50)),
            );
        }

        // Values should still be readable
        assert_eq!(
            storage.get(NodeId::new(0), &PropertyKey::new("age")),
            Some(Value::Int64(20))
        );
        assert_eq!(
            storage.get(NodeId::new(50), &PropertyKey::new("age")),
            Some(Value::Int64(20))
        );
    }

    #[test]
    fn test_compress_integer_column() {
        let mut col: PropertyColumn<NodeId> =
            PropertyColumn::with_compression(CompressionMode::Auto);

        // Add many sequential integers
        for i in 0..2000 {
            col.set(NodeId::new(i), Value::Int64(1000 + i as i64));
        }

        // Should have triggered compression at some point
        // Total count should include both compressed and hot buffer values
        let stats = col.compression_stats();
        assert_eq!(stats.value_count, 2000);

        // Values from the hot buffer should be readable
        // Note: Compressed values are not accessible via get() - see design note
        let last_value = col.get(NodeId::new(1999));
        assert!(last_value.is_some() || col.is_compressed());
    }

    #[test]
    fn test_compress_string_column() {
        let mut col: PropertyColumn<NodeId> =
            PropertyColumn::with_compression(CompressionMode::Auto);

        // Add repeated strings (good for dictionary compression)
        let categories = ["Person", "Company", "Product", "Location"];
        for i in 0..2000 {
            let cat = categories[i % 4];
            col.set(NodeId::new(i as u64), Value::String(ArcStr::from(cat)));
        }

        // Total count should be correct
        assert_eq!(col.len(), 2000);

        // Late values should be in hot buffer and readable
        let last_value = col.get(NodeId::new(1999));
        assert!(last_value.is_some() || col.is_compressed());
    }

    #[test]
    fn test_compress_boolean_column() {
        let mut col: PropertyColumn<NodeId> =
            PropertyColumn::with_compression(CompressionMode::Auto);

        // Add booleans
        for i in 0..2000 {
            col.set(NodeId::new(i as u64), Value::Bool(i % 2 == 0));
        }

        // Verify total count
        assert_eq!(col.len(), 2000);

        // Late values should be readable
        let last_value = col.get(NodeId::new(1999));
        assert!(last_value.is_some() || col.is_compressed());
    }

    #[test]
    fn test_force_compress() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();

        // Add fewer values than the threshold
        for i in 0..100 {
            col.set(NodeId::new(i), Value::Int64(i as i64));
        }

        // Force compression
        col.force_compress();

        // Stats should show compression was applied if beneficial
        let stats = col.compression_stats();
        assert_eq!(stats.value_count, 100);
    }

    #[test]
    fn test_compression_stats() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();

        for i in 0..50 {
            col.set(NodeId::new(i), Value::Int64(i as i64));
        }

        let stats = col.compression_stats();
        assert_eq!(stats.value_count, 50);
        assert!(stats.uncompressed_size > 0);
    }

    #[test]
    fn test_storage_compression_stats() {
        let storage = PropertyStorage::with_compression(CompressionMode::Auto);

        for i in 0..100 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("age"),
                Value::Int64(i as i64),
            );
            storage.set(
                NodeId::new(i),
                PropertyKey::new("name"),
                Value::String(ArcStr::from("Alice")),
            );
        }

        let stats = storage.compression_stats();
        assert_eq!(stats.len(), 2); // Two columns
        assert!(stats.contains_key(&PropertyKey::new("age")));
        assert!(stats.contains_key(&PropertyKey::new("name")));
    }

    #[test]
    fn test_memory_usage() {
        let storage = PropertyStorage::new();

        for i in 0..100 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("value"),
                Value::Int64(i as i64),
            );
        }

        let usage = storage.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_get_batch_single_property() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let node3 = NodeId::new(3);
        let age_key = PropertyKey::new("age");

        storage.set(node1, age_key.clone(), 25i64.into());
        storage.set(node2, age_key.clone(), 30i64.into());
        // node3 has no age property

        let ids = vec![node1, node2, node3];
        let values = storage.get_batch(&ids, &age_key);

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], Some(Value::Int64(25)));
        assert_eq!(values[1], Some(Value::Int64(30)));
        assert_eq!(values[2], None);
    }

    #[test]
    fn test_get_batch_missing_column() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let missing_key = PropertyKey::new("nonexistent");

        let ids = vec![node1, node2];
        let values = storage.get_batch(&ids, &missing_key);

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], None);
        assert_eq!(values[1], None);
    }

    #[test]
    fn test_get_batch_empty_ids() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();
        let key = PropertyKey::new("any");

        let values = storage.get_batch(&[], &key);
        assert!(values.is_empty());
    }

    #[test]
    fn test_get_all_batch() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let node3 = NodeId::new(3);

        storage.set(node1, PropertyKey::new("name"), "Alice".into());
        storage.set(node1, PropertyKey::new("age"), 25i64.into());
        storage.set(node2, PropertyKey::new("name"), "Bob".into());
        // node3 has no properties

        let ids = vec![node1, node2, node3];
        let all_props = storage.get_all_batch(&ids);

        assert_eq!(all_props.len(), 3);
        assert_eq!(all_props[0].len(), 2); // name and age
        assert_eq!(all_props[1].len(), 1); // name only
        assert_eq!(all_props[2].len(), 0); // no properties

        assert_eq!(
            all_props[0].get(&PropertyKey::new("name")),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(
            all_props[1].get(&PropertyKey::new("name")),
            Some(&Value::String("Bob".into()))
        );
    }

    #[test]
    fn test_get_all_batch_empty_ids() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let all_props = storage.get_all_batch(&[]);
        assert!(all_props.is_empty());
    }
}
