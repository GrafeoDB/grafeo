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

use crate::codec::CompressionCodec;
#[cfg(not(feature = "temporal"))]
use crate::codec::block::DEFAULT_BLOCK_ROWS;
#[cfg(not(feature = "temporal"))]
use crate::codec::{CompressedData, DictionaryBuilder, DictionaryEncoding, TypeSpecificCompressor};
use crate::index::zone_map::ZoneMapEntry;
#[cfg(not(feature = "temporal"))]
use arcstr::ArcStr;
#[cfg(feature = "temporal")]
use grafeo_common::temporal::VersionLog;
#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_common::utils::hash::FxHashSet;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

/// Compression mode for property columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
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
#[cfg(not(feature = "temporal"))]
const COMPRESSION_THRESHOLD: usize = 1000;

/// Size of the hot buffer for recent writes (before compression).
/// Larger buffer (4096) keeps more recent data uncompressed for faster reads.
/// This trades ~64KB of memory overhead per column for 1.5-2x faster point lookups
/// on recently-written data.
#[cfg(not(feature = "temporal"))]
const HOT_BUFFER_SIZE: usize = 4096;

/// Comparison operators used for zone map predicate checks.
///
/// These map directly to GQL comparison operators like `=`, `<`, `>=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Trait for IDs that can key into property storage.
///
/// Implemented for [`NodeId`] and [`EdgeId`] - you can store properties on both.
/// Provides safe conversions to/from `u64` for compression, replacing unsafe transmute.
pub trait EntityId: Copy + Eq + Hash + 'static {
    /// Returns the raw `u64` value.
    fn as_u64(self) -> u64;
    /// Creates an ID from a raw `u64` value.
    fn from_u64(v: u64) -> Self;
}

impl EntityId for NodeId {
    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

impl EntityId for EdgeId {
    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

/// The values of a spilled property column, held outside the heap (in a
/// spill file the engine wrote).
///
/// A column with a backing reads through it: a value written after the spill
/// stays in the column and wins, and an id removed after the spill reads as
/// absent. Every reader (queries, checkpoints, copies, search) sees the same
/// values whether the column is spilled or not.
///
/// The column relies on three things:
///
/// - The contents never change, and the backing stays readable until it is
///   dropped: the column counts and tombstones against them, and a vector
///   read may still hold the backing after a reload let go of it. So a
///   backing releases its file in `Drop`, never at the reload.
/// - [`ids`](Self::ids) and [`contains`](Self::contains) come from what the
///   backing holds in memory (its index), so they cannot fail; `ids` lists
///   each id once, and `contains` is true exactly for those ids.
/// - A backing never calls back into the property storage: [`get`](Self::get)
///   and [`contains`](Self::contains) run with the storage lock held.
///
/// Reading a value can fail (a spill file that cannot be read). A reader that
/// must not lose it (a reload, a checkpoint, a copy) stops on the error; a
/// query reads the value as absent. The backing reports its own read errors
/// (the engine's logs the first one of its column).
#[cfg(not(feature = "temporal"))]
pub trait ColumnBacking<Id: EntityId>: Send + Sync {
    /// Returns the stored value for `id`.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the value.
    fn get(&self, id: Id) -> std::io::Result<Option<Value>>;

    /// Returns whether a value is stored for `id`, without reading it.
    fn contains(&self, id: Id) -> bool;

    /// Returns every id with a stored value, each once, in any order.
    fn ids(&self) -> Vec<Id>;

    /// Returns the number of stored values.
    fn len(&self) -> usize;

    /// Returns whether no value is stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the heap bytes the backing holds (an index, a cache), not
    /// counting its file.
    fn heap_bytes(&self) -> usize;

    /// Calls `f` with the vector stored for `id`, and returns whether there
    /// was one. Vector search reads every distance through this, so a backing
    /// should hand the vector out without allocating; the default copies it
    /// through [`get`](Self::get). Runs without the storage lock held.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the vector.
    fn with_vector(&self, id: Id, f: &mut dyn FnMut(&[f32])) -> std::io::Result<bool> {
        Ok(match self.get(id)? {
            Some(Value::Vector(vector)) => {
                f(&vector);
                true
            }
            _ => false,
        })
    }
}

/// Where a vector read finds its vector: decided with the storage lock held,
/// read after it is released, so the caller's closure never runs under it.
#[cfg(not(feature = "temporal"))]
enum VectorSource<Id: EntityId> {
    /// The column's own vector.
    Own(Arc<[f32]>),
    /// The vector is in the backing.
    Backed(Arc<dyn ColumnBacking<Id>>),
}

/// The error of a backing that lists an id it holds no value for.
#[cfg(not(feature = "temporal"))]
fn missing_backed_value() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "a spill backing lists an id but holds no value for it",
    )
}

#[cfg(all(test, not(feature = "temporal")))]
thread_local! {
    /// How many compressed integer or boolean columns this thread decoded:
    /// tests read it before and after a call to count the call's decodes.
    static COMPRESSED_DECODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts one decode of a compressed integer or boolean column (tests only).
#[cfg(not(feature = "temporal"))]
fn count_compressed_decode() {
    #[cfg(test)]
    COMPRESSED_DECODES.set(COMPRESSED_DECODES.get() + 1);
}

/// The rows of one compressed integer or boolean column, decoded for the
/// reads of one call. Decoding one row of such a column decodes all of it, so
/// a batch read keeps the rows for its other ids instead of decoding the
/// column once per id. Empty until a read needs them; a temporal column is
/// never compressed, so there it stays empty.
#[derive(Default)]
struct DecodedRows {
    #[cfg(not(feature = "temporal"))]
    integers: Option<Vec<u64>>,
    #[cfg(not(feature = "temporal"))]
    booleans: Option<Vec<bool>>,
}

#[cfg(not(feature = "temporal"))]
impl DecodedRows {
    /// The rows of the integer column `data`, decoded on first use.
    fn integers(&mut self, data: &CompressedData) -> std::io::Result<&[u64]> {
        if self.integers.is_none() {
            count_compressed_decode();
            self.integers = Some(TypeSpecificCompressor::decompress_integers(data)?);
        }
        Ok(self.integers.as_deref().unwrap_or_default())
    }

    /// The rows of the boolean column `data`, decoded on first use.
    fn booleans(&mut self, data: &CompressedData) -> std::io::Result<&[bool]> {
        if self.booleans.is_none() {
            count_compressed_decode();
            self.booleans = Some(TypeSpecificCompressor::decompress_booleans(data)?);
        }
        Ok(self.booleans.as_deref().unwrap_or_default())
    }
}

/// The error of a compressed row that does not decode.
#[cfg(not(feature = "temporal"))]
fn undecodable_compressed_row() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "a compressed property column lists an id but holds no value for it",
    )
}

/// An in-memory [`ColumnBacking`] for tests: it counts the values copied out
/// of it, can fail its reads, and can run a hook the next time its ids are
/// read.
#[cfg(all(test, not(feature = "temporal")))]
pub(crate) mod test_backing {
    use super::{ColumnBacking, Value};
    use grafeo_common::types::NodeId;
    use grafeo_common::utils::hash::FxHashMap;
    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    type Hook = Box<dyn FnOnce() + Send>;

    pub(crate) struct MemoryBacking {
        values: FxHashMap<NodeId, Value>,
        /// Listed by `ids` (and `contains`) on top of the values: a backing
        /// that breaks its listing contract.
        extra_ids: Vec<NodeId>,
        /// How many values `get` copied out.
        pub(crate) copies: AtomicUsize,
        /// Every value read fails while set.
        failing: AtomicBool,
        on_ids: Mutex<Option<Hook>>,
    }

    impl MemoryBacking {
        /// What `heap_bytes` reports.
        pub(crate) const HEAP_BYTES: usize = 1000;

        pub(crate) fn of(entries: &[(NodeId, Value)]) -> Arc<Self> {
            Self::listing(entries, &[])
        }

        /// A backing holding `entries` that also lists `extra` ids: one it
        /// holds no value for, or one it lists twice, breaks its contract.
        pub(crate) fn listing(entries: &[(NodeId, Value)], extra: &[NodeId]) -> Arc<Self> {
            Arc::new(Self {
                values: entries.iter().cloned().collect(),
                extra_ids: extra.to_vec(),
                copies: AtomicUsize::new(0),
                failing: AtomicBool::new(false),
                on_ids: Mutex::new(None),
            })
        }

        /// Runs `hook` once, the next time the ids are read.
        pub(crate) fn on_ids(&self, hook: impl FnOnce() + Send + 'static) {
            *self.on_ids.lock() = Some(Box::new(hook));
        }

        /// Makes every value read fail (`true`) or succeed again.
        pub(crate) fn fail_reads(&self, failing: bool) {
            self.failing.store(failing, Ordering::Relaxed);
        }

        fn check(&self) -> std::io::Result<()> {
            if self.failing.load(Ordering::Relaxed) {
                Err(std::io::Error::other("the spill file cannot be read"))
            } else {
                Ok(())
            }
        }
    }

    impl ColumnBacking<NodeId> for MemoryBacking {
        fn get(&self, id: NodeId) -> std::io::Result<Option<Value>> {
            self.check()?;
            self.copies.fetch_add(1, Ordering::Relaxed);
            Ok(self.values.get(&id).cloned())
        }

        fn contains(&self, id: NodeId) -> bool {
            self.values.contains_key(&id) || self.extra_ids.contains(&id)
        }

        fn ids(&self) -> Vec<NodeId> {
            let hook = self.on_ids.lock().take();
            if let Some(hook) = hook {
                hook();
            }
            self.values
                .keys()
                .copied()
                .chain(self.extra_ids.iter().copied())
                .collect()
        }

        fn len(&self) -> usize {
            self.values.len()
        }

        fn heap_bytes(&self) -> usize {
            Self::HEAP_BYTES
        }

        fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> std::io::Result<bool> {
            self.check()?;
            Ok(match self.values.get(&id) {
                Some(Value::Vector(vector)) => {
                    f(vector);
                    true
                }
                _ => false,
            })
        }
    }
}

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
/// # #[cfg(not(feature = "temporal"))]
/// # {
/// use grafeo_core::graph::lpg::PropertyStorage;
/// use grafeo_common::types::{NodeId, PropertyKey};
///
/// let storage = PropertyStorage::new();
/// let alix = NodeId::new(1);
///
/// storage.set(alix, PropertyKey::new("name"), "Alix".into());
/// storage.set(alix, PropertyKey::new("age"), 30i64.into());
///
/// // Fetch all properties at once
/// let props = storage.get_all(alix);
/// assert_eq!(props.len(), 2);
/// # }
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
    #[cfg(not(feature = "temporal"))]
    pub fn set(&self, id: Id, key: PropertyKey, value: Value) {
        let mut columns = self.columns.write();
        let mode = self.default_compression;
        columns
            .entry(key)
            .or_insert_with(|| PropertyColumn::with_compression(mode))
            .set(id, value);
    }

    /// Sets a property value for an entity at a specific epoch.
    ///
    /// For non-transactional writes, pass the current epoch.
    /// For transactional writes, pass `EpochId::PENDING`.
    #[cfg(feature = "temporal")]
    pub fn set(&self, id: Id, key: PropertyKey, value: Value, epoch: EpochId) {
        let mut columns = self.columns.write();
        let mode = self.default_compression;
        columns
            .entry(key)
            .or_insert_with(|| PropertyColumn::with_compression(mode))
            .set(id, value, epoch);
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

    /// Returns the total memory usage of all columns (compressed size estimate).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let columns = self.columns.read();
        columns
            .values()
            .map(|col| col.compression_stats().compressed_size)
            .sum()
    }

    /// Returns estimated heap memory for all columns including hash map overhead.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        let columns = self.columns.read();
        // Outer hash map capacity
        let map_overhead = columns.capacity()
            * (std::mem::size_of::<PropertyKey>() + std::mem::size_of::<PropertyColumn<Id>>() + 1);
        // Sum of all column heap memory
        let column_bytes: usize = columns.values().map(|col| col.heap_memory_bytes()).sum();
        map_overhead + column_bytes
    }

    /// Gets a property value for an entity.
    #[must_use]
    pub fn get(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get(id))
    }

    /// [`get`](Self::get) as a fallible read: a spilled value that cannot be
    /// read is an error. For the readers that must not lose it, such as the
    /// before-image of a transaction's write.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value.
    pub fn try_get(&self, id: Id, key: &PropertyKey) -> Result<Option<Value>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => col.try_get(id).map_err(Error::Io),
            None => Ok(None),
        }
    }

    /// Removes a property value for an entity, returning it. A compressed or
    /// spilled value is read before it is hidden, so the caller can log and
    /// undo the removal.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the value (a spilled value whose file
    /// cannot be read); nothing changes then.
    #[cfg(not(feature = "temporal"))]
    pub fn remove(&self, id: Id, key: &PropertyKey) -> Result<Option<Value>> {
        let mut columns = self.columns.write();
        match columns.get_mut(key) {
            Some(col) => col.remove(id).map_err(Error::Io),
            None => Ok(None),
        }
    }

    /// Removes a property value for an entity without reading it, so it
    /// never fails: for a removal that does not need the value, such as the
    /// rollback of a write.
    #[cfg(not(feature = "temporal"))]
    pub fn discard(&self, id: Id, key: &PropertyKey) {
        if let Some(col) = self.columns.write().get_mut(key) {
            col.discard(id);
        }
    }

    /// Removes a property value for an entity (temporal: appends tombstone at epoch).
    #[cfg(feature = "temporal")]
    pub fn remove(&self, id: Id, key: &PropertyKey, epoch: EpochId) -> Option<Value> {
        let mut columns = self.columns.write();
        columns.get_mut(key).and_then(|col| col.remove(id, epoch))
    }

    /// Removes all properties for an entity.
    #[cfg(not(feature = "temporal"))]
    pub fn remove_all(&self, id: Id) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.discard(id);
        }
    }

    /// Removes all properties for an entity (temporal: tombstones at current epoch).
    #[cfg(feature = "temporal")]
    pub fn remove_all(&self, id: Id, epoch: EpochId) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove(id, epoch);
        }
    }

    /// Removes every property of an entity that never became visible, the
    /// rollback of the transaction that created it. Unlike `remove_all`, it
    /// keeps no history.
    pub fn purge(&self, id: Id) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            #[cfg(not(feature = "temporal"))]
            col.discard(id);
            #[cfg(feature = "temporal")]
            col.purge(id);
        }
    }

    /// Gets all properties for an entity, as a fallible read: a spilled value
    /// that cannot be read is an error, never left out. For the readers that
    /// must not lose a value (checkpoints, copies).
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value.
    pub fn try_get_all(&self, id: Id) -> Result<FxHashMap<PropertyKey, Value>> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.try_get(id).map_err(Error::Io)? {
                result.insert(key.clone(), value);
            }
        }
        Ok(result)
    }

    /// Gets all properties for an entity. A spilled value that cannot be read
    /// is left out (the backing reports the error); readers that must not
    /// lose it use [`try_get_all`](Self::try_get_all).
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
    ///
    /// A spilled value that cannot be read reads as `None` (the backing reports
    /// the error); readers that must not lose it use
    /// [`try_get_batch`](Self::try_get_batch).
    #[must_use]
    pub fn get_batch(&self, ids: &[Id], key: &PropertyKey) -> Vec<Option<Value>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => {
                // One decode of a compressed column for the whole batch.
                let mut decoded = DecodedRows::default();
                ids.iter()
                    .map(|&id| col.try_get_decoded(id, &mut decoded).ok().flatten())
                    .collect()
            }
            None => vec![None; ids.len()],
        }
    }

    /// [`get_batch`](Self::get_batch) as a fallible read: a spilled value that
    /// cannot be read is an error. A checkpoint streams a spilled column with
    /// [`column_ids`](Self::column_ids) and this, chunk by chunk, so it holds
    /// one chunk of values at a time.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value.
    pub fn try_get_batch(&self, ids: &[Id], key: &PropertyKey) -> Result<Vec<Option<Value>>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => {
                // One decode of a compressed column for the whole batch.
                let mut decoded = DecodedRows::default();
                ids.iter()
                    .map(|&id| col.try_get_decoded(id, &mut decoded).map_err(Error::Io))
                    .collect()
            }
            None => Ok(vec![None; ids.len()]),
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
        // One decode of each compressed column for the whole batch.
        let mut decoded: Vec<DecodedRows> =
            columns.keys().map(|_| DecodedRows::default()).collect();

        for &id in ids {
            // Pre-allocate HashMap with expected column count
            let mut result = FxHashMap::with_capacity_and_hasher(column_count, Default::default());
            for ((key, col), decoded) in columns.iter().zip(decoded.iter_mut()) {
                if let Some(value) = col.try_get_decoded(id, decoded).ok().flatten() {
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
        // One decode of each compressed column for the whole batch.
        let mut decoded: Vec<DecodedRows> = requested_columns
            .iter()
            .map(|_| DecodedRows::default())
            .collect();

        for &id in ids {
            let mut result =
                FxHashMap::with_capacity_and_hasher(requested_columns.len(), Default::default());
            // Only iterate requested columns, not all columns
            for ((key, col), decoded) in requested_columns.iter().zip(decoded.iter_mut()) {
                if let Some(value) = col.try_get_decoded(id, decoded).ok().flatten() {
                    result.insert((*key).clone(), value);
                }
            }
            results.push(result);
        }

        results
    }

    /// Returns the ids with a value for `key`, in id order. A spilled column
    /// is read through its backing.
    #[must_use]
    pub fn column_ids(&self, key: &PropertyKey) -> Vec<Id> {
        self.columns
            .read()
            .get(key)
            .map_or_else(Vec::new, PropertyColumn::ids)
    }

    /// Returns every `(id, value)` of `key`, in id order, read under one lock
    /// acquisition: compressed values and those of a spilled column included.
    /// A value that cannot be read is an error, never left out, so a snapshot
    /// or a copy of the column is complete.
    ///
    /// This holds the storage lock (every column of the entity kind) and
    /// loads the whole column: a checkpoint of a large spilled column streams
    /// it with [`column_ids`](Self::column_ids) and
    /// [`try_get_batch`](Self::try_get_batch) instead.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value or decoding a compressed
    /// one.
    pub fn try_column_entries(&self, key: &PropertyKey) -> Result<Vec<(Id, Value)>> {
        self.columns
            .read()
            .get(key)
            .map_or_else(|| Ok(Vec::new()), PropertyColumn::try_entries)
            .map_err(Error::Io)
    }

    /// Calls `f` with the vector stored for `id` under `key`, and returns its
    /// result, or `None` when there is no vector. A spilled vector is not
    /// copied out of its backing.
    ///
    /// `f` runs without the storage lock held, so it may read the store again
    /// (pairwise distances do). A spilled vector that cannot be read reads as
    /// `None` (the backing reports the error).
    #[cfg(not(feature = "temporal"))]
    pub fn with_vector<R>(
        &self,
        id: Id,
        key: &PropertyKey,
        f: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        // Decided under the lock, read after it is released.
        let source = self.columns.read().get(key)?.vector_source(id)?;
        match source {
            VectorSource::Own(vector) => Some(f(&vector)),
            VectorSource::Backed(backing) => {
                let mut f = Some(f);
                let mut result = None;
                // A vector that cannot be read reads as absent (the backing
                // reports the error).
                backing
                    .with_vector(id, &mut |vector| {
                        if let Some(f) = f.take() {
                            result = Some(f(vector));
                        }
                    })
                    .ok()?;
                result
            }
        }
    }

    /// Calls `f` with the latest vector stored for `id` under `key`, and
    /// returns its result, or `None` when there is no vector. `f` runs
    /// without the storage lock held.
    #[cfg(feature = "temporal")]
    pub fn with_vector<R>(
        &self,
        id: Id,
        key: &PropertyKey,
        f: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        let vector = self.columns.read().get(key)?.vector(id)?;
        Some(f(&vector))
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

    /// Removes all property data.
    pub fn clear(&self) {
        self.columns.write().clear();
    }

    // ── Column-level spill / reload ────────────────────────────────

    /// Spills the column `key` into `backing`, which holds the entries of
    /// `snapshot`: the column as [`try_column_entries`](Self::try_column_entries)
    /// returned it before the backing was written, without the lock held.
    /// A value changed since the snapshot stays in the column and one removed
    /// since stays removed; the other values leave the heap.
    ///
    /// Returns `false`, changing nothing, when the column is missing or
    /// spilled already; the caller then discards what it wrote.
    #[cfg(not(feature = "temporal"))]
    pub fn spill_column(
        &self,
        key: &PropertyKey,
        backing: Arc<dyn ColumnBacking<Id>>,
        snapshot: &[(Id, Value)],
    ) -> bool {
        self.columns
            .write()
            .get_mut(key)
            .is_some_and(|column| column.spill(backing, snapshot))
    }

    /// Moves the values of a spilled column back onto the heap and lets go
    /// of its backing. A value written or removed while the column was
    /// spilled keeps its change.
    ///
    /// The backing is read without the lock held, so other readers and
    /// writers wait only for the merge. Returns `Ok(false)` when the column is
    /// not spilled, or was reloaded (and maybe spilled again) meanwhile.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value; the column then keeps
    /// its backing, and nothing changes.
    #[cfg(not(feature = "temporal"))]
    pub fn reload_column(&self, key: &PropertyKey) -> Result<bool> {
        let Some(backing) = self
            .columns
            .read()
            .get(key)
            .and_then(|column| column.backing.clone())
        else {
            return Ok(false);
        };
        let mut entries = Vec::with_capacity(backing.len());
        for id in backing.ids() {
            let value = backing
                .get(id)?
                .ok_or_else(|| Error::Io(missing_backed_value()))?;
            entries.push((id, value));
        }
        let mut columns = self.columns.write();
        match columns.get_mut(key) {
            Some(column)
                if column
                    .backing
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &backing)) =>
            {
                column.unspill(entries);
                Ok(true)
            }
            // Reloaded (or cleared) by someone else meanwhile: what was read
            // is stale.
            _ => Ok(false),
        }
    }

    /// Returns the keys of the spilled columns, in key order.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn spilled_columns(&self) -> Vec<PropertyKey> {
        let mut keys: Vec<PropertyKey> = self
            .columns
            .read()
            .iter()
            .filter(|(_, column)| column.backing.is_some())
            .map(|(key, _)| key.clone())
            .collect();
        keys.sort_unstable();
        keys
    }

    /// Gets a column by key for bulk access.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<PropertyColumnRef<'_, Id>> {
        let columns = self.columns.read();
        if columns.contains_key(key) {
            Some(PropertyColumnRef {
                _guard: columns,
                _key: key.clone(),
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
            .map_or(true, |col| col.might_match(op, value)) // No column = assume might match (conservative)
    }

    /// Gets the zone map for a property column.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<ZoneMapEntry> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.zone_map().clone())
    }

    /// Returns the per-block zone maps for a property column, if any.
    ///
    /// Returns `None` when the column doesn't exist; returns `Some(empty)`
    /// when the column exists but is uncompressed (the hot buffer is
    /// unordered, so per-block pruning is meaningless there). Phase 4 will
    /// treat "no per-block stats" as "fall back to the column-level zone
    /// map".
    ///
    /// **Temporal mode:** always returns `Some(empty)` for any existing
    /// column. Compression is disabled for `VersionLog`-backed columns,
    /// so there is no sorted compressed array to chunk into blocks. Use
    /// the column-level [`zone_map`](Self::zone_map) instead.
    #[must_use]
    pub fn block_zone_maps_for(&self, key: &PropertyKey) -> Option<Vec<ZoneMapEntry>> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.block_zone_maps().to_vec())
    }

    /// Returns the number of compressed blocks for a property column.
    ///
    /// Returns `None` when the column doesn't exist; returns `Some(0)` for
    /// an uncompressed column.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn block_count_for(&self, key: &PropertyKey) -> Option<usize> {
        self.columns.read().get(key).map(|col| col.block_count())
    }

    /// Decodes a single compressed block of a property column into
    /// `(id, value)` pairs: exactly the rows that the entry of
    /// [`block_zone_maps_for`](Self::block_zone_maps_for) at the same index
    /// describes.
    ///
    /// Returns `None` when the column doesn't exist, the column is
    /// uncompressed, or `block_idx` is out of range.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn decode_block_for(
        &self,
        key: &PropertyKey,
        block_idx: usize,
    ) -> Option<DecodedBlock<Id>> {
        self.columns
            .read()
            .get(key)
            .and_then(|col| col.decode_block(block_idx))
    }

    /// Decodes every compressed block of a property column under a single
    /// read-lock acquisition, returning them as a `Vec`.
    ///
    /// Returns an empty `Vec` when the column doesn't exist or is
    /// uncompressed. Phase 4's iterator-bounds operator prefers
    /// [`Self::decode_block_for`] (after pruning by zone map) over
    /// decoding all blocks up-front; this method exists for tests and
    /// debug tools that want the whole picture.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn decoded_blocks_for(&self, key: &PropertyKey) -> Vec<DecodedBlock<Id>> {
        let columns = self.columns.read();
        match columns.get(key) {
            Some(col) => col.iter_decoded_blocks().collect(),
            None => Vec::new(),
        }
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
        columns.get(key).map_or(true, |col| {
            col.zone_map()
                .might_contain_range(min, max, min_inclusive, max_inclusive)
        }) // No column = assume might match (conservative)
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

// === Temporal-only methods for PropertyStorage ===
#[cfg(feature = "temporal")]
impl<Id: EntityId> PropertyStorage<Id> {
    /// Returns a write guard to the columns map for targeted rollback.
    pub(crate) fn columns_write(
        &self,
    ) -> parking_lot::RwLockWriteGuard<'_, FxHashMap<PropertyKey, PropertyColumn<Id>>> {
        self.columns.write()
    }

    /// Gets a property value at a specific epoch.
    #[must_use]
    pub fn get_at(&self, id: Id, key: &PropertyKey, epoch: EpochId) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get_at(id, epoch))
    }

    /// Gets all properties for an entity at a specific epoch.
    #[must_use]
    pub fn get_all_at(&self, id: Id, epoch: EpochId) -> FxHashMap<PropertyKey, Value> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.get_at(id, epoch) {
                result.insert(key.clone(), value);
            }
        }
        result
    }

    /// Replaces PENDING epochs with the real commit epoch in all columns.
    pub fn finalize_pending(&self, real_epoch: EpochId) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.finalize_pending(real_epoch);
        }
    }

    /// Removes all PENDING entries from all columns (transaction rollback).
    pub fn remove_pending(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove_pending();
        }
    }

    /// Garbage-collects old versions from all columns, and returns how many
    /// it dropped.
    pub fn gc(&self, min_epoch: EpochId) -> usize {
        let mut columns = self.columns.write();
        columns.values_mut().map(|col| col.gc(min_epoch)).sum()
    }

    /// Returns the full version history for all properties of an entity.
    ///
    /// Each entry is `(key, Vec<(epoch, value)>)`. Useful for snapshot
    /// export that preserves temporal history.
    #[must_use]
    pub fn get_all_history(&self, id: Id) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        let columns = self.columns.read();
        let mut result = Vec::new();
        for (key, col) in columns.iter() {
            if let Some(log) = col.values.get(&id) {
                let entries: Vec<(EpochId, Value)> = log
                    .history()
                    .iter()
                    .map(|(epoch, value)| (*epoch, value.clone()))
                    .collect();
                if !entries.is_empty() {
                    result.push((key.clone(), entries));
                }
            }
        }
        result
    }

    /// Returns the version history for a single property of an entity.
    ///
    /// More efficient than `get_all_history` when only one property is needed.
    #[must_use]
    pub fn get_history(&self, id: Id, key: &PropertyKey) -> Vec<(EpochId, Value)> {
        let columns = self.columns.read();
        columns
            .get(key)
            .and_then(|col| col.values.get(&id))
            .map(|log| log.history().iter().map(|(e, v)| (*e, v.clone())).collect())
            .unwrap_or_default()
    }
}

/// Compressed storage for a property column.
///
/// Holds the compressed representation of values along with the index
/// mapping entity IDs to positions in the compressed array.
#[cfg(not(feature = "temporal"))]
#[derive(Debug)]
#[non_exhaustive]
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

#[cfg(not(feature = "temporal"))]
impl CompressedColumnData {
    /// The ids of the compressed rows, in id order: row `i` belongs to the
    /// `i`-th.
    fn ids(&self) -> &[u64] {
        match self {
            Self::Integers { index_to_id, .. }
            | Self::Strings { index_to_id, .. }
            | Self::Booleans { index_to_id, .. } => index_to_id,
        }
    }

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
                encoding.code_count() * 4
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
}

/// A decoded compressed block of `(id, value)` pairs from a property column.
///
/// Phase 4's iterator-bounds operator consumes these after pruning via
/// per-block zone maps. The `entries` are sorted by entity id (matching
/// the underlying compressed layout).
#[cfg(not(feature = "temporal"))]
#[derive(Debug, Clone)]
pub struct DecodedBlock<Id: EntityId> {
    /// Per-block min/max/null/row counts populated when the block was
    /// compressed.
    pub zone_map: ZoneMapEntry,
    /// `(id, value)` pairs, sorted by id.
    pub entries: Vec<(Id, Value)>,
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
    #[cfg(not(feature = "temporal"))]
    values: FxHashMap<Id, Value>,
    /// Versioned storage: entity ID -> append-only version log.
    /// Each value is tagged with the epoch it was written in.
    #[cfg(feature = "temporal")]
    values: FxHashMap<Id, VersionLog<Value>>,
    /// Entities whose log holds more than one entry: the only ones garbage
    /// collection has work for, so it visits these instead of every log.
    #[cfg(feature = "temporal")]
    gc_candidates: FxHashSet<Id>,
    /// Zone map tracking min/max/null_count for predicate pushdown.
    zone_map: ZoneMapEntry,
    /// Whether zone map needs rebuild (after removes).
    zone_map_dirty: bool,
    /// Compression mode for this column.
    compression_mode: CompressionMode,
    /// Compressed data (when compression is enabled and triggered).
    #[cfg(not(feature = "temporal"))]
    compressed: Option<CompressedColumnData>,
    /// Number of values before last compression.
    #[cfg(not(feature = "temporal"))]
    compressed_count: usize,
    /// Where the values live while the column is spilled. `values` then holds
    /// only what was written after the spill, which wins over the backing.
    #[cfg(not(feature = "temporal"))]
    backing: Option<Arc<dyn ColumnBacking<Id>>>,
    /// Ids whose backed value was removed after the spill: each is held by
    /// the backing and absent from `values`.
    #[cfg(not(feature = "temporal"))]
    removed: FxHashSet<Id>,
    /// Per-block zone maps populated when the column is compressed.
    ///
    /// Each entry covers a contiguous slice of `DEFAULT_BLOCK_ROWS` rows of
    /// the sorted compressed array. Empty when the column is uncompressed
    /// (the hot buffer is a `HashMap` with no row order, so per-block
    /// pruning would be meaningless). Phase 4 consumes these for lazy
    /// `range_iter`-style scans.
    block_zone_maps: Vec<ZoneMapEntry>,
}

#[cfg(not(feature = "temporal"))]
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
            backing: None,
            removed: FxHashSet::default(),
            block_zone_maps: Vec::new(),
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
            backing: None,
            removed: FxHashSet::default(),
            block_zone_maps: Vec::new(),
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
        if !self.removed.is_empty() {
            self.removed.remove(&id);
        }
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

    /// Gets a value for an entity: the column's own value (the hot buffer)
    /// first, then a compressed one, then the backing of a spilled column. A
    /// value that cannot be read (a spilled value whose file cannot be read)
    /// reads as `None` (the backing reports the error);
    /// [`try_get`](Self::try_get) reports it.
    ///
    /// A compressed integer or boolean is decoded with the rest of its
    /// column, so a read by id of a compressed column costs a pass over it;
    /// the batch reads of [`PropertyStorage`] decode it once per call.
    #[must_use]
    pub fn get(&self, id: Id) -> Option<Value> {
        // A value that cannot be read reads as absent (the backing reports
        // the error).
        self.try_get(id).ok().flatten()
    }

    /// [`get`](Self::get) as a fallible read: a value that cannot be read is
    /// an error.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value or decoding a compressed
    /// one, or of a backing that lists `id` but holds no value for it.
    pub fn try_get(&self, id: Id) -> std::io::Result<Option<Value>> {
        self.try_get_decoded(id, &mut DecodedRows::default())
    }

    /// [`try_get`](Self::try_get), decoding the compressed integer or
    /// boolean rows into `decoded` the first time a read needs them and
    /// reusing them after: the reads of one call share one decode.
    fn try_get_decoded(&self, id: Id, decoded: &mut DecodedRows) -> std::io::Result<Option<Value>> {
        if let Some(value) = self.values.get(&id) {
            return Ok(Some(value.clone()));
        }
        self.stored_value(id, decoded)
    }

    /// The value of `id` the column holds outside its hot buffer: a
    /// compressed one, or one in the backing of a spilled column, unless it
    /// was removed.
    fn stored_value(&self, id: Id, decoded: &mut DecodedRows) -> std::io::Result<Option<Value>> {
        if (self.compressed.is_none() && self.backing.is_none()) || self.removed.contains(&id) {
            return Ok(None);
        }
        if let Some(value) = self.compressed_value(id, decoded)? {
            return Ok(Some(value));
        }
        match &self.backing {
            Some(backing) if backing.contains(id) => {
                backing.get(id)?.map(Some).ok_or_else(missing_backed_value)
            }
            _ => Ok(None),
        }
    }

    /// The compressed value of `id`, decoded (integer and boolean rows into
    /// `decoded`, if they are not there yet).
    fn compressed_value(
        &self,
        id: Id,
        decoded: &mut DecodedRows,
    ) -> std::io::Result<Option<Value>> {
        let Some(compressed) = &self.compressed else {
            return Ok(None);
        };
        let Ok(index) = compressed.ids().binary_search(&id.as_u64()) else {
            return Ok(None);
        };
        let value = match compressed {
            CompressedColumnData::Integers { data, .. } => decoded
                .integers(data)?
                .get(index)
                .map(|&value| Value::Int64(crate::codec::zigzag_decode(value))),
            CompressedColumnData::Strings { encoding, .. } => encoding
                .get(index)
                .map(|value| Value::String(ArcStr::from(value))),
            CompressedColumnData::Booleans { data, .. } => decoded
                .booleans(data)?
                .get(index)
                .map(|&value| Value::Bool(value)),
        };
        value.map(Some).ok_or_else(undecodable_compressed_row)
    }

    /// The compressed rows, decoded, in id order, removed ones included.
    fn decode_compressed(&self) -> std::io::Result<Vec<(Id, Value)>> {
        let Some(compressed) = &self.compressed else {
            return Ok(Vec::new());
        };
        let ids = compressed.ids();
        let values: Vec<Value> = match compressed {
            CompressedColumnData::Integers { data, .. } => {
                count_compressed_decode();
                TypeSpecificCompressor::decompress_integers(data)?
                    .into_iter()
                    .map(|value| Value::Int64(crate::codec::zigzag_decode(value)))
                    .collect()
            }
            CompressedColumnData::Strings { encoding, .. } => (0..ids.len())
                .map(|index| {
                    encoding
                        .get(index)
                        .map(|value| Value::String(ArcStr::from(value)))
                        .ok_or_else(undecodable_compressed_row)
                })
                .collect::<std::io::Result<_>>()?,
            CompressedColumnData::Booleans { data, .. } => {
                count_compressed_decode();
                TypeSpecificCompressor::decompress_booleans(data)?
                    .into_iter()
                    .map(Value::Bool)
                    .collect()
            }
        };
        if values.len() < ids.len() {
            return Err(undecodable_compressed_row());
        }
        Ok(ids
            .iter()
            .zip(values)
            .map(|(&id, value)| (Id::from_u64(id), value))
            .collect())
    }

    /// Removes a value for an entity, returning it. A compressed or spilled
    /// value is read before it is hidden, so the caller can log and undo the
    /// removal: when it cannot be read, nothing changes.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the value, as [`try_get`](Self::try_get).
    pub fn remove(&mut self, id: Id) -> std::io::Result<Option<Value>> {
        let removed = match self.values.remove(&id) {
            Some(value) => {
                self.hide_stored(id);
                Some(value)
            }
            None => {
                let value = self.stored_value(id, &mut DecodedRows::default())?;
                if value.is_some() {
                    self.removed.insert(id);
                }
                value
            }
        };
        if removed.is_some() {
            // Mark zone map as dirty - would need full rebuild for accurate min/max
            self.zone_map_dirty = true;
        }
        Ok(removed)
    }

    /// Removes a value for an entity without returning it, so a compressed
    /// or spilled value is not read.
    fn discard(&mut self, id: Id) {
        let from_values = self.values.remove(&id).is_some();
        let from_store = self.hide_stored(id);
        if from_values || from_store {
            self.zone_map_dirty = true;
        }
    }

    /// Hides the compressed or backed value of `id`, returning whether one
    /// was visible.
    fn hide_stored(&mut self, id: Id) -> bool {
        self.is_stored(id) && self.removed.insert(id)
    }

    /// Returns the ids with a value, in id order, each once: compressed and
    /// spilled values included.
    #[must_use]
    pub fn ids(&self) -> Vec<Id> {
        let mut ids: Vec<Id> = self.values.keys().copied().collect();
        if let Some(compressed) = &self.compressed {
            ids.extend(
                compressed
                    .ids()
                    .iter()
                    .map(|&id| Id::from_u64(id))
                    .filter(|id| !self.removed.contains(id)),
            );
        }
        if let Some(backing) = &self.backing {
            ids.extend(
                backing
                    .ids()
                    .into_iter()
                    .filter(|id| !self.removed.contains(id) && !self.values.contains_key(id)),
            );
        }
        ids.sort_unstable_by_key(|id| id.as_u64());
        ids.dedup();
        ids
    }

    /// Returns every `(id, value)`, in id order, compressed and spilled
    /// values included: a snapshot or a copy never leaves a value out.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value or decoding a compressed
    /// one, or of a backing that lists an id it holds no value for.
    pub fn try_entries(&self) -> std::io::Result<Vec<(Id, Value)>> {
        let mut entries: Vec<(Id, Value)> = self
            .values
            .iter()
            .map(|(id, value)| (*id, value.clone()))
            .collect();
        entries.extend(
            self.decode_compressed()?
                .into_iter()
                .filter(|(id, _)| !self.removed.contains(id) && !self.values.contains_key(id)),
        );
        if let Some(backing) = &self.backing {
            for id in backing.ids() {
                if self.removed.contains(&id) || self.values.contains_key(&id) {
                    continue;
                }
                let value = backing.get(id)?.ok_or_else(missing_backed_value)?;
                entries.push((id, value));
            }
        }
        entries.sort_unstable_by_key(|(id, _)| id.as_u64());
        // A backing that lists an id twice reads it twice.
        entries.dedup_by_key(|(id, _)| id.as_u64());
        Ok(entries)
    }

    /// Where the vector of `id` is: the column's own value first (a value
    /// that is not a vector gives `None`, as `get` would), then the backing.
    fn vector_source(&self, id: Id) -> Option<VectorSource<Id>> {
        if let Some(value) = self.values.get(&id) {
            return match value {
                Value::Vector(vector) => Some(VectorSource::Own(Arc::clone(vector))),
                _ => None,
            };
        }
        let backing = self.backing.as_ref()?;
        (!self.removed.contains(&id)).then(|| VectorSource::Backed(Arc::clone(backing)))
    }

    /// Hands the values over to `backing`, which holds the entries of
    /// `snapshot` it contains (see [`PropertyStorage::spill_column`]).
    /// Returns `false` when the column is spilled already, or compressed
    /// rows that do not decode.
    ///
    /// A compressed column is decompressed first: its rows are in the
    /// snapshot, so they leave the heap with the others, and a column is
    /// never compressed and spilled at once.
    fn spill(&mut self, backing: Arc<dyn ColumnBacking<Id>>, snapshot: &[(Id, Value)]) -> bool {
        if self.backing.is_some() {
            return false;
        }
        self.decompress_all();
        if self.compressed.is_some() {
            return false;
        }
        for (id, taken) in snapshot {
            // A value the backing does not hold (a vector file holds only
            // vectors) stays in the column.
            if !backing.contains(*id) {
                continue;
            }
            match self.values.get(id) {
                Some(current) if unchanged(current, taken) => {
                    self.values.remove(id);
                }
                // Written after the snapshot: newer than the backing.
                Some(_) => {}
                // Removed after the snapshot.
                None => {
                    self.removed.insert(*id);
                }
            }
        }
        self.values.shrink_to_fit();
        self.backing = Some(backing);
        true
    }

    /// Moves `entries`, read from the backing, back into the column, except
    /// values removed or written since the spill, and drops the backing.
    fn unspill(&mut self, entries: Vec<(Id, Value)>) {
        for (id, value) in entries {
            if !self.removed.contains(&id) {
                self.values.entry(id).or_insert(value);
            }
        }
        self.removed = FxHashSet::default();
        self.backing = None;
    }

    /// Returns the number of values in this column (hot + compressed +
    /// backed), each counted once.
    #[must_use]
    pub fn len(&self) -> usize {
        let stored = match (&self.compressed, &self.backing) {
            (None, None) => return self.values.len(),
            // A column is never compressed and spilled at once (see `spill`).
            (Some(_), _) => self.compressed_count,
            (None, Some(backing)) => backing.len(),
        };
        // Each removed id is a stored one (never a hot one), and a hot value
        // over a stored one counts once.
        let shadowing = self.values.keys().filter(|id| self.is_stored(**id)).count();
        // Saturating: a backing must not shrink while installed.
        self.values.len() - shadowing + stored.saturating_sub(self.removed.len())
    }

    /// Whether the column holds a value for `id` outside its hot buffer
    /// (compressed, or in the backing), removed or not.
    fn is_stored(&self, id: Id) -> bool {
        self.compressed
            .as_ref()
            .is_some_and(|compressed| compressed.ids().binary_search(&id.as_u64()).is_ok())
            || self
                .backing
                .as_ref()
                .is_some_and(|backing| backing.contains(id))
    }

    /// Returns true if this column is empty.
    #[cfg(test)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Returns estimated heap memory for this column.
    ///
    /// Includes the hot buffer hash map capacity, zone map, and any
    /// compressed data.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        // Hot buffer: FxHashMap<Id, Value> capacity
        let hot_bytes =
            self.values.capacity() * (std::mem::size_of::<Id>() + std::mem::size_of::<Value>() + 1);
        // Compressed data
        let compressed_bytes = self.compressed.as_ref().map_or(0, |c| c.memory_usage());
        // A spilled column's backing and its removed ids
        let backing_bytes = self
            .backing
            .as_ref()
            .map_or(0, |backing| backing.heap_bytes())
            + self.removed.capacity() * (std::mem::size_of::<Id>() + 1);
        // ZoneMapEntry is inline (no heap), so just hot + compressed + backing
        hot_bytes + compressed_bytes + backing_bytes
    }

    /// Returns whether the column has compressed data.
    #[must_use]
    #[cfg(test)]
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
    /// decompressing to re-compress with all values. A spilled column is
    /// not compressed either.
    pub fn compress(&mut self) {
        if self.values.is_empty() || self.backing.is_some() {
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
    fn compress_as_integers(&mut self) {
        // Extract integer values and their IDs
        let mut values: Vec<(u64, i64)> = Vec::new();
        let mut non_int_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::Int64(v) => {
                    let id_u64 = id.as_u64();
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
        let Ok(compressed) = TypeSpecificCompressor::compress_signed_integers(&int_values) else {
            return;
        };

        // Only use compression if it actually saves space
        if compressed.compression_ratio() > 1.2 {
            self.block_zone_maps =
                compute_block_zone_maps(int_values.iter().map(|v| Value::Int64(*v)));
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
    fn compress_as_strings(&mut self) {
        let mut values: Vec<(u64, ArcStr)> = Vec::new();
        let mut non_str_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::String(s) => {
                    values.push((id.as_u64(), s.clone()));
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
            self.block_zone_maps =
                compute_block_zone_maps(values.iter().map(|(_, s)| Value::String(s.clone())));
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
    fn compress_as_booleans(&mut self) {
        let mut values: Vec<(u64, bool)> = Vec::new();
        let mut non_bool_values: FxHashMap<Id, Value> = FxHashMap::default();

        for (&id, value) in &self.values {
            match value {
                Value::Bool(b) => {
                    values.push((id.as_u64(), *b));
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

        let Ok(compressed) = TypeSpecificCompressor::compress_booleans(&bool_values) else {
            return;
        };

        // Booleans always compress well (8x)
        self.block_zone_maps = compute_block_zone_maps(bool_values.iter().map(|b| Value::Bool(*b)));
        self.compressed = Some(CompressedColumnData::Booleans {
            data: compressed,
            id_to_index,
            index_to_id,
        });
        self.compressed_count = values.len();
        self.values = non_bool_values;
    }

    /// Decompresses all values back to the hot buffer. A value written over
    /// a compressed one wins, and a removed one stays removed. Rows that do
    /// not decode stay compressed rather than be dropped.
    fn decompress_all(&mut self) {
        if self.compressed.is_none() {
            return;
        }
        let Ok(rows) = self.decode_compressed() else {
            return;
        };
        for (id, value) in rows {
            if !self.removed.contains(&id) {
                self.values.entry(id).or_insert(value);
            }
        }
        // A compressed column has no backing (see `spill`), so every removed
        // id was a compressed one.
        self.removed = FxHashSet::default();
        self.compressed = None;
        self.compressed_count = 0;
        self.block_zone_maps.clear();
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

    /// Returns the per-block zone maps populated when the column was
    /// compressed.
    ///
    /// Each entry covers a contiguous slice of `DEFAULT_BLOCK_ROWS` rows
    /// of the sorted compressed array. Returns an empty slice when the
    /// column is uncompressed (the hot buffer is unordered, so per-block
    /// pruning would be meaningless). Phase 4 consumes these for lazy
    /// `range_iter`-style scans.
    #[must_use]
    pub fn block_zone_maps(&self) -> &[ZoneMapEntry] {
        &self.block_zone_maps
    }

    /// Returns the number of compressed blocks. Equal to
    /// `block_zone_maps().len()`. Returns 0 for uncompressed columns.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.block_zone_maps.len()
    }

    /// Decodes a single compressed block into `(id, value)` pairs.
    ///
    /// Returns `None` when the column is uncompressed or when `block_idx`
    /// is out of range. The block contains exactly the rows that
    /// [`block_zone_maps`](Self::block_zone_maps) at the same index
    /// describes (`row_count` entries).
    ///
    /// **Today** the implementation decodes the full compressed array
    /// once and slices the result; both calls are O(N). When Phase 5/6
    /// makes blocks independently decodable on-disk, the API contract
    /// stays the same and the implementation becomes per-block.
    #[must_use]
    pub fn decode_block(&self, block_idx: usize) -> Option<DecodedBlock<Id>> {
        let zone_map = self.block_zone_maps.get(block_idx)?.clone();
        let compressed = self.compressed.as_ref()?;

        let block_size = DEFAULT_BLOCK_ROWS as usize;
        let start = block_idx * block_size;
        let end = match self.compressed_count.min(start + block_size) {
            // Bounds-check: the last block may be short.
            n if n > start => n,
            _ => return None,
        };

        let entries = match compressed {
            CompressedColumnData::Integers {
                data, index_to_id, ..
            } => {
                let raw = TypeSpecificCompressor::decompress_integers(data).ok()?;
                let signed: Vec<i64> = raw
                    .iter()
                    .map(|&v| crate::codec::zigzag_decode(v))
                    .collect();
                index_to_id
                    .iter()
                    .zip(signed.iter())
                    .skip(start)
                    .take(end - start)
                    .map(|(&id_u64, &value)| (Id::from_u64(id_u64), Value::Int64(value)))
                    .collect()
            }
            CompressedColumnData::Strings {
                encoding,
                index_to_id,
                ..
            } => index_to_id
                .iter()
                .enumerate()
                .skip(start)
                .take(end - start)
                .filter_map(|(i, &id_u64)| {
                    encoding
                        .get(i)
                        .map(|s| (Id::from_u64(id_u64), Value::String(ArcStr::from(s))))
                })
                .collect(),
            CompressedColumnData::Booleans {
                data, index_to_id, ..
            } => {
                let raw = TypeSpecificCompressor::decompress_booleans(data).ok()?;
                index_to_id
                    .iter()
                    .zip(raw.iter())
                    .skip(start)
                    .take(end - start)
                    .map(|(&id_u64, &value)| (Id::from_u64(id_u64), Value::Bool(value)))
                    .collect()
            }
        };

        Some(DecodedBlock { zone_map, entries })
    }

    /// Iterates all compressed blocks, decoding each in turn.
    ///
    /// Empty for uncompressed columns. Phase 4's iterator-bounds operator
    /// will prefer `decode_block(idx)` after pruning via
    /// [`block_zone_maps`](Self::block_zone_maps); this iterator is the
    /// "decode everything" fallback.
    pub fn iter_decoded_blocks(&self) -> impl Iterator<Item = DecodedBlock<Id>> + '_ {
        (0..self.block_count()).filter_map(|idx| self.decode_block(idx))
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

    /// Rebuilds zone map from current values, compressed ones included.
    ///
    /// A spilled column keeps its zone map: it still covers the values only
    /// the backing holds, which a rebuild from the heap would drop. So does a
    /// column whose compressed rows do not decode.
    pub fn rebuild_zone_map(&mut self) {
        if self.backing.is_some() {
            return;
        }
        let Ok(compressed) = self.decode_compressed() else {
            return;
        };
        let mut zone_map = ZoneMapEntry::new();

        let stored = compressed
            .iter()
            .filter(|(id, _)| !self.removed.contains(id) && !self.values.contains_key(id))
            .map(|(_, value)| value);
        for value in self.values.values().chain(stored) {
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

// === Temporal implementation: VersionLog-backed property column ===
//
// **Zone map limitation**: zone maps track min/max across the *latest* values
// only (see `rebuild_zone_map`). For temporal queries at old epochs, the zone
// map may produce false negatives: it could reject a column based on current
// min/max even though historical values would match. This is a known
// trade-off: temporal queries are conservative but never return wrong results
// (the `zone_map_dirty` fallback returns `true` = "might match").
//
// **Compression**: disabled in temporal mode because the underlying codecs
// (DeltaBitPacked, Dictionary, BitVector) operate on flat `FxHashMap<Id, Value>`
// arrays, not `FxHashMap<Id, VersionLog<Value>>`. Per-epoch compression is a
// potential future optimization.
#[cfg(feature = "temporal")]
impl<Id: EntityId> PropertyColumn<Id> {
    /// Creates a new empty column.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            gc_candidates: FxHashSet::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: CompressionMode::None,
            block_zone_maps: Vec::new(),
        }
    }

    /// Creates a new column with the specified compression mode.
    #[must_use]
    pub fn with_compression(mode: CompressionMode) -> Self {
        Self {
            values: FxHashMap::default(),
            gc_candidates: FxHashSet::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
            compression_mode: mode,
            block_zone_maps: Vec::new(),
        }
    }

    /// Sets the compression mode for this column.
    pub fn set_compression_mode(&mut self, mode: CompressionMode) {
        self.compression_mode = mode;
    }

    /// Returns the compression mode for this column.
    #[must_use]
    pub fn compression_mode(&self) -> CompressionMode {
        self.compression_mode
    }

    /// Sets a value for an entity, appending to its version log.
    ///
    /// For non-transactional writes, pass the current epoch.
    /// For transactional writes, pass `EpochId::PENDING`.
    pub fn set(&mut self, id: Id, value: Value, epoch: EpochId) {
        self.update_zone_map_on_insert(&value);
        self.append(id, epoch, value);
    }

    /// Appends to an entity's log, noting it for garbage collection once it
    /// holds an older version.
    fn append(&mut self, id: Id, epoch: EpochId, value: Value) {
        let log = self.values.entry(id).or_default();
        log.append(epoch, value);
        if log.len() > 1 {
            self.gc_candidates.insert(id);
        }
    }

    /// Updates zone map when inserting a value.
    fn update_zone_map_on_insert(&mut self, value: &Value) {
        self.zone_map.row_count += 1;

        if matches!(value, Value::Null) {
            self.zone_map.null_count += 1;
            return;
        }

        match &self.zone_map.min {
            None => self.zone_map.min = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Less) {
                    self.zone_map.min = Some(value.clone());
                }
            }
        }

        match &self.zone_map.max {
            None => self.zone_map.max = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Greater) {
                    self.zone_map.max = Some(value.clone());
                }
            }
        }
    }

    /// Gets the latest value for an entity, filtering out tombstones (Null).
    #[must_use]
    pub fn get(&self, id: Id) -> Option<Value> {
        self.values
            .get(&id)
            .and_then(|log| log.latest())
            .filter(|v| !v.is_null())
            .cloned()
    }

    /// Returns the ids with a live value, in id order.
    #[must_use]
    pub fn ids(&self) -> Vec<Id> {
        let mut ids: Vec<Id> = self
            .values
            .iter()
            .filter(|(_, log)| log.latest().is_some_and(|v| !v.is_null()))
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable_by_key(|id| id.as_u64());
        ids
    }

    /// Returns every `(id, latest value)`, in id order.
    #[must_use]
    pub fn entries(&self) -> Vec<(Id, Value)> {
        self.ids()
            .into_iter()
            .filter_map(|id| self.get(id).map(|value| (id, value)))
            .collect()
    }

    /// [`get`](Self::get); a temporal column has no backing, so it never
    /// fails.
    ///
    /// # Errors
    ///
    /// None.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the same signature as the non-temporal column's fallible read"
    )]
    pub fn try_get(&self, id: Id) -> std::io::Result<Option<Value>> {
        Ok(self.get(id))
    }

    /// [`try_get`](Self::try_get): a temporal column is never compressed, so
    /// a batch read has nothing to decode once.
    fn try_get_decoded(
        &self,
        id: Id,
        _decoded: &mut DecodedRows,
    ) -> std::io::Result<Option<Value>> {
        self.try_get(id)
    }

    /// [`entries`](Self::entries); never fails.
    ///
    /// # Errors
    ///
    /// None.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the same signature as the non-temporal column's fallible read"
    )]
    pub fn try_entries(&self) -> std::io::Result<Vec<(Id, Value)>> {
        Ok(self.entries())
    }

    /// The latest vector stored for `id`.
    fn vector(&self, id: Id) -> Option<Arc<[f32]>> {
        match self.values.get(&id).and_then(|log| log.latest()) {
            Some(Value::Vector(vector)) => Some(Arc::clone(vector)),
            _ => None,
        }
    }

    /// Removes a value by appending a tombstone (Null) at the given epoch.
    pub fn remove(&mut self, id: Id, epoch: EpochId) -> Option<Value> {
        let previous = self.get(id);
        if previous.is_some() {
            self.append(id, epoch, Value::Null);
            self.zone_map_dirty = true;
        }
        previous
    }

    /// Returns the number of live (non-tombstoned) values in this column.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values
            .values()
            .filter(|log| log.latest().is_some_and(|v| !v.is_null()))
            .count()
    }

    /// Returns true if this column is empty.
    #[cfg(test)]
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns compression statistics for this column.
    ///
    /// In temporal mode, compression is not used. Reports live value count only.
    #[must_use]
    pub fn compression_stats(&self) -> CompressionStats {
        let live_count = self.len();
        let hot_size = live_count * std::mem::size_of::<Value>();

        CompressionStats {
            uncompressed_size: hot_size,
            compressed_size: hot_size,
            value_count: live_count,
            codec: None,
        }
    }

    /// Returns estimated heap memory for this column.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.values.capacity()
            * (std::mem::size_of::<Id>() + std::mem::size_of::<VersionLog<Value>>() + 1)
    }

    /// Compression is not supported in temporal mode (no-op).
    pub fn compress(&mut self) {}

    /// Forces compression (no-op in temporal mode).
    pub fn force_compress(&mut self) {}

    /// Returns the zone map for this column.
    #[must_use]
    pub fn zone_map(&self) -> &ZoneMapEntry {
        &self.zone_map
    }

    /// Returns the per-block zone maps for this column.
    ///
    /// Always empty in temporal mode: compression is disabled for
    /// `VersionLog`-backed columns (see module-level note), so there is
    /// no sorted compressed array to chunk into blocks.
    #[must_use]
    pub fn block_zone_maps(&self) -> &[ZoneMapEntry] {
        &self.block_zone_maps
    }

    /// Uses zone map to check if any values could satisfy the predicate.
    #[must_use]
    pub fn might_match(&self, op: CompareOp, value: &Value) -> bool {
        if self.zone_map_dirty {
            return true;
        }

        match op {
            CompareOp::Eq => self.zone_map.might_contain_equal(value),
            CompareOp::Ne => match (&self.zone_map.min, &self.zone_map.max) {
                (Some(min), Some(max)) => {
                    !(compare_values(min, value) == Some(Ordering::Equal)
                        && compare_values(max, value) == Some(Ordering::Equal))
                }
                _ => true,
            },
            CompareOp::Lt => self.zone_map.might_contain_less_than(value, false),
            CompareOp::Le => self.zone_map.might_contain_less_than(value, true),
            CompareOp::Gt => self.zone_map.might_contain_greater_than(value, false),
            CompareOp::Ge => self.zone_map.might_contain_greater_than(value, true),
        }
    }

    /// Rebuilds zone map from current (latest) values.
    pub fn rebuild_zone_map(&mut self) {
        let mut zone_map = ZoneMapEntry::new();

        for log in self.values.values() {
            if let Some(value) = log.latest() {
                zone_map.row_count += 1;

                if matches!(value, Value::Null) {
                    zone_map.null_count += 1;
                    continue;
                }

                match &zone_map.min {
                    None => zone_map.min = Some(value.clone()),
                    Some(current) => {
                        if compare_values(value, current) == Some(Ordering::Less) {
                            zone_map.min = Some(value.clone());
                        }
                    }
                }

                match &zone_map.max {
                    None => zone_map.max = Some(value.clone()),
                    Some(current) => {
                        if compare_values(value, current) == Some(Ordering::Greater) {
                            zone_map.max = Some(value.clone());
                        }
                    }
                }
            }
        }

        self.zone_map = zone_map;
        self.zone_map_dirty = false;
    }

    // === Temporal-only methods ===

    /// Gets the value at a specific epoch via binary search, filtering tombstones.
    #[must_use]
    pub fn get_at(&self, id: Id, epoch: EpochId) -> Option<Value> {
        self.values
            .get(&id)
            .and_then(|log| log.at(epoch))
            .filter(|v| !v.is_null())
            .cloned()
    }

    /// Replaces PENDING epochs with the real commit epoch in all version logs.
    pub fn finalize_pending(&mut self, real_epoch: EpochId) {
        for log in self.values.values_mut() {
            log.finalize_pending(real_epoch);
        }
    }

    /// Removes all PENDING entries from all version logs (transaction rollback).
    pub fn remove_pending(&mut self) {
        for log in self.values.values_mut() {
            log.remove_pending();
        }
        self.values.retain(|_, log| !log.is_empty());
    }

    /// Garbage-collects old versions, visiting only the logs that hold more
    /// than one entry. A log keeps the version visible at `min_epoch` and
    /// every later one, and stays a candidate while it has more than one.
    /// Returns how many versions it dropped.
    pub fn gc(&mut self, min_epoch: EpochId) -> usize {
        let candidates = std::mem::take(&mut self.gc_candidates);
        let mut dropped = 0;
        for id in candidates {
            let Some(log) = self.values.get_mut(&id) else {
                continue;
            };
            dropped += log.gc(min_epoch);
            if log.is_empty() {
                self.values.remove(&id);
            } else if log.len() > 1 {
                self.gc_candidates.insert(id);
            }
        }
        dropped
    }

    /// Removes up to `n` PENDING entries for a specific entity.
    ///
    /// Used by savepoint rollback to pop only the entries added after the
    /// savepoint, leaving earlier PENDING entries intact.
    pub fn pop_n_pending_for(&mut self, id: Id, n: usize) {
        if let Some(log) = self.values.get_mut(&id) {
            log.pop_n_pending(n);
            if log.is_empty() {
                self.values.remove(&id);
            }
        }
    }

    /// Replaces PENDING epochs with the commit epoch for one entity (commit
    /// of the transaction that wrote them).
    pub fn finalize_pending_for(&mut self, id: Id, real_epoch: EpochId) {
        if let Some(log) = self.values.get_mut(&id) {
            log.finalize_pending(real_epoch);
        }
    }

    /// Removes an entity's value and its history.
    pub fn purge(&mut self, id: Id) {
        self.values.remove(&id);
    }
}

/// Whether `current` is still the value `taken` from the column earlier: the
/// same vector allocation, or an equal value.
#[cfg(not(feature = "temporal"))]
fn unchanged(current: &Value, taken: &Value) -> bool {
    match (current, taken) {
        (Value::Vector(current), Value::Vector(taken)) => {
            Arc::ptr_eq(current, taken) || current == taken
        }
        _ => current == taken,
    }
}

/// Computes per-block zone maps for a sorted column by chunking the values
/// into blocks of [`DEFAULT_BLOCK_ROWS`] rows.
///
/// Used by `compress_as_*` to populate `PropertyColumn::block_zone_maps`.
/// The values must already be in the order in which they will be stored
/// (sorted by entity id today). Each block records min/max/null/row counts;
/// `Float64` and other variants without a defined `Ord` impl skip min/max
/// updates and contribute only to row/null counts (matching the column-
/// level zone map's behavior).
#[cfg(not(feature = "temporal"))]
fn compute_block_zone_maps(values: impl IntoIterator<Item = Value>) -> Vec<ZoneMapEntry> {
    let block_size = DEFAULT_BLOCK_ROWS as usize;
    let mut blocks: Vec<ZoneMapEntry> = Vec::new();
    let mut current = ZoneMapEntry::new();
    let mut current_rows: usize = 0;

    for value in values {
        if current_rows == block_size {
            blocks.push(current);
            current = ZoneMapEntry::new();
            current_rows = 0;
        }
        current.row_count += 1;
        current_rows += 1;

        if matches!(value, Value::Null) {
            current.null_count += 1;
            continue;
        }

        // Reflexive-comparison guard: only seed/update min/max with values
        // that have a defined ordering against themselves. This filters out
        // `Float64(NaN)` (and any future variant whose `compare_values` arm
        // returns `None`) so a NaN can never poison the running min/max.
        if compare_values(&value, &value) != Some(Ordering::Equal) {
            continue;
        }

        let is_less_than_min = match &current.min {
            None => true,
            Some(existing) => compare_values(&value, existing) == Some(Ordering::Less),
        };
        let is_greater_than_max = match &current.max {
            None => true,
            Some(existing) => compare_values(&value, existing) == Some(Ordering::Greater),
        };
        if is_less_than_min {
            current.min = Some(value.clone());
        }
        if is_greater_than_max {
            current.max = Some(value);
        }
    }

    if current_rows > 0 {
        blocks.push(current);
    }
    blocks
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
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        // Zoned datetimes, also against a timestamp: by their instant, as a
        // filter compares them, so zone maps prune them right.
        _ => a.compare_instants(b),
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
    _key: PropertyKey,
    _marker: PhantomData<Id>,
}

#[cfg(test)]
#[cfg(not(feature = "temporal"))]
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

        storage.set(node1, name_key.clone(), "Alix".into());
        storage.set(node1, age_key.clone(), 30i64.into());
        storage.set(node2, name_key.clone(), "Gus".into());

        assert_eq!(
            storage.get(node1, &name_key),
            Some(Value::String("Alix".into()))
        );
        assert_eq!(storage.get(node1, &age_key), Some(Value::Int64(30)));
        assert_eq!(
            storage.get(node2, &name_key),
            Some(Value::String("Gus".into()))
        );
        assert!(storage.get(node2, &age_key).is_none());
    }

    #[test]
    fn test_property_storage_remove() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        let key = PropertyKey::new("name");

        storage.set(node, key.clone(), "Alix".into());
        assert!(storage.get(node, &key).is_some());

        let removed = storage.remove(node, &key).unwrap();
        assert!(removed.is_some());
        assert!(storage.get(node, &key).is_none());
    }

    #[test]
    fn test_property_storage_get_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alix".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());
        storage.set(node, PropertyKey::new("active"), true.into());

        let props = storage.get_all(node);
        assert_eq!(props.len(), 3);
    }

    #[test]
    fn test_property_storage_remove_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alix".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());

        storage.remove_all(node);

        assert!(storage.get(node, &PropertyKey::new("name")).is_none());
        assert!(storage.get(node, &PropertyKey::new("age")).is_none());
    }

    #[test]
    fn test_property_column() {
        let mut col = PropertyColumn::new();

        col.set(NodeId::new(1), "Alix".into());
        col.set(NodeId::new(2), "Gus".into());

        assert_eq!(col.len(), 2);
        assert!(!col.is_empty());

        assert_eq!(col.get(NodeId::new(1)), Some(Value::String("Alix".into())));

        assert_eq!(
            col.remove(NodeId::new(1)).unwrap(),
            Some(Value::String("Alix".into()))
        );
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

        for i in 0u64..100 {
            let age = 20 + i64::try_from(i % 50).unwrap();
            storage.set(NodeId::new(i), PropertyKey::new("age"), Value::Int64(age));
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
        for i in 0u64..2000 {
            col.set(
                NodeId::new(i),
                Value::Int64(1000 + i64::try_from(i).unwrap()),
            );
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
        for i in 0u64..2000 {
            col.set(NodeId::new(i), Value::Bool(i % 2 == 0));
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
        for i in 0u64..100 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
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

        for i in 0u64..50 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }

        let stats = col.compression_stats();
        assert_eq!(stats.value_count, 50);
        assert!(stats.uncompressed_size > 0);
    }

    #[test]
    fn test_storage_compression_stats() {
        let storage = PropertyStorage::with_compression(CompressionMode::Auto);

        for i in 0u64..100 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("age"),
                Value::Int64(i64::try_from(i).unwrap()),
            );
            storage.set(
                NodeId::new(i),
                PropertyKey::new("name"),
                Value::String(ArcStr::from("Alix")),
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

        for i in 0u64..100 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("value"),
                Value::Int64(i64::try_from(i).unwrap()),
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
        assert_eq!(values, Vec::<Option<Value>>::new());
    }

    #[test]
    fn test_get_all_batch() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let node3 = NodeId::new(3);

        storage.set(node1, PropertyKey::new("name"), "Alix".into());
        storage.set(node1, PropertyKey::new("age"), 25i64.into());
        storage.set(node2, PropertyKey::new("name"), "Gus".into());
        // node3 has no properties

        let ids = vec![node1, node2, node3];
        let all_props = storage.get_all_batch(&ids);

        assert_eq!(all_props.len(), 3);
        assert_eq!(all_props[0].len(), 2); // name and age
        assert_eq!(all_props[1].len(), 1); // name only
        assert_eq!(all_props[2].len(), 0); // no properties

        assert_eq!(
            all_props[0].get(&PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(
            all_props[1].get(&PropertyKey::new("name")),
            Some(&Value::String("Gus".into()))
        );
    }

    #[test]
    fn test_get_all_batch_empty_ids() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();

        let all_props = storage.get_all_batch(&[]);
        assert_eq!(all_props, Vec::<FxHashMap<PropertyKey, Value>>::new());
    }

    // ── Phase 2d: per-block zone maps ─────────────────────────────────

    #[test]
    fn test_block_zone_maps_empty_for_uncompressed_column() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..50 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        assert!(col.block_zone_maps().is_empty());
    }

    #[test]
    fn test_block_zone_maps_integer_compressed() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        // 2500 sequential integers → 1024 + 1024 + 452 = 3 blocks at default size
        for i in 0u64..2500 {
            col.set(
                NodeId::new(i),
                Value::Int64(1000 + i64::try_from(i).unwrap()),
            );
        }
        col.force_compress();

        let blocks = col.block_zone_maps();
        assert_eq!(blocks.len(), 3, "2500 rows / 1024 = 3 blocks");

        assert_eq!(blocks[0].row_count, 1024);
        assert_eq!(blocks[0].min, Some(Value::Int64(1000)));
        assert_eq!(blocks[0].max, Some(Value::Int64(2023)));

        assert_eq!(blocks[1].row_count, 1024);
        assert_eq!(blocks[1].min, Some(Value::Int64(2024)));
        assert_eq!(blocks[1].max, Some(Value::Int64(3047)));

        assert_eq!(blocks[2].row_count, 452);
        assert_eq!(blocks[2].min, Some(Value::Int64(3048)));
        assert_eq!(blocks[2].max, Some(Value::Int64(3499)));
    }

    #[test]
    fn test_block_zone_maps_string_compressed() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        // Low-cardinality cycle (good for dictionary compression). Every
        // block contains all four strings, so per-block min/max match the
        // overall range, but row counts still segment into 1024+1024+rest.
        let strings = ["alpha", "bravo", "charlie", "delta"];
        for i in 0u64..2500 {
            col.set(
                NodeId::new(i),
                Value::String(ArcStr::from(strings[(i % 4) as usize])),
            );
        }
        col.force_compress();

        let blocks = col.block_zone_maps();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].row_count, 1024);
        assert_eq!(blocks[1].row_count, 1024);
        assert_eq!(blocks[2].row_count, 452);
        for block in blocks {
            assert_eq!(block.min, Some(Value::String(ArcStr::from("alpha"))));
            assert_eq!(block.max, Some(Value::String(ArcStr::from("delta"))));
        }
    }

    #[test]
    fn test_block_zone_maps_boolean_compressed() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        // 2500 alternating bools sorted by id: every block contains both true and false.
        for i in 0u64..2500 {
            col.set(NodeId::new(i), Value::Bool(i % 2 == 0));
        }
        col.force_compress();

        let blocks = col.block_zone_maps();
        assert_eq!(blocks.len(), 3);
        for block in blocks {
            assert_eq!(block.min, Some(Value::Bool(false)));
            assert_eq!(block.max, Some(Value::Bool(true)));
            assert_eq!(block.null_count, 0);
        }
    }

    #[test]
    fn test_block_might_match_prunes_disjoint_range() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(
                NodeId::new(i),
                Value::Int64(1000 + i64::try_from(i).unwrap()),
            );
        }
        col.force_compress();

        // Block 0 covers values 1000..=2023; querying for 5000 must prune all blocks.
        let target = Value::Int64(5000);
        let blocks = col.block_zone_maps();
        let any_match = blocks.iter().any(|zm| zm.might_contain_equal(&target));
        assert!(!any_match, "no block should claim to contain 5000");
    }

    #[test]
    fn test_storage_block_zone_maps_for_returns_blocks() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();
        for i in 0u64..2500 {
            storage.set(
                NodeId::new(i),
                PropertyKey::new("age"),
                Value::Int64(20 + i64::try_from(i).unwrap()),
            );
        }
        storage.force_compress_all();

        let blocks = storage
            .block_zone_maps_for(&PropertyKey::new("age"))
            .expect("compressed column must expose block stats");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].min, Some(Value::Int64(20)));
        assert_eq!(blocks.last().unwrap().max, Some(Value::Int64(2519)));
    }

    #[test]
    fn test_storage_block_zone_maps_for_missing_column_returns_none() {
        let storage: PropertyStorage<NodeId> = PropertyStorage::new();
        assert!(
            storage
                .block_zone_maps_for(&PropertyKey::new("missing"))
                .is_none()
        );
    }

    #[test]
    fn test_compute_block_zone_maps_float64_finite_min_max() {
        // Direct test of the helper: Float64 isn't dispatched by compress_as_*
        // today, but the helper must handle Float64 correctly so a future
        // compression path can feed Float64 streams without a behavior change.
        let values: Vec<Value> = (0u32..2500)
            .map(|i| Value::Float64(f64::from(i) * 0.5))
            .collect();
        let blocks = compute_block_zone_maps(values);

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].row_count, 1024);
        assert_eq!(blocks[0].min, Some(Value::Float64(0.0)));
        assert_eq!(blocks[0].max, Some(Value::Float64(1023.0 * 0.5)));
        assert_eq!(blocks[1].min, Some(Value::Float64(1024.0 * 0.5)));
    }

    // ── Phase 2e: vectorized batch decoders ──────────────────────────

    #[test]
    fn test_block_count_zero_for_uncompressed_column() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..50 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        assert_eq!(col.block_count(), 0);
    }

    #[test]
    fn test_block_count_matches_zone_maps_after_compression() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        col.force_compress();
        assert_eq!(col.block_count(), col.block_zone_maps().len());
        assert_eq!(col.block_count(), 3);
    }

    #[test]
    fn test_decode_block_returns_correct_pairs_integer() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(
                NodeId::new(i),
                Value::Int64(1000 + i64::try_from(i).unwrap()),
            );
        }
        col.force_compress();

        let block0 = col.decode_block(0).expect("block 0 must exist");
        assert_eq!(block0.entries.len(), 1024);
        assert_eq!(block0.entries[0], (NodeId::new(0), Value::Int64(1000)));
        assert_eq!(
            block0.entries[1023],
            (NodeId::new(1023), Value::Int64(2023))
        );
        assert_eq!(block0.zone_map.row_count, 1024);

        let block2 = col.decode_block(2).expect("block 2 must exist");
        assert_eq!(block2.entries.len(), 452);
        assert_eq!(block2.entries[0], (NodeId::new(2048), Value::Int64(3048)));
    }

    #[test]
    fn test_decode_block_returns_correct_pairs_boolean() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(NodeId::new(i), Value::Bool(i % 2 == 0));
        }
        col.force_compress();

        let block0 = col.decode_block(0).expect("block 0 must exist");
        assert_eq!(block0.entries.len(), 1024);
        assert_eq!(block0.entries[0], (NodeId::new(0), Value::Bool(true)));
        assert_eq!(block0.entries[1], (NodeId::new(1), Value::Bool(false)));
    }

    #[test]
    fn test_decode_block_returns_correct_pairs_string() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        let strings = ["alpha", "bravo", "charlie", "delta"];
        for i in 0u64..2500 {
            col.set(
                NodeId::new(i),
                Value::String(ArcStr::from(strings[(i % 4) as usize])),
            );
        }
        col.force_compress();

        let block0 = col.decode_block(0).expect("block 0 must exist");
        assert_eq!(block0.entries.len(), 1024);
        assert_eq!(
            block0.entries[0],
            (NodeId::new(0), Value::String(ArcStr::from("alpha")))
        );
        assert_eq!(
            block0.entries[3],
            (NodeId::new(3), Value::String(ArcStr::from("delta")))
        );
    }

    #[test]
    fn test_decode_block_out_of_range_returns_none() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        col.force_compress();

        assert!(col.decode_block(99).is_none());
    }

    #[test]
    fn test_decode_block_uncompressed_returns_none() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..50 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        // No force_compress — column is uncompressed.
        assert!(col.decode_block(0).is_none());
    }

    #[test]
    fn test_iter_decoded_blocks_yields_all_blocks() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..2500 {
            col.set(NodeId::new(i), Value::Int64(i64::try_from(i).unwrap()));
        }
        col.force_compress();

        let blocks: Vec<_> = col.iter_decoded_blocks().collect();
        assert_eq!(blocks.len(), 3);
        let total_rows: usize = blocks.iter().map(|b| b.entries.len()).sum();
        assert_eq!(total_rows, 2500);
    }

    #[test]
    fn test_compute_block_zone_maps_float64_nan_does_not_poison() {
        // NaN must never seed or displace min/max: comparisons against NaN
        // return None, which would otherwise leave min/max permanently
        // unrecoverable. Reflexive-comparison guard catches this.
        let mut values: Vec<Value> = vec![Value::Float64(f64::NAN)];
        values.extend((0u32..50).map(|i| Value::Float64(f64::from(i))));
        values.push(Value::Float64(f64::NAN));

        let blocks = compute_block_zone_maps(values);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].row_count, 52, "NaN values still count as rows");
        assert_eq!(blocks[0].null_count, 0, "NaN is not null");
        assert_eq!(blocks[0].min, Some(Value::Float64(0.0)));
        assert_eq!(blocks[0].max, Some(Value::Float64(49.0)));
    }

    // ── A column backed by a spill file (#594) ─────────────────────

    use super::test_backing::MemoryBacking;
    use std::sync::Arc;
    use std::sync::atomic::Ordering as AtomicOrdering;

    fn vector(values: &[f32]) -> Value {
        Value::Vector(values.into())
    }

    /// A storage whose `embedding` column holds `entries` and is spilled
    /// into a [`MemoryBacking`], which is returned too.
    fn spilled(entries: &[(NodeId, Value)]) -> (PropertyStorage, PropertyKey, Arc<MemoryBacking>) {
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("embedding");
        for (id, value) in entries {
            storage.set(*id, key.clone(), value.clone());
        }
        let snapshot = storage.try_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(storage.spill_column(&key, backing.clone(), &snapshot));
        (storage, key, backing)
    }

    fn overlay_ids(storage: &PropertyStorage, key: &PropertyKey) -> Vec<u64> {
        let columns = storage.columns.read();
        let mut ids: Vec<u64> = columns[key].values.keys().map(|id| id.0).collect();
        ids.sort_unstable();
        ids
    }

    fn ints(raw: &[(u64, i64)]) -> Vec<(NodeId, Value)> {
        raw.iter()
            .map(|&(id, value)| (NodeId::new(id), Value::Int64(value)))
            .collect()
    }

    /// Every read path returns a spilled value; the column keeps none of
    /// them on the heap.
    #[test]
    fn a_spilled_column_reads_through_its_backing() {
        let (alix, gus, vincent) = (NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let (storage, key, _backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);
        storage.set(vincent, PropertyKey::new("name"), "Vincent".into());

        assert_eq!(storage.spilled_columns(), vec![key.clone()]);
        assert_eq!(overlay_ids(&storage, &key), Vec::<u64>::new());
        assert_eq!(storage.get(alix, &key), Some(vector(&[3.0, 19.0])));
        assert_eq!(
            storage.get_batch(&[gus, vincent, alix], &key),
            vec![
                Some(vector(&[88.0, 3.19])),
                None,
                Some(vector(&[3.0, 19.0]))
            ]
        );
        assert_eq!(
            storage.try_get_batch(&[gus, vincent], &key).unwrap(),
            vec![Some(vector(&[88.0, 3.19])), None]
        );
        assert_eq!(storage.get_all(gus).get(&key), Some(&vector(&[88.0, 3.19])));
        assert_eq!(
            storage.try_get_all(alix).unwrap().get(&key),
            Some(&vector(&[3.0, 19.0]))
        );
        assert_eq!(
            storage.get_all_batch(&[alix])[0].get(&key),
            Some(&vector(&[3.0, 19.0]))
        );
        assert_eq!(
            storage.get_selective_batch(&[gus], std::slice::from_ref(&key))[0].get(&key),
            Some(&vector(&[88.0, 3.19]))
        );
        assert_eq!(storage.column_ids(&key), vec![alix, gus]);
        assert_eq!(storage.try_column_entries(&key).unwrap().len(), 2);
        assert_eq!(storage.try_column_entries(&key).unwrap().len(), 2);
    }

    /// A value written while the column is spilled wins over the spilled one,
    /// before and after the reload.
    #[test]
    fn a_write_while_spilled_wins_over_the_backing() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let (storage, key, _backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);

        storage.set(alix, key.clone(), vector(&[319.0, 1988.0]));
        assert_eq!(storage.get(alix, &key), Some(vector(&[319.0, 1988.0])));
        assert_eq!(storage.column_ids(&key), vec![alix, gus]);

        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(storage.spilled_columns(), Vec::<PropertyKey>::new());
        assert_eq!(storage.get(alix, &key), Some(vector(&[319.0, 1988.0])));
        assert_eq!(storage.get(gus, &key), Some(vector(&[88.0, 3.19])));
    }

    /// Removing a spilled value hides it, returns it (a change set records
    /// it), and it stays removed after the reload: the bug where a reload
    /// brought it back.
    #[test]
    fn a_removal_while_spilled_stays_removed() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let (storage, key, _backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);

        assert_eq!(
            storage.remove(alix, &key).unwrap(),
            Some(vector(&[3.0, 19.0]))
        );
        assert_eq!(storage.remove(alix, &key).unwrap(), None);
        assert_eq!(storage.get(alix, &key), None);
        assert!(!storage.get_all(alix).contains_key(&key));
        assert_eq!(storage.column_ids(&key), vec![gus]);

        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(storage.get(alix, &key), None);
        assert_eq!(storage.column_ids(&key), vec![gus]);
    }

    /// `remove_all` (a node delete) and `purge` (a rolled back create) hide
    /// spilled values too.
    #[test]
    fn remove_all_and_purge_hide_spilled_values() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let (storage, key, _backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);

        storage.remove_all(alix);
        storage.purge(gus);
        assert_eq!(storage.get_batch(&[alix, gus], &key), vec![None, None]);
        assert_eq!(storage.column_ids(&key), Vec::<NodeId>::new());
        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(storage.column_ids(&key), Vec::<NodeId>::new());
    }

    /// The spill writes its file from a snapshot without holding the lock;
    /// what changed in between wins: a newer value stays in the column, a
    /// removed value stays removed, and only unchanged values leave the heap.
    #[test]
    fn changes_between_the_snapshot_and_the_spill_are_kept() {
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("embedding");
        let ids: Vec<NodeId> = (1..=5).map(NodeId::new).collect();
        for (id, x) in ids.iter().zip([3.0, 19.0, 88.0, 319.0]) {
            storage.set(*id, key.clone(), vector(&[x, 3.19]));
        }
        let snapshot = storage.try_column_entries(&key).unwrap();
        assert_eq!(snapshot.len(), 4);

        storage.set(ids[0], key.clone(), vector(&[1988.0, 1988.0]));
        storage.remove(ids[1], &key).unwrap();
        storage.set(ids[2], key.clone(), snapshot[2].1.clone());
        storage.set(ids[4], key.clone(), vector(&[3.19, 3.19]));
        assert!(storage.spill_column(&key, MemoryBacking::of(&snapshot), &snapshot));

        assert_eq!(overlay_ids(&storage, &key), vec![1, 5]);
        assert_eq!(
            storage.get_batch(&ids, &key),
            vec![
                Some(vector(&[1988.0, 1988.0])),
                None,
                Some(vector(&[88.0, 3.19])),
                Some(vector(&[319.0, 3.19])),
                Some(vector(&[3.19, 3.19])),
            ]
        );
        assert_eq!(
            storage.column_ids(&key),
            vec![ids[0], ids[2], ids[3], ids[4]]
        );
    }

    /// A value the backing does not hold (a vector file holds only vectors)
    /// stays in the column instead of leaving the heap for nowhere.
    #[test]
    fn values_the_backing_does_not_hold_stay_in_the_column() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("embedding");
        storage.set(alix, key.clone(), vector(&[3.0, 19.0]));
        storage.set(gus, key.clone(), "not a vector".into());
        let snapshot = storage.try_column_entries(&key).unwrap();
        let vectors: Vec<(NodeId, Value)> = snapshot
            .iter()
            .filter(|(_, value)| matches!(value, Value::Vector(_)))
            .cloned()
            .collect();
        assert!(storage.spill_column(&key, MemoryBacking::of(&vectors), &snapshot));
        assert_eq!(storage.get(gus, &key), Some(Value::from("not a vector")));
        assert_eq!(storage.column_ids(&key), vec![alix, gus]);
    }

    /// `column_ids` lists each id once, in id order, whether its value is in
    /// the column, in the backing or in both.
    #[test]
    fn column_ids_come_in_id_order_without_duplicates() {
        let ids = |raw: &[u64]| -> Vec<NodeId> { raw.iter().copied().map(NodeId::new).collect() };
        let (storage, key, _backing) = spilled(&ints(&[(88, 88), (3, 3), (19, 19)]));
        storage.set(NodeId::new(5), key.clone(), Value::Int64(1988));
        storage.set(NodeId::new(3), key.clone(), Value::Int64(319));

        assert_eq!(storage.column_ids(&key), ids(&[3, 5, 19, 88]));
        assert_eq!(
            storage.try_column_entries(&key).unwrap(),
            ints(&[(3, 319), (5, 1988), (19, 19), (88, 88)])
        );
        assert_eq!(
            storage.column_ids(&PropertyKey::new("missing")),
            Vec::<NodeId>::new()
        );

        let plain = PropertyStorage::new();
        for raw in [88, 3, 19] {
            plain.set(NodeId::new(raw), key.clone(), Value::Int64(3));
        }
        assert_eq!(plain.column_ids(&key), ids(&[3, 19, 88]));
    }

    /// The column's length counts every visible value once.
    #[test]
    fn len_counts_each_visible_value_once() {
        let (storage, key, _backing) = spilled(&ints(&[(1, 3), (2, 19), (3, 88)]));
        let len = || storage.columns.read()[&key].len();
        assert_eq!(len(), 3);
        storage.set(NodeId::new(1), key.clone(), Value::Int64(319));
        assert_eq!(len(), 3, "a write over a spilled value");
        storage.set(NodeId::new(4), key.clone(), Value::Int64(1988));
        assert_eq!(len(), 4, "a new value");
        storage.remove(NodeId::new(2), &key).unwrap();
        assert_eq!(len(), 3, "a removed spilled value");
        storage.remove(NodeId::new(1), &key).unwrap();
        assert_eq!(len(), 2, "a removed value written over a spilled one");
        storage.set(NodeId::new(2), key.clone(), Value::Int64(19));
        assert_eq!(len(), 3, "a value written again after its removal");
        assert_eq!(
            storage.column_ids(&key),
            vec![NodeId::new(2), NodeId::new(3), NodeId::new(4)]
        );
    }

    /// Removing values the backing never held leaves no tombstone behind.
    #[test]
    fn removing_a_value_the_backing_never_held_leaves_no_tombstone() {
        let (storage, key, _backing) = spilled(&ints(&[(3, 3)]));
        storage.set(NodeId::new(19), key.clone(), Value::Int64(19));
        storage.remove(NodeId::new(19), &key).unwrap();
        storage.remove(NodeId::new(88), &key).unwrap();
        assert_eq!(storage.columns.read()[&key].len(), 1);
        assert!(
            storage.columns.read()[&key].removed.is_empty(),
            "no tombstone"
        );
    }

    /// A reload moves the values back into the column and lets go of the
    /// backing; a second reload has nothing to do.
    #[test]
    fn a_reload_moves_the_backing_back_into_the_column() {
        let (storage, key, backing) = spilled(&ints(&[(1, 3), (2, 19)]));
        storage.remove(NodeId::new(2), &key).unwrap();

        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(storage.spilled_columns(), Vec::<PropertyKey>::new());
        assert_eq!(Arc::strong_count(&backing), 1, "the column let go of it");
        assert_eq!(overlay_ids(&storage, &key), vec![1]);
        assert!(storage.columns.read()[&key].removed.is_empty());
        assert!(!storage.reload_column(&key).unwrap());
        assert!(!storage.reload_column(&PropertyKey::new("missing")).unwrap());
    }

    /// The reload reads the backing without holding the lock; a write and a
    /// removal made meanwhile win over what it read.
    #[test]
    fn a_reload_keeps_changes_made_while_it_reads_the_backing() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let storage = Arc::new(PropertyStorage::new());
        let key = PropertyKey::new("city");
        storage.set(alix, key.clone(), "Amsterdam".into());
        storage.set(gus, key.clone(), "Berlin".into());
        let snapshot = storage.try_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(storage.spill_column(&key, backing.clone(), &snapshot));

        let during = Arc::clone(&storage);
        let during_key = key.clone();
        backing.on_ids(move || {
            during.set(alix, during_key.clone(), "Paris".into());
            during.remove(gus, &during_key).unwrap();
        });
        assert!(storage.reload_column(&key).unwrap());

        assert_eq!(storage.get(alix, &key), Some(Value::from("Paris")));
        assert_eq!(storage.get(gus, &key), None);
    }

    /// A reload whose backing was reloaded and spilled again while it read
    /// leaves the newer spill alone: merging the stale backing would bring a
    /// removed value back and drop the newer backing.
    #[test]
    fn a_stale_reload_leaves_a_newer_spill_alone() {
        let (alix, gus, vincent) = (NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let storage = Arc::new(PropertyStorage::new());
        let key = PropertyKey::new("city");
        storage.set(alix, key.clone(), "Amsterdam".into());
        storage.set(gus, key.clone(), "Berlin".into());
        let snapshot = storage.try_column_entries(&key).unwrap();
        let first = MemoryBacking::of(&snapshot);
        assert!(storage.spill_column(&key, first.clone(), &snapshot));

        let during = Arc::clone(&storage);
        let during_key = key.clone();
        first.on_ids(move || {
            assert!(during.reload_column(&during_key).unwrap());
            during.remove(gus, &during_key).unwrap();
            during.set(vincent, during_key.clone(), "Prague".into());
            let snapshot = during.try_column_entries(&during_key).unwrap();
            assert!(during.spill_column(&during_key, MemoryBacking::of(&snapshot), &snapshot));
        });
        assert!(
            !storage.reload_column(&key).unwrap(),
            "the backing it read is gone"
        );
        assert_eq!(storage.get(gus, &key), None);
        assert_eq!(storage.column_ids(&key), vec![alix, vincent]);
        assert_eq!(storage.spilled_columns(), vec![key.clone()]);
    }

    /// A backing that cannot be read keeps its column spilled: the reload
    /// changes nothing, the fallible reads report the error and the queries
    /// read the values as absent. Once it reads again, nothing was lost.
    #[test]
    fn a_failed_read_keeps_the_column_spilled() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let (storage, key, backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);
        storage.set(gus, key.clone(), vector(&[319.0, 1988.0]));

        backing.fail_reads(true);
        assert!(storage.reload_column(&key).is_err());
        assert_eq!(storage.spilled_columns(), vec![key.clone()]);
        assert!(storage.try_column_entries(&key).is_err());
        assert!(storage.try_get_batch(&[alix], &key).is_err());
        assert!(storage.try_get_all(alix).is_err());
        assert_eq!(storage.get(alix, &key), None, "a query reads it as absent");
        assert_eq!(storage.with_vector(alix, &key, |v: &[f32]| v[0]), None);
        assert_eq!(
            storage.try_get_batch(&[gus], &key).unwrap(),
            vec![Some(vector(&[319.0, 1988.0]))],
            "the column's own value needs no read"
        );
        assert_eq!(storage.column_ids(&key), vec![alix, gus]);

        backing.fail_reads(false);
        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(
            storage.get_batch(&[alix, gus], &key),
            vec![Some(vector(&[3.0, 19.0])), Some(vector(&[319.0, 1988.0]))]
        );
    }

    /// A backing that lists an id it holds no value for, and another one
    /// twice, breaks its contract: the reload and the fallible reads refuse
    /// it rather than lose a value, and `column_ids` lists each id once.
    #[test]
    fn a_backing_that_lists_an_id_without_a_value_is_refused() {
        let (alix, vincent) = (NodeId::new(1), NodeId::new(3));
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("embedding");
        storage.set(alix, key.clone(), vector(&[3.0, 19.0]));
        let snapshot = storage.try_column_entries(&key).unwrap();
        let backing = MemoryBacking::listing(&snapshot, &[vincent, alix]);
        assert!(storage.spill_column(&key, backing, &snapshot));

        assert_eq!(storage.column_ids(&key), vec![alix, vincent]);
        assert!(storage.try_column_entries(&key).is_err());
        assert!(storage.reload_column(&key).is_err());
        assert_eq!(storage.spilled_columns(), vec![key.clone()]);
        assert_eq!(storage.get(alix, &key), Some(vector(&[3.0, 19.0])));
    }

    /// A column that is missing or already spilled is not spilled (again).
    #[test]
    fn spilling_a_missing_or_spilled_column_is_refused() {
        let (storage, key, _backing) = spilled(&ints(&[(1, 3)]));
        let snapshot = ints(&[(1, 3)]);
        assert!(!storage.spill_column(&key, MemoryBacking::of(&snapshot), &snapshot));
        assert!(!storage.spill_column(
            &PropertyKey::new("missing"),
            MemoryBacking::of(&snapshot),
            &snapshot
        ));
        assert_eq!(storage.get(NodeId::new(1), &key), Some(Value::Int64(3)));
    }

    /// `spilled_columns` lists the spilled keys in key order.
    #[test]
    fn spilled_columns_come_in_key_order() {
        let storage = PropertyStorage::new();
        for name in ["embedding", "city", "name", "age"] {
            let key = PropertyKey::new(name);
            storage.set(NodeId::new(1), key.clone(), Value::Int64(19));
            let snapshot = storage.try_column_entries(&key).unwrap();
            assert!(storage.spill_column(&key, MemoryBacking::of(&snapshot), &snapshot));
        }
        assert_eq!(
            storage.spilled_columns(),
            ["age", "city", "embedding", "name"].map(PropertyKey::new)
        );
    }

    /// Rebuilding the zone maps of a spilled column must not narrow them to
    /// the values still on the heap: the query would skip spilled matches.
    #[test]
    fn a_zone_map_rebuild_while_spilled_keeps_spilled_values_matchable() {
        let entries: Vec<(NodeId, Value)> = (1..=100)
            .map(|i| (NodeId::new(i), Value::Int64(i64::try_from(i).unwrap())))
            .collect();
        let (storage, key, _backing) = spilled(&entries);
        storage.set(NodeId::new(200), key.clone(), Value::Int64(1988));
        storage.remove(NodeId::new(3), &key).unwrap();

        storage.rebuild_zone_maps();
        assert!(storage.might_match(&key, CompareOp::Eq, &Value::Int64(88)));
        assert!(storage.might_match(&key, CompareOp::Eq, &Value::Int64(1988)));
    }

    /// Compression never moves the values of a spilled column out of reach.
    #[test]
    fn compression_leaves_a_spilled_column_readable() {
        let (storage, key, _backing) = spilled(&ints(&[(1, 3)]));
        for i in 1000..1100 {
            storage.set(
                NodeId::new(i),
                key.clone(),
                Value::Int64(i64::try_from(i).unwrap()),
            );
        }
        storage.enable_compression(&key, CompressionMode::Eager);
        storage.force_compress_all();
        storage.compress_all();

        assert_eq!(
            storage.get(NodeId::new(1088), &key),
            Some(Value::Int64(1088))
        );
        assert_eq!(storage.column_ids(&key).len(), 101);
    }

    /// A storage whose `key` column holds `entries`, compressed.
    fn compressed(entries: &[(NodeId, Value)]) -> (PropertyStorage, PropertyKey) {
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("score");
        for (id, value) in entries {
            storage.set(*id, key.clone(), value.clone());
        }
        storage.enable_compression(&key, CompressionMode::Eager);
        storage.force_compress_all();
        assert!(
            storage.columns.read()[&key].is_compressed(),
            "the column compressed"
        );
        (storage, key)
    }

    /// Integers, strings and booleans compressed in a column stay part of it:
    /// the enumerators that snapshots and checkpoints read list them, and
    /// they read by id.
    #[test]
    fn compressed_rows_are_read_by_every_reader() {
        let columns: [Vec<(NodeId, Value)>; 3] = [
            (0..100)
                .map(|i| {
                    (
                        NodeId::new(i),
                        Value::Int64(1000 + i64::try_from(i).unwrap()),
                    )
                })
                .collect(),
            (0..100)
                .map(|i| {
                    let city =
                        ["Amsterdam", "Berlin", "Paris", "Prague"][usize::try_from(i % 4).unwrap()];
                    (NodeId::new(i), Value::from(city))
                })
                .collect(),
            (0..100)
                .map(|i| (NodeId::new(i), Value::Bool(i % 3 == 0)))
                .collect(),
        ];
        for entries in columns {
            let (storage, key) = compressed(&entries);
            let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
            assert_eq!(storage.column_ids(&key), ids, "{:?}", entries[0].1);
            assert_eq!(
                storage.try_column_entries(&key).unwrap(),
                entries,
                "{:?}",
                entries[0].1
            );
            assert_eq!(
                storage.get(NodeId::new(19), &key),
                Some(entries[19].1.clone())
            );
            assert_eq!(
                storage.try_get(NodeId::new(88), &key).unwrap(),
                Some(entries[88].1.clone())
            );
            assert_eq!(
                storage.get_all(NodeId::new(3)).get(&key),
                Some(&entries[3].1)
            );
            assert_eq!(storage.columns.read()[&key].len(), 100);
        }
    }

    /// Decoding one row of a compressed integer or boolean column decodes
    /// all of it, so a batch read decodes the column once per call, not once
    /// per id; a single read decodes it once.
    #[test]
    fn a_batch_read_decodes_a_compressed_column_once() {
        let columns: [Vec<(NodeId, Value)>; 2] = [
            (0..100)
                .map(|i| {
                    (
                        NodeId::new(i),
                        Value::Int64(1000 + i64::try_from(i).unwrap()),
                    )
                })
                .collect(),
            (0..100)
                .map(|i| (NodeId::new(i), Value::Bool(i % 3 == 0)))
                .collect(),
        ];
        let decodes = |read: &dyn Fn()| {
            let before = COMPRESSED_DECODES.get();
            read();
            COMPRESSED_DECODES.get() - before
        };
        for entries in columns {
            let (storage, key) = compressed(&entries);
            let kind = format!("{:?}", entries[0].1);
            let ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
            let values: Vec<Option<Value>> = entries
                .iter()
                .map(|(_, value)| Some(value.clone()))
                .collect();
            let maps: Vec<FxHashMap<PropertyKey, Value>> = entries
                .iter()
                .map(|(_, value)| [(key.clone(), value.clone())].into_iter().collect())
                .collect();

            let get_batch = decodes(&|| assert_eq!(storage.get_batch(&ids, &key), values));
            assert_eq!(get_batch, 1, "get_batch of {kind}");
            let try_get_batch =
                decodes(&|| assert_eq!(storage.try_get_batch(&ids, &key).unwrap(), values));
            assert_eq!(try_get_batch, 1, "try_get_batch of {kind}");
            let get_all_batch = decodes(&|| assert_eq!(storage.get_all_batch(&ids), maps));
            assert_eq!(get_all_batch, 1, "get_all_batch of {kind}");
            let selective = decodes(&|| {
                assert_eq!(
                    storage.get_selective_batch(&ids, std::slice::from_ref(&key)),
                    maps
                );
            });
            assert_eq!(selective, 1, "get_selective_batch of {kind}");
            let entries_read = decodes(&|| {
                assert_eq!(storage.try_column_entries(&key).unwrap(), entries);
            });
            assert_eq!(entries_read, 1, "try_column_entries of {kind}");
            let single = decodes(&|| {
                assert_eq!(storage.get(NodeId::new(19), &key), values[19]);
            });
            assert_eq!(single, 1, "a single get of {kind}");
        }
    }

    /// A write over a compressed row wins, a removal hides it (and returns
    /// it), and decompressing keeps both changes.
    #[test]
    fn writes_over_compressed_rows_win() {
        let entries = ints(
            &(0..100)
                .map(|i| (i, 1000 + i64::try_from(i).unwrap()))
                .collect::<Vec<_>>(),
        );
        let (storage, key) = compressed(&entries);
        let (alix, gus, vincent) = (NodeId::new(3), NodeId::new(19), NodeId::new(88));

        storage.set(alix, key.clone(), Value::Int64(319));
        assert_eq!(storage.get(alix, &key), Some(Value::Int64(319)));
        assert_eq!(storage.remove(gus, &key).unwrap(), Some(Value::Int64(1019)));
        assert_eq!(storage.get(gus, &key), None);
        storage.remove_all(vincent);
        assert_eq!(storage.get(vincent, &key), None);
        let len = || storage.columns.read()[&key].len();
        assert_eq!(len(), 98);
        assert_eq!(storage.column_ids(&key).len(), 98);
        let entries = storage.try_column_entries(&key).unwrap();
        assert!(entries.contains(&(alix, Value::Int64(319))));
        assert!(!entries.iter().any(|(id, _)| *id == gus || *id == vincent));
        storage.rebuild_zone_maps();
        assert!(
            storage.might_match(&key, CompareOp::Eq, &Value::Int64(1050)),
            "a zone map rebuild without the compressed rows"
        );

        storage.enable_compression(&key, CompressionMode::None);
        assert!(!storage.columns.read()[&key].is_compressed());
        assert_eq!(storage.get(alix, &key), Some(Value::Int64(319)));
        assert_eq!(storage.get(gus, &key), None);
        assert_eq!(storage.get(vincent, &key), None);
        assert_eq!(len(), 98);
    }

    /// A compressed column spills whole: its snapshot holds the compressed
    /// rows, which then read through the backing.
    #[test]
    fn a_compressed_column_spills_whole() {
        let entries = ints(
            &(0..100)
                .map(|i| (i, 1000 + i64::try_from(i).unwrap()))
                .collect::<Vec<_>>(),
        );
        let (storage, key) = compressed(&entries);
        let snapshot = storage.try_column_entries(&key).unwrap();
        assert_eq!(snapshot, entries);
        assert!(storage.spill_column(&key, MemoryBacking::of(&snapshot), &snapshot));

        assert_eq!(storage.column_ids(&key).len(), 100);
        assert_eq!(storage.columns.read()[&key].len(), 100);
        assert_eq!(storage.get(NodeId::new(19), &key), Some(Value::Int64(1019)));
        assert!(storage.reload_column(&key).unwrap());
        assert_eq!(storage.try_column_entries(&key).unwrap(), entries);
    }

    /// A vector read hands out the spilled vector without copying it out of
    /// the backing; the column's own value wins and a removal hides it.
    #[test]
    fn a_vector_read_does_not_copy_from_the_backing() {
        let (alix, gus, vincent) = (NodeId::new(1), NodeId::new(2), NodeId::new(3));
        let (storage, key, backing) = spilled(&[
            (alix, vector(&[3.0, 19.0])),
            (gus, vector(&[88.0, 3.19])),
            (vincent, vector(&[319.0, 1988.0])),
        ]);
        let sum = |id| storage.with_vector(id, &key, |v: &[f32]| v.iter().sum::<f32>());

        assert_eq!(sum(alix), Some(22.0));
        assert_eq!(backing.copies.load(AtomicOrdering::Relaxed), 0);
        storage.set(gus, key.clone(), vector(&[3.0, 3.0]));
        assert_eq!(sum(gus), Some(6.0));
        storage.set(vincent, key.clone(), "not a vector".into());
        assert_eq!(sum(vincent), None, "the column's value is not a vector");
        storage.remove_all(alix);
        assert_eq!(sum(alix), None);
        assert_eq!(sum(NodeId::new(88)), None);
        assert_eq!(
            storage.with_vector(gus, &PropertyKey::new("missing"), |v: &[f32]| v.len()),
            None
        );
        assert_eq!(
            backing.copies.load(AtomicOrdering::Relaxed),
            0,
            "neither the reads nor the delete copied a value"
        );
    }

    /// A vector read inside a vector read (pairwise distances) finishes even
    /// with a writer waiting: the closure runs without the storage lock, so
    /// the inner read never queues behind the writer it blocks.
    #[test]
    fn a_nested_vector_read_does_not_wait_for_a_queued_writer() {
        let (alix, gus) = (NodeId::new(1), NodeId::new(2));
        let (storage, key, _backing) =
            spilled(&[(alix, vector(&[3.0, 19.0])), (gus, vector(&[88.0, 3.19]))]);
        let storage = Arc::new(storage);
        // One read from the backing, one from the column itself.
        storage.set(gus, key.clone(), vector(&[319.0, 1988.0]));

        let (done, finished) = std::sync::mpsc::channel();
        let reader = Arc::clone(&storage);
        std::thread::spawn(move || {
            let outer = reader.with_vector(alix, &key, |a: &[f32]| {
                let writer = Arc::clone(&reader);
                let writer_key = key.clone();
                let write = std::thread::spawn(move || {
                    writer.set(NodeId::new(3), writer_key, Value::Int64(88));
                });
                // Long enough for the writer to queue on a held lock.
                std::thread::sleep(std::time::Duration::from_millis(200));
                let inner = reader.with_vector(gus, &key, |b: &[f32]| a[0] + b[0]);
                write.join().unwrap();
                inner
            });
            done.send(outer).unwrap();
        });
        let outer = finished
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a nested vector read deadlocked behind a queued writer");
        assert_eq!(outer, Some(Some(322.0)));
    }

    /// The heap estimate of a spilled column is what the backing reports
    /// plus what the column still holds, tombstones included.
    #[test]
    fn heap_memory_of_a_spilled_column_counts_the_backing() {
        let (storage, key, _backing) = spilled(&ints(&[(1, 3), (2, 19)]));
        let heap = || storage.columns.read()[&key].heap_memory_bytes();
        assert_eq!(heap(), MemoryBacking::HEAP_BYTES);
        storage.remove(NodeId::new(2), &key).unwrap();
        assert!(heap() > MemoryBacking::HEAP_BYTES, "the tombstone counts");
    }

    /// Spills and reloads racing with writers and readers on real threads:
    /// no write is lost, no removed value comes back, an untouched value
    /// never reads as absent, and `column_ids` is always in order.
    #[test]
    fn spills_and_reloads_race_with_writers_without_losing_anything() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        let storage = Arc::new(PropertyStorage::new());
        let key = PropertyKey::new("city");
        let untouched: Vec<NodeId> = (1000..1050).map(NodeId::new).collect();
        for id in &untouched {
            storage.set(*id, key.clone(), Value::Int64(1988));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let (cycles, reads) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));

        let spiller = {
            let (storage, key, stop) = (Arc::clone(&storage), key.clone(), Arc::clone(&stop));
            let cycles = Arc::clone(&cycles);
            std::thread::spawn(move || {
                while !stop.load(AtomicOrdering::Relaxed) {
                    let snapshot = storage.try_column_entries(&key).unwrap();
                    if storage.spill_column(&key, MemoryBacking::of(&snapshot), &snapshot) {
                        cycles.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    storage.reload_column(&key).unwrap();
                }
            })
        };
        let reader = {
            let (storage, key, stop) = (Arc::clone(&storage), key.clone(), Arc::clone(&stop));
            let (untouched, reads) = (untouched.clone(), Arc::clone(&reads));
            std::thread::spawn(move || {
                while !stop.load(AtomicOrdering::Relaxed) {
                    for id in &untouched {
                        assert_eq!(storage.get(*id, &key), Some(Value::Int64(1988)));
                    }
                    let ids = storage.column_ids(&key);
                    assert!(ids.windows(2).all(|pair| pair[0].0 < pair[1].0));
                    reads.fetch_add(1, AtomicOrdering::Relaxed);
                }
            })
        };

        // The writer: random sets and removes on 100 ids, checked against a
        // model at every step, until both other threads have had their turns
        // (on a busy machine 20,000 steps can finish before they start).
        let mut model: std::collections::HashMap<u64, i64> = std::collections::HashMap::new();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut step = 0_i64;
        while step < 20_000
            || cycles.load(AtomicOrdering::Relaxed) < 19
            || reads.load(AtomicOrdering::Relaxed) < 19
        {
            assert!(step < 20_000_000, "the spiller or the reader never ran");
            step += 1;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let id = state % 100;
            if state.is_multiple_of(3) {
                let expected = model.remove(&id).map(Value::Int64);
                assert_eq!(
                    storage.remove(NodeId::new(id), &key).unwrap(),
                    expected,
                    "step {step}"
                );
            } else {
                storage.set(NodeId::new(id), key.clone(), Value::Int64(step));
                model.insert(id, step);
            }
            assert_eq!(
                storage.get(NodeId::new(id), &key),
                model.get(&id).copied().map(Value::Int64),
                "step {step}"
            );
        }
        stop.store(true, AtomicOrdering::Relaxed);
        spiller.join().unwrap();
        reader.join().unwrap();

        storage.reload_column(&key).unwrap();
        for id in 0..100 {
            assert_eq!(
                storage.get(NodeId::new(id), &key),
                model.get(&id).copied().map(Value::Int64)
            );
        }
        for id in &untouched {
            assert_eq!(storage.get(*id, &key), Some(Value::Int64(1988)));
        }
    }
}

#[cfg(test)]
#[cfg(feature = "temporal")]
mod temporal_tests {
    use super::*;

    /// `column_ids` and `try_column_entries` list the live values in id order,
    /// and `with_vector` reads the latest vector (#594).
    #[test]
    fn column_reads_see_the_latest_live_values() {
        let storage = PropertyStorage::new();
        let key = PropertyKey::new("embedding");
        let epoch = EpochId::new(1);
        for (raw, x) in [(7, 3.0), (2, 19.0), (5, 88.0)] {
            storage.set(
                NodeId::new(raw),
                key.clone(),
                Value::Vector(vec![x].into()),
                epoch,
            );
        }
        storage.set(
            NodeId::new(5),
            key.clone(),
            Value::Vector(vec![319.0].into()),
            EpochId::new(2),
        );
        storage.remove(NodeId::new(2), &key, EpochId::new(2));

        assert_eq!(
            storage.column_ids(&key),
            vec![NodeId::new(5), NodeId::new(7)]
        );
        let expected = vec![
            (NodeId::new(5), Value::Vector(vec![319.0].into())),
            (NodeId::new(7), Value::Vector(vec![3.0].into())),
        ];
        assert_eq!(storage.try_column_entries(&key).unwrap(), expected);
        assert_eq!(storage.try_column_entries(&key).unwrap(), expected);
        let first = |id| storage.with_vector(NodeId::new(id), &key, |v: &[f32]| v[0]);
        assert_eq!(first(5), Some(319.0));
        assert_eq!(first(2), None, "removed");
        assert_eq!(first(88), None);
    }

    /// Garbage collection visits only the entities with history, so its cost
    /// follows the changes and not the size of the column; a log stays on the
    /// list until it is down to one entry.
    #[test]
    fn gc_visits_only_entities_with_history() {
        let mut col: PropertyColumn<NodeId> = PropertyColumn::new();
        for i in 0u64..1000 {
            col.set(
                NodeId::new(i),
                Value::Int64(i64::try_from(i).unwrap()),
                EpochId::new(1),
            );
        }
        assert!(
            col.gc_candidates.is_empty(),
            "single versions: nothing to visit"
        );

        let (alix, gus) = (NodeId::new(7), NodeId::new(8));
        col.set(alix, Value::from("Paris"), EpochId::new(2));
        col.set(alix, Value::from("Prague"), EpochId::new(3));
        assert_eq!(col.remove(gus, EpochId::new(3)), Some(Value::Int64(8)));
        let candidates = |col: &PropertyColumn<NodeId>| {
            let mut ids: Vec<u64> = col.gc_candidates.iter().map(|id| id.as_u64()).collect();
            ids.sort_unstable();
            ids
        };
        assert_eq!(candidates(&col), [7, 8]);

        // Readers at epoch 3 still need the value before it: both logs keep
        // two entries and stay candidates; only Alix's first value goes.
        col.gc(EpochId::new(3));
        assert_eq!(candidates(&col), [7, 8]);
        assert_eq!(col.get_at(alix, EpochId::new(1)), None);
        assert_eq!(
            col.get_at(alix, EpochId::new(2)),
            Some(Value::from("Paris"))
        );
        assert_eq!(col.get_at(gus, EpochId::new(2)), Some(Value::Int64(8)));

        col.gc(EpochId::new(4));
        assert!(col.gc_candidates.is_empty());
        assert_eq!(col.get(alix), Some(Value::from("Prague")));
        assert_eq!(col.get(gus), None);
        assert_eq!(col.get_at(gus, EpochId::new(2)), None);
        assert_eq!(col.get(NodeId::new(9)), Some(Value::Int64(9)));
    }
}
