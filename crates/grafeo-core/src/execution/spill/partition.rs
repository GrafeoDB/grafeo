//! Hash partitioning for spillable aggregation.
//!
//! This module implements hash partitioning that allows aggregate state
//! to be partitioned and spilled to disk when memory pressure is high.
//!
//! # Design
//!
//! - Groups are assigned to partitions based on their key's hash
//! - In-memory partitions can be spilled to disk under memory pressure
//! - Cold (least recently accessed) partitions are spilled first
//! - When iterating results, spilled partitions are reloaded

use super::file::SpillFile;
use super::manager::SpillManager;
use super::serializer::{deserialize_row, serialize_row};
use crate::execution::memory::{
    GroupedMemoryOwner, checked_grouped_bytes, grouped_hash_table_bytes, grouped_io_error,
    grouped_memory_limit,
};
use crate::execution::operators::OperatorError;
use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::types::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

/// Default number of partitions for hash partitioning.
pub const DEFAULT_NUM_PARTITIONS: usize = 256;

/// A serialized key for use as a HashMap key.
/// We serialize Value vectors to bytes since Value doesn't implement Hash/Eq.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SerializedKey(Vec<u8>);

impl SerializedKey {
    fn from_values(values: &[Value]) -> Self {
        let mut buf = Vec::new();
        serialize_row(values, &mut buf).expect("serialization should not fail");
        Self(buf)
    }

    fn to_values(&self, num_columns: usize) -> std::io::Result<Vec<Value>> {
        deserialize_row(&mut self.0.as_slice(), num_columns)
    }
}

/// Entry in a partition: the original key columns count and value.
struct PartitionEntry<V> {
    num_key_columns: usize,
    value: V,
    owner: Option<GroupedMemoryOwner>,
}

/// Storage and admission share a lifetime; drop entries before their table grant.
struct ResidentPartition<V> {
    entries: HashMap<SerializedKey, PartitionEntry<V>>,
    table_owner: Option<GroupedMemoryOwner>,
    owned_bytes: usize,
}

impl<V> ResidentPartition<V> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            table_owner: None,
            owned_bytes: 0,
        }
    }
}

impl<V> std::ops::Deref for ResidentPartition<V> {
    type Target = HashMap<SerializedKey, PartitionEntry<V>>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl<V> std::ops::DerefMut for ResidentPartition<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries
    }
}

impl std::borrow::Borrow<[u8]> for SerializedKey {
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

/// Partitioned accumulator state for spillable aggregation.
///
/// Manages aggregate state across multiple partitions, with the ability
/// to spill cold partitions to disk under memory pressure.
pub struct PartitionedState<V> {
    /// Spill manager for file creation.
    manager: Arc<SpillManager>,
    /// Number of partitions.
    num_partitions: usize,
    /// In-memory partitions (None = spilled to disk).
    partitions: Vec<Option<ResidentPartition<V>>>,
    /// Spill files for spilled partitions.
    spill_files: Vec<Option<SpillFile>>,
    /// Number of groups per partition (for spilled partitions too).
    partition_sizes: Vec<usize>,
    /// Access timestamps for LRU eviction.
    access_times: Vec<u64>,
    /// Global timestamp counter.
    timestamp: u64,
    /// Serializer for V values.
    value_serializer: Box<dyn Fn(&V, &mut dyn Write) -> std::io::Result<()> + Send + Sync>,
    /// Deserializer for V values.
    value_deserializer: Box<dyn Fn(&mut dyn Read) -> std::io::Result<V> + Send + Sync>,
    /// Declared after the containers so their storage drops before admission.
    metadata_owner: Option<GroupedMemoryOwner>,
    admitted_deserializer:
        Option<fn(&mut dyn Read, &mut GroupedMemoryOwner) -> Result<V, OperatorError>>,
}

/// A bounded batch keeps every decoded/resident payload reservation alive.
pub(crate) struct OwnedPartitionBatch<V> {
    values: Vec<V>,
    owner: GroupedMemoryOwner,
}
impl<V> OwnedPartitionBatch<V> {
    pub(crate) fn into_parts(self) -> (Vec<V>, GroupedMemoryOwner) {
        (self.values, self.owner)
    }
}

struct ResidentDrain<V> {
    entries: std::collections::hash_map::IntoIter<SerializedKey, PartitionEntry<V>>,
    _table_owner: Option<GroupedMemoryOwner>,
}

/// Advances resident entries or one spilled record at a time. The source
/// state remains the cleanup authority until the cursor is exhausted/dropped.
pub(crate) struct OwnedPartitionDrainCursor<V> {
    resident: Option<ResidentDrain<V>>,
    reader: Option<(super::file::SpillFileReader, usize)>,
    index: usize,
    remaining: usize,
    state: PartitionedState<V>,
}

impl<V: Clone + Send + Sync + 'static> OwnedPartitionDrainCursor<V> {
    pub(crate) fn next_batch(&mut self) -> Result<Option<OwnedPartitionBatch<V>>, OperatorError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let capacity = self
            .remaining
            .min(crate::execution::chunk::DEFAULT_CHUNK_SIZE);
        let manager = Arc::clone(
            self.state
                .metadata_owner
                .as_ref()
                .expect("admitted drain retains metadata owner")
                .manager(),
        );
        let mut owner = GroupedMemoryOwner::try_new(
            manager,
            checked_grouped_bytes(capacity, std::mem::size_of::<V>())?,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| grouped_memory_limit())?;
        while values.len() < capacity {
            if let Some(resident) = self.resident.as_mut() {
                if let Some((key, mut entry)) = resident.entries.next() {
                    let key_bytes = key.0.capacity();
                    drop(key);
                    let mut entry_owner = entry.owner.take().expect("admitted entry has owner");
                    entry_owner.try_resize(
                        entry_owner.size().checked_sub(key_bytes).ok_or_else(|| {
                            OperatorError::Internal("group key reservation mismatch".to_string())
                        })?,
                    )?;
                    owner
                        .try_merge(entry_owner)
                        .map_err(|(error, _owner)| error)?;
                    values.push(entry.value);
                    self.remaining -= 1;
                    continue;
                }
                self.resident = None;
                self.state.partition_sizes[self.index] = 0;
                self.index += 1;
                continue;
            }
            if let Some((reader, remaining)) = self.reader.as_mut() {
                if *remaining == 0 {
                    self.reader = None; // Close the reader before Windows deletion.
                    let mut completed = [self.state.spill_files[self.index].take()];
                    release_spill_files(&self.state.manager, &mut completed);
                    self.state.partition_sizes[self.index] = 0;
                    self.index += 1;
                    continue;
                }
                let mut adapter = SpillReaderAdapter(reader);
                let mut key_len =
                    usize::try_from(read_u64(&mut adapter).map_err(grouped_io_error)?)
                        .map_err(|_| grouped_memory_limit())?;
                // The group state already carries the original output key.
                // Skip the canonical lookup bytes with fixed stack storage.
                let mut scratch = [0_u8; 4096];
                while key_len != 0 {
                    let count = key_len.min(scratch.len());
                    adapter
                        .read_exact(&mut scratch[..count])
                        .map_err(grouped_io_error)?;
                    key_len -= count;
                }
                let _columns = read_u64(&mut adapter).map_err(grouped_io_error)?;
                let decode = self
                    .state
                    .admitted_deserializer
                    .expect("admitted drain has decoder");
                let value = decode(&mut adapter, &mut owner)?;
                values.push(value);
                *remaining -= 1;
                self.remaining -= 1;
                continue;
            }
            if self.index >= self.state.num_partitions {
                return Err(grouped_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "spilled partition row count differs from resident metadata",
                )));
            }
            if let Some(partition) = self.state.partitions[self.index].take() {
                self.resident = Some(ResidentDrain {
                    entries: partition.entries.into_iter(),
                    _table_owner: partition.table_owner,
                });
            } else if let Some(file) = self.state.spill_files[self.index].as_ref() {
                let mut reader = file.reader().map_err(grouped_io_error)?;
                let count = usize::try_from(
                    read_u64(&mut SpillReaderAdapter(&mut reader)).map_err(grouped_io_error)?,
                )
                .map_err(|_| grouped_memory_limit())?;
                if count != self.state.partition_sizes[self.index] {
                    return Err(grouped_io_error(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "spilled partition row count differs from resident metadata",
                    )));
                }
                self.reader = Some((reader, count));
            } else {
                self.index += 1;
            }
        }
        Ok(Some(OwnedPartitionBatch { values, owner }))
    }
}

impl<V: Clone + Send + Sync + 'static> PartitionedState<V> {
    /// Creates a new partitioned state with custom serialization.
    pub fn new<S, D>(
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
    ) -> Self
    where
        S: Fn(&V, &mut dyn Write) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read) -> std::io::Result<V> + Send + Sync + 'static,
    {
        let mut partitions = Vec::with_capacity(num_partitions);
        let mut spill_files = Vec::with_capacity(num_partitions);
        for _ in 0..num_partitions {
            partitions.push(Some(ResidentPartition::new()));
            spill_files.push(None);
        }

        let partition_sizes = vec![0; num_partitions];
        let access_times = vec![0; num_partitions];

        Self {
            manager,
            num_partitions,
            partitions,
            spill_files,
            partition_sizes,
            access_times,
            timestamp: 0,
            value_serializer: Box::new(value_serializer),
            value_deserializer: Box::new(value_deserializer),
            metadata_owner: None,
            admitted_deserializer: None,
        }
    }

    /// Admits partition metadata before constructing its backing vectors.
    pub(crate) fn try_new_accounted<S, D>(
        buffer_manager: Arc<BufferManager>,
        manager: Arc<SpillManager>,
        num_partitions: usize,
        value_serializer: S,
        value_deserializer: D,
        admitted_deserializer: fn(
            &mut dyn Read,
            &mut GroupedMemoryOwner,
        ) -> Result<V, OperatorError>,
    ) -> Result<Self, OperatorError>
    where
        S: Fn(&V, &mut dyn Write) -> std::io::Result<()> + Send + Sync + 'static,
        D: Fn(&mut dyn Read) -> std::io::Result<V> + Send + Sync + 'static,
    {
        let slot_bytes = std::mem::size_of::<Option<ResidentPartition<V>>>()
            + std::mem::size_of::<Option<SpillFile>>()
            + std::mem::size_of::<usize>()
            + std::mem::size_of::<u64>();
        let metadata_bytes = checked_grouped_bytes(num_partitions, slot_bytes)?
            .checked_add(std::mem::size_of::<S>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<D>()))
            .and_then(|bytes| bytes.checked_add(super::file::BUFFER_SIZE))
            .ok_or_else(grouped_memory_limit)?;
        let owner = GroupedMemoryOwner::try_new(buffer_manager, metadata_bytes)?;
        let mut state = Self::new(
            manager,
            num_partitions,
            value_serializer,
            value_deserializer,
        );
        state.metadata_owner = Some(owner);
        state.admitted_deserializer = Some(admitted_deserializer);
        Ok(state)
    }

    /// Returns the partition index for a key.
    #[must_use]
    pub fn partition_for(&self, key: &[Value]) -> usize {
        let hash = hash_key(key);
        // reason: on 64-bit targets u64 == usize; on 32-bit the modulo handles overflow
        #[allow(clippy::cast_possible_truncation)]
        {
            hash as usize % self.num_partitions
        }
    }

    /// Updates access time for a partition.
    fn touch(&mut self, partition_idx: usize) {
        self.timestamp += 1;
        self.access_times[partition_idx] = self.timestamp;
    }

    /// Gets the in-memory partition, loading from disk if spilled.
    ///
    /// # Errors
    ///
    /// Returns an error if reading from disk fails.
    fn get_partition_mut(
        &mut self,
        partition_idx: usize,
    ) -> std::io::Result<&mut ResidentPartition<V>> {
        self.touch(partition_idx);

        // If partition is in memory, return it
        if self.partitions[partition_idx].is_some() {
            // Invariant: just checked is_some() above
            return Ok(self.partitions[partition_idx]
                .as_mut()
                .expect("partition is Some: checked on previous line"));
        }

        // Load from disk
        if let Some(spill_file) = self.spill_files[partition_idx].as_ref() {
            // Keep the file owner until the entire partition has decoded.
            let loaded = self.load_partition(spill_file)?;
            let mut completed = [self.spill_files[partition_idx].take()];
            release_spill_files(&self.manager, &mut completed);
            self.partitions[partition_idx] = Some(loaded);
        } else {
            // Neither in memory nor on disk - create empty partition
            self.partitions[partition_idx] = Some(ResidentPartition::new());
        }

        // Invariant: partition was either loaded from disk or created empty above
        Ok(self.partitions[partition_idx]
            .as_mut()
            .expect("partition is Some: set to Some in if/else branches above"))
    }

    /// Loads a partition from a spill file.
    fn load_partition(&self, spill_file: &SpillFile) -> std::io::Result<ResidentPartition<V>> {
        if let Some(decode) = self.admitted_deserializer {
            return self
                .load_accounted_partition(spill_file, decode)
                .map_err(std::io::Error::other);
        }
        let mut reader = spill_file.reader()?;
        let mut adapter = SpillReaderAdapter(&mut reader);

        // reason: deserialized counts are bounded by available data
        #[allow(clippy::cast_possible_truncation)]
        let num_entries = read_u64(&mut adapter)? as usize;
        let mut partition = HashMap::with_capacity(num_entries);

        for _ in 0..num_entries {
            // Read key
            // reason: deserialized lengths are bounded by available data
            #[allow(clippy::cast_possible_truncation)]
            let key_len = read_u64(&mut adapter)? as usize;
            let mut key_buf = vec![0u8; key_len];
            adapter.read_exact(&mut key_buf)?;
            let serialized_key = SerializedKey(key_buf);

            // Read number of key columns
            // reason: deserialized counts are bounded by available data
            #[allow(clippy::cast_possible_truncation)]
            let num_key_columns = read_u64(&mut adapter)? as usize;

            // Read value
            let value = (self.value_deserializer)(&mut adapter)?;

            partition.insert(
                serialized_key,
                PartitionEntry {
                    num_key_columns,
                    value,
                    owner: None,
                },
            );
        }

        Ok(ResidentPartition {
            entries: partition,
            table_owner: None,
            owned_bytes: 0,
        })
    }

    fn load_accounted_partition(
        &self,
        spill_file: &SpillFile,
        decode: fn(&mut dyn Read, &mut GroupedMemoryOwner) -> Result<V, OperatorError>,
    ) -> Result<ResidentPartition<V>, OperatorError> {
        let manager = Arc::clone(
            self.metadata_owner
                .as_ref()
                .expect("accounted state has metadata admission")
                .manager(),
        );
        // The metadata owner reserves one bounded reader/writer workspace.
        let mut reader = spill_file.reader().map_err(grouped_io_error)?;
        let mut adapter = SpillReaderAdapter(&mut reader);
        let count = usize::try_from(read_u64(&mut adapter).map_err(grouped_io_error)?)
            .map_err(|_| grouped_memory_limit())?;
        let table_owner = GroupedMemoryOwner::try_new(
            Arc::clone(&manager),
            grouped_hash_table_bytes::<(SerializedKey, PartitionEntry<V>)>(count)?,
        )?;
        let table_bytes = table_owner.size();
        let mut partition = ResidentPartition {
            entries: HashMap::new(),
            table_owner: Some(table_owner),
            owned_bytes: table_bytes,
        };
        partition
            .entries
            .try_reserve(count)
            .map_err(|_| grouped_memory_limit())?;
        for _ in 0..count {
            let key_len = usize::try_from(read_u64(&mut adapter).map_err(grouped_io_error)?)
                .map_err(|_| grouped_memory_limit())?;
            let mut owner = GroupedMemoryOwner::try_new(Arc::clone(&manager), key_len)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(key_len)
                .map_err(|_| grouped_memory_limit())?;
            bytes.resize(key_len, 0);
            adapter.read_exact(&mut bytes).map_err(grouped_io_error)?;
            let num_key_columns =
                usize::try_from(read_u64(&mut adapter).map_err(grouped_io_error)?)
                    .map_err(|_| grouped_memory_limit())?;
            let value = decode(&mut adapter, &mut owner)?;
            partition.owned_bytes = partition
                .owned_bytes
                .checked_add(owner.size())
                .ok_or_else(grouped_memory_limit)?;
            let old = partition.entries.insert(
                SerializedKey(bytes),
                PartitionEntry {
                    num_key_columns,
                    value,
                    owner: Some(owner),
                },
            );
            if old.is_some() {
                return Err(grouped_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "duplicate group key in spilled partition",
                )));
            }
        }
        Ok(partition)
    }

    /// Mutates one admitted group while keeping its storage reservation live.
    /// The temporary serialized lookup key is admitted before allocation.
    pub(crate) fn with_value_accounted<F, U>(
        &mut self,
        key: &[Value],
        default: F,
        update: U,
    ) -> Result<(), OperatorError>
    where
        F: FnOnce(&mut GroupedMemoryOwner) -> Result<V, OperatorError>,
        U: FnOnce(&mut V, &mut GroupedMemoryOwner) -> Result<(), OperatorError>,
    {
        let idx = self.partition_for(key);
        let mut counter = ByteCounter(0);
        serialize_row(key, &mut counter).map_err(grouped_io_error)?;
        // Retain the query's grant authority without a fresh pressure sweep
        // for every lookup. Growth still enforces the shared hard limit.
        let mut key_owner = self
            .metadata_owner
            .as_mut()
            .expect("accounted caller has partition admission")
            .split(0)?;
        key_owner.try_grow(counter.0)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(counter.0)
            .map_err(|_| grouped_memory_limit())?;
        serialize_row(key, &mut bytes).map_err(grouped_io_error)?;
        {
            let partition = self.get_partition_mut(idx).map_err(grouped_io_error)?;
            if !partition.entries.contains_key(bytes.as_slice()) {
                let needed = partition
                    .entries
                    .len()
                    .checked_add(1)
                    .ok_or_else(grouped_memory_limit)?;
                if needed > partition.entries.capacity() {
                    let new_bytes =
                        grouped_hash_table_bytes::<(SerializedKey, PartitionEntry<V>)>(needed)?;
                    let mut table_owner = match partition.table_owner.take() {
                        Some(owner) => owner,
                        None => GroupedMemoryOwner::try_new(Arc::clone(key_owner.manager()), 0)?,
                    };
                    let old_bytes = table_owner.size();
                    // Old and new tables coexist while hashbrown rehashes.
                    if let Err(error) = table_owner.try_grow(new_bytes) {
                        partition.table_owner = Some(table_owner);
                        return Err(error);
                    }
                    if partition.entries.try_reserve(1).is_err() {
                        table_owner.try_resize(table_owner.size() - new_bytes)?;
                        partition.table_owner = Some(table_owner);
                        return Err(grouped_memory_limit());
                    }
                    table_owner.try_resize(new_bytes)?;
                    partition.owned_bytes = partition.owned_bytes - old_bytes + new_bytes;
                    partition.table_owner = Some(table_owner);
                }
                let value = default(&mut key_owner)?;
                let before = key_owner.size();
                partition.owned_bytes = partition
                    .owned_bytes
                    .checked_add(before)
                    .ok_or_else(grouped_memory_limit)?;
                let entry =
                    partition
                        .entries
                        .entry(SerializedKey(bytes))
                        .or_insert(PartitionEntry {
                            num_key_columns: key.len(),
                            value,
                            owner: Some(key_owner),
                        });
                let owner = entry.owner.as_mut().expect("accounted group");
                let outcome = update(&mut entry.value, owner);
                partition.owned_bytes = partition.owned_bytes - before + owner.size();
                self.partition_sizes[idx] += 1;
                return outcome;
            }
            let entry = partition
                .entries
                .get_mut(bytes.as_slice())
                .expect("group existence checked above");
            let owner = entry
                .owner
                .as_mut()
                .expect("accounted entry retains its admission");
            let before = owner.size();
            let outcome = update(&mut entry.value, owner);
            partition.owned_bytes = partition.owned_bytes - before + owner.size();
            outcome
        }
    }

    pub(crate) fn into_owned_drain(self) -> Result<OwnedPartitionDrainCursor<V>, OperatorError> {
        if self.metadata_owner.is_none() || self.admitted_deserializer.is_none() {
            return Err(OperatorError::Internal(
                "owned drain requires admitted state".to_string(),
            ));
        }
        let remaining = self.total_size();
        Ok(OwnedPartitionDrainCursor {
            resident: None,
            reader: None,
            index: 0,
            remaining,
            state: self,
        })
    }

    /// Actual admitted resident storage, excluding authoritative disk files.
    pub(crate) fn accounted_bytes(&self) -> usize {
        let metadata = self
            .metadata_owner
            .as_ref()
            .map_or(0, GroupedMemoryOwner::size);
        self.partitions
            .iter()
            .flatten()
            .fold(metadata, |total, partition| {
                total.saturating_add(partition.owned_bytes)
            })
    }

    /// Returns whether a partition is in memory.
    #[must_use]
    pub fn is_in_memory(&self, partition_idx: usize) -> bool {
        self.partitions[partition_idx].is_some()
    }

    /// Returns the number of groups in a partition.
    #[must_use]
    pub fn partition_size(&self, partition_idx: usize) -> usize {
        self.partition_sizes[partition_idx]
    }

    /// Returns the total number of groups across all partitions.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.partition_sizes.iter().sum()
    }

    /// Returns the number of in-memory partitions.
    #[must_use]
    pub fn in_memory_count(&self) -> usize {
        self.partitions.iter().filter(|p| p.is_some()).count()
    }

    /// Returns the number of spilled partitions.
    #[must_use]
    pub fn spilled_count(&self) -> usize {
        self.spill_files.iter().filter(|f| f.is_some()).count()
    }

    /// Spills a specific partition to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_partition(&mut self, partition_idx: usize) -> std::io::Result<usize> {
        // Keep the resident owner until the entire write has completed.
        let Some(partition) = self.partitions[partition_idx].as_ref() else {
            return Ok(0);
        };
        if partition.is_empty() {
            return Ok(0);
        }
        let mut spill_file = self.manager.create_file("partition")?;
        let written = (|| {
            let mut writer = SpillWriterAdapter(&mut spill_file);
            write_u64(&mut writer, partition.len() as u64)?;
            for (key, entry) in partition.iter() {
                write_u64(&mut writer, key.0.len() as u64)?;
                writer.write_all(&key.0)?;
                write_u64(&mut writer, entry.num_key_columns as u64)?;
                (self.value_serializer)(&entry.value, &mut writer)?;
            }
            spill_file.finish_write()
        })();
        if let Err(error) = written {
            let path = spill_file.path().to_path_buf();
            match spill_file.delete() {
                Ok(()) => self.manager.unregister_file(&path),
                Err(remove_error) if remove_error.kind() == std::io::ErrorKind::NotFound => {
                    self.manager.unregister_file(&path);
                }
                Err(_) => {} // The manager retains authority over this path.
            }
            // This staged file never registered committed spilled bytes.
            return Err(error);
        }
        let bytes_written = spill_file.bytes_written();
        self.manager.register_spilled_bytes(bytes_written);
        self.partition_sizes[partition_idx] = partition.len();
        self.spill_files[partition_idx] = Some(spill_file);
        self.partitions[partition_idx] = None;
        usize::try_from(bytes_written)
            .map_err(|_| std::io::Error::other("spilled byte count exceeds usize"))
    }

    /// Spills the largest in-memory partition.
    ///
    /// Returns the number of bytes spilled, or 0 if no partition to spill.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_largest(&mut self) -> std::io::Result<usize> {
        // Find largest in-memory partition
        let largest_idx = self
            .partitions
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| p.as_ref().map(|m| (idx, m.len())))
            .max_by_key(|(_, size)| *size)
            .map(|(idx, _)| idx);

        match largest_idx {
            Some(idx) => self.spill_partition(idx),
            None => Ok(0),
        }
    }

    /// Spills the least recently used in-memory partition.
    ///
    /// Returns the number of bytes spilled, or 0 if no partition to spill.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_lru(&mut self) -> std::io::Result<usize> {
        // Find LRU in-memory partition
        let lru_idx = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_some())
            .min_by_key(|(idx, _)| self.access_times[*idx])
            .map(|(idx, _)| idx);

        match lru_idx {
            Some(idx) => self.spill_partition(idx),
            None => Ok(0),
        }
    }

    /// Inserts or updates a value for a key.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn insert(&mut self, key: Vec<Value>, value: V) -> std::io::Result<Option<V>> {
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        let serialized_key = SerializedKey::from_values(&key);
        let partition = self.get_partition_mut(partition_idx)?;

        let old = partition.insert(
            serialized_key,
            PartitionEntry {
                num_key_columns,
                value,
                owner: None,
            },
        );

        if old.is_none() {
            self.partition_sizes[partition_idx] += 1;
        }

        Ok(old.map(|e| e.value))
    }

    /// Gets a value for a key.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn get(&mut self, key: &[Value]) -> std::io::Result<Option<&V>> {
        let partition_idx = self.partition_for(key);
        let serialized_key = SerializedKey::from_values(key);
        let partition = self.get_partition_mut(partition_idx)?;
        Ok(partition.get(&serialized_key).map(|e| &e.value))
    }

    /// Gets a mutable value for a key, or inserts a default.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    ///
    /// # Panics
    ///
    /// Panics if the key was just inserted but cannot be found (invariant violation).
    pub fn get_or_insert_with<F>(&mut self, key: Vec<Value>, default: F) -> std::io::Result<&mut V>
    where
        F: FnOnce() -> V,
    {
        let partition_idx = self.partition_for(&key);
        let num_key_columns = key.len();
        let serialized_key = SerializedKey::from_values(&key);

        let was_new;
        {
            let partition = self.get_partition_mut(partition_idx)?;
            was_new = !partition.contains_key(&serialized_key);
            if was_new {
                partition.insert(
                    serialized_key.clone(),
                    PartitionEntry {
                        num_key_columns,
                        value: default(),
                        owner: None,
                    },
                );
            }
        }
        if was_new {
            self.partition_sizes[partition_idx] += 1;
        }

        let partition = self.get_partition_mut(partition_idx)?;
        // Invariant: key was either already present or inserted in the block above
        Ok(&mut partition
            .get_mut(&serialized_key)
            .expect("key exists: just inserted or already present in partition")
            .value)
    }

    /// Drains all entries from all partitions.
    ///
    /// Loads spilled partitions as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn drain_all(&mut self) -> std::io::Result<Vec<(Vec<Value>, V)>> {
        let mut result = Vec::with_capacity(self.total_size());

        for partition_idx in 0..self.num_partitions {
            let partition = self.get_partition_mut(partition_idx)?;
            for (serialized_key, entry) in partition.drain() {
                let key = serialized_key.to_values(entry.num_key_columns)?;
                result.push((key, entry.value));
            }
            self.partition_sizes[partition_idx] = 0;
        }

        // Clean up any remaining spill files
        for spill_file in &mut self.spill_files {
            if let Some(file) = spill_file.take() {
                let bytes = file.bytes_written();
                let _ = file.delete();
                self.manager.unregister_spilled_bytes(bytes);
            }
        }

        Ok(result)
    }

    /// Iterates over all entries without draining.
    ///
    /// Loads spilled partitions as needed.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from disk fails.
    pub fn iter_all(&mut self) -> std::io::Result<Vec<(Vec<Value>, V)>> {
        let mut result = Vec::with_capacity(self.total_size());

        for partition_idx in 0..self.num_partitions {
            let partition = self.get_partition_mut(partition_idx)?;
            for (serialized_key, entry) in partition.iter() {
                let key = serialized_key.to_values(entry.num_key_columns)?;
                result.push((key, entry.value.clone()));
            }
        }

        Ok(result)
    }

    /// Cleans up all spill files.
    pub fn cleanup(&mut self) {
        release_spill_files(&self.manager, &mut self.spill_files);

        self.spill_files.clear();
        self.partitions.clear();
        for _ in 0..self.num_partitions {
            self.spill_files.push(None);
            self.partitions.push(Some(ResidentPartition::new()));
        }
        self.partition_sizes = vec![0; self.num_partitions];
    }
}

impl<V> Drop for PartitionedState<V> {
    fn drop(&mut self) {
        release_spill_files(&self.manager, &mut self.spill_files);
    }
}

/// Releases completed files. Failed removals stay tracked by the manager,
/// which retains cleanup authority and byte accounting until its own cleanup.
fn release_spill_files(manager: &SpillManager, files: &mut [Option<SpillFile>]) {
    for slot in files {
        let Some(file) = slot.as_ref() else {
            continue;
        };
        let removed = match std::fs::remove_file(file.path()) {
            Ok(()) => true,
            Err(error) => error.kind() == std::io::ErrorKind::NotFound,
        };
        if removed {
            manager.unregister_file(file.path());
            manager.unregister_spilled_bytes(file.bytes_written());
        }
        // A completed spill file has no open writer. The manager still owns
        // the path of any file that could not be removed.
        *slot = None;
    }
}

/// Hashes a key (vector of values) to a u64.
fn hash_key(key: &[Value]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for value in key {
        match value {
            Value::Null => 0u8.hash(&mut hasher),
            Value::Bool(b) => {
                1u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Value::Int64(n) => {
                2u8.hash(&mut hasher);
                n.hash(&mut hasher);
            }
            Value::Float64(f) => {
                3u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            Value::String(s) => {
                4u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            Value::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Value::Timestamp(t) => {
                6u8.hash(&mut hasher);
                t.hash(&mut hasher);
            }
            Value::Date(d) => {
                10u8.hash(&mut hasher);
                d.hash(&mut hasher);
            }
            Value::Time(t) => {
                11u8.hash(&mut hasher);
                t.hash(&mut hasher);
            }
            Value::Duration(d) => {
                12u8.hash(&mut hasher);
                d.hash(&mut hasher);
            }
            Value::ZonedDatetime(zdt) => {
                14u8.hash(&mut hasher);
                zdt.hash(&mut hasher);
            }
            Value::List(l) => {
                7u8.hash(&mut hasher);
                l.len().hash(&mut hasher);
            }
            Value::Map(m) => {
                8u8.hash(&mut hasher);
                m.len().hash(&mut hasher);
            }
            Value::Vector(v) => {
                9u8.hash(&mut hasher);
                v.len().hash(&mut hasher);
                // Hash first few elements for distribution
                for &f in v.iter().take(4) {
                    f.to_bits().hash(&mut hasher);
                }
            }
            Value::Path { nodes, edges } => {
                13u8.hash(&mut hasher);
                nodes.len().hash(&mut hasher);
                edges.len().hash(&mut hasher);
            }
            Value::GCounter(counts) => {
                15u8.hash(&mut hasher);
                counts.len().hash(&mut hasher);
            }
            Value::OnCounter { pos, neg } => {
                16u8.hash(&mut hasher);
                pos.len().hash(&mut hasher);
                neg.len().hash(&mut hasher);
            }
            _ => {
                255u8.hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// Counts codec bytes without constructing a serialized value buffer.
struct ByteCounter(usize);
impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other(grouped_memory_limit()))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Helper to read u64 in little endian.
fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Helper to write u64 in little endian.
fn write_u64<W: Write>(writer: &mut W, value: u64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Streams the existing codec directly into the bounded file writer.
struct SpillWriterAdapter<'a>(&'a mut SpillFile);

impl Write for SpillWriterAdapter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Adapter to read from SpillFileReader through std::io::Read.
struct SpillReaderAdapter<'a>(&'a mut super::file::SpillFileReader);

impl Read for SpillReaderAdapter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read_exact(buf)?;
        Ok(buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Creates a test manager. Returns (TempDir, manager). TempDir must be kept alive.
    fn create_manager() -> (TempDir, Arc<SpillManager>) {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());
        (temp_dir, manager)
    }

    /// Simple i64 serializer for tests.
    #[allow(clippy::trivially_copy_pass_by_ref)] // Required by PartitionedState::new signature
    fn serialize_i64(value: &i64, w: &mut dyn Write) -> std::io::Result<()> {
        w.write_all(&value.to_le_bytes())
    }

    /// Simple i64 deserializer for tests.
    fn deserialize_i64(r: &mut dyn Read) -> std::io::Result<i64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn key(values: &[i64]) -> Vec<Value> {
        values.iter().map(|&v| Value::Int64(v)).collect()
    }

    #[test]
    fn test_partition_for() {
        let (_temp_dir, manager) = create_manager();
        let state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Same key should always go to same partition
        let k1 = key(&[1, 2, 3]);
        let p1 = state.partition_for(&k1);
        let p2 = state.partition_for(&k1);
        assert_eq!(p1, p2);

        // Partition should be in range
        assert!(p1 < 16);
    }

    #[test]
    fn test_insert_and_get() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Insert some values
        state.insert(key(&[1]), 100).unwrap();
        state.insert(key(&[2]), 200).unwrap();
        state.insert(key(&[3]), 300).unwrap();

        assert_eq!(state.total_size(), 3);

        // Get values
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&100));
        assert_eq!(state.get(&key(&[2])).unwrap(), Some(&200));
        assert_eq!(state.get(&key(&[3])).unwrap(), Some(&300));
        assert_eq!(state.get(&key(&[4])).unwrap(), None);
    }

    #[test]
    fn test_get_or_insert_with() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // First access creates the entry
        let v1 = state.get_or_insert_with(key(&[1]), || 42).unwrap();
        assert_eq!(*v1, 42);

        // Second access returns existing value
        let v2 = state.get_or_insert_with(key(&[1]), || 100).unwrap();
        assert_eq!(*v2, 42);

        // Mutate via returned reference
        *state.get_or_insert_with(key(&[1]), || 0).unwrap() = 999;
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&999));
    }

    #[test]
    fn failed_spill_retains_resident_state() {
        let (_directory, manager) = create_manager();
        let mut state = PartitionedState::new(
            Arc::clone(&manager),
            4,
            |_value: &i64, _writer: &mut dyn Write| {
                Err(std::io::Error::other("serializer refused"))
            },
            deserialize_i64,
        );
        let values = key(&[7]);
        let partition = state.partition_for(&values);
        state.insert(values.clone(), 42).unwrap();
        assert!(state.spill_partition(partition).is_err());
        assert!(
            state.is_in_memory(partition),
            "failed spill must keep the resident owner"
        );
        assert_eq!(state.total_size(), 1);
        assert_eq!(state.get(&values).unwrap(), Some(&42));
        assert_eq!(state.spilled_count(), 0);
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn test_spill_and_reload() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values that go to different partitions
        for i in 0..20 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        let initial_total = state.total_size();
        assert!(initial_total > 0);

        // Spill the largest partition
        let bytes_spilled = state.spill_largest().unwrap();
        assert!(bytes_spilled > 0);
        assert!(state.spilled_count() > 0);

        // Values should still be accessible (reloads from disk)
        for i in 0..20 {
            let expected = i * 10;
            assert_eq!(state.get(&key(&[i])).unwrap(), Some(&expected));
        }
    }

    #[test]
    fn test_spill_lru() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        state.insert(key(&[1]), 10).unwrap();
        state.insert(key(&[2]), 20).unwrap();
        state.insert(key(&[3]), 30).unwrap();

        // Access key 3 to make it recently used
        state.get(&key(&[3])).unwrap();

        // Spill LRU - should not spill partition containing key 3
        state.spill_lru().unwrap();

        // Key 3 should still be in memory
        let partition_idx = state.partition_for(&key(&[3]));
        assert!(state.is_in_memory(partition_idx));
    }

    #[test]
    fn test_drain_all() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        for i in 0..10 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        // Spill some partitions
        state.spill_largest().unwrap();
        state.spill_largest().unwrap();

        // Drain all
        let entries = state.drain_all().unwrap();
        assert_eq!(entries.len(), 10);

        // Verify all entries are present
        let mut values: Vec<i64> = entries.iter().map(|(_, v)| *v).collect();
        values.sort_unstable();
        assert_eq!(values, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);

        // State should be empty
        assert_eq!(state.total_size(), 0);
    }

    #[test]
    fn test_iter_all() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert values
        for i in 0..5 {
            state.insert(key(&[i]), i * 10).unwrap();
        }

        // Iterate without draining
        let entries = state.iter_all().unwrap();
        assert_eq!(entries.len(), 5);

        // State should still have values
        assert_eq!(state.total_size(), 5);

        // Should be able to iterate again
        let entries2 = state.iter_all().unwrap();
        assert_eq!(entries2.len(), 5);
    }

    #[test]
    fn test_many_groups() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 16, serialize_i64, deserialize_i64);

        // Insert many groups
        for i in 0..1000 {
            state.insert(key(&[i]), i).unwrap();
        }

        assert_eq!(state.total_size(), 1000);

        // Spill multiple partitions
        for _ in 0..8 {
            state.spill_largest().unwrap();
        }

        assert!(state.spilled_count() >= 8);

        // All values should still be retrievable
        for i in 0..1000 {
            assert_eq!(state.get(&key(&[i])).unwrap(), Some(&i));
        }
    }

    #[test]
    fn test_cleanup() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(Arc::clone(&manager), 4, serialize_i64, deserialize_i64);

        // Insert and spill
        for i in 0..20 {
            state.insert(key(&[i]), i).unwrap();
        }
        state.spill_largest().unwrap();
        state.spill_largest().unwrap();

        let spilled_before = manager.spilled_bytes();
        assert!(spilled_before > 0);

        // Cleanup
        state.cleanup();

        assert_eq!(state.total_size(), 0);
        assert_eq!(state.spilled_count(), 0);
    }

    #[test]
    fn test_multi_column_key() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 8, serialize_i64, deserialize_i64);

        // Insert with multi-column keys
        state
            .insert(vec![Value::String("a".into()), Value::Int64(1)], 100)
            .unwrap();
        state
            .insert(vec![Value::String("a".into()), Value::Int64(2)], 200)
            .unwrap();
        state
            .insert(vec![Value::String("b".into()), Value::Int64(1)], 300)
            .unwrap();

        assert_eq!(state.total_size(), 3);

        // Retrieve by multi-column key
        assert_eq!(
            state
                .get(&[Value::String("a".into()), Value::Int64(1)])
                .unwrap(),
            Some(&100)
        );
        assert_eq!(
            state
                .get(&[Value::String("a".into()), Value::Int64(2)])
                .unwrap(),
            Some(&200)
        );
        assert_eq!(
            state
                .get(&[Value::String("b".into()), Value::Int64(1)])
                .unwrap(),
            Some(&300)
        );
    }

    #[test]
    fn test_update_existing() {
        let (_temp_dir, manager) = create_manager();
        let mut state: PartitionedState<i64> =
            PartitionedState::new(manager, 4, serialize_i64, deserialize_i64);

        // Insert
        state.insert(key(&[1]), 100).unwrap();
        assert_eq!(state.total_size(), 1);

        // Update
        let old = state.insert(key(&[1]), 200).unwrap();
        assert_eq!(old, Some(100));
        assert_eq!(state.total_size(), 1); // Size shouldn't increase

        // Verify update
        assert_eq!(state.get(&key(&[1])).unwrap(), Some(&200));
    }
}
