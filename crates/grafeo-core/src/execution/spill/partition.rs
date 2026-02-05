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
    partitions: Vec<Option<HashMap<SerializedKey, PartitionEntry<V>>>>,
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
            partitions.push(Some(HashMap::new()));
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
        }
    }

    /// Returns the partition index for a key.
    #[must_use]
    pub fn partition_for(&self, key: &[Value]) -> usize {
        let hash = hash_key(key);
        hash as usize % self.num_partitions
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
    ) -> std::io::Result<&mut HashMap<SerializedKey, PartitionEntry<V>>> {
        self.touch(partition_idx);

        // If partition is in memory, return it
        if self.partitions[partition_idx].is_some() {
            // Invariant: just checked is_some() above
            return Ok(self.partitions[partition_idx]
                .as_mut()
                .expect("partition is Some: checked on previous line"));
        }

        // Load from disk
        if let Some(spill_file) = self.spill_files[partition_idx].take() {
            let loaded = self.load_partition(&spill_file)?;
            // Delete the spill file after loading
            let bytes = spill_file.bytes_written();
            let _ = spill_file.delete();
            self.manager.unregister_spilled_bytes(bytes);
            self.partitions[partition_idx] = Some(loaded);
        } else {
            // Neither in memory nor on disk - create empty partition
            self.partitions[partition_idx] = Some(HashMap::new());
        }

        // Invariant: partition was either loaded from disk or created empty above
        Ok(self.partitions[partition_idx]
            .as_mut()
            .expect("partition is Some: set to Some in if/else branches above"))
    }

    /// Loads a partition from a spill file.
    fn load_partition(
        &self,
        spill_file: &SpillFile,
    ) -> std::io::Result<HashMap<SerializedKey, PartitionEntry<V>>> {
        let mut reader = spill_file.reader()?;
        let mut adapter = SpillReaderAdapter(&mut reader);

        let num_entries = read_u64(&mut adapter)? as usize;
        let mut partition = HashMap::with_capacity(num_entries);

        for _ in 0..num_entries {
            // Read key
            let key_len = read_u64(&mut adapter)? as usize;
            let mut key_buf = vec![0u8; key_len];
            adapter.read_exact(&mut key_buf)?;
            let serialized_key = SerializedKey(key_buf);

            // Read number of key columns
            let num_key_columns = read_u64(&mut adapter)? as usize;

            // Read value
            let value = (self.value_deserializer)(&mut adapter)?;

            partition.insert(
                serialized_key,
                PartitionEntry {
                    num_key_columns,
                    value,
                },
            );
        }

        Ok(partition)
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
        // Get partition data
        let partition = match self.partitions[partition_idx].take() {
            Some(p) => p,
            None => return Ok(0), // Already spilled
        };

        if partition.is_empty() {
            return Ok(0);
        }

        // Create spill file
        let mut spill_file = self.manager.create_file("partition")?;

        // Write partition data
        let mut buf = Vec::new();
        write_u64(&mut buf, partition.len() as u64)?;

        for (key, entry) in &partition {
            // Write key bytes
            write_u64(&mut buf, key.0.len() as u64)?;
            buf.extend_from_slice(&key.0);

            // Write number of key columns
            write_u64(&mut buf, entry.num_key_columns as u64)?;

            // Write value
            (self.value_serializer)(&entry.value, &mut buf)?;
        }

        spill_file.write_all(&buf)?;
        spill_file.finish_write()?;

        let bytes_written = spill_file.bytes_written();
        self.manager.register_spilled_bytes(bytes_written);
        self.partition_sizes[partition_idx] = partition.len();
        self.spill_files[partition_idx] = Some(spill_file);

        Ok(bytes_written as usize)
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
        for spill_file in self.spill_files.iter_mut() {
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
        for file in self.spill_files.iter_mut().flatten() {
            let bytes = file.bytes_written();
            self.manager.unregister_spilled_bytes(bytes);
        }

        self.spill_files.clear();
        self.partitions.clear();
        for _ in 0..self.num_partitions {
            self.spill_files.push(None);
            self.partitions.push(Some(HashMap::new()));
        }
        self.partition_sizes = vec![0; self.num_partitions];
    }
}

impl<V> Drop for PartitionedState<V> {
    fn drop(&mut self) {
        // Unregister spilled bytes
        for file in self.spill_files.iter().flatten() {
            let bytes = file.bytes_written();
            self.manager.unregister_spilled_bytes(bytes);
        }
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
            Value::List(l) => {
                7u8.hash(&mut hasher);
                l.len().hash(&mut hasher);
            }
            Value::Map(m) => {
                8u8.hash(&mut hasher);
                m.len().hash(&mut hasher);
            }
        }
    }

    hasher.finish()
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

/// Adapter to read from SpillFileReader through std::io::Read.
struct SpillReaderAdapter<'a>(&'a mut super::file::SpillFileReader);

impl<'a> Read for SpillReaderAdapter<'a> {
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
        values.sort();
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
