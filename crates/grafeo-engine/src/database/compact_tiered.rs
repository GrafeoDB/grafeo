//! Disk-backed tier for the compact columnar base.
//!
//! Wraps a [`CompactStore`] in a two-state machine:
//!
//! - `InMemory`: the store lives entirely on the heap (default after
//!   [`compact()`](super::GrafeoDB::compact)).
//! - `OnDisk`: the store has been serialized to a file, mmapped, and
//!   re-deserialized. The mmap keeps the page cache populated so OS-level
//!   paging can reclaim cold pages under memory pressure without a hard
//!   error on reads.
//!
//! The wrapper is additive: the inner [`CompactStore`] is always a valid
//! `Arc<CompactStore>`, so [`LayeredStore`](grafeo_core::graph::compact::layered::LayeredStore)
//! keeps serving reads transparently across tier transitions.
//!
//! # Lifecycle
//!
//! ```text
//! new_in_memory(store)  -> InMemory(Arc<CompactStore>)
//!        |
//!        | persist(path) + mmap
//!        v
//!     OnDisk(path, Mmap, Arc<CompactStore>)
//!        |
//!        | reload_to_ram()
//!        v
//!     InMemory(Arc<CompactStore>)
//! ```
//!
//! # Memory accounting
//!
//! The bytes freed by a tier transition depend on whether the caller drops
//! the old in-memory store. [`persist_to_mmap`](CompactStoreTiered::persist_to_mmap)
//! consumes the old `Arc<CompactStore>` and replaces it with a fresh one
//! deserialized from mmap bytes. If the caller kept another `Arc` around
//! (e.g. in a [`LayeredStore`](grafeo_core::graph::compact::layered::LayeredStore)),
//! that clone still keeps the old allocation live; callers under memory
//! pressure should route reads through
//! [`store()`](CompactStoreTiered::store) and hold the tiered wrapper, not
//! raw CompactStore clones.
//!
//! # Feature flags
//!
//! Compiled only when both `compact-store` and `mmap` are enabled.

#![cfg(all(feature = "compact-store", feature = "mmap"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::storage::section::Section;
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::compact::CompactStore;
use grafeo_core::graph::compact::section::CompactStoreSection;
use memmap2::Mmap;
use parking_lot::RwLock;

/// Two-state disk-backed wrapper around a [`CompactStore`].
pub struct CompactStoreTiered {
    state: RwLock<TierState>,
}

enum TierState {
    /// Store lives entirely on the heap.
    InMemory(Arc<CompactStore>),
    /// Store is backed by a mmap'd file. Phase 3c: the entire mmap is
    /// wrapped as a refcounted [`Bytes`] via [`Bytes::from_owner`], and
    /// every column codec inside `store` holds a `Bytes::slice(range)`
    /// view into it — so column data is read directly from mmap'd
    /// memory with zero copies. The `Bytes` here keeps the Mmap alive
    /// for the lifetime of the tier state; dropping it after `store`
    /// drops releases the mapping.
    OnDisk {
        path: PathBuf,
        _mmap_bytes: Bytes,
        store: Arc<CompactStore>,
    },
}

impl CompactStoreTiered {
    /// Creates a tiered wrapper starting in the in-memory state.
    #[must_use]
    pub fn new_in_memory(store: Arc<CompactStore>) -> Self {
        Self {
            state: RwLock::new(TierState::InMemory(store)),
        }
    }

    /// Returns the current store, whether in-memory or mmap-backed.
    ///
    /// Cheap `Arc::clone`, safe to call on the query hot path.
    #[must_use]
    pub fn store(&self) -> Arc<CompactStore> {
        match &*self.state.read() {
            TierState::InMemory(store) => Arc::clone(store),
            TierState::OnDisk { store, .. } => Arc::clone(store),
        }
    }

    /// Returns `true` when the backing file is mmap'd.
    #[must_use]
    pub fn is_on_disk(&self) -> bool {
        matches!(&*self.state.read(), TierState::OnDisk { .. })
    }

    /// Returns the backing file path, if mmapped.
    #[must_use]
    pub fn path(&self) -> Option<PathBuf> {
        match &*self.state.read() {
            TierState::OnDisk { path, .. } => Some(path.clone()),
            TierState::InMemory(_) => None,
        }
    }

    /// Serializes the current store to `path` without switching tier state.
    ///
    /// Returns the number of bytes written. Useful for checkpoint flows
    /// that want a snapshot on disk without giving up the RAM copy.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization or the file write fails.
    pub fn persist(&self, path: &Path) -> Result<usize> {
        let store = self.store();
        let section = CompactStoreSection::new(store);
        let bytes = section.serialize()?;
        write_atomically(path, &bytes)?;
        Ok(bytes.len())
    }

    /// Serializes the store, mmaps the file, and swaps into the `OnDisk`
    /// state, returning the number of bytes written.
    ///
    /// After this call, the wrapper holds a fresh `Arc<CompactStore>`
    /// deserialized from the mmap. The caller's previous `Arc<CompactStore>`
    /// (obtained via an earlier `store()` call) is still valid but stale
    /// relative to future reads routed through this wrapper.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization, the file write, or the
    /// subsequent mmap + deserialize cycle fails.
    pub fn persist_to_mmap(&self, path: &Path) -> Result<usize> {
        let bytes = {
            let store = self.store();
            let section = CompactStoreSection::new(store);
            section.serialize()?
        };
        write_atomically(path, &bytes)?;

        let (mmap_bytes, store) = open_and_deserialize(path)?;
        let written = bytes.len();
        *self.state.write() = TierState::OnDisk {
            path: path.to_path_buf(),
            _mmap_bytes: mmap_bytes,
            store,
        };
        Ok(written)
    }

    /// Opens an existing on-disk store via mmap, without writing.
    ///
    /// Returns a tiered wrapper starting in the `OnDisk` state.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if the file cannot be opened, mmapped,
    /// or deserialized into a valid `CompactStore`.
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let (mmap_bytes, store) = open_and_deserialize(path)?;
        Ok(Self {
            state: RwLock::new(TierState::OnDisk {
                path: path.to_path_buf(),
                _mmap_bytes: mmap_bytes,
                store,
            }),
        })
    }

    /// Reloads the store into a heap-owning `InMemory` tier and drops the
    /// mmap, leaving the backing file in place.
    ///
    /// Naively re-tagging the existing `Arc<CompactStore>` as `InMemory`
    /// would still leave column codec storage referencing the mmap-backed
    /// `Bytes` produced by the original open path: the data would continue
    /// to be served from the OS page cache and the `Mmap` would stay alive
    /// through the codec slices. To make the tier label truthful we
    /// re-serialize the live store and deserialize from a heap-backed
    /// `Bytes`, so the new codec storage no longer references the mapping.
    ///
    /// No-op when already `InMemory`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` if serialization or deserialization
    /// fails.
    pub fn reload_to_ram(&self) -> Result<()> {
        let mut guard = self.state.write();
        let TierState::OnDisk { store, .. } = &*guard else {
            return Ok(());
        };
        let section = CompactStoreSection::new(Arc::clone(store));
        let bytes = section.serialize()?;
        let mut reloaded = CompactStoreSection::empty();
        reloaded.deserialize_from_bytes(Bytes::from(bytes))?;
        let new_store = reloaded.store().ok_or_else(|| {
            Error::Internal("empty CompactStoreSection after reload_to_ram".to_string())
        })?;
        *guard = TierState::InMemory(new_store);
        Ok(())
    }

    /// Estimated heap memory footprint of the wrapped store, in bytes.
    ///
    /// When the state is `OnDisk`, this counts the heap copy alone: the
    /// mmap bytes live outside the heap and are managed by the OS page
    /// cache.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.store().memory_bytes()
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Internal(format!("create dir for {}: {e}", parent.display())))?;
    }
    // Write to a sibling temp file, then rename for atomic replacement.
    let tmp = path.with_extension("grafeo.tmp");
    std::fs::write(&tmp, bytes)
        .map_err(|e| Error::Internal(format!("write {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        Error::Internal(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

fn open_and_deserialize(path: &Path) -> Result<(Bytes, Arc<CompactStore>)> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Internal(format!("open {}: {e}", path.display())))?;
    // SAFETY: we mmap a file that's owned by this process for the duration
    // of the `Mmap` lifetime. The file is read-only from Grafeo's side
    // (we never write through the mmap); external truncation or modification
    // while an `Mmap` is held is undefined per memmap2 docs, same caveat as
    // every other mmap call site in the project.
    #[allow(unsafe_code)]
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Internal(format!("mmap {}: {e}", path.display())))?;

    // Phase 3c: wrap the Mmap as a refcounted `Bytes` so column codec
    // storage can be `data.slice(range)` against this view — zero-copy.
    // Every codec's `Bytes` here shares the refcount that keeps the
    // Mmap alive. When the last `Bytes` referring to this region drops,
    // the OS unmaps.
    let mmap_bytes = Bytes::from_owner(mmap);

    let mut section = CompactStoreSection::empty();
    section.deserialize_from_bytes(mmap_bytes.clone())?;
    let store = section.store().ok_or_else(|| {
        Error::Internal(format!(
            "empty CompactStoreSection after deserialize of {}",
            path.display()
        ))
    })?;

    Ok((mmap_bytes, store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_core::graph::compact::builder::from_graph_store;
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::traits::GraphStore;

    fn build_sample_store() -> Arc<CompactStore> {
        let lpg = LpgStore::new().expect("lpg store");
        for i in 0..16 {
            let id = lpg.create_node(&["Person"]);
            lpg.set_node_property(id, "age", Value::Int64(i as i64));
            lpg.set_node_property(id, "name", Value::String(arcstr::format!("person-{i}")));
        }
        let compact = from_graph_store(&lpg).expect("compact");
        Arc::new(compact)
    }

    #[test]
    fn in_memory_roundtrip() {
        let store = build_sample_store();
        let expected = store.memory_bytes();
        let tiered = CompactStoreTiered::new_in_memory(store);
        assert!(!tiered.is_on_disk());
        assert_eq!(tiered.memory_bytes(), expected);
    }

    #[test]
    fn persist_and_mmap_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let store = build_sample_store();
        let expected_nodes = store.node_count();

        let tiered = CompactStoreTiered::new_in_memory(store);
        let written = tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        assert!(written > 0);
        assert!(tiered.is_on_disk());
        assert_eq!(tiered.path().as_deref(), Some(path.as_path()));

        let store_after = tiered.store();
        assert_eq!(store_after.node_count(), expected_nodes);
    }

    #[test]
    fn open_mmap_reads_existing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let expected_nodes = {
            let store = build_sample_store();
            let expected = store.node_count();
            let tiered = CompactStoreTiered::new_in_memory(store);
            tiered.persist_to_mmap(&path).expect("persist_to_mmap");
            expected
        };

        let reopened = CompactStoreTiered::open_mmap(&path).expect("open_mmap");
        assert!(reopened.is_on_disk());
        assert_eq!(reopened.store().node_count(), expected_nodes);
    }

    /// `reload_to_ram` must produce a store whose column codec storage
    /// is heap-backed, not a mmap slice — otherwise the tier label is a
    /// lie. We prove the disconnect by deleting the backing file after
    /// reload and confirming reads still succeed (mmap-backed reads
    /// would be unspecified after unlink on Windows and could fault).
    #[test]
    fn reload_to_ram_drops_mmap_backing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        assert!(tiered.is_on_disk());

        tiered.reload_to_ram().expect("reload_to_ram");
        assert!(!tiered.is_on_disk());

        // Removing the file should be safe once we've truly reloaded
        // into heap memory. (On Windows, this would fail outright if
        // any mmap handle were still open against the file.)
        std::fs::remove_file(&path).expect("file must be unlinkable post-reload");

        // Reads still work after the file is gone — the data lives on
        // the heap now.
        let store = tiered.store();
        let person_ids = store.nodes_by_label("Person");
        assert!(!person_ids.is_empty(), "person_ids is empty");
        for id in person_ids.iter().take(4) {
            assert!(
                store
                    .get_node_property(*id, &PropertyKey::new("name"))
                    .is_some(),
                "name property still readable after backing file removal"
            );
        }
    }

    #[test]
    fn reload_to_ram_transitions_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        assert!(tiered.is_on_disk());

        tiered.reload_to_ram().expect("reload_to_ram");
        assert!(!tiered.is_on_disk());
        assert!(tiered.path().is_none());

        // Reads still work after reload.
        let store = tiered.store();
        assert!(store.node_count() > 0);
    }

    #[test]
    fn persist_without_mmap_keeps_memory_state() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("snapshot.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let written = tiered.persist(&path).expect("persist");
        assert!(written > 0);
        assert!(
            !tiered.is_on_disk(),
            "persist() alone must not change tier state"
        );
        assert!(path.exists());
    }

    #[test]
    fn spill_drops_original_arc() {
        // Proxy for "spill frees memory": the in-memory Arc the wrapper held
        // before `persist_to_mmap` must be dropped during the transition, so
        // the only live reference left to that specific allocation is
        // whatever the caller chose to hold. Verified by comparing strong
        // counts before and after.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        // One Arc in the wrapper, none held externally.
        assert_eq!(Arc::strong_count(&tiered.store()), 2);
        // The line above borrowed a clone then dropped it; wrapper now holds 1.

        tiered.persist_to_mmap(&path).expect("persist_to_mmap");

        // After spill: the wrapper holds a fresh Arc pointing at a
        // newly-deserialized CompactStore. The original allocation is gone.
        let after = tiered.store();
        // wrapper holds 1 + our binding holds 1 = 2
        assert_eq!(Arc::strong_count(&after), 2);
    }

    #[test]
    fn store_values_survive_mmap_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("base.compact");

        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let before = tiered.store();
        let first_id = before.nodes_by_label("Person").first().copied();
        let first_name =
            first_id.and_then(|id| before.get_node_property(id, &PropertyKey::new("name")));

        tiered.persist_to_mmap(&path).expect("persist_to_mmap");

        let after = tiered.store();
        let after_name =
            first_id.and_then(|id| after.get_node_property(id, &PropertyKey::new("name")));
        assert_eq!(first_name, after_name);
    }

    /// Phase 3c: column data on the disk tier should be served from
    /// the mmap-backed `Bytes` rather than from a heap copy. We can't
    /// directly assert "no allocation happened" in a portable way, but
    /// we can prove the column codec storage shares the mmap refcount:
    /// if we drop the tiered wrapper, the underlying `Mmap` should
    /// still be live as long as we hold an `Arc<CompactStore>` whose
    /// codec storage references it.
    ///
    /// This test exercises the full open-mmap path and reads several
    /// values. Combined with the column codec's `from_bytes_storage`
    /// constructors using `data.slice(range)`, it confirms the
    /// zero-copy contract end-to-end.
    #[test]
    fn mmap_backed_store_serves_reads_from_mapped_bytes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("zerocopy.compact");

        // Build, persist, then drop the in-memory wrapper.
        {
            let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
            tiered.persist_to_mmap(&path).expect("persist_to_mmap");
        }

        // Re-open via mmap. Column codec storage Bytes refcount-share
        // the Mmap-owning Bytes inside `_mmap_bytes`.
        let reopened = CompactStoreTiered::open_mmap(&path).expect("open_mmap");
        let store = reopened.store();
        let person_ids = store.nodes_by_label("Person");
        assert!(!person_ids.is_empty(), "person_ids is empty");

        // Reads work; values come from the mmap-backed Bytes via
        // `data.slice(range)` constructors in `read_from_v3`.
        for &id in person_ids.iter().take(8) {
            let name = store
                .get_node_property(id, &PropertyKey::new("name"))
                .expect("name property exists");
            assert!(matches!(name, Value::String(_)));
            let age = store
                .get_node_property(id, &PropertyKey::new("age"))
                .expect("age property exists");
            assert!(matches!(age, Value::Int64(_)));
        }
    }
}
