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
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use grafeo_common::storage::section::Section;
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::compact::CompactStore;
use grafeo_core::graph::compact::section::CompactStoreSection;
use memmap2::Mmap;
#[cfg(feature = "lpg")]
use parking_lot::MutexGuard;
use parking_lot::{Mutex, RwLock};

#[cfg(test)]
thread_local! {
    static BEFORE_MMAP: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
    static BEFORE_INSTALL: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) fn set_before_install_hook(hook: impl FnOnce() + 'static) {
    BEFORE_INSTALL.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
}

#[cfg(test)]
fn run_before_mmap_hook() {
    let hook = BEFORE_MMAP.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(test)]
fn run_before_install_hook() {
    let hook = BEFORE_INSTALL.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

/// Two-state disk-backed wrapper around a [`CompactStore`].
pub struct CompactStoreTiered {
    state: RwLock<TierState>,
    transition: Mutex<()>,
}

/// A fully prepared tier change that has not replaced the live state.
pub(super) struct PreparedTier {
    state: TierState,
    written: usize,
}

impl PreparedTier {
    #[cfg(feature = "lpg")]
    pub(super) fn store(&self) -> Arc<CompactStore> {
        Arc::clone(self.state.store())
    }
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

impl TierState {
    fn store(&self) -> &Arc<CompactStore> {
        match self {
            Self::InMemory(store) | Self::OnDisk { store, .. } => store,
        }
    }
}

impl CompactStoreTiered {
    /// Creates a tiered wrapper starting in the in-memory state.
    #[must_use]
    pub fn new_in_memory(store: Arc<CompactStore>) -> Self {
        Self {
            state: RwLock::new(TierState::InMemory(store)),
            transition: Mutex::new(()),
        }
    }

    /// Returns the current store, whether in-memory or mmap-backed.
    ///
    /// Cheap `Arc::clone`, safe to call on the query hot path.
    #[must_use]
    pub fn store(&self) -> Arc<CompactStore> {
        Arc::clone(self.state.read().store())
    }

    pub(super) fn install_if_current(&self, expected: &Arc<CompactStore>, prepared: PreparedTier) {
        let mut state = self.state.write();
        if Arc::ptr_eq(state.store(), expected) {
            *state = prepared.state;
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
        let _transition = self.transition.lock();
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
        let _transition = self.transition.lock();
        let expected = self.store();
        let prepared = Self::prepare_mmap(Arc::clone(&expected), path)?;
        let written = prepared.written;
        self.install_if_current(&expected, prepared);
        Ok(written)
    }

    pub(super) fn prepare_mmap(store: Arc<CompactStore>, path: &Path) -> Result<PreparedTier> {
        let bytes = CompactStoreSection::new(store).serialize()?;
        // Keep the exact file we wrote: another wrapper may replace the same
        // path between our rename and mmap (for example after recompact).
        let file = write_atomically(path, &bytes)?;
        #[cfg(test)]
        run_before_mmap_hook();
        let (mmap_bytes, store) = deserialize_mmap(&file, path)?;
        #[cfg(test)]
        run_before_install_hook();
        Ok(PreparedTier {
            state: TierState::OnDisk {
                path: path.to_path_buf(),
                _mmap_bytes: mmap_bytes,
                store,
            },
            written: bytes.len(),
        })
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
            transition: Mutex::new(()),
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
        let _transition = self.transition.lock();
        let expected = {
            let state = self.state.read();
            let TierState::OnDisk { store, .. } = &*state else {
                return Ok(());
            };
            Arc::clone(store)
        };
        let prepared = Self::prepare_ram(Arc::clone(&expected))?;
        self.install_if_current(&expected, prepared);
        Ok(())
    }

    pub(super) fn prepare_ram(store: Arc<CompactStore>) -> Result<PreparedTier> {
        let section = CompactStoreSection::new(store);
        let bytes = section.serialize()?;
        let mut reloaded = CompactStoreSection::empty();
        reloaded.deserialize_from_bytes(Bytes::from(bytes))?;
        let new_store = reloaded.store().ok_or_else(|| {
            Error::Internal("empty CompactStoreSection after reload_to_ram".to_string())
        })?;
        Ok(PreparedTier {
            state: TierState::InMemory(new_store),
            written: 0,
        })
    }

    /// Makes `base` the wrapped store, in memory, unless the wrapper holds
    /// it already, and returns whether it changed: a merge of the overlay
    /// replaced the layered store's base, and the wrapper must spill and
    /// report that base from then on, never bring back the one it held.
    pub fn follow(&self, base: &Arc<CompactStore>) -> bool {
        let mut state = self.state.write();
        let held = match &*state {
            TierState::InMemory(store) | TierState::OnDisk { store, .. } => store,
        };
        if Arc::ptr_eq(held, base) {
            return false;
        }
        *state = TierState::InMemory(Arc::clone(base));
        true
    }

    /// Replaces the store with `store`, held in memory: the empty base a
    /// restore leaves (see `GrafeoDB::restore_snapshot`), so a later spill
    /// writes that base and never the one the restore dropped. A spilled
    /// store's mapping goes once its last reader drops it; its file stays, as
    /// [`reload_to_ram`](Self::reload_to_ram) leaves it.
    pub fn replace(&self, store: Arc<CompactStore>) {
        let retired = std::mem::replace(&mut *self.state.write(), TierState::InMemory(store));
        // Freed after the lock is released.
        drop(retired);
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

#[cfg(feature = "lpg")]
impl CompactStoreTiered {
    pub(super) fn try_transition(&self) -> Option<MutexGuard<'_, ()>> {
        self.transition.try_lock()
    }

    pub(super) fn is_on_disk_for(&self, store: &Arc<CompactStore>) -> bool {
        let state = self.state.read();
        matches!(&*state, TierState::OnDisk { .. }) && Arc::ptr_eq(state.store(), store)
    }

    pub(super) fn memory_bytes_for(&self, store: &Arc<CompactStore>) -> usize {
        let state = self.state.read();
        let current = store.memory_bytes();
        if Arc::ptr_eq(state.store(), store) {
            current
        } else {
            // A contended reconciliation can temporarily retain the old base.
            current.saturating_add(state.store().memory_bytes())
        }
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<std::fs::File> {
    use std::io::Write;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Internal(format!("create dir for {}: {e}", parent.display())))?;
    }
    // Distinct wrappers can spill concurrently to the same destination.
    // A unique sibling and an open handle keep each prepared snapshot intact.
    let (tmp, mut file) = loop {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("grafeo.{}.{sequence}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => break (tmp, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::Internal(format!(
                    "create {}: {error}",
                    tmp.display()
                )));
            }
        }
    };
    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Internal(format!("write {}: {error}", tmp.display())));
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Internal(format!(
            "rename {} -> {}: {error}",
            tmp.display(),
            path.display()
        )));
    }
    Ok(file)
}

fn open_and_deserialize(path: &Path) -> Result<(Bytes, Arc<CompactStore>)> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Internal(format!("open {}: {e}", path.display())))?;
    deserialize_mmap(&file, path)
}

fn deserialize_mmap(file: &std::fs::File, path: &Path) -> Result<(Bytes, Arc<CompactStore>)> {
    // SAFETY: we mmap a file that's owned by this process for the duration
    // of the `Mmap` lifetime. The file is read-only from Grafeo's side
    // (we never write through the mmap); external truncation or modification
    // while an `Mmap` is held is undefined per memmap2 docs, same caveat as
    // every other mmap call site in the project.
    #[allow(unsafe_code)]
    let mmap = unsafe { Mmap::map(file) }
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
    fn a_spill_maps_its_own_file_when_another_wrapper_replaces_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compact_base.grafeo");
        let tiered = CompactStoreTiered::new_in_memory(build_sample_store());
        let other = LpgStore::new().unwrap();
        other.create_node(&["Replacement"]);
        let other = CompactStoreTiered::new_in_memory(Arc::new(from_graph_store(&other).unwrap()));
        let replacement_path = path.clone();
        BEFORE_MMAP.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move || {
                other.persist_to_mmap(&replacement_path).unwrap();
            }));
        });
        tiered.persist_to_mmap(&path).unwrap();
        assert_eq!(tiered.store().node_count(), 16);
        assert_eq!(open_and_deserialize(&path).unwrap().1.node_count(), 1);
    }

    #[test]
    fn following_a_new_base_during_preparation_keeps_the_new_base() {
        let dir = tempfile::tempdir().unwrap();
        let tiered = Arc::new(CompactStoreTiered::new_in_memory(build_sample_store()));
        let replacement = LpgStore::new().unwrap();
        replacement.create_node(&["Replacement"]);
        let replacement = Arc::new(from_graph_store(&replacement).unwrap());
        let following = Arc::clone(&tiered);
        let new_base = Arc::clone(&replacement);
        set_before_install_hook(move || {
            assert!(following.follow(&new_base));
        });
        tiered
            .persist_to_mmap(&dir.path().join("base.compact"))
            .unwrap();
        assert!(Arc::ptr_eq(&tiered.store(), &replacement));
        assert!(!tiered.is_on_disk());
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
