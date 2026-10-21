//! Column spill (P8): a node property column leaves the heap under a memory
//! limit, as `LpgStore`'s spilled columns do, and comes back.
//!
//! A spill encodes each row group's column into cold chunks (`column`), a
//! vector of rows each, and hands their bytes to a [`ColumnSpill`]: the
//! engine writes them into a new spill file and maps it, so the store does
//! no file I/O. The column then holds cold chunks whose bytes are the
//! mapping's, and reads each row through the chunk's point view: queries,
//! vector search and the section writer of a checkpoint read the file, and
//! the column stays cold. A write to the column thaws it (#412 decision Q9):
//! the row group's column comes back into memory and lets go of its file. A
//! reload thaws every row group's column. The mapping goes, and with it the
//! file, when no chunk reads it any more: after a thaw, or when the store
//! is dropped.
//!
//! A row group's column with update chains is not spilled: a cold column
//! holds no chains, and the readers that must not see a write read the value
//! a chain kept. A collect that drops the chains lets the next spill take
//! it. A column whose chunks would have no point view (strings, values of
//! several kinds) is not spilled either: its first read would decode it
//! back onto the heap.
//!
//! A spill file is temporary: it holds a copy of what the column holds, and
//! a checkpoint writes the column's values into the database file as it
//! writes any column. After a crash the data comes from the database file
//! and its WAL, and the files a crash leaves are the engine's to remove.
//! Later (H3) the same chunks get block ids in the database file, which the
//! buffer manager reloads from there.

use bytes::Bytes;
use grafeo_common::types::PropertyKey;
use grafeo_common::utils::error::{Error, Result};
use parking_lot::RwLockUpgradableReadGuard;

use super::RowGroupStore;
use super::column::SpillPlan;

/// Where a spill writes a column's chunks: a file the engine writes and maps
/// (the store does no file I/O).
pub trait ColumnSpill: Send + Sync {
    /// Writes `bytes` into a new spill file and returns them as the file
    /// holds them, mapped. The returned bytes, and every slice of them, read
    /// the file until the last of them is dropped, which removes the file.
    ///
    /// # Errors
    ///
    /// Returns the error of creating, writing or mapping the file; nothing
    /// is left behind then.
    fn write(&self, bytes: &[u8]) -> std::io::Result<Bytes>;
}

/// What a spill of a column did, over its row groups.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ColumnSpillReport {
    /// The row groups whose column went into a spill file of its own.
    pub spilled_groups: usize,
    /// The row groups whose column stayed in memory for its update chains.
    pub kept_with_chains: usize,
    /// The row groups whose column stayed in memory for a chunk without a
    /// point view.
    pub kept_unsupported: usize,
    /// The heap bytes the spilled columns held, less what their cold chunks
    /// hold on the heap (an estimate, as the store's memory figures are).
    pub freed_bytes: usize,
    /// The bytes written to spill files.
    pub file_bytes: usize,
}

impl RowGroupStore {
    /// Spills the node property column of `key` into files `spill` writes,
    /// a row group at a time (see the module docs): each row group's column
    /// that holds values, no update chains and only kinds with a point view
    /// goes into a file of its own, and reads from it from then on.
    ///
    /// Each row group is spilled under the store's upgradable lock, so no
    /// write changes the column between its encoding and the swap: the
    /// readers that hold the lock go on while its file is written, a writer
    /// waits for the file (and, the lock being fair, the readers after it
    /// wait too).
    ///
    /// # Errors
    ///
    /// Returns the error of writing a file (as [`Error::Io`]) or of encoding
    /// a value; the row groups spilled before it stay spilled, the others
    /// stay as they were.
    pub fn spill_node_property_column(
        &self,
        key: &PropertyKey,
        spill: &dyn ColumnSpill,
    ) -> Result<ColumnSpillReport> {
        let mut report = ColumnSpillReport::default();
        let (key, groups) = {
            let inner = self.inner.read();
            let Some(key) = inner.keys.get_id(key.as_str()) else {
                return Ok(report);
            };
            (key, inner.nodes.keys().copied().collect::<Vec<u64>>())
        };
        for number in groups {
            let inner = self.inner.upgradable_read();
            let Some(group) = inner.nodes.get(&number) else {
                continue;
            };
            let chunks = match group.columns.plan_spill(key)? {
                SpillPlan::Nothing => continue,
                SpillPlan::Chained => {
                    report.kept_with_chains += 1;
                    continue;
                }
                SpillPlan::Unsupported => {
                    report.kept_unsupported += 1;
                    continue;
                }
                SpillPlan::Write(chunks) => chunks,
            };
            let mapped = spill.write(&chunks.bytes).map_err(Error::Io)?;
            let resident = chunks.resident;
            let file_bytes = chunks.bytes.len();
            let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
            let group = inner
                .nodes
                .get_mut(&number)
                .ok_or_else(|| Error::Internal("a spilled row group is gone".to_string()))?;
            let after = group.columns.install_spill(key, chunks, &mapped)?;
            report.spilled_groups += 1;
            report.file_bytes += file_bytes;
            report.freed_bytes += resident.saturating_sub(after);
        }
        Ok(report)
    }

    /// Brings the spilled node property column of `key` back into memory in
    /// every row group, letting go of its files; returns whether a row group
    /// held it spilled.
    ///
    /// # Errors
    ///
    /// Returns a chunk's decode error; that row group's column then stays
    /// spilled, the ones reloaded before it stay reloaded.
    pub fn reload_node_property_column(&self, key: &PropertyKey) -> Result<bool> {
        let mut inner = self.inner.write();
        let Some(key) = inner.keys.get_id(key.as_str()) else {
            return Ok(false);
        };
        let mut reloaded = false;
        for group in inner.nodes.values_mut() {
            reloaded |= group.columns.reload(key)?;
        }
        Ok(reloaded)
    }

    /// The keys of the node property columns some row group holds spilled,
    /// in key order.
    #[must_use]
    pub fn spilled_node_property_columns(&self) -> Vec<PropertyKey> {
        let inner = self.inner.read();
        let mut ids: Vec<u32> = inner
            .nodes
            .values()
            .flat_map(|group| group.columns.spilled_keys())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let mut keys: Vec<PropertyKey> = ids
            .into_iter()
            .filter_map(|id| inner.keys.get_name(id))
            .map(|name| PropertyKey::new(name.as_str()))
            .collect();
        keys.sort_unstable();
        keys
    }
}

#[cfg(test)]
mod tests;
