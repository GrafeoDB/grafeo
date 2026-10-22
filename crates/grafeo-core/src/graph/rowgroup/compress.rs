//! Compression on request (H5): `compact()` encodes the hot property columns
//! of every row group into cold chunks in memory, in a database file or in
//! memory alike, so a column a write or a read decoded goes cold again.
//!
//! A column is encoded as a spill encodes it (`column`), a vector of rows a
//! chunk, and keeps the chunks on the heap instead of in a file: reads go
//! through each chunk's point view, a write thaws the column (#412 decision
//! Q9). What snapshot isolation needs stays: a column with update chains
//! stays hot, as the readers that must not see a write read the values the
//! chains kept (a cold column holds none), so compressing after a collect
//! takes every column no open reader still needs the chains of; a column
//! whose chunks would have no point view (strings, values of several kinds)
//! stays hot too, as its first read would decode it back. The rows' version
//! info is not touched: a collect folds it. Each node row group also merges
//! its adjacency deltas, dropping the edges a collect took out.
//!
//! Compression changes how the store holds its values, not what any reader
//! at any view reads, so it marks nothing for the next checkpoint.

use grafeo_common::utils::error::Result;
use parking_lot::RwLockUpgradableReadGuard;

use super::RowGroupStore;
use super::column::Compressed;

/// What [`RowGroupStore::compress`] did, over a root and its named graphs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompressReport {
    /// The nodes and edges of the row groups in which it encoded a column.
    pub rows: usize,
    /// The columns it encoded: one property key in one row group each.
    pub columns: usize,
    /// The columns it left hot for their update chains.
    pub kept_with_chains: usize,
    /// The columns it left hot because their chunks would have no point
    /// view.
    pub kept_unsupported: usize,
    /// The heap bytes the encoded columns held, less what their chunks hold
    /// (an estimate, as the store's memory figures are).
    pub freed_bytes: usize,
}

impl CompressReport {
    fn add(&mut self, other: Self) {
        self.rows += other.rows;
        self.columns += other.columns;
        self.kept_with_chains += other.kept_with_chains;
        self.kept_unsupported += other.kept_unsupported;
        self.freed_bytes += other.freed_bytes;
    }

    fn count(&mut self, compressed: &Compressed, rows: usize) {
        self.columns += compressed.columns;
        self.kept_with_chains += compressed.kept_with_chains;
        self.kept_unsupported += compressed.kept_unsupported;
        self.freed_bytes += compressed.freed_bytes;
        if compressed.columns > 0 {
            self.rows += rows;
        }
    }
}

impl RowGroupStore {
    /// Encodes the hot property columns of this store's row groups, and of
    /// its named graphs', into cold chunks in memory (see the module docs).
    ///
    /// Each row group is encoded under the store's upgradable lock, so
    /// readers go on while it is encoded and no write changes it between the
    /// encoding and the swap; the swap takes the write lock.
    ///
    /// # Errors
    ///
    /// Returns the encoder's error for a value the format cannot hold; the
    /// row groups compressed before it stay so, the others stay as they
    /// were.
    pub fn compress(&self) -> Result<CompressReport> {
        let mut report = self.compress_graph()?;
        for name in self.graph_names() {
            if let Some(graph) = self.graph(&name) {
                report.add(graph.compress_graph()?);
            }
        }
        Ok(report)
    }

    /// [`compress`](Self::compress) of this store alone.
    fn compress_graph(&self) -> Result<CompressReport> {
        let mut report = CompressReport::default();
        let (nodes, edges) = {
            let inner = self.inner.read();
            (
                inner.nodes.keys().copied().collect::<Vec<u64>>(),
                inner.edges.keys().copied().collect::<Vec<u64>>(),
            )
        };
        for index in nodes {
            let inner = self.inner.upgradable_read();
            let Some(group) = inner.nodes.get(&index) else {
                continue;
            };
            let plan = group.columns.plan_compress()?;
            let merge = !group.outgoing.is_merged() || !group.incoming.is_merged();
            if plan.is_empty() && !merge {
                report.count(&plan.counts(), 0);
                continue;
            }
            let rows = group.versions.rows().count();
            let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
            let Some(group) = inner.nodes.get_mut(&index) else {
                continue;
            };
            let compressed = group.columns.install_compressed(plan)?;
            if merge {
                group.outgoing.merge();
                group.incoming.merge();
            }
            report.count(&compressed, rows);
        }
        for index in edges {
            let inner = self.inner.upgradable_read();
            let Some(group) = inner.edges.get(&index) else {
                continue;
            };
            let plan = group.columns.plan_compress()?;
            if plan.is_empty() {
                report.count(&plan.counts(), 0);
                continue;
            }
            let rows = group.versions.rows().count();
            let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
            let Some(group) = inner.edges.get_mut(&index) else {
                continue;
            };
            let compressed = group.columns.install_compressed(plan)?;
            report.count(&compressed, rows);
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
