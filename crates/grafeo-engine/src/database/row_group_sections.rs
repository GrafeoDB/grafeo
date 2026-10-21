//! Checkpoint and load of the row-group store (#412 S8), the database's
//! store since the switch (H2d).
//!
//! A checkpoint of the root goes through the database's sources, flush and
//! file ([`CheckpointSources`](super::sections::CheckpointSources)): the
//! catalog section, which reads the root's named graphs and index
//! definitions, and
//! the LPG section of version 4, written at the checkpoint's epoch as a
//! reader at it sees each row, so it holds the committed state without a
//! committed copy or the open transactions' change sets. An open reads the
//! sections back into a root ([`load_row_group_sections`]), builds every
//! graph's indexes from the rows ([`restore_row_group_indexes`]) and moves
//! the root, its named graphs and the transaction manager to one epoch
//! ([`continue_row_group_epochs`]).
//!
//! No vector or text index section is written: the row-group store's text
//! and vector indexes hold each node's newest text and vector, open
//! transactions' included, so the load builds them from the rows at the
//! checkpoint's epoch instead. A property index is built from the rows too.
//!
//! A database that keeps history (H2c) checkpoints it from its history
//! floor: the LPG section's `History` chunks and the `TIME_INDEX` section.
//! The load gives them back to the root and the transaction manager, the
//! root keeps history from then on (replay keeps what it replaces), and the
//! open collects at the collect horizon, which keeps the retention window.

use std::sync::Arc;

use grafeo_common::storage::{ImageSource, Section, SectionType};
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::Result;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::rowgroup::section::RowGroupStoreSection;

use super::catalog_section::GraphIndexes;
use crate::catalog::Catalog;
use crate::config::Config;
use crate::transaction::{HistoryImage, TransactionManager};

/// The row-group root a database with `config` builds: one that keeps each
/// node's incoming edges, or with `backward_edges` off keeps none, reads
/// them from the edge table and checkpoints no incoming adjacency (10b). An
/// open passes the configuration it applied the file's stored one to
/// (#551), so a database reopens with the backward edges it was created
/// with; its named graphs, and the ones a load creates, follow the root.
pub(super) fn row_group_root(config: &Config) -> Arc<RowGroupStore> {
    Arc::new(RowGroupStore::with_backward_edges(config.backward_edges()))
}

/// Loads the sections of `image` into `root`, a row-group root, and
/// `catalog` (for [`load_sections`](super::sections::load_sections)): the
/// catalog first (with its refusal of data this
/// build cannot read, for the image of the database file `file`), then the
/// LPG section into the root and its named graphs (version 4, or migrated
/// from version 3 or 0.5.x, see `RowGroupStoreSection::read_from`), the RDF
/// triples, the compacted base of a 0.5.x file folded into the root (whose
/// LPG section holds the base's overlay), and every graph's indexes, built
/// from the rows.
///
/// # Errors
///
/// The errors of reading a section and of folding a compacted base; the
/// refusal of data this build cannot read; [`Error::Serialization`](grafeo_common::utils::error::Error::Serialization) for an
/// LPG section of a version no release wrote.
pub(super) fn load_row_group_sections(
    image: &dyn ImageSource,
    file: Option<&std::path::Path>,
    root: &Arc<RowGroupStore>,
    catalog: &Arc<Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
) -> Result<Option<HistoryImage>> {
    let (indexes, catalog_epoch) = super::sections::read_catalog(image, file, root, catalog)?;
    let history = super::time_index_section::read_history(image)?;
    if let Some(source) = image.section_source(SectionType::LpgStore) {
        RowGroupStoreSection::new(Arc::clone(root)).read_from(&*source)?;
    }
    #[cfg(feature = "triple-store")]
    super::sections::load_triples(image, rdf_store)?;
    // Before the indexes: they are built from all the data.
    #[cfg(feature = "grafeo-file")]
    super::sections::fold_compacted_base(image, &**root)?;
    #[cfg(not(feature = "grafeo-file"))]
    if image.section_source(SectionType::CompactStore).is_some() {
        return Err(grafeo_common::utils::error::Error::Serialization(
            "the image holds the compacted base of a 0.5.x file, which only a build with the              `grafeo-file` feature folds"
                .to_string(),
        ));
    }
    root.sync_epoch(EpochId::new(catalog_epoch));
    restore_row_group_indexes(root, indexes);
    Ok(history)
}

/// Gives the history a load read (`history`, the image of the file's
/// `TIME_INDEX` section) to `transaction_manager`, before a replay records
/// its groups: a file without one retains nothing before the epoch `root`
/// was loaded at. From here on `root` keeps history as the database does
/// (replay keeps what it replaces), and answers reads from the floor on.
pub(super) fn restore_row_group_history(
    root: &RowGroupStore,
    transaction_manager: &TransactionManager,
    history: Option<HistoryImage>,
) {
    transaction_manager.restore_history_or_floor(history, root.current_epoch());
    root.set_keeps_history(transaction_manager.history_image().is_some());
    root.set_history_floor(transaction_manager.history_floor());
}

/// Builds the indexes `graphs` defines (the catalog's) in `root` and its
/// named graphs from their rows: property indexes, and vector and text
/// indexes with the parameters and options their definitions hold. A graph
/// the root does not hold is skipped, as `restore_indexes` skips it. An
/// index that cannot be built is left out with a warning, as
/// `GrafeoDB::finish_load` leaves it: the data is intact, and creating the
/// index builds it again.
pub(super) fn restore_row_group_indexes(root: &Arc<RowGroupStore>, graphs: Vec<GraphIndexes>) {
    for graph in graphs {
        let target = match &graph.graph {
            None => Arc::clone(root),
            Some(name) => match root.graph(name) {
                Some(target) => target,
                None => continue,
            },
        };
        for property in &graph.property {
            target.create_property_index(property);
        }
        #[cfg(feature = "vector-index")]
        for def in &graph.vector {
            let quantization = def.quantization.and_then(super::index::quantization_name);
            match super::index::vector_index_from_data(
                &*target,
                &def.label,
                &def.property,
                Some(def.dimensions),
                Some(def.metric.name()),
                Some(def.m),
                Some(def.ef_construction),
                quantization,
            ) {
                Ok(index) => target.add_vector_index(&def.label, &def.property, Arc::new(index)),
                Err(err) => grafeo_common::grafeo_warn!(
                    "vector index :{}({}) not rebuilt: {err}",
                    def.label,
                    def.property
                ),
            }
        }
        #[cfg(feature = "text-index")]
        for def in &graph.text {
            let index = super::index::text_index_from_data(
                &*target,
                &def.label,
                &def.property,
                &def.options,
            );
            target.add_text_index(
                &def.label,
                &def.property,
                Arc::new(parking_lot::RwLock::new(index)),
            );
        }
    }
}

/// Moves `root`, every named graph and `transaction_manager` to one epoch,
/// the highest a store reached (for `GrafeoDB::continue_epochs`): a load
/// gives each graph the section's epoch, and the
/// transaction manager starts at 0. Reads then see every loaded version, a
/// checkpoint records that epoch, and new commits continue above it.
///
/// No reader runs before the open ends, so the versions the load and the
/// replay of a WAL stamped are collected at the collect horizon: that epoch
/// for a database without history, so a replayed or migrated database
/// costs what one loaded from a section of version 4 does and what the
/// replay deleted is gone; the start of the retention window for one that
/// keeps history, whose history then reaches back as before the reopen.
/// The history goes on from that epoch.
pub(super) fn continue_row_group_epochs(
    root: &RowGroupStore,
    transaction_manager: &TransactionManager,
) {
    let graphs: Vec<Arc<RowGroupStore>> = root
        .graph_names()
        .iter()
        .filter_map(|name| root.graph(name))
        .collect();
    let epoch = graphs
        .iter()
        .map(|graph| graph.current_epoch())
        .fold(root.current_epoch(), std::cmp::max);
    root.sync_epoch(epoch);
    for graph in &graphs {
        graph.sync_epoch(epoch);
    }
    transaction_manager.sync_epoch(epoch);
    root.gc_versions(transaction_manager.collect_horizon());
    transaction_manager.anchor_history();
}

#[cfg(test)]
mod tests;
