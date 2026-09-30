//! The sections of a database: what a checkpoint writes, and loading them
//! back. Opening a `.grafeo` file and `to_memory()` both load through
//! [`load_sections`], so a copy holds what a reopen would.

use std::sync::Arc;

use grafeo_common::storage::Section;
#[cfg(feature = "lpg")]
use grafeo_common::storage::SectionType;
#[cfg(feature = "lpg")]
use grafeo_common::utils::error::Result;

#[cfg(feature = "lpg")]
use super::catalog_section::{CatalogSection, GraphIndexes};
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;

/// Everything a checkpoint writes: the database's complete state.
///
/// A checkpoint's container holds only the sections built here, so every
/// checkpoint path (`close()`, `wal_checkpoint()`, backups, the async
/// snapshot and the periodic timer) builds them from these sources, and
/// `to_memory()` copies a database through them.
#[derive(Clone)]
pub(super) struct CheckpointSources {
    /// The LPG store; the overlay after `compact()`.
    #[cfg(feature = "lpg")]
    pub store: Option<Arc<grafeo_core::graph::lpg::LpgStore>>,
    /// The compacted base and overlay, after `compact()`.
    #[cfg(all(feature = "compact-store", feature = "lpg"))]
    pub layered: Option<Arc<grafeo_core::graph::compact::layered::LayeredStore>>,
    #[cfg(feature = "lpg")]
    pub catalog: Arc<crate::catalog::Catalog>,
    pub transaction_manager: Arc<crate::transaction::TransactionManager>,
    #[cfg(feature = "triple-store")]
    pub rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
}

impl CheckpointSources {
    /// Builds every section of the database.
    pub fn sections(&self) -> Vec<Box<dyn Section>> {
        #[cfg_attr(
            not(any(feature = "lpg", feature = "triple-store")),
            expect(
                unused_mut,
                reason = "only the lpg and triple-store features add sections"
            )
        )]
        let mut sections: Vec<Box<dyn Section>> = Vec::new();

        #[cfg(feature = "lpg")]
        if let Some(store) = &self.store {
            let transaction_manager = Arc::clone(&self.transaction_manager);
            sections.push(Box::new(CatalogSection::new(
                Arc::clone(&self.catalog),
                Arc::clone(store),
                move || transaction_manager.current_epoch().as_u64(),
            )));

            #[cfg(feature = "compact-store")]
            let layered = self.push_layered(&mut sections);
            #[cfg(not(feature = "compact-store"))]
            let layered = false;
            if !layered {
                sections.push(Box::new(grafeo_core::graph::lpg::LpgStoreSection::new(
                    Arc::clone(store),
                )));
            }

            // Vector indexes: persist HNSW topology to avoid rebuild on load
            #[cfg(feature = "vector-index")]
            {
                let indexes = store.vector_index_entries();
                if !indexes.is_empty() {
                    sections.push(Box::new(
                        grafeo_core::index::vector::VectorStoreSection::new(indexes),
                    ));
                }
            }

            // Text indexes: persist BM25 postings to avoid rebuild on load
            #[cfg(feature = "text-index")]
            {
                let indexes = store.text_index_entries();
                if !indexes.is_empty() {
                    sections.push(Box::new(grafeo_core::index::text::TextIndexSection::new(
                        indexes,
                    )));
                }
            }
        }

        #[cfg(feature = "triple-store")]
        if !self.rdf_store.is_empty() || self.rdf_store.graph_count() > 0 {
            sections.push(Box::new(grafeo_core::graph::rdf::RdfStoreSection::new(
                Arc::clone(&self.rdf_store),
            )));
        }

        #[cfg(feature = "ring-index")]
        if self.rdf_store.ring().is_some() {
            sections.push(Box::new(grafeo_core::index::ring::RdfRingSection::new(
                Arc::clone(&self.rdf_store),
            )));
        }

        sections
    }

    /// Adds the compacted base, the overlay and the overlay's deletions, or
    /// returns `false` when the database is not compacted.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    fn push_layered(&self, sections: &mut Vec<Box<dyn Section>>) -> bool {
        use grafeo_core::graph::compact::deletions_section::OverlayDeletionsSection;
        use grafeo_core::graph::compact::section::CompactStoreSection;

        let Some(layered) = &self.layered else {
            return false;
        };
        sections.push(Box::new(CompactStoreSection::new(layered.base_store_arc())));
        sections.push(Box::new(grafeo_core::graph::lpg::LpgStoreSection::new(
            layered.overlay_store(),
        )));
        // Tombstones for base nodes and edges not yet merged into the base:
        // without them a reopen would bring deleted entities back.
        let deletions = OverlayDeletionsSection::from_layered(Arc::clone(layered));
        if deletions.is_empty() {
            layered.mark_deletions_clean();
        } else {
            sections.push(Box::new(deletions));
        }
        true
    }
}

/// Reads a section's bytes by type: from a file, or from memory.
#[cfg(feature = "lpg")]
pub(super) type ReadSection<'a> = dyn FnMut(SectionType) -> Result<Option<Vec<u8>>> + 'a;

/// What loading leaves for [`GrafeoDB::finish_load`](super::GrafeoDB) to do
/// once the database is built.
#[cfg(feature = "lpg")]
#[derive(Default)]
pub(super) struct LoadedSections {
    /// The compacted base; the loaded LPG store is its overlay.
    #[cfg(feature = "compact-store")]
    compact_base: Option<Arc<grafeo_core::graph::compact::CompactStore>>,
    /// Base nodes and edges the overlay deleted.
    #[cfg(feature = "compact-store")]
    overlay_deletions: Option<(
        Vec<grafeo_common::types::NodeId>,
        Vec<grafeo_common::types::EdgeId>,
    )>,
    /// Vector and text indexes the sections did not hold, built from the
    /// data once the database is built.
    unbuilt: Vec<GraphIndexes>,
}

/// Loads the sections that `read` returns into the stores and the catalog,
/// and puts the indexes back: property indexes from the data, the default
/// graph's vector and text indexes from their own sections.
///
/// The indexes exist before WAL recovery, which keeps them current. What is
/// left (see [`LoadedSections`]) is finished by `GrafeoDB::finish_load`.
///
/// # Errors
///
/// Returns an error if a section cannot be read or decoded.
#[cfg(feature = "lpg")]
pub(super) fn load_sections(
    read: &mut ReadSection<'_>,
    store: &Arc<LpgStore>,
    catalog: &Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
) -> Result<LoadedSections> {
    let mut loaded = LoadedSections::default();

    // The catalog first: the schema is needed before the data.
    let mut indexes = Vec::new();
    if let Some(data) = read(SectionType::Catalog)? {
        let transaction_manager = Arc::new(crate::transaction::TransactionManager::new());
        let mut section = CatalogSection::new(Arc::clone(catalog), Arc::clone(store), move || {
            transaction_manager.current_epoch().as_u64()
        });
        section.deserialize(&data)?;
        indexes = section.take_loaded_indexes();
    }

    // With a compacted base, this is the overlay.
    if let Some(data) = read(SectionType::LpgStore)? {
        grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store)).deserialize(&data)?;
    }

    #[cfg(feature = "triple-store")]
    if let Some(data) = read(SectionType::RdfStore)? {
        grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(rdf_store)).deserialize(&data)?;
    }

    #[cfg(feature = "ring-index")]
    if let Some(data) = read(SectionType::RdfRing)? {
        grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store)).deserialize(&data)?;
    }

    #[cfg(feature = "compact-store")]
    {
        use grafeo_core::graph::compact::deletions_section::OverlayDeletionsSection;
        use grafeo_core::graph::compact::section::CompactStoreSection;

        if let Some(data) = read(SectionType::CompactStore)? {
            let mut section = CompactStoreSection::empty();
            section.deserialize(&data)?;
            loaded.compact_base = section.store();
        }
        if let Some(data) = read(SectionType::OverlayDeletions)? {
            let mut section = OverlayDeletionsSection::empty();
            section.deserialize(&data)?;
            loaded.overlay_deletions = Some(section.take());
        }
    }

    loaded.unbuilt = restore_indexes(read, store, indexes)?;
    Ok(loaded)
}

/// Builds the property indexes of every graph and restores the default
/// graph's vector and text indexes from their sections.
///
/// Returns the vector and text indexes left to build from the data: those
/// of named graphs (the sections hold only the default graph's), quantized
/// ones (the section holds the HNSW graph, not the quantized codes) and any
/// a section did not hold.
#[cfg(feature = "lpg")]
fn restore_indexes(
    read: &mut ReadSection<'_>,
    store: &Arc<LpgStore>,
    graphs: Vec<GraphIndexes>,
) -> Result<Vec<GraphIndexes>> {
    let mut unbuilt = Vec::new();
    let mut from_sections = GraphIndexes::default();
    for graph in graphs {
        let target = match &graph.graph {
            None => Arc::clone(store),
            Some(name) => match store.graph(name) {
                Some(target) => target,
                None => continue,
            },
        };
        for property in &graph.property {
            target.create_property_index(property);
        }

        let default_graph = graph.graph.is_none();
        let mut rest = GraphIndexes {
            graph: graph.graph,
            ..GraphIndexes::default()
        };
        for def in graph.vector {
            if default_graph && def.quantization.is_none() {
                from_sections.vector.push(def);
            } else {
                rest.vector.push(def);
            }
        }
        if default_graph {
            from_sections.text = graph.text;
        } else {
            rest.text = graph.text;
        }
        if !rest.is_empty() {
            unbuilt.push(rest);
        }
    }

    let missing = restore_from_sections(read, store, from_sections)?;
    if !missing.is_empty() {
        unbuilt.push(missing);
    }
    Ok(unbuilt)
}

/// Restores the default graph's vector and text indexes from their
/// sections, and returns those the sections did not hold. A section that
/// cannot be decoded is skipped with a warning: its indexes are then built
/// from the data, which the index only mirrors.
#[cfg(feature = "lpg")]
#[cfg_attr(
    not(any(feature = "vector-index", feature = "text-index")),
    expect(
        unused_variables,
        reason = "without vector and text indexes there is nothing to restore"
    )
)]
fn restore_from_sections(
    read: &mut ReadSection<'_>,
    store: &LpgStore,
    indexes: GraphIndexes,
) -> Result<GraphIndexes> {
    #[cfg_attr(
        not(any(feature = "vector-index", feature = "text-index")),
        expect(unused_mut, reason = "only vector and text indexes are restored")
    )]
    let mut missing = GraphIndexes::default();

    #[cfg(feature = "vector-index")]
    if !indexes.vector.is_empty() {
        use grafeo_core::index::vector::{QuantizationType, VectorStoreSection};

        for def in &indexes.vector {
            let shell = super::GrafeoDB::build_vector_index(
                def.dimensions,
                def.metric,
                Some(def.m),
                Some(def.ef_construction),
                QuantizationType::None,
                0,
            );
            store.add_vector_index(&def.label, &def.property, Arc::new(shell));
        }
        if let Some(data) = read(SectionType::VectorStore)?
            && let Err(err) =
                VectorStoreSection::new(store.vector_index_entries()).deserialize(&data)
        {
            grafeo_common::grafeo_warn!("rebuilding vector indexes from the data: {err}");
            for def in &indexes.vector {
                store.remove_vector_index(&def.label, &def.property);
            }
        }
        for def in indexes.vector {
            let restored = store
                .get_vector_index(&def.label, &def.property)
                .is_some_and(|index| !index.is_empty());
            if !restored {
                store.remove_vector_index(&def.label, &def.property);
                missing.vector.push(def);
            }
        }
    }

    #[cfg(feature = "text-index")]
    if !indexes.text.is_empty() {
        use grafeo_core::index::text::{BM25Config, InvertedIndex, TextIndexSection};

        for (label, property) in &indexes.text {
            let shell = InvertedIndex::new(BM25Config::default());
            store.add_text_index(label, property, Arc::new(parking_lot::RwLock::new(shell)));
        }
        if let Some(data) = read(SectionType::TextIndex)?
            && let Err(err) = TextIndexSection::new(store.text_index_entries()).deserialize(&data)
        {
            grafeo_common::grafeo_warn!("rebuilding text indexes from the data: {err}");
            for (label, property) in &indexes.text {
                store.remove_text_index(label, property);
            }
        }
        for (label, property) in indexes.text {
            let restored = store
                .get_text_index(&label, &property)
                .is_some_and(|index| !index.read().is_empty());
            if !restored {
                store.remove_text_index(&label, &property);
                missing.text.push((label, property));
            }
        }
    }

    Ok(missing)
}

#[cfg(feature = "lpg")]
impl super::GrafeoDB {
    /// Finishes a load once the database is built: puts a compacted base
    /// under its overlay, then builds the indexes the sections did not hold
    /// from all the data.
    ///
    /// # Errors
    ///
    /// Returns an error if the compacted base cannot be wired.
    #[cfg_attr(
        not(feature = "compact-store"),
        expect(
            clippy::unnecessary_wraps,
            reason = "only wiring a compacted base can fail"
        )
    )]
    #[cfg_attr(
        not(feature = "compact-store"),
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "only wiring a compacted base changes the database"
        )
    )]
    pub(super) fn finish_load(&mut self, loaded: LoadedSections) -> Result<()> {
        #[cfg(feature = "compact-store")]
        if let Some(base) = loaded.compact_base {
            self.wire_layered_after_load(base, loaded.overlay_deletions)?;
        }
        for indexes in loaded.unbuilt {
            self.build_indexes(indexes);
        }
        Ok(())
    }

    /// Builds vector and text indexes from the data. One that cannot be
    /// built is left out with a warning: the data is intact, and
    /// `create_vector_index` or `create_text_index` builds it again.
    #[cfg_attr(
        not(any(feature = "vector-index", feature = "text-index")),
        expect(
            unused_variables,
            clippy::needless_pass_by_value,
            reason = "only vector and text indexes are built after a load"
        )
    )]
    fn build_indexes(&self, indexes: GraphIndexes) {
        #[cfg(feature = "vector-index")]
        for def in indexes.vector {
            let quantization = def.quantization.and_then(super::index::quantization_name);
            if let Err(err) = self.create_vector_index_in(
                indexes.graph.as_deref(),
                &def.label,
                &def.property,
                Some(def.dimensions),
                Some(def.metric.name()),
                Some(def.m),
                Some(def.ef_construction),
                quantization,
            ) {
                grafeo_common::grafeo_warn!(
                    "vector index :{}({}) not rebuilt: {err}",
                    def.label,
                    def.property
                );
            }
        }
        #[cfg(feature = "text-index")]
        for (label, property) in indexes.text {
            if let Err(err) = self.create_text_index_in(indexes.graph.as_deref(), &label, &property)
            {
                grafeo_common::grafeo_warn!("text index :{label}({property}) not rebuilt: {err}");
            }
        }
    }
}

#[cfg(all(
    test,
    feature = "gql",
    feature = "vector-index",
    feature = "text-index"
))]
mod tests {
    use super::*;
    use crate::GrafeoDB;

    /// The default graph's HNSW and text indexes come back from their
    /// sections, not from the data; quantized indexes and those of named
    /// graphs are left to build from the data once the database is built.
    #[test]
    fn the_default_graph_indexes_come_from_their_sections() {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Doc {emb: vector([1.0, 0.0]), body: 'graph'}), \
             (:Note {emb: vector([0.0, 1.0])})",
        )
        .unwrap();
        db.create_vector_index("Doc", "emb", None, None, None, None, None)
            .unwrap();
        db.create_vector_index("Note", "emb", None, None, None, None, Some("scalar"))
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        db.create_graph("model").unwrap();
        let model = db.graph("model").unwrap();
        model
            .execute("INSERT (:Doc {emb: vector([1.0, 0.0])})")
            .unwrap();
        model
            .execute("CREATE VECTOR INDEX model_emb ON :Doc(emb)")
            .unwrap();

        let mut sections: Vec<(SectionType, Vec<u8>)> = db
            .checkpoint_sources()
            .sections()
            .iter()
            .map(|section| (section.section_type(), section.serialize().unwrap()))
            .collect();
        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load_sections(
            &mut |section_type| {
                Ok(sections
                    .iter()
                    .position(|(stored, _)| *stored == section_type)
                    .map(|at| sections.swap_remove(at).1))
            },
            &store,
            &Arc::new(crate::catalog::Catalog::new()),
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        )
        .unwrap();

        assert_eq!(store.get_vector_index("Doc", "emb").unwrap().len(), 1);
        assert_eq!(store.get_text_index("Doc", "body").unwrap().read().len(), 1);
        assert!(
            store.get_vector_index("Note", "emb").is_none(),
            "a quantized index is built after the load"
        );
        let unbuilt: Vec<(Option<&str>, Vec<&str>)> = loaded
            .unbuilt
            .iter()
            .map(|graph| {
                let labels = graph.vector.iter().map(|def| def.label.as_str()).collect();
                (graph.graph.as_deref(), labels)
            })
            .collect();
        assert_eq!(
            unbuilt,
            [(None, vec!["Note"]), (Some("model"), vec!["Doc"])]
        );
    }
}
