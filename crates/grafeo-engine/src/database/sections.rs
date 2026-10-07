//! The sections of a database: what a checkpoint writes, and loading them
//! back. Opening a `.grafeo` file and `to_memory()` both load through
//! [`load_sections`], so a copy holds what a reopen would.

use std::sync::Arc;

use grafeo_common::storage::Section;
#[cfg(all(
    feature = "lpg",
    any(
        feature = "vector-index",
        feature = "text-index",
        feature = "ring-index"
    )
))]
use grafeo_common::storage::{ChunkMeta, SectionSource};
#[cfg(feature = "lpg")]
use grafeo_common::storage::{ImageSource, SectionType};
#[cfg(feature = "lpg")]
use grafeo_common::utils::error::Result;

use crate::transaction::CommitsHeld;

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
    /// Builds every section of the database. The caller holds commits off
    /// (`_commits`, see
    /// [`TransactionManager::hold_commits`](crate::transaction::TransactionManager)):
    /// no commit is in the middle of being written while the sections are
    /// built and serialized, so they hold every commit whole, and none that
    /// did not complete.
    pub fn sections(&self, _commits: &CommitsHeld<'_>) -> Vec<Box<dyn Section>> {
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

/// Loads the sections of `image` into the stores and the catalog, and puts
/// the indexes back: property indexes from the data, the default graph's
/// vector and text indexes from their own sections.
///
/// The image is the active checkpoint of a `.grafeo` file (served by
/// `GrafeoFileManager::read_image`), the sections of a 0.5.x file, or a copy
/// in memory. `read_image` holds the file manager's file lock while it runs
/// this, so nothing here, nor anything it calls, may break the rule in
/// `read_image`'s doc: no file manager method that takes the file lock, and
/// no lock a checkpoint holds while it waits for it (`checkpoint_guard`, the
/// commit hold).
///
/// Each section type is asked for once, so an image that serves each
/// section once (`ServedOnce`, which frees a section once it is loaded)
/// serves them all.
///
/// The indexes exist before WAL recovery, which keeps them current. What is
/// left (see [`LoadedSections`]) is finished by `GrafeoDB::finish_load`.
///
/// # Errors
///
/// Returns an error if a section cannot be read or decoded. A vector, text
/// or ring index section that can be read but not decoded is no error: its
/// indexes are built from the data, which they only mirror.
#[cfg(feature = "lpg")]
pub(super) fn load_sections(
    image: &dyn ImageSource,
    store: &Arc<LpgStore>,
    catalog: &Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
) -> Result<LoadedSections> {
    let mut loaded = LoadedSections::default();

    // The catalog first: the schema is needed before the data.
    let mut indexes = Vec::new();
    if let Some(source) = image.section_source(SectionType::Catalog) {
        let transaction_manager = Arc::new(crate::transaction::TransactionManager::new());
        let mut section = CatalogSection::new(Arc::clone(catalog), Arc::clone(store), move || {
            transaction_manager.current_epoch().as_u64()
        });
        section.read_from(&*source)?;
        indexes = section.take_loaded_indexes();
    }

    // With a compacted base, this is the overlay.
    if let Some(source) = image.section_source(SectionType::LpgStore) {
        grafeo_core::graph::lpg::LpgStoreSection::new(Arc::clone(store)).read_from(&*source)?;
    }

    #[cfg(feature = "triple-store")]
    if let Some(source) = image.section_source(SectionType::RdfStore) {
        grafeo_core::graph::rdf::RdfStoreSection::new(Arc::clone(rdf_store)).read_from(&*source)?;
    }

    // The ring only mirrors the triples, loaded just before: a ring section
    // that does not decode is built from them instead.
    #[cfg(feature = "ring-index")]
    if let Some(source) = image.section_source(SectionType::RdfRing)
        && let Some(err) = read_mirror(
            &mut grafeo_core::index::ring::RdfRingSection::new(Arc::clone(rdf_store)),
            &*source,
        )?
    {
        grafeo_common::grafeo_warn!("rebuilding the RDF ring from the triples: {err}");
        rdf_store.rebuild_ring();
    }

    #[cfg(feature = "compact-store")]
    {
        use grafeo_core::graph::compact::deletions_section::OverlayDeletionsSection;
        use grafeo_core::graph::compact::section::CompactStoreSection;

        if let Some(source) = image.section_source(SectionType::CompactStore) {
            let mut section = CompactStoreSection::empty();
            section.read_from(&*source)?;
            loaded.compact_base = section.store();
        }
        if let Some(source) = image.section_source(SectionType::OverlayDeletions) {
            let mut section = OverlayDeletionsSection::empty();
            section.read_from(&*source)?;
            loaded.overlay_deletions = Some(section.take());
        }
    }

    loaded.unbuilt = restore_indexes(image, store, indexes)?;
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
    image: &dyn ImageSource,
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

    let missing = restore_from_sections(image, store, from_sections)?;
    if !missing.is_empty() {
        unbuilt.push(missing);
    }
    Ok(unbuilt)
}

/// Restores the default graph's vector and text indexes from their
/// sections, and returns those the sections did not hold. A section that
/// cannot be decoded is skipped with a warning: its indexes are then built
/// from the data, which the index only mirrors. A section whose chunks
/// cannot be read (an I/O, checksum or decryption failure) fails the load:
/// the image is damaged.
#[cfg(feature = "lpg")]
#[cfg_attr(
    not(any(feature = "vector-index", feature = "text-index")),
    expect(
        unused_variables,
        reason = "without vector and text indexes there is nothing to restore"
    )
)]
fn restore_from_sections(
    image: &dyn ImageSource,
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
        if let Some(source) = image.section_source(SectionType::VectorStore)
            && let Some(err) = read_mirror(
                &mut VectorStoreSection::new(store.vector_index_entries()),
                &*source,
            )?
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
        if let Some(source) = image.section_source(SectionType::TextIndex)
            && let Some(err) = read_mirror(
                &mut TextIndexSection::new(store.text_index_entries()),
                &*source,
            )?
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

/// Reads an index section, which only mirrors the data, from `source`.
///
/// Returns the error of a section that cannot be decoded, whose indexes the
/// caller then builds from the data instead. An error fetching a chunk (I/O,
/// checksum, decryption) means the image is damaged: it is returned as an
/// error, as for any other section.
///
/// # Errors
///
/// Returns the error of a chunk that cannot be fetched.
#[cfg(all(
    feature = "lpg",
    any(
        feature = "vector-index",
        feature = "text-index",
        feature = "ring-index"
    )
))]
fn read_mirror(
    section: &mut dyn Section,
    source: &dyn SectionSource,
) -> Result<Option<grafeo_common::utils::error::Error>> {
    let watched = FetchWatch {
        source,
        failed: std::cell::Cell::new(false),
    };
    match section.read_from(&watched) {
        Ok(()) => Ok(None),
        Err(error) if watched.failed.get() => Err(error),
        Err(error) => Ok(Some(error)),
    }
}

/// Serves the chunks of a section and remembers whether fetching one failed,
/// so an unreadable section can be told apart from one that does not decode.
#[cfg(all(
    feature = "lpg",
    any(
        feature = "vector-index",
        feature = "text-index",
        feature = "ring-index"
    )
))]
struct FetchWatch<'a> {
    source: &'a dyn SectionSource,
    failed: std::cell::Cell<bool>,
}

#[cfg(all(
    feature = "lpg",
    any(
        feature = "vector-index",
        feature = "text-index",
        feature = "ring-index"
    )
))]
impl SectionSource for FetchWatch<'_> {
    fn chunks(&self) -> &[ChunkMeta] {
        self.source.chunks()
    }

    fn fetch(&self, index: usize) -> Result<bytes::Bytes> {
        let fetched = self.source.fetch(index);
        if fetched.is_err() {
            self.failed.set(true);
        }
        fetched
    }

    fn section_version(&self) -> u8 {
        self.source.section_version()
    }
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
    use grafeo_common::storage::{ChunkKind, MemoryImage, SectionSink, ServedOnce, legacy_bytes};

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

        let commits = db.transaction_manager.hold_commits().unwrap();
        let sections = db.checkpoint_sources().sections(&commits);
        let refs: Vec<&dyn Section> = sections.iter().map(AsRef::as_ref).collect();
        let image = MemoryImage::from_sections(&refs).unwrap();
        drop(commits);
        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load_sections(
            &image,
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

    /// The sections of a 0.5.x file reach their `deserialize` through raw chunks.
    #[test]
    fn a_raw_image_loads_through_the_0_5_readers() {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:City {name: 'Amsterdam'})-[:ROUTE {km: 653}]->(:City {name: 'Berlin'})",
        )
        .unwrap();
        db.execute("CREATE CONSTRAINT city_name FOR (c:City) ON (c.name) UNIQUE")
            .unwrap();
        let commits = db.transaction_manager.hold_commits().unwrap();
        let raw: Vec<(SectionType, Vec<u8>)> = db
            .checkpoint_sources()
            .sections(&commits)
            .iter()
            .map(|section| (section.section_type(), section.serialize().unwrap()))
            .collect();
        drop(commits);
        let store = Arc::new(LpgStore::new().unwrap());
        let catalog = Arc::new(crate::catalog::Catalog::new());
        load_sections(
            &MemoryImage::from_raw(raw).unwrap(),
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        )
        .unwrap();
        assert_eq!((store.node_count(), store.edge_count()), (2, 1));
        assert_eq!(catalog.constraints().len(), 1);
    }

    /// A database with a vector and a text index on the default graph.
    fn indexed_database() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Doc {emb: vector([3.0, 19.0]), body: 'Amsterdam'}), \
             (:Doc {emb: vector([88.0, 3.0]), body: 'Berlin'})",
        )
        .unwrap();
        db.create_vector_index("Doc", "emb", None, None, None, None, None)
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        db
    }

    /// A database whose checkpoint holds every kind of section this build
    /// has: the catalog, the LPG store, a vector and a text index and, with
    /// the features for them, RDF triples with their ring and a compacted
    /// base with a deletion.
    fn every_kind_of_section() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'}), \
             (:City {name: 'Paris'})",
        )
        .unwrap();
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        {
            db.execute_sparql(
                "INSERT DATA { <http://example.org/mia> <http://example.org/lives_in> \
                 <http://example.org/prague> }",
            )
            .unwrap();
            #[cfg(feature = "ring-index")]
            db.rdf_store().rebuild_ring();
        }
        #[cfg(feature = "compact-store")]
        let db = {
            let mut db = db;
            db.compact().unwrap();
            db.execute("MATCH (c:City {name: 'Berlin'}) DELETE c")
                .unwrap();
            db
        };
        db.execute("INSERT (:Doc {emb: vector([3.0, 19.0]), body: 'Prague'})")
            .unwrap();
        db.create_vector_index("Doc", "emb", None, None, None, None, None)
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        db
    }

    /// The section types [`every_kind_of_section`] gives a checkpoint in this
    /// build, ordered by their byte.
    fn every_section_type() -> Vec<SectionType> {
        let mut types = vec![
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::VectorStore,
            SectionType::TextIndex,
        ];
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        types.push(SectionType::RdfStore);
        #[cfg(all(feature = "sparql", feature = "ring-index"))]
        types.push(SectionType::RdfRing);
        #[cfg(feature = "compact-store")]
        types.extend([SectionType::CompactStore, SectionType::OverlayDeletions]);
        types.sort_by_key(|section_type| section_type.to_u8());
        types
    }

    /// `types`, ordered by their byte.
    fn by_byte(mut types: Vec<SectionType>) -> Vec<SectionType> {
        types.sort_by_key(|section_type| section_type.to_u8());
        types
    }

    /// The image of a checkpoint of `db`. A section for which `replace`
    /// returns bytes is written as one raw chunk of them instead of its own
    /// chunks.
    fn image_with(db: &GrafeoDB, replace: impl Fn(&dyn Section) -> Option<Vec<u8>>) -> MemoryImage {
        let commits = db.transaction_manager.hold_commits().unwrap();
        let mut image = MemoryImage::new();
        for section in db.checkpoint_sources().sections(&commits) {
            image
                .begin_section(section.section_type(), section.version())
                .unwrap();
            match replace(&*section) {
                Some(bytes) => image.write_chunk(ChunkMeta::raw(), &bytes).unwrap(),
                None => section.write_to(&mut image).unwrap(),
            }
        }
        image
    }

    /// The image of a checkpoint of `db`.
    fn image_of(db: &GrafeoDB) -> MemoryImage {
        image_with(db, |_| None)
    }

    fn load(image: &dyn ImageSource, store: &Arc<LpgStore>) -> Result<LoadedSections> {
        load_sections(
            image,
            store,
            &Arc::new(crate::catalog::Catalog::new()),
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        )
    }

    /// The RDF ring comes back from its own section: nothing else builds it
    /// during a load.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn the_ring_comes_back_from_its_section() {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
             <http://example.org/gus> . <http://example.org/gus> \
             <http://example.org/lives_in> <http://example.org/berlin> }",
        )
        .unwrap();
        db.rdf_store().rebuild_ring();
        let image = image_of(&db);
        assert!(
            image.section_types().contains(&SectionType::RdfRing),
            "the image has a ring section"
        );

        let rdf_store = Arc::new(grafeo_core::graph::rdf::RdfStore::new());
        load_sections(
            &image,
            &Arc::new(LpgStore::new().unwrap()),
            &Arc::new(crate::catalog::Catalog::new()),
            &rdf_store,
        )
        .unwrap();
        assert_eq!(rdf_store.len(), 2, "the triples are loaded");
        assert_eq!(
            rdf_store.ring().map(|ring| ring.len()),
            Some(2),
            "the ring is loaded from its section"
        );
    }

    /// A database with two RDF triples and their ring.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    fn ringed_database() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
             <http://example.org/gus> . <http://example.org/gus> \
             <http://example.org/lives_in> <http://example.org/berlin> }",
        )
        .unwrap();
        db.rdf_store().rebuild_ring();
        db
    }

    /// The image of a checkpoint of `db` in which `edit` may change the
    /// bytes of each chunk of `section_type` as the section wrote it.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    fn image_editing(
        db: &GrafeoDB,
        section_type: SectionType,
        edit: impl Fn(&ChunkMeta, &mut Vec<u8>),
    ) -> MemoryImage {
        let commits = db.transaction_manager.hold_commits().unwrap();
        let mut image = MemoryImage::new();
        for section in db.checkpoint_sources().sections(&commits) {
            image
                .begin_section(section.section_type(), section.version())
                .unwrap();
            if section.section_type() != section_type {
                section.write_to(&mut image).unwrap();
                continue;
            }
            let own = MemoryImage::from_sections(&[&*section]).unwrap();
            let source = own.section_source(section_type).unwrap();
            for (index, meta) in source.chunks().iter().enumerate() {
                let mut bytes = source.fetch(index).unwrap().to_vec();
                edit(meta, &mut bytes);
                image.write_chunk(*meta, &bytes).unwrap();
            }
        }
        image
    }

    /// A ring section that can be read but not decoded is no error: the
    /// ring, which only mirrors the triples, is built from them (with a
    /// warning), and answers.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn a_ring_section_that_does_not_decode_is_rebuilt() {
        use grafeo_core::graph::rdf::{RdfStore, Term, TriplePattern};
        use grafeo_core::index::ring::RdfRingSection;

        let db = ringed_database();
        // The subjects (stream 1) lose their magic bytes.
        let image = image_editing(&db, SectionType::RdfRing, |meta, bytes| {
            if meta.kind == grafeo_common::storage::ChunkKind::Stream
                && meta.column_id == 1
                && meta.row_start == 0
            {
                bytes[0] ^= 0xFF;
            }
        });
        let source = image.section_source(SectionType::RdfRing).unwrap();
        let error = RdfRingSection::new(Arc::new(RdfStore::new()))
            .read_from(&*source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("bad magic"),
            "the crafted section does not decode: {error}"
        );
        drop(source);

        let rdf_store = Arc::new(RdfStore::new());
        load_sections(
            &image,
            &Arc::new(LpgStore::new().unwrap()),
            &Arc::new(crate::catalog::Catalog::new()),
            &rdf_store,
        )
        .expect("the load succeeds");
        assert_eq!(rdf_store.len(), 2, "the triples are loaded");
        let ring = rdf_store
            .ring()
            .expect("the ring is built from the triples");
        assert_eq!(ring.len(), 2);
        let knows = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://example.org/knows")),
            object: None,
        };
        assert_eq!(ring.count(&knows), 1, "the ring answers");
    }

    /// A ring section as 0.5.x wrote it (one raw chunk of the version 2
    /// envelope) that does not decode no longer fails the open: the ring is
    /// built from the triples, as for a section of this release.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn a_0_5_ring_section_that_does_not_decode_is_rebuilt() {
        use grafeo_core::graph::rdf::{RdfStore, Term, TriplePattern};

        let db = ringed_database();
        let commits = db.transaction_manager.hold_commits().unwrap();
        let raw: Vec<(SectionType, Vec<u8>)> = db
            .checkpoint_sources()
            .sections(&commits)
            .iter()
            .map(|section| {
                let mut bytes = section.serialize().unwrap();
                if section.section_type() == SectionType::RdfRing {
                    // A byte of the body: the envelope's checksum fails.
                    let middle = bytes.len() / 2;
                    bytes[middle] ^= 0xFF;
                }
                (section.section_type(), bytes)
            })
            .collect();
        drop(commits);
        assert!(
            raw.iter()
                .any(|(section_type, _)| *section_type == SectionType::RdfRing),
            "the 0.5.x image has a ring section"
        );

        let rdf_store = Arc::new(RdfStore::new());
        load_sections(
            &MemoryImage::from_raw(raw).unwrap(),
            &Arc::new(LpgStore::new().unwrap()),
            &Arc::new(crate::catalog::Catalog::new()),
            &rdf_store,
        )
        .expect("the load succeeds");
        let ring = rdf_store
            .ring()
            .expect("the ring is built from the triples");
        let lives_in = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://example.org/lives_in")),
            object: None,
        };
        assert_eq!((ring.len(), ring.count(&lives_in)), (2, 1));
    }

    /// Terms whose N-Triples string does not parse back to them (two blank
    /// nodes that differ in a trailing space, a language tag with one) do
    /// not come back from the ring section with shifted ids: the section is
    /// refused and the ring built from the loaded triples, whose answers it
    /// then gives.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn terms_that_do_not_survive_their_string_get_a_rebuilt_ring() {
        use grafeo_core::graph::rdf::{RdfStore, Term, Triple, TriplePattern};

        let db = ringed_database();
        for triple in [
            Triple::new(
                Term::blank("b"),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/vincent"),
            ),
            Triple::new(
                Term::blank("b "),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/jules"),
            ),
            Triple::new(
                Term::iri("http://example.org/jules"),
                Term::iri("http://example.org/motto"),
                Term::lang_literal("gezellig", "nl "),
            ),
        ] {
            db.rdf_store().insert(triple);
        }
        db.rdf_store().rebuild_ring();
        let image = image_of(&db);

        let rdf_store = Arc::new(RdfStore::new());
        load_sections(
            &image,
            &Arc::new(LpgStore::new().unwrap()),
            &Arc::new(crate::catalog::Catalog::new()),
            &rdf_store,
        )
        .expect("the load succeeds");
        let ring = rdf_store.ring().expect("a ring");
        let loaded = rdf_store.triples();
        assert_eq!(ring.len(), loaded.len(), "one ring entry per loaded triple");
        for triple in &loaded {
            let pattern = TriplePattern {
                subject: Some(triple.subject().clone()),
                predicate: Some(triple.predicate().clone()),
                object: Some(triple.object().clone()),
            };
            assert_eq!(ring.count(&pattern), 1, "the ring holds {triple:?}");
            let by_subject = TriplePattern {
                subject: Some(triple.subject().clone()),
                predicate: None,
                object: None,
            };
            assert_eq!(
                ring.count(&by_subject),
                loaded
                    .iter()
                    .filter(|other| other.subject() == triple.subject())
                    .count(),
                "triples of {:?}",
                triple.subject()
            );
        }
    }

    /// A section that writes the chunks it was given, as a file held them.
    #[cfg(all(feature = "ring-index", feature = "sparql", feature = "grafeo-file"))]
    struct Replayed {
        section_type: SectionType,
        version: u8,
        chunks: Vec<(ChunkMeta, bytes::Bytes)>,
    }

    #[cfg(all(feature = "ring-index", feature = "sparql", feature = "grafeo-file"))]
    impl Section for Replayed {
        fn section_type(&self) -> SectionType {
            self.section_type
        }
        fn version(&self) -> u8 {
            self.version
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Err(grafeo_common::utils::error::Error::Internal(
                "a replayed section only writes its chunks".to_string(),
            ))
        }
        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Err(grafeo_common::utils::error::Error::Internal(
                "a replayed section is not read".to_string(),
            ))
        }
        fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
            for (meta, bytes) in &self.chunks {
                sink.write_chunk(*meta, bytes)?;
            }
            Ok(())
        }
        fn is_dirty(&self) -> bool {
            true
        }
        fn mark_clean(&self) {}
        fn memory_usage(&self) -> usize {
            0
        }
    }

    /// The open of a file whose ring section does not decode succeeds, and
    /// the ring is built from the triples: the file holds a checkpoint whose
    /// ring section has its subjects (stream 1) without their magic bytes,
    /// with valid chunk checksums.
    #[cfg(all(feature = "ring-index", feature = "sparql", feature = "grafeo-file"))]
    #[test]
    fn a_file_whose_ring_section_does_not_decode_opens() {
        use grafeo_core::graph::rdf::{Term, TriplePattern};
        use grafeo_storage::file::{CheckpointHeader, GrafeoFileManager};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ring.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        db.execute_sparql(
            "INSERT DATA { <http://example.org/mia> <http://example.org/knows> \
             <http://example.org/vincent> . <http://example.org/vincent> \
             <http://example.org/lives_in> <http://example.org/amsterdam> }",
        )
        .unwrap();
        db.rdf_store().rebuild_ring();
        db.close().unwrap();
        drop(db);

        let fm = GrafeoFileManager::open(&path, None).unwrap();
        let sections: Vec<Replayed> = fm
            .read_image(|image| {
                let mut sections = Vec::new();
                for section_type in (0..=u8::MAX).filter_map(SectionType::from_u8) {
                    let Some(source) = image.section_source(section_type) else {
                        continue;
                    };
                    let mut chunks = Vec::new();
                    for (index, meta) in source.chunks().iter().enumerate() {
                        let mut bytes = source.fetch(index)?.to_vec();
                        if section_type == SectionType::RdfRing
                            && meta.kind == grafeo_common::storage::ChunkKind::Stream
                            && meta.column_id == 1
                            && meta.row_start == 0
                        {
                            bytes[0] ^= 0xFF;
                        }
                        chunks.push((*meta, bytes::Bytes::from(bytes)));
                    }
                    sections.push(Replayed {
                        section_type,
                        version: source.section_version(),
                        chunks,
                    });
                }
                Ok(sections)
            })
            .unwrap();
        assert!(
            sections
                .iter()
                .any(|section| section.section_type == SectionType::RdfRing
                    && section.chunks.len() > 1),
            "the file holds a streamed ring section"
        );
        let active = fm.active_header();
        let refs: Vec<&dyn Section> = sections
            .iter()
            .map(|section| section as &dyn Section)
            .collect();
        fm.write_checkpoint(
            &refs,
            &CheckpointHeader {
                checkpoint_lsn: active.checkpoint_lsn,
                epoch: active.epoch,
                last_transaction_id: active.last_transaction_id,
                node_count: active.node_count,
                edge_count: active.edge_count,
            },
        )
        .unwrap();
        fm.close().unwrap();
        drop(fm);

        let db = GrafeoDB::open(&path).expect("the open succeeds");
        assert_eq!(db.rdf_store().len(), 2, "the triples are loaded");
        let ring = db
            .rdf_store()
            .ring()
            .expect("the ring is built from the triples");
        let lives_in = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://example.org/lives_in")),
            object: None,
        };
        assert_eq!(
            (ring.len(), ring.count(&lives_in)),
            (2, 1),
            "the ring answers"
        );
    }

    /// A ring section whose chunks cannot be read fails the load, as an
    /// unreadable vector or text index section does: the image is damaged.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn a_ring_section_that_cannot_be_read_fails_the_load() {
        let db = ringed_database();
        let image = image_of(&db);
        let store = Arc::new(LpgStore::new().unwrap());
        assert!(load(&image, &store).is_ok(), "the image itself loads");
        let error = load(
            &Unreadable {
                image: &image,
                unreadable: SectionType::RdfRing,
            },
            &store,
        )
        .map(|_| ())
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("RdfRing") && error.contains("fails its checksum"),
            "the fetch error is returned: {error}"
        );
    }

    /// `to_memory` loads the copy from every section but the LPG store's: a
    /// compacted base keeps the deletions of its overlay.
    #[cfg(feature = "compact-store")]
    #[test]
    fn to_memory_keeps_the_deletions_of_a_compacted_base() {
        let mut db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'}), \
             (:City {name: 'Paris'})",
        )
        .unwrap();
        db.compact().unwrap();
        db.execute("MATCH (c:City {name: 'Berlin'}) DELETE c")
            .unwrap();

        let copy = db.to_memory().unwrap();
        let names = copy
            .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
            .unwrap();
        assert_eq!(
            names.rows(),
            [
                [grafeo_common::types::Value::from("Amsterdam")],
                [grafeo_common::types::Value::from("Paris")]
            ],
            "Berlin, deleted from the compacted base, stays deleted in the copy"
        );
    }

    /// `to_memory` copies the RDF ring from its section.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn to_memory_keeps_the_ring() {
        let db = GrafeoDB::new_in_memory();
        db.execute_sparql(
            "INSERT DATA { <http://example.org/mia> <http://example.org/lives_in> \
             <http://example.org/prague> }",
        )
        .unwrap();
        db.rdf_store().rebuild_ring();

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.rdf_store().len(), 1);
        assert_eq!(
            copy.rdf_store().ring().map(|ring| ring.len()),
            Some(1),
            "the copy has the ring"
        );
    }

    /// A vector section that decodes its first index and fails on the
    /// second, and a text section that does not decode, are no error: the
    /// load keeps none of their indexes, not even the one restored before
    /// the failure, and leaves them all to build from the data, which they
    /// mirror.
    #[test]
    fn an_index_section_that_does_not_decode_is_built_from_the_data() {
        use grafeo_core::index::vector::{
            DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind, VectorStoreSection,
        };

        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Doc {emb: vector([3.0, 19.0]), body: 'Amsterdam'}), \
             (:Note {emb: vector([88.0, 3.0])})",
        )
        .unwrap();
        for label in ["Doc", "Note"] {
            db.create_vector_index(label, "emb", None, None, None, None, None)
                .unwrap();
        }
        db.create_text_index("Doc", "body").unwrap();
        // The vector section without its last byte, which belongs to the
        // index it holds last; text bytes that are no text section.
        let image = image_with(&db, |section| match section.section_type() {
            SectionType::VectorStore => {
                let mut bytes = section.serialize().unwrap();
                bytes.pop();
                Some(bytes)
            }
            SectionType::TextIndex => Some(b"Vincent".to_vec()),
            _ => None,
        });

        // The cut section restores its first index before it fails.
        let source = image.section_source(SectionType::VectorStore).unwrap();
        let cut = legacy_bytes(&*source).unwrap().unwrap();
        let shells: Vec<(String, Arc<VectorIndexKind>)> = ["Doc:emb", "Note:emb"]
            .into_iter()
            .map(|key| {
                let config = HnswConfig::new(2, DistanceMetric::Cosine);
                (
                    key.to_string(),
                    Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config))),
                )
            })
            .collect();
        assert!(
            VectorStoreSection::new(shells.clone())
                .deserialize(&cut)
                .is_err()
        );
        assert_eq!(
            shells.iter().filter(|(_, index)| !index.is_empty()).count(),
            1,
            "the cut section restores one index, then fails"
        );

        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load(&image, &store).unwrap();
        assert_eq!(store.node_count(), 2, "the data is loaded");
        for label in ["Doc", "Note"] {
            assert!(
                store.get_vector_index(label, "emb").is_none(),
                "the vector index {label}:emb of a section that failed is not kept"
            );
        }
        assert!(
            store.get_text_index("Doc", "body").is_none(),
            "the text index the section could not restore is left to build from the data"
        );
        assert_eq!(loaded.unbuilt.len(), 1, "one graph: the default one");
        let graph = &loaded.unbuilt[0];
        let mut labels: Vec<&str> = graph.vector.iter().map(|def| def.label.as_str()).collect();
        labels.sort_unstable();
        assert_eq!(
            (graph.graph.as_deref(), labels, &graph.text[..]),
            (
                None,
                vec!["Doc", "Note"],
                &[("Doc".to_string(), "body".to_string())][..]
            ),
            "every index is left to build from the data"
        );
    }

    /// A version 3 vector section whose stream ends early does not decode,
    /// which is no error: the load leaves the index to build from the data,
    /// as for any index section that does not decode. The intact section
    /// restores it.
    #[test]
    fn a_vector_section_whose_stream_ends_early_is_built_from_the_data() {
        /// Keeps the chunks a section writes.
        struct Kept(Vec<(ChunkMeta, Vec<u8>)>);

        impl SectionSink for Kept {
            fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
                self.0.push((meta, bytes.to_vec()));
                Ok(())
            }
        }

        let db = indexed_database();
        let commits = db.transaction_manager.hold_commits().unwrap();
        let mut image = MemoryImage::new();
        for section in db.checkpoint_sources().sections(&commits) {
            image
                .begin_section(section.section_type(), section.version())
                .unwrap();
            if section.section_type() == SectionType::VectorStore {
                assert_eq!(section.version(), 3, "the vector section streams");
                let mut kept = Kept(Vec::new());
                section.write_to(&mut kept).unwrap();
                let (last, _) = kept.0.pop().unwrap();
                assert_eq!(last.kind, ChunkKind::Stream, "the last piece is dropped");
                for (meta, bytes) in kept.0 {
                    image.write_chunk(meta, &bytes).unwrap();
                }
            } else {
                section.write_to(&mut image).unwrap();
            }
        }
        drop(commits);

        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load(&image, &store).unwrap();
        assert_eq!(store.node_count(), 2, "the data is loaded");
        assert!(
            store.get_vector_index("Doc", "emb").is_none(),
            "the index whose stream ends early is not kept"
        );
        let unbuilt: Vec<&str> = loaded
            .unbuilt
            .iter()
            .flat_map(|graph| graph.vector.iter().map(|def| def.label.as_str()))
            .collect();
        assert_eq!(unbuilt, ["Doc"], "the index is left to build from the data");

        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load(&image_of(&db), &store).unwrap();
        assert_eq!(
            store
                .get_vector_index("Doc", "emb")
                .map(|index| index.len()),
            Some(2),
            "the intact section restores the index"
        );
        assert!(loaded.unbuilt.iter().all(|graph| graph.vector.is_empty()));
    }

    /// A streamed text section whose second stream lost its last piece is no
    /// error: the first index, restored before the failure, is not kept
    /// either, and both are left to build from the data.
    #[test]
    fn a_text_section_that_fails_in_its_second_stream_is_built_from_the_data() {
        use grafeo_common::storage::ChunkKind;

        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Doc {body: 'Amsterdam canal bridge'}), \
             (:Note {text: 'Berlin museum Prague'})",
        )
        .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        db.create_text_index("Note", "text").unwrap();

        let commits = db.transaction_manager.hold_commits().unwrap();
        let mut image = MemoryImage::new();
        for section in db.checkpoint_sources().sections(&commits) {
            image
                .begin_section(section.section_type(), section.version())
                .unwrap();
            if section.section_type() != SectionType::TextIndex {
                section.write_to(&mut image).unwrap();
                continue;
            }
            let mut whole = MemoryImage::new();
            whole
                .begin_section(SectionType::TextIndex, section.version())
                .unwrap();
            section.write_to(&mut whole).unwrap();
            let source = whole.section_source(SectionType::TextIndex).unwrap();
            // Stream 1 holds "Note:text", the second key in order.
            let last = source
                .chunks()
                .iter()
                .rposition(|meta| meta.kind == ChunkKind::Stream && meta.column_id == 1)
                .expect("the second index has a stream");
            assert!(
                source
                    .chunks()
                    .iter()
                    .any(|meta| meta.kind == ChunkKind::Stream && meta.column_id == 0),
                "the first index has a stream"
            );
            for (index, meta) in source.chunks().iter().enumerate() {
                if index != last {
                    image
                        .write_chunk(*meta, &source.fetch(index).unwrap())
                        .unwrap();
                }
            }
        }
        drop(commits);

        let store = Arc::new(LpgStore::new().unwrap());
        let loaded = load(&image, &store).unwrap();
        assert_eq!(store.node_count(), 2, "the data is loaded");
        for (label, property) in [("Doc", "body"), ("Note", "text")] {
            assert!(
                store.get_text_index(label, property).is_none(),
                "the text index {label}:{property} of a section that failed is not kept"
            );
        }
        assert_eq!(loaded.unbuilt.len(), 1, "one graph: the default one");
        let mut text = loaded.unbuilt[0].text.clone();
        text.sort_unstable();
        assert_eq!(
            text,
            [
                ("Doc".to_string(), "body".to_string()),
                ("Note".to_string(), "text".to_string())
            ],
            "both text indexes are left to build from the data"
        );
    }

    /// A section read through `read_mirror` sees the version it was written
    /// with.
    #[test]
    fn read_mirror_passes_on_the_section_version() {
        /// Notes the version its source reports.
        struct Versioned(Option<u8>);

        impl Section for Versioned {
            fn section_type(&self) -> SectionType {
                SectionType::TextIndex
            }
            fn serialize(&self) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }
            fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
                Ok(())
            }
            fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
                self.0 = Some(source.section_version());
                Ok(())
            }
            fn is_dirty(&self) -> bool {
                false
            }
            fn mark_clean(&self) {}
            fn memory_usage(&self) -> usize {
                0
            }
        }

        let mut image = MemoryImage::new();
        image.begin_section(SectionType::TextIndex, 3).unwrap();
        image.write_chunk(ChunkMeta::raw(), b"Gus").unwrap();
        let source = image.section_source(SectionType::TextIndex).unwrap();
        let mut section = Versioned(None);
        assert!(read_mirror(&mut section, &*source).unwrap().is_none());
        assert_eq!(
            section.0,
            Some(3),
            "the section reads the version it was written with"
        );
    }

    /// An image that notes every section type asked for.
    struct Counting<'a> {
        image: &'a dyn ImageSource,
        asked: std::cell::RefCell<Vec<SectionType>>,
    }

    impl ImageSource for Counting<'_> {
        fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
            self.asked.borrow_mut().push(section_type);
            self.image.section_source(section_type)
        }
    }

    /// `load_sections` asks for each section type once and reads every
    /// section of the image: an image that serves each section once serves
    /// them all, and holds none of them once the load returns.
    #[test]
    fn load_sections_asks_for_each_section_once() {
        let db = every_kind_of_section();
        let image = ServedOnce::new(image_of(&db));
        let held = by_byte(image.section_types());
        assert_eq!(
            held,
            every_section_type(),
            "the image holds every kind of section"
        );

        let counting = Counting {
            image: &image,
            asked: std::cell::RefCell::default(),
        };
        let store = Arc::new(LpgStore::new().unwrap());
        load(&counting, &store).unwrap();
        let asked = counting.asked.into_inner();
        for section_type in &held {
            assert_eq!(
                asked.iter().filter(|asked| *asked == section_type).count(),
                1,
                "{section_type:?} is asked for once: {asked:?}"
            );
        }
        assert!(
            image.section_types().is_empty(),
            "every section was served, and freed once loaded"
        );
        assert_eq!(
            store
                .get_vector_index("Doc", "emb")
                .map(|index| index.len()),
            Some(1),
            "the vector index comes from its section"
        );
    }

    /// The image `to_memory` loads holds every section of a checkpoint but
    /// the LPG store's, each with its version and chunks; the nodes and edges
    /// are copied into the target instead.
    #[test]
    fn the_copy_image_holds_every_section_but_the_lpg_store() {
        let db = every_kind_of_section();
        let target = GrafeoDB::new_in_memory();
        let copied = db.copy_into(&target).unwrap();

        let commits = db.transaction_manager.hold_commits().unwrap();
        let sections = db.checkpoint_sources().sections(&commits);
        let refs: Vec<&dyn Section> = sections
            .iter()
            .map(AsRef::as_ref)
            .filter(|section| section.section_type() != SectionType::LpgStore)
            .collect();
        let expected = MemoryImage::from_sections(&refs).unwrap();
        drop(commits);

        let mut kinds = every_section_type();
        kinds.retain(|section_type| *section_type != SectionType::LpgStore);
        assert_eq!(
            by_byte(copied.section_types()),
            kinds,
            "every section but the LPG store's"
        );
        assert_eq!(copied.section_types(), expected.section_types());
        for section_type in expected.section_types() {
            let copy = copied.section_source(section_type).unwrap();
            let checkpoint = expected.section_source(section_type).unwrap();
            assert_eq!(
                copy.section_version(),
                checkpoint.section_version(),
                "{section_type:?} keeps its version"
            );
            assert_eq!(copy.chunks(), checkpoint.chunks(), "{section_type:?}");
            for index in 0..checkpoint.chunks().len() {
                assert_eq!(
                    copy.fetch(index).unwrap(),
                    checkpoint.fetch(index).unwrap(),
                    "{section_type:?}, chunk {index}"
                );
            }
        }
        assert_eq!(
            target.lpg_store().node_count(),
            db.lpg_store().node_count(),
            "the nodes are copied store to store"
        );
        assert!(target.lpg_store().node_count() > 0);
    }

    /// An image whose `unreadable` section fails every fetch, as a chunk that
    /// fails its checksum does.
    struct Unreadable<'a> {
        image: &'a MemoryImage,
        unreadable: SectionType,
    }

    impl ImageSource for Unreadable<'_> {
        fn section_source(&self, section_type: SectionType) -> Option<Box<dyn SectionSource + '_>> {
            let source = self.image.section_source(section_type)?;
            if section_type == self.unreadable {
                Some(Box::new(FailingFetch(section_type, source)))
            } else {
                Some(source)
            }
        }
    }

    /// A section whose chunks are listed but cannot be fetched.
    struct FailingFetch<'a>(SectionType, Box<dyn SectionSource + 'a>);

    impl SectionSource for FailingFetch<'_> {
        fn chunks(&self) -> &[ChunkMeta] {
            self.1.chunks()
        }

        fn fetch(&self, _index: usize) -> Result<bytes::Bytes> {
            Err(grafeo_common::utils::error::Error::Serialization(format!(
                "chunk of section {:?} at offset 16384 fails its checksum",
                self.0
            )))
        }

        fn section_version(&self) -> u8 {
            self.1.section_version()
        }
    }

    /// A vector or text index section whose chunk cannot be read fails the
    /// load, as any other section does: the image is damaged, and building
    /// the index from the data would hide that.
    #[test]
    fn an_index_section_that_cannot_be_read_fails_the_load() {
        let db = indexed_database();
        let image = image_of(&db);
        let store = Arc::new(LpgStore::new().unwrap());
        assert!(load(&image, &store).is_ok(), "the image itself loads");
        for unreadable in [SectionType::VectorStore, SectionType::TextIndex] {
            let store = Arc::new(LpgStore::new().unwrap());
            let error = load(
                &Unreadable {
                    image: &image,
                    unreadable,
                },
                &store,
            )
            .map(|_| ())
            .unwrap_err()
            .to_string();
            assert!(
                error.contains(&format!("{unreadable:?}")) && error.contains("fails its checksum"),
                "the fetch error of {unreadable:?} is returned: {error}"
            );
        }
    }
}
