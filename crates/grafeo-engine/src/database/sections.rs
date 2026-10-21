//! The sections of a database: what a checkpoint writes, and loading them
//! back. Opening a `.grafeo` file and `to_memory()` both load through
//! [`load_sections`], so a copy holds what a reopen would.

use std::sync::Arc;

use grafeo_common::storage::Section;
#[cfg(all(feature = "lpg", feature = "ring-index"))]
use grafeo_common::storage::{ChunkMeta, SectionSource};
#[cfg(feature = "lpg")]
use grafeo_common::storage::{ImageSource, SectionType};
#[cfg(feature = "lpg")]
use grafeo_common::utils::error::Result;

use crate::transaction::CommitsHeld;

#[cfg(feature = "lpg")]
use super::catalog_section::{CatalogSection, CatalogStore, GraphIndexes};
#[cfg(feature = "lpg")]
use grafeo_core::graph::rowgroup::RowGroupStore;

/// Everything a checkpoint writes: the database's complete state.
///
/// A checkpoint's container holds only the sections built here, so every
/// checkpoint path (`close()`, `wal_checkpoint()`, backups, the async
/// snapshot and the periodic timer) builds them from these sources, and
/// `to_memory()` copies a database through them.
#[derive(Clone)]
pub(super) struct CheckpointSources {
    /// The root store with its named graphs, `None` with an external store.
    #[cfg(feature = "lpg")]
    pub store: Option<Arc<RowGroupStore>>,
    #[cfg(feature = "lpg")]
    pub catalog: Arc<crate::catalog::Catalog>,
    pub transaction_manager: Arc<crate::transaction::TransactionManager>,
    #[cfg(feature = "triple-store")]
    pub rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
    /// The WAL a checkpoint starts a new segment of and truncates, `None`
    /// without one.
    #[cfg(feature = "wal")]
    pub wal: Option<Arc<grafeo_storage::wal::Wal>>,
    /// The WAL's records written up to the last checkpoint, which each
    /// checkpoint sets (see `GrafeoDB::wal_status`).
    #[cfg(feature = "wal")]
    pub wal_records_at_checkpoint: Arc<std::sync::atomic::AtomicU64>,
}

impl CheckpointSources {
    /// The epoch a checkpoint or copy of the database records, which a
    /// reopen continues from: the root store's, which follows every commit,
    /// or the transaction manager's when that is ahead (`restore_snapshot`
    /// resets the store's epoch, and a direct write that fails still takes
    /// its epoch), so the file never names an epoch below one the database
    /// published. Taken while commits are held, both are the epoch of the
    /// last complete commit.
    #[cfg(any(feature = "lpg", feature = "grafeo-file"))]
    pub fn epoch(&self) -> u64 {
        let published = self.transaction_manager.current_epoch();
        #[cfg(feature = "lpg")]
        if let Some(store) = &self.store {
            return store.current_epoch().max(published).as_u64();
        }
        published.as_u64()
    }

    /// The default graph's committed nodes and edges, for the header.
    #[cfg(all(feature = "lpg", feature = "grafeo-file"))]
    pub(super) fn counts(&self) -> Option<(usize, usize)> {
        use grafeo_core::graph::GraphStore;
        self.store
            .as_ref()
            .map(|store| (store.node_count(), store.edge_count()))
    }

    /// Builds every section of the database. The caller holds commits off
    /// (`_commits`, see
    /// [`TransactionManager::hold_commits`](crate::transaction::TransactionManager)):
    /// no commit is in the middle of being written while the sections are
    /// built and serialized, so they hold every commit whole, and none that
    /// did not complete.
    ///
    /// The sections hold the committed state, also while transactions are
    /// open (the hold also holds their writes and rollbacks): the catalog
    /// section, which reads the root's named graphs and index definitions,
    /// and the LPG section of version 4, written at the checkpoint's epoch
    /// as a reader at it sees each row of every graph, so it needs neither
    /// the open transactions' change sets nor a committed copy. A database
    /// that keeps history writes it from its history floor, the oldest epoch
    /// a read may ask for (H2c). No vector or text index section is written:
    /// the indexes take what transactions write as they write it, and the
    /// load builds them from the rows (see
    /// `row_group_sections::load_row_group_sections`).
    pub fn sections(&self, commits: &CommitsHeld<'_>) -> Vec<Box<dyn Section>> {
        // The hold is what makes the sections consistent; nothing reads it.
        let _ = commits;
        let mut sections: Vec<Box<dyn Section>> = Vec::new();

        #[cfg(feature = "lpg")]
        if let Some(root) = &self.store {
            let transaction_manager = Arc::clone(&self.transaction_manager);
            sections.push(Box::new(CatalogSection::new(
                Arc::clone(&self.catalog),
                Arc::clone(root),
                move || transaction_manager.current_epoch().as_u64(),
            )));
            let section =
                grafeo_core::graph::rowgroup::section::RowGroupStoreSection::new(Arc::clone(root))
                    .at(grafeo_common::types::EpochId::new(self.epoch()));
            let section = if self.transaction_manager.history_image().is_some() {
                section.with_history_from(self.transaction_manager.history_floor())
            } else {
                section
            };
            sections.push(Box::new(section));
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

        // The history's time index, only while the database keeps history:
        // taken while commits are held, it holds every commit up to the
        // checkpoint's epoch.
        if let Some(history) = self.transaction_manager.history_image() {
            sections.push(Box::new(super::time_index_section::TimeIndexSection::new(
                history,
            )));
        }

        sections
    }
}

/// What loading leaves for [`GrafeoDB::finish_load`](super::GrafeoDB) to do
/// once the database is built.
#[cfg(feature = "lpg")]
#[derive(Default)]
pub(crate) struct LoadedSections {
    /// Vector and text indexes the WAL replay put, built from the data once
    /// the database is built (the sections' own are built at the load).
    unbuilt: Vec<GraphIndexes>,
    /// The history the image holds (its `TIME_INDEX` section), `None` for a
    /// database that keeps none.
    history: Option<crate::transaction::HistoryImage>,
}

#[cfg(feature = "lpg")]
impl LoadedSections {
    /// The vector and text indexes left to build once the database is
    /// built: WAL replay adds those its records put, and removes those its
    /// records drop (or whose graph they drop).
    #[cfg(feature = "wal")]
    pub(super) fn unbuilt_mut(&mut self) -> &mut Vec<GraphIndexes> {
        &mut self.unbuilt
    }

    /// The history the image held, for the transaction manager.
    pub(super) fn take_history(&mut self) -> Option<crate::transaction::HistoryImage> {
        self.history.take()
    }

    /// What a load that built every index of its sections leaves: the
    /// history it read.
    pub(super) fn with_history(history: Option<crate::transaction::HistoryImage>) -> Self {
        Self {
            unbuilt: Vec::new(),
            history,
        }
    }
}

/// Data a database can hold that a build reads only with a feature. A build
/// without the feature would open the database without it, and its next
/// checkpoint would write the file without it, losing it for good (also
/// when it migrates a 0.5.x database). So every open of a database refuses
/// such data, before it changes anything on disk: [`load_sections`] for the
/// image of a file ([`FEATURE_DATA`] says where it holds what), the replay of
/// a WAL for its records, and the load of a 0.5.x snapshot for the snapshot.
/// An import or restore of a snapshot refuses the triples and the index
/// definitions it holds the same way, before it changes anything (a
/// snapshot holds a compacted database as plain nodes and edges).
#[cfg(feature = "lpg")]
pub(super) struct FeatureData {
    /// What the data is, as the error names it.
    what: &'static str,
    /// The feature a build needs to read it.
    feature: &'static str,
    /// Whether this build has the feature.
    in_build: bool,
    /// Where the image of a file holds it.
    held_in: HeldIn,
}

/// Where the image of a database file holds the data of a [`FeatureData`].
#[cfg(feature = "lpg")]
enum HeldIn {
    /// Sections of these types.
    Sections(&'static [SectionType]),
    /// Vector index definitions in the catalog. The default graph's indexes
    /// are also in the `VectorStore` section, which only mirrors the data;
    /// the definition is what a build without the feature drops.
    VectorIndexDefinitions,
    /// Text index definitions in the catalog (the default graph's indexes
    /// also in the `TextIndex` section).
    TextIndexDefinitions,
}

/// RDF triples, and the Ring index over them. A WAL holds them as records
/// and a 0.5.x container v1 file in its snapshot.
#[cfg(feature = "lpg")]
pub(super) const RDF_TRIPLES: FeatureData = FeatureData {
    what: "RDF triples",
    feature: "triple-store",
    in_build: cfg!(feature = "triple-store"),
    held_in: HeldIn::Sections(&[SectionType::RdfStore, SectionType::RdfRing]),
};

/// Vector indexes: a build without the feature cannot build them, and its
/// checkpoint writes the catalog without their definitions.
#[cfg(feature = "lpg")]
pub(super) const VECTOR_INDEXES: FeatureData = FeatureData {
    what: "vector indexes",
    feature: "vector-index",
    in_build: cfg!(feature = "vector-index"),
    held_in: HeldIn::VectorIndexDefinitions,
};

/// Text indexes, as [`VECTOR_INDEXES`].
#[cfg(feature = "lpg")]
pub(super) const TEXT_INDEXES: FeatureData = FeatureData {
    what: "text indexes",
    feature: "text-index",
    in_build: cfg!(feature = "text-index"),
    held_in: HeldIn::TextIndexDefinitions,
};

/// Every kind of data a build reads only with a feature.
#[cfg(feature = "lpg")]
const FEATURE_DATA: [&FeatureData; 3] = [&RDF_TRIPLES, &VECTOR_INDEXES, &TEXT_INDEXES];

#[cfg(feature = "lpg")]
impl FeatureData {
    /// Whether this build has the feature that reads this data.
    pub(super) const fn in_build(&self) -> bool {
        self.in_build
    }

    /// Where the image holds this data, or `None` when it holds none:
    /// `sections RdfStore, RdfRing` (only which sections the image holds is
    /// looked at), or `on :Document(embedding), ...` for the indexes of this
    /// kind that `indexes`, the image's catalog, defines.
    fn held(&self, image: &dyn ImageSource, indexes: &[GraphIndexes]) -> Option<String> {
        let (kind, found): (&str, Vec<String>) =
            match self.held_in {
                HeldIn::Sections(section_types) => (
                    "sections",
                    section_types
                        .iter()
                        .filter(|section_type| image.section_source(**section_type).is_some())
                        .map(|section_type| format!("{section_type:?}"))
                        .collect(),
                ),
                HeldIn::VectorIndexDefinitions => (
                    "on",
                    indexes
                        .iter()
                        .flat_map(|graph| {
                            graph.vector.iter().map(|def| {
                                index_on(graph.graph.as_deref(), &def.label, &def.property)
                            })
                        })
                        .collect(),
                ),
                HeldIn::TextIndexDefinitions => (
                    "on",
                    indexes
                        .iter()
                        .flat_map(|graph| {
                            graph.text.iter().map(|def| {
                                index_on(graph.graph.as_deref(), &def.label, &def.property)
                            })
                        })
                        .collect(),
                ),
            };
        (!found.is_empty()).then(|| format!("{kind} {}", found.join(", ")))
    }
}

/// An index on `property` of the nodes with `label`, as a refusal names it:
/// `:Document(embedding)`, with ` in graph trips` for one of a named graph
/// (`graph`).
#[cfg(feature = "lpg")]
fn index_on(graph: Option<&str>, label: &str, property: &str) -> String {
    match graph {
        None => format!(":{label}({property})"),
        Some(graph) => format!(":{label}({property}) in graph {graph}"),
    }
}

/// The error that refuses `path` (a database file, or its WAL) for the data
/// in `found` this build cannot read, each with where `path` holds it.
#[cfg(feature = "lpg")]
pub(super) fn refusal(
    path: &std::path::Path,
    found: &[(&FeatureData, String)],
) -> grafeo_common::utils::error::Error {
    refusal_of(&path.display().to_string(), "open", found)
}

/// The error that refuses `subject` (what it names: a database file or its
/// WAL by its path, or `the snapshot`) for the data in `found` this build
/// cannot read, each with where `subject` holds it, and tells to `action` it
/// (`open`, `import`, `restore`) with a build that has the features.
#[cfg(feature = "lpg")]
pub(super) fn refusal_of(
    subject: &str,
    action: &str,
    found: &[(&FeatureData, String)],
) -> grafeo_common::utils::error::Error {
    let parts: Vec<String> = found
        .iter()
        .map(|(data, held)| {
            format!(
                "{} ({held}), data that only a build with the `{}` feature can read",
                data.what, data.feature
            )
        })
        .collect();
    let (holds, build) = match parts.split_last() {
        Some((last, before)) if !before.is_empty() => (
            format!("{}; and {last}", before.join("; ")),
            "a build that has these features",
        ),
        _ => (parts.concat(), "such a build"),
    };
    grafeo_common::utils::error::Error::InvalidValue(format!(
        "{subject} holds {holds}: {action} it with {build}"
    ))
}

/// Refuses the image of the database file at `path` when it holds data this
/// build cannot read (see [`FeatureData`]): sections of a type it does not
/// load, or definitions of indexes it cannot build among `indexes`, those
/// the image's catalog defines.
///
/// # Errors
///
/// Returns an error naming the database, what it holds and where, and the
/// features, in that case.
#[cfg(feature = "lpg")]
fn refuse_unreadable(
    image: &dyn ImageSource,
    indexes: &[GraphIndexes],
    path: &std::path::Path,
) -> Result<()> {
    let found: Vec<(&FeatureData, String)> = FEATURE_DATA
        .into_iter()
        .filter(|data| !data.in_build)
        .filter_map(|data| data.held(image, indexes).map(|held| (data, held)))
        .collect();
    if found.is_empty() {
        Ok(())
    } else {
        Err(refusal(path, &found))
    }
}

/// Loads the sections of `image` into the root store and its named graphs,
/// the catalog and the RDF store, and builds every graph's indexes from the
/// rows (see [`load_row_group_sections`](super::row_group_sections::load_row_group_sections)).
///
/// The image is the active checkpoint of a `.grafeo` file (served by
/// `GrafeoFileManager::read_image`) or the sections of a 0.5.x file, and
/// `file` names that database file; or it is a copy in memory, which this
/// build wrote (`file` is `None`). `read_image` holds the file manager's
/// file lock while it runs this, so nothing here, nor anything it calls, may
/// break the rule in `read_image`'s doc: no file manager method that takes
/// the file lock, and no lock a checkpoint holds while it waits for it
/// (`checkpoint_guard`, the commit hold).
///
/// Each section type is asked for once (a section this build cannot read
/// only by the refusal), so an image that serves each section once
/// (`ServedOnce`, which frees a section once it is loaded) serves them all.
/// The vector and text index sections an earlier build wrote are not read:
/// they only mirror the data, from which the load builds the indexes.
///
/// The indexes exist before WAL recovery, which keeps them current. What is
/// left (see [`LoadedSections`]) is finished by `GrafeoDB::finish_load`.
///
/// A build leaves out what it cannot read (the triples without
/// `triple-store`, the vector and text index definitions without
/// `vector-index` and `text-index`). So
/// the image of a database file is refused when it holds any of that (see
/// [`FeatureData`]), right after the catalog, which holds the index
/// definitions, and before any other section is read: a refused open has
/// changed nothing on disk.
///
/// # Errors
///
/// Returns an error if a section cannot be read or decoded, or if the image
/// of `file` holds data this build cannot read.
#[cfg(feature = "lpg")]
pub(super) fn load_sections(
    image: &dyn ImageSource,
    file: Option<&std::path::Path>,
    store: &Arc<RowGroupStore>,
    catalog: &Arc<crate::catalog::Catalog>,
    #[cfg(feature = "triple-store")] rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
) -> Result<LoadedSections> {
    let history = super::row_group_sections::load_row_group_sections(
        image,
        file,
        store,
        catalog,
        #[cfg(feature = "triple-store")]
        rdf_store,
    )?;
    Ok(LoadedSections::with_history(history))
}

/// Reads the catalog section of `image`, the database file `file` (`None`
/// for a copy in memory), into `catalog`, creating the schemas' graphs in
/// `store`, and refuses the image of a file when it holds data this build
/// cannot read (see [`FeatureData`]). Returns the index definitions it read,
/// for the loader to build once the data is in, and the epoch a 0.5.x
/// catalog names (0 for a 0.6 one).
///
/// # Errors
///
/// The section's errors, and the refusal.
#[cfg(feature = "lpg")]
pub(super) fn read_catalog<S: CatalogStore>(
    image: &dyn ImageSource,
    file: Option<&std::path::Path>,
    store: &Arc<S>,
    catalog: &Arc<crate::catalog::Catalog>,
) -> Result<(Vec<GraphIndexes>, u64)> {
    let mut indexes = Vec::new();
    let mut catalog_epoch = 0;
    if let Some(source) = image.section_source(SectionType::Catalog) {
        let transaction_manager = Arc::new(crate::transaction::TransactionManager::new());
        let mut section = CatalogSection::new(Arc::clone(catalog), Arc::clone(store), move || {
            transaction_manager.current_epoch().as_u64()
        });
        section.read_from(&*source)?;
        indexes = section.take_loaded_indexes();
        catalog_epoch = section.loaded_epoch();
    }
    if let Some(path) = file {
        refuse_unreadable(image, &indexes, path)?;
    }
    Ok((indexes, catalog_epoch))
}

/// Loads the RDF triples of `image` into `rdf_store`, and the Ring index
/// over them.
///
/// # Errors
///
/// The errors of the triples' section, and of a ring section whose chunks
/// cannot be read.
#[cfg(all(feature = "lpg", feature = "triple-store"))]
pub(super) fn load_triples(
    image: &dyn ImageSource,
    rdf_store: &Arc<grafeo_core::graph::rdf::RdfStore>,
) -> Result<()> {
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
    Ok(())
}

/// Folds the compacted base of a 0.5.x image (one `compact()` wrote) into
/// `store`, which holds its overlay: every base node and edge the overlay
/// neither copied nor deleted is created in it under its id (see
/// [`fold_0_5_base`](grafeo_core::graph::compact::fold::fold_0_5_base)).
/// The database then has one store, and its next checkpoint writes it as a
/// plain one. A deletion log without a base names nothing, and is dropped.
///
/// # Errors
///
/// Returns an error if a section cannot be read or is not the 0.5.x layout,
/// or if the base cannot be folded.
#[cfg(all(feature = "lpg", feature = "grafeo-file"))]
pub(super) fn fold_compacted_base<S: grafeo_core::graph::load::LoadTarget>(
    image: &dyn ImageSource,
    store: &S,
) -> Result<()> {
    let Some(base) = image.section_source(SectionType::CompactStore) else {
        return Ok(());
    };
    let deletions = image.section_source(SectionType::OverlayDeletions);
    let folded =
        grafeo_core::graph::compact::fold::fold_0_5_base(&*base, deletions.as_deref(), store)?;
    if folded.dangling_edges > 0 {
        grafeo_common::grafeo_warn!(
            "left out {} edges of the compacted base whose endpoint was deleted",
            folded.dangling_edges
        );
    }
    Ok(())
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
#[cfg(all(feature = "lpg", feature = "ring-index"))]
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
#[cfg(all(feature = "lpg", feature = "ring-index"))]
struct FetchWatch<'a> {
    source: &'a dyn SectionSource,
    failed: std::cell::Cell<bool>,
}

#[cfg(all(feature = "lpg", feature = "ring-index"))]
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

    fn stored_length(&self, index: usize) -> Result<u64> {
        self.source.stored_length(index)
    }

    fn section_version(&self) -> u8 {
        self.source.section_version()
    }
}

#[cfg(feature = "lpg")]
impl super::GrafeoDB {
    /// Finishes a load once the database is built: builds the indexes the
    /// sections did not hold from all the data.
    pub(super) fn finish_load(&self, loaded: LoadedSections) {
        self.build_unbuilt(loaded.unbuilt);
    }

    /// Builds the vector and text indexes of `unbuilt` from the data (see
    /// [`finish_load`](Self::finish_load)).
    pub(super) fn build_unbuilt(&self, unbuilt: Vec<GraphIndexes>) {
        for indexes in unbuilt {
            self.build_indexes(indexes);
        }
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
        for def in indexes.text {
            if let Err(err) = self.create_text_index_in(
                indexes.graph.as_deref(),
                &def.label,
                &def.property,
                &def.options,
            ) {
                grafeo_common::grafeo_warn!(
                    "text index :{}({}) not rebuilt: {err}",
                    def.label,
                    def.property
                );
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
    use grafeo_common::storage::{
        ChunkKind, ChunkMeta, MemoryImage, SectionSink, SectionSource, ServedOnce,
    };
    use grafeo_core::graph::{GraphStore as _, GraphStoreMut as _};

    /// The sections of a 0.5.x file reach their `deserialize` through raw
    /// chunks. The LPG section of this release is written as chunks only
    /// (version 4 has no blob form); the 0.5.x LPG blobs load from the
    /// fixtures of `row_group_sections`' legacy tests.
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
            .filter(|section| section.section_type() != SectionType::LpgStore)
            .map(|section| (section.section_type(), section.serialize().unwrap()))
            .collect();
        drop(commits);
        let store = Arc::new(RowGroupStore::new());
        let catalog = Arc::new(crate::catalog::Catalog::new());
        load_sections(
            &MemoryImage::from_raw(raw).unwrap(),
            None,
            &store,
            &catalog,
            #[cfg(feature = "triple-store")]
            &Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        )
        .unwrap();
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
    /// writes: the catalog, the LPG store, a vector and a text index and,
    /// with the features for them, RDF triples with their ring.
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
        let mut types = vec![SectionType::Catalog, SectionType::LpgStore];
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        types.push(SectionType::RdfStore);
        #[cfg(all(feature = "sparql", feature = "ring-index"))]
        types.push(SectionType::RdfRing);
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

    fn load(image: &dyn ImageSource, store: &Arc<RowGroupStore>) -> Result<LoadedSections> {
        load_sections(
            image,
            None,
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
            None,
            &Arc::new(RowGroupStore::new()),
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
            None,
            &Arc::new(RowGroupStore::new()),
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
            .filter(|section| section.section_type() != SectionType::LpgStore)
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
            None,
            &Arc::new(RowGroupStore::new()),
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
            None,
            &Arc::new(RowGroupStore::new()),
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
        fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
            grafeo_common::storage::read_raw(self, source)
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
        let store = Arc::new(RowGroupStore::new());
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

    /// The image of a database file 0.5.44 wrote with `compact()` between two
    /// sessions: the three sections of the core fixture `compact-0.5.44`, each
    /// one raw chunk. The base holds Alix, Gus, Vincent, Amsterdam, Berlin and
    /// two documents, with Alix knowing Gus and living in Amsterdam; the
    /// overlay copies of Alix (age 31, no score) and Amsterdam, and Mia, the
    /// Rijksmuseum (in Amsterdam) and a third document; the log Vincent.
    #[cfg(feature = "grafeo-file")]
    fn compacted_image() -> MemoryImage {
        use grafeo_common::storage::ChunkMeta;

        macro_rules! fixture {
            ($name:literal) => {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../grafeo-core/tests/fixtures/compact-0.5.44/",
                    $name
                ))
                .as_slice()
            };
        }
        let mut image = MemoryImage::new();
        for (section_type, bytes) in [
            (SectionType::LpgStore, fixture!("lpg_store.bin")),
            (SectionType::CompactStore, fixture!("compact_store.bin")),
            (
                SectionType::OverlayDeletions,
                fixture!("overlay_deletions.bin"),
            ),
        ] {
            image.begin_section(section_type, 1).unwrap();
            image.write_chunk(ChunkMeta::raw(), bytes).unwrap();
        }
        image
    }

    /// The load folds a compacted base into the store its overlay loads into,
    /// reading the base and the deletion log once: the base's nodes and edges
    /// come back under their ids, the overlay's copy of a changed one wins,
    /// and the deleted ones stay deleted.
    #[cfg(feature = "grafeo-file")]
    #[test]
    fn a_compacted_base_folds_into_the_store() {
        use grafeo_common::types::{PropertyKey, Value};

        let image = ServedOnce::new(compacted_image());
        let store = Arc::new(RowGroupStore::new());
        load(&image, &store).unwrap();

        assert!(
            image.section_types().is_empty(),
            "the overlay, the base and the deletion log are read"
        );
        let values = |label: &str, key: &str| {
            let mut values: Vec<String> = store
                .nodes_by_label(label)
                .into_iter()
                .filter_map(|id| store.get_node_property(id, &PropertyKey::from(key)))
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            values.sort();
            values
        };
        assert_eq!(values("Person", "name"), ["Alix", "Gus", "Mia"]);
        assert_eq!(values("City", "name"), ["Amsterdam", "Berlin"]);
        assert_eq!(
            values("Document", "title"),
            ["Bridges", "Canals", "Museums"]
        );

        let alix = store
            .nodes_by_label("Person")
            .into_iter()
            .find(|&id| {
                store.get_node_property(id, &PropertyKey::from("name")) == Some(Value::from("Alix"))
            })
            .unwrap();
        assert_eq!(
            store.get_node_property(alix, &PropertyKey::from("age")),
            Some(Value::Int64(31)),
            "the overlay's copy of Alix wins"
        );
        let knows: Vec<_> = store
            .edges_from(alix, grafeo_core::graph::Direction::Outgoing)
            .into_iter()
            .filter(|&(_, edge)| store.edge_type(edge).as_deref() == Some("KNOWS"))
            .collect();
        assert_eq!(knows.len(), 1);
        assert_eq!(
            store.get_edge_property(knows[0].1, &PropertyKey::from("since")),
            Some(Value::Int64(2019)),
            "a base edge keeps its properties"
        );
        assert_eq!(
            store.edge_count(),
            3,
            "knows and lives in from the base, the museum's from the overlay"
        );
        let highest = store.node_ids().into_iter().max().unwrap();
        assert!(
            store.create_node(&["City"]) > highest,
            "new ids stay clear of the base's and the overlay's"
        );
    }

    /// A database loaded from a compacted image is one plain store: its
    /// checkpoint writes the folded data in the LPG section and no compacted
    /// sections, and that image loads back to the same data.
    #[cfg(feature = "grafeo-file")]
    #[test]
    fn a_folded_database_checkpoints_as_a_plain_one() {
        let db = GrafeoDB::new_in_memory();
        let loaded = load(&compacted_image(), &db.lpg_store()).unwrap();
        db.finish_load(loaded);
        let names = |db: &GrafeoDB| {
            db.execute("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name ORDER BY n.name")
                .unwrap()
                .rows()
                .to_vec()
        };
        let before = names(&db);
        assert_eq!(
            before.len(),
            6,
            "Alix, Amsterdam, Berlin, Gus, Mia, Rijksmuseum"
        );

        let checkpoint = image_of(&db);
        assert_eq!(
            by_byte(checkpoint.section_types()),
            by_byte(vec![SectionType::Catalog, SectionType::LpgStore])
        );
        let reopened = GrafeoDB::new_in_memory();
        let loaded = load(&checkpoint, &reopened.lpg_store()).unwrap();
        reopened.finish_load(loaded);
        assert_eq!(names(&reopened), before);
        let edges = reopened
            .execute("MATCH ()-[r]->() RETURN count(r)")
            .unwrap();
        assert_eq!(edges.rows(), [[grafeo_common::types::Value::Int64(3)]]);
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

    /// The image of a checkpoint of `db` in which `section_type` claims the
    /// section version `version`, as a later release that changed its layout
    /// would write it (its chunks as this release writes them).
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    fn image_with_version(db: &GrafeoDB, section_type: SectionType, version: u8) -> MemoryImage {
        let commits = db.transaction_manager.hold_commits().unwrap();
        let mut image = MemoryImage::new();
        for section in db.checkpoint_sources().sections(&commits) {
            let claimed = if section.section_type() == section_type {
                version
            } else {
                section.version()
            };
            image
                .begin_section(section.section_type(), claimed)
                .unwrap();
            section.write_to(&mut image).unwrap();
        }
        image
    }

    /// An RDF ring section of a section version this build does not read is
    /// no error either: the ring is built from the triples it mirrors.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    #[test]
    fn a_ring_section_of_another_version_is_rebuilt() {
        use grafeo_core::graph::rdf::RdfStore;

        let db = ringed_database();
        let image = image_with_version(&db, SectionType::RdfRing, 250);
        let rdf_store = Arc::new(RdfStore::new());
        load_sections(
            &image,
            None,
            &Arc::new(RowGroupStore::new()),
            &Arc::new(crate::catalog::Catalog::new()),
            &rdf_store,
        )
        .expect("the load succeeds");
        assert_eq!(rdf_store.len(), 2, "the triples are loaded");
        let ring = rdf_store
            .ring()
            .expect("the ring is built from the triples");
        assert_eq!(ring.len(), 2);
    }

    /// A section read through `read_mirror` sees the version it was written
    /// with.
    #[cfg(feature = "ring-index")]
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
            fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
                grafeo_common::storage::write_raw(self, sink)
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
        let store = Arc::new(RowGroupStore::new());
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
            "the vector index is built from the rows"
        );
    }

    /// The image `to_memory` loads holds every section of a checkpoint, each
    /// with its version and chunks: the LPG section too, which writes every
    /// graph at the checkpoint's epoch.
    #[test]
    fn the_copy_image_holds_every_section_of_a_checkpoint() {
        let db = every_kind_of_section();
        let target = GrafeoDB::new_in_memory();
        let copied = db.copy_into(&target).unwrap();

        let commits = db.transaction_manager.hold_commits().unwrap();
        let sections = db.checkpoint_sources().sections(&commits);
        let refs: Vec<&dyn Section> = sections.iter().map(AsRef::as_ref).collect();
        let expected = MemoryImage::from_sections(&refs).unwrap();
        drop(commits);

        assert_eq!(
            by_byte(copied.section_types()),
            every_section_type(),
            "every section of a checkpoint"
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
    }

    /// An image whose `unreadable` section fails every fetch, as a chunk that
    /// fails its checksum does.
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    struct Unreadable<'a> {
        image: &'a MemoryImage,
        unreadable: SectionType,
    }

    #[cfg(all(feature = "ring-index", feature = "sparql"))]
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
    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    struct FailingFetch<'a>(SectionType, Box<dyn SectionSource + 'a>);

    #[cfg(all(feature = "ring-index", feature = "sparql"))]
    impl SectionSource for FailingFetch<'_> {
        fn chunks(&self) -> &[ChunkMeta] {
            self.1.chunks()
        }

        fn fetch(&self, _index: usize) -> Result<bytes::Bytes> {
            Err(grafeo_common::utils::error::Error::corruption_at(
                format!(
                    "chunk of section {:?} at offset 16384 fails its checksum",
                    self.0
                ),
                16384,
            ))
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            self.1.stored_length(index)
        }

        fn section_version(&self) -> u8 {
            self.1.section_version()
        }
    }

    /// A database whose checkpoint holds every section a checkpoint of this
    /// build writes: the catalog, the LPG store with a named graph, a vector
    /// and a text index and, with the features for them, RDF triples in the
    /// default and a named graph, with their ring.
    fn database_with_every_section() -> GrafeoDB {
        let db = indexed_database();
        db.create_graph("travel").unwrap();
        db.graph("travel")
            .unwrap()
            .execute("INSERT (:City {name: 'Prague'})-[:ROUTE {km: 88}]->(:City {name: 'Paris'})")
            .unwrap();
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        {
            db.execute_sparql(
                "INSERT DATA { <http://example.org/mia> <http://example.org/lives_in> \
                 <http://example.org/prague> . GRAPH <http://example.org/trips> { \
                 <http://example.org/jules> <http://example.org/visited> \
                 <http://example.org/berlin> } }",
            )
            .unwrap();
            #[cfg(feature = "ring-index")]
            db.rdf_store().rebuild_ring();
        }
        db
    }

    /// Every section a checkpoint writes streams chunks: none is one raw
    /// chunk, which only 0.5.x bytes are. Each holds one metadata chunk, first
    /// (the LPG store's last: it lists the names met while the chunks before
    /// it were written), and its data in chunks of their own.
    #[test]
    fn no_production_section_writes_a_raw_chunk() {
        let mut expected = vec![SectionType::Catalog, SectionType::LpgStore];
        #[cfg(all(feature = "sparql", feature = "triple-store"))]
        expected.push(SectionType::RdfStore);
        #[cfg(all(feature = "sparql", feature = "ring-index"))]
        expected.push(SectionType::RdfRing);
        let db = database_with_every_section();
        let commits = db.transaction_manager.hold_commits().unwrap();
        let sections = db.checkpoint_sources().sections(&commits);
        let refs: Vec<&dyn Section> = sections.iter().map(AsRef::as_ref).collect();
        let image = MemoryImage::from_sections(&refs).unwrap();
        drop(commits);
        assert_eq!(
            by_byte(image.section_types()),
            by_byte(expected),
            "the checkpoint holds every section the database has"
        );
        for section_type in image.section_types() {
            let source = image.section_source(section_type).unwrap();
            let chunks = source.chunks();
            assert!(
                chunks.iter().all(|meta| meta.kind != ChunkKind::Raw),
                "{section_type:?} writes no raw chunk: {chunks:?}"
            );
            let metadata: Vec<usize> = chunks
                .iter()
                .enumerate()
                .filter(|(_, meta)| meta.kind == ChunkKind::Meta)
                .map(|(index, _)| index)
                .collect();
            let place = if section_type == SectionType::LpgStore {
                chunks.len() - 1
            } else {
                0
            };
            assert_eq!(
                metadata,
                [place],
                "{section_type:?} has one metadata chunk, at {place} of its {} chunks",
                chunks.len()
            );
            assert!(
                chunks.len() > 1,
                "{section_type:?} holds its data in chunks of their own"
            );
        }
    }
}
