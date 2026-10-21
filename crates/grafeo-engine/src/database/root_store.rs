//! The root store a database's loads, replays and standalone changes write:
//! its named graphs, their indexes, its epoch, and its rows
//! ([`LoadTarget`]): the row-group store. The readers of earlier formats (a
//! 0.5.x file and its WAL, a 0.5.x WAL directory, a snapshot, the embeddings
//! an older build spilled), the replay of a standalone change and a graph
//! copy are written against this trait, which the row-group store answers
//! by its inherent methods of the same names (H4, #416, settles the store
//! contract these become part of).

use grafeo_common::change::DataOp;
use grafeo_common::types::EpochId;
use grafeo_core::graph::apply::ChangeTarget;
use grafeo_core::graph::load::LoadTarget;
use grafeo_core::graph::rowgroup::RowGroupStore;

use super::catalog_section::CatalogStore;

/// See the module docs.
pub(crate) trait RootStore: CatalogStore + LoadTarget + ChangeTarget {
    /// Drops the named graph `name`; `false` when there is none.
    fn drop_graph(&self, name: &str) -> bool;

    /// The named graph `name`, created when there is none.
    ///
    /// # Errors
    ///
    /// When every graph id was given out.
    #[cfg(feature = "wal")]
    fn graph_or_create(
        &self,
        name: &str,
    ) -> grafeo_common::utils::error::Result<std::sync::Arc<Self>>;

    /// Creates the property index on `key`, built from the data.
    fn create_property_index(&self, key: &str);

    /// Drops the property index on `key`; `false` when there is none.
    fn drop_property_index(&self, key: &str) -> bool;

    /// Installs `index` as the vector index on `label:property`.
    #[cfg(feature = "vector-index")]
    fn add_vector_index(
        &self,
        label: &str,
        property: &str,
        index: std::sync::Arc<grafeo_core::index::vector::VectorIndexKind>,
    );

    /// Removes the vector index on `label:property`; `false` when there is
    /// none.
    #[cfg(feature = "vector-index")]
    fn remove_vector_index(&self, label: &str, property: &str) -> bool;

    /// The vector index on `label:property`.
    #[cfg(all(feature = "grafeo-file", feature = "vector-index"))]
    fn get_vector_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<std::sync::Arc<grafeo_core::index::vector::VectorIndexKind>>;

    /// Fills values of `key` an older build spilled (#594) into the nodes
    /// of `label` that have none; returns how many it filled (see
    /// `RowGroupStore::fill_missing_node_values`).
    ///
    /// # Errors
    ///
    /// The error of reading a stored value, or of a value the store refuses.
    #[cfg(all(feature = "grafeo-file", feature = "vector-index"))]
    fn fill_missing_node_values(
        &self,
        label: &str,
        key: &grafeo_common::types::PropertyKey,
        values: Vec<(grafeo_common::types::NodeId, grafeo_common::types::Value)>,
    ) -> grafeo_common::utils::error::Result<usize>;

    /// Installs `index` as the text index on `label:property`.
    #[cfg(feature = "text-index")]
    fn add_text_index(
        &self,
        label: &str,
        property: &str,
        index: std::sync::Arc<parking_lot::RwLock<grafeo_core::index::text::InvertedIndex>>,
    );

    /// Removes the text index on `label:property`; `false` when there is
    /// none.
    #[cfg(feature = "text-index")]
    fn remove_text_index(&self, label: &str, property: &str) -> bool;

    /// Moves the store's epoch to `epoch` when it is later.
    fn sync_epoch(&self, epoch: EpochId);

    /// A commit of the WAL of the first log format, replayed: the store
    /// moves to a new epoch, so the rows the next commit writes are stamped
    /// after it.
    #[cfg(feature = "wal")]
    fn replayed_commit(&self);

    /// The create ops of every node, then every edge, a reader at `epoch`
    /// sees, with their ids, labels and values at `epoch`: what a graph copy
    /// logs and applies.
    fn copied(&self, epoch: EpochId) -> Box<dyn Iterator<Item = DataOp> + '_>;

    /// Loads the sections of `image`, the image of the database file `file`
    /// (`None` for a copy in memory), into `store`, `catalog` and the RDF
    /// store, as an open loads them (see `sections::load_sections`).
    ///
    /// # Errors
    ///
    /// The errors of reading a section, and the refusal of data this build
    /// cannot read.
    #[cfg(feature = "grafeo-file")]
    fn load_image(
        store: &std::sync::Arc<Self>,
        image: &dyn grafeo_common::storage::ImageSource,
        file: Option<&std::path::Path>,
        catalog: &std::sync::Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<
            grafeo_core::graph::rdf::RdfStore,
        >,
    ) -> grafeo_common::utils::error::Result<super::sections::LoadedSections>;
}

/// Implements the index and graph methods of [`RootStore`] for a store by
/// its inherent methods of the same names.
macro_rules! root_store_indexes {
    () => {
        fn drop_graph(&self, name: &str) -> bool {
            Self::drop_graph(self, name)
        }

        #[cfg(feature = "wal")]
        fn graph_or_create(
            &self,
            name: &str,
        ) -> grafeo_common::utils::error::Result<std::sync::Arc<Self>> {
            Ok(Self::graph_or_create(self, name)?)
        }

        fn create_property_index(&self, key: &str) {
            Self::create_property_index(self, key);
        }

        fn drop_property_index(&self, key: &str) -> bool {
            Self::drop_property_index(self, key)
        }

        #[cfg(feature = "vector-index")]
        fn add_vector_index(
            &self,
            label: &str,
            property: &str,
            index: std::sync::Arc<grafeo_core::index::vector::VectorIndexKind>,
        ) {
            Self::add_vector_index(self, label, property, index);
        }

        #[cfg(feature = "vector-index")]
        fn remove_vector_index(&self, label: &str, property: &str) -> bool {
            Self::remove_vector_index(self, label, property)
        }

        #[cfg(all(feature = "grafeo-file", feature = "vector-index"))]
        fn get_vector_index(
            &self,
            label: &str,
            property: &str,
        ) -> Option<std::sync::Arc<grafeo_core::index::vector::VectorIndexKind>> {
            Self::get_vector_index(self, label, property)
        }

        #[cfg(all(feature = "grafeo-file", feature = "vector-index"))]
        fn fill_missing_node_values(
            &self,
            label: &str,
            key: &grafeo_common::types::PropertyKey,
            values: Vec<(grafeo_common::types::NodeId, grafeo_common::types::Value)>,
        ) -> grafeo_common::utils::error::Result<usize> {
            Self::fill_missing_node_values(self, label, key, values)
        }

        #[cfg(feature = "text-index")]
        fn add_text_index(
            &self,
            label: &str,
            property: &str,
            index: std::sync::Arc<parking_lot::RwLock<grafeo_core::index::text::InvertedIndex>>,
        ) {
            Self::add_text_index(self, label, property, index);
        }

        #[cfg(feature = "text-index")]
        fn remove_text_index(&self, label: &str, property: &str) -> bool {
            Self::remove_text_index(self, label, property)
        }

        fn sync_epoch(&self, epoch: EpochId) {
            Self::sync_epoch(self, epoch);
        }
    };
}

impl RootStore for RowGroupStore {
    root_store_indexes!();

    /// Loads the sections into the root (see
    /// `row_group_sections::load_row_group_sections`), which builds the
    /// catalog's indexes from the rows: nothing is left for after the load.
    #[cfg(feature = "grafeo-file")]
    fn load_image(
        store: &std::sync::Arc<Self>,
        image: &dyn grafeo_common::storage::ImageSource,
        file: Option<&std::path::Path>,
        catalog: &std::sync::Arc<crate::catalog::Catalog>,
        #[cfg(feature = "triple-store")] rdf_store: &std::sync::Arc<
            grafeo_core::graph::rdf::RdfStore,
        >,
    ) -> grafeo_common::utils::error::Result<super::sections::LoadedSections> {
        let history = super::row_group_sections::load_row_group_sections(
            image,
            file,
            store,
            catalog,
            #[cfg(feature = "triple-store")]
            rdf_store,
        )?;
        Ok(super::sections::LoadedSections::with_history(history))
    }

    /// Each replayed commit gets an epoch of its own, as a commit of the
    /// log does: the rows the next one writes are stamped after it.
    #[cfg(feature = "wal")]
    fn replayed_commit(&self) {
        self.new_epoch();
    }

    fn copied(&self, epoch: EpochId) -> Box<dyn Iterator<Item = DataOp> + '_> {
        let nodes = self.nodes_at(epoch).map(|node| DataOp::CreateNode {
            id: node.id,
            labels: node.labels,
            properties: node.properties.into_iter().collect(),
        });
        let edges = self.edges_at(epoch).map(|edge| DataOp::CreateEdge {
            id: edge.id,
            src: edge.src,
            dst: edge.dst,
            edge_type: edge.edge_type,
            properties: edge.properties.into_iter().collect(),
        });
        Box::new(nodes.chain(edges))
    }
}
