//! The index records of a WAL of the format before 0.6.0 (`CreateIndex` and
//! `DropIndex`, #401), applied once the WAL is replayed: the migration of a
//! 0.5.x WAL directory, and a file's sidecar WAL of that format.
//!
//! Such a record names the index, its label, its property and its kind, but
//! not the graph it is in nor a vector index's parameters. An index belongs
//! to the graph the WAL had switched to when it was logged. A vector index
//! takes its dimensions from the first vector of its property, and the
//! parameters `CREATE VECTOR INDEX` takes without options (the cosine
//! distance, the HNSW defaults), with a warning that names it. The records
//! are kept until the whole WAL is replayed, as an index can come before the
//! data it indexes.

use std::sync::Arc;

use grafeo_common::change::StandaloneOp;
use grafeo_common::storage::catalog_record::{
    CatalogRecord, IndexKindRecord, IndexNameKindRecord, IndexNameRecord,
};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_storage::wal::WalRecord;

use super::root_store::RootStore;
use super::standalone::{self, Applying};
use crate::catalog::Catalog;

/// An index a `CreateIndex` record defines, which no later `DropIndex`
/// dropped.
struct LegacyIndex {
    name: String,
    label: String,
    property: String,
    /// `property`, `btree`, `text` or `vector`.
    kind: String,
    /// The graph the WAL was on, `None` for the default graph.
    graph: Option<String>,
}

/// The indexes the index records of a WAL define, in the order of the WAL.
#[derive(Default)]
pub(super) struct LegacyIndexes(Vec<LegacyIndex>);

impl LegacyIndexes {
    /// Takes `record` when it is an index record, logged while the WAL was on
    /// `graph`: a create replaces an index of the same name, a drop removes it.
    pub(super) fn record(&mut self, record: &WalRecord, graph: Option<&str>) {
        match record {
            WalRecord::CreateIndex {
                name,
                label,
                property,
                index_type,
            } => {
                self.0.retain(|index| index.name != *name);
                self.0.push(LegacyIndex {
                    name: name.clone(),
                    label: label.clone(),
                    property: property.clone(),
                    kind: index_type.clone(),
                    graph: graph.map(str::to_string),
                });
            }
            WalRecord::DropIndex { name } => self.0.retain(|index| index.name != *name),
            _ => {}
        }
    }

    /// Defines each index, once the WAL of `wal` is replayed into `store`
    /// and `catalog`, as `CREATE INDEX` does: the index itself (a vector or
    /// text index left in `unbuilt`, built from the data once the database
    /// is built) and its name, for `SHOW INDEXES` and `DROP INDEX`. An index
    /// of a graph a later record dropped went with it.
    ///
    /// # Errors
    ///
    /// An index kind the format never wrote; a vector or text index this
    /// build cannot build (it would be lost, see
    /// [`FeatureData`](super::sections::FeatureData)).
    pub(super) fn apply<S: RootStore>(
        self,
        wal: &std::path::Path,
        store: &Arc<S>,
        catalog: &Catalog,
        unbuilt: &mut Vec<super::catalog_section::GraphIndexes>,
    ) -> Result<()> {
        for index in self.0 {
            let graph = match &index.graph {
                None => Arc::clone(store),
                Some(name) => match store.graph(name) {
                    Some(graph) => graph,
                    None => continue,
                },
            };
            let Some((kind, name_kind)) = index_kind(&index, &*graph, wal)? else {
                continue;
            };
            let named = name_kind.map(|kind| {
                StandaloneOp::PutCatalog(CatalogRecord::IndexName(IndexNameRecord {
                    name: index.name.clone(),
                    label: index.label.clone(),
                    property: index.property.clone(),
                    kind,
                }))
            });
            let put = super::index::put_index(index.graph.as_deref(), kind);
            for op in std::iter::once(put).chain(named) {
                standalone::apply(
                    &op,
                    None,
                    Some(store),
                    catalog,
                    &mut Applying::Replay { unbuilt },
                )?;
            }
        }
        Ok(())
    }
}

/// What `index` indexes and the kind of its name (none for a vector index,
/// which `CREATE VECTOR INDEX` does not name for `SHOW INDEXES` either), from
/// the data of `graph`; `None` for a vector index without a vector to take
/// its dimensions from.
fn index_kind<G: grafeo_core::graph::GraphStore + ?Sized>(
    index: &LegacyIndex,
    graph: &G,
    wal: &std::path::Path,
) -> Result<Option<(IndexKindRecord, Option<IndexNameKindRecord>)>> {
    let property = IndexKindRecord::Property {
        key: index.property.clone(),
    };
    match index.kind.as_str() {
        "property" => Ok(Some((property, Some(IndexNameKindRecord::Hash)))),
        "btree" => Ok(Some((property, Some(IndexNameKindRecord::BTree)))),
        "text" => text_index_kind(index, wal).map(Some),
        "vector" => vector_index_kind(index, graph, wal),
        other => Err(Error::from(StorageError::RecoveryFailed(format!(
            "the WAL {} defines the index '{}' of the unknown kind '{other}'",
            wal.display(),
            index.name
        )))),
    }
}

#[cfg(feature = "text-index")]
fn text_index_kind(
    index: &LegacyIndex,
    _wal: &std::path::Path,
) -> Result<(IndexKindRecord, Option<IndexNameKindRecord>)> {
    let kind = super::index::text_index_kind(
        &index.label,
        &index.property,
        &grafeo_core::index::text::TextIndexOptions::default(),
    )?;
    Ok((kind, Some(IndexNameKindRecord::FullText)))
}

#[cfg(not(feature = "text-index"))]
fn text_index_kind(
    index: &LegacyIndex,
    wal: &std::path::Path,
) -> Result<(IndexKindRecord, Option<IndexNameKindRecord>)> {
    Err(super::sections::refusal(
        wal,
        &[(
            &super::sections::TEXT_INDEXES,
            format!("WAL records: the text index '{}'", index.name),
        )],
    ))
}

#[cfg(feature = "vector-index")]
fn vector_index_kind<G: grafeo_core::graph::GraphStore + ?Sized>(
    index: &LegacyIndex,
    graph: &G,
    wal: &std::path::Path,
) -> Result<Option<(IndexKindRecord, Option<IndexNameKindRecord>)>> {
    use grafeo_common::types::{ReadView, Value};

    // An index build at the open: every write the replay applied.
    let view = ReadView::latest();
    let mut dimensions = None;
    for node in graph.nodes_by_label(&index.label, view)? {
        if let Some(Value::Vector(vector)) = graph.node_property(node, &index.property, view)? {
            dimensions = Some(vector.len());
            break;
        }
    }
    let Some(dimensions) = dimensions else {
        grafeo_common::grafeo_warn!(
            "the vector index '{}' on :{}({}) of the WAL {} has no vector to take its \
             dimensions from, so it is not defined: create it again",
            index.name,
            index.label,
            index.property,
            wal.display()
        );
        return Ok(None);
    };
    let metric = super::index::vector_index_metric(None)?;
    let built = super::GrafeoDB::build_vector_index(
        dimensions,
        metric,
        None,
        None,
        grafeo_core::index::vector::QuantizationType::None,
        0,
    );
    let record = super::index::vector_index_record(
        index.graph.as_deref(),
        &index.label,
        &index.property,
        &built,
    )?;
    grafeo_common::grafeo_warn!(
        "the vector index '{}' on :{}({}) of the WAL {} records no distance or HNSW \
         parameters: it is defined with those of CREATE VECTOR INDEX without options (the \
         {} distance); drop it and create it again if it used others",
        index.name,
        index.label,
        index.property,
        wal.display(),
        metric.name()
    );
    Ok(Some((record.index, None)))
}

#[cfg(not(feature = "vector-index"))]
fn vector_index_kind<G: grafeo_core::graph::GraphStore + ?Sized>(
    index: &LegacyIndex,
    _graph: &G,
    wal: &std::path::Path,
) -> Result<Option<(IndexKindRecord, Option<IndexNameKindRecord>)>> {
    Err(super::sections::refusal(
        wal,
        &[(
            &super::sections::VECTOR_INDEXES,
            format!("WAL records: the vector index '{}'", index.name),
        )],
    ))
}
