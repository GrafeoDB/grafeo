//! Standalone changes: graph commands, schema statements and the index API.
//!
//! None of them is an entry of a transaction's change set. Each statement or
//! call collects what it changes in a [`StandaloneChange`] while it holds
//! commits off, checking every op against the catalog and the store first,
//! then [`commit`] logs the change as one WAL group of its own and applies it
//! op by op through [`apply`], the function replay applies the logged ops
//! with ([`replay`]). So a statement refused by its checks leaves nothing,
//! in memory or in the log, and what replay rebuilds is what the statement
//! made, record for record.
//!
//! Such a change takes effect at once, also inside a transaction, whose
//! rollback keeps it (0.6.0 has no transactional DDL).
//!
//! An RDF graph operation (`CREATE`, `DROP`, `CLEAR`, `COPY`, `MOVE`, `ADD`)
//! is one too: [`commit_rdf`] logs it as a group of its own and applies it
//! to the RDF store through [`apply_rdf`], which replay applies it with.

use std::sync::Arc;

use grafeo_common::change::StandaloneOp;
use grafeo_common::storage::catalog_record::{
    CatalogKey, CatalogRecord, IndexKeyRecord, IndexKindRecord, IndexNameRecord, IndexRecord,
};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::LpgStore;

use super::catalog_records::{
    constraint_definition, edge_type_definition, graph_indexes, graph_type_definition, index_name,
    node_type_definition, procedure_definition, record_name,
};
use super::catalog_section::GraphIndexes;
use crate::catalog::{Catalog, CatalogError};
use crate::transaction::{BuiltIndex, CommitsHeld, StandaloneChange, TransactionManager};

/// Who applies an op.
pub(crate) enum Applying<'a> {
    /// A statement or call, whose checks passed and whose change was logged:
    /// an op that does not apply is a broken invariant.
    Live,
    /// Replay of a logged op, which can meet what the database image holds
    /// already (a crash between writing a checkpoint and trimming the WAL)
    /// or miss what a later op of the WAL removed: an op that finds its
    /// entry there (or gone, for a drop) changes nothing. The vector and
    /// text indexes it puts are kept in `unbuilt`, built from the data once
    /// the whole WAL is replayed.
    #[cfg_attr(
        not(feature = "wal"),
        expect(dead_code, reason = "only the replay of a WAL applies as replay")
    )]
    Replay {
        /// The indexes left to build once the database is built.
        unbuilt: &'a mut Vec<GraphIndexes>,
    },
}

impl Applying<'_> {
    /// `outcome` of a catalog call for `op`: a repeat is no error on replay.
    fn catalog(
        &self,
        op: &StandaloneOp,
        outcome: std::result::Result<(), CatalogError>,
    ) -> Result<()> {
        match outcome {
            Ok(()) => Ok(()),
            Err(error) if self.tolerates(&error) => {
                grafeo_common::grafeo_debug!("WAL replay skipped {}: {error}", describe(op));
                Ok(())
            }
            Err(error) => Err(not_applied(op, &error.to_string())),
        }
    }

    /// Whether `error` is one replay meets for an op whose effect is there
    /// already, or whose entry a later op removed (see [`Applying::Replay`]).
    fn tolerates(&self, error: &CatalogError) -> bool {
        matches!(self, Self::Replay { .. })
            && matches!(
                error,
                CatalogError::TypeAlreadyExists(_)
                    | CatalogError::TypeNotFound(_)
                    | CatalogError::SchemaAlreadyExists(_)
                    | CatalogError::SchemaNotFound(_)
                    | CatalogError::ConstraintAlreadyExists
                    | CatalogError::ConstraintNotFound(_)
            )
    }

    /// Fails a live op that found nothing to change (`what` says what it
    /// missed); replay goes on.
    fn missing(&self, op: &StandaloneOp, what: &str) -> Result<()> {
        match self {
            Self::Live => Err(not_applied(op, what)),
            Self::Replay { .. } => {
                grafeo_common::grafeo_debug!("WAL replay skipped {}: {what}", describe(op));
                Ok(())
            }
        }
    }
}

/// How errors name an op.
fn describe(op: &StandaloneOp) -> String {
    match op {
        StandaloneOp::CreateGraph { name } => format!("the creation of graph '{name}'"),
        StandaloneOp::DropGraph { name } => format!("the drop of graph '{name}'"),
        StandaloneOp::PutCatalog(record) => record_name(record),
        StandaloneOp::DropCatalog(key) => format!("the drop of {key:?}"),
        StandaloneOp::RdfGraph(op) => format!("the RDF graph operation {op:?}"),
    }
}

/// The error of an op that does not apply.
fn not_applied(op: &StandaloneOp, reason: &str) -> Error {
    Error::Internal(format!("{} did not apply: {reason}", describe(op)))
}

/// Holds commits off for a standalone change for as long as the guard
/// lives, so a checkpoint or `close()` sees all of it or none of it. With
/// `writes_too` (a graph drop) it also waits for the writes of open
/// transactions in progress and keeps new ones out meanwhile, so what their
/// change sets hold is what the stores hold (see
/// [`refuse_drop_with_open_changes`]).
///
/// # Errors
///
/// The database-closed error after `close()` of a persistent database, and
/// the incomplete-commit error after a commit that did not complete.
pub(crate) fn hold(manager: &TransactionManager, writes_too: bool) -> Result<CommitsHeld<'_>> {
    let held = if writes_too {
        let held = manager.hold_commits()?;
        manager.check_open()?;
        held
    } else {
        manager.hold_commits_for_change()?
    };
    // Tests start a checkpoint or `close()` here, which must wait.
    #[cfg(feature = "testing-statement-injection")]
    grafeo_common::testing::commit_hook::run_during_held_change();
    Ok(held)
}

/// Refuses to drop the graph with storage key `name` while an open
/// transaction (the dropping session's own one too) has changes in it: its
/// commit would log them under the graph's name, and replay would create
/// the graph again for them. The caller holds commits off and the writes
/// of open transactions out (`held`), so no transaction records a change in
/// the graph between this check and the drop, and the dropped graph's store
/// refuses every write after it.
///
/// # Errors
///
/// A write conflict naming the graph.
pub(crate) fn refuse_drop_with_open_changes(
    manager: &TransactionManager,
    held: &CommitsHeld<'_>,
    name: &str,
) -> Result<()> {
    let changed = manager
        .open_change_sets(held)
        .iter()
        .any(|changes| changes.writes_graph(Some(name)));
    if changed {
        return Err(Error::Transaction(
            grafeo_common::utils::error::TransactionError::WriteConflict(format!(
                "graph '{name}' has changes of an open transaction: drop it once that \
                 transaction commits or rolls back"
            )),
        ));
    }
    Ok(())
}

/// Logs `change` as one WAL group of its own (to `wal`, when the database
/// logs), then applies its ops in order. The caller checked every op while
/// holding commits off (`_held`), so a checkpoint or `close()` sees all of
/// the change or none of it.
///
/// # Errors
///
/// The error of writing the group: nothing is applied then. An op that does
/// not apply after the group was logged is a broken invariant: the error
/// comes back and the database is poisoned (no commit until it is reopened,
/// whose replay applies the group).
pub(crate) fn commit(
    change: StandaloneChange,
    _held: &CommitsHeld<'_>,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    store: &Arc<LpgStore>,
    catalog: &Catalog,
    manager: &TransactionManager,
) -> Result<()> {
    if change.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "wal")]
    if let Some(wal) = wal {
        wal.log_batch(&crate::transaction::v1_group::standalone_group(&change)?)?;
    }
    for (op, built) in change.into_ops() {
        if let Err(error) = apply(&op, built, store, catalog, &mut Applying::Live) {
            manager.poison(&format!("a logged change did not apply: {error}"));
            return Err(error);
        }
    }
    Ok(())
}

/// Applies `op` to `store` (the root store, which holds the named graphs)
/// and `catalog`: what a statement does once its change is logged, and what
/// replay does with the logged op. `built` is the index a statement built
/// for the put of an index record.
///
/// # Errors
///
/// Live: an op the statement's checks should have refused. Replay: an op
/// that does not apply for another reason than a repeat (see
/// [`Applying::Replay`]), or a put of an index whose definition does not fit
/// this platform.
pub(crate) fn apply(
    op: &StandaloneOp,
    built: Option<BuiltIndex>,
    store: &Arc<LpgStore>,
    catalog: &Catalog,
    applying: &mut Applying<'_>,
) -> Result<()> {
    match op {
        StandaloneOp::CreateGraph { name } => {
            if store.create_graph(name)? {
                Ok(())
            } else {
                applying.missing(op, "the graph exists")
            }
        }
        StandaloneOp::DropGraph { name } => {
            if let Applying::Replay { unbuilt } = applying {
                unbuilt.retain(|graph| graph.graph.as_deref() != Some(name.as_str()));
            }
            if store.drop_graph(name) {
                Ok(())
            } else {
                applying.missing(op, "the graph does not exist")
            }
        }
        StandaloneOp::PutCatalog(record) => put(op, record, built, store, catalog, applying),
        StandaloneOp::DropCatalog(key) => drop_entry(op, key, store, catalog, applying),
        // A change of the RDF store, which `commit_rdf` and `replay` apply
        // through `apply_rdf`: a change of the labeled property graphs and
        // the catalog never holds one.
        StandaloneOp::RdfGraph(_) => Err(not_applied(
            op,
            "an RDF graph operation applies to the RDF store, through `apply_rdf`",
        )),
    }
}

/// Logs the RDF graph operation `op` as one WAL group of its own (to `wal`,
/// when the database logs), then applies it to `rdf`. The caller checked it
/// while holding commits off (`_held`), so a checkpoint or `close()` sees
/// all of it or none of it.
///
/// # Errors
///
/// As [`commit`].
#[cfg(feature = "triple-store")]
pub(crate) fn commit_rdf(
    op: grafeo_common::change::RdfGraphOp,
    _held: &CommitsHeld<'_>,
    #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    rdf: &grafeo_core::graph::rdf::RdfStore,
    manager: &TransactionManager,
) -> Result<()> {
    let op = StandaloneOp::RdfGraph(op);
    #[cfg(feature = "wal")]
    if let Some(wal) = wal {
        let mut change = StandaloneChange::new();
        change.push(op.clone());
        wal.log_batch(&crate::transaction::v1_group::standalone_group(&change)?)?;
    }
    if let Err(error) = apply_rdf(&op, rdf, &Applying::Live) {
        manager.poison(&format!("a logged change did not apply: {error}"));
        return Err(error);
    }
    Ok(())
}

/// Applies `op`, an RDF graph operation, to `rdf`: what a statement does
/// once it is logged, and what replay does with the logged op. A create of a
/// graph that exists, or a drop of a named graph that does not, applies
/// nothing: an error live (the statement's checks refuse it), skipped on
/// replay.
///
/// # Errors
///
/// Live: an op the statement's checks should have refused.
#[cfg(feature = "triple-store")]
pub(crate) fn apply_rdf(
    op: &StandaloneOp,
    rdf: &grafeo_core::graph::rdf::RdfStore,
    applying: &Applying<'_>,
) -> Result<()> {
    let StandaloneOp::RdfGraph(graph_op) = op else {
        return Err(not_applied(op, "it is no RDF graph operation"));
    };
    if rdf.apply_graph_op(graph_op) {
        return Ok(());
    }
    match graph_op {
        grafeo_common::change::RdfGraphOp::Create { .. } => {
            applying.missing(op, "the graph exists")
        }
        _ => applying.missing(op, "the graph does not exist"),
    }
}

/// Applies the put of `record` (see [`apply`]): the record replaces the
/// entry with its key, or adds it.
fn put(
    op: &StandaloneOp,
    record: &CatalogRecord,
    built: Option<BuiltIndex>,
    store: &Arc<LpgStore>,
    catalog: &Catalog,
    applying: &mut Applying<'_>,
) -> Result<()> {
    match record {
        CatalogRecord::Schema(schema) => {
            let outcome = catalog.register_schema_namespace(schema.name.clone());
            applying.catalog(op, outcome)
        }
        CatalogRecord::NodeType(record) => {
            catalog.register_or_replace_node_type(node_type_definition(record.clone()));
            Ok(())
        }
        CatalogRecord::EdgeType(record) => {
            catalog.register_or_replace_edge_type_def(edge_type_definition(record.clone())?);
            Ok(())
        }
        CatalogRecord::GraphType(record) => {
            let def = graph_type_definition(record.clone());
            // A graph type replaced keeps the graphs bound to it (`ALTER
            // GRAPH TYPE`); a new one drops the bindings a dropped graph type
            // of its name left (see `register_graph_type`).
            if catalog.get_graph_type_def(&def.name).is_some() {
                catalog.register_or_replace_graph_type(def);
                Ok(())
            } else {
                let outcome = catalog.register_graph_type(def);
                applying.catalog(op, outcome)
            }
        }
        CatalogRecord::GraphBinding(binding) => {
            let outcome = catalog.bind_graph_type(&binding.graph, binding.graph_type.clone());
            applying.catalog(op, outcome)
        }
        CatalogRecord::Constraint(record) => {
            // A constraint of the name there already (replay over an image
            // that holds it) is made again: its node type may have been
            // replaced since without the type constraints that enforce it.
            let def = constraint_definition(record.clone());
            if catalog.constraint(&def.name).is_some() {
                applying.catalog(op, catalog.drop_constraint(&def.name))?;
            }
            let outcome = catalog.create_constraint(def);
            applying.catalog(op, outcome)
        }
        CatalogRecord::Index(record) => put_index(op, record, built, store, applying),
        CatalogRecord::IndexName(record) => {
            if !has_index_name(catalog, record) {
                let name = index_name(record.clone());
                let label = catalog.get_or_create_label(&name.label);
                let property = catalog.get_or_create_property_key(&name.property);
                catalog.create_index(&name.name, label, property, name.index_type);
            }
            Ok(())
        }
        CatalogRecord::Procedure(record) => {
            let outcome = catalog.replace_procedure(procedure_definition(record.clone()));
            applying.catalog(op, outcome)
        }
    }
}

/// Whether the catalog has the index name `record` names, with its label,
/// property and kind: `CREATE INDEX` names one index per property, so a
/// name can come more than once, but each of these once.
fn has_index_name(catalog: &Catalog, record: &IndexNameRecord) -> bool {
    let wanted = index_name(record.clone());
    catalog.all_indexes().into_iter().any(|def| {
        def.name == wanted.name
            && def.index_type == wanted.index_type
            && catalog
                .get_label_name(def.label)
                .is_some_and(|label| *label == *wanted.label)
            && catalog
                .get_property_key_name(def.property_key)
                .is_some_and(|property| *property == *wanted.property)
    })
}

/// The store of the graph with storage key `graph` (`None` for the default
/// graph), or `None` when no such graph exists.
fn graph_store(store: &Arc<LpgStore>, graph: Option<&str>) -> Option<Arc<LpgStore>> {
    match graph {
        None => Some(Arc::clone(store)),
        Some(key) => store.graph(key),
    }
}

/// Applies the put of an index record (see [`apply`]): installs the index a
/// statement built, or a property index, which is built from the data at
/// once. Replay keeps a vector or text index for the build after the replay
/// instead, in place of an index of the same key the image held.
fn put_index(
    op: &StandaloneOp,
    record: &IndexRecord,
    built: Option<BuiltIndex>,
    store: &Arc<LpgStore>,
    applying: &mut Applying<'_>,
) -> Result<()> {
    let Some(target) = graph_store(store, record.graph.as_deref()) else {
        return applying.missing(op, "the graph does not exist");
    };
    if let IndexKindRecord::Property { key } = &record.index {
        target.create_property_index(key);
        return Ok(());
    }
    match (applying, built) {
        (Applying::Replay { unbuilt }, _) => {
            let key = index_key(&record.index);
            remove_index(&target, &key);
            forget_unbuilt(unbuilt, record.graph.as_deref(), &key);
            let mut indexes = graph_indexes(vec![record.clone()])?;
            match unbuilt.iter_mut().find(|graph| graph.graph == record.graph) {
                Some(graph) => {
                    let indexes = indexes.remove(0);
                    graph.vector.extend(indexes.vector);
                    graph.text.extend(indexes.text);
                }
                None => unbuilt.append(&mut indexes),
            }
            Ok(())
        }
        (Applying::Live, Some(index)) => install(&target, &record.index, index),
        (Applying::Live, None) => Err(not_applied(op, "the index was not built")),
    }
}

/// Installs `index`, built for `kind`, in `target`, in place of an index of
/// the same label and property.
///
/// # Errors
///
/// An index built for another kind of record than `kind`: a broken
/// invariant (a statement builds the index its record describes).
#[cfg_attr(
    not(any(feature = "vector-index", feature = "text-index")),
    expect(
        unused_variables,
        clippy::needless_pass_by_value,
        reason = "without vector and text indexes no index is ever built"
    )
)]
fn install(target: &LpgStore, kind: &IndexKindRecord, index: BuiltIndex) -> Result<()> {
    match index {
        #[cfg(feature = "vector-index")]
        BuiltIndex::Vector(index) => {
            let IndexKindRecord::Vector {
                label, property, ..
            } = kind
            else {
                return Err(Error::Internal(format!(
                    "a vector index built for {kind:?}"
                )));
            };
            target.add_vector_index(label, property, Arc::new(index));
            Ok(())
        }
        #[cfg(feature = "text-index")]
        BuiltIndex::Text(index) => {
            let IndexKindRecord::Text {
                label, property, ..
            } = kind
            else {
                return Err(Error::Internal(format!("a text index built for {kind:?}")));
            };
            target.add_text_index(label, property, Arc::new(parking_lot::RwLock::new(index)));
            Ok(())
        }
    }
}

/// What identifies the index of `kind`.
fn index_key(kind: &IndexKindRecord) -> IndexKeyRecord {
    match kind {
        IndexKindRecord::Property { key } => IndexKeyRecord::Property { key: key.clone() },
        IndexKindRecord::Vector {
            label, property, ..
        } => IndexKeyRecord::Vector {
            label: label.clone(),
            property: property.clone(),
        },
        IndexKindRecord::Text {
            label, property, ..
        } => IndexKeyRecord::Text {
            label: label.clone(),
            property: property.clone(),
        },
    }
}

/// Removes the index `key` names from `target`; whether there was one.
fn remove_index(target: &LpgStore, key: &IndexKeyRecord) -> bool {
    match key {
        IndexKeyRecord::Property { key } => target.drop_property_index(key),
        #[cfg(feature = "vector-index")]
        IndexKeyRecord::Vector { label, property } => target.remove_vector_index(label, property),
        #[cfg(feature = "text-index")]
        IndexKeyRecord::Text { label, property } => target.remove_text_index(label, property),
        #[cfg(not(all(feature = "vector-index", feature = "text-index")))]
        _ => false,
    }
}

/// Removes the vector or text index `key` names in the graph `graph` from
/// the indexes left to build.
fn forget_unbuilt(unbuilt: &mut Vec<GraphIndexes>, graph: Option<&str>, key: &IndexKeyRecord) {
    for indexes in unbuilt
        .iter_mut()
        .filter(|indexes| indexes.graph.as_deref() == graph)
    {
        match key {
            IndexKeyRecord::Property { .. } => {}
            IndexKeyRecord::Vector { label, property } => indexes
                .vector
                .retain(|def| (&def.label, &def.property) != (label, property)),
            IndexKeyRecord::Text { label, property } => indexes
                .text
                .retain(|def| (&def.label, &def.property) != (label, property)),
        }
    }
    unbuilt.retain(|indexes| !indexes.is_empty());
}

/// Applies the drop of the entry `key` names (see [`apply`]).
fn drop_entry(
    op: &StandaloneOp,
    key: &CatalogKey,
    store: &Arc<LpgStore>,
    catalog: &Catalog,
    applying: &mut Applying<'_>,
) -> Result<()> {
    match key {
        CatalogKey::Schema(name) => {
            let outcome = catalog.drop_schema_namespace(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::NodeType(name) => {
            let outcome = catalog.drop_node_type(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::EdgeType(name) => {
            let outcome = catalog.drop_edge_type_def(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::GraphType(name) => {
            let outcome = catalog.drop_graph_type(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::Constraint(name) => {
            let outcome = catalog.drop_constraint(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::Procedure(name) => {
            let outcome = catalog.drop_procedure(name);
            applying.catalog(op, outcome)
        }
        CatalogKey::Index { graph, index } => {
            if let Applying::Replay { unbuilt } = applying {
                forget_unbuilt(unbuilt, graph.as_deref(), index);
            }
            match graph_store(store, graph.as_deref()) {
                Some(target) if remove_index(&target, index) => Ok(()),
                Some(_) => applying.missing(op, "there is no such index"),
                None => applying.missing(op, "the graph does not exist"),
            }
        }
        CatalogKey::IndexName(name) => match catalog.find_index_by_name(name) {
            Some(id) => {
                catalog.drop_index(id);
                Ok(())
            }
            None => applying.missing(op, "there is no index of that name"),
        },
        // No statement drops a binding on its own: dropping a graph leaves
        // its binding, which binds nothing until a graph of its name exists.
        CatalogKey::GraphBinding(_) => Err(not_applied(
            op,
            "a graph type binding is not dropped on its own by this release",
        )),
    }
}

/// Replays the standalone op the WAL record `record` holds (a framed WAL v2
/// log record): applies it as the statement that logged it did (see
/// [`Applying::Replay`]), the vector and text indexes it puts kept in
/// `unbuilt`. `wal` names the WAL in errors.
///
/// # Errors
///
/// Returns an error naming the WAL when the record does not decode, holds
/// no standalone op, or holds one that does not apply; and when it puts a
/// vector or text index this build cannot build (it would be lost: see
/// `sections::FeatureData`).
#[cfg(feature = "wal")]
pub(crate) fn replay(
    record: &[u8],
    wal: &std::path::Path,
    store: &Arc<LpgStore>,
    #[cfg(feature = "triple-store")] rdf: &grafeo_core::graph::rdf::RdfStore,
    catalog: &Catalog,
    unbuilt: &mut Vec<GraphIndexes>,
) -> Result<()> {
    use grafeo_common::storage::{LogRecord, read_log_records};
    use grafeo_common::utils::error::StorageError;

    let failed = |reason: String| {
        Error::Storage(StorageError::RecoveryFailed(format!(
            "cannot replay a standalone change of the WAL {}: {reason}",
            wal.display()
        )))
    };
    let mut ops = Vec::new();
    read_log_records(record, &mut |record| {
        ops.push(record);
        Ok(())
    })
    .map_err(|error| failed(error.to_string()))?;
    for record in ops {
        let LogRecord::Standalone(op) = record else {
            return Err(failed(format!("{record:?} is no standalone change")));
        };
        refuse_unbuildable(&op, wal)?;
        let applied = if matches!(op, StandaloneOp::RdfGraph(_)) {
            #[cfg(feature = "triple-store")]
            {
                apply_rdf(&op, rdf, &Applying::Replay { unbuilt })
            }
            // This build cannot replay it: the next checkpoint would write
            // the file without its change, and remove the WAL.
            #[cfg(not(feature = "triple-store"))]
            {
                return Err(super::sections::refusal(
                    wal,
                    &[(&super::sections::RDF_TRIPLES, "WAL records".to_string())],
                ));
            }
        } else {
            apply(&op, None, store, catalog, &mut Applying::Replay { unbuilt })
        };
        applied.map_err(|error| failed(error.to_string()))?;
    }
    Ok(())
}

/// Refuses the put of a vector or text index this build cannot build: the
/// next checkpoint would write the catalog without it.
#[cfg(feature = "wal")]
fn refuse_unbuildable(op: &StandaloneOp, wal: &std::path::Path) -> Result<()> {
    let StandaloneOp::PutCatalog(CatalogRecord::Index(record)) = op else {
        return Ok(());
    };
    let data = match &record.index {
        IndexKindRecord::Vector { .. } if !super::sections::VECTOR_INDEXES.in_build() => {
            &super::sections::VECTOR_INDEXES
        }
        IndexKindRecord::Text { .. } if !super::sections::TEXT_INDEXES.in_build() => {
            &super::sections::TEXT_INDEXES
        }
        _ => return Ok(()),
    };
    Err(super::sections::refusal(
        wal,
        &[(
            data,
            format!(
                "WAL records: {}",
                record_name(&CatalogRecord::Index(record.clone()))
            ),
        )],
    ))
}
