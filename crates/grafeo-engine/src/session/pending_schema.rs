//! Schema statements in a transaction: they queue for its commit.
//!
//! A statement runs its checks as it does outside a transaction, then claims its names
//! and queues its ops in the transaction's change set. The session's view of the catalog
//! is a copy of its own from the first such statement on, with the queued ops applied, so
//! the transaction sees them and nobody else does. At the commit the ops are checked again
//! against the catalog as committed, the vector and text indexes built, and everything
//! logged as one group with the data; they apply once the data is stamped, under the
//! commit lock. A rollback drops the queue and the copy, and leaves nothing to undo.

use std::sync::Arc;

use grafeo_common::change::StandaloneOp;
use grafeo_common::storage::catalog_record::{CatalogRecord, IndexRecord};
use grafeo_common::types::{EpochId, ReadView, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};

use grafeo_core::graph::GraphStore;

use super::Session;
use crate::catalog::{Catalog, CatalogConstraintValidator, RuleScope, check_data_against_rules};
use crate::database::standalone::{
    Applying, apply, apply_to_catalog, claims_of, copy_of_catalog, refuse_drop_with_changes_in,
};
use crate::query::cache::QueryCache;
use crate::transaction::{BuiltIndex, QueuedSchemaOp, StandaloneChange};

/// The plans a transaction with its own catalog makes, kept from every other session.
const PRIVATE_PLANS: usize = 64;

/// Which named RDF graphs exist as a commit's queued operations apply in order, for a check
/// ahead of logging: one that does not apply once logged poisons the database.
#[cfg(feature = "triple-store")]
struct RdfGraphsAhead<'a> {
    store: &'a grafeo_core::graph::rdf::RdfStore,
    /// The graphs the operations so far made or took away.
    changed: std::collections::HashMap<String, bool>,
    /// Whether one took away every named graph there was.
    all_gone: bool,
}

#[cfg(feature = "triple-store")]
impl<'a> RdfGraphsAhead<'a> {
    fn new(store: &'a grafeo_core::graph::rdf::RdfStore) -> Self {
        Self {
            store,
            changed: std::collections::HashMap::new(),
            all_gone: false,
        }
    }

    fn exists(&self, name: &str) -> bool {
        self.changed
            .get(name)
            .copied()
            .unwrap_or_else(|| !self.all_gone && self.store.graph(name).is_some())
    }

    /// Whether `op` applies after those before it, as `RdfStore::apply_graph_op` would.
    fn applies(&mut self, op: &grafeo_common::change::RdfGraphOp) -> bool {
        use grafeo_common::change::RdfGraphOp;
        use grafeo_common::storage::log_record::RdfGraphTarget;
        match op {
            RdfGraphOp::Create { name } => {
                if self.exists(name) {
                    return false;
                }
                self.changed.insert(name.clone(), true);
            }
            RdfGraphOp::Drop {
                target: RdfGraphTarget::Named(name),
            } => {
                if !self.exists(name) {
                    return false;
                }
                self.changed.insert(name.clone(), false);
            }
            RdfGraphOp::Drop {
                target: RdfGraphTarget::All | RdfGraphTarget::AllNamed,
            }
            | RdfGraphOp::Clear {
                target: RdfGraphTarget::All | RdfGraphTarget::AllNamed,
            } => {
                self.changed.clear();
                self.all_gone = true;
            }
            RdfGraphOp::Drop { .. } | RdfGraphOp::Clear { .. } => {}
            RdfGraphOp::Copy { source, target }
            | RdfGraphOp::Move { source, target }
            | RdfGraphOp::Add { source, target } => {
                if source != target {
                    if matches!(op, RdfGraphOp::Move { .. })
                        && let Some(source) = source
                    {
                        self.changed.insert(source.clone(), false);
                    }
                    if let Some(target) = target {
                        self.changed.insert(target.clone(), true);
                    }
                }
            }
        }
        true
    }
}

/// A transaction's own view of the catalog.
pub(super) struct PendingSchema {
    /// The catalog at its first schema statement: a savepoint rollback reapplies the
    /// queue on this, so its view does not move.
    base: Arc<Catalog>,
    /// The base with the queue applied.
    overlay: Arc<Catalog>,
    /// The plans it makes, which read the overlay.
    plans: Arc<QueryCache>,
}

/// The property indexes that serve a label and a key in a graph.
#[cfg(feature = "gql")]
#[derive(Clone, Copy)]
pub(super) struct PropertyIndexesOn {
    /// The index on the key, on every label.
    pub every_label: bool,
    /// The index on the key of this label.
    pub this_label: bool,
}

#[cfg(feature = "gql")]
impl PropertyIndexesOn {
    /// Whether either is there.
    pub fn any(self) -> bool {
        self.every_label || self.this_label
    }
}

impl Session {
    /// The catalog this session's statements read: the open transaction's own view, when
    /// it ran a schema statement.
    pub(super) fn effective_catalog(&self) -> Arc<Catalog> {
        match &*self.pending_schema.lock() {
            Some(pending) => Arc::clone(&pending.overlay),
            None => Arc::clone(&self.catalog),
        }
    }

    /// The cache of this session's plans: one of its own while the open transaction has
    /// its own view, whose plans no other session may run, nor it theirs.
    pub(super) fn plan_cache(&self) -> Arc<QueryCache> {
        match &*self.pending_schema.lock() {
            Some(pending) => Arc::clone(&pending.plans),
            None => Arc::clone(&self.query_cache),
        }
    }

    /// Queues `change`, whose checks passed, for the commit of the open transaction.
    ///
    /// # Errors
    ///
    /// A write conflict (`GRAFEO-T001`) when another open transaction claimed one of its
    /// names: the statement leaves nothing.
    pub(super) fn queue_schema_change(&self, change: StandaloneChange) -> Result<()> {
        let (Some(transaction), Some(changes)) = (
            *self.current_transaction.lock(),
            self.changes.lock().clone(),
        ) else {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "a schema statement needs the transaction's change set".to_string(),
            )));
        };
        let mark = self.transaction_manager.mark_claims(transaction)?;
        for claim in change.ops().flat_map(claims_of) {
            if let Err(error) = self.transaction_manager.record_key(transaction, claim) {
                self.transaction_manager.restore_claims(transaction, mark)?;
                return Err(error);
            }
        }
        let mut pending = self.pending_schema.lock();
        let overlay = match &*pending {
            Some(pending) => Arc::clone(&pending.overlay),
            None => {
                let base = Arc::new(copy_of_catalog(&self.catalog)?);
                let overlay = Arc::new(copy_of_catalog(&base)?);
                *pending = Some(PendingSchema {
                    base,
                    overlay: Arc::clone(&overlay),
                    plans: Arc::new(QueryCache::new(PRIVATE_PLANS)),
                });
                overlay
            }
        };
        drop(pending);
        // What a statement built reads no write of the transaction: the commit builds it
        // again. Built here, it told the statement's errors early.
        for (op, _) in change.into_ops() {
            apply_to_catalog(&op, &overlay)?;
            let graph = match (&op, self.root_store()) {
                (StandaloneOp::CreateGraph { .. }, Some(root)) => {
                    Some(root.new_unnamed_graph().map_err(|error| {
                        Error::Internal(format!("the graph could not be made: {error}"))
                    })?)
                }
                _ => None,
            };
            changes.queue_schema(op, graph);
        }
        self.refresh_pending_graphs(&changes);
        self.plan_cache().clear();
        Ok(())
    }

    /// The graphs the queue created and dropped, for the transaction's own statements.
    fn refresh_pending_graphs(&self, changes: &crate::transaction::TransactionChanges) {
        let mut graphs = self.stores.pending_graphs.write();
        graphs.created.clear();
        graphs.dropped.clear();
        graphs.made.clear();
        changes.read_schema(|queue| {
            for queued in queue {
                match (&queued.op, &queued.graph) {
                    (StandaloneOp::CreateGraph { name }, Some(graph)) => {
                        graphs.created.insert(name.clone(), Arc::clone(graph));
                        graphs.made.insert(name.clone());
                    }
                    (StandaloneOp::DropGraph { name }, _) => {
                        graphs.created.remove(name);
                        graphs.dropped.insert(name.clone());
                    }
                    _ => {}
                }
            }
        });
    }

    /// Back to the first `len` queued ops, at a savepoint rollback: the view is rebuilt
    /// on the base, not on what others committed since.
    pub(super) fn schema_back_to(&self, len: usize) -> Result<()> {
        let Some(changes) = self.changes.lock().clone() else {
            return Ok(());
        };
        if changes.schema_len() == len {
            return Ok(());
        }
        let current = self.current_graph_created_here();
        changes.truncate_schema(len);
        let mut pending = self.pending_schema.lock();
        if let Some(pending) = &mut *pending {
            let overlay = copy_of_catalog(&pending.base)?;
            changes.read_schema(|queue| {
                queue
                    .iter()
                    .try_for_each(|queued| apply_to_catalog(&queued.op, &overlay))
            })?;
            pending.overlay = Arc::new(overlay);
            pending.plans = Arc::new(QueryCache::new(PRIVATE_PLANS));
        }
        drop(pending);
        self.refresh_pending_graphs(&changes);
        self.leave_graph_if_gone(current);
        Ok(())
    }

    /// Whether the session is in a graph the transaction created, also one it has dropped
    /// since, by its name.
    fn current_graph_created_here(&self) -> Option<String> {
        let current = self.current_graph.lock().clone()?;
        let key = self.effective_graph_key(&current);
        self.stores
            .pending_graphs
            .read()
            .made
            .contains(&key)
            .then_some(current)
    }

    /// Leaves `created`, a graph the transaction made, when an undo took it away.
    fn leave_graph_if_gone(&self, created: Option<String>) {
        if let Some(name) = created
            && self.named_graph(&self.effective_graph_key(&name)).is_none()
        {
            *self.current_graph.lock() = None;
        }
    }

    /// Drops the view and the graphs of a transaction that ended without its commit.
    pub(super) fn forget_pending_schema(&self) {
        let current = self.current_graph_created_here();
        *self.pending_schema.lock() = None;
        {
            let mut graphs = self.stores.pending_graphs.write();
            graphs.created.clear();
            graphs.dropped.clear();
            graphs.made.clear();
        }
        self.leave_graph_if_gone(current);
    }

    /// Checks the queue against what is committed, and builds its vector and text
    /// indexes, under the commit lock: the indexes the ops put, by position.
    ///
    /// # Errors
    ///
    /// A write conflict (`GRAFEO-T001`) for an op that no longer applies, or a graph
    /// another transaction writes; the error of a build. Nothing is logged then.
    pub(super) fn ready_schema_ops(
        &self,
        transaction: TransactionId,
        queue: &[QueuedSchemaOp],
    ) -> Result<Vec<Option<BuiltIndex>>> {
        if queue.is_empty() {
            return Ok(Vec::new());
        }
        let conflict = |what: String| {
            Error::Transaction(TransactionError::WriteConflict(format!(
                "{what}: another commit changed it since this transaction's statement"
            )))
        };
        let check = copy_of_catalog(&self.catalog)?;
        let root = self.root_store();
        let open = self.transaction_manager.open_change_sets_while_frozen();
        let mut made = std::collections::HashSet::new();
        let view = ReadView::transaction(transaction, EpochId::LATEST);
        let mut built = Vec::with_capacity(queue.len());
        #[cfg(feature = "triple-store")]
        let mut rdf_graphs = RdfGraphsAhead::new(&self.rdf_store);
        for queued in queue {
            apply_to_catalog(&queued.op, &check).map_err(|error| conflict(error.to_string()))?;
            let mut index = None;
            match &queued.op {
                StandaloneOp::CreateGraph { name } => {
                    if root.as_ref().is_some_and(|root| root.graph(name).is_some())
                        && !queue_dropped_before(queue, queued, name)
                    {
                        return Err(conflict(format!("graph '{name}'")));
                    }
                    made.insert(name.clone());
                }
                StandaloneOp::DropGraph { name } => {
                    if !made.remove(name) {
                        if root.as_ref().is_some_and(|root| root.graph(name).is_none()) {
                            return Err(conflict(format!("graph '{name}'")));
                        }
                        refuse_drop_with_changes_in(&open, name)?;
                    }
                }
                #[cfg(feature = "triple-store")]
                StandaloneOp::RdfGraph(op) => {
                    crate::transaction::refuse_graph_op_with_changes_in(&open, op)?;
                    if !rdf_graphs.applies(op) {
                        return Err(conflict(format!("the RDF graph of {op:?}")));
                    }
                }
                StandaloneOp::PutCatalog(CatalogRecord::Index(record)) => {
                    index = self.build_for_commit(record, view)?;
                }
                _ => {}
            }
            built.push(index);
        }
        // Again under the commit lock: what others committed since the statement counts.
        let ops: Vec<&StandaloneOp> = queue.iter().map(|queued| &queued.op).collect();
        self.check_change_over_data(&ops, view, &self.catalog)?;
        Ok(built)
    }

    /// What a statement of this session reads the data at: its transaction's view, or
    /// every commit so far.
    pub(super) fn statement_view(&self) -> ReadView {
        match (*self.current_transaction.lock(), &*self.changes.lock()) {
            (Some(transaction), Some(changes)) => {
                ReadView::transaction(transaction, changes.snapshot())
            }
            _ => ReadView::latest(),
        }
    }

    /// The graphs a uniqueness rule on `label` and `key`, which `change` adds, reaches with no
    /// index on `key`: by storage key (`None` for the default graph). Where a write of the
    /// label would meet the rule, as the check over the data reads it. An index is there as
    /// the transaction's queue leaves it, in order: one it creates counts, one it drops does
    /// not.
    ///
    /// # Errors
    ///
    /// A record of the change that does not apply to a copy of the catalog: damage.
    #[cfg(feature = "gql")]
    pub(super) fn graphs_a_rule_needs_an_index_in(
        &self,
        change: &StandaloneChange,
        label: &str,
        key: &str,
    ) -> Result<Vec<Option<String>>> {
        let what_if = Arc::new(what_if_catalog(change.ops(), &self.effective_catalog())?);
        Ok(self
            .reachable_graphs()
            .into_iter()
            .filter(|(graph, store)| {
                !self.property_indexes_on(graph, &**store, label, key).any()
                    && validator_of(&what_if, graph.as_deref())
                        .node_type(label)
                        .is_some()
            })
            .map(|(graph, _)| graph)
            .collect())
    }

    /// The property indexes that serve `label` and `key` in `store`, the graph `graph`, as the
    /// transaction's queue leaves them, in order: one it creates counts, one it drops does not.
    #[cfg(feature = "gql")]
    pub(super) fn property_indexes_on(
        &self,
        graph: &Option<String>,
        store: &dyn GraphStore,
        label: &str,
        key: &str,
    ) -> PropertyIndexesOn {
        use grafeo_common::storage::catalog_record::{CatalogKey, IndexKeyRecord, IndexKindRecord};

        let mut there = PropertyIndexesOn {
            every_label: store.index_serves(&[], key),
            this_label: store.has_label_index(label, key),
        };
        let Some(changes) = self.changes.lock().clone() else {
            return there;
        };
        changes.read_schema(|queue| {
            for queued in queue {
                match &queued.op {
                    StandaloneOp::PutCatalog(CatalogRecord::Index(record))
                        if record.graph == *graph
                            && matches!(&record.index, IndexKindRecord::Property { key: on } if on == key) =>
                    {
                        there.every_label = true;
                    }
                    StandaloneOp::DropCatalog(CatalogKey::Index {
                        graph: on_graph,
                        index: IndexKeyRecord::Property { key: on },
                    }) if on_graph == graph && on == key => there.every_label = false,
                    StandaloneOp::PutCatalog(CatalogRecord::LabelIndex(record))
                        if record.graph == *graph
                            && record.label == label
                            && record.key == key =>
                    {
                        there.this_label = true;
                    }
                    StandaloneOp::DropCatalog(CatalogKey::LabelIndex {
                        graph: on_graph,
                        label: on_label,
                        key: on,
                    }) if on_graph == graph && on_label == label && on == key => {
                        there.this_label = false;
                    }
                    _ => {}
                }
            }
        });
        there
    }

    /// Every graph a statement of this session can reach, by storage key (`None` for the
    /// default graph).
    fn reachable_graphs(&self) -> Vec<(Option<String>, Arc<dyn GraphStore>)> {
        let mut graphs: Vec<(Option<String>, Arc<dyn GraphStore>)> =
            vec![(None, Arc::clone(&self.stores.graph_store))];
        for name in self
            .root_store()
            .map(|root| root.graph_names())
            .unwrap_or_default()
        {
            if let Some(graph) = self.named_graph(&name) {
                graphs.push((Some(name), graph as Arc<dyn GraphStore>));
            }
        }
        let pending = self.stores.pending_graphs.read();
        for (name, graph) in &pending.created {
            if !graphs.iter().any(|(key, _)| key.as_ref() == Some(name)) {
                graphs.push((Some(name.clone()), Arc::clone(graph) as Arc<dyn GraphStore>));
            }
        }
        graphs
    }

    /// Fails when the data, as `view` sees it, breaks a rule `ops` add to `base`: a named
    /// constraint, a node or edge type, or a graph type a graph has. The one check for
    /// every statement that adds a rule, and for the commit of one in a transaction.
    ///
    /// # Errors
    ///
    /// `GRAFEO-V001`, naming the rule and a value; a graph type's with GQLSTATUS G2000.
    pub(super) fn check_change_over_data(
        &self,
        ops: &[&StandaloneOp],
        view: ReadView,
        base: &Catalog,
    ) -> Result<()> {
        use crate::database::catalog_records::{
            constraint_definition, edge_type_definition, node_type_definition,
        };

        let mut scope = RuleScope::default();
        // Every type the statement changes, also to one with nothing left to check by itself: a
        // closed graph type that lists it may then no longer allow what a graph holds.
        let mut changed_labels = Vec::new();
        let mut changed_edge_types = Vec::new();
        let mut constraints = Vec::new();
        let mut graph_types = Vec::new();
        let mut bound_graphs = Vec::new();
        let mut dropped_graphs = Vec::new();
        let mut creates_a_graph = false;
        for op in ops {
            match op {
                StandaloneOp::DropGraph { name } => dropped_graphs.push(name),
                StandaloneOp::CreateGraph { .. } => creates_a_graph = true,
                _ => {}
            }
            let StandaloneOp::PutCatalog(record) = op else {
                continue;
            };
            match record {
                CatalogRecord::Constraint(record) => {
                    constraints.push(constraint_definition(record.clone()));
                }
                CatalogRecord::NodeType(record) => {
                    let def = node_type_definition(record.clone())?;
                    changed_labels.push(short_name(&def.name).to_string());
                    if !def.properties.is_empty() || !def.constraints.is_empty() {
                        scope.labels.push(short_name(&def.name).to_string());
                    }
                }
                CatalogRecord::EdgeType(record) => {
                    let def = edge_type_definition(record.clone())?;
                    changed_edge_types.push(short_name(&def.name).to_string());
                    if !def.properties.is_empty()
                        || !def.source_node_types.is_empty()
                        || !def.target_node_types.is_empty()
                    {
                        scope.edge_types.push(short_name(&def.name).to_string());
                    }
                }
                CatalogRecord::GraphType(record) => graph_types.push(record.name.clone()),
                CatalogRecord::GraphBinding(record) => bound_graphs.push(record.graph.clone()),
                _ => {}
            }
        }
        // Types declared with a graph or a graph type are theirs (`CREATE PROPERTY GRAPH`,
        // `CREATE GRAPH TYPE`): a new graph starts empty and a new type binds none, so what
        // other graphs hold of their labels is not theirs to judge. A graph that has the
        // type is checked whole below.
        if creates_a_graph || !graph_types.is_empty() {
            scope.labels.clear();
            scope.edge_types.clear();
            changed_labels.clear();
            changed_edge_types.clear();
        }
        let graphs: Vec<_> = self
            .reachable_graphs()
            .into_iter()
            .filter(|(key, _)| {
                key.as_ref()
                    .is_none_or(|key| !dropped_graphs.contains(&key))
            })
            .collect();
        if constraints.is_empty()
            && changed_labels.is_empty()
            && changed_edge_types.is_empty()
            && graph_types.is_empty()
            && bound_graphs.is_empty()
        {
            return Ok(());
        }
        let what_if = Arc::new(what_if_catalog(ops.iter().copied(), base)?);
        for def in &constraints {
            for (key, store) in &graphs {
                // Only where a write of the label would meet it.
                if validator_of(&what_if, key.as_deref())
                    .node_type(&def.label)
                    .is_some()
                {
                    crate::catalog::check_rule_over_nodes(def, &**store, view)
                        .map_err(crate::query::executor::convert_operator_error)?;
                }
            }
        }
        let bindings = what_if.all_graph_type_bindings();
        for (key, store) in graphs {
            let whole_graph = key.as_ref().is_some_and(|key| {
                bound_graphs.contains(key)
                    || bindings
                        .iter()
                        .any(|(graph, bound)| graph == key && graph_types.contains(bound))
            });
            let validator = validator_of(&what_if, key.as_deref());
            let mut scope = RuleScope {
                whole_graph,
                ..scope.clone()
            };
            // On this graph, a changed type its closed type lists is checked as a write of its
            // nodes or edges would be: only those are read.
            for label in &changed_labels {
                if validator.closed_type_lists_node(label) {
                    scope.in_its_type = true;
                    if !scope.labels.contains(label) {
                        scope.labels.push(label.clone());
                    }
                }
            }
            for edge_type in &changed_edge_types {
                if validator.closed_type_lists_edge(edge_type) {
                    scope.in_its_type = true;
                    if !scope.edge_types.contains(edge_type) {
                        scope.edge_types.push(edge_type.clone());
                    }
                }
            }
            if scope.labels.is_empty() && scope.edge_types.is_empty() && !whole_graph {
                continue;
            }
            if let Some(key) = &key {
                refuse_emptied_type(key, &*store, base, &what_if, view)?;
            }
            check_data_against_rules(&validator, &*store, &scope, view)
                .map_err(crate::query::executor::convert_operator_error)?;
        }
        Ok(())
    }

    /// Fails when `source`, the data of a `CREATE GRAPH ... AS COPY OF`, breaks the graph
    /// type `change` gives the new graph `key`.
    ///
    /// # Errors
    ///
    /// `GRAFEO-V001` with GQLSTATUS G2000 (ISO/IEC 39075 12.4).
    #[cfg(feature = "gql")]
    pub(super) fn check_copy_against_its_type(
        &self,
        change: &StandaloneChange,
        source: &dyn GraphStore,
        key: &str,
    ) -> Result<()> {
        let typed = change.ops().any(|op| {
            matches!(op, StandaloneOp::PutCatalog(CatalogRecord::GraphBinding(record)) if record.graph == key)
        });
        if !typed {
            return Ok(());
        }
        let what_if = Arc::new(what_if_catalog(change.ops(), &self.effective_catalog())?);
        let scope = RuleScope {
            whole_graph: true,
            ..RuleScope::default()
        };
        check_data_against_rules(
            &validator_of(&what_if, Some(key)),
            source,
            &scope,
            ReadView::latest(),
        )
        .map_err(crate::query::executor::convert_operator_error)
    }

    /// Checks what the transaction wrote against the rules, as it stands, when they
    /// changed since it began. One integer compare for every other commit.
    ///
    /// # Errors
    ///
    /// `GRAFEO-V001`, what a write of it would meet now.
    pub(super) fn check_writes_against_new_rules(
        &self,
        transaction: TransactionId,
        changes: Option<&crate::transaction::TransactionChanges>,
    ) -> Result<()> {
        let began = self
            .rules_at_begin
            .load(std::sync::atomic::Ordering::Relaxed);
        let Some(changes) = changes.filter(|_| self.catalog.rules_generation() != began) else {
            return Ok(());
        };
        let view = ReadView::transaction(transaction, EpochId::LATEST);
        let error = crate::query::executor::convert_operator_error;
        changes.read(|set| {
            for written in set.written_entities() {
                let store = self.stores.store_for_key(written.graph);
                let validator = validator_of(&self.catalog, written.graph)
                    .with_store(Arc::clone(&store))
                    .with_transaction_context(EpochId::LATEST, Some(transaction));
                for id in written.nodes {
                    if let Some(node) = store.node(id, view)? {
                        validator.check_node_as_it_stands(&node).map_err(error)?;
                    }
                }
                for id in written.edges {
                    if let Some(edge) = store.edge(id, view)? {
                        validator
                            .check_edge_as_it_stands(&*store, &edge, true, view)
                            .map_err(error)?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Refuses a SPARQL statement after a queued SPARQL graph operation: the RDF store
    /// holds only what is committed, so its reads would still see a graph the transaction
    /// dropped, and its triples apply ahead of the operation. Until reads of RDF are
    /// isolated by triple (#718).
    ///
    /// # Errors
    ///
    /// An invalid transaction state. The transaction goes on.
    #[cfg(feature = "triple-store")]
    pub(super) fn refuse_sparql_after_graph_op(&self) -> Result<()> {
        let queued = self.changes.lock().as_ref().is_some_and(|changes| {
            changes.read_schema(|queue| {
                queue
                    .iter()
                    .any(|queued| matches!(queued.op, StandaloneOp::RdfGraph(_)))
            })
        });
        if queued {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "this transaction ran a SPARQL graph operation, which applies at its commit: \
                 no SPARQL statement can run after it until then"
                    .to_string(),
            )));
        }
        Ok(())
    }

    /// Fails when another open transaction claimed the name a statement creates: the
    /// conflict, not what that transaction's view hides from this one.
    ///
    /// # Errors
    ///
    /// A write conflict (`GRAFEO-T001`).
    #[cfg(feature = "gql")]
    pub(super) fn refuse_name_claimed(&self, kind: &str, name: &str) -> Result<()> {
        let own = *self.current_transaction.lock();
        self.transaction_manager
            .refuse_claimed(&crate::database::standalone::name_claim(kind, name), own)
    }

    /// [`refuse_name_claimed`](Self::refuse_name_claimed) of the name `cmd` creates.
    ///
    /// # Errors
    ///
    /// A write conflict (`GRAFEO-T001`).
    #[cfg(feature = "gql")]
    pub(super) fn refuse_claimed_name(
        &self,
        cmd: &grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> Result<()> {
        use grafeo_adapters::query::gql::ast::SchemaStatement;

        match cmd {
            SchemaStatement::CreateNodeType(stmt) => {
                self.refuse_name_claimed("node type", &self.effective_type_key(&stmt.name))
            }
            SchemaStatement::CreateEdgeType(stmt) => {
                self.refuse_name_claimed("edge type", &self.effective_type_key(&stmt.name))
            }
            SchemaStatement::CreateGraphType(stmt) => {
                self.refuse_name_claimed("graph type", &self.effective_type_key(&stmt.name))
            }
            SchemaStatement::CreateIndex(stmt) => {
                self.refuse_name_claimed("index name", &stmt.name)
            }
            SchemaStatement::CreateConstraint(stmt) => stmt
                .name
                .as_deref()
                .map_or(Ok(()), |name| self.refuse_name_claimed("constraint", name)),
            SchemaStatement::CreateSchema { name, .. } => self.refuse_name_claimed("schema", name),
            SchemaStatement::CreateProcedure(stmt) => {
                self.refuse_name_claimed("procedure", &stmt.name)
            }
            _ => Ok(()),
        }
    }

    /// Takes out of `indexes` those the transaction's own statements dropped: they stand
    /// in the store until the commit.
    #[cfg(feature = "gql")]
    pub(super) fn hide_indexes_dropped_here(
        &self,
        indexes: &mut Vec<crate::database::index::ListedIndex>,
    ) {
        use grafeo_common::storage::catalog_record::{CatalogKey, IndexKeyRecord};

        let Some(changes) = self.changes.lock().clone() else {
            return;
        };
        changes.read_schema(|queue| {
            for queued in queue {
                if let StandaloneOp::DropCatalog(CatalogKey::LabelIndex { label, key, .. }) =
                    &queued.op
                {
                    indexes.retain(|listed| {
                        listed.name.is_some()
                            || listed.property != *key
                            || listed.label.as_deref() != Some(label.as_str())
                    });
                    continue;
                }
                let StandaloneOp::DropCatalog(CatalogKey::Index { index, .. }) = &queued.op else {
                    continue;
                };
                indexes.retain(|listed| {
                    listed.name.is_some()
                        || match index {
                            IndexKeyRecord::Property { key } => {
                                listed.property != *key || listed.label.is_some()
                            }
                            IndexKeyRecord::Vector { label, property }
                            | IndexKeyRecord::Text { label, property } => {
                                listed.property != *property
                                    || listed.label.as_deref() != Some(label.as_str())
                            }
                        }
                });
            }
        });
    }

    /// The vector or text index `record` puts, read at `view`.
    fn build_for_commit(&self, record: &IndexRecord, view: ReadView) -> Result<Option<BuiltIndex>> {
        match self.stores.lpg_store(record.graph.as_deref()) {
            Some(store) => crate::database::index::built_for_commit(&*store, record, view),
            None => Ok(None),
        }
    }

    /// Applies the queue: its log group is written and its data stamped.
    ///
    /// # Errors
    ///
    /// An op that does not apply though it was checked: the commit stays unpublished,
    /// which poisons the database until a reopen replays the group.
    pub(super) fn apply_schema_ops(
        &self,
        queue: &[QueuedSchemaOp],
        built: Vec<Option<BuiltIndex>>,
    ) -> Result<()> {
        for (queued, index) in queue.iter().zip(built) {
            match (&queued.op, &queued.graph, self.root_store()) {
                (StandaloneOp::CreateGraph { name }, Some(graph), Some(root)) => {
                    root.install_graph(name, Arc::clone(graph));
                }
                #[cfg(feature = "triple-store")]
                (op @ StandaloneOp::RdfGraph(_), _, _) => {
                    // A repeat is no error: it was checked at its statement, and an
                    // operation outside a transaction may have got there first.
                    let mut unbuilt = Vec::new();
                    crate::database::standalone::apply_rdf(
                        op,
                        &self.rdf_store,
                        &Applying::Replay {
                            unbuilt: &mut unbuilt,
                        },
                    )?;
                }
                (op, _, root) => {
                    apply(op, index, root.as_ref(), &self.catalog, &mut Applying::Live)?;
                }
            }
        }
        Ok(())
    }

    /// What the session does once the queue applied: other sessions plan anew, and it
    /// leaves a graph or a schema that is gone.
    pub(super) fn schema_committed(&self, queue: &[QueuedSchemaOp]) {
        {
            let mut graphs = self.stores.pending_graphs.write();
            graphs.created.clear();
            graphs.dropped.clear();
            graphs.made.clear();
        }
        if queue.is_empty() {
            return;
        }
        for queued in queue {
            match &queued.op {
                StandaloneOp::DropGraph { name } => {
                    let mut current = self.current_graph.lock();
                    if current
                        .as_deref()
                        .is_some_and(|graph| graph.eq_ignore_ascii_case(name))
                    {
                        *current = None;
                    }
                }
                StandaloneOp::DropCatalog(
                    grafeo_common::storage::catalog_record::CatalogKey::Schema(name),
                ) => {
                    let mut current = self.current_schema.lock();
                    if current
                        .as_deref()
                        .is_some_and(|schema| schema.eq_ignore_ascii_case(name))
                    {
                        *current = None;
                    }
                }
                _ => {}
            }
        }
        self.query_cache.clear();
    }
}

/// Fails when the change to `what_if` empties a side of the closed graph type the graph
/// `key` has, and the graph holds elements on that side: ISO/IEC 39075 4.13.2.1 has an empty
/// closed type allow none, where the write checks read an empty list as open.
fn refuse_emptied_type(
    key: &str,
    store: &dyn GraphStore,
    base: &Catalog,
    what_if: &Catalog,
    view: ReadView,
) -> Result<()> {
    let Some(name) = what_if.get_graph_type_binding(key) else {
        return Ok(());
    };
    let (Some(before), Some(after)) = (
        base.get_graph_type_def(&name),
        what_if.get_graph_type_def(&name),
    ) else {
        return Ok(());
    };
    if after.open {
        return Ok(());
    }
    let refuse = |side: &str| {
        crate::query::executor::convert_operator_error(
            grafeo_core::execution::operators::OperatorError::schema_violation(format!(
                "graph type '{name}' would allow no {side}, and graph '{key}' holds some"
            )),
        )
    };
    if after.allowed_node_types.is_empty()
        && !before.allowed_node_types.is_empty()
        && store.node_count(view)? > 0
    {
        return Err(refuse("nodes"));
    }
    if after.allowed_edge_types.is_empty()
        && !before.allowed_edge_types.is_empty()
        && store.edge_count(view)? > 0
    {
        return Err(refuse("edges"));
    }
    Ok(())
}

/// The name of a type without its schema.
fn short_name(name: &str) -> &str {
    name.rsplit_once('/').map_or(name, |(_, short)| short)
}

/// A copy of `base` with `ops` applied, to check data against before they apply.
fn what_if_catalog<'a>(
    ops: impl Iterator<Item = &'a StandaloneOp>,
    base: &Catalog,
) -> Result<Catalog> {
    let what_if = copy_of_catalog(base)?;
    for op in ops {
        apply_to_catalog(op, &what_if)?;
    }
    Ok(what_if)
}

/// The checks a write to the graph with storage key `key` goes through under `catalog`,
/// without what reads the store: uniqueness takes a pass of its own.
fn validator_of(catalog: &Arc<Catalog>, key: Option<&str>) -> CatalogConstraintValidator {
    CatalogConstraintValidator::of_graph(catalog, key)
}

/// Whether the queue drops graph `name` before `queued`, so a graph of that name may
/// stand until then.
fn queue_dropped_before(queue: &[QueuedSchemaOp], queued: &QueuedSchemaOp, name: &str) -> bool {
    queue
        .iter()
        .take_while(|other| !std::ptr::eq(*other, queued))
        .any(|other| matches!(&other.op, StandaloneOp::DropGraph { name: dropped } if dropped == name))
}
