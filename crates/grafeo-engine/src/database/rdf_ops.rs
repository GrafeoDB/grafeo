//! RDF-specific operations for GrafeoDB.
//!
//! This module consolidates all RDF functionality that was previously scattered
//! across `query.rs`, `crud.rs`, `admin.rs`, and `mod.rs`. The entire module
//! is gated behind `#[cfg(feature = "triple-store")]` in the parent.

use std::sync::Arc;

use grafeo_common::utils::error::Result;
use grafeo_core::graph::rdf::RdfStore;

use super::GrafeoDB;

// =========================================================================
// Query operations
// =========================================================================

impl GrafeoDB {
    /// Executes a SPARQL query and returns the result.
    ///
    /// SPARQL queries operate on the RDF triple store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<super::QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner::rdf::RdfPlanner, translators::sparql,
        };

        // Parse and translate the SPARQL query to a logical plan
        let logical_plan = sparql::translate(query)?;

        // Optimize the plan using RDF-specific statistics
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // EXPLAIN: return the physical plan tree without executing
        if optimized_plan.explain {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store));
            let (_, entries) = planner.plan_profiled(&optimized_plan)?;
            use crate::query::processor::physical_explain_result;
            return Ok(physical_explain_result(&optimized_plan, entries));
        }

        // No transaction here: each statement's WAL records are written as
        // one implicit group once it has run.
        #[cfg(feature = "wal")]
        let wal_buffer = self.wal.as_ref().map(|wal| {
            Arc::new(crate::transaction::wal_buffer::WalBuffer::new(Arc::clone(
                wal,
            )))
        });
        #[cfg(feature = "wal")]
        let flush_wal = || {
            if let Some(ref buffer) = wal_buffer
                && let Err(e) = buffer.flush_implicit()
            {
                grafeo_common::grafeo_warn!("Failed to write SPARQL update to WAL: {}", e);
            }
        };

        // EXPLAIN ANALYZE: execute with profiling, report actual stats
        if optimized_plan.profile {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store));
            #[cfg(feature = "wal")]
            let planner = planner.with_wal(wal_buffer.clone());
            let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

            let start = std::time::Instant::now();
            let executor = Executor::with_columns(physical_plan.columns.clone());
            let result = executor.execute(physical_plan.operator.as_mut());
            #[cfg(feature = "wal")]
            flush_wal();
            let _result = result?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let tree = crate::query::profile::build_profile_tree(
                &optimized_plan.root,
                &mut entries.into_iter(),
            );
            return Ok(crate::query::profile::profile_result(&tree, elapsed_ms));
        }

        // Convert to physical plan using RDF planner
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store));
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(wal_buffer.clone());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());
        #[cfg(feature = "wal")]
        flush_wal();
        result
    }

    /// Returns the RDF store.
    ///
    /// This provides direct access to the RDF store for triple operations.
    #[must_use]
    pub fn rdf_store(&self) -> &Arc<RdfStore> {
        &self.rdf_store
    }
}

// =========================================================================
// CRUD operations
// =========================================================================

impl GrafeoDB {
    /// Batch-inserts RDF triples into the RDF store.
    ///
    /// Delegates to `RdfStore::batch_insert`, which acquires each index lock
    /// once for the entire batch. Duplicates are silently skipped.
    ///
    /// Returns the number of triples that were newly inserted.
    pub fn batch_insert_rdf(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
    ) -> usize {
        self.rdf_store.batch_insert(triples)
    }
}

// =========================================================================
// Admin operations
// =========================================================================

impl GrafeoDB {
    /// Returns RDF schema information.
    ///
    /// Only available when the RDF feature is enabled.
    #[must_use]
    pub fn rdf_schema(&self) -> crate::admin::SchemaInfo {
        let stats = self.rdf_store.stats();

        let predicates = self
            .rdf_store
            .predicates()
            .into_iter()
            .map(|predicate| {
                let count = self.rdf_store.triples_with_predicate(&predicate).len();
                crate::admin::PredicateInfo {
                    iri: predicate.to_string(),
                    count,
                }
            })
            .collect();

        crate::admin::SchemaInfo::Rdf(crate::admin::RdfSchemaInfo {
            predicates,
            named_graphs: Vec::new(),
            subject_count: stats.subject_count,
            object_count: stats.object_count,
        })
    }
}

// =========================================================================
// SHACL validation
// =========================================================================

#[cfg(feature = "shacl")]
impl GrafeoDB {
    /// Validates the default graph against SHACL shapes in a named graph.
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or the shapes graph doesn't exist.
    pub fn validate_shacl(
        &self,
        shapes_graph: &str,
    ) -> grafeo_common::utils::error::Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        let session = self.session();
        session.validate_shacl(shapes_graph)
    }
}

// =========================================================================
// WAL replay helper
// =========================================================================

/// Replays a single RDF WAL record into the RDF store.
///
/// Returns `true` if the record was handled, `false` if it was not an RDF record.
#[cfg(feature = "wal")]
pub(super) fn replay_rdf_wal_record(
    rdf_store: &Arc<RdfStore>,
    record: &grafeo_storage::wal::WalRecord,
) {
    use grafeo_core::graph::rdf::Term;
    use grafeo_storage::wal::WalRecord;

    match record {
        WalRecord::InsertRdfTriple {
            subject,
            predicate,
            object,
            graph,
        } => {
            if let (Some(s), Some(p), Some(o)) = (
                Term::from_ntriples(subject),
                Term::from_ntriples(predicate),
                Term::from_ntriples(object),
            ) {
                let triple = grafeo_core::graph::rdf::Triple::new(s, p, o);
                let target = match graph {
                    Some(name) => rdf_store.graph_or_create(name),
                    None => Arc::clone(rdf_store),
                };
                target.insert(triple);
            }
        }
        WalRecord::DeleteRdfTriple {
            subject,
            predicate,
            object,
            graph,
        } => {
            if let (Some(s), Some(p), Some(o)) = (
                Term::from_ntriples(subject),
                Term::from_ntriples(predicate),
                Term::from_ntriples(object),
            ) {
                let triple = grafeo_core::graph::rdf::Triple::new(s, p, o);
                let target = match graph {
                    Some(name) => rdf_store.graph_or_create(name),
                    None => Arc::clone(rdf_store),
                };
                target.remove(&triple);
            }
        }
        WalRecord::ClearRdfGraph { graph } => {
            rdf_store.clear_graph(graph.as_deref());
        }
        WalRecord::CreateRdfGraph { name } => {
            let _ = rdf_store.create_graph(name);
        }
        WalRecord::DropRdfGraph { name } => match name {
            None => rdf_store.clear(),
            Some(graph_name) => {
                rdf_store.drop_graph(graph_name);
            }
        },
        _ => {}
    }
}
