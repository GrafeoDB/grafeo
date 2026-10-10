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
    /// Returns an error if the query fails; for an update, also on a
    /// read-only database, after `close()` of a persistent database, and after
    /// a commit that did not complete.
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

        // An update on a read-only database fails, as in a session.
        let mutates = optimized_plan.root.has_mutations();
        if mutates && self.read_only {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::ReadOnly,
            ));
        }

        // EXPLAIN: return the physical plan tree without executing
        if optimized_plan.explain {
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
                .with_shuffle_unordered(self.config.shuffle_unordered);
            let (_, entries) = planner.plan_profiled(&optimized_plan)?;
            use crate::query::processor::physical_explain_result;
            return Ok(physical_explain_result(&optimized_plan, entries));
        }

        // No transaction here: an update changes the store at once and its
        // WAL records are written as one implicit group once it has run, so
        // it holds commits off meanwhile (a checkpoint or `close()` sees all
        // of it or none), and fails once the database is closed.
        let held = if mutates {
            Some(self.transaction_manager.hold_commits_for_change()?)
        } else {
            None
        };
        // Tests start a checkpoint or `close()` here, which must wait.
        #[cfg(feature = "testing-statement-injection")]
        if held.is_some() {
            grafeo_common::testing::commit_hook::run_during_held_change();
        }
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
            let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
                .with_shuffle_unordered(self.config.shuffle_unordered);
            #[cfg(feature = "wal")]
            let planner = planner.with_wal(wal_buffer.clone());
            let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

            let start = std::time::Instant::now();
            let executor = Executor::with_columns(physical_plan.columns.clone());
            let result = executor.execute(physical_plan.operator.as_mut());
            #[cfg(feature = "wal")]
            flush_wal();
            drop(held);
            let _result = result?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let tree = crate::query::profile::build_profile_tree(
                &optimized_plan.root,
                &mut entries.into_iter(),
            );
            return Ok(crate::query::profile::profile_result(&tree, elapsed_ms));
        }

        // Convert to physical plan using RDF planner
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_shuffle_unordered(self.config.shuffle_unordered);
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(wal_buffer.clone());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());
        #[cfg(feature = "wal")]
        flush_wal();
        // The update and its WAL group are written: commits may go on.
        drop(held);
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
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; `triples` is collected before, without blocking anything,
    /// and only once the database takes the insert: a refused call never
    /// pulls the iterator.
    ///
    /// # Errors
    ///
    /// Returns the read-only error on a read-only database, the
    /// database-closed error after `close()` of a persistent database
    /// (read-only or not), and the incomplete-commit error after a commit
    /// that did not complete.
    pub fn batch_insert_rdf(
        &self,
        triples: impl IntoIterator<Item = grafeo_core::graph::rdf::Triple>,
    ) -> Result<usize> {
        // Refused before the iterator is pulled: it may parse or compute the
        // triples, which a refused call should not wait for.
        self.check_import_allowed()?;
        // Collected before commits are held off, for the same reason.
        let triples: Vec<_> = triples.into_iter().collect();
        let _held = self.hold_commits_for_import()?;
        Ok(self.rdf_store.batch_insert(triples))
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

/// Replays a single RDF WAL record into the RDF store; other records change
/// nothing.
///
/// # Errors
///
/// Returns [`Error::Corruption`](grafeo_common::utils::error::Error::Corruption)
/// naming the record, its graph and the term when a triple's term is not an
/// N-Triples term: replay never drops a triple.
#[cfg(feature = "wal")]
pub(super) fn replay_rdf_wal_record(
    rdf_store: &Arc<RdfStore>,
    record: &grafeo_storage::wal::WalRecord,
) -> Result<()> {
    use grafeo_storage::wal::WalRecord;

    match record {
        WalRecord::InsertRdfTriple {
            subject,
            predicate,
            object,
            graph,
        } => {
            let triple = wal_triple(
                "InsertRdfTriple",
                graph.as_deref(),
                subject,
                predicate,
                object,
            )?;
            let target = match graph {
                Some(name) => rdf_store.graph_or_create(name),
                None => Arc::clone(rdf_store),
            };
            target.insert(triple);
        }
        WalRecord::DeleteRdfTriple {
            subject,
            predicate,
            object,
            graph,
        } => {
            let triple = wal_triple(
                "DeleteRdfTriple",
                graph.as_deref(),
                subject,
                predicate,
                object,
            )?;
            let target = match graph {
                Some(name) => rdf_store.graph_or_create(name),
                None => Arc::clone(rdf_store),
            };
            target.remove(&triple);
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
    Ok(())
}

/// The triple of an RDF WAL record of kind `record` in `graph`, its terms
/// read from their N-Triples strings as the live statement wrote them.
#[cfg(feature = "wal")]
fn wal_triple(
    record: &str,
    graph: Option<&str>,
    subject: &str,
    predicate: &str,
    object: &str,
) -> Result<grafeo_core::graph::rdf::Triple> {
    use grafeo_common::utils::error::Error;
    use grafeo_core::graph::rdf::{Term, Triple};

    let term = |text: &str, role: &str| {
        Term::from_ntriples(text).map_err(|error| {
            let graph = graph.map_or_else(
                || "the default graph".to_string(),
                |name| format!("graph {name:?}"),
            );
            Error::corruption(format!("WAL record {record} in {graph}, {role}: {error}"))
        })
    };
    // Unchecked: the store holds what the statement gave it, as it was
    // written.
    Ok(Triple::new_unchecked(
        term(subject, "subject")?,
        term(predicate, "predicate")?,
        term(object, "object")?,
    ))
}

#[cfg(all(test, feature = "wal"))]
mod tests {
    use std::sync::Arc;

    use grafeo_core::graph::rdf::{RdfStore, Term, Triple};
    use grafeo_storage::wal::WalRecord;

    use super::replay_rdf_wal_record;

    fn insert(subject: &str, object: &str, graph: Option<&str>) -> WalRecord {
        WalRecord::InsertRdfTriple {
            subject: subject.to_string(),
            predicate: "<http://example.org/name>".to_string(),
            object: object.to_string(),
            graph: graph.map(str::to_string),
        }
    }

    #[test]
    fn replay_keeps_non_ascii_terms_and_whitespace() {
        let store = Arc::new(RdfStore::new());
        let terms = [
            Term::literal("Kraków"),
            Term::lang_literal("阿姆斯特丹", "zh"),
            Term::literal("🚲 naar Amsterdam"),
            Term::typed_literal("Ámsterdam", "http://example.org/stad"),
            Term::blank("b0 "),
        ];
        for term in &terms {
            for graph in [None, Some("http://example.org/Kraków")] {
                let record = insert("_:gus ", &term.to_string(), graph);
                replay_rdf_wal_record(&store, &record).unwrap();
            }
        }
        let named = store.graph("http://example.org/Kraków").unwrap();
        for term in terms {
            let triple = Triple::new(
                Term::blank("gus "),
                Term::iri("http://example.org/name"),
                term,
            );
            assert!(store.contains(&triple), "{triple}");
            assert!(named.contains(&triple), "{triple}");
        }
        assert_eq!(store.len(), 5);

        let delete = WalRecord::DeleteRdfTriple {
            subject: "_:gus ".to_string(),
            predicate: "<http://example.org/name>".to_string(),
            object: Term::literal("Kraków").to_string(),
            graph: None,
        };
        replay_rdf_wal_record(&store, &delete).unwrap();
        assert_eq!(store.len(), 4, "the delete found the non-ASCII literal");
    }

    #[test]
    fn a_record_whose_term_does_not_parse_fails_the_replay() {
        let store = Arc::new(RdfStore::new());
        let record = insert(
            "<http://example.org/alix>",
            "<<not a term",
            Some("http://example.org/g"),
        );
        let error = replay_rdf_wal_record(&store, &record)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("WAL record InsertRdfTriple in graph \"http://example.org/g\", object")
                && error.contains("<<not a term"),
            "{error}"
        );
        let delete = WalRecord::DeleteRdfTriple {
            subject: "Alix".to_string(),
            predicate: "<http://example.org/name>".to_string(),
            object: "\"Alix\"".to_string(),
            graph: None,
        };
        let error = replay_rdf_wal_record(&store, &delete)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("WAL record DeleteRdfTriple in the default graph, subject"),
            "{error}"
        );
    }
}
