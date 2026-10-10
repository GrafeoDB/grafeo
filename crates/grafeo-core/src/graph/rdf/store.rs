//! RDF Triple Store.
//!
//! Provides an in-memory triple store with efficient indexing for
//! subject, predicate, and object queries.

use super::sink::TripleSink;
use super::term::Term;
use super::triple::{Triple, TriplePattern};
use grafeo_common::change::RdfGraphOp;
use grafeo_common::storage::log_record::RdfGraphTarget;
use grafeo_common::types::TransactionId;
use grafeo_common::utils::hash::FxHashSet;
use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

#[cfg(test)]
mod path_lookup_tests {
    use super::super::path_budget::PathBudget;
    use super::*;
    use crate::execution::operators::OperatorError;
    use std::time::{Duration, Instant};

    #[test]
    fn path_lookup_polls_rejected_index_candidates() {
        let store = RdfStore::new();
        for n in 0..64 {
            store.insert(Triple::new(
                Term::iri("s"),
                Term::iri("p"),
                Term::literal(n.to_string()),
            ));
        }
        let pattern = TriplePattern {
            subject: Some(Term::iri("s")),
            predicate: Some(Term::iri("p")),
            object: Some(Term::literal("missing")),
        };
        let mut budget = PathBudget::new(4096, None);
        budget.expire_after_polls(8);
        let mut visits = 0;
        let result = store.visit_matches_with_pending(&pattern, None, &mut budget, &mut |_, _| {
            visits += 1;
            Ok(())
        });
        assert!(matches!(result, Err(OperatorError::Timeout)));
        assert_eq!(visits, 0);
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the legacy per-transaction buffer supported until 0.7.0"
    )]
    fn path_lookup_polls_rejected_pending_operations() {
        let store = RdfStore::new();
        let tx = TransactionId::new(1);
        for n in 0..64 {
            store.insert_in_transaction(
                tx,
                Triple::new(
                    Term::iri("s"),
                    Term::iri("other"),
                    Term::literal(n.to_string()),
                ),
            );
        }
        let mut budget = PathBudget::new(4096, None);
        budget.expire_after_polls(8);
        let result = store.visit_matches_with_pending(
            &TriplePattern::with_predicate(Term::iri("p")),
            Some(tx),
            &mut budget,
            &mut |_, _| Ok(()),
        );
        assert!(matches!(result, Err(OperatorError::Timeout)));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn path_lookup_deadline_bounds_index_lock_wait() {
        let store = RdfStore::new();
        std::thread::scope(|scope| {
            let (held_tx, held_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let store_ref = &store;
            scope.spawn(move || {
                let _guard = store_ref.sp_index.write();
                held_tx.send(()).unwrap();
                // Bound the fixture too, so a regressed lock wait fails
                // instead of preventing the test suite from finishing.
                let _ = release_rx.recv_timeout(Duration::from_secs(2));
            });
            held_rx.recv().unwrap();
            let deadline = Instant::now() + Duration::from_millis(2);
            let mut budget = PathBudget::new(4096, Some(deadline));
            let pattern = TriplePattern {
                subject: Some(Term::iri("s")),
                predicate: Some(Term::iri("p")),
                object: None,
            };
            let result =
                store.visit_matches_with_pending(&pattern, None, &mut budget, &mut |_, _| Ok(()));
            assert!(store.sp_index.try_read().is_none());
            release_tx.send(()).unwrap();
            assert!(matches!(result, Err(OperatorError::Timeout)));
            assert!(Instant::now() >= deadline);
        });
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the legacy per-transaction buffer supported until 0.7.0"
    )]
    fn path_lookup_pending_net_is_bounded_and_last_operation_wins() {
        let store = RdfStore::new();
        let tx = TransactionId::new(1);
        let triple = Triple::new(Term::iri("s"), Term::iri("p"), Term::iri("o"));
        store.insert(triple.clone());
        store.remove_in_transaction(tx, triple.clone());
        store.insert_in_transaction(tx, triple);
        let mut empty_budget = PathBudget::new(0, None);
        assert!(matches!(
            store.visit_matches_with_pending(
                &TriplePattern::any(),
                Some(tx),
                &mut empty_budget,
                &mut |_, _| Ok(())
            ),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(empty_budget.used(), 0);
        let mut budget = PathBudget::new(4096, None);
        let mut visits = 0;
        store
            .visit_matches_with_pending(
                &TriplePattern::any(),
                Some(tx),
                &mut budget,
                &mut |_, _| {
                    visits += 1;
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(visits, 1);
        assert_eq!(budget.used(), 0);
    }
}

/// A pending operation in a transaction buffer.
#[derive(Debug, Clone)]
enum PendingOp {
    /// Insert a triple.
    Insert(Triple),
    /// Delete a triple.
    Delete(Triple),
}

/// Transaction buffer for pending operations.
#[derive(Debug, Default)]
struct TransactionBuffer {
    /// Pending operations for each transaction.
    buffers: HashMap<TransactionId, Vec<PendingOp>>,
}

/// Configuration for the RDF store.
#[derive(Debug, Clone)]
pub struct RdfStoreConfig {
    /// Initial capacity for triple storage.
    pub initial_capacity: usize,
    /// Whether to build object index (for reverse lookups).
    pub index_objects: bool,
}

impl Default for RdfStoreConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            index_objects: true,
        }
    }
}

/// An in-memory RDF triple store.
///
/// The store maintains multiple indexes for efficient querying:
/// - SPO (Subject, Predicate, Object): primary storage
/// - POS (Predicate, Object, Subject): for predicate-based queries
/// - OSP (Object, Subject, Predicate): for object-based queries (optional)
///
/// The store holds committed triples only: a database transaction records
/// its RDF writes in its change set and applies them when it commits. The
/// deprecated per-transaction buffer (`insert_in_transaction` and the
/// methods around it) is no part of a database transaction.
pub struct RdfStore {
    /// Configuration.
    config: RdfStoreConfig,
    /// All triples (primary storage).
    triples: RwLock<FxHashSet<Arc<Triple>>>,
    /// Subject index: subject -> triples.
    subject_index: RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Predicate index: predicate -> triples.
    predicate_index:
        RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Object index: object -> triples (optional).
    object_index:
        RwLock<Option<hashbrown::HashMap<Term, Vec<Arc<Triple>>, foldhash::fast::RandomState>>>,
    /// Subject+Predicate composite index: (subject, predicate) -> triples.
    sp_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Predicate+Object composite index: (predicate, object) -> triples.
    po_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// Object+Subject composite index: (object, subject) -> triples.
    os_index:
        RwLock<hashbrown::HashMap<(Term, Term), Vec<Arc<Triple>>, foldhash::fast::RandomState>>,
    /// The deprecated per-transaction buffers (removed in 0.7.0).
    tx_buffer: RwLock<TransactionBuffer>,
    /// Named graphs, each a separate `RdfStore` partition.
    named_graphs: RwLock<HashMap<String, Arc<RdfStore>>>,
    /// Cached RDF statistics for query optimization. Invalidated on any mutation.
    statistics_cache: RwLock<Option<Arc<crate::statistics::RdfStatistics>>>,
    /// Cached term dictionary for dictionary-encoded triple scans. Invalidated on any mutation.
    dictionary_cache: RwLock<Option<Arc<super::dictionary::TermDictionary>>>,
    /// Compact Ring Index for SPARQL query acceleration. Built during `bulk_load()`
    /// or explicitly via `rebuild_ring()`. Marked stale on incremental mutations,
    /// falling back to HashMap indexes until rebuilt.
    #[cfg(feature = "ring-index")]
    ring: RwLock<Option<Arc<crate::index::ring::TripleRing>>>,
    /// Whether the Ring is stale (triples changed since last build).
    #[cfg(feature = "ring-index")]
    ring_stale: std::sync::atomic::AtomicBool,
}

impl RdfStore {
    /// Creates a new RDF store with default configuration.
    pub fn new() -> Self {
        Self::with_config(RdfStoreConfig::default())
    }

    /// Creates a new RDF store with the given configuration.
    pub fn with_config(config: RdfStoreConfig) -> Self {
        let object_index = if config.index_objects {
            Some(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            ))
        } else {
            None
        };

        Self {
            triples: RwLock::new(FxHashSet::default()),
            subject_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            predicate_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            object_index: RwLock::new(object_index),
            sp_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            po_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            os_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                foldhash::fast::RandomState::default(),
            )),
            tx_buffer: RwLock::new(TransactionBuffer::default()),
            named_graphs: RwLock::new(HashMap::new()),
            statistics_cache: RwLock::new(None),
            dictionary_cache: RwLock::new(None),
            #[cfg(feature = "ring-index")]
            ring: RwLock::new(None),
            #[cfg(feature = "ring-index")]
            ring_stale: std::sync::atomic::AtomicBool::new(false),
            config,
        }
    }

    /// Inserts a triple into the store.
    ///
    /// Returns `true` if the triple was newly inserted, `false` if it already existed.
    pub fn insert(&self, triple: Triple) -> bool {
        let triple = Arc::new(triple);

        // Check if already exists
        {
            let triples = self.triples.read();
            if triples.contains(&triple) {
                return false;
            }
        }

        // Insert into primary storage
        {
            let mut triples = self.triples.write();
            if !triples.insert(Arc::clone(&triple)) {
                return false;
            }
        }

        // Update indexes
        {
            let mut subject_index = self.subject_index.write();
            subject_index
                .entry(triple.subject().clone())
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut predicate_index = self.predicate_index.write();
            predicate_index
                .entry(triple.predicate().clone())
                .or_default()
                .push(Arc::clone(&triple));
        }

        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index {
                index
                    .entry(triple.object().clone())
                    .or_default()
                    .push(Arc::clone(&triple));
            }
        }

        // Update composite indexes
        {
            let mut sp = self.sp_index.write();
            sp.entry((triple.subject().clone(), triple.predicate().clone()))
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut po = self.po_index.write();
            po.entry((triple.predicate().clone(), triple.object().clone()))
                .or_default()
                .push(Arc::clone(&triple));
        }

        {
            let mut os = self.os_index.write();
            os.entry((triple.object().clone(), triple.subject().clone()))
                .or_default()
                .push(triple);
        }

        self.invalidate_statistics_cache();
        true
    }

    /// Inserts a batch of triples with single lock acquisition per index.
    ///
    /// Much more efficient than calling [`Self::insert`] in a loop because each
    /// index lock is acquired once for the entire batch rather than once per
    /// triple (4 lock acquisitions total vs 4 * N).
    ///
    /// Returns the number of triples that were newly inserted (duplicates are
    /// skipped).
    pub fn batch_insert(&self, triples: impl IntoIterator<Item = Triple>) -> usize {
        // Phase 1: deduplicate against primary storage (single lock)
        let mut new_triples = Vec::new();
        {
            let mut primary = self.triples.write();
            for triple in triples {
                let arc = Arc::new(triple);
                if primary.insert(Arc::clone(&arc)) {
                    new_triples.push(arc);
                }
            }
        }

        if new_triples.is_empty() {
            return 0;
        }

        let count = new_triples.len();

        // Phase 2: update subject index (single lock)
        {
            let mut subject_index = self.subject_index.write();
            for triple in &new_triples {
                subject_index
                    .entry(triple.subject().clone())
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 3: update predicate index (single lock)
        {
            let mut predicate_index = self.predicate_index.write();
            for triple in &new_triples {
                predicate_index
                    .entry(triple.predicate().clone())
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 4: update object index if enabled (single lock)
        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index {
                for triple in &new_triples {
                    index
                        .entry(triple.object().clone())
                        .or_default()
                        .push(Arc::clone(triple));
                }
            }
        }

        // Phase 5: update SP composite index (single lock)
        {
            let mut sp = self.sp_index.write();
            for triple in &new_triples {
                sp.entry((triple.subject().clone(), triple.predicate().clone()))
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 6: update PO composite index (single lock)
        {
            let mut po = self.po_index.write();
            for triple in &new_triples {
                po.entry((triple.predicate().clone(), triple.object().clone()))
                    .or_default()
                    .push(Arc::clone(triple));
            }
        }

        // Phase 7: update OS composite index (single lock)
        {
            let mut os = self.os_index.write();
            for triple in new_triples {
                os.entry((triple.object().clone(), triple.subject().clone()))
                    .or_default()
                    .push(triple);
            }
        }

        if count > 0 {
            self.invalidate_statistics_cache();
        }
        count
    }

    /// Removes a triple from the store.
    ///
    /// Returns `true` if the triple was found and removed.
    pub fn remove(&self, triple: &Triple) -> bool {
        // Remove from primary storage
        let removed = {
            let mut triples = self.triples.write();
            triples.remove(triple)
        };

        if !removed {
            return false;
        }

        // Update indexes
        {
            let mut subject_index = self.subject_index.write();
            if let Some(vec) = subject_index.get_mut(triple.subject()) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    subject_index.remove(triple.subject());
                }
            }
        }

        {
            let mut predicate_index = self.predicate_index.write();
            if let Some(vec) = predicate_index.get_mut(triple.predicate()) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    predicate_index.remove(triple.predicate());
                }
            }
        }

        if self.config.index_objects {
            let mut object_index = self.object_index.write();
            if let Some(ref mut index) = *object_index
                && let Some(vec) = index.get_mut(triple.object())
            {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    index.remove(triple.object());
                }
            }
        }

        // Remove from composite indexes
        {
            let mut sp = self.sp_index.write();
            let key = (triple.subject().clone(), triple.predicate().clone());
            if let Some(vec) = sp.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    sp.remove(&key);
                }
            }
        }

        {
            let mut po = self.po_index.write();
            let key = (triple.predicate().clone(), triple.object().clone());
            if let Some(vec) = po.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    po.remove(&key);
                }
            }
        }

        {
            let mut os = self.os_index.write();
            let key = (triple.object().clone(), triple.subject().clone());
            if let Some(vec) = os.get_mut(&key) {
                vec.retain(|t| t.as_ref() != triple);
                if vec.is_empty() {
                    os.remove(&key);
                }
            }
        }

        self.invalidate_statistics_cache();
        true
    }

    /// Returns the number of triples in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triples.read().len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triples.read().is_empty()
    }

    /// Estimates the heap footprint of the store for memory reporting.
    ///
    /// Returns `(total_triples, triples_and_indexes_bytes, term_dictionary_bytes,
    /// ring_index_bytes, named_graph_count)`. Recurses into named graphs so the
    /// totals reflect the full RDF memory, not just the default graph.
    ///
    /// The index-bytes figure is an approximation: each `HashMap` entry is
    /// charged for a pointer-sized key plus capacity-based Vec overhead. It
    /// undercounts the per-`Term` heap (a `Term::IRI(String)` carries its
    /// payload) and overcounts hash-map empty buckets. Good enough to surface
    /// "RDF is eating the heap" in an introspection breakdown.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize, usize, usize) {
        use std::mem::size_of;

        let triples = self.triples.read();
        let triple_arc_bytes = triples.capacity() * (size_of::<Arc<Triple>>() + size_of::<u64>());
        let triple_payload_bytes = triples.len() * size_of::<Triple>();
        drop(triples);

        let index_entry = size_of::<(Term, Vec<Arc<Triple>>)>();
        let composite_entry = size_of::<((Term, Term), Vec<Arc<Triple>>)>();

        let subject_bytes = {
            let g = self.subject_index.read();
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let predicate_bytes = {
            let g = self.predicate_index.read();
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let object_bytes = self.object_index.read().as_ref().map_or(0, |g| {
            g.capacity() * index_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        });
        let sp_bytes = {
            let g = self.sp_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let po_bytes = {
            let g = self.po_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };
        let os_bytes = {
            let g = self.os_index.read();
            g.capacity() * composite_entry
                + g.values()
                    .map(|v| v.capacity() * size_of::<Arc<Triple>>())
                    .sum::<usize>()
        };

        let mut triples_and_indexes = triple_arc_bytes
            + triple_payload_bytes
            + subject_bytes
            + predicate_bytes
            + object_bytes
            + sp_bytes
            + po_bytes
            + os_bytes;

        let term_dict_bytes = self
            .dictionary_cache
            .read()
            .as_ref()
            .map_or(0, |d| d.len() * (size_of::<Term>() + size_of::<u32>()) * 2);

        #[cfg(feature = "ring-index")]
        let ring_bytes = self
            .ring
            .read()
            .as_ref()
            .map_or(0, |r| r.len() * size_of::<(u32, u32, u32)>());
        #[cfg(not(feature = "ring-index"))]
        let ring_bytes = 0usize;

        let named_graphs_guard = self.named_graphs.read();
        let named_graph_count = named_graphs_guard.len();
        let mut total_triples = self.len();
        let mut total_term_dict = term_dict_bytes;
        let mut total_ring = ring_bytes;
        for (_name, graph) in named_graphs_guard.iter() {
            let (ng_triples, ng_store, ng_dict, ng_ring, _) = graph.heap_memory_bytes();
            total_triples += ng_triples;
            triples_and_indexes += ng_store;
            total_term_dict += ng_dict;
            total_ring += ng_ring;
        }

        (
            total_triples,
            triples_and_indexes,
            total_term_dict,
            total_ring,
            named_graph_count,
        )
    }

    /// Checks if a triple exists in the store.
    #[must_use]
    pub fn contains(&self, triple: &Triple) -> bool {
        self.triples.read().contains(triple)
    }

    /// Returns all triples in the store.
    pub fn triples(&self) -> Vec<Arc<Triple>> {
        self.triples.read().iter().cloned().collect()
    }

    /// The bytes the triple set and index maps of this graph (not its named
    /// graphs) have allocated.
    #[cfg(test)]
    pub(crate) fn index_allocation_bytes(&self) -> usize {
        self.triples.read().allocation_size()
            + self.subject_index.read().allocation_size()
            + self.predicate_index.read().allocation_size()
            + self
                .object_index
                .read()
                .as_ref()
                .map_or(0, hashbrown::HashMap::allocation_size)
            + self.sp_index.read().allocation_size()
            + self.po_index.read().allocation_size()
            + self.os_index.read().allocation_size()
    }

    /// Calls `visit` for every triple of this graph (not its named graphs),
    /// in the order of [`sorted_triples`](Self::sorted_triples), which it
    /// takes first: writers wait only while its references are taken, not
    /// while `visit` runs.
    ///
    /// # Errors
    ///
    /// Returns the first error of `visit`, which ends the walk.
    pub fn for_each_triple(
        &self,
        visit: &mut dyn FnMut(&Triple) -> grafeo_common::utils::error::Result<()>,
    ) -> grafeo_common::utils::error::Result<()> {
        for triple in &self.sorted_triples() {
            visit(triple)?;
        }
        Ok(())
    }

    /// The triples of this graph (not its named graphs) present now, in a
    /// defined order: by subject, then predicate, then object, a term
    /// ordered by its kind (IRIs, blank nodes, literals) and then by its
    /// strings (a literal by value, datatype and language). Stores that hold
    /// the same triples give them in the same order, whatever order the
    /// triples came in.
    ///
    /// Takes one reference per triple under one read lock, then sorts them
    /// after releasing it.
    #[must_use]
    pub fn sorted_triples(&self) -> Vec<Arc<Triple>> {
        let mut triples = self.triples();
        triples.sort_unstable_by(|a, b| {
            term_order(a.subject(), b.subject())
                .then_with(|| term_order(a.predicate(), b.predicate()))
                .then_with(|| term_order(a.object(), b.object()))
        });
        triples
    }

    /// Returns triples matching the given pattern.
    ///
    /// Uses composite indexes for 2-bound and 3-bound queries (O(1) lookup),
    /// single-term indexes for 1-bound queries, and full scan for unbound.
    pub fn find(&self, pattern: &TriplePattern) -> Vec<Arc<Triple>> {
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            // 3-bound: use SP composite, filter on O (at most 1 result)
            (Some(s), Some(p), Some(o)) => {
                let index = self.sp_index.read();
                if let Some(triples) = index.get(&(s.clone(), p.clone())) {
                    triples
                        .iter()
                        .filter(|t| t.object() == o)
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                }
            }
            // 2-bound: use composite indexes (O(1) lookup, no filtering)
            (Some(s), Some(p), None) => {
                let index = self.sp_index.read();
                index
                    .get(&(s.clone(), p.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            (Some(s), None, Some(o)) => {
                let index = self.os_index.read();
                index
                    .get(&(o.clone(), s.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            (None, Some(p), Some(o)) => {
                let index = self.po_index.read();
                index
                    .get(&(p.clone(), o.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            // 1-bound: use single-term indexes
            (Some(s), None, None) => {
                let index = self.subject_index.read();
                index.get(s).cloned().unwrap_or_default()
            }
            (None, Some(p), None) => {
                let index = self.predicate_index.read();
                index.get(p).cloned().unwrap_or_default()
            }
            (None, None, Some(o)) if self.config.index_objects => {
                let index = self.object_index.read();
                if let Some(ref idx) = *index {
                    idx.get(o).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                }
            }
            // 0-bound or O-only without object index: full scan
            _ => self
                .triples
                .read()
                .iter()
                .filter(|t| pattern.matches(t))
                .cloned()
                .collect(),
        }
    }

    /// Bounded, query-local visitor for native paths. The callback must not
    /// re-enter the store: it runs under a pending-buffer read lock and one
    /// index read lock, and may only update the query's charged collections.
    pub(crate) fn visit_matches_with_pending(
        &self,
        pattern: &TriplePattern,
        transaction_id: Option<TransactionId>,
        budget: &mut super::path_budget::PathBudget,
        visit: &mut dyn FnMut(
            &Triple,
            &mut super::path_budget::PathBudget,
        ) -> Result<(), crate::execution::operators::OperatorError>,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        use grafeo_common::utils::hash::FxHashMap;
        let buffer = match transaction_id {
            Some(_) => Some(budget.read(&self.tx_buffer)?),
            None => None,
        };
        // Borrow terms already owned by the buffer. No Arc/Triple clone or
        // eager result vector is needed to compute the last operation.
        let mut net: FxHashMap<&Triple, bool> = FxHashMap::default();
        let result = (|| {
            if let (Some(tx), Some(buffer)) = (transaction_id, buffer.as_ref())
                && let Some(ops) = buffer.buffers.get(&tx)
            {
                for op in ops {
                    budget.check()?;
                    let (triple, present) = match op {
                        PendingOp::Insert(triple) => (triple, true),
                        PendingOp::Delete(triple) => (triple, false),
                    };
                    if pattern.matches(triple) {
                        if !net.contains_key(triple) {
                            budget.reserve_map(&mut net, 1)?;
                        }
                        net.insert(triple, present);
                    }
                }
            }
            self.visit_path_base(pattern, budget, &mut |triple, budget| {
                if let Some(present) = net.get_mut(triple) {
                    if !*present {
                        return Ok(());
                    }
                    // A pending insertion already present in committed state
                    // is visited here, never again in the pending-only pass.
                    *present = false;
                }
                visit(triple, budget)
            })?;
            for (triple, present) in &net {
                budget.check()?;
                if *present {
                    visit(triple, budget)?;
                }
            }
            Ok(())
        })();
        let allocation = net.allocation_size();
        drop(net);
        budget.release(allocation);
        result
    }

    pub(crate) fn visit_path_base(
        &self,
        pattern: &TriplePattern,
        budget: &mut super::path_budget::PathBudget,
        visit: &mut dyn FnMut(
            &Triple,
            &mut super::path_budget::PathBudget,
        ) -> Result<(), crate::execution::operators::OperatorError>,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            (Some(s), Some(p), _) => {
                let index = budget.read(&self.sp_index)?;
                Self::visit_path_candidates(
                    index.get(&(s.clone(), p.clone())).into_iter().flatten(),
                    pattern,
                    budget,
                    visit,
                )
            }
            (Some(s), None, Some(o)) => {
                let index = budget.read(&self.os_index)?;
                Self::visit_path_candidates(
                    index.get(&(o.clone(), s.clone())).into_iter().flatten(),
                    pattern,
                    budget,
                    visit,
                )
            }
            (None, Some(p), Some(o)) => {
                let index = budget.read(&self.po_index)?;
                Self::visit_path_candidates(
                    index.get(&(p.clone(), o.clone())).into_iter().flatten(),
                    pattern,
                    budget,
                    visit,
                )
            }
            (Some(s), None, None) => {
                let index = budget.read(&self.subject_index)?;
                Self::visit_path_candidates(
                    index.get(s).into_iter().flatten(),
                    pattern,
                    budget,
                    visit,
                )
            }
            (None, Some(p), None) => {
                let index = budget.read(&self.predicate_index)?;
                Self::visit_path_candidates(
                    index.get(p).into_iter().flatten(),
                    pattern,
                    budget,
                    visit,
                )
            }
            (None, None, Some(o)) if self.config.index_objects => {
                let index = budget.read(&self.object_index)?;
                if let Some(index) = index.as_ref() {
                    return Self::visit_path_candidates(
                        index.get(o).into_iter().flatten(),
                        pattern,
                        budget,
                        visit,
                    );
                }
                drop(index);
                let triples = budget.read(&self.triples)?;
                Self::visit_path_candidates(triples.iter(), pattern, budget, visit)
            }
            _ => {
                let triples = budget.read(&self.triples)?;
                Self::visit_path_candidates(triples.iter(), pattern, budget, visit)
            }
        }
    }

    fn visit_path_candidates<'a>(
        candidates: impl IntoIterator<Item = &'a Arc<Triple>>,
        pattern: &TriplePattern,
        budget: &mut super::path_budget::PathBudget,
        visit: &mut dyn FnMut(
            &Triple,
            &mut super::path_budget::PathBudget,
        ) -> Result<(), crate::execution::operators::OperatorError>,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        for triple in candidates {
            budget.check()?; // Rejected candidates must still obey the deadline.
            if pattern.matches(triple) {
                visit(triple, budget)?;
            }
        }
        Ok(())
    }

    /// Visits named partitions without copying an unbounded graph-name list.
    /// The callback is query-local and must not re-enter this store.
    pub(crate) fn visit_path_graphs(
        &self,
        names: Option<&[String]>,
        budget: &mut super::path_budget::PathBudget,
        visit: &mut dyn FnMut(
            &str,
            &Arc<RdfStore>,
            &mut super::path_budget::PathBudget,
        ) -> Result<(), crate::execution::operators::OperatorError>,
    ) -> Result<(), crate::execution::operators::OperatorError> {
        let graphs = budget.read(&self.named_graphs)?;
        match names {
            Some(names) => {
                for name in names {
                    budget.check()?;
                    if let Some(store) = graphs.get(name) {
                        visit(name, store, budget)?;
                    }
                }
            }
            None => {
                for (name, store) in graphs.iter() {
                    budget.check()?;
                    visit(name, store, budget)?;
                }
            }
        }
        Ok(())
    }

    /// Returns triples with the given subject.
    pub fn triples_with_subject(&self, subject: &Term) -> Vec<Arc<Triple>> {
        let index = self.subject_index.read();
        index.get(subject).cloned().unwrap_or_default()
    }

    /// Returns triples with the given predicate.
    pub fn triples_with_predicate(&self, predicate: &Term) -> Vec<Arc<Triple>> {
        let index = self.predicate_index.read();
        index.get(predicate).cloned().unwrap_or_default()
    }

    /// Returns triples with the given object.
    pub fn triples_with_object(&self, object: &Term) -> Vec<Arc<Triple>> {
        let index = self.object_index.read();
        if let Some(ref idx) = *index {
            idx.get(object).cloned().unwrap_or_default()
        } else {
            // Fall back to full scan if object index is disabled
            self.triples
                .read()
                .iter()
                .filter(|t| t.object() == object)
                .cloned()
                .collect()
        }
    }

    /// Returns all unique subjects in the store.
    pub fn subjects(&self) -> Vec<Term> {
        self.subject_index.read().keys().cloned().collect()
    }

    /// Returns all unique predicates in the store.
    pub fn predicates(&self) -> Vec<Term> {
        self.predicate_index.read().keys().cloned().collect()
    }

    /// Returns all unique objects in the store.
    pub fn objects(&self) -> Vec<Term> {
        if self.config.index_objects {
            let index = self.object_index.read();
            if let Some(ref idx) = *index {
                return idx.keys().cloned().collect();
            }
        }
        // Fall back to collecting from triples
        let triples = self.triples.read();
        let mut objects = FxHashSet::default();
        for triple in triples.iter() {
            objects.insert(triple.object().clone());
        }
        objects.into_iter().collect()
    }

    /// Clears all triples from the store.
    pub fn clear(&self) {
        self.triples.write().clear();
        self.subject_index.write().clear();
        self.predicate_index.write().clear();
        if let Some(ref mut idx) = *self.object_index.write() {
            idx.clear();
        }
        self.sp_index.write().clear();
        self.po_index.write().clear();
        self.os_index.write().clear();
        self.invalidate_statistics_cache();
    }

    /// Returns store statistics.
    #[must_use]
    pub fn stats(&self) -> RdfStoreStats {
        RdfStoreStats {
            triple_count: self.len(),
            subject_count: self.subject_index.read().len(),
            predicate_count: self.predicate_index.read().len(),
            object_count: if self.config.index_objects {
                self.object_index.read().as_ref().map_or(0, |i| i.len())
            } else {
                0
            },
            graph_count: self.named_graphs.read().len(),
        }
    }

    /// Collects detailed RDF statistics for query optimization.
    ///
    /// Iterates all triples to compute per-predicate cardinality estimates,
    /// distinct subject/object counts, and index access pattern costs.
    #[must_use]
    pub fn collect_statistics(&self) -> crate::statistics::RdfStatistics {
        let mut collector = crate::statistics::RdfStatisticsCollector::new();
        let triples = self.triples.read();
        for triple in triples.iter() {
            collector.record_triple(
                &triple.subject().to_string(),
                &triple.predicate().to_string(),
                &triple.object().to_string(),
            );
        }
        collector.build()
    }

    /// Returns cached RDF statistics, computing them on first call.
    ///
    /// The cache is invalidated by any mutation (insert, delete, bulk load).
    /// This avoids the full-table-scan overhead of `collect_statistics()` on
    /// every query, which was a measurable regression for larger datasets.
    #[must_use]
    pub fn get_or_collect_statistics(&self) -> Arc<crate::statistics::RdfStatistics> {
        // Fast path: return cached statistics if available.
        if let Some(cached) = self.statistics_cache.read().as_ref() {
            return Arc::clone(cached);
        }

        // Slow path: compute and cache.
        let stats = Arc::new(self.collect_statistics());
        *self.statistics_cache.write() = Some(Arc::clone(&stats));
        stats
    }

    /// Invalidates the cached RDF statistics and term dictionary. Called after any mutation.
    fn invalidate_statistics_cache(&self) {
        *self.statistics_cache.write() = None;
        *self.dictionary_cache.write() = None;
        #[cfg(feature = "ring-index")]
        {
            self.ring_stale
                .store(true, std::sync::atomic::Ordering::Relaxed);
            *self.ring.write() = None;
        }
    }

    /// Returns a cached term dictionary, building it on first call.
    ///
    /// The dictionary maps each unique `Term` in the store to a compact `u32` ID.
    /// It is invalidated by any mutation (insert, delete, bulk load).
    #[must_use]
    pub fn get_or_build_dictionary(&self) -> Arc<super::dictionary::TermDictionary> {
        // Fast path: return cached dictionary
        {
            let cache = self.dictionary_cache.read();
            if let Some(dict) = cache.as_ref() {
                return Arc::clone(dict);
            }
        }

        // Slow path: build dictionary from all triples
        let triples = self.triples.read();
        let mut dict =
            super::dictionary::TermDictionary::with_capacity(triples.len().saturating_mul(3));
        for triple in triples.iter() {
            dict.get_or_insert(triple.subject());
            dict.get_or_insert(triple.predicate());
            dict.get_or_insert(triple.object());
        }

        let dict = Arc::new(dict);
        *self.dictionary_cache.write() = Some(Arc::clone(&dict));
        dict
    }

    /// Returns the cached term dictionary if it exists, without building it.
    #[must_use]
    pub fn term_dictionary(&self) -> Option<Arc<super::dictionary::TermDictionary>> {
        self.dictionary_cache.read().clone()
    }

    /// Returns the Ring Index if it exists and is not stale.
    #[cfg(feature = "ring-index")]
    #[must_use]
    pub fn ring(&self) -> Option<Arc<crate::index::ring::TripleRing>> {
        if self.ring_stale.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        self.ring.read().clone()
    }

    /// Builds or rebuilds the Ring Index from current triples.
    ///
    /// The Ring provides ~3x memory reduction and O(log sigma) pattern counting.
    /// It is automatically built during `bulk_load()` and can be rebuilt
    /// explicitly after incremental mutations.
    #[cfg(feature = "ring-index")]
    pub fn rebuild_ring(&self) {
        let triples = self.triples.read();
        if triples.is_empty() {
            *self.ring.write() = None;
            return;
        }
        let ring = crate::index::ring::TripleRing::from_triples(
            triples.iter().map(|t| t.as_ref().clone()),
        );
        *self.ring.write() = Some(Arc::new(ring));
        self.ring_stale
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Sets the Ring Index directly (used during container deserialization).
    #[cfg(feature = "ring-index")]
    pub fn set_ring(&self, ring: crate::index::ring::TripleRing) {
        *self.ring.write() = Some(Arc::new(ring));
        self.ring_stale
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    // =========================================================================
    // Bulk loading
    // =========================================================================

    /// Loads triples in bulk, replacing all existing data.
    ///
    /// Much faster than `batch_insert()` for initial data loading:
    /// - Skips duplicate checking entirely (caller must ensure no duplicates)
    /// - Builds all indexes in a single pass using pre-sized `HashMap`s
    /// - Computes [`RdfStatistics`](crate::statistics::RdfStatistics) during the
    ///   same traversal (no extra scan needed)
    ///
    /// **Warning**: This replaces all existing triples and indexes in the store.
    /// Any previously stored data will be lost.
    pub fn bulk_load(&self, triples: impl IntoIterator<Item = Triple>) -> BulkLoadResult {
        let arcs: Vec<Arc<Triple>> = triples.into_iter().map(Arc::new).collect();
        let count = arcs.len();

        if count == 0 {
            self.clear();
            return BulkLoadResult {
                triple_count: 0,
                statistics: crate::statistics::RdfStatistics::new(),
            };
        }

        // Build all indexes in local HashMaps (no locks during build)
        let hasher = || foldhash::fast::RandomState::default();
        let mut subject_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 4, hasher());
        let mut predicate_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 8, hasher());
        let mut object_idx_map = hashbrown::HashMap::with_capacity_and_hasher(count / 4, hasher());
        let mut sp_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());
        let mut po_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());
        let mut os_idx = hashbrown::HashMap::with_capacity_and_hasher(count / 2, hasher());

        let mut stats_collector = crate::statistics::RdfStatisticsCollector::new();

        for triple in &arcs {
            // Single-term indexes
            subject_idx
                .entry(triple.subject().clone())
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            predicate_idx
                .entry(triple.predicate().clone())
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            if self.config.index_objects {
                object_idx_map
                    .entry(triple.object().clone())
                    .or_insert_with(Vec::new)
                    .push(Arc::clone(triple));
            }

            // Composite indexes
            sp_idx
                .entry((triple.subject().clone(), triple.predicate().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            po_idx
                .entry((triple.predicate().clone(), triple.object().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));
            os_idx
                .entry((triple.object().clone(), triple.subject().clone()))
                .or_insert_with(Vec::new)
                .push(Arc::clone(triple));

            // Collect statistics in the same pass
            stats_collector.record_triple(
                &triple.subject().to_string(),
                &triple.predicate().to_string(),
                &triple.object().to_string(),
            );
        }

        // Swap indexes into the store (one lock acquisition per index)
        let primary: FxHashSet<Arc<Triple>> = arcs.into_iter().collect();
        *self.triples.write() = primary;
        *self.subject_index.write() = subject_idx;
        *self.predicate_index.write() = predicate_idx;
        *self.object_index.write() = if self.config.index_objects {
            Some(object_idx_map)
        } else {
            None
        };
        *self.sp_index.write() = sp_idx;
        *self.po_index.write() = po_idx;
        *self.os_index.write() = os_idx;

        let stats = stats_collector.build();
        // Cache the freshly-computed statistics from the bulk load pass.
        *self.statistics_cache.write() = Some(Arc::new(stats.clone()));
        // Invalidate the dictionary cache: the old dictionary mapped terms
        // from the previous dataset, not the newly loaded one.
        *self.dictionary_cache.write() = None;

        // Build Ring Index automatically during bulk load (when feature is enabled).
        #[cfg(feature = "ring-index")]
        {
            let ring = crate::index::ring::TripleRing::from_triples(
                self.triples.read().iter().map(|t| t.as_ref().clone()),
            );
            *self.ring.write() = Some(Arc::new(ring));
            self.ring_stale
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }

        BulkLoadResult {
            triple_count: count,
            statistics: stats,
        }
    }

    /// Parses and loads an N-Triples document, replacing all existing data.
    ///
    /// Each line is parsed as `<subject> <predicate> <object> .` per the
    /// [N-Triples spec](https://www.w3.org/TR/n-triples/). Empty lines and
    /// comment lines (starting with `#`) are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or if a line cannot be parsed.
    pub fn load_ntriples(
        &self,
        reader: impl std::io::BufRead,
    ) -> Result<BulkLoadResult, NTriplesError> {
        let mut triples = Vec::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(NTriplesError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let triple = parse_ntriples_line(trimmed).map_err(|reason| NTriplesError::Parse {
                line: line_no + 1,
                content: format!("{reason}: {line}"),
            })?;
            triples.push(triple);
        }
        Ok(self.bulk_load(triples))
    }

    /// Parses and loads a Turtle document, replacing all existing data.
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` on parse failure.
    pub fn load_turtle(&self, input: &str) -> Result<BulkLoadResult, super::turtle::TurtleError> {
        let triples = super::turtle::TurtleParser::new().parse(input)?;
        Ok(self.bulk_load(triples))
    }

    /// Parses a Turtle document and streams triples into the store via batched inserts.
    ///
    /// Unlike [`load_turtle`](Self::load_turtle), this does not replace existing data.
    /// Triples are inserted incrementally in batches, keeping memory bounded regardless
    /// of document size. Duplicate triples are skipped during each batch insert.
    ///
    /// # Errors
    ///
    /// Returns a `TurtleError` on parse failure.
    pub fn load_turtle_streaming(
        &self,
        input: &str,
        batch_size: usize,
    ) -> Result<usize, super::turtle::TurtleError> {
        let mut sink = super::sink::BatchInsertSink::new(self, batch_size);
        let mut parser = super::turtle::TurtleParser::new();
        parser.parse_into(input, &mut sink)?;
        Ok(sink.total_inserted())
    }

    /// Reads a Turtle document from a buffered reader and streams triples into the store.
    ///
    /// The entire reader is consumed into a string first (Turtle requires random access
    /// for prefix resolution), then parsed with batched inserts. For pure streaming from
    /// a reader without replacing existing data, this is the recommended entry point.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if reading fails, or a `TurtleError` on parse failure.
    pub fn load_turtle_reader(
        &self,
        reader: impl std::io::Read,
        batch_size: usize,
    ) -> Result<usize, NTriplesError> {
        let mut input = String::new();
        std::io::Read::read_to_string(&mut { reader }, &mut input).map_err(NTriplesError::Io)?;
        self.load_turtle_streaming(&input, batch_size)
            .map_err(|e| NTriplesError::Parse {
                line: e.line,
                content: e.message,
            })
    }

    /// Parses N-Triples from a reader, streaming triples into the store via batched inserts.
    ///
    /// Unlike [`load_ntriples`](Self::load_ntriples), this does not replace existing data.
    /// Triples are inserted incrementally in batches, keeping memory bounded.
    ///
    /// # Errors
    ///
    /// Returns an `NTriplesError` on I/O or parse failure.
    pub fn load_ntriples_streaming(
        &self,
        reader: impl std::io::BufRead,
        batch_size: usize,
    ) -> Result<usize, NTriplesError> {
        let mut sink = super::sink::BatchInsertSink::new(self, batch_size);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(NTriplesError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let triple = parse_ntriples_line(trimmed).map_err(|reason| NTriplesError::Parse {
                line: line_no + 1,
                content: format!("{reason}: {line}"),
            })?;
            sink.emit(triple).map_err(|e| NTriplesError::Parse {
                line: line_no + 1,
                content: e,
            })?;
        }
        sink.finish().map_err(|e| NTriplesError::Parse {
            line: 0,
            content: e,
        })?;
        Ok(sink.total_inserted())
    }

    /// Serializes this store's triples to Turtle format.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if serialization fails.
    pub fn to_turtle(&self) -> std::io::Result<String> {
        super::turtle::TurtleSerializer::new().to_string(&self.triples())
    }

    /// Serializes this store (default + named graphs) to N-Quads format.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if serialization fails.
    pub fn to_nquads(&self) -> std::io::Result<String> {
        super::nquads::to_nquads_string(self)
    }

    // =========================================================================
    // Named graph support
    // =========================================================================

    /// The configuration of a new named graph: this store's, with indexes
    /// that start empty and grow as triples arrive (a store can hold many
    /// named graphs, most of them small or empty).
    fn named_graph_config(&self) -> RdfStoreConfig {
        RdfStoreConfig {
            initial_capacity: 0,
            ..self.config.clone()
        }
    }

    /// Returns a named graph by IRI, or `None` if it doesn't exist.
    #[must_use]
    pub fn graph(&self, name: &str) -> Option<Arc<RdfStore>> {
        self.named_graphs.read().get(name).cloned()
    }

    /// Returns a named graph, creating it if it doesn't exist.
    pub fn graph_or_create(&self, name: &str) -> Arc<RdfStore> {
        {
            let graphs = self.named_graphs.read();
            if let Some(g) = graphs.get(name) {
                return Arc::clone(g);
            }
        }
        let mut graphs = self.named_graphs.write();
        Arc::clone(
            graphs
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(RdfStore::with_config(self.named_graph_config()))),
        )
    }

    /// Creates a named graph. Returns `false` if it already exists.
    pub fn create_graph(&self, name: &str) -> bool {
        let mut graphs = self.named_graphs.write();
        if graphs.contains_key(name) {
            return false;
        }
        graphs.insert(
            name.to_string(),
            Arc::new(RdfStore::with_config(self.named_graph_config())),
        );
        true
    }

    /// Drops a named graph. Returns `false` if it didn't exist.
    pub fn drop_graph(&self, name: &str) -> bool {
        self.named_graphs.write().remove(name).is_some()
    }

    /// Returns all named graph IRIs.
    #[must_use]
    pub fn graph_names(&self) -> Vec<String> {
        self.named_graphs.read().keys().cloned().collect()
    }

    /// Returns the number of named graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.named_graphs.read().len()
    }

    /// Clears a specific graph, or the default graph if `name` is `None`.
    pub fn clear_graph(&self, name: Option<&str>) {
        match name {
            None => self.clear(),
            Some(n) => {
                if let Some(g) = self.named_graphs.read().get(n) {
                    g.clear();
                }
            }
        }
    }

    /// Clears all named graphs (but not the default graph).
    pub fn clear_all_named(&self) {
        self.named_graphs.write().clear();
    }

    /// Copies all triples from source graph to destination graph. A copy of
    /// a graph onto itself does nothing (SPARQL 1.1 Update, section 3.2.3).
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if source == dest {
            return;
        }
        let triples = match source {
            None => self.triples(),
            Some(n) => self.graph(n).map(|g| g.triples()).unwrap_or_default(),
        };
        let dest_store: Arc<RdfStore> = match dest {
            None => {
                // Copying into default graph: clear and re-insert
                self.clear();
                // We need to insert directly, so just do it inline
                for t in triples {
                    self.insert((*t).clone());
                }
                return;
            }
            Some(n) => self.graph_or_create(n),
        };
        dest_store.clear();
        for t in triples {
            dest_store.insert((*t).clone());
        }
    }

    /// Moves all triples from source graph to destination graph. A move of
    /// a graph onto itself does nothing (SPARQL 1.1 Update, section 3.2.4):
    /// it used to drop the graph.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn move_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if source == dest {
            return;
        }
        self.copy_graph(source, dest);
        match source {
            None => self.clear(),
            Some(n) => {
                self.drop_graph(n);
            }
        }
    }

    /// Adds all triples from source graph into destination graph (union). An
    /// add of a graph onto itself does nothing.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn add_graph(&self, source: Option<&str>, dest: Option<&str>) {
        if source == dest {
            return;
        }
        let triples = match source {
            None => self.triples(),
            Some(n) => self.graph(n).map(|g| g.triples()).unwrap_or_default(),
        };
        match dest {
            None => {
                for t in triples {
                    self.insert((*t).clone());
                }
            }
            Some(n) => {
                let dest_store = self.graph_or_create(n);
                for t in triples {
                    dest_store.insert((*t).clone());
                }
            }
        }
    }

    /// Applies a whole-graph operation: what a SPARQL `CREATE`, `DROP`,
    /// `CLEAR`, `COPY`, `MOVE` or `ADD` changes, and what the replay of its
    /// logged operation changes. `DROP` and `CLEAR` of the default graph
    /// empty it (it always exists); of `ALL`, they empty the default graph
    /// and remove every named graph; of `NAMED`, they remove every named
    /// graph. `CLEAR` of one named graph empties it and keeps it.
    ///
    /// Returns `false` when the operation found nothing to change that it
    /// names: a graph to create that exists, or a named graph to drop that
    /// does not.
    pub fn apply_graph_op(&self, op: &RdfGraphOp) -> bool {
        match op {
            RdfGraphOp::Create { name } => self.create_graph(name),
            RdfGraphOp::Drop {
                target: RdfGraphTarget::Named(name),
            } => self.drop_graph(name),
            RdfGraphOp::Drop { target } | RdfGraphOp::Clear { target } => {
                match target {
                    RdfGraphTarget::Default => self.clear(),
                    RdfGraphTarget::Named(name) => self.clear_graph(Some(name)),
                    RdfGraphTarget::AllNamed => self.clear_all_named(),
                    RdfGraphTarget::All => {
                        self.clear();
                        self.clear_all_named();
                    }
                }
                true
            }
            RdfGraphOp::Copy { source, target } => {
                self.copy_graph(source.as_deref(), target.as_deref());
                true
            }
            RdfGraphOp::Move { source, target } => {
                self.move_graph(source.as_deref(), target.as_deref());
                true
            }
            RdfGraphOp::Add { source, target } => {
                self.add_graph(source.as_deref(), target.as_deref());
                true
            }
        }
    }

    /// Inserts `triple` into `graph` (`None` for the default graph), creating
    /// a named graph that does not exist. Returns whether the triple is new.
    pub fn insert_into(&self, graph: Option<&str>, triple: Triple) -> bool {
        match graph {
            None => self.insert(triple),
            Some(name) => self.graph_or_create(name).insert(triple),
        }
    }

    /// Removes `triple` from `graph` (`None` for the default graph). Returns
    /// whether the graph held it; a named graph that does not exist holds
    /// nothing and is not created.
    pub fn remove_from(&self, graph: Option<&str>, triple: &Triple) -> bool {
        match graph {
            None => self.remove(triple),
            Some(name) => self.graph(name).is_some_and(|store| store.remove(triple)),
        }
    }

    /// Finds triples across specific graphs.
    ///
    /// - `graphs = None` searches the default graph only (backward compatible).
    /// - `graphs = Some(&[])` searches all named graphs (excluding the default graph).
    /// - `graphs = Some(&["g1", "g2"])` searches those named graphs only.
    pub fn find_in_graphs(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        match graphs {
            None => {
                // Default graph only
                self.find(pattern).into_iter().map(|t| (None, t)).collect()
            }
            Some([]) => {
                // All named graphs (excludes default graph per SPARQL spec sec 13.3)
                let mut results = Vec::new();
                for (name, store) in self.named_graphs.read().iter() {
                    for t in store.find(pattern) {
                        results.push((Some(name.clone()), t));
                    }
                }
                results
            }
            Some(names) => {
                // Specific named graphs
                let mut results = Vec::new();
                let graphs = self.named_graphs.read();
                for name in names {
                    if let Some(store) = graphs.get(*name) {
                        for t in store.find(pattern) {
                            results.push((Some((*name).to_string()), t));
                        }
                    }
                }
                results
            }
        }
    }

    /// Like [`find_in_graphs`](Self::find_in_graphs), but as seen by
    /// `transaction_id`: each graph's pending changes from that transaction
    /// are applied (see [`find_with_pending`](Self::find_with_pending)).
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn find_in_graphs_with_pending(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
        transaction_id: Option<TransactionId>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        let Some(transaction_id) = transaction_id else {
            return self.find_in_graphs(pattern, graphs);
        };
        let tx = Some(transaction_id);
        match graphs {
            None => self
                .buffered_find(pattern, tx)
                .into_iter()
                .map(|t| (None, t))
                .collect(),
            Some([]) => {
                let mut results = Vec::new();
                for (name, store) in self.named_graphs.read().iter() {
                    for t in store.buffered_find(pattern, tx) {
                        results.push((Some(name.clone()), t));
                    }
                }
                results
            }
            Some(names) => {
                let mut results = Vec::new();
                let graphs = self.named_graphs.read();
                for name in names {
                    if let Some(store) = graphs.get(*name) {
                        for t in store.buffered_find(pattern, tx) {
                            results.push((Some((*name).to_string()), t));
                        }
                    }
                }
                results
            }
        }
    }

    // =========================================================================
    // Transaction support
    // =========================================================================

    /// Inserts a triple within a transaction context.
    ///
    /// The insert is buffered until the transaction is committed.
    /// If the transaction is rolled back, the insert is discarded.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn insert_in_transaction(&self, transaction_id: TransactionId, triple: Triple) {
        let mut buffer = self.tx_buffer.write();
        buffer
            .buffers
            .entry(transaction_id)
            .or_default()
            .push(PendingOp::Insert(triple));
    }

    /// Removes a triple within a transaction context.
    ///
    /// The removal is buffered until the transaction is committed.
    /// If the transaction is rolled back, the removal is discarded.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn remove_in_transaction(&self, transaction_id: TransactionId, triple: Triple) {
        let mut buffer = self.tx_buffer.write();
        buffer
            .buffers
            .entry(transaction_id)
            .or_default()
            .push(PendingOp::Delete(triple));
    }

    /// Commits a transaction, applying all buffered operations in order, in
    /// this graph and in every named graph.
    ///
    /// Returns the number of operations applied.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn commit_transaction(&self, transaction_id: TransactionId) -> usize {
        let ops = {
            let mut buffer = self.tx_buffer.write();
            buffer.buffers.remove(&transaction_id).unwrap_or_default()
        };

        let mut count = ops.len();
        for op in ops {
            match op {
                PendingOp::Insert(triple) => {
                    self.insert(triple);
                }
                PendingOp::Delete(triple) => {
                    self.remove(&triple);
                }
            }
        }
        for graph in self.named_graph_stores() {
            count += graph.commit_transaction(transaction_id);
        }
        count
    }

    /// Rolls back a transaction, discarding all buffered operations in this
    /// graph and in every named graph.
    ///
    /// Returns the number of operations discarded.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn rollback_transaction(&self, transaction_id: TransactionId) -> usize {
        let mut count = self
            .tx_buffer
            .write()
            .buffers
            .remove(&transaction_id)
            .map_or(0, |ops| ops.len());
        for graph in self.named_graph_stores() {
            count += graph.rollback_transaction(transaction_id);
        }
        count
    }

    /// Snapshot of the named graph stores, so their transaction buffers can be
    /// committed or rolled back without holding the graph map lock.
    fn named_graph_stores(&self) -> Vec<Arc<RdfStore>> {
        self.named_graphs.read().values().cloned().collect()
    }

    /// Checks if a transaction has pending operations.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    #[must_use]
    pub fn has_pending_ops(&self, transaction_id: TransactionId) -> bool {
        let buffer = self.tx_buffer.read();
        buffer
            .buffers
            .get(&transaction_id)
            .is_some_and(|ops| !ops.is_empty())
    }

    /// Returns triples matching the given pattern as seen by the specified
    /// transaction: committed triples with that transaction's pending changes
    /// applied (read-your-writes). Pending operations take effect in order, so
    /// the result is exactly what the store will hold once the transaction
    /// commits: a triple inserted and then deleted is absent, one deleted and
    /// then re-inserted is present, and a triple is never returned twice.
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    pub fn find_with_pending(
        &self,
        pattern: &TriplePattern,
        transaction_id: Option<TransactionId>,
    ) -> Vec<Arc<Triple>> {
        self.buffered_find(pattern, transaction_id)
    }

    /// What [`find_with_pending`](Self::find_with_pending) returns.
    fn buffered_find(
        &self,
        pattern: &TriplePattern,
        transaction_id: Option<TransactionId>,
    ) -> Vec<Arc<Triple>> {
        let mut results = self.find(pattern);

        let Some(tx) = transaction_id else {
            return results;
        };
        let buffer = self.tx_buffer.read();
        let Some(ops) = buffer.buffers.get(&tx) else {
            return results;
        };

        // Net effect per matching triple: `true` = present after the
        // transaction, `false` = absent. The last operation wins.
        let mut net: HashMap<&Triple, bool> = HashMap::new();
        for op in ops {
            match op {
                PendingOp::Insert(triple) if pattern.matches(triple) => {
                    net.insert(triple, true);
                }
                PendingOp::Delete(triple) if pattern.matches(triple) => {
                    net.insert(triple, false);
                }
                _ => {}
            }
        }
        if net.is_empty() {
            return results;
        }

        // Committed triples the transaction deleted disappear; committed
        // triples it re-inserted are already in `results`.
        results.retain(|t| net.get(t.as_ref()).copied().unwrap_or(true));
        let added: Vec<Arc<Triple>> = {
            let committed: FxHashSet<&Triple> = results.iter().map(AsRef::as_ref).collect();
            net.iter()
                .filter(|&(triple, &present)| present && !committed.contains(*triple))
                .map(|(triple, _)| Arc::new((*triple).clone()))
                .collect()
        };
        results.extend(added);
        results
    }

    /// Whether `triple` is present as seen by `transaction_id` (committed
    /// state plus that transaction's pending changes).
    #[deprecated(
        since = "0.6.0",
        note = "a database transaction records its RDF writes in its change set; this buffer is no part of it (removed in 0.7.0)"
    )]
    #[must_use]
    pub fn contains_with_pending(&self, triple: &Triple, transaction_id: TransactionId) -> bool {
        let pattern = TriplePattern {
            subject: Some(triple.subject().clone()),
            predicate: Some(triple.predicate().clone()),
            object: Some(triple.object().clone()),
        };
        !self
            .buffered_find(&pattern, Some(transaction_id))
            .is_empty()
    }
}

impl Default for RdfStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an RDF store.
#[derive(Debug, Clone, Copy)]
pub struct RdfStoreStats {
    /// Total number of triples.
    pub triple_count: usize,
    /// Number of unique subjects.
    pub subject_count: usize,
    /// Number of unique predicates.
    pub predicate_count: usize,
    /// Number of unique objects (0 if object index disabled).
    pub object_count: usize,
    /// Number of named graphs.
    pub graph_count: usize,
}

/// Result of a bulk load operation.
#[derive(Debug, Clone)]
pub struct BulkLoadResult {
    /// Number of triples loaded.
    pub triple_count: usize,
    /// Statistics computed during the load pass.
    pub statistics: crate::statistics::RdfStatistics,
}

/// Error from parsing an N-Triples document.
#[derive(Debug)]
#[non_exhaustive]
pub enum NTriplesError {
    /// I/O error while reading.
    Io(std::io::Error),
    /// A line could not be parsed as a valid N-Triples triple.
    Parse {
        /// 1-based line number.
        line: usize,
        /// What is wrong: for a line that is not an N-Triples triple, the
        /// reason followed by the line; otherwise the parser's or the sink's
        /// message.
        content: String,
    },
}

impl std::fmt::Display for NTriplesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Parse { line, content } => {
                write!(f, "parse error at line {line}: {content}")
            }
        }
    }
}

impl std::error::Error for NTriplesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Parse { .. } => None,
        }
    }
}

/// The order of [`RdfStore::for_each_triple`]: IRIs, then blank nodes, then
/// literals; within a kind by the IRI, the id, or a literal's value,
/// datatype and language. Only equal terms compare equal.
fn term_order(a: &Term, b: &Term) -> std::cmp::Ordering {
    fn rank(term: &Term) -> u8 {
        match term {
            Term::Iri(_) => 0,
            Term::BlankNode(_) => 1,
            Term::Literal(_) => 2,
        }
    }
    match (a, b) {
        (Term::Iri(a), Term::Iri(b)) => a.as_str().cmp(b.as_str()),
        (Term::BlankNode(a), Term::BlankNode(b)) => a.id().cmp(b.id()),
        (Term::Literal(a), Term::Literal(b)) => a
            .value()
            .cmp(b.value())
            .then_with(|| a.datatype().cmp(b.datatype()))
            .then_with(|| a.language().cmp(&b.language())),
        _ => rank(a).cmp(&rank(b)),
    }
}

/// Extracts the next N-Triples term from a string, returning `(term_str, rest)`.
fn next_ntriples_term(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('<') {
        // IRI: find closing >
        let end = rest.find('>')?;
        Some((&s[..end + 2], &rest[end + 1..]))
    } else if s.starts_with("_:") {
        // Blank node: until whitespace or end
        let end = s.find(|c: char| c.is_whitespace()).unwrap_or(s.len());
        Some((&s[..end], &s[end..]))
    } else if s.starts_with('"') {
        // Literal: find closing quote (handling escapes), then optional suffix
        let bytes = s.as_bytes();
        let mut pos = 1;
        while pos < bytes.len() {
            if bytes[pos] == b'\\' {
                // Skip the escape sequence; an escape at the end of the
                // line ends the term, which then has no closing quote.
                pos = (pos + 2).min(bytes.len());
            } else if bytes[pos] == b'"' {
                pos += 1;
                // Check for datatype or language suffix
                if s[pos..].starts_with("^^<") {
                    if let Some(end) = s[pos..].find('>') {
                        pos += end + 1;
                    }
                } else if s[pos..].starts_with('@') {
                    let lang_end = s[pos..]
                        .find(|c: char| c.is_whitespace())
                        .unwrap_or(s.len() - pos);
                    pos += lang_end;
                }
                break;
            } else {
                pos += 1;
            }
        }
        Some((&s[..pos], &s[pos..]))
    } else {
        None
    }
}

/// Parses a single N-Triples line into a `Triple`.
///
/// Expected format: `<subject> <predicate> <object> .`
///
/// # Errors
///
/// Returns what is wrong: a line that is not three terms followed by `.`,
/// or the term that does not parse, with its [`TermParseError`](super::TermParseError).
fn parse_ntriples_line(line: &str) -> Result<Triple, String> {
    let shape = || "expected three terms followed by '.'".to_string();
    let (subj_str, rest) = next_ntriples_term(line).ok_or_else(shape)?;
    let (pred_str, rest) = next_ntriples_term(rest).ok_or_else(shape)?;
    let (obj_str, rest) = next_ntriples_term(rest).ok_or_else(shape)?;

    let term = |text: &str, role: &str| {
        Term::from_ntriples(text).map_err(|error| format!("{role}: {error}"))
    };
    let subject = term(subj_str, "subject")?;
    let predicate = term(pred_str, "predicate")?;
    let object = term(obj_str, "object")?;

    // Expect trailing ` .`
    if !rest.trim().starts_with('.') {
        return Err(shape());
    }
    Ok(Triple::new(subject, predicate, object))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/age"),
                Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
            ),
            Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/gus"),
            ),
            Triple::new(
                Term::iri("http://example.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
        ]
    }

    #[test]
    fn for_each_triple_visits_this_graph_in_term_order() {
        let iri = |local: &str| Term::iri(format!("http://example.org/{local}"));
        let expected = vec![
            Triple::new(iri("alix"), iri("knows"), iri("gus")),
            Triple::new(iri("alix"), iri("knows"), Term::blank("b0")),
            Triple::new(
                iri("alix"),
                iri("knows"),
                Term::typed_literal("Gus", "http://example.org/name"),
            ),
            Triple::new(iri("alix"), iri("knows"), Term::literal("Gus")),
            Triple::new(iri("alix"), iri("knows"), Term::literal("Gus ")),
            Triple::new(iri("alix"), iri("likes"), iri("amsterdam")),
            Triple::new(iri("gus"), iri("knows"), iri("alix")),
            Triple::new(Term::blank("b0"), iri("knows"), iri("alix")),
        ];
        let store = RdfStore::new();
        for triple in expected.iter().rev() {
            store.insert(triple.clone());
        }
        store
            .graph_or_create("http://example.org/trips")
            .insert(Triple::new(iri("mia"), iri("visits"), iri("paris")));
        let mut visited = Vec::new();
        store
            .for_each_triple(&mut |triple| {
                visited.push(triple.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(visited, expected, "in term order, without the named graph");

        // The first error ends the walk.
        let mut seen = 0;
        let error = store
            .for_each_triple(&mut |_| {
                seen += 1;
                if seen == 2 {
                    Err(grafeo_common::utils::error::Error::Internal(
                        "Mia stops here".to_string(),
                    ))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(seen, 2);
        assert!(error.to_string().contains("Mia stops here"), "{error}");
    }

    #[test]
    fn ntriples_lines_decode_non_ascii_text_and_refuse_bad_escapes() {
        // `~` stands for a backslash.
        let input = concat!(
            "<http://example.org/Kraków> <http://example.org/name> \"Krak~u00F3w\"@pl .\n",
            "_:b0 <http://example.org/name> \"Kraków 🚲\" .\n",
        )
        .replace('~', "\\");
        let store = RdfStore::new();
        store.load_ntriples(input.as_bytes()).unwrap();
        assert!(store.contains(&Triple::new(
            Term::iri("http://example.org/Kraków"),
            Term::iri("http://example.org/name"),
            Term::lang_literal("Kraków", "pl"),
        )));
        assert!(store.contains(&Triple::new(
            Term::blank("b0"),
            Term::iri("http://example.org/name"),
            Term::literal("Kraków 🚲"),
        )));
        let bad = "<http://example.org/a> <http://example.org/b> \"~x\" .\n".replace('~', "\\");
        assert!(matches!(
            RdfStore::new().load_ntriples(bad.as_bytes()),
            Err(NTriplesError::Parse { line: 1, .. })
        ));
    }

    /// A literal that ends in a backslash ran the N-Triples term scanner past
    /// the end of the line, which panicked; it is a parse error naming why.
    #[test]
    fn a_literal_ending_in_a_backslash_is_a_parse_error() {
        // `~` stands for a backslash.
        for line in [
            "<http://example.org/a> <http://example.org/b> \"abc~",
            "<http://example.org/a> <http://example.org/b> \"~",
            "<http://example.org/a> <http://example.org/b> \"Kraków~",
        ] {
            let line = line.replace('~', "\\");
            match RdfStore::new().load_ntriples(line.as_bytes()) {
                Err(NTriplesError::Parse { line: 1, content }) => assert!(
                    content.contains("an escape without its character"),
                    "{line}: {content}"
                ),
                other => panic!("{line}: {other:?}"),
            }
            match RdfStore::new().load_ntriples_streaming(line.as_bytes(), 3) {
                Err(NTriplesError::Parse { line: 1, content }) => assert!(
                    content.contains("an escape without its character"),
                    "{line}: {content}"
                ),
                other => panic!("{line}: {other:?}"),
            }
        }
    }

    /// A parse error says what is wrong with the line, not only which line.
    #[test]
    fn an_ntriples_parse_error_keeps_its_reason() {
        let bad = "<http://example.org/a> <http://example.org/b> \"~x\" .\n".replace('~', "\\");
        match RdfStore::new().load_ntriples(bad.as_bytes()) {
            Err(NTriplesError::Parse { line: 1, content }) => assert!(
                content.contains("object: not an N-Triples term (an unknown escape")
                    && content.contains("<http://example.org/a>"),
                "{content}"
            ),
            other => panic!("{other:?}"),
        }
        let short = "<http://example.org/a> <http://example.org/b> .\n";
        match RdfStore::new().load_ntriples(short.as_bytes()) {
            Err(NTriplesError::Parse { line: 1, content }) => {
                assert!(content.contains("three terms"), "{content}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// Named graphs start with empty indexes, which grow as triples arrive;
    /// the default graph keeps the capacity of its configuration. Each named
    /// graph used to preallocate about 1.3 MB.
    #[test]
    fn named_graphs_start_without_preallocated_indexes() {
        let store = RdfStore::new();
        assert!(
            store.index_allocation_bytes() > 0,
            "the default graph keeps its capacity"
        );
        for index in 0..300 {
            store.create_graph(&format!("http://example.org/trips/{index}"));
            store.graph_or_create(&format!("http://example.org/visits/{index}"));
        }
        assert_eq!(store.graph_count(), 600);
        for name in store.graph_names() {
            let graph = store.graph(&name).unwrap();
            assert_eq!(graph.index_allocation_bytes(), 0, "{name}");
        }
        let trips = store.graph("http://example.org/trips/3").unwrap();
        let visit = Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://example.org/visits"),
            Term::iri("http://example.org/prague"),
        );
        trips.insert(visit.clone());
        assert!(trips.contains(&visit));
        assert_eq!(trips.triples_with_subject(visit.subject()).len(), 1);
    }

    /// `term_order` calls two terms equal exactly when they are equal, so the
    /// order of `for_each_triple` never depends on the hash set's order.
    #[test]
    fn term_order_is_equal_exactly_for_equal_terms() {
        let mut terms = crate::graph::rdf::term::unusual_terms();
        terms.extend([
            Term::lang_literal("Praha", "cs"),
            Term::lang_literal("Praha", "sk"),
            Term::lang_literal("Praha", "CS"),
            Term::typed_literal("19", crate::graph::rdf::Literal::XSD_INTEGER),
            Term::typed_literal("19", crate::graph::rdf::Literal::XSD_DECIMAL),
            Term::literal("19"),
            Term::typed_literal("19", crate::graph::rdf::Literal::XSD_STRING),
            Term::iri("19"),
            Term::blank("19"),
        ]);
        for a in &terms {
            for b in &terms {
                let order = term_order(a, b);
                assert_eq!(order == std::cmp::Ordering::Equal, a == b, "{a} and {b}");
                assert_eq!(order, term_order(b, a).reverse(), "{a} and {b}");
            }
        }
    }

    #[test]
    fn test_insert_and_contains() {
        let store = RdfStore::new();
        let triples = sample_triples();

        for triple in &triples {
            assert!(store.insert(triple.clone()));
        }

        assert_eq!(store.len(), 4);

        for triple in &triples {
            assert!(store.contains(triple));
        }

        // Inserting duplicate should return false
        assert!(!store.insert(triples[0].clone()));
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn test_remove() {
        let store = RdfStore::new();
        let triples = sample_triples();

        for triple in &triples {
            store.insert(triple.clone());
        }

        assert!(store.remove(&triples[0]));
        assert_eq!(store.len(), 3);
        assert!(!store.contains(&triples[0]));

        // Removing non-existent should return false
        assert!(!store.remove(&triples[0]));
    }

    #[test]
    fn test_query_by_subject() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let alix = Term::iri("http://example.org/alix");
        let alice_triples = store.triples_with_subject(&alix);

        assert_eq!(alice_triples.len(), 3);
        for triple in &alice_triples {
            assert_eq!(triple.subject(), &alix);
        }
    }

    #[test]
    fn test_query_by_predicate() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let name_pred = Term::iri("http://xmlns.com/foaf/0.1/name");
        let name_triples = store.triples_with_predicate(&name_pred);

        assert_eq!(name_triples.len(), 2);
        for triple in &name_triples {
            assert_eq!(triple.predicate(), &name_pred);
        }
    }

    #[test]
    fn test_query_by_object() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let gus = Term::iri("http://example.org/gus");
        let bob_triples = store.triples_with_object(&gus);

        assert_eq!(bob_triples.len(), 1);
        assert_eq!(bob_triples[0].object(), &gus);
    }

    #[test]
    fn test_pattern_matching() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // Find all triples with subject alix and predicate knows
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/knows")),
            object: None,
        };

        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::iri("http://example.org/gus"));
    }

    #[test]
    fn test_stats() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let stats = store.stats();
        assert_eq!(stats.triple_count, 4);
        assert_eq!(stats.subject_count, 2); // alix, gus
        assert_eq!(stats.predicate_count, 3); // name, age, knows
    }

    #[test]
    fn test_clear() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        assert!(!store.is_empty());
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the deprecated per-transaction buffer until 0.7.0 removes it"
    )]
    fn test_find_with_pending_filters_deletes() {
        let store = RdfStore::new();
        let triples = sample_triples();

        // Insert all triples into committed storage
        for triple in &triples {
            store.insert(triple.clone());
        }

        // Create a transaction and add a pending delete
        let transaction_id = TransactionId::new(1);
        store.remove_in_transaction(transaction_id, triples[0].clone()); // Delete Alix's name triple

        // Query with transaction context - should NOT see the deleted triple
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: None,
            object: None,
        };

        let results = store.find_with_pending(&pattern, Some(transaction_id));
        assert_eq!(results.len(), 2); // Should be 2, not 3 (one deleted)

        // Verify the deleted triple is not in results
        let deleted = &triples[0];
        for result in &results {
            assert_ne!(result.as_ref(), deleted);
        }

        // Query without transaction context - should still see all 3
        let results_no_tx = store.find_with_pending(&pattern, None);
        assert_eq!(results_no_tx.len(), 3);

        // Verify pending inserts are still included
        let new_triple = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/email"),
            Term::literal("alix@example.org"),
        );
        store.insert_in_transaction(transaction_id, new_triple.clone());

        let results_with_insert = store.find_with_pending(&pattern, Some(transaction_id));
        assert_eq!(results_with_insert.len(), 3); // 2 committed - 1 deleted + 1 inserted

        // Verify the new triple is in results
        let found_new = results_with_insert
            .iter()
            .any(|t| t.as_ref() == &new_triple);
        assert!(found_new, "Pending insert should be visible");
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the deprecated per-transaction buffer until 0.7.0 removes it"
    )]
    fn test_find_with_pending_applies_ops_in_order() {
        let store = RdfStore::new();
        let committed = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/p"),
            Term::literal("committed"),
        );
        let fresh = Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/p"),
            Term::literal("fresh"),
        );
        store.insert(committed.clone());
        let tx = TransactionId::new(7);
        let all = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: None,
            object: None,
        };
        let seen = |store: &RdfStore| -> Vec<Triple> {
            let mut triples: Vec<Triple> = store
                .find_with_pending(&all, Some(tx))
                .into_iter()
                .map(|t| (*t).clone())
                .collect();
            triples.sort_by_key(|t| t.object().to_string());
            triples
        };

        // Inserting an already committed triple does not duplicate it.
        store.insert_in_transaction(tx, committed.clone());
        assert_eq!(seen(&store), vec![committed.clone()]);

        // Inserted then deleted: absent.
        store.insert_in_transaction(tx, fresh.clone());
        store.remove_in_transaction(tx, fresh.clone());
        assert_eq!(seen(&store), vec![committed.clone()]);
        assert!(!store.contains_with_pending(&fresh, tx));

        // Deleted then re-inserted: present once.
        store.remove_in_transaction(tx, committed.clone());
        assert!(!store.contains_with_pending(&committed, tx));
        store.insert_in_transaction(tx, committed.clone());
        assert_eq!(seen(&store), vec![committed.clone()]);
        assert!(store.contains_with_pending(&committed, tx));

        // Commit produces exactly what the transaction saw.
        store.commit_transaction(tx);
        assert_eq!(store.len(), 1);
        assert!(store.contains(&committed));
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the deprecated per-transaction buffer until 0.7.0 removes it"
    )]
    fn test_transaction_commit_and_rollback_reach_named_graphs() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://example.org/p"),
            Term::literal("named"),
        );
        let everything = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        // Rollback discards a buffered write in a named graph.
        let tx = TransactionId::new(1);
        store
            .graph_or_create("http://example.org/g")
            .insert_in_transaction(tx, triple.clone());
        let visible = store.find_in_graphs_with_pending(&everything, Some(&[]), Some(tx));
        assert_eq!(
            visible.len(),
            1,
            "the transaction sees its own named-graph write"
        );
        assert!(
            store.find_in_graphs(&everything, Some(&[])).is_empty(),
            "other readers do not"
        );
        assert_eq!(store.rollback_transaction(tx), 1);
        let graph = store.graph("http://example.org/g").unwrap();
        assert!(graph.is_empty());
        assert!(!graph.has_pending_ops(tx));

        // Commit applies a buffered write in a named graph.
        let tx = TransactionId::new(2);
        graph.insert_in_transaction(tx, triple.clone());
        assert_eq!(store.commit_transaction(tx), 1);
        assert!(graph.contains(&triple));
        assert!(!graph.has_pending_ops(tx));
    }

    #[test]
    fn test_named_graph_crud() {
        let store = RdfStore::new();

        // Create named graph
        assert!(store.create_graph("http://example.org/g1"));
        assert!(!store.create_graph("http://example.org/g1")); // already exists
        assert_eq!(store.graph_count(), 1);

        // Insert into named graph
        let g1 = store.graph("http://example.org/g1").unwrap();
        g1.insert(Triple::new(
            Term::iri("http://example.org/s1"),
            Term::iri("http://example.org/p1"),
            Term::literal("o1"),
        ));
        assert_eq!(g1.len(), 1);

        // Default graph is still empty
        assert_eq!(store.len(), 0);

        // Query named graph
        let results = g1.find(&TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        });
        assert_eq!(results.len(), 1);

        // Drop graph
        assert!(store.drop_graph("http://example.org/g1"));
        assert!(!store.drop_graph("http://example.org/g1"));
        assert_eq!(store.graph_count(), 0);
    }

    #[test]
    fn test_named_graph_isolation() {
        let store = RdfStore::new();

        // Insert into default graph
        store.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("default"),
        ));

        // Insert into named graph
        let g1 = store.graph_or_create("http://example.org/g1");
        g1.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("named"),
        ));

        // Each graph sees only its own triples
        assert_eq!(store.len(), 1);
        assert_eq!(g1.len(), 1);
        assert_eq!(store.triples()[0].object(), &Term::literal("default"));
        assert_eq!(g1.triples()[0].object(), &Term::literal("named"));
    }

    #[test]
    fn test_find_in_graphs() {
        let store = RdfStore::new();
        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        store.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("default"),
        ));

        let g1 = store.graph_or_create("http://example.org/g1");
        g1.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("g1"),
        ));

        // Default only
        let results = store.find_in_graphs(&pattern, None);
        assert_eq!(results.len(), 1);
        assert!(results[0].0.is_none());

        // All named graphs (excludes default)
        let results = store.find_in_graphs(&pattern, Some(&[]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_deref(), Some("http://example.org/g1"));

        // Specific named graph
        let results = store.find_in_graphs(&pattern, Some(&["http://example.org/g1"]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_deref(), Some("http://example.org/g1"));
    }

    /// Each whole-graph operation changes the graphs it names, and only
    /// them; `ALL` and `NAMED` reach every named graph, `ALL` the default
    /// graph too. An operation that finds nothing to change says so.
    #[test]
    fn a_graph_operation_changes_the_graphs_it_names() {
        let knows = |subject: &str| {
            Triple::new(
                Term::iri(format!("http://example.org/{subject}")),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/gus"),
            )
        };
        let paris = "http://example.org/paris";
        let berlin = "http://example.org/berlin";
        let filled = || {
            let store = RdfStore::new();
            store.insert_into(None, knows("alix"));
            store.insert_into(Some(paris), knows("vincent"));
            store.insert_into(Some(berlin), knows("mia"));
            store
        };
        // (graph, triple count) of the default graph, then of each named graph.
        let shape = |store: &RdfStore| {
            let mut named: Vec<(String, usize)> = store
                .graph_names()
                .into_iter()
                .map(|name| {
                    let len = store.graph(&name).unwrap().len();
                    (name, len)
                })
                .collect();
            named.sort();
            (store.len(), named)
        };
        let named = |graphs: &[(&str, usize)]| -> Vec<(String, usize)> {
            graphs
                .iter()
                .map(|(name, len)| ((*name).to_string(), *len))
                .collect()
        };
        let target = |name: &str| RdfGraphTarget::Named(name.to_string());

        let cases: Vec<(RdfGraphOp, (usize, Vec<(String, usize)>))> = vec![
            (
                RdfGraphOp::Clear {
                    target: target(paris),
                },
                (1, named(&[(berlin, 1), (paris, 0)])),
            ),
            (
                RdfGraphOp::Drop {
                    target: target(paris),
                },
                (1, named(&[(berlin, 1)])),
            ),
            (
                RdfGraphOp::Clear {
                    target: RdfGraphTarget::Default,
                },
                (0, named(&[(berlin, 1), (paris, 1)])),
            ),
            (
                RdfGraphOp::Drop {
                    target: RdfGraphTarget::AllNamed,
                },
                (1, Vec::new()),
            ),
            (
                RdfGraphOp::Clear {
                    target: RdfGraphTarget::All,
                },
                (0, Vec::new()),
            ),
            (
                RdfGraphOp::Copy {
                    source: Some(paris.to_string()),
                    target: None,
                },
                (1, named(&[(berlin, 1), (paris, 1)])),
            ),
            (
                RdfGraphOp::Move {
                    source: Some(paris.to_string()),
                    target: Some(berlin.to_string()),
                },
                (1, named(&[(berlin, 1)])),
            ),
            (
                RdfGraphOp::Add {
                    source: None,
                    target: Some(berlin.to_string()),
                },
                (1, named(&[(berlin, 2), (paris, 1)])),
            ),
        ];
        for (op, expected) in cases {
            let store = filled();
            assert!(store.apply_graph_op(&op), "{op:?}");
            assert_eq!(shape(&store), expected, "{op:?}");
        }

        let store = filled();
        assert!(!store.apply_graph_op(&RdfGraphOp::Create {
            name: paris.to_string()
        }));
        assert!(!store.apply_graph_op(&RdfGraphOp::Drop {
            target: target("http://example.org/prague"),
        }));
        assert_eq!(shape(&store), (1, named(&[(berlin, 1), (paris, 1)])));
    }

    /// A copy, move or add of a graph onto itself keeps the graph as it was:
    /// a move used to drop it (or clear the default graph).
    #[test]
    fn a_graph_onto_itself_is_kept() {
        let paris = "http://example.org/paris";
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://example.org/vincent"),
            Term::iri("http://example.org/knows"),
            Term::iri("http://example.org/mia"),
        );
        store.insert_into(None, triple.clone());
        store.insert_into(Some(paris), triple);
        for graph in [None, Some(paris)] {
            store.copy_graph(graph, graph);
            store.move_graph(graph, graph);
            store.add_graph(graph, graph);
        }
        assert_eq!(store.len(), 1);
        assert_eq!(store.graph(paris).map(|graph| graph.len()), Some(1));
        let prague = Some("http://example.org/prague");
        store.move_graph(prague, prague);
        assert!(store.graph("http://example.org/prague").is_none());
    }

    /// A triple applied to a named graph that does not exist creates it on
    /// insert and changes nothing on remove.
    #[test]
    fn a_triple_applies_to_its_graph() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://example.org/jules"),
            Term::iri("http://example.org/knows"),
            Term::iri("http://example.org/butch"),
        );
        let prague = Some("http://example.org/prague");
        assert!(!store.remove_from(prague, &triple));
        assert_eq!(
            store.graph_names(),
            Vec::<String>::new(),
            "a remove creates no graph"
        );
        assert!(store.insert_into(prague, triple.clone()));
        assert!(!store.insert_into(prague, triple.clone()));
        assert!(store.is_empty(), "the default graph holds nothing");
        assert!(store.remove_from(prague, &triple));
        assert!(!store.remove_from(prague, &triple));
    }

    #[test]
    fn test_copy_move_add_graph() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        );

        // Insert into default graph
        store.insert(triple.clone());

        // Copy default -> named
        store.copy_graph(None, Some("http://example.org/copy"));
        assert_eq!(store.len(), 1); // default still has it
        let copy = store.graph("http://example.org/copy").unwrap();
        assert_eq!(copy.len(), 1);

        // Add named -> another named (union)
        let g2 = store.graph_or_create("http://example.org/g2");
        g2.insert(Triple::new(
            Term::iri("http://example.org/s2"),
            Term::iri("http://example.org/p2"),
            Term::literal("extra"),
        ));
        store.add_graph(
            Some("http://example.org/copy"),
            Some("http://example.org/g2"),
        );
        assert_eq!(g2.len(), 2); // original + added

        // Move named -> named
        store.move_graph(Some("http://example.org/g2"), Some("http://example.org/g3"));
        assert!(store.graph("http://example.org/g2").is_none());
        let g3 = store.graph("http://example.org/g3").unwrap();
        assert_eq!(g3.len(), 2);
    }

    #[test]
    #[expect(
        deprecated,
        reason = "tests the deprecated per-transaction buffer until 0.7.0 removes it"
    )]
    fn test_transaction_commit_and_rollback() {
        let store = RdfStore::new();
        let triples = sample_triples();

        // Insert initial triples
        for triple in &triples {
            store.insert(triple.clone());
        }
        assert_eq!(store.len(), 4);

        // Test rollback
        let tx1 = TransactionId::new(1);
        store.remove_in_transaction(tx1, triples[0].clone());
        assert!(store.has_pending_ops(tx1));

        let discarded = store.rollback_transaction(tx1);
        assert_eq!(discarded, 1);
        assert!(!store.has_pending_ops(tx1));
        assert_eq!(store.len(), 4); // No change

        // Test commit
        let tx2 = TransactionId::new(2);
        store.remove_in_transaction(tx2, triples[0].clone());

        let applied = store.commit_transaction(tx2);
        assert_eq!(applied, 1);
        assert_eq!(store.len(), 3); // Triple removed
        assert!(!store.contains(&triples[0]));
    }

    #[test]
    fn test_batch_insert() {
        let store = RdfStore::new();
        let triples = sample_triples();

        let inserted = store.batch_insert(triples.clone());
        assert_eq!(inserted, 4);
        assert_eq!(store.len(), 4);

        // All triples should be queryable
        for triple in &triples {
            assert!(store.contains(triple));
        }

        // Indexes should be populated correctly
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_batch_insert_with_duplicates() {
        let store = RdfStore::new();

        // Insert one triple first
        let triples = sample_triples();
        store.insert(triples[0].clone());
        assert_eq!(store.len(), 1);

        // Batch insert all 4: only 3 should be new
        let inserted = store.batch_insert(triples.clone());
        assert_eq!(inserted, 3);
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn test_batch_insert_empty() {
        let store = RdfStore::new();
        let inserted = store.batch_insert(Vec::<Triple>::new());
        assert_eq!(inserted, 0);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_composite_index_sp_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+P bound: should use SP composite index (no filtering)
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Alix"));
    }

    #[test]
    fn test_composite_index_po_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // P+O bound: should use PO composite index
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject(), &Term::iri("http://example.org/alix"));
    }

    #[test]
    fn test_composite_index_os_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+O bound: should use OS composite index
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: None,
            object: Some(Term::iri("http://example.org/gus")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].predicate(),
            &Term::iri("http://xmlns.com/foaf/0.1/knows")
        );
    }

    #[test]
    fn test_composite_index_spo_lookup() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // S+P+O fully bound: existence check via SP composite
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);

        // Non-existent triple
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("NotAlix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_composite_index_removal() {
        let store = RdfStore::new();
        let triples = sample_triples();
        for triple in &triples {
            store.insert(triple.clone());
        }

        // Remove alix's name triple
        store.remove(&triples[0]);

        // SP lookup should no longer find it
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);

        // PO lookup should only find gus's name
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Alix")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_composite_index_batch_insert() {
        let store = RdfStore::new();
        let triples = sample_triples();
        store.batch_insert(triples);

        // Verify composite indexes are populated by batch_insert
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/knows")),
            object: None,
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::iri("http://example.org/gus"));

        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Gus")),
        };
        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bulk_load() {
        let store = RdfStore::new();

        // Insert some existing data that should be replaced
        store.insert(Triple::new(
            Term::iri("http://example.org/old"),
            Term::iri("http://example.org/p"),
            Term::literal("old"),
        ));

        let result = store.bulk_load(sample_triples());
        assert_eq!(result.triple_count, 4);
        assert_eq!(store.len(), 4);

        // Old data should be gone
        assert!(!store.contains(&Triple::new(
            Term::iri("http://example.org/old"),
            Term::iri("http://example.org/p"),
            Term::literal("old"),
        )));

        // All indexes should work (single-term)
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);

        // Composite indexes should work
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(store.find(&pattern).len(), 1);

        // Statistics should be computed
        assert_eq!(result.statistics.total_triples, 4);
        assert_eq!(result.statistics.subject_count, 2);
        assert_eq!(result.statistics.predicate_count, 3);
    }

    #[test]
    fn test_bulk_load_empty() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("v"),
        ));

        let result = store.bulk_load(Vec::<Triple>::new());
        assert_eq!(result.triple_count, 0);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_parse_ntriples_line() {
        // Simple IRI triple
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" ."#,
        );
        assert!(triple.is_ok());
        let triple = triple.unwrap();
        assert_eq!(triple.subject(), &Term::iri("http://example.org/alix"));
        assert_eq!(
            triple.predicate(),
            &Term::iri("http://xmlns.com/foaf/0.1/name")
        );
        assert_eq!(triple.object(), &Term::literal("Alix"));

        // Typed literal
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> ."#,
        );
        assert!(triple.is_ok());
        let triple = triple.unwrap();
        assert_eq!(
            triple.object(),
            &Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer")
        );

        // Language-tagged literal
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix"@en ."#,
        );
        assert!(triple.is_ok());
        assert_eq!(triple.unwrap().object(), &Term::lang_literal("Alix", "en"));

        // Blank node subject
        let triple = parse_ntriples_line(r#"_:b0 <http://xmlns.com/foaf/0.1/name> "Gus" ."#);
        assert!(triple.is_ok());
        assert_eq!(triple.unwrap().subject(), &Term::blank("b0"));

        // IRI object
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> ."#,
        );
        assert!(triple.is_ok());
        assert_eq!(
            triple.unwrap().object(),
            &Term::iri("http://example.org/gus")
        );

        // Invalid line (no dot)
        assert!(
            parse_ntriples_line(r#"<http://example.org/s> <http://example.org/p> "v""#,).is_err()
        );
    }

    #[test]
    fn test_load_ntriples() {
        let ntriples = "\
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> \"Alix\" .
# This is a comment
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .

<http://example.org/gus> <http://xmlns.com/foaf/0.1/name> \"Gus\" .
";
        let store = RdfStore::new();
        let result = store.load_ntriples(ntriples.as_bytes()).unwrap();
        assert_eq!(result.triple_count, 3);
        assert_eq!(store.len(), 3);
        assert_eq!(result.statistics.total_triples, 3);

        // Verify composite index works after load
        let pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: Some(Term::literal("Gus")),
        };
        assert_eq!(store.find(&pattern).len(), 1);
    }

    #[test]
    fn test_load_turtle_roundtrip() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix a foaf:Person ;
                foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        let result = store.load_turtle(turtle).unwrap();
        assert_eq!(result.triple_count, 4);
        assert_eq!(store.len(), 4);

        // Verify subject/predicate indexes are populated.
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_to_turtle_roundtrip() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        store.load_turtle(turtle).unwrap();
        assert_eq!(store.len(), 3);

        // Serialize to Turtle and re-parse.
        let output = store.to_turtle().unwrap();
        assert!(!output.is_empty(), "output is empty");

        let store2 = RdfStore::new();
        let result2 = store2.load_turtle(&output).unwrap();
        assert_eq!(result2.triple_count, 3);

        // Verify structural equivalence: same subject/predicate/object triples exist.
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alix")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store2.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Alix"));

        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/gus")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        let results = store2.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::literal("Gus"));
    }

    #[test]
    fn test_load_ntriples_parse_error() {
        let bad_ntriples = "\
<http://example.org/s> <http://example.org/p> \"ok\" .
this is not valid ntriples
<http://example.org/s2> <http://example.org/p2> \"ok2\" .
";
        let store = RdfStore::new();
        let result = store.load_ntriples(bad_ntriples.as_bytes());
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            NTriplesError::Parse { line, .. } => assert_eq!(line, 2),
            _ => panic!("expected Parse error"),
        }
    }

    // =========================================================================
    // Streaming load tests (TripleSink-based)
    // =========================================================================

    #[test]
    fn test_load_turtle_streaming_inserts_incrementally() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:alix a foaf:Person ;
                foaf:name "Alix" ;
                foaf:knows ex:gus .

            ex:gus foaf:name "Gus" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 2).unwrap();
        assert_eq!(count, 4);
        assert_eq!(store.len(), 4);

        // Verify indexes work
        let alix = Term::iri("http://example.org/alix");
        assert_eq!(store.triples_with_subject(&alix).len(), 3);
    }

    #[test]
    fn test_load_turtle_streaming_does_not_replace_existing() {
        let store = RdfStore::new();
        // Pre-load one triple
        store.insert(Triple::new(
            Term::iri("http://example.org/existing"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        ));
        assert_eq!(store.len(), 1);

        let turtle = r#"
            <http://example.org/new> <http://example.org/p> "added" .
        "#;
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        assert_eq!(count, 1);
        // Both the existing and new triple should be present
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_load_turtle_streaming_deduplicates() {
        let turtle = r#"
            <http://example.org/s> <http://example.org/p> "o" .
            <http://example.org/s> <http://example.org/p> "o" .
            <http://example.org/s> <http://example.org/p> "o" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        assert_eq!(count, 1, "duplicates should be filtered by batch_insert");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_load_ntriples_streaming() {
        let ntriples = "\
<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> \"Alix\" .
<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
<http://example.org/gus> <http://xmlns.com/foaf/0.1/name> \"Gus\" .
";
        let store = RdfStore::new();
        let count = store
            .load_ntriples_streaming(ntriples.as_bytes(), 2)
            .unwrap();
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_load_ntriples_streaming_does_not_replace_existing() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/existing"),
            Term::iri("http://example.org/p"),
            Term::literal("value"),
        ));

        let ntriples = "<http://example.org/new> <http://example.org/p> \"added\" .\n";
        let count = store
            .load_ntriples_streaming(ntriples.as_bytes(), 100)
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_load_turtle_reader() {
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alix ex:name "Alix" .
            ex:gus ex:name "Gus" .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_reader(turtle.as_bytes(), 100).unwrap();
        assert_eq!(count, 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_parse_into_with_count_sink() {
        use crate::graph::rdf::sink::CountSink;
        use crate::graph::rdf::turtle::TurtleParser;

        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:a ex:p "1" .
            ex:b ex:p "2" .
            ex:c ex:p "3" .
        "#;

        let mut sink = CountSink::new();
        let mut parser = TurtleParser::new();
        parser.parse_into(turtle, &mut sink).unwrap();
        assert_eq!(sink.count(), 3);
    }

    #[test]
    fn test_streaming_turtle_with_collections() {
        // Collections emit rdf:first/rdf:rest triples through the sink
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:list ex:items ( "a" "b" "c" ) .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        // 1 (ex:list ex:items _:head) + 3*(first+rest) = 7
        assert_eq!(count, 7);
        assert_eq!(store.len(), 7);
    }

    #[test]
    fn test_streaming_turtle_with_blank_node_property_list() {
        // Blank node property lists emit triples through the sink
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:alix ex:address [ ex:city "Amsterdam" ; ex:country "NL" ] .
        "#;

        let store = RdfStore::new();
        let count = store.load_turtle_streaming(turtle, 100).unwrap();
        // 1 (alix address _:b) + 2 (city, country) = 3
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_built_during_bulk_load() {
        let store = RdfStore::new();
        assert!(
            store.ring().is_none(),
            "ring should not exist before bulk load"
        );

        store.bulk_load(vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/knows"),
                Term::iri("http://ex.org/gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/knows"),
                Term::iri("http://ex.org/alix"),
            ),
        ]);

        let ring = store.ring().expect("ring should exist after bulk load");
        assert_eq!(ring.len(), 2);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_stale_after_insert() {
        let store = RdfStore::new();
        store.bulk_load(vec![Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::iri("http://ex.org/b"),
        )]);
        assert!(store.ring().is_some());

        // Incremental insert marks ring stale
        store.insert(Triple::new(
            Term::iri("http://ex.org/c"),
            Term::iri("http://ex.org/p"),
            Term::iri("http://ex.org/d"),
        ));
        assert!(
            store.ring().is_none(),
            "ring should be stale after incremental insert"
        );

        // Rebuild restores ring
        store.rebuild_ring();
        let ring = store.ring().expect("ring should exist after rebuild");
        assert_eq!(ring.len(), 2);
    }

    #[test]
    #[cfg(feature = "ring-index")]
    fn test_ring_find_matches_store_find() {
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ];

        let store = RdfStore::new();
        store.bulk_load(triples);

        let ring = store.ring().expect("ring should exist");

        // Fully unbound: ring.count should match store.len
        let all_pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };
        assert_eq!(ring.count(&all_pattern), store.len());

        // Predicate-bound: count names
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);

        // Subject-bound: triples about alix
        let alix_pattern = TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: None,
            object: None,
        };
        assert_eq!(ring.count(&alix_pattern), 2);
    }

    #[test]
    fn test_batch_insert_composite_indexes() {
        let store = RdfStore::new();
        let triples = vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://ex.org/age"),
                Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
            ),
        ];
        let inserted = store.batch_insert(triples);
        assert_eq!(inserted, 3);
        assert_eq!(store.len(), 3);

        // Verify composite indexes via find
        let sp_result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: Some(Term::iri("http://ex.org/name")),
            object: None,
        });
        assert_eq!(sp_result.len(), 1);
        let po_result = store.find(&TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://ex.org/name")),
            object: Some(Term::literal("Gus")),
        });
        assert_eq!(po_result.len(), 1);
        let os_result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/alix")),
            predicate: None,
            object: Some(Term::literal("Alix")),
        });
        assert_eq!(os_result.len(), 1);
    }

    #[test]
    fn test_batch_insert_deduplication() {
        let store = RdfStore::new();
        let triple = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/b"),
            Term::literal("c"),
        );
        let inserted = store.batch_insert(vec![triple.clone(), triple.clone(), triple]);
        assert_eq!(inserted, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_composite_index_after_remove() {
        let store = RdfStore::new();
        let t1 = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v1"),
        );
        let t2 = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v2"),
        );
        store.insert(t1.clone());
        store.insert(t2);
        assert_eq!(
            store
                .find(&TriplePattern {
                    subject: Some(Term::iri("http://ex.org/a")),
                    predicate: Some(Term::iri("http://ex.org/p")),
                    object: None
                })
                .len(),
            2
        );
        store.remove(&t1);
        assert_eq!(
            store
                .find(&TriplePattern {
                    subject: Some(Term::iri("http://ex.org/a")),
                    predicate: Some(Term::iri("http://ex.org/p")),
                    object: None
                })
                .len(),
            1
        );
    }

    #[test]
    fn test_named_graph_operations() {
        let store = RdfStore::new();
        assert!(store.create_graph("http://ex.org/g1"));
        assert!(!store.create_graph("http://ex.org/g1")); // already exists
        let g = store.graph("http://ex.org/g1").unwrap();
        g.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/b"),
            Term::literal("c"),
        ));
        assert_eq!(g.len(), 1);
        assert!(store.drop_graph("http://ex.org/g1"));
        assert!(store.graph("http://ex.org/g1").is_none());
    }

    #[test]
    fn test_statistics_cache_invalidation() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        ));
        let stats1 = store.get_or_collect_statistics();
        // Term::iri.to_string() produces "<http://ex.org/p>"
        assert!(stats1.get_predicate("<http://ex.org/p>").is_some());
        // Insert more data: cache should be invalidated
        store.insert(Triple::new(
            Term::iri("http://ex.org/b"),
            Term::iri("http://ex.org/q"),
            Term::literal("w"),
        ));
        let stats2 = store.get_or_collect_statistics();
        assert!(stats2.get_predicate("<http://ex.org/q>").is_some());
    }

    #[test]
    fn test_find_three_bound() {
        let store = RdfStore::new();
        let t = Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("v"),
        );
        store.insert(t.clone());
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            Term::iri("http://ex.org/p"),
            Term::literal("other"),
        ));
        let result = store.find(&TriplePattern {
            subject: Some(Term::iri("http://ex.org/a")),
            predicate: Some(Term::iri("http://ex.org/p")),
            object: Some(Term::literal("v")),
        });
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_ref(), &t);
    }
}
