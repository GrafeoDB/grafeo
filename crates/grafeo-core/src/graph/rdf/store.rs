//! RDF Triple Store.
//!
//! Provides an in-memory triple store with efficient indexing for
//! subject, predicate, and object queries.

use super::sink::TripleSink;
use super::term::Term;
use super::triple::{Triple, TriplePattern};
use grafeo_common::types::TransactionId;
use grafeo_common::utils::hash::FxHashSet;
use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

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
/// The store also supports transactional operations through buffering.
/// When operations are performed within a transaction context, they are
/// buffered until commit (applied) or rollback (discarded).
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
    /// Transaction buffers for pending operations.
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
            let triple = parse_ntriples_line(trimmed).ok_or_else(|| NTriplesError::Parse {
                line: line_no + 1,
                content: line.clone(),
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
            let triple = parse_ntriples_line(trimmed).ok_or_else(|| NTriplesError::Parse {
                line: line_no + 1,
                content: line.clone(),
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
                .or_insert_with(|| Arc::new(RdfStore::with_config(self.config.clone()))),
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
            Arc::new(RdfStore::with_config(self.config.clone())),
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

    /// Copies all triples from source graph to destination graph.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) {
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

    /// Moves all triples from source graph to destination graph.
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn move_graph(&self, source: Option<&str>, dest: Option<&str>) {
        self.copy_graph(source, dest);
        match source {
            None => self.clear(),
            Some(n) => {
                self.drop_graph(n);
            }
        }
    }

    /// Adds all triples from source graph into destination graph (union).
    ///
    /// `None` = default graph, `Some(iri)` = named graph.
    pub fn add_graph(&self, source: Option<&str>, dest: Option<&str>) {
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
                .find_with_pending(pattern, tx)
                .into_iter()
                .map(|t| (None, t))
                .collect(),
            Some([]) => {
                let mut results = Vec::new();
                for (name, store) in self.named_graphs.read().iter() {
                    for t in store.find_with_pending(pattern, tx) {
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
                        for t in store.find_with_pending(pattern, tx) {
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
    pub fn find_with_pending(
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
    #[must_use]
    pub fn contains_with_pending(&self, triple: &Triple, transaction_id: TransactionId) -> bool {
        let pattern = TriplePattern {
            subject: Some(triple.subject().clone()),
            predicate: Some(triple.predicate().clone()),
            object: Some(triple.object().clone()),
        };
        !self
            .find_with_pending(&pattern, Some(transaction_id))
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
        /// The raw line content.
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
                pos += 2; // skip escape sequence
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
fn parse_ntriples_line(line: &str) -> Option<Triple> {
    let (subj_str, rest) = next_ntriples_term(line)?;
    let (pred_str, rest) = next_ntriples_term(rest)?;
    let (obj_str, rest) = next_ntriples_term(rest)?;

    // Expect trailing ` .`
    let rest = rest.trim();
    if !rest.starts_with('.') {
        return None;
    }

    let subject = Term::from_ntriples(subj_str)?;
    let predicate = Term::from_ntriples(pred_str)?;
    let object = Term::from_ntriples(obj_str)?;
    Some(Triple::new(subject, predicate, object))
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
        assert!(triple.is_some());
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
        assert!(triple.is_some());
        let triple = triple.unwrap();
        assert_eq!(
            triple.object(),
            &Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer")
        );

        // Language-tagged literal
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix"@en ."#,
        );
        assert!(triple.is_some());
        assert_eq!(triple.unwrap().object(), &Term::lang_literal("Alix", "en"));

        // Blank node subject
        let triple = parse_ntriples_line(r#"_:b0 <http://xmlns.com/foaf/0.1/name> "Gus" ."#);
        assert!(triple.is_some());
        assert_eq!(triple.unwrap().subject(), &Term::blank("b0"));

        // IRI object
        let triple = parse_ntriples_line(
            r#"<http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> ."#,
        );
        assert!(triple.is_some());
        assert_eq!(
            triple.unwrap().object(),
            &Term::iri("http://example.org/gus")
        );

        // Invalid line (no dot)
        assert!(
            parse_ntriples_line(r#"<http://example.org/s> <http://example.org/p> "v""#,).is_none()
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
        assert!(!output.is_empty());

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
