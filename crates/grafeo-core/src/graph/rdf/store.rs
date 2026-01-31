//! RDF Triple Store.
//!
//! Provides an in-memory triple store with efficient indexing for
//! subject, predicate, and object queries.

use super::term::Term;
use super::triple::{Triple, TriplePattern};
use grafeo_common::types::TxId;
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
    buffers: HashMap<TxId, Vec<PendingOp>>,
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
    subject_index: RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, ahash::RandomState>>,
    /// Predicate index: predicate -> triples.
    predicate_index: RwLock<hashbrown::HashMap<Term, Vec<Arc<Triple>>, ahash::RandomState>>,
    /// Object index: object -> triples (optional).
    object_index: RwLock<Option<hashbrown::HashMap<Term, Vec<Arc<Triple>>, ahash::RandomState>>>,
    /// Transaction buffers for pending operations.
    tx_buffer: RwLock<TransactionBuffer>,
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
                ahash::RandomState::new(),
            ))
        } else {
            None
        };

        Self {
            triples: RwLock::new(FxHashSet::default()),
            subject_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                ahash::RandomState::new(),
            )),
            predicate_index: RwLock::new(hashbrown::HashMap::with_capacity_and_hasher(
                config.initial_capacity,
                ahash::RandomState::new(),
            )),
            object_index: RwLock::new(object_index),
            tx_buffer: RwLock::new(TransactionBuffer::default()),
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
                    .push(triple);
            }
        }

        true
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
            if let Some(ref mut index) = *object_index {
                if let Some(vec) = index.get_mut(triple.object()) {
                    vec.retain(|t| t.as_ref() != triple);
                    if vec.is_empty() {
                        index.remove(triple.object());
                    }
                }
            }
        }

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
    pub fn find(&self, pattern: &TriplePattern) -> Vec<Arc<Triple>> {
        // Use the most selective index
        match (&pattern.subject, &pattern.predicate, &pattern.object) {
            (Some(s), _, _) => {
                // Use subject index
                let index = self.subject_index.read();
                if let Some(triples) = index.get(s) {
                    triples
                        .iter()
                        .filter(|t| pattern.matches(t))
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                }
            }
            (None, Some(p), _) => {
                // Use predicate index
                let index = self.predicate_index.read();
                if let Some(triples) = index.get(p) {
                    triples
                        .iter()
                        .filter(|t| pattern.matches(t))
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                }
            }
            (None, None, Some(o)) if self.config.index_objects => {
                // Use object index
                let index = self.object_index.read();
                if let Some(ref idx) = *index {
                    if let Some(triples) = idx.get(o) {
                        triples
                            .iter()
                            .filter(|t| pattern.matches(t))
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            }
            _ => {
                // Full scan
                self.triples
                    .read()
                    .iter()
                    .filter(|t| pattern.matches(t))
                    .cloned()
                    .collect()
            }
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
    }

    /// Returns store statistics.
    #[must_use]
    pub fn stats(&self) -> RdfStoreStats {
        RdfStoreStats {
            triple_count: self.len(),
            subject_count: self.subject_index.read().len(),
            predicate_count: self.predicate_index.read().len(),
            object_count: if self.config.index_objects {
                self.object_index
                    .read()
                    .as_ref()
                    .map(|i| i.len())
                    .unwrap_or(0)
            } else {
                0
            },
        }
    }

    // =========================================================================
    // Transaction support
    // =========================================================================

    /// Inserts a triple within a transaction context.
    ///
    /// The insert is buffered until the transaction is committed.
    /// If the transaction is rolled back, the insert is discarded.
    pub fn insert_in_tx(&self, tx_id: TxId, triple: Triple) {
        let mut buffer = self.tx_buffer.write();
        buffer
            .buffers
            .entry(tx_id)
            .or_default()
            .push(PendingOp::Insert(triple));
    }

    /// Removes a triple within a transaction context.
    ///
    /// The removal is buffered until the transaction is committed.
    /// If the transaction is rolled back, the removal is discarded.
    pub fn remove_in_tx(&self, tx_id: TxId, triple: Triple) {
        let mut buffer = self.tx_buffer.write();
        buffer
            .buffers
            .entry(tx_id)
            .or_default()
            .push(PendingOp::Delete(triple));
    }

    /// Commits a transaction, applying all buffered operations.
    ///
    /// Returns the number of operations applied.
    pub fn commit_tx(&self, tx_id: TxId) -> usize {
        let ops = {
            let mut buffer = self.tx_buffer.write();
            buffer.buffers.remove(&tx_id).unwrap_or_default()
        };

        let count = ops.len();
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
        count
    }

    /// Rolls back a transaction, discarding all buffered operations.
    ///
    /// Returns the number of operations discarded.
    pub fn rollback_tx(&self, tx_id: TxId) -> usize {
        let mut buffer = self.tx_buffer.write();
        buffer
            .buffers
            .remove(&tx_id)
            .map(|ops| ops.len())
            .unwrap_or(0)
    }

    /// Checks if a transaction has pending operations.
    #[must_use]
    pub fn has_pending_ops(&self, tx_id: TxId) -> bool {
        let buffer = self.tx_buffer.read();
        buffer
            .buffers
            .get(&tx_id)
            .map(|ops| !ops.is_empty())
            .unwrap_or(false)
    }

    /// Returns triples matching the given pattern, including pending inserts
    /// from the specified transaction (for read-your-writes within a transaction).
    pub fn find_with_pending(
        &self,
        pattern: &TriplePattern,
        tx_id: Option<TxId>,
    ) -> Vec<Arc<Triple>> {
        let mut results = self.find(pattern);

        // Include pending inserts from the current transaction
        if let Some(tx) = tx_id {
            let buffer = self.tx_buffer.read();
            if let Some(ops) = buffer.buffers.get(&tx) {
                for op in ops {
                    if let PendingOp::Insert(triple) = op {
                        if pattern.matches(triple) {
                            results.push(Arc::new(triple.clone()));
                        }
                    }
                }
            }
        }

        results
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<Triple> {
        vec![
            Triple::new(
                Term::iri("http://example.org/alice"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alice"),
            ),
            Triple::new(
                Term::iri("http://example.org/alice"),
                Term::iri("http://xmlns.com/foaf/0.1/age"),
                Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
            ),
            Triple::new(
                Term::iri("http://example.org/alice"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/bob"),
            ),
            Triple::new(
                Term::iri("http://example.org/bob"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Bob"),
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

        let alice = Term::iri("http://example.org/alice");
        let alice_triples = store.triples_with_subject(&alice);

        assert_eq!(alice_triples.len(), 3);
        for triple in &alice_triples {
            assert_eq!(triple.subject(), &alice);
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

        let bob = Term::iri("http://example.org/bob");
        let bob_triples = store.triples_with_object(&bob);

        assert_eq!(bob_triples.len(), 1);
        assert_eq!(bob_triples[0].object(), &bob);
    }

    #[test]
    fn test_pattern_matching() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        // Find all triples with subject alice and predicate knows
        let pattern = TriplePattern {
            subject: Some(Term::iri("http://example.org/alice")),
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/knows")),
            object: None,
        };

        let results = store.find(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object(), &Term::iri("http://example.org/bob"));
    }

    #[test]
    fn test_stats() {
        let store = RdfStore::new();
        for triple in sample_triples() {
            store.insert(triple);
        }

        let stats = store.stats();
        assert_eq!(stats.triple_count, 4);
        assert_eq!(stats.subject_count, 2); // alice, bob
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
}
