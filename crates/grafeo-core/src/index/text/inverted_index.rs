//! BM25-scored inverted index for full-text search.

use super::tokenizer::{SimpleTokenizer, Tokenizer};
use grafeo_common::types::NodeId;
use std::collections::HashMap;

/// Configuration for BM25 scoring.
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Term frequency saturation parameter (default 1.2).
    ///
    /// Higher values give more weight to term frequency.
    pub k1: f64,
    /// Length normalization parameter (default 0.75).
    ///
    /// 0.0 = no length normalization, 1.0 = full normalization.
    pub b: f64,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// A posting entry: document ID and term frequency.
#[derive(Debug, Clone)]
struct Posting {
    node_id: NodeId,
    term_freq: u32,
}

/// A posting list for a single term.
#[derive(Debug, Clone, Default)]
struct PostingList {
    postings: Vec<Posting>,
}

/// An in-memory inverted index with Okapi BM25 scoring.
///
/// Supports insert, remove, and ranked search operations. Designed
/// for indexing text properties on graph nodes.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "text-index")]
/// # {
/// use grafeo_core::index::text::{InvertedIndex, BM25Config};
/// use grafeo_common::types::NodeId;
///
/// let mut index = InvertedIndex::new(BM25Config::default());
/// index.insert(NodeId::new(1), "rust graph database");
/// index.insert(NodeId::new(2), "python web framework");
///
/// let results = index.search("graph database", 10);
/// assert_eq!(results[0].0, NodeId::new(1));
/// # }
/// ```
pub struct InvertedIndex {
    /// Term → posting list.
    postings: HashMap<String, PostingList>,
    /// Document lengths (in tokens).
    doc_lengths: HashMap<NodeId, u32>,
    /// Sum of all document lengths (for average calculation).
    total_length: u64,
    /// Tokenizer used for indexing and querying.
    tokenizer: Box<dyn Tokenizer>,
    /// BM25 configuration.
    config: BM25Config,
}

impl InvertedIndex {
    /// Creates a new inverted index with the given BM25 configuration.
    #[must_use]
    pub fn new(config: BM25Config) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            tokenizer: Box::new(SimpleTokenizer::new()),
            config,
        }
    }

    /// Creates a new inverted index with a custom tokenizer.
    pub fn with_tokenizer(config: BM25Config, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            tokenizer,
            config,
        }
    }

    /// Indexes a document (node text) into the inverted index.
    ///
    /// If the node was already indexed, it is first removed and re-indexed.
    pub fn insert(&mut self, id: NodeId, text: &str) {
        // Remove existing entry if present
        if self.doc_lengths.contains_key(&id) {
            self.remove(id);
        }

        let tokens = self.tokenizer.tokenize(text);
        // reason: document token count fits u32 for practical text sizes
        #[allow(clippy::cast_possible_truncation)]
        let doc_len = tokens.len() as u32;

        if doc_len == 0 {
            return;
        }

        // Count term frequencies
        let mut term_freqs: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.as_str()).or_insert(0) += 1;
        }

        // Add to posting lists
        for (term, freq) in term_freqs {
            self.postings
                .entry(term.to_string())
                .or_default()
                .postings
                .push(Posting {
                    node_id: id,
                    term_freq: freq,
                });
        }

        self.doc_lengths.insert(id, doc_len);
        self.total_length += u64::from(doc_len);
    }

    /// Removes a document from the index.
    ///
    /// Returns `true` if the document was found and removed.
    pub fn remove(&mut self, id: NodeId) -> bool {
        let Some(doc_len) = self.doc_lengths.remove(&id) else {
            return false;
        };

        self.total_length -= u64::from(doc_len);

        // Remove from all posting lists
        self.postings.retain(|_, list| {
            list.postings.retain(|p| p.node_id != id);
            !list.postings.is_empty()
        });

        true
    }

    /// BM25 term score: IDF * TF-component for a single term occurrence.
    ///
    /// `df` is the document frequency (number of documents containing the term),
    /// `tf` is the term frequency in this document, `dl` is the document length,
    /// `n` is the corpus size, and `avg_dl` is the average document length.
    #[inline]
    fn bm25_term_score(&self, df: f64, tf: f64, dl: f64, n: f64, avg_dl: f64) -> f64 {
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        let tf_component = (tf * (self.config.k1 + 1.0))
            / (tf + self.config.k1 * (1.0 - self.config.b + self.config.b * dl / avg_dl));
        idf * tf_component
    }

    /// Searches the index using BM25 scoring.
    ///
    /// Returns up to `k` results sorted by descending BM25 score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(NodeId, f64)> {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() || self.doc_lengths.is_empty() {
            return Vec::new();
        }

        let n = self.doc_lengths.len() as f64;
        let avg_dl = self.total_length as f64 / n;
        let mut scores: HashMap<NodeId, f64> = HashMap::new();

        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list.postings.len() as f64;
            for posting in &posting_list.postings {
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(self.doc_lengths.get(&posting.node_id).copied().unwrap_or(0));
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n, avg_dl);
            }
        }

        let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Scores a single document against a query using BM25.
    ///
    /// Looks up each query term in its posting list, finds the entry for the
    /// given node ID, and computes BM25 with corpus statistics. Returns `0.0`
    /// if the document has no matching terms or doesn't exist.
    ///
    /// This is O(query_terms) per call and is intended for per-row evaluation.
    #[must_use]
    pub fn score_document(&self, id: NodeId, query: &str) -> f64 {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() || self.doc_lengths.is_empty() {
            return 0.0;
        }
        let Some(&doc_len) = self.doc_lengths.get(&id) else {
            return 0.0;
        };
        let n = self.doc_lengths.len() as f64;
        let avg_dl = self.total_length as f64 / n;
        let dl = f64::from(doc_len);
        let mut score = 0.0;
        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list.postings.len() as f64;
            let tf = posting_list
                .postings
                .iter()
                .find(|p| p.node_id == id)
                .map_or(0.0, |p| f64::from(p.term_freq));
            if tf > 0.0 {
                score += self.bm25_term_score(df, tf, dl, n, avg_dl);
            }
        }
        score
    }

    /// Returns all documents scoring at or above `threshold` using BM25.
    ///
    /// Unlike [`search()`] (top-k), this returns every document above the
    /// threshold, sorted by score descending. Intended for index-accelerated
    /// text search with WHERE predicates.
    #[must_use]
    pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() || self.doc_lengths.is_empty() {
            return Vec::new();
        }
        let n = self.doc_lengths.len() as f64;
        let avg_dl = self.total_length as f64 / n;
        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list.postings.len() as f64;
            for posting in &posting_list.postings {
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(self.doc_lengths.get(&posting.node_id).copied().unwrap_or(0));
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n, avg_dl);
            }
        }
        let mut results: Vec<(NodeId, f64)> = scores
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Returns true if the given node is indexed.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.doc_lengths.contains_key(&id)
    }

    /// Returns the number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_lengths.is_empty()
    }

    /// Returns the number of unique terms in the index.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Returns the BM25 configuration.
    #[must_use]
    pub fn config(&self) -> &BM25Config {
        &self.config
    }

    /// Snapshot the index for serialization.
    ///
    /// Returns (postings, doc_lengths, total_length) where postings is
    /// a vec of (term, vec of (node_id, term_freq)).
    #[must_use]
    pub fn snapshot(&self) -> (Vec<(String, Vec<(NodeId, u32)>)>, Vec<(NodeId, u32)>, u64) {
        let mut postings: Vec<(String, Vec<(NodeId, u32)>)> = self
            .postings
            .iter()
            .map(|(term, pl)| {
                let entries: Vec<(NodeId, u32)> = pl
                    .postings
                    .iter()
                    .map(|p| (p.node_id, p.term_freq))
                    .collect();
                (term.clone(), entries)
            })
            .collect();
        postings.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut doc_lengths: Vec<(NodeId, u32)> = self
            .doc_lengths
            .iter()
            .map(|(id, len)| (*id, *len))
            .collect();
        doc_lengths.sort_by_key(|(id, _)| *id);

        (postings, doc_lengths, self.total_length)
    }

    /// Override the BM25 configuration parameters.
    pub fn set_config(&mut self, config: BM25Config) {
        self.config = config;
    }

    /// Restore the index from a snapshot. Replaces all current data.
    pub fn restore(
        &mut self,
        postings: Vec<(String, Vec<(NodeId, u32)>)>,
        doc_lengths: Vec<(NodeId, u32)>,
        total_length: u64,
    ) {
        self.postings.clear();
        for (term, entries) in postings {
            let posting_list = PostingList {
                postings: entries
                    .into_iter()
                    .map(|(node_id, term_freq)| Posting { node_id, term_freq })
                    .collect(),
            };
            self.postings.insert(term, posting_list);
        }
        self.doc_lengths = doc_lengths.into_iter().collect();
        self.total_length = total_length;
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        // Postings map: term strings + PostingList vecs
        let postings_overhead = self.postings.capacity()
            * (std::mem::size_of::<String>() + std::mem::size_of::<PostingList>() + 1);
        let postings_data: usize = self
            .postings
            .iter()
            .map(|(term, pl)| term.len() + pl.postings.capacity() * std::mem::size_of::<Posting>())
            .sum();
        // Doc lengths map
        let doc_lengths_bytes = self.doc_lengths.capacity()
            * (std::mem::size_of::<NodeId>() + std::mem::size_of::<u32>() + 1);
        postings_overhead + postings_data + doc_lengths_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(
            NodeId::new(1),
            "the quick brown fox jumps over the lazy dog",
        );
        index.insert(NodeId::new(2), "a fast red car drives on the highway");
        index.insert(NodeId::new(3), "the brown dog sleeps all day");

        let results = index.search("brown dog", 10);
        assert!(!results.is_empty());
        // Node 3 mentions both "brown" and "dog" in a shorter document
        assert_eq!(results[0].0, NodeId::new(3));
    }

    #[test]
    fn test_empty_index_search() {
        let index = InvertedIndex::new(BM25Config::default());
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_stop_word_only_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("the a an", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        index.insert(NodeId::new(2), "hello rust");

        assert_eq!(index.len(), 2);
        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);

        let results = index.search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(2));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = InvertedIndex::new(BM25Config::default());
        assert!(!index.remove(NodeId::new(999)));
    }

    #[test]
    fn test_reinsert() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "old text");
        index.insert(NodeId::new(1), "new text completely different");

        assert_eq!(index.len(), 1);
        let results = index.search("old", 10);
        assert!(results.is_empty());

        let results = index.search("completely different", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_contains() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");

        assert!(index.contains(NodeId::new(1)));
        assert!(!index.contains(NodeId::new(2)));
    }

    #[test]
    fn test_term_count() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        index.insert(NodeId::new(2), "hello rust");

        // "hello", "world", "rust" (stop words removed)
        assert_eq!(index.term_count(), 3);
    }

    #[test]
    fn test_k_limit() {
        let mut index = InvertedIndex::new(BM25Config::default());
        for i in 1..=10 {
            index.insert(NodeId::new(i), &format!("document number {}", i));
        }

        let results = index.search("document", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_bm25_scoring_prefers_shorter_docs() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // Short doc with the term
        index.insert(NodeId::new(1), "rust database");
        // Long doc with the same term buried in noise
        index.insert(
            NodeId::new(2),
            "rust programming language systems web server framework database engine query optimizer",
        );

        let results = index.search("rust database", 10);
        assert_eq!(results.len(), 2);
        // Shorter doc should score higher (length normalization)
        assert_eq!(results[0].0, NodeId::new(1));
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_no_match() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("nonexistent term", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_idf_weighting() {
        let mut index = InvertedIndex::new(BM25Config::default());
        // "common" appears in all docs, "rare" only in one
        index.insert(NodeId::new(1), "common rare word");
        index.insert(NodeId::new(2), "common another word");
        index.insert(NodeId::new(3), "common third word");

        let results = index.search("rare", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));

        // "common" matches all three
        let results = index.search("common", 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_score_document_matches_search() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "the quick brown fox jumps over the lazy dog");
        index.insert(NodeId::new(2), "a fast red car drives on the highway");
        index.insert(NodeId::new(3), "the brown dog sleeps all day");

        let query = "brown dog";
        let search_results = index.search(query, 10);

        // Verify score_document returns the same score as search() for matching docs
        for (node_id, search_score) in &search_results {
            let doc_score = index.score_document(*node_id, query);
            assert!(
                (doc_score - search_score).abs() < 1e-10,
                "score_document({:?}) = {doc_score} but search gave {search_score}",
                node_id
            );
        }

        // Node 2 has no matching terms — should score 0.0
        let no_match_score = index.score_document(NodeId::new(2), query);
        assert_eq!(no_match_score, 0.0, "non-matching doc should score 0.0");

        // Non-existent doc should score 0.0
        let nonexistent_score = index.score_document(NodeId::new(999), query);
        assert_eq!(nonexistent_score, 0.0, "non-existent doc should score 0.0");
    }

    #[test]
    fn test_search_with_threshold() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database query engine");
        index.insert(NodeId::new(2), "python web framework django flask");
        index.insert(NodeId::new(3), "rust systems programming language");
        index.insert(NodeId::new(4), "graph theory algorithms data structures");
        index.insert(NodeId::new(5), "database indexing storage engine optimization");

        let query = "rust graph database";

        // Get search results to calibrate the threshold
        let search_results = index.search(query, 10);
        assert!(search_results.len() >= 2, "need at least 2 matching docs for this test");

        // Use the score of the second-highest result as our mid threshold
        let mid_threshold = search_results[1].1;

        // threshold=0 should return all matching docs (same set as search with no k limit)
        let all_results = index.search_with_threshold(query, 0.0);
        assert_eq!(
            all_results.len(),
            search_results.len(),
            "threshold=0 should return all matching docs"
        );

        // Results should be sorted descending by score
        for i in 1..all_results.len() {
            assert!(
                all_results[i - 1].1 >= all_results[i].1,
                "results should be sorted descending"
            );
        }

        // mid_threshold should filter out lower-scoring docs
        let filtered = index.search_with_threshold(query, mid_threshold);
        assert!(
            filtered.len() <= search_results.len(),
            "mid-threshold should not exceed total matches"
        );
        for (_, score) in &filtered {
            assert!(
                *score >= mid_threshold,
                "all returned docs should score >= threshold"
            );
        }

        // Very high threshold should return nothing
        let empty_results = index.search_with_threshold(query, 1_000_000.0);
        assert!(
            empty_results.is_empty(),
            "very high threshold should return no results"
        );

        // Empty query should return nothing
        let empty_query_results = index.search_with_threshold("", 0.0);
        assert!(
            empty_query_results.is_empty(),
            "empty query should return no results"
        );
    }
}
