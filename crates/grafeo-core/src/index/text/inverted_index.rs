//! BM25-scored inverted index for full-text search.

use super::options::{BM25Config, TextIndexOptions};
use super::tokenizer::{OptionsTokenizer, Tokenizer};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Receives an [`InvertedIndex`] one posting list at a time, from
/// [`InvertedIndex::visit`].
///
/// An error from any method ends the visit, which returns it.
pub trait PostingsVisitor {
    /// Receives the BM25 parameters, the sum of all document lengths, and
    /// how many posting lists and document lengths follow.
    ///
    /// # Errors
    ///
    /// An error ends the visit.
    fn header(
        &mut self,
        config: &BM25Config,
        total_length: u64,
        term_count: usize,
        doc_count: usize,
    ) -> Result<()>;

    /// Receives the posting list of `term`: `count` pairs of a node and the
    /// term's frequency in its document, in node order.
    ///
    /// # Errors
    ///
    /// An error ends the visit.
    fn posting_list(
        &mut self,
        term: &str,
        count: usize,
        postings: &mut dyn Iterator<Item = (NodeId, u32)>,
    ) -> Result<()>;

    /// Receives the length (in tokens) of the document of `node`.
    ///
    /// # Errors
    ///
    /// An error ends the visit.
    fn doc_length(&mut self, node: NodeId, length: u32) -> Result<()>;
}

/// The length of the document of `tokens` and the frequency of each of its
/// terms, as the index stores them.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] when there are more than `u32::MAX`
/// tokens. Every term frequency is then at most the length, so the checked
/// increments below cannot fail either.
fn count_terms<'t>(
    tokens: impl ExactSizeIterator<Item = &'t str>,
) -> Result<(u32, HashMap<&'t str, u32>)> {
    let token_count = tokens.len();
    let length = u32::try_from(token_count).map_err(|_| {
        Error::InvalidValue(format!(
            "a text index counts at most {} tokens in a document, this one has {token_count} tokens",
            u32::MAX
        ))
    })?;
    let mut frequencies: HashMap<&str, u32> = HashMap::new();
    if length == 0 {
        return Ok((0, frequencies));
    }
    for token in tokens {
        let frequency = frequencies.entry(token).or_insert(0);
        *frequency = frequency.checked_add(1).ok_or_else(|| {
            Error::InvalidValue(format!(
                "a text index counts a term at most {} times in a document",
                u32::MAX
            ))
        })?;
    }
    Ok((length, frequencies))
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
    /// Tokenizer used for indexing and querying: the one `options` names.
    tokenizer: Box<dyn Tokenizer>,
    /// The options the index was made with; their BM25 parameters are the
    /// ones it scores with.
    options: TextIndexOptions,
}

impl InvertedIndex {
    /// Creates a new inverted index with the given BM25 configuration, the
    /// [`simple`](super::TokenizerKind::Simple) tokenizer and its stop words.
    #[must_use]
    pub fn new(config: BM25Config) -> Self {
        Self::with_options(TextIndexOptions::new().with_bm25(config))
    }

    /// Creates a new inverted index with `options`: its BM25 parameters, its
    /// tokenizer and its stop words, which [`options`](Self::options) gives
    /// back. The options are not checked (see [`TextIndexOptions::check`]).
    #[must_use]
    pub fn with_options(options: TextIndexOptions) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            tokenizer: Box::new(OptionsTokenizer::new(&options)),
            options,
        }
    }

    /// Creates a new inverted index with a custom tokenizer.
    ///
    /// A database keeps a text index by its [`options`](Self::options), and
    /// a custom tokenizer is none of them: the options of such an index name
    /// the `simple` tokenizer, which a database opened again would use.
    #[deprecated(
        since = "0.6.0",
        note = "use `InvertedIndex::with_options` and a `TokenizerKind`: a database cannot \
                keep a custom tokenizer"
    )]
    pub fn with_tokenizer(config: BM25Config, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            tokenizer,
            options: TextIndexOptions::new().with_bm25(config),
        }
    }

    /// The options of the index: its BM25 parameters (also after a restore,
    /// which sets them), its tokenizer and its stop words.
    #[must_use]
    pub fn options(&self) -> &TextIndexOptions {
        &self.options
    }

    /// Indexes a document (node text) into the inverted index.
    ///
    /// If the node was already indexed, it is first removed and re-indexed.
    /// A document the index cannot count (more than `u32::MAX` tokens, a
    /// string of at least 8 GiB) is left out of the index, as a vector of the
    /// wrong dimension is left out of a vector index: [`Self::try_insert`]
    /// returns the error instead.
    pub fn insert(&mut self, id: NodeId, text: &str) {
        if self.try_insert(id, text).is_err() {
            debug_assert!(
                !self.contains(id),
                "a document the index cannot count is not indexed"
            );
        }
    }

    /// Indexes a document (node text) into the inverted index, as
    /// [`Self::insert`] does.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] when the document has more than
    /// `u32::MAX` tokens, or when the sum of all document lengths would pass
    /// `u64::MAX`. The node is then not in the index (an earlier text of it is
    /// removed).
    pub fn try_insert(&mut self, id: NodeId, text: &str) -> Result<()> {
        // Remove existing entry if present
        if self.doc_lengths.contains_key(&id) {
            self.remove(id);
        }

        let tokens = self.tokenizer.tokenize(text);
        let (doc_len, term_freqs) = count_terms(tokens.iter().map(String::as_str))?;
        if doc_len == 0 {
            return Ok(());
        }
        let total_length = self
            .total_length
            .checked_add(u64::from(doc_len))
            .ok_or_else(|| {
                Error::InvalidValue(format!(
                    "a text index cannot add a document of {doc_len} tokens to its {} tokens",
                    self.total_length
                ))
            })?;

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
        self.total_length = total_length;
        Ok(())
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
        let BM25Config { k1, b } = *self.options.bm25();
        let tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
        idf * tf_component
    }

    /// The BM25 score of every document of a node `keep` accepts that holds a
    /// term of `query`. The corpus statistics count every document.
    fn scores(&self, query: &str, keep: impl Fn(NodeId) -> bool) -> HashMap<NodeId, f64> {
        let query_tokens = self.tokenizer.tokenize(query);
        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        if query_tokens.is_empty() || self.doc_lengths.is_empty() {
            return scores;
        }

        let n = self.doc_lengths.len() as f64;
        let avg_dl = self.total_length as f64 / n;
        for token in &query_tokens {
            let Some(posting_list) = self.postings.get(token.as_str()) else {
                continue;
            };
            let df = posting_list.postings.len() as f64;
            for posting in &posting_list.postings {
                if !keep(posting.node_id) {
                    continue;
                }
                let tf = f64::from(posting.term_freq);
                let dl = f64::from(self.doc_lengths.get(&posting.node_id).copied().unwrap_or(0));
                *scores.entry(posting.node_id).or_insert(0.0) +=
                    self.bm25_term_score(df, tf, dl, n, avg_dl);
            }
        }
        scores
    }

    /// The `k` best of `scores`, by descending score.
    fn top(scores: HashMap<NodeId, f64>, k: usize) -> Vec<(NodeId, f64)> {
        let mut results: Vec<(NodeId, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Searches the index using BM25 scoring.
    ///
    /// Returns up to `k` results sorted by descending BM25 score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(NodeId, f64)> {
        Self::top(self.scores(query, |_| true), k)
    }

    /// Searches the index as [`Self::search`] does, among the nodes of
    /// `allowlist` only: up to `k` of them, sorted by descending BM25 score.
    ///
    /// The scores are the ones [`Self::search`] gives the same nodes: the
    /// corpus statistics (the number of documents, their average length and
    /// each term's document frequency) count every document of the index,
    /// not only the allowed ones, so a filter narrows the results without
    /// changing how a document scores.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &str,
        k: usize,
        allowlist: &HashSet<NodeId>,
    ) -> Vec<(NodeId, f64)> {
        if allowlist.is_empty() {
            return Vec::new();
        }
        Self::top(self.scores(query, |node| allowlist.contains(&node)), k)
    }

    /// Scores a single document against a query using BM25.
    ///
    /// Looks up each query term in its posting list, finds the entry for the
    /// given node ID, and computes BM25 with corpus statistics. Returns `0.0`
    /// if the document has no matching terms or doesn't exist.
    ///
    /// Cost is O(query_terms × average posting-list length) per call: for each
    /// query term, the matching node is found by linear scan of that term's
    /// posting list. Intended for per-row evaluation, where a few hundred
    /// per-document scores are cheaper than reorganizing posting lists into
    /// per-document maps.
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
    /// Unlike [`Self::search`] (top-k), this returns every document above the
    /// threshold, sorted by score descending. Intended for index-accelerated
    /// text search with WHERE predicates.
    #[must_use]
    pub fn search_with_threshold(&self, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        let mut scores = self.scores(query, |_| true);
        scores.retain(|_, score| *score >= threshold);
        let count = scores.len();
        Self::top(scores, count)
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
        self.options.bm25()
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
        self.options.set_bm25(config);
    }

    /// Restore the index from a snapshot. Replaces all current data and
    /// keeps the BM25 configuration.
    pub fn restore(
        &mut self,
        postings: Vec<(String, Vec<(NodeId, u32)>)>,
        doc_lengths: Vec<(NodeId, u32)>,
        total_length: u64,
    ) {
        self.begin_restore(self.options.bm25().clone(), total_length);
        for (term, entries) in postings {
            self.restore_posting_list(term, entries);
        }
        for (node, length) in doc_lengths {
            self.restore_doc_length(node, length);
        }
    }

    /// Hands the index to `visitor`: the header, every document length in
    /// node order, then every posting list in term order, each list in node
    /// order.
    ///
    /// The order depends only on what the index holds, not on its hash maps
    /// or the order documents were inserted in, so the same postings are
    /// always handed over the same way. A list the index holds in node order
    /// (as a restored index, or one built from nodes in id order, does) is
    /// handed over as an iterator over the index's own list; another is
    /// copied and sorted first (16 bytes per posting, one list at a time).
    /// To sort, the visit also gathers references to every term (16 bytes
    /// each) and a copy of the document lengths (16 bytes each).
    ///
    /// The index cannot change during the visit, so whoever calls it holds
    /// the index's lock for its length: a checkpoint holds the read lock
    /// while it writes the index to its sink.
    ///
    /// # Errors
    ///
    /// Returns the first error of `visitor`, which ends the visit.
    pub fn visit(&self, visitor: &mut dyn PostingsVisitor) -> Result<()> {
        visitor.header(
            self.options.bm25(),
            self.total_length,
            self.postings.len(),
            self.doc_lengths.len(),
        )?;
        let mut lengths: Vec<(NodeId, u32)> = self
            .doc_lengths
            .iter()
            .map(|(node, length)| (*node, *length))
            .collect();
        lengths.sort_unstable_by_key(|(node, _)| *node);
        for (node, length) in lengths {
            visitor.doc_length(node, length)?;
        }
        let mut terms: Vec<(&String, &PostingList)> = self.postings.iter().collect();
        terms.sort_unstable_by_key(|(term, _)| *term);
        for (term, list) in terms {
            let count = list.postings.len();
            if list.postings.is_sorted_by_key(|posting| posting.node_id) {
                let mut postings = list
                    .postings
                    .iter()
                    .map(|posting| (posting.node_id, posting.term_freq));
                visitor.posting_list(term, count, &mut postings)?;
            } else {
                let mut sorted: Vec<(NodeId, u32)> = list
                    .postings
                    .iter()
                    .map(|posting| (posting.node_id, posting.term_freq))
                    .collect();
                sorted.sort_unstable_by_key(|(node, _)| *node);
                visitor.posting_list(term, count, &mut sorted.into_iter())?;
            }
        }
        Ok(())
    }

    /// Empties the index and sets its BM25 configuration and the sum of its
    /// document lengths, for [`restore_posting_list`](Self::restore_posting_list)
    /// and [`restore_doc_length`](Self::restore_doc_length) to fill. The
    /// tokenizer stays.
    pub fn begin_restore(&mut self, config: BM25Config, total_length: u64) {
        self.postings = HashMap::new();
        self.doc_lengths = HashMap::new();
        self.total_length = total_length;
        self.options.set_bm25(config);
    }

    /// Sets the posting list of `term`: pairs of a node and the term's
    /// frequency in its document, kept in this order.
    pub fn restore_posting_list(&mut self, term: String, postings: Vec<(NodeId, u32)>) {
        let postings = postings
            .into_iter()
            .map(|(node_id, term_freq)| Posting { node_id, term_freq })
            .collect();
        self.postings.insert(term, PostingList { postings });
    }

    /// Sets the length (in tokens) of the document of `node`. The sum of
    /// all lengths stays what [`begin_restore`](Self::begin_restore) set.
    pub fn restore_doc_length(&mut self, node: NodeId, length: u32) {
        self.doc_lengths.insert(node, length);
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
        assert!(!results.is_empty(), "results is empty");
        // Node 3 mentions both "brown" and "dog" in a shorter document
        assert_eq!(results[0].0, NodeId::new(3));
    }

    #[test]
    fn term_counts_hold_the_document_length_and_each_term_frequency() {
        let (length, frequencies) =
            count_terms(["graph", "notes", "graph"].into_iter()).expect("three tokens fit u32");
        assert_eq!(length, 3, "the document length counts every token");
        assert_eq!(frequencies.get("graph"), Some(&2));
        assert_eq!(frequencies.get("notes"), Some(&1));
        assert_eq!(frequencies.len(), 2, "{frequencies:?}");
    }

    /// A length past `u32::MAX` used to be cast: 2^32 tokens became a
    /// document of length 0, which the index left out without a word, and
    /// 2^32 + 3 tokens a document of length 3.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn a_document_with_more_tokens_than_u32_counts_is_an_error() {
        let too_many = usize::try_from(u64::from(u32::MAX) + 1).expect("64-bit usize");
        let result = count_terms(std::iter::repeat_n("graph", too_many));
        assert!(
            matches!(&result, Err(Error::InvalidValue(message)) if message.contains("4294967296 tokens")),
            "{:?}",
            result.map(|(length, _)| length)
        );
    }

    #[test]
    fn test_empty_index_search() {
        let index = InvertedIndex::new(BM25Config::default());
        let results = index.search("anything", 10);
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_empty_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("", 10);
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_stop_word_only_query() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "hello world");
        let results = index.search("the a an", 10);
        assert!(results.is_empty(), "{results:?}");
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
        assert!(results.is_empty(), "{results:?}");

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
        assert!(results.is_empty(), "{results:?}");
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
        index.insert(
            NodeId::new(1),
            "the quick brown fox jumps over the lazy dog",
        );
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

        // Node 2 has no matching terms: it scores 0.0
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
        index.insert(
            NodeId::new(5),
            "database indexing storage engine optimization",
        );

        let query = "rust graph database";

        // Get search results to calibrate the threshold
        let search_results = index.search(query, 10);
        assert!(
            search_results.len() >= 2,
            "need at least 2 matching docs for this test"
        );

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

    /// Three notes in the standard tokenizer: Alix in Amsterdam (3 terms),
    /// Gus in Berlin (2), Alix three times in Berlin (4).
    fn berlin_notes(options: TextIndexOptions) -> InvertedIndex {
        let mut index = InvertedIndex::with_options(
            options.with_tokenizer(super::super::TokenizerKind::Standard),
        );
        index.insert(NodeId::new(3), "Alix Amsterdam Amsterdam");
        index.insert(NodeId::new(19), "Gus Berlin");
        index.insert(NodeId::new(88), "Alix Berlin Berlin Berlin");
        index
    }

    /// BM25 of the term "berlin" in [`berlin_notes`]: 3 documents of 3
    /// terms on average, 2 with the term, so its inverse document frequency
    /// is ln(1 + 1.5 / 2.5) = 0.470003629245736 for both. Gus's note has it
    /// once in 2 terms, the other three times in 4:
    /// idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / 3)).
    #[test]
    fn scores_follow_k1_and_b_as_bm25_says() {
        for (k1, b, gus, alix) in [
            (1.2, 0.75, 0.544_214_728_600_325_5, 0.689_338_656_227_079),
            (0.0, 0.75, 0.470_003_629_245_735_6, 0.470_003_629_245_735_6),
            (2.0, 0.0, 0.470_003_629_245_735_6, 0.846_006_532_642_324_1),
            (1.2, 1.0, 0.574_448_880_189_232_6, 0.674_353_033_265_620_8),
            (0.3, 0.19, 0.476_974_799_390_676_2, 0.552_279_046_115_808_7),
        ] {
            let index = berlin_notes(TextIndexOptions::new().with_k1(k1).with_b(b));
            let mut found = index.search("Berlin", 10);
            found.sort_by_key(|(node, _)| *node);
            assert_eq!(found.len(), 2, "k1 {k1}, b {b}: {found:?}");
            for ((node, score), expected) in found.iter().zip([gus, alix]) {
                assert!(
                    (score - expected).abs() < 1e-12,
                    "k1 {k1}, b {b}: node {node:?} scores {score}, BM25 says {expected}"
                );
            }
        }
    }

    #[test]
    fn the_default_index_scores_as_bm25_1_2_and_0_75() {
        let default = berlin_notes(TextIndexOptions::new());
        let explicit = berlin_notes(TextIndexOptions::new().with_k1(1.2).with_b(0.75));
        assert_eq!(default.search("berlin", 10).len(), 2);
        let mut found = default.search("berlin", 10);
        let mut expected = explicit.search("berlin", 10);
        found.sort_by_key(|(node, _)| *node);
        expected.sort_by_key(|(node, _)| *node);
        assert_eq!(found, expected);
        assert_eq!(
            InvertedIndex::new(BM25Config::default()).options(),
            &TextIndexOptions::new(),
            "new() makes an index with the default options"
        );
    }

    #[test]
    fn documents_and_queries_are_tokenized_by_the_options_tokenizer() {
        use super::super::TokenizerKind;

        let mut chinese = InvertedIndex::with_options(
            TextIndexOptions::new().with_tokenizer(TokenizerKind::CjkBigram),
        );
        chinese.insert(NodeId::new(3), "阿利克斯住在柏林");
        chinese.insert(NodeId::new(19), "古斯住在阿姆斯特丹");
        let found: Vec<u64> = chinese
            .search("柏林", 10)
            .into_iter()
            .map(|(node, _)| node.0)
            .collect();
        assert_eq!(found, [3], "Berlin in Chinese is found inside the sentence");

        let mut simple = InvertedIndex::new(BM25Config::default());
        simple.insert(NodeId::new(3), "阿利克斯住在柏林");
        assert_eq!(
            simple.search("柏林", 10),
            [],
            "the simple tokenizer keeps the sentence as one term"
        );

        let mut russian = InvertedIndex::with_options(
            TextIndexOptions::new()
                .with_tokenizer(TokenizerKind::Standard)
                .with_stop_words(["и", "в"]),
        );
        russian.insert(NodeId::new(3), "Аликс и Гас едут в Берлин");
        assert_eq!(
            russian.search("и в", 10),
            [],
            "a query of stop words finds nothing"
        );
        assert_eq!(
            russian.search("БЕРЛИН", 10).len(),
            1,
            "queries are lowercased"
        );
        assert_eq!(russian.len(), 1);
        assert_eq!(russian.term_count(), 4, "аликс, гас, едут, берлин");
    }

    #[test]
    fn a_restore_sets_the_bm25_parameters_and_keeps_the_tokenizer_and_stop_words() {
        use super::super::TokenizerKind;

        let options = TextIndexOptions::new()
            .with_tokenizer(TokenizerKind::CjkBigram)
            .with_stop_words(["住在"]);
        let mut index = InvertedIndex::with_options(options.clone());
        index.begin_restore(BM25Config { k1: 0.3, b: 0.19 }, 0);
        assert_eq!(
            index.options(),
            &options.with_k1(0.3).with_b(0.19),
            "the restored parameters, the same tokenizer and stop words"
        );
    }

    /// Five notes, three about canals: Gus's three times, Alix's in a shorter
    /// note than Mia's.
    fn canal_notes() -> InvertedIndex {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(3), "Alix walks the canals of Amsterdam");
        index.insert(
            NodeId::new(19),
            "Gus cycles along canals, canals and more canals",
        );
        index.insert(NodeId::new(88), "Vincent visits Berlin");
        index.insert(NodeId::new(4), "Mia sees the old canals of Berlin at night");
        index.insert(NodeId::new(5), "Jules naps");
        index
    }

    /// The nodes with these ids, as an allowlist.
    fn nodes(ids: &[u64]) -> HashSet<NodeId> {
        ids.iter().copied().map(NodeId::new).collect()
    }

    #[test]
    fn a_filtered_search_keeps_the_allowed_nodes_with_the_scores_of_the_whole_index() {
        let index = canal_notes();
        let allowlist = nodes(&[3, 4, 5, 88]);
        let everything = index.search("canals Berlin", 10);
        let expected: Vec<(NodeId, f64)> = everything
            .iter()
            .copied()
            .filter(|(node, _)| allowlist.contains(node))
            .collect();
        assert_eq!(
            expected.iter().map(|(node, _)| node.0).collect::<Vec<_>>(),
            vec![4, 88, 3],
            "Mia has both terms; Berlin, the rarer one, outscores the canals of Alix: \
             {everything:?}"
        );
        assert_eq!(
            index.search_with_filter("canals Berlin", 10, &allowlist),
            expected,
            "the order and the scores of the unfiltered search, Gus left out"
        );
        assert_eq!(
            index.search_with_filter("canals Berlin", 2, &allowlist),
            expected[..2],
            "k counts the allowed nodes"
        );
    }

    #[test]
    fn a_filtered_search_fills_k_from_the_allowed_nodes() {
        let index = canal_notes();
        let top_two: Vec<u64> = index
            .search("canals", 2)
            .into_iter()
            .map(|(node, _)| node.0)
            .collect();
        assert_eq!(top_two, vec![19, 3], "Gus and Alix outscore Mia");
        let found = index.search_with_filter("canals", 1, &nodes(&[4, 88]));
        assert_eq!(
            found.iter().map(|(node, _)| node.0).collect::<Vec<_>>(),
            vec![4],
            "the best allowed match, not a top-k filtered afterwards: {found:?}"
        );
    }

    #[test]
    fn a_filtered_search_without_allowed_nodes_finds_nothing() {
        let index = canal_notes();
        assert_eq!(
            index.search_with_filter("canals", 10, &HashSet::new()),
            [],
            "no node is allowed"
        );
        assert_eq!(
            index.search_with_filter("canals", 10, &nodes(&[5, 88])),
            [],
            "Jules and Vincent never mention canals"
        );
    }

    /// One call [`InvertedIndex::visit`] made.
    #[derive(Debug, PartialEq)]
    enum Visited {
        /// k1 and b as bits, the total length, the term and document counts.
        Header(u64, u64, u64, usize, usize),
        /// A term, the count announced and the postings handed over.
        List(String, usize, Vec<(NodeId, u32)>),
        /// A node and the length of its document.
        Length(NodeId, u32),
    }

    /// The calls of a visit, in order; refuses the call after the first
    /// `fail_after` calls, when set.
    #[derive(Debug, Default)]
    struct Recorded {
        calls: Vec<Visited>,
        fail_after: Option<usize>,
    }

    impl Recorded {
        /// Refuses the call when `fail_after` calls were recorded.
        fn record(&mut self, call: Visited) -> Result<()> {
            if self.fail_after == Some(self.calls.len()) {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Gus stops the visit".to_string(),
                ));
            }
            self.calls.push(call);
            Ok(())
        }
    }

    impl PostingsVisitor for Recorded {
        fn header(
            &mut self,
            config: &BM25Config,
            total_length: u64,
            term_count: usize,
            doc_count: usize,
        ) -> Result<()> {
            self.record(Visited::Header(
                config.k1.to_bits(),
                config.b.to_bits(),
                total_length,
                term_count,
                doc_count,
            ))
        }

        fn posting_list(
            &mut self,
            term: &str,
            count: usize,
            postings: &mut dyn Iterator<Item = (NodeId, u32)>,
        ) -> Result<()> {
            self.record(Visited::List(term.to_string(), count, postings.collect()))
        }

        fn doc_length(&mut self, node: NodeId, length: u32) -> Result<()> {
            self.record(Visited::Length(node, length))
        }
    }

    /// Restores the index it holds from what `visit` hands over.
    struct Restorer<'a>(&'a mut InvertedIndex);

    impl PostingsVisitor for Restorer<'_> {
        fn header(
            &mut self,
            config: &BM25Config,
            total_length: u64,
            _term_count: usize,
            _doc_count: usize,
        ) -> Result<()> {
            self.0.begin_restore(config.clone(), total_length);
            Ok(())
        }

        fn posting_list(
            &mut self,
            term: &str,
            _count: usize,
            postings: &mut dyn Iterator<Item = (NodeId, u32)>,
        ) -> Result<()> {
            self.0
                .restore_posting_list(term.to_string(), postings.collect());
            Ok(())
        }

        fn doc_length(&mut self, node: NodeId, length: u32) -> Result<()> {
            self.0.restore_doc_length(node, length);
            Ok(())
        }
    }

    /// Three documents inserted out of node order: Paris twice in node 88,
    /// once in node 3.
    fn three_cities() -> InvertedIndex {
        let mut index = InvertedIndex::new(BM25Config { k1: 1.9, b: 0.3 });
        index.insert(NodeId::new(88), "Paris Mia Paris");
        index.insert(NodeId::new(3), "Berlin Paris");
        index.insert(NodeId::new(19), "Amsterdam");
        index
    }

    /// The calls a visit of [`three_cities`] makes.
    fn three_cities_visited() -> Vec<Visited> {
        vec![
            Visited::Header(1.9f64.to_bits(), 0.3f64.to_bits(), 6, 4, 3),
            Visited::Length(NodeId::new(3), 2),
            Visited::Length(NodeId::new(19), 1),
            Visited::Length(NodeId::new(88), 3),
            Visited::List("amsterdam".to_string(), 1, vec![(NodeId::new(19), 1)]),
            Visited::List("berlin".to_string(), 1, vec![(NodeId::new(3), 1)]),
            Visited::List("mia".to_string(), 1, vec![(NodeId::new(88), 1)]),
            // In node order, although node 88 was inserted first.
            Visited::List(
                "paris".to_string(),
                2,
                vec![(NodeId::new(3), 1), (NodeId::new(88), 2)],
            ),
        ]
    }

    #[test]
    fn visit_hands_over_the_documents_in_node_order_then_the_terms_in_order() {
        let mut recorded = Recorded::default();
        three_cities().visit(&mut recorded).unwrap();
        assert_eq!(recorded.calls, three_cities_visited());
    }

    #[test]
    fn a_visitor_error_ends_the_visit() {
        let all = three_cities_visited();
        for fail_after in 0..all.len() {
            let mut visitor = Recorded {
                fail_after: Some(fail_after),
                ..Recorded::default()
            };
            let error = three_cities().visit(&mut visitor).unwrap_err();
            assert!(error.to_string().contains("Gus stops the visit"), "{error}");
            assert_eq!(
                visitor.calls[..],
                all[..fail_after],
                "no call after the error in call {fail_after}"
            );
        }
    }

    #[test]
    fn a_restore_replaces_the_whole_index_and_searches_as_the_original() {
        let original = three_cities();
        let mut restored = InvertedIndex::new(BM25Config::default());
        restored.insert(NodeId::new(5), "Gus Prague");
        original.visit(&mut Restorer(&mut restored)).unwrap();

        let (mut postings, lengths, total) = original.snapshot();
        for (_, list) in &mut postings {
            list.sort_by_key(|(node, _)| *node);
        }
        assert_eq!(
            restored.snapshot(),
            (postings, lengths, total),
            "the original with every list in node order"
        );
        assert!(!restored.contains(NodeId::new(5)), "Gus is gone");
        assert_eq!(
            (restored.config().k1, restored.config().b),
            (1.9, 0.3),
            "the BM25 parameters come with the restore"
        );
        for query in ["paris", "berlin paris", "mia amsterdam"] {
            let mut expected = original.search(query, 10);
            let mut found = restored.search(query, 10);
            expected.sort_by_key(|(node, _)| *node);
            found.sort_by_key(|(node, _)| *node);
            assert_eq!(found, expected, "{query}");
        }
    }

    #[test]
    fn begin_restore_empties_the_index_and_keeps_the_total_it_is_given() {
        let mut index = three_cities();
        index.begin_restore(BM25Config { k1: 0.3, b: 0.88 }, 19);
        assert!(index.is_empty() && index.term_count() == 0);
        assert_eq!(index.snapshot(), (Vec::new(), Vec::new(), 19));
        assert_eq!((index.config().k1, index.config().b), (0.3, 0.88));

        index.restore_doc_length(NodeId::new(3), 19);
        index.restore_posting_list("jules".to_string(), vec![(NodeId::new(3), 19)]);
        assert_eq!(
            index.snapshot(),
            (
                vec![("jules".to_string(), vec![(NodeId::new(3), 19)])],
                vec![(NodeId::new(3), 19)],
                19
            ),
            "a document length leaves the total as begin_restore set it"
        );
    }
}
