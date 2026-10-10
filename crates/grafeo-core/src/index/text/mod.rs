//! Full-text search with BM25 scoring and hybrid score fusion.
//!
//! This module provides text search capabilities for graph node properties,
//! enabling keyword-based retrieval alongside vector similarity search.
//!
//! # Components
//!
//! | Component | Feature | Description |
//! |-----------|---------|-------------|
//! | [`TextIndexOptions`] | always | BM25 parameters, tokenizer and stop words of an index |
//! | [`Tokenizer`] | `text-index` | Trait for text tokenization |
//! | [`SimpleTokenizer`] | `text-index` | Unicode-aware tokenizer with English stop words (the default) |
//! | [`StandardTokenizer`] | `text-index` | Every word, any language that separates words |
//! | [`CjkBigramTokenizer`] | `text-index` | Pairs of characters for Chinese, Japanese and Korean |
//! | [`InvertedIndex`] | `text-index` | BM25-scored inverted index |
//! | [`FusionMethod`] | `hybrid-search` | Score fusion for combining search results |
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "text-index")]
//! # {
//! use grafeo_core::index::text::{InvertedIndex, BM25Config};
//! use grafeo_common::types::NodeId;
//!
//! let mut index = InvertedIndex::new(BM25Config::default());
//! index.insert(NodeId::new(1), "the quick brown fox");
//! index.insert(NodeId::new(2), "the lazy brown dog");
//!
//! let results = index.search("quick fox", 10);
//! assert_eq!(results[0].0, NodeId::new(1));
//! # }
//! ```

#[cfg(feature = "text-index")]
mod inverted_index;
mod options;
#[cfg(feature = "text-index")]
pub mod section;
#[cfg(feature = "text-index")]
mod tokenizer;

#[cfg(feature = "text-index")]
pub use inverted_index::{InvertedIndex, PostingsVisitor};
pub use options::{BM25Config, MAX_STOP_WORD_BYTES, TextIndexOptions, TokenizerKind};
#[cfg(feature = "text-index")]
pub use section::TextIndexSection;
#[cfg(feature = "text-index")]
pub use tokenizer::{CjkBigramTokenizer, SimpleTokenizer, StandardTokenizer, Tokenizer};

#[cfg(feature = "hybrid-search")]
mod fusion;
#[cfg(feature = "hybrid-search")]
pub use fusion::{FusionMethod, fuse_results};
