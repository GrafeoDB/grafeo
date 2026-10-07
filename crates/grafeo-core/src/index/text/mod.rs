//! Full-text search with BM25 scoring and hybrid score fusion.
//!
//! This module provides text search capabilities for graph node properties,
//! enabling keyword-based retrieval alongside vector similarity search.
//!
//! # Components
//!
//! | Component | Feature | Description |
//! |-----------|---------|-------------|
//! | [`Tokenizer`] | `text-index` | Trait for text tokenization |
//! | [`SimpleTokenizer`] | `text-index` | Unicode-aware tokenizer with stop words |
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

mod inverted_index;
pub mod section;
mod tokenizer;

pub use inverted_index::{BM25Config, InvertedIndex, PostingsVisitor};
pub use section::TextIndexSection;
pub use tokenizer::{SimpleTokenizer, Tokenizer};

#[cfg(feature = "hybrid-search")]
mod fusion;
#[cfg(feature = "hybrid-search")]
pub use fusion::{FusionMethod, fuse_results};
