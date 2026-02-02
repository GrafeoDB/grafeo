//! Succinct data structures for compact indexing with O(1) operations.
//!
//! This module provides space-efficient data structures that support fast rank/select
//! operations, essential for compact graph indexing and the Ring Index pattern.
//!
//! | Structure | Space | Operations | Use Case |
//! |-----------|-------|------------|----------|
//! | [`SuccinctBitVector`] | n + o(n) bits | O(1) rank/select | Sparse label sets |
//! | [`EliasFano`] | n⌈log(u/n)⌉ + 2n bits | O(1) access | Node ID lists per label |
//! | [`WaveletTree`] | n log σ + o(n log σ) | O(log σ) rank/select | Ring Index, sequence indexing |
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::storage::succinct::{SuccinctBitVector, EliasFano};
//!
//! // Succinct bitvector with O(1) rank/select
//! let bits: Vec<bool> = (0..10000).map(|i| i % 3 == 0).collect();
//! let sbv = SuccinctBitVector::from_bools(&bits);
//!
//! assert_eq!(sbv.rank1(100), 34);  // Count of 1s in [0, 100)
//! assert_eq!(sbv.select1(10), Some(30));  // Position of 11th 1-bit
//!
//! // Elias-Fano for sparse monotonic sequences
//! let node_ids: Vec<u64> = vec![100, 150, 200, 1000, 5000];
//! let ef = EliasFano::new(&node_ids);
//!
//! assert_eq!(ef.get(2), 200);
//! assert!(ef.contains(1000));
//! ```
//!
//! # References
//!
//! - pasta-flat (SEA 2022): Engineering Compact Data Structures for Rank and Select
//! - Elias-Fano: Quasi-Succinct Indices (WSDM 2013)
//! - Wavelet Trees: Compressed Suffix Arrays and Suffix Trees (ESA 2000)

mod elias_fano;
mod rank_select;
mod wavelet;

pub use elias_fano::EliasFano;
pub use rank_select::SuccinctBitVector;
pub use wavelet::WaveletTree;
