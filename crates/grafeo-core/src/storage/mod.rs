//! Compression and encoding for graph property storage.
//!
//! Graph properties can take up a lot of space - especially string-heavy data like
//! names and labels. This module provides several encoding strategies to shrink
//! your data without losing information.
//!
//! | Data type | Best codec | Typical savings |
//! | --------- | ---------- | --------------- |
//! | Sorted integers (IDs, timestamps) | [`DeltaBitPacked`] | 5-20x smaller |
//! | Small integers (ages, counts) | [`BitPackedInts`] | 2-16x smaller |
//! | Repeated strings (labels, categories) | [`DictionaryEncoding`] | 2-50x smaller |
//! | Booleans (flags, markers) | [`BitVector`] | 8x smaller |
//!
//! Use [`CodecSelector`] to automatically pick the best codec for your data,
//! or choose manually when you know your data characteristics.
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::storage::{TypeSpecificCompressor, CodecSelector};
//!
//! // Compress sorted integers
//! let values: Vec<u64> = (100..200).collect();
//! let compressed = TypeSpecificCompressor::compress_integers(&values);
//! println!("Compression ratio: {:.1}x", compressed.compression_ratio());
//!
//! // Compress booleans
//! let bools = vec![true, false, true, true, false];
//! let compressed = TypeSpecificCompressor::compress_booleans(&bools);
//! ```

pub mod bitpack;
pub mod bitvec;
pub mod codec;
pub mod delta;
pub mod dictionary;
#[cfg(feature = "tiered-storage")]
pub mod epoch_store;
pub mod runlength;
#[cfg(feature = "succinct-indexes")]
pub mod succinct;

// Re-export commonly used types
pub use bitpack::{BitPackedInts, DeltaBitPacked};
pub use bitvec::BitVector;
pub use codec::{
    CodecSelector, CompressedData, CompressionCodec, CompressionMetadata, TypeSpecificCompressor,
};
pub use delta::{DeltaEncoding, zigzag_decode, zigzag_encode};
pub use dictionary::{DictionaryBuilder, DictionaryEncoding};
pub use runlength::{Run, RunLengthAnalyzer, RunLengthEncoding, SignedRunLengthEncoding};

// Tiered storage exports (feature-gated)
#[cfg(feature = "tiered-storage")]
pub use epoch_store::{
    CompressedEpochBlock, CompressionType, EpochBlockHeader, EpochStore, EpochStoreStats,
    IndexEntry, ZoneMap,
};

// Succinct data structure exports (feature-gated)
#[cfg(feature = "succinct-indexes")]
pub use succinct::{EliasFano, SuccinctBitVector, WaveletTree};
