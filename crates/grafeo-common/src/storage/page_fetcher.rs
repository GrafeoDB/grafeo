//! Page-fetching abstraction for tiered storage.
//!
//! Provides an indirection layer that lets sections receive paged byte
//! access without knowing whether the bytes come from a memory-mapped
//! file, an explicit buffer pool, or some other source. The mmap-backed
//! implementation lives in `grafeo-storage`; future implementations may
//! include vmcache (TUM, 2024) or an explicit pager with pointer
//! swizzling (LeanStore, VLDB 2024).
//!
//! See `docs/internals/storage/disk-storage-decisions.md` for the
//! rationale (decision D1).

use std::io;

/// Hint about the expected access pattern for a range of bytes.
///
/// Best-effort: implementations may ignore hints on platforms that don't
/// support them (for example, Windows lacks a portable `madvise` analogue
/// without unsafe FFI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessHint {
    /// Pages will be accessed in sequence; readahead encouraged.
    Sequential,
    /// Pages will be accessed in random order; no readahead.
    Random,
    /// Pages will be accessed soon; prefetch into cache.
    WillNeed,
    /// Pages won't be accessed for a while; release from cache.
    DontNeed,
}

/// Fetches byte ranges from a paged storage source.
///
/// Offsets are relative to the start of the fetcher's logical region
/// (i.e., 0 corresponds to the first byte of the section, not the file).
pub trait PageFetcher: Send + Sync {
    /// Fetch `len` bytes starting at `offset` within the region.
    ///
    /// Returns a borrowed slice into the underlying storage; the borrow
    /// must not outlive a mutation of the underlying source.
    ///
    /// # Errors
    ///
    /// - [`io::ErrorKind::InvalidInput`] if `offset + len` overflows.
    /// - [`io::ErrorKind::UnexpectedEof`] if the requested range exceeds
    ///   the region length.
    fn fetch(&self, offset: usize, len: usize) -> io::Result<&[u8]>;

    /// Total length of the addressable region in bytes.
    fn len(&self) -> usize;

    /// Whether the region is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advise the OS about the expected access pattern for a range.
    ///
    /// Best-effort: out-of-range or unsupported hints silently no-op.
    fn advise(&self, offset: usize, len: usize, hint: AccessHint);
}
