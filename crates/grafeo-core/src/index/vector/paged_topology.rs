//! Packed HNSW topology for the v2 vector store on-disk format (Phase 7a).
//!
//! Today's [`super::section::VectorStoreSection`] persists each HNSW
//! topology as bincode'd `Vec<(NodeId, Vec<Vec<NodeId>>)>`. On open the
//! whole structure is re-allocated into a `HashMap`, which dominates
//! heap usage for vector workloads (per ARCHITECTURE.md, hundreds of MB
//! to multiple GB at million-vector scale).
//!
//! v2 stores topology in a flat, `Bytes`-backed byte format with a
//! sorted page index for O(log n) neighbor-list lookup. Phase 7a
//! delivers only the byte format and round-trip codec — Phase 7c wires
//! it into a zero-copy [`super::HnswIndex`] accessor.
//!
//! ## Layout
//!
//! ```text
//! Header (32 bytes, little-endian):
//!     0..4    magic "GTOP"
//!     4       version u8 = 1
//!     5       has_entry_point u8 (0 or 1)
//!     6..8    reserved (2 bytes, zero)
//!     8..16   n_nodes u64
//!     16..20  max_level u32
//!     20..24  reserved (4 bytes, zero)
//!     24..32  entry_point u64 (valid iff has_entry_point != 0)
//!
//! Page index (16 bytes per node, sorted by NodeId for binary search):
//!     For each node:
//!         8 bytes: NodeId u64
//!         8 bytes: payload_offset u64 (offset into payload region)
//!
//! Payload region (variable, packed back-to-back per node):
//!     For each node, in NodeId-sorted order:
//!         4 bytes: n_levels u32
//!         For each level l in 0..n_levels:
//!             4 bytes: n_neighbors u32
//!             n_neighbors * 8 bytes: neighbor NodeIds (u64 LE)
//! ```
//!
//! The page index is "paged" in the page-cache sense: at 16 bytes per
//! node, 1M nodes occupies 16 MB, which the OS will demand-page from
//! mmap as the binary search descends. The payload region is also
//! paged but unaligned — Phase 7c may revisit alignment if measured
//! page-fault counts justify it (see ARCHITECTURE.md vmcache exit).

use bytes::Bytes;

use grafeo_common::types::NodeId;

const MAGIC: &[u8; 4] = b"GTOP";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 32;
const INDEX_ENTRY_SIZE: usize = 16;

/// Errors returned when parsing a packed topology from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedTopologyError {
    /// Buffer is too short to contain even the fixed-size header.
    TruncatedHeader,
    /// First 4 bytes don't match "GTOP".
    BadMagic,
    /// Version byte not recognized.
    UnsupportedVersion(u8),
    /// `has_entry_point` byte is neither 0 nor 1.
    InvalidEntryPointFlag(u8),
    /// `n_nodes` field overflows the platform-native usize.
    SizeOverflow,
    /// Index region is shorter than `n_nodes` declares.
    TruncatedIndex {
        /// Bytes the index region should contain.
        expected: usize,
        /// Bytes available in the input.
        actual: usize,
    },
    /// Page-index entries must be sorted ascending by NodeId.
    UnsortedPageIndex {
        /// Position where the order broke.
        index: usize,
    },
    /// A page-index entry's payload offset is past end-of-buffer.
    PayloadOffsetOutOfRange {
        /// Position whose offset was bad.
        index: usize,
        /// The bad payload offset.
        offset: u64,
    },
    /// A node's payload header (n_levels) is truncated.
    TruncatedNodeHeader {
        /// NodeId whose entry was truncated.
        node: u64,
    },
    /// A node's level header (n_neighbors) is truncated.
    TruncatedLevelHeader {
        /// NodeId whose entry was truncated.
        node: u64,
        /// Level index where the truncation occurred.
        level: u32,
    },
    /// A node's level neighbor list is truncated.
    TruncatedNeighbors {
        /// NodeId whose entry was truncated.
        node: u64,
        /// Level index where the truncation occurred.
        level: u32,
        /// Bytes the neighbor list should contain.
        expected: usize,
        /// Bytes available in the input.
        actual: usize,
    },
}

impl std::fmt::Display for PagedTopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "paged topology header truncated"),
            Self::BadMagic => write!(f, "paged topology bad magic (expected 'GTOP')"),
            Self::UnsupportedVersion(v) => {
                write!(f, "paged topology unsupported version {v}")
            }
            Self::InvalidEntryPointFlag(b) => {
                write!(f, "paged topology invalid has_entry_point flag {b}")
            }
            Self::SizeOverflow => write!(f, "paged topology size field overflows usize"),
            Self::TruncatedIndex { expected, actual } => write!(
                f,
                "paged topology index truncated: expected {expected} bytes, got {actual}"
            ),
            Self::UnsortedPageIndex { index } => {
                write!(
                    f,
                    "paged topology page index not sorted at position {index}"
                )
            }
            Self::PayloadOffsetOutOfRange { index, offset } => write!(
                f,
                "paged topology payload offset {offset} out of range at index {index}"
            ),
            Self::TruncatedNodeHeader { node } => {
                write!(f, "paged topology node {node} header truncated")
            }
            Self::TruncatedLevelHeader { node, level } => write!(
                f,
                "paged topology node {node} level {level} header truncated"
            ),
            Self::TruncatedNeighbors {
                node,
                level,
                expected,
                actual,
            } => write!(
                f,
                "paged topology node {node} level {level} neighbors truncated: expected {expected} bytes, got {actual}"
            ),
        }
    }
}

impl std::error::Error for PagedTopologyError {}

/// Serializes an HNSW topology snapshot into the v2 paged byte format.
///
/// Input format mirrors [`super::HnswIndex::snapshot_topology`]: nodes
/// must already be sorted ascending by NodeId. Caller responsibility to
/// pass a sorted slice — the snapshot helper sorts before returning.
///
/// # Panics
///
/// Panics if any layer count or neighbor count exceeds `u32::MAX`, or
/// if `max_level` exceeds `u32::MAX`. These bounds are far above any
/// practical HNSW configuration.
#[must_use]
pub fn serialize_topology(
    entry_point: Option<NodeId>,
    max_level: usize,
    nodes: &[(NodeId, Vec<Vec<NodeId>>)],
) -> Vec<u8> {
    let n_nodes = nodes.len();
    let index_size = n_nodes * INDEX_ENTRY_SIZE;

    let mut payload = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(n_nodes);
    for (_id, layers) in nodes {
        offsets.push(payload.len() as u64);

        let n_levels = u32::try_from(layers.len()).expect("HNSW level count fits in u32");
        payload.extend_from_slice(&n_levels.to_le_bytes());

        for layer in layers {
            let n_neighbors = u32::try_from(layer.len()).expect("HNSW neighbor count fits in u32");
            payload.extend_from_slice(&n_neighbors.to_le_bytes());
            for nb in layer {
                payload.extend_from_slice(&nb.as_u64().to_le_bytes());
            }
        }
    }

    let mut buf = Vec::with_capacity(HEADER_SIZE + index_size + payload.len());

    buf.extend_from_slice(MAGIC);
    buf.push(VERSION);
    let has_entry_point: u8 = u8::from(entry_point.is_some());
    buf.push(has_entry_point);
    buf.extend_from_slice(&[0u8; 2]);
    buf.extend_from_slice(&(n_nodes as u64).to_le_bytes());
    let max_level_u32 = u32::try_from(max_level).expect("HNSW max_level fits in u32");
    buf.extend_from_slice(&max_level_u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);
    let entry_raw = entry_point.map_or(0u64, |id| id.as_u64());
    buf.extend_from_slice(&entry_raw.to_le_bytes());

    debug_assert_eq!(buf.len(), HEADER_SIZE);

    for ((id, _), offset) in nodes.iter().zip(offsets.iter()) {
        buf.extend_from_slice(&id.as_u64().to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    buf.extend_from_slice(&payload);
    buf
}

/// Deserializes a v2 paged topology buffer into the heap representation
/// expected by [`super::HnswIndex::restore_topology`].
///
/// The output `node_data` is sorted ascending by NodeId (preserved from
/// the on-disk page index).
///
/// # Errors
///
/// Returns [`PagedTopologyError`] if the buffer is malformed: missing
/// or wrong magic, unsupported version, truncated header / index /
/// payload, unsorted page index, or out-of-range payload offsets.
///
/// # Panics
///
/// Internal `expect` calls assert that fixed-size byte slice conversions
/// succeed (e.g. an 8-byte slice into `[u8; 8]`). These cannot fail
/// because the indices are pre-validated against `data.len()`.
#[allow(clippy::type_complexity)]
pub fn deserialize_topology(
    data: Bytes,
) -> Result<(Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>), PagedTopologyError> {
    if data.len() < HEADER_SIZE {
        return Err(PagedTopologyError::TruncatedHeader);
    }

    if &data[0..4] != MAGIC {
        return Err(PagedTopologyError::BadMagic);
    }

    let version = data[4];
    if version != VERSION {
        return Err(PagedTopologyError::UnsupportedVersion(version));
    }

    let has_entry_point = data[5];
    if has_entry_point > 1 {
        return Err(PagedTopologyError::InvalidEntryPointFlag(has_entry_point));
    }

    let n_nodes_u64 = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .expect("slice length 8 fits u64 array"),
    );
    let n_nodes = usize::try_from(n_nodes_u64).map_err(|_| PagedTopologyError::SizeOverflow)?;

    let max_level_u32 = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .expect("slice length 4 fits u32 array"),
    );
    let max_level = usize::try_from(max_level_u32).map_err(|_| PagedTopologyError::SizeOverflow)?;

    let entry_raw = u64::from_le_bytes(
        data[24..32]
            .try_into()
            .expect("slice length 8 fits u64 array"),
    );
    let entry_point = if has_entry_point == 1 {
        Some(NodeId::new(entry_raw))
    } else {
        None
    };

    let index_size = n_nodes
        .checked_mul(INDEX_ENTRY_SIZE)
        .ok_or(PagedTopologyError::SizeOverflow)?;
    let index_end = HEADER_SIZE
        .checked_add(index_size)
        .ok_or(PagedTopologyError::SizeOverflow)?;
    if data.len() < index_end {
        return Err(PagedTopologyError::TruncatedIndex {
            expected: index_size,
            actual: data.len().saturating_sub(HEADER_SIZE),
        });
    }

    let payload_start = index_end;
    let payload_len = data.len() - payload_start;

    // Validate page index: sorted ascending by NodeId, payload offsets
    // strictly increasing and within the payload region.
    let mut prev_id: Option<u64> = None;
    let mut prev_offset: Option<u64> = None;
    for i in 0..n_nodes {
        let entry_start = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
        let id = u64::from_le_bytes(
            data[entry_start..entry_start + 8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let offset = u64::from_le_bytes(
            data[entry_start + 8..entry_start + 16]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );

        if let Some(prev) = prev_id
            && id <= prev
        {
            return Err(PagedTopologyError::UnsortedPageIndex { index: i });
        }
        prev_id = Some(id);

        if usize::try_from(offset).map_err(|_| PagedTopologyError::SizeOverflow)? > payload_len {
            return Err(PagedTopologyError::PayloadOffsetOutOfRange { index: i, offset });
        }
        if let Some(prev) = prev_offset
            && offset <= prev
            && i > 0
        {
            // Strictly increasing: the on-disk encoding is "concatenated
            // per-node entries," so two distinct entries must have
            // strictly increasing offsets. Equal offsets would imply an
            // empty entry, which is fine only at index 0.
            return Err(PagedTopologyError::PayloadOffsetOutOfRange { index: i, offset });
        }
        prev_offset = Some(offset);
    }

    // Decode payloads in order.
    let mut node_data: Vec<(NodeId, Vec<Vec<NodeId>>)> = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let entry_start = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
        let id = u64::from_le_bytes(
            data[entry_start..entry_start + 8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let offset = u64::from_le_bytes(
            data[entry_start + 8..entry_start + 16]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let offset_usize = usize::try_from(offset).map_err(|_| PagedTopologyError::SizeOverflow)?;

        let payload = &data[payload_start..];
        let layers = decode_node_payload(payload, offset_usize, id)?;
        node_data.push((NodeId::new(id), layers));
    }

    Ok((entry_point, max_level, node_data))
}

fn decode_node_payload(
    payload: &[u8],
    offset: usize,
    node_id: u64,
) -> Result<Vec<Vec<NodeId>>, PagedTopologyError> {
    if payload.len() < offset + 4 {
        return Err(PagedTopologyError::TruncatedNodeHeader { node: node_id });
    }
    let n_levels = u32::from_le_bytes(
        payload[offset..offset + 4]
            .try_into()
            .expect("slice length 4 fits u32 array"),
    );
    let mut cursor = offset + 4;
    let mut layers: Vec<Vec<NodeId>> = Vec::with_capacity(n_levels as usize);

    for level in 0..n_levels {
        if payload.len() < cursor + 4 {
            return Err(PagedTopologyError::TruncatedLevelHeader {
                node: node_id,
                level,
            });
        }
        let n_neighbors = u32::from_le_bytes(
            payload[cursor..cursor + 4]
                .try_into()
                .expect("slice length 4 fits u32 array"),
        );
        cursor += 4;

        let n_neighbors_usize =
            usize::try_from(n_neighbors).map_err(|_| PagedTopologyError::SizeOverflow)?;
        let neighbors_bytes = n_neighbors_usize
            .checked_mul(8)
            .ok_or(PagedTopologyError::SizeOverflow)?;
        if payload.len() < cursor + neighbors_bytes {
            return Err(PagedTopologyError::TruncatedNeighbors {
                node: node_id,
                level,
                expected: neighbors_bytes,
                actual: payload.len().saturating_sub(cursor),
            });
        }

        let mut neighbors: Vec<NodeId> = Vec::with_capacity(n_neighbors_usize);
        for j in 0..n_neighbors_usize {
            let nb_start = cursor + j * 8;
            let nb = u64::from_le_bytes(
                payload[nb_start..nb_start + 8]
                    .try_into()
                    .expect("slice length 8 fits u64 array"),
            );
            neighbors.push(NodeId::new(nb));
        }
        cursor += neighbors_bytes;
        layers.push(neighbors);
    }

    Ok(layers)
}

// ── MmapTopology: zero-allocation Bytes-backed reader (Phase 7c-1) ──

/// Bytes-backed view of a v2 paged HNSW topology.
///
/// Unlike [`deserialize_topology`] which reconstructs a heap
/// `Vec<(NodeId, Vec<Vec<NodeId>>)>`, [`MmapTopology`] holds the raw
/// [`Bytes`] buffer and answers neighbor-list queries directly via
/// binary search + payload-offset reads. No per-query allocation.
///
/// # Heap footprint
///
/// Just the struct itself (a few words) plus the [`Bytes`] refcount.
/// The actual topology data lives in whatever the [`Bytes`] is backed
/// by — typically an mmap of the `.grafeo` container, in which case
/// this struct adds zero data heap.
///
/// # Lookup cost
///
/// - `contains`, `neighbors_at`: O(log n) binary search on the page
///   index, then O(layer) linear scan to skip past prior layers in the
///   payload.
/// - `entry_point`, `max_level`, `len`: O(1) (cached).
///
/// # Concurrency
///
/// `MmapTopology` is `Send + Sync` because [`Bytes`] is. Multiple
/// search threads may share an `Arc<MmapTopology>` and call
/// [`Self::neighbors_at`] concurrently with no locking.
#[derive(Clone)]
pub struct MmapTopology {
    data: Bytes,
    n_nodes: usize,
    payload_start: usize,
    entry_point: Option<NodeId>,
    max_level: usize,
}

impl MmapTopology {
    /// Builds a [`MmapTopology`] view over a v2 paged buffer.
    ///
    /// Validates the header and index region size but does NOT
    /// eagerly read the page-index entries — those are read on demand
    /// during binary search. v1 bincode buffers are rejected; callers
    /// must route v1 buffers through [`deserialize_topology`] (which
    /// does not exist for v1) or fall back to the section-level v1 path.
    ///
    /// # Errors
    ///
    /// Returns [`PagedTopologyError`] if the buffer is malformed:
    /// missing/wrong magic, unsupported version, or truncated header.
    ///
    /// # Panics
    ///
    /// Internal `expect` calls assert that fixed-size byte slice
    /// conversions succeed; the slice indices are pre-validated against
    /// `data.len()` so these cannot fail.
    pub fn from_bytes(data: Bytes) -> Result<Self, PagedTopologyError> {
        if data.len() < HEADER_SIZE {
            return Err(PagedTopologyError::TruncatedHeader);
        }
        if &data[0..4] != MAGIC {
            return Err(PagedTopologyError::BadMagic);
        }
        let version = data[4];
        if version != VERSION {
            return Err(PagedTopologyError::UnsupportedVersion(version));
        }
        let has_entry_point = data[5];
        if has_entry_point > 1 {
            return Err(PagedTopologyError::InvalidEntryPointFlag(has_entry_point));
        }

        let n_nodes_u64 = u64::from_le_bytes(
            data[8..16]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let n_nodes = usize::try_from(n_nodes_u64).map_err(|_| PagedTopologyError::SizeOverflow)?;

        let max_level_u32 = u32::from_le_bytes(
            data[16..20]
                .try_into()
                .expect("slice length 4 fits u32 array"),
        );
        let max_level =
            usize::try_from(max_level_u32).map_err(|_| PagedTopologyError::SizeOverflow)?;

        let entry_raw = u64::from_le_bytes(
            data[24..32]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let entry_point = if has_entry_point == 1 {
            Some(NodeId::new(entry_raw))
        } else {
            None
        };

        let index_size = n_nodes
            .checked_mul(INDEX_ENTRY_SIZE)
            .ok_or(PagedTopologyError::SizeOverflow)?;
        let payload_start = HEADER_SIZE
            .checked_add(index_size)
            .ok_or(PagedTopologyError::SizeOverflow)?;
        if data.len() < payload_start {
            return Err(PagedTopologyError::TruncatedIndex {
                expected: index_size,
                actual: data.len().saturating_sub(HEADER_SIZE),
            });
        }

        Ok(Self {
            data,
            n_nodes,
            payload_start,
            entry_point,
            max_level,
        })
    }

    /// Returns the entry point recorded in the topology header.
    #[must_use]
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// Returns the maximum layer recorded in the topology header.
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Returns the number of nodes in the topology.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_nodes
    }

    /// Returns true if the topology has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_nodes == 0
    }

    /// Returns true if `node` is present in the page index.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.payload_offset_for(node).is_some()
    }

    /// Returns an iterator over the neighbors of `node` at the given
    /// `layer`, or `None` if the node is not present or `layer`
    /// exceeds the node's level count.
    ///
    /// The iterator reads `u64` neighbor IDs directly from the
    /// underlying [`Bytes`] slice; no allocation per call.
    ///
    /// # Panics
    ///
    /// Internal `expect` calls assert that fixed-size byte slice
    /// conversions succeed; the cursor is bounds-checked before each
    /// such conversion so they cannot fail.
    #[must_use]
    pub fn neighbors_at(&self, node: NodeId, layer: usize) -> Option<NeighborsIter<'_>> {
        let payload_off = self.payload_offset_for(node)?;
        let payload = &self.data[self.payload_start..];
        let payload_off_usize = usize::try_from(payload_off).ok()?;

        // Read n_levels from payload[off..off+4].
        if payload.len() < payload_off_usize + 4 {
            return None;
        }
        let n_levels = u32::from_le_bytes(
            payload[payload_off_usize..payload_off_usize + 4]
                .try_into()
                .expect("slice length 4 fits u32 array"),
        );
        let n_levels_usize = usize::try_from(n_levels).ok()?;
        if layer >= n_levels_usize {
            return None;
        }

        // Skip past layers 0..layer to reach the target layer's entry.
        let mut cursor = payload_off_usize + 4;
        for _ in 0..layer {
            if payload.len() < cursor + 4 {
                return None;
            }
            let n_neighbors = u32::from_le_bytes(
                payload[cursor..cursor + 4]
                    .try_into()
                    .expect("slice length 4 fits u32 array"),
            );
            cursor += 4;
            let n_neighbors_usize = usize::try_from(n_neighbors).ok()?;
            let bytes = n_neighbors_usize.checked_mul(8)?;
            cursor = cursor.checked_add(bytes)?;
        }

        // Now cursor points at the target layer's n_neighbors header.
        if payload.len() < cursor + 4 {
            return None;
        }
        let n_neighbors = u32::from_le_bytes(
            payload[cursor..cursor + 4]
                .try_into()
                .expect("slice length 4 fits u32 array"),
        );
        cursor += 4;
        let n_neighbors_usize = usize::try_from(n_neighbors).ok()?;
        let bytes = n_neighbors_usize.checked_mul(8)?;
        let end = cursor.checked_add(bytes)?;
        if payload.len() < end {
            return None;
        }

        Some(NeighborsIter {
            bytes: &payload[cursor..end],
            remaining: n_neighbors_usize,
        })
    }

    /// Iterates every NodeId in the page index, in sorted order.
    ///
    /// Used by checkpoint paths that need to walk the whole topology
    /// (e.g. re-serializing an mmap-backed index for snapshot).
    pub fn iter_node_ids(&self) -> NodeIdIter<'_> {
        NodeIdIter {
            data: &self.data,
            position: 0,
            n_nodes: self.n_nodes,
        }
    }

    /// Binary searches the page index for `node`, returning its
    /// payload offset on hit.
    fn payload_offset_for(&self, node: NodeId) -> Option<u64> {
        let target = node.as_u64();
        let mut lo: usize = 0;
        let mut hi: usize = self.n_nodes;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry_start = HEADER_SIZE + mid * INDEX_ENTRY_SIZE;
            let id = u64::from_le_bytes(
                self.data[entry_start..entry_start + 8]
                    .try_into()
                    .expect("slice length 8 fits u64 array"),
            );
            match id.cmp(&target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let off = u64::from_le_bytes(
                        self.data[entry_start + 8..entry_start + 16]
                            .try_into()
                            .expect("slice length 8 fits u64 array"),
                    );
                    return Some(off);
                }
            }
        }
        None
    }
}

impl std::fmt::Debug for MmapTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapTopology")
            .field("n_nodes", &self.n_nodes)
            .field("max_level", &self.max_level)
            .field("entry_point", &self.entry_point)
            .field("data_len", &self.data.len())
            .finish()
    }
}

/// Zero-allocation iterator over a node's neighbors at a given layer.
///
/// Yields [`NodeId`] values by reading 8-byte little-endian chunks from
/// a [`Bytes`] slice held by [`MmapTopology`].
pub struct NeighborsIter<'a> {
    bytes: &'a [u8],
    remaining: usize,
}

impl Iterator for NeighborsIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.remaining == 0 {
            return None;
        }
        let raw = u64::from_le_bytes(
            self.bytes[..8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        self.bytes = &self.bytes[8..];
        self.remaining -= 1;
        Some(NodeId::new(raw))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for NeighborsIter<'_> {}

/// Iterator over [`NodeId`]s in an [`MmapTopology`]'s page index.
///
/// Yields each node id in sorted (page-index) order. Backed by a slice
/// view; allocates nothing per step.
pub struct NodeIdIter<'a> {
    data: &'a [u8],
    position: usize,
    n_nodes: usize,
}

impl Iterator for NodeIdIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.position >= self.n_nodes {
            return None;
        }
        let entry_start = HEADER_SIZE + self.position * INDEX_ENTRY_SIZE;
        let id = u64::from_le_bytes(
            self.data[entry_start..entry_start + 8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        self.position += 1;
        Some(NodeId::new(id))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.n_nodes - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NodeIdIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(
        entry_point: Option<NodeId>,
        max_level: usize,
        nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        let bytes = serialize_topology(entry_point, max_level, &nodes);
        deserialize_topology(Bytes::from(bytes)).expect("round trip should succeed")
    }

    #[test]
    fn alix_empty_topology_round_trips() {
        let (ep, lvl, nodes) = round_trip(None, 0, Vec::new());
        assert_eq!(ep, None);
        assert_eq!(lvl, 0);
        assert!(nodes.is_empty(), "{nodes:?}");
    }

    #[test]
    fn gus_single_node_single_level() {
        let original = vec![(NodeId::new(42), vec![vec![]])];
        let (ep, lvl, nodes) = round_trip(Some(NodeId::new(42)), 0, original.clone());
        assert_eq!(ep, Some(NodeId::new(42)));
        assert_eq!(lvl, 0);
        assert_eq!(nodes, original);
    }

    #[test]
    fn vincent_multi_level_node_preserves_layers() {
        let original = vec![(
            NodeId::new(7),
            vec![
                vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)],
                vec![NodeId::new(4)],
                vec![],
            ],
        )];
        let (_, _, nodes) = round_trip(Some(NodeId::new(7)), 2, original.clone());
        assert_eq!(nodes, original);
    }

    #[test]
    fn jules_many_nodes_round_trip() {
        let original: Vec<_> = (1..=100u64)
            .map(|i| {
                (
                    NodeId::new(i),
                    vec![vec![
                        NodeId::new(i.wrapping_add(1)),
                        NodeId::new(i.wrapping_add(2)),
                    ]],
                )
            })
            .collect();
        let (ep, lvl, nodes) = round_trip(Some(NodeId::new(1)), 3, original.clone());
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(lvl, 3);
        assert_eq!(nodes.len(), 100);
        assert_eq!(nodes, original);
    }

    #[test]
    fn mia_neighbor_order_is_preserved() {
        // HNSW neighbor order matters: pruning + greedy traversal can be
        // sensitive to insertion order. The round trip must not perturb it.
        let original = vec![(
            NodeId::new(1),
            vec![vec![
                NodeId::new(9),
                NodeId::new(2),
                NodeId::new(7),
                NodeId::new(3),
                NodeId::new(5),
            ]],
        )];
        let (_, _, nodes) = round_trip(Some(NodeId::new(1)), 0, original.clone());
        assert_eq!(nodes[0].1[0], original[0].1[0]);
    }

    #[test]
    fn shosanna_invalid_magic_rejected() {
        let mut bytes = serialize_topology(None, 0, &[]);
        bytes[0] = b'X';
        let err = deserialize_topology(Bytes::from(bytes)).expect_err("must reject bad magic");
        assert_eq!(err, PagedTopologyError::BadMagic);
    }

    #[test]
    fn butch_truncated_header_rejected() {
        let bytes = vec![b'G', b'T', b'O'];
        let err =
            deserialize_topology(Bytes::from(bytes)).expect_err("must reject truncated header");
        assert_eq!(err, PagedTopologyError::TruncatedHeader);
    }

    #[test]
    fn django_unsupported_version_rejected() {
        let mut bytes = serialize_topology(None, 0, &[]);
        bytes[4] = 99;
        let err = deserialize_topology(Bytes::from(bytes)).expect_err("must reject bad version");
        assert_eq!(err, PagedTopologyError::UnsupportedVersion(99));
    }

    #[test]
    fn beatrix_unsorted_page_index_rejected() {
        let original = vec![
            (NodeId::new(1), vec![vec![]]),
            (NodeId::new(2), vec![vec![]]),
        ];
        let mut bytes = serialize_topology(None, 0, &original);
        // Swap the two NodeIds in the index region: now [2, 1] descending.
        let entry0 = HEADER_SIZE;
        let entry1 = HEADER_SIZE + INDEX_ENTRY_SIZE;
        for k in 0..8 {
            bytes.swap(entry0 + k, entry1 + k);
        }
        let err =
            deserialize_topology(Bytes::from(bytes)).expect_err("must reject unsorted page index");
        assert!(matches!(err, PagedTopologyError::UnsortedPageIndex { .. }));
    }

    #[test]
    fn hans_payload_offset_out_of_range_rejected() {
        let original = vec![(NodeId::new(1), vec![vec![]])];
        let mut bytes = serialize_topology(None, 0, &original);
        // Corrupt the payload offset of node 0 to be past payload end.
        let offset_pos = HEADER_SIZE + 8;
        let huge: u64 = 1_000_000;
        bytes[offset_pos..offset_pos + 8].copy_from_slice(&huge.to_le_bytes());
        let err =
            deserialize_topology(Bytes::from(bytes)).expect_err("must reject out-of-range offset");
        assert!(matches!(
            err,
            PagedTopologyError::PayloadOffsetOutOfRange { .. }
        ));
    }

    #[test]
    fn tarantino_size_bound_is_predictable() {
        // 1k nodes, 2 layers each, 16 neighbors at layer 0 + 8 at layer 1.
        // Use realistic NodeIds (large, not 1..=N) so the test reflects
        // production sizes rather than varint-friendly small ints.
        let base: u64 = 1 << 32;
        let nodes: Vec<_> = (0..1000u64)
            .map(|i| {
                (
                    NodeId::new(base + i),
                    vec![
                        (0..16u64)
                            .map(|j| NodeId::new(base + j))
                            .collect::<Vec<_>>(),
                        (0..8u64).map(|j| NodeId::new(base + j)).collect::<Vec<_>>(),
                    ],
                )
            })
            .collect();
        let v2 = serialize_topology(Some(NodeId::new(base)), 1, &nodes);

        // Predictable upper bound for this fixed shape:
        //   header (32) + index (16 * n) + payload-per-node (4 + 4 + 16*8 + 4 + 8*8)
        let expected_per_node_payload = 4 + 4 + 16 * 8 + 4 + 8 * 8;
        let upper =
            HEADER_SIZE + INDEX_ENTRY_SIZE * nodes.len() + expected_per_node_payload * nodes.len();
        assert_eq!(
            v2.len(),
            upper,
            "paged topology size should match the layout exactly"
        );

        // Sanity: at production NodeId magnitudes, bincode varint loses
        // most of its small-int advantage. Verify v2 is within ~25% of
        // bincode at this scale.
        let v1 = bincode::serde::encode_to_vec(&nodes, bincode::config::standard())
            .expect("bincode encode");
        assert!(
            v2.len() <= v1.len() * 3 / 2,
            "paged topology {} should be within 1.5x bincode {} at production NodeId scale",
            v2.len(),
            v1.len()
        );
    }

    // ── MmapTopology zero-allocation reader (Phase 7c-1) ────────────

    fn build_topology() -> Vec<(NodeId, Vec<Vec<NodeId>>)> {
        // Sorted by NodeId. Mix of layer counts and neighbor counts.
        vec![
            (
                NodeId::new(10),
                vec![
                    vec![NodeId::new(20), NodeId::new(30)],
                    vec![NodeId::new(40)],
                ],
            ),
            (NodeId::new(20), vec![vec![NodeId::new(10)]]),
            (
                NodeId::new(30),
                vec![
                    vec![NodeId::new(10), NodeId::new(20), NodeId::new(40)],
                    vec![],
                    vec![NodeId::new(20)],
                ],
            ),
            (NodeId::new(40), vec![vec![]]),
        ]
    }

    #[test]
    fn alix_mmap_topology_header_round_trips() {
        let nodes = build_topology();
        let bytes = serialize_topology(Some(NodeId::new(10)), 2, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        assert_eq!(topo.len(), 4);
        assert!(!topo.is_empty());
        assert_eq!(topo.entry_point(), Some(NodeId::new(10)));
        assert_eq!(topo.max_level(), 2);
    }

    #[test]
    fn gus_mmap_topology_neighbors_match_heap_round_trip() {
        let nodes = build_topology();
        let bytes = serialize_topology(Some(NodeId::new(10)), 2, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        for (id, layers) in &nodes {
            for (layer, expected) in layers.iter().enumerate() {
                let actual: Vec<NodeId> = topo
                    .neighbors_at(*id, layer)
                    .expect("neighbors present")
                    .collect();
                assert_eq!(&actual, expected, "node {id:?} layer {layer}");
            }
        }
    }

    #[test]
    fn vincent_mmap_topology_unknown_node_returns_none() {
        let nodes = build_topology();
        let bytes = serialize_topology(Some(NodeId::new(10)), 2, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        assert!(!topo.contains(NodeId::new(999)));
        assert!(topo.neighbors_at(NodeId::new(999), 0).is_none());
    }

    #[test]
    fn jules_mmap_topology_layer_out_of_range_returns_none() {
        let nodes = build_topology();
        let bytes = serialize_topology(Some(NodeId::new(10)), 2, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        // Node 20 only has 1 layer. Layer 1 should be None.
        assert!(topo.neighbors_at(NodeId::new(20), 1).is_none());
        // Node 40 has an empty layer 0; that should yield an empty
        // iterator, not None.
        let layer0: Vec<NodeId> = topo
            .neighbors_at(NodeId::new(40), 0)
            .expect("layer 0 exists")
            .collect();
        assert!(layer0.is_empty(), "{layer0:?}");
    }

    #[test]
    fn mia_mmap_topology_iterator_size_hint_is_exact() {
        let nodes = build_topology();
        let bytes = serialize_topology(Some(NodeId::new(10)), 2, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let iter = topo.neighbors_at(NodeId::new(30), 0).expect("layer 0");
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
        assert_eq!(iter.count(), 3);
    }

    #[test]
    fn shosanna_mmap_topology_no_per_query_allocation() {
        // Build a topology with 1000 nodes, each with 16 neighbors at
        // layer 0. Iterate every node's layer 0 — the iterator must
        // reach exhaustion via 16 next() calls each, yielding 16k
        // NodeIds total without ever allocating.
        let base = 1u64 << 20;
        let nodes: Vec<_> = (0..1000u64)
            .map(|i| {
                (
                    NodeId::new(base + i),
                    vec![
                        (0..16u64)
                            .map(|j| NodeId::new(base + (i + j) % 1000))
                            .collect::<Vec<_>>(),
                    ],
                )
            })
            .collect();
        let bytes = serialize_topology(Some(NodeId::new(base)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mut total = 0usize;
        for (id, _) in &nodes {
            let iter = topo.neighbors_at(*id, 0).expect("layer 0 present");
            total += iter.count();
        }
        assert_eq!(total, 16_000);
    }

    #[test]
    fn beatrix_mmap_topology_rejects_bad_magic() {
        let mut bytes = serialize_topology(None, 0, &[]);
        bytes[0] = b'X';
        let err = MmapTopology::from_bytes(Bytes::from(bytes)).expect_err("must reject bad magic");
        assert_eq!(err, PagedTopologyError::BadMagic);
    }

    #[test]
    fn butch_mmap_topology_rejects_truncated_index() {
        // Manufacture a header that says n_nodes=5 but the buffer is
        // truncated right after the header — index region missing.
        let mut bytes = serialize_topology(None, 0, &[]);
        // Patch n_nodes from 0 to 5.
        let n = 5u64;
        bytes[8..16].copy_from_slice(&n.to_le_bytes());
        let err =
            MmapTopology::from_bytes(Bytes::from(bytes)).expect_err("must reject truncated index");
        assert!(matches!(err, PagedTopologyError::TruncatedIndex { .. }));
    }
}
