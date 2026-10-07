//! Compressed Sparse Row adjacency representation.
//!
//! For node i, its neighbors are `targets[offsets[i]..offsets[i+1]]`.
//! Uses u32 for both offsets and targets (max ~4B nodes/edges per table).

/// Compressed Sparse Row adjacency structure.
///
/// Stores a directed graph in two flat arrays: `offsets` (one per node + 1
/// sentinel) and `targets` (concatenated neighbor lists). This layout is
/// cache-friendly for forward traversal and has O(1) neighbor access.
#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    /// One entry per node plus a trailing sentinel.
    /// `offsets[i]..offsets[i+1]` is the range in `targets` for node `i`.
    offsets: Vec<u32>,
    /// Concatenated target node offsets, grouped by source.
    targets: Vec<u32>,
    /// Optional per-edge auxiliary data, parallel to `targets`.
    /// For backward CSRs, stores the corresponding forward CSR position.
    edge_data: Option<Vec<u32>>,
}

impl CsrAdjacency {
    /// Builds a CSR from pre-sorted `(src, dst)` pairs.
    ///
    /// The input **must** be sorted by `src` (ties broken arbitrarily).
    /// `num_nodes` is the total number of source nodes, nodes beyond the
    /// highest `src` in `edges` are treated as having zero out-degree.
    ///
    /// # Panics
    ///
    /// Panics if `edges` is not sorted by source.
    #[must_use]
    pub fn from_sorted_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        assert!(
            edges.windows(2).all(|w| w[0].0 <= w[1].0),
            "edges must be sorted by source"
        );

        let mut offsets = vec![0u32; num_nodes + 1];

        // Count edges per source.
        for &(src, _) in edges {
            offsets[src as usize + 1] += 1;
        }

        // Prefix sum.
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let targets: Vec<u32> = edges.iter().map(|&(_, dst)| dst).collect();

        Self {
            offsets,
            targets,
            edge_data: None,
        }
    }

    /// Sets optional per-edge auxiliary data parallel to `targets`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `self.targets.len()`.
    pub fn set_edge_data(&mut self, data: Vec<u32>) {
        assert_eq!(
            data.len(),
            self.targets.len(),
            "edge_data length must equal targets length"
        );
        self.edge_data = Some(data);
    }

    /// Returns `true` if per-edge auxiliary data has been set.
    #[must_use]
    pub fn has_edge_data(&self) -> bool {
        self.edge_data.is_some()
    }

    /// Returns the auxiliary data for the edge at the given CSR position.
    ///
    /// Returns `None` if no edge data has been set, or if the position is
    /// out of bounds.
    #[must_use]
    pub fn edge_data_at(&self, position: usize) -> Option<u32> {
        self.edge_data.as_ref()?.get(position).copied()
    }

    /// Returns the number of nodes in this CSR.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        // offsets has num_nodes + 1 entries.
        self.offsets.len().saturating_sub(1)
    }

    /// Returns the total number of edges in this CSR.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.targets.len()
    }

    /// Returns the neighbors (target offsets) of the given node.
    ///
    /// Returns an empty slice if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn neighbors(&self, node_offset: u32) -> &[u32] {
        let i = node_offset as usize;
        if i + 1 >= self.offsets.len() {
            return &[];
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.targets[start..end]
    }

    /// Returns the out-degree of the given node.
    ///
    /// Returns 0 if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn degree(&self, node_offset: u32) -> usize {
        self.neighbors(node_offset).len()
    }

    /// Finds the source node for a given CSR position via binary search.
    ///
    /// The CSR position is an index into `targets`. This method returns the
    /// node offset `i` such that `offsets[i] <= position < offsets[i+1]`.
    /// Returns `None` if `position` is out of range.
    #[must_use]
    pub fn source_for_position(&self, position: u32) -> Option<u32> {
        if position as usize >= self.targets.len() {
            return None;
        }

        // Binary search: find the last offset <= position.
        // offsets is monotonically non-decreasing with len = num_nodes + 1.
        let num_nodes = self.num_nodes();
        let mut lo = 0usize;
        let mut hi = num_nodes;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.offsets[mid + 1] <= position {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // reason: CSR position index fits u32
        #[allow(clippy::cast_possible_truncation)]
        Some(lo as u32)
    }

    /// Returns the starting CSR position (index into `targets`) for the given node.
    ///
    /// This is `offsets[node_offset]`, the index at which this node's
    /// neighbor list begins in the targets array.
    ///
    /// Returns 0 if `node_offset` is out of range.
    #[inline]
    #[must_use]
    pub fn offset_of(&self, node_offset: u32) -> u32 {
        let i = node_offset as usize;
        if i >= self.offsets.len() {
            return 0;
        }
        self.offsets[i]
    }

    /// Reconstructs from pre-built raw parts.
    ///
    /// Used by section deserialization.
    #[must_use]
    pub fn from_raw_parts(
        offsets: Vec<u32>,
        targets: Vec<u32>,
        edge_data: Option<Vec<u32>>,
    ) -> Self {
        Self {
            offsets,
            targets,
            edge_data,
        }
    }

    /// Returns the raw offsets array.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Returns the raw targets array.
    #[must_use]
    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    /// Returns the raw edge_data array, if set.
    #[must_use]
    pub fn edge_data(&self) -> Option<&[u32]> {
        self.edge_data.as_deref()
    }

    /// Serializes this CSR to a byte buffer.
    ///
    /// # Errors
    ///
    /// Fails if an array length does not fit the format's `u32` fields.
    pub fn write_to(&self, buf: &mut Vec<u8>) -> grafeo_common::utils::error::Result<()> {
        // offsets
        write_usize_as_u32(buf, self.offsets.len())?;
        for &o in &self.offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        // targets
        write_usize_as_u32(buf, self.targets.len())?;
        for &t in &self.targets {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        // edge_data
        match &self.edge_data {
            Some(ed) => {
                buf.push(1);
                write_usize_as_u32(buf, ed.len())?;
                for &d in ed {
                    buf.extend_from_slice(&d.to_le_bytes());
                }
            }
            None => buf.push(0),
        }
        Ok(())
    }

    /// Deserializes a CSR from a byte buffer at the given offset.
    ///
    /// The counts and offsets come from a file, so they are checked before
    /// they are trusted: no array is allocated larger than the bytes left can
    /// fill, the offsets start at 0, never decrease and end at the number of
    /// targets (so `neighbors` never slices past them), and edge data, when
    /// present, has one entry per target.
    ///
    /// # Errors
    ///
    /// Returns an error string if data is truncated or does not describe an
    /// adjacency.
    pub fn read_from(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let offsets = read_u32_array(data, pos, "CSR offsets do not fit the bytes left")?;
        let targets = read_u32_array(data, pos, "CSR targets do not fit the bytes left")?;
        let has_edge_data = *data.get(*pos).ok_or("truncated edge_data flag")?;
        *pos += 1;
        let edge_data = match has_edge_data {
            0 => None,
            1 => Some(read_u32_array(
                data,
                pos,
                "CSR edge data does not fit the bytes left",
            )?),
            _ => return Err("CSR edge data flag is neither 0 nor 1"),
        };
        match offsets.first() {
            Some(0) => {}
            Some(_) => return Err("CSR offsets do not start at 0"),
            None => return Err("CSR has no offsets"),
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("CSR offsets decrease");
        }
        if offsets.last().map(|&last| last as usize) != Some(targets.len()) {
            return Err("CSR offsets do not end at the number of targets");
        }
        if edge_data
            .as_ref()
            .is_some_and(|edge_data| edge_data.len() != targets.len())
        {
            return Err("CSR edge data and targets differ in length");
        }
        Ok(Self::from_raw_parts(offsets, targets, edge_data))
    }

    /// Returns the approximate heap memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<u32>()
            + self.targets.len() * std::mem::size_of::<u32>()
            + self
                .edge_data
                .as_ref()
                .map_or(0, |d| d.len() * std::mem::size_of::<u32>())
    }
}

fn write_usize_as_u32(buf: &mut Vec<u8>, v: usize) -> grafeo_common::utils::error::Result<()> {
    let n = crate::codec::limits::checked_u32(v, "compact store adjacency size")?;
    buf.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

/// Reads a `u32` count and that many `u32` values, refusing a count the
/// bytes left cannot hold before anything is allocated for it.
fn read_u32_array(
    data: &[u8],
    pos: &mut usize,
    too_long: &'static str,
) -> Result<Vec<u32>, &'static str> {
    let count = read_u32_le(data, pos)? as usize;
    let left = data.len().saturating_sub(*pos);
    if count.checked_mul(4).is_none_or(|needed| needed > left) {
        return Err(too_long);
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_u32_le(data, pos)?);
    }
    Ok(values)
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_csr() {
        // 3 nodes, edges: 0->1, 0->2, 1->2
        let edges = vec![(0u32, 1u32), (0, 2), (1, 2)];
        let csr = CsrAdjacency::from_sorted_edges(3, &edges);

        assert_eq!(csr.num_nodes(), 3);
        assert_eq!(csr.num_edges(), 3);

        // Node 0: neighbors [1, 2]
        assert_eq!(csr.neighbors(0), &[1, 2]);
        assert_eq!(csr.degree(0), 2);

        // Node 1: neighbors [2]
        assert_eq!(csr.neighbors(1), &[2]);
        assert_eq!(csr.degree(1), 1);

        // Node 2: no neighbors
        assert_eq!(csr.neighbors(2), &[] as &[u32]);
        assert_eq!(csr.degree(2), 0);
    }

    #[test]
    fn test_source_for_position() {
        // 3 nodes, edges: 0->1, 0->2, 1->2
        // CSR targets: [1, 2, 2]
        // offsets:      [0, 2, 3, 3]
        // position 0 -> source 0 (0->1)
        // position 1 -> source 0 (0->2)
        // position 2 -> source 1 (1->2)
        let edges = vec![(0u32, 1u32), (0, 2), (1, 2)];
        let csr = CsrAdjacency::from_sorted_edges(3, &edges);

        assert_eq!(csr.source_for_position(0), Some(0));
        assert_eq!(csr.source_for_position(1), Some(0));
        assert_eq!(csr.source_for_position(2), Some(1));

        // Out of range.
        assert_eq!(csr.source_for_position(3), None);
        assert_eq!(csr.source_for_position(100), None);
    }

    #[test]
    fn test_empty_graph() {
        // 0 nodes, 0 edges.
        let csr = CsrAdjacency::from_sorted_edges(0, &[]);
        assert_eq!(csr.num_nodes(), 0);
        assert_eq!(csr.num_edges(), 0);
        assert_eq!(csr.source_for_position(0), None);
        assert_eq!(csr.memory_bytes(), 4); // 1 offset entry (sentinel)
    }

    /// An adjacency as `read_from` reads it: the counts as given, then the
    /// values.
    fn encoded(
        offsets: (u32, &[u32]),
        targets: (u32, &[u32]),
        edge_data: Option<(u32, &[u32])>,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (count, values) in [offsets, targets] {
            bytes.extend_from_slice(&count.to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        match edge_data {
            Some((count, values)) => {
                bytes.push(1);
                bytes.extend_from_slice(&count.to_le_bytes());
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            None => bytes.push(0),
        }
        bytes
    }

    /// Counts and offsets read from a file are untrusted: a count the bytes
    /// left cannot hold, offsets that do not describe the targets and edge
    /// data of another length are errors, never an abort or a later panic
    /// in `neighbors`.
    #[test]
    fn crafted_adjacencies_are_refused() {
        let valid = encoded((3, &[0, 1, 2]), (2, &[1, 0]), Some((2, &[1, 0])));
        let csr = CsrAdjacency::read_from(&valid, &mut 0).unwrap();
        assert_eq!((csr.num_nodes(), csr.num_edges()), (2, 2));

        for (case, bytes, expected) in [
            (
                "an offset count",
                encoded((u32::MAX, &[]), (0, &[]), None),
                "offsets",
            ),
            (
                "a target count",
                encoded((3, &[0, 1, 2]), (u32::MAX, &[]), None),
                "targets",
            ),
            (
                "an edge data count",
                encoded((3, &[0, 1, 2]), (2, &[1, 0]), Some((u32::MAX, &[]))),
                "edge data",
            ),
            ("no offsets", encoded((0, &[]), (0, &[]), None), "offsets"),
            (
                "a first offset past 0",
                encoded((2, &[1, 1]), (1, &[0]), None),
                "offsets",
            ),
            (
                "offsets that run backwards",
                encoded((3, &[0, 2, 1]), (2, &[1, 0]), None),
                "offsets",
            ),
            (
                "offsets that dip and end right",
                encoded((4, &[0, 2, 1, 2]), (2, &[1, 0]), None),
                "decrease",
            ),
            (
                "a last offset short of the targets",
                encoded((3, &[0, 1, 1]), (2, &[1, 0]), None),
                "offsets",
            ),
            (
                "edge data of another length",
                encoded((3, &[0, 1, 2]), (2, &[1, 0]), Some((1, &[0]))),
                "edge data",
            ),
        ] {
            let error = CsrAdjacency::read_from(&bytes, &mut 0).unwrap_err();
            assert!(error.contains(expected), "{case}: {error}");
        }
    }
}
