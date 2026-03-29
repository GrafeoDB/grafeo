//! NodeId and EdgeId encoding for CompactStore.
//!
//! Packs a (table_id, offset) pair into a single u64. Bit 63 is always 0
//! to survive the `as i64` round-trip in existing operator code
//! (mutation.rs, project.rs, filter.rs).

use grafeo_common::types::{EdgeId, NodeId};

/// Mask for the lower 48 bits (offset field).
const OFFSET_MASK: u64 = (1 << 48) - 1;

/// Maximum table_id value (15 bits).
const MAX_TABLE_ID: u16 = (1 << 15) - 1;

/// Maximum offset value (48 bits).
const MAX_OFFSET: u64 = OFFSET_MASK;

/// Encodes a table ID and offset into a [`NodeId`].
///
/// Layout: `[63] = 0 | [62:48] = table_id (15 bits) | [47:0] = offset (48 bits)`.
///
/// # Panics
///
/// Panics if `table_id > 32767` or `offset > 2^48 - 1`.
#[inline]
#[must_use]
pub fn encode_node_id(table_id: u16, offset: u64) -> NodeId {
    assert!(
        table_id <= MAX_TABLE_ID,
        "table_id {table_id} exceeds 15-bit max {MAX_TABLE_ID}"
    );
    assert!(
        offset <= MAX_OFFSET,
        "offset {offset} exceeds 48-bit max {MAX_OFFSET}"
    );
    let raw = (u64::from(table_id) << 48) | (offset & OFFSET_MASK);
    NodeId::new(raw)
}

/// Decodes a [`NodeId`] back into its (table_id, offset) pair.
///
/// Returns the table_id in the upper 15 bits and the offset in the lower 48 bits.
/// Bit 63 is masked off.
#[inline]
#[must_use]
pub fn decode_node_id(id: NodeId) -> (u16, u64) {
    let raw = id.as_u64();
    let table_id = ((raw >> 48) & 0x7FFF) as u16;
    let offset = raw & OFFSET_MASK;
    (table_id, offset)
}

/// Encodes a relationship table ID and CSR position into an [`EdgeId`].
///
/// Layout: `[63] = 0 | [62:48] = rel_table_id (15 bits) | [47:0] = csr_position (48 bits)`.
///
/// # Panics
///
/// Panics if `rel_table_id > 32767` or `csr_position > 2^48 - 1`.
#[inline]
#[must_use]
pub fn encode_edge_id(rel_table_id: u16, csr_position: u64) -> EdgeId {
    assert!(
        rel_table_id <= MAX_TABLE_ID,
        "rel_table_id {rel_table_id} exceeds 15-bit max {MAX_TABLE_ID}"
    );
    assert!(
        csr_position <= MAX_OFFSET,
        "csr_position {csr_position} exceeds 48-bit max {MAX_OFFSET}"
    );
    let raw = (u64::from(rel_table_id) << 48) | (csr_position & OFFSET_MASK);
    EdgeId::new(raw)
}

/// Decodes an [`EdgeId`] back into its (rel_table_id, csr_position) pair.
///
/// Returns the rel_table_id in the upper 15 bits and the csr_position in the
/// lower 48 bits. Bit 63 is masked off.
#[inline]
#[must_use]
pub fn decode_edge_id(id: EdgeId) -> (u16, u64) {
    let raw = id.as_u64();
    let rel_table_id = ((raw >> 48) & 0x7FFF) as u16;
    let csr_position = raw & OFFSET_MASK;
    (rel_table_id, csr_position)
}

/// Returns `true` if the given [`NodeId`] was produced by [`encode_node_id`].
///
/// Compact IDs always have bit 63 = 0 and are never [`NodeId::INVALID`].
#[inline]
#[must_use]
pub fn is_compact_id(id: NodeId) -> bool {
    let raw = id.as_u64();
    // Bit 63 must be 0 and the ID must not be INVALID (u64::MAX, which has bit 63 = 1).
    (raw & (1 << 63)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_round_trip() {
        let table_id = 42u16;
        let offset = 123_456u64;
        let id = encode_node_id(table_id, offset);
        let (decoded_table, decoded_offset) = decode_node_id(id);
        assert_eq!(decoded_table, table_id);
        assert_eq!(decoded_offset, offset);
    }

    #[test]
    fn test_edge_id_round_trip() {
        let rel_table_id = 7u16;
        let csr_position = 999_999u64;
        let id = encode_edge_id(rel_table_id, csr_position);
        let (decoded_table, decoded_pos) = decode_edge_id(id);
        assert_eq!(decoded_table, rel_table_id);
        assert_eq!(decoded_pos, csr_position);
    }

    #[test]
    fn test_i64_cast_survival() {
        // Bit 63 is always 0, so casting to i64 and back must preserve the value.
        let id = encode_node_id(100, 500);
        let raw = id.as_u64();
        let as_signed = raw as i64;
        assert!(as_signed >= 0, "compact NodeId must survive as i64 cast");
        let back = as_signed as u64;
        assert_eq!(back, raw);

        let edge_id = encode_edge_id(200, 1000);
        let raw = edge_id.as_u64();
        let as_signed = raw as i64;
        assert!(as_signed >= 0, "compact EdgeId must survive as i64 cast");
        let back = as_signed as u64;
        assert_eq!(back, raw);
    }

    #[test]
    fn test_max_values() {
        let max_table = MAX_TABLE_ID; // 32767
        let max_offset = MAX_OFFSET; // 2^48 - 1

        let id = encode_node_id(max_table, max_offset);
        let (decoded_table, decoded_offset) = decode_node_id(id);
        assert_eq!(decoded_table, max_table);
        assert_eq!(decoded_offset, max_offset);

        // Must still survive i64 cast at max values.
        let raw = id.as_u64();
        let as_signed = raw as i64;
        assert!(as_signed >= 0);
        assert_eq!(as_signed as u64, raw);
    }

    #[test]
    fn test_never_collides_with_invalid() {
        // NodeId::INVALID is u64::MAX which has bit 63 = 1.
        // Any compact ID has bit 63 = 0, so they can never be INVALID.
        let id = encode_node_id(MAX_TABLE_ID, MAX_OFFSET);
        assert_ne!(id, NodeId::INVALID);

        let edge_id = encode_edge_id(MAX_TABLE_ID, MAX_OFFSET);
        assert_ne!(edge_id, EdgeId::INVALID);

        // Even (0, 0) must not be INVALID.
        let zero_id = encode_node_id(0, 0);
        assert_ne!(zero_id, NodeId::INVALID);
    }

    #[test]
    fn test_is_compact_id() {
        assert!(is_compact_id(encode_node_id(0, 0)));
        assert!(is_compact_id(encode_node_id(1, 42)));
        assert!(is_compact_id(encode_node_id(MAX_TABLE_ID, MAX_OFFSET)));

        // NodeId::INVALID has bit 63 = 1, so it is not a compact ID.
        assert!(!is_compact_id(NodeId::INVALID));
    }

    #[test]
    fn test_zero_table_id() {
        let id = encode_node_id(0, 12345);
        let (table_id, offset) = decode_node_id(id);
        assert_eq!(table_id, 0);
        assert_eq!(offset, 12345);
    }

    #[test]
    fn test_zero_offset() {
        let id = encode_node_id(500, 0);
        let (table_id, offset) = decode_node_id(id);
        assert_eq!(table_id, 500);
        assert_eq!(offset, 0);
    }
}
