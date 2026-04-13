//! ID offset computation for hybrid store ID partitioning.
//!
//! Compact IDs live below the offset, overlay IDs above it.

use crate::graph::compact::CompactStore;
use crate::graph::compact::id::{encode_edge_id, encode_node_id};

/// Computes the ID boundary for a compact store.
///
/// Scans all node tables AND relationship tables to find the maximum
/// encoded ID (node or edge), then returns `max + 1`. Overlay LpgStore
/// IDs start at this value, guaranteeing no overlap with compact IDs.
///
/// Both node IDs and edge IDs share the same bit layout
/// (`[63]=0 | [62:48]=table_id | [47:0]=offset`), so the offset must
/// cover the maximum across both ID types to prevent collisions.
///
/// Returns 0 for an empty CompactStore.
#[must_use]
pub fn compute_id_offset(compact: &CompactStore) -> u64 {
    let mut has_entities = false;
    let mut max_id: u64 = 0;

    for (table_id, label) in compact.table_id_to_label().iter().enumerate() {
        if let Some(nt) = compact.node_table(label) {
            let len = nt.len();
            if len > 0 {
                has_entities = true;
                let id = encode_node_id(table_id as u16, (len - 1) as u64);
                max_id = max_id.max(id.as_u64());
            }
        }
    }

    for (rel_id, rt) in compact.rel_tables_by_id().iter().enumerate() {
        let num = rt.num_edges();
        if num > 0 {
            has_entities = true;
            let id = encode_edge_id(rel_id as u16, (num - 1) as u64);
            max_id = max_id.max(id.as_u64());
        }
    }

    if has_entities { max_id + 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::CompactStoreBuilder;

    fn empty_compact() -> CompactStore {
        CompactStoreBuilder::new().build().unwrap()
    }

    fn compact_with_students(n: usize) -> CompactStore {
        let grades: Vec<u64> = (0..n).map(|i| i as u64).collect();
        CompactStoreBuilder::new()
            .node_table("Student", |b| {
                b.column_bitpacked("grade", &grades, 8)
            })
            .build()
            .unwrap()
    }

    #[test]
    fn offset_empty_compact() {
        let compact = empty_compact();
        assert_eq!(compute_id_offset(&compact), 0);
    }

    #[test]
    fn offset_single_table() {
        let compact = compact_with_students(100);
        let offset = compute_id_offset(&compact);
        assert!(offset > 0);
        assert_eq!(
            offset,
            crate::graph::compact::id::encode_node_id(0, 99).as_u64() + 1
        );
    }

    #[test]
    fn offset_accounts_for_edge_ids() {
        // Edge IDs use the same bit layout as node IDs:
        //   [62:48]=table_id, [47:0]=offset
        // With a single node table (table_id=0) and a single rel table
        // (rel_table_id=0), the offset field determines which is larger.
        // Here 1000 edges > 5 nodes, so max_edge_id > max_node_id.
        let grades: Vec<u64> = (0..5).collect();
        let edges: Vec<(u32, u32)> = (0..1000).map(|i| (i % 5, i % 5)).collect();
        let compact = CompactStoreBuilder::new()
            .node_table("A", |b| b.column_bitpacked("x", &grades, 4))
            .rel_table("R", "A", "A", |b| b.edges(edges))
            .build()
            .unwrap();
        let offset = compute_id_offset(&compact);
        // Both table_id=0, so max is determined by offset field
        let max_edge = crate::graph::compact::id::encode_edge_id(0, 999).as_u64();
        let max_node = crate::graph::compact::id::encode_node_id(0, 4).as_u64();
        assert!(max_edge > max_node);
        assert_eq!(offset, max_edge + 1);
    }

    #[test]
    fn offset_multiple_tables() {
        let grades: Vec<u64> = (0..50).collect();
        let levels: Vec<u64> = (0..200).collect();
        let compact = CompactStoreBuilder::new()
            .node_table("Student", |b| {
                b.column_bitpacked("grade", &grades, 8)
            })
            .node_table("Course", |b| {
                b.column_bitpacked("level", &levels, 8)
            })
            .build()
            .unwrap();
        let offset = compute_id_offset(&compact);
        let max_id = crate::graph::compact::id::encode_node_id(1, 199).as_u64();
        assert_eq!(offset, max_id + 1);
    }
}
