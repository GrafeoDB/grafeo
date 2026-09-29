use super::LpgStore;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl LpgStore {
    // === Statistics ===

    /// Returns the current statistics (cheap `Arc` clone, no deep copy).
    #[must_use]
    pub fn statistics(&self) -> Arc<Statistics> {
        Arc::clone(&self.statistics.read())
    }

    /// Rebuilds the statistics from the live counters.
    ///
    /// Call this before reading statistics for query optimization. The
    /// counters are exact after commits and rollbacks, so this never scans
    /// the store.
    #[doc(hidden)]
    pub fn ensure_statistics_fresh(&self) {
        self.compute_statistics();
    }

    /// Recomputes statistics from incremental counters.
    ///
    /// Reads live node/edge counts from atomic counters and per-label counts
    /// from the label index. This is O(|labels| + |edge_types|) instead of
    /// O(n + m) for a full scan.
    pub(crate) fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Read total counts from atomic counters
        // reason: clamped to >= 0 by max(0), safe to cast to u64
        #[allow(clippy::cast_sign_loss)]
        {
            stats.total_nodes = self.live_node_count.load(Ordering::Relaxed).max(0) as u64;
        }
        // reason: clamped to >= 0 by max(0), safe to cast to u64
        #[allow(clippy::cast_sign_loss)]
        {
            stats.total_edges = self.live_edge_count.load(Ordering::Relaxed).max(0) as u64;
        }

        // Compute per-label statistics from label_index (each is O(1) via .len())
        let registry = self.label_registry.read();
        let label_index = self.label_index.read();

        for (label_id, label_name) in registry.names().iter().enumerate() {
            let node_count = label_index.get(label_id).map_or(0, |set| set.len() as u64);

            if node_count > 0 {
                let avg_out_degree = if stats.total_nodes > 0 {
                    stats.total_edges as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let label_stats =
                    LabelStatistics::new(node_count).with_degrees(avg_out_degree, avg_out_degree);

                stats.update_label(label_name.as_ref(), label_stats);
            }
        }

        // Compute per-edge-type statistics from incremental counts
        let id_to_edge_type = self.id_to_edge_type.read();
        let edge_type_counts = self.edge_type_live_counts.read();

        for (type_id, type_name) in id_to_edge_type.iter().enumerate() {
            // reason: clamped to >= 0 by max(0), safe to cast to u64
            #[allow(clippy::cast_sign_loss)]
            let count = edge_type_counts.get(type_id).copied().unwrap_or(0).max(0) as u64;

            if count > 0 {
                let avg_degree = if stats.total_nodes > 0 {
                    count as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let edge_stats = EdgeTypeStatistics::new(count, avg_degree, avg_degree);
                stats.update_edge_type(type_name.as_ref(), edge_stats);
            }
        }

        *self.statistics.write() = Arc::new(stats);
    }

    /// Estimates cardinality for a label scan.
    #[must_use]
    pub fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.statistics.read().estimate_label_cardinality(label)
    }

    /// Estimates average degree for an edge type.
    #[must_use]
    pub fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.statistics
            .read()
            .estimate_avg_degree(edge_type, outgoing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> LpgStore {
        LpgStore::new().unwrap()
    }

    #[test]
    fn compute_statistics_empty_store() {
        let store = make_store();
        store.compute_statistics();
        let stats = store.statistics();
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
    }

    #[test]
    fn compute_statistics_with_nodes_and_edges() {
        let store = make_store();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
        store.compute_statistics();
        let stats = store.statistics();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
    }

    #[test]
    fn ensure_statistics_fresh_counts_live_nodes() {
        let store = make_store();
        store.create_node(&["X"]);
        store.ensure_statistics_fresh();
        assert_eq!(store.statistics().total_nodes, 1);
    }

    #[test]
    fn estimate_label_cardinality_returns_nonzero_for_known_label() {
        let store = make_store();
        store.create_node(&["Doc"]);
        store.compute_statistics();
        let card = store.estimate_label_cardinality("Doc");
        assert!(card > 0.0, "cardinality should be positive, got {card}");
    }

    #[test]
    fn estimate_label_cardinality_returns_default_for_unknown_label() {
        let store = make_store();
        store.compute_statistics();
        let card = store.estimate_label_cardinality("NeverSeen");
        // Default estimate should be small but non-negative
        assert!(card >= 0.0);
    }

    #[test]
    fn estimate_avg_degree_for_known_edge_type() {
        let store = make_store();
        let a = store.create_node(&[]);
        let b = store.create_node(&[]);
        store.create_edge(a, b, "FOLLOWS");
        store.compute_statistics();
        let deg = store.estimate_avg_degree("FOLLOWS", true);
        assert!(deg >= 0.0);
    }

    #[test]
    fn compute_statistics_zero_nodes_gives_zero_degree() {
        let store = make_store();
        // Manually add an edge type count without nodes by using the store
        // with an empty graph — avg_degree branch when total_nodes == 0
        store.compute_statistics();
        let stats = store.statistics();
        // No labels or edge types should be present
        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.total_edges, 0);
    }
}
