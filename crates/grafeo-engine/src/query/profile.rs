//! PROFILE statement: per-operator execution metrics.
//!
//! After a profiled query executes, the results are collected into a
//! [`ProfileNode`] tree that mirrors the physical operator tree, annotated
//! with actual row counts, timing, and call counts.

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::profile::{ProfileStats, SharedProfileStats};
use parking_lot::Mutex;

use super::plan::LogicalOperator;
use crate::database::QueryResult;

/// A node in the profile output tree, corresponding to one physical operator.
#[derive(Debug)]
pub struct ProfileNode {
    /// Operator name (e.g., "NodeScan", "Filter", "Expand").
    pub name: String,
    /// Display label (e.g., "(n:Person)", "(n.age > 25) [label-first]").
    pub label: String,
    /// Shared stats handle, populated during execution.
    pub stats: SharedProfileStats,
    /// Child nodes.
    pub children: Vec<ProfileNode>,
}

/// An entry collected during physical planning, used to build the profile tree.
pub struct ProfileEntry {
    /// Operator name from `Operator::name()`.
    pub name: String,
    /// Human-readable label from the logical operator.
    pub label: String,
    /// Shared stats handle passed to the `ProfiledOperator` wrapper.
    pub stats: SharedProfileStats,
}

impl ProfileEntry {
    /// Creates a new profile entry with fresh (empty) stats.
    pub fn new(name: &str, label: String) -> (Self, SharedProfileStats) {
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let entry = Self {
            name: name.to_string(),
            label,
            stats: Arc::clone(&stats),
        };
        (entry, stats)
    }
}

/// Builds a `ProfileNode` tree from the logical plan and a list of
/// [`ProfileEntry`] items collected during physical planning.
///
/// The entries must be in **post-order** (children before parents),
/// matching the order in which `plan_operator()` processes operators.
///
/// # Panics
///
/// Panics if the iterator yields fewer entries than there are logical operators.
pub fn build_profile_tree(
    logical: &LogicalOperator,
    entries: &mut impl Iterator<Item = ProfileEntry>,
) -> ProfileNode {
    // Recurse into children first (post-order)
    let children: Vec<ProfileNode> = logical
        .children()
        .into_iter()
        .map(|child| build_profile_tree(child, entries))
        .collect();

    // Consume the entry for this node
    let entry = entries.next().unwrap_or_else(|| {
        panic!(
            "profile entry count must match logical operator count — \
             ran out of entries while building tree node for logical \
             operator: {:?} (label='{}')",
            std::mem::discriminant(logical),
            logical.display_label(),
        )
    });

    ProfileNode {
        name: entry.name,
        label: entry.label,
        stats: entry.stats,
        children,
    }
}

/// Formats a `ProfileNode` tree into a human-readable text representation
/// and wraps it in a `QueryResult` with a single "profile" column.
pub fn profile_result(root: &ProfileNode, total_time_ms: f64) -> QueryResult {
    let mut output = String::new();
    format_node(&mut output, root, 0);
    let _ = writeln!(output);
    let _ = write!(output, "Total time: {total_time_ms:.2}ms");

    QueryResult {
        columns: vec!["profile".to_string()],
        column_types: vec![LogicalType::String],
        rows: vec![vec![Value::String(output.into())]],
        execution_time_ms: Some(total_time_ms),
        rows_scanned: None,
        status_message: None,
        gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
    }
}

/// Recursively formats a profile node with indentation.
fn format_node(out: &mut String, node: &ProfileNode, depth: usize) {
    let indent = "  ".repeat(depth);

    // Compute self-time before locking stats (self_time_ns also locks).
    let self_time_ns = self_time_ns(node);
    let self_time_ms = self_time_ns as f64 / 1_000_000.0;

    let rows_out = node.stats.lock().rows_out;

    let _ = writeln!(
        out,
        "{indent}{name} ({label})  rows={rows}  time={time:.2}ms",
        name = node.name,
        label = node.label,
        rows = rows_out,
        time = self_time_ms,
    );

    for child in &node.children {
        format_node(out, child, depth + 1);
    }
}

/// Computes self-time: wall time minus children's wall time.
fn self_time_ns(node: &ProfileNode) -> u64 {
    let own_time = node.stats.lock().time_ns;
    let child_time: u64 = node.children.iter().map(|c| c.stats.lock().time_ns).sum();
    own_time.saturating_sub(child_time)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        FilterOp, LogicalExpression, LogicalOperator, NodeScanOp, ReturnItem, ReturnOp,
    };

    /// Builds a simple `Return(Filter(NodeScan))` tree for profile-tree tests.
    fn three_level_plan() -> LogicalOperator {
        LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                pushdown_hint: None,
            })),
        })
    }

    /// Verifies the builder walks in post-order: children must be consumed
    /// before their parent. Also checks that self-time subtracts child time.
    #[test]
    fn test_profile_tree_post_order() {
        let plan = three_level_plan();

        // Provide three entries, ordered post-order: NodeScan, Filter, Return.
        // Pre-set distinct wall-times on each so we can check self_time subtraction.
        let (scan_entry, scan_stats) = ProfileEntry::new("NodeScan", "n:Person".to_string());
        let (filter_entry, filter_stats) = ProfileEntry::new("Filter", "true".to_string());
        let (return_entry, return_stats) = ProfileEntry::new("Return", "n".to_string());

        scan_stats.lock().time_ns = 100;
        scan_stats.lock().rows_out = 10;
        filter_stats.lock().time_ns = 250; // Includes child (100) + 150 own
        filter_stats.lock().rows_out = 10;
        return_stats.lock().time_ns = 400; // Includes child (250) + 150 own
        return_stats.lock().rows_out = 10;

        let mut iter = vec![scan_entry, filter_entry, return_entry].into_iter();
        let root = build_profile_tree(&plan, &mut iter);

        // Root is Return, its child is Filter, whose child is NodeScan.
        assert_eq!(root.name, "Return");
        assert_eq!(root.children.len(), 1);
        let filter = &root.children[0];
        assert_eq!(filter.name, "Filter");
        assert_eq!(filter.children.len(), 1);
        let scan = &filter.children[0];
        assert_eq!(scan.name, "NodeScan");
        assert!(scan.children.is_empty());

        // Self-time subtracts children: Return = 400 - 250 = 150; Filter = 250 - 100 = 150.
        assert_eq!(self_time_ns(&root), 150);
        assert_eq!(self_time_ns(filter), 150);
        // Leaf has no children, so self = own time.
        assert_eq!(self_time_ns(scan), 100);
    }

    /// Sanity check for the formatted output produced by `profile_result`.
    #[test]
    fn test_profile_result_formatting_and_saturating_self_time() {
        // Build a tree where child time exceeds parent time: saturating_sub
        // should produce 0 rather than underflowing.
        let plan = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            pushdown_hint: None,
        });
        let (scan_entry, scan_stats) = ProfileEntry::new("NodeScan", "n:*".to_string());
        let (filter_entry, filter_stats) = ProfileEntry::new("Filter", "true".to_string());
        // Child dwarfs parent: child=500, parent=100 -> self should saturate at 0.
        scan_stats.lock().time_ns = 500;
        filter_stats.lock().time_ns = 100;
        filter_stats.lock().rows_out = 3;

        let mut iter = vec![scan_entry, filter_entry].into_iter();
        let root = build_profile_tree(&plan, &mut iter);
        assert_eq!(self_time_ns(&root), 0);

        let result = profile_result(&root, 1.23);
        assert_eq!(result.columns, vec!["profile".to_string()]);
        let text = match &result.rows[0][0] {
            grafeo_common::types::Value::String(s) => s.to_string(),
            other => panic!("expected String, got {other:?}"),
        };
        assert!(text.contains("Filter"));
        assert!(text.contains("NodeScan"));
        assert!(text.contains("Total time: 1.23ms"));
    }
}
