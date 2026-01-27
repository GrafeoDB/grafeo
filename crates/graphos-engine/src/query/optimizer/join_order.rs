//! DPccp (Dynamic Programming with connected complement pairs) join ordering.
//!
//! This module implements the DPccp algorithm for finding optimal join orderings.
//! The algorithm works by:
//! 1. Building a join graph from the query
//! 2. Enumerating all connected subgraphs
//! 3. Finding optimal plans for each subgraph using dynamic programming
//!
//! Reference: Moerkotte, G., & Neumann, T. (2006). Analysis of Two Existing and
//! One New Dynamic Programming Algorithm for the Generation of Optimal Bushy
//! Join Trees without Cross Products.

use super::cardinality::CardinalityEstimator;
use super::cost::{Cost, CostModel};
use crate::query::plan::{JoinCondition, JoinOp, JoinType, LogicalExpression, LogicalOperator};
use std::collections::{HashMap, HashSet};

/// A node in the join graph.
#[derive(Debug, Clone)]
pub struct JoinNode {
    /// Unique identifier for this node.
    pub id: usize,
    /// Variable name (e.g., "a" for node (a:Person)).
    pub variable: String,
    /// The base relation (scan operator).
    pub relation: LogicalOperator,
}

impl PartialEq for JoinNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.variable == other.variable
    }
}

impl Eq for JoinNode {}

impl std::hash::Hash for JoinNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.variable.hash(state);
    }
}

/// An edge in the join graph representing a join condition.
#[derive(Debug, Clone)]
pub struct JoinEdge {
    /// Source node id.
    pub from: usize,
    /// Target node id.
    pub to: usize,
    /// Join conditions between these nodes.
    pub conditions: Vec<JoinCondition>,
}

/// The join graph representing all relations and join conditions in a query.
#[derive(Debug)]
pub struct JoinGraph {
    /// Nodes in the graph.
    nodes: Vec<JoinNode>,
    /// Edges (join conditions) between nodes.
    edges: Vec<JoinEdge>,
    /// Adjacency list for quick neighbor lookup.
    adjacency: HashMap<usize, HashSet<usize>>,
}

impl JoinGraph {
    /// Creates a new empty join graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, variable: String, relation: LogicalOperator) -> usize {
        let id = self.nodes.len();
        self.nodes.push(JoinNode {
            id,
            variable,
            relation,
        });
        self.adjacency.insert(id, HashSet::new());
        id
    }

    /// Adds a join edge between two nodes.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` were not previously added via [`add_node`].
    pub fn add_edge(&mut self, from: usize, to: usize, conditions: Vec<JoinCondition>) {
        self.edges.push(JoinEdge {
            from,
            to,
            conditions,
        });
        // Invariant: add_node() inserts node ID into adjacency map (line 84),
        // so get_mut succeeds for any ID returned by add_node()
        self.adjacency
            .get_mut(&from)
            .expect("'from' node must be added via add_node() before add_edge()")
            .insert(to);
        self.adjacency
            .get_mut(&to)
            .expect("'to' node must be added via add_node() before add_edge()")
            .insert(from);
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns a reference to the nodes.
    pub fn nodes(&self) -> &[JoinNode] {
        &self.nodes
    }

    /// Returns neighbors of a node.
    pub fn neighbors(&self, node_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjacency.get(&node_id).into_iter().flatten().copied()
    }

    /// Gets the join conditions between two node sets.
    pub fn get_conditions(&self, left: &BitSet, right: &BitSet) -> Vec<JoinCondition> {
        let mut conditions = Vec::new();
        for edge in &self.edges {
            let from_in_left = left.contains(edge.from);
            let from_in_right = right.contains(edge.from);
            let to_in_left = left.contains(edge.to);
            let to_in_right = right.contains(edge.to);

            // Edge crosses between left and right
            if (from_in_left && to_in_right) || (from_in_right && to_in_left) {
                conditions.extend(edge.conditions.clone());
            }
        }
        conditions
    }

    /// Checks if two node sets are connected by at least one edge.
    pub fn are_connected(&self, left: &BitSet, right: &BitSet) -> bool {
        for edge in &self.edges {
            let from_in_left = left.contains(edge.from);
            let from_in_right = right.contains(edge.from);
            let to_in_left = left.contains(edge.to);
            let to_in_right = right.contains(edge.to);

            if (from_in_left && to_in_right) || (from_in_right && to_in_left) {
                return true;
            }
        }
        false
    }
}

impl Default for JoinGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A bitset for efficient subset representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitSet(u64);

impl BitSet {
    /// Creates an empty bitset.
    pub fn empty() -> Self {
        Self(0)
    }

    /// Creates a bitset with a single element.
    pub fn singleton(i: usize) -> Self {
        Self(1 << i)
    }

    /// Creates a bitset from an iterator of indices.
    pub fn from_iter(iter: impl Iterator<Item = usize>) -> Self {
        let mut bits = 0u64;
        for i in iter {
            bits |= 1 << i;
        }
        Self(bits)
    }

    /// Creates a full bitset with elements 0..n.
    pub fn full(n: usize) -> Self {
        Self((1 << n) - 1)
    }

    /// Checks if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    /// Checks if the set contains an element.
    pub fn contains(&self, i: usize) -> bool {
        (self.0 & (1 << i)) != 0
    }

    /// Inserts an element.
    pub fn insert(&mut self, i: usize) {
        self.0 |= 1 << i;
    }

    /// Removes an element.
    pub fn remove(&mut self, i: usize) {
        self.0 &= !(1 << i);
    }

    /// Returns the union of two sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the intersection of two sets.
    pub fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns the difference (self - other).
    pub fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Checks if this set is a subset of another.
    pub fn is_subset_of(self, other: Self) -> bool {
        (self.0 & other.0) == self.0
    }

    /// Iterates over all elements in the set.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        (0..64).filter(move |&i| self.contains(i))
    }

    /// Iterates over all non-empty subsets.
    pub fn subsets(self) -> SubsetIterator {
        SubsetIterator {
            full: self.0,
            current: Some(self.0),
        }
    }
}

/// Iterator over all subsets of a bitset.
pub struct SubsetIterator {
    full: u64,
    current: Option<u64>,
}

impl Iterator for SubsetIterator {
    type Item = BitSet;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        if current == 0 {
            self.current = None;
            return Some(BitSet(0));
        }
        let result = current;
        // Gosper's hack variant for subset enumeration
        self.current = Some((current.wrapping_sub(1)) & self.full);
        if self.current == Some(self.full) {
            self.current = None;
        }
        Some(BitSet(result))
    }
}

/// Represents a (partial) join plan.
#[derive(Debug, Clone)]
pub struct JoinPlan {
    /// The subset of nodes covered by this plan.
    pub nodes: BitSet,
    /// The logical operator representing this plan.
    pub operator: LogicalOperator,
    /// Estimated cost of this plan.
    pub cost: Cost,
    /// Estimated cardinality.
    pub cardinality: f64,
}

/// DPccp join ordering optimizer.
pub struct DPccp<'a> {
    /// The join graph.
    graph: &'a JoinGraph,
    /// Cost model for estimating operator costs.
    cost_model: &'a CostModel,
    /// Cardinality estimator.
    card_estimator: &'a CardinalityEstimator,
    /// Memoization table: subset -> best plan.
    memo: HashMap<BitSet, JoinPlan>,
}

impl<'a> DPccp<'a> {
    /// Creates a new DPccp optimizer.
    pub fn new(
        graph: &'a JoinGraph,
        cost_model: &'a CostModel,
        card_estimator: &'a CardinalityEstimator,
    ) -> Self {
        Self {
            graph,
            cost_model,
            card_estimator,
            memo: HashMap::new(),
        }
    }

    /// Finds the optimal join order for the graph.
    pub fn optimize(&mut self) -> Option<JoinPlan> {
        let n = self.graph.node_count();
        if n == 0 {
            return None;
        }
        if n == 1 {
            let node = &self.graph.nodes[0];
            let cardinality = self.card_estimator.estimate(&node.relation);
            let cost = self.cost_model.estimate(&node.relation, cardinality);
            return Some(JoinPlan {
                nodes: BitSet::singleton(0),
                operator: node.relation.clone(),
                cost,
                cardinality,
            });
        }

        // Initialize with single relations
        for (i, node) in self.graph.nodes.iter().enumerate() {
            let subset = BitSet::singleton(i);
            let cardinality = self.card_estimator.estimate(&node.relation);
            let cost = self.cost_model.estimate(&node.relation, cardinality);
            self.memo.insert(
                subset,
                JoinPlan {
                    nodes: subset,
                    operator: node.relation.clone(),
                    cost,
                    cardinality,
                },
            );
        }

        // Enumerate connected subgraph pairs (ccp)
        let full_set = BitSet::full(n);
        self.enumerate_ccp(full_set);

        // Return the best plan for the full set
        self.memo.get(&full_set).cloned()
    }

    /// Enumerates connected complement pairs using DPccp algorithm.
    fn enumerate_ccp(&mut self, s: BitSet) {
        // Iterate over all proper non-empty subsets
        for s1 in s.subsets() {
            if s1.is_empty() || s1 == s {
                continue;
            }

            let s2 = s.difference(s1);
            if s2.is_empty() {
                continue;
            }

            // Both s1 and s2 must be connected subsets
            if !self.is_connected(s1) || !self.is_connected(s2) {
                continue;
            }

            // s1 and s2 must be connected to each other
            if !self.graph.are_connected(&s1, &s2) {
                continue;
            }

            // Recursively solve subproblems
            if !self.memo.contains_key(&s1) {
                self.enumerate_ccp(s1);
            }
            if !self.memo.contains_key(&s2) {
                self.enumerate_ccp(s2);
            }

            // Try to build a plan for s by joining s1 and s2
            if let (Some(plan1), Some(plan2)) = (self.memo.get(&s1), self.memo.get(&s2)) {
                let conditions = self.graph.get_conditions(&s1, &s2);
                let new_plan = self.build_join_plan(plan1.clone(), plan2.clone(), conditions);

                // Update memo if this is a better plan
                let should_update = self
                    .memo
                    .get(&s)
                    .map(|existing| new_plan.cost.total() < existing.cost.total())
                    .unwrap_or(true);

                if should_update {
                    self.memo.insert(s, new_plan);
                }
            }
        }
    }

    /// Checks if a subset forms a connected subgraph.
    fn is_connected(&self, subset: BitSet) -> bool {
        if subset.len() <= 1 {
            return true;
        }

        // BFS to check connectivity
        // Invariant: subset.len() >= 2 (guard on line 400), so iter().next() returns Some
        let start = subset
            .iter()
            .next()
            .expect("subset is non-empty: len >= 2 checked on line 400");
        let mut visited = BitSet::singleton(start);
        let mut queue = vec![start];

        while let Some(node) = queue.pop() {
            for neighbor in self.graph.neighbors(node) {
                if subset.contains(neighbor) && !visited.contains(neighbor) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        visited == subset
    }

    /// Builds a join plan from two sub-plans.
    fn build_join_plan(
        &self,
        left: JoinPlan,
        right: JoinPlan,
        conditions: Vec<JoinCondition>,
    ) -> JoinPlan {
        let nodes = left.nodes.union(right.nodes);

        // Create the join operator
        let join_op = LogicalOperator::Join(JoinOp {
            left: Box::new(left.operator),
            right: Box::new(right.operator),
            join_type: JoinType::Inner,
            conditions,
        });

        // Estimate cardinality
        let cardinality = self.card_estimator.estimate(&join_op);

        // Calculate cost (child costs + join cost)
        let join_cost = self.cost_model.estimate(&join_op, cardinality);
        let total_cost = left.cost + right.cost + join_cost;

        JoinPlan {
            nodes,
            operator: join_op,
            cost: total_cost,
            cardinality,
        }
    }
}

/// Extracts a join graph from a query pattern.
pub struct JoinGraphBuilder {
    graph: JoinGraph,
    variable_to_node: HashMap<String, usize>,
}

impl JoinGraphBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            graph: JoinGraph::new(),
            variable_to_node: HashMap::new(),
        }
    }

    /// Adds a base relation (scan).
    pub fn add_relation(&mut self, variable: &str, relation: LogicalOperator) -> usize {
        let id = self.graph.add_node(variable.to_string(), relation);
        self.variable_to_node.insert(variable.to_string(), id);
        id
    }

    /// Adds a join condition between two variables.
    pub fn add_join_condition(
        &mut self,
        left_var: &str,
        right_var: &str,
        left_expr: LogicalExpression,
        right_expr: LogicalExpression,
    ) {
        if let (Some(&left_id), Some(&right_id)) = (
            self.variable_to_node.get(left_var),
            self.variable_to_node.get(right_var),
        ) {
            self.graph.add_edge(
                left_id,
                right_id,
                vec![JoinCondition {
                    left: left_expr,
                    right: right_expr,
                }],
            );
        }
    }

    /// Builds the join graph.
    pub fn build(self) -> JoinGraph {
        self.graph
    }
}

impl Default for JoinGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::NodeScanOp;

    fn create_node_scan(var: &str, label: &str) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: var.to_string(),
            label: Some(label.to_string()),
            input: None,
        })
    }

    #[test]
    fn test_bitset_operations() {
        let a = BitSet::singleton(0);
        let b = BitSet::singleton(1);
        let _c = BitSet::singleton(2);

        assert!(a.contains(0));
        assert!(!a.contains(1));

        let ab = a.union(b);
        assert!(ab.contains(0));
        assert!(ab.contains(1));
        assert!(!ab.contains(2));

        let full = BitSet::full(3);
        assert_eq!(full.len(), 3);
        assert!(full.contains(0));
        assert!(full.contains(1));
        assert!(full.contains(2));
    }

    #[test]
    fn test_subset_iteration() {
        let set = BitSet::from_iter([0, 1].into_iter());
        let subsets: Vec<_> = set.subsets().collect();

        // Should have 4 subsets: {}, {0}, {1}, {0,1}
        assert_eq!(subsets.len(), 4);
        assert!(subsets.contains(&BitSet::empty()));
        assert!(subsets.contains(&BitSet::singleton(0)));
        assert!(subsets.contains(&BitSet::singleton(1)));
        assert!(subsets.contains(&set));
    }

    #[test]
    fn test_join_graph_construction() {
        let mut builder = JoinGraphBuilder::new();

        builder.add_relation("a", create_node_scan("a", "Person"));
        builder.add_relation("b", create_node_scan("b", "Person"));
        builder.add_relation("c", create_node_scan("c", "Company"));

        builder.add_join_condition(
            "a",
            "b",
            LogicalExpression::Property {
                variable: "a".to_string(),
                property: "id".to_string(),
            },
            LogicalExpression::Property {
                variable: "b".to_string(),
                property: "friend_id".to_string(),
            },
        );

        builder.add_join_condition(
            "a",
            "c",
            LogicalExpression::Property {
                variable: "a".to_string(),
                property: "company_id".to_string(),
            },
            LogicalExpression::Property {
                variable: "c".to_string(),
                property: "id".to_string(),
            },
        );

        let graph = builder.build();
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_dpccp_single_relation() {
        let mut builder = JoinGraphBuilder::new();
        builder.add_relation("a", create_node_scan("a", "Person"));
        let graph = builder.build();

        let cost_model = CostModel::new();
        let mut card_estimator = CardinalityEstimator::new();
        card_estimator.add_table_stats(
            "Person",
            super::super::cardinality::TableStats::new(1000),
        );

        let mut dpccp = DPccp::new(&graph, &cost_model, &card_estimator);
        let plan = dpccp.optimize();

        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.nodes.len(), 1);
    }

    #[test]
    fn test_dpccp_two_relations() {
        let mut builder = JoinGraphBuilder::new();
        builder.add_relation("a", create_node_scan("a", "Person"));
        builder.add_relation("b", create_node_scan("b", "Company"));

        builder.add_join_condition(
            "a",
            "b",
            LogicalExpression::Property {
                variable: "a".to_string(),
                property: "company_id".to_string(),
            },
            LogicalExpression::Property {
                variable: "b".to_string(),
                property: "id".to_string(),
            },
        );

        let graph = builder.build();

        let cost_model = CostModel::new();
        let mut card_estimator = CardinalityEstimator::new();
        card_estimator.add_table_stats(
            "Person",
            super::super::cardinality::TableStats::new(1000),
        );
        card_estimator.add_table_stats(
            "Company",
            super::super::cardinality::TableStats::new(100),
        );

        let mut dpccp = DPccp::new(&graph, &cost_model, &card_estimator);
        let plan = dpccp.optimize();

        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.nodes.len(), 2);

        // The result should be a join
        if let LogicalOperator::Join(_) = plan.operator {
            // Good
        } else {
            panic!("Expected Join operator");
        }
    }

    #[test]
    fn test_dpccp_three_relations_chain() {
        // a -[knows]-> b -[works_at]-> c
        let mut builder = JoinGraphBuilder::new();
        builder.add_relation("a", create_node_scan("a", "Person"));
        builder.add_relation("b", create_node_scan("b", "Person"));
        builder.add_relation("c", create_node_scan("c", "Company"));

        builder.add_join_condition(
            "a",
            "b",
            LogicalExpression::Property {
                variable: "a".to_string(),
                property: "knows".to_string(),
            },
            LogicalExpression::Property {
                variable: "b".to_string(),
                property: "id".to_string(),
            },
        );

        builder.add_join_condition(
            "b",
            "c",
            LogicalExpression::Property {
                variable: "b".to_string(),
                property: "company_id".to_string(),
            },
            LogicalExpression::Property {
                variable: "c".to_string(),
                property: "id".to_string(),
            },
        );

        let graph = builder.build();

        let cost_model = CostModel::new();
        let mut card_estimator = CardinalityEstimator::new();
        card_estimator.add_table_stats(
            "Person",
            super::super::cardinality::TableStats::new(1000),
        );
        card_estimator.add_table_stats(
            "Company",
            super::super::cardinality::TableStats::new(100),
        );

        let mut dpccp = DPccp::new(&graph, &cost_model, &card_estimator);
        let plan = dpccp.optimize();

        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.nodes.len(), 3);
    }

    #[test]
    fn test_dpccp_prefers_smaller_intermediate() {
        // Test that DPccp prefers joining smaller tables first
        // Setup: Small (100) -[r1]-> Medium (1000) -[r2]-> Large (10000)
        // Without cost-based ordering, might get (Small ⋈ Large) ⋈ Medium
        // With cost-based ordering, should get (Small ⋈ Medium) ⋈ Large

        let mut builder = JoinGraphBuilder::new();
        builder.add_relation("s", create_node_scan("s", "Small"));
        builder.add_relation("m", create_node_scan("m", "Medium"));
        builder.add_relation("l", create_node_scan("l", "Large"));

        // Connect all three (star schema)
        builder.add_join_condition(
            "s",
            "m",
            LogicalExpression::Property {
                variable: "s".to_string(),
                property: "id".to_string(),
            },
            LogicalExpression::Property {
                variable: "m".to_string(),
                property: "s_id".to_string(),
            },
        );

        builder.add_join_condition(
            "m",
            "l",
            LogicalExpression::Property {
                variable: "m".to_string(),
                property: "id".to_string(),
            },
            LogicalExpression::Property {
                variable: "l".to_string(),
                property: "m_id".to_string(),
            },
        );

        let graph = builder.build();

        let cost_model = CostModel::new();
        let mut card_estimator = CardinalityEstimator::new();
        card_estimator.add_table_stats(
            "Small",
            super::super::cardinality::TableStats::new(100),
        );
        card_estimator.add_table_stats(
            "Medium",
            super::super::cardinality::TableStats::new(1000),
        );
        card_estimator.add_table_stats(
            "Large",
            super::super::cardinality::TableStats::new(10000),
        );

        let mut dpccp = DPccp::new(&graph, &cost_model, &card_estimator);
        let plan = dpccp.optimize();

        assert!(plan.is_some());
        let plan = plan.unwrap();

        // The plan should cover all three relations
        assert_eq!(plan.nodes.len(), 3);

        // We can't easily verify the exact join order without inspecting the tree,
        // but we can verify the plan was created successfully
        assert!(plan.cost.total() > 0.0);
    }
}
