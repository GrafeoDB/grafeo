//! The count of each row's edges of one pattern, read from the adjacency
//! without listing the edges.
//!
//! `MATCH (a)-[:E]->() RETURN a.id, count(*)` counts the edges of each `a`:
//! the expand would turn every edge into a row only for the aggregate to
//! count them. This operator adds the count to each row instead, and the
//! aggregate sums it (see the planner's degree count). Without groups
//! (`MATCH (a)-[:E]->() RETURN count(*)`) it totals the counts itself, in
//! one row (see [`EdgeCountOperator::totalled`]).

use std::sync::Arc;

use grafeo_common::types::{ArcStr, LogicalType, ReadView};

use super::chain_counter::{ChainCounter, CountedHop, MiddleSum};
use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::execution::vector::ValueVector;
use crate::graph::{Direction, GraphStore};

/// Each input row with one more column: how many edges of the node in
/// `source_column` in `direction`, of one of `edge_types` (any type when
/// empty), end at a node with every label of `end_labels`, as `view` sees
/// them (see [`GraphStore::count_edges`]). A row whose count is 0 (a node
/// without such an edge, or a null) is dropped, as the expand it stands in
/// for gives that row no match.
pub struct EdgeCountOperator {
    store: Arc<dyn GraphStore>,
    input: Box<dyn Operator>,
    source_column: usize,
    direction: Direction,
    edge_types: Vec<ArcStr>,
    end_labels: Vec<ArcStr>,
    view: ReadView,
    reads: Option<Arc<ReadCounter>>,
    /// The number of columns of the one row of totals, for an operator that
    /// totals the counts (see [`totalled`](Self::totalled)).
    total_columns: Option<usize>,
    /// Whether the row of totals is out.
    totalled: bool,
    /// The counter of a chain of hops, when the count goes on past this one.
    chain: Option<ChainCounter>,
    /// The sum of two hops, when each input row is their middle node.
    middle_sum: Option<MiddleSum>,
}

impl EdgeCountOperator {
    /// The count of the edges of the node in `source_column` of each row of
    /// `input`, in `direction`, of one of `edge_types` (the stored names),
    /// to a node with every label of `end_labels`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
        end_labels: Vec<String>,
        view: ReadView,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types: edge_types.into_iter().map(ArcStr::from).collect(),
            end_labels: end_labels.into_iter().map(ArcStr::from).collect(),
            view,
            reads: None,
            total_columns: None,
            totalled: false,
            chain: None,
            middle_sum: None,
        }
    }

    /// Counts the paths through each input row's node of `sum`'s two hops, in place
    /// of this operator's own hop. Set after the read counter.
    #[must_use]
    pub fn with_middle_sum(mut self, sum: MiddleSum) -> Self {
        self.middle_sum = Some(sum);
        self
    }

    /// Whether what it counts is not the edges it reads, which it counts itself.
    fn counts_its_own_reads(&self) -> bool {
        self.chain.is_some() || self.middle_sum.is_some()
    }

    /// Counts the paths of this hop and then `rest`, where the edges of each pair
    /// of `different` hops (by position, this hop first) differ, without listing
    /// them: each node's lists are read once. Set after the read counter.
    #[must_use]
    pub fn with_chain(mut self, rest: Vec<CountedHop>, different: &[(usize, usize)]) -> Self {
        let mut hops = vec![CountedHop {
            direction: self.direction,
            edge_types: self.edge_types.clone(),
            end_labels: self.end_labels.clone(),
        }];
        hops.extend(rest);
        self.chain = Some(ChainCounter::new(
            Arc::clone(&self.store),
            hops,
            different,
            self.view,
            self.reads.clone(),
        ));
        self
    }

    /// Counts the edges counted in `reads`, as the expand counts the edges
    /// it reads.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// The operator that gives one row of `columns` integer columns, each
    /// the total of the counts of every input row, in place of the rows:
    /// the counts of `count(*)` and `count(x)` without groups over the
    /// expand, which are 0 over no rows.
    #[must_use]
    pub fn totalled(mut self, columns: usize) -> Self {
        self.total_columns = Some(columns);
        self
    }

    /// The row of totals of an operator that totals its counts.
    fn next_total(&mut self, columns: usize) -> OperatorResult {
        if self.totalled {
            return Ok(None);
        }
        let mut total: usize = 0;
        while let Some(chunk) = self.input.next()? {
            crate::execution::deadline::check()?;
            let source = chunk.column(self.source_column).ok_or_else(|| {
                OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
            })?;
            let mut counted: usize = 0;
            for row in chunk.selected_indices() {
                counted = counted
                    .checked_add(self.count_of(source, row)?)
                    .ok_or_else(total_overflow)?;
            }
            // A chain counts the lists it reads, not the paths it counts.
            if !self.counts_its_own_reads() {
                super::read_counter::count(self.reads.as_deref(), counted);
            }
            total = total.checked_add(counted).ok_or_else(total_overflow)?;
        }
        self.totalled = true;
        let total = i64::try_from(total).map_err(|_| total_overflow())?;
        let columns = (0..columns)
            .map(|_| {
                let mut column = ValueVector::with_capacity(LogicalType::Int64, 1);
                column.push_int64(total);
                column
            })
            .collect();
        Ok(Some(DataChunk::new(columns)))
    }

    /// The count of the edges of the node in `row` of `column`: 0 for a null.
    fn count_of(&mut self, column: &ValueVector, row: usize) -> Result<usize, OperatorError> {
        if column.is_null(row) {
            return Ok(0);
        }
        let node = column
            .get_node_id(row)
            .ok_or_else(|| OperatorError::Internal("Expected node ID in source column".into()))?;
        if let Some(chain) = &mut self.chain {
            return chain.count(node);
        }
        if let Some(sum) = &self.middle_sum {
            return sum.count(self.store.as_ref(), node, self.view, self.reads.as_deref());
        }
        Ok(self.store.count_edges(
            node,
            self.direction,
            &self.edge_types,
            &self.end_labels,
            self.view,
        )?)
    }
}

/// The error of a total no 64-bit integer holds (22003).
fn total_overflow() -> OperatorError {
    OperatorError::numeric_out_of_range("a count of paths past 64 bits")
}

impl Operator for EdgeCountOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(columns) = self.total_columns {
            return self.next_total(columns);
        }
        loop {
            crate::execution::deadline::check()?;
            let Some(mut chunk) = self.input.next()? else {
                return Ok(None);
            };
            chunk.flatten();
            let rows = chunk.row_count();
            let source = chunk.column(self.source_column).ok_or_else(|| {
                OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
            })?;
            let mut counts: Vec<i64> = Vec::with_capacity(rows);
            for row in 0..rows {
                let count = self.count_of(source, row)?;
                counts.push(i64::try_from(count).map_err(|_| total_overflow())?);
            }
            if !self.counts_its_own_reads() {
                let read: usize = counts
                    .iter()
                    .map(|count| usize::try_from(*count).unwrap_or(0))
                    .sum();
                super::read_counter::count(self.reads.as_deref(), read);
            }
            let kept = counts.iter().filter(|&&count| count > 0).count();
            if kept == 0 {
                continue;
            }
            if kept == rows {
                let mut column = ValueVector::with_capacity(LogicalType::Int64, rows);
                for &count in &counts {
                    column.push_int64(count);
                }
                chunk.push_column(column);
                return Ok(Some(chunk));
            }
            // Some rows have no edge: copy the others.
            let mut schema: Vec<LogicalType> = chunk
                .columns()
                .iter()
                .map(|c| c.data_type().clone())
                .collect();
            schema.push(LogicalType::Int64);
            let mut output = DataChunk::with_capacity(&schema, kept);
            let width = chunk.column_count();
            for (row, &count) in counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                for index in 0..width {
                    if let (Some(from), Some(to)) = (chunk.column(index), output.column_mut(index))
                    {
                        from.copy_row_to(row, to);
                    }
                }
                if let Some(to) = output.column_mut(width) {
                    to.push_int64(count);
                }
            }
            output.set_count(kept);
            return Ok(Some(output));
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.totalled = false;
    }

    fn name(&self) -> &'static str {
        "EdgeCount"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::rowgroup::RowGroupStore;

    /// The counts of the edges of each node with `label`: Alix has two
    /// KNOWS edges (one to a City) and a self-loop, Gus one LIVES_IN edge,
    /// Vincent none.
    fn counting(
        label: &str,
        direction: Direction,
        types: &[&str],
        end_labels: &[&str],
    ) -> EdgeCountOperator {
        let store = Arc::new(RowGroupStore::new());
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let _vincent = store.create_node(&["Person"]);
        let amsterdam = store.create_node(&["City"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, amsterdam, "KNOWS");
        store.create_edge(alix, alix, "KNOWS");
        store.create_edge(gus, amsterdam, "LIVES_IN");
        let dyn_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            label,
            ReadView::latest(),
        ));
        EdgeCountOperator::new(
            dyn_store,
            scan,
            0,
            direction,
            types.iter().map(ToString::to_string).collect(),
            end_labels.iter().map(ToString::to_string).collect(),
            ReadView::latest(),
        )
    }

    /// Each person with an edge of the pattern, and its count.
    fn rows(direction: Direction, types: &[&str], end_labels: &[&str]) -> Vec<(u64, i64)> {
        let mut counts = counting("Person", direction, types, end_labels);
        let mut found = Vec::new();
        while let Some(chunk) = counts.next().unwrap() {
            for row in 0..chunk.row_count() {
                let node = chunk.column(0).unwrap().get_node_id(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                found.push((node.as_u64(), count));
            }
        }
        found
    }

    #[test]
    fn counts_the_edges_of_each_row_and_drops_rows_without_one() {
        assert_eq!(rows(Direction::Outgoing, &[], &[]), [(0, 3), (1, 1)]);
        assert_eq!(rows(Direction::Outgoing, &["KNOWS"], &[]), [(0, 3)]);
        assert_eq!(rows(Direction::Outgoing, &["NONE"], &[]), []);
        assert_eq!(rows(Direction::Outgoing, &["KNOWS"], &["Person"]), [(0, 2)]);
        assert_eq!(rows(Direction::Outgoing, &[], &["City"]), [(0, 1), (1, 1)]);
        assert_eq!(rows(Direction::Outgoing, &[], &["Nobody"]), []);
    }

    #[test]
    fn an_undirected_count_lists_a_self_loop_once() {
        // Alix: Gus, Amsterdam and the loop; Gus: Alix and Amsterdam.
        assert_eq!(rows(Direction::Both, &[], &[]), [(0, 3), (1, 2)]);
        assert_eq!(rows(Direction::Incoming, &[], &[]), [(0, 1), (1, 1)]);
    }

    /// The chunks of the totals of the counts of each node with `label`,
    /// in two columns.
    fn totals(label: &str, direction: Direction, end_labels: &[&str]) -> Vec<Vec<i64>> {
        let mut counts = counting(label, direction, &[], end_labels).totalled(2);
        let mut found = Vec::new();
        while let Some(chunk) = counts.next().unwrap() {
            assert_eq!((chunk.row_count(), chunk.column_count()), (1, 2));
            found.push(
                (0..2)
                    .map(|column| chunk.column(column).unwrap().get_int64(0).unwrap())
                    .collect(),
            );
        }
        found
    }

    #[test]
    fn a_totalled_count_is_one_row_also_without_edges_or_rows() {
        assert_eq!(totals("Person", Direction::Outgoing, &[]), [[4, 4]]);
        assert_eq!(totals("Person", Direction::Both, &[]), [[5, 5]]);
        assert_eq!(totals("Person", Direction::Outgoing, &["City"]), [[2, 2]]);
        // No edge to such a node, and no node to start from.
        assert_eq!(totals("Person", Direction::Outgoing, &["Nobody"]), [[0, 0]]);
        assert_eq!(totals("Nobody", Direction::Outgoing, &[]), [[0, 0]]);
        // A reset gives the row again.
        let mut counts = counting("Person", Direction::Outgoing, &[], &[]).totalled(1);
        assert_eq!(counts.next().unwrap().unwrap().row_count(), 1);
        assert!(counts.next().unwrap().is_none());
        counts.reset();
        let again = counts.next().unwrap().unwrap();
        assert_eq!(again.column(0).unwrap().get_int64(0), Some(4));
    }
}
