//! The count of each row's edges of one pattern, read from the adjacency
//! without listing the edges.
//!
//! `MATCH (a)-[:E]->() RETURN a.id, count(*)` counts the edges of each `a`:
//! the expand would turn every edge into a row only for the aggregate to
//! count them. This operator adds the count to each row instead, and the
//! aggregate sums it (see the planner's degree count).

use std::sync::Arc;

use grafeo_common::types::{ArcStr, LogicalType, ReadView};

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
        }
    }

    /// Counts the edges counted in `reads`, as the expand counts the edges
    /// it reads.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// The count of the edges of the node in `row` of `column`: 0 for a null.
    fn count_of(&self, column: &ValueVector, row: usize) -> Result<usize, OperatorError> {
        if column.is_null(row) {
            return Ok(0);
        }
        let node = column
            .get_node_id(row)
            .ok_or_else(|| OperatorError::Internal("Expected node ID in source column".into()))?;
        Ok(self.store.count_edges(
            node,
            self.direction,
            &self.edge_types,
            &self.end_labels,
            self.view,
        )?)
    }
}

impl Operator for EdgeCountOperator {
    fn next(&mut self) -> OperatorResult {
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
            let mut counts = Vec::with_capacity(rows);
            for row in 0..rows {
                counts.push(self.count_of(source, row)?);
            }
            super::read_counter::count(self.reads.as_deref(), counts.iter().sum());
            let kept = counts.iter().filter(|&&count| count > 0).count();
            if kept == 0 {
                continue;
            }
            let as_integer = |count: usize| i64::try_from(count).unwrap_or(i64::MAX);
            if kept == rows {
                let mut column = ValueVector::with_capacity(LogicalType::Int64, rows);
                for &count in &counts {
                    column.push_int64(as_integer(count));
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
                    to.push_int64(as_integer(count));
                }
            }
            output.set_count(kept);
            return Ok(Some(output));
        }
    }

    fn reset(&mut self) {
        self.input.reset();
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

    /// Alix has two KNOWS edges (one to a City) and a self-loop, Gus one
    /// LIVES_IN edge, Vincent none.
    fn rows(direction: Direction, types: &[&str], end_labels: &[&str]) -> Vec<(u64, i64)> {
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
            "Person",
            ReadView::latest(),
        ));
        let mut counts = EdgeCountOperator::new(
            dyn_store,
            scan,
            0,
            direction,
            types.iter().map(ToString::to_string).collect(),
            end_labels.iter().map(ToString::to_string).collect(),
            ReadView::latest(),
        );
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
}
