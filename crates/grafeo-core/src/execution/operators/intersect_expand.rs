//! Two hops that meet: the nodes both adjacency lists hold, not a walk of either.
//!
//! The last two hops of a cycle, `(b)-[e1]->(c)-[e2]->(a)` with `a` and `b` bound,
//! find each `c` as a node both `b`'s and `a`'s lists hold. Listing `b`'s edges and
//! looking each far end up reads every edge of a hub for each row; the intersection
//! gallops from the shorter list, or hashes it on a store that cannot read the lists
//! where they lie. Neither reads fewer entries on every shape.
//!
//! The other way round, `(b)-[e1]->(c)` and `(a)-[e2]->(c)` meet at the far end of
//! both. The rows are those of the two expands: a row per pair of parallel edges,
//! with the columns of the two, `[input..., e1, c, e2, a]` or `[input..., e1, c, e2, c]`.

use std::sync::Arc;

use grafeo_common::types::{ArcStr, LogicalType, NodeId, ReadView};

use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::graph::traits::{CommonEdge, HopFrom};
use crate::graph::{Direction, GraphStore};

/// The most rows a chunk of the operator holds.
const CHUNK_ROWS: usize = 2048;

/// One side of the intersection.
pub struct IntersectSide {
    /// The input column of the bound node the side reads the list of.
    pub column: usize,
    /// The way the list goes from that node.
    pub direction: Direction,
    /// The edge types it holds, as stored (every type when empty).
    pub edge_types: Vec<ArcStr>,
}

/// For each input row, the nodes both sides' lists hold, each with its pairs of edges.
pub struct IntersectExpandOperator {
    store: Arc<dyn GraphStore>,
    input: Box<dyn Operator>,
    near: IntersectSide,
    far: IntersectSide,
    view: ReadView,
    reads: Option<Arc<ReadCounter>>,
    /// Whether the last column is the node they meet at, not the far side's own.
    ends_at_common: bool,
    current_input: Option<DataChunk>,
    current_row: usize,
    common: Vec<CommonEdge>,
    next_common: usize,
}

impl IntersectExpandOperator {
    /// The intersection of the `near` list, whose far end and edge come first in
    /// the output, and the `far` list, whose bound node comes last.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        near: IntersectSide,
        far: IntersectSide,
        view: ReadView,
    ) -> Self {
        Self {
            store,
            input,
            near,
            far,
            view,
            reads: None,
            ends_at_common: false,
            current_input: None,
            current_row: 0,
            common: Vec::new(),
            next_common: 0,
        }
    }

    /// Counts the entries of the lists it reads in `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Gives the node they meet at as the last column, for hops that both end there.
    #[must_use]
    pub fn ending_at_the_common_node(mut self) -> Self {
        self.ends_at_common = true;
        self
    }

    /// The node in `column` of the current row, when it holds one.
    fn node_at(&self, column: usize) -> Option<NodeId> {
        let chunk = self.current_input.as_ref()?;
        let column = chunk.column(column)?;
        if column.is_null(self.current_row) {
            return None;
        }
        column.get_node_id(self.current_row)
    }

    /// The edges in common of the current row.
    fn load_common(&mut self) -> Result<(), OperatorError> {
        self.next_common = 0;
        self.common.clear();
        let (Some(near), Some(far)) = (
            self.node_at(self.near.column),
            self.node_at(self.far.column),
        ) else {
            return Ok(());
        };
        let (common, read) = self.store.common_edges(
            &HopFrom {
                node: near,
                direction: self.near.direction,
                edge_types: &self.near.edge_types,
            },
            &HopFrom {
                node: far,
                direction: self.far.direction,
                edge_types: &self.far.edge_types,
            },
            self.view,
        )?;
        super::read_counter::count(self.reads.as_deref(), read);
        self.common = common;
        Ok(())
    }

    /// Moves on to the next input row, `false` when there is none.
    fn next_row(&mut self) -> Result<bool, OperatorError> {
        loop {
            crate::execution::deadline::check()?;
            if let Some(chunk) = &self.current_input
                && self.current_row + 1 < chunk.row_count()
            {
                self.current_row += 1;
                self.load_common()?;
                return Ok(true);
            }
            match self.input.next()? {
                Some(mut chunk) => {
                    chunk.flatten();
                    if chunk.row_count() == 0 {
                        continue;
                    }
                    self.current_input = Some(chunk);
                    self.current_row = 0;
                    self.load_common()?;
                    return Ok(true);
                }
                None => {
                    self.current_input = None;
                    return Ok(false);
                }
            }
        }
    }
}

impl Operator for IntersectExpandOperator {
    fn next(&mut self) -> OperatorResult {
        let mut output: Option<DataChunk> = None;
        let mut count = 0;
        let mut columns = 0;
        while count < CHUNK_ROWS {
            if self.next_common >= self.common.len() {
                if !self.next_row()? {
                    break;
                }
                continue;
            }
            let Some(input) = self.current_input.as_ref() else {
                break;
            };
            let chunk = output.get_or_insert_with(|| {
                columns = input.column_count();
                let mut schema: Vec<LogicalType> = input
                    .columns()
                    .iter()
                    .map(|column| column.data_type().clone())
                    .collect();
                schema.extend([
                    LogicalType::Edge,
                    LogicalType::Node,
                    LogicalType::Edge,
                    LogicalType::Node,
                ]);
                DataChunk::with_capacity(&schema, CHUNK_ROWS)
            });
            for index in 0..columns {
                if let (Some(from), Some(to)) = (input.column(index), chunk.column_mut(index)) {
                    from.copy_row_to(self.current_row, to);
                }
            }
            let common = self.common[self.next_common];
            let bound = if self.ends_at_common {
                Some(common.node)
            } else {
                input
                    .column(self.far.column)
                    .and_then(|column| column.get_node_id(self.current_row))
            };
            if let Some(column) = chunk.column_mut(columns) {
                column.push_edge_id(common.near_edge);
            }
            if let Some(column) = chunk.column_mut(columns + 1) {
                column.push_node_id(common.node);
            }
            if let Some(column) = chunk.column_mut(columns + 2) {
                column.push_edge_id(common.far_edge);
            }
            if let (Some(column), Some(bound)) = (chunk.column_mut(columns + 3), bound) {
                column.push_node_id(bound);
            }
            self.next_common += 1;
            count += 1;
        }
        Ok(output.map(|mut chunk| {
            chunk.set_count(count);
            chunk
        }))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.current_row = 0;
        self.common.clear();
        self.next_common = 0;
    }

    fn name(&self) -> &'static str {
        "IntersectExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
