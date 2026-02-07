//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{LogicalType, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// A projection expression.
pub enum ProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Property access on a node/edge column.
    PropertyAccess {
        /// The column containing the node or edge ID.
        column: usize,
        /// The property name to access.
        property: String,
    },
    /// Edge type accessor (for type(r) function).
    EdgeType {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Full expression evaluation (for CASE WHEN, etc.).
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
}

/// A project operator that selects and transforms columns.
pub struct ProjectOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<ProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
    /// Optional store for property access.
    store: Option<Arc<LpgStore>>,
}

impl ProjectOperator {
    /// Creates a new project operator.
    pub fn new(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: None,
        }
    }

    /// Creates a new project operator with store access for property lookups.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<LpgStore>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some(store),
        }
    }

    /// Creates a project operator that selects specific columns.
    pub fn select_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        types: Vec<LogicalType>,
    ) -> Self {
        let projections = columns.into_iter().map(ProjectExpr::Column).collect();
        Self::new(child, projections, types)
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult {
        // Get next chunk from child
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Create output chunk
        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        // Evaluate each projection
        for (i, proj) in self.projections.iter().enumerate() {
            match proj {
                ProjectExpr::Column(col_idx) => {
                    // Copy column from input to output
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;

                    let output_col = output.column_mut(i).unwrap();

                    // Copy selected rows
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        }
                    }
                }
                ProjectExpr::Constant(value) => {
                    // Push constant for each row
                    let output_col = output.column_mut(i).unwrap();
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                ProjectExpr::PropertyAccess { column, property } => {
                    // Access property from node/edge in the specified column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output.column_mut(i).unwrap();

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for property access".to_string())
                    })?;

                    // Extract property for each row
                    for row in input.selected_indices() {
                        // Try to get node ID first, then edge ID
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            store
                                .get_node(node_id)
                                .and_then(|node| node.get_property(property).cloned())
                                .unwrap_or(Value::Null)
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .get_edge(edge_id)
                                .and_then(|edge| edge.get_property(property).cloned())
                                .unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeType { column } => {
                    // Get edge type string from an edge column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output.column_mut(i).unwrap();

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge type access".to_string())
                    })?;

                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .edge_type(edge_id)
                                .map(Value::String)
                                .unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let output_col = output.column_mut(i).unwrap();

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution(
                            "Store required for expression evaluation".to_string(),
                        )
                    })?;

                    // Use the ExpressionPredicate for expression evaluation
                    let evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    );

                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row).unwrap_or(Value::Null);
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Project"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use grafeo_common::types::Value;

    struct MockScanOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }
    }

    #[test]
    fn test_project_select_columns() {
        // Create input with 3 columns: [int, string, int]
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::Int64]);

        builder.column_mut(0).unwrap().push_int64(1);
        builder.column_mut(1).unwrap().push_string("hello");
        builder.column_mut(2).unwrap().push_int64(100);
        builder.advance_row();

        builder.column_mut(0).unwrap().push_int64(2);
        builder.column_mut(1).unwrap().push_string("world");
        builder.column_mut(2).unwrap().push_int64(200);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project to select columns 2 and 0 (reordering)
        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![2, 0],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 2);

        // Check values are reordered
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(100));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(1));
    }

    #[test]
    fn test_project_constant() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project with a constant
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::String("constant".into())),
            ],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column(1).unwrap().get_string(0), Some("constant"));
        assert_eq!(result.column(1).unwrap().get_string(1), Some("constant"));
    }
}
