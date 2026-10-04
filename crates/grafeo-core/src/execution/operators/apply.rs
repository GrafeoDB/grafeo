//! Apply operator (lateral join / correlated subquery).
//!
//! For each row from the outer input, the inner subplan is reset and executed.
//! Results are the concatenation of outer columns with inner columns.
//!
//! This operator is the backend for:
//! - Cypher: `CALL { subquery }`
//! - GQL: `VALUE { subquery }`
//! - Pattern comprehensions (with a Collect aggregate wrapper)

use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};

use super::parameter_scan::ParameterState;
use super::{DataChunk, Operator, OperatorResult};
use crate::execution::chunk::ColumnTypes;
use crate::execution::vector::ValueVector;

/// Apply (lateral join) operator.
///
/// Evaluates `inner` once for each row of `outer`. The result schema is
/// `outer_columns ++ inner_columns`. If the inner plan produces zero rows
/// for a given outer row, that outer row is omitted (inner join semantics).
///
/// When `param_state` is set, outer row values for the specified column indices
/// are injected into the shared [`ParameterState`] before each inner execution,
/// allowing the inner plan's [`ParameterScanOperator`](super::ParameterScanOperator) to read them.
///
/// A row keeps the column types of the rows it combines (and the injected
/// values those of their outer columns): a node or edge stays one.
pub struct ApplyOperator {
    outer: Box<dyn Operator>,
    inner: Box<dyn Operator>,
    /// Shared parameter state for correlated subqueries.
    param_state: Option<Arc<ParameterState>>,
    /// Indices of outer columns to inject into the inner plan.
    param_col_indices: Vec<usize>,
    /// When true, outer rows with no inner results emit NULLs (left-join).
    optional: bool,
    /// Number of columns the inner plan produces (needed for NULL-padding).
    inner_column_count: usize,
    /// EXISTS mode: Some(true) = semi-join (keep if inner has rows),
    /// Some(false) = anti-join (keep if inner has NO rows).
    /// Inner columns are NOT appended in EXISTS mode.
    exists_mode: Option<bool>,
    /// EXISTS flag mode: instead of filtering rows, append a boolean column
    /// indicating whether the inner plan produced results. All outer rows
    /// are preserved. Used for EXISTS inside OR predicates.
    exists_flag: bool,
    /// Buffered outer rows waiting to be combined with inner results.
    state: ApplyState,
}

enum ApplyState {
    /// Pull next outer chunk, process row-by-row.
    Init,
    /// Processing a chunk of outer rows. `outer_chunk` is the current batch,
    /// `outer_row` is the next row index to process.
    Processing {
        outer_chunk: DataChunk,
        outer_row: usize,
        /// Accumulated output rows (combined outer + inner).
        output: Vec<Vec<Value>>,
        /// The column types of the inner chunks of these rows.
        inner_types: ColumnTypes,
    },
    /// All outer input exhausted.
    Done,
}

impl ApplyOperator {
    /// Creates a new Apply operator (uncorrelated: no parameter injection).
    pub fn new(outer: Box<dyn Operator>, inner: Box<dyn Operator>) -> Self {
        Self {
            outer,
            inner,
            param_state: None,
            param_col_indices: Vec::new(),
            optional: false,
            inner_column_count: 0,
            exists_mode: None,
            exists_flag: false,
            state: ApplyState::Init,
        }
    }

    /// Creates a correlated Apply operator that injects outer row values.
    ///
    /// `param_state` is shared with a [`ParameterScanOperator`](super::ParameterScanOperator) in the inner plan.
    /// `param_col_indices` specifies which outer columns to inject (by index).
    pub fn new_correlated(
        outer: Box<dyn Operator>,
        inner: Box<dyn Operator>,
        param_state: Arc<ParameterState>,
        param_col_indices: Vec<usize>,
    ) -> Self {
        Self {
            outer,
            inner,
            param_state: Some(param_state),
            param_col_indices,
            optional: false,
            inner_column_count: 0,
            exists_mode: None,
            exists_flag: false,
            state: ApplyState::Init,
        }
    }

    /// Enables optional (left-join) semantics with the given inner column count.
    ///
    /// When enabled, outer rows that produce no inner results will be emitted
    /// with NULL values for the inner columns instead of being dropped.
    pub fn with_optional(mut self, inner_column_count: usize) -> Self {
        self.optional = true;
        self.inner_column_count = inner_column_count;
        self
    }

    /// Enables EXISTS mode: semi-join (`keep_matches=true`) or anti-join
    /// (`keep_matches=false`). Inner columns are NOT appended to the output.
    pub fn with_exists_mode(mut self, keep_matches: bool) -> Self {
        self.exists_mode = Some(keep_matches);
        self
    }

    /// Enables EXISTS flag mode: instead of filtering, appends a boolean
    /// column indicating whether the inner plan produced results. All outer
    /// rows are preserved. Used for EXISTS inside OR predicates where
    /// semi-join filtering would be incorrect.
    pub fn with_exists_flag(mut self) -> Self {
        self.exists_flag = true;
        self
    }

    /// Extracts all values from a single row of a DataChunk.
    fn extract_row(chunk: &DataChunk, row: usize) -> Vec<Value> {
        let mut values = Vec::with_capacity(chunk.num_columns());
        for col_idx in 0..chunk.num_columns() {
            let val = chunk
                .column(col_idx)
                .and_then(|col| col.get_value(row))
                .unwrap_or(Value::Null);
            values.push(val);
        }
        values
    }

    /// The column types of output rows from `outer_chunk`: its own, then the
    /// EXISTS flag's or the inner chunks' (see [`ColumnTypes`]).
    fn output_types(
        exists_flag: bool,
        exists_mode: Option<bool>,
        outer_chunk: &DataChunk,
        inner_types: &ColumnTypes,
    ) -> Vec<LogicalType> {
        let mut types = outer_chunk.column_types();
        if exists_flag {
            types.push(LogicalType::Bool);
        } else if exists_mode.is_none() {
            types.extend_from_slice(inner_types.types());
        }
        types
    }

    /// Builds a DataChunk from accumulated rows, in `types` (a column past
    /// them, from inner chunks never seen, is of any type).
    fn build_chunk(rows: &[Vec<Value>], types: &[LogicalType]) -> DataChunk {
        if rows.is_empty() {
            return DataChunk::empty();
        }
        let num_cols = rows[0].len();
        let mut columns: Vec<ValueVector> = (0..num_cols)
            .map(|i| {
                let column_type = types.get(i).cloned().unwrap_or(LogicalType::Any);
                ValueVector::with_capacity(column_type, rows.len())
            })
            .collect();

        for row in rows {
            for (col_idx, val) in row.iter().enumerate() {
                if col_idx < columns.len() {
                    columns[col_idx].push_value(val.clone());
                }
            }
        }
        DataChunk::new(columns)
    }
}

impl Operator for ApplyOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            match &mut self.state {
                ApplyState::Init => match self.outer.next()? {
                    Some(chunk) => {
                        self.state = ApplyState::Processing {
                            outer_chunk: chunk,
                            outer_row: 0,
                            output: Vec::new(),
                            inner_types: ColumnTypes::default(),
                        };
                    }
                    None => {
                        self.state = ApplyState::Done;
                        return Ok(None);
                    }
                },
                ApplyState::Processing {
                    outer_chunk,
                    outer_row,
                    output,
                    inner_types,
                } => {
                    let selected: Vec<usize> = outer_chunk.selected_indices().collect();
                    while *outer_row < selected.len() {
                        let row = selected[*outer_row];
                        let outer_values = Self::extract_row(outer_chunk, row);

                        // Inject outer values into the inner plan's parameter state,
                        // with the types of the columns they come from
                        if let Some(ref param_state) = self.param_state {
                            let injected: Vec<Value> = self
                                .param_col_indices
                                .iter()
                                .map(|&idx| outer_values.get(idx).cloned().unwrap_or(Value::Null))
                                .collect();
                            let types: Vec<LogicalType> = self
                                .param_col_indices
                                .iter()
                                .map(|&idx| {
                                    outer_chunk
                                        .column(idx)
                                        .map_or(LogicalType::Any, |col| col.data_type().clone())
                                })
                                .collect();
                            param_state.set_typed_values(injected, types);
                        }

                        // Reset and run inner plan for this outer row
                        self.inner.reset();

                        // EXISTS flag mode: append boolean column, keep all rows
                        if self.exists_flag {
                            let has_results = self.inner.next()?.is_some();
                            let mut combined = outer_values;
                            combined.push(Value::Bool(has_results));
                            output.push(combined);
                        }
                        // EXISTS mode: check for row existence without appending inner columns
                        else if let Some(keep_matches) = self.exists_mode {
                            let has_results = self.inner.next()?.is_some();
                            if has_results == keep_matches {
                                output.push(outer_values);
                            }
                        } else {
                            let pre_len = output.len();
                            while let Some(inner_chunk) = self.inner.next()? {
                                inner_types.add(&inner_chunk);
                                for inner_row in inner_chunk.selected_indices() {
                                    let inner_values = Self::extract_row(&inner_chunk, inner_row);
                                    let mut combined = outer_values.clone();
                                    combined.extend(inner_values);
                                    output.push(combined);
                                }
                            }

                            // OPTIONAL: emit outer row with NULLs when inner produced nothing
                            if self.optional && output.len() == pre_len {
                                let mut combined = outer_values;
                                combined.extend(std::iter::repeat_n(
                                    Value::Null,
                                    self.inner_column_count,
                                ));
                                output.push(combined);
                            }
                        }

                        *outer_row += 1;

                        // Flush when we have enough rows
                        if output.len() >= 1024 {
                            let types = Self::output_types(
                                self.exists_flag,
                                self.exists_mode,
                                outer_chunk,
                                inner_types,
                            );
                            let chunk = Self::build_chunk(output, &types);
                            output.clear();
                            *inner_types = ColumnTypes::default();
                            return Ok(Some(chunk));
                        }
                    }

                    // Finished this outer chunk; flush any remaining output
                    if !output.is_empty() {
                        let types = Self::output_types(
                            self.exists_flag,
                            self.exists_mode,
                            outer_chunk,
                            inner_types,
                        );
                        let chunk = Self::build_chunk(output, &types);
                        output.clear();
                        self.state = ApplyState::Init;
                        return Ok(Some(chunk));
                    }

                    // Move to next outer chunk
                    self.state = ApplyState::Init;
                }
                ApplyState::Done => return Ok(None),
            }
        }
    }

    fn reset(&mut self) {
        self.outer.reset();
        self.inner.reset();
        self.state = ApplyState::Init;
    }

    fn name(&self) -> &'static str {
        "Apply"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::NodeId;

    use super::super::ParameterScanOperator;
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            let chunk = self.chunks.get(self.position).cloned();
            self.position += 1;
            Ok(chunk)
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Outer rows with the nodes 101 and 102.
    fn outer_nodes() -> Box<dyn Operator> {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        for id in [101, 102] {
            builder.column_mut(0).unwrap().push_node_id(NodeId::new(id));
            builder.advance_row();
        }
        Box::new(MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        })
    }

    /// The inner plan reads the injected outer node as a node, and the joined
    /// rows keep the column types of both sides.
    #[test]
    fn apply_keeps_the_column_types_of_outer_and_inner_rows() {
        let state = Arc::new(ParameterState::new(vec!["a".to_string()]));
        let inner = Box::new(ParameterScanOperator::new(Arc::clone(&state)));
        let mut apply = ApplyOperator::new_correlated(outer_nodes(), inner, state, vec![0]);

        let chunk = apply.next().unwrap().unwrap();
        assert_eq!(chunk.column_types(), [LogicalType::Node, LogicalType::Node]);
        let rows: Vec<(u64, u64)> = chunk
            .selected_indices()
            .map(|row| {
                (
                    chunk.column(0).unwrap().get_node_id(row).unwrap().as_u64(),
                    chunk.column(1).unwrap().get_node_id(row).unwrap().as_u64(),
                )
            })
            .collect();
        assert_eq!(rows, [(101, 101), (102, 102)]);
        assert!(chunk.column(1).unwrap().get_edge_id(0).is_none());
        assert!(apply.next().unwrap().is_none());
    }

    /// The EXISTS flag is a boolean column after the outer row's own types.
    #[test]
    fn an_exists_flag_follows_the_outer_types() {
        let state = Arc::new(ParameterState::new(vec!["a".to_string()]));
        let inner = Box::new(ParameterScanOperator::new(Arc::clone(&state)));
        let mut apply =
            ApplyOperator::new_correlated(outer_nodes(), inner, state, vec![0]).with_exists_flag();

        let chunk = apply.next().unwrap().unwrap();
        assert_eq!(chunk.column_types(), [LogicalType::Node, LogicalType::Bool]);
        assert_eq!(
            chunk.column(1).unwrap().get_value(0),
            Some(Value::Bool(true))
        );
    }
}
