//! Apply operator (lateral join / correlated subquery).
//!
//! For each row from the outer input, the inner subplan is reset and executed.
//! Results are the concatenation of outer columns with inner columns.
//!
//! This operator is the backend for:
//! - Cypher: `CALL { subquery }`
//! - GQL: `VALUE { subquery }`
//! - Pattern comprehensions (with a Collect aggregate wrapper)
//! - Cypher: `FOREACH`, and a `CALL { subquery }` without a final `RETURN`,
//!   in unit mode

use std::collections::VecDeque;
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
/// In unit mode ([`with_unit`](Self::with_unit)) the inner plan runs only for
/// its writes and each outer row comes out once, unchanged.
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
    /// What an outer row becomes, given the inner plan's rows for it.
    mode: RowMode,
    /// Whether the whole outer input is read before the inner plan runs.
    outer_first: bool,
    /// The outer input's chunks, when it is read first.
    outer_chunks: Option<VecDeque<DataChunk>>,
    /// Buffered outer rows waiting to be combined with inner results.
    state: ApplyState,
}

/// What the Apply makes of an outer row, given the rows the inner plan
/// produces for it. The builder that sets a mode last decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowMode {
    /// One row per inner row: the outer columns, then the inner ones. An
    /// outer row without inner rows is dropped (inner join semantics).
    Join,
    /// As `Join`, and an outer row without inner rows comes out once with
    /// `inner_column_count` nulls for the inner columns (left join).
    Optional {
        /// Number of columns the inner plan produces.
        inner_column_count: usize,
    },
    /// EXISTS: the outer row alone, when the inner plan has a row
    /// (`keep_matches`, a semi-join) or when it has none (an anti-join).
    Exists {
        /// Whether a row with inner rows is kept (or one without).
        keep_matches: bool,
    },
    /// EXISTS flag: every outer row, with a boolean column that says whether
    /// the inner plan has a row. Used for EXISTS inside OR predicates.
    ExistsFlag,
    /// Unit: the outer row once, as it came in, after the inner plan ran to
    /// its end for what it writes (`FOREACH`).
    Unit,
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
            mode: RowMode::Join,
            outer_first: false,
            outer_chunks: None,
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
            mode: RowMode::Join,
            outer_first: false,
            outer_chunks: None,
            state: ApplyState::Init,
        }
    }

    /// Enables optional (left-join) semantics with the given inner column count.
    ///
    /// When enabled, outer rows that produce no inner results will be emitted
    /// with NULL values for the inner columns instead of being dropped.
    pub fn with_optional(mut self, inner_column_count: usize) -> Self {
        self.mode = RowMode::Optional { inner_column_count };
        self
    }

    /// Enables EXISTS mode: semi-join (`keep_matches=true`) or anti-join
    /// (`keep_matches=false`). Inner columns are NOT appended to the output.
    pub fn with_exists_mode(mut self, keep_matches: bool) -> Self {
        self.mode = RowMode::Exists { keep_matches };
        self
    }

    /// Enables EXISTS flag mode: instead of filtering, appends a boolean
    /// column indicating whether the inner plan produced results. All outer
    /// rows are preserved. Used for EXISTS inside OR predicates where
    /// semi-join filtering would be incorrect.
    pub fn with_exists_flag(mut self) -> Self {
        self.mode = RowMode::ExistsFlag;
        self
    }

    /// Enables unit mode, for an inner plan that runs only for its writes
    /// (Cypher `FOREACH`, a unit `CALL` subquery): for each outer row the
    /// inner plan runs to its end, whatever rows it produces are dropped, and
    /// the outer row comes out once, as it came in. No inner columns are
    /// appended, so a row is neither repeated for an inner plan of several
    /// rows nor dropped for one of none.
    #[must_use]
    pub fn with_unit(mut self) -> Self {
        self.mode = RowMode::Unit;
        self
    }

    /// Reads the whole outer input before the inner plan runs, so that the
    /// inner plan sees what the outer input writes: in `UNWIND ... CREATE
    /// (:N {k: i}) WITH i CALL { WITH i MATCH (t:N {k: 3001 - i}) ... }` the
    /// first rows look for nodes that later rows create.
    #[must_use]
    pub fn with_outer_first(mut self) -> Self {
        self.outer_first = true;
        self
    }

    /// The next outer chunk, from the buffer when the outer input was read
    /// first.
    fn next_outer(&mut self) -> OperatorResult {
        match &mut self.outer_chunks {
            Some(chunks) => Ok(chunks.pop_front()),
            None => self.outer.next(),
        }
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
        mode: RowMode,
        outer_chunk: &DataChunk,
        inner_types: &ColumnTypes,
    ) -> Vec<LogicalType> {
        let mut types = outer_chunk.column_types();
        match mode {
            RowMode::ExistsFlag => types.push(LogicalType::Bool),
            RowMode::Join | RowMode::Optional { .. } => {
                types.extend_from_slice(inner_types.types());
            }
            RowMode::Exists { .. } | RowMode::Unit => {}
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
        if self.outer_first && self.outer_chunks.is_none() {
            let mut chunks = VecDeque::new();
            while let Some(chunk) = self.outer.next()? {
                chunks.push_back(chunk);
            }
            self.outer_chunks = Some(chunks);
        }
        loop {
            match &mut self.state {
                ApplyState::Init => match self.next_outer()? {
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

                        match self.mode {
                            // EXISTS flag mode: append boolean column, keep all rows
                            RowMode::ExistsFlag => {
                                let has_results = self.inner.next()?.is_some();
                                let mut combined = outer_values;
                                combined.push(Value::Bool(has_results));
                                output.push(combined);
                            }
                            // EXISTS mode: check for row existence without
                            // appending inner columns
                            RowMode::Exists { keep_matches } => {
                                let has_results = self.inner.next()?.is_some();
                                if has_results == keep_matches {
                                    output.push(outer_values);
                                }
                            }
                            // Unit mode: run the inner plan to its end for its
                            // writes, keep the outer row once
                            RowMode::Unit => {
                                while self.inner.next()?.is_some() {}
                                output.push(outer_values);
                            }
                            RowMode::Join | RowMode::Optional { .. } => {
                                let pre_len = output.len();
                                while let Some(inner_chunk) = self.inner.next()? {
                                    inner_types.add(&inner_chunk);
                                    for inner_row in inner_chunk.selected_indices() {
                                        let inner_values =
                                            Self::extract_row(&inner_chunk, inner_row);
                                        let mut combined = outer_values.clone();
                                        combined.extend(inner_values);
                                        output.push(combined);
                                    }
                                }

                                // OPTIONAL: emit outer row with NULLs when inner
                                // produced nothing
                                if let RowMode::Optional { inner_column_count } = self.mode
                                    && output.len() == pre_len
                                {
                                    let mut combined = outer_values;
                                    combined.extend(std::iter::repeat_n(
                                        Value::Null,
                                        inner_column_count,
                                    ));
                                    output.push(combined);
                                }
                            }
                        }

                        *outer_row += 1;

                        // Flush when we have enough rows
                        if output.len() >= 1024 {
                            let types = Self::output_types(self.mode, outer_chunk, inner_types);
                            let chunk = Self::build_chunk(output, &types);
                            output.clear();
                            *inner_types = ColumnTypes::default();
                            return Ok(Some(chunk));
                        }
                    }

                    // Finished this outer chunk; flush any remaining output
                    if !output.is_empty() {
                        let types = Self::output_types(self.mode, outer_chunk, inner_types);
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
        self.outer_chunks = None;
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

    /// An outer input that "writes" one unit per chunk it returns: three
    /// chunks of one row each.
    struct WritingOperator {
        written: Arc<std::sync::atomic::AtomicI64>,
        position: i64,
    }

    impl Operator for WritingOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position == 3 {
                return Ok(None);
            }
            self.position += 1;
            self.written
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
            builder
                .column_mut(0)
                .unwrap()
                .push_value(Value::Int64(self.position));
            builder.advance_row();
            Ok(Some(builder.finish()))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Writing"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// An inner plan that reads what the outer input wrote so far: one row.
    struct ReadingOperator {
        written: Arc<std::sync::atomic::AtomicI64>,
        done: bool,
    }

    impl Operator for ReadingOperator {
        fn next(&mut self) -> OperatorResult {
            if self.done {
                return Ok(None);
            }
            self.done = true;
            let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
            builder.column_mut(0).unwrap().push_value(Value::Int64(
                self.written.load(std::sync::atomic::Ordering::SeqCst),
            ));
            builder.advance_row();
            Ok(Some(builder.finish()))
        }

        fn reset(&mut self) {
            self.done = false;
        }

        fn name(&self) -> &'static str {
            "Reading"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// The (outer, seen) pairs `apply` returns: each outer row with what the
    /// inner plan saw written when it ran for that row.
    fn seen(apply: &mut ApplyOperator) -> Vec<(i64, i64)> {
        let mut rows = Vec::new();
        while let Some(chunk) = apply.next().unwrap() {
            for row in chunk.selected_indices() {
                let value = |column: usize| match chunk.column(column).unwrap().get_value(row) {
                    Some(Value::Int64(value)) => value,
                    other => panic!("expected an integer, got {other:?}"),
                };
                rows.push((value(0), value(1)));
            }
        }
        rows
    }

    /// With the outer input read first, the inner plan of every row sees all
    /// of what the outer input wrote; without, a row sees only what the rows
    /// before it wrote. After a reset, the outer input is read again.
    #[test]
    fn reading_the_outer_input_first_shows_the_inner_plan_all_its_writes() {
        let apply = |outer_first: bool| {
            let written = Arc::new(std::sync::atomic::AtomicI64::new(0));
            let outer = Box::new(WritingOperator {
                written: Arc::clone(&written),
                position: 0,
            });
            let inner = Box::new(ReadingOperator {
                written,
                done: false,
            });
            let apply = ApplyOperator::new(outer, inner);
            if outer_first {
                apply.with_outer_first()
            } else {
                apply
            }
        };

        assert_eq!(seen(&mut apply(false)), [(1, 1), (2, 2), (3, 3)]);

        let mut first = apply(true);
        assert_eq!(seen(&mut first), [(1, 3), (2, 3), (3, 3)]);
        first.reset();
        assert_eq!(
            seen(&mut first),
            [(1, 6), (2, 6), (3, 6)],
            "a reset reads the outer input again, before the inner plan"
        );
    }

    /// An inner plan that "writes" one unit per row of its input it passes on.
    struct CountingOperator {
        input: Box<dyn Operator>,
        written: Arc<std::sync::atomic::AtomicI64>,
    }

    impl Operator for CountingOperator {
        fn next(&mut self) -> OperatorResult {
            let chunk = self.input.next()?;
            if let Some(chunk) = &chunk {
                let rows = i64::try_from(chunk.row_count()).unwrap();
                self.written
                    .fetch_add(rows, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(chunk)
        }

        fn reset(&mut self) {
            self.input.reset();
        }

        fn name(&self) -> &'static str {
            "Counting"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// In unit mode (FOREACH), the inner plan unwinds the list of each outer
    /// row and "writes" once per item: every outer row comes out once, as it
    /// came in and without inner columns, whether its list has two items, none
    /// or is null. Without it, the row of two items comes out twice and the
    /// others not at all.
    #[test]
    fn unit_mode_keeps_each_outer_row_once() {
        let apply = |unit: bool| {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Any, LogicalType::Int64]);
            for (list, tag) in [
                (Value::from(vec![Value::Int64(1), Value::Int64(2)]), 3),
                (Value::from(Vec::<Value>::new()), 19),
                (Value::Null, 88),
            ] {
                builder.column_mut(0).unwrap().push_value(list);
                builder.column_mut(1).unwrap().push_value(Value::Int64(tag));
                builder.advance_row();
            }
            let outer = Box::new(MockOperator {
                chunks: vec![builder.finish()],
                position: 0,
            });
            let state = Arc::new(ParameterState::new(vec!["list".to_string()]));
            let unwind = super::super::UnwindOperator::new(
                Box::new(ParameterScanOperator::new(Arc::clone(&state))),
                0,
                "item".to_string(),
                vec![LogicalType::Any, LogicalType::Any],
                false,
                false,
            );
            let written = Arc::new(std::sync::atomic::AtomicI64::new(0));
            let inner = Box::new(CountingOperator {
                input: Box::new(unwind),
                written: Arc::clone(&written),
            });
            let apply = ApplyOperator::new_correlated(outer, inner, state, vec![0]);
            (if unit { apply.with_unit() } else { apply }, written)
        };
        let tags = |apply: &mut ApplyOperator| {
            let mut rows = Vec::new();
            while let Some(chunk) = apply.next().unwrap() {
                for row in chunk.selected_indices() {
                    rows.push((
                        chunk.column_count(),
                        chunk.column(1).unwrap().get_value(row),
                    ));
                }
            }
            rows
        };

        let (mut unit, written) = apply(true);
        assert_eq!(
            tags(&mut unit),
            [3, 19, 88].map(|tag| (2, Some(Value::Int64(tag)))),
            "each outer row once, with its own two columns"
        );
        assert_eq!(written.load(std::sync::atomic::Ordering::SeqCst), 2);
        unit.reset();
        assert_eq!(tags(&mut unit).len(), 3, "a reset runs every row again");
        assert_eq!(written.load(std::sync::atomic::Ordering::SeqCst), 4);

        let (mut joined, written) = apply(false);
        assert_eq!(
            tags(&mut joined),
            [(4, Some(Value::Int64(3))), (4, Some(Value::Int64(3)))],
            "without unit mode, a row per inner row"
        );
        assert_eq!(written.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
