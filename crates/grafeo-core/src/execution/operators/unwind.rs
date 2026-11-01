//! Unwind operator for expanding lists into individual rows.

use super::{Operator, OperatorResult};
use crate::execution::chunk::{DataChunk, DataChunkBuilder, copied_column_types};
use grafeo_common::types::{LogicalType, PropertyKey, Value};

/// Unwind operator that expands a list column into individual rows.
///
/// For each input row, if the list column contains N elements, this operator
/// produces N output rows, each with one element from the list.
pub struct UnwindOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Index of the column containing the list to unwind.
    list_col_idx: usize,
    /// Name of the new variable for the unwound elements.
    variable_name: String,
    /// Schema of output columns (inherited from input plus the new column).
    output_schema: Vec<LogicalType>,
    /// Whether to emit a 1-based ORDINALITY column.
    emit_ordinality: bool,
    /// Whether to emit a 0-based OFFSET column.
    emit_offset: bool,
    /// Whether the rows pass on the list column: a variable that holds the
    /// list stays in scope after the UNWIND (see
    /// [`without_the_list`](Self::without_the_list)).
    passes_on_the_list: bool,
    /// How many rows a chunk takes (see
    /// [`row_at_a_time`](Self::row_at_a_time)).
    rows_per_chunk: RowsPerChunk,
    /// What the list column holds.
    list_column: ListColumn,
    /// The input chunk being unwound.
    current: Option<InputChunk>,
    /// Position in `current`'s rows.
    current_row: usize,
    /// Current index within the list being unwound.
    current_list_idx: u64,
    /// Current list being unwound.
    current_list: Option<Items>,
}

/// What an UNWIND's list column holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListColumn {
    /// The list.
    List,
    /// The bounds of a `range` call (see
    /// [`range_bounds`](UnwindOperator::range_bounds)).
    RangeBounds,
}

/// The items of the list being unwound.
enum Items {
    /// The items of a list.
    List(Vec<Value>),
    /// The `len` integers `start`, `start + step`, ... of a `range` call,
    /// made as they are unwound (see [`UnwindOperator::range_bounds`]).
    Range { start: i64, step: i64, len: u64 },
}

impl Items {
    /// The integers of `range(start, stop, step)`: from `start` towards
    /// `stop` in steps of `step` (not 0), with `stop` when a step lands on
    /// it; none when `stop` lies the other way.
    fn range(start: i64, stop: i64, step: i64) -> Self {
        let span = i128::from(stop) - i128::from(start);
        let len = if span != 0 && (span < 0) != (step < 0) {
            0
        } else {
            // The span and the step have the same sign, so the quotient
            // rounds down and every item lies between the bounds.
            u64::try_from(span / i128::from(step) + 1).unwrap_or(u64::MAX)
        };
        Self::Range { start, step, len }
    }

    /// The items of the range whose bounds `value` holds, `[start, stop]` or
    /// `[start, stop, step]`; none when a bound is no integer or the step is
    /// 0, for which `range` gives null.
    fn of_range_bounds(value: Option<Value>) -> Option<Self> {
        let Some(Value::List(bounds)) = value else {
            return None;
        };
        let bound = |index: usize| match bounds.get(index) {
            Some(Value::Int64(bound)) => Some(*bound),
            _ => None,
        };
        let step = match bounds.len() {
            2 => 1,
            3 => bound(2)?,
            _ => return None,
        };
        if step == 0 {
            return None;
        }
        Some(Self::range(bound(0)?, bound(1)?, step))
    }

    /// How many items there are.
    fn len(&self) -> u64 {
        match self {
            Self::List(items) => u64::try_from(items.len()).unwrap_or(u64::MAX),
            Self::Range { len, .. } => *len,
        }
    }

    /// The item at `index`, which is below [`len`](Self::len).
    fn item(&self, index: u64) -> Value {
        match self {
            Self::List(items) => {
                items[usize::try_from(index).expect("below the length of a list")].clone()
            }
            Self::Range { start, step, .. } => {
                let item = i128::from(*start) + i128::from(index) * i128::from(*step);
                Value::Int64(i64::try_from(item).expect("an item lies between the bounds"))
            }
        }
    }
}

/// How many rows an UNWIND's chunk takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowsPerChunk {
    /// Up to the chunk's capacity.
    Full,
    /// One: each row goes out alone.
    One,
}

/// An input chunk being unwound: its rows (the selected ones, by physical
/// index) and the types of the rows it gives, computed once for the chunk.
struct InputChunk {
    chunk: DataChunk,
    rows: Vec<usize>,
    /// The output types: the input's copied columns keep their own types (a
    /// node stays one, see `ColumnTypes`), the new ones have the declared
    /// types.
    types: Vec<LogicalType>,
    /// How many input columns are copied.
    copied: usize,
}

impl UnwindOperator {
    /// Creates a new unwind operator.
    ///
    /// # Arguments
    /// * `child` - The input operator
    /// * `list_col_idx` - The column index containing the list to unwind
    /// * `variable_name` - The name of the new variable
    /// * `output_schema` - The schema for output (should include the new column type)
    /// * `emit_ordinality` - Whether to emit a 1-based index column
    /// * `emit_offset` - Whether to emit a 0-based index column
    pub fn new(
        child: Box<dyn Operator>,
        list_col_idx: usize,
        variable_name: String,
        output_schema: Vec<LogicalType>,
        emit_ordinality: bool,
        emit_offset: bool,
    ) -> Self {
        Self {
            child,
            list_col_idx,
            variable_name,
            output_schema,
            emit_ordinality,
            emit_offset,
            passes_on_the_list: true,
            rows_per_chunk: RowsPerChunk::Full,
            list_column: ListColumn::List,
            current: None,
            current_row: 0,
            current_list_idx: 0,
            current_list: None,
        }
    }

    /// Leaves the list column null in the rows: for a list the planner
    /// computed for the UNWIND alone (a literal, a parameter, a property),
    /// which no variable holds, so the rows do not carry the whole list along
    /// with each of its items. A variable that holds the list stays in scope
    /// (`WITH [1, 2] AS l UNWIND l AS x RETURN l, x`), so by default the rows
    /// pass the list on.
    #[must_use]
    pub fn without_the_list(mut self) -> Self {
        self.passes_on_the_list = false;
        self
    }

    /// Gives each row in a chunk of its own, for a statement that writes:
    /// the rows of one chunk are computed together, before any of them
    /// writes, while a row must see what the rows before it wrote
    /// (`UNWIND [3, 19] AS x SET p.n = coalesce(p.n, 0) + x`, and `FOREACH`,
    /// whose items see the updates before them). Otherwise the rows go out
    /// in full chunks.
    #[must_use]
    pub fn row_at_a_time(mut self) -> Self {
        self.rows_per_chunk = RowsPerChunk::One;
        self
    }

    /// Reads the list column as the bounds of a `range` call, `[start, stop]`
    /// or `[start, stop, step]`, and makes the range's integers as they are
    /// unwound, never holding them as a list: `UNWIND range(1, 3000000000)
    /// AS x` would first build a list of three billion values, and a second
    /// `UNWIND range(1, n)` one list of `n` values per row of the first. A
    /// bound that is no integer, or a step of 0, gives no rows, as the null
    /// `range` gives for them does.
    #[must_use]
    pub fn range_bounds(mut self) -> Self {
        self.list_column = ListColumn::RangeBounds;
        self
    }

    /// Returns the variable name for the unwound elements.
    #[must_use]
    pub fn variable_name(&self) -> &str {
        &self.variable_name
    }

    /// The index of the unwound element's column: after the declared input
    /// columns, before any ordinality and offset columns.
    fn element_col_idx(&self) -> usize {
        let extra_cols = usize::from(self.emit_ordinality) + usize::from(self.emit_offset);
        self.output_schema.len() - 1 - extra_cols
    }

    /// The input chunk `chunk` with its rows and output types.
    fn input_chunk(&self, chunk: DataChunk) -> InputChunk {
        let element_col_idx = self.element_col_idx();
        let mut types = chunk.column_types();
        types.truncate(element_col_idx);
        let copied = types.len();
        let mut types = copied_column_types(&types, &self.output_schema);
        types.extend(self.output_schema.iter().skip(copied).cloned());
        let rows = chunk.selected_indices().collect();
        InputChunk {
            chunk,
            rows,
            types,
            copied,
        }
    }

    /// The list of the current row, when it holds one with elements: a
    /// `Value::List` or a `Value::Vector`, or the bounds of a range (see
    /// [`range_bounds`](Self::range_bounds)). Moves past rows without one.
    fn next_list(&mut self) -> Option<Items> {
        let input = self.current.as_ref()?;
        while let Some(&row) = input.rows.get(self.current_row) {
            let value = input.chunk.column(self.list_col_idx)?.get_value(row);
            let items = if self.list_column == ListColumn::RangeBounds {
                Items::of_range_bounds(value)
            } else {
                match value {
                    Some(Value::List(list)) => Some(Items::List(list.iter().cloned().collect())),
                    Some(Value::Vector(vector)) => Some(Items::List(
                        vector
                            .iter()
                            .map(|f| Value::Float64(f64::from(*f)))
                            .collect(),
                    )),
                    _ => None,
                }
            };
            if let Some(items) = items.filter(|items| items.len() > 0) {
                return Some(items);
            }
            self.current_row += 1;
        }
        None
    }

    /// Appends the row of the current element to `builder`, and moves to
    /// the next element.
    fn push_row(&mut self, builder: &mut DataChunkBuilder, element: Value) {
        let element_col_idx = self.element_col_idx();
        let input = self
            .current
            .as_ref()
            .expect("an input chunk while its rows are unwound");
        let row = input.rows[self.current_row];
        // Copy the input columns, the list too unless it is left out (a
        // variable that holds the list stays in scope; it read null).
        for col_idx in 0..input.copied {
            let value = if col_idx == self.list_col_idx && !self.passes_on_the_list {
                Value::Null
            } else {
                input
                    .chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .unwrap_or(Value::Null)
            };
            if let Some(out_col) = builder.column_mut(col_idx) {
                out_col.push_value(value);
            }
        }
        if let Some(out_col) = builder.column_mut(element_col_idx) {
            out_col.push_value(element);
        }
        // A list holds fewer than i64::MAX items, so its index fits; a range
        // stops long before its index could reach it.
        let index = i64::try_from(self.current_list_idx).unwrap_or(i64::MAX);
        let mut next_col = element_col_idx + 1;
        if self.emit_ordinality {
            if let Some(out_col) = builder.column_mut(next_col) {
                out_col.push_value(Value::Int64(index + 1));
            }
            next_col += 1;
        }
        if self.emit_offset
            && let Some(out_col) = builder.column_mut(next_col)
        {
            out_col.push_value(Value::Int64(index));
        }
        builder.advance_row();

        self.current_list_idx += 1;
        if self
            .current_list
            .as_ref()
            .is_some_and(|items| self.current_list_idx >= items.len())
        {
            self.current_list = None;
            self.current_list_idx = 0;
            self.current_row += 1;
        }
    }
}

impl Operator for UnwindOperator {
    /// Up to a chunk of rows from one input chunk (one row with
    /// [`row_at_a_time`](UnwindOperator::row_at_a_time)). A node or edge item that
    /// is no ID goes into a column of any value; a chunk holds one type per
    /// column, so such an item starts a new chunk, and so does the item
    /// after it.
    fn next(&mut self) -> OperatorResult {
        let element_col_idx = self.element_col_idx();
        let entity_items = matches!(
            self.output_schema[element_col_idx],
            LogicalType::Node | LogicalType::Edge
        );
        // The rows so far, and whether their element column is of any value.
        let mut out: Option<(DataChunkBuilder, bool)> = None;
        loop {
            if self.current_list.is_none() {
                if let Some(list) = self.next_list() {
                    self.current_list = Some(list);
                    self.current_list_idx = 0;
                } else {
                    // The input chunk is done: its rows go out before the next
                    // one, whose column types may differ.
                    self.current = None;
                    if let Some((builder, _)) = out.take() {
                        return Ok(Some(builder.finish()));
                    }
                    // Rows without items read chunk after chunk here.
                    crate::execution::deadline::check()?;
                    match self.child.next()? {
                        Some(chunk) => {
                            self.current = Some(self.input_chunk(chunk));
                            self.current_row = 0;
                            continue;
                        }
                        None => return Ok(None),
                    }
                }
            }
            let items = self.current_list.as_ref().expect("a list: set above");
            // An item of a node or edge list is an ID (a returned node or
            // edge, a map with `_id`, stands for its ID) or null; any other
            // item goes into a column of any value, so it is never read as a
            // node or an edge.
            let element = entity_item(
                items.item(self.current_list_idx),
                &self.output_schema[element_col_idx],
            );
            let any = entity_items && !matches!(element, Value::Int64(_) | Value::Null);
            if out.as_ref().is_some_and(|(_, out_any)| *out_any != any) {
                let (builder, _) = out.take().expect("checked above");
                return Ok(Some(builder.finish()));
            }
            let (builder, _) = out.get_or_insert_with(|| {
                let input = self
                    .current
                    .as_ref()
                    .expect("an input chunk while its rows are unwound");
                let mut types = input.types.clone();
                if any {
                    types[element_col_idx] = LogicalType::Any;
                }
                (DataChunkBuilder::new(&types), any)
            });
            self.push_row(builder, element);
            if self.rows_per_chunk == RowsPerChunk::One || builder.is_full() {
                let (builder, _) = out.take().expect("a builder: just filled");
                return Ok(Some(builder.finish()));
            }
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.current = None;
        self.current_row = 0;
        self.current_list_idx = 0;
        self.current_list = None;
    }

    fn name(&self) -> &'static str {
        "Unwind"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// The item of a node or edge list as the entity's ID: a map with `_id` (a
/// node or edge a subquery returned) stands for that ID. Other items, and the
/// items of other lists, stay as they are.
fn entity_item(item: Value, item_type: &LogicalType) -> Value {
    match (item_type, &item) {
        (LogicalType::Node | LogicalType::Edge, Value::Map(map)) => {
            match map.get(&PropertyKey::new("_id")) {
                Some(id @ Value::Int64(_)) => id.clone(),
                _ => item,
            }
        }
        _ => item,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use std::sync::Arc;

    /// The rows `unwind` gives, one chunk each, with their chunk's column
    /// types: the tests read one row at a time.
    fn rows_of(unwind: &mut UnwindOperator) -> Vec<DataChunk> {
        let mut rows = Vec::new();
        while let Some(chunk) = unwind.next().unwrap() {
            rows.extend((0..chunk.row_count()).map(|row| chunk.slice(row, 1)));
        }
        rows
    }

    /// The rows of one input chunk come out together, up to a chunk's
    /// capacity, not one chunk per item.
    #[test]
    fn the_items_of_an_input_chunk_come_out_in_full_chunks() {
        use crate::execution::chunk::DEFAULT_CHUNK_SIZE;

        let items = DEFAULT_CHUNK_SIZE + 100;
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        for row in 0..2 {
            let list: Vec<Value> = (0..items / 2 + row)
                .map(|i| Value::Int64(i64::try_from(i).unwrap()))
                .collect();
            builder
                .column_mut(0)
                .unwrap()
                .push_value(Value::List(list.into()));
            builder.advance_row();
        }
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        );
        let mut sizes = Vec::new();
        while let Some(chunk) = unwind.next().unwrap() {
            sizes.push(chunk.row_count());
        }
        assert_eq!(sizes, [DEFAULT_CHUNK_SIZE, items + 1 - DEFAULT_CHUNK_SIZE]);
    }

    /// In a statement that writes, every row goes out in a chunk of its own.
    #[test]
    fn row_at_a_time_gives_one_row_per_chunk() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(Value::List(
            (0..5).map(Value::Int64).collect::<Vec<_>>().into(),
        ));
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        )
        .row_at_a_time();
        let mut sizes = Vec::new();
        while let Some(chunk) = unwind.next().unwrap() {
            sizes.push(chunk.row_count());
        }
        assert_eq!(sizes, [1, 1, 1, 1, 1]);
    }

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockOperator {
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
            "MockOperator"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// The input columns keep their types: a node stays a node.
    #[test]
    fn unwind_keeps_the_input_column_types() {
        use grafeo_common::types::NodeId;

        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_node_id(NodeId::new(101));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(vec![Value::Int64(1), Value::Int64(2)].into()));
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            1,
            "k".to_string(),
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Any],
            false,
            false,
        );

        let mut elements = Vec::new();
        for chunk in rows_of(&mut unwind) {
            assert_eq!(chunk.column_types()[0], LogicalType::Node);
            assert_eq!(
                chunk.column(0).unwrap().get_node_id(0).unwrap().as_u64(),
                101
            );
            assert!(chunk.column(0).unwrap().get_edge_id(0).is_none());
            elements.push(chunk.column(2).unwrap().get_value(0));
        }
        assert_eq!(elements, [Some(Value::Int64(1)), Some(Value::Int64(2))]);
    }

    /// The items of a node list go into a node column: IDs, a returned node (a
    /// map with `_id`) as its ID, and null. Any other item goes into a column
    /// of any value, so it is never read as a node.
    #[test]
    fn a_node_list_gives_nodes_and_other_items_stay_values() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let returned_node = Value::Map(Arc::new(BTreeMap::from([(
            PropertyKey::new("_id"),
            Value::Int64(5),
        )])));
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(Value::List(
            vec![
                Value::Int64(3),
                returned_node,
                Value::String("x".into()),
                Value::Null,
            ]
            .into(),
        ));
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "n".to_string(),
            vec![LogicalType::Node],
            false,
            false,
        );

        let mut items = Vec::new();
        for chunk in rows_of(&mut unwind) {
            let column = chunk.column(0).unwrap();
            items.push((
                column.data_type().clone(),
                column.get_node_id(0).map(|id| id.as_u64()),
                column.get_value(0),
            ));
        }
        assert_eq!(items[0].0, LogicalType::Node);
        assert_eq!(items[0].1, Some(3));
        assert_eq!(items[1].0, LogicalType::Node);
        assert_eq!(items[1].1, Some(5));
        assert_eq!(items[2].0, LogicalType::Any);
        assert_eq!(items[2].2, Some(Value::String("x".into())));
        assert_eq!(items[3].0, LogicalType::Node);
        assert_eq!(items[3].1, None);
        assert_eq!(items.len(), 4);
    }

    /// The row is as wide as the declared schema: an input column past the
    /// declared ones does not take the place of the unwound element.
    #[test]
    fn unwind_passes_on_only_the_declared_input_columns() {
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Any, LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_value(Value::Int64(7));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(vec![Value::Int64(1), Value::Int64(2)].into()));
        builder.column_mut(2).unwrap().push_value(Value::Int64(99));
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        // Declared: a column, the list and the element; the input's third
        // column is not one of them.
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            1,
            "k".to_string(),
            vec![LogicalType::Int64, LogicalType::Any, LogicalType::Any],
            false,
            false,
        );

        let mut rows = Vec::new();
        for chunk in rows_of(&mut unwind) {
            assert_eq!(chunk.column_count(), 3);
            rows.push((
                chunk.column(0).unwrap().get_value(0),
                chunk.column(2).unwrap().get_value(0),
            ));
        }
        assert_eq!(
            rows,
            [
                (Some(Value::Int64(7)), Some(Value::Int64(1))),
                (Some(Value::Int64(7)), Some(Value::Int64(2)))
            ]
        );
    }

    /// The rows pass on the list column, which a variable holds and which
    /// stays in scope (`WITH [3, 19] AS l UNWIND l AS x RETURN l, x`; it read
    /// null), unless the list is left out for being the UNWIND's own.
    #[test]
    fn unwind_passes_on_its_list_unless_left_out() {
        let list = Value::List(vec![Value::Int64(3), Value::Int64(19)].into());
        let run = |left_out: bool| {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
            builder.column_mut(0).unwrap().push_value(list.clone());
            builder.advance_row();
            let child = MockOperator {
                chunks: vec![builder.finish()],
                position: 0,
            };
            let unwind = UnwindOperator::new(
                Box::new(child),
                0,
                "x".to_string(),
                vec![LogicalType::Any, LogicalType::Any],
                false,
                false,
            );
            let mut unwind = if left_out {
                unwind.without_the_list()
            } else {
                unwind
            };
            let mut rows = Vec::new();
            for chunk in rows_of(&mut unwind) {
                rows.push((
                    chunk.column(0).unwrap().get_value(0),
                    chunk.column(1).unwrap().get_value(0),
                ));
            }
            rows
        };
        assert_eq!(
            run(false),
            [
                (Some(list.clone()), Some(Value::Int64(3))),
                (Some(list.clone()), Some(Value::Int64(19))),
            ]
        );
        assert_eq!(
            run(true),
            [
                (Some(Value::Null), Some(Value::Int64(3))),
                (Some(Value::Null), Some(Value::Int64(19))),
            ]
        );
    }

    #[test]
    fn test_unwind_basic() {
        // Create a chunk with a list column [1, 2, 3]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]); // Any for list
        let list = Value::List(Arc::new([
            Value::Int64(1),
            Value::Int64(2),
            Value::Int64(3),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Create unwind operator
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64], // Output is just the unwound element
            false,
            false,
        );

        // Should produce 3 rows
        let mut results = Vec::new();
        for chunk in rows_of(&mut unwind) {
            results.push(chunk);
        }

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_unwind_empty_list() {
        // A list with zero elements should produce no output rows
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        let mut results = Vec::new();
        for chunk in rows_of(&mut unwind) {
            results.push(chunk);
        }

        assert_eq!(results.len(), 0, "Empty list should produce no rows");
    }

    #[test]
    fn test_unwind_empty_input() {
        // No chunks at all
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        assert!(unwind.next().unwrap().is_none());
    }

    #[test]
    fn test_unwind_multiple_rows() {
        // Two rows with lists of different sizes
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);

        let list1 = Value::List(Arc::new([Value::Int64(10), Value::Int64(20)]));
        builder.column_mut(0).unwrap().push_value(list1);
        builder.advance_row();

        let list2 = Value::List(Arc::new([Value::Int64(30)]));
        builder.column_mut(0).unwrap().push_value(list2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Int64],
            false,
            false,
        );

        let mut count = 0;
        for _chunk in rows_of(&mut unwind) {
            count += 1;
        }

        // 2 from first list + 1 from second list = 3 rows
        assert_eq!(count, 3);
    }

    #[test]
    fn test_unwind_single_element_list() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([Value::String("hello".into())]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "item".to_string(),
            vec![LogicalType::String],
            false,
            false,
        );

        let mut results = Vec::new();
        for chunk in rows_of(&mut unwind) {
            results.push(chunk);
        }

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_unwind_variable_name() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "my_var".to_string(),
            vec![LogicalType::Any],
            false,
            false,
        );

        assert_eq!(unwind.variable_name(), "my_var");
    }

    #[test]
    fn test_unwind_name() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };

        let unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any],
            false,
            false,
        );

        assert_eq!(unwind.name(), "Unwind");
    }

    #[test]
    fn test_unwind_with_ordinality() {
        // Create a chunk with a list column [10, 20, 30]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([
            Value::Int64(10),
            Value::Int64(20),
            Value::Int64(30),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Output schema: [element (Any), ordinality (Int64)]
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Int64],
            true,  // emit_ordinality
            false, // emit_offset
        );

        let mut ordinalities = Vec::new();
        for chunk in rows_of(&mut unwind) {
            if let Some(col) = chunk.column(1)
                && let Some(Value::Int64(v)) = col.get_value(0)
            {
                ordinalities.push(v);
            }
        }

        // ORDINALITY is 1-based
        assert_eq!(ordinalities, vec![1, 2, 3]);
    }

    #[test]
    fn test_unwind_with_offset() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        let list = Value::List(Arc::new([
            Value::Int64(10),
            Value::Int64(20),
            Value::Int64(30),
        ]));
        builder.column_mut(0).unwrap().push_value(list);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Output schema: [element (Any), offset (Int64)]
        let mut unwind = UnwindOperator::new(
            Box::new(mock),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Int64],
            false, // emit_ordinality
            true,  // emit_offset
        );

        let mut offsets = Vec::new();
        for chunk in rows_of(&mut unwind) {
            if let Some(col) = chunk.column(1)
                && let Some(Value::Int64(v)) = col.get_value(0)
            {
                offsets.push(v);
            }
        }

        // OFFSET is 0-based
        assert_eq!(offsets, vec![0, 1, 2]);
    }

    #[test]
    fn test_unwind_into_any() {
        let mock = MockOperator {
            chunks: vec![],
            position: 0,
        };
        let op = UnwindOperator::new(
            Box::new(mock),
            0,
            "items".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<UnwindOperator>().is_ok());
    }

    /// The rows an UNWIND of the range with bounds `bounds` gives: the
    /// items, and the ordinality of each.
    fn range_rows(bounds: Value) -> Vec<(Value, Value)> {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder.column_mut(0).unwrap().push_value(bounds);
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Int64],
            true,
            false,
        )
        .without_the_list()
        .range_bounds();
        rows_of(&mut unwind)
            .iter()
            .map(|row| {
                assert_eq!(row.column(0).unwrap().get_value(0), Some(Value::Null));
                (
                    row.column(1).unwrap().get_value(0).unwrap(),
                    row.column(2).unwrap().get_value(0).unwrap(),
                )
            })
            .collect()
    }

    fn bounds(bounds: &[i64]) -> Value {
        Value::List(
            bounds
                .iter()
                .map(|&bound| Value::Int64(bound))
                .collect::<Vec<_>>()
                .into(),
        )
    }

    /// The integers of `range(start, stop[, step])`, numbered from 1, as the
    /// list `range` gives would unwind.
    fn items(items: &[i64]) -> Vec<(Value, Value)> {
        items
            .iter()
            .zip(1..)
            .map(|(&item, ordinal)| (Value::Int64(item), Value::Int64(ordinal)))
            .collect()
    }

    /// A range unwinds the integers `range` gives, in order, with the same
    /// ends, steps and empty ranges, and stops before a step that overflows.
    #[test]
    fn a_range_unwinds_the_integers_range_gives() {
        assert_eq!(range_rows(bounds(&[1, 5])), items(&[1, 2, 3, 4, 5]));
        assert_eq!(range_rows(bounds(&[3, 3])), items(&[3]));
        assert_eq!(range_rows(bounds(&[1, 10, 3])), items(&[1, 4, 7, 10]));
        assert_eq!(range_rows(bounds(&[1, 9, 3])), items(&[1, 4, 7]));
        assert_eq!(range_rows(bounds(&[10, 1, -4])), items(&[10, 6, 2]));
        assert_eq!(range_rows(bounds(&[5, 1])), items(&[]));
        assert_eq!(range_rows(bounds(&[1, 5, -1])), items(&[]));
        assert_eq!(
            range_rows(bounds(&[i64::MAX - 1, i64::MAX])),
            items(&[i64::MAX - 1, i64::MAX])
        );
        assert_eq!(
            range_rows(bounds(&[i64::MAX - 3, i64::MAX, 2])),
            items(&[i64::MAX - 3, i64::MAX - 1])
        );
        assert_eq!(
            range_rows(bounds(&[i64::MIN + 1, i64::MIN, -1])),
            items(&[i64::MIN + 1, i64::MIN])
        );
        assert_eq!(
            range_rows(bounds(&[i64::MIN, i64::MAX, i64::MAX])),
            items(&[i64::MIN, -1, i64::MAX - 1])
        );
    }

    /// Bounds `range` gives null for (a step of 0, a bound that is no
    /// integer, a missing bound) give no rows, as null does.
    #[test]
    fn a_range_without_integers_unwinds_nothing() {
        assert_eq!(range_rows(bounds(&[1, 5, 0])), items(&[]));
        assert_eq!(range_rows(bounds(&[1])), items(&[]));
        assert_eq!(
            range_rows(Value::List(
                vec![Value::Float64(1.0), Value::Int64(5)].into()
            )),
            items(&[])
        );
        assert_eq!(
            range_rows(Value::List(vec![Value::Int64(1), Value::Null].into())),
            items(&[])
        );
        assert_eq!(range_rows(Value::Null), items(&[]));
    }

    /// A range far longer than memory holds unwinds a chunk at a time.
    #[test]
    fn a_range_is_unwound_without_its_list() {
        use crate::execution::chunk::DEFAULT_CHUNK_SIZE;

        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(bounds(&[1, i64::MAX]));
        builder.advance_row();
        let child = MockOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        )
        .range_bounds();
        for first in [1, 1 + DEFAULT_CHUNK_SIZE] {
            let chunk = unwind.next().unwrap().unwrap();
            assert_eq!(chunk.row_count(), DEFAULT_CHUNK_SIZE);
            assert_eq!(
                chunk.column(1).unwrap().get_value(0),
                Some(Value::Int64(i64::try_from(first).unwrap()))
            );
        }
    }

    /// Input rows without items are read under the query's deadline: an
    /// UNWIND that finds no item fails with the timeout once it has passed.
    #[test]
    fn rows_without_items_are_read_under_the_deadline() {
        let chunks = (0..3)
            .map(|_| {
                let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
                builder.column_mut(0).unwrap().push_value(Value::Null);
                builder.advance_row();
                builder.finish()
            })
            .collect();
        let child = MockOperator {
            chunks,
            position: 0,
        };
        let mut unwind = UnwindOperator::new(
            Box::new(child),
            0,
            "x".to_string(),
            vec![LogicalType::Any, LogicalType::Any],
            false,
            false,
        );
        let _deadline =
            crate::execution::deadline::enter(Some(grafeo_common::clock::Instant::now()));
        assert!(matches!(
            unwind.next(),
            Err(super::super::OperatorError::Timeout)
        ));
    }
}
