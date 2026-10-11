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
    /// Current input chunk being processed.
    current_chunk: Option<DataChunk>,
    /// Current row index within the chunk.
    current_row: usize,
    /// Current index within the list being unwound.
    current_list_idx: usize,
    /// Current list being unwound.
    current_list: Option<Vec<Value>>,
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
            current_chunk: None,
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

    /// Returns the variable name for the unwound elements.
    #[must_use]
    pub fn variable_name(&self) -> &str {
        &self.variable_name
    }

    /// Advances to the next row or fetches the next chunk.
    fn advance(&mut self) -> OperatorResult {
        loop {
            // If we have a current list, try to get the next element
            if let Some(list) = &self.current_list
                && self.current_list_idx < list.len()
            {
                // Still have elements in the current list
                return Ok(Some(self.emit_row()?));
            }

            // Need to move to the next row
            self.current_list_idx = 0;
            self.current_list = None;

            // Get the next chunk if needed
            if self.current_chunk.is_none() {
                self.current_chunk = self.child.next()?;
                self.current_row = 0;
                if self.current_chunk.is_none() {
                    return Ok(None); // No more data
                }
            }

            let chunk = self
                .current_chunk
                .as_ref()
                .expect("current_chunk is Some: checked above");

            // Find the next row with a list value
            while self.current_row < chunk.row_count() {
                if let Some(col) = chunk.column(self.list_col_idx)
                    && let Some(value) = col.get_value(self.current_row)
                {
                    // Extract list elements from either Value::List or Value::Vector
                    let list = match value {
                        Value::List(list_arc) => list_arc.iter().cloned().collect::<Vec<Value>>(),
                        Value::Vector(vec_arc) => {
                            vec_arc.iter().map(|f| Value::Float64(*f as f64)).collect()
                        }
                        _ => {
                            self.current_row += 1;
                            continue;
                        }
                    };
                    if !list.is_empty() {
                        self.current_list = Some(list);
                        return Ok(Some(self.emit_row()?));
                    }
                }
                self.current_row += 1;
            }

            // Exhausted current chunk, get next one
            self.current_chunk = None;
        }
    }

    /// Emits a single row with the current list element.
    fn emit_row(&mut self) -> Result<DataChunk, super::OperatorError> {
        let chunk = self
            .current_chunk
            .as_ref()
            .expect("current_chunk is Some: set before emit_row call");
        let list = self
            .current_list
            .as_ref()
            .expect("current_list is Some: set before emit_row call");
        let element = list[self.current_list_idx].clone();

        // The unwound element comes after the declared input columns, followed
        // by any ordinality/offset columns.
        let extra_cols = usize::from(self.emit_ordinality) + usize::from(self.emit_offset);
        let element_col_idx = self.output_schema.len() - 1 - extra_cols;

        // Build output row: copy the declared input columns + add the unwound
        // element. The copied columns keep the input's types (a node or edge
        // stays one, see `ColumnTypes`); the new ones have the declared types.
        // The row is as wide as the declared schema.
        let mut types = chunk.column_types();
        types.truncate(element_col_idx);
        let copied = types.len();
        let mut types = copied_column_types(&types, &self.output_schema);
        types.extend(self.output_schema.iter().skip(copied).cloned());
        // An item of a node or edge list is an ID (a returned node or edge, a
        // map with `_id`, stands for its ID) or null; any other item goes into
        // a column of any value, so it is never read as a node or an edge.
        let element = entity_item(element, &self.output_schema[element_col_idx]);
        if matches!(
            self.output_schema[element_col_idx],
            LogicalType::Node | LogicalType::Edge
        ) && !matches!(element, Value::Int64(_) | Value::Null)
        {
            types[element_col_idx] = LogicalType::Any;
        }
        let mut builder = DataChunkBuilder::new(&types);

        // Copy the input columns, the list too unless it is left out (a
        // variable that holds the list stays in scope; it read null).
        for col_idx in 0..copied {
            if col_idx == self.list_col_idx && !self.passes_on_the_list {
                if let Some(out_col) = builder.column_mut(col_idx) {
                    out_col.push_value(Value::Null);
                }
                continue;
            }
            if let Some(col) = chunk.column(col_idx)
                && let Some(value) = col.get_value(self.current_row)
                && let Some(out_col) = builder.column_mut(col_idx)
            {
                out_col.push_value(value);
            }
        }

        // Add the unwound element column.
        if let Some(out_col) = builder.column_mut(element_col_idx) {
            out_col.push_value(element);
        }

        // Add ORDINALITY (1-based) if requested
        let mut next_col = element_col_idx + 1;
        if self.emit_ordinality {
            if let Some(out_col) = builder.column_mut(next_col) {
                // reason: list index fits i64 for practical sizes
                #[allow(clippy::cast_possible_wrap)]
                out_col.push_value(Value::Int64((self.current_list_idx + 1) as i64));
            }
            next_col += 1;
        }

        // Add OFFSET (0-based) if requested
        if self.emit_offset
            && let Some(out_col) = builder.column_mut(next_col)
        {
            // reason: list index fits i64 for practical sizes
            #[allow(clippy::cast_possible_wrap)]
            out_col.push_value(Value::Int64(self.current_list_idx as i64));
        }

        builder.advance_row();
        self.current_list_idx += 1;

        // If we've exhausted this list, move to the next row
        if self.current_list_idx >= list.len() {
            self.current_row += 1;
        }

        Ok(builder.finish())
    }
}

impl Operator for UnwindOperator {
    fn next(&mut self) -> OperatorResult {
        self.advance()
    }

    fn reset(&mut self) {
        self.child.reset();
        self.current_chunk = None;
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
        while let Some(chunk) = unwind.next().unwrap() {
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
        while let Some(chunk) = unwind.next().unwrap() {
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
        while let Some(chunk) = unwind.next().unwrap() {
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
            while let Some(chunk) = unwind.next().unwrap() {
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
        while let Ok(Some(chunk)) = unwind.next() {
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
        while let Ok(Some(chunk)) = unwind.next() {
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
        while let Ok(Some(_chunk)) = unwind.next() {
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
        while let Ok(Some(chunk)) = unwind.next() {
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
        while let Ok(Some(chunk)) = unwind.next() {
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
        while let Ok(Some(chunk)) = unwind.next() {
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
}
