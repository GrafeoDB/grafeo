//! Unwind operator for expanding lists into individual rows.

use super::{Operator, OperatorResult};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use grafeo_common::types::{LogicalType, Value};

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
    pub fn new(
        child: Box<dyn Operator>,
        list_col_idx: usize,
        variable_name: String,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            list_col_idx,
            variable_name,
            output_schema,
            current_chunk: None,
            current_row: 0,
            current_list_idx: 0,
            current_list: None,
        }
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

            let chunk = self.current_chunk.as_ref().unwrap();

            // Find the next row with a list value
            while self.current_row < chunk.row_count() {
                if let Some(col) = chunk.column(self.list_col_idx)
                    && let Some(value) = col.get_value(self.current_row)
                    && let Value::List(list_arc) = value
                {
                    // Found a list - store it and return first element
                    let list: Vec<Value> = list_arc.iter().cloned().collect();
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
        let chunk = self.current_chunk.as_ref().unwrap();
        let list = self.current_list.as_ref().unwrap();
        let element = list[self.current_list_idx].clone();

        // Build output row: copy all columns from input + add the unwound element
        let mut builder = DataChunkBuilder::new(&self.output_schema);

        // Copy existing columns (except the list column which we're replacing)
        for col_idx in 0..chunk.column_count() {
            if col_idx == self.list_col_idx {
                continue; // Skip the list column
            }
            if let Some(col) = chunk.column(col_idx)
                && let Some(value) = col.get_value(self.current_row)
                && let Some(out_col) = builder.column_mut(col_idx)
            {
                out_col.push_value(value);
            }
        }

        // Add the unwound element as the last column
        let new_col_idx = self.output_schema.len() - 1;
        if let Some(out_col) = builder.column_mut(new_col_idx) {
            out_col.push_value(element);
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
        );

        // Should produce 3 rows
        let mut results = Vec::new();
        while let Ok(Some(chunk)) = unwind.next() {
            results.push(chunk);
        }

        assert_eq!(results.len(), 3);
    }
}
