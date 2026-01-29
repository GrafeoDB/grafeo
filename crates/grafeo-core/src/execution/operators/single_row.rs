//! Single row operator for producing a single empty row.
//!
//! This is used for queries like `UNWIND [1,2,3] AS x RETURN x` that don't have
//! a MATCH clause but need an initial row to start with.

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;

/// An operator that produces exactly one empty row.
///
/// This is useful for UNWIND clauses that operate on literal lists
/// without a prior MATCH clause.
pub struct SingleRowOperator {
    /// Whether the single row has been produced.
    produced: bool,
}

impl SingleRowOperator {
    /// Creates a new single row operator.
    #[must_use]
    pub fn new() -> Self {
        Self { produced: false }
    }
}

impl Default for SingleRowOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for SingleRowOperator {
    fn next(&mut self) -> OperatorResult {
        if self.produced {
            return Ok(None);
        }

        self.produced = true;

        // Create a single row with no columns
        let mut chunk = DataChunk::with_capacity(&[], 1);
        chunk.set_count(1);

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.produced = false;
    }

    fn name(&self) -> &'static str {
        "SingleRow"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_row_operator() {
        let mut op = SingleRowOperator::new();

        // First call produces one row
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Second call produces None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset, produces one row again
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
    }
}
