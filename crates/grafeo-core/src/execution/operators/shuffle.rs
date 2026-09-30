//! Shuffle operator: returns its input's rows in random order.

use grafeo_common::types::{LogicalType, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// Returns its input's rows in random order.
///
/// Planned at the root of a query without `ORDER BY` when the database's
/// `shuffle_unordered` test option is on, so that tests find code relying on
/// a row order that is unspecified.
pub struct ShuffleOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Materialized chunks.
    chunks: Vec<DataChunk>,
    /// The rows as `(chunk, row)`, in the order they are returned.
    rows: Vec<(usize, usize)>,
    /// Whether the input is materialized and shuffled.
    shuffled: bool,
    /// Current position in `rows`.
    position: usize,
}

impl ShuffleOperator {
    /// Creates a shuffle operator over `child`.
    pub fn new(child: Box<dyn Operator>, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            output_schema,
            chunks: Vec::new(),
            rows: Vec::new(),
            shuffled: false,
            position: 0,
        }
    }

    /// Materializes the input and puts its rows in random order.
    fn shuffle(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            let chunk_index = self.chunks.len();
            for row in chunk.selected_indices() {
                self.rows.push((chunk_index, row));
            }
            self.chunks.push(chunk);
        }

        // Fisher-Yates, with a SplitMix64 stream seeded differently per run.
        let mut state = random_seed();
        for i in (1..self.rows.len()).rev() {
            let bound = u64::try_from(i + 1).unwrap_or(u64::MAX);
            let j = usize::try_from(next_random(&mut state) % bound).unwrap_or(i);
            self.rows.swap(i, j);
        }
        self.shuffled = true;
        Ok(())
    }
}

/// A seed that differs between calls and processes, without a dependency on
/// a random number crate: std seeds each `RandomState` from the system.
fn random_seed() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(0x5eed_u64)
}

/// The next value of a SplitMix64 stream.
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Operator for ShuffleOperator {
    fn next(&mut self) -> OperatorResult {
        if !self.shuffled {
            self.shuffle()?;
        }
        if self.position >= self.rows.len() {
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);
        while self.position < self.rows.len() && !builder.is_full() {
            let (chunk_index, row) = self.rows[self.position];
            let source_chunk = &self.chunks[chunk_index];
            for col_idx in 0..source_chunk.column_count() {
                if let (Some(src_col), Some(dst_col)) =
                    (source_chunk.column(col_idx), builder.column_mut(col_idx))
                {
                    dst_col.push_value(src_col.get_value(row).unwrap_or(Value::Null));
                }
            }
            builder.advance_row();
            self.position += 1;
        }
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.chunks.clear();
        self.rows.clear();
        self.shuffled = false;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Shuffle"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns its chunks once.
    struct Chunks(Vec<DataChunk>);

    impl Operator for Chunks {
        fn next(&mut self) -> OperatorResult {
            Ok(self.0.pop())
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Chunks"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// The values 0 to 99, in three chunks, returned by a shuffle.
    fn shuffled() -> Vec<i64> {
        let chunks = [0..40, 40..70, 70..100]
            .into_iter()
            .rev()
            .map(|range| {
                let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
                for value in range {
                    builder.column_mut(0).unwrap().push_int64(value);
                    builder.advance_row();
                }
                builder.finish()
            })
            .collect();
        let mut shuffle = ShuffleOperator::new(Box::new(Chunks(chunks)), vec![LogicalType::Int64]);
        let mut values = Vec::new();
        while let Some(chunk) = shuffle.next().unwrap() {
            for row in chunk.selected_indices() {
                match chunk.column(0).unwrap().get_value(row) {
                    Some(Value::Int64(value)) => values.push(value),
                    other => panic!("unexpected {other:?}"),
                }
            }
        }
        values
    }

    #[test]
    fn every_row_comes_back_once() {
        let mut values = shuffled();
        values.sort_unstable();
        assert_eq!(values, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn the_order_changes_between_runs() {
        let orders: std::collections::HashSet<Vec<i64>> = (0..5).map(|_| shuffled()).collect();
        assert!(orders.len() > 1, "five runs gave the same order");
        assert!(!orders.contains(&(0..100).collect::<Vec<_>>()));
    }
}
