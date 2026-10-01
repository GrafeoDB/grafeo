//! Shuffle operator: returns its input's rows in random order.

use grafeo_common::types::Value;

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder};

/// Returns its input's rows in random order.
///
/// Planned at the root of a query without `ORDER BY` when the database's
/// `shuffle_unordered` test option is on, so that tests find code relying on
/// a row order that is unspecified. A stream shuffles each chunk on its own
/// (see [`per_chunk`](Self::per_chunk)). The rows keep their columns' types
/// and values.
pub struct ShuffleOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Whether each input chunk is shuffled on its own.
    per_chunk: bool,
    /// The state of the random stream, seeded differently per operator.
    state: u64,
    /// Materialized chunks (the current one, per chunk).
    chunks: Vec<DataChunk>,
    /// The rows as `(chunk, row)`, in the order they are returned.
    rows: Vec<(usize, usize)>,
    /// Whether the input is materialized and shuffled.
    shuffled: bool,
    /// Current position in `rows`.
    position: usize,
}

impl ShuffleOperator {
    /// Creates a shuffle operator over `child` that returns all of its rows
    /// in random order, after reading the whole input.
    pub fn new(child: Box<dyn Operator>) -> Self {
        Self {
            child,
            per_chunk: false,
            state: random_seed(),
            chunks: Vec::new(),
            rows: Vec::new(),
            shuffled: false,
            position: 0,
        }
    }

    /// Creates a shuffle operator over `child` that reads one input chunk at
    /// a time and returns its rows in random order: memory stays at one
    /// chunk, as a stream needs, and rows stay within their chunk.
    pub fn per_chunk(child: Box<dyn Operator>) -> Self {
        Self {
            per_chunk: true,
            ..Self::new(child)
        }
    }

    /// Materializes the input and puts its rows in random order.
    fn shuffle(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            self.take(chunk);
        }
        self.shuffle_rows();
        self.shuffled = true;
        Ok(())
    }

    /// Reads input chunks until one has rows and puts them in random order;
    /// `false` once the input is exhausted.
    fn shuffle_next_chunk(&mut self) -> Result<bool, OperatorError> {
        self.chunks.clear();
        self.rows.clear();
        self.position = 0;
        while let Some(chunk) = self.child.next()? {
            self.take(chunk);
            if !self.rows.is_empty() {
                self.shuffle_rows();
                return Ok(true);
            }
            self.chunks.clear();
        }
        Ok(false)
    }

    /// Keeps `chunk` and lists its rows.
    fn take(&mut self, chunk: DataChunk) {
        let chunk_index = self.chunks.len();
        for row in chunk.selected_indices() {
            self.rows.push((chunk_index, row));
        }
        self.chunks.push(chunk);
    }

    /// Fisher-Yates over the listed rows, with a SplitMix64 stream.
    fn shuffle_rows(&mut self) {
        for i in (1..self.rows.len()).rev() {
            let bound = u64::try_from(i + 1).unwrap_or(u64::MAX);
            let j = usize::try_from(next_random(&mut self.state) % bound).unwrap_or(i);
            self.rows.swap(i, j);
        }
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
        if self.per_chunk {
            if self.position >= self.rows.len() && !self.shuffle_next_chunk()? {
                return Ok(None);
            }
        } else if !self.shuffled {
            self.shuffle()?;
        }
        if self.position >= self.rows.len() {
            return Ok(None);
        }

        let mut column_types = ColumnTypes::default();
        for chunk in &self.chunks {
            column_types.add(chunk);
        }
        let mut builder = DataChunkBuilder::with_capacity(column_types.types(), 2048);
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
        self.state = random_seed();
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use grafeo_common::types::LogicalType;

    /// Returns its chunks once, counting how many it handed out.
    struct Chunks(Vec<DataChunk>, Arc<AtomicUsize>);

    impl Operator for Chunks {
        fn next(&mut self) -> OperatorResult {
            let chunk = self.0.pop();
            if chunk.is_some() {
                self.1.fetch_add(1, Ordering::Relaxed);
            }
            Ok(chunk)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Chunks"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// The values 0 to 99 as three input chunks (0..40 first), and the
    /// counter of chunks handed out.
    fn input() -> (Chunks, Arc<AtomicUsize>) {
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
        let pulled = Arc::new(AtomicUsize::new(0));
        (Chunks(chunks, Arc::clone(&pulled)), pulled)
    }

    fn values(chunk: &DataChunk) -> Vec<i64> {
        chunk
            .selected_indices()
            .map(|row| match chunk.column(0).unwrap().get_value(row) {
                Some(Value::Int64(value)) => value,
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    /// The values 0 to 99, in three chunks, returned by a shuffle.
    fn shuffled() -> Vec<i64> {
        let mut shuffle = ShuffleOperator::new(Box::new(input().0));
        let mut all = Vec::new();
        while let Some(chunk) = shuffle.next().unwrap() {
            all.extend(values(&chunk));
        }
        all
    }

    /// A per-chunk shuffle reads one input chunk per output chunk and keeps
    /// each chunk's rows together, in an order that changes between runs.
    #[test]
    fn a_per_chunk_shuffle_reads_one_chunk_at_a_time() {
        let mut first_chunks = std::collections::HashSet::new();
        for _ in 0..5 {
            let (chunks, pulled) = input();
            let mut shuffle = ShuffleOperator::per_chunk(Box::new(chunks));
            let mut outputs = Vec::new();
            while let Some(chunk) = shuffle.next().unwrap() {
                outputs.push(values(&chunk));
                assert_eq!(
                    pulled.load(Ordering::Relaxed),
                    outputs.len(),
                    "one input chunk per output chunk"
                );
            }
            let sorted: Vec<Vec<i64>> = outputs
                .iter()
                .map(|chunk| {
                    let mut chunk = chunk.clone();
                    chunk.sort_unstable();
                    chunk
                })
                .collect();
            assert_eq!(
                sorted,
                [
                    (0..40).collect::<Vec<_>>(),
                    (40..70).collect(),
                    (70..100).collect()
                ]
            );
            first_chunks.insert(outputs[0].clone());
        }
        assert!(first_chunks.len() > 1, "five runs gave the same order");
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
