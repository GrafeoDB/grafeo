//! Eager operator: reads its whole input before it returns a row.

use std::collections::VecDeque;

use super::{DataChunk, Operator, OperatorResult};

/// Reads its whole input before it returns the first chunk, then returns the
/// input's chunks unchanged, in order.
///
/// Placed above a writing input, it lets what reads the store above it see
/// all of what the input writes, as clause-at-a-time semantics ask: in
/// `UNWIND ... MERGE (h:Hub) CREATE (h)-[:R]->(:Q) WITH h MATCH (h)-[:R]->(q)`
/// each row sees every edge, also those that later rows create. Without it,
/// rows flow up one at a time and a row sees only what the rows before it
/// wrote.
pub struct EagerOperator {
    input: Box<dyn Operator>,
    /// The input's chunks, once it is read.
    chunks: Option<VecDeque<DataChunk>>,
}

impl EagerOperator {
    /// Creates an eager operator over `input`.
    pub fn new(input: Box<dyn Operator>) -> Self {
        Self {
            input,
            chunks: None,
        }
    }
}

impl Operator for EagerOperator {
    fn next(&mut self) -> OperatorResult {
        if self.chunks.is_none() {
            let mut chunks = VecDeque::new();
            while let Some(chunk) = self.input.next()? {
                chunks.push_back(chunk);
            }
            self.chunks = Some(chunks);
        }
        Ok(self.chunks.as_mut().and_then(VecDeque::pop_front))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.chunks = None;
    }

    fn name(&self) -> &'static str {
        "Eager"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, Ordering};

    use grafeo_common::types::{LogicalType, Value};

    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

    /// An input that "writes" one unit per chunk it returns: three chunks of
    /// one row each, holding 1, 2 and 3.
    struct Writing {
        written: Arc<AtomicI64>,
        position: i64,
    }

    impl Operator for Writing {
        fn next(&mut self) -> OperatorResult {
            if self.position == 3 {
                return Ok(None);
            }
            self.position += 1;
            self.written.fetch_add(1, Ordering::SeqCst);
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

    /// The (row, written) pairs read from `operator`: each row with what was
    /// written when it came out.
    fn seen(operator: &mut dyn Operator, written: &AtomicI64) -> Vec<(i64, i64)> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                let Some(Value::Int64(value)) = chunk.column(0).unwrap().get_value(row) else {
                    panic!("expected an integer");
                };
                rows.push((value, written.load(Ordering::SeqCst)));
            }
        }
        rows
    }

    /// Every row comes out after the whole input was read (written), in the
    /// input's order; the input alone passes each row on as it writes it.
    /// After a reset, the input is read again.
    #[test]
    fn every_row_comes_out_after_the_whole_input_is_read() {
        let written = Arc::new(AtomicI64::new(0));
        let mut lazy = Writing {
            written: Arc::clone(&written),
            position: 0,
        };
        assert_eq!(seen(&mut lazy, &written), [(1, 1), (2, 2), (3, 3)]);

        let written = Arc::new(AtomicI64::new(0));
        let mut eager = EagerOperator::new(Box::new(Writing {
            written: Arc::clone(&written),
            position: 0,
        }));
        assert_eq!(seen(&mut eager, &written), [(1, 3), (2, 3), (3, 3)]);
        assert!(eager.next().unwrap().is_none(), "the input stays read");
        eager.reset();
        assert_eq!(
            seen(&mut eager, &written),
            [(1, 6), (2, 6), (3, 6)],
            "a reset reads the input again"
        );
        assert_eq!(eager.name(), "Eager");
    }
}
