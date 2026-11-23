//! A limit after a spilled sort stops the merge of its runs: the merge reads
//! back fewer rows than the runs hold, by the spill manager's count of rows
//! read back from its runs.
//!
//! The count is `SpillManager::rows_read_back`: the rows the merges of its
//! runs have read back from their files, every pass counted.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test main spilled_sort_reads_back::
//! ```

#![cfg(feature = "spill")]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_core::execution::operators::push::{SortKey, SpillableSortPushOperator};
use grafeo_core::execution::operators::{LimitPushOperator, OperatorError};
use grafeo_core::execution::{
    CollectorSink, DataChunk, Pipeline, Source, SpillManager, ValueVector,
};

/// Eight chunks of 2,048 numbers, a run each.
struct Numbers(i64);

impl Source for Numbers {
    fn next_chunk(&mut self, _size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.0 == 8 {
            return Ok(None);
        }
        let start = self.0 * 2048;
        self.0 += 1;
        let values: Vec<Value> = (start..start + 2048)
            .map(|i| Value::Int64(i * 7919 % 16_384))
            .collect();
        Ok(Some(DataChunk::new(vec![ValueVector::from_values(
            &values,
        )])))
    }

    fn reset(&mut self) {
        self.0 = 0;
    }

    fn name(&self) -> &'static str {
        "Numbers"
    }
}

#[test]
fn a_limit_after_a_spilled_sort_reads_back_fewer_rows_than_the_runs_hold() {
    let dir = tempfile::tempdir().unwrap();
    let manager = Arc::new(SpillManager::new(dir.path().join("spill")).unwrap());
    let sort = SpillableSortPushOperator::with_spilling(
        vec![SortKey::ascending(0)],
        Arc::clone(&manager),
        2048,
    );
    let mut pipeline = Pipeline::new(
        Box::new(Numbers(0)),
        vec![Box::new(sort), Box::new(LimitPushOperator::new(10))],
        Box::new(CollectorSink::new()),
    );
    pipeline.execute().unwrap();
    let read_back = manager.rows_read_back();
    assert!(read_back > 0, "the merge read its runs");
    assert!(
        read_back < 16_384 / 2,
        "read back {read_back} of the 16,384 rows the runs hold"
    );
    let sink = pipeline
        .into_sink()
        .into_any()
        .downcast::<CollectorSink>()
        .unwrap();
    let rows: Vec<Value> = sink
        .into_chunks()
        .iter()
        .flat_map(|chunk| {
            (0..chunk.len())
                .map(|row| chunk.column(0).unwrap().get_value(row).unwrap())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(rows, (0..10).map(Value::Int64).collect::<Vec<_>>());
}
