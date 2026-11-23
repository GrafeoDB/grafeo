//! The streaming merge of sorted runs: its rows against the in-memory sort,
//! its memory against its grant, and its files against its lifetime.

use std::cmp::Ordering;
use std::sync::Arc;

use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::types::Value;
use tempfile::TempDir;

use super::{
    MERGE_FAN_IN, MERGE_MIN_READ_BUFFER, MERGE_READ_BUFFER, RowOrder, SortedRuns, SpillManager,
};
use crate::execution::memory::{OperatorMemory, OperatorMemoryContext};
use crate::execution::operators::OperatorError;

/// Orders rows by their first column, an integer or null, nulls last; the
/// second column is the row's place in the input, which only the tie rule
/// orders.
struct ByFirst;

impl RowOrder for ByFirst {
    type Key = Option<i64>;

    fn key(&mut self, row: &[Value]) -> Result<Self::Key, OperatorError> {
        Ok(match row[0] {
            Value::Int64(value) => Some(value),
            _ => None,
        })
    }

    fn compare(&self, a: &Self::Key, b: &Self::Key) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }

    fn key_bytes(&self, _key: &Self::Key) -> usize {
        std::mem::size_of::<Self::Key>()
    }
}

/// Rows of the input: seven key values and a null, so most rows tie, and a
/// string column so a row holds heap memory.
fn input(rows: usize) -> Vec<Vec<Value>> {
    (0..rows)
        .map(|place| {
            let key = match place * 7919 % 8 {
                7 => Value::Null,
                k => Value::Int64(i64::try_from(k).unwrap() - 3),
            };
            vec![
                key,
                Value::Int64(i64::try_from(place).unwrap()),
                Value::from(format!("row {place}")),
            ]
        })
        .collect()
}

/// The in-memory sort: a stable sort, so ties keep input order.
fn in_memory(rows: &[Vec<Value>]) -> Vec<Vec<Value>> {
    let mut order = ByFirst;
    let mut keyed: Vec<_> = rows
        .iter()
        .map(|row| (order.key(row).unwrap(), row.clone()))
        .collect();
    keyed.sort_by(|a, b| order.compare(&a.0, &b.0));
    keyed.into_iter().map(|(_, row)| row).collect()
}

fn manager() -> (TempDir, Arc<SpillManager>) {
    let dir = TempDir::new().unwrap();
    let manager = Arc::new(SpillManager::new(dir.path().join("spill")).unwrap());
    (dir, manager)
}

/// A statement's context capped at `limit`, spilling through `manager`.
fn context(manager: &Arc<SpillManager>, limit: usize) -> OperatorMemoryContext {
    OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
        .with_spill_manager(Some(Arc::clone(manager)))
        .with_query_limit(limit)
}

/// The memory of an operator in a statement capped at `limit`.
fn memory(manager: &Arc<SpillManager>, limit: usize) -> OperatorMemory {
    OperatorMemory::new(
        OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
            .with_spill_manager(Some(Arc::clone(manager)))
            .with_query_limit(limit),
    )
}

/// `rows` cut into runs of `run_rows` in input order, each sorted in memory,
/// as a sort spills them.
fn runs_of(
    manager: &Arc<SpillManager>,
    rows: &[Vec<Value>],
    run_rows: usize,
) -> SortedRuns<ByFirst> {
    let mut runs = SortedRuns::new(Arc::clone(manager), 3, ByFirst);
    for run in rows.chunks(run_rows) {
        runs.spill(in_memory(run)).unwrap();
    }
    runs
}

/// Files left in the spill directory.
fn files_left(dir: &TempDir) -> usize {
    std::fs::read_dir(dir.path().join("spill")).map_or(0, |entries| entries.count())
}

/// The passes a merge of `runs` runs takes at `fan_in`: groups of `fan_in`
/// consecutive runs merge into one until at most `fan_in` are left, then the
/// last pass.
fn passes_for(mut runs: usize, fan_in: usize) -> usize {
    let mut passes = 1;
    while runs > fan_in {
        runs = runs.div_ceil(fan_in);
        passes += 1;
    }
    passes
}

/// Every batch of a merge, checking each is at most `batch_rows` rows.
fn drain(
    merge: &mut super::RunMerge<ByFirst>,
    batch_rows: usize,
) -> Result<Vec<Vec<Value>>, OperatorError> {
    let mut out = Vec::new();
    while let Some(batch) = merge.next_batch()? {
        assert!(
            !batch.is_empty() && batch.len() <= batch_rows,
            "{}",
            batch.len()
        );
        out.extend(batch);
    }
    Ok(out)
}

/// At 1, 15, 16, 17 and 300 runs (one pass, a full fan-in, one pass past it,
/// three passes), the merge gives exactly the in-memory sort's rows in its
/// order: ties by run and then by place, nulls last. Nothing stays on disk.
#[test]
fn a_merge_gives_the_in_memory_sort_at_every_run_count() {
    for runs in [1, 15, 16, 17, 300] {
        let (dir, manager) = manager();
        let rows = input(runs * 8);
        let sorted = runs_of(&manager, &rows, 8);
        assert_eq!(sorted.run_count(), runs);
        let mut merge = sorted.merge(memory(&manager, usize::MAX), 100).unwrap();
        assert_eq!(
            drain(&mut merge, 100).unwrap(),
            in_memory(&rows),
            "{runs} runs"
        );
        assert_eq!(merge.fan_in(), MERGE_FAN_IN.min(runs).max(2), "{runs} runs");
        assert_eq!(
            merge.passes(),
            passes_for(runs, merge.fan_in()),
            "{runs} runs"
        );
        drop(merge);
        assert_eq!(files_left(&dir), 0, "{runs} runs");
    }
}

/// No runs: no rows.
#[test]
fn a_merge_of_no_runs_gives_nothing() {
    let (_dir, manager) = manager();
    let sorted = SortedRuns::new(Arc::clone(&manager), 3, ByFirst);
    let mut merge = sorted.merge(memory(&manager, usize::MAX), 100).unwrap();
    assert_eq!(merge.next_batch().unwrap(), None);
}

/// A grant that holds five read buffers but not sixteen: the merge reads
/// fewer runs at once and takes more passes, gives the same rows, and what it
/// counts at each batch covers its read buffers, its grant within the limit
/// (a batch ends where the grant does, so its count may pass it by a row).
#[test]
fn a_tight_grant_narrows_the_fan_in_and_holds() {
    let (dir, manager) = manager();
    let rows = input(300 * 8);
    let limit = 5 * MERGE_READ_BUFFER + (64 << 10);
    let mut merge = runs_of(&manager, &rows, 8)
        .merge(memory(&manager, limit), 100)
        .unwrap();
    assert!(
        merge.fan_in() >= 2 && merge.fan_in() < MERGE_FAN_IN,
        "fan-in {}",
        merge.fan_in()
    );
    assert_eq!(merge.passes(), passes_for(300, merge.fan_in()));
    assert!(merge.passes() > passes_for(300, MERGE_FAN_IN));
    let mut out = Vec::new();
    while let Some(batch) = merge.next_batch().unwrap() {
        let counted = merge.memory().bytes();
        assert!(
            counted >= merge.open_runs() * merge.read_buffer(),
            "{counted} counted"
        );
        assert!(
            merge.memory().granted() <= limit,
            "{} granted",
            merge.memory().granted()
        );
        out.extend(batch);
    }
    assert_eq!(out, in_memory(&rows));
    drop(merge);
    assert_eq!(files_left(&dir), 0);
}

/// The floor of a merge: what it holds at least for its runs. No runs need
/// nothing, one run one smallest read buffer and room for its row, two runs
/// two of those, and three or more a pass's write buffer as well. One byte
/// below each floor refuses with the query memory limit and leaves no run
/// behind; at the floor the merge gives the in-memory sort.
#[test]
fn a_merge_refuses_below_its_floor_and_answers_at_it() {
    // A run's smallest read buffer and its next row, and the batch's first row.
    let unit = 2 * MERGE_MIN_READ_BUFFER;
    let first = MERGE_MIN_READ_BUFFER;
    for (runs, floor) in [
        (0, 0),
        (1, unit + first),
        (2, 2 * unit + first),
        (3, 2 * unit + first + MERGE_READ_BUFFER),
    ] {
        let rows = input(runs * 8);
        if floor > 0 {
            let (dir, manager) = manager();
            let refused = runs_of(&manager, &rows, 8).merge(memory(&manager, floor - 1), 100);
            let error = refused.err().expect("refused");
            assert!(
                error.to_string().contains("query memory limit"),
                "{runs} runs: {error}"
            );
            assert_eq!(files_left(&dir), 0, "{runs} runs");
        }
        let (dir, manager) = manager();
        let mut merge = runs_of(&manager, &rows, 8)
            .merge(memory(&manager, floor.max(1)), 100)
            .unwrap_or_else(|error| panic!("{runs} runs at {floor}: {error}"));
        assert_eq!(
            drain(&mut merge, 100).unwrap(),
            in_memory(&rows),
            "{runs} runs"
        );
        drop(merge);
        assert_eq!(files_left(&dir), 0, "{runs} runs");
    }
}

/// The floor holds rows up to the size a plan keeps room for: two runs of
/// rows of about 3.5 KiB are refused when the merge is planned below it, not
/// partway through, and every row comes out at it.
#[test]
fn rows_near_the_planned_size_are_refused_up_front_or_merged() {
    let rows: Vec<Vec<Value>> = (0..16_i64)
        .map(|place| {
            vec![
                Value::Int64(place * 7 % 16),
                Value::Int64(place),
                Value::from("x".repeat(3500)),
            ]
        })
        .collect();
    let floor = 2 * 2 * MERGE_MIN_READ_BUFFER + MERGE_MIN_READ_BUFFER;
    let (dir, spill) = manager();
    let refused =
        runs_of(&spill, &rows, 8).merge(memory(&spill, floor - MERGE_MIN_READ_BUFFER), 100);
    let error = refused.err().expect("refused when planned");
    assert!(error.to_string().contains("query memory limit"), "{error}");
    assert_eq!(files_left(&dir), 0);
    let (dir, manager) = manager();
    let mut merge = runs_of(&manager, &rows, 8)
        .merge(memory(&manager, floor), 100)
        .unwrap_or_else(|error| panic!("at the floor: {error}"));
    assert_eq!(drain(&mut merge, 100).unwrap(), in_memory(&rows));
    drop(merge);
    assert_eq!(files_left(&dir), 0);
}

/// A grant that holds a pass's write buffer but not two full read buffers
/// beside it: the merge reads two runs at a time through smaller buffers, and
/// gives the same rows.
#[test]
fn a_grant_short_of_two_full_buffers_shrinks_them() {
    let (dir, manager) = manager();
    let rows = input(300 * 8);
    let limit = MERGE_READ_BUFFER + MERGE_READ_BUFFER / 2;
    let mut merge = runs_of(&manager, &rows, 8)
        .merge(memory(&manager, limit), 100)
        .unwrap();
    assert_eq!(merge.fan_in(), 2);
    assert!(
        merge.read_buffer() < MERGE_READ_BUFFER && merge.read_buffer() >= MERGE_MIN_READ_BUFFER,
        "{}",
        merge.read_buffer()
    );
    assert_eq!(merge.passes(), passes_for(300, 2));
    assert_eq!(drain(&mut merge, 100).unwrap(), in_memory(&rows));
    drop(merge);
    assert_eq!(files_left(&dir), 0);
}

/// A batch ends where the grant does: with room for the buffers, their next
/// rows and a batch's first row, and a few rows more, the merge hands out
/// short batches, and every row in order.
#[test]
fn a_batch_ends_at_the_grant() {
    let (_dir, manager) = manager();
    let rows = input(2 * 500);
    let limit = 2 * MERGE_READ_BUFFER + 8 * 1024 + MERGE_MIN_READ_BUFFER;
    let mut merge = runs_of(&manager, &rows, 500)
        .merge(memory(&manager, limit), 10_000)
        .unwrap();
    assert_eq!(merge.read_buffer(), MERGE_READ_BUFFER);
    let mut out = Vec::new();
    let mut batches = 0;
    while let Some(batch) = merge.next_batch().unwrap() {
        assert!(!batch.is_empty() && batch.len() < 10_000, "{}", batch.len());
        batches += 1;
        out.extend(batch);
    }
    assert!(batches > 2, "{batches} batches");
    assert_eq!(out, in_memory(&rows));
}

/// A merge dropped before its end (the query was cancelled, or failed
/// downstream), in its last pass and with intermediate runs written, leaves
/// no file; so do runs that never merge.
#[test]
fn a_merge_dropped_midway_leaves_no_file() {
    for runs in [3, 40] {
        let (dir, manager) = manager();
        let mut merge = runs_of(&manager, &input(runs * 8), 8)
            .merge(memory(&manager, usize::MAX), 10)
            .unwrap();
        assert!(merge.next_batch().unwrap().is_some());
        assert!(files_left(&dir) > 0, "{runs} runs");
        drop(merge);
        assert_eq!(files_left(&dir), 0, "{runs} runs");
    }
    let (dir, manager) = manager();
    let sorted = runs_of(&manager, &input(24), 8);
    assert_eq!(files_left(&dir), 3);
    drop(sorted);
    assert_eq!(files_left(&dir), 0);
}

/// Between batches the merge counts only its open runs, and its grant holds
/// about that: the room a batch took goes back for the operators after it,
/// which take the rows.
#[test]
fn a_merge_between_batches_holds_only_its_open_runs() {
    // Under a 1 MiB cap the grants grow by steps of at most 16 KiB; a batch of
    // 1,000 rows takes far more.
    let (_dir, manager) = manager();
    let ctx = context(&manager, 1 << 20);
    let rows = input(4000);
    let mut merge = runs_of(&manager, &rows, 1000)
        .merge(OperatorMemory::new(ctx.clone()), 1000)
        .unwrap();
    while let Some(batch) = merge.next_batch().unwrap() {
        // The plan's room: the open runs' read buffers and a next row each,
        // and the next batch's first row; rounded by the grant rule.
        let planned = merge.open_runs() * (merge.read_buffer() + MERGE_MIN_READ_BUFFER)
            + MERGE_MIN_READ_BUFFER;
        let granted = merge.memory().granted();
        // It never gives back its open runs' read buffers.
        assert!(
            granted >= merge.open_runs() * merge.read_buffer(),
            "{granted}"
        );
        assert!(
            granted < planned + (16 << 10),
            "{granted} granted for a plan of {planned} after a batch of {}",
            batch.len()
        );
        // What it gave back left the statement's memory too.
        assert_eq!(ctx.query().used(), granted);
    }
}

/// A merge whose rows grow (short rows, then rows of about 3 KiB, under the
/// 4 KiB a plan keeps for each) hands out every row while another operator of
/// the statement takes all the memory the merge gives back between batches:
/// the trim keeps the room the plan promised. Sixteen runs take keys in turn,
/// so their heads grow together, by more than a grant step (32 KiB here).
#[test]
fn a_merge_keeps_its_planned_room_while_others_fill_the_cap() {
    let (_dir, manager) = manager();
    let limit = 2 << 20;
    let ctx = context(&manager, limit);
    // Run `r` holds the keys `j * 16 + r`: short rows first, then long ones.
    let rows: Vec<Vec<Value>> = (0..16_i64)
        .flat_map(|run| {
            (0..40_i64).map(move |j| {
                let size = if j < 20 { 8 } else { 3000 };
                vec![
                    Value::Int64(j * 16 + run),
                    Value::Int64(j * 16 + run),
                    Value::from("x".repeat(size)),
                ]
            })
        })
        .collect();
    let mut merge = runs_of(&manager, &rows, 40)
        .merge(OperatorMemory::new(ctx.clone()), 50)
        .unwrap();
    assert_eq!(merge.open_runs(), 16, "one pass");
    let mut other = OperatorMemory::new(ctx.clone());
    let mut got = Vec::new();
    while let Some(batch) = merge
        .next_batch()
        .unwrap_or_else(|error| panic!("after {} rows: {error}", got.len()))
    {
        got.extend(batch);
        // The rest of the cap, taken by the operator after the merge.
        other
            .set(other.bytes() + (limit - ctx.query().used()))
            .unwrap();
    }
    assert_eq!(got, in_memory(&rows));
}

/// The merge gives back what it counts once it ends.
#[test]
fn a_finished_merge_gives_back_its_grant() {
    let (_dir, manager) = manager();
    let rows = input(17 * 8);
    let mut merge = runs_of(&manager, &rows, 8)
        .merge(memory(&manager, 4 << 20), 100)
        .unwrap();
    drain(&mut merge, 100).unwrap();
    assert_eq!(merge.memory().granted(), 0);
}
