//! The merge of a spilled sort's runs: sorted runs on disk, read back a few
//! at a time and handed out in batches, within the operator's grant.
//!
//! A sort that spills writes its buffer as a sorted run each time it is
//! full, in input order, so run `i` holds rows that came before run `i + 1`'s.
//! The merge reads up to [`MERGE_FAN_IN`] runs at once. When there are more,
//! it merges groups of consecutive runs into longer runs first (a pass), so
//! the runs keep their input order. A tie goes to the earlier run, and within
//! a run to the earlier row, which is the order a stable in-memory sort
//! gives: a spilled sort returns exactly the rows, in exactly the order, of
//! the same sort in memory.
//!
//! Each open run holds a read buffer of [`MERGE_READ_BUFFER`] bytes and its
//! next row, and a pass writes through a buffer of as many. The merge counts
//! these, and the batch it builds, against the operator's grant, planning
//! room for a row of up to [`MERGE_MIN_READ_BUFFER`] bytes per open run, and
//! one more for a batch's first row.
//! When the grant cannot hold a full fan-in's buffers, the merge reads fewer
//! runs at once and takes more passes; at two runs, smaller read buffers, down
//! to [`MERGE_MIN_READ_BUFFER`]; short of that it refuses with the bound that
//! refused (`GRAFEO-S001`). A batch ends early where the grant does, and
//! between batches the merge gives back the room a batch took, for the
//! operators after it; when not even a batch's first row fits again, the
//! merge refuses, naming itself. The
//! plan is made once: a run's row past [`MERGE_MIN_READ_BUFFER`] bytes (a
//! long literal, which a SPARQL run holds as a value and as its key) counts
//! as it is, and under a tight grant can refuse where a smaller plan would
//! have fit.
//!
//! The runs are files the merge owns: each is deleted when it is merged or
//! when the merge drops, after an error or a cancel too.

use std::cmp::Ordering;
use std::io::{Read, Write};
use std::sync::Arc;

use grafeo_common::types::Value;

use super::file::{SpillFile, SpillFileReader};
use super::manager::SpillManager;
use super::serializer::{deserialize_row, serialize_row};
use super::{SpillOwner, spill_error};
use crate::execution::memory::OperatorMemory;
use crate::execution::operators::OperatorError;

/// The most runs a merge reads at once.
pub const MERGE_FAN_IN: usize = 16;

/// The bytes an open run's read buffer holds (and a pass's write buffer).
pub const MERGE_READ_BUFFER: usize = super::file::BUFFER_SIZE;

/// The smallest read buffer a merge narrows to, and the room it plans for an
/// open run's next row.
pub const MERGE_MIN_READ_BUFFER: usize = 4 * 1024;

/// How a sort orders its rows: the key a row sorts by, made when the merge
/// reads the row back from its run.
pub trait RowOrder: Send {
    /// What rows are compared by.
    type Key;

    /// The key of `row`, a row as the sort spilled it.
    ///
    /// # Errors
    ///
    /// When the key cannot be made from the row.
    fn key(&mut self, row: &[Value]) -> Result<Self::Key, OperatorError>;

    /// How the rows of keys `a` and `b` order; `Equal` is a tie, which the
    /// merge breaks by input order.
    fn compare(&self, a: &Self::Key, b: &Self::Key) -> Ordering;

    /// The bytes `key` holds, counted in the grant.
    fn key_bytes(&self, key: &Self::Key) -> usize;
}

/// A sorted run on disk, deleted when it drops.
struct Run {
    file: Option<SpillFile>,
    rows: u64,
}

impl Run {
    fn file(&self) -> &SpillFile {
        self.file
            .as_ref()
            .expect("a run holds its file until it drops")
    }

    fn file_mut(&mut self) -> &mut SpillFile {
        self.file
            .as_mut()
            .expect("a run holds its file until it drops")
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            // A file that cannot be removed now stays with its manager, whose
            // cleanup removes it at the end of the query.
            drop(file.delete());
        }
    }
}

/// Writes a run, row by row.
struct RunWriter<'a>(&'a mut SpillFile);

impl Write for RunWriter<'_> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.write_all(data)?;
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Reads a run, row by row (the serializer reads whole values).
struct RunReader(SpillFileReader);

impl Read for RunReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read_exact(buf)?;
        Ok(buf.len())
    }
}

/// The bytes a row holds: its values' slots and what they hold on the heap.
fn row_bytes(row: &[Value]) -> usize {
    row.iter()
        .map(|value| std::mem::size_of::<Value>() + value.estimated_size_bytes())
        .sum()
}

/// A spilled sort's sorted runs, in input order: what [`merge`](Self::merge)
/// merges.
pub struct SortedRuns<O: RowOrder> {
    manager: Arc<SpillManager>,
    /// The columns of a row.
    columns: usize,
    order: O,
    runs: Vec<Run>,
}

impl<O: RowOrder> SortedRuns<O> {
    /// No runs yet, of rows of `columns` columns ordered by `order`, written
    /// through `manager`.
    #[must_use]
    pub fn new(manager: Arc<SpillManager>, columns: usize, order: O) -> Self {
        Self {
            manager,
            columns,
            order,
            runs: Vec::new(),
        }
    }

    /// The runs written so far.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Writes `rows`, already sorted by the order (ties in input order), as
    /// the next run. No rows: no run. The file's 64 KiB write buffer is not
    /// counted here: the caller writes what it holds and counted already, and
    /// gives that back after.
    ///
    /// # Errors
    ///
    /// When the run cannot be written, or the spill account refuses it.
    pub fn spill<I>(&mut self, rows: I) -> Result<(), OperatorError>
    where
        I: IntoIterator<Item = Vec<Value>>,
        I::IntoIter: ExactSizeIterator,
    {
        let rows = rows.into_iter();
        if rows.len() == 0 {
            return Ok(());
        }
        let mut run = new_run(&self.manager, rows.len() as u64)?;
        let file = run.file_mut();
        for row in rows {
            serialize_row(&row, &mut RunWriter(file)).map_err(spill_error)?;
        }
        file.finish_write().map_err(spill_error)?;
        self.runs.push(run);
        Ok(())
    }

    /// The merge of the runs, counted in `memory`, handing out batches of at
    /// most `batch_rows` rows. Past [`MERGE_FAN_IN`] runs, or past what
    /// `memory`'s grant holds, the passes that merge runs into longer ones
    /// run here.
    ///
    /// # Errors
    ///
    /// When the grant cannot hold two runs' buffers (`GRAFEO-S001` naming
    /// the bound), or a run cannot be read or written. The runs are deleted.
    pub fn merge(
        self,
        memory: OperatorMemory,
        batch_rows: usize,
    ) -> Result<RunMerge<O>, OperatorError> {
        let Self {
            manager,
            columns,
            order,
            runs,
        } = self;
        let mut merge = RunMerge {
            open: Vec::new(),
            runs,
            manager,
            columns,
            order,
            memory,
            batch_rows: batch_rows.max(1),
            fan_in: 0,
            read_buffer: MERGE_READ_BUFFER,
            passes: 0,
            rows: 0,
        };
        merge.rows = merge.runs.iter().map(|run| run.rows).sum();
        merge.fan_in = merge.plan_fan_in()?;
        while merge.runs.len() > merge.fan_in {
            merge.pass()?;
        }
        merge.open_last_pass()?;
        Ok(merge)
    }
}

/// A run file, created with its row count first.
fn new_run(manager: &SpillManager, rows: u64) -> Result<Run, OperatorError> {
    let file = manager
        .create_file(SpillOwner::Sort, "sort_run")
        .map_err(spill_error)?;
    let mut run = Run {
        file: Some(file),
        rows,
    };
    run.file_mut().write_u64_le(rows).map_err(spill_error)?;
    Ok(run)
}

/// An open run during a merge: its reader, its next row and that row's key.
struct OpenRun<K> {
    reader: RunReader,
    /// Rows still on disk after `head`.
    left: u64,
    /// The next row and its key; `None` once the run is read.
    head: Option<(K, Vec<Value>)>,
    /// What `head` holds, as counted.
    head_bytes: usize,
}

/// The merge of a spilled sort's runs (see the module docs). Its rows come
/// from [`next_batch`](Self::next_batch).
pub struct RunMerge<O: RowOrder> {
    /// The runs of the last pass being read, in run order. Declared before
    /// `runs`, so the readers close before their files are deleted.
    open: Vec<OpenRun<O::Key>>,
    /// The runs of the last pass (or of the next pass while one runs).
    runs: Vec<Run>,
    manager: Arc<SpillManager>,
    columns: usize,
    order: O,
    memory: OperatorMemory,
    batch_rows: usize,
    fan_in: usize,
    /// The bytes of an open run's read buffer.
    read_buffer: usize,
    passes: usize,
    /// The rows its runs held when it began.
    rows: u64,
}

impl<O: RowOrder> RunMerge<O> {
    /// The most runs it reads at once.
    #[must_use]
    pub fn fan_in(&self) -> usize {
        self.fan_in
    }

    /// The passes it takes, the last one included.
    #[must_use]
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// The rows the sort spilled: every row it merges.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The bytes of an open run's read buffer.
    #[must_use]
    pub fn read_buffer(&self) -> usize {
        self.read_buffer
    }

    /// The runs it reads now (each holds a read buffer).
    #[must_use]
    pub fn open_runs(&self) -> usize {
        self.open.len()
    }

    /// What it counts and holds.
    #[must_use]
    pub fn memory(&self) -> &OperatorMemory {
        &self.memory
    }

    /// The fan-in the grant holds: from the run count (at most
    /// [`MERGE_FAN_IN`]), halved while the grant cannot hold that many read
    /// buffers and a row each, and a write buffer when passes are needed;
    /// at two, the read buffers halve down to [`MERGE_MIN_READ_BUFFER`].
    fn plan_fan_in(&mut self) -> Result<usize, OperatorError> {
        let runs = self.runs.len();
        let mut fan_in = runs.clamp(2, MERGE_FAN_IN);
        loop {
            let writer = usize::from(runs > fan_in) * MERGE_READ_BUFFER;
            // Only the runs it opens at once: none or one when that is all.
            let readers = planned_room(runs.min(fan_in), self.read_buffer);
            match self.memory.set(readers + writer) {
                Ok(()) => return Ok(fan_in),
                Err(_) if fan_in > 2 => fan_in = (fan_in / 2).max(2),
                Err(_) if self.read_buffer > MERGE_MIN_READ_BUFFER => {
                    self.read_buffer = (self.read_buffer / 2).max(MERGE_MIN_READ_BUFFER);
                }
                Err(refusal) => return Err(refusal),
            }
        }
    }

    /// One pass: groups of consecutive runs, from the first, merge into one
    /// run each, until no more than `fan_in` runs are left or every run was
    /// read once. Runs past the last group stay as they are, and every run
    /// keeps its place in input order.
    fn pass(&mut self) -> Result<(), OperatorError> {
        // How many runs fewer the last pass needs.
        let mut excess = self.runs.len().saturating_sub(self.fan_in);
        let mut runs = std::mem::take(&mut self.runs).into_iter();
        while excess > 0 {
            // A group of `n` runs leaves `n - 1` fewer.
            let wanted = (excess + 1).min(self.fan_in);
            // Declared before its readers, so they close before it deletes
            // the group's files.
            let group: Vec<Run> = runs.by_ref().take(wanted).collect();
            if group.len() < 2 {
                self.runs.extend(group);
                break;
            }
            excess -= group.len() - 1;
            let mut open = group
                .iter()
                .map(|run| {
                    open_run(
                        &self.manager,
                        &mut self.order,
                        self.columns,
                        self.read_buffer,
                        run,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut out = new_run(&self.manager, group.iter().map(|run| run.rows).sum())?;
            let file = out.file_mut();
            while let Some(index) = first(&self.order, &open) {
                count(&mut self.memory, &open, self.read_buffer, 1, 0)?;
                let Some((_, row)) = open[index].head.take() else {
                    break;
                };
                serialize_row(&row, &mut RunWriter(file)).map_err(spill_error)?;
                advance(
                    &self.manager,
                    &mut self.order,
                    self.columns,
                    &mut open[index],
                )?;
            }
            file.finish_write().map_err(spill_error)?;
            drop(open);
            drop(group);
            self.runs.push(out);
        }
        self.runs.extend(runs);
        self.passes += 1;
        Ok(())
    }

    /// Opens the runs of the last pass.
    fn open_last_pass(&mut self) -> Result<(), OperatorError> {
        for run in &self.runs {
            let open = open_run(
                &self.manager,
                &mut self.order,
                self.columns,
                self.read_buffer,
                run,
            )?;
            self.open.push(open);
        }
        self.passes += 1;
        count(&mut self.memory, &self.open, self.read_buffer, 0, 0)
    }

    /// The next rows in order, at most the batch size; `None` at the end,
    /// when the runs are deleted and the grant given back.
    ///
    /// # Errors
    ///
    /// When a run cannot be read, or the batch's rows pass the grant
    /// (`GRAFEO-S001`).
    pub fn next_batch(&mut self) -> Result<Option<Vec<Vec<Value>>>, OperatorError> {
        // The open runs' buffers and rows must fit before a batch starts.
        count(&mut self.memory, &self.open, self.read_buffer, 0, 0)?;
        let mut batch = Vec::with_capacity(self.batch_rows.min(1024));
        let mut batch_bytes = 0;
        while batch.len() < self.batch_rows {
            let Some(index) = first(&self.order, &self.open) else {
                break;
            };
            let Some((_, row)) = self.open[index].head.take() else {
                break;
            };
            batch_bytes += row_bytes(&row);
            batch.push(row);
            advance(
                &self.manager,
                &mut self.order,
                self.columns,
                &mut self.open[index],
            )?;
            // Where the grant ends, so does the batch. When not even its
            // first row fits, the merge refuses: a one-row batch is never
            // handed on past its grant.
            if let Err(refusal) = count(
                &mut self.memory,
                &self.open,
                self.read_buffer,
                0,
                batch_bytes,
            ) {
                if batch.len() == 1 {
                    return Err(merge_refused(refusal));
                }
                break;
            }
        }
        if batch.is_empty() {
            self.open.clear();
            self.runs.clear();
            self.memory.release();
            return Ok(None);
        }
        // The rows go on to the operators after it, which count what they
        // keep: it counts its open runs again and gives the batch's room back,
        // keeping its plan's (a read buffer and a next row per open run, and
        // the next batch's first row), so a row the plan allows always fits.
        count(&mut self.memory, &self.open, self.read_buffer, 0, 0)?;
        self.memory
            .trim_to(planned_room(self.open.len(), self.read_buffer));
        Ok(Some(batch))
    }
}

/// The room a merge plans for, and keeps between batches, with `open` runs
/// open through read buffers of `read_buffer` bytes: each run's buffer and
/// next row, and the next batch's first row (rows of up to
/// [`MERGE_MIN_READ_BUFFER`] bytes); nothing for no runs.
fn planned_room(open: usize, read_buffer: usize) -> usize {
    if open == 0 {
        return 0;
    }
    open * (read_buffer + MERGE_MIN_READ_BUFFER) + MERGE_MIN_READ_BUFFER
}

/// `error` naming the merge as what asked, when it is a memory refusal.
fn merge_refused(error: OperatorError) -> OperatorError {
    use grafeo_common::utils::error::{Error, LimitRefusal};

    let OperatorError::Wrapped(inner) = &error else {
        return error;
    };
    let Some(&LimitRefusal::Memory {
        bound,
        requested,
        used,
        limit,
        ..
    }) = inner.limit_refusal()
    else {
        return error;
    };
    OperatorError::from(Error::from(LimitRefusal::Memory {
        bound,
        requested,
        used,
        limit,
        detail: Some("the merge of a spilled sort has no room for its next row".into()),
    }))
}

/// Opens `run` and reads its first row.
fn open_run<O: RowOrder>(
    manager: &SpillManager,
    order: &mut O,
    columns: usize,
    read_buffer: usize,
    run: &Run,
) -> Result<OpenRun<O::Key>, OperatorError> {
    let mut reader = RunReader(
        run.file()
            .reader_with_buffer(read_buffer)
            .map_err(spill_error)?,
    );
    let rows = reader.0.read_u64_le().map_err(spill_error)?;
    if rows != run.rows {
        return Err(spill_error(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "a sorted run's file claims {rows} rows, the sort wrote {}",
                run.rows
            ),
        )));
    }
    let mut open = OpenRun {
        reader,
        left: rows,
        head: None,
        head_bytes: 0,
    };
    advance(manager, order, columns, &mut open)?;
    Ok(open)
}

/// Reads `open`'s next row into its head, or empties the head at its end,
/// counting the row on `manager` (see [`SpillManager::rows_read_back`]).
fn advance<O: RowOrder>(
    manager: &SpillManager,
    order: &mut O,
    columns: usize,
    open: &mut OpenRun<O::Key>,
) -> Result<(), OperatorError> {
    if open.left == 0 {
        open.head = None;
        open.head_bytes = 0;
        return Ok(());
    }
    open.left -= 1;
    let row = deserialize_row(&mut open.reader, columns).map_err(spill_error)?;
    manager.note_row_read_back();
    let key = order.key(&row)?;
    open.head_bytes = row_bytes(&row) + order.key_bytes(&key);
    open.head = Some((key, row));
    Ok(())
}

/// The open run whose head comes first, the earlier run on a tie.
fn first<O: RowOrder>(order: &O, open: &[OpenRun<O::Key>]) -> Option<usize> {
    let mut first: Option<(usize, &O::Key)> = None;
    for (index, run) in open.iter().enumerate() {
        if let Some((key, _)) = &run.head
            && first.is_none_or(|(_, best)| order.compare(key, best) == Ordering::Less)
        {
            first = Some((index, key));
        }
    }
    first.map(|(index, _)| index)
}

/// Counts the open runs' read buffers of `read_buffer` bytes and their next
/// rows, `writers` write buffers and `rows` bytes of rows more.
fn count<K>(
    memory: &mut OperatorMemory,
    open: &[OpenRun<K>],
    read_buffer: usize,
    writers: usize,
    rows: usize,
) -> Result<(), OperatorError> {
    let heads: usize = open.iter().map(|run| run.head_bytes).sum();
    memory.set(open.len() * read_buffer + writers * MERGE_READ_BUFFER + heads + rows)
}
