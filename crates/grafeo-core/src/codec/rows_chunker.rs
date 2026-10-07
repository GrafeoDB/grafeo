//! Cuts the rows of one row group of a table into column chunks.
//!
//! A table section (the LPG node and edge tables, the RDF triples) writes
//! each row group through a [`RowsChunker`]: the rows come in ascending, one
//! cell per column, and every chunk the chunker cuts holds the same rows in
//! each of its columns.

use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, SectionSink};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashSet;

use crate::codec::column_chunk::{chunk_overhead, encode_column_chunk, value_bound};

/// One column a [`RowsChunker`] writes: its chunk kind
/// ([`ChunkKind::Column`] or [`ChunkKind::History`]) and column id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkColumn {
    /// [`ChunkKind::Column`] for the current values, [`ChunkKind::History`]
    /// for the older versions of a column.
    pub kind: ChunkKind,
    /// The column's id in its table.
    pub column_id: u32,
}

/// What one column holds in the open chunk.
#[derive(Debug, Default)]
struct OpenCells {
    /// The values, at their offset from the chunk's first row, ascending.
    values: Vec<(u32, Value)>,
    /// One epoch per value.
    epochs: Vec<u64>,
    /// Whether one of `epochs` is not 0.
    versioned: bool,
    /// The sum of the values' [`value_bound`]s.
    bound: usize,
}

impl OpenCells {
    fn clear(&mut self) {
        self.values.clear();
        self.epochs.clear();
        self.versioned = false;
        self.bound = 0;
    }
}

/// Writes the column chunks of one row group,
/// `[group_start, group_start + caps.max_rows)`, of one table.
///
/// The columns share their chunk boundaries; a chunk is cut before a row that
/// would make any column's bound
/// ([`chunk_overhead`] plus the [`value_bound`] of each value) exceed
/// `caps.max_bytes` (a chunk always takes its first row, so a value larger
/// than the cap gets a chunk of its own), and each cut writes one chunk per
/// column that has a value in it, in the order of `columns`. A chunk's rows
/// run from the first row pushed into it to the last; rows never pushed
/// between two chunks belong to neither. A `Column` chunk carries epochs when
/// one of its values has an epoch other than 0; a `History` chunk never does
/// (its values hold their versions' epochs themselves).
///
/// The chunker holds one open chunk per column. [`finish`](Self::finish)
/// writes the last one: a chunker dropped without it loses those rows. After
/// an error the chunker's state is unspecified; drop it.
#[must_use = "call finish to write the open chunk"]
#[derive(Debug)]
pub struct RowsChunker {
    graph_id: u32,
    columns: Vec<ChunkColumn>,
    /// The first column of `columns` listed a second time, which
    /// [`push`](Self::push) refuses: its chunks would share an identity.
    repeated: Option<ChunkColumn>,
    caps: ChunkCaps,
    group_start: u64,
    /// The first row of the open chunk, once a row was pushed.
    chunk_start: u64,
    /// The last row pushed, `None` before the first.
    last_row: Option<u64>,
    /// The open chunk, one entry per column.
    cells: Vec<OpenCells>,
}

impl RowsChunker {
    /// A chunker of the row group starting at `group_start` of graph
    /// `graph_id`, writing `columns`.
    ///
    /// A column listed twice (one kind and column id) is refused by the
    /// first [`push`](Self::push): its chunks would share an identity.
    pub fn new(
        graph_id: u32,
        columns: Vec<ChunkColumn>,
        group_start: u64,
        caps: ChunkCaps,
    ) -> Self {
        let cells = columns.iter().map(|_| OpenCells::default()).collect();
        let mut seen = FxHashSet::default();
        let repeated = columns
            .iter()
            .find(|column| !seen.insert((column.kind.to_byte(), column.column_id)))
            .copied();
        Self {
            graph_id,
            columns,
            repeated,
            caps,
            group_start,
            chunk_start: group_start,
            last_row: None,
            cells,
        }
    }

    /// Adds row `row` with one cell per column (`None`: no value there). Rows
    /// ascend within the group.
    ///
    /// A `History` cell's epoch is 0: its value holds the epochs of its
    /// versions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] for caps that [`ChunkCaps::validate`]
    /// refuses, and [`Error::Internal`] naming the graph, column and row for
    /// a column listed twice, a row outside the group or not after the last
    /// one, a cell count other than the column count, a column of another
    /// kind than `Column` or `History`, or a `History` cell with an epoch; a
    /// refused row changes nothing. Returns the encoder's or the sink's error
    /// when the row cuts the open chunk.
    pub fn push(
        &mut self,
        sink: &mut dyn SectionSink,
        row: u64,
        cells: Vec<Option<(Value, u64)>>,
    ) -> Result<()> {
        self.check_row(row, &cells)?;
        let bounds: Vec<usize> = cells
            .iter()
            .map(|cell| cell.as_ref().map_or(0, |(value, _)| value_bound(value)))
            .collect();
        let cut = match self.last_row {
            Some(_) => self.passes_cap(row, &cells, &bounds)?,
            None => true,
        };
        if cut {
            self.write_open_chunk(sink)?;
            self.chunk_start = row;
        }
        let offset = self.offset(row)?;
        for ((open, cell), bound) in self.cells.iter_mut().zip(cells).zip(bounds) {
            if let Some((value, epoch)) = cell {
                open.values.push((offset, value));
                open.epochs.push(epoch);
                open.versioned |= epoch != 0;
                open.bound = open.bound.saturating_add(bound);
            }
        }
        self.last_row = Some(row);
        Ok(())
    }

    /// Writes the open chunk.
    ///
    /// # Errors
    ///
    /// Returns the encoder's or the sink's error.
    pub fn finish(mut self, sink: &mut dyn SectionSink) -> Result<()> {
        self.write_open_chunk(sink)
    }

    /// Refuses a row [`push`](Self::push) must not take, before anything
    /// changes.
    fn check_row(&self, row: u64, cells: &[Option<(Value, u64)>]) -> Result<()> {
        self.caps.validate()?;
        let graph = self.graph_id;
        if let Some(column) = self.repeated {
            return Err(Error::Internal(format!(
                "graph {graph}: {:?} column {} is listed twice, so two of its chunks would \
                 share an identity",
                column.kind, column.column_id
            )));
        }
        if cells.len() != self.columns.len() {
            return Err(Error::Internal(format!(
                "graph {graph}, row {row}: expected one cell per column ({}), got {}",
                self.columns.len(),
                cells.len()
            )));
        }
        let in_group = row
            .checked_sub(self.group_start)
            .is_some_and(|offset| offset < u64::from(self.caps.max_rows));
        if !in_group {
            return Err(Error::Internal(format!(
                "graph {graph}: row {row} is outside the row group of {} rows from row {}",
                self.caps.max_rows, self.group_start
            )));
        }
        if let Some(last) = self.last_row
            && row <= last
        {
            return Err(Error::Internal(format!(
                "graph {graph}: row {row} does not come after row {last}"
            )));
        }
        for (column, cell) in self.columns.iter().zip(cells) {
            let column_id = column.column_id;
            match column.kind {
                ChunkKind::Column => {}
                ChunkKind::History => {
                    if let Some((_, epoch)) = cell
                        && *epoch != 0
                    {
                        return Err(Error::Internal(format!(
                            "graph {graph}, history column {column_id}, row {row}: epoch {epoch} \
                             on a history value, which holds its versions' epochs itself"
                        )));
                    }
                }
                other => {
                    return Err(Error::Internal(format!(
                        "graph {graph}, column {column_id}: a row chunker writes Column and \
                         History chunks, not {other:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Whether adding `cells` (whose values have `bounds`) at `row` makes a
    /// column's bound exceed `max_bytes`.
    fn passes_cap(
        &self,
        row: u64,
        cells: &[Option<(Value, u64)>],
        bounds: &[usize],
    ) -> Result<bool> {
        let rows_after = self.row_count(row)?;
        // A cap past `usize` cannot be passed.
        let max_bytes = usize::try_from(self.caps.max_bytes).unwrap_or(usize::MAX);
        for (((column, open), cell), bound) in
            self.columns.iter().zip(&self.cells).zip(cells).zip(bounds)
        {
            let (values_after, versioned_after) = match cell {
                Some((_, epoch)) => (open.values.len() + 1, open.versioned || *epoch != 0),
                None => (open.values.len(), open.versioned),
            };
            if values_after == 0 {
                // Writes nothing, whatever its overhead.
                continue;
            }
            let with_epochs = column.kind == ChunkKind::Column && versioned_after;
            let size = chunk_overhead(rows_after, values_after, with_epochs)
                .saturating_add(open.bound)
                .saturating_add(*bound);
            if size > max_bytes {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Writes one chunk per column that has a value in the open chunk, in
    /// the order of `columns`, and empties them.
    fn write_open_chunk(&mut self, sink: &mut dyn SectionSink) -> Result<()> {
        let Some(last_row) = self.last_row else {
            return Ok(());
        };
        let row_count = self.row_count(last_row)?;
        let (graph_id, row_start) = (self.graph_id, self.chunk_start);
        for (column, open) in self.columns.iter().zip(&mut self.cells) {
            if open.values.is_empty() {
                continue;
            }
            let epochs = (column.kind == ChunkKind::Column && open.versioned)
                .then_some(open.epochs.as_slice());
            let (codec, bytes) = encode_column_chunk(row_count, &open.values, epochs)?;
            let (column_id, codec) = (column.column_id, codec.to_byte());
            let meta = if column.kind == ChunkKind::History {
                ChunkMeta::history(graph_id, column_id, row_start, row_count, codec)
            } else {
                ChunkMeta::column(graph_id, column_id, row_start, row_count, codec)
            };
            sink.write_chunk(meta, &bytes)?;
            open.clear();
        }
        Ok(())
    }

    /// The offset of `row` from the open chunk's first row.
    fn offset(&self, row: u64) -> Result<u32> {
        row.checked_sub(self.chunk_start)
            .and_then(|offset| u32::try_from(offset).ok())
            .ok_or_else(|| {
                Error::Internal(format!(
                    "graph {}: row {row} lies outside the chunk from row {}",
                    self.graph_id, self.chunk_start
                ))
            })
    }

    /// The rows of the open chunk if it ended at `row`.
    fn row_count(&self, row: u64) -> Result<u32> {
        // `check_row` keeps `row` below `group_start + max_rows`, so the
        // count is at most `max_rows`, a u32.
        self.offset(row)?.checked_add(1).ok_or_else(|| {
            Error::Internal(format!(
                "graph {}: the chunk from row {} to row {row} holds more than u32::MAX rows",
                self.graph_id, self.chunk_start
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, SectionSink};
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::{Error, Result};

    use super::{ChunkColumn, RowsChunker};
    use crate::codec::column_chunk::{
        ColumnChunk, chunk_overhead, decode_column_chunk, value_bound,
    };

    /// Keeps every chunk written, in order.
    #[derive(Default)]
    struct Recorder(Vec<(ChunkMeta, Vec<u8>)>);

    impl SectionSink for Recorder {
        fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
            self.0.push((meta, bytes.to_vec()));
            Ok(())
        }
    }

    /// Refuses every chunk.
    struct Refusing;

    impl SectionSink for Refusing {
        fn write_chunk(&mut self, _meta: ChunkMeta, _bytes: &[u8]) -> Result<()> {
            Err(Error::Internal("Jules refuses the chunk".into()))
        }
    }

    fn column(column_id: u32) -> ChunkColumn {
        ChunkColumn {
            kind: ChunkKind::Column,
            column_id,
        }
    }

    fn history(column_id: u32) -> ChunkColumn {
        ChunkColumn {
            kind: ChunkKind::History,
            column_id,
        }
    }

    /// One `Int64` cell with epoch 0.
    fn int(value: i64) -> Option<(Value, u64)> {
        Some((Value::Int64(value), 0))
    }

    fn decode(meta: &ChunkMeta, bytes: &[u8]) -> ColumnChunk {
        decode_column_chunk(bytes, meta.codec, meta.row_count).unwrap()
    }

    /// The first row and row count of every chunk, in order.
    fn ranges(sink: &Recorder) -> Vec<(u64, u32)> {
        sink.0
            .iter()
            .map(|(meta, _)| (meta.row_start, meta.row_count))
            .collect()
    }

    /// Kind, column, first row and row count of every chunk, in order.
    fn written(sink: &Recorder) -> Vec<(ChunkKind, u32, u64, u32)> {
        sink.0
            .iter()
            .map(|(meta, _)| (meta.kind, meta.column_id, meta.row_start, meta.row_count))
            .collect()
    }

    /// Every value the `kind` chunks of `column_id` hold, at its row in the
    /// table, with its epoch (0 when the chunk carries none).
    fn cells_of(sink: &Recorder, kind: ChunkKind, column_id: u32) -> Vec<(u64, Value, u64)> {
        let mut cells = Vec::new();
        for (meta, bytes) in &sink.0 {
            if meta.kind != kind || meta.column_id != column_id {
                continue;
            }
            let chunk = decode(meta, bytes);
            for (index, (offset, value)) in chunk.values.into_iter().enumerate() {
                let epoch = chunk.epochs.as_ref().map_or(0, |epochs| epochs[index]);
                cells.push((meta.row_start + u64::from(offset), value, epoch));
            }
        }
        cells
    }

    /// `bytes` as a byte cap.
    fn cap(bytes: usize) -> u32 {
        u32::try_from(bytes).unwrap()
    }

    #[test]
    fn a_chunk_never_exceeds_the_byte_cap_unless_it_holds_one_value() {
        let caps = ChunkCaps {
            max_rows: 64,
            max_bytes: 300,
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
        let mut pushed = Vec::new();
        for row in 0..64u64 {
            let index = usize::try_from(row).unwrap();
            let name = ["Alix", "Gus", "Vincent", "Mia", "Jules"][index % 5].repeat(1 + index % 7);
            pushed.push((row, Value::from(name.as_str()), 0));
            chunker
                .push(&mut sink, row, vec![Some((Value::from(name), 0))])
                .unwrap();
        }
        chunker.finish(&mut sink).unwrap();
        assert!(sink.0.len() > 1, "the cap cut the group");
        for (meta, bytes) in &sink.0 {
            let chunk = decode(meta, bytes);
            assert!(
                bytes.len() <= 300 || chunk.values.len() == 1,
                "{} bytes, {} values",
                bytes.len(),
                chunk.values.len()
            );
        }
        // the chunks cover the 64 rows exactly once, in order
        let mut next = 0;
        for (row_start, row_count) in ranges(&sink) {
            assert_eq!(
                row_start,
                next,
                "chunks follow each other: {:?}",
                ranges(&sink)
            );
            next += u64::from(row_count);
        }
        assert_eq!(next, 64, "the last chunk ends at the last row");
        assert_eq!(
            cells_of(&sink, ChunkKind::Column, 16),
            pushed,
            "every value comes back at its row"
        );
    }

    /// The bound the chunker keeps for `cells` in a chunk of `rows` rows.
    fn bound_of(cells: &[(u64, Value, u64)], rows: u32) -> usize {
        let with_epochs = cells.iter().any(|(_, _, epoch)| *epoch != 0);
        chunk_overhead(rows, cells.len(), with_epochs)
            + cells
                .iter()
                .map(|(_, value, _)| value_bound(value))
                .sum::<usize>()
    }

    #[test]
    fn a_chunk_is_cut_only_when_its_next_row_would_pass_the_cap() {
        // The first rows carry epochs, the later ones do not: each chunk
        // counts only its own values and epochs.
        let caps = ChunkCaps {
            max_rows: 64,
            max_bytes: 300,
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
        let mut pushed = Vec::new();
        for row in 0..64u64 {
            let index = usize::try_from(row).unwrap();
            let name = Value::from(
                ["Alix", "Gus", "Vincent", "Mia", "Jules"][index % 5].repeat(1 + index % 7),
            );
            let epoch = if row < 8 { 19 } else { 0 };
            pushed.push((row, name.clone(), epoch));
            chunker
                .push(&mut sink, row, vec![Some((name, epoch))])
                .unwrap();
        }
        chunker.finish(&mut sink).unwrap();
        assert_eq!(cells_of(&sink, ChunkKind::Column, 16), pushed);
        assert!(sink.0.len() > 2, "{:?}", ranges(&sink));
        for (start, rows) in ranges(&sink) {
            let start = usize::try_from(start).unwrap();
            let end = start + usize::try_from(rows).unwrap();
            let held = &pushed[start..end];
            assert!(
                held.len() == 1 || bound_of(held, rows) <= 300,
                "rows {start} to {end} pass the cap"
            );
            if end < pushed.len() {
                assert!(
                    bound_of(&pushed[start..=end], rows + 1) > 300,
                    "row {end} fits the chunk from row {start}"
                );
            }
        }
    }

    #[test]
    fn a_value_larger_than_the_cap_gets_a_chunk_of_its_own() {
        let caps = ChunkCaps {
            max_rows: 8,
            max_bytes: 256,
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(3, vec![column(16)], 8, caps);
        chunker
            .push(&mut sink, 8, vec![Some((Value::from("Berlin"), 0))])
            .unwrap();
        chunker
            .push(
                &mut sink,
                9,
                vec![Some((Value::from("Prague".repeat(100)), 0))],
            )
            .unwrap();
        chunker
            .push(&mut sink, 10, vec![Some((Value::from("Paris"), 0))])
            .unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(ranges(&sink), [(8, 1), (9, 1), (10, 1)]);
        assert!(
            sink.0
                .iter()
                .all(|(meta, _)| meta.graph_id == 3 && meta.kind == ChunkKind::Column),
            "{:?}",
            written(&sink)
        );
        assert_eq!(
            cells_of(&sink, ChunkKind::Column, 16)[1],
            (9, Value::from("Prague".repeat(100)), 0),
            "the large value is written whole"
        );
    }

    /// A column with no value in the open chunk still cuts it when its next
    /// value alone passes the cap.
    #[test]
    fn a_value_past_the_cap_cuts_even_in_a_column_empty_so_far() {
        let caps = ChunkCaps {
            max_rows: 8,
            max_bytes: 256,
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(1), column(2)], 0, caps);
        chunker.push(&mut sink, 0, vec![int(3), None]).unwrap();
        let prague = Value::from("Prague".repeat(100));
        chunker
            .push(&mut sink, 1, vec![None, Some((prague.clone(), 0))])
            .unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(ranges(&sink), [(0, 1), (1, 1)]);
        assert_eq!(
            written(&sink),
            [(ChunkKind::Column, 1, 0, 1), (ChunkKind::Column, 2, 1, 1)]
        );
        assert_eq!(cells_of(&sink, ChunkKind::Column, 2), [(1, prague, 0)]);
    }

    #[test]
    fn columns_share_chunk_boundaries_and_keep_their_order() {
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 1 << 20,
        };
        let mut sink = Recorder::default();
        let edge = vec![column(1), column(2), column(3)];
        let mut chunker = RowsChunker::new(0, edge, 0, caps);
        for row in [0u64, 2] {
            chunker
                .push(&mut sink, row, vec![int(3), int(19), int(0)])
                .unwrap();
        }
        chunker.finish(&mut sink).unwrap();
        let written: Vec<(u32, u64, u32)> = sink
            .0
            .iter()
            .map(|(meta, _)| (meta.column_id, meta.row_start, meta.row_count))
            .collect();
        assert_eq!(written, [(1, 0, 3), (2, 0, 3), (3, 0, 3)]);
        assert_eq!(
            cells_of(&sink, ChunkKind::Column, 2),
            [(0, Value::Int64(19), 0), (2, Value::Int64(19), 0)]
        );
    }

    #[test]
    fn a_cut_for_one_column_cuts_every_column() {
        // The Int64 column alone would fit four rows in one chunk; the
        // string column fits one row per chunk, so both cut at every row.
        let long = Value::from("Prague".repeat(10));
        let max_bytes = chunk_overhead(4, 4, false) + 4 * value_bound(&Value::Int64(3));
        assert!(chunk_overhead(2, 2, false) + 2 * value_bound(&long) > max_bytes);
        let caps = ChunkCaps {
            max_rows: 8,
            max_bytes: cap(max_bytes),
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(1), column(2)], 0, caps);
        for row in 0..4u64 {
            chunker
                .push(&mut sink, row, vec![int(3), Some((long.clone(), 0))])
                .unwrap();
        }
        chunker.finish(&mut sink).unwrap();
        let expected: Vec<(ChunkKind, u32, u64, u32)> = (0..4u64)
            .flat_map(|row| {
                [
                    (ChunkKind::Column, 1, row, 1),
                    (ChunkKind::Column, 2, row, 1),
                ]
            })
            .collect();
        assert_eq!(written(&sink), expected);
    }

    #[test]
    fn epochs_count_toward_the_cap() {
        // Two values without epochs fit the cap exactly; an epoch on one of
        // them makes the chunk's bound pass it.
        let max_bytes = chunk_overhead(2, 2, false) + 2 * value_bound(&Value::Int64(3));
        let caps = ChunkCaps {
            max_rows: 4,
            max_bytes: cap(max_bytes),
        };
        let write = |epochs: [u64; 2]| {
            let mut sink = Recorder::default();
            let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
            for (row, epoch) in (0u64..).zip(epochs) {
                chunker
                    .push(&mut sink, row, vec![Some((Value::Int64(3), epoch))])
                    .unwrap();
            }
            chunker.finish(&mut sink).unwrap();
            sink
        };
        let plain = write([0, 0]);
        assert_eq!(ranges(&plain), [(0, 2)], "a bound equal to the cap fits");
        assert_eq!(decode(&plain.0[0].0, &plain.0[0].1).epochs, None);
        let versioned = write([0, 19]);
        assert_eq!(
            ranges(&versioned),
            [(0, 1), (1, 1)],
            "the epochs pass the cap"
        );
        assert_eq!(
            cells_of(&versioned, ChunkKind::Column, 16),
            [(0, Value::Int64(3), 0), (1, Value::Int64(3), 19)]
        );
    }

    #[test]
    fn rows_without_a_value_count_toward_the_cap() {
        // Two values 64 rows apart fit the cap exactly; one row further, the
        // presence bitmap needs another word.
        let max_bytes = chunk_overhead(64, 2, false) + 2 * value_bound(&Value::Int64(3));
        assert!(chunk_overhead(65, 2, false) > chunk_overhead(64, 2, false));
        let caps = ChunkCaps {
            max_rows: 128,
            max_bytes: cap(max_bytes),
        };
        let write = |last: u64| {
            let mut sink = Recorder::default();
            let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
            chunker.push(&mut sink, 0, vec![int(3)]).unwrap();
            chunker.push(&mut sink, last, vec![int(88)]).unwrap();
            chunker.finish(&mut sink).unwrap();
            ranges(&sink)
        };
        assert_eq!(write(63), [(0, 64)]);
        assert_eq!(write(64), [(0, 1), (64, 1)]);
        // A column without a value in the new row still counts its rows:
        // column 1 passes the cap at row 64, column 2 alone would fit.
        assert!(
            chunk_overhead(65, 1, false) + value_bound(&Value::Int64(88)) <= max_bytes,
            "column 2 alone fits"
        );
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(1), column(2)], 0, caps);
        chunker.push(&mut sink, 0, vec![int(3), None]).unwrap();
        chunker.push(&mut sink, 1, vec![int(19), None]).unwrap();
        chunker.push(&mut sink, 64, vec![None, int(88)]).unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(
            written(&sink),
            [(ChunkKind::Column, 1, 0, 2), (ChunkKind::Column, 2, 64, 1)]
        );
    }

    #[test]
    fn a_column_without_a_value_in_a_chunk_writes_nothing() {
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![history(16), column(16)], 0, ChunkCaps::DEFAULT);
        chunker
            .push(
                &mut sink,
                0,
                vec![None, Some((Value::from("Amsterdam"), 0))],
            )
            .unwrap();
        chunker
            .push(&mut sink, 1, vec![None, Some((Value::from("Berlin"), 0))])
            .unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(written(&sink), [(ChunkKind::Column, 16, 0, 2)]);
    }

    #[test]
    fn a_history_chunk_comes_before_its_column_and_carries_no_epochs() {
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![history(16), column(16)], 0, ChunkCaps::DEFAULT);
        let older = Value::List(
            vec![Value::List(
                vec![Value::Int64(3), Value::from("Paris")].into(),
            )]
            .into(),
        );
        chunker
            .push(
                &mut sink,
                0,
                vec![Some((older.clone(), 0)), Some((Value::from("Prague"), 19))],
            )
            .unwrap();
        chunker
            .push(&mut sink, 1, vec![None, Some((Value::from("Berlin"), 88))])
            .unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(
            written(&sink),
            [
                (ChunkKind::History, 16, 0, 2),
                (ChunkKind::Column, 16, 0, 2)
            ]
        );
        assert_eq!(decode(&sink.0[0].0, &sink.0[0].1).epochs, None);
        assert_eq!(cells_of(&sink, ChunkKind::History, 16), [(0, older, 0)]);
        assert_eq!(
            cells_of(&sink, ChunkKind::Column, 16),
            [
                (0, Value::from("Prague"), 19),
                (1, Value::from("Berlin"), 88)
            ]
        );
    }

    #[test]
    fn rows_must_ascend_within_the_group() {
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(
            0,
            vec![column(16)],
            4,
            ChunkCaps {
                max_rows: 4,
                max_bytes: 1 << 20,
            },
        );
        chunker.push(&mut sink, 5, vec![int(3)]).unwrap();
        assert!(
            chunker.push(&mut sink, 5, vec![int(19)]).is_err(),
            "a repeated row"
        );
        assert!(
            chunker.push(&mut sink, 3, vec![int(19)]).is_err(),
            "before the group"
        );
        assert!(
            chunker.push(&mut sink, 8, vec![int(19)]).is_err(),
            "past the group"
        );
        // A refused row changes nothing.
        chunker.push(&mut sink, 7, vec![int(88)]).unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(ranges(&sink), [(5, 3)]);
        assert_eq!(
            cells_of(&sink, ChunkKind::Column, 16),
            [(5, Value::Int64(3), 0), (7, Value::Int64(88), 0)]
        );
    }

    #[test]
    fn max_rows_of_one_gives_one_row_per_chunk() {
        let caps = ChunkCaps {
            max_rows: 1,
            max_bytes: 1 << 20,
        };
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(0, vec![column(1), column(2)], 3, caps);
        chunker.push(&mut sink, 3, vec![int(3), int(19)]).unwrap();
        assert!(
            chunker.push(&mut sink, 4, vec![int(88), int(3)]).is_err(),
            "the group is one row"
        );
        chunker.finish(&mut sink).unwrap();
        assert_eq!(
            written(&sink),
            [(ChunkKind::Column, 1, 3, 1), (ChunkKind::Column, 2, 3, 1)]
        );
    }

    #[test]
    fn a_chunker_without_values_writes_nothing() {
        let mut sink = Recorder::default();
        RowsChunker::new(0, vec![column(16)], 0, ChunkCaps::DEFAULT)
            .finish(&mut sink)
            .unwrap();
        let mut chunker = RowsChunker::new(0, vec![column(16)], 0, ChunkCaps::DEFAULT);
        chunker.push(&mut sink, 0, vec![None]).unwrap();
        chunker.push(&mut sink, 3, vec![None]).unwrap();
        chunker.finish(&mut sink).unwrap();
        assert!(sink.0.is_empty(), "{:?}", written(&sink));
    }

    #[test]
    fn misuse_is_refused_naming_the_graph_and_column() {
        let mut sink = Recorder::default();
        let caps = ChunkCaps::DEFAULT;
        let refused = |columns: Vec<ChunkColumn>, caps: ChunkCaps, cells| {
            let mut sink = Recorder::default();
            let error = RowsChunker::new(3, columns, 0, caps)
                .push(&mut sink, 0, cells)
                .unwrap_err();
            assert!(sink.0.is_empty(), "nothing is written");
            error
        };
        let error = refused(vec![column(16)], caps, vec![int(3), int(19)]);
        assert!(
            matches!(&error, Error::Internal(text)
                if text.contains("graph 3")
                    && text.contains("one cell per column (1)")
                    && text.contains("got 2")),
            "{error}"
        );
        let error = refused(
            vec![history(16)],
            caps,
            vec![Some((Value::from("Mia"), 19))],
        );
        assert!(
            matches!(&error, Error::Internal(text)
                if text.contains("graph 3")
                    && text.contains("column 16")
                    && text.contains("epoch 19")),
            "{error}"
        );
        let stream = ChunkColumn {
            kind: ChunkKind::Stream,
            column_id: 16,
        };
        let error = refused(vec![stream], caps, vec![None]);
        assert!(
            matches!(&error, Error::Internal(text)
                if text.contains("column 16") && text.contains("Stream")),
            "{error}"
        );
        let error = refused(
            vec![column(16)],
            ChunkCaps {
                max_rows: 4,
                max_bytes: 0,
            },
            vec![int(3)],
        );
        assert!(matches!(error, Error::InvalidValue(_)), "{error}");
        let error = RowsChunker::new(3, vec![column(16)], 8, caps)
            .push(&mut sink, 7, vec![int(3)])
            .unwrap_err();
        assert!(
            matches!(&error, Error::Internal(text)
                if text.contains("graph 3") && text.contains("row 7")),
            "{error}"
        );
    }

    /// A (kind, column) listed twice would write two chunks with one
    /// identity: the chunker refuses it before writing anything. A column's
    /// `Column` and `History` chunks are two identities.
    #[test]
    fn a_column_listed_twice_is_refused() {
        let caps = ChunkCaps::DEFAULT;
        for (case, columns, cells) in [
            (
                "a Column column twice",
                vec![column(16), column(19), column(16)],
                vec![int(3), int(19), int(88)],
            ),
            (
                "a History column twice",
                vec![history(16), column(16), history(16)],
                vec![None, int(3), None],
            ),
        ] {
            let mut sink = Recorder::default();
            let mut chunker = RowsChunker::new(3, columns, 0, caps);
            let error = chunker.push(&mut sink, 0, cells).unwrap_err();
            assert!(
                matches!(&error, Error::Internal(text)
                    if text.contains("graph 3")
                        && text.contains("column 16")
                        && text.contains("listed twice")),
                "{case}: {error}"
            );
            chunker.finish(&mut sink).unwrap();
            assert!(sink.0.is_empty(), "{case}: {:?}", written(&sink));
        }
        let mut sink = Recorder::default();
        let mut chunker = RowsChunker::new(3, vec![history(16), column(16)], 0, caps);
        chunker
            .push(&mut sink, 0, vec![Some((Value::from("Mia"), 0)), int(3)])
            .unwrap();
        chunker.finish(&mut sink).unwrap();
        assert_eq!(
            written(&sink),
            [
                (ChunkKind::History, 16, 0, 1),
                (ChunkKind::Column, 16, 0, 1)
            ]
        );
    }

    #[test]
    fn a_sink_error_comes_back_from_push_and_finish() {
        let caps = ChunkCaps {
            max_rows: 4,
            max_bytes: 1,
        };
        let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
        chunker.push(&mut Refusing, 0, vec![int(3)]).unwrap();
        let error = chunker.push(&mut Refusing, 1, vec![int(19)]).unwrap_err();
        assert!(error.to_string().contains("Jules"), "{error}");
        let mut chunker = RowsChunker::new(0, vec![column(16)], 0, caps);
        chunker.push(&mut Refusing, 0, vec![int(3)]).unwrap();
        let error = chunker.finish(&mut Refusing).unwrap_err();
        assert!(error.to_string().contains("Jules"), "{error}");
    }
}
