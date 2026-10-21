//! The hot columns of a row group: one per property key, a typed vector
//! with presence bits.
//!
//! A column holds whatever values its rows get: it starts typed by its first
//! value (`Int64`, `Float64`, `Bool`, `String`) and turns into a column of
//! any values when a row gets a value of another type. Property columns are
//! not typed per graph (change target R11): no value is refused for its
//! type.
//!
//! A column holds the newest value of each row; what a row held before the
//! writes some readers must not see is in the column's update chains
//! (`updates`), which the reads at a version consult.
//!
//! A column is cold when it holds its values as encoded chunks: as a load
//! left them (in memory), or spilled (P8): written to a spill file whose
//! mapping the chunks read, so its values leave the heap. A spilled column
//! reads each row through its chunk's point view, as a loaded one does, and
//! comes back into memory when it is written (#412 decision Q9) or reloaded.

use std::ops::Range;
use std::sync::OnceLock;

use bytes::Bytes;
use grafeo_common::change::PendingVersion;
use grafeo_common::types::{ArcStr, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::Mutex;

use super::Read;
use super::bitset::Bitset;
use super::updates::Updates;
use super::versions::VECTOR_ROWS;
use crate::codec::column_chunk::{PointView, decode_column_chunk_bytes, encode_column_chunk};

/// The rows of a chunk a spill encodes: a vector of rows, so each chunk
/// stays small to encode and its zone map tight.
const SPILL_CHUNK_ROWS: usize = VECTOR_ROWS;

/// The values of a column's rows, by row.
#[derive(Debug, Clone)]
enum Data {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Bool(Bitset),
    String(Vec<ArcStr>),
    /// Values of any type, once the rows' types differ.
    Values(Vec<Value>),
}

impl Data {
    /// An empty column for values like `value`.
    fn for_value(value: &Value) -> Self {
        match value {
            Value::Int64(_) => Self::Int64(Vec::new()),
            Value::Float64(_) => Self::Float64(Vec::new()),
            Value::Bool(_) => Self::Bool(Bitset::default()),
            Value::String(_) => Self::String(Vec::new()),
            _ => Self::Values(Vec::new()),
        }
    }

    /// Writes `value` at `row` when it is of this column's type; returns
    /// whether it was.
    fn put(&mut self, row: usize, value: &Value) -> bool {
        fn grow<T: Clone>(values: &mut Vec<T>, row: usize, filler: T) {
            if row >= values.len() {
                values.resize(row + 1, filler);
            }
        }
        match (self, value) {
            (Self::Int64(values), Value::Int64(v)) => {
                grow(values, row, 0);
                values[row] = *v;
            }
            (Self::Float64(values), Value::Float64(v)) => {
                grow(values, row, 0.0);
                values[row] = *v;
            }
            (Self::Bool(bits), Value::Bool(v)) => bits.put(row, *v),
            (Self::String(values), Value::String(v)) => {
                grow(values, row, ArcStr::default());
                values[row] = v.clone();
            }
            (Self::Values(values), v) => {
                grow(values, row, Value::Null);
                values[row] = v.clone();
            }
            _ => return false,
        }
        true
    }

    /// The value at `row`, which the column holds.
    fn get(&self, row: usize) -> Value {
        match self {
            Self::Int64(values) => Value::Int64(values[row]),
            Self::Float64(values) => Value::Float64(values[row]),
            Self::Bool(bits) => Value::Bool(bits.get(row)),
            Self::String(values) => Value::String(values[row].clone()),
            Self::Values(values) => values[row].clone(),
        }
    }

    /// Frees what `row` holds on the heap.
    fn reset(&mut self, row: usize) {
        match self {
            Self::Int64(_) | Self::Float64(_) => {}
            Self::Bool(bits) => bits.clear(row),
            Self::String(values) => {
                if let Some(value) = values.get_mut(row) {
                    *value = ArcStr::default();
                }
            }
            Self::Values(values) => {
                if let Some(value) = values.get_mut(row) {
                    *value = Value::Null;
                }
            }
        }
    }

    /// The heap bytes the values hold besides the column's own vectors: the
    /// components of vectors and the bytes of strings.
    fn payload_bytes(&self) -> usize {
        let payload = |value: &Value| match value {
            Value::Vector(vector) => std::mem::size_of_val::<[f32]>(vector),
            Value::String(string) => string.len(),
            _ => 0,
        };
        match self {
            Self::Int64(_) | Self::Float64(_) | Self::Bool(_) => 0,
            Self::String(values) => values.iter().map(|value| value.len()).sum(),
            Self::Values(values) => values.iter().map(payload).sum(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Int64(values) => values.capacity() * std::mem::size_of::<i64>(),
            Self::Float64(values) => values.capacity() * std::mem::size_of::<f64>(),
            Self::Bool(bits) => bits.heap_bytes(),
            Self::String(values) => values.capacity() * std::mem::size_of::<ArcStr>(),
            Self::Values(values) => values.capacity() * std::mem::size_of::<Value>(),
        }
    }
}

/// A column's values, decoded: typed vectors with presence bits.
#[derive(Debug)]
struct Hot {
    /// The rows with a value.
    present: Bitset,
    data: Data,
    /// What rows held before the writes some readers must not see: a
    /// column with update chains is hot (a write thaws a cold column).
    updates: Updates<Option<Value>>,
    /// The least and the greatest value the column held since it became
    /// hot: a zone map of every version a reader may still see.
    bound: Bound,
}

/// The values a hot column held, as a range that only widens (a value
/// taken out leaves it as it is, so it covers the kept versions too): a
/// zone map the planner prunes filters with.
#[derive(Debug, Clone, Default)]
enum Bound {
    /// No value yet.
    #[default]
    Empty,
    /// Every value lies between these.
    Range { min: Value, max: Value },
    /// Values that do not order against each other: anything may match.
    Any,
}

impl Bound {
    /// Takes `value` in.
    fn widen(&mut self, value: &Value) {
        use std::cmp::Ordering;

        match self {
            Self::Any => {}
            Self::Empty => {
                *self = Self::Range {
                    min: value.clone(),
                    max: value.clone(),
                };
            }
            Self::Range { min, max } => {
                match (
                    super::read::compare_for_range(value, min),
                    super::read::compare_for_range(value, max),
                ) {
                    (Some(below), Some(above)) => {
                        if below == Ordering::Less {
                            *min = value.clone();
                        }
                        if above == Ordering::Greater {
                            *max = value.clone();
                        }
                    }
                    _ => *self = Self::Any,
                }
            }
        }
    }

    /// Whether a value of the column may meet `value` by `op`.
    fn might_match(&self, op: crate::graph::lpg::CompareOp, value: &Value) -> bool {
        use std::cmp::Ordering;

        use crate::graph::lpg::CompareOp;

        let (min, max) = match self {
            Self::Empty => return false,
            Self::Any => return true,
            Self::Range { min, max } => (min, max),
        };
        let (Some(to_min), Some(to_max)) = (
            super::read::compare_for_range(value, min),
            super::read::compare_for_range(value, max),
        ) else {
            return true;
        };
        match op {
            CompareOp::Eq => to_min != Ordering::Less && to_max != Ordering::Greater,
            CompareOp::Ne => !(to_min == Ordering::Equal && to_max == Ordering::Equal),
            CompareOp::Lt => to_min == Ordering::Greater,
            CompareOp::Le => to_min != Ordering::Less,
            CompareOp::Gt => to_max == Ordering::Less,
            CompareOp::Ge => to_max != Ordering::Greater,
        }
    }
}

impl Hot {
    /// The value at `row`, if it has one.
    fn get(&self, row: usize) -> Option<Value> {
        self.present.get(row).then(|| self.data.get(row))
    }

    /// Sets the value at `row`.
    fn set(&mut self, row: usize, value: &Value) {
        self.bound.widen(value);
        if !self.data.put(row, value) {
            if self.present.is_empty() {
                self.data = Data::for_value(value);
            } else {
                // Another type: every row's value moves to a column of any
                // values.
                let mut values = Vec::new();
                for present in self.present.rows() {
                    if present >= values.len() {
                        values.resize(present + 1, Value::Null);
                    }
                    values[present] = self.data.get(present);
                }
                self.data = Data::Values(values);
            }
            let typed = self.data.put(row, value);
            debug_assert!(typed, "a column of any values takes every value");
        }
        self.present.set(row);
    }

    /// Takes the value at `row` out, if it has one.
    fn take(&mut self, row: usize) -> Option<Value> {
        let value = self.get(row)?;
        self.present.clear(row);
        self.data.reset(row);
        Some(value)
    }

    /// The value at `row` as `read` sees it: the newest, unless a write
    /// `read` does not see replaced what it reads. A column without chains,
    /// the common case, reads in place after one test.
    fn get_at(&self, row: usize, read: Read) -> Option<Value> {
        if !self.updates.is_empty()
            && let Some(saved) = self.updates.saved(row, read)
        {
            return saved.clone();
        }
        self.get(row)
    }

    fn heap_bytes(&self) -> usize {
        self.present.heap_bytes() + self.data.heap_bytes()
    }

    /// The decoded values of `chunks`.
    fn decode(chunks: &[ColdChunk]) -> Result<Self> {
        let mut hot = Self {
            present: Bitset::default(),
            data: Data::Values(Vec::new()),
            updates: Updates::default(),
            bound: Bound::Empty,
        };
        for chunk in chunks {
            let decoded = decode_column_chunk_bytes(&chunk.bytes, chunk.codec, chunk.row_count)?;
            for (offset, value) in decoded.values {
                hot.set(chunk.first_row + offset as usize, &value);
            }
        }
        Ok(hot)
    }
}

/// Where a cold chunk's bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Home {
    /// In memory, as the load fetched them.
    Memory,
    /// In a spill file, mapped: off the heap.
    Spilled,
}

/// A column chunk as the file holds it, kept encoded: the values of rows
/// `[first_row, first_row + row_count)` of the row group.
#[derive(Debug)]
struct ColdChunk {
    first_row: usize,
    row_count: u32,
    codec: u8,
    bytes: Bytes,
    point: Option<PointView>,
    home: Home,
}

/// What a spill of a column does (see [`Columns::plan_spill`]).
pub(super) enum SpillPlan {
    /// Nothing: the column has no value, or is spilled already.
    Nothing,
    /// The column stays in memory: it has update chains, which a cold column
    /// cannot hold (a write thaws it, Q9), and which the readers that must
    /// not see a write read.
    Chained,
    /// The column stays in memory: a chunk of it would have no point view
    /// (strings, values of several kinds), so its first read would decode
    /// the whole column back onto the heap.
    Unsupported,
    /// The chunks to write to a spill file.
    Write(SpillChunks),
}

/// A column's chunks encoded for a spill: the bytes a spill file holds and
/// where each chunk is in them.
pub(super) struct SpillChunks {
    /// The chunks, one after the other: what the spill file holds.
    pub(super) bytes: Vec<u8>,
    places: Vec<ChunkPlace>,
    /// The heap bytes the column holds now, which the spill frees.
    pub(super) resident: usize,
}

/// Where a chunk is in a spill file, and what its directory entry says.
struct ChunkPlace {
    first_row: usize,
    row_count: u32,
    codec: u8,
    range: Range<usize>,
}

impl SpillChunks {
    fn new(resident: usize) -> Self {
        Self {
            bytes: Vec::new(),
            places: Vec::new(),
            resident,
        }
    }

    fn push(&mut self, first_row: usize, row_count: u32, codec: u8, bytes: &[u8]) {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        self.places.push(ChunkPlace {
            first_row,
            row_count,
            codec,
            range: start..self.bytes.len(),
        });
    }
}

/// One property key's values in a row group.
///
/// A loaded column holds its chunks cold, as their encoded bytes. Scalar
/// reads use those bytes directly (a point view per chunk); a write, or a
/// read of a chunk without a point view, decodes the column into the hot
/// form once. A write thaws a cold column (#412 decision Q9), so the update
/// chains of a column live with its decoded values, and a cold column has
/// none.
#[derive(Debug)]
pub(super) struct Column {
    /// The chunks not decoded yet; emptied when the hot form is published.
    cold: Mutex<Vec<ColdChunk>>,
    /// The decoded values, set when a caller needs the hot form.
    hot: OnceLock<Hot>,
}

impl Column {
    /// An empty column for values like `value`.
    fn for_value(value: &Value) -> Self {
        Self {
            cold: Mutex::new(Vec::new()),
            hot: OnceLock::from(Hot {
                present: Bitset::default(),
                data: Data::for_value(value),
                updates: Updates::default(),
                bound: Bound::Empty,
            }),
        }
    }

    /// An empty column that takes cold chunks.
    fn cold() -> Self {
        Self {
            cold: Mutex::new(Vec::new()),
            hot: OnceLock::new(),
        }
    }

    /// The value at `row` as `read` sees it. A cold column has no update
    /// chains (a write thaws it), so every reader reads its one value, from
    /// the chunk's point view where it has one.
    fn value_at(&self, row: usize, read: Read) -> Option<Value> {
        match self.hot.get() {
            Some(hot) => hot.get_at(row, read),
            None => self.get(row),
        }
    }

    /// The update chains, `None` while the column is cold (it has none).
    fn updates(&self) -> Option<&Updates<Option<Value>>> {
        self.hot.get().map(|hot| &hot.updates)
    }

    /// The update chains, thawing the column.
    fn updates_mut(&mut self) -> &mut Updates<Option<Value>> {
        &mut self.hot_mut().updates
    }

    /// Keeps the value at `row` for the readers that must not see what
    /// `keep` (the writer's stamp and the epoch it reads at) writes next;
    /// `None` keeps nothing. Returns what the write does to the writer's
    /// version of the value.
    fn keep(&mut self, row: usize, keep: Option<(u64, u64)>) -> PendingVersion {
        let Some((stamp, snapshot)) = keep else {
            return PendingVersion::Created;
        };
        let hot = self.hot_mut();
        let current = hot.get(row);
        if hot.updates.keep(row, stamp, snapshot, || current) {
            PendingVersion::Created
        } else {
            PendingVersion::Replaced
        }
    }

    /// The decoded values, decoding the cold chunks when needed.
    fn hot(&self) -> &Hot {
        if let Some(hot) = self.hot.get() {
            return hot;
        }
        // Keep the chunks locked until OnceLock publishes the decoded form.
        // Otherwise a concurrent point read could find neither form.
        let mut cold = self.cold.lock();
        self.hot.get_or_init(|| {
            // The load decoded each chunk once to check it, and a spill
            // checked that its file holds the chunks it encoded.
            let hot = Hot::decode(&cold).expect("a chunk the load or the spill checked");
            *cold = Vec::new();
            hot
        })
    }

    /// Decodes the cold chunks into the hot form, as [`hot`](Self::hot)
    /// does, returning a chunk's decode error instead (the column then stays
    /// as it was).
    fn thaw(&mut self) -> Result<()> {
        if self.hot.get().is_some() {
            return Ok(());
        }
        let cold = self.cold.get_mut();
        let hot = Hot::decode(cold)?;
        *cold = Vec::new();
        self.hot = OnceLock::from(hot);
        Ok(())
    }

    /// Whether the row has a value, reading a cold column's presence bits
    /// without decoding a value.
    fn has_value(&self, row: usize) -> bool {
        if let Some(hot) = self.hot.get() {
            return hot.present.get(row);
        }
        let cold = self.cold.lock();
        if let Some(hot) = self.hot.get() {
            return hot.present.get(row);
        }
        let Some(chunk) = cold.iter().find(|chunk| {
            row.checked_sub(chunk.first_row)
                .is_some_and(|offset| offset < chunk.row_count as usize)
        }) else {
            return false;
        };
        match (&chunk.point, u32::try_from(row - chunk.first_row)) {
            (Some(point), Ok(offset)) => point.has(offset),
            _ => {
                drop(cold);
                self.hot().present.get(row)
            }
        }
    }

    /// What a spill of the column writes (see [`SpillPlan`]): the chunks of
    /// its hot form encoded a vector of rows each, or the chunks a load left
    /// in memory as they are.
    ///
    /// # Errors
    ///
    /// Returns the encoder's error for a value the format cannot hold.
    fn plan_spill(&self) -> Result<SpillPlan> {
        if self.hot.get().is_none() {
            let cold = self.cold.lock();
            // A read that decoded the column meanwhile left it hot.
            if self.hot.get().is_none() {
                return Ok(Self::plan_cold_spill(&cold));
            }
        }
        let hot = self.hot();
        if !hot.updates.is_empty() {
            return Ok(SpillPlan::Chained);
        }
        if hot.present.is_empty() {
            return Ok(SpillPlan::Nothing);
        }
        let mut chunks = SpillChunks::new(hot.heap_bytes() + hot.data.payload_bytes());
        let rows = u32::try_from(SPILL_CHUNK_ROWS).expect("a vector of rows fits a chunk");
        let mut window: Vec<(u32, Value)> = Vec::new();
        let mut first_row = 0;
        // Encodes the window's values as one chunk; `false` for a chunk
        // without a point view.
        let mut flush = |first_row: usize, window: &mut Vec<(u32, Value)>| -> Result<bool> {
            if window.is_empty() {
                return Ok(true);
            }
            let (codec, bytes) = encode_column_chunk(rows, window, None)?;
            window.clear();
            if !PointView::supports(codec) {
                return Ok(false);
            }
            chunks.push(first_row, rows, codec.to_byte(), &bytes);
            Ok(true)
        };
        for row in hot.present.rows() {
            let start = row - row % SPILL_CHUNK_ROWS;
            if start != first_row {
                if !flush(first_row, &mut window)? {
                    return Ok(SpillPlan::Unsupported);
                }
                first_row = start;
            }
            let offset = u32::try_from(row - start).expect("an offset in a vector of rows");
            window.push((offset, hot.data.get(row)));
        }
        if !flush(first_row, &mut window)? {
            return Ok(SpillPlan::Unsupported);
        }
        Ok(SpillPlan::Write(chunks))
    }

    /// [`plan_spill`](Self::plan_spill) of a cold column: its chunks as a
    /// load left them, unless they are spilled already.
    fn plan_cold_spill(cold: &[ColdChunk]) -> SpillPlan {
        if cold.is_empty() || cold.iter().any(|chunk| chunk.home == Home::Spilled) {
            return SpillPlan::Nothing;
        }
        if cold.iter().any(|chunk| chunk.point.is_none()) {
            return SpillPlan::Unsupported;
        }
        let resident = cold.iter().map(|chunk| chunk.bytes.len()).sum();
        let mut chunks = SpillChunks::new(resident);
        for chunk in cold {
            chunks.push(chunk.first_row, chunk.row_count, chunk.codec, &chunk.bytes);
        }
        SpillPlan::Write(chunks)
    }

    /// Makes the column cold, its chunks read from `mapped`: the spill file
    /// that holds `chunks` (from [`plan_spill`](Self::plan_spill), with no
    /// write since).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], changing nothing, when `mapped` does not
    /// hold the bytes `chunks` encoded, or a chunk gets no point view.
    fn install_spill(&mut self, chunks: SpillChunks, mapped: &Bytes) -> Result<()> {
        if mapped.as_ref() != chunks.bytes.as_slice() {
            return Err(Error::Internal(
                "a spill file does not hold the chunks written to it".to_string(),
            ));
        }
        let mut cold = Vec::with_capacity(chunks.places.len());
        for place in chunks.places {
            let bytes = mapped.slice(place.range);
            // The bytes are those the encoder wrote or the load checked.
            let point = PointView::from_validated(&bytes, place.codec, place.row_count)?
                .ok_or_else(|| {
                    Error::Internal("a spilled chunk without a point view".to_string())
                })?;
            cold.push(ColdChunk {
                first_row: place.first_row,
                row_count: place.row_count,
                codec: place.codec,
                bytes,
                point: Some(point),
                home: Home::Spilled,
            });
        }
        *self.cold.get_mut() = cold;
        self.hot = OnceLock::new();
        Ok(())
    }

    /// Whether its chunks are in a spill file.
    fn is_spilled(&self) -> bool {
        self.is_cold()
            && self
                .cold
                .lock()
                .iter()
                .any(|chunk| chunk.home == Home::Spilled)
    }

    /// Reads a scalar from its chunk, decoding only for unsupported kinds.
    fn get(&self, row: usize) -> Option<Value> {
        if let Some(hot) = self.hot.get() {
            return hot.get(row);
        }
        let cold = self.cold.lock();
        if let Some(hot) = self.hot.get() {
            return hot.get(row);
        }
        let chunk = cold.iter().find(|chunk| {
            row.checked_sub(chunk.first_row)
                .is_some_and(|offset| offset < chunk.row_count as usize)
        })?;
        if let Some(point) = &chunk.point {
            let offset = u32::try_from(row - chunk.first_row).ok()?;
            return point.get(offset);
        }
        drop(cold);
        self.hot().get(row)
    }

    fn hot_mut(&mut self) -> &mut Hot {
        self.hot();
        self.hot.get_mut().expect("set by the line above")
    }

    /// Whether the column still holds its chunks encoded.
    fn is_cold(&self) -> bool {
        self.hot.get().is_none()
    }

    /// The heap bytes of the chunks it still holds encoded: their metadata,
    /// the point views' indexes, and the bytes of the chunks in memory (a
    /// spilled chunk's are in its file).
    fn cold_bytes(&self) -> usize {
        let cold = self.cold.lock();
        cold.capacity() * std::mem::size_of::<ColdChunk>()
            + cold
                .iter()
                .map(|chunk| {
                    let bytes = match chunk.home {
                        Home::Memory => chunk.bytes.len(),
                        Home::Spilled => 0,
                    };
                    bytes + chunk.point.as_ref().map_or(0, PointView::index_heap_bytes)
                })
                .sum::<usize>()
    }

    /// The bytes of its chunks in a spill file.
    fn spilled_bytes(&self) -> usize {
        self.cold
            .lock()
            .iter()
            .filter(|chunk| chunk.home == Home::Spilled)
            .map(|chunk| chunk.bytes.len())
            .sum()
    }

    /// The heap bytes of its decoded values, none while it is cold.
    fn hot_bytes(&self) -> usize {
        self.hot.get().map_or(0, Hot::heap_bytes)
    }
}

/// The columns of a row group, by property key id.
#[derive(Debug, Default)]
pub(super) struct Columns {
    by_key: FxHashMap<u32, Column>,
}

impl Columns {
    /// The value of key `key` at `row`.
    pub(super) fn get(&self, key: u32, row: usize) -> Option<Value> {
        self.by_key.get(&key)?.get(row)
    }

    /// Adds a chunk of key `key` as the file holds it, for rows from
    /// `first_row`. The load has already decoded and validated the chunk.
    pub(super) fn add_cold(
        &mut self,
        key: u32,
        first_row: usize,
        row_count: u32,
        codec: u8,
        bytes: Bytes,
    ) {
        let column = self.by_key.entry(key).or_insert_with(Column::cold);
        if column.is_cold() {
            let point = PointView::from_validated(&bytes, codec, row_count)
                .expect("a chunk the load checked");
            column.cold.lock().push(ColdChunk {
                first_row,
                row_count,
                codec,
                bytes,
                point,
                home: Home::Memory,
            });
        } else {
            // Written to already: decode into the hot values.
            let decoded = decode_column_chunk_bytes(&bytes, codec, row_count)
                .expect("a chunk the load checked");
            let hot = column.hot_mut();
            for (offset, value) in decoded.values {
                hot.set(first_row + offset as usize, &value);
            }
        }
    }

    /// Whether a value of key `key` may meet `value` by `op`: by the zone
    /// map of the hot column; a cold one (its chunks' zone maps are not read
    /// here yet) may always. `None` without a column of the key.
    pub(super) fn might_match(
        &self,
        key: u32,
        op: crate::graph::lpg::CompareOp,
        value: &Value,
    ) -> Option<bool> {
        let column = self.by_key.get(&key)?;
        if let Some(hot) = column.hot.get() {
            return Some(hot.bound.might_match(op, value));
        }
        // Under the chunks' lock: a thaw publishes the hot form first.
        let cold = column.cold.lock();
        Some(match column.hot.get() {
            Some(hot) => hot.bound.might_match(op, value),
            None => !cold.is_empty(),
        })
    }

    /// The columns that still hold their chunks encoded.
    pub(super) fn cold_count(&self) -> usize {
        self.by_key
            .values()
            .filter(|column| column.is_cold())
            .count()
    }

    /// Sets the value of key `key` at `row`.
    pub(super) fn set(&mut self, key: u32, row: usize, value: &Value) {
        self.by_key
            .entry(key)
            .or_insert_with(|| Column::for_value(value))
            .hot_mut()
            .set(row, value);
    }

    /// Takes the value of key `key` at `row` out.
    pub(super) fn take(&mut self, key: u32, row: usize) -> Option<Value> {
        self.by_key.get_mut(&key)?.hot_mut().take(row)
    }

    /// The value of key `key` at `row` as `read` sees it.
    pub(super) fn get_at(&self, key: u32, row: usize, read: Read) -> Option<Value> {
        self.by_key.get(&key)?.value_at(row, read)
    }

    /// Writes the value of key `key` at `row` (`None` takes it out), first
    /// keeping the value it replaces for the readers that must not see the
    /// write (`keep`: the writer's stamp and the epoch it reads at; `None`
    /// keeps nothing). Returns what the write does to the writer's version
    /// of the value.
    pub(super) fn write(
        &mut self,
        key: u32,
        row: usize,
        value: Option<&Value>,
        keep: Option<(u64, u64)>,
    ) -> PendingVersion {
        let column = match value {
            Some(value) => self
                .by_key
                .entry(key)
                .or_insert_with(|| Column::for_value(value)),
            None => match self.by_key.get_mut(&key) {
                Some(column) => column,
                None => return PendingVersion::Created,
            },
        };
        let version = column.keep(row, keep);
        let hot = column.hot_mut();
        match value {
            Some(value) => hot.set(row, value),
            None => {
                hot.take(row);
            }
        }
        version
    }

    /// Writes `value` back at `row` of key `key` (`None` takes the value
    /// out), keeping nothing: an undo.
    pub(super) fn restore(&mut self, key: u32, row: usize, value: Option<&Value>) {
        match value {
            Some(value) => self.set(key, row, value),
            None => {
                self.take(key, row);
            }
        }
    }

    /// Commits the newest kept value of key `key` at `row` at `at`, if
    /// transaction stamp `pending` kept it (see [`Updates::stamp`]).
    pub(super) fn stamp(&mut self, key: u32, row: usize, pending: u64, at: u64) -> Option<u64> {
        self.by_key
            .get_mut(&key)?
            .updates_mut()
            .stamp(row, pending, at)
    }

    /// Rolls back transaction stamp `pending`'s first write of key `key` at
    /// `row`: writes back what its undo node kept, what the row held before
    /// the transaction wrote it, and takes it out of the node (see
    /// [`Updates::undo`]). Returns `false`, changing nothing, when no node
    /// of the transaction holds the row.
    pub(super) fn undo(&mut self, key: u32, row: usize, pending: u64) -> bool {
        let Some(column) = self.by_key.get_mut(&key) else {
            return false;
        };
        let Some(kept) = column.updates_mut().undo(row, pending) else {
            return false;
        };
        let hot = column.hot_mut();
        match kept {
            Some(value) => hot.set(row, &value),
            None => {
                hot.take(row);
            }
        }
        true
    }

    /// Drops the kept values every reader at `horizon` or later sees past;
    /// returns how many it dropped.
    pub(super) fn collect(&mut self, horizon: u64) -> usize {
        let mut dropped = 0;
        for column in self.by_key.values_mut() {
            if column.updates().is_some_and(|updates| !updates.is_empty()) {
                dropped += column.updates_mut().collect(horizon);
            }
        }
        dropped
    }

    /// [`collect`](Self::collect) of the rows of vector `vector` alone.
    pub(super) fn collect_vector(&mut self, vector: usize, horizon: u64) -> usize {
        let mut dropped = 0;
        for column in self.by_key.values_mut() {
            if column
                .updates()
                .is_some_and(|updates| updates.has_chain(vector))
            {
                dropped += column.updates_mut().collect_vector(vector, horizon);
            }
        }
        dropped
    }

    /// Whether a column has a chain in vector `vector`.
    pub(super) fn has_chain(&self, vector: usize) -> bool {
        self.by_key
            .values()
            .filter_map(Column::updates)
            .any(|updates| updates.has_chain(vector))
    }

    /// Every value at `row`, by key id.
    pub(super) fn row(&self, row: usize) -> impl Iterator<Item = (u32, Value)> + '_ {
        self.by_key
            .iter()
            .filter_map(move |(key, column)| column.get(row).map(|value| (*key, value)))
    }

    /// Every value at `row` as `read` sees it, by key id.
    pub(super) fn row_at(&self, row: usize, read: Read) -> impl Iterator<Item = (u32, Value)> + '_ {
        self.by_key
            .iter()
            .filter_map(move |(key, column)| column.value_at(row, read).map(|value| (*key, value)))
    }

    /// Takes every value at `row` out, with what any chain kept of it. A
    /// cold column without a value at the row stays cold.
    pub(super) fn clear_row(&mut self, row: usize) {
        for column in self.by_key.values_mut() {
            if column.is_cold() && !column.has_value(row) {
                continue;
            }
            let hot = column.hot_mut();
            hot.take(row);
            if !hot.updates.is_empty() {
                hot.updates.remove_row(row);
            }
        }
    }

    /// The heap bytes of the update chains.
    pub(super) fn updates_bytes(&self) -> usize {
        self.by_key
            .values()
            .filter_map(Column::updates)
            .map(|updates| updates.heap_bytes(|_| 0))
            .sum()
    }

    /// The undo nodes of the update chains.
    #[cfg(test)]
    pub(super) fn update_nodes(&self) -> usize {
        self.by_key
            .values()
            .filter_map(Column::updates)
            .map(Updates::nodes)
            .sum()
    }

    /// What the update chains of key `key` kept of `row`, oldest first, each
    /// with the stamp of the write over it (see [`Updates::kept_stamped`]).
    pub(super) fn kept(
        &self,
        key: u32,
        row: usize,
    ) -> impl Iterator<Item = (u64, &Option<Value>)> + '_ {
        self.by_key
            .get(&key)
            .and_then(Column::updates)
            .into_iter()
            .flat_map(move |updates| updates.kept_stamped(row))
    }

    /// Restores the update chains of key `key` a load reads from a file's
    /// history (see [`Updates::load`]), thawing its column: a column with
    /// chains is hot (Q9). A key with no value left in the rows (every row
    /// had it removed last) gets an empty column to hold them.
    pub(super) fn load_kept(&mut self, key: u32, kept: Vec<(usize, u64, Option<Value>)>) {
        self.by_key
            .entry(key)
            .or_insert_with(Column::cold)
            .updates_mut()
            .load(kept);
    }

    /// The rows the update chains of key `key` hold, `None` when they hold
    /// none.
    pub(super) fn chained_rows(&self, key: u32) -> Option<Bitset> {
        self.by_key.get(&key)?.updates()?.rows()
    }

    /// Each column written out, by key id ascending.
    #[cfg(test)]
    pub(super) fn canonical(&self) -> Vec<(u32, String)> {
        let mut columns: Vec<(u32, String)> = self
            .by_key
            .iter()
            .map(|(key, column)| (*key, format!("{column:?}")))
            .collect();
        columns.sort_unstable();
        columns
    }

    /// The key ids it has a column of.
    pub(super) fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.by_key.keys().copied()
    }

    /// The heap bytes of the decoded values and of the columns themselves.
    pub(super) fn hot_bytes(&self) -> usize {
        self.by_key.values().map(Column::hot_bytes).sum::<usize>()
            + self.by_key.capacity() * (std::mem::size_of::<u32>() + std::mem::size_of::<Column>())
    }

    /// The heap bytes of the chunks its columns still hold encoded.
    pub(super) fn cold_bytes(&self) -> usize {
        self.by_key.values().map(Column::cold_bytes).sum::<usize>()
    }

    /// What a spill of key `key`'s column writes (see [`SpillPlan`]).
    ///
    /// # Errors
    ///
    /// Returns the encoder's error for a value the format cannot hold.
    pub(super) fn plan_spill(&self, key: u32) -> Result<SpillPlan> {
        match self.by_key.get(&key) {
            Some(column) => column.plan_spill(),
            None => Ok(SpillPlan::Nothing),
        }
    }

    /// Makes key `key`'s column cold, read from `mapped`, the spill file of
    /// `chunks` (planned with no write since); returns the column's heap
    /// bytes after.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], changing nothing, when `mapped` does not
    /// hold the planned bytes, or the column is gone.
    pub(super) fn install_spill(
        &mut self,
        key: u32,
        chunks: SpillChunks,
        mapped: &Bytes,
    ) -> Result<usize> {
        let column = self
            .by_key
            .get_mut(&key)
            .ok_or_else(|| Error::Internal("a spilled column is gone".to_string()))?;
        column.install_spill(chunks, mapped)?;
        Ok(column.cold_bytes())
    }

    /// Brings key `key`'s spilled column back into memory; returns whether
    /// it was spilled.
    ///
    /// # Errors
    ///
    /// Returns a chunk's decode error; the column then stays spilled.
    pub(super) fn reload(&mut self, key: u32) -> Result<bool> {
        match self.by_key.get_mut(&key) {
            Some(column) if column.is_spilled() => column.thaw().map(|()| true),
            _ => Ok(false),
        }
    }

    /// The key ids of its spilled columns.
    pub(super) fn spilled_keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.by_key
            .iter()
            .filter(|(_, column)| column.is_spilled())
            .map(|(key, _)| *key)
    }

    /// The bytes of its columns' chunks in spill files.
    pub(super) fn spilled_bytes(&self) -> usize {
        self.by_key.values().map(Column::spilled_bytes).sum()
    }

    /// The columns whose chunks are in spill files.
    pub(super) fn spilled_count(&self) -> usize {
        self.by_key
            .values()
            .filter(|column| column.is_spilled())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use bytes::Bytes;
    use grafeo_common::types::Value;

    use super::Columns;
    use crate::codec::column_chunk::{decode_column_chunk_bytes, encode_column_chunk};

    fn add_chunk(columns: &mut Columns, first: usize, rows: u32, values: &[(u32, Value)]) {
        let epochs = vec![17; values.len()];
        let (codec, bytes) = encode_column_chunk(rows, values, Some(&epochs)).unwrap();
        let bytes = Bytes::from(bytes);
        decode_column_chunk_bytes(&bytes, codec.to_byte(), rows).unwrap();
        columns.add_cold(7, first, rows, codec.to_byte(), bytes);
    }

    /// A point read must not inflate the column into an owned Value per row.
    #[test]
    fn point_reads_of_a_loaded_integer_column_keep_it_cold() {
        let values: Vec<_> = (0..65_536)
            .map(|row| (row, Value::Int64(i64::from(row))))
            .collect();
        let (codec, encoded) = encode_column_chunk(65_536, &values, None).unwrap();
        let bytes = bytes::Bytes::from(encoded);
        decode_column_chunk_bytes(&bytes, codec.to_byte(), 65_536).unwrap();
        let mut columns = Columns::default();
        columns.add_cold(7, 0, 65_536, codec.to_byte(), bytes);
        let resident = columns.hot_bytes() + columns.cold_bytes();

        for (row, expected) in [(0, 0), (63, 63), (64, 64), (65_535, 65_535)] {
            assert_eq!(columns.get(7, row), Some(Value::Int64(expected)));
        }
        assert_eq!(columns.get(7, 65_536), None);
        assert_eq!(
            columns.cold_count(),
            1,
            "point reads must retain the encoded column"
        );
        assert_eq!(
            (columns.hot_bytes() + columns.cold_bytes()),
            resident,
            "point reads must not allocate a decoded column"
        );
    }

    #[test]
    fn sparse_point_reads_cross_words_and_chunks_without_decoding() {
        let mut columns = Columns::default();
        let values = [0, 63, 64, 127, 128, 255].map(|row| (row, Value::Int64(-i64::from(row) - 1)));
        add_chunk(&mut columns, 300, 256, &values);
        add_chunk(&mut columns, 0, 2, &[(1, Value::Bool(false))]);
        let resident = columns.hot_bytes() + columns.cold_bytes();
        for row in 0..600 {
            let expected = if row == 1 {
                Some(Value::Bool(false))
            } else {
                values.iter().find_map(|(offset, value)| {
                    (row == 300 + *offset as usize).then(|| value.clone())
                })
            };
            assert_eq!(columns.get(7, row), expected, "row {row}");
            assert_eq!(
                columns.row(row).collect::<Vec<_>>(),
                expected
                    .into_iter()
                    .map(|value| (7, value))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(columns.get(7, usize::MAX), None);
        assert_eq!(columns.cold_count(), 1);
        assert_eq!((columns.hot_bytes() + columns.cold_bytes()), resident);
    }

    #[test]
    fn writes_and_removals_preserve_other_cold_rows() {
        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            65,
            &[
                (0, Value::Int64(11)),
                (63, Value::Int64(22)),
                (64, Value::Int64(33)),
            ],
        );
        assert_eq!(columns.get(7, 63), Some(Value::Int64(22)));
        assert_eq!(columns.cold_count(), 1);
        columns.set(7, 63, &Value::from("changed type"));
        assert_eq!(columns.cold_count(), 0);
        assert_eq!(columns.get(7, 0), Some(Value::Int64(11)));
        assert_eq!(columns.get(7, 63), Some(Value::from("changed type")));
        assert_eq!(columns.take(7, 64), Some(Value::Int64(33)));
        assert_eq!(columns.get(7, 64), None);

        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            2,
            &[(0, Value::Int64(11)), (1, Value::Int64(22))],
        );
        assert_eq!(columns.take(7, 0), Some(Value::Int64(11)));
        assert_eq!(columns.get(7, 1), Some(Value::Int64(22)));
        columns.clear_row(1);
        assert_eq!(columns.get(7, 1), None);
    }

    #[test]
    fn concurrent_point_reads_and_fallback_reads_keep_every_value() {
        for _ in 0..16 {
            let mut columns = Columns::default();
            let values = (0..4096)
                .map(|row| (row, Value::Int64(i64::from(row))))
                .collect::<Vec<_>>();
            add_chunk(&mut columns, 0, 4096, &values);
            add_chunk(&mut columns, 4096, 1, &[(0, Value::from("fallback"))]);
            let columns = Arc::new(columns);
            let start = Arc::new(Barrier::new(4));
            std::thread::scope(|scope| {
                for reader in 0..4 {
                    let columns = Arc::clone(&columns);
                    let start = Arc::clone(&start);
                    scope.spawn(move || {
                        start.wait();
                        for _ in 0..32 {
                            let (row, expected) = if reader == 0 {
                                (4096, Value::from("fallback"))
                            } else {
                                (4095, Value::Int64(4095))
                            };
                            assert_eq!(columns.get(7, row), Some(expected));
                        }
                    });
                }
            });
            assert_eq!(columns.cold_count(), 0);
        }
    }

    #[test]
    fn a_column_keeps_typed_values_and_turns_into_any_values_on_another_type() {
        let mut columns = Columns::default();
        columns.set(7, 3, &Value::Int64(19));
        columns.set(7, 0, &Value::Int64(88));
        assert_eq!(columns.get(7, 3), Some(Value::Int64(19)));
        assert_eq!(columns.get(7, 1), None, "a row without a value");
        columns.set(7, 5, &Value::from("Amsterdam"));
        assert_eq!(
            columns.get(7, 0),
            Some(Value::Int64(88)),
            "kept on the change"
        );
        assert_eq!(columns.get(7, 5), Some(Value::from("Amsterdam")));
        assert_eq!(columns.take(7, 3), Some(Value::Int64(19)));
        assert_eq!(columns.get(7, 3), None);
        assert_eq!(columns.take(7, 3), None);
        columns.set(8, 2, &Value::Bool(false));
        columns.set(8, 4, &Value::Bool(true));
        assert_eq!(
            columns.get(8, 2),
            Some(Value::Bool(false)),
            "a false value is present"
        );
        let mut row: Vec<(u32, Value)> = columns.row(0).collect();
        row.sort_by_key(|(key, _)| *key);
        assert_eq!(row, [(7, Value::Int64(88))]);
    }

    #[test]
    fn an_emptied_column_takes_the_type_of_its_next_value() {
        let mut columns = Columns::default();
        columns.set(1, 0, &Value::Int64(3));
        columns.take(1, 0);
        columns.set(1, 0, &Value::from("Gus"));
        assert_eq!(columns.get(1, 0), Some(Value::from("Gus")));
        columns.set(1, 1, &Value::Null);
        assert_eq!(
            columns.get(1, 1),
            Some(Value::Null),
            "a null is a value the row holds"
        );
    }
}
