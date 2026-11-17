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
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use grafeo_common::change::PendingVersion;
use grafeo_common::memory::buffer::{BufferManager, MemoryGrant};
use grafeo_common::types::{ArcStr, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::Mutex;

use super::Read;
use super::account::{ThawGrant, ThawSink, granted_chunk};
use super::bitset::Bitset;
use super::slots::{GroupRows, vector_start};
use super::updates::Updates;
use super::versions::VECTOR_ROWS;
use crate::codec::column_chunk::{
    ColumnChunk, DictView, PointView, ZoneMap, decode_checked_chunk, decode_column_chunk_bytes,
    encode_column_chunk, read_zone_map,
};

/// The rows of a chunk a spill encodes: a vector of rows, so each chunk
/// stays small to encode and its zone map tight.
const SPILL_CHUNK_ROWS: usize = VECTOR_ROWS;

/// The values of a column's rows, by row, from the vector of the group's
/// first row with a value (see [`GroupRows`]).
#[derive(Debug, Clone)]
enum Data {
    Int64(GroupRows<i64>),
    Float64(GroupRows<f64>),
    Bool(Bitset),
    String(GroupRows<ArcStr>),
    /// Values of any type, once the rows' types differ.
    Values(GroupRows<Value>),
}

impl Data {
    /// An empty column for values like `value`.
    fn for_value(value: &Value) -> Self {
        match value {
            Value::Int64(_) => Self::Int64(GroupRows::default()),
            Value::Float64(_) => Self::Float64(GroupRows::default()),
            Value::Bool(_) => Self::Bool(Bitset::default()),
            Value::String(_) => Self::String(GroupRows::default()),
            _ => Self::Values(GroupRows::default()),
        }
    }

    /// Writes `value` at `row` when it is of this column's type; returns
    /// whether it was. The vectors grow through [`GroupRows::put`]; what they
    /// hold is read through [`heap_bytes`](Self::heap_bytes).
    fn put(&mut self, row: usize, value: &Value) -> bool {
        match (self, value) {
            (Self::Int64(values), Value::Int64(v)) => values.put(row, *v, 0),
            (Self::Float64(values), Value::Float64(v)) => values.put(row, *v, 0.0),
            (Self::Bool(bits), Value::Bool(v)) => bits.put(row, *v),
            (Self::String(values), Value::String(v)) => {
                values.put(row, v.clone(), ArcStr::default());
            }
            (Self::Values(values), v) => values.put(row, v.clone(), Value::Null),
            _ => return false,
        }
        true
    }

    /// The value at `row`, which the column holds.
    fn get(&self, row: usize) -> Value {
        match self {
            Self::Int64(values) => Value::Int64(*values.at(row)),
            Self::Float64(values) => Value::Float64(*values.at(row)),
            Self::Bool(bits) => Value::Bool(bits.get(row)),
            Self::String(values) => Value::String(values.at(row).clone()),
            Self::Values(values) => values.at(row).clone(),
        }
    }

    /// Calls `read` with the value at `row`, which the column holds; a value
    /// of a column of any values is lent, not copied.
    fn with_value<R>(&self, row: usize, read: impl FnOnce(&Value) -> R) -> R {
        match self {
            Self::Values(values) => read(values.at(row)),
            _ => read(&self.get(row)),
        }
    }

    /// How the value at `a` orders against the value at `b`, both held, as
    /// [`compare_for_range`](super::read::compare_for_range) orders them: in
    /// place, without making either a [`Value`].
    fn compare_rows(&self, a: usize, b: usize) -> Option<std::cmp::Ordering> {
        match self {
            Self::Int64(values) => Some(values.at(a).cmp(values.at(b))),
            Self::Float64(values) => values.at(a).partial_cmp(values.at(b)),
            Self::Bool(bits) => Some(bits.get(a).cmp(&bits.get(b))),
            Self::String(values) => Some(values.at(a).cmp(values.at(b))),
            Self::Values(values) => super::read::compare_for_range(values.at(a), values.at(b)),
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

    /// The heap bytes the value at `row` holds besides the column's vector,
    /// as [`payload_bytes`](Self::payload_bytes) counts it; 0 past the rows.
    fn payload_at(&self, row: usize) -> usize {
        match self {
            Self::Int64(_) | Self::Float64(_) | Self::Bool(_) => 0,
            Self::String(values) => values.get(row).map_or(0, |value| value.len()),
            Self::Values(values) => values.get(row).map_or(0, Value::estimated_size_bytes),
        }
    }

    /// The heap bytes the values hold besides the column's own vectors: the
    /// bytes of strings, and of every value of another kind that holds any
    /// (vectors, bytes, lists, maps, paths, counters), as
    /// [`Value::estimated_size_bytes`] counts them. A value several rows
    /// share is counted for each.
    fn payload_bytes(&self) -> usize {
        match self {
            Self::Int64(_) | Self::Float64(_) | Self::Bool(_) => 0,
            Self::String(values) => values.as_slice().iter().map(|value| value.len()).sum(),
            Self::Values(values) => values
                .as_slice()
                .iter()
                .map(Value::estimated_size_bytes)
                .sum(),
        }
    }

    /// The heap bytes of the column's own vectors: their capacity, the one
    /// figure of their size both the walk and the account read.
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Int64(values) => values.heap_bytes(),
            Self::Float64(values) => values.heap_bytes(),
            Self::Bool(bits) => bits.heap_bytes(),
            Self::String(values) => values.heap_bytes(),
            Self::Values(values) => values.heap_bytes(),
        }
    }
}

/// A column's values, decoded: typed vectors with presence bits.
#[derive(Debug)]
struct Hot {
    /// The rows with a value.
    present: Bitset,
    data: Data,
    /// The heap bytes the values hold besides the vectors (see
    /// [`Data::payload_bytes`]), kept as the values change.
    payload: usize,
    /// What rows held before the writes some readers must not see: a
    /// column with update chains is hot (a write thaws a cold column).
    updates: Updates<Option<Value>>,
    /// The rows of the least and the greatest value: a zone map of every
    /// version a reader may still see, which holds no value of its own.
    bound: Bound,
}

/// What the values of a hot column, in every version a reader may still
/// see, lie between: a zone map the planner prunes filters with. It names
/// the rows that hold the least and the greatest value and owns no value
/// (#712), so a value taken out or rolled back is not kept by the bound. A
/// side whose row is written with a value less extreme there, or taken
/// out, goes stale (`None`) and prunes nothing, while the other side still
/// prunes its half; a collect that leaves no kept version rebuilds it from
/// the live values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Bound {
    /// No value yet.
    #[default]
    Empty,
    /// The rows of the least and the greatest value, each `None` once
    /// stale: later values do not narrow a stale side.
    Range {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Values that do not order against each other: anything may match.
    Any,
}

impl Bound {
    /// `self` with the value of `row` in `data`, which holds it, taken in;
    /// `self` names no row whose value the write replaced unless
    /// [`written`](Self::written) kept it.
    fn widened(self, data: &Data, row: usize) -> Self {
        use std::cmp::Ordering;

        let (min, max) = match self {
            Self::Any => return self,
            Self::Empty => {
                return Self::Range {
                    min: Some(row),
                    max: Some(row),
                };
            }
            Self::Range { min, max } => (min, max),
        };
        // A side the value does not order against: anything may match.
        let side = |witness: Option<usize>, moves: Ordering| match witness {
            None => Some(None),
            Some(witness) => data
                .compare_rows(row, witness)
                .map(|order| Some(if order == moves { row } else { witness })),
        };
        match (side(min, Ordering::Less), side(max, Ordering::Greater)) {
            (Some(min), Some(max)) => Self::Range { min, max },
            _ => Self::Any,
        }
    }

    /// What is left of `self` once `value` replaces the value of `row` in
    /// `data`, read before the write: a side whose row it is stays when the
    /// value is at least as extreme there (the one it replaces lies inside
    /// the new range), and goes stale otherwise; a row that is both sides
    /// keeps them only with an equal value, as other rows and readers may
    /// still hold the old one.
    fn written(self, data: &Data, row: usize, value: &Value) -> Self {
        use std::cmp::Ordering;

        let Self::Range { min, max } = self else {
            return self;
        };
        let (on_min, on_max) = (min == Some(row), max == Some(row));
        if !on_min && !on_max {
            return self;
        }
        let order = data.with_value(row, |old| super::read::compare_for_range(value, old));
        let keeps = |moves: Ordering| matches!(order, Some(order) if order == moves || order == Ordering::Equal);
        if on_min && on_max {
            return if order == Some(Ordering::Equal) {
                self
            } else {
                Self::Range {
                    min: None,
                    max: None,
                }
            };
        }
        Self::Range {
            min: min.filter(|_| !on_min || keeps(Ordering::Less)),
            max: max.filter(|_| !on_max || keeps(Ordering::Greater)),
        }
    }

    /// What is left of `self` once the value of `row` is taken out: a side
    /// whose row it is goes stale.
    fn taken(self, row: usize) -> Self {
        match self {
            Self::Range { min, max } => Self::Range {
                min: min.filter(|&witness| witness != row),
                max: max.filter(|&witness| witness != row),
            },
            Self::Empty | Self::Any => self,
        }
    }
}

/// Whether `value` is a null, which never compares and never raises, or of
/// `kind`'s one kind.
fn value_holds_only_kind(
    value: &Value,
    kind: crate::execution::operators::value_utils::ValueKinds,
) -> bool {
    kind.is_one()
        && (matches!(value, Value::Null)
            || crate::execution::operators::value_utils::ValueKinds::of(value) == kind)
}

/// Whether every value a hot column's bound covers is of `kind`'s kind or a
/// null: `None` when the bound saw values that do not order, or a side is
/// stale, which says nothing about the kinds.
fn bound_holds_only_kind(
    hot: &Hot,
    kind: crate::execution::operators::value_utils::ValueKinds,
) -> Option<bool> {
    match hot.bound {
        Bound::Empty => Some(true),
        Bound::Range {
            min: Some(min),
            max: Some(max),
        } => Some(
            hot.data
                .with_value(min, |min| value_holds_only_kind(min, kind))
                && hot
                    .data
                    .with_value(max, |max| value_holds_only_kind(max, kind)),
        ),
        Bound::Range { .. } | Bound::Any => None,
    }
}

impl Hot {
    /// The least and the greatest value the bound covers, when it is a
    /// range with no stale side.
    fn bound_range(&self) -> Option<(Value, Value)> {
        match self.bound {
            Bound::Range {
                min: Some(min),
                max: Some(max),
            } => Some((self.data.get(min), self.data.get(max))),
            Bound::Empty | Bound::Range { .. } | Bound::Any => None,
        }
    }

    /// Whether a value the bound covers may meet `value` by `op`: a stale
    /// side bounds nothing.
    fn might_match(&self, op: crate::graph::lpg::CompareOp, value: &Value) -> bool {
        use std::cmp::Ordering;

        use crate::graph::lpg::CompareOp;

        let (min, max) = match self.bound {
            Bound::Empty => return false,
            Bound::Any => return true,
            Bound::Range { min, max } => (min, max),
        };
        // How `value` orders against a live side; `Err` when it does not.
        let side = |witness: Option<usize>| match witness {
            None => Ok(None),
            Some(witness) => self
                .data
                .with_value(witness, |bound| {
                    super::read::compare_for_range(value, bound)
                })
                .map(Some)
                .ok_or(()),
        };
        let (Ok(to_min), Ok(to_max)) = (side(min), side(max)) else {
            return true;
        };
        // Whether the side, when live, allows the match.
        let at_min = |allows: fn(Ordering) -> bool| to_min.is_none_or(allows);
        let at_max = |allows: fn(Ordering) -> bool| to_max.is_none_or(allows);
        match op {
            CompareOp::Eq => {
                at_min(|order| order != Ordering::Less)
                    && at_max(|order| order != Ordering::Greater)
            }
            CompareOp::Ne => !(to_min == Some(Ordering::Equal) && to_max == Some(Ordering::Equal)),
            CompareOp::Lt => at_min(|order| order == Ordering::Greater),
            CompareOp::Le => at_min(|order| order != Ordering::Less),
            CompareOp::Gt => at_max(|order| order == Ordering::Less),
            CompareOp::Ge => at_max(|order| order != Ordering::Greater),
        }
    }

    /// Builds a bound with a stale side anew from the live values, once no
    /// version is kept that a reader could see instead: the live side comes
    /// out as it was (its row holds the extreme of the live values).
    fn rebuild_stale_bound(&mut self) {
        let stale =
            matches!(self.bound, Bound::Range { min, max } if min.is_none() || max.is_none());
        if stale && self.updates.is_empty() {
            self.bound = self
                .present
                .rows()
                .fold(Bound::Empty, |bound, row| bound.widened(&self.data, row));
        }
    }

    /// The value at `row`, if it has one.
    fn get(&self, row: usize) -> Option<Value> {
        self.present.get(row).then(|| self.data.get(row))
    }

    /// Sets the value at `row`.
    fn set(&mut self, row: usize, value: &Value) {
        let replaced = self.data.payload_at(row);
        self.set_replacing(row, value, replaced);
    }

    /// [`set`](Self::set) of a row whose value holds `replaced` bytes (see
    /// [`Data::payload_at`]), which the caller measured: each value is
    /// measured once.
    fn set_replacing(&mut self, row: usize, value: &Value, replaced: usize) {
        if self.present.get(row) {
            self.bound = self.bound.written(&self.data, row, value);
        }
        if self.data.put(row, value) {
            self.payload = super::sub_count(self.payload + self.data.payload_at(row), replaced);
        } else {
            if self.present.is_empty() {
                // Every row was taken out: the values hold nothing.
                self.data = Data::for_value(value);
            } else {
                // Another type: every row's value moves to a column of any
                // values.
                let mut values = GroupRows::default();
                for present in self.present.rows() {
                    values.put(present, self.data.get(present), Value::Null);
                }
                self.data = Data::Values(values);
            }
            self.payload = self.data.payload_bytes();
            let replaced = self.data.payload_at(row);
            let typed = self.data.put(row, value);
            debug_assert!(typed, "a column of any values takes every value");
            self.payload = super::sub_count(self.payload + self.data.payload_at(row), replaced);
        }
        self.present.set(row);
        self.bound = self.bound.widened(&self.data, row);
    }

    /// Takes the value at `row` out, if it has one.
    fn take(&mut self, row: usize) -> Option<Value> {
        let replaced = self.data.payload_at(row);
        self.take_replacing(row, replaced)
    }

    /// [`take`](Self::take) of a row whose value holds `replaced` bytes,
    /// which the caller measured.
    fn take_replacing(&mut self, row: usize, replaced: usize) -> Option<Value> {
        let value = self.get(row)?;
        self.bound = self.bound.taken(row);
        self.present.clear(row);
        self.payload = super::sub_count(self.payload, replaced);
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

    /// Gives back the value slots below `head` when they are due (see
    /// [`trims`](super::slots::trims)); the presence bits stay.
    fn trim_dead_head(&mut self, head: usize) -> usize {
        match &mut self.data {
            Data::Int64(values) => values.trim_dead_head(head),
            Data::Float64(values) => values.trim_dead_head(head),
            Data::Bool(_) => 0,
            Data::String(values) => values.trim_dead_head(head),
            Data::Values(values) => values.trim_dead_head(head),
        }
    }

    /// The heap bytes of its presence bits, its values and what they hold
    /// (the update chains are counted with the versions): a walk of every
    /// value, for the oracle.
    fn heap_bytes(&self) -> usize {
        self.present.heap_bytes() + self.data.heap_bytes() + self.data.payload_bytes()
    }

    /// [`heap_bytes`](Self::heap_bytes) from what the column keeps as its
    /// values change: a few loads.
    fn counted_bytes(&self) -> usize {
        self.present.heap_bytes() + self.data.heap_bytes() + self.payload
    }

    /// The decoded values of `chunks`.
    fn decode(chunks: &[ColdChunk]) -> Result<Self> {
        let mut hot = Self {
            present: Bitset::default(),
            // Typed by its first value (see `set`), as a written column is.
            data: Data::Int64(GroupRows::default()),
            payload: 0,
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
    point: Option<Arc<PointView>>,
    home: Home,
    /// What it takes decoded ([`decoded_size`]).
    decoded: usize,
}

impl ColdChunk {
    /// What the chunk takes decoded (see [`decoded_size`]), for the grant a
    /// thaw of it asks for.
    fn decoded_bytes(&self) -> usize {
        self.decoded
    }
}

/// How a vector grows as [`grow_rows`] (or a [`Bitset`]) grows it to a
/// length at a time: as the standard library's amortized growth does, to
/// twice its capacity, at least the length asked for, at least four.
#[derive(Debug, Default, Clone, Copy)]
struct Growth {
    len: usize,
    cap: usize,
}

impl Growth {
    fn to(&mut self, len: usize) {
        if len > self.len {
            if len > self.cap {
                self.cap = (self.cap * 2).max(len).max(4);
            }
            self.len = len;
        }
    }
}

/// What a chunk's `values` (its rows' offsets from `first_row`, in order,
/// none null) take decoded in a column of their own, as
/// [`Hot::counted_bytes`] counts a [`Hot::decode`] of the chunk: the
/// vector of values to the last row and the presence bits as they grow,
/// what the values hold, and the bounds' values. Measured from the decode a
/// load makes to check the chunk, or the values a spill encodes.
pub(super) fn decoded_size(first_row: usize, values: &[(u32, Value)]) -> usize {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Int64,
        Float64,
        Bool,
        String,
        Any,
    }
    let kind_of = |value: &Value| match value {
        Value::Int64(_) => Kind::Int64,
        Value::Float64(_) => Kind::Float64,
        Value::Bool(_) => Kind::Bool,
        Value::String(_) => Kind::String,
        _ => Kind::Any,
    };
    let held = |kind: Kind, value: &Value| match (kind, value) {
        (Kind::String, Value::String(text)) => text.len(),
        (Kind::Any, value) => value.estimated_size_bytes(),
        _ => 0,
    };
    let mut kind: Option<Kind> = None;
    let mut data = Growth::default();
    let mut present = Growth::default();
    let mut payload = 0;
    // A typed column's values start at the vector of its first row with a
    // value (see [`GroupRows::put`]): a put past the end fills to the row,
    // then pushes it.
    let start = values
        .first()
        .map_or(0, |(offset, _)| vector_start(first_row + *offset as usize));
    let put = |data: &mut Growth, row: usize| {
        data.to(row - start);
        data.to(row - start + 1);
    };
    for (index, (offset, value)) in values.iter().enumerate() {
        let row = first_row + *offset as usize;
        let this = kind_of(value);
        let current = match kind {
            None => this,
            Some(current) if current == this || current == Kind::Any => current,
            // Another type: the rows so far move to a column of any values,
            // rebuilt row by row.
            Some(_) => {
                data = Growth::default();
                for (offset, _) in &values[..index] {
                    put(&mut data, first_row + *offset as usize);
                }
                payload = values[..index]
                    .iter()
                    .map(|(_, value)| value.estimated_size_bytes())
                    .sum();
                Kind::Any
            }
        };
        kind = Some(current);
        match current {
            // A false is a bit left unset: nothing grows.
            Kind::Bool => {
                if matches!(value, Value::Bool(true)) {
                    data.to(row / 64 + 1);
                }
            }
            _ => put(&mut data, row),
        }
        payload += held(current, value);
        present.to(row / 64 + 1);
    }
    let slot = match kind {
        Some(Kind::Int64) => std::mem::size_of::<i64>(),
        Some(Kind::Float64) => std::mem::size_of::<f64>(),
        Some(Kind::Bool) => std::mem::size_of::<u64>(),
        Some(Kind::String) => std::mem::size_of::<ArcStr>(),
        Some(Kind::Any) => std::mem::size_of::<Value>(),
        None => 0,
    };
    present.cap * std::mem::size_of::<u64>() + data.cap * slot + payload
}

/// The heap bytes of a cold column's chunk list and point views, besides
/// the chunks' bytes.
fn cold_meta_bytes(cold: &Vec<ColdChunk>) -> usize {
    cold.capacity() * std::mem::size_of::<ColdChunk>()
        + cold
            .iter()
            .map(|chunk| {
                chunk.point.as_ref().map_or(0, |point| {
                    std::mem::size_of::<PointView>() + point.index_heap_bytes()
                })
            })
            .sum::<usize>()
}

/// The bytes of a cold column's chunks in memory (a spilled chunk's are in
/// its file).
fn cold_chunk_bytes(cold: &[ColdChunk]) -> usize {
    cold.iter()
        .map(|chunk| match chunk.home {
            Home::Memory => chunk.bytes.len(),
            Home::Spilled => 0,
        })
        .sum()
}

/// A column as a read of all its rows finds it (see [`Columns::read`]).
pub(super) enum ColumnRead<'a> {
    /// Decoded.
    Hot(HotRows<'a>),
    /// Still encoded: what to read, each chunk's bytes held apart from the column.
    Cold(Vec<ColdRows>),
}

/// A hot column's values, as far as they compare by their kind.
pub(super) enum HotValues<'a> {
    /// Integers by row: only the rows with a value mean anything.
    Int64(&'a GroupRows<i64>),
    /// Floats by row: only the rows with a value mean anything.
    Float64(&'a GroupRows<f64>),
    /// Strings by row: only the rows with a value mean anything.
    String(&'a GroupRows<ArcStr>),
    /// Anything else: read a row with [`HotRows::get`].
    Other,
}

/// A hot column, read.
pub(super) struct HotRows<'a>(&'a Hot);

impl<'a> HotRows<'a> {
    /// The rows with a newest value.
    pub(super) fn present(&self) -> &'a Bitset {
        &self.0.present
    }

    /// Its values.
    pub(super) fn values(&self) -> HotValues<'a> {
        match &self.0.data {
            Data::Int64(values) => HotValues::Int64(values),
            Data::Float64(values) => HotValues::Float64(values),
            Data::String(values) => HotValues::String(values),
            Data::Bool(_) | Data::Values(_) => HotValues::Other,
        }
    }

    /// The newest value at `row`, if it has one.
    pub(super) fn get(&self, row: usize) -> Option<Value> {
        self.0.get(row)
    }

    /// Whether a value the column held in any version a reader may see may equal one of `sorted`,
    /// integers: one search of a long list, by the column's bounds.
    pub(super) fn might_hold_an_integer_of(&self, sorted: &[i64]) -> bool {
        match self.0.bound_range() {
            Some((Value::Int64(min), Value::Int64(max))) => sorted
                .get(sorted.partition_point(|target| *target < min))
                .is_some_and(|target| *target <= max),
            _ => sorted
                .iter()
                .any(|target| self.might_equal(&Value::Int64(*target))),
        }
    }

    /// [`might_hold_an_integer_of`](Self::might_hold_an_integer_of), of strings.
    pub(super) fn might_hold_a_string_of(&self, sorted: &[ArcStr]) -> bool {
        match self.0.bound_range() {
            Some((Value::String(min), Value::String(max))) => sorted
                .get(sorted.partition_point(|target| *target < min))
                .is_some_and(|target| *target <= max),
            _ => sorted
                .iter()
                .any(|target| self.might_equal(&Value::String(target.clone()))),
        }
    }

    /// Whether a value the column held in any version a reader may see may equal `value`, of the
    /// kind of the column's values.
    pub(super) fn might_equal(&self, value: &Value) -> bool {
        self.0.might_match(crate::graph::lpg::CompareOp::Eq, value)
    }
}

/// A cold chunk, read: its bytes outlive the column's lock, and a thaw.
pub(super) struct ColdRows {
    first_row: usize,
    row_count: u32,
    codec: u8,
    bytes: Bytes,
    point: Option<Arc<PointView>>,
}

impl ColdRows {
    /// The rows of the row group it covers.
    pub(super) fn rows(&self) -> Range<usize> {
        self.first_row..self.first_row + self.row_count as usize
    }

    /// The zone map its header holds.
    ///
    /// # Errors
    ///
    /// Returns the error of a header it cannot read.
    pub(super) fn zone_map(&self) -> Result<Option<ZoneMap>> {
        read_zone_map(&self.bytes, self.codec, self.row_count)
    }

    /// Its point view, for a chunk of values of one fixed size.
    pub(super) fn point(&self) -> Option<&PointView> {
        self.point.as_deref()
    }

    /// Its dictionary, read in place, for a chunk of strings, asking `finds` of each entry's bytes.
    ///
    /// # Errors
    ///
    /// Returns the error of a layout it cannot read.
    pub(super) fn dict(&self, finds: impl FnMut(&[u8]) -> bool) -> Result<Option<DictView>> {
        DictView::read_validated(&self.bytes, self.codec, self.row_count, finds)
    }

    /// Its values, decoded. The load or the spill checked the chunk.
    ///
    /// # Errors
    ///
    /// Returns the error of a chunk it cannot decode.
    pub(super) fn decode(&self) -> Result<ColumnChunk> {
        decode_checked_chunk(&self.bytes, self.codec, self.row_count)
    }
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
    /// What the chunk takes decoded ([`decoded_size`]).
    decoded: usize,
}

impl SpillChunks {
    fn new(resident: usize) -> Self {
        Self {
            bytes: Vec::new(),
            places: Vec::new(),
            resident,
        }
    }

    fn push(&mut self, first_row: usize, row_count: u32, codec: u8, bytes: &[u8], decoded: usize) {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        self.places.push(ChunkPlace {
            first_row,
            row_count,
            codec,
            range: start..self.bytes.len(),
            decoded,
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
pub(super) struct Column {
    /// The chunks not decoded yet; emptied when the hot form is published.
    cold: Mutex<Vec<ColdChunk>>,
    /// The decoded values, set when a caller needs the hot form.
    hot: OnceLock<Hot>,
    /// Where a thaw reports what it gained and freed (`account`); `None`
    /// for a column that was never cold.
    thaws: Option<ThawSink>,
}

/// Its chunks and its decoded values, without where its thaw reports: what
/// it holds, which two stores with the same values show alike.
impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Column")
            .field("cold", &self.cold)
            .field("hot", &self.hot)
            .finish()
    }
}

/// What a column holds, as the account counts it: its decoded bytes, its
/// cold chunks' metadata and bytes, and its update chains.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct ColumnCounts {
    /// The decoded values, with the columns' entries.
    pub(super) hot: usize,
    /// The cold chunks' metadata.
    pub(super) cold: usize,
    /// The cold chunks' bytes in memory.
    pub(super) chunks: usize,
    /// The update chains.
    pub(super) updates: usize,
}

impl ColumnCounts {
    /// Adds what changed from `before` to `after`. The counts move by
    /// differences, so they wrap (see `account`).
    fn moved(&mut self, before: Self, after: Self) {
        self.hot = self.hot.wrapping_add(after.hot.wrapping_sub(before.hot));
        self.cold = self.cold.wrapping_add(after.cold.wrapping_sub(before.cold));
        self.chunks = self
            .chunks
            .wrapping_add(after.chunks.wrapping_sub(before.chunks));
        self.updates = self
            .updates
            .wrapping_add(after.updates.wrapping_sub(before.updates));
    }
}

impl Column {
    /// An empty column for values like `value`.
    fn for_value(value: &Value) -> Self {
        Self {
            cold: Mutex::new(Vec::new()),
            hot: OnceLock::from(Hot {
                present: Bitset::default(),
                data: Data::for_value(value),
                payload: 0,
                updates: Updates::default(),
                bound: Bound::Empty,
            }),
            thaws: None,
        }
    }

    /// An empty column that takes cold chunks, reporting its thaw to
    /// `thaws`.
    fn cold(thaws: Option<ThawSink>) -> Self {
        Self {
            cold: Mutex::new(Vec::new()),
            hot: OnceLock::new(),
            thaws,
        }
    }

    /// What it holds now, as the account counts it: a few loads while it
    /// is hot, its chunks' figures while it is cold.
    fn counts(&self) -> ColumnCounts {
        let of_hot = |hot: &Hot| ColumnCounts {
            hot: hot.counted_bytes(),
            updates: hot.updates.counted_bytes(),
            ..ColumnCounts::default()
        };
        if let Some(hot) = self.hot.get() {
            return of_hot(hot);
        }
        let cold = self.cold.lock();
        // Under the chunks' lock: a thaw publishes the hot form first.
        match self.hot.get() {
            Some(hot) => of_hot(hot),
            None => ColumnCounts {
                cold: cold_meta_bytes(&cold),
                chunks: cold_chunk_bytes(&cold),
                ..ColumnCounts::default()
            },
        }
    }

    /// The hot form, thawing the column (a thaw the store's thaws count),
    /// with what it holds as the account counts it.
    fn hot_counted(&mut self) -> (&mut Hot, ColumnCounts) {
        let hot = self.hot_mut();
        let counts = ColumnCounts {
            hot: hot.counted_bytes(),
            updates: hot.updates.counted_bytes(),
            ..ColumnCounts::default()
        };
        (hot, counts)
    }

    /// Reports the thaw of `cold` into `hot` to the store's thaws, with the
    /// grant a read took for it.
    fn report_thaw(&self, cold: &Vec<ColdChunk>, hot: &Hot, granted: Option<MemoryGrant>) {
        if let Some(thaws) = &self.thaws {
            thaws.thawed(
                hot.counted_bytes() + hot.updates.counted_bytes(),
                cold_meta_bytes(cold),
                cold_chunk_bytes(cold),
                granted,
            );
        }
    }

    /// The decoded values for a read, thawing the column when the memory
    /// limit has room for them; `None` when it has not (C4): the read then
    /// decodes the one chunk it needs ([`decode_one`](Self::decode_one)),
    /// slower but within the limit.
    fn hot_for_read(&self) -> Option<&Hot> {
        if let Some(hot) = self.hot.get() {
            return Some(hot);
        }
        let Some(thaws) = &self.thaws else {
            return Some(self.hot());
        };
        match thaws.grant_thaw(self.thaw_bytes()) {
            ThawGrant::Refused => None,
            grant => Some(self.hot_granted(grant.into_grant())),
        }
    }

    /// The value at `row`, decoded from its chunk alone (a thaw the memory
    /// limit refused).
    fn decode_one(&self, row: usize) -> Option<Value> {
        let cold = self.cold.lock();
        if let Some(hot) = self.hot.get() {
            return hot.get(row);
        }
        let chunk = cold.iter().find(|chunk| {
            row.checked_sub(chunk.first_row)
                .is_some_and(|offset| offset < chunk.row_count as usize)
        })?;
        let offset = u32::try_from(row - chunk.first_row).ok()?;
        // The load decoded each chunk once to check it, and a spill checked
        // that its file holds the chunks it encoded.
        let decoded = decode_column_chunk_bytes(&chunk.bytes, chunk.codec, chunk.row_count)
            .expect("a chunk the load or the spill checked");
        decoded
            .values
            .into_iter()
            .find_map(|(at, value)| (at == offset).then_some(value))
    }

    /// The column as a read of all its rows finds it.
    fn read(&self) -> ColumnRead<'_> {
        if let Some(hot) = self.hot.get() {
            return ColumnRead::Hot(HotRows(hot));
        }
        let cold = self.cold.lock();
        // Under the chunks' lock: a thaw publishes the hot form first.
        match self.hot.get() {
            Some(hot) => ColumnRead::Hot(HotRows(hot)),
            None => {
                let mut chunks: Vec<ColdRows> = cold
                    .iter()
                    .map(|chunk| ColdRows {
                        first_row: chunk.first_row,
                        row_count: chunk.row_count,
                        codec: chunk.codec,
                        bytes: chunk.bytes.clone(),
                        point: chunk.point.clone(),
                    })
                    .collect();
                chunks.sort_unstable_by_key(|chunk| chunk.first_row);
                ColumnRead::Cold(chunks)
            }
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
    /// `held` is what the value at `row` holds on the heap (see
    /// [`Data::payload_at`]), which the caller measured.
    fn keep(&mut self, row: usize, keep: Option<(u64, u64)>, held: usize) -> PendingVersion {
        let Some((stamp, snapshot)) = keep else {
            return PendingVersion::Created;
        };
        let hot = self.hot_mut();
        let current = hot.get(row);
        if hot
            .updates
            .keep_with(row, stamp, snapshot, || current, |_| held)
        {
            PendingVersion::Created
        } else {
            PendingVersion::Replaced
        }
    }

    /// The decoded values, decoding the cold chunks when needed (a grant
    /// whatever the limit: a write's admission covered it).
    fn hot(&self) -> &Hot {
        self.hot_granted(None)
    }

    /// [`hot`](Self::hot), the thaw holding `granted` when it is this
    /// call's (a read's grant; dropped, so given back, when another call
    /// thawed the column first).
    fn hot_granted(&self, granted: Option<MemoryGrant>) -> &Hot {
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
            self.report_thaw(&cold, &hot, granted);
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
        let cold = std::mem::take(self.cold.get_mut());
        let hot = match Hot::decode(&cold) {
            Ok(hot) => hot,
            Err(error) => {
                *self.cold.get_mut() = cold;
                return Err(error);
            }
        };
        self.report_thaw(&cold, &hot, None);
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
                match self.hot_for_read() {
                    Some(hot) => hot.present.get(row),
                    None => self.decode_one(row).is_some(),
                }
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
        let mut chunks = SpillChunks::new(hot.counted_bytes());
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
            let decoded = decoded_size(first_row, window);
            window.clear();
            if !PointView::supports(codec) {
                return Ok(false);
            }
            chunks.push(first_row, rows, codec.to_byte(), &bytes, decoded);
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
            chunks.push(
                chunk.first_row,
                chunk.row_count,
                chunk.codec,
                &chunk.bytes,
                chunk.decoded,
            );
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
    fn install_spill(
        &mut self,
        chunks: SpillChunks,
        mapped: &Bytes,
        thaws: &ThawSink,
    ) -> Result<()> {
        if mapped.as_ref() != chunks.bytes.as_slice() {
            return Err(Error::Internal(
                "a spill file does not hold the chunks written to it".to_string(),
            ));
        }
        self.install_chunks(chunks.places, mapped, Home::Spilled, thaws)
    }

    /// Makes the column cold, its chunks `chunks` (from
    /// [`plan_spill`](Self::plan_spill), with no write since) kept on the
    /// heap, holding a grant of their bytes from `memory`: compression on
    /// request (`compress`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`], changing nothing, when a chunk gets no
    /// point view.
    fn install_in_memory(
        &mut self,
        chunks: SpillChunks,
        memory: Option<&Arc<BufferManager>>,
        thaws: &ThawSink,
    ) -> Result<()> {
        let SpillChunks { bytes, places, .. } = chunks;
        let bytes = granted_chunk(Bytes::from(bytes), memory);
        self.install_chunks(places, &bytes, Home::Memory, thaws)
    }

    /// Makes the column cold, the chunk of each of `places` read from its
    /// range of `bytes`, which are kept at `home`; a later thaw reports to
    /// `thaws`.
    fn install_chunks(
        &mut self,
        places: Vec<ChunkPlace>,
        bytes: &Bytes,
        home: Home,
        thaws: &ThawSink,
    ) -> Result<()> {
        let mut cold = Vec::with_capacity(places.len());
        for place in places {
            let bytes = bytes.slice(place.range);
            // The bytes are those the encoder wrote or the load checked.
            let point = PointView::from_validated(&bytes, place.codec, place.row_count)?
                .ok_or_else(|| Error::Internal("a cold chunk without a point view".to_string()))?;
            cold.push(ColdChunk {
                first_row: place.first_row,
                row_count: place.row_count,
                codec: place.codec,
                bytes,
                point: Some(Arc::new(point)),
                home,
                decoded: place.decoded,
            });
        }
        *self.cold.get_mut() = cold;
        self.hot = OnceLock::new();
        if self.thaws.is_none() {
            self.thaws = Some(thaws.clone());
        }
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
        match self.hot_for_read() {
            Some(hot) => hot.get(row),
            None => self.decode_one(row),
        }
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
    #[cfg(test)]
    fn cold_bytes(&self) -> usize {
        self.cold_meta_bytes() + self.cold_chunk_bytes()
    }

    /// The metadata of its cold chunks (see [`cold_bytes`](Self::cold_bytes)).
    fn cold_meta_bytes(&self) -> usize {
        cold_meta_bytes(&self.cold.lock())
    }

    /// The bytes of its cold chunks in memory.
    fn cold_chunk_bytes(&self) -> usize {
        cold_chunk_bytes(&self.cold.lock())
    }

    /// The heap bytes a spill of it would move to disk, as
    /// [`plan_spill`](Self::plan_spill) decides: a hot column's values when
    /// it holds any and has no update chains (a chained one stays), a cold
    /// one's chunks in memory when each has a point view, none once it is
    /// spilled.
    fn spillable_bytes(&self) -> usize {
        let hot_bytes = |hot: &Hot| {
            if hot.updates.is_empty() && !hot.present.is_empty() {
                hot.counted_bytes()
            } else {
                0
            }
        };
        if let Some(hot) = self.hot.get() {
            return hot_bytes(hot);
        }
        let cold = self.cold.lock();
        match self.hot.get() {
            // A thaw published the hot form meanwhile.
            Some(hot) => hot_bytes(hot),
            None if cold
                .iter()
                .any(|chunk| chunk.home == Home::Spilled || chunk.point.is_none()) =>
            {
                0
            }
            None => cold_chunk_bytes(&cold),
        }
    }

    /// About what a thaw of it decodes: 0 for a hot column.
    fn thaw_bytes(&self) -> usize {
        if self.hot.get().is_some() {
            return 0;
        }
        self.cold.lock().iter().map(ColdChunk::decoded_bytes).sum()
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

/// What a compression of a row group's columns encodes (see
/// [`Columns::plan_compress`]).
pub(super) struct CompressPlan {
    /// The hot columns to make cold, each with its chunks.
    columns: Vec<(u32, SpillChunks)>,
    kept_with_chains: usize,
    kept_unsupported: usize,
}

impl CompressPlan {
    /// Whether it encodes no column.
    pub(super) fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The columns it leaves hot, as a [`Compressed`] that encoded none.
    pub(super) fn counts(&self) -> Compressed {
        Compressed {
            kept_with_chains: self.kept_with_chains,
            kept_unsupported: self.kept_unsupported,
            ..Compressed::default()
        }
    }
}

/// What a compression did to a row group's columns.
#[derive(Debug, Default)]
pub(super) struct Compressed {
    /// The columns made cold.
    pub(super) columns: usize,
    /// The columns left hot for their update chains.
    pub(super) kept_with_chains: usize,
    /// The columns left hot for a chunk without a point view.
    pub(super) kept_unsupported: usize,
    /// The heap bytes the columns made cold held, less what their chunks
    /// hold.
    pub(super) freed_bytes: usize,
}

/// The column of `key`, made for values like `value` when there is none;
/// whether it was made. One lookup.
fn column_of<'a>(
    by_key: &'a mut FxHashMap<u32, Column>,
    key: u32,
    value: &Value,
) -> (&'a mut Column, bool) {
    match by_key.entry(key) {
        hashbrown::hash_map::Entry::Occupied(entry) => (entry.into_mut(), false),
        hashbrown::hash_map::Entry::Vacant(entry) => (entry.insert(Column::for_value(value)), true),
    }
}

/// The columns of a row group, by property key id.
///
/// Each write measures the column it changes before and after, hot (after
/// any thaw, which the store's thaws count), and adds the difference to
/// what it counts ([`counts`](Self::counts)); the account reads that count
/// before and after an op.
#[derive(Debug, Default)]
pub(super) struct Columns {
    by_key: FxHashMap<u32, Column>,
    /// What the writes moved its columns by, thaws aside (see `account`).
    counted: ColumnCounts,
}

impl Columns {
    /// What its columns hold as the account counts it: what the writes
    /// moved them by, with the columns' entries. Read before and after a
    /// write, the difference is the write's.
    pub(super) fn counts(&self) -> ColumnCounts {
        ColumnCounts {
            hot: self.counted.hot.wrapping_add(self.entries_bytes()),
            ..self.counted
        }
    }

    /// The heap bytes of the columns' entries in the map: its whole table as hashbrown
    /// allocates it (buckets, padded entries, control bytes).
    fn entries_bytes(&self) -> usize {
        self.by_key.allocation_size()
    }

    /// What a compression of the hot columns encodes: each hot column with
    /// values, no update chains and only kinds with a point view, a vector
    /// of rows a chunk (as a spill encodes it). Cold columns stay as they
    /// are.
    ///
    /// # Errors
    ///
    /// Returns the encoder's error for a value the format cannot hold.
    pub(super) fn plan_compress(&self) -> Result<CompressPlan> {
        let mut plan = CompressPlan {
            columns: Vec::new(),
            kept_with_chains: 0,
            kept_unsupported: 0,
        };
        for (key, column) in &self.by_key {
            if column.is_cold() {
                continue;
            }
            match column.plan_spill()? {
                SpillPlan::Nothing => {}
                SpillPlan::Chained => plan.kept_with_chains += 1,
                SpillPlan::Unsupported => plan.kept_unsupported += 1,
                SpillPlan::Write(chunks) => plan.columns.push((*key, chunks)),
            }
        }
        Ok(plan)
    }

    /// Makes the columns of `plan` (planned with no write since) cold, their
    /// chunks on the heap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when a planned column is gone or a chunk
    /// gets no point view; the columns before it stay cold.
    pub(super) fn install_compressed(
        &mut self,
        plan: CompressPlan,
        memory: Option<&Arc<BufferManager>>,
        thaws: &ThawSink,
    ) -> Result<Compressed> {
        let mut done = plan.counts();
        for (key, chunks) in plan.columns {
            let column = self
                .by_key
                .get_mut(&key)
                .ok_or_else(|| Error::Internal("a compressed column is gone".to_string()))?;
            let resident = chunks.resident;
            let before = column.counts();
            column.install_in_memory(chunks, memory, thaws)?;
            let after = column.counts();
            self.counted.moved(before, after);
            done.columns += 1;
            done.freed_bytes += resident.saturating_sub(after.cold + after.chunks);
        }
        Ok(done)
    }

    /// The column of key `key` as a read of all its rows finds it, `None` without one. A cold
    /// column stays cold.
    pub(super) fn read(&self, key: u32) -> Option<ColumnRead<'_>> {
        Some(self.by_key.get(&key)?.read())
    }

    /// The value of key `key` at `row`.
    pub(super) fn get(&self, key: u32, row: usize) -> Option<Value> {
        self.by_key.get(&key)?.get(row)
    }

    /// Adds a chunk of key `key` as the file holds it, for rows from
    /// `first_row`. The load has already decoded and validated the chunk.
    #[cfg(test)]
    pub(super) fn add_cold(
        &mut self,
        key: u32,
        first_row: usize,
        row_count: u32,
        codec: u8,
        bytes: Bytes,
    ) {
        let decoded = decode_column_chunk_bytes(&bytes, codec, row_count)
            .map_or(0, |decoded| decoded_size(first_row, &decoded.values));
        self.add_cold_reporting(key, first_row, row_count, codec, bytes, None, decoded);
    }

    /// Adds a chunk of key `key` as the file holds it, for rows from
    /// `first_row`, a new column reporting its thaw to `thaws`. The load has
    /// already decoded and validated the chunk, and measured what it takes
    /// decoded ([`decoded_size`]).
    #[expect(
        clippy::too_many_arguments,
        reason = "a chunk's directory entry, its bytes and the measure of its decode, as the load has them"
    )]
    pub(super) fn add_cold_reporting(
        &mut self,
        key: u32,
        first_row: usize,
        row_count: u32,
        codec: u8,
        bytes: Bytes,
        thaws: Option<ThawSink>,
        decoded: usize,
    ) {
        let column = self
            .by_key
            .entry(key)
            .or_insert_with(|| Column::cold(thaws));
        let before = column.counts();
        if column.is_cold() {
            let point = PointView::from_validated(&bytes, codec, row_count)
                .expect("a chunk the load checked")
                .map(Arc::new);
            column.cold.lock().push(ColdChunk {
                first_row,
                row_count,
                codec,
                bytes,
                point,
                home: Home::Memory,
                decoded,
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
        let after = column.counts();
        self.counted.moved(before, after);
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
            return Some(hot.might_match(op, value));
        }
        // Under the chunks' lock: a thaw publishes the hot form first.
        let cold = column.cold.lock();
        Some(match column.hot.get() {
            Some(hot) => hot.might_match(op, value),
            None => !cold.is_empty(),
        })
    }

    /// Whether every value the column of key `key` holds is of `kind`'s kind
    /// or a null: by the hot column's bound, or by the cold chunks' zone maps.
    /// `None` when a part of the column keeps no bounds; `Some(true)` without a
    /// column of the key, which holds nothing.
    pub(super) fn holds_only_kind(
        &self,
        key: u32,
        kind: crate::execution::operators::value_utils::ValueKinds,
    ) -> Option<bool> {
        let Some(column) = self.by_key.get(&key) else {
            return Some(true);
        };
        if let Some(hot) = column.hot.get() {
            return bound_holds_only_kind(hot, kind);
        }
        // Under the chunks' lock: a thaw publishes the hot form first.
        let cold = column.cold.lock();
        if let Some(hot) = column.hot.get() {
            return bound_holds_only_kind(hot, kind);
        }
        for chunk in cold.iter() {
            let zone = read_zone_map(&chunk.bytes, chunk.codec, chunk.row_count).ok()??;
            let holds = match &zone {
                ZoneMap::Values(min, max) => {
                    value_holds_only_kind(min, kind) && value_holds_only_kind(max, kind)
                }
                // A dictionary chunk holds strings only.
                ZoneMap::Strings(_) => {
                    kind == crate::execution::operators::value_utils::ValueKinds::STRINGS
                }
            };
            if !holds {
                return Some(false);
            }
        }
        Some(true)
    }

    /// The columns that still hold their chunks encoded.
    pub(super) fn cold_count(&self) -> usize {
        self.by_key
            .values()
            .filter(|column| column.is_cold())
            .count()
    }

    /// Sets the value of key `key` at `row`.
    /// Returns whether it made the column.
    pub(super) fn set(&mut self, key: u32, row: usize, value: &Value) -> bool {
        let (column, made) = column_of(&mut self.by_key, key, value);
        let (hot, _) = column.hot_counted();
        let held = hot.counted_bytes();
        hot.set(row, value);
        self.counted.hot = self
            .counted
            .hot
            .wrapping_add(hot.counted_bytes().wrapping_sub(held));
        made
    }

    /// Takes the value of key `key` at `row` out.
    pub(super) fn take(&mut self, key: u32, row: usize) -> Option<Value> {
        let (hot, _) = self.by_key.get_mut(&key)?.hot_counted();
        let held = hot.counted_bytes();
        let taken = hot.take(row);
        self.counted.hot = self
            .counted
            .hot
            .wrapping_add(hot.counted_bytes().wrapping_sub(held));
        taken
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
    ) -> (PendingVersion, bool) {
        let (column, made) = match value {
            Some(value) => column_of(&mut self.by_key, key, value),
            None => match self.by_key.get_mut(&key) {
                Some(column) => (column, false),
                None => return (PendingVersion::Created, false),
            },
        };
        let (hot, before) = column.hot_counted();
        // What the replaced value holds, measured once: for the chain that
        // keeps it and for the values' count.
        let replaced = hot.data.payload_at(row);
        let version = column.keep(row, keep, replaced);
        let hot = column.hot_mut();
        match value {
            Some(value) => hot.set_replacing(row, value, replaced),
            None => {
                hot.take_replacing(row, replaced);
            }
        }
        let (_, after) = column.hot_counted();
        self.counted.moved(before, after);
        (version, made)
    }

    /// Writes `value` back at `row` of key `key` (`None` takes the value
    /// out), keeping nothing: an undo.
    pub(super) fn restore(&mut self, key: u32, row: usize, value: Option<&Value>) {
        match value {
            // The column of a value an undo writes back is there: the write
            // undone was to it.
            Some(value) => {
                self.set(key, row, value);
            }
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
        let (_, before) = column.hot_counted();
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
        let (_, after) = column.hot_counted();
        self.counted.moved(before, after);
        true
    }

    /// Drops the kept values every reader at `horizon` or later sees past;
    /// returns how many it dropped.
    pub(super) fn collect(&mut self, horizon: u64) -> usize {
        let mut dropped = 0;
        for column in self.by_key.values_mut() {
            if column.updates().is_some_and(|updates| !updates.is_empty()) {
                let (_, before) = column.hot_counted();
                dropped += column.updates_mut().collect(horizon);
                let (_, after) = column.hot_counted();
                self.counted.moved(before, after);
            }
            if let Some(hot) = column.hot.get_mut() {
                hot.rebuild_stale_bound();
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
                let (_, before) = column.hot_counted();
                dropped += column.updates_mut().collect_vector(vector, horizon);
                let (_, after) = column.hot_counted();
                self.counted.moved(before, after);
            }
            if let Some(hot) = column.hot.get_mut() {
                hot.rebuild_stale_bound();
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
    /// Gives back each hot column's value slots below `head`, the row
    /// group's first vector with a row in any version, when they are due (see
    /// [`trims`](super::slots::trims)). A cold column holds only the chunks of
    /// its rows and stays as it is. Returns the heap bytes freed.
    pub(super) fn trim_dead_head(&mut self, head: usize) -> usize {
        let mut freed = 0;
        for column in self.by_key.values_mut() {
            if column.hot.get().is_none() {
                continue;
            }
            let (hot, before) = column.hot_counted();
            freed += hot.trim_dead_head(head);
            let (_, after) = column.hot_counted();
            self.counted.moved(before, after);
        }
        freed
    }

    pub(super) fn clear_row(&mut self, row: usize) {
        for column in self.by_key.values_mut() {
            if column.is_cold() && !column.has_value(row) {
                continue;
            }
            let (hot, before) = column.hot_counted();
            hot.take(row);
            if !hot.updates.is_empty() {
                hot.updates.remove_row(row);
            }
            let (_, after) = column.hot_counted();
            self.counted.moved(before, after);
        }
    }

    /// The heap bytes of the update chains.
    pub(super) fn updates_bytes(&self) -> usize {
        self.by_key
            .values()
            .filter_map(Column::updates)
            .map(|updates| {
                updates.heap_bytes(|kept| kept.as_ref().map_or(0, Value::estimated_size_bytes))
            })
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
        let column = self.by_key.entry(key).or_insert_with(|| Column::cold(None));
        let (_, before) = column.hot_counted();
        column.updates_mut().load(kept);
        let (_, after) = column.hot_counted();
        self.counted.moved(before, after);
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
        self.by_key.values().map(Column::hot_bytes).sum::<usize>() + self.entries_bytes()
    }

    /// The heap bytes of the chunks its columns still hold encoded.
    #[cfg(test)]
    pub(super) fn cold_bytes(&self) -> usize {
        self.by_key.values().map(Column::cold_bytes).sum::<usize>()
    }

    /// The metadata of the chunks its columns still hold encoded.
    pub(super) fn cold_meta_bytes(&self) -> usize {
        self.by_key
            .values()
            .map(Column::cold_meta_bytes)
            .sum::<usize>()
    }

    /// The bytes of the chunks its columns still hold encoded, in memory.
    pub(super) fn cold_chunk_bytes(&self) -> usize {
        self.by_key
            .values()
            .map(Column::cold_chunk_bytes)
            .sum::<usize>()
    }

    /// The heap bytes a spill of key `key`'s column would move to disk
    /// (see `Column::spillable_bytes`).
    pub(super) fn spillable_bytes(&self, key: u32) -> usize {
        self.by_key.get(&key).map_or(0, Column::spillable_bytes)
    }

    /// About what a write of key `key` thaws: the decoded bytes of its
    /// column while it is cold.
    pub(super) fn thaw_bytes(&self, key: u32) -> usize {
        self.by_key.get(&key).map_or(0, Column::thaw_bytes)
    }

    /// Sets its count to what its columns hold: after a load that built them
    /// without counting, so its count moves with them from then on.
    pub(super) fn retally(&mut self) {
        let mut counted = ColumnCounts::default();
        for column in self.by_key.values() {
            counted.moved(ColumnCounts::default(), column.counts());
        }
        self.counted = counted;
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
        thaws: &ThawSink,
    ) -> Result<usize> {
        let column = self
            .by_key
            .get_mut(&key)
            .ok_or_else(|| Error::Internal("a spilled column is gone".to_string()))?;
        let before = column.counts();
        column.install_spill(chunks, mapped, thaws)?;
        let after = column.counts();
        self.counted.moved(before, after);
        Ok(after.cold + after.chunks)
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

    use super::{Columns, Data, Hot};
    use crate::codec::column_chunk::{decode_column_chunk_bytes, encode_column_chunk};

    /// What [`super::decoded_size`] says a chunk takes decoded is what a
    /// thaw of it holds, for each kind of column and a column whose rows'
    /// types differ, from a row group's first row and from further in.
    #[test]
    fn a_chunks_decoded_size_is_what_its_thaw_holds() {
        let cases: Vec<Vec<(u32, Value)>> = vec![
            (0..300).map(|i| (i, Value::Int64(i64::from(i)))).collect(),
            (0..70)
                .map(|i| (i * 3, Value::Float64(f64::from(i))))
                .collect(),
            (0..130).map(|i| (i, Value::Bool(i % 3 == 0))).collect(),
            (0..90)
                .map(|i| (i, Value::from(format!("Vincent {i}").as_str())))
                .collect(),
            vec![
                (0, Value::Int64(5)),
                (4, Value::from("Jules")),
                (9, Value::Float64(0.5)),
                (11, Value::Bool(true)),
            ],
        ];
        for first_row in [0, 128] {
            for values in &cases {
                let rows = values.last().unwrap().0 + 1;
                let (codec, bytes) = encode_column_chunk(rows, values, None).unwrap();
                let chunk = super::ColdChunk {
                    first_row,
                    row_count: rows,
                    codec: codec.to_byte(),
                    bytes: Bytes::from(bytes),
                    point: None,
                    home: super::Home::Memory,
                    decoded: 0,
                };
                let hot = Hot::decode(std::slice::from_ref(&chunk)).unwrap();
                assert_eq!(
                    super::decoded_size(first_row, values),
                    hot.counted_bytes(),
                    "{:?} from row {first_row}",
                    values.first()
                );
            }
        }
    }

    fn add_chunk(columns: &mut Columns, first: usize, rows: u32, values: &[(u32, Value)]) {
        let epochs = vec![17; values.len()];
        let (codec, bytes) = encode_column_chunk(rows, values, Some(&epochs)).unwrap();
        let bytes = Bytes::from(bytes);
        decode_column_chunk_bytes(&bytes, codec.to_byte(), rows).unwrap();
        columns.add_cold(7, first, rows, codec.to_byte(), bytes);
    }

    /// The column of key 7, after a write thawed it.
    fn thawed<'a>(columns: &'a mut Columns, row: usize, value: &Value) -> &'a Hot {
        columns.set(7, row, value);
        assert_eq!(columns.cold_count(), 0, "a write thaws the column");
        columns.by_key[&7].hot.get().expect("thawed")
    }

    fn all_rows(rows: u32, value: impl Fn(u32) -> Value) -> Vec<(u32, Value)> {
        (0..rows).map(|row| (row, value(row))).collect()
    }

    /// A thawed column is as typed as one written row by row, so a compare can read it as a slice and
    /// it costs what a typed column costs.
    #[test]
    fn a_thawed_integer_column_is_typed() {
        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            4_096,
            &all_rows(4_096, |row| Value::Int64(i64::from(row))),
        );
        add_chunk(
            &mut columns,
            4_096,
            4_096,
            &all_rows(4_096, |row| Value::Int64(-i64::from(row))),
        );
        let hot = thawed(&mut columns, 8_192, &Value::Int64(5));
        assert!(matches!(hot.data, Data::Int64(_)), "{:?}", hot.data);
        assert!(
            columns.hot_bytes() <= 8_193 * 17 + 64,
            "eight bytes a row, room for as many as a vector doubles, and a presence bit: {} bytes",
            columns.hot_bytes()
        );
        assert_eq!(columns.get(7, 3), Some(Value::Int64(3)));
        assert_eq!(columns.get(7, 4_099), Some(Value::Int64(-3)));
        assert_eq!(columns.get(7, 8_192), Some(Value::Int64(5)));
    }

    #[test]
    fn a_thawed_column_of_each_kind_is_typed() {
        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            64,
            &all_rows(64, |row| Value::Float64(f64::from(row) / 2.0)),
        );
        let hot = thawed(&mut columns, 64, &Value::Float64(0.5));
        assert!(matches!(hot.data, Data::Float64(_)), "{:?}", hot.data);
        assert_eq!(columns.get(7, 3), Some(Value::Float64(1.5)));

        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            64,
            &all_rows(64, |row| Value::Bool(row % 2 == 0)),
        );
        let hot = thawed(&mut columns, 64, &Value::Bool(true));
        assert!(matches!(hot.data, Data::Bool(_)), "{:?}", hot.data);
        assert_eq!(columns.get(7, 3), Some(Value::Bool(false)));

        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            64,
            &all_rows(64, |row| Value::from(format!("k{row}").as_str())),
        );
        let hot = thawed(&mut columns, 64, &Value::from("Alix"));
        assert!(matches!(hot.data, Data::String(_)), "{:?}", hot.data);
        assert_eq!(columns.get(7, 3), Some(Value::from("k3")));
    }

    /// Chunks of two kinds hold each row's own value, whichever kind came first, and a write of
    /// another kind into a typed one moves it to any values with nothing lost.
    #[test]
    fn a_thawed_column_of_mixed_kinds_stays_right() {
        for (first, second) in [
            (Value::Int64(1), Value::from("Gus")),
            (Value::from("Gus"), Value::Int64(1)),
            (Value::Float64(1.5), Value::Int64(1)),
        ] {
            let mut columns = Columns::default();
            add_chunk(&mut columns, 0, 64, &all_rows(64, |_| first.clone()));
            add_chunk(&mut columns, 64, 64, &all_rows(64, |_| second.clone()));
            let written = Value::Bool(true);
            let hot = thawed(&mut columns, 128, &written);
            assert!(matches!(hot.data, Data::Values(_)), "{:?}", hot.data);
            assert_eq!(columns.get(7, 0), Some(first.clone()));
            assert_eq!(columns.get(7, 63), Some(first.clone()));
            assert_eq!(columns.get(7, 64), Some(second.clone()));
            assert_eq!(columns.get(7, 127), Some(second.clone()));
            assert_eq!(columns.get(7, 128), Some(written));
        }
        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            64,
            &all_rows(64, |row| Value::Int64(i64::from(row))),
        );
        let hot = thawed(&mut columns, 3, &Value::from("Gus"));
        assert!(matches!(hot.data, Data::Values(_)), "{:?}", hot.data);
        assert_eq!(columns.get(7, 2), Some(Value::Int64(2)));
        assert_eq!(columns.get(7, 3), Some(Value::from("Gus")));
    }

    /// A chunk with gaps keeps them: a row with no value is not read as a zero, in a typed column.
    #[test]
    fn a_thawed_column_keeps_rows_without_a_value() {
        let values: Vec<(u32, Value)> = (0..64)
            .filter(|row| row % 3 == 0)
            .map(|row| (row, Value::Int64(i64::from(row) + 100)))
            .collect();
        let mut columns = Columns::default();
        add_chunk(&mut columns, 0, 64, &values);
        let hot = thawed(&mut columns, 70, &Value::Int64(1));
        assert!(matches!(hot.data, Data::Int64(_)), "{:?}", hot.data);
        assert_eq!(columns.get(7, 3), Some(Value::Int64(103)));
        assert_eq!(columns.get(7, 4), None);
        assert_eq!(columns.get(7, 65), None);
        assert!(!columns.by_key[&7].hot.get().unwrap().present.get(4));
    }

    /// The bounds hold every value of a thawed column, so a lookup can still prune by them.
    #[test]
    fn a_thawed_column_keeps_its_bounds() {
        use crate::graph::lpg::CompareOp;

        let mut columns = Columns::default();
        add_chunk(
            &mut columns,
            0,
            64,
            &all_rows(64, |row| Value::Int64(i64::from(row))),
        );
        thawed(&mut columns, 64, &Value::Int64(1_000));
        assert_eq!(
            columns.might_match(7, CompareOp::Eq, &Value::Int64(1_000)),
            Some(true)
        );
        assert_eq!(
            columns.might_match(7, CompareOp::Eq, &Value::Int64(5)),
            Some(true)
        );
        assert_eq!(
            columns.might_match(7, CompareOp::Eq, &Value::Int64(1_001)),
            Some(false)
        );
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

    /// The values of `key` every row shows to `read`, in row order.
    fn visible(columns: &Columns, key: u32, read: super::Read) -> Vec<Value> {
        (0..8)
            .filter_map(|row| columns.get_at(key, row, read))
            .collect()
    }

    /// A filter by any operator and probe never skips the column while a value `read` sees
    /// meets it: the bound covers every version a reader may still see.
    fn assert_no_wrong_prune(
        columns: &Columns,
        key: u32,
        read: super::Read,
        probes: &[Value],
        step: &str,
    ) {
        use std::cmp::Ordering;

        use crate::graph::lpg::CompareOp;
        use crate::graph::traits::compare_for_range;

        let values = visible(columns, key, read);
        for probe in probes {
            for op in [
                CompareOp::Eq,
                CompareOp::Ne,
                CompareOp::Lt,
                CompareOp::Le,
                CompareOp::Gt,
                CompareOp::Ge,
            ] {
                let meets = values.iter().any(|value| {
                    compare_for_range(value, probe).is_some_and(|order| match op {
                        CompareOp::Eq => order == Ordering::Equal,
                        CompareOp::Ne => order != Ordering::Equal,
                        CompareOp::Lt => order == Ordering::Less,
                        CompareOp::Le => order != Ordering::Greater,
                        CompareOp::Gt => order == Ordering::Greater,
                        CompareOp::Ge => order != Ordering::Less,
                    })
                });
                if meets {
                    assert_eq!(
                        columns.might_match(key, op, probe),
                        Some(true),
                        "{step}: {read:?} sees {values:?}, one meets {op:?} {probe:?}, \
                         and the bound pruned the column"
                    );
                }
            }
        }
    }

    /// The live least and greatest value of key 0, and what the bound says they are.
    fn live_and_bound(columns: &Columns) -> (Option<(Value, Value)>, Option<(Value, Value)>) {
        use crate::graph::traits::compare_for_range;

        let mut live = visible(columns, 0, super::Read::At(u64::MAX >> 2));
        live.sort_by(|a, b| compare_for_range(a, b).expect("one kind"));
        let live = live.first().cloned().zip(live.last().cloned());
        let hot = columns.by_key[&0].hot();
        (live, hot.bound_range())
    }

    /// The bound of a hot column owns no value (#712): after an overwrite, a removal, a rollback
    /// and a collapse of each extreme, with a reader of an older snapshot open, a filter never
    /// skips the column while a version some reader sees meets it; once the kept versions are
    /// collected, the bound is the live least and greatest value, so it holds no copy of a
    /// deleted or rolled-back one. Run for integers and for strings.
    #[test]
    fn the_bound_covers_every_visible_version_and_then_only_the_live_values() {
        /// The stamp of transaction `id`'s pending writes.
        fn pending(id: u64) -> u64 {
            super::super::PENDING | id
        }

        let families: [(&str, fn(i64) -> Value); 2] = [
            ("integers", Value::Int64),
            ("strings", |rank| Value::from(format!("v{rank:03}"))),
        ];
        for (family, make) in families {
            let mut columns = Columns::default();
            // Rows 0..8 hold 10, 20, ..., 80, written with no reader to keep anything for.
            for (row, rank) in (0..8).zip((10..=80).step_by(10)) {
                columns.write(0, row, Some(&make(rank)), None);
            }
            let probes: Vec<Value> = (0..=20).map(|rank| make(rank * 5)).collect();
            let old = super::Read::At(3);
            let check = |columns: &Columns, now: u64, step: &str| {
                let step = format!("{family}, {step}");
                assert_no_wrong_prune(columns, 0, old, &probes, &step);
                assert_no_wrong_prune(columns, 0, super::Read::At(now), &probes, &step);
            };
            check(&columns, 3, "written");

            // The greatest value (80, row 7) is overwritten with 45 and committed at 4.
            columns.write(0, 7, Some(&make(45)), Some((pending(19), 3)));
            columns.stamp(0, 7, pending(19), 4);
            check(&columns, 4, "the greatest overwritten");

            // The least value (10, row 0) is taken out and committed at 5.
            columns.write(0, 0, None, Some((pending(21), 4)));
            columns.stamp(0, 0, pending(21), 5);
            check(&columns, 5, "the least taken out");

            // A new greatest (99, row 3) is written and rolled back.
            columns.write(0, 3, Some(&make(99)), Some((pending(22), 5)));
            assert!(columns.undo(0, 3, pending(22)), "the rollback");
            check(&columns, 5, "a new greatest rolled back");

            // A new least (5, row 4) is committed at 6, then overwritten with 50 at 7.
            columns.write(0, 4, Some(&make(5)), Some((pending(23), 5)));
            columns.stamp(0, 4, pending(23), 6);
            columns.write(0, 4, Some(&make(50)), Some((pending(24), 6)));
            columns.stamp(0, 4, pending(24), 7);
            check(&columns, 7, "a new least, then overwritten");

            // The old reader ends; every kept version is collected.
            columns.collect(8);
            check(&columns, 8, "collected");
            let (live, bound) = live_and_bound(&columns);
            assert_eq!(bound, live, "{family}: the bound once nothing is kept");

            // The row holding the greatest (70, row 6) is deleted and collapsed.
            columns.clear_row(6);
            columns.collect(9);
            check(&columns, 9, "the greatest's row collapsed");
            let (live, bound) = live_and_bound(&columns);
            assert_eq!(
                live,
                Some((make(20), make(60))),
                "{family}: the live values"
            );
            assert_eq!(bound, live, "{family}: the bound once the row is gone");
        }
    }

    /// A write to the row of the greatest value that leaves it at least the greatest, or to the
    /// row of the least that leaves it at least the least, keeps the bound (a counter or a
    /// timestamp that only grows): the value it replaced lies inside the new range, and the
    /// bound still owns none. A write that moves an extreme inward makes it stale. A row that
    /// is both the least and the greatest keeps it only with a value equal to both.
    #[test]
    fn a_witness_written_with_a_value_still_as_extreme_keeps_the_bound() {
        use crate::graph::lpg::CompareOp;

        fn pending(id: u64) -> u64 {
            super::super::PENDING | id
        }
        let families: [(&str, fn(i64) -> Value); 2] = [
            ("integers", Value::Int64),
            ("strings", |rank| Value::from(format!("v{rank:03}"))),
        ];
        for (family, make) in families {
            let mut columns = Columns::default();
            for (row, rank) in (0..4).zip((10..=40).step_by(10)) {
                columns.write(0, row, Some(&make(rank)), None);
            }
            let probes: Vec<Value> = (0..=20).map(|rank| make(rank * 5)).collect();
            let bound = |columns: &Columns, key: u32| columns.by_key[&key].hot().bound_range();

            // The greatest (40, row 3) grows to 45: the reader at 3 still sees 40, inside.
            columns.write(0, 3, Some(&make(45)), Some((pending(31), 3)));
            columns.stamp(0, 3, pending(31), 4);
            assert_eq!(
                bound(&columns, 0),
                Some((make(10), make(45))),
                "{family}: the greatest grew"
            );
            assert_no_wrong_prune(&columns, 0, super::Read::At(3), &probes, family);
            assert_no_wrong_prune(&columns, 0, super::Read::At(4), &probes, family);

            // The least (10, row 0) falls to 5.
            columns.write(0, 0, Some(&make(5)), Some((pending(32), 4)));
            columns.stamp(0, 0, pending(32), 5);
            assert_eq!(
                bound(&columns, 0),
                Some((make(5), make(45))),
                "{family}: the least fell"
            );

            // The greatest written again with the same value.
            columns.write(0, 3, Some(&make(45)), Some((pending(33), 5)));
            columns.stamp(0, 3, pending(33), 6);
            assert_eq!(
                bound(&columns, 0),
                Some((make(5), make(45))),
                "{family}: the greatest written with itself"
            );

            // The greatest moves inward: that side is stale, so nothing a reader sees is
            // pruned, while the live least side still prunes below 5.
            columns.write(0, 3, Some(&make(44)), Some((pending(34), 6)));
            columns.stamp(0, 3, pending(34), 7);
            assert_eq!(
                bound(&columns, 0),
                None,
                "{family}: the greatest moved inward"
            );
            assert_no_wrong_prune(&columns, 0, super::Read::At(6), &probes, family);
            assert_no_wrong_prune(&columns, 0, super::Read::At(7), &probes, family);
            assert_eq!(
                columns.might_match(0, CompareOp::Lt, &make(5)),
                Some(false),
                "{family}: the live least side prunes"
            );
            assert_eq!(
                columns.might_match(0, CompareOp::Gt, &make(100)),
                Some(true),
                "{family}: the stale greatest side bounds nothing"
            );
            columns.collect(8);
            assert_eq!(
                bound(&columns, 0),
                Some((make(5), make(44))),
                "{family}: the stale side rebuilt"
            );

            // A column of one row: the row is both sides, so only an equal value keeps the
            // bound, and a greater or a smaller one makes it stale. A reader at the epoch
            // before each write sees the value the write replaced, which is never pruned.
            columns.write(1, 0, Some(&make(10)), None);
            let steps = [
                ("an equal", 10, true),
                ("a greater", 11, false),
                ("a smaller", 9, false),
            ];
            for ((step, rank, keeps), epoch) in steps.into_iter().zip(8..) {
                columns.collect(epoch);
                assert!(
                    bound(&columns, 1).is_some(),
                    "{family}: one row, before {step} value"
                );
                columns.write(1, 0, Some(&make(rank)), Some((pending(40 + epoch), epoch)));
                columns.stamp(1, 0, pending(40 + epoch), epoch + 1);
                assert_eq!(
                    bound(&columns, 1).is_some(),
                    keeps,
                    "{family}: one row written with {step} value"
                );
                assert_no_wrong_prune(&columns, 1, super::Read::At(epoch), &probes, family);
                assert_no_wrong_prune(&columns, 1, super::Read::At(epoch + 1), &probes, family);
            }
        }
    }
}
