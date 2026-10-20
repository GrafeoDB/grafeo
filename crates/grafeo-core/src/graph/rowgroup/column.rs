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

use std::sync::OnceLock;

use bytes::Bytes;
use grafeo_common::change::PendingVersion;
use grafeo_common::types::{ArcStr, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::Mutex;

use super::Read;
use super::bitset::Bitset;
use super::updates::Updates;
use crate::codec::column_chunk::decode_column_chunk_bytes;

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
}

impl Hot {
    /// The value at `row`, if it has one.
    fn get(&self, row: usize) -> Option<Value> {
        self.present.get(row).then(|| self.data.get(row))
    }

    /// Sets the value at `row`.
    fn set(&mut self, row: usize, value: &Value) {
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
}

/// A column chunk as the file holds it, kept encoded: the values of rows
/// `[first_row, first_row + row_count)` of the row group.
#[derive(Debug)]
struct ColdChunk {
    first_row: usize,
    row_count: u32,
    codec: u8,
    bytes: Bytes,
}

/// One property key's values in a row group.
///
/// A column a load filled holds its chunks cold, as their encoded bytes,
/// until its first use: a read or a write of any of its rows decodes them
/// all into the hot form, once. The other columns of the row group stay
/// cold. A write thaws a cold column (#412 decision Q9), so the update
/// chains of a column live with its decoded values.
#[derive(Debug)]
pub(super) struct Column {
    /// The chunks not decoded yet; emptied by the first use.
    cold: Mutex<Vec<ColdChunk>>,
    /// The decoded values, set by the first use (or by the first write of a
    /// column no load filled).
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

    /// The value at `row` as `read` sees it.
    fn value_at(&self, row: usize, read: Read) -> Option<Value> {
        self.hot().get_at(row, read)
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

    /// The decoded values, decoding the cold chunks on the first use.
    fn hot(&self) -> &Hot {
        self.hot.get_or_init(|| {
            let mut hot = Hot {
                present: Bitset::default(),
                data: Data::Values(Vec::new()),
                updates: Updates::default(),
            };
            for chunk in std::mem::take(&mut *self.cold.lock()) {
                // The load decoded each chunk once to check it.
                let decoded = decode_column_chunk_bytes(&chunk.bytes, chunk.codec, chunk.row_count)
                    .expect("a chunk the load checked");
                for (offset, value) in decoded.values {
                    hot.set(chunk.first_row + offset as usize, &value);
                }
            }
            hot
        })
    }

    fn hot_mut(&mut self) -> &mut Hot {
        self.hot();
        self.hot.get_mut().expect("set by the line above")
    }

    /// Whether the column still holds its chunks encoded.
    fn is_cold(&self) -> bool {
        self.hot.get().is_none()
    }

    /// The bytes of the chunks it still holds encoded.
    fn cold_bytes(&self) -> usize {
        self.cold
            .lock()
            .iter()
            .map(|chunk| chunk.bytes.len())
            .sum::<usize>()
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
        self.by_key.get(&key)?.hot().get(row)
    }

    /// Adds a chunk of key `key` as the file holds it, for rows from
    /// `first_row`: it stays encoded until the column's first use. The chunk
    /// decodes (the load checked it).
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
            column.cold.lock().push(ColdChunk {
                first_row,
                row_count,
                codec,
                bytes,
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
            .filter_map(move |(key, column)| column.hot().get(row).map(|value| (*key, value)))
    }

    /// Every value at `row` as `read` sees it, by key id.
    pub(super) fn row_at(&self, row: usize, read: Read) -> impl Iterator<Item = (u32, Value)> + '_ {
        self.by_key
            .iter()
            .filter_map(move |(key, column)| column.value_at(row, read).map(|value| (*key, value)))
    }

    /// Takes every value at `row` out, with what any chain kept of it.
    pub(super) fn clear_row(&mut self, row: usize) {
        for column in self.by_key.values_mut() {
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

    /// The bytes of the chunks its columns still hold encoded.
    pub(super) fn cold_bytes(&self) -> usize {
        self.by_key.values().map(Column::cold_bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Value;

    use super::Columns;

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
