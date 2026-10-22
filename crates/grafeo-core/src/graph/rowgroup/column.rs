//! The hot columns of a row group: one per property key, a typed vector
//! with presence bits.
//!
//! A column holds whatever values its rows get: it starts typed by its first
//! value (`Int64`, `Float64`, `Bool`, `String`) and turns into a column of
//! any values when a row gets a value of another type. Property columns are
//! not typed per graph (change target R11): no value is refused for its
//! type.

use std::sync::OnceLock;

use bytes::Bytes;
use grafeo_common::types::{ArcStr, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::Mutex;

use super::bitset::Bitset;
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
/// cold.
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

    /// The decoded values, decoding the cold chunks on the first use.
    fn hot(&self) -> &Hot {
        self.hot.get_or_init(|| {
            let mut hot = Hot {
                present: Bitset::default(),
                data: Data::Values(Vec::new()),
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

    fn heap_bytes(&self) -> usize {
        self.cold
            .lock()
            .iter()
            .map(|chunk| chunk.bytes.len())
            .sum::<usize>()
            + self.hot.get().map_or(0, Hot::heap_bytes)
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

    /// Every value at `row`, by key id.
    pub(super) fn row(&self, row: usize) -> impl Iterator<Item = (u32, Value)> + '_ {
        self.by_key
            .iter()
            .filter_map(move |(key, column)| column.hot().get(row).map(|value| (*key, value)))
    }

    /// Takes every value at `row` out.
    pub(super) fn clear_row(&mut self, row: usize) {
        for column in self.by_key.values_mut() {
            column.hot_mut().take(row);
        }
    }

    pub(super) fn heap_bytes(&self) -> usize {
        self.by_key.values().map(Column::heap_bytes).sum::<usize>()
            + self.by_key.capacity() * (std::mem::size_of::<u32>() + std::mem::size_of::<Column>())
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
