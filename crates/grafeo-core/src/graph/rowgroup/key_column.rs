//! Finding a key's holders by reading its column, one row group at a time, among the rows a scope
//! leaves.
//!
//! An index on a label holds every node with the label in its newest labels or a kept version,
//! under every value it holds in any version (see `index`). So the rows left out by the newest bits
//! of indexed labels are all in those indexes, and only the others are read here.
//!
//! A column of integers or floats is compared in place: such a value is found only by being one of
//! the forms a key names (see `MaybeEqual`), so the forms of its kind say all [`MaybeEqual::finds`]
//! would. A column of strings is too against string keys, as a string `=` a string only when their
//! bytes are equal (`'042' = '42'` is false). Other columns ask `finds` per row. Whatever reads a
//! row's value by itself reads a slice of rows at a time, so a `LIMIT` costs about what a scan's
//! first batch does. Bounds rule out a group or a chunk only when they bound what the key can find:
//! an integer or float column's own kind, or strings for string keys, as `'042' = 42` must be
//! found.
//!
//! The answer is at the reader's view: visibility by the word of found rows, which is one `AND` in
//! a vector without stamps, and a row a chain holds is read as the view sees it.

use crate::execution::operators::value_utils::{EqualAcrossKinds, ValueKinds, holds_null};
use std::ops::Range;
use std::sync::OnceLock;

use grafeo_common::types::{ArcStr, NodeId, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use smallvec::{SmallVec, smallvec};

use super::bitset::Bitset;
use super::column::{ColdRows, ColumnRead, HotRows, HotValues};
use super::index::MaybeEqual;
use super::slots::GroupRows;
use super::{Inner, LabelIds, ROWS_PER_GROUP, Read};
use crate::codec::column_chunk::{DictView, ZoneMap, bits_among};
use crate::graph::traits::KeyColumnGroup;

/// The words of a row group's bits.
const GROUP_WORDS: usize = (ROWS_PER_GROUP / 64) as usize;

/// The rows of a row group.
const GROUP_ROWS: usize = GROUP_WORDS * 64;

/// The rows a read of values one by one reads at once: a scan's first batch.
const SLICE_ROWS: usize = 2048;

/// A read of a key's column: which rows to look at and which values to find, kept ready for every
/// row group it reads (see `GraphStore::key_column_group`).
#[derive(Debug)]
pub struct KeyColumnQuery {
    /// Labels every result has.
    required: Vec<String>,
    /// Labels of which every result has one; none: no such need.
    one_of: Vec<String>,
    /// Labels whose nodes the caller finds otherwise, through their indexes.
    except: Vec<String>,
    property: String,
    /// What it looks for; `None` when `=` finds nothing equal to any value (NULL).
    targets: Option<Targets>,
    /// A chunk it decoded and reads a slice at a time.
    decoded: Option<DecodedChunk>,
    /// Its names as the store's ids, for the next group.
    ids: Option<ReadIds>,
    /// What the statement's `=` gives across kinds.
    equal: EqualAcrossKinds,
}

/// A read's names as a store's ids. An id is never given twice, so ids are kept once every name has
/// one; a name without one may get one by the next read.
#[derive(Debug)]
struct ReadIds {
    key: u32,
    required: LabelIds,
    one_of: LabelIds,
    except: LabelIds,
    every_name_known: bool,
}

/// What a read keeps of a cold chunk between its slices: a cold column's values do not change, as
/// a write thaws it, and the read then finds it hot.
#[derive(Debug)]
struct DecodedChunk {
    group: u64,
    first_row: usize,
    end_row: usize,
    values: KeptValues,
}

/// A cold chunk's values as a read compares them.
#[derive(Debug)]
enum KeptValues {
    /// Its dictionary, with the entries found.
    Dict(DictView),
    /// Its values, by offset, compared one by one.
    Decoded(Vec<(u32, Value)>),
}

impl DecodedChunk {
    /// Whether no row of it can be found: a dictionary with no entry found.
    fn finds_nothing(&self) -> bool {
        matches!(&self.values, KeptValues::Dict(view) if view.finds_nothing())
    }

    /// Whether a read from `row` of `group` goes on through it, a slice at a time.
    fn goes_on_at(&self, group: u64, row: usize) -> bool {
        self.group == group
            && (self.first_row..self.end_row).contains(&row)
            && !self.finds_nothing()
    }
}

impl KeyColumnQuery {
    /// A read of `property`'s column for `values`.
    #[must_use]
    pub fn new(
        required: &[&str],
        one_of: &[&str],
        except: &[&str],
        property: &str,
        values: impl Into<Vec<Value>>,
    ) -> Self {
        let owned = |labels: &[&str]| labels.iter().map(ToString::to_string).collect();
        Self {
            required: owned(required),
            one_of: owned(one_of),
            except: owned(except),
            property: property.to_string(),
            targets: Targets::new(values.into()),
            decoded: None,
            ids: None,
            equal: EqualAcrossKinds::False,
        }
    }

    /// Reads as a statement whose `=` gives `equal` across kinds compares: where it raises
    /// (SQL/PGQ), a row of the read's labels whose value is of another kind than the keys' fails
    /// the read (22G04), as the scan's `=` would meet it, whatever the read finds.
    #[must_use]
    pub fn with_equal_across_kinds(mut self, equal: EqualAcrossKinds) -> Self {
        self.equal = equal;
        self
    }

    /// What the statement's `=` gives across kinds (see
    /// [`with_equal_across_kinds`](Self::with_equal_across_kinds)).
    #[must_use]
    pub fn equal_across_kinds(&self) -> EqualAcrossKinds {
        self.equal
    }

    /// The key it reads.
    #[must_use]
    pub fn property(&self) -> &str {
        &self.property
    }

    /// Reads the rows of the labels left out too, whose indexes the caller could not read. Their
    /// nodes were to come from the indexes, so a need of one of some labels takes them in: over
    /// `n:A|B` with A's index read, the column read B's nodes only, and now reads A's as well.
    pub fn read_every_label(&mut self) {
        let left_out = std::mem::take(&mut self.except);
        if !self.one_of.is_empty() {
            for label in left_out {
                if !self.one_of.contains(&label) {
                    self.one_of.push(label);
                }
            }
        }
        self.ids = None;
    }

    /// Lets go of what it kept for a next read.
    pub fn forget_decoded(&mut self) {
        self.decoded = None;
    }
}

/// What a read of a key's column looks for. What it compares against is built when a column first
/// asks: a long `IN` pays only for what the column's kind reads.
#[derive(Debug)]
struct Targets {
    /// The keys, none of them NULL.
    values: Vec<Value>,
    /// Whether every key is an integer or a float other than NaN, so an integer or float found
    /// is `=` the key, not only maybe.
    number_keys: bool,
    /// Whether every key is a string.
    string_keys: bool,
    finders: OnceLock<Vec<MaybeEqual>>,
    forms: OnceLock<NumberForms>,
    strings: OnceLock<Vec<ArcStr>>,
}

/// The numbers a read finds.
#[derive(Debug)]
struct NumberForms {
    /// The integers found, sorted.
    integers: Vec<i64>,
    /// The bits of the floats found, sorted: `=`'s two zeros are both among them, and a NaN is
    /// found as the index finds it, for the filter to refuse.
    floats: Vec<u64>,
}

impl Targets {
    /// What finds `values`; `None` when `=` finds nothing equal to any: each is or holds a null.
    fn new(mut values: Vec<Value>) -> Option<Self> {
        values.retain(|value| !holds_null(value));
        if values.is_empty() {
            return None;
        }
        let number_keys = values.iter().all(|value| match value {
            Value::Int64(_) => true,
            Value::Float64(value) => !value.is_nan(),
            _ => false,
        });
        let string_keys = values.iter().all(|value| matches!(value, Value::String(_)));
        Some(Self {
            values,
            number_keys,
            string_keys,
            finders: OnceLock::new(),
            forms: OnceLock::new(),
            strings: OnceLock::new(),
        })
    }

    fn finders(&self) -> &[MaybeEqual] {
        self.finders
            .get_or_init(|| self.values.iter().filter_map(MaybeEqual::new).collect())
    }

    fn forms(&self) -> &NumberForms {
        self.forms.get_or_init(|| {
            let (mut integers, mut floats) = (Vec::new(), Vec::new());
            if self.number_keys {
                for key in &self.values {
                    MaybeEqual::number_forms(key, &mut integers, &mut floats);
                }
            } else {
                for find in self.finders() {
                    integers.extend(find.integers());
                    floats.extend(find.floats().map(f64::to_bits));
                }
            }
            integers.sort_unstable();
            integers.dedup();
            floats.sort_unstable();
            floats.dedup();
            NumberForms { integers, floats }
        })
    }

    /// The string keys, sorted: a stored string `=` a key only when it is one of them, by its
    /// bytes, as no string equals a number.
    fn strings(&self) -> &[ArcStr] {
        self.strings.get_or_init(|| {
            let mut strings: Vec<ArcStr> = self
                .values
                .iter()
                .filter_map(|value| match value {
                    Value::String(key) => Some(key.clone()),
                    _ => None,
                })
                .collect();
            strings.sort_unstable();
            strings.dedup();
            strings
        })
    }

    /// The kinds of the keys.
    fn kinds(&self) -> ValueKinds {
        self.values
            .iter()
            .fold(ValueKinds::default(), |kinds, key| {
                kinds.union(ValueKinds::of(key))
            })
    }

    /// Whether every value of `column` in the rows of `span` is of the keys' one kind, by what
    /// the column says of itself: a hot column of integers, floats or strings, and cold chunks
    /// whose zone maps bound integers, floats or strings. Values a chain holds are not counted.
    fn column_is_of_their_kind(
        &self,
        column: &ColumnRead<'_>,
        span: &Range<usize>,
    ) -> Result<bool> {
        let keys = self.kinds();
        if !keys.is_one() {
            return Ok(false);
        }
        match column {
            ColumnRead::Hot(hot) => Ok(match hot.values() {
                HotValues::Int64(_) | HotValues::Float64(_) => keys == ValueKinds::NUMBERS,
                HotValues::String(_) => keys == ValueKinds::STRINGS,
                HotValues::Other => false,
            }),
            ColumnRead::Cold(chunks) => {
                for chunk in chunks {
                    let rows = chunk.rows();
                    if rows.end <= span.start || rows.start >= span.end {
                        continue;
                    }
                    let kinds = match chunk.zone_map()? {
                        Some(ZoneMap::Values(Value::Int64(_) | Value::Float64(_), _)) => {
                            ValueKinds::NUMBERS
                        }
                        Some(ZoneMap::Strings(_)) => ValueKinds::STRINGS,
                        _ => return Ok(false),
                    };
                    if kinds != keys {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// The error of a key `=` a value of another kind meets, in a statement where that raises.
    fn not_comparable(&self, found: &Value) -> Error {
        let kind = ValueKinds::of(found);
        let key = self
            .values
            .iter()
            .find(|key| ValueKinds::of(key) != kind)
            .or(self.values.first());
        let message = match key {
            Some(key) => format!(
                "a value of type {} ({key}) cannot be compared with a value of type {} ({found})",
                key.type_name(),
                found.type_name()
            ),
            None => format!("a value of type {} ({found})", found.type_name()),
        };
        Error::from(QueryError::new(
            QueryErrorKind::ValuesNotComparable,
            message,
        ))
    }

    /// Whether a found `value` is `=` a key, not only maybe: a string found is one of the keys.
    fn is_exact(&self, value: &Value) -> bool {
        match value {
            Value::Int64(_) | Value::Float64(_) => self.number_keys,
            Value::String(_) => true,
            _ => false,
        }
    }

    /// Whether `text` is one of the string keys.
    fn is_a_key(&self, text: &[u8]) -> bool {
        match self.strings() {
            [] => false,
            // Keys that share their start, as ids do, differ at their end.
            [key] => {
                let key = key.as_bytes();
                key.last() == text.last() && key == text
            }
            keys => keys
                .binary_search_by(|key| key.as_bytes().cmp(text))
                .is_ok(),
        }
    }

    /// Whether the lookup finds `value`, as [`MaybeEqual::finds`] says.
    fn finds(&self, value: &Value) -> bool {
        match value {
            Value::Int64(value) => self.forms().integers.binary_search(value).is_ok(),
            Value::Float64(value) => self.forms().floats.binary_search(&value.to_bits()).is_ok(),
            Value::String(text) => self.is_a_key(text.as_bytes()),
            _ => self.finders().iter().any(|find| find.finds(value)),
        }
    }

    /// Whether a dictionary's entry, a string by its bytes, is found.
    fn entry_is_found(&self, entry: &[u8]) -> bool {
        self.is_a_key(entry)
    }

    /// Whether a hot column is read a slice at a time: its values are read one by one, as strings
    /// are, where integers and floats compare at a fraction of that.
    fn reads_in_slices(hot: &HotRows<'_>) -> bool {
        !matches!(hot.values(), HotValues::Int64(_) | HotValues::Float64(_))
    }

    /// Whether a hot column may hold a value it finds, by the bounds of what the column held.
    fn may_be_in(&self, hot: &HotRows<'_>) -> bool {
        match hot.values() {
            HotValues::Int64(_) => hot.might_hold_an_integer_of(&self.forms().integers),
            HotValues::Float64(_) => self
                .forms()
                .floats
                .iter()
                .any(|target| hot.might_equal(&Value::Float64(f64::from_bits(*target)))),
            HotValues::String(_) => hot.might_hold_a_string_of(self.strings()),
            HotValues::Other => true,
        }
    }

    /// Whether a chunk with the zone map `zone` holds nothing it finds. Zone maps leave NaN out.
    fn rules_out(&self, zone: &ZoneMap) -> bool {
        match zone {
            ZoneMap::Values(Value::Int64(min), Value::Int64(max)) => {
                let integers = &self.forms().integers;
                integers
                    .get(integers.partition_point(|target| target < min))
                    .is_none_or(|target| target > max)
            }
            ZoneMap::Values(Value::Float64(min), Value::Float64(max)) => {
                !self.forms().floats.iter().any(|target| {
                    let target = f64::from_bits(*target);
                    target.is_nan() || (*min..=*max).contains(&target)
                })
            }
            ZoneMap::Strings(zone) => !self.strings().iter().any(|string| zone.may_hold(string)),
            ZoneMap::Values(..) => false,
        }
    }

    /// The rows of `to_compare`, those of word `word` of a hot column, whose value it finds,
    /// clearing `exact` for one found other than exactly.
    fn found_in_word(
        &self,
        hot: &HotRows<'_>,
        word: usize,
        to_compare: u64,
        exact: &mut bool,
    ) -> u64 {
        if to_compare == 0 {
            return 0;
        }
        let found = match hot.values() {
            HotValues::Int64(values) => {
                let found = among(values, word, &self.forms().integers, |value| value);
                *exact &= self.number_keys || found & to_compare == 0;
                found
            }
            HotValues::Float64(values) => {
                let found = among(values, word, &self.forms().floats, f64::to_bits);
                *exact &= self.number_keys || found & to_compare == 0;
                found
            }
            HotValues::String(values) => {
                let mut found = 0;
                let mut rows = to_compare;
                while rows != 0 {
                    let bit = rows.trailing_zeros() as usize;
                    rows &= rows - 1;
                    if values
                        .get(word * 64 + bit)
                        .is_some_and(|value| self.is_a_key(value.as_bytes()))
                    {
                        found |= 1 << bit;
                    }
                }
                found
            }
            HotValues::Other => {
                let mut found = 0;
                let mut rows = to_compare;
                while rows != 0 {
                    let bit = rows.trailing_zeros() as usize;
                    rows &= rows - 1;
                    if let Some(value) = hot.get(word * 64 + bit)
                        && self.finds(&value)
                    {
                        found |= 1 << bit;
                        *exact &= self.is_exact(&value);
                    }
                }
                found
            }
        };
        found & to_compare
    }

    /// The rows of `span` that `scoped` names, a word per word of a group, in a cold column's
    /// `chunks` and that hold a value it finds, likewise; `None` when none does. Adds to `answer`
    /// the rows it read and clears its `exact` for one found other than exactly. Nothing thaws. A
    /// chunk whose values it must read one by one, or whose dictionary holds a key, ends `span` at
    /// a slice, and stays in `decoded` for the next read.
    fn compare_chunks(
        &self,
        chunks: &[ColdRows],
        group: u64,
        span: &mut Range<usize>,
        mut scoped: Vec<u64>,
        answer: &mut KeyColumnGroup,
        decoded: &mut Option<DecodedChunk>,
    ) -> Result<Option<Vec<u64>>> {
        let mut found = vec![0_u64; GROUP_WORDS];
        let mut any = false;
        for chunk in chunks {
            let rows = chunk.rows();
            if rows.end <= span.start {
                continue;
            }
            if rows.start >= span.end {
                break;
            }
            let first = rows.start;
            let mut hit = |offset: u32| {
                let row = first + offset as usize;
                if let Some(word) = found.get_mut(row / 64) {
                    *word |= 1 << (row % 64);
                    any = true;
                }
            };
            if !decoded
                .as_ref()
                .is_some_and(|kept| kept.group == group && kept.first_row == first)
            {
                let words = (first / 64).max(span.start / 64)
                    ..rows.end.div_ceil(64).min(span.end.div_ceil(64));
                let in_scope: usize = scoped.get(words).map_or(0, |words| {
                    words.iter().map(|word| word.count_ones()).sum::<u32>()
                }) as usize;
                if in_scope == 0 || chunk.zone_map()?.is_some_and(|zone| self.rules_out(&zone)) {
                    continue;
                }
                let scope = |word: usize| chunk_scope(&scoped, first, word);
                let point = chunk.point();
                if let Some(point) = point
                    && in_scope * point.rows_per_point_read() < rows.len()
                {
                    for (word, at) in (0..rows.len().div_ceil(64)).zip((0_u32..).step_by(64)) {
                        let mut bits = scope(word);
                        while bits != 0 {
                            let offset = at + bits.trailing_zeros();
                            bits &= bits - 1;
                            let Some(value) = point.get(offset) else {
                                continue;
                            };
                            answer.compared += 1;
                            if self.finds(&value) {
                                hit(offset);
                                answer.exact &= self.is_exact(&value);
                            }
                        }
                    }
                    continue;
                }
                let mut numbers_found = false;
                let read = point.and_then(|point| {
                    let forms = self.forms();
                    point.find_numbers(&forms.integers, &forms.floats, scope, |offset| {
                        hit(offset);
                        numbers_found = true;
                    })
                });
                if let Some(read) = read {
                    answer.compared += read;
                    answer.exact &= self.number_keys || !numbers_found;
                    continue;
                }
                let dict = chunk.dict(|entry| self.entry_is_found(entry))?;
                let values = match dict {
                    Some(view) => KeptValues::Dict(view),
                    None => KeptValues::Decoded(chunk.decode()?.values),
                };
                *decoded = Some(DecodedChunk {
                    group,
                    first_row: first,
                    end_row: rows.end,
                    values,
                });
            }
            let Some(kept) = decoded.as_ref() else {
                continue;
            };
            if kept.finds_nothing() {
                continue;
            }
            // A slice at a time. A later chunk the slice reaches into is compared up to its end,
            // and the rest in the next read, which counts it.
            let start = span.start.max(first);
            span.end = span.end.min(slice_end(start));
            if let Some(past) = scoped.get_mut(span.end / 64..) {
                past.fill(0);
            }
            match &kept.values {
                KeptValues::Dict(view) => {
                    // An entry found is a string key, which is exactly `=`.
                    answer.compared +=
                        view.find_codes(|word| chunk_scope(&scoped, first, word), hit);
                }
                KeptValues::Decoded(values) => {
                    let from =
                        values.partition_point(|(offset, _)| first + (*offset as usize) < start);
                    for (offset, value) in &values[from..] {
                        let row = first + *offset as usize;
                        if row >= span.end {
                            break;
                        }
                        if scoped
                            .get(row / 64)
                            .is_none_or(|rows| rows & (1 << (row % 64)) == 0)
                        {
                            continue;
                        }
                        answer.compared += 1;
                        if self.finds(value) {
                            hit(*offset);
                            answer.exact &= self.is_exact(value);
                        }
                    }
                }
            }
        }
        Ok(any.then_some(found))
    }
}

/// The end of the slice that holds row `start`.
fn slice_end(start: usize) -> usize {
    (start / SLICE_ROWS + 1) * SLICE_ROWS
}

/// The rows `scoped` names, a word per word of a group, in word `word` of a chunk from row `first`,
/// which may fall in a word.
fn chunk_scope(scoped: &[u64], first: usize, word: usize) -> u64 {
    let at = |index: usize| scoped.get(index).copied().unwrap_or(0);
    let (index, shift) = (first / 64 + word, first % 64);
    if shift == 0 {
        at(index)
    } else {
        (at(index) >> shift) | (at(index + 1) << (64 - shift))
    }
}

/// Calls `read` with each of `words` that has a row in one of `within`, or with every one of them
/// without it: a lookup on a rare label finds its words in a tight loop.
fn words_to_read(within: Option<&[&Bitset]>, words: Range<usize>, mut read: impl FnMut(usize)) {
    match within {
        None => words.for_each(read),
        Some([rows]) => {
            let held = rows.words();
            if let Some(held) = held.get(words.start..words.end.min(held.len())) {
                for (at, bits) in held.iter().enumerate() {
                    if *bits != 0 {
                        read(words.start + at);
                    }
                }
            }
        }
        Some(sets) => {
            let end = sets.iter().map(|rows| rows.words().len()).max();
            for word in words.start..words.end.min(end.unwrap_or(0)) {
                if sets.iter().any(|rows| rows.word(word) != 0) {
                    read(word);
                }
            }
        }
    }
}

/// The bits of word `word` of `values` whose key is among `targets`, sorted. A column's values
/// start at a vector and end at its highest row with a value, so a word below the start holds
/// none and the last word may be short.
fn among<T: Copy, K: Copy + Ord>(
    values: &GroupRows<T>,
    word: usize,
    targets: &[K],
    key: impl Fn(T) -> K,
) -> u64 {
    let Some(rest) = (word * 64)
        .checked_sub(values.first_row())
        .and_then(|from| values.as_slice().get(from..))
    else {
        return 0;
    };
    bits_among(&rest[..rest.len().min(64)], targets, key)
}

impl Inner {
    /// `query`'s names as ids; `None` when no node has the key, or a label every result has: no
    /// node has it in any version.
    fn read_ids(
        &self,
        required: &[String],
        one_of: &[String],
        except: &[String],
        property: &str,
    ) -> Option<ReadIds> {
        let key = self.keys.get_id(property)?;
        let required = required
            .iter()
            .map(|label| self.label_id(label))
            .collect::<Option<LabelIds>>()?;
        let known = |labels: &[String]| -> LabelIds {
            labels
                .iter()
                .filter_map(|label| self.label_id(label))
                .collect()
        };
        let (one_of_ids, except_ids) = (known(one_of), known(except));
        Some(ReadIds {
            key,
            required,
            every_name_known: one_of_ids.len() == one_of.len() && except_ids.len() == except.len(),
            one_of: one_of_ids,
            except: except_ids,
        })
    }

    /// See `GraphStore::key_column_group`.
    ///
    /// # Errors
    ///
    /// Returns a cold chunk's decode error.
    pub(super) fn key_column_group(
        &self,
        query: &mut KeyColumnQuery,
        from: u64,
        read: Read,
    ) -> Result<KeyColumnGroup> {
        let mut answer = KeyColumnGroup::default();
        let KeyColumnQuery {
            required,
            one_of,
            except,
            property,
            targets,
            decoded,
            ids,
            equal,
        } = query;
        let raises = *equal == EqualAcrossKinds::Raises;
        let Some(targets) = &*targets else {
            return Ok(answer);
        };
        if ids.as_ref().is_none_or(|ids| !ids.every_name_known) {
            *ids = self.read_ids(required, one_of, except, property);
        }
        let Some(ids) = &*ids else {
            return Ok(answer);
        };
        let any_of_labels = !one_of.is_empty();
        let (key, required, one_of, except) = (ids.key, &ids.required, &ids.one_of, &ids.except);
        let Some((number, group)) = self.nodes.range(from / ROWS_PER_GROUP..).next() else {
            return Ok(answer);
        };
        let base = number * ROWS_PER_GROUP;
        answer.end = Some(base + ROWS_PER_GROUP);
        answer.exact = targets.number_keys || targets.string_keys;
        // Built only past the tests below, and outliving what borrows them.
        let (value_chains, label_chains);
        let bits = |labels: &[u32]| -> SmallVec<[&Bitset; 4]> {
            labels
                .iter()
                .filter_map(|id| group.labels.get(id))
                .collect()
        };
        let (required_bits, one_of_bits, except_bits) =
            (bits(required), bits(one_of), bits(except));
        // As slices, and the one test as a flag: `in_scope` runs for each word of a group.
        let (required_bits, one_of_bits, except_bits) = (
            required_bits.as_slice(),
            one_of_bits.as_slice(),
            except_bits.as_slice(),
        );
        let a_required_label_is_missing = required_bits.len() < required.len();
        // A group with no row of the scope's labels, and no chain that may keep one: a lookup on a
        // rare label pays one test per other group.
        let no_row_in_scope = a_required_label_is_missing
            || required_bits.iter().any(|label| label.is_empty())
            || (any_of_labels && one_of_bits.iter().all(|label| label.is_empty()));
        if no_row_in_scope && group.label_updates.is_empty() {
            return Ok(answer);
        }
        let Some(column) = group.columns.read(key) else {
            return Ok(answer);
        };
        // A hot column's bound covers every version a reader may see, chained ones too, so no read
        // of it can find a value outside it. Where `=` raises across kinds, the values it holds are
        // read all the same, below.
        if let ColumnRead::Hot(hot) = &column
            && !raises
            && !targets.may_be_in(hot)
        {
            return Ok(answer);
        }
        // Every row with a label of the scope in its newest labels, less those the indexes hold.
        // A label's bits are empty in a group without it.
        let in_scope = |word: usize| -> u64 {
            let mut rows = if a_required_label_is_missing {
                0
            } else {
                u64::MAX
            };
            for label in required_bits {
                rows &= label.word(word);
            }
            if any_of_labels {
                rows &= one_of_bits
                    .iter()
                    .fold(0, |any, label| any | label.word(word));
            }
            for label in except_bits {
                rows &= !label.word(word);
            }
            rows
        };
        value_chains = group.columns.chained_rows(key);
        label_chains = group.label_updates.rows();
        let word_of = |chains: &Option<Bitset>, word: usize| {
            chains.as_ref().map_or(0, |rows| rows.word(word))
        };
        // A row a chain holds may differ at a reader's view whatever it holds now, in its labels
        // or its value: it is read as the view sees it.
        let chained = |word: usize, scoped: u64| {
            word_of(&label_chains, word) | (word_of(&value_chains, word) & scoped)
        };
        // The rows of this read: from `from`, on a word as the last read ended on one, up to the
        // group's end or a slice's.
        let mut span = usize::try_from(from.saturating_sub(base)).unwrap_or(GROUP_ROWS)..GROUP_ROWS;
        // What holds every row in scope, when something other than every row does: its rarest
        // required label, or the labels it has one of.
        let scope_rows: Option<SmallVec<[&Bitset; 4]>> = if a_required_label_is_missing {
            Some(SmallVec::new())
        } else if let Some(rarest) = required_bits
            .iter()
            .copied()
            .min_by_key(|label| label.words().len())
        {
            Some(smallvec![rarest])
        } else if any_of_labels {
            Some(SmallVec::from_slice(one_of_bits))
        } else {
            None
        };
        // The rows the check of kinds reads, where `=` raises: those of the scope, and those whose
        // labels a chain holds.
        let kind_scope = raises.then(|| {
            scope_rows.clone().map(|mut rows| {
                rows.extend(label_chains.as_ref());
                rows
            })
        });
        let mut cold_found = None;
        let may_find = match &column {
            ColumnRead::Hot(hot) => {
                if Targets::reads_in_slices(hot) {
                    span.end = span.end.min(slice_end(span.start));
                }
                true
            }
            ColumnRead::Cold(chunks) => {
                // Going on through a kept chunk, the scope is that of the slice alone.
                if decoded
                    .as_ref()
                    .is_some_and(|kept| kept.goes_on_at(*number, span.start))
                {
                    span.end = slice_end(span.start);
                }
                let mut to_compare = vec![0_u64; GROUP_WORDS];
                words_to_read(
                    scope_rows.as_deref(),
                    span.start / 64..span.end.div_ceil(64),
                    |word| {
                        let scoped = in_scope(word);
                        to_compare[word] = scoped & !chained(word, scoped);
                    },
                );
                cold_found = targets.compare_chunks(
                    chunks,
                    *number,
                    &mut span,
                    to_compare,
                    &mut answer,
                    decoded,
                )?;
                cold_found.is_some()
            }
        };
        if span.end < GROUP_ROWS {
            answer.end = Some(base + span.end as u64);
        }
        // A row found is in scope, and a chained row is read whatever it holds now: with nothing to
        // find, which only a cold column gets to and which has no value chains, only the rows whose
        // labels are chained are looked at.
        let to_read: Option<SmallVec<[&Bitset; 4]>> = if may_find {
            scope_rows.map(|mut rows| {
                rows.extend(label_chains.as_ref());
                rows
            })
        } else {
            Some(label_chains.iter().collect())
        };
        // The labels of the scope, as the view sees a chained row's.
        let labels_in_scope = |row: usize| {
            if required.is_empty() && !any_of_labels {
                return true;
            }
            let labels = group.labels_at(row, read);
            required.iter().all(|label| labels.contains(label))
                && (!any_of_labels || one_of.iter().any(|label| labels.contains(label)))
        };
        words_to_read(
            to_read.as_deref(),
            span.start / 64..span.end.div_ceil(64),
            |word| {
                let scoped = in_scope(word);
                let always = chained(word, scoped);
                let found = match &column {
                    ColumnRead::Hot(hot) => {
                        let to_compare = scoped & hot.present().word(word) & !always;
                        answer.compared += to_compare.count_ones() as usize;
                        targets.found_in_word(hot, word, to_compare, &mut answer.exact)
                    }
                    ColumnRead::Cold(_) => cold_found.as_ref().map_or(0, |found| found[word]),
                };
                answer.compared += always.count_ones() as usize;
                let mut rows = group.versions.visible_word(word, found, read) | always;
                while rows != 0 {
                    let bit = rows.trailing_zeros() as usize;
                    rows &= rows - 1;
                    let row = word * 64 + bit;
                    if always & (1 << bit) != 0 {
                        if !group.versions.visible(row, read) || !labels_in_scope(row) {
                            continue;
                        }
                        let Some(value) = group.columns.get_at(key, row, read) else {
                            continue;
                        };
                        if !targets.finds(&value) {
                            continue;
                        }
                        answer.exact &= targets.is_exact(&value);
                    }
                    answer.nodes.push(NodeId::new(base + row as u64));
                }
            },
        );
        // Where `=` raises across kinds, a row of the scope the view sees whose value is of another
        // kind than the keys' fails the read, as the scan's `=` would meet it. A column of the
        // keys' one kind has such a value only in a chain.
        if let Some(kind_scope) = kind_scope {
            let keys = targets.kinds();
            let of_their_kind = targets.column_is_of_their_kind(&column, &span)?;
            let mut met = None;
            words_to_read(
                kind_scope.as_deref(),
                span.start / 64..span.end.div_ceil(64),
                |word| {
                    if met.is_some() {
                        return;
                    }
                    let labels_chained = word_of(&label_chains, word);
                    let mut rows = if of_their_kind {
                        word_of(&value_chains, word) & (in_scope(word) | labels_chained)
                    } else {
                        in_scope(word) | labels_chained
                    };
                    while rows != 0 {
                        let bit = rows.trailing_zeros() as usize;
                        rows &= rows - 1;
                        let row = word * 64 + bit;
                        if !group.versions.visible(row, read)
                            || (labels_chained & (1 << bit) != 0 && !labels_in_scope(row))
                        {
                            continue;
                        }
                        if let Some(value) = group.columns.get_at(key, row, read)
                            && !value.is_null()
                            && ValueKinds::of(&value) != keys
                        {
                            met = Some(value);
                            return;
                        }
                    }
                },
            );
            if let Some(found) = met {
                return Err(targets.not_comparable(&found));
            }
        }
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use bytes::Bytes;
    use grafeo_common::change::DataOp;
    use grafeo_common::types::{
        Date, Decimal, NodeId, PropertyKey, ReadView, TemporalComparison, Value, ZonedDatetime,
    };

    use super::super::column::{ColumnRead, Columns};
    use super::super::{ROWS_PER_GROUP, RowGroupStore};
    use super::{GROUP_ROWS, GROUP_WORDS, KeyColumnQuery, MaybeEqual, Targets};
    use crate::codec::column_chunk::encode_column_chunk;
    use crate::execution::operators::value_utils::values_equal;
    use crate::graph::apply::{GraphBackend, Writer};
    use crate::graph::conformance::{apply_all, labels, next_epoch, properties};
    use crate::graph::traits::{GraphStore, KeyColumnGroup};

    const KEY: u32 = 7;

    fn decimal(mantissa: i128, scale: u8) -> Value {
        Value::Decimal(Decimal::new(mantissa, scale).unwrap())
    }

    fn string(text: &str) -> Value {
        Value::from(text)
    }

    /// Strings as `=` reads them against a number: by `parse::<i64>` or `parse::<f64>`, so every
    /// spelling those take, and the near ones they do not.
    const SPELLINGS: &[&str] = &[
        "42",
        "042",
        "0042",
        "+42",
        "-42",
        "-042",
        "42.0",
        "42.00",
        "+42.0",
        "4.2e1",
        "4.2E1",
        "420e-1",
        "0.42e2",
        " 42",
        "42 ",
        "4 2",
        "0x2A",
        "1_000",
        "42a",
        "a42",
        "--42",
        "+-42",
        "4.2.1",
        "e1",
        "1e",
        ".",
        "+",
        "-",
        "0",
        "00",
        "-0",
        "+0",
        "0.0",
        "-0.0",
        ".0",
        "0.",
        ".5",
        "5.",
        "1000",
        "1e3",
        "1E3",
        "1.0e3",
        "nan",
        "NaN",
        "NAN",
        "+nan",
        "-nan",
        "inf",
        "Inf",
        "INF",
        "+inf",
        "-inf",
        "infinity",
        "Infinity",
        "-Infinity",
        "INFINITY",
        "infin",
        "1e400",
        "-1e400",
        "1e-400",
        "9007199254740993",
        "9007199254740992",
        "9007199254740992.0",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775808",
        "true",
        "false",
        "",
        " ",
        "abc",
        "Alix",
        "é",
        "42é",
    ];

    /// One value of each kind and spelling `=` treats specially, as `equality_with_index` has them.
    fn every_value() -> Vec<Value> {
        let big = 9_007_199_254_740_993_i64;
        let map = |value: Value| {
            let mut map = BTreeMap::new();
            map.insert(PropertyKey::new("a"), value);
            Value::Map(Arc::new(map))
        };
        let mut values = vec![
            Value::Int64(42),
            Value::Int64(-42),
            decimal(420, 1),
            decimal(425, 1),
            string("42"),
            string("-42"),
            string("042"),
            string("+42"),
            string("42.0"),
            string("4.2e1"),
            string("abc"),
            decimal(3, 1),
            string("0.3"),
            Value::Int64(0),
            decimal(0, 1),
            Value::Int64(1),
            decimal(10, 1),
            Value::Float64(0.1 + 0.2),
            Value::Float64(0.3),
            string("0.30000000000000004"),
            Value::Float64(0.0),
            Value::Float64(-0.0),
            string("-0"),
            Value::Float64(1.0),
            Value::Float64(0.999_999_999_999_999_9),
            Value::Float64(1.0e-300),
            Value::Float64(-1.0e-300),
            Value::Float64(f64::NAN),
            Value::Float64(f64::INFINITY),
            Value::Float64(f64::NEG_INFINITY),
            Value::Float64(1.0e300),
            Value::Float64(42.0),
            string("1e3"),
            Value::Int64(1_000),
            Value::Int64(2),
            string("2"),
            Value::Bool(true),
            Value::Bool(false),
            string("true"),
            Value::Int64(big),
            Value::Int64(big - 1),
            Value::Float64(9_007_199_254_740_992.0),
            string("9007199254740993"),
            string("NaN"),
            string("inf"),
            Value::Int64(i64::MAX),
            Value::Int64(i64::MIN),
            Value::Float64(9_223_372_036_854_775_808.0),
            Value::List(vec![Value::Int64(42)].into()),
            Value::List(vec![string("42")].into()),
            map(Value::Int64(42)),
            map(string("42")),
            Value::Date(Date::parse("2024-03-15").unwrap()),
            Value::ZonedDatetime(ZonedDatetime::parse("2024-03-15T10:00:00+01:00").unwrap()),
            Value::ZonedDatetime(ZonedDatetime::parse("2024-03-15T09:00:00Z").unwrap()),
        ];
        values.extend(SPELLINGS.iter().map(|text| string(text)));
        values
    }

    /// `=` itself: what a read must not miss, whatever else it leaves to a check.
    fn equal(key: &Value, value: &Value) -> bool {
        values_equal(key, value, TemporalComparison::default())
    }

    /// The columns a chunk may be: each kind alone picks its own codec.
    fn pools() -> Vec<(&'static str, Vec<Value>)> {
        let values = every_value();
        let only = |test: fn(&Value) -> bool| -> Vec<Value> {
            values.iter().filter(|v| test(v)).cloned().collect()
        };
        vec![
            ("integers", only(|v| matches!(v, Value::Int64(_)))),
            ("small integers", (0..40).map(Value::Int64).collect()),
            ("in order", vec![]),
            ("floats", only(|v| matches!(v, Value::Float64(_)))),
            ("strings", only(|v| matches!(v, Value::String(_)))),
            ("one string", vec![string("42")]),
            (
                "spellings",
                SPELLINGS.iter().map(|text| string(text)).collect(),
            ),
            (
                "no number",
                vec![
                    string("Amsterdam"),
                    string("Berlin"),
                    string(""),
                    string("Paris"),
                ],
            ),
            (
                "many strings",
                (0..300).map(|i| string(&format!("name-{i}"))).collect(),
            ),
            ("booleans", only(|v| matches!(v, Value::Bool(_)))),
            ("decimals", only(|v| matches!(v, Value::Decimal(_)))),
            ("all kinds", values.clone()),
        ]
    }

    /// A chunk: its first row, its rows, and a value on every `every`th row.
    #[derive(Clone, Copy)]
    struct Chunk {
        first: usize,
        rows: u32,
        every: u32,
    }

    const fn chunk(first: usize, rows: u32, every: u32) -> Chunk {
        Chunk { first, rows, every }
    }

    /// Dense and sparse chunks, on a 64-row boundary and off one.
    fn layouts() -> Vec<(&'static str, Vec<Chunk>)> {
        vec![
            ("dense", vec![chunk(0, 300, 1)]),
            ("sparse", vec![chunk(0, 300, 3)]),
            ("very sparse", vec![chunk(0, 300, 17)]),
            ("two aligned", vec![chunk(0, 128, 1), chunk(128, 172, 2)]),
            ("off a boundary", vec![chunk(0, 37, 1), chunk(37, 263, 1)]),
            (
                "off a boundary, sparse",
                vec![chunk(0, 100, 2), chunk(100, 200, 3)],
            ),
            ("first row inside a word", vec![chunk(37, 263, 1)]),
            ("sparse, inside a word", vec![chunk(61, 239, 5)]),
            ("a gap between", vec![chunk(5, 50, 1), chunk(200, 100, 1)]),
            ("a short chunk", vec![chunk(62, 5, 1)]),
        ]
    }

    /// The rows looked at: all of them, every other, few enough for point reads, and a run over a
    /// word boundary.
    fn scopes() -> Vec<(&'static str, Vec<u64>)> {
        let of = |rows: &[usize]| {
            let mut words = vec![0_u64; GROUP_WORDS];
            for row in rows {
                words[row / 64] |= 1 << (row % 64);
            }
            words
        };
        vec![
            ("all", vec![u64::MAX; GROUP_WORDS]),
            (
                "odd",
                of(&(0..300).filter(|row| row % 2 == 1).collect::<Vec<_>>()),
            ),
            ("few", of(&[1, 38, 66, 130, 262])),
            ("over a word", of(&(58..70).collect::<Vec<_>>())),
            ("none", vec![0; GROUP_WORDS]),
        ]
    }

    /// The value of each row of a layout, by row.
    fn rows_of(pool: &[Value], layout: &[Chunk]) -> Vec<(usize, Value)> {
        let mut rows = Vec::new();
        for part in layout {
            for offset in (0..part.rows).step_by(part.every as usize) {
                let row = part.first + offset as usize;
                let value = if pool.is_empty() {
                    Value::Int64(i64::try_from(row).unwrap())
                } else {
                    pool[row % pool.len()].clone()
                };
                rows.push((row, value));
            }
        }
        rows
    }

    fn cold(pool: &[Value], layout: &[Chunk]) -> Columns {
        let mut columns = Columns::default();
        for part in layout {
            let values: Vec<(u32, Value)> = rows_of(pool, std::slice::from_ref(part))
                .into_iter()
                .map(|(row, value)| (u32::try_from(row - part.first).unwrap(), value))
                .collect();
            let (codec, bytes) = encode_column_chunk(part.rows, &values, None).unwrap();
            columns.add_cold(
                KEY,
                part.first,
                part.rows,
                codec.to_byte(),
                Bytes::from(bytes),
            );
        }
        columns
    }

    fn hot(pool: &[Value], layout: &[Chunk]) -> Columns {
        let mut columns = Columns::default();
        for (row, value) in rows_of(pool, layout) {
            columns.set(KEY, row, &value);
        }
        columns
    }

    fn set_rows(words: &[u64]) -> Vec<usize> {
        let mut rows = Vec::new();
        for (word, bits) in words.iter().enumerate() {
            let mut bits = *bits;
            while bits != 0 {
                rows.push(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        rows
    }

    /// What a read found, row by row: nothing `=` finds is missing, nothing `MaybeEqual::finds` would
    /// not find is there, and an answer called exact is `=`'s to the row.
    fn bounded(
        found: &[usize],
        exact: bool,
        targets: &Targets,
        keys: &[Value],
        rows: &[(usize, Value)],
        scope: &[u64],
        name: &str,
    ) {
        for (row, value) in rows {
            let is_found = found.binary_search(row).is_ok();
            if scope[row / 64] & (1 << (row % 64)) == 0 {
                assert!(
                    !is_found,
                    "{name}: row {row} is out of scope, keys {keys:?}"
                );
                continue;
            }
            let is_equal = keys.iter().any(|key| equal(key, value));
            assert!(
                !is_equal || is_found,
                "{name}: {value:?} = {keys:?}, not found"
            );
            let maybe = targets.finders().iter().any(|find| find.finds(value));
            assert!(
                !is_found || maybe,
                "{name}: {value:?} found, no `finds` of {keys:?} finds it"
            );
            assert!(
                !exact || is_equal == is_found,
                "{name}: called exact, {value:?} against {keys:?}"
            );
        }
        for found in found {
            assert!(
                rows.iter().any(|(row, _)| row == found),
                "{name}: row {found} holds nothing"
            );
        }
    }

    /// Reads a cold column whole, a slice at a time as the store does: each row once.
    fn read_cold(targets: &Targets, columns: &Columns, scope: &[u64]) -> (Vec<usize>, usize, bool) {
        let Some(ColumnRead::Cold(chunks)) = columns.read(KEY) else {
            panic!("a cold column");
        };
        let mut answer = KeyColumnGroup {
            exact: targets.number_keys || targets.string_keys,
            ..KeyColumnGroup::default()
        };
        let mut decoded = None;
        let mut found = vec![0_u64; GROUP_WORDS];
        let mut start = 0;
        while start < GROUP_ROWS {
            let mut span = start..GROUP_ROWS;
            let mut asked = scope.to_vec();
            asked[..start / 64].fill(0);
            let read = targets
                .compare_chunks(&chunks, 0, &mut span, asked, &mut answer, &mut decoded)
                .unwrap();
            if let Some(read) = read {
                for word in span.start / 64..span.end.div_ceil(64) {
                    found[word] |= read[word];
                }
            }
            start = span.end;
        }
        (set_rows(&found), answer.compared, answer.exact)
    }

    /// The cold read and the hot read keep within what `=` and `MaybeEqual::finds` say, row by row.
    fn check(
        name: &str,
        targets: &Targets,
        keys: &[Value],
        pool: &[Value],
        layout: &[Chunk],
        scope: &[u64],
    ) {
        let rows = rows_of(pool, layout);
        let held_in_scope = rows
            .iter()
            .filter(|(row, _)| scope[row / 64] & (1 << (row % 64)) != 0)
            .count();
        let in_scope = |value: &&(usize, Value)| scope[value.0 / 64] & (1 << (value.0 % 64)) != 0;
        let only_numbers = rows
            .iter()
            .filter(in_scope)
            .all(|(_, v)| matches!(v, Value::Int64(_) | Value::Float64(_)));
        let only_strings = rows
            .iter()
            .filter(in_scope)
            .all(|(_, v)| matches!(v, Value::String(_)));
        let stays_exact =
            (targets.number_keys && only_numbers) || (targets.string_keys && only_strings);

        let (found, compared, exact) = read_cold(targets, &cold(pool, layout), scope);
        bounded(
            &found,
            exact,
            targets,
            keys,
            &rows,
            scope,
            &format!("cold: {name}"),
        );
        assert!(!stays_exact || exact, "cold, exact: {name}, keys {keys:?}");
        assert!(
            compared >= found.len() && compared <= held_in_scope,
            "cold, compared {compared} of {held_in_scope}, found {}: {name}, keys {keys:?}",
            found.len()
        );

        let columns = hot(pool, layout);
        let Some(ColumnRead::Hot(hot)) = columns.read(KEY) else {
            panic!("{name}: a hot column");
        };
        let mut words = vec![0_u64; GROUP_WORDS];
        let mut exact = targets.number_keys || targets.string_keys;
        if targets.may_be_in(&hot) {
            for word in 0..GROUP_WORDS {
                let to_compare = scope[word] & hot.present().word(word);
                words[word] = targets.found_in_word(&hot, word, to_compare, &mut exact);
            }
        }
        bounded(
            &set_rows(&words),
            exact,
            targets,
            keys,
            &rows,
            scope,
            &format!("hot: {name}"),
        );
        assert!(!stays_exact || exact, "hot, exact: {name}, keys {keys:?}");
    }

    /// Against [`MaybeEqual::finds`](super::MaybeEqual::finds), over every key: the sparse path and
    /// a scope off a word's boundary are where a compare in place can go wrong without the filter
    /// above it showing that it missed a row.
    #[test]
    fn the_read_finds_what_finds_finds() {
        for (pool_name, pool) in pools() {
            for (layout_name, layout) in layouts() {
                for (scope_name, scope) in scopes() {
                    let name = format!("{pool_name}, {layout_name}, {scope_name}");
                    for key in every_value() {
                        let keys = [key];
                        let targets = Targets::new(keys.to_vec()).expect("no key here is NULL");
                        check(&name, &targets, &keys, &pool, &layout, &scope);
                    }
                }
            }
        }
    }

    /// A list of keys of each kind, sorted or not, finds what any of them finds.
    #[test]
    fn a_list_of_keys_finds_what_any_of_them_finds() {
        let lists = [
            vec![Value::Int64(42), Value::Int64(0)],
            vec![Value::Int64(1_000), Value::Int64(2), Value::Int64(-42)],
            vec![Value::Int64(1), Value::Float64(0.3), string("abc")],
            vec![string("42"), string("abc")],
            vec![Value::Float64(f64::NAN), Value::Int64(42)],
            vec![Value::Float64(0.0), Value::Float64(1.0)],
            vec![decimal(3, 1), Value::Bool(true), string("inf")],
        ];
        for (pool_name, pool) in pools() {
            for (layout_name, layout) in layouts() {
                for (scope_name, scope) in scopes() {
                    let name = format!("{pool_name}, {layout_name}, {scope_name}");
                    for keys in &lists {
                        let targets = Targets::new(keys.clone()).unwrap();
                        check(&name, &targets, keys, &pool, &layout, &scope);
                    }
                }
            }
        }
    }

    /// A read that must decode a chunk ends at a slice, and a chunk it compares in place may lie past
    /// that end: each row is found and counted once, when its slice comes.
    #[test]
    fn a_slice_boundary_inside_a_mixed_column_reads_each_row_once() {
        let mixed: Vec<Value> = vec![
            Value::Int64(7),
            string("7"),
            Value::Float64(7.0),
            string("Gus"),
            Value::Bool(true),
        ];
        let numbers: Vec<Value> = (0..50).map(Value::Int64).collect();
        let strings: Vec<Value> = vec![string("7"), string("Berlin"), string("07")];
        let later: Vec<Value> = vec![string("Prague"), string("Berlin"), string("7.0")];
        let parts: [&[(usize, u32, &[Value])]; 7] = [
            // A mixed chunk over two slices, then a number chunk across the rest.
            &[(0, 5_000, &mixed), (5_000, 9_000, &numbers)],
            // A number chunk, then a mixed one that starts inside a slice and a word.
            &[
                (0, 3_001, &numbers),
                (3_001, 4_000, &mixed),
                (7_001, 3_000, &strings),
            ],
            // A mixed chunk that ends just past a slice's end.
            &[(0, 2_049, &mixed), (2_049, 4_000, &strings)],
            // A gap, and a mixed chunk after it.
            &[(100, 50, &numbers), (6_000, 3_000, &mixed)],
            // Two dictionary chunks, a slice's end between them and one inside the second.
            &[(0, 2_048, &strings), (2_048, 3_000, &later)],
            // Two dictionary chunks with their boundary inside a slice and a word.
            &[(0, 3_001, &strings), (3_001, 3_000, &later)],
            // A key in a later chunk, after a chunk with no such entry, then a mixed one.
            &[
                (0, 3_000, &later),
                (3_000, 2_000, &strings),
                (5_000, 2_000, &mixed),
            ],
        ];
        for keys in [
            vec![Value::Int64(7)],
            vec![Value::Float64(7.0)],
            vec![string("7")],
            vec![string("07")],
            vec![string("Gus"), Value::Int64(7)],
            vec![string("Prague")],
            vec![string("Berlin")],
            vec![string("Oslo")],
            vec![Value::Int64(8)],
        ] {
            let targets = Targets::new(keys.clone()).unwrap();
            for (case, parts) in parts.iter().enumerate() {
                let mut columns = Columns::default();
                let mut rows = Vec::new();
                for (first, count, pool) in *parts {
                    let values: Vec<(u32, Value)> = (0..*count)
                        .map(|offset| (offset, pool[offset as usize % pool.len()].clone()))
                        .collect();
                    rows.extend(
                        values
                            .iter()
                            .map(|(offset, value)| (first + *offset as usize, value.clone())),
                    );
                    let (codec, bytes) = encode_column_chunk(*count, &values, None).unwrap();
                    columns.add_cold(KEY, *first, *count, codec.to_byte(), Bytes::from(bytes));
                }
                let scope = vec![u64::MAX; GROUP_WORDS];
                let (found, compared, exact) = read_cold(&targets, &columns, &scope);
                let name = format!("case {case}, keys {keys:?}");
                bounded(&found, exact, &targets, &keys, &rows, &scope, &name);
                assert!(
                    compared <= rows.len(),
                    "{name}: compared {compared} of {} rows",
                    rows.len()
                );
            }
        }
    }

    /// The pins of a string key: its bytes against a string, and never a number, whatever it spells.
    #[test]
    fn a_string_key_against_each_kind() {
        let targets = |key: &str| Targets::new(vec![string(key)]).unwrap();
        for (key, value, wanted) in [
            ("042", string("42"), false),
            ("042", string("042"), true),
            ("42", string("042"), false),
            ("042", Value::Int64(42), false),
            ("42", Value::Int64(42), false),
            ("4.2e1", Value::Int64(42), false),
            ("42.0", Value::Int64(42), false),
            ("4.2e1", Value::Float64(42.0), false),
            ("42.0", Value::Float64(42.0), false),
            ("42", Value::Float64(42.0), false),
            ("abc", Value::Int64(42), false),
            ("", string(""), true),
            ("", Value::Int64(0), false),
            ("nan", string("nan"), true),
            ("nan", Value::Float64(f64::NAN), false),
        ] {
            let key_value = string(key);
            assert_eq!(
                equal(&key_value, &value),
                wanted,
                "`=` of {key:?} and {value:?}"
            );
            let targets = targets(key);
            if wanted {
                assert!(targets.finds(&value), "{key:?} finds {value:?}");
            }
            if targets.is_exact(&value) && targets.finds(&value) {
                assert!(wanted, "{key:?} calls {value:?} exact");
            }
        }
    }

    /// `number_forms` is a second statement of the rule in `probes`: for every integer and float key
    /// both name the same numbers, as sets, so a long `IN` over numbers finds what a lookup does.
    #[test]
    fn number_forms_say_what_the_probes_say() {
        let two_to_53 = 9_007_199_254_740_992_i64;
        let mut keys = vec![
            Value::Int64(0),
            Value::Int64(1),
            Value::Int64(-1),
            Value::Int64(42),
            Value::Int64(-42),
            Value::Int64(1_000),
            Value::Int64(two_to_53 - 1),
            Value::Int64(two_to_53),
            Value::Int64(two_to_53 + 1),
            Value::Int64(-two_to_53 - 1),
            Value::Int64(i64::MAX),
            Value::Int64(i64::MIN),
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Float64(1.0),
            Value::Float64(42.0),
            Value::Float64(42.5),
            Value::Float64(0.1 + 0.2),
            Value::Float64(0.3),
            Value::Float64(0.999_999_999_999_999_9),
            Value::Float64(1.0e-300),
            Value::Float64(-1.0e-300),
            Value::Float64(1.0e300),
            Value::Float64(f64::NAN),
            Value::Float64(f64::INFINITY),
            Value::Float64(f64::NEG_INFINITY),
            Value::Float64(9_007_199_254_740_992.0),
            Value::Float64(9_223_372_036_854_775_808.0),
            Value::Float64(-9_223_372_036_854_775_808.0),
        ];
        keys.extend(
            every_value()
                .into_iter()
                .filter(|value| matches!(value, Value::Int64(_) | Value::Float64(_))),
        );
        for key in keys {
            let find = MaybeEqual::new(&key).expect("no key here is NULL");
            let mut from_probes: Vec<i64> = find.integers().collect();
            let mut bits_from_probes: Vec<u64> = find.floats().map(f64::to_bits).collect();
            let (mut integers, mut floats) = (Vec::new(), Vec::new());
            MaybeEqual::number_forms(&key, &mut integers, &mut floats);
            for list in [&mut from_probes, &mut integers] {
                list.sort_unstable();
                list.dedup();
            }
            for list in [&mut bits_from_probes, &mut floats] {
                list.sort_unstable();
                list.dedup();
            }
            assert_eq!(integers, from_probes, "the integers of {key:?}");
            assert_eq!(floats, bits_from_probes, "the floats of {key:?}");
        }
    }

    /// A list of numbers read through the forms alone, and one that also holds a string, find the
    /// same numbers.
    #[test]
    fn a_long_list_of_numbers_finds_what_each_of_its_keys_finds() {
        let keys: Vec<Value> = (0..2_000).map(|i| Value::Int64(i * 3)).collect();
        let targets = Targets::new(keys.clone()).unwrap();
        for value in [
            Value::Int64(3),
            Value::Int64(4),
            Value::Float64(3.0),
            Value::Float64(3.5),
            string("3"),
            string("03"),
            string("3.0"),
            decimal(30, 1),
        ] {
            let by_key = keys.iter().any(|key| equal(key, &value));
            assert_eq!(targets.finds(&value), by_key, "{value:?}");
        }
    }

    /// A number key against a string is answered in place of `=`, so it answers to `values_equal`
    /// itself: a change on either side fails here. A row it finds, it reports exact.
    #[test]
    fn text_against_number_keys_answers_as_values_equal() {
        const TEXTS: &[&str] = &[
            "42",
            "042",
            "+42",
            "-0",
            "0",
            "4.2e1",
            "4.2E1",
            "42.0",
            ".5",
            "5.",
            "1e400",
            "-1e400",
            "inf",
            "-inf",
            "infinity",
            "nan",
            "NaN",
            "9007199254740992",
            "9007199254740993",
            "9223372036854775807",
            "9223372036854775808",
            "-9223372036854775809",
            "",
            " 42",
            "42 ",
            "abc",
            "0x2A",
            "1_000",
        ];
        let two_to_53 = 9_007_199_254_740_992_i64;
        let lists: Vec<Vec<Value>> = vec![
            vec![Value::Int64(42)],
            vec![Value::Float64(42.0)],
            vec![Value::Int64(0)],
            vec![Value::Float64(0.0)],
            vec![Value::Float64(-0.0)],
            vec![Value::Float64(0.5)],
            vec![Value::Float64(f64::INFINITY)],
            vec![Value::Float64(f64::NEG_INFINITY)],
            vec![Value::Int64(two_to_53)],
            vec![Value::Float64(two_to_53 as f64)],
            vec![Value::Int64(two_to_53 + 1)],
            vec![Value::Int64(i64::MAX)],
            vec![Value::Float64(9_223_372_036_854_775_808.0)],
            vec![Value::Int64(42), Value::Float64(0.5)],
            vec![Value::Float64(42.0), Value::Int64(0)],
            vec![
                Value::Int64(0),
                Value::Float64(f64::INFINITY),
                Value::Int64(two_to_53),
            ],
            vec![
                Value::Float64(9_223_372_036_854_775_808.0),
                Value::Int64(i64::MAX),
                Value::Float64(-0.0),
            ],
        ];
        for keys in &lists {
            let targets = Targets::new(keys.clone()).unwrap();
            assert!(targets.number_keys, "{keys:?}");
            for text in TEXTS {
                let value = string(text);
                let by_equals = keys.iter().any(|key| equal(key, &value));
                assert_eq!(
                    targets.finds(&value),
                    by_equals,
                    "{text:?} against {keys:?}"
                );
                if by_equals {
                    assert!(
                        targets.is_exact(&value),
                        "{text:?} found by {keys:?}, not called exact"
                    );
                }
            }
        }
    }

    /// NULL finds nothing, and a list of only NULL too.
    #[test]
    fn null_is_no_target() {
        assert!(Targets::new(vec![Value::Null]).is_none());
        assert!(Targets::new(vec![Value::Null, Value::Null]).is_none());
        assert!(Targets::new(vec![Value::Null, Value::Int64(1)]).is_some());
    }

    /// Nodes of `labels` with `k: 5` at `ids`, as later writes do.
    fn create_at(
        store: &RowGroupStore,
        ids: &[u64],
        node_labels: &[&str],
        values: &[(&str, Value)],
    ) {
        let ops: Vec<DataOp> = ids
            .iter()
            .map(|&id| DataOp::CreateNode {
                id: NodeId::new(id),
                labels: labels(node_labels),
                properties: properties(values),
            })
            .collect();
        apply_all(
            store,
            &ops,
            Writer::Replay {
                epoch: next_epoch(store),
            },
        )
        .unwrap();
    }

    /// A store with room for ids in two groups, and a node of `A` with `k: 5` in the first.
    fn two_groups() -> RowGroupStore {
        let store = RowGroupStore::new();
        store.reserve_node_ids(ROWS_PER_GROUP + 8).unwrap();
        create_at(&store, &[1], &["A"], &[("k", Value::Int64(5))]);
        store
    }

    /// Reads from `from` to the end; the nodes found, as raw ids.
    fn read_on(store: &RowGroupStore, query: &mut KeyColumnQuery, mut from: u64) -> Vec<u64> {
        let mut found = Vec::new();
        loop {
            let group = store
                .key_column_group(query, from, ReadView::latest())
                .unwrap();
            found.extend(group.nodes.iter().map(|id| id.as_u64()));
            match group.end {
                Some(end) => from = end,
                None => return found,
            }
        }
    }

    /// The first read of one query, then the second group's: what appears between them.
    fn first_group(store: &RowGroupStore, query: &mut KeyColumnQuery) -> Vec<u64> {
        let group = store
            .key_column_group(query, 0, ReadView::latest())
            .unwrap();
        group.nodes.iter().map(|id| id.as_u64()).collect()
    }

    const SECOND_GROUP: u64 = ROWS_PER_GROUP;

    /// A label of a union no node had at the first read is read by the next: its id is not kept as
    /// missing.
    #[test]
    fn a_label_of_a_union_that_appears_between_two_reads_is_read() {
        let store = two_groups();
        let mut query = KeyColumnQuery::new(&[], &["A", "B"], &[], "k", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), [1]);
        create_at(
            &store,
            &[SECOND_GROUP + 1],
            &["B"],
            &[("k", Value::Int64(5))],
        );
        assert_eq!(
            read_on(&store, &mut query, SECOND_GROUP),
            [SECOND_GROUP + 1]
        );
    }

    /// A label left out that no node had at the first read still leaves out its rows after it appears:
    /// the index that holds them may have been read by then.
    #[test]
    fn a_label_left_out_that_appears_between_two_reads_stays_out() {
        let store = two_groups();
        let mut query = KeyColumnQuery::new(&[], &[], &["B"], "k", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), [1]);
        create_at(
            &store,
            &[SECOND_GROUP + 1],
            &["B"],
            &[("k", Value::Int64(5))],
        );
        create_at(
            &store,
            &[SECOND_GROUP + 2],
            &["A"],
            &[("k", Value::Int64(5))],
        );
        assert_eq!(
            read_on(&store, &mut query, SECOND_GROUP),
            [SECOND_GROUP + 2]
        );
    }

    /// A required label no node had finds nothing, and once it has a node a read finds it: nothing
    /// was kept from the read that knew no such name.
    #[test]
    fn a_required_label_that_appears_between_two_reads_is_read() {
        let store = two_groups();
        let mut query = KeyColumnQuery::new(&["B"], &[], &[], "k", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), Vec::<u64>::new());
        create_at(
            &store,
            &[SECOND_GROUP + 1],
            &["B"],
            &[("k", Value::Int64(5))],
        );
        assert_eq!(
            read_on(&store, &mut query, SECOND_GROUP),
            [SECOND_GROUP + 1]
        );
        assert_eq!(read_on(&store, &mut query, 0), [SECOND_GROUP + 1]);
    }

    /// As for a label, a key no node had.
    #[test]
    fn a_key_that_appears_between_two_reads_is_read() {
        let store = two_groups();
        let mut query = KeyColumnQuery::new(&["A"], &[], &[], "late", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), Vec::<u64>::new());
        create_at(
            &store,
            &[SECOND_GROUP + 1],
            &["A"],
            &[("late", Value::Int64(5))],
        );
        assert_eq!(
            read_on(&store, &mut query, SECOND_GROUP),
            [SECOND_GROUP + 1]
        );
    }

    /// With every name known the ids are kept, and a label that appears later changes nothing.
    #[test]
    fn a_new_label_leaves_a_read_of_known_names_as_it_is() {
        let store = two_groups();
        create_at(&store, &[2], &["B"], &[("k", Value::Int64(5))]);
        let mut query = KeyColumnQuery::new(&["A"], &[], &["B"], "k", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), [1]);
        create_at(
            &store,
            &[SECOND_GROUP + 1],
            &["C"],
            &[("k", Value::Int64(5))],
        );
        create_at(
            &store,
            &[SECOND_GROUP + 2],
            &["A", "C"],
            &[("k", Value::Int64(5))],
        );
        create_at(
            &store,
            &[SECOND_GROUP + 3],
            &["A", "B"],
            &[("k", Value::Int64(5))],
        );
        assert_eq!(
            read_on(&store, &mut query, SECOND_GROUP),
            [SECOND_GROUP + 2]
        );
    }

    /// Reading the labels left out too, after a read that kept their ids.
    #[test]
    fn reading_every_label_after_a_first_read_reads_the_rows_left_out() {
        let store = two_groups();
        create_at(&store, &[2], &["B"], &[("k", Value::Int64(5))]);
        let mut query = KeyColumnQuery::new(&[], &[], &["B"], "k", [Value::Int64(5)]);
        assert_eq!(first_group(&store, &mut query), [1]);
        query.read_every_label();
        assert_eq!(read_on(&store, &mut query, 0), [1, 2]);
    }

    /// Text that cannot reach a number key is left out by its last byte, and what `=` finds is not:
    /// the keys' last digits with the sign and the leading zeros of each, the extremes, and text
    /// ending in anything else.
    #[test]
    fn the_last_digit_rules_out_only_what_equals_rules_out() {
        let lists = [
            vec![Value::Int64(0)],
            vec![Value::Int64(10)],
            vec![Value::Int64(-7)],
            vec![Value::Int64(42), Value::Int64(-42), Value::Int64(1_003)],
            vec![Value::Int64(i64::MIN)],
            vec![Value::Int64(i64::MAX)],
        ];
        let texts = [
            "",
            " ",
            "0",
            "-0",
            "+0",
            "00",
            "10",
            "010",
            "+10",
            "-10",
            "7",
            "-7",
            "07",
            "-07",
            "+7",
            "42",
            "042",
            "+42",
            "-42",
            "-042",
            "42.0",
            "4.2e1",
            "42 ",
            " 42",
            "42a",
            "a42",
            "1003",
            "01003",
            "-9223372036854775808",
            "9223372036854775808",
            "9223372036854775807",
            "+9223372036854775807",
            "\u{0664}\u{0662}",
            "42\u{0660}",
        ];
        for keys in lists {
            let targets = Targets::new(keys.clone()).unwrap();
            for text in texts {
                let value = string(text);
                let by_equals = keys.iter().any(|key| equal(key, &value));
                assert_eq!(
                    targets.finds(&value),
                    by_equals,
                    "{text:?} against {keys:?}"
                );
                assert_eq!(
                    targets.entry_is_found(text.as_bytes()),
                    by_equals,
                    "the dictionary's test, {text:?} against {keys:?}"
                );
            }
        }
    }

    /// One string key compares its last byte first: keys and text that share an end, a start, or
    /// have nothing to compare.
    #[test]
    fn one_string_key_finds_only_itself() {
        for key in ["", "a", "person_12", "person_22", "\u{00e9}"] {
            let targets = Targets::new(vec![string(key)]).unwrap();
            for text in [
                "",
                "a",
                "b",
                "person_12",
                "person_22",
                "person_2",
                "erson_12",
                "\u{00e9}",
                "e",
            ] {
                assert_eq!(
                    targets.entry_is_found(text.as_bytes()),
                    text == key,
                    "{text:?} against {key:?}"
                );
            }
        }
    }
}
