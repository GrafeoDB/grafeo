//! The version info of a row group's rows, per vector of 2,048 rows
//! (masterplan 6.6, after DuckDB's version info per vector).
//!
//! A vector is flat while every reader sees its rows alike: one bit per row
//! says whether the row exists. The first write that some readers must not
//! see yet (a transaction's create or delete, a commit newer than a
//! reader's epoch) stamps the vector, as DuckDB's `ChunkVectorInfo` does:
//! who created its rows is one constant stamp while they share it (a vector
//! one transaction filled, open and once committed) and a stamp per row
//! once they differ; who deleted them is, from the vector's first delete
//! on, a bit per row that has a delete stamp and the stamps themselves: a
//! short list while few rows have one, a stamp per row past that. When
//! every stamp of a vector is committed at or before the horizon (the
//! oldest epoch a reader can still read at), the vector collapses back to
//! its bits and the rows deleted by then are gone. So rows cost one bit
//! each for their versions, but where a reader can still tell them apart,
//! and a collapse costs what the vector's deletes do, not its rows: a
//! collect after each commit (see `collect.rs`) visits a vector a few
//! deletes met for a few words, not for its 2,048 rows, and allocates no
//! array for them.

use std::ops::Range;

use smallvec::SmallVec;

use super::bitset::Bitset;
use super::{ABSENT, PENDING, Read, RowVersion};

/// The rows of a vector.
pub(crate) const VECTOR_ROWS: usize = 2_048;

/// The version of a row every reader sees.
const ALWAYS: RowVersion = RowVersion {
    created: 0,
    deleted: ABSENT,
};

/// The words of a bit per row of a vector.
const WORDS: usize = VECTOR_ROWS / 64;

/// A stamp per row of a vector, [`ABSENT`] where it has none.
fn stamp_array() -> Box<[u64]> {
    vec![ABSENT; VECTOR_ROWS].into_boxed_slice()
}

/// How many rows of a vector keep their delete stamps in a short list;
/// the vector's next delete takes a stamp per row.
const FEW_DELETES: usize = 32;

/// The delete stamps of a vector's rows.
#[derive(Debug)]
enum DeleteStamps {
    /// The stamps of at most [`FEW_DELETES`] rows, by row ascending: a
    /// vector a few deletes met costs a short list, not a stamp per row.
    Few(SmallVec<[(u16, u64); 4]>),
    /// A stamp per row, [`ABSENT`] where no one deleted it.
    PerRow(Box<[u64]>),
}

/// Who deleted a vector's rows, from its first delete on.
#[derive(Debug)]
struct Deletes {
    /// A bit per row with a delete stamp: the rows a collapse, its check
    /// and [`Versions::deleted_by`] visit, a word at a time, so they cost
    /// what the vector's deletes do, not its rows.
    rows: [u64; WORDS],
    stamps: DeleteStamps,
}

impl Deletes {
    fn new() -> Box<Self> {
        Box::new(Self {
            rows: [0; WORDS],
            stamps: DeleteStamps::Few(SmallVec::new()),
        })
    }

    /// Who deleted `row` ([`ABSENT`]: no one).
    ///
    /// Out of line: inlined into every visibility check, it made the scans
    /// over flat vectors that never reach it slower (the in-degree scan of a
    /// store without backward edges took twice as long in `store_apply`).
    #[inline(never)]
    fn get(&self, row: usize) -> u64 {
        if self.rows[row / 64] & (1 << (row % 64)) == 0 {
            return ABSENT;
        }
        match &self.stamps {
            DeleteStamps::Few(few) => few
                .binary_search_by_key(&key(row), |(at, _)| *at)
                .map_or(ABSENT, |place| few[place].1),
            DeleteStamps::PerRow(stamps) => stamps[row],
        }
    }

    /// Sets who deleted `row` ([`ABSENT`]: no one).
    fn set(&mut self, row: usize, stamp: u64) {
        let bit = 1 << (row % 64);
        if stamp == ABSENT {
            self.rows[row / 64] &= !bit;
        } else {
            self.rows[row / 64] |= bit;
        }
        match &mut self.stamps {
            DeleteStamps::PerRow(stamps) => stamps[row] = stamp,
            DeleteStamps::Few(few) => {
                match (few.binary_search_by_key(&key(row), |(at, _)| *at), stamp) {
                    (Ok(place), ABSENT) => {
                        few.remove(place);
                    }
                    (Ok(place), _) => few[place].1 = stamp,
                    (Err(_), ABSENT) => {}
                    (Err(place), _) if few.len() < FEW_DELETES => {
                        few.insert(place, (key(row), stamp));
                    }
                    (Err(_), _) => {
                        let mut stamps = stamp_array();
                        for (at, stamp) in few.iter() {
                            stamps[usize::from(*at)] = *stamp;
                        }
                        stamps[row] = stamp;
                        self.stamps = DeleteStamps::PerRow(stamps);
                    }
                }
            }
        }
    }

    /// The rows of `exists` with a delete stamp, ascending, each with it.
    fn of<'a>(&'a self, exists: &'a Bitset) -> impl Iterator<Item = (usize, u64)> + 'a {
        self.rows
            .iter()
            .enumerate()
            .flat_map(move |(index, word)| {
                let mut bits = word & exists.word(index);
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some(index * 64 + bit)
                })
            })
            .map(|row| (row, self.get(row)))
    }

    fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.stamps {
                DeleteStamps::Few(few) if few.spilled() => {
                    few.capacity() * std::mem::size_of::<(u16, u64)>()
                }
                DeleteStamps::Few(_) => 0,
                DeleteStamps::PerRow(stamps) => std::mem::size_of_val::<[u64]>(stamps),
            }
    }
}

/// A row of a vector as the short list of delete stamps keeps it.
fn key(row: usize) -> u16 {
    u16::try_from(row).expect("a row of a vector is below 2,048")
}

/// Who created a stamped vector's rows.
#[derive(Debug)]
enum Inserts {
    /// Every row that exists has this stamp (DuckDB's same inserted id).
    Constant(u64),
    /// A stamp per row.
    PerRow(Box<[u64]>),
}

/// The stamps of a vector's rows.
#[derive(Debug)]
struct Stamped {
    /// The rows that exist (created, by anyone, and not undone), so an
    /// existence check reads a bit, as in a flat vector.
    exists: Bitset,
    created: Inserts,
    /// Who deleted each row, from the vector's first delete on.
    deleted: Option<Box<Deletes>>,
}

impl Stamped {
    /// The stamps of the rows of `exists`, every reader's.
    fn from_flat(exists: &Bitset) -> Self {
        Self {
            exists: exists.clone(),
            created: Inserts::Constant(ALWAYS.created),
            deleted: None,
        }
    }

    /// Who created `row`, a row that exists.
    fn creator(&self, row: usize) -> u64 {
        match &self.created {
            Inserts::Constant(stamp) => *stamp,
            Inserts::PerRow(stamps) => stamps[row],
        }
    }

    /// Who deleted `row` ([`ABSENT`]: no one).
    fn deleter(&self, row: usize) -> u64 {
        self.deleted
            .as_ref()
            .map_or(ABSENT, |deletes| deletes.get(row))
    }

    fn get(&self, row: usize) -> RowVersion {
        if !self.exists.get(row) {
            return RowVersion::NONE;
        }
        RowVersion {
            created: self.creator(row),
            deleted: self.deleter(row),
        }
    }

    /// Whether every row that exists, `row` aside, was created by `stamp`.
    fn others_created_by(&self, row: usize, stamp: u64) -> bool {
        match &self.created {
            Inserts::Constant(constant) => {
                *constant == stamp || self.exists.rows().all(|other| other == row)
            }
            Inserts::PerRow(_) => false,
        }
    }

    fn set_created(&mut self, row: usize, stamp: u64) {
        match &mut self.created {
            Inserts::PerRow(stamps) => stamps[row] = stamp,
            Inserts::Constant(constant) => {
                let constant = *constant;
                if self.others_created_by(row, stamp) {
                    self.created = Inserts::Constant(stamp);
                } else {
                    // Rows of two creators: a stamp per row from now on.
                    let mut stamps = stamp_array();
                    for other in self.exists.rows() {
                        stamps[other] = constant;
                    }
                    stamps[row] = stamp;
                    self.created = Inserts::PerRow(stamps);
                }
            }
        }
        self.exists.set(row);
    }

    fn set_deleted(&mut self, row: usize, stamp: u64) {
        match &mut self.deleted {
            Some(deletes) => deletes.set(row, stamp),
            None if stamp == ABSENT => {}
            None => {
                let mut deletes = Deletes::new();
                deletes.set(row, stamp);
                self.deleted = Some(deletes);
            }
        }
    }

    fn remove(&mut self, row: usize) {
        self.exists.clear(row);
        if let Inserts::PerRow(stamps) = &mut self.created {
            stamps[row] = ABSENT;
        }
        if let Some(deletes) = &mut self.deleted {
            deletes.set(row, ABSENT);
        }
    }

    /// Commits `from`'s creates among `rows` at `to`; returns the rows of
    /// `rows` `from` created, ascending. A constant `from` takes `to` for
    /// the whole vector, as the transaction's whole commit stamps in one
    /// turn; its rows a later call names are then found at `to`.
    fn commit_created(&mut self, rows: Range<usize>, from: u64, to: u64) -> Vec<usize> {
        match &mut self.created {
            Inserts::Constant(constant) if *constant == from || *constant == to => {
                *constant = to;
                self.exists
                    .rows()
                    .filter(|row| rows.contains(row))
                    .collect()
            }
            Inserts::Constant(_) => Vec::new(),
            Inserts::PerRow(stamps) => {
                let mut committed = Vec::new();
                for row in rows {
                    if stamps[row] == from {
                        stamps[row] = to;
                        committed.push(row);
                    }
                }
                committed
            }
        }
    }

    /// Whether every stamp of a row that exists is a commit at or before
    /// `horizon`.
    fn settled_by(&self, horizon: u64) -> bool {
        let settled = |stamp: u64| stamp & PENDING == 0 && stamp <= horizon;
        let created = match &self.created {
            Inserts::Constant(stamp) => self.exists.is_empty() || settled(*stamp),
            Inserts::PerRow(stamps) => self.exists.rows().all(|row| settled(stamps[row])),
        };
        created
            && self.deleted.as_ref().is_none_or(|deletes| {
                deletes
                    .of(&self.exists)
                    .all(|(_, stamp)| stamp == ABSENT || settled(stamp))
            })
    }

    fn heap_bytes(&self) -> usize {
        let array = |stamps: &[u64]| std::mem::size_of_val::<[u64]>(stamps);
        std::mem::size_of::<Self>()
            + self.exists.heap_bytes()
            + match &self.created {
                Inserts::Constant(_) => 0,
                Inserts::PerRow(stamps) => array(stamps),
            }
            + self
                .deleted
                .as_ref()
                .map_or(0, |deletes| deletes.heap_bytes())
    }
}

/// The versions of one vector of rows.
#[derive(Debug)]
enum Vector {
    /// Every reader sees the rows alike: the rows that exist.
    Flat(Bitset),
    /// Who created and who deleted each row.
    Stamped(Box<Stamped>),
}

impl Default for Vector {
    fn default() -> Self {
        Self::Flat(Bitset::default())
    }
}

/// The versions of a row group's rows.
#[derive(Debug, Default)]
pub(super) struct Versions {
    vectors: Vec<Vector>,
}

impl Versions {
    /// The version of `row`: [`RowVersion::NONE`] for a row that does not
    /// exist.
    pub(super) fn get(&self, row: usize) -> RowVersion {
        match self.vectors.get(row / VECTOR_ROWS) {
            None => RowVersion::NONE,
            Some(Vector::Flat(exists)) => {
                if exists.get(row % VECTOR_ROWS) {
                    ALWAYS
                } else {
                    RowVersion::NONE
                }
            }
            Some(Vector::Stamped(stamped)) => stamped.get(row % VECTOR_ROWS),
        }
    }

    /// The stamps of vector `index`, stamping it if it is flat.
    fn stamped_mut(&mut self, index: usize) -> &mut Stamped {
        if index >= self.vectors.len() {
            self.vectors.resize_with(index + 1, Vector::default);
        }
        let vector = &mut self.vectors[index];
        if let Vector::Flat(exists) = vector {
            *vector = Vector::Stamped(Box::new(Stamped::from_flat(exists)));
        }
        match vector {
            Vector::Stamped(stamped) => stamped,
            Vector::Flat(_) => unreachable!("stamped by the lines above"),
        }
    }

    /// Whether `row` exists in any reader's view (created, by anyone, and
    /// not undone): a bit test in either form of its vector.
    /// Whether `read` sees `row`, in one lookup: a row of a flat vector,
    /// the common case, is seen by every reader once it exists (a bit
    /// test); a stamped vector's row by its stamps.
    pub(super) fn visible(&self, row: usize, read: Read) -> bool {
        match self.vectors.get(row / VECTOR_ROWS) {
            None => false,
            Some(Vector::Flat(exists)) => exists.get(row % VECTOR_ROWS),
            Some(Vector::Stamped(stamped)) => stamped.get(row % VECTOR_ROWS).visible(read),
        }
    }

    pub(super) fn exists(&self, row: usize) -> bool {
        match self.vectors.get(row / VECTOR_ROWS) {
            None => false,
            Some(Vector::Flat(exists)) => exists.get(row % VECTOR_ROWS),
            Some(Vector::Stamped(stamped)) => stamped.exists.get(row % VECTOR_ROWS),
        }
    }

    /// Who deleted `row`, a row that exists ([`ABSENT`]: no one), without
    /// reading who created it.
    pub(super) fn deleted(&self, row: usize) -> u64 {
        match self.vectors.get(row / VECTOR_ROWS) {
            Some(Vector::Stamped(stamped)) => stamped.deleter(row % VECTOR_ROWS),
            Some(Vector::Flat(_)) | None => ABSENT,
        }
    }

    /// The rows of `rows` that exist, ascending: a word of bits at a time.
    pub(super) fn existing<'a>(&'a self, rows: &'a Bitset) -> impl Iterator<Item = usize> + 'a {
        const WORDS: usize = VECTOR_ROWS / 64;
        rows.words()
            .iter()
            .enumerate()
            .flat_map(move |(index, word)| {
                let exists = match self.vectors.get(index / WORDS) {
                    None => 0,
                    Some(Vector::Flat(exists)) => exists.word(index % WORDS),
                    Some(Vector::Stamped(stamped)) => stamped.exists.word(index % WORDS),
                };
                let mut bits = word & exists;
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some(index * 64 + bit)
                })
            })
    }

    /// Sets who created `row`: its vector holds stamps from then on.
    pub(super) fn set_created(&mut self, row: usize, stamp: u64) {
        self.stamped_mut(row / VECTOR_ROWS)
            .set_created(row % VECTOR_ROWS, stamp);
    }

    /// Sets who deleted `row` ([`ABSENT`]: no one); the row exists.
    pub(super) fn set_deleted(&mut self, row: usize, stamp: u64) {
        self.stamped_mut(row / VECTOR_ROWS)
            .set_deleted(row % VECTOR_ROWS, stamp);
    }

    /// Commits transaction stamp `from`'s creates among `rows` at epoch
    /// `to`, as `stamp` does, for one row or a bulk range; returns the rows
    /// of `rows` `from` created, as runs, ascending. A vector whose rows
    /// `from` created alone keeps one constant stamp: it takes `to` whole
    /// at the first row committed, so every row a later call of the same
    /// commit names is found at `to` (`to` is this commit's epoch alone).
    pub(super) fn commit_created(
        &mut self,
        rows: Range<usize>,
        from: u64,
        to: u64,
    ) -> Vec<Range<usize>> {
        let mut committed: Vec<Range<usize>> = Vec::new();
        let mut start = rows.start;
        while start < rows.end {
            let index = start / VECTOR_ROWS;
            let base = index * VECTOR_ROWS;
            let end = rows.end.min(base + VECTOR_ROWS);
            if let Some(Vector::Stamped(stamped)) = self.vectors.get_mut(index) {
                for row in stamped.commit_created(start - base..end - base, from, to) {
                    let row = base + row;
                    match committed.last_mut() {
                        Some(run) if run.end == row => run.end += 1,
                        _ => committed.push(row..row + 1),
                    }
                }
            }
            start = end;
        }
        committed
    }

    /// Makes `row` a row every reader sees, as a load does: a flat vector
    /// stays flat.
    pub(super) fn set_always(&mut self, row: usize) {
        let index = row / VECTOR_ROWS;
        if index >= self.vectors.len() {
            self.vectors.resize_with(index + 1, Vector::default);
        }
        let offset = row % VECTOR_ROWS;
        match &mut self.vectors[index] {
            Vector::Flat(exists) => exists.set(offset),
            Vector::Stamped(stamped) => {
                stamped.set_created(offset, ALWAYS.created);
                stamped.set_deleted(offset, ABSENT);
            }
        }
    }

    /// Takes `row` out: it no longer exists for any reader. A flat vector
    /// stays flat, as a load's deleted rows leave it.
    pub(super) fn remove(&mut self, row: usize) {
        let offset = row % VECTOR_ROWS;
        match self.vectors.get_mut(row / VECTOR_ROWS) {
            None => {}
            Some(Vector::Flat(exists)) => exists.clear(offset),
            Some(Vector::Stamped(stamped)) => stamped.remove(offset),
        }
    }

    /// The rows that exist, ascending, each with its version.
    pub(super) fn rows(&self) -> impl Iterator<Item = (usize, RowVersion)> + '_ {
        self.vectors.iter().enumerate().flat_map(|(index, vector)| {
            let base = index * VECTOR_ROWS;
            let rows: Box<dyn Iterator<Item = (usize, RowVersion)> + '_> = match vector {
                Vector::Flat(exists) => Box::new(exists.rows().map(|row| (row, ALWAYS))),
                Vector::Stamped(stamped) => Box::new(
                    stamped
                        .exists
                        .rows()
                        .map(move |row| (row, stamped.get(row))),
                ),
            };
            rows.map(move |(row, version)| (base + row, version))
        })
    }

    /// Collapses every vector whose stamps are all committed at or before
    /// `horizon`: its rows deleted by then are gone, the others become rows
    /// every reader sees. Returns the rows that are gone, ascending; the
    /// caller takes their data out.
    pub(super) fn collapse(&mut self, horizon: u64) -> Vec<usize> {
        let mut gone = Vec::new();
        for index in 0..self.vectors.len() {
            self.collapse_into(index, horizon, &mut gone);
        }
        gone
    }

    /// [`collapse`](Self::collapse) of vector `index` alone.
    pub(super) fn collapse_vector(&mut self, index: usize, horizon: u64) -> Vec<usize> {
        let mut gone = Vec::new();
        self.collapse_into(index, horizon, &mut gone);
        gone
    }

    /// Collapses vector `index` if its stamps are all committed at or before
    /// `horizon`, pushing its rows that are gone onto `gone`.
    fn collapse_into(&mut self, index: usize, horizon: u64, gone: &mut Vec<usize>) {
        let Some(vector) = self.vectors.get_mut(index) else {
            return;
        };
        let Vector::Stamped(stamped) = vector else {
            return;
        };
        if !stamped.settled_by(horizon) {
            return;
        }
        // Built anew, a word at a time in ascending order as a load sets
        // its rows, so it holds what a load would; the rows with a delete
        // stamp alone are looked at.
        let mut exists = Bitset::default();
        let base = index * VECTOR_ROWS;
        for (at, word) in stamped.exists.words().iter().enumerate() {
            let mut kept = *word;
            if let Some(deletes) = &stamped.deleted {
                let mut bits = kept & deletes.rows[at];
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if deletes.get(at * 64 + bit) != ABSENT {
                        kept &= !(1 << bit);
                        gone.push(base + at * 64 + bit);
                    }
                }
            }
            exists.set_word(at, kept);
        }
        *vector = Vector::Flat(exists);
    }

    /// Whether vector `index` holds stamps.
    pub(super) fn is_stamped(&self, index: usize) -> bool {
        matches!(self.vectors.get(index), Some(Vector::Stamped(_)))
    }

    /// Whether no row exists in any version: every vector is flat and
    /// empty.
    pub(super) fn is_empty(&self) -> bool {
        self.vectors
            .iter()
            .all(|vector| matches!(vector, Vector::Flat(exists) if exists.is_empty()))
    }

    /// The vectors that hold stamps.
    #[cfg(test)]
    pub(super) fn stamped(&self) -> usize {
        self.vectors
            .iter()
            .filter(|vector| matches!(vector, Vector::Stamped(_)))
            .count()
    }

    /// The heap bytes the versions take.
    pub(super) fn heap_bytes(&self) -> usize {
        self.vectors.capacity() * std::mem::size_of::<Vector>()
            + self
                .vectors
                .iter()
                .map(|vector| match vector {
                    Vector::Flat(exists) => exists.heap_bytes(),
                    Vector::Stamped(stamped) => stamped.heap_bytes(),
                })
                .sum::<usize>()
    }

    /// The rows of the vectors that hold stamps, ascending: the rows whose
    /// versions some readers may tell apart (the others every reader sees
    /// alike). The text and vector indexes bring an index they take in
    /// line with them.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub(super) fn stamped_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.stamped_vectors()
            .flat_map(|(base, stamped)| stamped.exists.rows().map(move |row| base + row))
    }

    /// The rows a delete committed at or before `horizon` took out of every
    /// view at the horizon or later, ascending, whether or not their vector
    /// collapses yet: what an index keeps for them no reader there needs.
    pub(super) fn deleted_by(&self, horizon: u64) -> impl Iterator<Item = usize> + '_ {
        self.stamped_vectors().flat_map(move |(base, stamped)| {
            stamped
                .deleted
                .iter()
                .flat_map(|deletes| deletes.of(&stamped.exists))
                .filter(move |(_, deleter)| {
                    *deleter != ABSENT && deleter & PENDING == 0 && *deleter <= horizon
                })
                .map(move |(row, _)| base + row)
        })
    }

    /// The vectors that hold stamps, with their first row.
    fn stamped_vectors(&self) -> impl Iterator<Item = (usize, &Stamped)> + '_ {
        self.vectors
            .iter()
            .enumerate()
            .filter_map(|(index, vector)| match vector {
                Vector::Stamped(stamped) => Some((index * VECTOR_ROWS, &**stamped)),
                Vector::Flat(_) => None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{ALWAYS, FEW_DELETES, VECTOR_ROWS, Versions};
    use crate::graph::conformance::Rng;
    use crate::graph::rowgroup::{ABSENT, PENDING, RowVersion};

    fn existing(versions: &Versions) -> Vec<usize> {
        versions.rows().map(|(row, _)| row).collect()
    }

    /// The bytes of a stamp per row of a vector.
    const ARRAY: usize = VECTOR_ROWS * std::mem::size_of::<u64>();

    /// The bytes of a bit per row of a vector.
    const BITS: usize = VECTOR_ROWS / 8;

    #[test]
    fn loaded_rows_stay_flat_and_a_write_stamps_its_vector_alone() {
        let mut versions = Versions::default();
        for row in [0, 5, VECTOR_ROWS + 3] {
            versions.set_always(row);
        }
        let flat = versions.heap_bytes();
        assert_eq!(versions.get(5), ALWAYS);
        assert_eq!(versions.get(6), RowVersion::NONE);
        assert_eq!(versions.get(10 * VECTOR_ROWS), RowVersion::NONE);
        assert_eq!(existing(&versions), [0, 5, VECTOR_ROWS + 3]);

        let transaction = 19;
        versions.set_created(7, PENDING | transaction);
        assert_eq!(
            versions.get(5),
            ALWAYS,
            "the vector's other rows keep their version"
        );
        assert_eq!(versions.get(7).created, PENDING | transaction);
        assert_eq!(existing(&versions), [0, 5, 7, VECTOR_ROWS + 3]);
        assert_eq!(versions.stamped(), 1, "the other vector stays flat");
        assert!(
            versions.heap_bytes() >= ARRAY,
            "a write among loaded rows makes a stamp per row"
        );
        assert!(
            versions.heap_bytes() < flat + 2 * ARRAY,
            "and no delete stamps: {} bytes",
            versions.heap_bytes()
        );
    }

    #[test]
    fn a_row_taken_out_of_a_flat_vector_keeps_it_flat() {
        let mut versions = Versions::default();
        for row in [0, 5, 6] {
            versions.set_always(row);
        }
        let flat = versions.heap_bytes();
        versions.remove(5);
        versions.remove(10 * VECTOR_ROWS);
        assert_eq!(versions.get(5), RowVersion::NONE);
        assert_eq!(existing(&versions), [0, 6]);
        assert_eq!(versions.heap_bytes(), flat, "no vector is stamped");

        let transaction = 19;
        versions.set_created(7, PENDING | transaction);
        versions.set_deleted(6, PENDING | transaction);
        versions.remove(7);
        versions.remove(6);
        assert_eq!(versions.get(7), RowVersion::NONE, "a stamped row goes too");
        assert_eq!(versions.get(6), RowVersion::NONE, "and its delete stamp");
        assert_eq!(existing(&versions), [0]);
    }

    #[test]
    fn a_vector_collapses_once_every_stamp_is_at_or_before_the_horizon() {
        let mut versions = Versions::default();
        let transaction = 19;
        versions.set_created(1, 3);
        versions.set_created(2, 3);
        versions.set_deleted(2, 7);
        versions.set_created(VECTOR_ROWS, PENDING | transaction);
        let stamped = versions.heap_bytes();

        assert_eq!(
            versions.collapse(6),
            Vec::<usize>::new(),
            "a reader at 6 still sees row 2"
        );
        assert_eq!(versions.get(2).deleted, 7);
        assert_eq!(versions.collapse(7), [2], "deleted at the horizon: gone");
        assert_eq!(versions.get(1), ALWAYS);
        assert_eq!(versions.get(2), RowVersion::NONE);
        assert_eq!(
            versions.get(VECTOR_ROWS).created,
            PENDING | transaction,
            "a vector with a pending stamp stays"
        );
        assert!(versions.heap_bytes() < stamped);
        assert_eq!(existing(&versions), [1, VECTOR_ROWS]);
    }

    /// `existing` keeps the rows of a set that exist, in flat and stamped
    /// vectors and past the last vector.
    #[test]
    fn existing_rows_of_a_set_are_listed_a_word_at_a_time() {
        use crate::graph::rowgroup::bitset::Bitset;

        let mut versions = Versions::default();
        for row in [0, 63, 64, 700] {
            versions.set_always(row);
        }
        let id = 19;
        for row in [VECTOR_ROWS + 1, VECTOR_ROWS + 64] {
            versions.set_created(row, PENDING | id);
        }
        versions.set_created(VECTOR_ROWS + 2, 3);
        versions.remove(VECTOR_ROWS + 2);
        let mut rows = Bitset::default();
        for row in [
            0,
            1,
            63,
            64,
            700,
            VECTOR_ROWS + 1,
            VECTOR_ROWS + 2,
            VECTOR_ROWS + 64,
            5 * VECTOR_ROWS,
        ] {
            rows.set(row);
        }
        assert_eq!(
            versions.existing(&rows).collect::<Vec<_>>(),
            [0, 63, 64, 700, VECTOR_ROWS + 1, VECTOR_ROWS + 64]
        );
    }

    /// Two vectors one transaction filled hold one constant stamp each,
    /// open and once committed row by row or by a range across them.
    #[test]
    fn a_vector_filled_by_one_transaction_holds_one_constant_stamp() {
        let id = 19;
        let transaction = PENDING | id;
        let mut versions = Versions::default();
        for row in 0..2 * VECTOR_ROWS {
            versions.set_created(row, transaction);
        }
        let constant = versions.heap_bytes();
        assert!(
            constant < 1_024,
            "two vectors, a bit per row and a constant each: {constant} bytes"
        );
        for row in 0..3 {
            assert_eq!(
                versions.commit_created(row..row + 1, transaction, 88),
                vec![row..row + 1]
            );
        }
        assert_eq!(
            versions.commit_created(3..2 * VECTOR_ROWS, transaction, 88),
            vec![3..2 * VECTOR_ROWS],
            "the rest, across both vectors, found at the commit's epoch"
        );
        assert_eq!(versions.heap_bytes(), constant);
        assert!(versions.rows().all(|(_, version)| version.created == 88));
        assert_eq!(
            versions.commit_created(0..2 * VECTOR_ROWS, PENDING | 3, 89),
            Vec::<std::ops::Range<usize>>::new(),
            "another transaction created none of them"
        );
    }

    /// A second writer's row in a constant vector turns the inserts into a
    /// stamp per row, each row keeping its creator; its commit stamps its
    /// own rows alone.
    #[test]
    fn a_second_writers_row_turns_the_inserts_into_an_array() {
        let (first, second) = (PENDING | 1, PENDING | 2);
        let mut versions = Versions::default();
        for row in 0..10 {
            versions.set_created(row, first);
        }
        let constant = versions.heap_bytes();
        versions.set_created(10, second);
        assert!(versions.heap_bytes() >= constant + ARRAY);
        assert!((0..10).all(|row| versions.get(row).created == first));
        assert_eq!(versions.get(10).created, second);
        assert_eq!(
            versions.commit_created(0..VECTOR_ROWS, second, 5),
            vec![10..11]
        );
        assert_eq!(versions.get(9).created, first);
        assert_eq!(versions.get(10).created, 5);
        assert_eq!(
            versions.commit_created(0..VECTOR_ROWS, first, 6),
            vec![0..10]
        );
        assert_eq!(versions.collapse(6), Vec::<usize>::new());
        assert_eq!(versions.stamped(), 0);
    }

    /// A vector holds no delete stamps until its first delete, which
    /// allocates a bit per row and a short list; past [`FEW_DELETES`] rows
    /// the vector takes a stamp per row. Undoing a delete keeps the stamps
    /// until the vector collapses, which drops them all.
    #[test]
    fn the_first_delete_allocates_the_delete_stamps() {
        let transaction = PENDING | 1;
        let mut versions = Versions::default();
        for row in 0..100 {
            versions.set_created(row, 3);
        }
        let before = versions.heap_bytes();
        versions.set_deleted(50, ABSENT);
        assert_eq!(
            versions.heap_bytes(),
            before,
            "taking no delete off allocates none"
        );
        versions.set_deleted(50, transaction);
        let few = versions.heap_bytes();
        assert!(
            few > before + BITS && few < before + BITS + 1_024,
            "a bit per row and a short list: {} bytes",
            few - before
        );
        assert_eq!(versions.get(50).deleted, transaction);
        assert_eq!(versions.get(49).deleted, ABSENT);
        versions.set_deleted(51, transaction);
        versions.set_deleted(50, ABSENT);
        versions.set_deleted(51, 4);
        assert_eq!(versions.get(50).deleted, ABSENT);
        assert_eq!(versions.get(51).deleted, 4);
        assert_eq!(versions.collapse(4), [51]);
        assert!(
            versions.heap_bytes() < before,
            "the collapse drops every stamp"
        );

        // Past the short list: a stamp per row, each row keeping its own.
        let mut versions = Versions::default();
        for row in 0..VECTOR_ROWS {
            versions.set_created(row, 3);
        }
        let before = versions.heap_bytes();
        for row in 0..=FEW_DELETES {
            versions.set_deleted(2 * row, 10 + row as u64);
        }
        assert!(
            versions.heap_bytes() >= before + ARRAY + BITS,
            "{} deletes take a stamp per row",
            FEW_DELETES + 1
        );
        for row in 0..VECTOR_ROWS {
            let expected = if row % 2 == 0 && row / 2 <= FEW_DELETES {
                10 + (row / 2) as u64
            } else {
                ABSENT
            };
            assert_eq!(versions.get(row).deleted, expected, "row {row}");
        }
        let horizon = 10 + FEW_DELETES as u64;
        let gone: Vec<usize> = (0..=FEW_DELETES).map(|row| 2 * row).collect();
        assert_eq!(versions.deleted_by(horizon).collect::<Vec<_>>(), gone);
        assert_eq!(versions.collapse(horizon), gone);
        assert_eq!(versions.stamped(), 0);
    }

    /// Random creates, deletes, commits (row by row and by range, in any
    /// order), removals, loads and collapses agree with a reference of one
    /// version per row.
    #[test]
    fn versions_agree_with_a_version_per_row() {
        const ROWS: usize = 3 * VECTOR_ROWS;
        for seed in 1..=24 {
            let mut rng = Rng::new(seed);
            let mut versions = Versions::default();
            let mut model = vec![RowVersion::NONE; ROWS];
            let transactions = [PENDING | 1, PENDING | 2];
            // Every commit and immediate write at an epoch of its own.
            let mut epoch = 1_u64;
            for row in 0..rng.below(VECTOR_ROWS) {
                if rng.below(4) != 0 {
                    versions.set_always(row);
                    model[row] = ALWAYS;
                }
            }
            for step in 0..3_000 {
                let start = rng.below(ROWS);
                let end = (start + 1 + rng.below(40)).min(ROWS);
                let stamp = if rng.below(3) == 0 {
                    epoch += 1;
                    epoch
                } else {
                    transactions[rng.below(2)]
                };
                let what = format!("seed {seed}, step {step}");
                match rng.below(9) {
                    0..=2 => {
                        for row in start..end {
                            if model[row].created == ABSENT {
                                versions.set_created(row, stamp);
                                model[row].created = stamp;
                            }
                        }
                    }
                    3 | 4 => {
                        for row in start..end {
                            if model[row].created != ABSENT && model[row].deleted == ABSENT {
                                versions.set_deleted(row, stamp);
                                model[row].deleted = stamp;
                            }
                        }
                    }
                    5 => {
                        // A transaction commits all it wrote, in pieces.
                        let from = transactions[rng.below(2)];
                        epoch += 1;
                        let to = epoch;
                        let mut pieces: Vec<std::ops::Range<usize>> = Vec::new();
                        let mut at = 0;
                        while at < ROWS {
                            let next = (at + 1 + rng.below(300)).min(ROWS);
                            pieces.push(at..next);
                            at = next;
                        }
                        while !pieces.is_empty() {
                            let piece = pieces.swap_remove(rng.below(pieces.len()));
                            let expected: Vec<usize> = piece
                                .clone()
                                .filter(|row| model[*row].created == from)
                                .collect();
                            let committed: Vec<usize> = versions
                                .commit_created(piece.clone(), from, to)
                                .into_iter()
                                .flatten()
                                .collect();
                            assert_eq!(committed, expected, "{what}: commit of {piece:?}");
                            for row in piece {
                                if model[row].created == from {
                                    model[row].created = to;
                                }
                                if model[row].deleted == from {
                                    versions.set_deleted(row, to);
                                    model[row].deleted = to;
                                }
                            }
                        }
                    }
                    6 => {
                        for row in start..end {
                            versions.remove(row);
                            model[row] = RowVersion::NONE;
                        }
                    }
                    7 => {
                        let horizon = rng.below(usize::try_from(epoch).unwrap() + 1) as u64;
                        let deleted: Vec<usize> = versions.deleted_by(horizon).collect();
                        let expected_deleted: Vec<usize> = (0..ROWS)
                            .filter(|row| {
                                let deleter = model[*row].deleted;
                                model[*row].created != ABSENT
                                    && deleter != ABSENT
                                    && deleter & PENDING == 0
                                    && deleter <= horizon
                            })
                            .collect();
                        assert_eq!(deleted, expected_deleted, "{what}: deleted by {horizon}");
                        let gone = versions.collapse(horizon);
                        let mut expected_gone = Vec::new();
                        // A collapsed vector holds bits: its rows are every
                        // reader's.
                        for (index, chunk) in model.chunks_mut(VECTOR_ROWS).enumerate() {
                            let settled = |stamp: u64| stamp & PENDING == 0 && stamp <= horizon;
                            let collapsed = chunk.iter().all(|version| {
                                version.created == ABSENT
                                    || (settled(version.created)
                                        && (version.deleted == ABSENT || settled(version.deleted)))
                            });
                            if collapsed {
                                for (offset, version) in chunk.iter_mut().enumerate() {
                                    if version.created == ABSENT {
                                        continue;
                                    }
                                    if version.deleted == ABSENT {
                                        *version = ALWAYS;
                                    } else {
                                        expected_gone.push(index * VECTOR_ROWS + offset);
                                        *version = RowVersion::NONE;
                                    }
                                }
                            }
                        }
                        assert_eq!(gone, expected_gone, "{what}: the rows gone");
                    }
                    _ => {
                        for row in start..end {
                            versions.set_always(row);
                            model[row] = ALWAYS;
                        }
                    }
                }
                for row in [start, end - 1, end.min(ROWS - 1), start.saturating_sub(1)] {
                    assert_eq!(versions.get(row), model[row], "{what}: row {row}");
                }
            }
            for (row, version) in model.iter().enumerate() {
                assert_eq!(versions.get(row), *version, "seed {seed}: row {row}");
            }
            let listed: Vec<(usize, RowVersion)> = versions.rows().collect();
            let expected: Vec<(usize, RowVersion)> = model
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, version)| version.created != ABSENT)
                .collect();
            assert_eq!(listed, expected, "seed {seed}: the rows");
        }
    }
}
