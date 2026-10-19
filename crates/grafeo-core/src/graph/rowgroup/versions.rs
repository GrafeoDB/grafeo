//! The version info of a row group's rows, per vector of 2,048 rows
//! (masterplan 6.6, after DuckDB's version info per vector).
//!
//! A vector is flat while every reader sees its rows alike: one bit per row
//! says whether the row exists. The first write that some readers must not
//! see yet (a transaction's create or delete, a commit newer than a
//! reader's epoch) turns the vector into one stamp pair per row: who created
//! the row and who deleted it. Once every such write is committed at or
//! before the horizon (the oldest epoch a reader can still read at), the
//! vector collapses back to its bits, and the rows deleted by then are gone.
//! So rows cost one bit each for their versions, but where a reader can
//! still tell them apart.

use super::bitset::Bitset;
use super::{ABSENT, PENDING, RowVersion};

/// The rows of a vector.
pub(crate) const VECTOR_ROWS: usize = 2_048;

/// The version of a row every reader sees.
const ALWAYS: RowVersion = RowVersion {
    created: 0,
    deleted: ABSENT,
};

/// The versions of one vector of rows.
#[derive(Debug)]
enum Vector {
    /// Every reader sees the rows alike: the rows that exist.
    Flat(Bitset),
    /// A stamp pair per row.
    Stamped(Box<[RowVersion]>),
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
            Some(Vector::Stamped(stamps)) => stamps[row % VECTOR_ROWS],
        }
    }

    /// The version of `row`, to change: its vector gets a stamp pair per
    /// row.
    pub(super) fn get_mut(&mut self, row: usize) -> &mut RowVersion {
        let index = row / VECTOR_ROWS;
        if index >= self.vectors.len() {
            self.vectors.resize_with(index + 1, Vector::default);
        }
        let vector = &mut self.vectors[index];
        if let Vector::Flat(exists) = vector {
            let mut stamps = vec![RowVersion::NONE; VECTOR_ROWS].into_boxed_slice();
            for row in exists.rows() {
                stamps[row] = ALWAYS;
            }
            *vector = Vector::Stamped(stamps);
        }
        match vector {
            Vector::Stamped(stamps) => &mut stamps[row % VECTOR_ROWS],
            Vector::Flat(_) => unreachable!("stamped by the lines above"),
        }
    }

    /// Makes `row` a row every reader sees, as a load does: a flat vector
    /// stays flat.
    pub(super) fn set_always(&mut self, row: usize) {
        let index = row / VECTOR_ROWS;
        if index >= self.vectors.len() {
            self.vectors.resize_with(index + 1, Vector::default);
        }
        match &mut self.vectors[index] {
            Vector::Flat(exists) => exists.set(row % VECTOR_ROWS),
            Vector::Stamped(stamps) => stamps[row % VECTOR_ROWS] = ALWAYS,
        }
    }

    /// Takes `row` out: it no longer exists for any reader. A flat vector
    /// stays flat, as a load's deleted rows leave it.
    pub(super) fn remove(&mut self, row: usize) {
        match self.vectors.get_mut(row / VECTOR_ROWS) {
            None => {}
            Some(Vector::Flat(exists)) => exists.clear(row % VECTOR_ROWS),
            Some(Vector::Stamped(stamps)) => stamps[row % VECTOR_ROWS] = RowVersion::NONE,
        }
    }

    /// The rows that exist, ascending, each with its version.
    pub(super) fn rows(&self) -> impl Iterator<Item = (usize, RowVersion)> + '_ {
        self.vectors.iter().enumerate().flat_map(|(index, vector)| {
            let rows: Box<dyn Iterator<Item = (usize, RowVersion)> + '_> = match vector {
                Vector::Flat(exists) => Box::new(exists.rows().map(|row| (row, ALWAYS))),
                Vector::Stamped(stamps) => Box::new(
                    stamps
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, version)| version.exists()),
                ),
            };
            rows.map(move |(row, version)| (index * VECTOR_ROWS + row, version))
        })
    }

    /// Collapses every vector whose stamps are all committed at or before
    /// `horizon`: its rows deleted by then are gone, the others become rows
    /// every reader sees. Returns the rows that are gone, ascending; the
    /// caller takes their data out.
    pub(super) fn collapse(&mut self, horizon: u64) -> Vec<usize> {
        let settled = |stamp: u64| stamp & PENDING == 0 && stamp <= horizon;
        let mut gone = Vec::new();
        for (index, vector) in self.vectors.iter_mut().enumerate() {
            let Vector::Stamped(stamps) = vector else {
                continue;
            };
            let collapsible = stamps.iter().all(|version| {
                !version.exists()
                    || (settled(version.created)
                        && (version.deleted == ABSENT || settled(version.deleted)))
            });
            if !collapsible {
                continue;
            }
            let mut exists = Bitset::default();
            for (row, version) in stamps.iter().enumerate() {
                if !version.exists() {
                    continue;
                }
                if version.deleted == ABSENT {
                    exists.set(row);
                } else {
                    gone.push(index * VECTOR_ROWS + row);
                }
            }
            *vector = Vector::Flat(exists);
        }
        gone
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
                    Vector::Stamped(stamps) => std::mem::size_of_val::<[RowVersion]>(stamps),
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::{ALWAYS, VECTOR_ROWS, Versions};
    use crate::graph::rowgroup::{ABSENT, PENDING, RowVersion};

    fn existing(versions: &Versions) -> Vec<usize> {
        versions.rows().map(|(row, _)| row).collect()
    }

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
        *versions.get_mut(7) = RowVersion {
            created: PENDING | transaction,
            deleted: ABSENT,
        };
        assert_eq!(
            versions.get(5),
            ALWAYS,
            "the vector's other rows keep their version"
        );
        assert_eq!(existing(&versions), [0, 5, 7, VECTOR_ROWS + 3]);
        // Stamping replaces the vector's bits with a stamp pair per row; the
        // other vector keeps its bits.
        let stamps = VECTOR_ROWS * std::mem::size_of::<RowVersion>();
        assert!(
            versions.heap_bytes() >= stamps,
            "the written vector holds a stamp pair per row"
        );
        assert!(
            versions.heap_bytes() < flat + stamps,
            "the other vector stays flat"
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
        *versions.get_mut(7) = RowVersion {
            created: PENDING | transaction,
            deleted: ABSENT,
        };
        versions.remove(7);
        assert_eq!(versions.get(7), RowVersion::NONE, "a stamped row goes too");
        assert_eq!(existing(&versions), [0, 6]);
    }

    #[test]
    fn a_vector_collapses_once_every_stamp_is_at_or_before_the_horizon() {
        let mut versions = Versions::default();
        let transaction = 19;
        *versions.get_mut(1) = RowVersion {
            created: 3,
            deleted: ABSENT,
        };
        *versions.get_mut(2) = RowVersion {
            created: 3,
            deleted: 7,
        };
        *versions.get_mut(VECTOR_ROWS) = RowVersion {
            created: PENDING | transaction,
            deleted: ABSENT,
        };
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
}
