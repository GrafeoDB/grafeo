//! Page allocator for container v3.
//!
//! A checkpoint writes copy-on-write: its chunks go into pages the active
//! database header does not reach. The allocator is built from the runs the
//! active image uses; every other page from [`DATA_START_PAGE`] up to the end
//! of the last used run is free. Allocation is first fit, and appends at the
//! end when no gap is large enough. Nothing is released during a checkpoint.

use std::collections::BTreeMap;

use grafeo_common::utils::error::{Error, Result};

use super::header::{DATA_START_PAGE, PAGE_SIZE};

/// A run of contiguous pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PageRun {
    /// Page number of the first page of the run.
    pub first: u64,
    /// Number of pages in the run.
    pub count: u64,
}

impl PageRun {
    /// Byte offset of the first page (`first * PAGE_SIZE`).
    ///
    /// # Errors
    ///
    /// Returns an error naming the run when the offset overflows `u64`, which
    /// a run from [`PageAllocator`] never does (a run built by hand can).
    pub fn offset(self) -> Result<u64> {
        self.first.checked_mul(PAGE_SIZE).ok_or_else(|| {
            Error::Internal(format!(
                "page run {self:?} starts beyond the file offset range"
            ))
        })
    }

    /// Page number one past the last page of the run (`first + count`).
    ///
    /// # Errors
    ///
    /// Returns an error naming the run when the sum overflows `u64`, which a
    /// run from [`PageAllocator`] never does (a run built by hand can).
    pub fn end(self) -> Result<u64> {
        self.first.checked_add(self.count).ok_or_else(|| {
            Error::Internal(format!(
                "page run {self:?} ends beyond the last page number"
            ))
        })
    }

    /// Number of pages needed for `len` bytes (0 for 0).
    #[must_use]
    pub fn for_bytes(len: u64) -> u64 {
        len.div_ceil(PAGE_SIZE)
    }
}

/// First-fit allocator over the pages of a v3 file.
#[derive(Debug, Clone)]
pub struct PageAllocator {
    /// Free gaps, keyed by first page, valued by page count. Gaps never touch.
    free: BTreeMap<u64, u64>,
    /// Page number one past the last page in use or allocated.
    end: u64,
}

impl PageAllocator {
    /// Builds an allocator from the runs the active image uses. Everything
    /// from `DATA_START_PAGE` up to the end of the last used run that no used
    /// run covers is free. Zero-length runs are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error naming the runs when two used runs overlap, when a run
    /// starts below `DATA_START_PAGE`, or when a run's end (in pages or in
    /// bytes) overflows `u64`.
    pub fn from_used(used: impl IntoIterator<Item = PageRun>) -> Result<Self> {
        let mut runs: Vec<PageRun> = used.into_iter().filter(|run| run.count > 0).collect();
        runs.sort();

        let mut free = BTreeMap::new();
        let mut end = DATA_START_PAGE;
        for run in runs {
            if run.first < DATA_START_PAGE {
                return Err(Error::Serialization(format!(
                    "page run {run:?} starts below the data start page {DATA_START_PAGE}"
                )));
            }
            let run_end = run
                .first
                .checked_add(run.count)
                .filter(|run_end| run_end.checked_mul(PAGE_SIZE).is_some())
                .ok_or_else(|| {
                    Error::Serialization(format!(
                        "page run {run:?} overflows the file offset range"
                    ))
                })?;
            if run.first < end {
                return Err(Error::Serialization(format!(
                    "page run {run:?} overlaps an earlier run that ends at page {end}"
                )));
            }
            if run.first > end {
                free.insert(end, run.first - end);
            }
            end = run_end;
        }
        Ok(Self { free, end })
    }

    /// Allocates `count` contiguous pages: the first gap that fits, else at
    /// the end. `allocate(0)` returns a zero-length run at page 0 and changes
    /// nothing (a zero-length chunk takes no pages).
    ///
    /// Every run it returns lies within the byte-offset range: its
    /// [`offset`](PageRun::offset) and [`end`](PageRun::end) never fail, and
    /// neither does `end * PAGE_SIZE`.
    ///
    /// # Errors
    ///
    /// Returns an error naming the request when no gap fits and the run at
    /// the end would pass the last page number or the last byte offset; the
    /// allocator is then unchanged.
    pub fn allocate(&mut self, count: u64) -> Result<PageRun> {
        if count == 0 {
            return Ok(PageRun { first: 0, count: 0 });
        }
        let fit = self
            .free
            .iter()
            .find(|&(_, &gap)| gap >= count)
            .map(|(&first, &gap)| (first, gap));
        if let Some((first, gap)) = fit {
            // A gap lies below `end`, so the run does too.
            self.free.remove(&first);
            if gap > count {
                self.free.insert(first + count, gap - count);
            }
            return Ok(PageRun { first, count });
        }
        let first = self.end;
        let end = first
            .checked_add(count)
            .filter(|end| end.checked_mul(PAGE_SIZE).is_some())
            .ok_or_else(|| {
                Error::Internal(format!(
                    "cannot allocate {count} pages at page {first}: the run would end beyond \
                     the file offset range"
                ))
            })?;
        self.end = end;
        Ok(PageRun { first, count })
    }

    /// Page number one past the last page in use or allocated.
    #[must_use]
    pub fn end_page(&self) -> u64 {
        self.end
    }

    /// The free gaps below the end, sorted by page number.
    #[must_use]
    pub fn free_runs(&self) -> Vec<PageRun> {
        self.free
            .iter()
            .map(|(&first, &count)| PageRun { first, count })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_between_used_runs_are_free_and_reused_first() {
        let used = [
            PageRun { first: 3, count: 2 },
            PageRun { first: 8, count: 1 },
        ];
        let mut pages = PageAllocator::from_used(used).unwrap();
        assert_eq!(pages.free_runs(), [PageRun { first: 5, count: 3 }]);
        assert_eq!(pages.allocate(2).unwrap(), PageRun { first: 5, count: 2 });
        assert_eq!(
            pages.allocate(2).unwrap(),
            PageRun { first: 9, count: 2 },
            "no gap of 2 left: the end"
        );
        assert_eq!(pages.allocate(1).unwrap(), PageRun { first: 7, count: 1 });
        assert_eq!(pages.end_page(), 11);
    }

    #[test]
    fn an_empty_file_allocates_from_the_data_start() {
        let mut pages = PageAllocator::from_used([]).unwrap();
        assert_eq!(
            pages.allocate(1).unwrap(),
            PageRun {
                first: DATA_START_PAGE,
                count: 1
            }
        );
    }

    #[test]
    fn overlapping_used_runs_are_corruption() {
        let used = [
            PageRun { first: 3, count: 4 },
            PageRun { first: 5, count: 1 },
        ];
        assert!(PageAllocator::from_used(used).is_err());
    }

    #[test]
    fn bytes_round_up_to_pages() {
        assert_eq!(PageRun::for_bytes(0), 0);
        assert_eq!(PageRun::for_bytes(1), 1);
        assert_eq!(PageRun::for_bytes(4096), 1);
        assert_eq!(PageRun::for_bytes(4097), 2);
    }

    #[test]
    fn a_run_below_the_data_start_is_corruption() {
        let error = PageAllocator::from_used([PageRun { first: 1, count: 2 }]).unwrap_err();
        assert!(
            error.to_string().contains("first: 1"),
            "error names the run: {error}"
        );
    }

    #[test]
    fn zero_length_runs_are_ignored() {
        let used = [
            PageRun { first: 0, count: 0 },
            PageRun { first: 4, count: 0 },
            PageRun { first: 3, count: 1 },
        ];
        let pages = PageAllocator::from_used(used).unwrap();
        assert_eq!(pages.free_runs().len(), 0, "no gaps");
        assert_eq!(pages.end_page(), 4);
    }

    #[test]
    fn allocating_zero_pages_changes_nothing() {
        let mut pages = PageAllocator::from_used([PageRun { first: 5, count: 1 }]).unwrap();
        let before = (pages.free_runs(), pages.end_page());
        assert_eq!(pages.allocate(0).unwrap(), PageRun { first: 0, count: 0 });
        assert_eq!((pages.free_runs(), pages.end_page()), before);
    }

    #[test]
    fn a_run_that_overflows_is_an_error_not_a_panic() {
        let huge_count = [PageRun {
            first: 3,
            count: u64::MAX,
        }];
        assert!(PageAllocator::from_used(huge_count).is_err());
        let huge_offset = [PageRun {
            first: u64::MAX / PAGE_SIZE,
            count: 1,
        }];
        assert!(PageAllocator::from_used(huge_offset).is_err());
    }

    #[test]
    fn unsorted_input_and_touching_runs_leave_no_adjacent_gaps() {
        let used = [
            PageRun {
                first: 10,
                count: 1,
            },
            PageRun { first: 3, count: 1 },
            PageRun { first: 4, count: 2 },
        ];
        let pages = PageAllocator::from_used(used).unwrap();
        assert_eq!(pages.free_runs(), [PageRun { first: 6, count: 4 }]);
    }

    #[test]
    fn a_run_offset_is_its_first_page_times_the_page_size() {
        let run = PageRun { first: 3, count: 2 };
        assert_eq!(run.offset().unwrap(), 12288);
        assert_eq!(run.end().unwrap(), 5);
    }

    /// A request whose pages would end past `u64` pages, or past the byte
    /// offset range, fails and leaves the allocator as it was; the largest
    /// request that fits still succeeds.
    #[test]
    fn an_allocation_beyond_the_offset_range_is_an_error_and_changes_nothing() {
        let mut pages = PageAllocator::from_used([PageRun { first: 3, count: 1 }]).unwrap();
        let before = (pages.free_runs(), pages.end_page());
        for count in [u64::MAX, u64::MAX - 3, u64::MAX / PAGE_SIZE] {
            let error = pages
                .allocate(count)
                .expect_err("the run would end beyond the offset range")
                .to_string();
            assert!(
                error.contains(&count.to_string()),
                "the error names the request: {error}"
            );
            assert_eq!(
                (pages.free_runs(), pages.end_page()),
                before,
                "a refused allocation changes nothing"
            );
        }
        let largest = u64::MAX / PAGE_SIZE - 4;
        let run = pages.allocate(largest).unwrap();
        assert_eq!(
            run,
            PageRun {
                first: 4,
                count: largest
            }
        );
        assert_eq!(
            run.end().unwrap() * PAGE_SIZE,
            (u64::MAX / PAGE_SIZE) * PAGE_SIZE
        );
    }

    /// A run built by hand can lie beyond the offset range: its offset and
    /// end are errors, not saturated values.
    #[test]
    fn a_run_beyond_the_offset_range_has_no_offset_or_end() {
        let far = PageRun {
            first: u64::MAX / PAGE_SIZE + 1,
            count: 1,
        };
        assert!(far.offset().is_err(), "{far:?} starts past the last byte");
        let long = PageRun {
            first: 3,
            count: u64::MAX,
        };
        assert!(long.end().is_err(), "{long:?} ends past the last page");
        assert_eq!(long.offset().unwrap(), 12288);
    }
}
