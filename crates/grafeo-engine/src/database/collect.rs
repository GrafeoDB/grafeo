//! What runs after each immediate call and each commit: the collect of the
//! versions no reader needs any more, and the transaction manager's cleanup
//! (#412 S6).
//!
//! A store that collects after every write (the row-group store, whose
//! immediate writes keep what they replace and whose collect visits only
//! the vectors written since the last one) collects right after each
//! immediate call and commit, at the read horizon. `LpgStore`, which the
//! engine runs on until the store switch, collects every `gc_interval`
//! commits, as before: the store type's
//! `COLLECTS_AFTER_EVERY_WRITE` decides, so the switch changes the type and
//! nothing here. The transaction manager drops its finished transactions
//! every `gc_interval` commits with either store; an interval of 0 turns
//! that, and `LpgStore`'s collect, off.

use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_core::graph::lpg::LpgStore;

use crate::transaction::TransactionManager;

/// What runs after the `count`th immediate call or commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AfterWrite {
    /// Whether the written stores collect at the read horizon: always when
    /// the periodic part runs.
    pub collect: bool,
    /// Whether the transaction manager drops its finished transactions:
    /// every `gc_interval`th call or commit.
    pub periodic: bool,
}

impl AfterWrite {
    /// What runs after the `count`th call or commit (from 1), for stores
    /// that collect after every write (`every_write`) or not.
    pub(crate) fn after(count: usize, gc_interval: usize, every_write: bool) -> Self {
        let periodic = gc_interval > 0 && count.is_multiple_of(gc_interval);
        Self {
            collect: every_write || periodic,
            periodic,
        }
    }

    /// Counts an immediate call or a commit in `commits` and returns what
    /// runs after it, on the engine's store type.
    pub(crate) fn next(commits: &AtomicUsize, gc_interval: usize) -> Self {
        let count = commits.fetch_add(1, Ordering::Relaxed) + 1;
        Self::after(count, gc_interval, LpgStore::COLLECTS_AFTER_EVERY_WRITE)
    }

    /// Runs it after a call or a commit that wrote `stores`: the root store
    /// among them, so that its collect also covers what every named graph
    /// lists once the root is a row-group store (its `gc_versions` does).
    /// A store named twice collects once.
    pub(crate) fn run(self, manager: &TransactionManager, stores: &[&LpgStore]) {
        if self.collect {
            let horizon = manager.read_horizon();
            for (index, store) in stores.iter().enumerate() {
                if !stores[..index]
                    .iter()
                    .any(|earlier| std::ptr::eq(*earlier, *store))
                {
                    store.gc_versions(horizon);
                }
            }
        }
        if self.periodic {
            manager.gc();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AfterWrite;

    /// A store that collects after every write collects after each call and
    /// commit, `gc_interval` aside; one that does not collects every
    /// `gc_interval`th one; the transaction manager's cleanup runs every
    /// `gc_interval`th one for both, and never with an interval of 0.
    #[test]
    fn every_write_stores_collect_each_time_and_the_cleanup_stays_periodic() {
        let every_write: Vec<AfterWrite> = (1..=4)
            .map(|count| AfterWrite::after(count, 2, true))
            .collect();
        assert!(every_write.iter().all(|after| after.collect));
        assert_eq!(
            every_write
                .iter()
                .map(|after| after.periodic)
                .collect::<Vec<_>>(),
            [false, true, false, true]
        );
        let periodic: Vec<AfterWrite> = (1..=4)
            .map(|count| AfterWrite::after(count, 2, false))
            .collect();
        assert_eq!(
            periodic
                .iter()
                .map(|after| (after.collect, after.periodic))
                .collect::<Vec<_>>(),
            [(false, false), (true, true), (false, false), (true, true)]
        );
        assert_eq!(
            AfterWrite::after(100, 0, false),
            AfterWrite {
                collect: false,
                periodic: false
            },
            "an interval of 0 turns it off"
        );
        assert_eq!(
            AfterWrite::after(100, 0, true),
            AfterWrite {
                collect: true,
                periodic: false
            },
            "but not a collect after every write"
        );
    }
}
