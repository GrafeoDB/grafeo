//! The spill budget of an open: one account for everything it spills
//! (sorts' runs, aggregates' partitions, vector indexes' embedding columns),
//! capped by [`Config::spill_limit`](crate::config::Config::spill_limit) and
//! keeping the free space the database needs for its next checkpoint on the
//! spill volume.
//!
//! The account exists only when the open has a spill directory (see
//! `spill_directory`): an in-memory database without a spill path and an
//! encrypted one spill nothing, and their spill settings do nothing. The
//! default cap is half the spill volume's free space when the database
//! opens, measured after the open removed what crashed queries left there
//! (`spill_directory::remove_leftovers`); when the free space cannot be read
//! (or the build has no probe), the memory budget, with one warning.

use std::path::Path;
#[cfg(feature = "grafeo-file")]
use std::path::PathBuf;
use std::sync::Arc;

use grafeo_core::execution::spill::{SpillAccount, SpillCap, SpillRoom};
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;

use crate::config::SpillLimit;

/// The account of an open whose spill directory is `root` (`None`: the open
/// spills nothing, and has no account), capped by `limit`, with `budget` the
/// open's memory budget (the default cap when the free space cannot be
/// read). `database` is the open's file manager, for the room the next
/// checkpoint needs; a `read_only` open never checkpoints, and keeps none.
pub(super) fn open_account(
    limit: SpillLimit,
    root: Option<&Path>,
    budget: usize,
    #[cfg(feature = "grafeo-file")] database: Option<&Arc<GrafeoFileManager>>,
    read_only: bool,
) -> Option<Arc<SpillAccount>> {
    let root = root?;
    #[cfg(feature = "grafeo-file")]
    let room = Some(room(root, database.filter(|_| !read_only)));
    #[cfg(not(feature = "grafeo-file"))]
    let _ = read_only;
    // No probe in this build (C17): the cap alone bounds.
    #[cfg(not(feature = "grafeo-file"))]
    let room: Option<SpillRoom> = None;
    let cap = match limit {
        SpillLimit::Bytes(bytes) => SpillCap::Bytes(bytes),
        SpillLimit::Unlimited => SpillCap::Unlimited,
        // `SpillLimit` is non-exhaustive: a later kind reads as the default.
        _ => default_cap(root, room.as_ref(), budget),
    };
    Some(SpillAccount::new(cap, room))
}

/// Half the free space of `root`'s volume, or the memory budget (with one
/// warning) when it cannot be read.
fn default_cap(root: &Path, room: Option<&SpillRoom>, budget: usize) -> SpillCap {
    let free = room.map(|room| (room.probe)(root));
    match free {
        Some(Ok(free)) => SpillCap::Bytes(free / 2),
        failed => {
            let why = match failed {
                Some(Err(error)) => format!("its free space cannot be read ({error})"),
                _ => "this build cannot read its free space".to_string(),
            };
            grafeo_common::grafeo_warn!(
                "the spill directory {}: {why}, so the spill limit is the memory budget, \
                 {budget} bytes; set it with Config::with_spill_limit",
                root.display()
            );
            SpillCap::Bytes(budget as u64)
        }
    }
}

/// The free-space probe of `root`'s volume and the room the database's next
/// checkpoint needs there.
#[cfg(feature = "grafeo-file")]
fn room(root: &Path, database: Option<&Arc<GrafeoFileManager>>) -> SpillRoom {
    let warned = std::sync::atomic::AtomicBool::new(false);
    let probe = move |path: &Path| {
        let answer = grafeo_storage::file::space::free_space(&existing(path));
        if let Err(error) = &answer
            && !warned.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            grafeo_common::grafeo_warn!(
                "the free space of the spill volume {} cannot be read ({error}): spills are \
                 bounded by the spill limit alone",
                path.display()
            );
        }
        answer
    };
    // Weak: the account outlives no database, and holds no file open.
    let database = database.map(Arc::downgrade);
    let same_volume = database
        .as_ref()
        .and_then(std::sync::Weak::upgrade)
        .is_none_or(|database| same_volume(database.path(), root));
    let needed = move || {
        database
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
            .map_or(0, |database| {
                let headroom = database.checkpoint_headroom();
                if same_volume {
                    // A checkpoint may write the whole file anew beside it.
                    database.file_size().unwrap_or(0).saturating_add(headroom)
                } else {
                    headroom
                }
            })
    };
    SpillRoom {
        volume: root.to_path_buf(),
        probe: Box::new(probe),
        needed: Box::new(needed),
    }
}

/// `path`, or its nearest ancestor that exists: a spill directory is
/// created with its first spill, and the probe reads the volume it will be
/// on.
#[cfg(feature = "grafeo-file")]
fn existing(path: &Path) -> PathBuf {
    let mut at = path;
    while !at.exists() {
        match at.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => at = parent,
            _ => return PathBuf::from("."),
        }
    }
    at.to_path_buf()
}

/// Whether `a` and `b` are on one volume: the device on Unix, the drive or
/// share of the canonical path on Windows. Unsure means one volume, the
/// stricter bound.
#[cfg(feature = "grafeo-file")]
fn same_volume(a: &Path, b: &Path) -> bool {
    let (Ok(a), Ok(b)) = (
        std::fs::canonicalize(existing(a)),
        std::fs::canonicalize(existing(b)),
    ) else {
        return true;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(&a), std::fs::metadata(&b)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev(),
            _ => true,
        }
    }
    #[cfg(not(unix))]
    {
        let prefix = |path: &Path| path.components().next().map(|c| c.as_os_str().to_owned());
        match (prefix(&a), prefix(&b)) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(&b),
            _ => true,
        }
    }
}

#[cfg(all(test, feature = "grafeo-file"))]
mod tests {
    use super::*;

    #[test]
    fn a_missing_directory_is_probed_at_its_nearest_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("lille.grafeo.spill").join("query_3");
        assert_eq!(existing(&missing), dir.path());
    }

    #[test]
    fn two_paths_in_one_directory_are_on_one_volume() {
        let dir = tempfile::tempdir().unwrap();
        assert!(same_volume(
            &dir.path().join("paris.grafeo"),
            &dir.path().join("paris.grafeo.spill")
        ));
    }

    #[test]
    fn an_explicit_limit_is_the_cap_without_a_probe_of_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let account = open_account(
            SpillLimit::Bytes(88),
            Some(dir.path()),
            1 << 20,
            None,
            false,
        )
        .expect("a spill directory has an account");
        assert_eq!(account.cap(), SpillCap::Bytes(88));
        let account = open_account(
            SpillLimit::Unlimited,
            Some(dir.path()),
            1 << 20,
            None,
            false,
        )
        .expect("a spill directory has an account");
        assert_eq!(account.cap(), SpillCap::Unlimited);
        assert!(open_account(SpillLimit::Default, None, 1 << 20, None, false).is_none());
    }

    /// A read-write open keeps the room of the file's next checkpoint free;
    /// a read-only one never checkpoints, and keeps none.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_read_only_open_keeps_no_room_for_a_checkpoint() {
        use grafeo_core::execution::spill::SpillOwner;
        use grafeo_storage::file::space::with_free_space;

        let dir = tempfile::tempdir().unwrap();
        let database =
            Arc::new(GrafeoFileManager::create(dir.path().join("ghent.grafeo"), None).unwrap());
        let needed = database.file_size().unwrap() + database.checkpoint_headroom();
        assert!(needed > 0);
        // Exactly the room the next checkpoint needs is free.
        with_free_space(Some(needed), || {
            let open = |read_only| {
                open_account(
                    SpillLimit::Unlimited,
                    Some(dir.path()),
                    1 << 20,
                    Some(&database),
                    read_only,
                )
                .expect("a spill directory has an account")
            };
            let refusal = open(false)
                .reserve(SpillOwner::Sort, 1, true)
                .expect_err("a read-write open keeps the room");
            assert_eq!(
                refusal.bound(),
                grafeo_common::utils::error::LimitBound::CheckpointRoom
            );
            open(true)
                .reserve(SpillOwner::Sort, 1, true)
                .expect("a read-only open keeps none");
        });
    }
}
