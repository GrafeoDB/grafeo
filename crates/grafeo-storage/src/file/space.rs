//! The free-space check before a checkpoint (#726).
//!
//! A checkpoint writes a new image next to the active one (copy-on-write),
//! so in the worst case it needs as much room as the file takes, and the
//! WAL must still be able to grow by a segment afterwards. Before it writes
//! anything, [`GrafeoFileManager::write_checkpoint`](super::GrafeoFileManager::write_checkpoint)
//! asks [`free_space`] how much room the disk has and fails with
//! [`InsufficientSpace`] when that is less. The check is a guard, not the
//! guarantee: a disk that fills after it still fails the write, and the
//! file keeps its active image.

use std::path::{Path, PathBuf};

use grafeo_common::utils::error::Error;

/// A checkpoint did not start: the disk holding the database file has less
/// free space than the checkpoint may need. Nothing was written, and the
/// database stays usable; a later checkpoint with room succeeds.
///
/// It reaches callers as an [`Error::Io`] of kind
/// [`StorageFull`](std::io::ErrorKind::StorageFull) that carries this value:
///
/// ```
/// use grafeo_common::utils::error::Error;
/// use grafeo_storage::file::InsufficientSpace;
///
/// let error = Error::from(InsufficientSpace {
///     path: "people.grafeo".into(),
///     needed: 1988,
///     headroom: 1024,
///     available: 319,
/// });
/// let Error::Io(io) = &error else { panic!("an I/O error") };
/// assert_eq!(io.kind(), std::io::ErrorKind::StorageFull);
/// let space = io.get_ref().and_then(|inner| inner.downcast_ref::<InsufficientSpace>());
/// assert_eq!(space.map(|space| space.available), Some(319));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsufficientSpace {
    /// The database file.
    pub path: PathBuf,
    /// The free space the checkpoint may need, in bytes: the file's size
    /// plus the headroom.
    pub needed: u64,
    /// The room, in bytes, a checkpoint leaves for the WAL beyond the file's
    /// size: one WAL segment of the configured size, or 0 without a WAL.
    pub headroom: u64,
    /// The free space the disk reported, in bytes.
    pub available: u64,
}

impl std::fmt::Display for InsufficientSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "not enough free disk space for a checkpoint of {}: it may need {} bytes (the size \
             of the database file, which a copy-on-write checkpoint may write again",
            self.path.display(),
            self.needed
        )?;
        if self.headroom > 0 {
            write!(
                f,
                ", plus {} bytes of room for one WAL segment",
                self.headroom
            )?;
        }
        write!(f, "), and {} bytes are free", self.available)
    }
}

impl std::error::Error for InsufficientSpace {}

impl From<InsufficientSpace> for Error {
    fn from(space: InsufficientSpace) -> Self {
        Error::Io(std::io::Error::new(std::io::ErrorKind::StorageFull, space))
    }
}

/// The free space, in bytes, of the disk that holds `path`, as far as this
/// process may use it.
///
/// With the `testing-crash-injection` feature, a test can replace the answer
/// on its thread with [`with_free_space`], or for the paths under a
/// directory on every thread with [`with_free_space_under`].
///
/// # Errors
///
/// The error of the probe (an unsupported filesystem, a missing path).
pub fn free_space(path: &Path) -> std::io::Result<u64> {
    #[cfg(feature = "testing-crash-injection")]
    match fake::FREE_SPACE.with(std::cell::Cell::get) {
        fake::Answer::Disk => {}
        fake::Answer::Bytes(bytes) => return Ok(bytes),
        fake::Answer::Fails => {
            return Err(std::io::Error::other("the free-space probe failed (faked)"));
        }
    }
    #[cfg(feature = "testing-crash-injection")]
    match fake::under(path) {
        fake::Answer::Disk => {}
        fake::Answer::Bytes(bytes) => return Ok(bytes),
        fake::Answer::Fails => {
            return Err(std::io::Error::other("the free-space probe failed (faked)"));
        }
    }
    fs2::available_space(path)
}

/// Runs `f` while [`free_space`] reports `bytes` for every path under `dir`,
/// on every thread (a background checkpoint's too), or fails when `bytes`
/// is `None`; the real probe comes back afterwards, also when `f` panics.
/// An answer [`with_free_space`] gives on a thread comes first.
#[cfg(feature = "testing-crash-injection")]
pub fn with_free_space_under<T>(dir: &Path, bytes: Option<u64>, f: impl FnOnce() -> T) -> T {
    /// Takes the answer back when dropped.
    struct Restore(u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            fake::UNDER.lock().retain(|(id, _, _)| *id != self.0);
        }
    }
    let id = fake::NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let answer = bytes.map_or(fake::Answer::Fails, fake::Answer::Bytes);
    fake::UNDER.lock().push((id, dir.to_path_buf(), answer));
    let _restore = Restore(id);
    f()
}

/// Runs `f` while [`free_space`] reports `bytes` on this thread, or fails
/// when `bytes` is `None`; the real probe comes back afterwards, also when
/// `f` panics.
#[cfg(feature = "testing-crash-injection")]
pub fn with_free_space<T>(bytes: Option<u64>, f: impl FnOnce() -> T) -> T {
    /// Puts back the answer before [`with_free_space`] when dropped.
    struct Restore(fake::Answer);
    impl Drop for Restore {
        fn drop(&mut self) {
            fake::FREE_SPACE.with(|fake| fake.set(self.0));
        }
    }
    let answer = bytes.map_or(fake::Answer::Fails, fake::Answer::Bytes);
    let _restore = Restore(fake::FREE_SPACE.with(|fake| fake.replace(answer)));
    f()
}

#[cfg(feature = "testing-crash-injection")]
mod fake {
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;

    /// The answers [`with_free_space_under`](super::with_free_space_under)
    /// gives on every thread: an id, the directory and the answer; the
    /// newest first matches.
    pub(super) static UNDER: parking_lot::Mutex<Vec<(u64, PathBuf, Answer)>> =
        parking_lot::Mutex::new(Vec::new());

    /// The id of the next answer in [`UNDER`].
    pub(super) static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    /// The answer for `path` from [`UNDER`], or the disk's when none holds
    /// it.
    pub(super) fn under(path: &Path) -> Answer {
        UNDER
            .lock()
            .iter()
            .rev()
            .find(|(_, dir, _)| path.starts_with(dir))
            .map_or(Answer::Disk, |(_, _, answer)| *answer)
    }

    /// What [`free_space`](super::free_space) answers on a thread.
    #[derive(Debug, Clone, Copy)]
    pub(super) enum Answer {
        /// The disk's free space.
        Disk,
        /// This many bytes.
        Bytes(u64),
        /// A failed probe.
        Fails,
    }

    thread_local! {
        /// The answer [`free_space`](super::free_space) gives on this thread.
        pub(super) static FREE_SPACE: Cell<Answer> = const { Cell::new(Answer::Disk) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_real_probe_reports_room_on_the_temporary_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(free_space(dir.path()).unwrap() > 0);
    }

    #[test]
    fn the_message_names_both_numbers_and_the_wal_headroom() {
        let message = InsufficientSpace {
            path: "people.grafeo".into(),
            needed: 1988,
            headroom: 1024,
            available: 319,
        }
        .to_string();
        assert!(message.contains("1988 bytes"), "{message}");
        assert!(message.contains("319 bytes are free"), "{message}");
        assert!(
            message.contains("1024 bytes of room for one WAL segment"),
            "{message}"
        );
    }

    /// Without a WAL there is no headroom, and the message names none.
    #[test]
    fn without_headroom_the_message_names_no_wal_segment() {
        let message = InsufficientSpace {
            path: "people.grafeo".into(),
            needed: 1988,
            headroom: 0,
            available: 319,
        }
        .to_string();
        assert!(!message.contains("WAL"), "{message}");
        assert!(
            message.contains("may write again), and 319 bytes"),
            "{message}"
        );
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn an_answer_under_a_directory_holds_on_every_thread_and_is_taken_back() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let file = dir.path().join("people.grafeo");
        with_free_space_under(dir.path(), Some(88), || {
            assert_eq!(free_space(&file).unwrap(), 88);
            let on_a_thread = std::thread::spawn({
                let file = file.clone();
                move || free_space(&file).unwrap()
            });
            assert_eq!(on_a_thread.join().unwrap(), 88);
            assert_ne!(free_space(other.path()).unwrap(), 88, "another directory");
            with_free_space(Some(19), || assert_eq!(free_space(&file).unwrap(), 19));
            with_free_space_under(dir.path(), None, || assert!(free_space(&file).is_err()));
            assert_eq!(free_space(&file).unwrap(), 88);
        });
        assert_ne!(free_space(&file).unwrap_or(0), 88);
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_faked_probe_answers_on_this_thread_only_and_is_put_back() {
        let dir = tempfile::tempdir().unwrap();
        with_free_space(Some(88), || {
            assert_eq!(free_space(dir.path()).unwrap(), 88);
            let other = std::thread::spawn({
                let path = dir.path().to_path_buf();
                move || free_space(&path).unwrap()
            });
            assert_ne!(other.join().unwrap(), 88);
            with_free_space(None, || assert!(free_space(dir.path()).is_err()));
            assert_eq!(free_space(dir.path()).unwrap(), 88);
        });
        assert_ne!(free_space(dir.path()).unwrap(), 88);
    }
}
