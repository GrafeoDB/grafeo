//! Pause points for testing races between processes.
//!
//! When the `testing-crash-injection` feature is enabled, [`maybe_pause`]
//! stops the process at a named point when the environment asks for it:
//! [`POINT_VAR`] names the point and [`FLAG_VAR`] a flag path. At that point
//! the process creates `<flag>.paused` and waits until `<flag>.resume` exists
//! (polling, for up to a minute), so a test that runs it as a child process
//! can let another process act in between, deterministically.
//!
//! When the feature is **disabled**, [`maybe_pause`] compiles to a no-op.

/// The environment variable naming the point to pause at.
pub const POINT_VAR: &str = "GRAFEO_TEST_PAUSE_POINT";

/// The environment variable naming the flag path: the paused process creates
/// `<flag>.paused` and waits for `<flag>.resume`.
pub const FLAG_VAR: &str = "GRAFEO_TEST_PAUSE_FLAG";

#[cfg(feature = "testing-crash-injection")]
mod inner {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// How long a paused process waits for its resume file.
    const RESUME_WAIT: Duration = Duration::from_secs(60);

    /// Pauses at `point` when [`super::POINT_VAR`] names it and
    /// [`super::FLAG_VAR`] names a flag path (see the module docs).
    ///
    /// # Panics
    ///
    /// Panics if the paused process cannot create `<flag>.paused`, or if
    /// `<flag>.resume` does not appear within a minute.
    pub fn maybe_pause(point: &str) {
        let (Ok(wanted), Some(flag)) = (
            std::env::var(super::POINT_VAR),
            std::env::var_os(super::FLAG_VAR),
        ) else {
            return;
        };
        pause_at(point, &wanted, Path::new(&flag));
    }

    /// Pauses at `point` if it is `wanted`: creates `<flag>.paused`, then
    /// waits until `<flag>.resume` exists.
    pub(super) fn pause_at(point: &str, wanted: &str, flag: &Path) {
        if point != wanted {
            return;
        }
        std::fs::write(with_suffix(flag, ".paused"), point)
            .unwrap_or_else(|error| panic!("pause at {point}: cannot create the flag: {error}"));
        let resume = with_suffix(flag, ".resume");
        let deadline = Instant::now() + RESUME_WAIT;
        while !resume.exists() {
            assert!(
                Instant::now() < deadline,
                "pause at {point}: {} did not appear within a minute",
                resume.display()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        PathBuf::from(name)
    }
}

#[cfg(not(feature = "testing-crash-injection"))]
mod inner {
    /// No-op when crash injection is disabled.
    #[inline(always)]
    pub fn maybe_pause(_point: &str) {}
}

pub use inner::maybe_pause;

#[cfg(all(test, feature = "testing-crash-injection"))]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use super::inner::pause_at;

    /// A flag path of this test run, with no flag files next to it.
    fn flag(name: &str) -> PathBuf {
        let flag = std::env::temp_dir().join(format!("grafeo-pause-{}-{name}", std::process::id()));
        for suffix in [".paused", ".resume"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", flag.display()));
        }
        flag
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri isolation forbids file system access")]
    fn another_point_does_not_pause() {
        let flag = flag("other");
        pause_at("open:after_check", "migrate:after_old", &flag);
        assert!(
            !PathBuf::from(format!("{}.paused", flag.display())).exists(),
            "only the named point pauses"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri isolation forbids file system access")]
    fn the_named_point_waits_for_its_resume_file() {
        let flag = flag("named");
        let paused = PathBuf::from(format!("{}.paused", flag.display()));
        let resume = PathBuf::from(format!("{}.resume", flag.display()));
        let (paused_file, resume_file) = (paused.clone(), resume.clone());
        let resumer = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            while !paused_file.exists() {
                assert!(Instant::now() < deadline, "the pause never started");
                std::thread::sleep(Duration::from_millis(5));
            }
            // Still paused a moment later: the pause waits for the resume file.
            std::thread::sleep(Duration::from_millis(50));
            let resumed_at = Instant::now();
            std::fs::write(&resume_file, b"go").unwrap();
            resumed_at
        });
        pause_at("migrate:after_old", "migrate:after_old", &flag);
        let returned_at = Instant::now();
        let resumed_at = resumer.join().unwrap();
        assert!(
            returned_at >= resumed_at,
            "the pause ends only once the resume file exists"
        );
        std::fs::remove_file(paused).unwrap();
        std::fs::remove_file(resume).unwrap();
    }
}
