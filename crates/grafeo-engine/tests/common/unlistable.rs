//! A directory nobody may list, for tests of an open that cannot read a WAL
//! directory. Include it with `#[path = "common/unlistable.rs"] mod unlistable;`.

use std::path::{Path, PathBuf};

/// Denies listing a directory until dropped, which allows it again, so the
/// temporary directory holding it can be removed.
pub struct Unlistable {
    dir: PathBuf,
}

impl Unlistable {
    /// Denies everyone the listing of `dir`: the "list folder" right of
    /// Everyone is denied with `icacls` on Windows, the mode becomes `0300` on
    /// Unix. Checks that listing then fails; when it still works (as root,
    /// permissions do not apply), says why on stderr and returns `None`, so
    /// the caller skips its test.
    ///
    /// # Panics
    ///
    /// Panics if the permissions cannot be changed.
    pub fn deny(dir: &Path) -> Option<Self> {
        if let Err(error) = set_listing(dir, false) {
            panic!("{error}");
        }
        let guard = Self {
            dir: dir.to_path_buf(),
        };
        match std::fs::read_dir(dir) {
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => Some(guard),
            outcome => {
                eprintln!(
                    "skipped: {} can still be listed after denying it ({:?}); permissions do \
                     not apply to this user",
                    dir.display(),
                    outcome.map(|_| "listed")
                );
                None
            }
        }
    }
}

impl Drop for Unlistable {
    /// Allows listing again. A failure (the directory moved away, as an open
    /// under test may do) is reported, not raised: a panic in a drop during a
    /// test failure would abort the test binary.
    fn drop(&mut self) {
        if let Err(error) = set_listing(&self.dir, true) {
            eprintln!("cannot allow listing again: {error}");
        }
    }
}

/// Denies (`allowed` false) or allows again the listing of `dir` to everyone.
#[cfg(windows)]
fn set_listing(dir: &Path, allowed: bool) -> Result<(), String> {
    // Everyone (S-1-1-0): an explicit deny wins over every allow.
    let mut command = std::process::Command::new("icacls");
    command.arg(dir);
    if allowed {
        command.args(["/remove:d", "*S-1-1-0"]);
    } else {
        command.args(["/deny", "*S-1-1-0:(RD)"]);
    }
    let output = command
        .output()
        .map_err(|error| format!("cannot run icacls: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "icacls {} failed: {}{}",
        dir.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// Denies (`allowed` false) or allows again the listing of `dir` to everyone.
#[cfg(unix)]
fn set_listing(dir: &Path, allowed: bool) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let mode = if allowed { 0o700 } else { 0o300 };
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode))
        .map_err(|error| format!("chmod {mode:o} {}: {error}", dir.display()))
}
