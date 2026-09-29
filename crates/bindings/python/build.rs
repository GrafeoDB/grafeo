//! Records the git commit this module is built from, for `grafeo.build_info()`.
//!
//! Sets `GRAFEO_BUILD_COMMIT` (the full commit hash) and `GRAFEO_BUILD_DIRTY`
//! (`true` when tracked files had uncommitted changes) when the crate is built
//! from a git checkout of Grafeo. Both stay unset when git is not available or
//! the sources are not tracked (a build from an sdist), and `build_info()`
//! then reports `None`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Sources compiled into the module, relative to this crate: a change to any
/// of them can change the dirty flag, so the script reruns when one changes.
const SOURCES: &[&str] = &[
    "src",
    "Cargo.toml",
    "pyproject.toml",
    "../common/src",
    "../common/Cargo.toml",
    "../../grafeo-common",
    "../../grafeo-core",
    "../../grafeo-storage",
    "../../grafeo-adapters",
    "../../grafeo-engine",
    "../../../Cargo.toml",
    "../../../Cargo.lock",
];

/// Runs git in `dir` and returns its trimmed output, or `None` when git is
/// missing or the command fails.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        // Never take the index lock: `git status` would otherwise refresh the
        // index, whose change would rerun this script on every build.
        .arg("--no-optional-locks")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8(output.stdout)
            .ok()?
            .trim_end()
            .to_string(),
    )
}

fn main() {
    let crate_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR")
            .expect("cargo sets CARGO_MANIFEST_DIR for build scripts"),
    );
    println!("cargo:rerun-if-changed=build.rs");

    // Only a checkout that tracks this crate describes these sources: an sdist
    // unpacked inside some other repository must not report that commit.
    if git(&crate_dir, &["ls-files", "--error-unmatch", "Cargo.toml"]).is_none() {
        return;
    }
    let (Some(commit), Some(status)) = (
        git(&crate_dir, &["rev-parse", "HEAD"]),
        git(
            &crate_dir,
            &["status", "--porcelain", "--untracked-files=no"],
        ),
    ) else {
        return;
    };
    println!("cargo:rustc-env=GRAFEO_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=GRAFEO_BUILD_DIRTY={}", !status.is_empty());

    // Rerun when HEAD moves (checkout), the current branch moves (commit) or
    // the index changes (add, reset). `--git-path` resolves each file inside
    // linked worktrees too.
    let mut git_files = vec!["HEAD".to_string(), "index".to_string()];
    if let Some(branch) = git(&crate_dir, &["symbolic-ref", "-q", "HEAD"]) {
        git_files.push(branch);
        git_files.push("packed-refs".to_string());
    }
    for name in git_files {
        if let Some(path) = git(&crate_dir, &["rev-parse", "--git-path", &name])
            && crate_dir.join(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
}
