//! The integration tests of `grafeo-cli`, built as one test binary: each file
//! in this directory is a module here, so the crate is linked into one binary, not
//! one per file.
//!
//! ```bash
//! cargo test -p grafeo-cli --all-features --test main                  # all of them
//! cargo test -p grafeo-cli --all-features --test main some_file::      # one file
//! ```
//!
//! A new test file needs its `mod` line here: Cargo does not look for test files in
//! this crate. A file that needs a process of its own, or a build with few features,
//! is a `[[test]]` entry in `Cargo.toml` instead, with a comment saying why.

/// The `grafeo` binary with `--no-color` and a temp directory of its own (`TMPDIR`, `TEMP`,
/// `TMP`), so nothing other processes leave in the shared one reaches its output. `temp` must
/// live until the command has run.
fn grafeo_command(temp: &std::path::Path) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_grafeo"));
    command
        .env("TMPDIR", temp)
        .env("TEMP", temp)
        .env("TMP", temp)
        .arg("--no-color");
    command
}

mod cli_runs;
mod cli_warnings;
mod commands;
mod default_features_open;
