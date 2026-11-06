//! Query warnings in the CLI (0.6.2): each warning of a statement on standard error as
//! `warning <code>: <message>`, the rows on standard output unchanged, in every output format, so
//! a script that reads the output reads what it read before. `--quiet` hides progress and status
//! messages, and a warning is neither, so it still prints.
//!
//! ```bash
//! cargo test -p grafeo-cli --all-features --test main cli_warnings::
//! ```

use std::process::{Command, Output};

/// Runs the `grafeo` binary with `args`, after checking that it succeeded.
fn grafeo(args: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_grafeo"))
        .arg("--no-color")
        .args(args)
        .output()
        .expect("run grafeo");
    assert!(
        output.status.success(),
        "grafeo {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn database() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("warnings.grafeo");
    let path = path.to_str().unwrap().to_string();
    grafeo(&["init", &path]);
    (dir, path)
}

#[test]
fn a_warning_goes_to_standard_error_with_its_code() {
    let (_dir, path) = database();
    let out = grafeo(&[
        "query",
        &path,
        "RETURN ZONED DATETIME '2024-01-01T10:00:00+01:00' IS NOT NULL AS v",
    ]);
    let lines: Vec<String> = stderr(&out)
        .lines()
        .filter(|l| l.starts_with("warning "))
        .map(str::to_string)
        .collect();
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("warning 01X01: "), "{}", lines[0]);
    assert!(lines[0].contains("0.7.0"), "{}", lines[0]);
}

#[test]
fn two_warnings_come_in_code_order() {
    let (_dir, path) = database();
    let out = grafeo(&[
        "query",
        &path,
        "RETURN size([1] + [2]) AS a, ZONED DATETIME '2024-01-01T10:00:00+01:00' IS NOT NULL AS b",
    ]);
    let codes: Vec<String> = stderr(&out)
        .lines()
        .filter_map(|l| l.strip_prefix("warning "))
        .map(|rest| rest[..5].to_string())
        .collect();
    assert_eq!(codes, ["01X01", "01X02"]);
}

/// JSON output stays the rows array it was: the warning is on standard error, not in the rows.
#[test]
fn json_output_keeps_its_shape() {
    let (_dir, path) = database();
    let out = grafeo(&[
        "--format",
        "json",
        "query",
        &path,
        "RETURN size([1] + [2]) AS v",
    ]);
    let rows: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the rows parse as JSON");
    assert_eq!(rows, serde_json::json!([{"v": 2}]));
    assert!(stderr(&out).contains("warning 01X02: "), "{}", stderr(&out));
}

#[test]
fn no_warning_writes_nothing_to_standard_error() {
    let (_dir, path) = database();
    let out = grafeo(&["query", &path, "RETURN size([1] || [2]) AS v"]);
    assert!(
        !stderr(&out).lines().any(|l| l.starts_with("warning ")),
        "{}",
        stderr(&out)
    );
}

/// `--quiet` suppresses progress and status messages, not warnings: the rows on standard output,
/// exactly one warning line on standard error, and the command still succeeds.
#[test]
fn quiet_keeps_the_warning() {
    let (_dir, path) = database();
    let out = grafeo(&[
        "--quiet",
        "--format",
        "json",
        "query",
        &path,
        "RETURN size([1] + [2]) AS v",
    ]);
    assert!(out.status.success());
    let rows: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the rows parse as JSON");
    assert_eq!(rows, serde_json::json!([{"v": 2}]));
    let warnings: Vec<String> = stderr(&out)
        .lines()
        .filter(|l| l.starts_with("warning "))
        .map(str::to_string)
        .collect();
    assert_eq!(warnings.len(), 1, "{}", stderr(&out));
    assert!(
        warnings[0].starts_with("warning 01X02: "),
        "{}",
        warnings[0]
    );
}
