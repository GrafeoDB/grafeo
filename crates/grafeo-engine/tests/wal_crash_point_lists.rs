//! Every crash point of the WAL is named by a test, and the lists that pin a flow's points in
//! order agree with the code (0.6.2, the guard added with #725's new point). Three tests list the
//! points of a flow in the order it reaches them: writing (`WRITE_POINTS` in grafeo-storage's
//! `wal_v2_crash.rs`), a checkpoint (`CHECKPOINT_POINTS` in `crash_injection_single_file.rs`) and
//! an open after a migration (`OPEN_WAL_POINTS` in `migration.rs`). Each makes a segment, so each
//! holds the rotation's run of points; a point added to that run, or anywhere in the WAL, fails
//! here until the lists and a test name it.
//!
//! The run is the points named `wal:after_segment_*` and `wal:before_dir_sync`, in the order a
//! real checkpoint reaches them (a panic at each point in turn), not the order the source holds
//! them in.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main wal_crash_point_lists::
//! ```

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The `crates/` directory.
fn crates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Every `.rs` file under `dir`, recursively.
fn sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The `wal:` points of `maybe_crash("...")` calls in `text`, in the order they appear.
fn crash_calls(text: &str, prefix: &str) -> Vec<String> {
    text.match_indices(&format!("maybe_crash(\"{prefix}"))
        .filter_map(|(at, call)| {
            let start = at + call.len() - prefix.len();
            let end = text[start..].find('"')? + start;
            Some(text[start..end].to_string())
        })
        .collect()
}

/// The quoted crash points of `text` (`"wal:..."`, `"flush:..."` and `"checkpoint:..."`), in the
/// order they appear.
fn quoted_points(text: &str) -> Vec<String> {
    let mut found: Vec<(usize, String)> = ["wal:", "flush:", "checkpoint:"]
        .into_iter()
        .flat_map(|prefix| {
            text.match_indices(&format!("\"{prefix}"))
                .filter_map(move |(at, _)| {
                    let start = at + 1;
                    let end = text[start..].find('"')? + start;
                    let point = &text[start..end];
                    let name = &point[prefix.len()..];
                    (!name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
                        .then(|| (at, point.to_string()))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    found.sort_unstable_by_key(|(at, _)| *at);
    found.into_iter().map(|(_, point)| point).collect()
}

/// The quoted points of the `const NAME` array in `file`.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
fn listed(file: &Path, name: &str) -> Vec<String> {
    let text = read(file);
    let start = text
        .find(&format!("const {name}"))
        .unwrap_or_else(|| panic!("{}: no `const {name}`", file.display()));
    let end = text[start..].find("];").unwrap() + start;
    quoted_points(&text[start..end])
}

/// A point of the rotation's run, by its name.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
fn in_rotation(point: &str) -> bool {
    point.starts_with("wal:after_segment_") || point == "wal:before_dir_sync"
}

/// Points no test names, with the reason. Empty: every point is tested.
const UNTESTED: &[(&str, &str)] = &[];

#[test]
fn every_wal_and_flush_crash_point_in_the_code_is_named_by_a_test() {
    let mut code = BTreeSet::new();
    for file in sources(&crates().join("grafeo-storage/src")) {
        code.extend(crash_calls(&read(&file), "wal:"));
    }
    for file in sources(&crates().join("grafeo-engine/src")) {
        code.extend(crash_calls(&read(&file), "flush:"));
    }
    assert!(code.len() > 10, "the WAL's points were found: {code:?}");
    let mut named = BTreeSet::new();
    for dir in std::fs::read_dir(crates()).unwrap() {
        for file in sources(&dir.unwrap().path().join("tests")) {
            named.extend(quoted_points(&read(&file)));
        }
    }
    let missing: Vec<&String> = code
        .iter()
        .filter(|point| !named.contains(*point))
        .filter(|point| !UNTESTED.iter().any(|(untested, _)| untested == point))
        .collect();
    assert!(
        missing.is_empty(),
        "crash points no test names: {missing:?}"
    );
}

/// Every crash point a checkpoint reaches, in order: the checkpoint's points walked in turn, a
/// panic at each, on a database built outside the closure.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
fn checkpoint_run() -> Vec<String> {
    use grafeo_common::testing::crash::{disable_crash, enable_crash_at};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut run = Vec::new();
    for count in 1..=64 {
        let dir = tempfile::tempdir().unwrap();
        let db = grafeo_engine::GrafeoDB::with_config(grafeo_engine::Config::persistent(
            dir.path().join("run.grafeo"),
        ))
        .unwrap();
        db.session()
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        enable_crash_at(count);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = db.wal_checkpoint();
        }));
        disable_crash();
        let Err(payload) = outcome else {
            return run;
        };
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        run.push(message.rsplit(' ').next().unwrap_or_default().to_string());
    }
    panic!("a checkpoint ran past 64 crash points")
}

#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[test]
fn the_flow_lists_hold_the_rotation_run_a_checkpoint_reaches() {
    let run: Vec<String> = checkpoint_run()
        .into_iter()
        .filter(|point| in_rotation(point))
        .collect();
    assert!(run.len() >= 3, "the rotation's points were found: {run:?}");
    for (file, name) in [
        ("grafeo-storage/tests/wal_v2_crash.rs", "WRITE_POINTS"),
        (
            "grafeo-engine/tests/crash_injection_single_file.rs",
            "CHECKPOINT_POINTS",
        ),
        ("grafeo-engine/tests/migration.rs", "OPEN_WAL_POINTS"),
    ] {
        let points = listed(&crates().join(file), name);
        let in_list: Vec<String> = points
            .into_iter()
            .filter(|point| in_rotation(point))
            .collect();
        assert_eq!(
            in_list, run,
            "{file} `{name}`: the rotation's points, in the code's order"
        );
    }
}

/// `CHECKPOINT_POINTS` is every crash point a checkpoint reaches, in its order: a point added to
/// the checkpoint fails here until the list names it.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[test]
fn the_checkpoint_list_is_every_point_a_checkpoint_reaches() {
    let listed = listed(
        &crates().join("grafeo-engine/tests/crash_injection_single_file.rs"),
        "CHECKPOINT_POINTS",
    );
    assert_eq!(listed, checkpoint_run());
}
