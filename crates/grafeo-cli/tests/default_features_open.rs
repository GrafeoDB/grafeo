//! #763: the CLI's default build opens every file the `grafeo` crate's default profile writes,
//! and an open it must refuse names the feature to install.
//!
//! The fixtures (`fixtures/facade/`, see its README) are written by the `grafeo` crate: one with
//! its default profile (documents with a vector index and a text index), one with its `rdf`
//! profile (RDF triples, which the default profile does not write). The tests run the `grafeo`
//! binary on a copy of each, as a user does.
//!
//! The case that matters runs with the CLI's default features (`--all-features` opens both files
//! and skips the refusal):
//!
//! ```bash
//! cargo test -p grafeo-cli --test main default_features_open::
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A copy of fixture `name` in `dir`: an open may checkpoint, which would change the fixture.
fn copy_of(name: &str, dir: &Path) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/facade")
        .join(name);
    let to = dir.join(name);
    std::fs::copy(&from, &to).unwrap_or_else(|error| panic!("{}: {error}", from.display()));
    to
}

fn grafeo(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_grafeo"))
        .arg("--no-color")
        .args(args)
        .output()
        .expect("run grafeo")
}

/// The rows `grafeo --format json query` printed for `query`, after checking that it succeeded.
fn rows(db: &Path, query: &str) -> Vec<serde_json::Value> {
    let output = grafeo(&["--format", "json", "query", db.to_str().unwrap(), query]);
    assert!(
        output.status.success(),
        "`{query}` on {}: {}",
        db.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8(output.stdout).unwrap();
    match serde_json::from_str(&out) {
        Ok(serde_json::Value::Array(rows)) => rows,
        other => panic!("`{query}` printed {out:?}: {other:?}"),
    }
}

fn titles(rows: &[serde_json::Value]) -> Vec<String> {
    rows.iter()
        .map(|row| row["t"].as_str().expect("a title").to_string())
        .collect()
}

/// A file the default profile wrote with a vector index and a text index opens, its documents
/// read back, both indexes answer, and after the CLI's close the file opens again with them.
#[test]
fn the_default_build_opens_a_file_of_the_facades_default_profile() {
    let dir = tempfile::tempdir().unwrap();
    let db = copy_of("default-profile.grafeo", dir.path());
    let all = "MATCH (d:Document) RETURN d.title AS t ORDER BY t";
    let text =
        "MATCH (d:Document) WHERE text_match(d.body, 'canals') RETURN d.title AS t ORDER BY t";
    let vector = "CALL grafeo.search.vector('Document', 'embedding', [1.0, 0.0, 0.0], 1) \
                  YIELD node_id MATCH (d:Document) WHERE id(d) = node_id RETURN d.title AS t";
    for round in ["first open", "after the CLI closed it"] {
        assert_eq!(
            titles(&rows(&db, all)),
            ["Amsterdam", "Rotterdam", "Utrecht"],
            "{round}"
        );
        assert_eq!(
            titles(&rows(&db, text)),
            ["Amsterdam", "Utrecht"],
            "{round}: the text index"
        );
        assert_eq!(
            titles(&rows(&db, vector)),
            ["Amsterdam"],
            "{round}: the vector index"
        );
    }
}

/// A file holding data this build cannot read (RDF triples, without `triple-store`) is refused,
/// and the error names the feature and how to build the CLI with it.
#[cfg(not(feature = "triple-store"))]
#[test]
fn a_refused_open_names_the_feature_to_install() {
    let dir = tempfile::tempdir().unwrap();
    let db = copy_of("rdf-profile.grafeo", dir.path());
    let before = std::fs::read(&db).unwrap();
    let output = grafeo(&["info", db.to_str().unwrap()]);
    assert!(!output.status.success(), "the open is refused");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("triple-store"),
        "names the feature: {stderr}"
    );
    assert!(
        stderr.contains("--features"),
        "says how to build the CLI with it: {stderr}"
    );
    assert_eq!(std::fs::read(&db).unwrap(), before, "the file is untouched");
}
