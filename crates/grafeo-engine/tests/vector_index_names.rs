//! A vector index keeps the name `CREATE VECTOR INDEX` (or `CREATE INDEX ...
//! USING VECTOR`) gave it: `SHOW INDEXES` lists it as Vector, `DROP INDEX`
//! drops it, and both hold across a close, a checkpoint and the replay of
//! the log after a crash. A name already in use is refused. The name was
//! stored nowhere, so after a reopen `SHOW INDEXES` did not list the index
//! and `DROP INDEX` of its name failed; `USING VECTOR` recorded it as Hash.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test vector_index_names
//! ```

#![cfg(all(feature = "vector-index", feature = "wal", feature = "gql"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use grafeo_common::testing::child_process;
use grafeo_engine::GrafeoDB;

const STEP_VAR: &str = "GRAFEO_VECTOR_NAMES_STEP";
const PATH_VAR: &str = "GRAFEO_VECTOR_NAMES_PATH";

const DOCS: &str = "INSERT (:Doc {title: 'Canals', emb: vector([1.0, 0.0])}), \
                    (:Doc {title: 'Bikes', emb: vector([0.0, 1.0])}), \
                    (:Doc {title: 'Graphs', emb: vector([0.8, 0.6])})";
const CREATE: &str = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 2 METRIC 'cosine'";

/// `SHOW INDEXES` as `name type label property` lines.
fn indexes(db: &GrafeoDB) -> Vec<String> {
    db.execute("SHOW INDEXES")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    grafeo_common::types::Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Whether a vector search on `Doc.emb` runs (an index serves it).
fn searches(db: &GrafeoDB) -> bool {
    db.vector_search("Doc", "emb", &[1.0, 0.0], 1, None, None)
        .is_ok()
}

fn database(path: &Path) -> GrafeoDB {
    GrafeoDB::open(path).unwrap()
}

/// Runs this binary's child test for `step` at `path`.
fn child(step: &str, path: &Path) {
    let status = child_process::run(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "vector_names_child", "--nocapture"])
            .env(STEP_VAR, step)
            .env(PATH_VAR, path),
    )
    .unwrap();
    assert_eq!(status.code(), Some(0), "child step {step}");
}

/// Child-process entry: each step ends the process without closing the
/// database, as a crash does (no `Drop`, which would checkpoint and close).
/// A no-op when run directly.
#[test]
fn vector_names_child() {
    let (Some(step), Some(path)) = (std::env::var_os(STEP_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let path = PathBuf::from(path);
    let db = database(&path);
    match step.to_str().unwrap() {
        // Create, and leave it in the log only.
        "create" => {
            db.execute(DOCS).unwrap();
            db.execute(CREATE).unwrap();
        }
        // Create, then drop, both in the log only.
        "create_drop" => {
            db.execute(DOCS).unwrap();
            db.execute(CREATE).unwrap();
            db.execute("DROP INDEX doc_emb").unwrap();
        }
        // Create, checkpoint, then a write after the checkpoint.
        "create_checkpoint" => {
            db.execute(DOCS).unwrap();
            db.execute(CREATE).unwrap();
            db.wal_checkpoint().unwrap();
            db.execute("INSERT (:Doc {title: 'Trams', emb: vector([0.6, 0.8])})")
                .unwrap();
        }
        // Drop after a checkpoint that holds the index.
        "checkpoint_drop" => {
            db.execute(DOCS).unwrap();
            db.execute(CREATE).unwrap();
            db.wal_checkpoint().unwrap();
            db.execute("DROP INDEX doc_emb").unwrap();
        }
        other => panic!("unknown step {other}"),
    }
    std::process::exit(0);
}

/// A named vector index survives a close: listed as Vector, it searches,
/// and `DROP INDEX` of its name after the reopen drops it, also for the
/// next open.
#[test]
fn a_named_vector_index_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.grafeo");
    {
        let db = database(&path);
        db.execute(DOCS).unwrap();
        db.execute(CREATE).unwrap();
        assert_eq!(indexes(&db), ["doc_emb Vector Doc emb"]);
        db.close().unwrap();
    }
    {
        let db = database(&path);
        assert_eq!(indexes(&db), ["doc_emb Vector Doc emb"]);
        assert!(searches(&db));
        db.execute("DROP INDEX doc_emb").unwrap();
        assert_eq!(indexes(&db), Vec::<String>::new());
        assert!(!searches(&db));
        db.close().unwrap();
    }
    let db = database(&path);
    assert_eq!(indexes(&db), Vec::<String>::new());
    assert!(!searches(&db));
}

/// The replay of the log after a crash brings the name back with the index,
/// and a drop in the same log stays dropped.
#[test]
fn the_log_replays_a_create_and_a_drop() {
    for (step, listed) in [("create", true), ("create_drop", false)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.grafeo");
        child(step, &path);
        let db = database(&path);
        let expected: &[&str] = if listed {
            &["doc_emb Vector Doc emb"]
        } else {
            &[]
        };
        assert_eq!(indexes(&db), expected, "{step}");
        assert_eq!(searches(&db), listed, "{step}");
    }
}

/// A checkpoint holds the name: after a crash past the checkpoint the index
/// is listed and searches (the later insert too), and a drop after the
/// checkpoint replays as a drop.
#[test]
fn a_checkpoint_holds_the_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.grafeo");
    child("create_checkpoint", &path);
    let db = database(&path);
    assert_eq!(indexes(&db), ["doc_emb Vector Doc emb"]);
    let found = db
        .vector_search("Doc", "emb", &[0.6, 0.8], 1, None, None)
        .unwrap();
    assert_eq!(found.len(), 1);
    drop(db);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.grafeo");
    child("checkpoint_drop", &path);
    let db = database(&path);
    assert_eq!(indexes(&db), Vec::<String>::new());
    assert!(!searches(&db));
}

/// A vector index's name is unique among every index name: another vector
/// index, a property index and a text index of that name are refused, and
/// the first index stays.
#[test]
fn a_name_in_use_is_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute(DOCS).unwrap();
    db.execute(CREATE).unwrap();
    for statement in [
        "CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 2 METRIC 'cosine'",
        "CREATE VECTOR INDEX doc_emb ON :Doc(other) DIMENSION 2 METRIC 'cosine'",
        "CREATE INDEX doc_emb FOR (d:Doc) ON (d.title)",
    ] {
        let error = db.execute(statement).expect_err(statement);
        assert!(
            error.to_string().contains("Index 'doc_emb' already exists"),
            "{statement}: {error}"
        );
    }
    db.execute("CREATE INDEX doc_title FOR (d:Doc) ON (d.title)")
        .unwrap();
    let error = db
        .execute("CREATE VECTOR INDEX doc_title ON :Doc(emb) DIMENSION 2 METRIC 'cosine'")
        .expect_err("a vector index under a property index's name");
    assert!(
        error
            .to_string()
            .contains("Index 'doc_title' already exists"),
        "{error}"
    );
    assert_eq!(
        indexes(&db),
        ["doc_emb Vector Doc emb", "doc_title Hash Doc title"]
    );
    assert!(searches(&db));
}

/// `CREATE INDEX ... USING VECTOR` (Cypher) records its name as Vector, and
/// `DROP INDEX` of it drops the vector index, leaving a property index on
/// the same key.
#[cfg(feature = "cypher")]
#[test]
fn using_vector_is_listed_as_vector() {
    let db = GrafeoDB::new_in_memory();
    db.execute(DOCS).unwrap();
    db.execute("CREATE INDEX emb_eq FOR (d:Doc) ON (d.emb)")
        .unwrap();
    db.execute_language(
        "CREATE INDEX emb_vec FOR (d:Doc) ON (d.emb) USING VECTOR {dimensions: 2, metric: 'cosine'}",
        "cypher",
        None,
    )
    .unwrap();
    assert_eq!(
        indexes(&db),
        ["emb_eq Hash Doc emb", "emb_vec Vector Doc emb"]
    );
    assert!(searches(&db));
    db.execute("DROP INDEX emb_vec").unwrap();
    assert_eq!(indexes(&db), ["emb_eq Hash Doc emb"]);
    assert!(!searches(&db));
}
