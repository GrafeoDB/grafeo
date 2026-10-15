//! A database file stores the configuration it was created with (#551):
//! `GrafeoDB::open(path)` and `Config::read_only(path)` apply it to every
//! setting they do not set themselves, after a close, after a crash, and in
//! another process; a value set for an open applies to that open only;
//! `set_stored_config` changes what the file stores; the graph model and the
//! backward edges are fixed at creation; files that store nothing (0.5.x)
//! open with the defaults.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test stored_config
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]
#![allow(
    missing_docs,
    reason = "a test crate has no public API: each test documents itself"
)]

#[path = "common/replay.rs"]
mod replay;

use std::path::{Path, PathBuf};

use grafeo_common::storage::{SectionType, TierOverride};
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};

use replay::reopened_after_crash;

const MEMORY_LIMIT: usize = 88 * 1024 * 1024;

/// The configuration of `amsterdam.grafeo` in `dir`: every stored setting
/// away from its default.
fn created(dir: &Path) -> Config {
    created_at(&dir.join("amsterdam.grafeo"))
}

/// [`created`] for the database file `path`.
fn created_at(path: &Path) -> Config {
    let dir = path.parent().unwrap();
    Config::persistent(path)
        .with_memory_limit(MEMORY_LIMIT)
        .without_backward_edges()
        .with_schema_constraints()
        .with_wal_durability(DurabilityMode::Sync)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceRam)
        .with_spill_path(dir.join("amsterdam.spill"))
}

/// The stored settings of `config`, comparable.
fn stored(
    config: &Config,
) -> (
    Option<usize>,
    bool,
    bool,
    bool,
    DurabilityMode,
    Option<PathBuf>,
    Option<TierOverride>,
) {
    (
        config.memory_limit(),
        config.backward_edges(),
        config.schema_constraints(),
        config.wal_enabled(),
        config.wal_durability(),
        config.spill_path().map(Path::to_path_buf),
        config
            .section_configs()
            .get(&SectionType::VectorStore)
            .map(|memory| memory.tier),
    )
}

fn expected(
    dir: &Path,
) -> (
    Option<usize>,
    bool,
    bool,
    bool,
    DurabilityMode,
    Option<PathBuf>,
    Option<TierOverride>,
) {
    (
        Some(MEMORY_LIMIT),
        false,
        true,
        true,
        DurabilityMode::Sync,
        Some(dir.join("amsterdam.spill")),
        Some(TierOverride::ForceRam),
    )
}

/// The defaults, as a database that stores nothing opens with.
fn defaults() -> (
    Option<usize>,
    bool,
    bool,
    bool,
    DurabilityMode,
    Option<PathBuf>,
    Option<TierOverride>,
) {
    let config = Config::persistent("default.grafeo");
    (None, true, false, true, config.wal_durability(), None, None)
}

/// A reopen after `close()` applies every stored setting; the memory limit
/// also reaches the database (the issue's reproduction).
#[test]
fn a_reopen_applies_the_stored_configuration() {
    let dir = tempfile::tempdir().unwrap();
    GrafeoDB::with_config(created(dir.path()))
        .unwrap()
        .close()
        .unwrap();
    let db = GrafeoDB::open(dir.path().join("amsterdam.grafeo")).unwrap();
    assert_eq!(stored(db.config()), expected(dir.path()));
    assert_eq!(db.memory_limit(), Some(MEMORY_LIMIT));
    db.close().unwrap();
}

/// A read-only open applies the stored settings too.
#[test]
fn a_read_only_open_applies_the_stored_configuration() {
    let dir = tempfile::tempdir().unwrap();
    GrafeoDB::with_config(created(dir.path()))
        .unwrap()
        .close()
        .unwrap();
    let db = GrafeoDB::with_config(Config::read_only(dir.path().join("amsterdam.grafeo"))).unwrap();
    let (memory, backward, schema, _wal, durability, spill, tier) = stored(db.config());
    let (e_memory, e_backward, e_schema, _, e_durability, _, e_tier) = expected(dir.path());
    assert_eq!(
        (memory, backward, schema, durability, tier),
        (e_memory, e_backward, e_schema, e_durability, e_tier)
    );
    assert_eq!(spill, Some(dir.path().join("amsterdam.spill")));
}

/// A database that exits without `close()` (its WAL holds what it wrote
/// after the creation) reopens with the stored settings: the file stored
/// them when it was created.
#[test]
fn a_reopen_after_a_crash_applies_the_stored_configuration() {
    let (dir, db) = reopened_after_crash(
        "a_reopen_after_a_crash_applies_the_stored_configuration",
        |path| {
            if path.exists() {
                GrafeoDB::open(path).unwrap()
            } else {
                GrafeoDB::with_config(created_at(path)).unwrap()
            }
        },
        |db| {
            db.execute("INSERT (:City {name: 'Prague'})").unwrap();
        },
    );
    // The crash helper names the file `db.grafeo`.
    assert_eq!(stored(db.config()), expected(dir.path()));
    assert_eq!(
        db.execute("MATCH (c:City) RETURN count(c)").unwrap().rows()[0][0],
        grafeo_common::types::Value::Int64(1)
    );
}

/// A value set for an open wins for that open only: the next open without
/// it gets the stored value again.
#[test]
fn a_value_set_on_open_wins_for_that_open_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    GrafeoDB::with_config(created(dir.path()))
        .unwrap()
        .close()
        .unwrap();

    let db = GrafeoDB::with_config(
        Config::persistent(&path)
            .with_memory_limit(19 * 1024 * 1024)
            .without_schema_constraints()
            .with_wal_durability(DurabilityMode::NoSync),
    )
    .unwrap();
    assert_eq!(db.config().memory_limit(), Some(19 * 1024 * 1024));
    assert!(!db.config().schema_constraints());
    assert_eq!(db.config().wal_durability(), DurabilityMode::NoSync);
    assert!(
        !db.config().backward_edges(),
        "what the open does not set comes from the file"
    );
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        stored(db.config()),
        expected(dir.path()),
        "the stored values stay"
    );
    db.close().unwrap();
}

/// `set_stored_config` changes what the file stores for the next open; the
/// running database keeps its configuration.
#[test]
fn set_stored_config_changes_the_next_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    GrafeoDB::with_config(created(dir.path()))
        .unwrap()
        .close()
        .unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    db.set_stored_config(
        &Config::in_memory()
            .with_memory_limit(3 * 1024 * 1024)
            .with_wal_durability(DurabilityMode::NoSync),
    )
    .unwrap();
    assert_eq!(
        db.config().memory_limit(),
        Some(MEMORY_LIMIT),
        "this database keeps what it opened with"
    );
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.config().memory_limit(), Some(3 * 1024 * 1024));
    assert_eq!(db.config().wal_durability(), DurabilityMode::NoSync);
    assert!(
        db.config().schema_constraints(),
        "a setting the change does not set stays"
    );
    db.close().unwrap();
}

/// The graph model and the backward edges are fixed at creation: an open or
/// a change that sets another value is refused, and the file is unchanged.
#[test]
fn the_backward_edges_are_fixed_at_creation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    GrafeoDB::with_config(Config::persistent(&path).without_backward_edges())
        .unwrap()
        .close()
        .unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    let error = db
        .set_stored_config(&Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf))
        .unwrap_err();
    assert!(error.to_string().contains("fixed"), "{error}");
    db.close().unwrap();

    let error = GrafeoDB::with_config(
        Config::persistent(&path).with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .err()
    .expect("an open as another graph model is refused");
    assert!(error.to_string().contains("graph model"), "{error}");
    let db = GrafeoDB::open(&path).unwrap();
    assert!(!db.config().backward_edges());
    db.close().unwrap();
}

/// An in-memory database stores no configuration.
#[test]
fn an_in_memory_database_stores_no_configuration() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .set_stored_config(&Config::in_memory().with_memory_limit(MEMORY_LIMIT))
        .unwrap_err();
    assert!(error.to_string().contains("in-memory"), "{error}");
}

/// A database the file of 0.5.44 migrates stores no configuration: it opens
/// with the defaults, as before, and so does its next open.
#[test]
fn a_migrated_0_5_44_file_opens_with_the_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closed.grafeo");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/closed.grafeo"),
        &path,
    )
    .unwrap();
    for open in ["the migrating open", "the next open"] {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(stored(db.config()), defaults(), "{open}");
        db.close().unwrap();
    }
}

/// A database created without its WAL stores that: the next open logs
/// nothing either.
#[test]
fn a_database_created_without_its_wal_reopens_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    GrafeoDB::with_config(Config::persistent(&path).without_wal())
        .unwrap()
        .close()
        .unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert!(!db.config().wal_enabled());
    assert!(db.wal().is_none(), "no WAL is opened");
    db.close().unwrap();
}

/// An RDF database reopens as one.
#[cfg(feature = "triple-store")]
#[test]
fn an_rdf_database_reopens_as_rdf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    GrafeoDB::with_config(
        Config::persistent(&path).with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap()
    .close()
    .unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.config().graph_model(), grafeo_engine::GraphModel::Rdf);
    db.close().unwrap();
}
