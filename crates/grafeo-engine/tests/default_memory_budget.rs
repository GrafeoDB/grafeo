//! The default memory budget (#751): with no limit set, every way to open takes 0.75 of the
//! memory detected for the process (a container's limit where one binds, never the fallback on
//! a machine whose memory can be read). A fraction is taken of the memory detected at open,
//! applies to the open that names it and is not stored; an explicit limit is stored and holds on
//! reopen, unless a fraction set after it replaces it. A fraction that is NaN, infinite, not
//! above zero or too small to give a byte is refused.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main default_memory_budget::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]

use std::sync::Arc;

use grafeo_common::memory::system::fraction_of;
use grafeo_common::memory::{DEFAULT_MEMORY_FRACTION, MemorySource, detect_memory};
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_engine::{Config, ConfigError, GrafeoDB, ProjectionSpec};

const MIB: usize = 1024 * 1024;

/// Detection finds the memory on every platform the tests run on; the fallback is for a system
/// where nothing can be read.
#[test]
fn detection_finds_memory_here() {
    let found = detect_memory();
    assert!(found.bytes > 0);
    if cfg!(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos"
    )) {
        assert_ne!(found.source, MemorySource::Fallback, "{found:?}");
    }
}

/// With no limit set, each way to open takes the default budget, 0.75 of the detected memory,
/// and reports no configured limit.
#[test]
fn every_open_takes_the_default_budget() {
    let budget = detect_memory().default_budget();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a memory size times 0.75, the default's own arithmetic"
    )]
    let by_hand = (detect_memory().bytes as f64 * DEFAULT_MEMORY_FRACTION) as usize;
    assert_eq!(budget, by_hand);

    let config_open = GrafeoDB::with_config(Config::in_memory()).unwrap();
    let store_open = GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        Config::in_memory(),
    )
    .unwrap();
    let source = GrafeoDB::new_in_memory();
    source
        .create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
        .unwrap();
    let read_open = GrafeoDB::with_read_store(
        source.projection("people").unwrap().shared(),
        Config::in_memory(),
    )
    .unwrap();
    for (open, db) in [
        ("with_config", &config_open),
        ("with_store", &store_open),
        ("with_read_store", &read_open),
    ] {
        assert_eq!(db.buffer_manager().budget(), budget, "{open}");
        assert_eq!(db.memory_limit(), None, "{open}: no limit configured");
    }
}

/// A fraction is taken of the memory detected at open (a container's limit, not the host's),
/// applies to the open that names it and is not stored: reopened without a limit, the file
/// takes the default. A limit is stored when the file is created with it or through
/// `set_stored_config`; given to a later open, it holds for that open only.
#[test]
fn a_fraction_applies_to_its_open_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fraction.grafeo");
    let detected = detect_memory();
    let half = fraction_of(detected.bytes, 0.5);
    assert_eq!(
        Config::in_memory().with_memory_fraction(0.5).memory_limit(),
        Some(half),
        "the Config reads the detection when asked"
    );

    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_fraction(0.5)).unwrap();
    assert_eq!(db.memory_limit(), Some(half));
    assert_eq!(db.buffer_manager().budget(), half);
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.memory_limit(), None, "the fraction was stored");
    assert_eq!(db.buffer_manager().budget(), detected.default_budget());
    db.close().unwrap();
    drop(db);

    // A limit given to an open of an existing file holds for that open only; the file keeps
    // its stored settings.
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(48 * MIB)).unwrap();
    assert_eq!(db.buffer_manager().budget(), 48 * MIB);
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.memory_limit(), None, "an open's limit is not stored");
    assert_eq!(db.buffer_manager().budget(), detected.default_budget());

    // Storing a fraction leaves the stored limit as it was, as a fraction is not a limit.
    db.set_stored_config(&Config::in_memory().with_memory_fraction(0.5))
        .unwrap();
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.memory_limit(), None, "a stored fraction is not a limit");
    assert_eq!(db.buffer_manager().budget(), detected.default_budget());

    // An explicit limit stored on the open file holds on every later open.
    db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
        .unwrap();
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.memory_limit(), Some(48 * MIB), "a stored limit stays");
    assert_eq!(db.buffer_manager().budget(), 48 * MIB);
    db.close().unwrap();
    drop(db);

    // An explicit limit given when the file is created is stored with it.
    let created = dir.path().join("created.grafeo");
    let db =
        GrafeoDB::with_config(Config::persistent(&created).with_memory_limit(48 * MIB)).unwrap();
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&created)).unwrap();
    assert_eq!(
        db.memory_limit(),
        Some(48 * MIB),
        "the limit it was created with"
    );
    assert_eq!(db.buffer_manager().budget(), 48 * MIB);
    db.close().unwrap();
}

/// A fraction that is NaN, not above zero, or so small that its budget comes to 0 bytes is
/// refused as a zero limit; an infinite one is refused as not finite (it would saturate to an
/// unbounded budget); above 1 is allowed.
#[test]
fn a_fraction_must_be_above_zero() {
    for fraction in [f64::INFINITY, f64::NEG_INFINITY] {
        let error = Config::in_memory()
            .with_memory_fraction(fraction)
            .validate()
            .unwrap_err();
        assert_eq!(error, ConfigError::MemoryFractionNotFinite, "{fraction}");
        assert_eq!(error.to_string(), "memory_fraction must be a finite number");
    }
    for fraction in [0.0, -0.5, f64::NAN, 1e-300] {
        assert_eq!(
            Config::in_memory()
                .with_memory_fraction(fraction)
                .validate()
                .unwrap_err(),
            ConfigError::ZeroMemoryLimit,
            "{fraction}"
        );
    }
    for fraction in [0.5, 1.5] {
        assert!(
            Config::in_memory()
                .with_memory_fraction(fraction)
                .validate()
                .is_ok(),
            "{fraction}"
        );
    }
}

/// Storing a Config stores its limit only when its last memory call set one: a fraction set
/// after a limit replaces it, so storing that Config leaves the stored limit as it was; a limit
/// set after a fraction is stored.
#[test]
fn the_later_memory_call_decides_what_is_stored() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("order.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(48 * MIB)).unwrap();
    db.set_stored_config(
        &Config::in_memory()
            .with_memory_limit(64 * MIB)
            .with_memory_fraction(0.5),
    )
    .unwrap();
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        db.memory_limit(),
        Some(48 * MIB),
        "the fraction set last replaced the limit: the stored one stays"
    );

    db.set_stored_config(
        &Config::in_memory()
            .with_memory_fraction(0.5)
            .with_memory_limit(64 * MIB),
    )
    .unwrap();
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        db.memory_limit(),
        Some(64 * MIB),
        "the limit set last is stored"
    );
    assert_eq!(db.buffer_manager().budget(), 64 * MIB);
    db.close().unwrap();
}
