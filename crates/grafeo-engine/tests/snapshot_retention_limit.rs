//! The snapshot retention limit as a setting: `Config::with_snapshot_retention_limit(bytes)`,
//! at least 1 MiB, and by default a quarter of the memory limit (or of the detected memory).
//! Past it an explicit transaction's snapshot expires: its next call fails with the retryable
//! `GRAFEO-T012`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main snapshot_retention_limit::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::config::{ConfigError, MIN_SNAPSHOT_RETENTION_LIMIT};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

const MIB: u64 = 1 << 20;

/// Opens `config` with Alix (`bio` "Amsterdam"), and a session whose explicit transaction read
/// her.
fn open_with_reader(config: Config) -> (GrafeoDB, Session) {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
        .unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(read_bio(&reader), Ok(Value::from("Amsterdam")));
    (db, reader)
}

/// Replaces Alix's `bio` with a fresh 64 KiB string `times` times, each in a commit of its own.
fn rewrite_bio(db: &GrafeoDB, times: usize) {
    for round in 0..times {
        let bio = Value::from(format!("{round}{}", "Vincent ".repeat(8 * 1024)).as_str());
        db.execute_with_params(
            "MATCH (p:Person {name: 'Alix'}) SET p.bio = $bio",
            HashMap::from([("bio".to_string(), bio)]),
        )
        .unwrap();
    }
}

/// Alix's `bio` as `session` reads it, or the error's code.
fn read_bio(session: &Session) -> Result<Value, String> {
    session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio")
        .map(|result| result.rows()[0][0].clone())
        .map_err(|error| error.error_code().as_str().to_string())
}

/// Row 28: below 1 MiB the configuration is refused, naming the minimum; 1 MiB is accepted.
#[test]
fn a_limit_below_1_mib_is_refused() {
    let config = Config::in_memory().with_snapshot_retention_limit(512 * 1024);
    let refused = config.validate().unwrap_err();
    assert_eq!(
        refused,
        ConfigError::SnapshotRetentionLimitTooSmall(512 * 1024)
    );
    assert!(refused.to_string().contains("at least 1 MiB"), "{refused}");
    assert!(GrafeoDB::with_config(config).is_err());
    Config::in_memory()
        .with_snapshot_retention_limit(MIN_SNAPSHOT_RETENTION_LIMIT)
        .validate()
        .expect("1 MiB is the minimum");
}

/// The setting reaches the open: 20 rewrites of 64 KiB pass a 1 MiB limit, and the reader's
/// next read fails with `GRAFEO-T012`; under a 64 MiB limit it reads its snapshot.
#[test]
fn the_setting_bounds_an_open_transaction() {
    let (db, reader) = open_with_reader(Config::in_memory().with_snapshot_retention_limit(MIB));
    rewrite_bio(&db, 20);
    assert_eq!(read_bio(&reader), Err("GRAFEO-T012".to_string()));

    let (db, reader) =
        open_with_reader(Config::in_memory().with_snapshot_retention_limit(64 * MIB));
    rewrite_bio(&db, 20);
    assert_eq!(read_bio(&reader), Ok(Value::from("Amsterdam")));
}

/// Without the setting the limit is a quarter of the memory limit: 4 MiB under a 16 MiB
/// limit, so 40 rewrites of 64 KiB (2.5 MiB) keep the reader and 80 (5 MiB) expire it.
#[test]
fn the_default_is_a_quarter_of_the_memory_limit() {
    let (db, reader) = open_with_reader(Config::in_memory().with_memory_limit(16 << 20));
    rewrite_bio(&db, 40);
    assert_eq!(read_bio(&reader), Ok(Value::from("Amsterdam")));
    rewrite_bio(&db, 40);
    assert_eq!(read_bio(&reader), Err("GRAFEO-T012".to_string()));
}
