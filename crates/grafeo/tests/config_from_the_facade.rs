//! Every type a `Config` method takes is exported by the `grafeo` crate, so a
//! user configures a database through the facade alone, without depending on
//! `grafeo-engine` or `grafeo-common`.
//!
//! ```bash
//! cargo test -p grafeo --features encryption --test config_from_the_facade
//! ```

use std::time::Duration;

use grafeo::{Config, DurabilityMode, StorageFormat};

#[test]
fn storage_format_and_durability_come_from_the_facade() {
    let config = Config::persistent("amsterdam.grafeo")
        .with_storage_format(StorageFormat::Auto)
        .with_wal_durability(DurabilityMode::batch(Duration::from_millis(19), 88));
    assert_eq!(config.storage_format, StorageFormat::Auto);
    assert!(
        matches!(
            config.wal_durability,
            DurabilityMode::Batch {
                max_delay_ms: 19,
                max_records: 88,
                ..
            }
        ),
        "{:?}",
        config.wal_durability
    );
    assert_eq!(config.validate(), Ok(()));

    let config = Config::persistent("berlin.grafeo")
        .with_wal_durability(DurabilityMode::adaptive(Duration::from_millis(3)));
    assert!(
        matches!(
            config.wal_durability,
            DurabilityMode::Adaptive {
                target_interval_ms: 3,
                ..
            }
        ),
        "{:?}",
        config.wal_durability
    );
}

#[cfg(feature = "cdc")]
#[test]
fn cdc_retention_comes_from_the_facade() {
    use grafeo::CdcRetentionConfig;

    let config = Config::in_memory()
        .with_cdc()
        .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_events(88));
    assert_eq!(
        (
            config.cdc_retention.max_epochs,
            config.cdc_retention.max_events
        ),
        (None, Some(88))
    );
}

/// The facade's `encryption` feature turns encryption at rest on: a database
/// configured through the facade is written encrypted and opens only with
/// its key.
#[cfg(all(feature = "encryption", feature = "grafeo-file"))]
#[test]
fn an_encrypted_database_is_configured_through_the_facade() {
    use std::sync::Arc;

    use grafeo::{EncryptionConfig, GrafeoDB, KeyChain};

    let dir = std::env::temp_dir().join(format!("grafeo-facade-encryption-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("prague.grafeo");
    let key = || EncryptionConfig::new(Arc::new(KeyChain::new([19; 32])));

    let db = GrafeoDB::with_config(Config::persistent(&path).with_encryption(key())).unwrap();
    db.create_node(&["Person"]).unwrap();
    db.close().unwrap();
    drop(db);

    let error = GrafeoDB::with_config(Config::persistent(&path))
        .err()
        .expect("an encrypted database does not open without its key");
    assert!(
        error.to_string().contains("needs its key"),
        "the error says the key is missing: {error}"
    );

    let db = GrafeoDB::with_config(Config::persistent(&path).with_encryption(key())).unwrap();
    assert_eq!(db.node_count(), 1, "the key opens the database");
    db.close().unwrap();
    drop(db);
    let _ = std::fs::remove_dir_all(&dir);
}
