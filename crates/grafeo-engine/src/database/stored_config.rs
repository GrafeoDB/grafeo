//! The configuration a database file stores (#551).
//!
//! A database created as a file stores what defines or protects it: its
//! graph model, backward edges, schema constraints, WAL and durability mode,
//! memory limit, section tier overrides, and its spill path when that lies in
//! the database's directory. The thread count and other paths belong to the
//! machine and are not stored. It is a record of the catalog
//! ([`StoredConfigRecord`]), which every checkpoint writes.
//!
//! An open reads the record from the file before it builds anything (the
//! memory limit sizes the buffer manager, the backward edges shape the
//! store), and applies each stored value its configuration did not set with a
//! builder ([`Config::was_set`]): a value set for the open applies to that
//! open only. The graph model and the backward edges are fixed when the
//! database is created, so an open that sets another value is refused. The
//! record changes only through [`GrafeoDB::set_stored_config`], which
//! checkpoints at once: the record is always in the file's image, never only
//! in its WAL, which the open replays after it built the stores.
//!
//! [`GrafeoDB::set_stored_config`]: super::GrafeoDB::set_stored_config

use std::path::{Path, PathBuf};
use std::time::Duration;

use grafeo_common::storage::catalog_record::{
    DurabilityRecord, GraphModelRecord, HistoryRetentionRecord, SectionMemoryRecord,
    StoredConfigRecord, TierRecord,
};
use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::utils::error::{Error, Result};
use grafeo_storage::file::GrafeoFileManager;

use super::encryption::DatabaseKeys;
use crate::config::{Config, DurabilityMode, GraphModel, HistoryRetention, SetFields};

/// What a new database at `path` stores of `config`: every stored setting's
/// value in effect, set or default.
pub(super) fn record_of(config: &Config, path: &Path) -> StoredConfigRecord {
    let mut sections: Vec<SectionMemoryRecord> = config
        .section_configs()
        .iter()
        .map(|(section, memory)| section_record(*section, memory))
        .collect();
    sections.sort_by_key(|section| section.section);
    StoredConfigRecord {
        graph_model: match config.graph_model() {
            GraphModel::Lpg => GraphModelRecord::Lpg,
            GraphModel::Rdf => GraphModelRecord::Rdf,
        },
        backward_edges: config.backward_edges(),
        schema_constraints: config.schema_constraints(),
        wal_enabled: config.wal_enabled(),
        wal_durability: durability_record(config.wal_durability()),
        memory_limit: config
            .memory_limit()
            .map(|limit| u64::try_from(limit).unwrap_or(u64::MAX)),
        spill_path: config
            .spill_path()
            .and_then(|spill| relative_spill_path(spill, path)),
        sections,
        history_retention: retention_record(config.history_retention()),
    }
}

/// The record of `stored` with the settings `changes` set (see
/// [`Config::was_set`]), for a database at `path`.
///
/// # Errors
///
/// [`Error::InvalidValue`] when `changes` sets the graph model or the
/// backward edges to another value: they are fixed at creation.
pub(super) fn changed(
    stored: StoredConfigRecord,
    changes: &Config,
    path: &Path,
) -> Result<StoredConfigRecord> {
    let wanted = record_of(changes, path);
    check_fixed(&stored, changes)?;
    let set = |field| changes.was_set(field);
    Ok(StoredConfigRecord {
        graph_model: stored.graph_model,
        backward_edges: stored.backward_edges,
        schema_constraints: if set(SetFields::SCHEMA_CONSTRAINTS) {
            wanted.schema_constraints
        } else {
            stored.schema_constraints
        },
        wal_enabled: if set(SetFields::WAL_ENABLED) {
            wanted.wal_enabled
        } else {
            stored.wal_enabled
        },
        wal_durability: if set(SetFields::WAL_DURABILITY) {
            wanted.wal_durability
        } else {
            stored.wal_durability
        },
        memory_limit: if set(SetFields::MEMORY_LIMIT) {
            wanted.memory_limit
        } else {
            stored.memory_limit
        },
        spill_path: if set(SetFields::SPILL_PATH) {
            wanted.spill_path
        } else {
            stored.spill_path
        },
        sections: if set(SetFields::SECTION_CONFIGS) {
            wanted.sections
        } else {
            stored.sections
        },
        history_retention: if set(SetFields::HISTORY_RETENTION) {
            wanted.history_retention
        } else {
            stored.history_retention
        },
    })
}

/// `config` with the values of `stored` (a database at `path`) for every
/// setting it did not set.
///
/// # Errors
///
/// [`Error::InvalidValue`] when `config` sets the graph model or the backward
/// edges to another value than the database was created with.
pub(super) fn applied(config: Config, stored: &StoredConfigRecord, path: &Path) -> Result<Config> {
    check_fixed(stored, &config)?;
    let mut config = config;
    let set = |config: &Config, field| config.was_set(field);
    if !set(&config, SetFields::GRAPH_MODEL) {
        config = config.with_graph_model(match stored.graph_model {
            GraphModelRecord::Lpg => GraphModel::Lpg,
            GraphModelRecord::Rdf => GraphModel::Rdf,
        });
    }
    if !set(&config, SetFields::BACKWARD_EDGES) && !stored.backward_edges {
        config = config.without_backward_edges();
    }
    if !set(&config, SetFields::SCHEMA_CONSTRAINTS) {
        config = if stored.schema_constraints {
            config.with_schema_constraints()
        } else {
            config.without_schema_constraints()
        };
    }
    if !set(&config, SetFields::WAL_ENABLED) {
        config = if stored.wal_enabled {
            config.with_wal()
        } else {
            config.without_wal()
        };
    }
    if !set(&config, SetFields::WAL_DURABILITY) {
        config = config.with_wal_durability(durability_mode(stored.wal_durability));
    }
    if !set(&config, SetFields::MEMORY_LIMIT)
        && let Some(limit) = stored.memory_limit
    {
        config = config.with_memory_limit(usize::try_from(limit).unwrap_or(usize::MAX));
    }
    if !set(&config, SetFields::SPILL_PATH)
        && let Some(spill) = &stored.spill_path
    {
        config = config.with_spill_path(database_dir(path).join(spill));
    }
    if !set(&config, SetFields::SECTION_CONFIGS) {
        for section in &stored.sections {
            // A section this build does not know is left out.
            if let Some((section_type, memory)) = section_config(section) {
                config = config.with_section_config(section_type, memory);
            }
        }
    }
    if !set(&config, SetFields::HISTORY_RETENTION) {
        config = config.with_history_retention(history_retention(stored.history_retention));
    }
    Ok(config)
}

/// Refuses `config` when it sets the graph model or the backward edges to
/// another value than `stored` holds.
fn check_fixed(stored: &StoredConfigRecord, config: &Config) -> Result<()> {
    let model = match config.graph_model() {
        GraphModel::Lpg => GraphModelRecord::Lpg,
        GraphModel::Rdf => GraphModelRecord::Rdf,
    };
    if config.was_set(SetFields::GRAPH_MODEL) && model != stored.graph_model {
        return Err(Error::InvalidValue(format!(
            "the database was created with the graph model {:?}, which is fixed: it cannot open \
             or be stored as {:?}",
            stored.graph_model, model
        )));
    }
    if config.was_set(SetFields::BACKWARD_EDGES) && config.backward_edges() != stored.backward_edges
    {
        return Err(Error::InvalidValue(format!(
            "the database was created {} backward edges, which is fixed",
            if stored.backward_edges {
                "with"
            } else {
                "without"
            }
        )));
    }
    Ok(())
}

/// The configuration the database file at `path` stores, read from its
/// active image before the open builds anything; `None` for a file that
/// stores none (written by 0.5.x or 0.6.0-beta.1).
///
/// # Errors
///
/// The errors of opening the file read-only (a lock another process holds
/// exclusively, a wrong key) and of reading its catalog section.
pub(super) fn read(path: &Path, keys: &DatabaseKeys) -> Result<Option<StoredConfigRecord>> {
    let fm =
        GrafeoFileManager::open_read_only_with_cipher_for(path, |id| keys.container_cipher(id))?;
    let stored = fm.read_image(|image| match image.section_source(SectionType::Catalog) {
        Some(source) => super::catalog_section::read_stored_config(&*source),
        None => Ok(None),
    });
    fm.close()?;
    stored
}

/// The directory of the database file at `path`.
fn database_dir(path: &Path) -> PathBuf {
    path.parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// `spill` relative to the directory of the database at `path`, when it lies
/// in it; `None` otherwise (an absolute path elsewhere belongs to the
/// machine).
fn relative_spill_path(spill: &Path, path: &Path) -> Option<String> {
    let absolute = |p: &Path| std::path::absolute(p).ok();
    let (spill, dir) = (absolute(spill)?, absolute(&database_dir(path))?);
    spill
        .strip_prefix(&dir)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .and_then(|relative| relative.to_str().map(str::to_string))
}

/// The record of a history retention.
pub(super) fn retention_record(retention: HistoryRetention) -> HistoryRetentionRecord {
    match retention {
        HistoryRetention::Duration(window) => HistoryRetentionRecord::Duration {
            millis: u64::try_from(window.as_millis()).unwrap_or(u64::MAX),
        },
        HistoryRetention::Forever => HistoryRetentionRecord::Forever,
        // `HistoryRetention` is non-exhaustive for callers only.
        HistoryRetention::Off => HistoryRetentionRecord::Off,
    }
}

/// The history retention a record holds.
pub(super) fn history_retention(record: HistoryRetentionRecord) -> HistoryRetention {
    match record {
        HistoryRetentionRecord::Off => HistoryRetention::Off,
        HistoryRetentionRecord::Duration { millis } => {
            HistoryRetention::Duration(Duration::from_millis(millis))
        }
        HistoryRetentionRecord::Forever => HistoryRetention::Forever,
    }
}

fn durability_record(mode: DurabilityMode) -> DurabilityRecord {
    match mode {
        DurabilityMode::Sync => DurabilityRecord::Sync,
        DurabilityMode::Batch {
            max_delay_ms,
            max_records,
            ..
        } => DurabilityRecord::Batch {
            max_delay_ms,
            max_records,
        },
        DurabilityMode::Adaptive {
            target_interval_ms, ..
        } => DurabilityRecord::Adaptive { target_interval_ms },
        DurabilityMode::NoSync => DurabilityRecord::NoSync,
    }
}

fn durability_mode(record: DurabilityRecord) -> DurabilityMode {
    match record {
        DurabilityRecord::Sync => DurabilityMode::Sync,
        DurabilityRecord::Batch {
            max_delay_ms,
            max_records,
        } => DurabilityMode::batch(Duration::from_millis(max_delay_ms), max_records),
        DurabilityRecord::Adaptive { target_interval_ms } => {
            DurabilityMode::adaptive(Duration::from_millis(target_interval_ms))
        }
        DurabilityRecord::NoSync => DurabilityMode::NoSync,
    }
}

fn section_record(section: SectionType, memory: &SectionMemoryConfig) -> SectionMemoryRecord {
    SectionMemoryRecord {
        section: u8::try_from(section as u32).unwrap_or(u8::MAX),
        max_ram: memory
            .max_ram
            .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX)),
        tier: match memory.tier {
            TierOverride::ForceRam => TierRecord::ForceRam,
            TierOverride::ForceDisk => TierRecord::ForceDisk,
            _ => TierRecord::Auto,
        },
    }
}

fn section_config(record: &SectionMemoryRecord) -> Option<(SectionType, SectionMemoryConfig)> {
    let section = SectionType::from_u8(record.section)?;
    Some((
        section,
        SectionMemoryConfig {
            max_ram: record
                .max_ram
                .map(|bytes| usize::try_from(bytes).unwrap_or(usize::MAX)),
            tier: match record.tier {
                TierRecord::Auto => TierOverride::Auto,
                TierRecord::ForceRam => TierOverride::ForceRam,
                TierRecord::ForceDisk => TierOverride::ForceDisk,
            },
        },
    ))
}
