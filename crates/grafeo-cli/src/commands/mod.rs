//! CLI command implementations.

use std::path::Path;

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

pub mod backup;
pub mod compact;
pub mod data;
pub mod import;
pub mod index;
pub mod info;
pub mod init;
pub mod query;
pub mod schema;
pub mod stats;
pub mod validate;
pub mod version;
pub mod wal;

/// Open an existing database, returning a clear error if the path does not exist.
///
/// Use this instead of `GrafeoDB::open` for commands that expect a pre-existing database.
/// The `init` command should use `GrafeoDB::open` directly since it intentionally creates.
pub fn open_existing(path: &Path) -> Result<GrafeoDB> {
    // Without trailing separators: `db/` names the database file `db` too,
    // but no longer reaches it as a path once a 0.5.x directory became a file.
    let path: std::path::PathBuf = path.components().collect();
    let path = path.as_path();
    if !path.exists() {
        anyhow::bail!(
            "Database not found: {}\n\
             Use `grafeo init {}` to create a new database.",
            path.display(),
            path.display()
        );
    }
    GrafeoDB::open(path).with_context(|| format!("Failed to open database at {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::open_existing;

    /// `db/` names the database file `db`: the CLI opens it, as a 0.5.x WAL
    /// directory named `db/` becomes the file `db` when 0.6 migrates it.
    #[test]
    fn a_path_with_a_trailing_separator_opens_the_database_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);

        let written = dir.path().join("db/");
        let db = open_existing(&written).unwrap_or_else(|error| panic!("{error:#}"));
        assert_eq!(db.node_count(), 1, "the database at {}", written.display());
        db.close().unwrap();
    }
}
