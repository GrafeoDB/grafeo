//! Backup management commands.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

use crate::output;
use crate::{BackupCommands, OutputFormat};

/// Run backup commands.
pub fn run(cmd: BackupCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        BackupCommands::Create { path, output: out } => {
            output::status(&format!("Creating backup of {}...", path.display()), quiet);

            let db = super::open_existing(&path)?;
            db.save(&out)
                .with_context(|| format!("Failed to create backup at {}", out.display()))?;

            output::success(&format!("Backup created at {}", out.display()), quiet);
        }
        BackupCommands::Restore {
            backup,
            path,
            force,
        } => {
            if path.exists() && !force {
                anyhow::bail!(
                    "Target path {} already exists. Use --force to overwrite.",
                    path.display()
                );
            }

            if path.exists() && force {
                output::status(
                    &format!("Removing existing database at {}...", path.display()),
                    quiet,
                );
                remove_database(&path)?;
            }

            output::status(&format!("Restoring from {}...", backup.display()), quiet);

            // Read-only: restoring never changes the backup (a 0.5.x backup
            // is read in place, not migrated).
            if !backup.exists() {
                anyhow::bail!("Backup not found: {}", backup.display());
            }
            let db = GrafeoDB::open_read_only(&backup)
                .with_context(|| format!("Failed to open backup at {}", backup.display()))?;
            db.save(&path)
                .with_context(|| format!("Failed to restore to {}", path.display()))?;

            output::success(&format!("Database restored to {}", path.display()), quiet);
        }
        BackupCommands::Full { path, output: out } => {
            output::status(
                &format!("Creating full backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db
                .backup_full(&out)
                .with_context(|| format!("Failed to create full backup at {}", out.display()))?;

            output::success(
                &format!(
                    "Full backup created: {} ({} bytes, epoch 0-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Incremental { path, output: out } => {
            output::status(
                &format!("Creating incremental backup of {}...", path.display()),
                quiet,
            );

            let db = super::open_existing(&path)?;
            let segment = db.backup_incremental(&out).with_context(|| {
                format!("Failed to create incremental backup at {}", out.display())
            })?;

            output::success(
                &format!(
                    "Incremental backup created: {} ({} bytes, epoch {}-{})",
                    segment.filename,
                    segment.size_bytes,
                    segment.start_epoch.as_u64(),
                    segment.end_epoch.as_u64()
                ),
                quiet,
            );
        }
        BackupCommands::Status { path } => {
            let manifest = GrafeoDB::read_backup_manifest(&path)
                .with_context(|| format!("Failed to read manifest at {}", path.display()))?;

            match manifest {
                Some(m) => {
                    println!("Backup manifest (version {})", m.version);
                    println!("Segments: {}", m.segments.len());
                    if let Some((start, end)) = m.epoch_range() {
                        println!("Epoch range: {} - {}", start.as_u64(), end.as_u64());
                    }
                    println!();
                    for (i, seg) in m.segments.iter().enumerate() {
                        println!(
                            "  [{i}] {:?} {} (epoch {}-{}, {} bytes)",
                            seg.kind,
                            seg.filename,
                            seg.start_epoch.as_u64(),
                            seg.end_epoch.as_u64(),
                            seg.size_bytes
                        );
                    }
                }
                None => {
                    println!("No backup manifest found at {}", path.display());
                }
            }
        }
        BackupCommands::RestoreToEpoch {
            backup_dir,
            epoch,
            output: out,
        } => {
            output::status(
                &format!(
                    "Restoring to epoch {epoch} from {}...",
                    backup_dir.display()
                ),
                quiet,
            );

            let target = grafeo_common::types::EpochId::new(epoch);
            GrafeoDB::restore_to_epoch(&backup_dir, target, &out)
                .with_context(|| format!("Failed to restore to epoch {epoch}"))?;

            output::success(
                &format!("Database restored to epoch {epoch} at {}", out.display()),
                quiet,
            );
        }
    }

    Ok(())
}

/// Removes the database at `path`: a single file and its sidecar WAL
/// `<path>.wal/` (whose records would otherwise be replayed into the database
/// restored at that path), or a 0.5.x WAL directory.
fn remove_database(path: &Path) -> Result<()> {
    let path: PathBuf = path.components().collect();
    if path.is_dir() {
        fs::remove_dir_all(&path)
    } else {
        fs::remove_file(&path)
    }
    .with_context(|| format!("Failed to remove {}", path.display()))?;
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    let sidecar = PathBuf::from(sidecar);
    if sidecar.exists() {
        fs::remove_dir_all(&sidecar)
            .with_context(|| format!("Failed to remove {}", sidecar.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use grafeo_engine::GrafeoDB;

    use super::run;
    use crate::{BackupCommands, OutputFormat};

    fn restore(backup: &Path, path: &Path, force: bool) -> anyhow::Result<()> {
        run(
            BackupCommands::Restore {
                backup: backup.to_path_buf(),
                path: path.to_path_buf(),
                force,
            },
            OutputFormat::Auto,
            true,
        )
    }

    /// A closed database at `path` holding one person.
    fn database_with(path: &Path, name: &str) {
        let db = GrafeoDB::open(path).unwrap();
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
        db.close().unwrap();
    }

    /// The names of the people in the database at `path`, read without
    /// changing it.
    fn names(path: &Path) -> Vec<String> {
        let db = GrafeoDB::open_read_only(path).unwrap();
        let names = db
            .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].as_str().expect("a name").to_owned())
            .collect();
        db.close().unwrap();
        names
    }

    fn sidecar(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// `--force` replaces an existing single-file database, and removes its
    /// sidecar WAL, which would otherwise replay the old database's last
    /// commits into the restored one.
    #[test]
    fn restore_with_force_replaces_a_database_file_and_its_wal() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        let live = dir.path().join("live.grafeo");
        database_with(&backup, "Alix");
        database_with(&live, "Gus");
        std::fs::create_dir_all(sidecar(&live)).unwrap();

        restore(&backup, &live, false).expect_err("an existing target needs --force");
        restore(&backup, &live, true).unwrap_or_else(|error| panic!("{error:#}"));

        assert!(live.is_file(), "the restored database is a file");
        assert!(!sidecar(&live).exists(), "the old sidecar WAL is gone");
        assert_eq!(names(&live), vec!["Alix".to_string()]);
    }

    /// Restoring reads the backup and changes nothing in it: a backup written
    /// by 0.5.x is read in place, never migrated.
    #[test]
    fn restore_leaves_a_0_5_backup_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let backup = dir.path().join("backup.grafeo");
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../grafeo-engine/tests/fixtures/released/0.5.44/closed.grafeo");
        std::fs::copy(&fixture, &backup).unwrap();
        let before = std::fs::read(&backup).unwrap();

        let restored = dir.path().join("restored.grafeo");
        restore(&backup, &restored, false).unwrap_or_else(|error| panic!("{error:#}"));

        assert!(
            std::fs::read(&backup).unwrap() == before,
            "the backup is unchanged"
        );
        assert!(
            !dir.path().join("backup.grafeo.pre-0.6").exists(),
            "the backup was not migrated"
        );
        assert!(
            !names(&restored).is_empty(),
            "the restored database holds the backup's people"
        );
    }
}
