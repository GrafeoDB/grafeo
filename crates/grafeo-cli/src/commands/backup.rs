//! Backup management commands.

use std::fs;

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

use crate::output;
use crate::{BackupCommands, OutputFormat};

/// Run backup commands.
pub fn run(cmd: BackupCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        BackupCommands::Create { path, output: out } => {
            output::status(&format!("Creating backup of {}...", path.display()), quiet);

            let db = GrafeoDB::open(&path)?;
            db.save(&out)
                .with_context(|| format!("Failed to create backup at {}", out.display()))?;

            output::success(&format!("Backup created at {}", out.display()), quiet);
        }
        BackupCommands::Restore {
            backup,
            path,
            force,
        } => {
            // Check if target exists
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
                fs::remove_dir_all(&path)
                    .with_context(|| format!("Failed to remove {}", path.display()))?;
            }

            output::status(&format!("Restoring from {}...", backup.display()), quiet);

            // Open backup and save to new location
            let db = GrafeoDB::open(&backup)
                .with_context(|| format!("Failed to open backup at {}", backup.display()))?;
            db.save(&path)
                .with_context(|| format!("Failed to restore to {}", path.display()))?;

            output::success(&format!("Database restored to {}", path.display()), quiet);
        }
    }

    Ok(())
}
