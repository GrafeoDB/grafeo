//! Data export/import commands.

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

use crate::output;
use crate::{DataCommands, OutputFormat};

/// Run data commands.
pub fn run(cmd: DataCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        DataCommands::Dump {
            path,
            output: out,
            format: dump_format,
        } => {
            let format_name = dump_format.as_deref().unwrap_or("parquet");
            output::status(
                &format!(
                    "Exporting {} to {} (format: {})...",
                    path.display(),
                    out.display(),
                    format_name
                ),
                quiet,
            );

            let db = GrafeoDB::open(&path)
                .with_context(|| format!("Failed to open database at {}", path.display()))?;

            // Get counts for progress reporting
            let info = db.info();

            // TODO: Implement actual export when format handlers are available
            // For now, use the native backup format
            db.save(&out)
                .with_context(|| format!("Failed to export to {}", out.display()))?;

            output::success(
                &format!(
                    "Exported {} nodes and {} edges to {}",
                    info.node_count,
                    info.edge_count,
                    out.display()
                ),
                quiet,
            );
        }
        DataCommands::Load { input, path } => {
            output::status(
                &format!("Importing {} into {}...", input.display(), path.display()),
                quiet,
            );

            // TODO: Implement format detection and import
            // For now, use native backup format
            let db = GrafeoDB::open(&input)
                .with_context(|| format!("Failed to open dump at {}", input.display()))?;
            db.save(&path)
                .with_context(|| format!("Failed to save to {}", path.display()))?;

            let info = db.info();
            output::success(
                &format!(
                    "Imported {} nodes and {} edges to {}",
                    info.node_count,
                    info.edge_count,
                    path.display()
                ),
                quiet,
            );
        }
    }

    Ok(())
}
