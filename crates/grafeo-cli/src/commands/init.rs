//! Database initialization command.

use std::path::Path;

use anyhow::{Context, Result};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

use crate::output::{self, Format};
use crate::{GraphMode, OutputFormat};

/// Run the init command.
pub fn run(path: &Path, mode: GraphMode, format: OutputFormat, quiet: bool) -> Result<()> {
    if path.exists() {
        anyhow::bail!(
            "Path {} already exists. Remove it first or choose a different path.",
            path.display()
        );
    }

    let graph_model = match mode {
        GraphMode::Lpg => GraphModel::Lpg,
        GraphMode::Rdf => GraphModel::Rdf,
    };

    let config = Config::persistent(path).with_graph_model(graph_model);
    let db = GrafeoDB::with_config(config)
        .with_context(|| format!("Failed to create database at {}", path.display()))?;

    let info = db.info();

    let fmt: Format = format.into();
    match fmt {
        Format::Json => {
            if !quiet {
                let output = serde_json::json!({
                    "path": path.display().to_string(),
                    "mode": format!("{:?}", info.mode),
                    "wal_enabled": info.wal_enabled,
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
        Format::Table | Format::Csv => {
            output::success(
                &format!("Created {:?} database at {}", graph_model, path.display()),
                quiet,
            );
        }
    }

    Ok(())
}
