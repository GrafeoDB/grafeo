//! Database info command.

use std::path::Path;

use anyhow::Result;
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

/// Information about a database.
#[derive(Serialize)]
struct DatabaseInfoOutput {
    mode: String,
    node_count: usize,
    edge_count: usize,
    is_persistent: bool,
    path: Option<String>,
    wal_enabled: bool,
    version: String,
}

/// Run the info command.
pub fn run(path: &Path, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = GrafeoDB::open(path)?;
    let info = db.info();

    let output = DatabaseInfoOutput {
        mode: format!("{:?}", info.mode),
        node_count: info.node_count,
        edge_count: info.edge_count,
        is_persistent: info.is_persistent,
        path: info.path.map(|p| p.display().to_string()),
        wal_enabled: info.wal_enabled,
        version: info.version,
    };

    let fmt: Format = format.into();
    match fmt {
        Format::Json => {
            if !quiet {
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
        Format::Table => {
            let items = vec![
                ("Mode", output.mode),
                ("Nodes", output.node_count.to_string()),
                ("Edges", output.edge_count.to_string()),
                ("Persistent", output.is_persistent.to_string()),
                (
                    "Path",
                    output.path.unwrap_or_else(|| "(in-memory)".to_string()),
                ),
                ("WAL Enabled", output.wal_enabled.to_string()),
                ("Version", output.version),
            ];
            output::print_key_value_table(&items, fmt, quiet);
        }
    }

    Ok(())
}
