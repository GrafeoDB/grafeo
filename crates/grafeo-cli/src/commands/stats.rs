//! Database statistics command.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format, format_bytes};

/// Detailed database statistics.
#[derive(Serialize)]
struct StatsOutput {
    node_count: usize,
    edge_count: usize,
    label_count: usize,
    edge_type_count: usize,
    property_key_count: usize,
    index_count: usize,
    memory_bytes: usize,
    disk_bytes: Option<usize>,
}

/// Run the stats command.
pub fn run(path: &Path, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let stats = db.detailed_stats();

    let output = StatsOutput {
        node_count: stats.node_count,
        edge_count: stats.edge_count,
        label_count: stats.label_count,
        edge_type_count: stats.edge_type_count,
        property_key_count: stats.property_key_count,
        index_count: stats.index_count,
        memory_bytes: stats.memory_bytes,
        disk_bytes: stats.disk_bytes,
    };

    let fmt: Format = format.into();
    match fmt {
        Format::Json => {
            if !quiet {
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
        }
        Format::Table | Format::Csv => {
            let items = vec![
                ("Nodes", output.node_count.to_string()),
                ("Edges", output.edge_count.to_string()),
                ("Labels", output.label_count.to_string()),
                ("Edge Types", output.edge_type_count.to_string()),
                ("Property Keys", output.property_key_count.to_string()),
                ("Indexes", output.index_count.to_string()),
                ("Memory Usage", format_bytes(output.memory_bytes)),
                (
                    "Disk Usage",
                    output
                        .disk_bytes
                        .map_or_else(|| "N/A".to_string(), format_bytes),
                ),
            ];
            output::print_key_value_table(&items, fmt, quiet);
        }
    }

    Ok(())
}
