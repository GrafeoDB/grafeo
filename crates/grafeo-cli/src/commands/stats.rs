//! Database statistics command.

use std::path::Path;

use anyhow::Result;
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

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

/// Format bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Run the stats command.
pub fn run(path: &Path, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = GrafeoDB::open(path)?;
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
        Format::Table => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(1), "1 bytes");
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(10240), "10.00 KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.00 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 100), "100.00 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 * 2), "2.00 GB");
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.50 GB"
        );
    }
}
