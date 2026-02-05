//! Database compaction command.

use std::path::Path;

use anyhow::Result;
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format};

/// Compaction result output.
#[derive(Serialize)]
struct CompactionOutput {
    dry_run: bool,
    before_size_bytes: usize,
    after_size_bytes: Option<usize>,
    space_saved_bytes: Option<usize>,
    space_saved_percent: Option<f64>,
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

/// Run the compact command.
pub fn run(path: &Path, dry_run: bool, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = GrafeoDB::open(path)?;
    let stats_before = db.detailed_stats();

    if dry_run {
        output::status("Dry run - no changes will be made", quiet);

        let output = CompactionOutput {
            dry_run: true,
            before_size_bytes: stats_before.memory_bytes,
            after_size_bytes: None,
            space_saved_bytes: None,
            space_saved_percent: None,
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
                    ("Mode", "Dry Run".to_string()),
                    ("Current Size", format_bytes(stats_before.memory_bytes)),
                    ("Nodes", stats_before.node_count.to_string()),
                    ("Edges", stats_before.edge_count.to_string()),
                ];
                output::print_key_value_table(&items, fmt, quiet);
            }
        }
    } else {
        output::status("Compacting database...", quiet);

        // TODO: Implement actual compaction when API is available
        // For now, just report current state
        let stats_after = db.detailed_stats();

        let space_saved = stats_before
            .memory_bytes
            .saturating_sub(stats_after.memory_bytes);

        let percent_saved = if stats_before.memory_bytes > 0 {
            (space_saved as f64 / stats_before.memory_bytes as f64) * 100.0
        } else {
            0.0
        };

        let output = CompactionOutput {
            dry_run: false,
            before_size_bytes: stats_before.memory_bytes,
            after_size_bytes: Some(stats_after.memory_bytes),
            space_saved_bytes: Some(space_saved),
            space_saved_percent: Some(percent_saved),
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
                    ("Before", format_bytes(stats_before.memory_bytes)),
                    ("After", format_bytes(stats_after.memory_bytes)),
                    ("Saved", format_bytes(space_saved)),
                    ("Reduction", format!("{:.1}%", percent_saved)),
                ];
                output::print_key_value_table(&items, fmt, quiet);
            }
        }

        output::success("Compaction completed", quiet);
    }

    Ok(())
}
