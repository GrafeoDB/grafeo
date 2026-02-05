//! WAL management commands.

use anyhow::Result;
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::output::{self, Format};
use crate::{OutputFormat, WalCommands};

/// WAL status output.
#[derive(Serialize)]
struct WalStatusOutput {
    enabled: bool,
    path: Option<String>,
    size_bytes: usize,
    size_human: String,
    record_count: usize,
    last_checkpoint: Option<u64>,
    current_epoch: u64,
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

/// Run WAL commands.
pub fn run(cmd: WalCommands, format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        WalCommands::Status { path } => {
            let db = GrafeoDB::open(&path)?;
            let status = db.wal_status();

            let output = WalStatusOutput {
                enabled: status.enabled,
                path: status.path.map(|p| p.display().to_string()),
                size_bytes: status.size_bytes,
                size_human: format_bytes(status.size_bytes),
                record_count: status.record_count,
                last_checkpoint: status.last_checkpoint,
                current_epoch: status.current_epoch,
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
                        ("Enabled", output.enabled.to_string()),
                        ("Path", output.path.unwrap_or_else(|| "N/A".to_string())),
                        ("Size", output.size_human),
                        ("Records", output.record_count.to_string()),
                        (
                            "Last Checkpoint",
                            output
                                .last_checkpoint
                                .map(|ts| {
                                    // Format timestamp
                                    format!("{ts}")
                                })
                                .unwrap_or_else(|| "Never".to_string()),
                        ),
                        ("Current Epoch", output.current_epoch.to_string()),
                    ];
                    output::print_key_value_table(&items, fmt, quiet);
                }
            }
        }
        WalCommands::Checkpoint { path } => {
            output::status("Forcing WAL checkpoint...", quiet);

            let db = GrafeoDB::open(&path)?;
            db.wal_checkpoint()?;

            output::success("WAL checkpoint completed", quiet);
        }
    }

    Ok(())
}
