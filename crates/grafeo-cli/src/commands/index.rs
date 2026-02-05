//! Index management commands.

use anyhow::Result;
use grafeo_engine::GrafeoDB;
use serde::Serialize;

use crate::output::{self, Format};
use crate::{IndexCommands, OutputFormat};

/// Index information output.
#[derive(Serialize)]
struct IndexOutput {
    name: String,
    index_type: String,
    target: String,
    property: Option<String>,
    entry_count: usize,
}

/// Index statistics output.
#[derive(Serialize)]
struct IndexStatsOutput {
    total_indexes: usize,
    indexes: Vec<IndexOutput>,
}

/// Run index commands.
pub fn run(cmd: IndexCommands, format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        IndexCommands::List { path } => {
            let db = GrafeoDB::open(&path)?;
            let stats = db.detailed_stats();

            let fmt: Format = format.into();
            match fmt {
                Format::Json => {
                    if !quiet {
                        let output = IndexStatsOutput {
                            total_indexes: stats.index_count,
                            indexes: vec![], // TODO: Get actual index list when API available
                        };
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    }
                }
                Format::Table => {
                    if !quiet {
                        println!("Total indexes: {}\n", stats.index_count);

                        let mut table = output::create_table();
                        output::add_header(
                            &mut table,
                            &["Name", "Type", "Target", "Property", "Entries"],
                        );
                        // TODO: Populate with actual index data when API available
                        println!("{table}");
                    }
                }
            }
        }
        IndexCommands::Stats { path } => {
            let db = GrafeoDB::open(&path)?;
            let stats = db.detailed_stats();

            let fmt: Format = format.into();
            let items = vec![
                ("Total Indexes", stats.index_count.to_string()),
                ("Labels Indexed", stats.label_count.to_string()),
                ("Edge Types Indexed", stats.edge_type_count.to_string()),
            ];
            output::print_key_value_table(&items, fmt, quiet);
        }
    }

    Ok(())
}
