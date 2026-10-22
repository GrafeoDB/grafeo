//! Database compaction command.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::OutputFormat;
use crate::output::{self, Format, format_bytes};

/// Compaction result output.
#[derive(Serialize)]
struct CompactionOutput {
    dry_run: bool,
    before_size_bytes: usize,
    after_size_bytes: Option<usize>,
}

/// Run the compact command.
pub fn run(path: &Path, dry_run: bool, format: OutputFormat, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let stats_before = db.detailed_stats();

    if dry_run {
        output::status("Dry run, no changes will be made", quiet);

        let result = CompactionOutput {
            dry_run: true,
            before_size_bytes: stats_before.memory_bytes,
            after_size_bytes: None,
        };

        let fmt: Format = format.into();
        match fmt {
            Format::Json => {
                if !quiet {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Format::Table | Format::Csv => {
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
        db.gc();

        let stats_after = db.detailed_stats();

        let result = CompactionOutput {
            dry_run: false,
            before_size_bytes: stats_before.memory_bytes,
            after_size_bytes: Some(stats_after.memory_bytes),
        };

        let fmt: Format = format.into();
        match fmt {
            Format::Json => {
                if !quiet {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
            }
            Format::Table | Format::Csv => {
                let reclaimed = stats_before
                    .memory_bytes
                    .saturating_sub(stats_after.memory_bytes);
                let items = vec![
                    ("Before", format_bytes(stats_before.memory_bytes)),
                    ("After", format_bytes(stats_after.memory_bytes)),
                    ("Reclaimed", format_bytes(reclaimed)),
                ];
                output::print_key_value_table(&items, fmt, quiet);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) {
        let db = grafeo_engine::GrafeoDB::open(dir).expect("create db");
        let n1 = db.create_node(&["Person"]).unwrap();
        let n2 = db.create_node(&["Person"]).unwrap();
        db.set_node_property(n1, "name", grafeo_common::types::Value::from("Alix"))
            .unwrap();
        db.set_node_property(n2, "name", grafeo_common::types::Value::from("Gus"))
            .unwrap();
        db.create_edge(n1, n2, "KNOWS").unwrap();
        db.close().expect("close db");
    }

    #[test]
    fn test_compact_dry_run() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        create_test_db(&db_path);

        // Dry run should succeed without modifying the database
        run(&db_path, true, OutputFormat::Table, true).expect("dry run should succeed");
    }

    #[test]
    fn test_compact_actual() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        create_test_db(&db_path);

        // Actual compaction
        run(&db_path, false, OutputFormat::Table, true).expect("compaction should succeed");
    }

    #[test]
    fn test_compact_json_format() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        create_test_db(&db_path);

        run(&db_path, true, OutputFormat::Json, true).expect("json format dry run should succeed");
        run(&db_path, false, OutputFormat::Json, true)
            .expect("json format compaction should succeed");
    }

    #[test]
    fn test_compact_nonexistent_database() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("nonexistent.grafeo");

        let result = run(&db_path, true, OutputFormat::Table, true);
        assert!(result.is_err());
    }
}
