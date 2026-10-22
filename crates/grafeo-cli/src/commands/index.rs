//! Index management commands.

use anyhow::Result;
use serde::Serialize;

use crate::output::{self, Format};
use crate::{IndexCommands, OutputFormat};

/// Index information for serialization.
#[derive(Serialize)]
struct IndexOutput {
    name: String,
    index_type: String,
    target: String,
    unique: bool,
}

/// Run index commands.
pub fn run(cmd: IndexCommands, format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        IndexCommands::List { path } => {
            let db = super::open_existing(&path)?;
            let indexes = db.list_indexes();

            let fmt: Format = format.into();
            match fmt {
                Format::Json => {
                    if !quiet {
                        let output: Vec<IndexOutput> = indexes
                            .iter()
                            .map(|idx| IndexOutput {
                                name: idx.name.clone(),
                                index_type: idx.index_type.clone(),
                                target: idx.target.clone(),
                                unique: idx.unique,
                            })
                            .collect();
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    }
                }
                Format::Table | Format::Csv => {
                    if indexes.is_empty() {
                        if !quiet {
                            println!("No indexes.");
                        }
                    } else {
                        let headers = vec![
                            "Name".to_string(),
                            "Type".to_string(),
                            "Target".to_string(),
                            "Unique".to_string(),
                        ];
                        let rows: Vec<Vec<String>> = indexes
                            .iter()
                            .map(|idx| {
                                vec![
                                    idx.name.clone(),
                                    idx.index_type.clone(),
                                    idx.target.clone(),
                                    if idx.unique { "yes" } else { "no" }.to_string(),
                                ]
                            })
                            .collect();
                        output::print_result_table(&headers, &rows, fmt, quiet);
                        if !quiet {
                            println!(
                                "{} index{}",
                                indexes.len(),
                                if indexes.len() == 1 { "" } else { "es" }
                            );
                        }
                    }
                }
            }
        }
        IndexCommands::Stats { path } => {
            let db = super::open_existing(&path)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) {
        let db = grafeo_engine::GrafeoDB::open(dir).expect("create db");
        let n1 = db.create_node(&["Person"]).unwrap();
        let n2 = db.create_node(&["Company"]).unwrap();
        db.set_node_property(n1, "name", grafeo_common::types::Value::from("Alix"))
            .unwrap();
        db.set_node_property(n2, "name", grafeo_common::types::Value::from("Acme"))
            .unwrap();
        db.create_edge(n1, n2, "WORKS_AT").unwrap();
        db.close().expect("close db");
    }

    #[test]
    fn test_index_list_empty() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        create_test_db(&db_path);

        // List should succeed even with no explicit indexes
        run(
            IndexCommands::List {
                path: db_path.clone(),
            },
            OutputFormat::Table,
            true,
        )
        .expect("list should succeed");

        // Also test JSON format
        run(
            IndexCommands::List { path: db_path },
            OutputFormat::Json,
            true,
        )
        .expect("list json should succeed");
    }

    #[test]
    fn test_index_stats() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        create_test_db(&db_path);

        run(
            IndexCommands::Stats {
                path: db_path.clone(),
            },
            OutputFormat::Table,
            true,
        )
        .expect("stats table should succeed");

        run(
            IndexCommands::Stats { path: db_path },
            OutputFormat::Json,
            true,
        )
        .expect("stats json should succeed");
    }

    #[test]
    fn test_index_list_nonexistent_database() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("nonexistent.grafeo");

        let result = run(
            IndexCommands::List { path: db_path },
            OutputFormat::Table,
            true,
        );
        assert!(result.is_err());
    }
}
