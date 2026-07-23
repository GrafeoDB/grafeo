//! Data export/import commands.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

use anyhow::{Context, Result};

use grafeo_common::types::Value;

use crate::output;
use crate::{DataCommands, OutputFormat};

/// Run data commands.
pub fn run(cmd: DataCommands, _format: OutputFormat, quiet: bool) -> Result<()> {
    match cmd {
        DataCommands::Dump {
            path,
            output: out,
            export_format,
        } => {
            let format_name = export_format.as_deref().unwrap_or("json");

            match format_name {
                "json" | "jsonl" => dump_jsonl(&path, &out, quiet)?,
                "arrow" | "arrow-ipc" | "ipc" => dump_arrow(&path, &out, quiet)?,
                "gexf" => dump_gexf(&path, &out, quiet)?,
                "graphml" => dump_graphml(&path, &out, quiet)?,
                _ => {
                    anyhow::bail!(
                        "Unsupported export format: {format_name}\n\
                         Supported formats: json, arrow, gexf, graphml"
                    );
                }
            }
        }
        DataCommands::Load { input, path } => {
            let file = std::fs::File::open(&input)
                .with_context(|| format!("Failed to open input file: {}", input.display()))?;
            let reader = BufReader::new(file);

            let db = if path.exists() {
                super::open_existing(&path)?
            } else {
                grafeo_engine::GrafeoDB::open(&path)
                    .with_context(|| format!("Failed to create database at {}", path.display()))?
            };

            let mut node_count = 0usize;
            let mut edge_count = 0usize;

            for (line_num, line) in reader.lines().enumerate() {
                let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let record: serde_json::Value = serde_json::from_str(trimmed)
                    .with_context(|| format!("Invalid JSON on line {}", line_num + 1))?;

                match record.get("type").and_then(|t| t.as_str()) {
                    Some("node") => {
                        let labels: Vec<&str> = record
                            .get("labels")
                            .and_then(|l| l.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                            .unwrap_or_default();

                        let id = db.create_node(&labels)?;

                        if let Some(props) = record.get("properties").and_then(|p| p.as_object()) {
                            for (key, val) in props {
                                let value: Value =
                                    serde_json::from_value(val.clone()).unwrap_or(Value::Null);
                                db.set_node_property(id, key, value);
                            }
                        }
                        node_count += 1;
                    }
                    Some("edge") => {
                        let source = record
                            .get("source")
                            .and_then(|v| v.as_u64())
                            .map(grafeo_common::types::NodeId)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'source' on line {}", line_num + 1)
                            })?;
                        let target = record
                            .get("target")
                            .and_then(|v| v.as_u64())
                            .map(grafeo_common::types::NodeId)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'target' on line {}", line_num + 1)
                            })?;
                        let edge_type = record
                            .get("edge_type")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                anyhow::anyhow!("Missing 'edge_type' on line {}", line_num + 1)
                            })?;

                        let id = db.create_edge(source, target, edge_type);

                        if let Some(props) = record.get("properties").and_then(|p| p.as_object()) {
                            for (key, val) in props {
                                let value: Value =
                                    serde_json::from_value(val.clone()).unwrap_or(Value::Null);
                                db.set_edge_property(id, key, value);
                            }
                        }
                        edge_count += 1;
                    }
                    other => {
                        anyhow::bail!("Unknown record type {:?} on line {}", other, line_num + 1);
                    }
                }
            }

            output::status(
                &format!(
                    "Loaded {} nodes and {} edges into {}",
                    node_count,
                    edge_count,
                    path.display()
                ),
                quiet,
            );
        }
    }

    Ok(())
}

fn dump_jsonl(path: &std::path::Path, out: &std::path::Path, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let file = std::fs::File::create(out)
        .with_context(|| format!("Failed to create output file: {}", out.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    let mut node_count = 0usize;
    for node in db.iter_nodes() {
        let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
        let properties: HashMap<String, Value> = node
            .properties
            .to_btree_map()
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect();
        let record = serde_json::json!({
            "type": "node",
            "id": node.id.0,
            "labels": labels,
            "properties": properties,
        });
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        node_count += 1;
    }

    let mut edge_count = 0usize;
    for edge in db.iter_edges() {
        let properties: HashMap<String, Value> = edge
            .properties
            .to_btree_map()
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect();
        let record = serde_json::json!({
            "type": "edge",
            "id": edge.id.0,
            "source": edge.src.0,
            "target": edge.dst.0,
            "edge_type": edge.edge_type.as_str(),
            "properties": properties,
        });
        serde_json::to_writer(&mut writer, &record)?;
        writeln!(writer)?;
        edge_count += 1;
    }

    writer.flush()?;
    output::status(
        &format!(
            "Exported {} nodes and {} edges to {}",
            node_count,
            edge_count,
            out.display()
        ),
        quiet,
    );
    Ok(())
}

#[cfg(feature = "arrow-export")]
fn dump_arrow(path: &std::path::Path, out: &std::path::Path, quiet: bool) -> Result<()> {
    use grafeo_common::LogicalType;

    let db = super::open_existing(path)?;

    // Build nodes RecordBatch
    let nodes: Vec<_> = db.iter_nodes().collect();
    let mut property_keys = std::collections::BTreeSet::new();
    for node in &nodes {
        for (key, _) in node.properties.iter() {
            property_keys.insert(key.clone());
        }
    }

    let node_columns: Vec<String> = std::iter::once("_id".to_string())
        .chain(std::iter::once("_labels".to_string()))
        .chain(property_keys.iter().map(|k| k.as_str().to_string()))
        .collect();

    let node_types: Vec<LogicalType> = std::iter::once(LogicalType::Int64)
        .chain(std::iter::once(LogicalType::String))
        .chain(property_keys.iter().map(|_| LogicalType::Any))
        .collect();

    let node_rows: Vec<Vec<Value>> = nodes
        .iter()
        .map(|node| {
            let mut row = Vec::with_capacity(node_columns.len());
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            row.push(Value::Int64(node.id.0 as i64));
            let labels_str: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            row.push(Value::String(labels_str.join(",").into()));
            for key in &property_keys {
                row.push(node.properties.get(key).cloned().unwrap_or(Value::Null));
            }
            row
        })
        .collect();

    // Build edges RecordBatch
    let edges: Vec<_> = db.iter_edges().collect();
    let mut edge_prop_keys = std::collections::BTreeSet::new();
    for edge in &edges {
        for (key, _) in edge.properties.iter() {
            edge_prop_keys.insert(key.clone());
        }
    }

    let edge_columns: Vec<String> = ["_id", "_source", "_target", "_type"]
        .iter()
        .map(|s| (*s).to_string())
        .chain(edge_prop_keys.iter().map(|k| k.as_str().to_string()))
        .collect();

    let edge_types: Vec<LogicalType> = [
        LogicalType::Int64,
        LogicalType::Int64,
        LogicalType::Int64,
        LogicalType::String,
    ]
    .into_iter()
    .chain(edge_prop_keys.iter().map(|_| LogicalType::Any))
    .collect();

    let edge_rows: Vec<Vec<Value>> = edges
        .iter()
        .map(|edge| {
            let mut row = Vec::with_capacity(edge_columns.len());
            // reason: Node/edge IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            {
                row.push(Value::Int64(edge.id.0 as i64));
                row.push(Value::Int64(edge.src.0 as i64));
                row.push(Value::Int64(edge.dst.0 as i64));
            }
            row.push(Value::String(edge.edge_type.clone()));
            for key in &edge_prop_keys {
                row.push(edge.properties.get(key).cloned().unwrap_or(Value::Null));
            }
            row
        })
        .collect();

    // Write both batches to IPC stream
    let file = std::fs::File::create(out)
        .with_context(|| format!("Failed to create output file: {}", out.display()))?;
    let mut file_writer = std::io::BufWriter::new(file);

    let node_batch = grafeo_engine::database::arrow::query_result_to_record_batch(
        &node_columns,
        &node_types,
        &node_rows,
    )
    .map_err(|e| anyhow::anyhow!("Arrow node conversion failed: {e}"))?;

    let edge_batch = grafeo_engine::database::arrow::query_result_to_record_batch(
        &edge_columns,
        &edge_types,
        &edge_rows,
    )
    .map_err(|e| anyhow::anyhow!("Arrow edge conversion failed: {e}"))?;

    // Write nodes batch
    let node_ipc = grafeo_engine::database::arrow::record_batch_to_ipc_stream(&node_batch)
        .map_err(|e| anyhow::anyhow!("Arrow IPC serialization failed: {e}"))?;
    let edge_ipc = grafeo_engine::database::arrow::record_batch_to_ipc_stream(&edge_batch)
        .map_err(|e| anyhow::anyhow!("Arrow IPC serialization failed: {e}"))?;

    // Write a simple container: 4-byte length prefix for each stream
    let node_len = u32::try_from(node_ipc.len())
        .map_err(|_| anyhow::anyhow!("Node IPC stream exceeds 4 GiB limit"))?;
    let edge_len = u32::try_from(edge_ipc.len())
        .map_err(|_| anyhow::anyhow!("Edge IPC stream exceeds 4 GiB limit"))?;
    file_writer.write_all(&node_len.to_le_bytes())?;
    file_writer.write_all(&node_ipc)?;
    file_writer.write_all(&edge_len.to_le_bytes())?;
    file_writer.write_all(&edge_ipc)?;
    file_writer.flush()?;

    let node_count = nodes.len();
    let edge_count = edges.len();
    output::status(
        &format!(
            "Exported {} nodes and {} edges to {} (Arrow IPC)",
            node_count,
            edge_count,
            out.display()
        ),
        quiet,
    );
    Ok(())
}

#[cfg(not(feature = "arrow-export"))]
fn dump_arrow(_path: &std::path::Path, _out: &std::path::Path, _quiet: bool) -> Result<()> {
    anyhow::bail!(
        "Arrow export is not enabled.\n\
         Rebuild with --features arrow-export to enable Arrow IPC export."
    )
}

fn dump_gexf(path: &std::path::Path, out: &std::path::Path, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let nodes: Vec<_> = db.iter_nodes().collect();
    let edges: Vec<_> = db.iter_edges().collect();

    let file = std::fs::File::create(out)
        .with_context(|| format!("Failed to create output file: {}", out.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    grafeo_engine::export::gexf::write_gexf(&mut writer, &nodes, &edges)
        .map_err(|e| anyhow::anyhow!("GEXF export failed: {e}"))?;
    writer.flush()?;

    output::status(
        &format!(
            "Exported {} nodes and {} edges to {} (GEXF)",
            nodes.len(),
            edges.len(),
            out.display()
        ),
        quiet,
    );
    Ok(())
}

fn dump_graphml(path: &std::path::Path, out: &std::path::Path, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let nodes: Vec<_> = db.iter_nodes().collect();
    let edges: Vec<_> = db.iter_edges().collect();

    let file = std::fs::File::create(out)
        .with_context(|| format!("Failed to create output file: {}", out.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    grafeo_engine::export::graphml::write_graphml(&mut writer, &nodes, &edges)
        .map_err(|e| anyhow::anyhow!("GraphML export failed: {e}"))?;
    writer.flush()?;

    output::status(
        &format!(
            "Exported {} nodes and {} edges to {} (GraphML)",
            nodes.len(),
            edges.len(),
            out.display()
        ),
        quiet,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) -> grafeo_engine::GrafeoDB {
        let db = grafeo_engine::GrafeoDB::open(dir).expect("create db");
        let n1 = db.create_node(&["Person"]);
        let n2 = db.create_node(&["Company"]);
        db.set_node_property(n1, "name", Value::from("Alix"));
        db.set_node_property(n1, "age", Value::Int64(30));
        db.set_node_property(n2, "name", Value::from("Acme"));
        let e = db.create_edge(n1, n2, "WORKS_AT");
        db.set_edge_property(e, "since", Value::Int64(2020));
        db
    }

    #[test]
    fn test_dump_and_load_roundtrip() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("source.grafeo");
        let dump_path = temp.path().join("dump.jsonl");
        let target_path = temp.path().join("target.grafeo");

        let db = create_test_db(&db_path);
        drop(db);

        // Dump
        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true, // quiet
        )
        .expect("dump should succeed");

        // Verify dump file exists and has content
        let content = std::fs::read_to_string(&dump_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3); // 2 nodes + 1 edge

        // Load into new database
        run(
            DataCommands::Load {
                input: dump_path,
                path: target_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load should succeed");

        // Verify loaded data
        let loaded = grafeo_engine::GrafeoDB::open(&target_path).unwrap();
        let info = loaded.info();
        assert_eq!(info.node_count, 2);
        assert_eq!(info.edge_count, 1);
    }

    #[test]
    fn test_dump_explicit_json_format() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.json");

        // Create and drop the database so the CLI can reopen it
        drop(create_test_db(&db_path));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: Some("json".to_string()),
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump with explicit json format should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        assert!(!content.is_empty());
    }

    #[test]
    fn test_dump_invalid_format_rejected() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.parquet");

        let _db = create_test_db(&db_path);

        let result = run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path,
                export_format: Some("parquet".to_string()),
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unsupported export format"));
    }

    #[test]
    fn test_load_invalid_json_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("bad.jsonl");
        let db_path = temp.path().join("target.grafeo");

        std::fs::write(&input_path, "not valid json\n").unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("line 1"));
    }

    #[test]
    fn test_load_unknown_type_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("unknown.jsonl");
        let db_path = temp.path().join("target.grafeo");

        std::fs::write(&input_path, "{\"type\": \"widget\"}\n").unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown record type"));
    }

    #[test]
    fn test_load_skips_empty_lines() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("sparse.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "\n{\"type\":\"node\",\"labels\":[\"A\"],\"properties\":{}}\n\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("should handle empty lines");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_edge_missing_source_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("bad_edge.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"type\":\"edge\",\"target\":1,\"edge_type\":\"KNOWS\"}\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("source"));
    }

    #[test]
    fn test_load_edge_missing_target_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("no_target.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"type\":\"edge\",\"source\":1,\"edge_type\":\"KNOWS\"}\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("target"),
            "expected 'target' in: {err_msg}"
        );
    }

    #[test]
    fn test_load_edge_missing_edge_type_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("no_edge_type.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"type\":\"edge\",\"source\":1,\"target\":2}\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("edge_type"),
            "expected 'edge_type' in: {err_msg}"
        );
    }

    #[test]
    fn test_load_record_with_no_type_field_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("no_type.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"labels\":[\"Person\"],\"properties\":{}}\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Unknown record type"),
            "expected 'Unknown record type' in: {err_msg}"
        );
    }

    #[test]
    fn test_load_into_existing_database() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("existing.grafeo");
        let input_path = temp.path().join("data.jsonl");

        // Create a database with one node already in it
        {
            let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
            db.create_node(&["Existing"]);
        }

        let content = "{\"type\":\"node\",\"labels\":[\"Imported\"],\"properties\":{}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load into existing db should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 2);
    }

    #[test]
    fn test_load_nonexistent_input_file_fails() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("does_not_exist.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to open input file"),
            "expected file open error in: {err_msg}"
        );
    }

    #[test]
    fn test_dump_nonexistent_database_fails() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("nonexistent.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        let result = run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path,
                export_format: None,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Database not found"),
            "expected 'Database not found' in: {err_msg}"
        );
    }

    #[test]
    fn test_dump_jsonl_format_accepted() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        drop(create_test_db(&db_path));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: Some("jsonl".to_string()),
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump with jsonl format should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        assert!(!content.is_empty());
    }

    #[test]
    fn test_dump_verifies_node_json_structure() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        drop(create_test_db(&db_path));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();

        // Parse first line as a node record and verify structure
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "node");
        assert!(first["id"].is_number(), "node id should be a number");
        assert!(first["labels"].is_array(), "labels should be an array");
        assert!(
            first["properties"].is_object(),
            "properties should be an object"
        );
    }

    #[test]
    fn test_dump_verifies_edge_json_structure() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        drop(create_test_db(&db_path));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();

        // Find the edge record (last line)
        let edge: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(edge["type"], "edge");
        assert!(edge["id"].is_number(), "edge id should be a number");
        assert!(edge["source"].is_number(), "source should be a number");
        assert!(edge["target"].is_number(), "target should be a number");
        assert_eq!(edge["edge_type"], "WORKS_AT");
        assert!(
            edge["properties"].is_object(),
            "properties should be an object"
        );
        // Value is serialized with its enum variant, e.g. {"Int64": 2020}
        assert_eq!(edge["properties"]["since"]["Int64"], 2020);
    }

    #[test]
    fn test_dump_non_quiet_mode() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        drop(create_test_db(&db_path));

        // Run with quiet=false to cover the output::status branch
        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path,
                export_format: None,
            },
            OutputFormat::Json,
            false,
        )
        .expect("dump non-quiet should succeed");
    }

    #[test]
    fn test_load_non_quiet_mode() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("data.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content =
            "{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Gus\"}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            false,
        )
        .expect("load non-quiet should succeed");
    }

    #[test]
    fn test_load_node_without_labels() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("no_labels.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // Missing labels field entirely
        let content = "{\"type\":\"node\",\"properties\":{}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load node without labels should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_node_without_properties() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("no_props.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // No properties field at all
        let content = "{\"type\":\"node\",\"labels\":[\"City\"]}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load node without properties should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_edge_with_properties() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("edge_props.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // Create two nodes first, then an edge with properties
        let content = "\
{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Vincent\"}}\n\
{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Jules\"}}\n\
{\"type\":\"edge\",\"source\":0,\"target\":1,\"edge_type\":\"KNOWS\",\"properties\":{\"since\":1994,\"close\":true}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load edge with properties should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 2);
        assert_eq!(db.info().edge_count, 1);
    }

    #[test]
    fn test_load_node_with_null_property_value() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("null_prop.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // Property value that cannot be deserialized into a Value falls back to Value::Null
        let content = "{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Mia\",\"unknown_field\":null}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load with null property should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_whitespace_only_lines_skipped() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("whitespace.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "   \n\t\n{\"type\":\"node\",\"labels\":[\"City\"],\"properties\":{\"name\":\"Berlin\"}}\n  \n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("whitespace-only lines should be skipped");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_dump_empty_database() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("empty.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        // Create empty database
        drop(grafeo_engine::GrafeoDB::open(&db_path).expect("create empty db"));

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump empty db should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        assert!(
            content.trim().is_empty(),
            "empty db dump should produce no lines"
        );
    }

    #[test]
    fn test_load_invalid_json_on_later_line() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("bad_line3.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "\
{\"type\":\"node\",\"labels\":[\"A\"],\"properties\":{}}\n\
{\"type\":\"node\",\"labels\":[\"B\"],\"properties\":{}}\n\
this is not json\n";
        std::fs::write(&input_path, content).unwrap();

        let result = run(
            DataCommands::Load {
                input: input_path,
                path: db_path,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("line 3"),
            "expected line 3 in error: {err_msg}"
        );
    }

    #[test]
    fn test_dump_node_with_multiple_labels() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("multi_label.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        {
            let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
            db.create_node(&["Person", "Employee"]);
        }

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let record: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        let labels = record["labels"].as_array().unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_load_edge_without_properties() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("edge_no_props.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "\
{\"type\":\"node\",\"labels\":[\"A\"],\"properties\":{}}\n\
{\"type\":\"node\",\"labels\":[\"B\"],\"properties\":{}}\n\
{\"type\":\"edge\",\"source\":0,\"target\":1,\"edge_type\":\"LINKS\"}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load edge without properties should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().edge_count, 1);
    }

    #[test]
    fn test_dump_to_invalid_output_path_fails() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.grafeo");

        drop(create_test_db(&db_path));

        // Try to write to a path inside a nonexistent directory
        let bad_output = temp.path().join("nonexistent_dir").join("dump.jsonl");

        let result = run(
            DataCommands::Dump {
                path: db_path,
                output: bad_output,
                export_format: None,
            },
            OutputFormat::Json,
            true,
        );

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Failed to create output file"),
            "expected file creation error in: {err_msg}"
        );
    }

    #[test]
    fn test_load_node_with_various_property_types() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("various_types.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // Test string, integer, float, boolean, and null property values
        let content = "{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\
            \"name\":\"Butch\",\
            \"age\":42,\
            \"height\":1.85,\
            \"active\":true,\
            \"nickname\":null\
        }}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load with various property types should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_dump_preserves_edge_properties() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("edge_props.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        {
            let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
            let n1 = db.create_node(&["Person"]);
            let n2 = db.create_node(&["Person"]);
            db.set_node_property(n1, "name", Value::from("Django"));
            db.set_node_property(n2, "name", Value::from("Shosanna"));
            let edge = db.create_edge(n1, n2, "FRIENDS_WITH");
            db.set_edge_property(edge, "year", Value::Int64(2012));
            db.set_edge_property(edge, "strong", Value::Bool(true));
        }

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 3, "should have 2 nodes + 1 edge");

        // Find the edge record
        let edge_line = lines
            .iter()
            .find(|line| line.contains("\"type\":\"edge\""))
            .expect("should have an edge record");
        let edge: serde_json::Value = serde_json::from_str(edge_line).unwrap();
        assert_eq!(edge["edge_type"], "FRIENDS_WITH");
        let props = edge["properties"].as_object().unwrap();
        // Value serializes with variant tags: {"Int64": 2012}, {"Bool": true}
        assert_eq!(props["year"]["Int64"], 2012);
        assert_eq!(props["strong"]["Bool"], true);
    }

    #[test]
    fn test_dump_node_properties_serialized() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("props.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        {
            let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
            let node = db.create_node(&["City"]);
            db.set_node_property(node, "name", Value::from("Amsterdam"));
            db.set_node_property(node, "population", Value::Int64(905_234));
            db.set_node_property(node, "capital", Value::Bool(true));
        }

        run(
            DataCommands::Dump {
                path: db_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        let content = std::fs::read_to_string(&dump_path).unwrap();
        let record: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(record["type"], "node");
        let props = record["properties"].as_object().unwrap();
        // Value serializes with variant tags: {"String": "Amsterdam"}, {"Int64": 905234}, {"Bool": true}
        assert_eq!(props["name"]["String"], "Amsterdam");
        assert_eq!(props["population"]["Int64"], 905_234);
        assert_eq!(props["capital"]["Bool"], true);
    }

    #[test]
    fn test_full_roundtrip_preserves_data() {
        let temp = TempDir::new().unwrap();
        let src_path = temp.path().join("source.grafeo");
        let dump_path = temp.path().join("roundtrip.jsonl");
        let dst_path = temp.path().join("dest.grafeo");

        // Build a richer graph
        {
            let db = grafeo_engine::GrafeoDB::open(&src_path).expect("create db");
            let hans = db.create_node(&["Person"]);
            let beatrix = db.create_node(&["Person"]);
            let paris = db.create_node(&["City"]);
            db.set_node_property(hans, "name", Value::from("Hans"));
            db.set_node_property(beatrix, "name", Value::from("Beatrix"));
            db.set_node_property(paris, "name", Value::from("Paris"));
            db.set_node_property(paris, "country", Value::from("France"));

            let edge1 = db.create_edge(hans, paris, "LIVES_IN");
            db.set_edge_property(edge1, "since", Value::Int64(2015));
            let edge2 = db.create_edge(beatrix, paris, "VISITED");
            db.set_edge_property(edge2, "year", Value::Int64(2023));
        }

        // Dump
        run(
            DataCommands::Dump {
                path: src_path,
                output: dump_path.clone(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump should succeed");

        // Load
        run(
            DataCommands::Load {
                input: dump_path,
                path: dst_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load should succeed");

        let db = grafeo_engine::GrafeoDB::open(&dst_path).unwrap();
        let info = db.info();
        assert_eq!(info.node_count, 3);
        assert_eq!(info.edge_count, 2);
    }

    #[test]
    fn test_load_node_with_empty_labels_array() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("empty_labels.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "{\"type\":\"node\",\"labels\":[],\"properties\":{\"name\":\"Gus\"}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load node with empty labels should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 1);
    }

    #[test]
    fn test_load_mixed_nodes_and_edges() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("mixed.jsonl");
        let db_path = temp.path().join("target.grafeo");

        let content = "\
{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Vincent\"}}\n\
{\"type\":\"node\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"Jules\"}}\n\
{\"type\":\"node\",\"labels\":[\"City\"],\"properties\":{\"name\":\"Amsterdam\"}}\n\
{\"type\":\"edge\",\"source\":0,\"target\":1,\"edge_type\":\"KNOWS\",\"properties\":{\"years\":5}}\n\
{\"type\":\"edge\",\"source\":0,\"target\":2,\"edge_type\":\"LIVES_IN\",\"properties\":{}}\n";
        std::fs::write(&input_path, content).unwrap();

        run(
            DataCommands::Load {
                input: input_path,
                path: db_path.clone(),
            },
            OutputFormat::Json,
            true,
        )
        .expect("load mixed content should succeed");

        let db = grafeo_engine::GrafeoDB::open(&db_path).unwrap();
        assert_eq!(db.info().node_count, 3);
        assert_eq!(db.info().edge_count, 2);
    }
}
