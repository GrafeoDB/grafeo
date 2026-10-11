//! Data export/import commands.

mod format;

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use grafeo_common::types::NodeId;
use grafeo_common::utils::hash::FxHashMap;
use grafeo_engine::{GrafeoDB, Session};

use self::format::Record;
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
        DataCommands::Load { input, path } => load_jsonl(&input, &path, quiet)?,
    }

    Ok(())
}

/// `grafeo data load`: adds the graph of a JSON Lines file (see
/// [`format`](mod@format)) to the database at `path`, which it creates when
/// there is none.
///
/// The load first reads the whole file without opening the database: every
/// record must read, values included, no two nodes may have one ID, and
/// every edge must join two nodes of the file (which may come after it).
/// Then it writes the file in one transaction: each node with its labels and
/// properties at once, then each edge between the nodes the file names. When
/// the database refuses a record (a constraint, the schema), the load rolls
/// back, so a load that fails leaves the database as it was.
///
/// Memory: the load reads one line at a time and holds a map from the file's
/// node IDs to the new ones (about 40 bytes per node with an ID). Until it
/// commits, its transaction also holds its own bookkeeping and the log
/// records of everything it wrote, which is about as much again as the
/// values it loads (measured in `load_holds_the_file_one_line_at_a_time`).
fn load_jsonl(input: &Path, path: &Path, quiet: bool) -> Result<()> {
    let checked = check_file(input)?;
    let db = if path.exists() {
        super::open_existing(path)?
    } else {
        GrafeoDB::open(path)
            .with_context(|| format!("Failed to create database at {}", path.display()))?
    };
    let (nodes, edges) = write_file(&db, input, &checked)?;
    output::status(
        &format!(
            "Loaded {nodes} nodes and {edges} edges into {}",
            path.display()
        ),
        quiet,
    );
    Ok(())
}

/// What [`check_file`] learned about a file.
struct CheckedFile {
    /// How many node records the file holds.
    nodes: usize,
    /// How many edge records the file holds.
    edges: usize,
    /// How many nodes have an ID: the size of the map the load builds.
    node_ids: usize,
    /// Whether every edge comes after every node, so that one pass can
    /// write both.
    nodes_first: bool,
}

/// An endpoint of an edge.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Endpoint {
    Source,
    Target,
}

impl Endpoint {
    /// The endpoint's field in an edge record.
    fn name(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Target => "target",
        }
    }
}

/// Reads every record of `input` and checks that the file can load: each
/// record reads, node IDs are unique, and each edge names nodes of the file.
fn check_file(input: &Path) -> Result<CheckedFile> {
    let metadata = std::fs::metadata(input)
        .with_context(|| format!("Failed to open input file: {}", input.display()))?;
    if !metadata.is_file() {
        bail!(
            "{} is not a file: a load reads its input twice, to check it and then to write it, \
             which a pipe or a directory does not allow",
            input.display()
        );
    }
    // The line of each node ID, and for each ID an edge names before any
    // node has it, the first line that names it.
    let mut node_lines: FxHashMap<u64, u64> = FxHashMap::default();
    let mut unresolved: FxHashMap<u64, (u64, Endpoint)> = FxHashMap::default();
    let mut nodes = 0_usize;
    let mut edges = 0_usize;
    let mut nodes_first = true;
    for_each_record(input, |line, record| {
        match record {
            Record::Node { id, .. } => {
                nodes += 1;
                nodes_first &= edges == 0;
                if let Some(id) = id {
                    if let Some(first) = node_lines.insert(id, line) {
                        bail!(
                            "line {line}: node id {id} is already the id of the node on line {first}"
                        );
                    }
                    unresolved.remove(&id);
                }
            }
            Record::Edge { source, target, .. } => {
                edges += 1;
                for (id, endpoint) in [(source, Endpoint::Source), (target, Endpoint::Target)] {
                    if !node_lines.contains_key(&id) {
                        unresolved.entry(id).or_insert((line, endpoint));
                    }
                }
            }
        }
        Ok(())
    })?;
    if let Some((id, (line, endpoint))) = unresolved
        .into_iter()
        .min_by_key(|(_, first_naming)| *first_naming)
    {
        bail!(
            "line {line}: edge {} {id} is not the id of a node in the file",
            endpoint.name()
        );
    }
    Ok(CheckedFile {
        nodes,
        edges,
        node_ids: node_lines.len(),
        nodes_first,
    })
}

/// Writes the records of a checked file to `db` in one transaction and
/// returns how many nodes and edges it created. On an error the transaction
/// rolls back.
fn write_file(db: &GrafeoDB, input: &Path, checked: &CheckedFile) -> Result<(usize, usize)> {
    let mut session = db.session();
    session
        .begin_transaction()
        .context("cannot start the transaction of the load")?;
    match write_records(&session, input, checked) {
        Ok(counts) => {
            session
                .commit()
                .map_err(|e| anyhow!("the load did not commit: {e}"))?;
            Ok(counts)
        }
        Err(error) => match session.rollback() {
            Ok(()) => Err(error),
            Err(rollback) => Err(anyhow!("{error}; the rollback failed too: {rollback}")),
        },
    }
}

/// Creates the nodes and edges of `input` in the session's transaction.
fn write_records(session: &Session, input: &Path, checked: &CheckedFile) -> Result<(usize, usize)> {
    // The new ID of each node by its ID in the file.
    let mut ids: FxHashMap<u64, NodeId> = FxHashMap::default();
    ids.reserve(checked.node_ids);
    let mut nodes = 0_usize;
    let mut edges = 0_usize;
    for_each_record(input, |line, record| match record {
        Record::Node {
            id,
            labels,
            properties,
        } => {
            let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
            let node = session
                .create_node_with_props(&labels, properties)
                .map_err(|e| anyhow!("line {line}: {e}"))?;
            if let Some(id) = id
                && ids.insert(id, node).is_some()
            {
                bail!("line {line}: node id {id} appears twice: the file changed during the load");
            }
            nodes += 1;
            Ok(())
        }
        Record::Edge { .. } if !checked.nodes_first => Ok(()),
        edge @ Record::Edge { .. } => {
            write_edge(session, &ids, line, edge)?;
            edges += 1;
            Ok(())
        }
    })?;
    if !checked.nodes_first {
        for_each_record(input, |line, record| {
            if matches!(record, Record::Edge { .. }) {
                write_edge(session, &ids, line, record)?;
                edges += 1;
            }
            Ok(())
        })?;
    }
    if (nodes, edges) != (checked.nodes, checked.edges) {
        bail!(
            "the file changed during the load: its check read {} nodes and {} edges, \
             and the load {nodes} nodes and {edges} edges",
            checked.nodes,
            checked.edges
        );
    }
    Ok((nodes, edges))
}

/// Creates the edge of `record` between the new nodes the file names.
fn write_edge(
    session: &Session,
    ids: &FxHashMap<u64, NodeId>,
    line: u64,
    record: Record,
) -> Result<()> {
    let Record::Edge {
        source,
        target,
        edge_type,
        properties,
    } = record
    else {
        return Ok(());
    };
    let node = |id: u64, endpoint: Endpoint| {
        ids.get(&id).copied().ok_or_else(|| {
            anyhow!(
                "line {line}: edge {} {id} is not the id of a node in the file",
                endpoint.name()
            )
        })
    };
    session
        .create_edge_with_props(
            node(source, Endpoint::Source)?,
            node(target, Endpoint::Target)?,
            &edge_type,
            properties,
        )
        .map_err(|e| anyhow!("line {line}: {e}"))?;
    Ok(())
}

/// Calls `each` with the number and record of every line of `input` that is
/// not blank, one line at a time.
fn for_each_record(input: &Path, mut each: impl FnMut(u64, Record) -> Result<()>) -> Result<()> {
    let file = std::fs::File::open(input)
        .with_context(|| format!("Failed to open input file: {}", input.display()))?;
    let mut reader = BufReader::new(file);
    let mut text = String::new();
    let mut line = 0_u64;
    loop {
        text.clear();
        line += 1;
        let read = reader
            .read_line(&mut text)
            .map_err(|e| anyhow!("line {line}: cannot read it: {e}"))?;
        if read == 0 {
            return Ok(());
        }
        // An editor may start the file with a byte order mark.
        let record = if line == 1 {
            text.trim_start_matches('\u{feff}')
        } else {
            &text
        }
        .trim();
        if record.is_empty() {
            continue;
        }
        let record =
            format::read_record(record).map_err(|reason| anyhow!("line {line}: {reason}"))?;
        each(line, record)?;
    }
}

fn dump_jsonl(path: &std::path::Path, out: &std::path::Path, quiet: bool) -> Result<()> {
    let db = super::open_existing(path)?;
    let file = std::fs::File::create(out)
        .with_context(|| format!("Failed to create output file: {}", out.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    let mut node_count = 0usize;
    for node in db.iter_nodes() {
        let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
        let properties = format::write_properties(&node.properties.to_btree_map())
            .map_err(|reason| anyhow!("node {}: {reason}", node.id.0))?;
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
        let properties = format::write_properties(&edge.properties.to_btree_map())
            .map_err(|reason| anyhow!("edge {}: {reason}", edge.id.0))?;
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
    use grafeo_common::types::Value;

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
    use grafeo_common::types::Value;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) -> grafeo_engine::GrafeoDB {
        let db = grafeo_engine::GrafeoDB::open(dir).expect("create db");
        let n1 = db.create_node(&["Person"]).unwrap();
        let n2 = db.create_node(&["Company"]).unwrap();
        db.set_node_property(n1, "name", Value::from("Alix"))
            .unwrap();
        db.set_node_property(n1, "age", Value::Int64(30)).unwrap();
        db.set_node_property(n2, "name", Value::from("Acme"))
            .unwrap();
        let e = db.create_edge(n1, n2, "WORKS_AT").unwrap();
        db.set_edge_property(e, "since", Value::Int64(2020))
            .unwrap();
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
        assert!(!content.is_empty(), "content is empty");
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
        assert!(
            err_msg.contains("line 1: unknown record type \"widget\""),
            "{err_msg}"
        );
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
            err_msg.contains("line 1: the record has no type"),
            "{err_msg}"
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
            db.create_node(&["Existing"]).unwrap();
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
        assert!(!content.is_empty(), "content is empty");
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
{\"type\":\"node\",\"id\":0,\"labels\":[\"Person\"],\"properties\":{\"name\":\"Vincent\"}}\n\
{\"type\":\"node\",\"id\":1,\"labels\":[\"Person\"],\"properties\":{\"name\":\"Jules\"}}\n\
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

        let (_, edges) = contents_at(&db_path);
        assert_eq!(
            edges,
            vec![(
                (
                    vec!["Person".to_string()],
                    vec![("name".to_string(), Value::from("Vincent"))]
                ),
                "KNOWS".to_string(),
                (
                    vec!["Person".to_string()],
                    vec![("name".to_string(), Value::from("Jules"))]
                ),
                vec![
                    ("close".to_string(), Value::Bool(true)),
                    ("since".to_string(), Value::Int64(1994)),
                ],
            )]
        );
    }

    #[test]
    fn test_load_node_with_null_property_value() {
        let temp = TempDir::new().unwrap();
        let input_path = temp.path().join("null_prop.jsonl");
        let db_path = temp.path().join("target.grafeo");

        // A null is no property: the node has its name only.
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

        assert_eq!(
            contents_at(&db_path).0,
            vec![(
                vec!["Person".to_string()],
                vec![("name".to_string(), Value::from("Mia"))]
            )]
        );
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
            db.create_node(&["Person", "Employee"]).unwrap();
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
{\"type\":\"node\",\"id\":0,\"labels\":[\"A\"],\"properties\":{}}\n\
{\"type\":\"node\",\"id\":1,\"labels\":[\"B\"],\"properties\":{}}\n\
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

        let (_, edges) = contents_at(&db_path);
        assert_eq!(
            edges,
            vec![(
                (vec!["A".to_string()], vec![]),
                "LINKS".to_string(),
                (vec!["B".to_string()], vec![]),
                vec![]
            )]
        );
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

        assert_eq!(
            contents_at(&db_path).0,
            vec![(
                vec!["Person".to_string()],
                vec![
                    ("active".to_string(), Value::Bool(true)),
                    ("age".to_string(), Value::Int64(42)),
                    ("height".to_string(), Value::Float64(1.85)),
                    ("name".to_string(), Value::from("Butch")),
                ]
            )]
        );
    }

    #[test]
    fn test_dump_preserves_edge_properties() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("edge_props.grafeo");
        let dump_path = temp.path().join("dump.jsonl");

        {
            let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
            let n1 = db.create_node(&["Person"]).unwrap();
            let n2 = db.create_node(&["Person"]).unwrap();
            db.set_node_property(n1, "name", Value::from("Django"))
                .unwrap();
            db.set_node_property(n2, "name", Value::from("Shosanna"))
                .unwrap();
            let edge = db.create_edge(n1, n2, "FRIENDS_WITH").unwrap();
            db.set_edge_property(edge, "year", Value::Int64(2012))
                .unwrap();
            db.set_edge_property(edge, "strong", Value::Bool(true))
                .unwrap();
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
            let node = db.create_node(&["City"]).unwrap();
            db.set_node_property(node, "name", Value::from("Amsterdam"))
                .unwrap();
            db.set_node_property(node, "population", Value::Int64(905_234))
                .unwrap();
            db.set_node_property(node, "capital", Value::Bool(true))
                .unwrap();
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
            let hans = db.create_node(&["Person"]).unwrap();
            let beatrix = db.create_node(&["Person"]).unwrap();
            let paris = db.create_node(&["City"]).unwrap();
            db.set_node_property(hans, "name", Value::from("Hans"))
                .unwrap();
            db.set_node_property(beatrix, "name", Value::from("Beatrix"))
                .unwrap();
            db.set_node_property(paris, "name", Value::from("Paris"))
                .unwrap();
            db.set_node_property(paris, "country", Value::from("France"))
                .unwrap();

            let edge1 = db.create_edge(hans, paris, "LIVES_IN").unwrap();
            db.set_edge_property(edge1, "since", Value::Int64(2015))
                .unwrap();
            let edge2 = db.create_edge(beatrix, paris, "VISITED").unwrap();
            db.set_edge_property(edge2, "year", Value::Int64(2023))
                .unwrap();
        }

        // Dump
        run(
            DataCommands::Dump {
                path: src_path.clone(),
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

        let expected = contents_at(&src_path);
        assert_eq!((expected.0.len(), expected.1.len()), (3, 2));
        assert_eq!(contents_at(&dst_path), expected);
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
{\"type\":\"node\",\"id\":0,\"labels\":[\"Person\"],\"properties\":{\"name\":\"Vincent\"}}\n\
{\"type\":\"node\",\"id\":1,\"labels\":[\"Person\"],\"properties\":{\"name\":\"Jules\"}}\n\
{\"type\":\"node\",\"id\":2,\"labels\":[\"City\"],\"properties\":{\"name\":\"Amsterdam\"}}\n\
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

        let named = |label: &str, name: &str| {
            (
                vec![label.to_string()],
                vec![("name".to_string(), Value::from(name))],
            )
        };
        let (nodes, edges) = contents_at(&db_path);
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            edges,
            vec![
                (
                    named("Person", "Vincent"),
                    "KNOWS".to_string(),
                    named("Person", "Jules"),
                    vec![("years".to_string(), Value::Int64(5))]
                ),
                (
                    named("Person", "Vincent"),
                    "LIVES_IN".to_string(),
                    named("City", "Amsterdam"),
                    vec![]
                ),
            ]
        );
    }

    // --- What a load must keep (#537) -------------------------------------------

    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    use std::sync::Arc;

    use grafeo_common::types::{Date, Duration, PropertyKey, Time, Timestamp, ZonedDatetime};
    use grafeo_engine::GrafeoDB;

    /// Runs `grafeo data load input db_path`.
    fn load(input: &Path, db_path: &Path) -> Result<()> {
        run(
            DataCommands::Load {
                input: input.to_path_buf(),
                path: db_path.to_path_buf(),
            },
            OutputFormat::Json,
            true,
        )
    }

    /// Runs `grafeo data dump db_path -o out`.
    fn dump(db_path: &Path, out: &Path) {
        run(
            DataCommands::Dump {
                path: db_path.to_path_buf(),
                output: out.to_path_buf(),
                export_format: None,
            },
            OutputFormat::Json,
            true,
        )
        .expect("dump");
    }

    /// A node as the tests compare it: its labels and its properties, both
    /// sorted. IDs are left out, since a load gives every node a new one.
    type NodeContent = (Vec<String>, Vec<(String, Value)>);

    /// An edge as the tests compare it: its source, type, target and
    /// properties.
    type EdgeContent = (NodeContent, String, NodeContent, Vec<(String, Value)>);

    /// The graph of `db`: every node and every edge by content, each list
    /// sorted, so two graphs compare equal when they hold the same nodes with
    /// the same edges between them, whatever their IDs.
    fn contents(db: &GrafeoDB) -> (Vec<NodeContent>, Vec<EdgeContent>) {
        let mut nodes = HashMap::new();
        for node in db.iter_nodes() {
            let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
            labels.sort();
            let properties = node
                .properties
                .to_btree_map()
                .into_iter()
                .map(|(key, value)| (key.as_str().to_string(), value))
                .collect();
            nodes.insert(node.id, (labels, properties));
        }
        let mut edges: Vec<EdgeContent> = db
            .iter_edges()
            .map(|edge| {
                let properties = edge
                    .properties
                    .to_btree_map()
                    .into_iter()
                    .map(|(key, value)| (key.as_str().to_string(), value))
                    .collect();
                (
                    nodes[&edge.src].clone(),
                    edge.edge_type.to_string(),
                    nodes[&edge.dst].clone(),
                    properties,
                )
            })
            .collect();
        let mut nodes: Vec<NodeContent> = nodes.into_values().collect();
        nodes.sort_by_key(|node| format!("{node:?}"));
        edges.sort_by_key(|edge| format!("{edge:?}"));
        (nodes, edges)
    }

    /// The contents of the database at `path`, opened and closed again.
    fn contents_at(path: &Path) -> (Vec<NodeContent>, Vec<EdgeContent>) {
        contents(&GrafeoDB::open(path).expect("open the database"))
    }

    fn map(entries: &[(&str, Value)]) -> Value {
        Value::Map(Arc::new(
            entries
                .iter()
                .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
                .collect::<BTreeMap<_, _>>(),
        ))
    }

    /// A property of every type a dump writes, with escaped and empty values.
    fn every_type() -> Vec<(&'static str, Value)> {
        vec![
            ("name", Value::from("Alix")),
            ("age", Value::Int64(88)),
            ("height", Value::Float64(1.88)),
            // A whole float stays a float: plain JSON would read `3.0` back as is,
            // an integer `3` as an integer.
            ("ratio", Value::Float64(3.0)),
            ("far", Value::Float64(f64::INFINITY)),
            ("near", Value::Float64(f64::NEG_INFINITY)),
            ("active", Value::Bool(true)),
            (
                "quote",
                Value::from("Mia said \"hi\",\n\ta \\ and caf\u{e9} \u{1F600}"),
            ),
            ("blank", Value::from("")),
            (
                "tags",
                Value::List(
                    vec![
                        Value::from("Paris"),
                        Value::Int64(3),
                        Value::Null,
                        Value::Float64(19.0),
                    ]
                    .into(),
                ),
            ),
            ("none", Value::List(Vec::<Value>::new().into())),
            (
                "address",
                map(&[
                    ("city", Value::from("Amsterdam")),
                    ("zip", Value::Int64(19)),
                    ("nested", map(&[("deep", Value::Bool(false))])),
                ]),
            ),
            ("nothing", map(&[])),
            (
                "born",
                Value::Date(Date::parse("1988-03-19").expect("a date")),
            ),
            (
                "wakes",
                Value::Time(Time::parse("08:19:03").expect("a time")),
            ),
            (
                "lunch",
                Value::Duration(Duration::parse("P3DT19H").expect("a duration")),
            ),
            (
                "seen",
                Value::Timestamp(Timestamp::from_micros(1_788_000_000_000_019)),
            ),
            (
                "meets",
                Value::ZonedDatetime(
                    ZonedDatetime::parse("2026-10-09T08:19:03+02:00").expect("a datetime"),
                ),
            ),
            ("photo", Value::Bytes(vec![0_u8, 3, 19, 88, 255].into())),
            (
                "embedding",
                Value::Vector(vec![0.1_f32, -3.5, 19.0, f32::INFINITY].into()),
            ),
            ("big", Value::Int64(i64::MAX)),
            ("small", Value::Int64(i64::MIN)),
        ]
    }

    /// Builds the source graph of the round trips at `path`. Its IDs do not
    /// start at 0 and have gaps (a node and an edge are deleted), one node has
    /// several labels and one none, labels, types and keys need escaping, and
    /// there are a self-loop, parallel edges and edges both ways.
    fn build_rich_graph(path: &Path) {
        let db = GrafeoDB::open(path).expect("create the source database");
        let gone = db.create_node(&["Gone"]).expect("a node to delete");
        let alix = db
            .create_node_with_props(&["Person", "Employee", "Caf\u{e9} \"Paris\""], every_type())
            .expect("Alix");
        let gus = db
            .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
            .expect("Gus");
        let gone_too = db.create_node(&["Gone"]).expect("another node to delete");
        let amsterdam = db
            .create_node_with_props(&["City"], [("name", Value::from("Amsterdam"))])
            .expect("Amsterdam");
        let prague = db
            .create_node_with_props(
                &[],
                [
                    ("name", Value::from("Prague")),
                    ("the \"key\"", Value::from("a\\b")),
                ],
            )
            .expect("Prague");
        let deleted_edge = db
            .create_edge(alix, amsterdam, "VISITED")
            .expect("an edge to delete");
        db.create_edge_with_props(
            alix,
            gus,
            "KNOWS",
            [
                ("since", Value::Int64(2019)),
                ("weight", Value::Float64(0.88)),
                (
                    "met",
                    Value::Date(Date::parse("2019-03-03").expect("a date")),
                ),
            ],
        )
        .expect("Alix knows Gus");
        db.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(3))])
            .expect("a parallel edge");
        db.create_edge(gus, alix, "KNOWS").expect("Gus knows Alix");
        db.create_edge(gus, amsterdam, "LIVES_IN")
            .expect("Gus lives in Amsterdam");
        db.create_edge_with_props(
            prague,
            prague,
            "TWINNED \"WITH\"",
            [("tags", Value::List(vec![Value::from("x")].into()))],
        )
        .expect("a self-loop");
        db.delete_edge(deleted_edge).expect("delete an edge");
        db.delete_node(gone).expect("delete a node");
        db.delete_node(gone_too).expect("delete another node");
    }

    /// A dump loaded into an empty database gives the same graph: every edge
    /// between the same nodes, every value with its type (#537).
    #[test]
    fn a_dump_loaded_into_an_empty_database_is_the_same_graph() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source.grafeo");
        let file = temp.path().join("dump.jsonl");
        let target = temp.path().join("target.grafeo");
        build_rich_graph(&source);
        dump(&source, &file);

        load(&file, &target).expect("load the dump");

        let expected = contents_at(&source);
        assert_eq!(expected.0.len(), 4, "the source graph has four nodes");
        assert_eq!(expected.1.len(), 5, "the source graph has five edges");
        assert_eq!(contents_at(&target), expected);
    }

    /// A dump loaded into a database with data adds its graph next to that
    /// data: no edge attaches to a node that was there before (#537).
    #[test]
    fn a_dump_loaded_into_a_database_with_data_adds_the_same_graph() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source.grafeo");
        let file = temp.path().join("dump.jsonl");
        let target = temp.path().join("target.grafeo");
        build_rich_graph(&source);
        dump(&source, &file);
        {
            let db = GrafeoDB::open(&target).expect("create the target");
            let vincent = db
                .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
                .unwrap();
            let jules = db
                .create_node_with_props(&["Person"], [("name", Value::from("Jules"))])
                .unwrap();
            db.create_edge(vincent, jules, "KNOWS").unwrap();
        }
        let before = contents_at(&target);

        load(&file, &target).expect("load the dump");

        let source_contents = contents_at(&source);
        let mut nodes = [before.0, source_contents.0].concat();
        let mut edges = [before.1, source_contents.1].concat();
        nodes.sort_by_key(|node| format!("{node:?}"));
        edges.sort_by_key(|edge| format!("{edge:?}"));
        assert_eq!(contents_at(&target), (nodes, edges));
    }

    /// The `id` of a node record is only a name in the file: IDs need not
    /// start at 0 or be dense, and edges name nodes by them (#537).
    #[test]
    fn edges_join_the_nodes_the_file_names_whatever_their_ids() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("people.jsonl");
        let target = temp.path().join("target.grafeo");
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"node","id":88,"labels":["Person"],"properties":{"name":"Mia"}}"#,
                "\n",
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Vincent"}}"#,
                "\n",
                r#"{"type":"node","id":19,"labels":["City"],"properties":{"name":"Berlin"}}"#,
                "\n",
                r#"{"type":"edge","source":3,"target":88,"edge_type":"KNOWS","properties":{}}"#,
                "\n",
                r#"{"type":"edge","source":88,"target":19,"edge_type":"LIVES_IN"}"#,
                "\n",
            ),
        )
        .unwrap();

        load(&file, &target).expect("load");

        let (_, edges) = contents_at(&target);
        let wiring: Vec<(String, String, String)> = edges
            .into_iter()
            .map(|(source, edge_type, target, _)| {
                (
                    format!("{:?}", source.1),
                    edge_type,
                    format!("{:?}", target.1),
                )
            })
            .collect();
        assert_eq!(
            wiring,
            vec![
                (
                    format!("{:?}", vec![("name".to_string(), Value::from("Mia"))]),
                    "LIVES_IN".to_string(),
                    format!("{:?}", vec![("name".to_string(), Value::from("Berlin"))]),
                ),
                (
                    format!("{:?}", vec![("name".to_string(), Value::from("Vincent"))]),
                    "KNOWS".to_string(),
                    format!("{:?}", vec![("name".to_string(), Value::from("Mia"))]),
                ),
            ]
        );
    }

    /// An edge may come before the nodes it joins.
    #[test]
    fn an_edge_may_come_before_its_nodes() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("edges_first.jsonl");
        let target = temp.path().join("target.grafeo");
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"edge","source":19,"target":3,"edge_type":"KNOWS","properties":{"since":2019}}"#,
                "\n",
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Jules"}}"#,
                "\n",
                r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Butch"}}"#,
                "\n",
            ),
        )
        .unwrap();

        load(&file, &target).expect("load");

        let butch = (
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::from("Butch"))],
        );
        let jules = (
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::from("Jules"))],
        );
        assert_eq!(
            contents_at(&target),
            (
                vec![butch.clone(), jules.clone()],
                vec![(
                    butch,
                    "KNOWS".to_string(),
                    jules,
                    vec![("since".to_string(), Value::Int64(2019))]
                )]
            )
        );
    }

    /// Plain JSON values keep their values and types: a hand-written file
    /// loads what it says, and a null is no property (#537).
    #[test]
    fn plain_json_values_keep_their_values_and_types() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("plain.jsonl");
        let target = temp.path().join("target.grafeo");
        let backslash = '\\';
        let line = format!(
            r#"{{"type":"node","id":3,"labels":["Person"],"properties":{{"name":"Gus","age":19,"height":1.88,"ratio":3.0,"negative":-88,"active":false,"nickname":null,"quote":"Mia said \"hi\",\n\ta \\ and caf{backslash}u00e9","blank":"","tags":["Paris",3,true,null],"none":[],"address":{{"city":"Prague","zip":88}},"nothing":{{}}}}}}"#
        );
        std::fs::write(&file, format!("{line}\n")).unwrap();

        load(&file, &target).expect("load");

        let (nodes, _) = contents_at(&target);
        let expected: Vec<(String, Value)> = vec![
            ("active".to_string(), Value::Bool(false)),
            (
                "address".to_string(),
                map(&[("city", Value::from("Prague")), ("zip", Value::Int64(88))]),
            ),
            ("age".to_string(), Value::Int64(19)),
            ("blank".to_string(), Value::from("")),
            ("height".to_string(), Value::Float64(1.88)),
            ("name".to_string(), Value::from("Gus")),
            ("negative".to_string(), Value::Int64(-88)),
            ("none".to_string(), Value::List(Vec::<Value>::new().into())),
            ("nothing".to_string(), map(&[])),
            (
                "quote".to_string(),
                Value::from("Mia said \"hi\",\n\ta \\ and caf\u{e9}"),
            ),
            ("ratio".to_string(), Value::Float64(3.0)),
            (
                "tags".to_string(),
                Value::List(
                    vec![
                        Value::from("Paris"),
                        Value::Int64(3),
                        Value::Bool(true),
                        Value::Null,
                    ]
                    .into(),
                ),
            ),
        ];
        assert_eq!(nodes, vec![(vec!["Person".to_string()], expected)]);
    }

    /// A dump of 0.5.44, whose IDs start at 1, loads with its wiring and
    /// types, a null inside a list included.
    #[test]
    fn a_dump_written_by_0_5_44_loads_with_its_wiring_and_types() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("dump-0.5.44.jsonl");
        let target = temp.path().join("target.grafeo");
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"node","id":1,"labels":["Person"],"properties":{"name":{"String":"Alix"},"tags":{"List":["Null",{"Int64":3}]}}}"#,
                "\n",
                r#"{"type":"node","id":2,"labels":["Person"],"properties":{"name":{"String":"Gus"}}}"#,
                "\n",
                r#"{"type":"node","id":3,"labels":["City"],"properties":{"name":{"String":"Amsterdam"}}}"#,
                "\n",
                r#"{"type":"edge","id":0,"source":1,"target":2,"edge_type":"KNOWS","properties":{"since":{"Int64":2019}}}"#,
                "\n",
                r#"{"type":"edge","id":1,"source":2,"target":3,"edge_type":"LIVES_IN","properties":{}}"#,
                "\n",
            ),
        )
        .unwrap();

        load(&file, &target).expect("load");

        let alix = (
            vec!["Person".to_string()],
            vec![
                ("name".to_string(), Value::from("Alix")),
                (
                    "tags".to_string(),
                    Value::List(vec![Value::Null, Value::Int64(3)].into()),
                ),
            ],
        );
        let gus = (
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::from("Gus"))],
        );
        let amsterdam = (
            vec!["City".to_string()],
            vec![("name".to_string(), Value::from("Amsterdam"))],
        );
        let (_, edges) = contents_at(&target);
        assert_eq!(
            edges,
            vec![
                (
                    alix,
                    "KNOWS".to_string(),
                    gus.clone(),
                    vec![("since".to_string(), Value::Int64(2019))]
                ),
                (gus, "LIVES_IN".to_string(), amsterdam, vec![]),
            ]
        );
    }

    /// NaN and infinities survive a round trip, at the top level, in a list
    /// and in a vector (a dump wrote them as null before).
    #[test]
    fn nan_and_infinities_survive_a_round_trip() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source.grafeo");
        let file = temp.path().join("dump.jsonl");
        let target = temp.path().join("target.grafeo");
        {
            let db = GrafeoDB::open(&source).unwrap();
            db.create_node_with_props(
                &["Reading"],
                [
                    ("nan", Value::Float64(f64::NAN)),
                    ("far", Value::Float64(f64::INFINITY)),
                    (
                        "list",
                        Value::List(
                            vec![Value::Float64(f64::NAN), Value::Float64(f64::NEG_INFINITY)]
                                .into(),
                        ),
                    ),
                    (
                        "vector",
                        Value::Vector(vec![f32::NAN, f32::NEG_INFINITY, 3.0].into()),
                    ),
                ],
            )
            .unwrap();
        }
        dump(&source, &file);

        load(&file, &target).expect("load");

        let db = GrafeoDB::open(&target).unwrap();
        let node = db.iter_nodes().next().expect("the node");
        let get = |key: &str| {
            node.properties
                .get(&PropertyKey::new(key))
                .cloned()
                .unwrap_or_else(|| panic!("property {key} was not loaded"))
        };
        assert!(
            matches!(get("nan"), Value::Float64(f) if f.is_nan()),
            "nan: {:?}",
            get("nan")
        );
        assert_eq!(get("far"), Value::Float64(f64::INFINITY));
        let Value::List(list) = get("list") else {
            panic!("list: {:?}", get("list"))
        };
        assert!(
            matches!(list[0], Value::Float64(f) if f.is_nan()),
            "{list:?}"
        );
        assert_eq!(list[1], Value::Float64(f64::NEG_INFINITY));
        let Value::Vector(vector) = get("vector") else {
            panic!("vector: {:?}", get("vector"))
        };
        assert!(vector[0].is_nan(), "{vector:?}");
        assert_eq!(&vector[1..], &[f32::NEG_INFINITY, 3.0]);
    }

    /// Writes `content` to a file and loads it into a database that holds one
    /// node already; returns the error and checks that the load left the
    /// database as it was.
    fn load_fails_and_changes_nothing(content: &str) -> String {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("input.jsonl");
        let target = temp.path().join("target.grafeo");
        std::fs::write(&file, content).unwrap();
        {
            let db = GrafeoDB::open(&target).unwrap();
            db.create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
                .unwrap();
        }
        let before = contents_at(&target);

        let error = load(&file, &target).expect_err("the load fails");

        assert_eq!(
            contents_at(&target),
            before,
            "a failed load leaves the database as it was ({error})"
        );
        error.to_string()
    }

    /// An edge to a node the file does not hold is an error naming its line,
    /// not an edge to some other node.
    #[test]
    fn an_edge_to_a_node_not_in_the_file_fails_naming_its_line() {
        let error = load_fails_and_changes_nothing(concat!(
            r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
            "\n",
            r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules"}}"#,
            "\n",
            r#"{"type":"edge","source":3,"target":19,"edge_type":"KNOWS"}"#,
            "\n",
            r#"{"type":"edge","source":3,"target":0,"edge_type":"KNOWS"}"#,
            "\n",
        ));
        assert!(
            error.contains("line 4") && error.contains("target 0"),
            "{error}"
        );
    }

    /// Two node records with one ID are an error naming both lines.
    #[test]
    fn a_duplicate_node_id_fails_naming_both_lines() {
        let error = load_fails_and_changes_nothing(concat!(
            r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
            "\n",
            r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules"}}"#,
            "\n",
            r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Butch"}}"#,
            "\n",
        ));
        assert!(
            error.contains("line 3") && error.contains("line 1") && error.contains("id 3"),
            "{error}"
        );
    }

    /// A value the load cannot read is an error naming its line and
    /// property, not a null.
    #[test]
    fn a_value_that_cannot_be_read_fails_naming_its_line_and_property() {
        for (value, what) in [
            (r#"{"Int64":"nineteen"}"#, "a tagged integer that is text"),
            (r#"{"Float64":null}"#, "a float a dump before 0.6.0 lost"),
            (r#"{"Date":"someday"}"#, "a tagged date that is not one"),
            ("18446744073709551615", "an integer beyond INT64"),
            (r#"{"List":[3]}"#, "an untagged value in a tagged list"),
        ] {
            let error = load_fails_and_changes_nothing(&format!(
                "{}\n{}\n",
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
                format_args!(
                    r#"{{"type":"node","id":19,"labels":["Person"],"properties":{{"name":"Jules","age":{value}}}}}"#
                ),
            ));
            assert!(
                error.contains("line 2") && error.contains("age"),
                "{what}: {error}"
            );
        }
    }

    /// A record the load cannot read is an error naming its line, and the
    /// database stays as it was.
    #[test]
    fn a_malformed_record_fails_naming_its_line() {
        for (record, what) in [
            ("this is not json", "a line that is not JSON"),
            (
                r#"{"type":"node","labels":"Person"}"#,
                "labels that are not a list",
            ),
            (
                r#"{"type":"node","labels":["Person",3]}"#,
                "a label that is not text",
            ),
            (
                r#"{"type":"node","properties":["name"]}"#,
                "properties that are not an object",
            ),
            (r#"{"type":"node","id":-3}"#, "a negative id"),
            (
                r#"{"type":"edge","source":"3","target":19,"edge_type":"KNOWS"}"#,
                "a source that is not an id",
            ),
            (r#"{"type":"edge","source":3,"target":19}"#, "no edge type"),
            (r#"{"type":"widget"}"#, "an unknown record type"),
        ] {
            let error = load_fails_and_changes_nothing(&format!(
                "{}\n{}\n{record}\n",
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
                r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules"}}"#,
            ));
            assert!(error.contains("line 3"), "{what}: {error}");
        }
    }

    /// A load that fails on a file it cannot read creates no database.
    #[test]
    fn a_failed_load_into_a_new_path_creates_no_database() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("input.jsonl");
        let target = temp.path().join("target.grafeo");
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
                "\n",
                r#"{"type":"edge","source":3,"target":88,"edge_type":"KNOWS"}"#,
                "\n",
            ),
        )
        .unwrap();

        load(&file, &target).expect_err("the edge names no node of the file");

        assert!(!target.exists(), "the failed load created {target:?}");
    }

    /// The load is one transaction: a record the database refuses halfway
    /// through (here a NODE KEY duplicate) leaves the database as it was, so
    /// running the load again after a fix adds the file once.
    #[test]
    fn a_record_the_database_refuses_leaves_the_database_as_it_was() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("input.jsonl");
        let target = temp.path().join("target.grafeo");
        {
            let db = GrafeoDB::open(&target).unwrap();
            db.execute("CREATE CONSTRAINT person_key FOR (n:Person) ON (n.name, n.city) NODE KEY")
                .expect("the constraint");
            db.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::from("Vincent")),
                    ("city", Value::from("Amsterdam")),
                ],
            )
            .unwrap();
        }
        let before = contents_at(&target);
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia","city":"Paris"}}"#,
                "\n",
                r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules","city":"Paris"}}"#,
                "\n",
                r#"{"type":"edge","source":3,"target":19,"edge_type":"KNOWS"}"#,
                "\n",
                r#"{"type":"node","id":88,"labels":["Person"],"properties":{"name":"Mia","city":"Paris"}}"#,
                "\n",
            ),
        )
        .unwrap();

        let error = load(&file, &target).expect_err("a NODE KEY duplicate");

        assert!(error.to_string().contains("line 4"), "{error}");
        assert_eq!(contents_at(&target), before, "the load changed nothing");
    }

    /// A database with a schema takes a load that meets it: each node and
    /// edge is created with its properties at once, so NOT NULL, NODE KEY and
    /// a required edge property hold at every step (#537).
    #[test]
    fn a_database_with_a_schema_takes_a_load_that_meets_it() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("input.jsonl");
        let target = temp.path().join("target.grafeo");
        {
            let db = GrafeoDB::open(&target).unwrap();
            for statement in [
                "CREATE NODE TYPE Person (name STRING NOT NULL, city STRING)",
                "CREATE CONSTRAINT person_key FOR (n:Person) ON (n.name, n.city) NODE KEY",
                "CREATE EDGE TYPE KNOWS (since INT64 NOT NULL)",
            ] {
                db.execute(statement).expect(statement);
            }
        }
        std::fs::write(
            &file,
            concat!(
                r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":{"String":"Mia"},"city":"Paris"}}"#,
                "\n",
                r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules","city":"Berlin"}}"#,
                "\n",
                r#"{"type":"edge","source":3,"target":19,"edge_type":"KNOWS","properties":{"since":{"Int64":2019}}}"#,
                "\n",
            ),
        )
        .unwrap();

        load(&file, &target).expect("the file meets the schema");

        let mia = (
            vec!["Person".to_string()],
            vec![
                ("city".to_string(), Value::from("Paris")),
                ("name".to_string(), Value::from("Mia")),
            ],
        );
        let jules = (
            vec!["Person".to_string()],
            vec![
                ("city".to_string(), Value::from("Berlin")),
                ("name".to_string(), Value::from("Jules")),
            ],
        );
        assert_eq!(
            contents_at(&target),
            (
                vec![jules.clone(), mia.clone()],
                vec![(
                    mia,
                    "KNOWS".to_string(),
                    jules,
                    vec![("since".to_string(), Value::Int64(2019))]
                )]
            )
        );
    }

    /// A file saved with a byte order mark and Windows line endings loads.
    #[test]
    fn a_file_with_a_byte_order_mark_and_crlf_line_endings_loads() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("windows.jsonl");
        let target = temp.path().join("target.grafeo");
        let mut content = String::from('\u{feff}');
        content
            .push_str(r#"{"type":"node","id":3,"labels":["City"],"properties":{"name":"Prague"}}"#);
        content.push_str("\r\n\r\n");
        content.push_str(r#"{"type":"edge","source":3,"target":3,"edge_type":"TWINNED"}"#);
        content.push_str("\r\n");
        std::fs::write(&file, content).unwrap();

        load(&file, &target).expect("load");

        let prague = (
            vec!["City".to_string()],
            vec![("name".to_string(), Value::from("Prague"))],
        );
        assert_eq!(
            contents_at(&target),
            (
                vec![prague.clone()],
                vec![(prague.clone(), "TWINNED".to_string(), prague, vec![])]
            )
        );
    }

    /// A load reads its input twice (to check it, then to write it), so it
    /// takes a file: a second read of a pipe would find nothing.
    #[test]
    fn a_load_takes_a_file_and_not_a_directory_or_a_pipe() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("target.grafeo");

        let error = load(temp.path(), &target).expect_err("a directory is no file");

        assert!(error.to_string().contains("is not a file"), "{error}");
        assert!(!target.exists(), "the failed load created {target:?}");
    }

    /// A file that loses records between the check and the write, as a pipe
    /// read twice would, fails the load and loads nothing of it.
    #[test]
    fn a_file_that_changes_during_the_load_loads_nothing() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("input.jsonl");
        let nodes = concat!(
            r#"{"type":"node","id":3,"labels":["Person"],"properties":{"name":"Mia"}}"#,
            "\n",
            r#"{"type":"node","id":19,"labels":["Person"],"properties":{"name":"Jules"}}"#,
            "\n",
        );
        std::fs::write(
            &file,
            format!(
                "{nodes}{}\n",
                r#"{"type":"edge","source":3,"target":19,"edge_type":"KNOWS"}"#
            ),
        )
        .unwrap();
        let db = GrafeoDB::new_in_memory();
        let checked = check_file(&file).unwrap();
        std::fs::write(&file, nodes).unwrap();

        let error = write_file(&db, &file, &checked).expect_err("the edge is gone");

        assert!(error.to_string().contains("changed"), "{error}");
        assert_eq!(db.node_count(), 0, "the nodes it wrote rolled back");
    }

    // --- Memory -----------------------------------------------------------------

    /// Counts the bytes each thread holds through the global allocator and
    /// the most it held since [`reset_peak`](counting::reset_peak), so that a
    /// test measures its own thread while the other tests run. Every call
    /// goes to the system allocator unchanged.
    #[expect(
        unsafe_code,
        reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
    )]
    mod counting {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static HELD: Cell<isize> = const { Cell::new(0) };
            static PEAK: Cell<isize> = const { Cell::new(0) };
        }

        /// Forwards every call to [`System`] and counts what it hands out.
        pub struct Counting;

        #[global_allocator]
        static COUNTING: Counting = Counting;

        fn grew(by: usize) {
            let by = isize::try_from(by).unwrap_or(isize::MAX);
            HELD.with(|held| {
                let now = held.get().saturating_add(by);
                held.set(now);
                PEAK.with(|peak| peak.set(peak.get().max(now)));
            });
        }

        fn shrank(by: usize) {
            let by = isize::try_from(by).unwrap_or(isize::MAX);
            HELD.with(|held| held.set(held.get().saturating_sub(by)));
        }

        // SAFETY: every method passes its arguments unchanged to `System`,
        // which meets the trait's contract, and returns what `System`
        // returned. The counting touches thread-locals without destructors
        // only and allocates nothing.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
                let block = unsafe { System.alloc(layout) };
                if !block.is_null() {
                    grew(layout.size());
                }
                block
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                // SAFETY: as in `alloc`.
                let block = unsafe { System.alloc_zeroed(layout) };
                if !block.is_null() {
                    grew(layout.size());
                }
                block
            }

            unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
                // SAFETY: the caller meets `dealloc`'s contract, which is `System`'s.
                unsafe { System.dealloc(block, layout) };
                shrank(layout.size());
            }

            unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                // SAFETY: the caller meets `realloc`'s contract, which is `System`'s.
                let moved = unsafe { System.realloc(block, layout, new_size) };
                if !moved.is_null() {
                    grew(new_size);
                    shrank(layout.size());
                }
                moved
            }
        }

        /// The bytes this thread holds now.
        pub fn held() -> isize {
            HELD.with(Cell::get)
        }

        /// Starts a new peak at what this thread holds now.
        pub fn reset_peak() {
            PEAK.with(|peak| peak.set(held()));
        }

        /// The most this thread held since the last [`reset_peak`].
        pub fn peak() -> isize {
            PEAK.with(Cell::get)
        }
    }

    /// The bytes of text each person of a test file carries: well above what
    /// a transaction keeps for a node and an edge until it commits.
    const NOTES: usize = 4096;

    /// Writes a file of `people` nodes, each with [`NOTES`] bytes of text,
    /// and as many edges.
    fn write_people(path: &Path, people: u64) {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        let notes = "x".repeat(NOTES);
        for id in 0..people {
            writeln!(
                out,
                r#"{{"type":"node","id":{id},"labels":["Person"],"properties":{{"name":"Mia {id}","notes":"{notes}"}}}}"#
            )
            .unwrap();
        }
        for id in 0..people {
            let next = (id + 1) % people;
            writeln!(
                out,
                r#"{{"type":"edge","source":{id},"target":{next},"edge_type":"KNOWS","properties":{{"since":2019}}}}"#
            )
            .unwrap();
        }
        out.flush().unwrap();
    }

    /// What the check and the write of a file of `people` hold at their
    /// peak on this thread, beyond what they leave held: (check, write).
    ///
    /// The write goes to a database without a WAL, so it measures what the
    /// load holds and not the log records its transaction buffers until it
    /// commits (see the size note on [`load_jsonl`]).
    fn load_peaks(dir: &Path, people: u64) -> (isize, isize) {
        let file = dir.join(format!("people-{people}.jsonl"));
        write_people(&file, people);
        let db = GrafeoDB::new_in_memory();

        let before = counting::held();
        counting::reset_peak();
        let checked = check_file(&file).unwrap();
        let check = counting::peak() - before;

        counting::reset_peak();
        let (nodes, edges) = write_file(&db, &file, &checked).unwrap();
        let write = counting::peak() - counting::held();
        let people = usize::try_from(people).unwrap();
        assert_eq!((nodes, edges), (people, people), "the load is complete");
        (check, write)
    }

    /// A load reads its file one line at a time: what it holds beyond the
    /// database grows with the number of nodes (the map of their IDs), not
    /// with the size of the file.
    ///
    /// Each person of the file is a node with 4 KiB of text and an edge, so a
    /// load that held the file, or its records, would grow by more than
    /// 4 KiB a person. Measured on Windows from 1,000 to 10,000 people, the
    /// check's peak grew by 41 bytes a person and the write's by 550, or by
    /// 1,050 in a build with the engine's `temporal` feature: what the
    /// transaction keeps of each write until it commits, whatever the size
    /// of the text (the same with 1 KiB of it). With 1 KiB of text, a load of
    /// the same file into a database on disk held about 1.4 KiB a person
    /// more until it committed, for the log records of its transaction.
    #[test]
    fn load_holds_the_file_one_line_at_a_time() {
        const PEOPLE: (u64, u64) = (1_000, 10_000);
        let temp = TempDir::new().unwrap();
        let small = load_peaks(temp.path(), PEOPLE.0);
        let large = load_peaks(temp.path(), PEOPLE.1);
        let added = isize::try_from(PEOPLE.1 - PEOPLE.0).unwrap();

        let check_growth = large.0 - small.0;
        assert!(
            check_growth <= added * 64 + 64 * 1024,
            "the check grew by {check_growth} bytes for {added} more people: \
             more than a map entry each ({small:?} to {large:?})"
        );
        let write_growth = large.1 - small.1;
        assert!(
            write_growth < added * isize::try_from(NOTES).unwrap(),
            "the write grew by {write_growth} bytes for {added} more people: \
             as much as their text ({small:?} to {large:?})"
        );
    }
}
