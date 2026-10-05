//! Bulk graph import from TSV edge lists and Matrix Market files.
//!
//! These importers bypass per-edge transaction overhead by batching all
//! operations into a single transaction. This is 10-100x faster than calling
//! `create_node`/`create_edge` in a loop for large graphs.
//!
//! An import reads and parses its input first, without blocking anything.
//! While it then changes the store, commits, new transactions, writes outside
//! a transaction and checkpoints wait for it: a checkpoint or `close()` holds
//! all of the import or none of it. Reads outside a transaction go on.
//!
//! # Supported Formats
//!
//! | Format | Extension | Description |
//! | ------ | --------- | ----------- |
//! | TSV | `.tsv`, `.txt`, `.edges` | Tab or space-separated edge list |
//! | MMIO | `.mtx` | Matrix Market coordinate format |
//!
//! # Example
//!
//! ```no_run
//! use grafeo_engine::GrafeoDB;
//!
//! let db = GrafeoDB::new_in_memory();
//! let (nodes, edges) = db.import_tsv("graph.tsv", "EDGE", true).unwrap();
//! println!("Loaded {} nodes, {} edges", nodes, edges);
//! ```

use std::io::{BufRead, BufReader};
use std::path::Path;

use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;

impl super::GrafeoDB {
    /// Bulk-imports a graph from a TSV/space-separated edge list into the LPG store.
    ///
    /// Each line should contain two integer IDs separated by whitespace (tab or space):
    /// `src_id dst_id` with an optional third column for edge weight.
    /// Lines starting with `#` or `%` are treated as comments and skipped.
    /// Empty lines are also skipped.
    ///
    /// Nodes are created on-demand as new external IDs are encountered.
    /// All nodes get the label `"_Imported"` and all edges get the given `edge_type`.
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; the file is read before, without blocking anything.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TSV file.
    /// * `edge_type` - Edge type label for all imported edges.
    /// * `directed` - If `true`, create one directed edge per line.
    ///   If `false`, create edges in both directions.
    ///
    /// # Returns
    ///
    /// `(node_count, edge_count)` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or contains malformed
    /// lines; the read-only error on a read-only database, the
    /// database-closed error after `close()` of a persistent database
    /// (read-only or not), and the incomplete-commit error after a commit
    /// that did not complete.
    pub fn import_tsv(
        &self,
        path: impl AsRef<Path>,
        edge_type: &str,
        directed: bool,
    ) -> Result<(usize, usize)> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Internal(format!("failed to open {}: {}", path.display(), e)))?;

        let reader = BufReader::new(file);
        let edges = parse_edge_list(reader)?;

        self.import_edge_list(&edges, edge_type, directed)
    }

    /// Bulk-imports from a string containing TSV edge list data.
    ///
    /// Same format as [`import_tsv`](Self::import_tsv) but reads from a string
    /// instead of a file. Useful for tests and embedded data.
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; the string is read before, without blocking anything.
    ///
    /// # Errors
    ///
    /// Returns an error if the data contains malformed lines; the read-only
    /// error on a read-only database, the database-closed error after
    /// `close()` of a persistent database (read-only or not), and the
    /// incomplete-commit error after a commit that did not complete.
    pub fn import_tsv_str(
        &self,
        data: &str,
        edge_type: &str,
        directed: bool,
    ) -> Result<(usize, usize)> {
        let reader = BufReader::new(data.as_bytes());
        let edges = parse_edge_list(reader)?;
        self.import_edge_list(&edges, edge_type, directed)
    }

    /// Bulk-imports from a Matrix Market (MMIO) coordinate format file.
    ///
    /// Handles the standard MMIO header:
    /// ```text
    /// %%MatrixMarket matrix coordinate real general
    /// % comment
    /// rows cols nnz
    /// row col [value]
    /// ```
    ///
    /// Symmetric matrices automatically create edges in both directions.
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; the file is read before, without blocking anything.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.mtx` file.
    /// * `edge_type` - Edge type label for all imported edges.
    ///
    /// # Returns
    ///
    /// `(node_count, edge_count)` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or has an invalid MMIO
    /// header or data; the read-only error on a read-only database, the
    /// database-closed error after `close()` of a persistent database
    /// (read-only or not), and the incomplete-commit error after a commit
    /// that did not complete.
    pub fn import_mmio(&self, path: impl AsRef<Path>, edge_type: &str) -> Result<(usize, usize)> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Internal(format!("failed to open {}: {}", path.display(), e)))?;

        let reader = BufReader::new(file);
        let (edges, symmetric) = parse_mmio(reader)?;
        self.import_edge_list(&edges, edge_type, !symmetric)
    }

    /// Bulk-imports a pre-parsed edge list into the LPG store.
    fn import_edge_list(
        &self,
        edges: &[(u64, u64)],
        edge_type: &str,
        directed: bool,
    ) -> Result<(usize, usize)> {
        let _held = self.hold_commits_for_import()?;
        let store = self.lpg_store();

        // Phase 1: Collect unique external IDs and create nodes.
        let mut ext_to_int: FxHashMap<u64, NodeId> = FxHashMap::default();

        for &(src, dst) in edges {
            if !ext_to_int.contains_key(&src) {
                let id = store.create_node(&["_Imported"]);
                ext_to_int.insert(src, id);
            }
            if !ext_to_int.contains_key(&dst) {
                let id = store.create_node(&["_Imported"]);
                ext_to_int.insert(dst, id);
            }
        }

        // Phase 2: Create edges in batch.
        let mut batch: Vec<(NodeId, NodeId, &str)> = Vec::with_capacity(if directed {
            edges.len()
        } else {
            edges.len() * 2
        });

        for &(src, dst) in edges {
            let src_id = ext_to_int[&src];
            let dst_id = ext_to_int[&dst];
            batch.push((src_id, dst_id, edge_type));
            if !directed {
                batch.push((dst_id, src_id, edge_type));
            }
        }

        store.batch_create_edges(&batch);

        let node_count = ext_to_int.len();
        let edge_count = batch.len();

        // Refresh statistics so the optimizer has fresh data.
        store.ensure_statistics_fresh();

        Ok((node_count, edge_count))
    }

    /// Bulk-imports a TSV edge list into the RDF store.
    ///
    /// Each edge `(src, dst)` becomes a triple:
    /// `<{base_uri}{src}> <{predicate_uri}> <{base_uri}{dst}>`
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; the file is read before, without blocking anything.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TSV file.
    /// * `predicate_uri` - Full IRI for the edge predicate (e.g., `"http://example.org/connects"`).
    /// * `base_uri` - Base IRI prefix for node identifiers (e.g., `"http://example.org/node/"`).
    ///
    /// # Returns
    ///
    /// `(node_count, edge_count)` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or contains malformed
    /// lines; the read-only error on a read-only database, the
    /// database-closed error after `close()` of a persistent database
    /// (read-only or not), and the incomplete-commit error after a commit
    /// that did not complete.
    #[cfg(feature = "triple-store")]
    pub fn import_tsv_rdf(
        &self,
        path: impl AsRef<Path>,
        predicate_uri: &str,
        base_uri: &str,
    ) -> Result<(usize, usize)> {
        use grafeo_core::graph::rdf::{Term, Triple};

        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Internal(format!("failed to open {}: {}", path.display(), e)))?;

        let reader = BufReader::new(file);
        let edges = parse_edge_list(reader)?;

        let predicate = Term::iri(predicate_uri);
        let mut unique_nodes = grafeo_common::utils::hash::FxHashSet::default();

        let triples: Vec<Triple> = edges
            .iter()
            .map(|&(src, dst)| {
                unique_nodes.insert(src);
                unique_nodes.insert(dst);
                Triple::new(
                    Term::iri(format!("{base_uri}{src}")),
                    predicate.clone(),
                    Term::iri(format!("{base_uri}{dst}")),
                )
            })
            .collect();

        let _held = self.hold_commits_for_import()?;
        let edge_count = self.rdf_store.batch_insert(triples);

        Ok((unique_nodes.len(), edge_count))
    }
}

/// Parses a TSV/space-separated edge list from a reader.
///
/// Each non-comment, non-empty line should contain at least two whitespace-separated
/// integer IDs. Additional columns (e.g., weights) are ignored.
fn parse_edge_list(reader: impl BufRead) -> Result<Vec<(u64, u64)>> {
    let mut edges = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line
            .map_err(|e| Error::Internal(format!("read error at line {}: {}", line_num + 1, e)))?;
        let trimmed = line.trim();

        // Skip comments and empty lines.
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let src_str = parts
            .next()
            .ok_or_else(|| Error::Internal(format!("line {}: missing source ID", line_num + 1)))?;
        let dst_str = parts
            .next()
            .ok_or_else(|| Error::Internal(format!("line {}: missing target ID", line_num + 1)))?;

        let src: u64 = src_str.parse().map_err(|_| {
            Error::Internal(format!(
                "line {}: invalid source ID '{}'",
                line_num + 1,
                src_str
            ))
        })?;
        let dst: u64 = dst_str.parse().map_err(|_| {
            Error::Internal(format!(
                "line {}: invalid target ID '{}'",
                line_num + 1,
                dst_str
            ))
        })?;

        edges.push((src, dst));
    }

    Ok(edges)
}

/// Parses a Matrix Market coordinate format file.
///
/// Returns the edge list and whether the matrix is symmetric.
fn parse_mmio(reader: impl BufRead) -> Result<(Vec<(u64, u64)>, bool)> {
    let mut lines = reader.lines();
    let mut symmetric = false;

    // Parse header line.
    let header = lines
        .next()
        .ok_or_else(|| Error::Internal("empty MMIO file".into()))?
        .map_err(|e| Error::Internal(format!("MMIO header read error: {e}")))?;

    if !header.starts_with("%%MatrixMarket") {
        return Err(Error::Internal(
            "invalid MMIO file: missing %%MatrixMarket header".into(),
        ));
    }

    let header_lower = header.to_lowercase();
    if header_lower.contains("symmetric") {
        symmetric = true;
    }

    // Skip comment lines, find the size line.
    let mut size_line = String::new();
    for line in &mut lines {
        let line = line.map_err(|e| Error::Internal(format!("MMIO read error: {e}")))?;
        let trimmed = line.trim();
        if trimmed.starts_with('%') || trimmed.is_empty() {
            continue;
        }
        size_line = trimmed.to_string();
        break;
    }

    // Parse size line: rows cols nnz
    let size_parts: Vec<&str> = size_line.split_whitespace().collect();
    if size_parts.len() < 3 {
        return Err(Error::Internal("invalid MMIO size line".into()));
    }
    let nnz: usize = size_parts[2]
        .parse()
        .map_err(|_| Error::Internal(format!("invalid nnz count: '{}'", size_parts[2])))?;

    // Parse data lines.
    let mut edges = Vec::with_capacity(nnz);
    for line in lines {
        let line = line.map_err(|e| Error::Internal(format!("MMIO read error: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let row_str = parts.next().unwrap_or("");
        let col_str = parts.next().unwrap_or("");

        let row: u64 = row_str
            .parse()
            .map_err(|_| Error::Internal(format!("invalid MMIO row: '{row_str}'")))?;
        let col: u64 = col_str
            .parse()
            .map_err(|_| Error::Internal(format!("invalid MMIO col: '{col_str}'")))?;

        edges.push((row, col));
    }

    Ok((edges, symmetric))
}

#[cfg(test)]
mod tests {
    use super::super::GrafeoDB;

    #[test]
    fn test_import_tsv_str_directed() {
        let db = GrafeoDB::new_in_memory();
        let data = "# comment\n1\t2\n2\t3\n3\t1\n";
        let (nodes, edges) = db.import_tsv_str(data, "CONNECTS", true).unwrap();

        assert_eq!(nodes, 3);
        assert_eq!(edges, 3);
        assert_eq!(db.node_count(), 3);
        assert_eq!(db.edge_count(), 3);
    }

    #[test]
    fn test_import_tsv_str_undirected() {
        let db = GrafeoDB::new_in_memory();
        let data = "1 2\n2 3\n";
        let (nodes, edges) = db.import_tsv_str(data, "CONNECTS", false).unwrap();

        assert_eq!(nodes, 3);
        assert_eq!(edges, 4); // 2 edges * 2 directions
    }

    #[test]
    fn test_import_tsv_str_with_weights() {
        let db = GrafeoDB::new_in_memory();
        // Third column (weight) should be ignored
        let data = "1\t2\t0.5\n2\t3\t1.0\n";
        let (nodes, edges) = db.import_tsv_str(data, "E", true).unwrap();

        assert_eq!(nodes, 3);
        assert_eq!(edges, 2);
    }

    #[test]
    fn test_import_tsv_str_comments_and_blanks() {
        let db = GrafeoDB::new_in_memory();
        let data = "# header\n% also a comment\n\n1 2\n\n3 4\n";
        let (nodes, edges) = db.import_tsv_str(data, "E", true).unwrap();

        assert_eq!(nodes, 4);
        assert_eq!(edges, 2);
    }

    #[test]
    fn test_import_tsv_str_empty() {
        let db = GrafeoDB::new_in_memory();
        let data = "# only comments\n% nothing here\n";
        let (nodes, edges) = db.import_tsv_str(data, "E", true).unwrap();

        assert_eq!(nodes, 0);
        assert_eq!(edges, 0);
    }

    #[test]
    fn test_import_tsv_str_duplicate_nodes() {
        let db = GrafeoDB::new_in_memory();
        // Node 1 appears in multiple edges
        let data = "1 2\n1 3\n1 4\n";
        let (nodes, edges) = db.import_tsv_str(data, "E", true).unwrap();

        assert_eq!(nodes, 4); // 1, 2, 3, 4
        assert_eq!(edges, 3);
    }

    #[test]
    fn test_import_mmio_str() {
        let db = GrafeoDB::new_in_memory();
        let data = "%%MatrixMarket matrix coordinate real general\n% comment\n3 3 3\n1 2 1.0\n2 3 1.0\n3 1 1.0\n";

        let reader = std::io::BufReader::new(data.as_bytes());
        let (edges, symmetric) = super::parse_mmio(reader).unwrap();

        assert!(!symmetric);
        assert_eq!(edges.len(), 3);

        let result = db.import_edge_list(&edges, "E", true);
        assert!(result.is_ok());
        let (nodes, edge_count) = result.unwrap();
        assert_eq!(nodes, 3);
        assert_eq!(edge_count, 3);
    }

    #[test]
    fn test_import_mmio_symmetric() {
        let data = "%%MatrixMarket matrix coordinate real symmetric\n3 3 2\n1 2 1.0\n2 3 1.0\n";

        let reader = std::io::BufReader::new(data.as_bytes());
        let (edges, symmetric) = super::parse_mmio(reader).unwrap();

        assert!(symmetric);
        assert_eq!(edges.len(), 2);

        let db = GrafeoDB::new_in_memory();
        // Symmetric = undirected = both directions
        let (nodes, edge_count) = db.import_edge_list(&edges, "E", false).unwrap();
        assert_eq!(nodes, 3);
        assert_eq!(edge_count, 4); // 2 edges * 2 directions
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_import_tsv_rdf() {
        use grafeo_core::graph::GraphStore;
        use grafeo_core::graph::rdf::RdfGraphStoreAdapter;

        let db = GrafeoDB::new_in_memory();

        // Write TSV to a temp file
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tsv");
        std::fs::write(&path, "1\t2\n2\t3\n3\t1\n").unwrap();

        let (nodes, edges) = db
            .import_tsv_rdf(
                &path,
                "http://example.org/connects",
                "http://example.org/node/",
            )
            .unwrap();

        assert_eq!(nodes, 3);
        assert_eq!(edges, 3);

        // Verify the adapter works on the imported RDF data
        let adapter = RdfGraphStoreAdapter::new(&db.rdf_store);
        assert_eq!(adapter.node_count(), 3);
        assert_eq!(adapter.edge_count(), 3);
    }
}
