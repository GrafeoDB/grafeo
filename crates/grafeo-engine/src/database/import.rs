//! Bulk graph import from TSV edge lists and Matrix Market files.
//!
//! An import is one bulk write (see
//! [`StreamingChanges`](crate::transaction::StreamingChanges)): its node and
//! edge ids are reserved in two ranges, each one entry of its change set
//! however many rows it writes, so its memory does not grow with the import;
//! its rows are written to the WAL as it goes and closed by one commit
//! marker at the end. This is 10-100x faster than calling
//! `create_node`/`create_edge` in a loop for large graphs.
//!
//! An import first checks that the database takes it: on a read-only
//! database, after `close()` and after a commit that did not complete it
//! refuses before it opens or parses anything. It then reads and parses its
//! input, without blocking anything. While it then changes the store,
//! commits, new transactions, writes outside a transaction and checkpoints
//! wait for it: a checkpoint or `close()` holds all of the import or none of
//! it, and so does the WAL after a crash. Once it returns it survives a
//! crash. Reads outside a transaction go on. An import reports no change
//! data capture events.
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
use std::sync::Arc;

use grafeo_common::change::{DataOp, Labels, Table};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId};
use grafeo_common::utils::error::{Error, Result};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::apply::ChangeTarget;

use crate::transaction::StreamingChanges;

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
    /// wait for it; the file is read before, without blocking anything, and
    /// only once the database takes the import: a refused call opens nothing.
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
        // Refused before the file is opened: a refused import does no work.
        self.check_import_allowed()?;
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open {}: {e}", path.display()),
            ))
        })?;

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
    /// wait for it; the string is parsed before, without blocking anything,
    /// and only once the database takes the import.
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
        self.check_import_allowed()?;
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
    /// wait for it; the file is read before, without blocking anything, and
    /// only once the database takes the import: a refused call opens nothing.
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
        self.check_import_allowed()?;
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open {}: {e}", path.display()),
            ))
        })?;

        let reader = BufReader::new(file);
        let (edges, symmetric) = parse_mmio(reader)?;
        self.import_edge_list(&edges, edge_type, !symmetric)
    }

    /// Bulk-imports a pre-parsed edge list into the LPG store, as one bulk
    /// write (see the [module docs](self)). The public imports call
    /// [`check_import_allowed`](Self::check_import_allowed) before they read
    /// their input; this checks again, under the hold.
    fn import_edge_list(
        &self,
        edges: &[(u64, u64)],
        edge_type: &str,
        directed: bool,
    ) -> Result<(usize, usize)> {
        self.import_edge_list_observed(edges, edge_type, directed, |_| {})
    }

    /// [`import_edge_list`](Self::import_edge_list), showing `observe` the
    /// import's changes after each row it writes.
    fn import_edge_list_observed(
        &self,
        edges: &[(u64, u64)],
        edge_type: &str,
        directed: bool,
        mut observe: impl FnMut(&StreamingChanges),
    ) -> Result<(usize, usize)> {
        // The row of each external id, in the order the edges name them.
        let mut rows: FxHashMap<u64, u64> = FxHashMap::default();
        let mut next = 0_u64;
        for &(src, dst) in edges {
            for external in [src, dst] {
                rows.entry(external).or_insert_with(|| {
                    next += 1;
                    next - 1
                });
            }
        }
        let per_line = if directed { 1 } else { 2 };
        let edge_count = edges.len().checked_mul(per_line).ok_or_else(|| {
            Error::Internal(format!(
                "{} lines: more edges than a store has",
                edges.len()
            ))
        })?;

        let held = self.hold_commits_for_import()?;
        let store: Arc<dyn ChangeTarget> = self.lpg_store();
        let mut changes = self.streaming_changes(&held);
        let nodes = changes.reserve(&store, Table::Nodes, rows.len())?;
        let labels: Labels = std::iter::once(ArcStr::from("_Imported")).collect();
        for raw in nodes.clone() {
            changes.apply(DataOp::CreateNode {
                id: NodeId::new(raw),
                labels: labels.clone(),
                properties: Vec::new(),
            })?;
            observe(&changes);
        }
        let node = |external: u64| NodeId::new(nodes.start + rows[&external]);
        let edge_ids = changes.reserve(&store, Table::Edges, edge_count)?;
        let edge_type = ArcStr::from(edge_type);
        let mut ids = edge_ids.map(EdgeId::new);
        for &(src, dst) in edges {
            let both = [(node(src), node(dst)), (node(dst), node(src))];
            for &(src, dst) in &both[..per_line] {
                let id = ids.next().ok_or_else(|| {
                    Error::Internal("an import wrote more edges than it reserved".to_string())
                })?;
                changes.apply(DataOp::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type: edge_type.clone(),
                    properties: Vec::new(),
                })?;
                observe(&changes);
            }
        }

        if !changes.is_empty() {
            // Commits wait, so no commit is between its epoch and its
            // completion: the next epoch is free.
            let epoch = EpochId::new(self.transaction_manager.current_epoch().as_u64() + 1);
            changes.commit(Some(epoch))?;
            self.transaction_manager.sync_epoch(epoch);
        }
        drop(held);

        // Refresh statistics so the optimizer has fresh data.
        self.lpg_store().ensure_statistics_fresh();

        Ok((rows.len(), edge_count))
    }

    /// Bulk-imports a TSV edge list into the RDF store.
    ///
    /// Each edge `(src, dst)` becomes a triple:
    /// `<{base_uri}{src}> <{predicate_uri}> <{base_uri}{dst}>`
    ///
    /// While it changes the store, commits, new transactions and checkpoints
    /// wait for it; the file is read before, without blocking anything, and
    /// only once the database takes the import: a refused call opens nothing.
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

        // Refused before the file is opened: a refused import does no work.
        self.check_import_allowed()?;
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open {}: {e}", path.display()),
            ))
        })?;

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

        let held = self.hold_commits_for_import()?;
        let edge_count = self.insert_triples_streamed(&held, triples)?;

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
        let line = line.map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read line {}: {e}", line_num + 1),
            ))
        })?;
        let trimmed = line.trim();

        // Skip comments and empty lines.
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let src_str = parts.next().ok_or_else(|| {
            Error::InvalidValue(format!("line {}: missing source ID", line_num + 1))
        })?;
        let dst_str = parts.next().ok_or_else(|| {
            Error::InvalidValue(format!("line {}: missing target ID", line_num + 1))
        })?;

        let src: u64 = src_str.parse().map_err(|_| {
            Error::InvalidValue(format!(
                "line {}: invalid source ID '{}'",
                line_num + 1,
                src_str
            ))
        })?;
        let dst: u64 = dst_str.parse().map_err(|_| {
            Error::InvalidValue(format!(
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
        .ok_or_else(|| Error::InvalidValue("empty MMIO file".into()))?
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read the MatrixMarket header: {e}"),
            ))
        })?;

    if !header.starts_with("%%MatrixMarket") {
        return Err(Error::InvalidValue(
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
        let line = line.map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read a MatrixMarket line: {e}"),
            ))
        })?;
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
        return Err(Error::InvalidValue("invalid MMIO size line".into()));
    }
    let nnz: usize = size_parts[2]
        .parse()
        .map_err(|_| Error::InvalidValue(format!("invalid nnz count: '{}'", size_parts[2])))?;

    // Parse data lines.
    let mut edges = Vec::with_capacity(nnz);
    for line in lines {
        let line = line.map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read a MatrixMarket line: {e}"),
            ))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let row_str = parts.next().unwrap_or("");
        let col_str = parts.next().unwrap_or("");

        let row: u64 = row_str
            .parse()
            .map_err(|_| Error::InvalidValue(format!("invalid MMIO row: '{row_str}'")))?;
        let col: u64 = col_str
            .parse()
            .map_err(|_| Error::InvalidValue(format!("invalid MMIO col: '{col_str}'")))?;

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

    /// An import's change set holds its two ranges whatever its size, and
    /// its WAL records wait for their write a run at a time: the memory of
    /// an import of 1,000 lines and one of 100,000 lines is the same.
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    #[test]
    fn an_import_keeps_constant_change_set_memory() {
        use crate::transaction::STREAMED_RECORDS;

        let mut most = Vec::new();
        for lines in [1_000_usize, 100_000] {
            let dir = tempfile::tempdir().unwrap();
            let db = GrafeoDB::open(dir.path().join("amsterdam.grafeo")).unwrap();
            let ring = u64::try_from(lines).unwrap();
            let edges: Vec<(u64, u64)> = (0..ring).map(|n| (n, (n + 3) % ring)).collect();
            let (mut bytes, mut pending) = (0, 0);
            let (nodes, edge_count) = db
                .import_edge_list_observed(&edges, "ROUTE", false, |changes| {
                    bytes = bytes.max(changes.approx_bytes());
                    pending = pending.max(changes.pending_records());
                })
                .unwrap();
            assert_eq!((nodes, edge_count), (lines, 2 * lines));
            assert!(
                pending < STREAMED_RECORDS,
                "{lines} lines: {pending} records waited for their write"
            );
            most.push(bytes);
            db.close().unwrap();
        }
        assert_eq!(
            most[0], most[1],
            "the change set of 100,000 lines holds what the one of 1,000 does"
        );
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
