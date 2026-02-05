//! Admin API types for database inspection, backup, and maintenance.
//!
//! These types support both LPG (Labeled Property Graph) and RDF (Resource Description Framework)
//! data models.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Database mode - either LPG (Labeled Property Graph) or RDF (Triple Store).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatabaseMode {
    /// Labeled Property Graph mode (nodes with labels and properties, typed edges).
    Lpg,
    /// RDF mode (subject-predicate-object triples).
    Rdf,
}

impl std::fmt::Display for DatabaseMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatabaseMode::Lpg => write!(f, "lpg"),
            DatabaseMode::Rdf => write!(f, "rdf"),
        }
    }
}

/// High-level database information returned by `db.info()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    /// Database mode (LPG or RDF).
    pub mode: DatabaseMode,
    /// Number of nodes (LPG) or subjects (RDF).
    pub node_count: usize,
    /// Number of edges (LPG) or triples (RDF).
    pub edge_count: usize,
    /// Whether the database is backed by a file.
    pub is_persistent: bool,
    /// Database file path, if persistent.
    pub path: Option<PathBuf>,
    /// Whether WAL is enabled.
    pub wal_enabled: bool,
    /// Database version.
    pub version: String,
}

/// Detailed database statistics returned by `db.stats()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    /// Number of nodes (LPG) or subjects (RDF).
    pub node_count: usize,
    /// Number of edges (LPG) or triples (RDF).
    pub edge_count: usize,
    /// Number of distinct labels (LPG) or classes (RDF).
    pub label_count: usize,
    /// Number of distinct edge types (LPG) or predicates (RDF).
    pub edge_type_count: usize,
    /// Number of distinct property keys.
    pub property_key_count: usize,
    /// Number of indexes.
    pub index_count: usize,
    /// Memory usage in bytes (approximate).
    pub memory_bytes: usize,
    /// Disk usage in bytes (if persistent).
    pub disk_bytes: Option<usize>,
}

/// Schema information for LPG databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpgSchemaInfo {
    /// All labels used in the database.
    pub labels: Vec<LabelInfo>,
    /// All edge types used in the database.
    pub edge_types: Vec<EdgeTypeInfo>,
    /// All property keys used in the database.
    pub property_keys: Vec<String>,
}

/// Information about a label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelInfo {
    /// The label name.
    pub name: String,
    /// Number of nodes with this label.
    pub count: usize,
}

/// Information about an edge type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTypeInfo {
    /// The edge type name.
    pub name: String,
    /// Number of edges with this type.
    pub count: usize,
}

/// Schema information for RDF databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfSchemaInfo {
    /// All predicates used in the database.
    pub predicates: Vec<PredicateInfo>,
    /// All named graphs.
    pub named_graphs: Vec<String>,
    /// Number of distinct subjects.
    pub subject_count: usize,
    /// Number of distinct objects.
    pub object_count: usize,
}

/// Information about an RDF predicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateInfo {
    /// The predicate IRI.
    pub iri: String,
    /// Number of triples using this predicate.
    pub count: usize,
}

/// Combined schema information supporting both LPG and RDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum SchemaInfo {
    /// LPG schema information.
    #[serde(rename = "lpg")]
    Lpg(LpgSchemaInfo),
    /// RDF schema information.
    #[serde(rename = "rdf")]
    Rdf(RdfSchemaInfo),
}

/// Index information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    /// Index name.
    pub name: String,
    /// Index type (hash, btree, fulltext, etc.).
    pub index_type: String,
    /// Target (label:property for LPG, predicate for RDF).
    pub target: String,
    /// Whether the index is unique.
    pub unique: bool,
    /// Estimated cardinality.
    pub cardinality: Option<usize>,
    /// Size in bytes.
    pub size_bytes: Option<usize>,
}

/// WAL (Write-Ahead Log) status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalStatus {
    /// Whether WAL is enabled.
    pub enabled: bool,
    /// WAL file path.
    pub path: Option<PathBuf>,
    /// WAL size in bytes.
    pub size_bytes: usize,
    /// Number of WAL records.
    pub record_count: usize,
    /// Last checkpoint timestamp (Unix epoch seconds).
    pub last_checkpoint: Option<u64>,
    /// Current epoch/LSN.
    pub current_epoch: u64,
}

/// Validation result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// List of validation errors (empty = valid).
    pub errors: Vec<ValidationError>,
    /// List of validation warnings.
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Returns true if validation passed (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// A validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// Optional context (e.g., affected entity ID).
    pub context: Option<String>,
}

/// A validation warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code.
    pub code: String,
    /// Human-readable warning message.
    pub message: String,
    /// Optional context.
    pub context: Option<String>,
}

/// Dump format for export operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DumpFormat {
    /// Apache Parquet format (default for LPG).
    Parquet,
    /// RDF Turtle format (default for RDF).
    Turtle,
    /// JSON Lines format.
    Json,
}

impl Default for DumpFormat {
    fn default() -> Self {
        DumpFormat::Parquet
    }
}

impl std::fmt::Display for DumpFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DumpFormat::Parquet => write!(f, "parquet"),
            DumpFormat::Turtle => write!(f, "turtle"),
            DumpFormat::Json => write!(f, "json"),
        }
    }
}

impl std::str::FromStr for DumpFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "parquet" => Ok(DumpFormat::Parquet),
            "turtle" | "ttl" => Ok(DumpFormat::Turtle),
            "json" | "jsonl" => Ok(DumpFormat::Json),
            _ => Err(format!("Unknown dump format: {}", s)),
        }
    }
}

/// Compaction statistics returned after a compact operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Bytes reclaimed.
    pub bytes_reclaimed: usize,
    /// Number of nodes compacted.
    pub nodes_compacted: usize,
    /// Number of edges compacted.
    pub edges_compacted: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Metadata for dump files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DumpMetadata {
    /// Grafeo version that created the dump.
    pub version: String,
    /// Database mode.
    pub mode: DatabaseMode,
    /// Dump format.
    pub format: DumpFormat,
    /// Number of nodes.
    pub node_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Timestamp (ISO 8601).
    pub created_at: String,
    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, String>,
}
