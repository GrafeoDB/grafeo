//! Admin API types for database inspection, backup, and maintenance.
//!
//! These types support both LPG (Labeled Property Graph) and RDF (Resource Description Framework)
//! data models.
//!
//! The result structs are read, not built: later releases may add fields, so
//! they are `#[non_exhaustive]` and a pattern names their fields with `..`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Database mode - either LPG (Labeled Property Graph) or RDF (Triple Store).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// Compiled feature flags (e.g. "gql", "cypher", "algos", "vector-index").
    pub features: Vec<String>,
}

/// Detailed database statistics returned by `db.stats()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct LabelInfo {
    /// The label name.
    pub name: String,
    /// Number of nodes with this label.
    pub count: usize,
}

/// Information about an edge type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EdgeTypeInfo {
    /// The edge type name.
    pub name: String,
    /// Number of edges with this type.
    pub count: usize,
}

/// Schema information for RDF databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct PredicateInfo {
    /// The predicate IRI.
    pub iri: String,
    /// Number of triples using this predicate.
    pub count: usize,
}

/// Combined schema information supporting both LPG and RDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode")]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum DumpFormat {
    /// Apache Parquet format (default for LPG).
    Parquet,
    /// RDF Turtle format (default for RDF).
    Turtle,
    /// JSON Lines format.
    Json,
    /// Arrow IPC stream format (zero-copy interop with DuckDB, Polars, pandas).
    Arrow,
    /// GEXF 1.3 format (Gephi, Gephi Lite, NetworkX).
    Gexf,
    /// GraphML format (Gephi, Cytoscape, yEd, igraph).
    GraphMl,
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
            DumpFormat::Arrow => write!(f, "arrow"),
            DumpFormat::Gexf => write!(f, "gexf"),
            DumpFormat::GraphMl => write!(f, "graphml"),
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
            "arrow" | "arrow-ipc" | "ipc" => Ok(DumpFormat::Arrow),
            "gexf" => Ok(DumpFormat::Gexf),
            "graphml" => Ok(DumpFormat::GraphMl),
            _ => Err(format!("Unknown dump format: {}", s)),
        }
    }
}

/// What [`GrafeoDB::compact`](crate::GrafeoDB::compact) did.
///
/// `compact()` may take on more work in later releases, and its report then
/// gains fields, so the struct is `#[non_exhaustive]`: read its fields, and
/// expect new ones.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompactReport {
    /// Whether a checkpoint of the database file was written: only for a
    /// persistent database that is not read-only.
    pub checkpointed: bool,
    /// The old versions dropped, in every graph: versions no open transaction
    /// can see any more.
    pub versions_collected: u64,
    /// How long `compact()` took, in milliseconds.
    pub duration_ms: u64,
}

/// Trait for administrative database operations.
///
/// Provides a uniform interface for introspection, validation, and
/// maintenance operations. Used by the CLI, REST API, and bindings
/// to inspect and manage a Grafeo database.
///
/// Implemented by [`GrafeoDB`](crate::GrafeoDB).
pub trait AdminService {
    /// Returns high-level database information (counts, mode, persistence).
    fn info(&self) -> DatabaseInfo;

    /// Returns detailed database statistics (memory, disk, indexes).
    fn detailed_stats(&self) -> DatabaseStats;

    /// Returns schema information (labels, edge types, property keys).
    fn schema(&self) -> SchemaInfo;

    /// Validates database integrity, returning errors and warnings.
    fn validate(&self) -> ValidationResult;

    /// Returns WAL (Write-Ahead Log) status.
    fn wal_status(&self) -> WalStatus;

    /// Forces a WAL checkpoint, flushing pending records to storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails.
    fn wal_checkpoint(&self) -> grafeo_common::utils::error::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- DatabaseMode ----

    #[test]
    fn test_database_mode_display() {
        assert_eq!(DatabaseMode::Lpg.to_string(), "lpg");
        assert_eq!(DatabaseMode::Rdf.to_string(), "rdf");
    }

    #[test]
    fn test_database_mode_serde_roundtrip() {
        let json = serde_json::to_string(&DatabaseMode::Lpg).unwrap();
        let mode: DatabaseMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, DatabaseMode::Lpg);

        let json = serde_json::to_string(&DatabaseMode::Rdf).unwrap();
        let mode: DatabaseMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, DatabaseMode::Rdf);
    }

    #[test]
    fn test_database_mode_equality() {
        assert_eq!(DatabaseMode::Lpg, DatabaseMode::Lpg);
        assert_ne!(DatabaseMode::Lpg, DatabaseMode::Rdf);
    }

    // ---- DumpFormat ----

    #[test]
    fn test_dump_format_default() {
        assert_eq!(DumpFormat::default(), DumpFormat::Parquet);
    }

    #[test]
    fn test_dump_format_display() {
        assert_eq!(DumpFormat::Parquet.to_string(), "parquet");
        assert_eq!(DumpFormat::Turtle.to_string(), "turtle");
        assert_eq!(DumpFormat::Json.to_string(), "json");
        assert_eq!(DumpFormat::Arrow.to_string(), "arrow");
        assert_eq!(DumpFormat::Gexf.to_string(), "gexf");
        assert_eq!(DumpFormat::GraphMl.to_string(), "graphml");
    }

    #[test]
    fn test_dump_format_from_str() {
        assert_eq!(
            "parquet".parse::<DumpFormat>().unwrap(),
            DumpFormat::Parquet
        );
        assert_eq!("turtle".parse::<DumpFormat>().unwrap(), DumpFormat::Turtle);
        assert_eq!("ttl".parse::<DumpFormat>().unwrap(), DumpFormat::Turtle);
        assert_eq!("json".parse::<DumpFormat>().unwrap(), DumpFormat::Json);
        assert_eq!("jsonl".parse::<DumpFormat>().unwrap(), DumpFormat::Json);
        assert_eq!("arrow".parse::<DumpFormat>().unwrap(), DumpFormat::Arrow);
        assert_eq!(
            "arrow-ipc".parse::<DumpFormat>().unwrap(),
            DumpFormat::Arrow
        );
        assert_eq!("ipc".parse::<DumpFormat>().unwrap(), DumpFormat::Arrow);
        assert_eq!("gexf".parse::<DumpFormat>().unwrap(), DumpFormat::Gexf);
        assert_eq!(
            "graphml".parse::<DumpFormat>().unwrap(),
            DumpFormat::GraphMl
        );
        assert_eq!(
            "PARQUET".parse::<DumpFormat>().unwrap(),
            DumpFormat::Parquet
        );
    }

    #[test]
    fn test_dump_format_from_str_invalid() {
        let result = "xml".parse::<DumpFormat>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown dump format"));
    }

    #[test]
    fn test_dump_format_serde_roundtrip() {
        for format in [
            DumpFormat::Parquet,
            DumpFormat::Turtle,
            DumpFormat::Json,
            DumpFormat::Arrow,
            DumpFormat::Gexf,
            DumpFormat::GraphMl,
        ] {
            let json = serde_json::to_string(&format).unwrap();
            let parsed: DumpFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, format);
        }
    }

    // ---- ValidationResult ----

    #[test]
    fn test_validation_result_default_is_valid() {
        let result = ValidationResult::default();
        assert!(result.is_valid());
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validation_result_with_errors() {
        let result = ValidationResult {
            errors: vec![ValidationError {
                code: "E001".to_string(),
                message: "Orphaned edge".to_string(),
                context: Some("edge_42".to_string()),
            }],
            warnings: Vec::new(),
        };
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validation_result_with_warnings_still_valid() {
        let result = ValidationResult {
            errors: Vec::new(),
            warnings: vec![ValidationWarning {
                code: "W001".to_string(),
                message: "Unused index".to_string(),
                context: None,
            }],
        };
        assert!(result.is_valid());
    }

    // ---- Serde roundtrips for complex types ----

    #[test]
    fn test_database_info_serde() {
        let info = DatabaseInfo {
            mode: DatabaseMode::Lpg,
            node_count: 100,
            edge_count: 200,
            is_persistent: true,
            path: Some(PathBuf::from("/tmp/db")),
            wal_enabled: true,
            version: "0.4.1".to_string(),
            features: vec!["gql".into(), "cypher".into()],
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: DatabaseInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_count, 100);
        assert_eq!(parsed.edge_count, 200);
        assert!(parsed.is_persistent);
    }

    #[test]
    fn test_database_stats_serde() {
        let stats = DatabaseStats {
            node_count: 50,
            edge_count: 75,
            label_count: 3,
            edge_type_count: 2,
            property_key_count: 10,
            index_count: 4,
            memory_bytes: 1024,
            disk_bytes: Some(2048),
        };
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: DatabaseStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_count, 50);
        assert_eq!(parsed.disk_bytes, Some(2048));
    }

    #[test]
    fn test_schema_info_lpg_serde() {
        let schema = SchemaInfo::Lpg(LpgSchemaInfo {
            labels: vec![LabelInfo {
                name: "Person".to_string(),
                count: 10,
            }],
            edge_types: vec![EdgeTypeInfo {
                name: "KNOWS".to_string(),
                count: 20,
            }],
            property_keys: vec!["name".to_string(), "age".to_string()],
        });
        let json = serde_json::to_string(&schema).unwrap();
        let parsed: SchemaInfo = serde_json::from_str(&json).unwrap();
        match parsed {
            SchemaInfo::Lpg(lpg) => {
                assert_eq!(lpg.labels.len(), 1);
                assert_eq!(lpg.labels[0].name, "Person");
                assert_eq!(lpg.edge_types[0].count, 20);
            }
            SchemaInfo::Rdf(_) => panic!("Expected LPG schema"),
        }
    }

    #[test]
    fn test_schema_info_rdf_serde() {
        let schema = SchemaInfo::Rdf(RdfSchemaInfo {
            predicates: vec![PredicateInfo {
                iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
                count: 5,
            }],
            named_graphs: vec!["default".to_string()],
            subject_count: 10,
            object_count: 15,
        });
        let json = serde_json::to_string(&schema).unwrap();
        let parsed: SchemaInfo = serde_json::from_str(&json).unwrap();
        match parsed {
            SchemaInfo::Rdf(rdf) => {
                assert_eq!(rdf.predicates.len(), 1);
                assert_eq!(rdf.subject_count, 10);
            }
            SchemaInfo::Lpg(_) => panic!("Expected RDF schema"),
        }
    }

    #[test]
    fn test_index_info_serde() {
        let info = IndexInfo {
            name: "idx_person_name".to_string(),
            index_type: "btree".to_string(),
            target: "Person:name".to_string(),
            unique: true,
            cardinality: Some(1000),
            size_bytes: Some(4096),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: IndexInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "idx_person_name");
        assert!(parsed.unique);
    }

    #[test]
    fn test_wal_status_serde() {
        let status = WalStatus {
            enabled: true,
            path: Some(PathBuf::from("/tmp/wal")),
            size_bytes: 8192,
            record_count: 42,
            last_checkpoint: Some(1700000000),
            current_epoch: 100,
        };
        let json = serde_json::to_string(&status).unwrap();
        let parsed: WalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.record_count, 42);
        assert_eq!(parsed.current_epoch, 100);
    }

    #[test]
    fn a_compact_report_round_trips_through_json() {
        let report = CompactReport {
            checkpointed: true,
            versions_collected: 19,
            duration_ms: 88,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            json, r#"{"checkpointed":true,"versions_collected":19,"duration_ms":88}"#,
            "the field names the bindings and server clients read"
        );
        let parsed: CompactReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, report);
    }

    #[test]
    fn test_validation_error_serde() {
        let error = ValidationError {
            code: "E001".to_string(),
            message: "Broken reference".to_string(),
            context: Some("node_id=42".to_string()),
        };
        let json = serde_json::to_string(&error).unwrap();
        let parsed: ValidationError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.code, "E001");
        assert_eq!(parsed.context, Some("node_id=42".to_string()));
    }

    #[test]
    fn test_validation_warning_serde() {
        let warning = ValidationWarning {
            code: "W001".to_string(),
            message: "High memory usage".to_string(),
            context: None,
        };
        let json = serde_json::to_string(&warning).unwrap();
        let parsed: ValidationWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.code, "W001");
        assert!(parsed.context.is_none());
    }
}
