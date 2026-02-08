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
        for format in [DumpFormat::Parquet, DumpFormat::Turtle, DumpFormat::Json] {
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
    fn test_compaction_stats_serde() {
        let stats = CompactionStats {
            bytes_reclaimed: 1024,
            nodes_compacted: 10,
            edges_compacted: 20,
            duration_ms: 150,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: CompactionStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.bytes_reclaimed, 1024);
        assert_eq!(parsed.duration_ms, 150);
    }

    #[test]
    fn test_dump_metadata_serde() {
        let metadata = DumpMetadata {
            version: "0.4.1".to_string(),
            mode: DatabaseMode::Lpg,
            format: DumpFormat::Parquet,
            node_count: 1000,
            edge_count: 5000,
            created_at: "2025-01-15T12:00:00Z".to_string(),
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: DumpMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_count, 1000);
        assert_eq!(parsed.format, DumpFormat::Parquet);
    }

    #[test]
    fn test_dump_metadata_with_extra() {
        let mut extra = HashMap::new();
        extra.insert("compression".to_string(), "zstd".to_string());
        let metadata = DumpMetadata {
            version: "0.4.1".to_string(),
            mode: DatabaseMode::Rdf,
            format: DumpFormat::Turtle,
            node_count: 0,
            edge_count: 0,
            created_at: "2025-01-15T12:00:00Z".to_string(),
            extra,
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: DumpMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.extra.get("compression").unwrap(), "zstd");
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
