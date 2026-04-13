//! Schema definitions for CompactStore tables.
//!
//! Each node table has a fixed schema (column names and types).
//! Each relationship table connects two node tables with a specific edge type.

use arcstr::ArcStr;

/// The type of data stored in a column.
///
/// Each variant describes both the logical type and the physical encoding
/// used in the columnar storage. This lets the query engine pick the right
/// decoder without any runtime dispatch.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ColumnType {
    /// Unsigned integer with a fixed bit width (1, 8, 16, 32, or 64).
    UInt {
        /// Number of bits per value.
        bits: u8,
    },
    /// Dictionary-encoded strings (each value is an index into a string table).
    DictString,
    /// Boolean values (1 bit per value, packed).
    Bool,
    /// Fixed-dimension 8-bit integer vector (for embeddings).
    Int8Vector {
        /// Number of dimensions in each vector.
        dimensions: u16,
    },
}

/// A single column within a table schema.
///
/// Pairs a column name with its [`ColumnType`]. The name is used for property
/// lookups; the type tells the storage layer how to encode/decode values.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnDef {
    /// Column name (matches the property key in queries).
    pub name: ArcStr,
    /// Physical and logical type of this column.
    pub column_type: ColumnType,
}

impl ColumnDef {
    /// Creates a new column definition.
    #[must_use]
    pub fn new(name: impl Into<ArcStr>, column_type: ColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
        }
    }
}

/// Schema for a node table, one per label.
///
/// All nodes with a given label share the same columnar layout defined here.
/// The `table_id` is encoded into [`NodeId`](grafeo_common::types::NodeId)
/// values via [`encode_node_id`](super::id::encode_node_id).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableSchema {
    /// The node label this table stores (e.g. "Person", "Movie").
    pub label: ArcStr,
    /// Unique table identifier, encoded into node IDs (15-bit max).
    pub table_id: u16,
    /// Columns in this table, in order.
    pub columns: Vec<ColumnDef>,
}

impl TableSchema {
    /// Creates a new table schema.
    #[must_use]
    pub fn new(label: impl Into<ArcStr>, table_id: u16, columns: Vec<ColumnDef>) -> Self {
        Self {
            label: label.into(),
            table_id,
            columns,
        }
    }

    /// Returns the column definition for the given name, if it exists.
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name.as_str() == name)
    }

    /// Returns the number of columns in this table.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

/// Schema for a relationship (edge) table.
///
/// Describes the edge type, which node tables it connects, and any property
/// columns stored on the edges. The `rel_table_id` is encoded into
/// [`EdgeId`](grafeo_common::types::EdgeId) values via
/// [`encode_edge_id`](super::id::encode_edge_id).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EdgeSchema {
    /// The edge type this table stores (e.g. "KNOWS", "ACTED_IN").
    pub edge_type: ArcStr,
    /// Unique relationship table identifier, encoded into edge IDs (15-bit max).
    pub rel_table_id: u16,
    /// Label of the source node table.
    pub src_label: ArcStr,
    /// Label of the destination node table.
    pub dst_label: ArcStr,
    /// Property columns on edges of this type.
    pub property_columns: Vec<ColumnDef>,
}

impl EdgeSchema {
    /// Creates a new edge schema.
    #[must_use]
    pub fn new(
        edge_type: impl Into<ArcStr>,
        rel_table_id: u16,
        src_label: impl Into<ArcStr>,
        dst_label: impl Into<ArcStr>,
        property_columns: Vec<ColumnDef>,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            rel_table_id,
            src_label: src_label.into(),
            dst_label: dst_label.into(),
            property_columns,
        }
    }

    /// Returns the number of property columns on edges of this type.
    #[must_use]
    pub fn property_count(&self) -> usize {
        self.property_columns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_def_creation() {
        let col = ColumnDef::new("age", ColumnType::UInt { bits: 8 });
        assert_eq!(col.name.as_str(), "age");
        assert_eq!(col.column_type, ColumnType::UInt { bits: 8 });
    }

    #[test]
    fn test_table_schema() {
        let schema = TableSchema::new(
            "Person",
            0,
            vec![
                ColumnDef::new("name", ColumnType::DictString),
                ColumnDef::new("age", ColumnType::UInt { bits: 8 }),
                ColumnDef::new("active", ColumnType::Bool),
            ],
        );

        assert_eq!(schema.label.as_str(), "Person");
        assert_eq!(schema.table_id, 0);
        assert_eq!(schema.column_count(), 3);
        assert!(schema.column_by_name("name").is_some());
        assert!(schema.column_by_name("age").is_some());
        assert!(schema.column_by_name("missing").is_none());
    }

    #[test]
    fn test_edge_schema() {
        let schema = EdgeSchema::new(
            "KNOWS",
            1,
            "Person",
            "Person",
            vec![ColumnDef::new("since", ColumnType::UInt { bits: 16 })],
        );

        assert_eq!(schema.edge_type.as_str(), "KNOWS");
        assert_eq!(schema.rel_table_id, 1);
        assert_eq!(schema.src_label.as_str(), "Person");
        assert_eq!(schema.dst_label.as_str(), "Person");
        assert_eq!(schema.property_count(), 1);
    }

    #[test]
    fn test_column_types() {
        // Verify all ColumnType variants are distinct.
        let types = [
            ColumnType::UInt { bits: 8 },
            ColumnType::UInt { bits: 32 },
            ColumnType::DictString,
            ColumnType::Bool,
            ColumnType::Int8Vector { dimensions: 128 },
        ];

        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_edge_schema_no_properties() {
        let schema = EdgeSchema::new("FOLLOWS", 2, "User", "User", vec![]);
        assert_eq!(schema.property_count(), 0);
    }
}
