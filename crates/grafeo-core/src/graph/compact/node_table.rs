//! Columnar node storage, a table per label set.
//!
//! Each `NodeTable` stores all nodes with one set of labels as typed columns.
//! Nodes are addressed by row offset; the `NodeId` encodes (table_id, offset).

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::id::encode_node_id;
use super::schema::TableSchema;

/// Columnar storage for the nodes with one set of labels.
///
/// All nodes with the same labels are stored in a single `NodeTable` with one
/// [`ColumnCodec`] per property. Row offsets are combined with the table ID
/// via [`encode_node_id`] to produce globally unique [`NodeId`] values.
#[derive(Debug)]
pub struct NodeTable {
    /// Schema describing the label, table ID, and column definitions.
    schema: TableSchema,
    /// Columns keyed by property name.
    columns: FxHashMap<PropertyKey, ColumnCodec>,
    /// Number of rows (nodes) in the table.
    len: usize,
}

impl NodeTable {
    /// Creates a table of `len` rows from its columns, as the section
    /// reader decoded them.
    #[must_use]
    pub fn from_columns(
        schema: TableSchema,
        columns: FxHashMap<PropertyKey, ColumnCodec>,
        len: usize,
    ) -> Self {
        Self {
            schema,
            columns,
            len,
        }
    }

    /// Returns the number of nodes in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the table's key, not a label: the labels of its nodes in name
    /// order, each with a `\` before every `\` and `|` it holds, joined with
    /// `|`. A table of single-label nodes has their label as its key only
    /// when the label holds neither character (the label `In|Out` is keyed
    /// `In\|Out`). The nodes without labels have the empty key, and the
    /// nodes whose one label is the empty label have the key `|`.
    #[must_use]
    pub fn label(&self) -> &str {
        self.schema.label.as_str()
    }

    /// Generates a [`NodeId`] for every row in this table.
    ///
    /// The IDs are returned in row order (offset 0, 1, 2, ...).
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let table_id = self.schema.table_id;
        (0..self.len)
            .map(|offset| encode_node_id(table_id, offset as u64))
            .collect()
    }

    /// Returns all properties for the node at the given row offset.
    ///
    /// Out-of-bounds offsets produce an empty map.
    #[must_use]
    pub fn get_all_properties(&self, offset: usize) -> FxHashMap<PropertyKey, Value> {
        let mut props = FxHashMap::default();
        if offset >= self.len {
            return props;
        }
        for (key, col) in &self.columns {
            if let Some(value) = col.get(offset) {
                props.insert(key.clone(), value);
            }
        }
        props
    }
}
