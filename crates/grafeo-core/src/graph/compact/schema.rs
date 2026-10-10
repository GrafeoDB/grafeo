//! The schemas of a 0.5.x base's tables: the key and id of a node table,
//! the edge type and id of a relationship table.

use arcstr::ArcStr;

/// Schema for a node table, one per label set. The `table_id` is encoded
/// into the table's node ids when the base does not keep ids (see
/// [`encode_node_id`](super::id::encode_node_id)).
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// The table's key: the labels of its nodes in name order, joined with
    /// `|` (a `\` escapes a `\` or `|` in a label), such as "Person" or
    /// "Actor|Person".
    pub label: ArcStr,
    /// Unique table identifier (15-bit max).
    pub table_id: u16,
}

impl TableSchema {
    /// Creates a new table schema.
    #[must_use]
    pub fn new(label: impl Into<ArcStr>, table_id: u16) -> Self {
        Self {
            label: label.into(),
            table_id,
        }
    }
}

/// Schema for a relationship (edge) table. The `rel_table_id` is encoded
/// into the table's edge ids when the base does not keep ids (see
/// [`encode_edge_id`](super::id::encode_edge_id)).
#[derive(Debug, Clone)]
pub struct EdgeSchema {
    /// The edge type this table stores (e.g. "KNOWS", "ACTED_IN").
    pub edge_type: ArcStr,
    /// Unique relationship table identifier (15-bit max).
    pub rel_table_id: u16,
}

impl EdgeSchema {
    /// Creates a new edge schema.
    #[must_use]
    pub fn new(edge_type: impl Into<ArcStr>, rel_table_id: u16) -> Self {
        Self {
            edge_type: edge_type.into(),
            rel_table_id,
        }
    }
}
