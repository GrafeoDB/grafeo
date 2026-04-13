//! Per-label columnar node storage.
//!
//! Each `NodeTable` stores all nodes of a single label as typed columns.
//! Nodes are addressed by row offset; the `NodeId` encodes (table_id, offset).

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::id::encode_node_id;
use super::schema::TableSchema;
use super::zone_map::ZoneMap;

/// Per-label columnar storage for nodes.
///
/// All nodes sharing a label are stored in a single `NodeTable` with one
/// [`ColumnCodec`] per property. Row offsets are combined with the table ID
/// via [`encode_node_id`] to produce globally unique [`NodeId`] values.
#[derive(Debug, Clone)]
pub struct NodeTable {
    /// Schema describing the label, table ID, and column definitions.
    schema: TableSchema,
    /// Columns keyed by property name.
    columns: FxHashMap<PropertyKey, ColumnCodec>,
    /// Per-column min/max statistics for predicate pushdown.
    zone_maps: FxHashMap<PropertyKey, ZoneMap>,
    /// Number of rows (nodes) in the table.
    len: usize,
}

impl NodeTable {
    /// Creates an empty table with the given schema.
    #[must_use]
    pub fn new(schema: TableSchema) -> Self {
        Self {
            schema,
            columns: FxHashMap::default(),
            zone_maps: FxHashMap::default(),
            len: 0,
        }
    }

    /// Creates a table from pre-built columns and zone maps.
    #[must_use]
    pub fn from_columns(
        schema: TableSchema,
        columns: FxHashMap<PropertyKey, ColumnCodec>,
        zone_maps: FxHashMap<PropertyKey, ZoneMap>,
        len: usize,
    ) -> Self {
        Self {
            schema,
            columns,
            zone_maps,
            len,
        }
    }

    /// Returns the number of nodes in this table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the table contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the table ID encoded into every [`NodeId`] from this table.
    #[must_use]
    pub fn table_id(&self) -> u16 {
        self.schema.table_id
    }

    /// Returns the label shared by all nodes in this table.
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

    /// Returns the decoded property value at the given row offset.
    ///
    /// Returns `None` if the column does not exist or the offset is out of bounds.
    #[must_use]
    pub fn get_property(&self, offset: usize, key: &PropertyKey) -> Option<Value> {
        self.columns.get(key)?.get(offset)
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

    /// Returns the raw `u64` stored at the given row offset for a bit-packed column.
    ///
    /// This is primarily useful for foreign-key columns where the raw encoded ID
    /// is needed rather than the `Value::Int64` conversion. Returns `None` for
    /// non-[`BitPacked`](ColumnCodec::BitPacked) columns or out-of-bounds offsets.
    #[must_use]
    pub fn get_raw_u64(&self, offset: usize, key: &PropertyKey) -> Option<u64> {
        self.columns.get(key)?.get_raw_u64(offset)
    }

    /// Returns the zone map for a column, if one exists.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<&ZoneMap> {
        self.zone_maps.get(key)
    }

    /// Returns the column codec for a property, if it exists.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<&ColumnCodec> {
        self.columns.get(key)
    }

    /// Returns all property keys present in this table.
    #[must_use]
    pub fn property_keys(&self) -> Vec<PropertyKey> {
        self.columns.keys().cloned().collect()
    }

    /// Returns an estimate of heap memory used by all columns in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.columns.values().map(|c| c.heap_bytes()).sum()
    }

    /// Updates the table ID embedded in the schema.
    ///
    /// Used by incremental compaction when tables are re-indexed after
    /// cloning from an old compact store.
    pub(crate) fn set_table_id(&mut self, table_id: u16) {
        self.schema.table_id = table_id;
    }

    /// Returns the schema for this table.
    #[must_use]
    pub(crate) fn schema(&self) -> &TableSchema {
        &self.schema
    }

    /// Returns the columns map for this table.
    #[must_use]
    pub(crate) fn columns(&self) -> &FxHashMap<PropertyKey, ColumnCodec> {
        &self.columns
    }

    /// Returns the zone maps for this table.
    #[must_use]
    pub(crate) fn zone_maps(&self) -> &FxHashMap<PropertyKey, ZoneMap> {
        &self.zone_maps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::BitPackedInts;
    use crate::graph::compact::id::decode_node_id;
    use crate::graph::compact::schema::{ColumnDef, ColumnType};

    /// Helper: build a `NodeTable` with 5 rows and two bit-packed columns
    /// ("rating" at 4 bits, "count" at 32 bits).
    fn sample_table() -> NodeTable {
        let schema = TableSchema::new(
            "Movie",
            3,
            vec![
                ColumnDef::new("rating", ColumnType::UInt { bits: 4 }),
                ColumnDef::new("count", ColumnType::UInt { bits: 32 }),
            ],
        );

        let ratings = vec![1u64, 5, 10, 15, 3];
        let counts = vec![100u64, 200, 300, 400, 500];

        let mut columns = FxHashMap::default();
        columns.insert(
            PropertyKey::new("rating"),
            ColumnCodec::BitPacked(BitPackedInts::pack(&ratings)),
        );
        columns.insert(
            PropertyKey::new("count"),
            ColumnCodec::BitPacked(BitPackedInts::pack(&counts)),
        );

        let mut zone_maps = FxHashMap::default();
        zone_maps.insert(
            PropertyKey::new("rating"),
            ZoneMap {
                min: Some(Value::Int64(1)),
                max: Some(Value::Int64(15)),
                null_count: 0,
                row_count: 5,
            },
        );
        zone_maps.insert(
            PropertyKey::new("count"),
            ZoneMap {
                min: Some(Value::Int64(100)),
                max: Some(Value::Int64(500)),
                null_count: 0,
                row_count: 5,
            },
        );

        NodeTable::from_columns(schema, columns, zone_maps, 5)
    }

    #[test]
    fn test_len_and_label() {
        let table = sample_table();
        assert_eq!(table.len(), 5);
        assert!(!table.is_empty());
        assert_eq!(table.table_id(), 3);
        assert_eq!(table.label(), "Movie");
    }

    #[test]
    fn test_empty_table() {
        let schema = TableSchema::new("Empty", 0, vec![]);
        let table = NodeTable::new(schema);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert!(table.node_ids().is_empty());
    }

    #[test]
    fn test_node_ids() {
        let table = sample_table();
        let ids = table.node_ids();
        assert_eq!(ids.len(), 5);

        for (i, id) in ids.iter().enumerate() {
            let (tid, offset) = decode_node_id(*id);
            assert_eq!(tid, 3);
            assert_eq!(offset, i as u64);
        }
    }

    #[test]
    fn test_get_property() {
        let table = sample_table();

        // First row
        assert_eq!(
            table.get_property(0, &PropertyKey::new("rating")),
            Some(Value::Int64(1))
        );
        assert_eq!(
            table.get_property(0, &PropertyKey::new("count")),
            Some(Value::Int64(100))
        );

        // Last row
        assert_eq!(
            table.get_property(4, &PropertyKey::new("rating")),
            Some(Value::Int64(3))
        );
        assert_eq!(
            table.get_property(4, &PropertyKey::new("count")),
            Some(Value::Int64(500))
        );
    }

    #[test]
    fn test_get_all_properties() {
        let table = sample_table();
        let props = table.get_all_properties(2);

        assert_eq!(props.len(), 2);
        assert_eq!(props[&PropertyKey::new("rating")], Value::Int64(10));
        assert_eq!(props[&PropertyKey::new("count")], Value::Int64(300));
    }

    #[test]
    fn test_get_raw_u64() {
        let table = sample_table();

        // Raw u64 is useful for FK lookups: verify it returns the original packed value.
        assert_eq!(table.get_raw_u64(0, &PropertyKey::new("count")), Some(100));
        assert_eq!(table.get_raw_u64(3, &PropertyKey::new("count")), Some(400));
        assert_eq!(table.get_raw_u64(4, &PropertyKey::new("rating")), Some(3));
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let table = sample_table();

        // Offset beyond table length.
        assert_eq!(table.get_property(5, &PropertyKey::new("rating")), None);
        assert_eq!(table.get_property(999, &PropertyKey::new("count")), None);
        assert_eq!(table.get_raw_u64(5, &PropertyKey::new("count")), None);

        // Non-existent property key.
        assert_eq!(table.get_property(0, &PropertyKey::new("missing")), None);
        assert_eq!(table.get_raw_u64(0, &PropertyKey::new("missing")), None);

        // get_all_properties on out-of-bounds offset returns empty map.
        let props = table.get_all_properties(100);
        assert!(props.is_empty());
    }

    #[test]
    fn test_zone_map_lookup() {
        let table = sample_table();

        let zm = table.zone_map(&PropertyKey::new("rating")).unwrap();
        assert_eq!(zm.min, Some(Value::Int64(1)));
        assert_eq!(zm.max, Some(Value::Int64(15)));
        assert_eq!(zm.row_count, 5);

        assert!(table.zone_map(&PropertyKey::new("missing")).is_none());
    }

    #[test]
    fn test_column_lookup() {
        let table = sample_table();

        assert!(table.column(&PropertyKey::new("rating")).is_some());
        assert!(table.column(&PropertyKey::new("count")).is_some());
        assert!(table.column(&PropertyKey::new("missing")).is_none());
    }

    #[test]
    fn test_property_keys() {
        let table = sample_table();
        let mut keys = table.property_keys();
        // Sort for deterministic assertion (hash map iteration order is unspecified).
        keys.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].as_ref(), "count");
        assert_eq!(keys[1].as_ref(), "rating");
    }
}
