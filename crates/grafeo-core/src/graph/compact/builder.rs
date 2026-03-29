//! Builder for constructing a `CompactStore` from raw data.
//!
//! The builder provides a fluent API for defining node tables, relationship
//! tables, and their columns. Data is loaded in bulk at construction time —
//! the resulting `CompactStore` is immutable (mutations go to the delta buffer).

use std::fmt;

use arcstr::ArcStr;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::CompactStore;
use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use crate::storage::{BitPackedInts, BitVector, DictionaryBuilder};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while building a [`CompactStore`].
#[derive(Debug, Clone)]
pub enum CompactStoreError {
    /// A relationship table references a node label that was not defined.
    LabelNotFound(String),
    /// A column was added with a length that does not match the table.
    ColumnLengthMismatch {
        /// Expected number of rows (inferred from the first column added).
        expected: usize,
        /// Actual number of rows in the column.
        got: usize,
    },
    /// Two node tables were defined with the same label.
    DuplicateLabel(String),
    /// A backward edge has no corresponding forward edge (data inconsistency).
    InconsistentEdgeData(String),
}

impl fmt::Display for CompactStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LabelNotFound(label) => {
                write!(f, "node label not found: {label:?}")
            }
            Self::ColumnLengthMismatch { expected, got } => {
                write!(
                    f,
                    "column length mismatch: expected {expected} rows, got {got}"
                )
            }
            Self::DuplicateLabel(label) => {
                write!(f, "duplicate node label: {label:?}")
            }
            Self::InconsistentEdgeData(msg) => {
                write!(f, "inconsistent edge data: {msg}")
            }
        }
    }
}

impl std::error::Error for CompactStoreError {}

// ---------------------------------------------------------------------------
// NodeTableBuilder
// ---------------------------------------------------------------------------

/// Builder for node table columns. Obtained through [`CompactStoreBuilder::node_table`].
pub struct NodeTableBuilder {
    label: ArcStr,
    columns: Vec<(PropertyKey, ColumnCodec)>,
    zone_maps: Vec<(PropertyKey, ZoneMap)>,
    len: Option<usize>,
    length_mismatch: Option<(usize, usize)>,
}

impl NodeTableBuilder {
    fn new(label: impl Into<ArcStr>) -> Self {
        Self {
            label: label.into(),
            columns: Vec::new(),
            zone_maps: Vec::new(),
            len: None,
            length_mismatch: None,
        }
    }

    /// Adds a bit-packed unsigned integer column.
    ///
    /// `bits` is the number of bits per value. Values are packed using
    /// [`BitPackedInts::pack_with_bits`].
    ///
    /// # Panics
    ///
    /// Panics if `bits > 63`. BitPacked columns are decoded via `as i64`, so
    /// values must fit in 63 bits to avoid sign-bit corruption.
    pub fn column_bitpacked(&mut self, name: &str, values: &[u64], bits: u8) -> &mut Self {
        assert!(
            bits <= 63,
            "BitPacked columns limited to 63 bits (values must fit in i64)"
        );
        self.record_len(values.len());

        let bp = BitPackedInts::pack_with_bits(values, bits);

        // Compute zone map from raw values.
        let zone_map = compute_zone_map_u64(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp)));
        self
    }

    /// Adds a dictionary-encoded string column.
    pub fn column_dict(&mut self, name: &str, values: &[&str]) -> &mut Self {
        self.record_len(values.len());

        let mut builder = DictionaryBuilder::new();
        for &v in values {
            builder.add(v);
        }
        let dict = builder.build();

        // Compute zone map for strings.
        let zone_map = compute_zone_map_strings(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::Dict(dict)));
        self
    }

    /// Adds an int8 quantised vector column (for embeddings).
    pub fn column_int8_vector(&mut self, name: &str, data: Vec<i8>, dimensions: u16) -> &mut Self {
        let row_count = if dimensions == 0 {
            0
        } else {
            data.len() / dimensions as usize
        };
        self.record_len(row_count);

        // No meaningful zone map for vector columns.
        self.columns.push((
            PropertyKey::new(name),
            ColumnCodec::Int8Vector { data, dimensions },
        ));
        self
    }

    /// Adds a boolean bitmap column.
    pub fn column_bitmap(&mut self, name: &str, values: &[bool]) -> &mut Self {
        self.record_len(values.len());

        let bv = BitVector::from_bools(values);

        // Zone map for booleans.
        let zone_map = compute_zone_map_bool(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::Bitmap(bv)));
        self
    }

    /// Adds a pre-built column codec (for advanced use).
    pub fn column(&mut self, name: &str, codec: ColumnCodec) -> &mut Self {
        self.record_len(codec.len());
        self.columns.push((PropertyKey::new(name), codec));
        self
    }

    /// Records the row count from the first column and validates subsequent ones.
    fn record_len(&mut self, col_len: usize) {
        match self.len {
            None => self.len = Some(col_len),
            Some(expected) => {
                if expected != col_len {
                    self.length_mismatch = Some((expected, col_len));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RelTableBuilder
// ---------------------------------------------------------------------------

/// Builder for relationship table edges and properties. Obtained through [`CompactStoreBuilder::rel_table`].
pub struct RelTableBuilder {
    edge_type: ArcStr,
    src_label: ArcStr,
    dst_label: ArcStr,
    edges: Vec<(u32, u32)>,
    backward: bool,
    properties: Vec<(PropertyKey, ColumnCodec)>,
}

impl RelTableBuilder {
    fn new(
        edge_type: impl Into<ArcStr>,
        src_label: impl Into<ArcStr>,
        dst_label: impl Into<ArcStr>,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            src_label: src_label.into(),
            dst_label: dst_label.into(),
            edges: Vec::new(),
            backward: false,
            properties: Vec::new(),
        }
    }

    /// Sets the `(src_offset, dst_offset)` edge pairs.
    pub fn edges(&mut self, pairs: impl Into<Vec<(u32, u32)>>) -> &mut Self {
        self.edges = pairs.into();
        self
    }

    /// Enables or disables backward CSR construction.
    pub fn backward(&mut self, enabled: bool) -> &mut Self {
        self.backward = enabled;
        self
    }

    /// Adds a bit-packed property column on edges.
    pub fn column_bitpacked(&mut self, name: &str, values: &[u64], bits: u8) -> &mut Self {
        let bp = BitPackedInts::pack_with_bits(values, bits);
        self.properties
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp)));
        self
    }
}

// ---------------------------------------------------------------------------
// CompactStoreBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`CompactStore`] from raw data.
///
/// # Example
///
/// ```ignore
/// let store = CompactStoreBuilder::new()
///     .node_table("Person", |t| {
///         t.column_bitpacked("age", &[25, 30, 35], 6)
///          .column_dict("name", &["Alice", "Bob", "Charlie"])
///     })
///     .build()
///     .unwrap();
/// ```
#[derive(Default)]
pub struct CompactStoreBuilder {
    node_table_builders: Vec<NodeTableBuilder>,
    rel_table_builders: Vec<RelTableBuilder>,
}

impl CompactStoreBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Defines a node table with the given label.
    ///
    /// The closure receives a [`NodeTableBuilder`] that can be used to add
    /// columns.
    pub fn node_table(
        mut self,
        label: &str,
        f: impl FnOnce(&mut NodeTableBuilder) -> &mut NodeTableBuilder,
    ) -> Self {
        let mut builder = NodeTableBuilder::new(label);
        f(&mut builder);
        self.node_table_builders.push(builder);
        self
    }

    /// Defines a relationship table connecting two node labels.
    ///
    /// The closure receives a [`RelTableBuilder`] that can be used to set
    /// edges, backward CSR, and properties.
    pub fn rel_table(
        mut self,
        edge_type: &str,
        src_label: &str,
        dst_label: &str,
        f: impl FnOnce(&mut RelTableBuilder) -> &mut RelTableBuilder,
    ) -> Self {
        let mut builder = RelTableBuilder::new(edge_type, src_label, dst_label);
        f(&mut builder);
        self.rel_table_builders.push(builder);
        self
    }

    /// Consumes the builder and constructs a [`CompactStore`].
    ///
    /// # Errors
    ///
    /// Returns [`CompactStoreError::LabelNotFound`] if a relationship table
    /// references a node label that was not defined.
    pub fn build(self) -> Result<CompactStore, CompactStoreError> {
        // Step 1: Validate column length mismatches recorded during building.
        for ntb in &self.node_table_builders {
            if let Some((expected, got)) = ntb.length_mismatch {
                return Err(CompactStoreError::ColumnLengthMismatch { expected, got });
            }
        }

        // Step 2: Validate no duplicate labels.
        {
            let mut seen_labels = FxHashSet::default();
            for ntb in &self.node_table_builders {
                if !seen_labels.insert(&ntb.label) {
                    return Err(CompactStoreError::DuplicateLabel(ntb.label.to_string()));
                }
            }
        }

        // Step 3: Assign sequential table IDs.
        let mut label_to_table_id: FxHashMap<ArcStr, u16> = FxHashMap::default();
        let mut table_id_to_label: Vec<ArcStr> = Vec::new();

        for (idx, ntb) in self.node_table_builders.iter().enumerate() {
            let table_id = idx as u16;
            label_to_table_id.insert(ntb.label.clone(), table_id);
            table_id_to_label.push(ntb.label.clone());
        }

        // Step 4: Build each NodeTable.
        let mut node_tables_by_id: Vec<NodeTable> =
            Vec::with_capacity(self.node_table_builders.len());

        for (idx, ntb) in self.node_table_builders.into_iter().enumerate() {
            let table_id = idx as u16;
            let row_count = ntb.len.unwrap_or(0);

            // Build column definitions for the schema.
            let col_defs: Vec<ColumnDef> = ntb
                .columns
                .iter()
                .map(|(key, codec)| {
                    let col_type = infer_column_type(codec);
                    ColumnDef::new(key.as_str(), col_type)
                })
                .collect();

            let schema = TableSchema::new(ntb.label.as_str(), table_id, col_defs);

            let columns: FxHashMap<PropertyKey, ColumnCodec> = ntb.columns.into_iter().collect();

            let zone_maps: FxHashMap<PropertyKey, ZoneMap> = ntb.zone_maps.into_iter().collect();

            let table = NodeTable::from_columns(schema, columns, zone_maps, row_count);
            node_tables_by_id.push(table);
        }

        // Step 5: Build each RelTable.
        let mut rel_tables_by_id: Vec<RelTable> = Vec::with_capacity(self.rel_table_builders.len());
        let mut edge_type_to_rel_id: FxHashMap<ArcStr, u16> = FxHashMap::default();
        let mut rel_table_id_to_type: Vec<ArcStr> = Vec::new();

        for (idx, rtb) in self.rel_table_builders.into_iter().enumerate() {
            let rel_table_id = idx as u16;
            rel_table_id_to_type.push(rtb.edge_type.clone());

            // Resolve labels to table IDs.
            let src_table_id = *label_to_table_id
                .get(&rtb.src_label)
                .ok_or_else(|| CompactStoreError::LabelNotFound(rtb.src_label.to_string()))?;
            let dst_table_id = *label_to_table_id
                .get(&rtb.dst_label)
                .ok_or_else(|| CompactStoreError::LabelNotFound(rtb.dst_label.to_string()))?;

            // Get source and destination node counts for CSR sizing.
            let src_node_count = node_tables_by_id
                .get(src_table_id as usize)
                .map_or(0, |t| t.len());
            let dst_node_count = node_tables_by_id
                .get(dst_table_id as usize)
                .map_or(0, |t| t.len());

            // Sort edges by source for forward CSR.
            let mut fwd_edges = rtb.edges.clone();
            fwd_edges.sort_by_key(|&(src, _dst)| src);
            let fwd = CsrAdjacency::from_sorted_edges(src_node_count, &fwd_edges);

            // Optionally build backward CSR + pre-compute bwd-to-fwd position mapping.
            let bwd =
                if rtb.backward {
                    let mut bwd_edges: Vec<(u32, u32)> =
                        rtb.edges.iter().map(|&(src, dst)| (dst, src)).collect();
                    bwd_edges.sort_by_key(|&(dst, _src)| dst);
                    let mut bwd_csr = CsrAdjacency::from_sorted_edges(dst_node_count, &bwd_edges);

                    // For each backward edge (dst -> src), find the forward CSR position
                    // of the corresponding (src -> dst) edge. This eliminates the O(degree)
                    // linear scan in edges_to_target at query time.
                    let mut mapping = Vec::with_capacity(bwd_edges.len());
                    for &(dst, src) in &bwd_edges {
                        let fwd_neighbors = fwd.neighbors(src);
                        let fwd_start = fwd.offset_of(src);
                        let local_idx = fwd_neighbors.iter().position(|&t| t == dst).ok_or_else(
                            || {
                                CompactStoreError::InconsistentEdgeData(format!(
                                    "backward edge ({dst}->{src}) has no corresponding forward edge"
                                ))
                            },
                        )?;
                        mapping.push(fwd_start + local_idx as u32);
                    }
                    bwd_csr.set_edge_data(mapping);

                    Some(bwd_csr)
                } else {
                    None
                };

            // Build edge property columns.
            let property_col_defs: Vec<ColumnDef> = rtb
                .properties
                .iter()
                .map(|(key, codec)| {
                    let col_type = infer_column_type(codec);
                    ColumnDef::new(key.as_str(), col_type)
                })
                .collect();

            let schema = EdgeSchema::new(
                rtb.edge_type.as_str(),
                rel_table_id,
                rtb.src_label.as_str(),
                rtb.dst_label.as_str(),
                property_col_defs,
            );

            let properties: FxHashMap<PropertyKey, ColumnCodec> =
                rtb.properties.into_iter().collect();

            let table = RelTable::new(schema, fwd, bwd, properties, src_table_id, dst_table_id);
            edge_type_to_rel_id.insert(rtb.edge_type.clone(), rel_table_id);
            rel_tables_by_id.push(table);
        }

        // Step 6: Compute initial Statistics.
        let mut stats = Statistics::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        for (idx, nt) in node_tables_by_id.iter().enumerate() {
            let count = nt.len() as u64;
            total_nodes += count;
            let label = &table_id_to_label[idx];
            stats.update_label(label.as_str(), LabelStatistics::new(count));
        }

        for (idx, rt) in rel_tables_by_id.iter().enumerate() {
            let count = rt.num_edges() as u64;
            total_edges += count;
            let edge_type = &rel_table_id_to_type[idx];
            stats.update_edge_type(edge_type.as_str(), EdgeTypeStatistics::new(count, 0.0, 0.0));
        }

        stats.total_nodes = total_nodes;
        stats.total_edges = total_edges;

        // Step 7: Construct the CompactStore.
        Ok(CompactStore::new(
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            stats,
        ))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Infers a [`ColumnType`] from a [`ColumnCodec`] variant.
fn infer_column_type(codec: &ColumnCodec) -> ColumnType {
    match codec {
        ColumnCodec::BitPacked(bp) => ColumnType::UInt {
            bits: bp.bits_per_value(),
        },
        ColumnCodec::Dict(_) => ColumnType::DictString,
        ColumnCodec::Bitmap(_) => ColumnType::Bool,
        ColumnCodec::Int8Vector { dimensions, .. } => ColumnType::Int8Vector {
            dimensions: *dimensions,
        },
    }
}

/// Computes a zone map from u64 values (bit-packed column).
///
/// If the maximum value exceeds `i64::MAX`, the zone map is returned without
/// min/max bounds (conservative — won't prune). This avoids incorrect ordering
/// comparisons caused by the `u64 as i64` sign-bit wrap.
fn compute_zone_map_u64(values: &[u64]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    if max > i64::MAX as u64 {
        // Values exceed i64 range — zone map would compare with wrong ordering.
        // Return conservative (no bounds) zone map.
        return ZoneMap {
            row_count: values.len(),
            ..ZoneMap::default()
        };
    }
    ZoneMap {
        min: Some(Value::Int64(min as i64)),
        max: Some(Value::Int64(max as i64)),
        null_count: 0,
        row_count: values.len(),
    }
}

/// Computes a zone map from string values (dict column).
fn compute_zone_map_strings(values: &[&str]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    ZoneMap {
        min: Some(Value::from(min)),
        max: Some(Value::from(max)),
        null_count: 0,
        row_count: values.len(),
    }
}

/// Computes a zone map from boolean values.
fn compute_zone_map_bool(values: &[bool]) -> ZoneMap {
    if values.is_empty() {
        return ZoneMap::new();
    }
    let has_false = values.iter().any(|&v| !v);
    let has_true = values.iter().any(|&v| v);
    let min = !has_false; // false if has_false, true if all true
    let max = has_true; // true if has_true, false if all false
    ZoneMap {
        min: Some(Value::Bool(min)),
        max: Some(Value::Bool(max)),
        null_count: 0,
        row_count: values.len(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphStore;

    #[test]
    fn test_builder_basic() {
        let store = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35, 40, 45], 6)
                    .column_dict("name", &["Alice", "Bob", "Charlie", "Diana", "Eve"])
            })
            .build()
            .unwrap();

        // Verify we can query it.
        let ids = store.nodes_by_label("Person");
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_builder_with_edges() {
        let store = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("val", &[1, 2, 3], 4))
            .node_table("B", |t| t.column_bitpacked("val", &[10, 20], 8))
            .rel_table("LINKS", "A", "B", |r| {
                r.edges([(0, 0), (0, 1), (1, 0), (2, 1)]).backward(true)
            })
            .build()
            .unwrap();

        let a_ids = store.nodes_by_label("A");
        assert_eq!(a_ids.len(), 3);
        let b_ids = store.nodes_by_label("B");
        assert_eq!(b_ids.len(), 2);
    }

    #[test]
    fn test_builder_label_not_found() {
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("val", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| {
                // "B" doesn't exist
                r.edges([(0, 0)])
            })
            .build();

        assert!(result.is_err());
    }
}
