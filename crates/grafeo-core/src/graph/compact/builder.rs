//! Builder for constructing a [`CompactStore`] from raw data.
//!
//! The builder provides a fluent API for defining node tables, relationship
//! tables, and their columns. Data is loaded in bulk at construction time,
//! producing an immutable, read-only store.

use arcstr::ArcStr;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use super::CompactStore;
use super::column::{ColumnCodec, CompactColumn};
use super::csr::CsrAdjacency;
use super::id::MAX_TABLE_ID;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::codec::{BitPackedInts, BitVector, DictionaryBuilder};
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while building a [`CompactStore`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum CompactStoreError {
    /// A relationship table references a node label that was not defined.
    #[error("node label not found: {0:?}")]
    LabelNotFound(String),
    /// A column was added with a length that does not match the table.
    #[error("column length mismatch: expected {expected} rows, got {got}")]
    ColumnLengthMismatch {
        /// Expected number of rows (inferred from the first column added).
        expected: usize,
        /// Actual number of rows in the column.
        got: usize,
    },
    /// Two node tables were defined with the same label.
    #[error("duplicate node label: {0:?}")]
    DuplicateLabel(String),
    /// Two relationship tables were defined with the same (edge type, src, dst) triple.
    #[error("duplicate edge type: {0:?}")]
    DuplicateEdgeType(String),
    /// A backward edge has no corresponding forward edge (data inconsistency).
    #[error("inconsistent edge data: {0}")]
    InconsistentEdgeData(String),
    /// A bit-packed column contains a value that exceeds `i64::MAX`.
    #[error("value overflow in column {column:?}: {value} exceeds i64::MAX ({max})")]
    ValueOverflow {
        /// Column name.
        column: String,
        /// The offending value.
        value: u64,
        /// Maximum allowed value.
        max: u64,
    },
    /// The number of tables exceeds the compact ID encoding limit (15-bit table ID).
    #[error("table count {count} exceeds compact ID limit of {max} ({kind} tables)")]
    TableCountOverflow {
        /// Kind of table ("node" or "relationship").
        kind: &'static str,
        /// Actual table count.
        count: usize,
        /// Maximum allowed table count.
        max: u16,
    },
}

// ---------------------------------------------------------------------------
// NodeTableBuilder
// ---------------------------------------------------------------------------

/// Builder for node table columns. Obtained through [`CompactStoreBuilder::node_table`].
pub struct NodeTableBuilder {
    label: ArcStr,
    columns: Vec<(PropertyKey, CompactColumn)>,
    zone_maps: Vec<(PropertyKey, ZoneMap)>,
    len: Option<usize>,
    length_mismatch: Option<(usize, usize)>,
    value_overflow: Option<(String, u64)>,
}

impl NodeTableBuilder {
    fn new(label: impl Into<ArcStr>) -> Self {
        Self {
            label: label.into(),
            columns: Vec::new(),
            zone_maps: Vec::new(),
            len: None,
            length_mismatch: None,
            value_overflow: None,
        }
    }

    /// Adds a bit-packed integer column.
    ///
    /// `bits` is the number of bits per value. Values are packed using
    /// [`BitPackedInts::pack_with_bits`]. All values must fit in `i64`
    /// (i.e., be at most `i64::MAX`); overflow is recorded and reported
    /// as [`CompactStoreError::ValueOverflow`] at build time.
    pub fn column_bitpacked(&mut self, name: &str, values: &[u64], bits: u8) -> &mut Self {
        self.record_len(values.len());

        // Validate that all values fit in i64.
        if let Some(&bad) = values.iter().find(|&&v| v > i64::MAX as u64) {
            self.value_overflow = Some((name.to_string(), bad));
        }

        let bp = BitPackedInts::pack_with_bits(values, bits);

        // Compute zone map from raw values.
        let zone_map = compute_zone_map_u64(values);
        self.zone_maps.push((PropertyKey::new(name), zone_map));

        self.columns
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp).into()));
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
            .push((PropertyKey::new(name), ColumnCodec::Dict(dict).into()));
        self
    }

    /// Adds an int8 quantised vector column (for embeddings).
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `dimensions`.
    pub fn column_int8_vector(&mut self, name: &str, data: Vec<i8>, dimensions: u16) -> &mut Self {
        let dims = dimensions as usize;
        let row_count = if dims == 0 {
            0
        } else {
            assert!(
                data.len().is_multiple_of(dims),
                "Int8Vector data length {} is not a multiple of dimensions {dimensions}",
                data.len(),
            );
            data.len() / dims
        };
        self.record_len(row_count);

        // No meaningful zone map for vector columns.
        self.columns.push((
            PropertyKey::new(name),
            ColumnCodec::int8_vector(data, dimensions).into(),
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
            .push((PropertyKey::new(name), ColumnCodec::Bitmap(bv).into()));
        self
    }

    /// Adds a pre-built column codec (for advanced use).
    pub fn column(&mut self, name: &str, codec: ColumnCodec) -> &mut Self {
        self.record_len(codec.len());
        self.columns.push((PropertyKey::new(name), codec.into()));
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
    properties: Vec<(PropertyKey, CompactColumn)>,
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
            .push((PropertyKey::new(name), ColumnCodec::BitPacked(bp).into()));
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
///          .column_dict("name", &["Alix", "Gus", "Vincent"])
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
        // Step 1: Validate column length mismatches and value overflows.
        for ntb in &self.node_table_builders {
            if let Some((expected, got)) = ntb.length_mismatch {
                return Err(CompactStoreError::ColumnLengthMismatch { expected, got });
            }
            if let Some((ref column, value)) = ntb.value_overflow {
                return Err(CompactStoreError::ValueOverflow {
                    column: column.clone(),
                    max: i64::MAX as u64,
                    value,
                });
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

        // Step 2b: Validate no duplicate (edge_type, src_label, dst_label) triples.
        {
            let mut seen_triples = FxHashSet::default();
            for rtb in &self.rel_table_builders {
                if !seen_triples.insert((&rtb.edge_type, &rtb.src_label, &rtb.dst_label)) {
                    return Err(CompactStoreError::DuplicateEdgeType(format!(
                        "{} ({} -> {})",
                        rtb.edge_type, rtb.src_label, rtb.dst_label
                    )));
                }
            }
        }

        // Step 2c: Validate table counts fit within the 15-bit compact ID encoding.
        let max_tables = usize::from(MAX_TABLE_ID) + 1; // 32768
        if self.node_table_builders.len() > max_tables {
            return Err(CompactStoreError::TableCountOverflow {
                kind: "node",
                count: self.node_table_builders.len(),
                max: MAX_TABLE_ID,
            });
        }
        if self.rel_table_builders.len() > max_tables {
            return Err(CompactStoreError::TableCountOverflow {
                kind: "relationship",
                count: self.rel_table_builders.len(),
                max: MAX_TABLE_ID,
            });
        }

        // Step 3: Assign sequential table IDs.
        let mut label_to_table_id: FxHashMap<ArcStr, u16> = FxHashMap::default();
        let mut table_id_to_label: Vec<ArcStr> = Vec::new();

        for (idx, ntb) in self.node_table_builders.iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "node",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
            label_to_table_id.insert(ntb.label.clone(), table_id);
            table_id_to_label.push(ntb.label.clone());
        }

        // Step 4: Build each NodeTable.
        let mut node_tables_by_id: Vec<NodeTable> =
            Vec::with_capacity(self.node_table_builders.len());

        for (idx, ntb) in self.node_table_builders.into_iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "node",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
            let row_count = ntb.len.unwrap_or(0);

            // Build column definitions for the schema.
            let col_defs: Vec<ColumnDef> = ntb
                .columns
                .iter()
                .map(|(key, column)| {
                    let col_type = infer_column_type(column.codec());
                    ColumnDef::new(key.as_str(), col_type)
                })
                .collect();

            let schema = TableSchema::new(ntb.label.as_str(), table_id, col_defs);

            let columns: FxHashMap<PropertyKey, CompactColumn> = ntb.columns.into_iter().collect();

            let zone_maps: FxHashMap<PropertyKey, ZoneMap> = ntb.zone_maps.into_iter().collect();

            // Phase 2c: compute per-block zone maps so range scans (Phase 4)
            // can skip entire blocks whose stats prove no match. Empty
            // when a column is empty; otherwise one entry per block.
            let block_zone_maps: FxHashMap<PropertyKey, Vec<ZoneMap>> = columns
                .iter()
                .map(|(key, column)| {
                    (
                        key.clone(),
                        super::zone_map::compute_block_zone_maps(column),
                    )
                })
                .collect();

            let table = NodeTable::from_columns_with_block_stats(
                schema,
                columns,
                zone_maps,
                block_zone_maps,
                row_count,
            );
            node_tables_by_id.push(table);
        }

        // Step 5: Build each RelTable.
        let mut rel_tables_by_id: Vec<RelTable> = Vec::with_capacity(self.rel_table_builders.len());
        let mut edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>> = FxHashMap::default();
        let mut rel_table_id_to_type: Vec<ArcStr> = Vec::new();

        for (idx, rtb) in self.rel_table_builders.into_iter().enumerate() {
            // Validated in Step 2c: count <= MAX_TABLE_ID + 1, so idx fits u16.
            let rel_table_id =
                u16::try_from(idx).map_err(|_| CompactStoreError::TableCountOverflow {
                    kind: "relationship",
                    count: idx,
                    max: MAX_TABLE_ID,
                })?;
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
                        // reason: local index within CSR neighbors fits u32
                        #[allow(clippy::cast_possible_truncation)]
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
                .map(|(key, column)| {
                    let col_type = infer_column_type(column.codec());
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

            let properties: FxHashMap<PropertyKey, CompactColumn> =
                rtb.properties.into_iter().collect();

            let table = RelTable::new(schema, fwd, bwd, properties, src_table_id, dst_table_id);
            edge_type_to_rel_id
                .entry(rtb.edge_type.clone())
                .or_default()
                .push(rel_table_id);
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

        let mut edge_type_counts: FxHashMap<&str, u64> = FxHashMap::default();
        for (idx, rt) in rel_tables_by_id.iter().enumerate() {
            let count = rt.num_edges() as u64;
            total_edges += count;
            let edge_type = &rel_table_id_to_type[idx];
            *edge_type_counts.entry(edge_type.as_str()).or_default() += count;
        }
        for (edge_type, count) in edge_type_counts {
            stats.update_edge_type(edge_type, EdgeTypeStatistics::new(count, 0.0, 0.0));
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
        ColumnCodec::Float64(_) => ColumnType::Float64,
        ColumnCodec::Float32Vector { dimensions, .. } => ColumnType::Float32Vector {
            dimensions: *dimensions,
        },
        ColumnCodec::RawI64(_) => ColumnType::Int64,
    }
}

/// Computes a zone map from u64 values (bit-packed column).
///
/// If the maximum value exceeds `i64::MAX`, the zone map is returned without
/// min/max bounds (conservative, won't prune). This avoids incorrect ordering
/// comparisons caused by the `u64 as i64` sign-bit wrap.
fn compute_zone_map_u64(values: &[u64]) -> ZoneMap {
    let Some(&min) = values.iter().min() else {
        return ZoneMap::new();
    };
    let max = *values.iter().max().expect("non-empty after min check");
    if max > i64::MAX as u64 {
        // Values exceed i64 range: zone map would compare with wrong ordering.
        // Return conservative (no bounds) zone map.
        return ZoneMap {
            row_count: values.len(),
            ..ZoneMap::default()
        };
    }
    // reason: max <= i64::MAX checked above, min <= max
    #[allow(clippy::cast_possible_wrap)]
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
// Conversion from GraphStore
// ---------------------------------------------------------------------------

/// Which columnar encoding to use for a property key, inferred from values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredType {
    /// All non-null values are `Value::Int64` with value >= 0.
    BitPacked,
    /// All non-null values are `Value::Int64`, with at least one negative.
    /// Uses the `ColumnCodec::RawI64` encoding so signed ordering works in
    /// `find_eq`, `find_in_range`, and zone-map comparisons, and the
    /// `Int64` type is preserved on decode.
    RawI64,
    /// All non-null values are `Value::Float64`, or mixed `Int64`+`Float64`.
    Float64,
    /// All non-null values are `Value::Bool`.
    Bitmap,
    /// All non-null values are `Value::Vector` with consistent dimensions.
    Float32Vector { dimensions: u16 },
    /// All non-null values are `Value::String`, or mixed/unsupported types.
    Dict,
}

/// Converts any [`GraphStore`](crate::graph::GraphStore) into a [`CompactStore`].
///
/// Reads all nodes grouped by label, infers column types from property values,
/// reads all edges grouped by type, and builds a `CompactStore` with backward
/// CSR enabled for every relationship table.
///
/// # Type mapping
///
/// | Source type | Codec | Notes |
/// |-------------|-------|-------|
/// | `Int64` (>= 0) | `BitPacked` | Auto bit-width via `BitPackedInts::pack` |
/// | `Bool` | `Bitmap` | |
/// | `String` | `Dict` | |
/// | All others | `Dict` | Serialized via `Display` |
///
/// Nodes with multiple labels use a canonical combined key (labels sorted,
/// joined with `|`). `Null` values are stored as zero/false/empty-string
/// depending on the inferred codec.
///
/// # Errors
///
/// Propagates any [`CompactStoreError`] from the underlying builder (e.g.
/// if there are more than 32,767 distinct labels or edge types).
pub fn from_graph_store(
    store: &dyn crate::graph::traits::GraphStore,
) -> Result<CompactStore, CompactStoreError> {
    // Step 1: Collect all nodes grouped by label, build ID mapping.
    let labels = store.all_labels();
    if labels.is_empty() {
        return CompactStoreBuilder::new().build();
    }

    // old_node_id -> (label_key, offset_within_label)
    let mut id_map: FxHashMap<grafeo_common::types::NodeId, (ArcStr, u32)> = FxHashMap::default();

    // label_key -> (ordered node IDs, property_key -> Vec<Value>)
    // We use Vec<Value> to collect per-column values in row order.
    let mut label_data: Vec<(
        ArcStr,
        Vec<grafeo_common::types::NodeId>,
        FxHashMap<PropertyKey, Vec<Value>>,
    )> = Vec::new();

    // Collect all node IDs per label. Nodes with multiple labels use a
    // compound key (sorted labels joined with "|").
    let mut seen_node_ids: FxHashSet<grafeo_common::types::NodeId> = FxHashSet::default();
    let mut label_key_index: FxHashMap<ArcStr, usize> = FxHashMap::default();

    for label in &labels {
        let node_ids = store.nodes_by_label(label);
        for &nid in &node_ids {
            if !seen_node_ids.insert(nid) {
                continue; // already assigned via an earlier label
            }

            // Get the node to check its full label set.
            let Some(node) = store.get_node(nid) else {
                continue;
            };

            let label_key: ArcStr = if node.labels.len() <= 1 {
                ArcStr::from(label.as_str())
            } else {
                let mut sorted: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
                sorted.sort_unstable();
                ArcStr::from(sorted.join("|"))
            };

            // Find or create the label_data entry.
            let entry_idx = if let Some(&idx) = label_key_index.get(&label_key) {
                idx
            } else {
                let idx = label_data.len();
                label_key_index.insert(label_key.clone(), idx);
                label_data.push((label_key.clone(), Vec::new(), FxHashMap::default()));
                idx
            };

            let (_, ref mut node_ids_vec, ref mut props_map) = label_data[entry_idx];
            // reason: node offset within a table fits u32
            #[allow(clippy::cast_possible_truncation)]
            let offset = node_ids_vec.len() as u32;
            node_ids_vec.push(nid);
            id_map.insert(nid, (label_key, offset));

            // Collect properties.
            for (key, value) in node.properties.iter() {
                let col = props_map
                    .entry(key.clone())
                    .or_insert_with(|| vec![Value::Null; offset as usize]);
                // Pad with nulls if this key appeared for the first time.
                while col.len() < offset as usize {
                    col.push(Value::Null);
                }
                col.push(value.clone());
            }

            // Pad all existing columns that this node didn't have.
            let expected_len = offset as usize + 1;
            for col in props_map.values_mut() {
                while col.len() < expected_len {
                    col.push(Value::Null);
                }
            }
        }
    }

    // Step 2: Infer column types and build CompactStoreBuilder.
    let mut builder = CompactStoreBuilder::new();

    for (label_key, node_ids_for_label, props_map) in &label_data {
        let node_count = node_ids_for_label.len();
        builder = builder.node_table(label_key.as_str(), |t| {
            // Ensure row count is set even when there are no properties.
            t.record_len(node_count);
            for (key, values) in props_map {
                let (column, zone_map) = encode_column(values);
                if let Some(zone_map) = zone_map {
                    t.zone_maps.push((key.clone(), zone_map));
                }
                t.record_len(column.len());
                t.columns.push((key.clone(), column));
            }
            t
        });
    }

    // Step 3: Collect all edges in a single pass, grouped by (edge_type, src_label, dst_label).
    // Key: (edge_type, src_label_key, dst_label_key) -> Vec<(src_offset, dst_offset)>
    type EdgeGroupKey = (ArcStr, ArcStr, ArcStr);
    let mut edge_groups: FxHashMap<EdgeGroupKey, Vec<(u32, u32)>> = FxHashMap::default();
    let mut edge_props_groups: FxHashMap<EdgeGroupKey, FxHashMap<PropertyKey, Vec<Value>>> =
        FxHashMap::default();

    // Iterate all nodes and their outgoing edges.
    for (_label_key, node_ids, _) in &label_data {
        for &nid in node_ids {
            let outgoing = store.edges_from(nid, crate::graph::Direction::Outgoing);
            for (_target_nid, edge_id) in outgoing {
                let Some(edge) = store.get_edge(edge_id) else {
                    continue;
                };

                let Some((src_label, src_offset)) = id_map.get(&edge.src) else {
                    continue;
                };
                let Some((dst_label, dst_offset)) = id_map.get(&edge.dst) else {
                    continue;
                };

                let group_key: EdgeGroupKey =
                    (edge.edge_type.clone(), src_label.clone(), dst_label.clone());

                let edges_vec = edge_groups.entry(group_key.clone()).or_default();
                let edge_idx = edges_vec.len();
                edges_vec.push((*src_offset, *dst_offset));

                // Collect edge properties.
                if !edge.properties.is_empty() {
                    let props = edge_props_groups.entry(group_key).or_default();
                    for (key, value) in edge.properties.iter() {
                        let col = props
                            .entry(key.clone())
                            .or_insert_with(|| vec![Value::Null; edge_idx]);
                        while col.len() < edge_idx {
                            col.push(Value::Null);
                        }
                        col.push(value.clone());
                    }
                    let expected_len = edge_idx + 1;
                    for col in props.values_mut() {
                        while col.len() < expected_len {
                            col.push(Value::Null);
                        }
                    }
                }
            }
        }
    }

    // Step 4: Add relationship tables to the builder.
    for ((edge_type, src_label, dst_label), edges) in &edge_groups {
        let edge_props =
            edge_props_groups.get(&(edge_type.clone(), src_label.clone(), dst_label.clone()));

        builder = builder.rel_table(
            edge_type.as_str(),
            src_label.as_str(),
            dst_label.as_str(),
            |r| {
                r.edges(edges.clone()).backward(true);

                // Add edge property columns.
                if let Some(props) = edge_props {
                    for (key, values) in props {
                        let (column, _) = encode_column(values);
                        r.properties.push((key.clone(), column));
                    }
                }

                r
            },
        );
    }

    builder.build()
}

/// Builds a [`CompactStore`] from any [`GraphStore`](crate::graph::GraphStore) with original ID preservation.
///
/// Same columnar conversion as [`from_graph_store`], but the resulting store
/// keeps a bidirectional mapping between the original `NodeId`/`EdgeId` values
/// and the internal compact positions. This enables layered storage where an
/// overlay store shares the same ID namespace.
///
/// # Errors
///
/// Same as [`from_graph_store`].
pub fn from_graph_store_preserving_ids(
    store: &dyn crate::graph::traits::GraphStore,
) -> Result<CompactStore, CompactStoreError> {
    let mut compact = from_graph_store(store)?;

    // ── Build node ID maps (replicate the label grouping logic) ────

    let labels = store.all_labels();
    if labels.is_empty() {
        compact.set_id_maps(
            FxHashMap::default(),
            FxHashMap::default(),
            Vec::new(),
            Vec::new(),
        );
        return Ok(compact);
    }

    let mut node_id_map: FxHashMap<grafeo_common::types::NodeId, (u16, u64)> = FxHashMap::default();
    let num_tables = compact.node_tables_by_id.len();
    let mut node_offset_to_id: Vec<Vec<grafeo_common::types::NodeId>> =
        vec![Vec::new(); num_tables];

    // Track per-label-key offset counters (same order as from_graph_store step 1).
    let mut seen: FxHashSet<grafeo_common::types::NodeId> = FxHashSet::default();
    let mut label_key_offsets: FxHashMap<ArcStr, u32> = FxHashMap::default();

    for label in &labels {
        let node_ids = store.nodes_by_label(label);
        for &nid in &node_ids {
            if !seen.insert(nid) {
                continue;
            }
            let Some(node) = store.get_node(nid) else {
                continue;
            };

            let label_key: ArcStr = if node.labels.len() <= 1 {
                ArcStr::from(label.as_str())
            } else {
                let mut sorted: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
                sorted.sort_unstable();
                ArcStr::from(sorted.join("|"))
            };

            let offset = label_key_offsets.entry(label_key.clone()).or_insert(0);
            let current_offset = *offset;
            *offset += 1;

            if let Some(&table_id) = compact.label_to_table_id.get(&label_key) {
                node_id_map.insert(nid, (table_id, u64::from(current_offset)));
                if let Some(rev) = node_offset_to_id.get_mut(table_id as usize) {
                    // Extend if needed (offsets should be sequential).
                    while rev.len() <= current_offset as usize {
                        rev.push(grafeo_common::types::NodeId::INVALID);
                    }
                    rev[current_offset as usize] = nid;
                }
            }
        }
    }

    // ── Build edge ID maps ─────────────────────────────────────────

    // Build (edge_type, src_table_id, dst_table_id) -> rel_table_id lookup.
    type RelKey = (ArcStr, u16, u16);
    let mut rel_key_to_id: FxHashMap<RelKey, u16> = FxHashMap::default();
    for (idx, rt) in compact.rel_tables_by_id.iter().enumerate() {
        let key = (rt.edge_type().clone(), rt.src_table_id(), rt.dst_table_id());
        let Ok(rel_id) = u16::try_from(idx) else {
            continue;
        };
        rel_key_to_id.insert(key, rel_id);
    }

    // Collect all edges grouped by (edge_type, src_table, dst_table), tracking
    // original EdgeId and (src_offset, dst_offset) for each.
    type EdgeGroupEntry = (grafeo_common::types::EdgeId, u32, u32); // (original_eid, src_off, dst_off)
    let mut edge_groups: FxHashMap<RelKey, Vec<EdgeGroupEntry>> = FxHashMap::default();

    let mut seen_edges: FxHashSet<grafeo_common::types::EdgeId> = FxHashSet::default();
    for &nid in node_id_map.keys() {
        let outgoing = store.edges_from(nid, crate::graph::Direction::Outgoing);
        for (_target_nid, edge_id) in outgoing {
            if !seen_edges.insert(edge_id) {
                continue;
            }
            let Some(edge) = store.get_edge(edge_id) else {
                continue;
            };
            let Some(&(src_tid, src_off)) = node_id_map.get(&edge.src) else {
                continue;
            };
            let Some(&(dst_tid, dst_off)) = node_id_map.get(&edge.dst) else {
                continue;
            };

            let key: RelKey = (edge.edge_type.clone(), src_tid, dst_tid);
            edge_groups.entry(key).or_default().push((
                edge_id,
                u32::try_from(src_off).unwrap_or(0),
                u32::try_from(dst_off).unwrap_or(0),
            ));
        }
    }

    // Sort each group by (src_offset, dst_offset) to match CSR construction order,
    // then build the edge_id_map from the resulting positions.
    let num_rel_tables = compact.rel_tables_by_id.len();
    let mut edge_id_map: FxHashMap<grafeo_common::types::EdgeId, (u16, u64)> = FxHashMap::default();
    let mut edge_offset_to_id: Vec<Vec<grafeo_common::types::EdgeId>> =
        vec![Vec::new(); num_rel_tables];

    for (key, mut entries) in edge_groups {
        let Some(&rel_table_id) = rel_key_to_id.get(&key) else {
            continue;
        };
        // Sort by (src_offset, dst_offset) to match CSR order.
        entries.sort_by_key(|&(_, src, dst)| (src, dst));

        let rev = &mut edge_offset_to_id[rel_table_id as usize];
        for (csr_pos, (original_eid, _src, _dst)) in entries.iter().enumerate() {
            edge_id_map.insert(*original_eid, (rel_table_id, csr_pos as u64));
            while rev.len() <= csr_pos {
                rev.push(grafeo_common::types::EdgeId::INVALID);
            }
            rev[csr_pos] = *original_eid;
        }
    }

    compact.set_id_maps(
        node_id_map,
        edge_id_map,
        node_offset_to_id,
        edge_offset_to_id,
    );
    Ok(compact)
}

/// Encodes one property column for [`from_graph_store`]: the values in the
/// codec their type calls for, and which rows have a value (a node or edge
/// without the property holds `Value::Null` here, and its row is marked
/// missing instead of holding the codec's empty value). Returns the column
/// and, for the types that keep one, its zone map.
fn encode_column(values: &[Value]) -> (CompactColumn, Option<ZoneMap>) {
    let present: Vec<bool> = values.iter().map(|v| !matches!(v, Value::Null)).collect();
    let (codec, zoned) = match infer_type_from_values(values) {
        InferredType::BitPacked => {
            let u64_values: Vec<u64> = values
                .iter()
                .map(|v| match v {
                    #[allow(
                        clippy::cast_sign_loss,
                        reason = "infer_type_from_values picks BitPacked only when no Int64 is negative"
                    )]
                    Value::Int64(n) => *n as u64,
                    _ => 0,
                })
                .collect();
            (
                ColumnCodec::BitPacked(BitPackedInts::pack(&u64_values)),
                true,
            )
        }
        InferredType::RawI64 => {
            let i64_values: Vec<i64> = values
                .iter()
                .map(|v| match v {
                    Value::Int64(n) => *n,
                    _ => 0,
                })
                .collect();
            (ColumnCodec::raw_i64(i64_values), true)
        }
        InferredType::Float64 => {
            let f64_values: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    Value::Float64(f) => *f,
                    Value::Int64(n) => *n as f64,
                    _ => 0.0,
                })
                .collect();
            (ColumnCodec::float64(f64_values), false)
        }
        InferredType::Float32Vector { dimensions } => {
            let mut flat: Vec<f32> = Vec::with_capacity(values.len() * dimensions as usize);
            for v in values {
                match v {
                    Value::Vector(vec) => flat.extend_from_slice(vec),
                    _ => flat.extend(std::iter::repeat_n(0.0f32, usize::from(dimensions))),
                }
            }
            (ColumnCodec::float32_vector(flat, dimensions), false)
        }
        InferredType::Bitmap => {
            let bool_values: Vec<bool> = values
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect();
            (
                ColumnCodec::Bitmap(BitVector::from_bools(&bool_values)),
                true,
            )
        }
        InferredType::Dict => {
            let mut dict_builder = DictionaryBuilder::new();
            for v in values {
                // A missing value takes the empty string; `present` masks it.
                match v {
                    Value::Null => dict_builder.add(""),
                    Value::String(s) => dict_builder.add(s.as_str()),
                    other => dict_builder.add(&format!("{other}")),
                };
            }
            (ColumnCodec::Dict(dict_builder.build()), true)
        }
    };
    let column = CompactColumn::with_present(codec, BitVector::from_bools(&present));
    let zone_map = zoned.then(|| super::zone_map::compute_zone_map(&column));
    (column, zone_map)
}

/// Infers the columnar encoding type from a slice of [`Value`]s.
///
/// Rules:
/// - If all non-null values are `Int64` with value >= 0, returns `BitPacked`.
/// - If all non-null values are `Bool`, returns `Bitmap`.
/// - Otherwise returns `Dict` (string fallback).
fn infer_type_from_values(values: &[Value]) -> InferredType {
    let mut saw_unsigned_int = false; // Value::Int64 with n >= 0
    let mut saw_signed_int = false; // Value::Int64 with n < 0
    let mut saw_float = false;
    let mut saw_bool = false;
    let mut saw_vector = false;
    let mut saw_other = false;
    let mut vector_dims: Option<u16> = None;

    for v in values {
        match v {
            Value::Null => {} // skip nulls
            Value::Int64(n) if *n >= 0 => saw_unsigned_int = true,
            Value::Int64(_) => saw_signed_int = true,
            Value::Float64(_) => saw_float = true,
            Value::Bool(_) => saw_bool = true,
            Value::Vector(vec) => {
                saw_vector = true;
                let Ok(dims) = u16::try_from(vec.len()) else {
                    saw_other = true; // too many dimensions for columnar storage
                    continue;
                };
                if let Some(prev) = vector_dims {
                    if prev != dims {
                        saw_other = true; // mixed dimensions → fallback
                    }
                } else {
                    vector_dims = Some(dims);
                }
            }
            _ => saw_other = true,
        }
    }

    let saw_any_int = saw_unsigned_int || saw_signed_int;

    // Vectors are exclusive; mixed with other types falls back to Dict.
    // Zero-dimension vectors cannot be round-tripped through the Float32Vector
    // codec (stride=0 means no row can be decoded), so those fall back to Dict
    // as well.
    if saw_vector
        && !saw_other
        && !saw_any_int
        && !saw_float
        && !saw_bool
        && let Some(dims) = vector_dims
        && dims > 0
    {
        return InferredType::Float32Vector { dimensions: dims };
    }

    // Mixed Int64+Float64 coalesces to Float64.
    // Vectors mixed with any other type fall back to Dict.
    if saw_other || saw_vector || (saw_any_int && saw_bool) || (saw_float && saw_bool) {
        InferredType::Dict
    } else if saw_float {
        InferredType::Float64
    } else if saw_signed_int {
        // Any negative value routes the whole column to RawI64, which uses
        // native i64 ordering. Non-negative-only columns still use the
        // more compact BitPacked encoding.
        InferredType::RawI64
    } else if saw_unsigned_int {
        InferredType::BitPacked
    } else if saw_bool {
        InferredType::Bitmap
    } else {
        // All nulls: default to Dict.
        InferredType::Dict
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
                    .column_dict("name", &["Alix", "Gus", "Vincent", "Jules", "Mia"])
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

    #[test]
    fn test_from_graph_store_round_trip() {
        // Build a CompactStore via the builder, then convert it back via
        // from_graph_store and verify the data survives the round-trip.
        let original = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35], 6)
                    .column_dict("name", &["Alix", "Gus", "Vincent"])
                    .column_bitmap("active", &[true, false, true])
            })
            .node_table("City", |t| t.column_dict("name", &["Amsterdam", "Berlin"]))
            .rel_table("LIVES_IN", "Person", "City", |r| {
                r.edges([(0, 0), (1, 1), (2, 0)]).backward(true)
            })
            .build()
            .unwrap();

        // Round-trip through from_graph_store.
        let converted = from_graph_store(&original).unwrap();

        // Verify node counts.
        assert_eq!(converted.nodes_by_label("Person").len(), 3);
        assert_eq!(converted.nodes_by_label("City").len(), 2);

        // Verify properties survived.
        let person_ids = converted.nodes_by_label("Person");
        let mut ages: Vec<i64> = person_ids
            .iter()
            .filter_map(|&id| {
                converted
                    .get_node_property(id, &PropertyKey::new("age"))
                    .and_then(|v| v.as_int64())
            })
            .collect();
        ages.sort_unstable();
        assert_eq!(ages, vec![25, 30, 35]);

        // Verify edges survived.
        let city_ids = converted.nodes_by_label("City");
        let mut total_edges = 0;
        for &pid in &person_ids {
            let edges = converted.edges_from(pid, crate::graph::Direction::Outgoing);
            total_edges += edges.len();
        }
        assert_eq!(total_edges, 3);

        // Verify backward edges (incoming to cities).
        for &cid in &city_ids {
            let incoming = converted.edges_from(cid, crate::graph::Direction::Incoming);
            assert!(!incoming.is_empty(), "incoming is empty");
        }
    }

    #[test]
    fn test_from_graph_store_empty() {
        let empty = CompactStoreBuilder::new().build().unwrap();
        let converted = from_graph_store(&empty).unwrap();
        assert_eq!(converted.nodes_by_label("Anything").len(), 0);
    }

    #[test]
    fn test_from_graph_store_with_lpg_store() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Insert nodes.
        let alix_id = store.create_node(&["Person"]);
        store.set_node_property(alix_id, "name", Value::from("Alix"));
        store.set_node_property(alix_id, "age", Value::Int64(30));

        let gus_id = store.create_node(&["Person"]);
        store.set_node_property(gus_id, "name", Value::from("Gus"));
        store.set_node_property(gus_id, "age", Value::Int64(25));

        let amsterdam_id = store.create_node(&["City"]);
        store.set_node_property(amsterdam_id, "name", Value::from("Amsterdam"));

        // Insert edges.
        store.create_edge(alix_id, amsterdam_id, "LIVES_IN");
        store.create_edge(gus_id, amsterdam_id, "LIVES_IN");

        // Convert.
        let compact = from_graph_store(&store).unwrap();

        // Verify.
        assert_eq!(compact.nodes_by_label("Person").len(), 2);
        assert_eq!(compact.nodes_by_label("City").len(), 1);

        // Check that properties are readable.
        let person_ids = compact.nodes_by_label("Person");
        let mut names: Vec<String> = person_ids
            .iter()
            .filter_map(|&id| {
                compact
                    .get_node_property(id, &PropertyKey::new("name"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
            .collect();
        names.sort();
        assert_eq!(names, vec!["Alix", "Gus"]);

        // Check edges: both persons should have outgoing edges.
        let mut total_outgoing = 0;
        for &pid in &person_ids {
            let edges = compact.edges_from(pid, crate::graph::Direction::Outgoing);
            total_outgoing += edges.len();
        }
        assert_eq!(total_outgoing, 2);

        // Check incoming edges on the city.
        let city_ids = compact.nodes_by_label("City");
        assert_eq!(city_ids.len(), 1);
        let incoming = compact.edges_from(city_ids[0], crate::graph::Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_from_graph_store_edge_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));

        // Edge with int property (BitPacked path).
        let e1 = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(e1, "since", Value::Int64(2020));

        // Edge with string property (Dict path).
        let e2 = store.create_edge(gus, alix, "KNOWS");
        store.set_edge_property(e2, "since", Value::Int64(2021));

        let compact = from_graph_store(&store).unwrap();

        // Verify edge count.
        let person_ids = compact.nodes_by_label("Person");
        let mut total_edges = 0;
        for &pid in &person_ids {
            total_edges += compact
                .edges_from(pid, crate::graph::Direction::Outgoing)
                .len();
        }
        assert_eq!(total_edges, 2);

        // Verify edge properties survived.
        for &pid in &person_ids {
            let edges = compact.edges_from(pid, crate::graph::Direction::Outgoing);
            for (_target, eid) in &edges {
                let edge = compact.get_edge(*eid).unwrap();
                let since = edge.properties.get(&PropertyKey::new("since")).unwrap();
                match since {
                    Value::Int64(v) => assert!(*v == 2020 || *v == 2021),
                    _ => panic!("expected Int64 for 'since', got {since:?}"),
                }
            }
        }
    }

    #[test]
    fn test_from_graph_store_edge_bool_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "active", Value::Bool(true));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Node");
        let edges = compact.edges_from(ids[0], crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let edge = compact.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("active")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_from_graph_store_edge_string_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "label", Value::from("primary"));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Node");
        let edges = compact.edges_from(ids[0], crate::graph::Direction::Outgoing);
        let edge = compact.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("label")),
            Some(&Value::String(ArcStr::from("primary")))
        );
    }

    #[test]
    fn test_from_graph_store_negative_int_preserves_int64_type() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "temp", Value::Int64(-10));

        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "temp", Value::Int64(5));

        let compact = from_graph_store(&store).unwrap();

        // Signed Int64 columns round-trip as Value::Int64 via the RawI64 codec.
        // Prior behaviour (<=0.5.40) stringified them into a Dict column,
        // breaking WHERE matches and promoting sum() to Float64.
        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 2);
        let mut temps: Vec<i64> = ids
            .iter()
            .filter_map(
                |&id| match compact.get_node_property(id, &PropertyKey::new("temp")) {
                    Some(Value::Int64(n)) => Some(n),
                    _ => None,
                },
            )
            .collect();
        temps.sort_unstable();
        assert_eq!(temps, vec![-10, 5]);
    }

    #[test]
    fn test_from_graph_store_float64_column() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Sensor"]);
        store.set_node_property(a, "reading", Value::Float64(98.6));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Sensor");
        assert_eq!(ids.len(), 1);

        // Float64 values are stored natively.
        let val = compact
            .get_node_property(ids[0], &PropertyKey::new("reading"))
            .unwrap();
        assert_eq!(val, Value::Float64(98.6));
    }

    #[test]
    fn test_from_graph_store_mixed_types_fall_back_to_dict() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Same property key with different types across nodes.
        let a = store.create_node(&["Thing"]);
        store.set_node_property(a, "value", Value::Int64(42));

        let b = store.create_node(&["Thing"]);
        store.set_node_property(b, "value", Value::Bool(true));

        let compact = from_graph_store(&store).unwrap();

        // Mixed Int64 + Bool should fall back to Dict.
        let ids = compact.nodes_by_label("Thing");
        assert_eq!(ids.len(), 2);

        for &id in &ids {
            let val = compact
                .get_node_property(id, &PropertyKey::new("value"))
                .unwrap();
            // All values should be strings (Dict encoding).
            assert!(
                matches!(val, Value::String(_)),
                "expected String (Dict fallback), got {val:?}"
            );
        }
    }

    #[test]
    fn test_from_graph_store_sparse_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Node A has both properties.
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "name", Value::from("alpha"));
        store.set_node_property(a, "score", Value::Int64(10));

        // Node B has only 'name', no 'score'.
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "name", Value::from("beta"));

        // Node C has only 'score', no 'name'.
        let c = store.create_node(&["Item"]);
        store.set_node_property(c, "score", Value::Int64(20));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 3);

        // All nodes should exist and have the properties they were given.
        // Missing properties should be null-padded (0 for BitPacked, "" for Dict).
        let mut name_count = 0;
        let mut score_count = 0;
        for &id in &ids {
            if let Some(Value::String(s)) = compact.get_node_property(id, &PropertyKey::new("name"))
                && !s.is_empty()
            {
                name_count += 1;
            }
            if let Some(Value::Int64(n)) = compact.get_node_property(id, &PropertyKey::new("score"))
                && n > 0
            {
                score_count += 1;
            }
        }
        // Two nodes have real names, two have real scores.
        assert_eq!(name_count, 2);
        assert_eq!(score_count, 2);
    }

    #[test]
    fn test_from_graph_store_multi_label_nodes() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        let a = store.create_node(&["Person", "Actor"]);
        store.set_node_property(a, "name", Value::from("Vincent"));

        let b = store.create_node(&["Person"]);
        store.set_node_property(b, "name", Value::from("Jules"));

        let compact = from_graph_store(&store).unwrap();

        // Single-label node goes to "Person" table.
        let person_ids = compact.nodes_by_label("Person");
        assert_eq!(person_ids.len(), 1);

        // Multi-label node goes to "Actor|Person" compound table.
        let compound_ids = compact.nodes_by_label("Actor|Person");
        assert_eq!(compound_ids.len(), 1);

        // Verify the multi-label node's property survived.
        let val = compact
            .get_node_property(compound_ids[0], &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(val, Value::String(ArcStr::from("Vincent")));
    }

    #[test]
    fn test_from_graph_store_all_null_column() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Two nodes with different property keys, creating gaps.
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "x", Value::Int64(1));

        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "y", Value::Int64(2));

        let compact = from_graph_store(&store).unwrap();

        let ids = compact.nodes_by_label("Item");
        assert_eq!(ids.len(), 2);

        // Node a has 'x' but 'y' is null-padded.
        // Node b has 'y' but 'x' is null-padded.
        // This exercises the null-padding logic for sparse properties.
    }

    #[test]
    fn test_infer_type_all_nulls() {
        assert_eq!(
            infer_type_from_values(&[Value::Null, Value::Null]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_int_only() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_bool_only() {
        assert_eq!(
            infer_type_from_values(&[Value::Bool(true), Value::Bool(false)]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_mixed_int_bool() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::Bool(true)]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_negative_int() {
        // Any negative Int64 routes the whole column to RawI64 (prior
        // behaviour fell back to Dict, which broke type round-trip and
        // ordered operations).
        assert_eq!(
            infer_type_from_values(&[Value::Int64(-5), Value::Int64(10)]),
            InferredType::RawI64
        );
        assert_eq!(
            infer_type_from_values(&[Value::Int64(-5)]),
            InferredType::RawI64
        );
        // Non-negative-only columns still use BitPacked for compression.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_float() {
        assert_eq!(
            infer_type_from_values(&[Value::Float64(1.5)]),
            InferredType::Float64
        );
    }

    #[test]
    fn test_infer_type_mixed_int_float_coalesces_to_float() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::Float64(2.5)]),
            InferredType::Float64
        );
    }

    #[test]
    fn test_infer_type_int_with_nulls() {
        assert_eq!(
            infer_type_from_values(&[Value::Int64(5), Value::Null, Value::Int64(10)]),
            InferredType::BitPacked
        );
    }

    /// Same edge type spanning multiple label pairs — normal in LPGs.
    /// Regression test for <https://github.com/GrafeoDB/grafeo/issues/221>.
    #[test]
    fn test_from_graph_store_multi_label_edge_type() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Three node types
        let m1 = store.create_node(&["Method"]);
        store.set_node_property(m1, "name", Value::from("foo"));
        let m2 = store.create_node(&["Method"]);
        store.set_node_property(m2, "name", Value::from("bar"));
        let c1 = store.create_node(&["Class"]);
        store.set_node_property(c1, "name", Value::from("MyClass"));
        let i1 = store.create_node(&["Interface"]);
        store.set_node_property(i1, "name", Value::from("MyInterface"));

        // CALLS edges between different label pairs
        store.create_edge(m1, m2, "CALLS"); // Method -> Method
        store.create_edge(c1, m1, "CALLS"); // Class -> Method
        // USES_TYPE edges between different label pairs
        store.create_edge(m1, c1, "USES_TYPE"); // Method -> Class
        store.create_edge(m1, i1, "USES_TYPE"); // Method -> Interface

        // This should not panic — same edge type across multiple label pairs is valid.
        let compact = from_graph_store(&store).unwrap();

        // Verify all nodes survived
        assert_eq!(compact.nodes_by_label("Method").len(), 2);
        assert_eq!(compact.nodes_by_label("Class").len(), 1);
        assert_eq!(compact.nodes_by_label("Interface").len(), 1);

        // Verify edges survived — check via rel_tables_for_type
        let calls_tables = compact.rel_tables_for_type("CALLS");
        let uses_tables = compact.rel_tables_for_type("USES_TYPE");

        // CALLS spans 2 label pairs: Method→Method, Class→Method
        assert_eq!(
            calls_tables.len(),
            2,
            "CALLS should have 2 rel tables (different label pairs)"
        );
        // USES_TYPE spans 2 label pairs: Method→Class, Method→Interface
        assert_eq!(
            uses_tables.len(),
            2,
            "USES_TYPE should have 2 rel tables (different label pairs)"
        );

        // Total edges across all CALLS tables
        let total_calls: usize = calls_tables.iter().map(|rt| rt.num_edges()).sum();
        assert_eq!(total_calls, 2, "Should have 2 CALLS edges total");
        // Total edges across all USES_TYPE tables
        let total_uses: usize = uses_tables.iter().map(|rt| rt.num_edges()).sum();
        assert_eq!(total_uses, 2, "Should have 2 USES_TYPE edges total");

        // Verify all_edge_types returns deduplicated type names
        use crate::graph::traits::GraphStore;
        let mut edge_types = compact.all_edge_types();
        edge_types.sort();
        assert_eq!(
            edge_types,
            vec!["CALLS", "USES_TYPE"],
            "all_edge_types should return each type once, not per rel table"
        );

        // Verify estimate_avg_degree deduplicates shared source labels.
        // USES_TYPE has Method→Class and Method→Interface — Method appears as source in both
        // rel tables but should only be counted once in the denominator.
        // 2 edges / 2 source nodes (Method) = 1.0 (not 2 edges / 4 = 0.5 if double-counted)
        let avg_out = compact.estimate_avg_degree("USES_TYPE", true);
        assert!(avg_out > 0.0, "USES_TYPE outgoing degree should be > 0");
        assert!(
            (avg_out - 1.0).abs() < f64::EPSILON,
            "USES_TYPE avg outgoing degree should be 1.0 (2 edges / 2 Method nodes), got {avg_out}"
        );

        // Verify unknown edge type returns 0
        let unknown = compact.estimate_avg_degree("NONEXISTENT", true);
        assert!(
            (unknown - 0.0).abs() < f64::EPSILON,
            "Unknown edge type should return 0.0 avg degree"
        );
    }

    // -------------------------------------------------------------------
    // Zone map helper tests
    // -------------------------------------------------------------------

    #[test]
    fn test_zone_map_u64_values_exceeding_i64_max() {
        // Values that exceed i64::MAX should produce a zone map without
        // min/max bounds (conservative, no pruning).
        let values = vec![0u64, i64::MAX as u64 + 1, u64::MAX];
        let zm = compute_zone_map_u64(&values);
        assert!(
            zm.min.is_none(),
            "min should be None when values overflow i64"
        );
        assert!(
            zm.max.is_none(),
            "max should be None when values overflow i64"
        );
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_u64_within_i64_range() {
        let values = vec![10u64, 20, 30];
        let zm = compute_zone_map_u64(&values);
        assert_eq!(zm.min, Some(Value::Int64(10)));
        assert_eq!(zm.max, Some(Value::Int64(30)));
        assert_eq!(zm.null_count, 0);
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_u64_empty_slice() {
        let zm = compute_zone_map_u64(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    #[test]
    fn test_zone_map_strings_empty_slice() {
        let zm = compute_zone_map_strings(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    #[test]
    fn test_zone_map_strings_sorted() {
        let values = &["Paris", "Amsterdam", "Berlin"];
        let zm = compute_zone_map_strings(values);
        assert_eq!(zm.min, Some(Value::from("Amsterdam")));
        assert_eq!(zm.max, Some(Value::from("Paris")));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_all_true() {
        let values = &[true, true, true];
        let zm = compute_zone_map_bool(values);
        // All true: min = true, max = true.
        assert_eq!(zm.min, Some(Value::Bool(true)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_all_false() {
        let values = &[false, false];
        let zm = compute_zone_map_bool(values);
        // All false: min = false, max = false.
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(false)));
        assert_eq!(zm.row_count, 2);
    }

    #[test]
    fn test_zone_map_bool_mixed() {
        let values = &[false, true, false];
        let zm = compute_zone_map_bool(values);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 3);
    }

    #[test]
    fn test_zone_map_bool_empty() {
        let zm = compute_zone_map_bool(&[]);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 0);
    }

    // -------------------------------------------------------------------
    // Type inference edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_infer_type_string_values() {
        assert_eq!(
            infer_type_from_values(&[Value::from("Alix"), Value::from("Gus")]),
            InferredType::Dict
        );
    }

    #[test]
    fn test_infer_type_int_and_null() {
        // Nulls are skipped, so pure Int64 with nulls remains BitPacked.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(0), Value::Null, Value::Int64(5)]),
            InferredType::BitPacked
        );
    }

    #[test]
    fn test_infer_type_bool_and_null() {
        // Nulls are skipped, so pure Bool with nulls remains Bitmap.
        assert_eq!(
            infer_type_from_values(&[Value::Bool(true), Value::Null]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_empty_values() {
        // Empty slice: no non-null values seen, defaults to Dict.
        assert_eq!(infer_type_from_values(&[]), InferredType::Dict);
    }

    // -------------------------------------------------------------------
    // from_graph_store: null properties and multi-label nodes
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_nodes_with_no_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Nodes with no properties at all.
        store.create_node(&["Marker"]);
        store.create_node(&["Marker"]);

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Marker");
        assert_eq!(ids.len(), 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store: vector-valued properties fall back to Dict
    // -------------------------------------------------------------------

    /// Nodes with `Value::Vector` properties of different dimensions produce
    /// a Dict column (vectors are not an inferred type). Covers the Dict
    /// fallback for "other" values in `infer_type_from_values` and the
    /// per-row Display serialization inside the Dict build path.
    #[test]
    fn test_from_graph_store_mixed_vector_dims() {
        use crate::graph::lpg::LpgStore;
        use std::sync::Arc;

        let store = LpgStore::new().unwrap();

        // Two nodes with differently-dimensioned embeddings.
        let alix = store.create_node(&["Doc"]);
        let short: Arc<[f32]> = Arc::from([0.1f32, 0.2, 0.3].as_slice());
        store.set_node_property(alix, "embedding", Value::Vector(short));

        let gus = store.create_node(&["Doc"]);
        let long: Arc<[f32]> = Arc::from([0.4f32, 0.5, 0.6, 0.7, 0.8].as_slice());
        store.set_node_property(gus, "embedding", Value::Vector(long));

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Doc");
        assert_eq!(ids.len(), 2);

        // Both embeddings should be readable as strings (Dict fallback).
        let mut seen_vec_strings = 0usize;
        for &id in &ids {
            let val = compact
                .get_node_property(id, &PropertyKey::new("embedding"))
                .expect("embedding property missing");
            match val {
                Value::String(s) => {
                    // Display format is lowercase "vector([...])"; compare
                    // case-insensitively to be resilient to formatting tweaks.
                    let lower = s.to_lowercase();
                    assert!(
                        lower.contains("vector"),
                        "dict-encoded vector should include the vector tag: {s}"
                    );
                    seen_vec_strings += 1;
                }
                other => panic!("expected Dict fallback (String), got {other:?}"),
            }
        }
        assert_eq!(seen_vec_strings, 2);
    }

    /// A `Value::Vector` with consistent dimensions across nodes round-trips
    /// through the Float32Vector column codec and comes back as `Value::Vector`
    /// with the original dimensions and values.
    #[test]
    fn test_from_graph_store_float32_vector() {
        use crate::graph::lpg::LpgStore;
        use std::sync::Arc;

        let store = LpgStore::new().unwrap();
        let expected: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        for name in ["Alix", "Gus", "Vincent"] {
            let id = store.create_node(&["Doc"]);
            store.set_node_property(id, "name", Value::from(name));
            let emb: Arc<[f32]> = Arc::from(expected.as_slice());
            store.set_node_property(id, "embedding", Value::Vector(emb));
        }

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Doc");
        assert_eq!(ids.len(), 3);

        for &id in &ids {
            let v = compact
                .get_node_property(id, &PropertyKey::new("embedding"))
                .expect("embedding missing");
            match v {
                Value::Vector(data) => {
                    assert_eq!(&*data, &expected, "unexpected vector contents");
                }
                other => panic!("expected Value::Vector, got {other:?}"),
            }
        }
    }

    /// Nodes where only ~10% of them carry a given property exercise the
    /// null-padding path in `from_graph_store`. Covers the `push(Value::Null)`
    /// fill loops that keep column lengths aligned to row count.
    #[test]
    fn test_from_graph_store_all_null() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let mut ids = Vec::new();
        for i in 0..20 {
            let nid = store.create_node(&["Item"]);
            // Only 2 out of 20 nodes (10%) have "flag" set.
            if i == 3 || i == 17 {
                store.set_node_property(nid, "flag", Value::Int64(i64::from(i)));
            }
            ids.push(nid);
        }

        let compact = from_graph_store(&store).unwrap();
        let rows = compact.nodes_by_label("Item");
        assert_eq!(rows.len(), 20);

        // Count rows whose "flag" decoded value is nonzero (BitPacked null
        // padding decodes as 0). We expect exactly 2.
        let mut nonzero = 0usize;
        for &id in &rows {
            if let Some(Value::Int64(v)) = compact.get_node_property(id, &PropertyKey::new("flag"))
                && v > 0
            {
                nonzero += 1;
            }
        }
        assert_eq!(
            nonzero, 2,
            "only 2 nodes had a real 'flag' value; the rest should be zero-padded"
        );
    }

    // -------------------------------------------------------------------
    // Zone map boundary cases
    // -------------------------------------------------------------------

    /// Zone map for a u64 column whose maximum equals `i64::MAX` exactly
    /// should keep both bounds (boundary-inclusive), while a max one above
    /// `i64::MAX` drops them. Guards against off-by-one errors in the
    /// overflow check.
    #[test]
    fn test_compute_zone_map_i64_boundary() {
        // Exactly at the boundary: keep bounds.
        let at_boundary = vec![0u64, 100, i64::MAX as u64];
        let zm = compute_zone_map_u64(&at_boundary);
        assert_eq!(zm.min, Some(Value::Int64(0)));
        assert_eq!(zm.max, Some(Value::Int64(i64::MAX)));
        assert_eq!(zm.row_count, 3);

        // One above the boundary: drop bounds, preserve row_count.
        let above_boundary = vec![0u64, i64::MAX as u64 + 1];
        let zm = compute_zone_map_u64(&above_boundary);
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.row_count, 2);
    }

    /// Zone map for a mixed true/false bool column: min is false, max is
    /// true. Distinct from the existing `test_zone_map_bool_mixed` in that
    /// the ratio of true/false is skewed to sanity-check the any()-based
    /// implementation.
    #[test]
    fn test_compute_zone_map_bool() {
        // Heavily skewed: one true, many false.
        let mostly_false = vec![false; 9]
            .into_iter()
            .chain(std::iter::once(true))
            .collect::<Vec<_>>();
        let zm = compute_zone_map_bool(&mostly_false);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 10);

        // Heavily skewed the other way: one false, many true.
        let mostly_true = std::iter::once(false)
            .chain(std::iter::repeat_n(true, 9))
            .collect::<Vec<_>>();
        let zm = compute_zone_map_bool(&mostly_true);
        assert_eq!(zm.min, Some(Value::Bool(false)));
        assert_eq!(zm.max, Some(Value::Bool(true)));
        assert_eq!(zm.row_count, 10);
    }

    #[test]
    fn test_from_graph_store_multi_label_sorted_key() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Labels "Zebra" and "Alpha" should be sorted to "Alpha|Zebra".
        let a = store.create_node(&["Zebra", "Alpha"]);
        store.set_node_property(a, "name", Value::from("Butch"));

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Alpha|Zebra");
        assert_eq!(ids.len(), 1);

        let val = compact
            .get_node_property(ids[0], &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(val, Value::String(ArcStr::from("Butch")));
    }

    // -------------------------------------------------------------------
    // infer_type_from_values: all Value::* fall-through branches
    // -------------------------------------------------------------------

    #[test]
    fn test_infer_type_timestamp_falls_back_to_dict() {
        use grafeo_common::types::Timestamp;
        let ts = Value::Timestamp(Timestamp::from_millis(1_700_000_000_000));
        assert_eq!(infer_type_from_values(&[ts]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_date_falls_back_to_dict() {
        use grafeo_common::types::Date;
        let d = Value::Date(Date::from_days(19000));
        assert_eq!(infer_type_from_values(&[d]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_vector_consistent_dim_picks_float32_vector() {
        // Consistent-dimension Float32 vectors get a dedicated column type.
        let v = Value::Vector(std::sync::Arc::from([0.1f32, 0.2, 0.3].as_slice()));
        assert_eq!(
            infer_type_from_values(&[v]),
            InferredType::Float32Vector { dimensions: 3 }
        );
    }

    #[test]
    fn test_infer_type_mixed_dim_vectors_fall_back_to_dict() {
        // Inconsistent dimensions force the Dict fallback.
        let v3 = Value::Vector(std::sync::Arc::from([0.1f32, 0.2, 0.3].as_slice()));
        let v5 = Value::Vector(std::sync::Arc::from(
            [0.1f32, 0.2, 0.3, 0.4, 0.5].as_slice(),
        ));
        assert_eq!(infer_type_from_values(&[v3, v5]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_bytes_falls_back_to_dict() {
        let b = Value::Bytes(std::sync::Arc::from(b"payload".as_slice()));
        assert_eq!(infer_type_from_values(&[b]), InferredType::Dict);
    }

    #[test]
    fn test_infer_type_null_with_bool_yields_bitmap() {
        // Nulls skipped; a pure Bool column stays Bitmap.
        assert_eq!(
            infer_type_from_values(&[Value::Null, Value::Bool(false), Value::Null]),
            InferredType::Bitmap
        );
    }

    #[test]
    fn test_infer_type_saw_int_and_other_yields_dict() {
        // Int + String (saw_other) should force Dict.
        assert_eq!(
            infer_type_from_values(&[Value::Int64(1), Value::from("hello")]),
            InferredType::Dict
        );
    }

    // -------------------------------------------------------------------
    // Builder error variants: DuplicateLabel, DuplicateEdgeType
    // -------------------------------------------------------------------

    #[test]
    fn test_builder_duplicate_label_error() {
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| t.column_bitpacked("age", &[30], 6))
            .node_table("Person", |t| t.column_bitpacked("age", &[40], 6))
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::DuplicateLabel(ref s)) if s == "Person"
        ));
    }

    #[test]
    fn test_builder_duplicate_edge_type_error() {
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::DuplicateEdgeType(_))
        ));
    }

    #[test]
    fn test_builder_column_length_mismatch_error() {
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("age", &[25, 30, 35], 6)
                    .column_dict("name", &["Alix", "Gus"]) // length 2, mismatch
            })
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::ColumnLengthMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn test_builder_value_overflow_error() {
        // Values exceeding i64::MAX must be flagged.
        let bad_value = (i64::MAX as u64) + 10;
        let result = CompactStoreBuilder::new()
            .node_table("Person", |t| {
                t.column_bitpacked("x", &[1u64, bad_value], 64)
            })
            .build();
        assert!(matches!(
            result,
            Err(CompactStoreError::ValueOverflow { ref column, value, .. })
                if column == "x" && value == bad_value
        ));
    }

    // -------------------------------------------------------------------
    // Pre-built codec passthrough: NodeTableBuilder::column
    // -------------------------------------------------------------------

    #[test]
    fn test_node_table_builder_prebuilt_column() {
        use crate::codec::BitPackedInts;

        let bp = BitPackedInts::pack(&[100u64, 200, 300]);
        let codec = ColumnCodec::BitPacked(bp);

        let store = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column("value", codec))
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Item");
        assert_eq!(ids.len(), 3);

        // Check the pre-built column values are readable.
        let mut values: Vec<i64> = ids
            .iter()
            .filter_map(|&id| {
                store
                    .get_node_property(id, &PropertyKey::new("value"))
                    .and_then(|v| v.as_int64())
            })
            .collect();
        values.sort_unstable();
        assert_eq!(values, vec![100, 200, 300]);
    }

    // -------------------------------------------------------------------
    // column_int8_vector: zero dimensions case (row_count=0)
    // -------------------------------------------------------------------

    #[test]
    fn test_node_table_builder_int8_vector_zero_dimensions() {
        // Zero dimensions yields 0 rows, no panic.
        let store = CompactStoreBuilder::new()
            .node_table("Item", |t| t.column_int8_vector("embed", Vec::new(), 0))
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Item");
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_node_table_builder_int8_vector_multi_row() {
        // Two 3-dim vectors packed in one flat array.
        let store = CompactStoreBuilder::new()
            .node_table("Doc", |t| {
                t.column_int8_vector("embed", vec![1i8, 2, 3, 4, 5, 6], 3)
            })
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Doc");
        assert_eq!(ids.len(), 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store / from_graph_store_preserving_ids edge cases
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_preserving_ids_empty() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        // No nodes, no edges.
        let compact = crate::graph::compact::from_graph_store_preserving_ids(&store).unwrap();
        assert_eq!(compact.node_count(), 0);
        assert_eq!(compact.edge_count(), 0);
    }

    #[test]
    fn test_from_graph_store_single_node_no_properties() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        store.create_node(&["Loner"]);

        let compact = from_graph_store(&store).unwrap();
        let ids = compact.nodes_by_label("Loner");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_from_graph_store_preserving_ids_with_data() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        let edge_id = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(edge_id, "since", Value::Int64(2020));

        let compact = crate::graph::compact::from_graph_store_preserving_ids(&store).unwrap();

        // Original IDs should still resolve to nodes.
        let alix_resolved = compact.get_node(alix);
        assert!(
            alix_resolved.is_some(),
            "original NodeId should remain resolvable after preserve_ids"
        );
        assert_eq!(
            alix_resolved
                .unwrap()
                .properties
                .get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Alix")))
        );

        let edge_resolved = compact.get_edge(edge_id);
        assert!(
            edge_resolved.is_some(),
            "original EdgeId should remain resolvable after preserve_ids"
        );
    }

    #[test]
    fn test_from_graph_store_skewed_properties() {
        // One label with 5 nodes, each has only one of three properties.
        // This stresses null-padding in sparse columns.
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        for (i, (name, key)) in [
            ("Alix", "a"),
            ("Gus", "b"),
            ("Vincent", "c"),
            ("Jules", "a"),
            ("Mia", "b"),
        ]
        .iter()
        .enumerate()
        {
            let nid = store.create_node(&["Person"]);
            store.set_node_property(nid, "name", Value::from(*name));
            // Property 'a', 'b', or 'c' stored as Int64.
            let score = i64::try_from(i).unwrap_or(0);
            store.set_node_property(nid, key, Value::Int64(score));
        }

        let compact = from_graph_store(&store).unwrap();
        assert_eq!(compact.nodes_by_label("Person").len(), 5);
    }

    // -------------------------------------------------------------------
    // Builder column methods: unique scenarios not covered by earlier tests.
    // `column_int8_vector` happy-path and zero-dim cases live at
    // `test_node_table_builder_int8_vector_multi_row` / `_zero_dimensions`;
    // `column` pre-built-codec passthrough lives at
    // `test_node_table_builder_prebuilt_column`. Keep those canonical.
    // -------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "is not a multiple of dimensions")]
    fn test_node_table_column_int8_vector_not_multiple_panics() {
        // 5 bytes with 2 dimensions is not a multiple.
        let _ = CompactStoreBuilder::new().node_table("Bad", |t| {
            t.column_int8_vector("vec", vec![1, 2, 3, 4, 5], 2)
        });
    }

    #[test]
    fn test_node_table_column_bitmap() {
        let store = CompactStoreBuilder::new()
            .node_table("Flag", |t| {
                t.column_bitmap("active", &[true, false, true, false])
            })
            .build()
            .unwrap();

        let ids = store.nodes_by_label("Flag");
        assert_eq!(ids.len(), 4);

        let v0 = store
            .get_node_property(ids[0], &PropertyKey::new("active"))
            .unwrap();
        assert_eq!(v0, Value::Bool(true));
    }

    // Error-path tests (`column_length_mismatch`, `value_overflow`,
    // `duplicate_label`, `duplicate_edge_type`) live at the matching
    // `*_error` tests above; keep those canonical.

    #[test]
    fn test_builder_same_edge_type_different_labels_allowed() {
        // Same edge type across different label pairs should NOT trigger
        // DuplicateEdgeType (issue #221 regression coverage).
        let result = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[1], 4))
            .node_table("C", |t| t.column_bitpacked("v", &[1], 4))
            .rel_table("LINKS", "A", "B", |r| r.edges([(0, 0)]))
            .rel_table("LINKS", "A", "C", |r| r.edges([(0, 0)]))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_error_types_trait_impls() {
        // Exercise Debug/Display/Clone on CompactStoreError variants.
        let err = CompactStoreError::LabelNotFound("Paris".to_string());
        let cloned = err.clone();
        assert!(format!("{cloned}").contains("Paris"));
        assert!(format!("{cloned:?}").contains("LabelNotFound"));

        let mismatch = CompactStoreError::ColumnLengthMismatch {
            expected: 10,
            got: 5,
        };
        assert!(format!("{mismatch}").contains("10"));
        assert!(format!("{mismatch}").contains('5'));

        let dup_label = CompactStoreError::DuplicateLabel("Berlin".to_string());
        assert!(format!("{dup_label}").contains("Berlin"));

        let dup_edge = CompactStoreError::DuplicateEdgeType("KNOWS".to_string());
        assert!(format!("{dup_edge}").contains("KNOWS"));

        let inconsistent = CompactStoreError::InconsistentEdgeData("boom".to_string());
        assert!(format!("{inconsistent}").contains("boom"));

        let overflow = CompactStoreError::ValueOverflow {
            column: "age".to_string(),
            value: u64::MAX,
            max: i64::MAX as u64,
        };
        assert!(format!("{overflow}").contains("age"));

        let table_overflow = CompactStoreError::TableCountOverflow {
            kind: "node",
            count: 99_999,
            max: MAX_TABLE_ID,
        };
        assert!(format!("{table_overflow}").contains("node"));
        assert!(format!("{table_overflow}").contains("99999"));
    }

    // -------------------------------------------------------------------
    // RelTableBuilder: bit-packed edge properties
    // -------------------------------------------------------------------

    #[test]
    fn test_rel_table_column_bitpacked() {
        // Exercise RelTableBuilder::column_bitpacked (pre-built codec injection on edges).
        let store = CompactStoreBuilder::new()
            .node_table("A", |t| t.column_bitpacked("v", &[1, 2], 4))
            .node_table("B", |t| t.column_bitpacked("v", &[3, 4], 4))
            .rel_table("LINKS", "A", "B", |r| {
                r.edges([(0, 0), (1, 1)])
                    .backward(true)
                    .column_bitpacked("weight", &[100, 200], 8)
            })
            .build()
            .unwrap();

        let a_ids = store.nodes_by_label("A");
        assert_eq!(a_ids.len(), 2);

        // Verify edges exist.
        let mut total_edges = 0;
        for &id in &a_ids {
            total_edges += store
                .edges_from(id, crate::graph::Direction::Outgoing)
                .len();
        }
        assert_eq!(total_edges, 2);
    }

    // -------------------------------------------------------------------
    // from_graph_store_preserving_ids: specialized cases.
    // The basic happy-path is covered by
    // `test_from_graph_store_preserving_ids_with_data`; multi-label and
    // CSR-edge-ordering are unique to this section.
    // -------------------------------------------------------------------

    #[test]
    fn test_from_graph_store_preserving_ids_multi_label() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();
        let butch = store.create_node(&["Person", "Boxer"]);
        store.set_node_property(butch, "name", Value::from("Butch"));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        // Multi-label key is sorted as "Boxer|Person".
        let name = compact
            .get_node_property(butch, &PropertyKey::new("name"))
            .and_then(|v| v.as_str().map(str::to_string));
        assert_eq!(name.as_deref(), Some("Butch"));
    }

    #[test]
    fn test_from_graph_store_preserving_ids_edges_sorted_by_csr_order() {
        use crate::graph::lpg::LpgStore;

        let store = LpgStore::new().unwrap();

        // Create nodes with deliberate insertion order to exercise CSR sorting.
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        // Insert edges in an order that differs from (src, dst) sort order.
        let e_c_a = store.create_edge(c, a, "LINK");
        let e_a_b = store.create_edge(a, b, "LINK");
        let e_b_c = store.create_edge(b, c, "LINK");

        let compact = from_graph_store_preserving_ids(&store).unwrap();

        // All three original edge IDs should resolve and round-trip.
        for eid in [e_c_a, e_a_b, e_b_c] {
            let rec = compact.get_edge(eid).unwrap();
            assert_eq!(rec.edge_type.as_str(), "LINK");
        }
    }
}
