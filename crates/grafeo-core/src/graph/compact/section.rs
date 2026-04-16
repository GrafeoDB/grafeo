//! [`Section`](grafeo_common::storage::section::Section) implementation for [`CompactStore`].
//!
//! Serializes/deserializes a CompactStore to/from the `.grafeo` container
//! format with versioned headers and CRC32 integrity.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EdgeId, NodeId, PropertyKey};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use super::CompactStore;
use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{ColumnDef, ColumnType, EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// Magic bytes identifying a CompactStore section.
const MAGIC: [u8; 4] = *b"GCST";

/// Current section format version.
const FORMAT_VERSION: u8 = 1;

/// Wraps a [`CompactStore`] as a container [`Section`].
pub struct CompactStoreSection {
    store: RwLock<Option<Arc<CompactStore>>>,
    dirty: AtomicBool,
}

impl CompactStoreSection {
    /// Creates a new section wrapping an existing store.
    #[must_use]
    pub fn new(store: Arc<CompactStore>) -> Self {
        Self {
            store: RwLock::new(Some(store)),
            dirty: AtomicBool::new(false),
        }
    }

    /// Creates an empty section (for deserialization).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            store: RwLock::new(None),
            dirty: AtomicBool::new(false),
        }
    }

    /// Marks this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Returns a reference to the inner store, if any.
    #[must_use]
    pub fn store(&self) -> Option<Arc<CompactStore>> {
        self.store.read().clone()
    }
}

impl Section for CompactStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::CompactStore
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION
    }

    fn serialize(&self) -> grafeo_common::utils::error::Result<Vec<u8>> {
        let guard = self.store.read();
        let store = guard.as_ref().ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal("no CompactStore to serialize".into())
        })?;

        let mut buf = Vec::with_capacity(store.memory_bytes());

        // Header.
        buf.extend_from_slice(&MAGIC);
        buf.push(FORMAT_VERSION);
        let flags: u8 = u8::from(store.preserves_ids());
        buf.push(flags);

        // Node tables.
        write_len(&mut buf, store.node_tables_by_id.len());
        for nt in &store.node_tables_by_id {
            write_str(&mut buf, nt.label());
            write_len(&mut buf, nt.len());
            let columns = nt.columns();
            let zone_maps = nt.zone_maps();
            write_len(&mut buf, columns.len());
            for (key, codec) in columns {
                write_str(&mut buf, key.as_str());
                // Zone map for this column.
                if let Some(zm) = zone_maps.get(key) {
                    buf.push(1);
                    write_zone_map(&mut buf, zm);
                } else {
                    buf.push(0);
                }
                codec.write_to(&mut buf);
            }
        }

        // Relationship tables.
        write_len(&mut buf, store.rel_tables_by_id.len());
        for rt in &store.rel_tables_by_id {
            write_str(&mut buf, rt.edge_type().as_str());
            write_u16(&mut buf, rt.src_table_id());
            write_u16(&mut buf, rt.dst_table_id());
            rt.fwd().write_to(&mut buf);
            if let Some(bwd) = rt.bwd() {
                buf.push(1);
                bwd.write_to(&mut buf);
            } else {
                buf.push(0);
            }
            let properties = rt.properties();
            write_len(&mut buf, properties.len());
            for (key, codec) in properties {
                write_str(&mut buf, key.as_str());
                codec.write_to(&mut buf);
            }
        }

        // ID maps.
        if store.preserves_ids() {
            if let Some(ref node_map) = store.node_id_map {
                write_len(&mut buf, node_map.len());
                for (&nid, &(tid, off)) in node_map {
                    write_u64(&mut buf, nid.as_u64());
                    write_u16(&mut buf, tid);
                    write_u64(&mut buf, off);
                }
            }
            if let Some(ref edge_map) = store.edge_id_map {
                write_len(&mut buf, edge_map.len());
                for (&eid, &(rtid, pos)) in edge_map {
                    write_u64(&mut buf, eid.as_u64());
                    write_u16(&mut buf, rtid);
                    write_u64(&mut buf, pos);
                }
            }
        }

        // CRC32 at end.
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        Ok(buf)
    }

    fn deserialize(&mut self, data: &[u8]) -> grafeo_common::utils::error::Result<()> {
        let store = deserialize_compact_store(data).map_err(|e| {
            grafeo_common::utils::error::Error::Internal(format!(
                "CompactStore deserialization failed: {e}"
            ))
        })?;

        *self.store.write() = Some(Arc::new(store));
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.read().as_ref().map_or(0, |s| s.memory_bytes())
    }
}

// ── Deserialization ────────────────────────────────────────────────

fn deserialize_compact_store(data: &[u8]) -> Result<CompactStore, String> {
    if data.len() < 10 {
        return Err("data too short for CompactStore section".into());
    }

    // Verify CRC32.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(format!(
            "CRC32 mismatch: stored {stored_crc:#010X}, computed {computed_crc:#010X}"
        ));
    }

    let mut pos = 0;

    // Header.
    if data[pos..pos + 4] != MAGIC {
        return Err("bad magic".into());
    }
    pos += 4;
    let version = data[pos];
    pos += 1;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported version {version}"));
    }
    let flags = data[pos];
    pos += 1;
    let preserves_ids = flags & 0x01 != 0;

    // Node tables.
    let num_node_tables = read_u32(data, &mut pos)? as usize;
    let mut node_tables = Vec::with_capacity(num_node_tables);
    let mut label_to_table_id: FxHashMap<arcstr::ArcStr, u16> = FxHashMap::default();
    let mut table_id_to_label: Vec<arcstr::ArcStr> = Vec::with_capacity(num_node_tables);

    for table_idx in 0..num_node_tables {
        let table_id = u16::try_from(table_idx).unwrap_or(0);
        let label = read_string(data, &mut pos)?;
        let label = arcstr::ArcStr::from(label.as_str());
        let row_count = read_u32(data, &mut pos)? as usize;
        let num_cols = read_u32(data, &mut pos)? as usize;

        let mut columns: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut zone_maps: FxHashMap<PropertyKey, ZoneMap> = FxHashMap::default();
        let mut col_defs = Vec::with_capacity(num_cols);

        for _ in 0..num_cols {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);

            let has_zm = *data.get(pos).ok_or("truncated zone map flag")?;
            pos += 1;
            if has_zm == 1 {
                let zm = read_zone_map(data, &mut pos)?;
                zone_maps.insert(key.clone(), zm);
            }

            let codec =
                ColumnCodec::read_from(data, &mut pos).map_err(|e| format!("codec: {e}"))?;
            let col_type = infer_column_type_from_codec(&codec);
            col_defs.push(ColumnDef::new(&key_str, col_type));
            columns.insert(key, codec);
        }

        let schema = TableSchema::new(label.as_str(), table_id, col_defs);
        let table = NodeTable::from_columns(schema, columns, zone_maps, row_count);
        node_tables.push(table);
        label_to_table_id.insert(label.clone(), table_id);
        table_id_to_label.push(label);
    }

    // Relationship tables.
    let num_rel_tables = read_u32(data, &mut pos)? as usize;
    let mut rel_tables = Vec::with_capacity(num_rel_tables);
    let mut edge_type_to_rel_id: FxHashMap<arcstr::ArcStr, Vec<u16>> = FxHashMap::default();
    let mut rel_table_id_to_type: Vec<arcstr::ArcStr> = Vec::with_capacity(num_rel_tables);

    for rel_idx in 0..num_rel_tables {
        let rel_table_id = u16::try_from(rel_idx).unwrap_or(0);
        let edge_type = read_string(data, &mut pos)?;
        let edge_type = arcstr::ArcStr::from(edge_type.as_str());
        let src_tid = read_u16(data, &mut pos)?;
        let dst_tid = read_u16(data, &mut pos)?;

        let fwd = CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("fwd CSR: {e}"))?;

        let has_bwd = *data.get(pos).ok_or("truncated bwd flag")?;
        pos += 1;
        let bwd = if has_bwd == 1 {
            Some(CsrAdjacency::read_from(data, &mut pos).map_err(|e| format!("bwd CSR: {e}"))?)
        } else {
            None
        };

        let num_props = read_u32(data, &mut pos)? as usize;
        let mut properties: FxHashMap<PropertyKey, ColumnCodec> = FxHashMap::default();
        let mut prop_defs = Vec::with_capacity(num_props);
        for _ in 0..num_props {
            let key_str = read_string(data, &mut pos)?;
            let key = PropertyKey::new(&key_str);
            let codec =
                ColumnCodec::read_from(data, &mut pos).map_err(|e| format!("edge codec: {e}"))?;
            let col_type = infer_column_type_from_codec(&codec);
            prop_defs.push(ColumnDef::new(&key_str, col_type));
            properties.insert(key, codec);
        }

        let src_label = table_id_to_label
            .get(src_tid as usize)
            .cloned()
            .unwrap_or_default();
        let dst_label = table_id_to_label
            .get(dst_tid as usize)
            .cloned()
            .unwrap_or_default();

        let schema = EdgeSchema::new(
            edge_type.as_str(),
            rel_table_id,
            src_label.as_str(),
            dst_label.as_str(),
            prop_defs,
        );

        let table = RelTable::new(schema, fwd, bwd, properties, src_tid, dst_tid);
        edge_type_to_rel_id
            .entry(edge_type.clone())
            .or_default()
            .push(rel_table_id);
        rel_table_id_to_type.push(edge_type);
        rel_tables.push(table);
    }

    // Compute statistics.
    let mut stats = Statistics::new();
    let mut total_nodes = 0u64;
    let mut total_edges = 0u64;
    for (idx, nt) in node_tables.iter().enumerate() {
        let c = nt.len() as u64;
        total_nodes += c;
        stats.update_label(table_id_to_label[idx].as_str(), LabelStatistics::new(c));
    }
    let mut edge_counts: FxHashMap<&str, u64> = FxHashMap::default();
    for (idx, rt) in rel_tables.iter().enumerate() {
        let c = rt.num_edges() as u64;
        total_edges += c;
        *edge_counts
            .entry(rel_table_id_to_type[idx].as_str())
            .or_default() += c;
    }
    for (et, count) in edge_counts {
        stats.update_edge_type(et, EdgeTypeStatistics::new(count, 0.0, 0.0));
    }
    stats.total_nodes = total_nodes;
    stats.total_edges = total_edges;

    let mut store = CompactStore::new(
        node_tables,
        label_to_table_id,
        rel_tables,
        edge_type_to_rel_id,
        table_id_to_label,
        rel_table_id_to_type,
        stats,
    );

    // ID maps.
    if preserves_ids {
        let node_map_len = read_u32(data, &mut pos)? as usize;
        let mut node_id_map = FxHashMap::with_capacity_and_hasher(node_map_len, Default::default());
        let num_tables = store.node_tables_by_id.len();
        let mut node_offset_to_id: Vec<Vec<NodeId>> = vec![Vec::new(); num_tables];
        for _ in 0..node_map_len {
            let nid = NodeId::new(read_u64(data, &mut pos)?);
            let tid = read_u16(data, &mut pos)?;
            let off = read_u64(data, &mut pos)?;
            node_id_map.insert(nid, (tid, off));
            let off_idx = usize::try_from(off).unwrap_or(usize::MAX);
            if let Some(rev) = node_offset_to_id.get_mut(tid as usize) {
                while rev.len() <= off_idx {
                    rev.push(NodeId::INVALID);
                }
                rev[off_idx] = nid;
            }
        }

        let edge_map_len = read_u32(data, &mut pos)? as usize;
        let mut edge_id_map = FxHashMap::with_capacity_and_hasher(edge_map_len, Default::default());
        let num_rel = store.rel_tables_by_id.len();
        let mut edge_offset_to_id: Vec<Vec<EdgeId>> = vec![Vec::new(); num_rel];
        for _ in 0..edge_map_len {
            let eid = EdgeId::new(read_u64(data, &mut pos)?);
            let rtid = read_u16(data, &mut pos)?;
            let csr_pos = read_u64(data, &mut pos)?;
            edge_id_map.insert(eid, (rtid, csr_pos));
            let pos_idx = usize::try_from(csr_pos).unwrap_or(usize::MAX);
            if let Some(rev) = edge_offset_to_id.get_mut(rtid as usize) {
                while rev.len() <= pos_idx {
                    rev.push(EdgeId::INVALID);
                }
                rev[pos_idx] = eid;
            }
        }

        store.set_id_maps(
            node_id_map,
            edge_id_map,
            node_offset_to_id,
            edge_offset_to_id,
        );
    }

    Ok(store)
}

// ── Write helpers ──────────────────────────────────────────────────

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_len(buf: &mut Vec<u8>, v: usize) {
    let n = u32::try_from(v).expect("length exceeds u32::MAX in compact section");
    buf.extend_from_slice(&n.to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let slen = u16::try_from(bytes.len()).expect("string exceeds u16::MAX in compact section");
    write_u16(buf, slen);
    buf.extend_from_slice(bytes);
}

fn write_zone_map(buf: &mut Vec<u8>, zm: &ZoneMap) {
    write_len(buf, zm.null_count);
    write_len(buf, zm.row_count);
    // Encode min/max as (tag, value) pairs.
    write_optional_value(buf, &zm.min);
    write_optional_value(buf, &zm.max);
}

fn write_optional_value(buf: &mut Vec<u8>, v: &Option<grafeo_common::types::Value>) {
    match v {
        None => buf.push(0),
        Some(grafeo_common::types::Value::Int64(n)) => {
            buf.push(1);
            // Store as raw i64 bytes to avoid sign-loss lint.
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Some(grafeo_common::types::Value::Bool(b)) => {
            buf.push(2);
            buf.push(u8::from(*b));
        }
        Some(grafeo_common::types::Value::String(s)) => {
            buf.push(3);
            write_str(buf, s.as_str());
        }
        Some(_) => {
            // Unsupported type for zone map: write as absent.
            buf.push(0);
        }
    }
}

// ── Read helpers ───────────────────────────────────────────────────

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, String> {
    if *pos + 2 > data.len() {
        return Err("truncated u16".into());
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let slen = read_u16(data, pos)? as usize;
    if *pos + slen > data.len() {
        return Err("truncated string".into());
    }
    let s =
        std::str::from_utf8(&data[*pos..*pos + slen]).map_err(|_| "invalid UTF-8".to_string())?;
    *pos += slen;
    Ok(s.to_string())
}

fn read_zone_map(data: &[u8], pos: &mut usize) -> Result<ZoneMap, String> {
    let null_count = read_u32(data, pos)? as usize;
    let row_count = read_u32(data, pos)? as usize;
    let min = read_optional_value(data, pos)?;
    let max = read_optional_value(data, pos)?;
    Ok(ZoneMap {
        min,
        max,
        null_count,
        row_count,
    })
}

fn read_optional_value(
    data: &[u8],
    pos: &mut usize,
) -> Result<Option<grafeo_common::types::Value>, String> {
    let tag = *data.get(*pos).ok_or("truncated value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            // Read raw i64 bytes (written via i64::to_le_bytes).
            if *pos + 8 > data.len() {
                return Err("truncated i64 value".into());
            }
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Some(grafeo_common::types::Value::Int64(v)))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated bool")?;
            *pos += 1;
            Ok(Some(grafeo_common::types::Value::Bool(b != 0)))
        }
        3 => {
            let s = read_string(data, pos)?;
            Ok(Some(grafeo_common::types::Value::String(
                arcstr::ArcStr::from(s.as_str()),
            )))
        }
        _ => Err(format!("unknown value tag {tag}")),
    }
}

fn infer_column_type_from_codec(codec: &ColumnCodec) -> ColumnType {
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
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;
    use crate::graph::traits::GraphStore;
    use grafeo_common::types::Value;

    #[test]
    fn test_round_trip_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));

        let bytes = section.serialize().unwrap();
        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();

        let restored = section2.store().unwrap();
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
    }

    #[test]
    fn test_round_trip_nodes_and_edges() {
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        assert!(compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(restored.preserves_ids());
        assert_eq!(restored.node_count(), 3);
        assert_eq!(restored.edge_count(), 2);

        // Verify original IDs survive.
        let alix_node = restored.get_node(alix).expect("Alix by original ID");
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(arcstr::ArcStr::from("Alix")))
        );
        assert_eq!(
            alix_node.properties.get(&PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );

        // Verify edge traversal.
        let neighbors = restored.neighbors(alix, crate::graph::Direction::Outgoing);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], amsterdam);
    }

    #[test]
    fn test_round_trip_without_id_preservation() {
        use crate::graph::compact::from_graph_store;

        let lpg = LpgStore::new().unwrap();
        let a = lpg.create_node(&["Node"]);
        lpg.set_node_property(a, "val", Value::Int64(42));
        let b = lpg.create_node(&["Node"]);
        lpg.set_node_property(b, "val", Value::Int64(99));
        lpg.create_edge(a, b, "LINK");

        let compact = from_graph_store(&lpg).unwrap();
        assert!(!compact.preserves_ids());

        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert!(!restored.preserves_ids());
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }

    #[test]
    fn test_crc_integrity() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Test"]);
        let compact = from_graph_store_preserving_ids(&store).unwrap();

        let section = CompactStoreSection::new(Arc::new(compact));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte in the middle.
        if bytes.len() > 10 {
            bytes[10] ^= 0xFF;
        }

        let mut section2 = CompactStoreSection::empty();
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn test_section_type_and_version() {
        let section = CompactStoreSection::empty();
        assert_eq!(section.section_type(), SectionType::CompactStore);
        assert_eq!(section.version(), FORMAT_VERSION);
        assert!(!section.is_dirty());
        assert_eq!(section.memory_usage(), 0);
    }

    #[test]
    fn test_dirty_tracking() {
        let section = CompactStoreSection::empty();
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn test_round_trip_bool_column() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Item"]);
        store.set_node_property(a, "active", Value::Bool(true));
        let b = store.create_node(&["Item"]);
        store.set_node_property(b, "active", Value::Bool(false));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        assert_eq!(
            restored.get_node_property(a, &PropertyKey::new("active")),
            Some(Value::Bool(true))
        );
        assert_eq!(
            restored.get_node_property(b, &PropertyKey::new("active")),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_round_trip_edge_properties() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let e = store.create_edge(a, b, "LINK");
        store.set_edge_property(e, "weight", Value::Int64(5));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let section = CompactStoreSection::new(Arc::new(compact));
        let bytes = section.serialize().unwrap();

        let mut section2 = CompactStoreSection::empty();
        section2.deserialize(&bytes).unwrap();
        let restored = section2.store().unwrap();

        // Find the edge via traversal.
        let edges = restored.edges_from(a, crate::graph::Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let edge = restored.get_edge(edges[0].1).unwrap();
        assert_eq!(
            edge.properties.get(&PropertyKey::new("weight")),
            Some(&Value::Int64(5))
        );
    }
}
