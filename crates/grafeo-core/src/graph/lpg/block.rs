//! Block-based binary format for LPG data sections (v2).
//!
//! Replaces the full-bincode serialization with a structured layout that
//! supports mmap, per-block CRC verification, and incremental writes.
//!
//! # Layout
//!
//! ```text
//! [SectionHeader 64B]
//! [BlockDirectory  block_count * 24B]
//! [StringTable     variable]
//! [NodeData        node_count * 16B]
//! [EdgeData        edge_count * 32B]
//! [LabelAssign     variable]
//! [PropertyColumn  variable, one per property key]
//! [NamedGraphs     variable]
//! ```
//!
//! All multi-byte integers are little-endian. Blocks are NOT page-aligned
//! within the section (the container handles page alignment of the section
//! as a whole).

use std::collections::BTreeMap;

use crate::codec::limits::{checked_u16, checked_u32};
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};

// ── Magic and version ──────────────────────────────────────────────

/// Magic bytes identifying block-based LPG section format.
pub const LPG_BLOCK_MAGIC: [u8; 4] = *b"LPGB";

/// Current block format version.
pub const LPG_BLOCK_VERSION: u8 = 1;

// ── Block types ────────────────────────────────────────────────────

/// Identifies the type of a block within the section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    /// String table: deduplicates all strings (labels, property keys, string values).
    StringTable = 1,
    /// Node data: packed array of node records.
    NodeData = 2,
    /// Edge data: packed array of edge records.
    EdgeData = 3,
    /// Label assignments: per-node label lists.
    LabelAssignment = 4,
    /// Property column: values for a single property key across all entities.
    PropertyColumn = 5,
    /// Named graph: self-contained sub-section for a named graph.
    NamedGraph = 6,
}

// ── Value type tags ────────────────────────────────────────────────

/// Type tags for property value encoding within property columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueTag {
    Null = 0,
    Bool = 1,
    Int64 = 2,
    Float64 = 3,
    String = 4,
    Bytes = 5,
    Date = 6,
    Time = 7,
    Timestamp = 8,
    ZonedDatetime = 9,
    Duration = 10,
    List = 11,
    Map = 12,
    Vector = 13,
    Path = 14,
}

// ── Block directory entry ──────────────────────────────────────────

/// A directory entry describing a single block within the section.
///
/// 24 bytes, packed sequentially after the section header.
#[derive(Debug, Clone, Copy)]
pub struct BlockDirEntry {
    /// Block type.
    pub block_type: u8,
    /// Reserved for future use.
    pub _reserved: [u8; 3],
    /// Byte offset from the start of the section.
    pub offset: u32,
    /// Length of the block in bytes.
    pub length: u32,
    /// CRC-32 of the block data.
    pub checksum: u32,
    /// For PropertyColumn: index into string table for the property key.
    /// For NamedGraph: index into string table for the graph name.
    /// Zero for other block types.
    pub key_string_index: u32,
    /// Sub-type or flags. For PropertyColumn: 0 = node property, 1 = edge property.
    pub sub_type: u32,
}

impl BlockDirEntry {
    const SIZE: usize = 24;

    /// Byte range of the block within the section, computed in `usize` so a
    /// corrupt entry cannot overflow the `u32` fields' sum.
    fn range(&self) -> std::ops::Range<usize> {
        let start = self.offset as usize;
        start..start.saturating_add(self.length as usize)
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.push(self.block_type);
        buf.extend_from_slice(&self._reserved);
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.length.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf.extend_from_slice(&self.key_string_index.to_le_bytes());
        buf.extend_from_slice(&self.sub_type.to_le_bytes());
    }

    fn read_from(data: &[u8]) -> Option<Self> {
        if data.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            block_type: data[0],
            _reserved: [data[1], data[2], data[3]],
            offset: u32::from_le_bytes(data[4..8].try_into().ok()?),
            length: u32::from_le_bytes(data[8..12].try_into().ok()?),
            checksum: u32::from_le_bytes(data[12..16].try_into().ok()?),
            key_string_index: u32::from_le_bytes(data[16..20].try_into().ok()?),
            sub_type: u32::from_le_bytes(data[20..24].try_into().ok()?),
        })
    }
}

// ── Section header ─────────────────────────────────────────────────

/// Fixed 64-byte header at the start of a block-based LPG section.
const HEADER_SIZE: usize = 64;

struct SectionHeader {
    magic: [u8; 4],
    version: u8,
    flags: u8,
    block_count: u16,
    node_count: u64,
    edge_count: u64,
    epoch: u64,
    named_graph_count: u32,
    _reserved: [u8; 28],
}

impl SectionHeader {
    fn write_to(&self, buf: &mut Vec<u8>) {
        let start = buf.len();
        buf.extend_from_slice(&self.magic); // 4
        buf.push(self.version); // 1
        buf.push(self.flags); // 1
        buf.extend_from_slice(&self.block_count.to_le_bytes()); // 2
        buf.extend_from_slice(&self.node_count.to_le_bytes()); // 8
        buf.extend_from_slice(&self.edge_count.to_le_bytes()); // 8
        buf.extend_from_slice(&self.epoch.to_le_bytes()); // 8
        buf.extend_from_slice(&self.named_graph_count.to_le_bytes()); // 4
        buf.extend_from_slice(&self._reserved); // 28 (pad to 64 total)
        debug_assert_eq!(buf.len() - start, HEADER_SIZE);
    }

    fn read_from(data: &[u8]) -> Option<Self> {
        if data.len() < HEADER_SIZE {
            return None;
        }
        let magic: [u8; 4] = data[0..4].try_into().ok()?;
        if magic != LPG_BLOCK_MAGIC {
            return None;
        }
        Some(Self {
            magic,
            version: data[4],
            flags: data[5],
            block_count: u16::from_le_bytes(data[6..8].try_into().ok()?),
            node_count: u64::from_le_bytes(data[8..16].try_into().ok()?),
            edge_count: u64::from_le_bytes(data[16..24].try_into().ok()?),
            epoch: u64::from_le_bytes(data[24..32].try_into().ok()?),
            named_graph_count: u32::from_le_bytes(data[32..36].try_into().ok()?),
            _reserved: {
                let mut r = [0u8; 28];
                r.copy_from_slice(&data[36..64]);
                r
            },
        })
    }
}

// ── String table ───────────────────────────────────────────────────

/// Builds a deduplicated string table and maps strings to indices.
struct StringTableBuilder {
    strings: Vec<String>,
    index: hashbrown::HashMap<String, u32>,
}

impl StringTableBuilder {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            index: hashbrown::HashMap::new(),
        }
    }

    /// Interns a string, returning its index.
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        // More than u32::MAX strings cannot be held in memory in practice; if it
        // ever happens the index saturates and `serialize` rejects the table
        // before anything is written.
        let idx = u32::try_from(self.strings.len()).unwrap_or(u32::MAX);
        self.strings.push(s.to_owned());
        self.index.insert(s.to_owned(), idx);
        idx
    }

    /// Serializes the string table: [count:u32] [offsets:u32*count] [packed strings]
    ///
    /// # Errors
    ///
    /// Fails if the string count, a string's length or the packed size does
    /// not fit the format's `u32` fields.
    fn serialize(&self) -> Result<Vec<u8>> {
        let count = checked_u32(self.strings.len(), "LPG string table count")?;
        let mut packed = Vec::new();
        let mut offsets = Vec::with_capacity(self.strings.len());
        for s in &self.strings {
            offsets.push(checked_u32(packed.len(), "LPG string table size")?);
            let bytes = s.as_bytes();
            packed.extend_from_slice(&checked_u32(bytes.len(), "LPG string length")?.to_le_bytes());
            packed.extend_from_slice(bytes);
        }

        let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + packed.len());
        buf.extend_from_slice(&count.to_le_bytes());
        for off in &offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        buf.extend_from_slice(&packed);
        Ok(buf)
    }
}

/// Read-only view into a serialized string table.
struct StringTableReader<'a> {
    data: &'a [u8],
    count: u32,
    offsets_start: usize,
    packed_start: usize,
}

impl<'a> StringTableReader<'a> {
    fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let count = u32::from_le_bytes(data[0..4].try_into().ok()?);
        let offsets_start: usize = 4;
        let packed_start = offsets_start.checked_add((count as usize).checked_mul(4)?)?;
        if data.len() < packed_start {
            return None;
        }
        Some(Self {
            data,
            count,
            offsets_start,
            packed_start,
        })
    }

    fn get(&self, index: u32) -> Option<&'a str> {
        if index >= self.count {
            return None;
        }
        let off_pos = self.offsets_start + (index as usize) * 4;
        let rel_offset =
            u32::from_le_bytes(self.data[off_pos..off_pos + 4].try_into().ok()?) as usize;
        let abs_offset = self.packed_start.checked_add(rel_offset)?;
        if abs_offset.checked_add(4)? > self.data.len() {
            return None;
        }
        let len =
            u32::from_le_bytes(self.data[abs_offset..abs_offset + 4].try_into().ok()?) as usize;
        let str_start = abs_offset + 4;
        if str_start + len > self.data.len() {
            return None;
        }
        std::str::from_utf8(&self.data[str_start..str_start + len]).ok()
    }
}

// ── Value encoding ─────────────────────────────────────────────────

/// Encodes a `Value` into the binary format used in property columns.
///
/// Format: [tag:u8] [payload:variable]
fn encode_value(val: &Value, strings: &mut StringTableBuilder, buf: &mut Vec<u8>) -> Result<()> {
    match val {
        Value::Null => buf.push(ValueTag::Null as u8),
        Value::Bool(b) => {
            buf.push(ValueTag::Bool as u8);
            buf.push(u8::from(*b));
        }
        Value::Int64(n) => {
            buf.push(ValueTag::Int64 as u8);
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Value::Float64(f) => {
            buf.push(ValueTag::Float64 as u8);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            buf.push(ValueTag::String as u8);
            let idx = strings.intern(s);
            buf.extend_from_slice(&idx.to_le_bytes());
        }
        Value::Bytes(b) => {
            buf.push(ValueTag::Bytes as u8);
            buf.extend_from_slice(&checked_u32(b.len(), "LPG bytes value length")?.to_le_bytes());
            buf.extend_from_slice(b);
        }
        Value::Date(d) => {
            buf.push(ValueTag::Date as u8);
            buf.extend_from_slice(&d.as_days().to_le_bytes());
        }
        Value::Time(t) => {
            buf.push(ValueTag::Time as u8);
            buf.extend_from_slice(&t.as_nanos().to_le_bytes());
            let offset = t.offset_seconds().unwrap_or(i32::MIN);
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        Value::Timestamp(ts) => {
            buf.push(ValueTag::Timestamp as u8);
            buf.extend_from_slice(&ts.as_millis().to_le_bytes());
        }
        Value::ZonedDatetime(zdt) => {
            buf.push(ValueTag::ZonedDatetime as u8);
            buf.extend_from_slice(&zdt.as_timestamp().as_millis().to_le_bytes());
            buf.extend_from_slice(&zdt.offset_seconds().to_le_bytes());
        }
        Value::Duration(d) => {
            buf.push(ValueTag::Duration as u8);
            buf.extend_from_slice(&d.months().to_le_bytes());
            buf.extend_from_slice(&d.days().to_le_bytes());
            buf.extend_from_slice(&d.nanos().to_le_bytes());
        }
        Value::List(items) => {
            buf.push(ValueTag::List as u8);
            buf.extend_from_slice(&checked_u32(items.len(), "LPG list length")?.to_le_bytes());
            for item in items.iter() {
                encode_value(item, strings, buf)?;
            }
        }
        Value::Map(map) => {
            buf.push(ValueTag::Map as u8);
            buf.extend_from_slice(&checked_u32(map.len(), "LPG map size")?.to_le_bytes());
            for (key, val) in map.iter() {
                let key_idx = strings.intern(key.as_ref());
                buf.extend_from_slice(&key_idx.to_le_bytes());
                encode_value(val, strings, buf)?;
            }
        }
        Value::Vector(v) => {
            buf.push(ValueTag::Vector as u8);
            buf.extend_from_slice(&checked_u32(v.len(), "LPG vector dimension")?.to_le_bytes());
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Path { nodes, edges } => {
            buf.push(ValueTag::Path as u8);
            buf.extend_from_slice(&checked_u32(nodes.len(), "LPG path node count")?.to_le_bytes());
            for n in nodes.iter() {
                encode_value(n, strings, buf)?;
            }
            buf.extend_from_slice(&checked_u32(edges.len(), "LPG path edge count")?.to_le_bytes());
            for e in edges.iter() {
                encode_value(e, strings, buf)?;
            }
        }
        // GCounter and OnCounter: encode as Map for forward compatibility
        _ => {
            buf.push(ValueTag::Null as u8);
        }
    }
    Ok(())
}

/// Decodes a `Value` from the binary format.
fn decode_value(data: &[u8], pos: &mut usize, strings: &StringTableReader<'_>) -> Result<Value> {
    if *pos >= data.len() {
        return Err(Error::Serialization(
            "unexpected end of data in value".to_string(),
        ));
    }
    let tag = data[*pos];
    *pos += 1;

    match tag {
        0 => Ok(Value::Null),
        1 => {
            // Bool
            ensure_remaining(data, *pos, 1)?;
            let v = data[*pos] != 0;
            *pos += 1;
            Ok(Value::Bool(v))
        }
        2 => {
            // Int64
            ensure_remaining(data, *pos, 8)?;
            let v = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Value::Int64(v))
        }
        3 => {
            // Float64
            ensure_remaining(data, *pos, 8)?;
            let v = f64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Value::Float64(v))
        }
        4 => {
            // String (index into string table)
            ensure_remaining(data, *pos, 4)?;
            let idx = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            let s = strings
                .get(idx)
                .ok_or_else(|| Error::Serialization(format!("invalid string index {idx}")))?;
            Ok(Value::String(s.into()))
        }
        5 => {
            // Bytes
            ensure_remaining(data, *pos, 4)?;
            let len = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            ensure_remaining(data, *pos, len)?;
            let bytes: Arc<[u8]> = data[*pos..*pos + len].into();
            *pos += len;
            Ok(Value::Bytes(bytes))
        }
        6 => {
            // Date (i32 days since epoch)
            ensure_remaining(data, *pos, 4)?;
            let days = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(Value::Date(grafeo_common::types::Date::from_days(days)))
        }
        7 => {
            // Time (u64 nanos + i32 offset)
            ensure_remaining(data, *pos, 12)?;
            let nanos = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            let offset = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            let mut time = grafeo_common::types::Time::from_nanos(nanos)
                .unwrap_or_else(|| grafeo_common::types::Time::from_nanos(0).unwrap());
            if offset != i32::MIN {
                time = time.with_offset(offset);
            }
            Ok(Value::Time(time))
        }
        8 => {
            // Timestamp (i64 millis)
            ensure_remaining(data, *pos, 8)?;
            let millis = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Value::Timestamp(
                grafeo_common::types::Timestamp::from_millis(millis),
            ))
        }
        9 => {
            // ZonedDatetime (i64 millis + i32 offset)
            ensure_remaining(data, *pos, 12)?;
            let millis = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            let offset = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(Value::ZonedDatetime(
                grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                    grafeo_common::types::Timestamp::from_millis(millis),
                    offset,
                ),
            ))
        }
        10 => {
            // Duration (i64 months + i64 days + i64 nanos = 24 bytes)
            ensure_remaining(data, *pos, 24)?;
            let months = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            let days = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            let nanos = i64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(Value::Duration(grafeo_common::types::Duration::new(
                months, days, nanos,
            )))
        }
        11 => {
            // List
            ensure_remaining(data, *pos, 4)?;
            let count = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let mut items = Vec::with_capacity(count.min(data.len()));
            for _ in 0..count {
                items.push(decode_value(data, pos, strings)?);
            }
            Ok(Value::List(items.into()))
        }
        12 => {
            // Map
            ensure_remaining(data, *pos, 4)?;
            let count = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let mut map = BTreeMap::new();
            for _ in 0..count {
                ensure_remaining(data, *pos, 4)?;
                let key_idx = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
                *pos += 4;
                let key_str = strings.get(key_idx).ok_or_else(|| {
                    Error::Serialization(format!("invalid map key string index {key_idx}"))
                })?;
                let val = decode_value(data, pos, strings)?;
                map.insert(key_str.into(), val);
            }
            Ok(Value::Map(Arc::new(map)))
        }
        13 => {
            // Vector
            ensure_remaining(data, *pos, 4)?;
            let count = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let byte_len = count
                .checked_mul(4)
                .ok_or_else(|| Error::Serialization("vector length overflow".to_string()))?;
            ensure_remaining(data, *pos, byte_len)?;
            let mut floats = Vec::with_capacity(count.min(data.len() / 4));
            for _ in 0..count {
                let f = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
                *pos += 4;
                floats.push(f);
            }
            Ok(Value::Vector(floats.into()))
        }
        14 => {
            // Path
            ensure_remaining(data, *pos, 4)?;
            let node_count = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let mut nodes = Vec::with_capacity(node_count.min(data.len()));
            for _ in 0..node_count {
                nodes.push(decode_value(data, pos, strings)?);
            }
            ensure_remaining(data, *pos, 4)?;
            let edge_count = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let mut edges = Vec::with_capacity(edge_count.min(data.len()));
            for _ in 0..edge_count {
                edges.push(decode_value(data, pos, strings)?);
            }
            Ok(Value::Path {
                nodes: nodes.into(),
                edges: edges.into(),
            })
        }
        other => Err(Error::Serialization(format!("unknown value tag {other}"))),
    }
}

fn ensure_remaining(data: &[u8], pos: usize, need: usize) -> Result<()> {
    let end = pos.checked_add(need).ok_or_else(|| {
        Error::Serialization(format!(
            "integer overflow: offset {pos} + need {need} exceeds usize"
        ))
    })?;
    if end > data.len() {
        return Err(Error::Serialization(format!(
            "unexpected end of data: need {} bytes at offset {}, have {}",
            need,
            pos,
            data.len()
        )));
    }
    Ok(())
}

use std::sync::Arc;

// ── Writer ─────────────────────────────────────────────────────────

/// Intermediate node representation for the block writer.
pub(crate) struct BlockNode {
    pub id: NodeId,
    pub labels: Vec<String>,
    pub properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

/// Intermediate edge representation for the block writer.
pub(crate) struct BlockEdge {
    pub id: EdgeId,
    pub src: NodeId,
    pub dst: NodeId,
    pub edge_type: String,
    pub properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

/// Intermediate named graph representation for the block writer.
pub(crate) struct BlockNamedGraph {
    pub name: String,
    pub nodes: Vec<BlockNode>,
    pub edges: Vec<BlockEdge>,
}

/// Serializes LPG data into the block-based binary format.
pub(crate) fn write_blocks(
    nodes: &[BlockNode],
    edges: &[BlockEdge],
    named_graphs: &[BlockNamedGraph],
    epoch: u64,
) -> Result<Vec<u8>> {
    let mut strings = StringTableBuilder::new();
    let mut blocks: Vec<(BlockType, Vec<u8>, u32, u32)> = Vec::new(); // (type, data, key_idx, sub_type)

    // Phase 1: intern all strings first so the table is complete before
    // we write any blocks that reference string indices.
    for node in nodes {
        for label in &node.labels {
            strings.intern(label);
        }
        for (key, entries) in &node.properties {
            strings.intern(key);
            intern_values_strings(entries, &mut strings);
        }
    }
    for edge in edges {
        strings.intern(&edge.edge_type);
        for (key, entries) in &edge.properties {
            strings.intern(key);
            intern_values_strings(entries, &mut strings);
        }
    }
    for graph in named_graphs {
        strings.intern(&graph.name);
        for node in &graph.nodes {
            for label in &node.labels {
                strings.intern(label);
            }
            for (key, entries) in &node.properties {
                strings.intern(key);
                intern_values_strings(entries, &mut strings);
            }
        }
        for edge in &graph.edges {
            strings.intern(&edge.edge_type);
            for (key, entries) in &edge.properties {
                strings.intern(key);
                intern_values_strings(entries, &mut strings);
            }
        }
    }

    // Block 0: String table
    let st_data = strings.serialize()?;
    blocks.push((BlockType::StringTable, st_data, 0, 0));

    // Block 1: Node data
    // Format: for each node [id:u64]
    let mut node_buf = Vec::with_capacity(nodes.len() * 8);
    for node in nodes {
        node_buf.extend_from_slice(&node.id.as_u64().to_le_bytes());
    }
    blocks.push((BlockType::NodeData, node_buf, 0, 0));

    // Block 2: Edge data
    // Format: for each edge [id:u64][src:u64][dst:u64][type_str_idx:u32]
    let mut edge_buf = Vec::with_capacity(edges.len() * 28);
    for edge in edges {
        edge_buf.extend_from_slice(&edge.id.as_u64().to_le_bytes());
        edge_buf.extend_from_slice(&edge.src.as_u64().to_le_bytes());
        edge_buf.extend_from_slice(&edge.dst.as_u64().to_le_bytes());
        let type_idx = strings.intern(&edge.edge_type);
        edge_buf.extend_from_slice(&type_idx.to_le_bytes());
    }
    blocks.push((BlockType::EdgeData, edge_buf, 0, 0));

    // Block 3: Label assignments
    // Format: [node_count:u32] for each node [label_count:u16][label_idx:u32*count]
    let mut label_buf = Vec::new();
    label_buf.extend_from_slice(&checked_u32(nodes.len(), "LPG node count")?.to_le_bytes());
    for node in nodes {
        label_buf.extend_from_slice(
            &checked_u16(node.labels.len(), "LPG labels per node")?.to_le_bytes(),
        );
        for label in &node.labels {
            let idx = strings.intern(label);
            label_buf.extend_from_slice(&idx.to_le_bytes());
        }
    }
    blocks.push((BlockType::LabelAssignment, label_buf, 0, 0));

    // Property columns: one block per (property_key, entity_type) pair
    // Node properties
    let node_prop_keys = collect_property_keys(nodes.iter().flat_map(|n| n.properties.iter()));
    for key in &node_prop_keys {
        let key_idx = strings.intern(key);
        let col_data = write_property_column(key, nodes.iter(), &mut strings, true)?;
        blocks.push((BlockType::PropertyColumn, col_data, key_idx, 0));
    }

    // Edge properties
    let edge_prop_keys = collect_property_keys(edges.iter().flat_map(|e| e.properties.iter()));
    for key in &edge_prop_keys {
        let key_idx = strings.intern(key);
        let col_data = write_property_column_edges(key, edges.iter(), &mut strings)?;
        blocks.push((BlockType::PropertyColumn, col_data, key_idx, 1));
    }

    // Named graphs: each is a self-contained sub-section
    for graph in named_graphs {
        let name_idx = strings.intern(&graph.name);
        let graph_data = write_blocks(&graph.nodes, &graph.edges, &[], epoch)?;
        blocks.push((BlockType::NamedGraph, graph_data, name_idx, 0));
    }

    // Re-serialize the string table now that all strings are interned
    // (property column writing may have added new strings)
    let final_st_data = strings.serialize()?;
    blocks[0].1 = final_st_data;

    // Assemble the final output
    let block_count = blocks.len();
    let dir_size = block_count * BlockDirEntry::SIZE;

    // Calculate offsets
    let mut data_offset = HEADER_SIZE + dir_size;
    let mut dir_entries = Vec::with_capacity(block_count);

    for (block_type, block_data, key_idx, sub_type) in &blocks {
        let checksum = crc32fast::hash(block_data);
        // The directory stores u32 offsets and lengths, so an LPG section is
        // limited to 4 GiB (#392); refuse to write a larger one.
        dir_entries.push(BlockDirEntry {
            block_type: *block_type as u8,
            _reserved: [0; 3],
            offset: checked_u32(data_offset, "LPG block offset")?,
            length: checked_u32(block_data.len(), "LPG block length")?,
            checksum,
            key_string_index: *key_idx,
            sub_type: *sub_type,
        });
        data_offset += block_data.len();
    }

    // Write output
    let total_size = data_offset;
    let mut output = Vec::with_capacity(total_size);

    // Header
    let header = SectionHeader {
        magic: LPG_BLOCK_MAGIC,
        version: LPG_BLOCK_VERSION,
        flags: 0,
        block_count: checked_u16(block_count, "LPG block count")?,
        node_count: nodes.len() as u64,
        edge_count: edges.len() as u64,
        epoch,
        named_graph_count: checked_u32(named_graphs.len(), "LPG named graph count")?,
        _reserved: [0; 28],
    };
    header.write_to(&mut output);

    // Directory
    for entry in &dir_entries {
        entry.write_to(&mut output);
    }

    // Block data
    for (_, block_data, _, _) in &blocks {
        output.extend_from_slice(block_data);
    }

    debug_assert_eq!(output.len(), total_size);
    Ok(output)
}

fn intern_values_strings(entries: &[(EpochId, Value)], strings: &mut StringTableBuilder) {
    for (_, value) in entries {
        intern_value_strings(value, strings);
    }
}

fn intern_value_strings(value: &Value, strings: &mut StringTableBuilder) {
    match value {
        Value::String(s) => {
            strings.intern(s);
        }
        Value::List(items) => {
            for item in items.iter() {
                intern_value_strings(item, strings);
            }
        }
        Value::Map(map) => {
            for (key, val) in map.iter() {
                strings.intern(key.as_ref());
                intern_value_strings(val, strings);
            }
        }
        Value::Path { nodes, edges } => {
            for n in nodes.iter() {
                intern_value_strings(n, strings);
            }
            for e in edges.iter() {
                intern_value_strings(e, strings);
            }
        }
        _ => {}
    }
}

fn collect_property_keys<'a>(
    props: impl Iterator<Item = &'a (String, Vec<(EpochId, Value)>)>,
) -> Vec<String> {
    let mut keys: Vec<String> = props.map(|(k, _)| k.clone()).collect();
    keys.sort();
    keys.dedup();
    keys
}

/// Writes a property column for node properties.
///
/// Format: [entry_count:u32] for each entry [entity_id:u64][version_count:u16]
///         for each version [epoch:u64][encoded_value]
fn write_property_column<'a>(
    key: &str,
    nodes: impl Iterator<Item = &'a BlockNode>,
    strings: &mut StringTableBuilder,
    _is_node: bool,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut entries: Vec<(u64, &[(EpochId, Value)])> = Vec::new();

    for node in nodes {
        for (k, versions) in &node.properties {
            if k == key && !versions.is_empty() {
                entries.push((node.id.as_u64(), versions));
            }
        }
    }

    buf.extend_from_slice(
        &checked_u32(entries.len(), "LPG property column entry count")?.to_le_bytes(),
    );
    for (entity_id, versions) in &entries {
        buf.extend_from_slice(&entity_id.to_le_bytes());
        buf.extend_from_slice(
            &checked_u16(versions.len(), "LPG property version count")?.to_le_bytes(),
        );
        for (epoch, value) in *versions {
            buf.extend_from_slice(&epoch.as_u64().to_le_bytes());
            encode_value(value, strings, &mut buf)?;
        }
    }

    Ok(buf)
}

/// Writes a property column for edge properties.
fn write_property_column_edges<'a>(
    key: &str,
    edges: impl Iterator<Item = &'a BlockEdge>,
    strings: &mut StringTableBuilder,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut entries: Vec<(u64, &[(EpochId, Value)])> = Vec::new();

    for edge in edges {
        for (k, versions) in &edge.properties {
            if k == key && !versions.is_empty() {
                entries.push((edge.id.as_u64(), versions));
            }
        }
    }

    buf.extend_from_slice(
        &checked_u32(entries.len(), "LPG property column entry count")?.to_le_bytes(),
    );
    for (entity_id, versions) in &entries {
        buf.extend_from_slice(&entity_id.to_le_bytes());
        buf.extend_from_slice(
            &checked_u16(versions.len(), "LPG property version count")?.to_le_bytes(),
        );
        for (epoch, value) in *versions {
            buf.extend_from_slice(&epoch.as_u64().to_le_bytes());
            encode_value(value, strings, &mut buf)?;
        }
    }

    Ok(buf)
}

// ── Reader ─────────────────────────────────────────────────────────

/// Reads block-based LPG section data and populates the store.
pub(crate) fn read_blocks(
    data: &[u8],
    populate: &mut dyn FnMut(
        Vec<BlockNode>,
        Vec<BlockEdge>,
        Vec<BlockNamedGraph>,
        u64,
    ) -> Result<()>,
) -> Result<()> {
    let header = SectionHeader::read_from(data)
        .ok_or_else(|| Error::Serialization("invalid LPG block section header".to_string()))?;

    if header.version > LPG_BLOCK_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported LPG block version {}, max supported is {LPG_BLOCK_VERSION}",
            header.version
        )));
    }

    let block_count = header.block_count as usize;
    let dir_start = HEADER_SIZE;
    let dir_end = dir_start + block_count * BlockDirEntry::SIZE;

    if data.len() < dir_end {
        return Err(Error::Serialization(
            "LPG block section too short for directory".to_string(),
        ));
    }

    // Parse directory
    let mut dir_entries = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let entry_start = dir_start + i * BlockDirEntry::SIZE;
        let entry = BlockDirEntry::read_from(&data[entry_start..])
            .ok_or_else(|| Error::Serialization(format!("invalid block directory entry {i}")))?;
        dir_entries.push(entry);
    }

    // Verify checksums and extract blocks
    for (i, entry) in dir_entries.iter().enumerate() {
        let (start, end) = (entry.range().start, entry.range().end);
        if end > data.len() {
            return Err(Error::Serialization(format!(
                "block {i} extends past end of data"
            )));
        }
        let actual_crc = crc32fast::hash(&data[start..end]);
        if actual_crc != entry.checksum {
            return Err(Error::Serialization(format!(
                "block {i} CRC mismatch: expected {:08x}, got {actual_crc:08x}",
                entry.checksum
            )));
        }
    }

    // Find string table (must be first or at least present)
    let st_entry = dir_entries
        .iter()
        .find(|e| e.block_type == BlockType::StringTable as u8)
        .ok_or_else(|| Error::Serialization("missing string table block".to_string()))?;
    let st_data = &data[st_entry.range()];
    let strings = StringTableReader::new(st_data)
        .ok_or_else(|| Error::Serialization("invalid string table".to_string()))?;

    // Read node data (cap capacity to prevent OOM from untrusted header)
    let node_count = usize::try_from(header.node_count).unwrap_or(usize::MAX);
    let mut nodes = Vec::with_capacity(node_count.min(data.len() / 8));
    if let Some(entry) = dir_entries
        .iter()
        .find(|e| e.block_type == BlockType::NodeData as u8)
    {
        let block = &data[entry.range()];
        let mut pos = 0;
        while pos + 8 <= block.len() {
            let id = NodeId::new(u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()));
            pos += 8;
            nodes.push(BlockNode {
                id,
                labels: Vec::new(),
                properties: Vec::new(),
            });
        }
    }

    // Read edge data
    let edge_count = usize::try_from(header.edge_count).unwrap_or(usize::MAX);
    let mut edges = Vec::with_capacity(edge_count.min(data.len() / 28));
    if let Some(entry) = dir_entries
        .iter()
        .find(|e| e.block_type == BlockType::EdgeData as u8)
    {
        let block = &data[entry.range()];
        let mut pos = 0;
        while pos + 28 <= block.len() {
            let id = EdgeId::new(u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()));
            pos += 8;
            let src = NodeId::new(u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()));
            pos += 8;
            let dst = NodeId::new(u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()));
            pos += 8;
            let type_idx = u32::from_le_bytes(block[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let edge_type = strings
                .get(type_idx)
                .ok_or_else(|| {
                    Error::Serialization(format!("invalid edge type string index {type_idx}"))
                })?
                .to_owned();
            edges.push(BlockEdge {
                id,
                src,
                dst,
                edge_type,
                properties: Vec::new(),
            });
        }
    }

    // Read label assignments
    if let Some(entry) = dir_entries
        .iter()
        .find(|e| e.block_type == BlockType::LabelAssignment as u8)
    {
        let block = &data[entry.range()];
        let mut pos = 0;
        ensure_remaining(block, pos, 4)?;
        let node_count = u32::from_le_bytes(block[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        for i in 0..node_count.min(nodes.len()) {
            ensure_remaining(block, pos, 2)?;
            let label_count = u16::from_le_bytes(block[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let mut labels = Vec::with_capacity(label_count);
            for _ in 0..label_count {
                ensure_remaining(block, pos, 4)?;
                let idx = u32::from_le_bytes(block[pos..pos + 4].try_into().unwrap());
                pos += 4;
                let label = strings.get(idx).ok_or_else(|| {
                    Error::Serialization(format!("invalid label string index {idx}"))
                })?;
                labels.push(label.to_owned());
            }
            nodes[i].labels = labels;
        }
    }

    // Read property columns
    // Build an index from entity_id to position in nodes/edges
    let node_id_map: hashbrown::HashMap<u64, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_u64(), i))
        .collect();
    let edge_id_map: hashbrown::HashMap<u64, usize> = edges
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_u64(), i))
        .collect();

    for entry in &dir_entries {
        if entry.block_type != BlockType::PropertyColumn as u8 {
            continue;
        }
        let block = &data[entry.range()];
        let key_name = strings.get(entry.key_string_index).ok_or_else(|| {
            Error::Serialization(format!(
                "invalid property key string index {}",
                entry.key_string_index
            ))
        })?;
        let is_edge_prop = entry.sub_type == 1;

        let mut pos = 0;
        ensure_remaining(block, pos, 4)?;
        let entry_count = u32::from_le_bytes(block[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        for _ in 0..entry_count {
            ensure_remaining(block, pos, 10)?; // entity_id(8) + version_count(2)
            let entity_id = u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let version_count =
                u16::from_le_bytes(block[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;

            let mut versions = Vec::with_capacity(version_count);
            for _ in 0..version_count {
                ensure_remaining(block, pos, 8)?;
                let epoch =
                    EpochId::new(u64::from_le_bytes(block[pos..pos + 8].try_into().unwrap()));
                pos += 8;
                let value = decode_value(block, &mut pos, &strings)?;
                versions.push((epoch, value));
            }

            if is_edge_prop {
                if let Some(&idx) = edge_id_map.get(&entity_id) {
                    edges[idx].properties.push((key_name.to_owned(), versions));
                }
            } else if let Some(&idx) = node_id_map.get(&entity_id) {
                nodes[idx].properties.push((key_name.to_owned(), versions));
            }
        }
    }

    // Read named graphs
    let mut named_graphs = Vec::new();
    for entry in &dir_entries {
        if entry.block_type != BlockType::NamedGraph as u8 {
            continue;
        }
        let block = &data[entry.range()];
        let graph_name = strings.get(entry.key_string_index).ok_or_else(|| {
            Error::Serialization(format!(
                "invalid graph name string index {}",
                entry.key_string_index
            ))
        })?;

        // Recursively read the named graph's sub-section
        let mut graph_nodes = Vec::new();
        let mut graph_edges = Vec::new();
        read_blocks(block, &mut |n, e, _, _| {
            graph_nodes = n;
            graph_edges = e;
            Ok(())
        })?;

        named_graphs.push(BlockNamedGraph {
            name: graph_name.to_owned(),
            nodes: graph_nodes,
            edges: graph_edges,
        });
    }

    populate(nodes, edges, named_graphs, header.epoch)?;
    Ok(())
}

/// Returns `true` if the given data starts with the LPG block magic bytes.
#[cfg(test)]
pub(crate) fn is_block_format(data: &[u8]) -> bool {
    data.len() >= 4 && data[0..4] == LPG_BLOCK_MAGIC
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::EpochId;

    #[test]
    fn test_empty_round_trip() {
        let data = write_blocks(&[], &[], &[], 0).unwrap();
        assert!(is_block_format(&data));

        let mut got_nodes = 0;
        let mut got_edges = 0;
        read_blocks(&data, &mut |nodes, edges, graphs, epoch| {
            got_nodes = nodes.len();
            got_edges = edges.len();
            assert!(graphs.is_empty());
            assert_eq!(epoch, 0);
            Ok(())
        })
        .unwrap();
        assert_eq!(got_nodes, 0);
        assert_eq!(got_edges, 0);
    }

    #[test]
    fn test_nodes_only_round_trip() {
        let nodes = vec![
            BlockNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
                properties: vec![(
                    "name".to_string(),
                    vec![(EpochId::new(1), Value::String("Alix".into()))],
                )],
            },
            BlockNode {
                id: NodeId::new(2),
                labels: vec!["Person".to_string(), "Employee".to_string()],
                properties: vec![
                    (
                        "name".to_string(),
                        vec![(EpochId::new(1), Value::String("Gus".into()))],
                    ),
                    ("age".to_string(), vec![(EpochId::new(1), Value::Int64(30))]),
                ],
            },
        ];

        let data = write_blocks(&nodes, &[], &[], 42).unwrap();
        assert!(is_block_format(&data));

        read_blocks(&data, &mut |decoded_nodes, edges, _, epoch| {
            assert_eq!(epoch, 42);
            assert_eq!(decoded_nodes.len(), 2);
            assert!(edges.is_empty());

            assert_eq!(decoded_nodes[0].id, NodeId::new(1));
            assert_eq!(decoded_nodes[0].labels, vec!["Person"]);
            assert_eq!(decoded_nodes[0].properties.len(), 1);
            assert_eq!(decoded_nodes[0].properties[0].0, "name");

            assert_eq!(decoded_nodes[1].id, NodeId::new(2));
            assert_eq!(decoded_nodes[1].labels.len(), 2);
            assert_eq!(decoded_nodes[1].properties.len(), 2);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_edges_round_trip() {
        let nodes = vec![
            BlockNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
                properties: vec![],
            },
            BlockNode {
                id: NodeId::new(2),
                labels: vec!["Person".to_string()],
                properties: vec![],
            },
        ];
        let edges = vec![BlockEdge {
            id: EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "KNOWS".to_string(),
            properties: vec![(
                "since".to_string(),
                vec![(EpochId::new(1), Value::Int64(2020))],
            )],
        }];

        let data = write_blocks(&nodes, &edges, &[], 5).unwrap();

        read_blocks(&data, &mut |_, decoded_edges, _, _| {
            assert_eq!(decoded_edges.len(), 1);
            assert_eq!(decoded_edges[0].edge_type, "KNOWS");
            assert_eq!(decoded_edges[0].src, NodeId::new(1));
            assert_eq!(decoded_edges[0].dst, NodeId::new(2));
            assert_eq!(decoded_edges[0].properties.len(), 1);
            assert_eq!(decoded_edges[0].properties[0].0, "since");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_all_value_types_round_trip() {
        use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};

        let props = vec![
            ("null_val".to_string(), vec![(EpochId::new(0), Value::Null)]),
            (
                "bool_val".to_string(),
                vec![(EpochId::new(0), Value::Bool(true))],
            ),
            (
                "int_val".to_string(),
                vec![(EpochId::new(0), Value::Int64(-42))],
            ),
            (
                "float_val".to_string(),
                vec![(EpochId::new(0), Value::Float64(2.72))],
            ),
            (
                "string_val".to_string(),
                vec![(EpochId::new(0), Value::String("hello".into()))],
            ),
            (
                "bytes_val".to_string(),
                vec![(EpochId::new(0), Value::Bytes(vec![1, 2, 3].into()))],
            ),
            (
                "date_val".to_string(),
                vec![(EpochId::new(0), Value::Date(Date::from_days(19000)))],
            ),
            (
                "time_val".to_string(),
                vec![(
                    EpochId::new(0),
                    Value::Time(
                        Time::from_nanos(43_200_000_000_000)
                            .unwrap()
                            .with_offset(3600),
                    ),
                )],
            ),
            (
                "ts_val".to_string(),
                vec![(
                    EpochId::new(0),
                    Value::Timestamp(Timestamp::from_millis(1_700_000_000_000)),
                )],
            ),
            (
                "zdt_val".to_string(),
                vec![(
                    EpochId::new(0),
                    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                        Timestamp::from_millis(1_700_000_000_000),
                        3600,
                    )),
                )],
            ),
            (
                "dur_val".to_string(),
                vec![(
                    EpochId::new(0),
                    Value::Duration(Duration::new(14, 3, 1_000_000)),
                )],
            ),
            (
                "list_val".to_string(),
                vec![(
                    EpochId::new(0),
                    Value::List(vec![Value::Int64(1), Value::String("two".into())].into()),
                )],
            ),
            (
                "vec_val".to_string(),
                vec![(EpochId::new(0), Value::Vector(vec![1.0, 2.0, 3.0].into()))],
            ),
        ];

        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec![],
            properties: props.clone(),
        }];

        let data = write_blocks(&nodes, &[], &[], 0).unwrap();

        read_blocks(&data, &mut |decoded_nodes, _, _, _| {
            let node = &decoded_nodes[0];
            assert_eq!(node.properties.len(), props.len());

            // Properties may come back in a different order (columnar storage
            // groups by key, directory order is alphabetical). Sort both sides.
            let mut decoded_sorted: Vec<_> = node.properties.clone();
            decoded_sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let mut original_sorted = props.clone();
            original_sorted.sort_by(|a, b| a.0.cmp(&b.0));

            for (decoded, original) in decoded_sorted.iter().zip(original_sorted.iter()) {
                assert_eq!(decoded.0, original.0, "property key mismatch");
                assert_eq!(
                    decoded.1.len(),
                    original.1.len(),
                    "version count mismatch for {}",
                    decoded.0
                );
                assert_eq!(
                    decoded.1[0].1, original.1[0].1,
                    "value mismatch for {}",
                    decoded.0
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_string_deduplication() {
        // Same label on multiple nodes should only appear once in string table
        let nodes = vec![
            BlockNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
                properties: vec![],
            },
            BlockNode {
                id: NodeId::new(2),
                labels: vec!["Person".to_string()],
                properties: vec![],
            },
            BlockNode {
                id: NodeId::new(3),
                labels: vec!["Person".to_string()],
                properties: vec![],
            },
        ];

        let data = write_blocks(&nodes, &[], &[], 0).unwrap();

        // Parse header and find string table to verify deduplication
        let _header = SectionHeader::read_from(&data).unwrap();
        let dir_start = HEADER_SIZE;
        let st_entry = BlockDirEntry::read_from(&data[dir_start..]).unwrap();
        let st_data = &data[st_entry.range()];
        let strings = StringTableReader::new(st_data).unwrap();

        // "Person" should only appear once in the string table
        assert_eq!(strings.count, 1);
        assert_eq!(strings.get(0), Some("Person"));
    }

    #[test]
    fn test_named_graph_round_trip() {
        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec!["Root".to_string()],
            properties: vec![],
        }];

        let named_graphs = vec![BlockNamedGraph {
            name: "social".to_string(),
            nodes: vec![BlockNode {
                id: NodeId::new(10),
                labels: vec!["Friend".to_string()],
                properties: vec![(
                    "name".to_string(),
                    vec![(EpochId::new(1), Value::String("Alix".into()))],
                )],
            }],
            edges: vec![],
        }];

        let data = write_blocks(&nodes, &[], &named_graphs, 7).unwrap();

        read_blocks(&data, &mut |decoded_nodes, _, decoded_graphs, epoch| {
            assert_eq!(epoch, 7);
            assert_eq!(decoded_nodes.len(), 1);
            assert_eq!(decoded_graphs.len(), 1);
            assert_eq!(decoded_graphs[0].name, "social");
            assert_eq!(decoded_graphs[0].nodes.len(), 1);
            assert_eq!(decoded_graphs[0].nodes[0].id, NodeId::new(10));
            assert_eq!(decoded_graphs[0].nodes[0].labels, vec!["Friend"]);
            Ok(())
        })
        .unwrap();
    }

    // ── Format limits (#392): refuse to write what cannot be read back ──

    fn write_error(nodes: &[BlockNode]) -> String {
        match write_blocks(nodes, &[], &[], 1) {
            Ok(data) => panic!("expected a serialization error, wrote {} bytes", data.len()),
            Err(err) => {
                assert!(matches!(err, Error::Serialization(_)), "{err:?}");
                err.to_string()
            }
        }
    }

    #[test]
    fn test_too_many_blocks_is_an_error() {
        // Each named graph is one block; with the 4 fixed blocks, 65,531 graphs
        // is exactly u16::MAX blocks and one more overflows the u16 count.
        let graphs = |count: usize| -> Vec<BlockNamedGraph> {
            (0..count)
                .map(|i| BlockNamedGraph {
                    name: format!("g{i}"),
                    nodes: Vec::new(),
                    edges: Vec::new(),
                })
                .collect()
        };
        let at_limit = usize::from(u16::MAX) - 4;

        let err = write_blocks(&[], &[], &graphs(at_limit + 1), 1).unwrap_err();
        assert!(matches!(err, Error::Serialization(_)), "{err:?}");
        assert!(err.to_string().contains("LPG block count: 65536"), "{err}");

        let data = write_blocks(&[], &[], &graphs(at_limit), 1).unwrap();
        read_blocks(&data, &mut |_, _, named, _| {
            assert_eq!(named.len(), at_limit);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_too_many_labels_on_a_node_is_an_error() {
        let node = BlockNode {
            id: NodeId::new(1),
            labels: (0..=usize::from(u16::MAX))
                .map(|i| format!("L{i}"))
                .collect(),
            properties: Vec::new(),
        };
        let err = write_error(&[node]);
        assert!(err.contains("LPG labels per node: 65536"), "{err}");
    }

    #[test]
    fn test_too_many_property_versions_is_an_error() {
        let versions = (0..=u64::from(u16::MAX))
            .map(|e| (EpochId::new(e), Value::Int64(1)))
            .collect();
        let node = BlockNode {
            id: NodeId::new(1),
            labels: Vec::new(),
            properties: vec![("v".to_string(), versions)],
        };
        let err = write_error(std::slice::from_ref(&node));
        assert!(err.contains("LPG property version count: 65536"), "{err}");

        // Edge property columns have the same limit.
        let edge = BlockEdge {
            id: EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(1),
            edge_type: "SELF".to_string(),
            properties: node.properties,
        };
        let node = BlockNode {
            id: NodeId::new(1),
            labels: Vec::new(),
            properties: Vec::new(),
        };
        let err = write_blocks(&[node], &[edge], &[], 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("LPG property version count: 65536"),
            "{err}"
        );
    }

    #[test]
    fn test_block_range_past_end_is_an_error_not_a_panic() {
        let data = write_blocks(&[], &[], &[], 1).unwrap();
        // Point the first directory entry near u32::MAX: offset + length would
        // overflow u32 arithmetic.
        let mut corrupt = data.clone();
        let entry = HEADER_SIZE;
        corrupt[entry + 4..entry + 8].copy_from_slice(&(u32::MAX - 1).to_le_bytes());
        corrupt[entry + 8..entry + 12].copy_from_slice(&16u32.to_le_bytes());
        let err = read_blocks(&corrupt, &mut |_, _, _, _| Ok(())).unwrap_err();
        assert!(
            err.to_string().contains("extends past end of data"),
            "{err}"
        );
    }

    #[test]
    fn test_crc_corruption_detected() {
        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec!["Test".to_string()],
            properties: vec![],
        }];

        let mut data = write_blocks(&nodes, &[], &[], 0).unwrap();

        // Corrupt a byte in the node data block
        let last = data.len() - 1;
        data[last] ^= 0xFF;

        let result = read_blocks(&data, &mut |_, _, _, _| Ok(()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("CRC mismatch"), "unexpected error: {err}");
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        assert!(!is_block_format(&data));
    }

    #[test]
    fn test_temporal_version_history() {
        // Multiple versions per property (temporal feature)
        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec![],
            properties: vec![(
                "name".to_string(),
                vec![
                    (EpochId::new(1), Value::String("Alix".into())),
                    (EpochId::new(5), Value::String("Gus".into())),
                    (EpochId::new(10), Value::String("Vincent".into())),
                ],
            )],
        }];

        let data = write_blocks(&nodes, &[], &[], 10).unwrap();

        read_blocks(&data, &mut |decoded, _, _, _| {
            let versions = &decoded[0].properties[0].1;
            assert_eq!(versions.len(), 3);
            assert_eq!(versions[0].0, EpochId::new(1));
            assert_eq!(versions[1].0, EpochId::new(5));
            assert_eq!(versions[2].0, EpochId::new(10));
            assert_eq!(versions[0].1, Value::String("Alix".into()));
            assert_eq!(versions[1].1, Value::String("Gus".into()));
            assert_eq!(versions[2].1, Value::String("Vincent".into()));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    // reason: test IDs 1..=1000 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_large_graph_round_trip() {
        // 1000 nodes, 2000 edges
        let nodes: Vec<BlockNode> = (1..=1000)
            .map(|i| BlockNode {
                id: NodeId::new(i),
                labels: vec!["Node".to_string()],
                properties: vec![(
                    "index".to_string(),
                    vec![(EpochId::new(0), Value::Int64(i as i64))],
                )],
            })
            .collect();

        let edges: Vec<BlockEdge> = (1..=2000)
            .map(|i| BlockEdge {
                id: EdgeId::new(i),
                src: NodeId::new((i % 1000) + 1),
                dst: NodeId::new(((i + 1) % 1000) + 1),
                edge_type: "LINK".to_string(),
                properties: vec![],
            })
            .collect();

        let data = write_blocks(&nodes, &edges, &[], 100).unwrap();

        read_blocks(&data, &mut |decoded_nodes, decoded_edges, _, epoch| {
            assert_eq!(epoch, 100);
            assert_eq!(decoded_nodes.len(), 1000);
            assert_eq!(decoded_edges.len(), 2000);

            // Verify IDs are preserved
            for (i, node) in decoded_nodes.iter().enumerate() {
                assert_eq!(node.id, NodeId::new((i + 1) as u64));
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_map_value_round_trip() {
        let mut map = BTreeMap::new();
        map.insert("key1".into(), Value::Int64(1));
        map.insert("key2".into(), Value::String("val".into()));

        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec![],
            properties: vec![(
                "metadata".to_string(),
                vec![(EpochId::new(0), Value::Map(Arc::new(map.clone())))],
            )],
        }];

        let data = write_blocks(&nodes, &[], &[], 0).unwrap();

        read_blocks(&data, &mut |decoded, _, _, _| {
            let val = &decoded[0].properties[0].1[0].1;
            assert_eq!(*val, Value::Map(Arc::new(map.clone())));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_truncated_header_rejected() {
        // Header requires at least HEADER_SIZE bytes
        let result = read_blocks(&[0; 4], &mut |_, _, _, _| Ok(()));
        assert!(result.is_err(), "truncated data should fail");
    }

    #[test]
    fn test_truncated_after_header_rejected() {
        // Valid magic + version but truncated before directory
        let mut data = Vec::new();
        data.extend_from_slice(&LPG_BLOCK_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes()); // version
        // Not enough bytes for the rest of the header
        let result = read_blocks(&data, &mut |_, _, _, _| Ok(()));
        assert!(result.is_err(), "truncated header should fail");
    }

    #[test]
    fn test_inflated_node_count_does_not_oom() {
        // Corrupt the node count in the header to u32::MAX.
        // The reader clamps Vec capacity to data.len() to prevent OOM,
        // then either errors or gracefully truncates.
        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec!["A".to_string()],
            properties: vec![],
        }];
        let mut data = write_blocks(&nodes, &[], &[], 0).unwrap();
        data[8..12].copy_from_slice(&u32::MAX.to_le_bytes());

        let result = read_blocks(&data, &mut |decoded_nodes, _, _, _| {
            assert!(decoded_nodes.len() < u32::MAX as usize);
            Ok(())
        });
        // Either error or graceful truncation is acceptable
        let _ = result;
    }

    #[test]
    fn test_inflated_edge_count_does_not_oom() {
        let edges = vec![BlockEdge {
            id: EdgeId::new(1),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "E".to_string(),
            properties: vec![],
        }];
        let mut data = write_blocks(&[], &edges, &[], 0).unwrap();
        data[12..16].copy_from_slice(&u32::MAX.to_le_bytes());

        let result = read_blocks(&data, &mut |_, decoded_edges, _, _| {
            assert!(decoded_edges.len() < u32::MAX as usize);
            Ok(())
        });
        let _ = result;
    }

    #[test]
    fn test_diverse_value_types_round_trip() {
        use grafeo_common::types::{Date, Time};

        let nodes = vec![BlockNode {
            id: NodeId::new(1),
            labels: vec![],
            properties: vec![
                (
                    "bool_val".to_string(),
                    vec![(EpochId::new(0), Value::Bool(true))],
                ),
                (
                    "float_val".to_string(),
                    vec![(EpochId::new(0), Value::Float64(1.234))],
                ),
                (
                    "bytes_val".to_string(),
                    vec![(EpochId::new(0), Value::Bytes(vec![0xDE, 0xAD].into()))],
                ),
                ("null_val".to_string(), vec![(EpochId::new(0), Value::Null)]),
                (
                    "date_val".to_string(),
                    vec![(
                        EpochId::new(0),
                        Value::Date(Date::from_ymd(2026, 4, 11).unwrap()),
                    )],
                ),
                (
                    "time_val".to_string(),
                    vec![(
                        EpochId::new(0),
                        Value::Time(Time::from_hms(14, 30, 0).unwrap()),
                    )],
                ),
                (
                    "list_val".to_string(),
                    vec![(
                        EpochId::new(0),
                        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into()),
                    )],
                ),
                (
                    "vector_val".to_string(),
                    vec![(EpochId::new(0), Value::Vector(vec![0.1, 0.2, 0.3].into()))],
                ),
            ],
        }];

        let data = write_blocks(&nodes, &[], &[], 0).unwrap();

        read_blocks(&data, &mut |decoded, _, _, _| {
            assert_eq!(decoded.len(), 1);
            let props = &decoded[0].properties;
            assert_eq!(
                props.len(),
                8,
                "all 8 property types should survive roundtrip"
            );

            // Verify each type decoded correctly
            let find_prop =
                |name: &str| -> &Value { &props.iter().find(|(k, _)| k == name).unwrap().1[0].1 };
            assert_eq!(*find_prop("bool_val"), Value::Bool(true));
            assert_eq!(*find_prop("float_val"), Value::Float64(1.234));
            assert_eq!(*find_prop("null_val"), Value::Null);
            assert!(matches!(find_prop("bytes_val"), Value::Bytes(_)));
            assert!(matches!(find_prop("date_val"), Value::Date(_)));
            assert!(matches!(find_prop("time_val"), Value::Time(_)));
            assert!(matches!(find_prop("list_val"), Value::List(_)));
            assert!(matches!(find_prop("vector_val"), Value::Vector(_)));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_empty_labels_and_properties() {
        let nodes = vec![
            BlockNode {
                id: NodeId::new(1),
                labels: vec![],     // no labels
                properties: vec![], // no properties
            },
            BlockNode {
                id: NodeId::new(2),
                labels: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                properties: vec![],
            },
        ];

        let data = write_blocks(&nodes, &[], &[], 0).unwrap();

        read_blocks(&data, &mut |decoded, _, _, _| {
            assert_eq!(decoded.len(), 2);
            assert!(decoded[0].labels.is_empty());
            assert_eq!(decoded[1].labels.len(), 3);
            Ok(())
        })
        .unwrap();
    }
}
